package benchmark

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

// recordTypeRunParams tags the run-parameter header row in a request-data
// JSONL file. Like recordTypeVLLMMetricsSample it rides in the same stream and
// is told apart by record_type; request rows carry no record_type at all.
const recordTypeRunParams = "run_params"

// runParamsSchemaVersion is bumped when the MEANING of an existing field
// changes. Adding a field does not need a bump — readers see the zero value,
// which is indistinguishable from "the run didn't set it" and is exactly how
// they must treat a file written before the field existed.
const runParamsSchemaVersion = 1

// runParamsRecord is a single header row describing the run that produced the
// request rows beneath it, written once when the JSONL is opened.
//
// It exists so downstream tooling stops having to be told, out of band, things
// the run already knew. `wekai benchmark visualize` needed --concurrency purely
// because the number was nowhere in the data; getting it wrong (or forgetting
// it) silently mis-sized the rolling-percentile window, and nothing in the
// report said so.
//
// Durations are stored as both a number and a string: the number is what code
// reads, the string is what a human greps for. Zero values mean "not set for
// this run" and every consumer MUST treat them the same as a file that predates
// this record — see runParams.concurrencyOr.
type runParamsRecord struct {
	RecordType string    `json:"record_type"`
	Version    int       `json:"params_version"`
	WrittenAt  time.Time `json:"ts"`

	// Identity
	Model string `json:"model"`
	Alias string `json:"alias,omitempty"`
	RunID string `json:"run_id,omitempty"`

	// Workload shape — the parameters that make two runs comparable.
	Concurrency          int `json:"concurrency"`
	HotSeriesConcurrency int `json:"hot_series_concurrency"`
	MaxSeries            int `json:"max_series"`
	StartSeries          int `json:"start_series"`
	MaxConcurrency       int `json:"max_concurrency"`

	// Budgets
	TimeoutSec        float64 `json:"timeout_sec,omitempty"`
	Timeout           string  `json:"timeout,omitempty"`
	RequestTimeoutSec float64 `json:"request_timeout_sec,omitempty"`
	RequestTimeout    string  `json:"request_timeout,omitempty"`
	TotalRequests     int     `json:"total_requests,omitempty"`
	MaxOutputTokens   int     `json:"max_output_tokens,omitempty"`

	// Source of work: router replay, dataset replay, or synthetic.
	RouterReplayFile      string  `json:"router_replay_file,omitempty"`
	RouterReplayRoles     string  `json:"router_replay_roles,omitempty"`
	ReplayOutputRatio     float64 `json:"replay_output_ratio,omitempty"`
	ReplayMinOutputTokens int     `json:"replay_min_output_tokens,omitempty"`
	ForceOutputVolume     bool    `json:"force_output_volume,omitempty"`
	FromDataset           string  `json:"from_dataset,omitempty"`
	ReplaySeries          int     `json:"replay_series,omitempty"`

	// Synthetic-prompt shaping (ignored in replay modes).
	Step                  int `json:"step,omitempty"`
	StepStartingTokens    int `json:"step_starting_tokens,omitempty"`
	Tokens                int `json:"tokens,omitempty"`
	SharedPrefixPerSeries int `json:"shared_prefix_per_series,omitempty"`

	// Behaviour switches worth knowing when comparing two runs.
	GlobalCacheHitRateTarget  float64 `json:"global_cache_hit_rate_target,omitempty"`
	EndpointOverloadThreshold float64 `json:"endpoint_overload_threshold,omitempty"`
	FIFOGateOrder             bool    `json:"fifo_gate_order,omitempty"`
	ExhaustSessions           bool    `json:"exhaust_sessions,omitempty"`
	DryRun                    bool    `json:"dry_run,omitempty"`
}

// buildRunParams snapshots cfg into a header record. Called after
// runSingleModelBenchmark has applied its defaults, so the recorded values are
// the ones the run actually used rather than the ones the caller typed.
func buildRunParams(cfg AutoBenchmarkConfig, now time.Time) runParamsRecord {
	p := runParamsRecord{
		RecordType: recordTypeRunParams,
		Version:    runParamsSchemaVersion,
		WrittenAt:  now,

		Model: cfg.Model,
		Alias: extractAlias(cfg.Model),
		RunID: cfg.RunID,

		Concurrency:          cfg.Concurrency,
		HotSeriesConcurrency: cfg.HotSeriesConcurrency,
		MaxSeries:            cfg.MaxSeries,
		StartSeries:          cfg.StartSeries,
		MaxConcurrency:       cfg.MaxConcurrency,

		TotalRequests:   cfg.Total,
		MaxOutputTokens: cfg.MaxOutputTokens,

		RouterReplayFile:      cfg.RouterReplayFile,
		RouterReplayRoles:     cfg.RouterReplayRoles,
		ReplayOutputRatio:     cfg.ReplayOutputRatio,
		ReplayMinOutputTokens: cfg.ReplayMinOutputTokens,
		ForceOutputVolume:     cfg.forceVolume(),
		FromDataset:           cfg.FromDataset,
		ReplaySeries:          cfg.ReplaySeries,

		Step:                  cfg.Step,
		StepStartingTokens:    cfg.StepStartingTokens,
		Tokens:                cfg.Tokens,
		SharedPrefixPerSeries: cfg.SharedPrefixPerSeries,

		GlobalCacheHitRateTarget:  cfg.GlobalCacheHitRateTarget,
		EndpointOverloadThreshold: cfg.EndpointOverloadThreshold,
		FIFOGateOrder:             cfg.FIFOGateOrder,
		ExhaustSessions:           cfg.ExhaustSessions,
		DryRun:                    cfg.DryRun,
	}
	if cfg.Timeout > 0 {
		p.TimeoutSec = cfg.Timeout.Seconds()
		p.Timeout = cfg.Timeout.String()
	}
	if cfg.RequestTimeout > 0 {
		p.RequestTimeoutSec = cfg.RequestTimeout.Seconds()
		p.RequestTimeout = cfg.RequestTimeout.String()
	}
	return p
}

// parseRunParams decodes a run_params line. Returns ok=false for anything that
// isn't a well-formed params row, so a caller can keep treating the file as
// legacy rather than acting on half-parsed values.
func parseRunParams(line []byte) (runParamsRecord, bool) {
	var p runParamsRecord
	if err := json.Unmarshal(line, &p); err != nil {
		return runParamsRecord{}, false
	}
	if p.RecordType != recordTypeRunParams {
		return runParamsRecord{}, false
	}
	return p, true
}

// effectiveConcurrency is the request-concurrency the run was held at, or 0
// when the run didn't pin one (hill-climber mode) — in which case there is no
// single number to report and callers fall back to deriving one from the data.
// The hot pool runs on its own gate ON TOP of the normal budget, so the total
// in-flight ceiling is the sum.
func (p runParamsRecord) effectiveConcurrency() int {
	if p.Concurrency <= 0 {
		return 0
	}
	c := p.Concurrency
	if p.HotSeriesConcurrency > 0 {
		c += p.HotSeriesConcurrency
	}
	return c
}

// summaryLine renders the workload shape as a compact one-liner for the report
// header — the parameters that decide whether two arms are comparable at all.
// Empty when the run recorded nothing useful.
func (p runParamsRecord) summaryLine() string {
	parts := make([]string, 0, 6)
	add := func(label string, v int) {
		if v > 0 {
			parts = append(parts, label+"="+strconv.Itoa(v))
		}
	}
	add("conc", p.Concurrency)
	add("hot", p.HotSeriesConcurrency)
	if p.MaxSeries > 0 && p.MaxSeries == p.StartSeries {
		add("series", p.MaxSeries)
	} else {
		add("start-series", p.StartSeries)
		add("max-series", p.MaxSeries)
	}
	add("total", p.TotalRequests)
	if p.Timeout != "" {
		parts = append(parts, "timeout="+p.Timeout)
	}
	if p.RouterReplayFile != "" {
		parts = append(parts, "router-replay")
	} else if p.FromDataset != "" {
		parts = append(parts, "dataset="+p.FromDataset)
	}
	if p.DryRun {
		parts = append(parts, "dry-run")
	}
	return strings.Join(parts, " ")
}
