package benchmark

// vLLM's cache-source metrics engine (see cache_metrics.go for the shared
// sampler/record-type/report machinery this plugs into).
//
// Eligibility is deliberately looser than type=openai_vllm (vllmEngine.
// Speculative = true). Any chat/completions endpoint — type=openai, the
// default type, or an autodiscovered bare host:port — may well be vLLM
// behind the scenes, and the only way to find out is to ask. The cost of
// guessing wrong is bounded by metricsSpeculativeBudget; the cost of not
// guessing is a whole benchmark run with no cache-source breakdown.
// type=openai_vllm still differs in one way: the operator asserted the
// server is vLLM, so a sampler for such a spec keeps retrying forever rather
// than giving up (see metricsSampler.noteFailure).
//
// The counter family is the fork's vllm:prompt_tokens_by_source
// (vllm/v1/metrics/loggers.py) with label source in {local_compute,
// local_cache_hit, external_kv_transfer} (vllm/v1/metrics/stats.py
// PromptTokenStats.ALL_SOURCES); the text exposition renders it as
// vllm:prompt_tokens_by_source_total. Values are summed across all other
// labels (model_name, engine index).
//
// vllmEngine.Derive projects those three source labels onto the canonical
// band set: local_compute -> compute (direct), local_cache_hit -> host (vLLM
// doesn't distinguish GPU-resident vs CPU-resident local cache, so "host" is
// the least-surprising catch-all), external_kv_transfer -> external (NOT
// "storage" — vLLM's external_kv_transfer means "any KV connector" per the
// vLLM fork's own source comments, not specifically WEKA/L3, so it must
// render in a different color from sglang's "storage" band on a merged
// chart — see mixBandStyleJS).

import (
	"bufio"
	"io"
	"regexp"
	"strconv"
	"strings"
)

const (
	vllmPromptTokensBySourceFamily = "vllm:prompt_tokens_by_source"
)

var promSourceLabelRe = regexp.MustCompile(`\bsource="([^"]*)"`)

// parsePromptTokensBySource scans Prometheus text exposition for the
// vllm:prompt_tokens_by_source family and returns cumulative values keyed by
// the source label, summed across all other label combinations (engine index,
// model_name).
//
// The series to match is vllm:prompt_tokens_by_source_total — prometheus_client
// appends the counter suffix, so that is the only form a real vLLM /metrics
// emits. The bare name is accepted too, purely as tolerance for an exposition
// path that doesn't append it; the two never coexist in one scrape. (The bare
// name does appear on the # HELP/# TYPE lines, but those are skipped as
// comments before any name matching.)
func parsePromptTokensBySource(r io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		rest, ok := strings.CutPrefix(line, vllmPromptTokensBySourceFamily)
		if !ok {
			continue
		}
		rest, _ = strings.CutPrefix(rest, "_total")
		// Reject sibling series like _created and other families sharing the
		// prefix: the next byte must open the label set.
		if len(rest) == 0 || rest[0] != '{' {
			continue
		}
		closeIdx := strings.LastIndex(rest, "}")
		if closeIdx < 0 {
			continue
		}
		fields := strings.Fields(rest[closeIdx+1:])
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			continue
		}
		m := promSourceLabelRe.FindStringSubmatch(rest[1:closeIdx])
		if m == nil {
			continue
		}
		out[m[1]] += v
	}
	return out, sc.Err()
}

// vllmEngine is the cacheEngine value for vLLM — see cache_metrics.go for
// what each field means.
var vllmEngine = cacheEngine{
	Name:        "vllm",
	SpecTypes:   []string{"openai_vllm"},
	Speculative: true,
	FamilyHint:  vllmPromptTokensBySourceFamily,
	FlagHint:    "type=openai_vllm",
	Parse:       parsePromptTokensBySource,
	Bands:       []string{"compute", "host", "external"},
	Derive: func(totals map[string]float64) map[string]float64 {
		return map[string]float64{
			"compute":  totals["local_compute"],
			"host":     totals["local_cache_hit"],
			"external": totals["external_kv_transfer"],
		}
	},
}
