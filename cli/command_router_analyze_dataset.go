package cli

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/benchmark"
)

// RouterAnalyzeDatasetCommand models the real-time replay's active dataset
// (the benchmark's active_dataset_tokens) over a long run and says when it
// settles.
type RouterAnalyzeDatasetCommand struct {
	Format         string  `long:"format" choice:"text" choice:"json" default:"text" description:"Output format"`
	ResponseScale  float64 `long:"response-scale" default:"3" description:"Each request runs for its captured total_ms times this. It stretches sessions, so it moves both the slot count and the settle time; take it from a run's [realtime] resp_ratio (about 3 for GLM-5.3 on B300 at full load)."`
	TargetInflight float64 `long:"target-inflight" description:"Model the TTFT governor as a load target: --start-series slots, then --admit-count more each --admit-every while the interval's mean in-flight requests stayed below this. Take it from a run (mean response time x requests per second). Exclusive with --series."`
	Series         int     `long:"series" description:"Fix the slot count instead of modelling the governor. Exclusive with --target-inflight."`
	StartSeries    int     `long:"start-series" default:"32" description:"Governor: slots at start (benchmark default under realtime)"`
	AdmitEvery     string  `long:"admit-every" default:"1m" description:"Governor: admission interval"`
	AdmitCount     int     `long:"admit-count" default:"16" description:"Governor: slots added per open interval"`
	MaxSeries      int     `long:"max-series" description:"Governor: slot cap (0 = none)"`
	NoRetire       bool    `long:"no-retire" description:"Governor: never retire slots (the benchmark's --ttft-retire=false)"`
	RetireMargin   float64 `long:"retire-margin" default:"0.15" description:"Governor: a finished session's slot is retired while the last interval ran this fraction above --target-inflight"`
	Strategy       string  `long:"strategy" choice:"reverse-from-end" choice:"beginning" choice:"random" default:"reverse-from-end" description:"Session order per lap, as --replay-selection-strategy"`
	Horizon        string  `long:"horizon" default:"30d" description:"How long to simulate (units s, m, h, d)"`
	SampleEvery    string  `long:"sample-every" default:"10m" description:"Sampling interval"`
	ReportEvery    string  `long:"report-every" default:"12h" description:"Timeline row interval in the text report"`
	Tolerance      float64 `long:"tolerance" default:"0.05" description:"Settled = the 6h trailing mean stays within this fraction of the long-run value"`
	TokenScale     float64 `long:"token-scale" default:"1" description:"Multiply captured prompt tokens by this to express the dataset in the served model's tokens (measured run per-slot dataset / simulated per-slot dataset)"`

	Args struct {
		Path string `positional-arg-name:"PATH" description:"replay file (replay-v4/v5 with think_ms)"`
	} `positional-args:"yes" required:"yes"`
}

type datasetLightRequest struct {
	RequestID   uint64 `json:"request_id"`
	Ts          string `json:"ts"`
	InputTokens int    `json:"input_tokens"`
	TotalMs     int64  `json:"total_ms"`
	ThinkMs     *int64 `json:"think_ms,omitempty"`
	ThinkRef    string `json:"think_ref,omitempty"`
}

type datasetLightSession struct {
	SessionID string `json:"session_id"`
	StartTs   string `json:"start_ts"`
	Instances []struct {
		ParentSpawnRequestID uint64                `json:"parent_spawn_request_id"`
		Requests             []datasetLightRequest `json:"requests"`
	} `json:"instances"`
}

func (l *datasetLightSession) toReplay() *benchmark.RouterReplaySession {
	s := &benchmark.RouterReplaySession{SessionID: l.SessionID, StartTs: l.StartTs}
	s.Instances = make([]benchmark.RouterReplayInstance, len(l.Instances))
	for i, in := range l.Instances {
		s.Instances[i].ParentSpawnRequestID = in.ParentSpawnRequestID
		rs := make([]benchmark.RouterReplayRequest, len(in.Requests))
		for j, r := range in.Requests {
			rs[j] = benchmark.RouterReplayRequest{RequestID: r.RequestID, Ts: r.Ts, InputTokens: r.InputTokens, TotalMs: r.TotalMs, ThinkMs: r.ThinkMs, ThinkRef: r.ThinkRef}
		}
		s.Instances[i].Requests = rs
	}
	return s
}

// loadDatasetSessions parses the replay file in parallel, keeping only each
// session's timeline, in file order.
func loadDatasetSessions(path string, respScale, tokenScale float64) ([]benchmark.DatasetSession, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	type job struct {
		idx  int
		line []byte
	}
	jobs := make(chan job, runtime.NumCPU()*2)
	var (
		mu     sync.Mutex
		out    = map[int]benchmark.DatasetSession{}
		bad    error
		wg     sync.WaitGroup
		thinks int
	)
	for range runtime.NumCPU() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				var l datasetLightSession
				if err := json.Unmarshal(j.line, &l); err != nil {
					mu.Lock()
					bad = fmt.Errorf("session line %d: %w", j.idx, err)
					mu.Unlock()
					continue
				}
				withThink := 0
				for _, in := range l.Instances {
					for _, r := range in.Requests {
						if r.ThinkMs != nil {
							withThink++
						}
					}
				}
				ds := benchmark.BuildDatasetSession(l.toReplay(), respScale)
				if tokenScale != 1 {
					for k := range ds.Requests {
						ds.Requests[k].Prompt = int64(math.Round(float64(ds.Requests[k].Prompt) * tokenScale))
					}
					ds.PromptMsArea *= tokenScale
				}
				mu.Lock()
				out[j.idx] = ds
				thinks += withThink
				mu.Unlock()
			}
		}()
	}
	r := bufio.NewReaderSize(f, 16<<20)
	idx := 0
	for {
		line, rerr := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if len(line) > 0 && !bytes.HasPrefix(line, []byte(`{"_schema"`)) && !benchmark.IsReplayIndexLine(line) {
			jobs <- job{idx, line}
			idx++
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			close(jobs)
			wg.Wait()
			return nil, rerr
		}
	}
	close(jobs)
	wg.Wait()
	if bad != nil {
		return nil, bad
	}
	if thinks == 0 {
		return nil, fmt.Errorf("%s has no think_ms: real-time think pacing cannot be modelled (regenerate it with 'wekai router replay-prepare')", path)
	}
	sessions := make([]benchmark.DatasetSession, idx)
	for i := range idx {
		sessions[i] = out[i]
	}
	return sessions, nil
}

func parseSpan(name, v string) (time.Duration, error) {
	ws, err := benchmark.ParseRetentionWindows(v)
	if err != nil || len(ws) != 1 {
		return 0, fmt.Errorf("--%s %q: want one duration (units s, m, h, d)", name, v)
	}
	return ws[0], nil
}

// datasetDurationBuckets split slot time by the length of the session holding
// the slot: the share above a day is what a one-day run cannot have settled.
var datasetDurationBuckets = benchmark.DatasetLifeBounds

type datasetBucket struct {
	Label       string  `json:"label"`
	Sessions    int     `json:"sessions"`
	SlotTimePct float64 `json:"slot_time_pct"`
	DatasetPct  float64 `json:"dataset_time_pct"`
	RequestsPct float64 `json:"requests_pct"`
	InflightPct float64 `json:"busy_time_pct"`
	MeanPromptK float64 `json:"mean_tracked_prompt_k"`
}

// datasetWaitBucket groups sessions by their longest think gap.
type datasetWaitBucket struct {
	Label       string  `json:"label"`
	Sessions    int     `json:"sessions"`
	SessionsPct float64 `json:"sessions_pct"`
	SlotTimePct float64 `json:"slot_time_pct"`
	RequestsPct float64 `json:"requests_pct"`
	// GapsAtLeast is the number of gaps, over all sessions, at or above the
	// bucket's lower edge.
	GapsAtLeast int `json:"gaps_at_least"`
}

type datasetReport struct {
	Sessions        int                         `json:"sessions"`
	ResponseScale   float64                     `json:"response_scale"`
	TokenScale      float64                     `json:"token_scale"`
	DurationP       map[string]time.Duration    `json:"session_duration"`
	Buckets         []datasetBucket             `json:"buckets"`
	Waits           []datasetWaitBucket         `json:"max_wait"`
	Stationary      benchmark.DatasetStationary `json:"stationary"`
	StationarySlots float64                     `json:"stationary_slots,omitempty"`
	StationaryData  float64                     `json:"stationary_dataset_tokens"`
	// Plateau is the simulated mean over the horizon's last quarter: what the
	// settle and reach times are measured against.
	PlateauData  float64                   `json:"plateau_dataset_tokens"`
	PlateauSlots float64                   `json:"plateau_slots"`
	LapTime      time.Duration             `json:"lap_time"`
	Settle       map[string]time.Duration  `json:"settle"`
	Reach        map[string]time.Duration  `json:"reach_dataset"`
	Samples      []benchmark.DatasetSample `json:"samples"`
}

func (c *RouterAnalyzeDatasetCommand) Execute(args []string) error {
	if (c.Series > 0) == (c.TargetInflight > 0) {
		return fmt.Errorf("pass exactly one of --target-inflight (model the governor) and --series (fixed slots)")
	}
	horizon, err := parseSpan("horizon", c.Horizon)
	if err != nil {
		return err
	}
	sampleEvery, err := parseSpan("sample-every", c.SampleEvery)
	if err != nil {
		return err
	}
	reportEvery, err := parseSpan("report-every", c.ReportEvery)
	if err != nil {
		return err
	}
	admitEvery, err := parseSpan("admit-every", c.AdmitEvery)
	if err != nil {
		return err
	}
	if c.ResponseScale <= 0 || c.TokenScale <= 0 {
		return fmt.Errorf("--response-scale and --token-scale must be positive")
	}

	sessions, err := loadDatasetSessions(c.Args.Path, c.ResponseScale, c.TokenScale)
	if err != nil {
		return err
	}
	cfg := benchmark.DatasetSimConfig{
		Strategy:       c.Strategy,
		Series:         c.Series,
		TargetInflight: c.TargetInflight,
		StartSeries:    c.StartSeries,
		AdmitEvery:     admitEvery,
		AdmitCount:     c.AdmitCount,
		MaxSeries:      c.MaxSeries,
		Retire:         !c.NoRetire,
		RetireMargin:   c.RetireMargin,
		Horizon:        horizon,
		SampleEvery:    sampleEvery,
	}
	samples, err := benchmark.SimulateDataset(sessions, cfg)
	if err != nil {
		return err
	}

	rep := c.buildReport(sessions, samples, sampleEvery)
	if c.Format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	c.printText(rep, reportEvery, sampleEvery)
	return nil
}

func (c *RouterAnalyzeDatasetCommand) buildReport(sessions []benchmark.DatasetSession, samples []benchmark.DatasetSample, sampleEvery time.Duration) datasetReport {
	rep := datasetReport{Sessions: len(sessions), ResponseScale: c.ResponseScale, TokenScale: c.TokenScale, Samples: samples}
	durs := make([]int64, len(sessions))
	for i, s := range sessions {
		durs[i] = s.DurationMs
	}
	sort.Slice(durs, func(a, b int) bool { return durs[a] < durs[b] })
	pct := func(p float64) time.Duration {
		return time.Duration(durs[min(len(durs)-1, int(p*float64(len(durs))))]) * time.Millisecond
	}
	rep.DurationP = map[string]time.Duration{"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": pct(1)}

	var totDur, totArea, totBusy float64
	var totReq int
	for _, s := range sessions {
		totDur += float64(s.DurationMs)
		totArea += s.PromptMsArea
		totBusy += float64(s.BusyMs)
		totReq += len(s.Requests)
	}
	bounds := append([]time.Duration{0}, datasetDurationBuckets...)
	for b := range bounds {
		lo := bounds[b]
		hi := time.Duration(math.MaxInt64)
		label := "> " + fmtSpan(lo)
		if b+1 < len(bounds) {
			hi = bounds[b+1]
			label = fmtSpan(lo) + "–" + fmtSpan(hi)
			if lo == 0 {
				label = "< " + fmtSpan(hi)
			}
		}
		var bk datasetBucket
		var dur, area, busy float64
		var reqs int
		for _, s := range sessions {
			d := time.Duration(s.DurationMs) * time.Millisecond
			if d < lo || d >= hi {
				continue
			}
			bk.Sessions++
			dur += float64(s.DurationMs)
			area += s.PromptMsArea
			busy += float64(s.BusyMs)
			reqs += len(s.Requests)
		}
		bk.Label = label
		bk.SlotTimePct = 100 * dur / totDur
		bk.DatasetPct = 100 * area / totArea
		bk.RequestsPct = 100 * float64(reqs) / float64(totReq)
		bk.InflightPct = 100 * busy / totBusy
		if dur > 0 {
			bk.MeanPromptK = area / dur / 1000
		}
		rep.Buckets = append(rep.Buckets, bk)
	}

	wb := append([]time.Duration{0}, benchmark.DatasetWaitBounds...)
	for b := range wb {
		lo, hi := wb[b], time.Duration(math.MaxInt64)
		label := "≥ " + fmtSpan(lo)
		if b+1 < len(wb) {
			hi = wb[b+1]
			label = fmtSpan(lo) + "–" + fmtSpan(hi)
			if lo == 0 {
				label = "< " + fmtSpan(hi)
			}
		}
		wk := datasetWaitBucket{Label: label}
		var dur float64
		var reqs int
		for _, s := range sessions {
			if b > 0 {
				wk.GapsAtLeast += s.Waits[b-1]
			}
			w := time.Duration(s.MaxWaitMs) * time.Millisecond
			if w < lo || w >= hi {
				continue
			}
			wk.Sessions++
			dur += float64(s.DurationMs)
			reqs += len(s.Requests)
		}
		wk.SessionsPct = 100 * float64(wk.Sessions) / float64(len(sessions))
		wk.SlotTimePct = 100 * dur / totDur
		wk.RequestsPct = 100 * float64(reqs) / float64(totReq)
		rep.Waits = append(rep.Waits, wk)
	}

	st := benchmark.StationaryDataset(sessions)
	rep.Stationary = st
	if c.TargetInflight > 0 && st.PerSlotInflight > 0 {
		rep.StationarySlots = c.TargetInflight / st.PerSlotInflight
	} else {
		rep.StationarySlots = float64(c.Series)
	}
	rep.StationaryData = rep.StationarySlots * st.PerSlotDataset
	if rep.StationarySlots > 0 {
		rep.LapTime = time.Duration(float64(st.LapSlotTime) / rep.StationarySlots)
	}

	tail := samples[len(samples)*3/4:]
	var plateauPer float64
	for _, s := range tail {
		rep.PlateauData += float64(s.Dataset)
		rep.PlateauSlots += float64(s.Slots)
	}
	if len(tail) > 0 {
		rep.PlateauData /= float64(len(tail))
		rep.PlateauSlots /= float64(len(tail))
	}
	if rep.PlateauSlots > 0 {
		plateauPer = rep.PlateauData / rep.PlateauSlots
	}

	window := max(1, int(6*time.Hour/sampleEvery))
	rep.Settle = map[string]time.Duration{
		"dataset": benchmark.SettleTime(samples, func(s benchmark.DatasetSample) float64 { return float64(s.Dataset) }, rep.PlateauData, c.Tolerance, window),
		"slots":   benchmark.SettleTime(samples, func(s benchmark.DatasetSample) float64 { return float64(s.Slots) }, rep.PlateauSlots, c.Tolerance, window),
		"per_slot": benchmark.SettleTime(samples, func(s benchmark.DatasetSample) float64 {
			if s.Slots == 0 {
				return 0
			}
			return float64(s.Dataset) / float64(s.Slots)
		}, plateauPer, c.Tolerance, window),
	}
	rep.Reach = map[string]time.Duration{}
	for _, f := range []float64{0.5, 0.75, 0.9} {
		key := fmt.Sprintf("%.0f%%", f*100)
		rep.Reach[key] = -1
		mean := trailingMeans(samples, window)
		for i, s := range samples {
			if mean[i] >= f*rep.PlateauData {
				rep.Reach[key] = s.At
				break
			}
		}
	}
	return rep
}

func trailingMeans(samples []benchmark.DatasetSample, window int) []float64 {
	out := make([]float64, len(samples))
	var sum float64
	for i, s := range samples {
		sum += float64(s.Dataset)
		if i >= window {
			sum -= float64(samples[i-window].Dataset)
		}
		out[i] = sum / float64(min(i+1, window))
	}
	return out
}

func fmtSpan(d time.Duration) string {
	switch {
	case d < 0:
		return "never"
	case d == 0:
		return "0"
	case d%(24*time.Hour) == 0:
		return fmt.Sprintf("%dd", d/(24*time.Hour))
	case d >= 48*time.Hour:
		return fmt.Sprintf("%.1fd", d.Hours()/24)
	case d >= time.Hour:
		return fmt.Sprintf("%.1fh", d.Hours())
	default:
		return fmt.Sprintf("%.0fm", d.Minutes())
	}
}

func fmtTok(v float64) string {
	switch {
	case v >= 1e6:
		return fmt.Sprintf("%.1fM", v/1e6)
	case v >= 1e3:
		return fmt.Sprintf("%.0fk", v/1e3)
	default:
		return fmt.Sprintf("%.0f", v)
	}
}

func (c *RouterAnalyzeDatasetCommand) printText(rep datasetReport, reportEvery, sampleEvery time.Duration) {
	var b strings.Builder
	fmt.Fprintf(&b, "Active dataset model: %d sessions, responses x%.2f of captured, %s order", rep.Sessions, rep.ResponseScale, c.Strategy)
	if rep.TokenScale != 1 {
		fmt.Fprintf(&b, ", tokens x%.3f", rep.TokenScale)
	}
	if c.TargetInflight > 0 {
		fmt.Fprintf(&b, ", governor at %.1f in flight\n\n", c.TargetInflight)
	} else {
		fmt.Fprintf(&b, ", %d fixed slots\n\n", c.Series)
	}

	fmt.Fprintf(&b, "Session life on the replay clock: p50 %s  p90 %s  p99 %s  max %s\n\n",
		fmtSpan(rep.DurationP["p50"]), fmtSpan(rep.DurationP["p90"]), fmtSpan(rep.DurationP["p99"]), fmtSpan(rep.DurationP["max"]))
	fmt.Fprintf(&b, "%-10s %9s %10s %10s %10s %10s %12s\n", "life", "sessions", "slot-time", "dataset", "requests", "busy", "mean prompt")
	for _, bk := range rep.Buckets {
		fmt.Fprintf(&b, "%-10s %9d %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.0fk\n", bk.Label, bk.Sessions, bk.SlotTimePct, bk.DatasetPct, bk.RequestsPct, bk.InflightPct, bk.MeanPromptK)
	}
	b.WriteString("  slot-time / dataset: share of a slot's life / of its dataset-time spent in sessions of that life;\n")
	b.WriteString("  busy: share of in-flight request time (the load the governor sees).\n\n")

	b.WriteString("Sessions by their longest wait (think gap after a request end):\n")
	fmt.Fprintf(&b, "%-10s %9s %9s %10s %10s %14s\n", "max wait", "sessions", "share", "slot-time", "requests", "gaps >= lower")
	for _, w := range rep.Waits {
		gaps := "-"
		if w.GapsAtLeast > 0 {
			gaps = fmt.Sprintf("%d", w.GapsAtLeast)
		}
		fmt.Fprintf(&b, "%-10s %9d %8.1f%% %9.1f%% %9.1f%% %14s\n", w.Label, w.Sessions, w.SessionsPct, w.SlotTimePct, w.RequestsPct, gaps)
	}
	b.WriteString("  gaps >= lower: think gaps across all sessions at least the bucket's lower edge.\n\n")

	st := rep.Stationary
	fmt.Fprintf(&b, "Long-run (whole laps): %s tokens and %.3f requests in flight per slot", fmtTok(st.PerSlotDataset), st.PerSlotInflight)
	if c.TargetInflight > 0 {
		fmt.Fprintf(&b, " -> %.0f slots, dataset %s\n", rep.StationarySlots, fmtTok(rep.StationaryData))
	} else {
		fmt.Fprintf(&b, " -> dataset %s\n", fmtTok(rep.StationaryData))
	}
	fmt.Fprintf(&b, "One lap over the corpus at that slot count: %s\n\n", fmtSpan(rep.LapTime))

	fmt.Fprintf(&b, "%8s %4s %6s %9s %10s %9s %9s %9s\n", "t", "lap", "slots", "inflight", "dataset", "per-slot", "% plateau", "mean age")
	step := max(1, int(reportEvery/sampleEvery))
	for i := step - 1; i < len(rep.Samples); i += step {
		s := rep.Samples[i]
		per := 0.0
		if s.Slots > 0 {
			per = float64(s.Dataset) / float64(s.Slots)
		}
		fmt.Fprintf(&b, "%8s %4d %6d %9.1f %10s %9s %8.0f%% %9s\n", fmtSpan(s.At), s.Lap, s.Slots, s.Inflight, fmtTok(float64(s.Dataset)), fmtTok(per), 100*float64(s.Dataset)/rep.PlateauData, fmtSpan(s.MeanAge))
	}
	b.WriteString("\nLive slots by the life of the session they hold: % of slots (% of dataset); busy = slots with a request in flight\n")
	fmt.Fprintf(&b, "%8s", "t")
	for _, bk := range rep.Buckets {
		fmt.Fprintf(&b, " %13s", bk.Label)
	}
	fmt.Fprintf(&b, " %7s\n", "busy")
	for i := step - 1; i < len(rep.Samples); i += step {
		s := rep.Samples[i]
		fmt.Fprintf(&b, "%8s", fmtSpan(s.At))
		busy := 0
		for _, l := range s.ByLife {
			fmt.Fprintf(&b, " %5.1f%% (%3.0f%%)", pctOf(l.Slots, s.Slots), 100*float64(l.Dataset)/math.Max(1, float64(s.Dataset)))
			busy += l.Busy
		}
		fmt.Fprintf(&b, " %6.1f%%\n", pctOf(busy, s.Slots))
	}
	fmt.Fprintf(&b, "\nPlateau (mean of the last quarter of %s): %.0f slots, dataset %s\n", fmtSpan(rep.Samples[len(rep.Samples)-1].At), rep.PlateauSlots, fmtTok(rep.PlateauData))
	fmt.Fprintf(&b, "Dataset reaches (6h mean) 50%% at %s, 75%% at %s, 90%% at %s of the plateau.\n",
		fmtSpan(rep.Reach["50%"]), fmtSpan(rep.Reach["75%"]), fmtSpan(rep.Reach["90%"]))
	fmt.Fprintf(&b, "Settled within ±%.0f%% (6h mean, to the end of the horizon): dataset %s, slots %s, per-slot %s.\n",
		c.Tolerance*100, fmtSpan(rep.Settle["dataset"]), fmtSpan(rep.Settle["slots"]), fmtSpan(rep.Settle["per_slot"]))
	fmt.Print(b.String())
}

func pctOf(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}
