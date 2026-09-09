package benchmark

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestAggregateCrossChecksAgainstJS is the point of this file: the Go port
// in aggregate.go and the embedded JS in visualize.go (computeDerived,
// windowStats/seriesStats, SUMMARY_METRICS, BASELINE_INDEX/classifyAlias,
// summaryRatioPct) are two independent implementations of the same math, and
// they WILL drift silently if only one of them is ever touched again. This
// test builds one fixture, runs the JS half of the report under node (the
// same reportDOMStub pattern visualize_samples_test.go uses) and the Go half
// via aggregate.go, and asserts every rolling-percentile series, every
// error bar, every summary stat and every baseline ratio agree -- reporting
// the actual maximum divergence measured, not just a pass/fail.
//
// Skips (t.Skip) when node isn't installed, matching every other JS test in
// this package.
func TestAggregateCrossChecksAgainstJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; aggregate cross-check skipped")
	}

	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)

	// Two arms, deterministic but varied: different record counts, spacing,
	// and recorded concurrency (so each gets its OWN rolling window per
	// WindowSize's precedence rule), a scattering of errors, a couple of
	// extreme-latency outliers, and requests that never reported a first
	// token (ttft <= 0, which the rolling TTFT lines and windowStats both
	// treat specially). armA's alias makes it the HBM/no-offload baseline
	// (classifyAlias => "gpu"); armB is a weka arm.
	genRecords := func(alias string, n int, spacingMs int) []requestDataRecord {
		model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=" + alias
		out := make([]requestDataRecord, 0, n)
		for i := 0; i < n; i++ {
			st := base.Add(time.Duration(i*spacingMs) * time.Millisecond)
			r := requestDataRecord{
				StartTime:    st,
				EndTime:      st.Add(time.Second),
				Model:        model,
				SeriesNum:    1 + i/20,
				RequestNum:   i + 1,
				InputTokens:  80 + (i%25)*7,
				CachedTokens: 300 + (i%11)*13,
				OutputTokens: 40 + (i%19)*3,
				TTFT:         float64(100 + (i%30)*5),
				ResponseMs:   float64(800 + (i%40)*10),
			}
			if i%29 == 0 {
				r.TTFT = 0 // no first token reported
			}
			if i%53 == 52 {
				r.ResponseMs = 45000 + float64(i) // extreme latency outlier
			}
			if i%17 == 16 {
				r.IsError = true
			}
			out = append(out, r)
		}
		return out
	}

	armAName, armBName := "hbm-c28", "weka-c9"
	armARecords := genRecords(armAName, 210, 500) // 210 reqs, 500ms apart => ~104.5s
	armBRecords := genRecords(armBName, 260, 400) // 260 reqs, 400ms apart => ~103.6s

	armACfg := AutoBenchmarkConfig{Concurrency: 28}
	armBCfg := AutoBenchmarkConfig{Concurrency: 9}

	dir := t.TempDir()
	writeArmWithParams(t, dir, armAName, armACfg, base, armARecords)
	writeArmWithParams(t, dir, armBName, armBCfg, base, armBRecords)

	htmlPath, err := GenerateVisualization(dir, 0)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	html, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(html), "<script>")
	end := strings.Index(string(html), "</script>")
	if start < 0 || end < 0 {
		t.Fatal("script block not found in generated report")
	}
	script := string(html)[start+len("<script>") : end]

	probe := `
const perSeries = seriesStats();
const summary = DATA.map((s, i) => {
  const st = perSeries[i];
  const metrics = {};
  SUMMARY_METRICS.forEach(m => {
    let ratioPct = null;
    if (BASELINE_INDEX >= 0 && i !== BASELINE_INDEX) {
      ratioPct = summaryRatioPct(m.val(st), m.val(perSeries[BASELINE_INDEX]));
    }
    metrics[m.key] = { val: m.val(st), ratioPct: ratioPct };
  });
  return {
    name: s.name, isBaseline: i === BASELINE_INDEX,
    winSize: s._winSize, winConc: s._winConc, winConcSource: s._winConcSource,
    respP50: s._respP50, respP10: s._respP10, respP90: s._respP90,
    ttftP50: s._ttftP50, ttftP95: s._ttftP95, errBars: s._errBars,
    stats: st, metrics: metrics,
  };
});
console.log("CROSSCHECK_JSON:" + JSON.stringify({ baselineIndex: BASELINE_INDEX, series: summary }));
`
	jsPath := filepath.Join(dir, "crosscheck.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil {
		t.Fatalf("node cross-check script failed: %v\n%s", err, out)
	}
	var payload string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "CROSSCHECK_JSON:") {
			payload = strings.TrimPrefix(line, "CROSSCHECK_JSON:")
			break
		}
	}
	if payload == "" {
		t.Fatalf("no CROSSCHECK_JSON line in node output:\n%s", out)
	}
	var jsResult jsCrossCheck
	if err := json.Unmarshal([]byte(payload), &jsResult); err != nil {
		t.Fatalf("unmarshal JS result: %v\nraw: %s", err, payload)
	}
	byName := map[string]jsSeries{}
	for _, s := range jsResult.Series {
		byName[s.Name] = s
	}

	// --- Go half: the exact same fixture, through aggregate.go. ---
	goSeries := map[string]struct {
		agg   AggSeries
		stats AggSummaryStats
		norm  []AggRecord
	}{}
	for name, records := range map[string][]requestDataRecord{armAName: armARecords, armBName: armBRecords} {
		norm := NormalizeSeriesRecords(records)
		var conc int
		switch name {
		case armAName:
			conc = armACfg.Concurrency
		case armBName:
			conc = armBCfg.Concurrency
		}
		agg := ComputeAggSeries(norm, conc, 0)
		goSeries[name] = struct {
			agg   AggSeries
			stats AggSummaryStats
			norm  []AggRecord
		}{agg: agg, norm: norm}
	}
	tMin, tMax := GlobalTimeRange([][]AggRecord{goSeries[armAName].norm, goSeries[armBName].norm})
	names := []string{armAName, armBName} // order doesn't matter for BuildSummaryTable's math
	stats := make([]AggSummaryStats, len(names))
	for i, name := range names {
		e := goSeries[name]
		e.stats = ComputeSummaryStats(e.norm, tMin, tMax)
		goSeries[name] = e
		stats[i] = e.stats
	}
	rows := BuildSummaryTable(names, stats)
	rowByName := map[string]AggSummaryRow{}
	for _, r := range rows {
		rowByName[r.Name] = r
	}

	// --- Compare, tracking the max divergence per series/metric. Every key
	// this test ever tracks is seeded at 0 up front so a perfect (zero-
	// divergence) match still shows up in the report below instead of
	// silently vanishing from the map (Go map reads default to zero, so an
	// untouched key and a key whose max diff really is 0 are otherwise
	// indistinguishable).
	maxDiff := map[string]float64{
		"respP50": 0, "respP10": 0, "respP90": 0, "ttftP50": 0, "ttftP95": 0,
		"errBars.errRate": 0, "errBars.respAvg": 0,
		"stats.spanSec": 0, "stats.ttft50": 0, "stats.ttft95": 0,
	}
	for _, m := range AggSummaryMetrics {
		maxDiff["metric."+string(m.Key)] = 0
		maxDiff["ratioPct."+string(m.Key)] = 0
	}
	track := func(key string, diff float64) {
		if diff > maxDiff[key] {
			maxDiff[key] = diff
		}
	}
	const tol = 1e-6

	for _, name := range names {
		js, ok := byName[name]
		if !ok {
			t.Fatalf("JS result missing series %q", name)
		}
		g := goSeries[name].agg

		if g.WinSize != js.WinSize || g.WinConc != js.WinConc || g.WinConcSource != js.WinConcSource {
			t.Errorf("%s: window mismatch: go=(%d,%d,%q) js=(%d,%d,%q)",
				name, g.WinSize, g.WinConc, g.WinConcSource, js.WinSize, js.WinConc, js.WinConcSource)
		}

		comparePoints := func(label string, goPts []AggPoint, jsPts []jsPoint) {
			if len(goPts) != len(jsPts) {
				t.Errorf("%s/%s: point count go=%d js=%d", name, label, len(goPts), len(jsPts))
				return
			}
			for i := range goPts {
				if d := math.Abs(goPts[i].T - jsPts[i].T); d > tol {
					t.Errorf("%s/%s[%d]: T go=%v js=%v", name, label, i, goPts[i].T, jsPts[i].T)
				}
				track(label, math.Abs(goPts[i].V-jsPts[i].V))
			}
		}
		comparePoints("respP50", g.RespP50, js.RespP50)
		comparePoints("respP10", g.RespP10, js.RespP10)
		comparePoints("respP90", g.RespP90, js.RespP90)
		comparePoints("ttftP50", g.TTFTP50, js.TTFTP50)
		comparePoints("ttftP95", g.TTFTP95, js.TTFTP95)

		if len(g.ErrBars) != len(js.ErrBars) {
			t.Errorf("%s/errBars: count go=%d js=%d", name, len(g.ErrBars), len(js.ErrBars))
		} else {
			for i := range g.ErrBars {
				gb, jb := g.ErrBars[i], js.ErrBars[i]
				if math.Abs(gb.T-jb.T) > tol {
					t.Errorf("%s/errBars[%d]: T go=%v js=%v", name, i, gb.T, jb.T)
				}
				if gb.Errs != jb.Errs || gb.Total != jb.Total {
					t.Errorf("%s/errBars[%d]: counts go=(%d/%d) js=(%d/%d)", name, i, gb.Errs, gb.Total, jb.Errs, jb.Total)
				}
				track("errBars.errRate", math.Abs(gb.ErrRate-jb.ErrRate))
				track("errBars.respAvg", math.Abs(gb.RespAvg-jb.RespAvg))
			}
		}

		// Summary stats.
		gs := goSeries[name].stats
		if gs.OK != js.Stats.OK || gs.Err != js.Stats.Err || gs.Total != js.Stats.Total ||
			gs.InTok != js.Stats.InTok || gs.CaTok != js.Stats.CaTok || gs.OutTok != js.Stats.OutTok ||
			gs.Prompt != js.Stats.Prompt || gs.TTFTN != js.Stats.TTFTN {
			t.Errorf("%s: integer stats mismatch: go=%+v js=%+v", name, gs, js.Stats)
		}
		track("stats.spanSec", math.Abs(gs.SpanSec-js.Stats.SpanSec))
		track("stats.ttft50", math.Abs(gs.TTFT50-js.Stats.TTFT50))
		track("stats.ttft95", math.Abs(gs.TTFT95-js.Stats.TTFT95))

		// SUMMARY_METRICS values + baseline ratios.
		row := rowByName[name]
		if row.IsBaseline != js.IsBaseline {
			t.Errorf("%s: IsBaseline go=%v js=%v", name, row.IsBaseline, js.IsBaseline)
		}
		for _, m := range AggSummaryMetrics {
			key := string(m.Key)
			jm, ok := js.Metrics[key]
			if !ok {
				t.Fatalf("%s: JS result missing metric %q", name, key)
			}
			goVal := m.Val(gs)
			track("metric."+key, math.Abs(goVal-jm.Val))

			if row.IsBaseline || row.Ratios == nil {
				if jm.RatioPct.OK {
					t.Errorf("%s/%s: JS reports a ratio for a row Go says has none (%v)", name, key, jm.RatioPct.Val)
				}
				continue
			}
			mi := metricIndex(m.Key)
			goRatio := row.Ratios[mi]
			if goRatio.OK != jm.RatioPct.OK {
				t.Errorf("%s/%s: ratio gating go=%v js=%v", name, key, goRatio.OK, jm.RatioPct.OK)
				continue
			}
			if goRatio.OK {
				track("ratioPct."+key, math.Abs(goRatio.Pct-jm.RatioPct.Val))
			}
		}
	}

	t.Logf("max divergence measured (go vs js):")
	for _, key := range sortedKeys(maxDiff) {
		t.Logf("  %-16s %.3e", key, maxDiff[key])
	}

	// Downsampling has no JS counterpart (see DownsampleAggPoints), so it
	// isn't part of the parity check above -- log point counts on this same
	// fixture at a realistic interval so the reduction it buys is visible
	// against the full-resolution counts just verified.
	for _, name := range names {
		full := goSeries[name].agg.RespP50
		down := DownsampleAggPoints(full, 30000)
		t.Logf("%s: respP50 full=%d points, downsampled@30s=%d points", name, len(full), len(down))
	}
	for key, d := range maxDiff {
		if d > tol {
			t.Errorf("max divergence for %s = %v exceeds tolerance %v", key, d, tol)
		}
	}
}

func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j-1] > keys[j]; j-- {
			keys[j-1], keys[j] = keys[j], keys[j-1]
		}
	}
	return keys
}

// writeArmWithParams writes one arm's JSONL file: a run_params header
// (giving it a recorded concurrency, so ComputeAggSeries/computeDerived both
// pick it up as WinConcSourceRecorded) followed by its request records.
func writeArmWithParams(t *testing.T, dir, name string, cfg AutoBenchmarkConfig, writtenAt time.Time, records []requestDataRecord) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, name+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	if err := enc.Encode(buildRunParams(cfg, writtenAt)); err != nil {
		t.Fatal(err)
	}
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
}

// --- JS-side JSON shapes (unmarshalled from the node probe's output). ---

type jsPoint struct {
	T float64 `json:"t"`
	V float64 `json:"v"`
}

type jsErrBar struct {
	T       float64 `json:"t"`
	ErrRate float64 `json:"errRate"`
	Errs    int     `json:"errs"`
	Total   int     `json:"total"`
	RespAvg float64 `json:"respAvg"`
}

type jsStats struct {
	OK      int     `json:"ok"`
	Err     int     `json:"err"`
	Total   int     `json:"total"`
	InTok   int     `json:"inTok"`
	CaTok   int     `json:"caTok"`
	OutTok  int     `json:"outTok"`
	Prompt  int     `json:"prompt"`
	SpanSec float64 `json:"spanSec"`
	TTFT50  float64 `json:"ttft50"`
	TTFT95  float64 `json:"ttft95"`
	TTFTN   int     `json:"ttftN"`
}

// jsRatioPct unmarshals summaryRatioPct's own return value, which is either
// a number or "" (the JS-side "no ratio" sentinel) -- or JSON null, which
// this test's probe uses for "not compared at all" (the baseline row).
type jsRatioPct struct {
	Val float64
	OK  bool
}

func (r *jsRatioPct) UnmarshalJSON(b []byte) error {
	s := strings.TrimSpace(string(b))
	if s == "null" || s == `""` {
		r.Val, r.OK = 0, false
		return nil
	}
	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return err
	}
	r.Val, r.OK = f, true
	return nil
}

type jsMetricEntry struct {
	Val      float64    `json:"val"`
	RatioPct jsRatioPct `json:"ratioPct"`
}

type jsSeries struct {
	Name          string                   `json:"name"`
	IsBaseline    bool                     `json:"isBaseline"`
	WinSize       int                      `json:"winSize"`
	WinConc       int                      `json:"winConc"`
	WinConcSource string                   `json:"winConcSource"`
	RespP50       []jsPoint                `json:"respP50"`
	RespP10       []jsPoint                `json:"respP10"`
	RespP90       []jsPoint                `json:"respP90"`
	TTFTP50       []jsPoint                `json:"ttftP50"`
	TTFTP95       []jsPoint                `json:"ttftP95"`
	ErrBars       []jsErrBar               `json:"errBars"`
	Stats         jsStats                  `json:"stats"`
	Metrics       map[string]jsMetricEntry `json:"metrics"`
}

type jsCrossCheck struct {
	BaselineIndex int        `json:"baselineIndex"`
	Series        []jsSeries `json:"series"`
}
