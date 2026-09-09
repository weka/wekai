package benchmark

import (
	"math"
	"testing"
	"time"
)

func TestPercentileNearestRank(t *testing.T) {
	if v := Percentile(nil, 0.5); v != 0 {
		t.Errorf("empty => 0, got %v", v)
	}
	if v := Percentile([]float64{7}, 0.5); v != 7 {
		t.Errorf("singleton p50, got %v", v)
	}
	if v := Percentile([]float64{7}, 0.95); v != 7 {
		t.Errorf("singleton p95, got %v", v)
	}
	if v := Percentile([]float64{4, 1, 3, 2}, 0.5); v != 2 {
		t.Errorf("p50 of 1..4 = 2 (nearest-rank), got %v", v)
	}
	if v := Percentile([]float64{4, 1, 3, 2}, 0.95); v != 4 {
		t.Errorf("p95 of 1..4 = 4, got %v", v)
	}
	hundred := make([]float64, 100)
	for i := range hundred {
		hundred[i] = float64(i + 1)
	}
	if v := Percentile(hundred, 0.5); v != 50 {
		t.Errorf("p50 of 1..100 = 50, got %v", v)
	}
	if v := Percentile(hundred, 0.95); v != 95 {
		t.Errorf("p95 of 1..100 = 95, got %v", v)
	}
}

func TestPercentileSortedAgreesWithPercentile(t *testing.T) {
	raw := []float64{5, 1, 4, 2, 3}
	srt := append([]float64(nil), raw...)
	// sort ascending, same as Float64Array.prototype.sort() default.
	for i := 1; i < len(srt); i++ {
		for j := i; j > 0 && srt[j-1] > srt[j]; j-- {
			srt[j-1], srt[j] = srt[j], srt[j-1]
		}
	}
	for _, p := range []float64{0.5, 0.95, 0.0, 1.0} {
		if got, want := PercentileSorted(srt, p), Percentile(raw, p); got != want {
			t.Errorf("p=%v: PercentileSorted=%v, Percentile=%v", p, got, want)
		}
	}
	if v := PercentileSorted(nil, 0.5); v != 0 {
		t.Errorf("empty => 0, got %v", v)
	}
	if v := PercentileSorted([]float64{7}, 0.95); v != 7 {
		t.Errorf("single sample, got %v", v)
	}
}

func TestWindowSizePrecedence(t *testing.T) {
	cases := []struct {
		name                 string
		seriesConc, flagConc int
		wantSize, wantConc   int
		wantSource           string
	}{
		{"recorded wins over flag", 28, 4, 84, 28, WinConcSourceRecorded},
		{"flag wins when no recorded", 0, 4, 12, 4, WinConcSourceFlag},
		{"default when neither", 0, 0, AggDefaultWindowReqs, 0, WinConcSourceDefault},
		{"recorded 32 (28+4 hot)", 32, 0, 96, 32, WinConcSourceRecorded},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			size, conc, source := WindowSize(c.seriesConc, c.flagConc)
			if size != c.wantSize || conc != c.wantConc || source != c.wantSource {
				t.Errorf("got (%d, %d, %q), want (%d, %d, %q)", size, conc, source, c.wantSize, c.wantConc, c.wantSource)
			}
		})
	}
}

func TestNormalizeSeriesRecordsDeltaEncodesFromOwnMin(t *testing.T) {
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	recs := []requestDataRecord{
		{StartTime: base.Add(30 * time.Second), TTFT: 100, ResponseMs: 900, InputTokens: 1, OutputTokens: 2, CachedTokens: 3},
		{StartTime: base, TTFT: 50, ResponseMs: 800, InputTokens: 4, OutputTokens: 5, CachedTokens: 6},
		{StartTime: base.Add(60 * time.Second), IsError: true},
	}
	got := NormalizeSeriesRecords(recs)
	if len(got) != 3 {
		t.Fatalf("want 3 records, got %d", len(got))
	}
	// Order is preserved (not sorted) -- only the T delta changes.
	if got[0].T != 30000 || got[1].T != 0 || got[2].T != 60000 {
		t.Errorf("unexpected deltas: %v", got)
	}
	if got[1].In != 4 || got[1].Ca != 6 || got[1].Out != 5 || got[1].TTFT != 50 || got[1].Resp != 800 {
		t.Errorf("fields not carried through: %+v", got[1])
	}
	if !got[2].Err {
		t.Errorf("IsError not carried through")
	}
	if NormalizeSeriesRecords(nil) != nil {
		t.Errorf("empty input => nil")
	}
}

func TestGlobalTimeRange(t *testing.T) {
	tMin, tMax := GlobalTimeRange([][]AggRecord{
		{{T: 0}, {T: 500}},
		{{T: 0}, {T: 900}},
	})
	if tMin != 0 || tMax != 900 {
		t.Errorf("got (%v, %v), want (0, 900)", tMin, tMax)
	}
	if tMin, tMax := GlobalTimeRange(nil); tMin != 0 || tMax != 0 {
		t.Errorf("empty => (0,0), got (%v, %v)", tMin, tMax)
	}
	// Degenerate: every series collapses to one instant => +1000ms fallback.
	if tMin, tMax := GlobalTimeRange([][]AggRecord{{{T: 5}}}); tMin != 5 || tMax != 1005 {
		t.Errorf("degenerate => tMin+1000, got (%v, %v)", tMin, tMax)
	}
}

func TestComputeAggSeriesWindowAndAnchors(t *testing.T) {
	// 10 non-error records at t=0,10,...,90ms, concurrency forces winSize=3*... use small window directly via WindowSize(1,0) => winSize=3.
	var recs []AggRecord
	for i := 0; i < 10; i++ {
		recs = append(recs, AggRecord{T: float64(i * 10), Resp: float64(i + 1), TTFT: float64(i + 1)})
	}
	s := ComputeAggSeries(recs, 1, 0) // seriesConc=1 => winSize=3
	if s.WinSize != 3 || s.WinConc != 1 || s.WinConcSource != WinConcSourceRecorded {
		t.Fatalf("unexpected window: %+v", s)
	}
	if len(s.RespP50) != 10 {
		t.Fatalf("stride=1 (10 < 2000) => one anchor per record, got %d", len(s.RespP50))
	}
	// Anchor at i=2 (t=20): window is records 0,1,2 (resp 1,2,3) => nearest-rank
	// p50 index = ceil(3*0.5)-1 = 1 => value 2.
	if s.RespP50[2].V != 2 {
		t.Errorf("windowed p50 at 3rd anchor, want 2, got %v", s.RespP50[2].V)
	}
	// Anchor at i=9 (t=90): window is records 7,8,9 (resp 8,9,10) => p50 index
	// ceil(3*0.5)-1=1 => value 9.
	if s.RespP50[9].V != 9 {
		t.Errorf("windowed p50 at last anchor, want 9, got %v", s.RespP50[9].V)
	}
	// Early anchor at i=0: partial window of just record 0 (resp 1).
	if s.RespP50[0].V != 1 {
		t.Errorf("first anchor is a 1-record window, want 1, got %v", s.RespP50[0].V)
	}
}

func TestComputeAggSeriesErrorBarsAndRespAvg(t *testing.T) {
	// winSize=3 (seriesConc=1). byT = all 6 records including 2 errors.
	// Tumbling windows at i=2,5 (0-indexed, winSize-1=2, then +3=5).
	recs := []AggRecord{
		{T: 0, Resp: 10},
		{T: 10, Resp: 20},
		{T: 20, Resp: 30, Err: true}, // window [0,2]: 1 error
		{T: 30, Resp: 40},
		{T: 40, Resp: 50},
		{T: 50, Resp: 60, Err: true}, // window [3,5]: 1 error
	}
	s := ComputeAggSeries(recs, 1, 0)
	if len(s.ErrBars) != 2 {
		t.Fatalf("want 2 error bars (both tumbling windows have an error), got %d: %+v", len(s.ErrBars), s.ErrBars)
	}
	if s.ErrBars[0].Errs != 1 || s.ErrBars[0].Total != 3 || s.ErrBars[0].T != 20 {
		t.Errorf("first bar wrong: %+v", s.ErrBars[0])
	}
	if s.ErrBars[1].Errs != 1 || s.ErrBars[1].Total != 3 || s.ErrBars[1].T != 50 {
		t.Errorf("second bar wrong: %+v", s.ErrBars[1])
	}
	// RespAvg is the resp-p50 line's value at the first anchor whose T >= the
	// bar's own T; sorted (non-error) records are t=0,10,30,40, so p50 anchors
	// land at those same t's. The bar at t=20 should read the p50 anchor at
	// t=30 (first anchor >= 20).
	var wantAt30 float64
	for _, p := range s.RespP50 {
		if p.T == 30 {
			wantAt30 = p.V
		}
	}
	if s.ErrBars[0].RespAvg != wantAt30 {
		t.Errorf("errbar[0].RespAvg=%v, want the p50 anchor at t=30 (%v)", s.ErrBars[0].RespAvg, wantAt30)
	}
}

func TestDownsampleAggPoints(t *testing.T) {
	pts := []AggPoint{{T: 0, V: 1}, {T: 5, V: 2}, {T: 12, V: 3}, {T: 19, V: 4}, {T: 25, V: 5}}
	got := DownsampleAggPoints(pts, 10)
	// Buckets anchored at t0=0: [0,10) -> last is t=5 (v=2); [10,20) -> last is
	// t=19 (v=4); final point t=25 (v=5) always kept.
	want := []AggPoint{{T: 5, V: 2}, {T: 19, V: 4}, {T: 25, V: 5}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("point %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
	if got := DownsampleAggPoints(nil, 10); got != nil {
		t.Errorf("nil input => nil, got %v", got)
	}
	if got := DownsampleAggPoints(pts, 0); len(got) != len(pts) {
		t.Errorf("intervalMs<=0 => unchanged, got %v", got)
	}
}

func TestClassifyAndSortAliases(t *testing.T) {
	// ClassifyArmAlias mirrors classifyAlias(a) in visualize.go, which always
	// receives an already-lowercased alias (getAlias lowercases before
	// classifyAlias ever sees it) -- its regexes carry no /i flag, so a
	// mixed-case input here would not classify the way real callers see it.
	cases := map[string]string{
		"gpu": "gpu", "hbm-c28": "gpu", "hbm_run": "gpu",
		"weka-64r8w": "weka", "gds-run": "weka", "dram-cache": "dram",
		"custom-run": "other",
		// "no-hbm-here" is a faithful (if surprising) port: the JS regex
		// (?:^|[-_])hbm(?:$|[-_]) matches the "-hbm-" substring regardless of
		// what surrounds it, so this also classifies as the gpu baseline in
		// the original -- not a bug introduced by the port.
		"no-hbm-here": "gpu",
	}
	for alias, want := range cases {
		if got := ClassifyArmAlias(alias); got != want {
			t.Errorf("ClassifyArmAlias(%q) = %q, want %q", alias, got, want)
		}
	}
	if got := GetArmAlias("dynamic/http://x,alias=DS3H_weka-64r8w"); got != "ds3h_weka-64r8w" {
		t.Errorf("GetArmAlias with alias= param, got %q", got)
	}
	if got := GetArmAlias("plain-name"); got != "plain-name" {
		t.Errorf("GetArmAlias falls back to lowercased whole name, got %q", got)
	}

	names := []string{"weka-c28", "hbm-c28", "dram-run", "misc-arm"}
	order := SortArmIndices(names)
	var sorted []string
	for _, i := range order {
		sorted = append(sorted, names[i])
	}
	want := []string{"hbm-c28", "dram-run", "misc-arm", "weka-c28"}
	for i, w := range want {
		if sorted[i] != w {
			t.Errorf("sort order = %v, want %v", sorted, want)
			break
		}
	}
}

func TestFindBaselineIndex(t *testing.T) {
	if i := FindBaselineIndex([]string{"hbm-only"}); i != -1 {
		t.Errorf("single arm => -1, got %d", i)
	}
	if i := FindBaselineIndex([]string{"weka-a", "weka-b"}); i != -1 {
		t.Errorf("no hbm arm => -1, got %d", i)
	}
	if i := FindBaselineIndex([]string{"weka-a", "hbm-a"}); i != 1 {
		t.Errorf("hbm at index 1, got %d", i)
	}
}

func TestRatioPctGating(t *testing.T) {
	if pct, ok := RatioPct(300, 100); !ok || pct != 300 {
		t.Errorf("300/100 => 300%%, got (%v, %v)", pct, ok)
	}
	if pct, ok := RatioPct(1000, 4000); !ok || pct != 25 {
		t.Errorf("1000/4000 => 25%%, got (%v, %v)", pct, ok)
	}
	if _, ok := RatioPct(5, 0); ok {
		t.Errorf("zero base => no ratio")
	}
	if _, ok := RatioPct(5, -1); ok {
		t.Errorf("negative base => no ratio")
	}
	if _, ok := RatioPct(-1, 5); ok {
		t.Errorf("negative value => no ratio")
	}
	if _, ok := RatioPct(math.Inf(1), 5); ok {
		t.Errorf("+Inf value => no ratio")
	}
	if _, ok := RatioPct(math.NaN(), 5); ok {
		t.Errorf("NaN value => no ratio")
	}
	if pct, ok := RatioPct(0, 5); !ok || pct != 0 {
		t.Errorf("0/5 => 0%%, got (%v, %v)", pct, ok)
	}

	if good, ok := RatioIsImprovement(300, 100, BetterUp); !ok || !good {
		t.Errorf("more is better and ratio>1 => improvement")
	}
	if good, ok := RatioIsImprovement(1000, 4000, BetterDown); !ok || !good {
		t.Errorf("less is better and ratio<1 => improvement")
	}
	if good, ok := RatioIsImprovement(100, 100, BetterUp); !ok || good {
		t.Errorf("ratio==1 => not an improvement, but still gated ok")
	}
	if _, ok := RatioIsImprovement(5, 0, BetterUp); ok {
		t.Errorf("no baseline => not gated")
	}
}

func TestComputeSummaryStatsMatchesHandComputation(t *testing.T) {
	// 4 requests at t=0/10/20/30s, the last an error -- mirrors the JS
	// TestSummaryPanelJS fixture exactly, so the expected numbers can be
	// checked by hand against that test's documented derivation.
	recs := []AggRecord{
		{T: 0, TTFT: 150, Resp: 2000, In: 100, Ca: 400, Out: 50},
		{T: 10000, TTFT: 150, Resp: 2000, In: 100, Ca: 400, Out: 50},
		{T: 20000, TTFT: 150, Resp: 2000, In: 100, Ca: 400, Out: 50},
		{T: 30000, TTFT: 150, Resp: 2000, In: 200, Ca: 800, Out: 100, Err: true},
	}
	st := ComputeSummaryStats(recs, 0, 30000)
	if st.Prompt != 2500 || st.OutTok != 250 || st.OK != 3 || st.Err != 1 || st.Total != 4 {
		t.Fatalf("unexpected stats: %+v", st)
	}
	if st.SpanSec != 30 {
		t.Errorf("span = 30s, got %v", st.SpanSec)
	}
	if st.TTFT50 != 150 || st.TTFT95 != 150 || st.TTFTN != 3 {
		t.Errorf("ttft stats: %+v", st)
	}
	for _, m := range AggSummaryMetrics {
		v := m.Val(st)
		switch m.Key {
		case MetricInput:
			if v != 2500 {
				t.Errorf("input=%v", v)
			}
		case MetricOutput:
			if v != 250 {
				t.Errorf("output=%v", v)
			}
		case MetricReqs:
			if v != 3 {
				t.Errorf("reqs=%v", v)
			}
		case MetricInRate:
			if math.Abs(v-2500.0/30) > 1e-9 {
				t.Errorf("inrate=%v", v)
			}
		case MetricOutRate:
			if math.Abs(v-250.0/30) > 1e-9 {
				t.Errorf("outrate=%v", v)
			}
		case MetricTTFT50, MetricTTFT95:
			if v != 150 {
				t.Errorf("%s=%v", m.Key, v)
			}
		case MetricErr1k:
			if v != 250 {
				t.Errorf("err1k=%v, want 250 (1/4*1000)", v)
			}
		}
	}
}

func TestBuildSummaryTableRatios(t *testing.T) {
	// hbm does 4 requests of 500 prompt tokens, weka 12 of the same, both over
	// the same 36s extent -- mirrors TestBaselineRatiosJS exactly (300% on
	// every volume/rate metric, ttft inverted so weka's is 25% of hbm's).
	mk := func(n int, ttft float64) []AggRecord {
		var out []AggRecord
		step := 36000.0 / float64(n-1)
		for i := 0; i < n; i++ {
			out = append(out, AggRecord{T: float64(i) * step, TTFT: ttft, Resp: 900, In: 100, Ca: 400, Out: 50})
		}
		return out
	}
	names := []string{"hbm-c28", "weka-c28"}
	stats := []AggSummaryStats{
		ComputeSummaryStats(mk(4, 4000), 0, 36000),
		ComputeSummaryStats(mk(12, 1000), 0, 36000),
	}
	rows := BuildSummaryTable(names, stats)
	if !rows[0].IsBaseline || rows[1].IsBaseline {
		t.Fatalf("hbm (index 0) must be the baseline: %+v", rows)
	}
	if rows[0].Ratios != nil {
		t.Errorf("baseline row carries no ratios")
	}
	for _, key := range []AggMetricKey{MetricInput, MetricOutput, MetricReqs, MetricInRate, MetricOutRate} {
		mi := metricIndex(key)
		if r := rows[1].Ratios[mi]; !r.OK || r.Pct != 300 {
			t.Errorf("%s ratio = 300%%, got %+v", key, r)
		}
	}
	if r := rows[1].Ratios[metricIndex(MetricTTFT50)]; !r.OK || r.Pct != 25 {
		t.Errorf("ttft50 ratio = 25%%, got %+v", r)
	}
}

func metricIndex(key AggMetricKey) int {
	for i, m := range AggSummaryMetrics {
		if m.Key == key {
			return i
		}
	}
	return -1
}
