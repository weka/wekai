package benchmark

import (
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// This file is a server-side Go port of the aggregation math that
// benchmark/visualize.go's embedded JS performs client-side: the rolling-
// percentile series computeDerived() builds per arm, the per-scope summary
// windowStats()/seriesStats() computes, and the SUMMARY_METRICS/BASELINE_INDEX
// ratio-to-baseline derivation. It is pure data in, pure data out -- no
// html/template, no DOM -- so a server-side caller (and a --public,
// aggregate-only report) can compute the exact same numbers the interactive
// report shows without embedding a browser.
//
// It deliberately does NOT port the cache-mix / active-dataset overlays
// (mix/adt) -- those are a separate visual layer with their own geometry and
// aren't part of the rolling-percentile or summary math this phase covers.
//
// Every function here has a JS counterpart named in its doc comment; keep
// the two in lockstep -- benchmark/aggregate_crosscheck_test.go asserts they
// agree on a shared fixture and will fail loudly the moment they drift.

// AggDefaultWindowReqs mirrors DEFAULT_WINDOW_REQS in visualize.go: the
// rolling-percentile window used when neither the arm's recorded
// concurrency nor a caller-supplied concurrency says what to use.
const AggDefaultWindowReqs = 96

// aggTargetLinePoints mirrors TARGET_LINE_POINTS in visualize.go: the
// rolling-percentile line is anchored at roughly this many x-positions
// rather than one per request, by striding over the sorted, non-error
// records (see ComputeAggSeries).
const aggTargetLinePoints = 2000

// Window-concurrency-source labels, mirroring the exact strings
// computeDerived assigns to s._winConcSource.
const (
	WinConcSourceRecorded = "recorded"
	WinConcSourceFlag     = "--concurrency"
	WinConcSourceDefault  = "default"
)

// AggRecord is one request, time-normalized to elapsed milliseconds from its
// OWN series' first record -- exactly the "t" field of REC_FIELDS /
// vizRecord.T that visualize.go's computeDerived operates on. Build one
// slice per arm with NormalizeSeriesRecords.
type AggRecord struct {
	T    float64 // elapsed ms since this series' own first record
	TTFT float64 // ms; 0 means "no first token reported"
	Resp float64 // ms
	Err  bool
	In   int // net-of-cache input tokens
	Ca   int // server-cached prompt tokens
	Out  int // completion tokens
}

// NormalizeSeriesRecords converts one arm's raw requestDataRecord rows into
// AggRecords with T delta-encoded against the series' own minimum
// StartTime -- the same per-series normalization generateVisualization
// performs before embedding vizRecord.T (see seriesData.T0) and that
// visualize.go's DATA-normalization pass repeats for every series
// independently (each arm's x=0 is ITS OWN start, not a shared wall clock).
// The result is not sorted; ComputeAggSeries and ComputeSummaryStats sort as
// they need to. Returns nil for an empty input.
func NormalizeSeriesRecords(records []requestDataRecord) []AggRecord {
	if len(records) == 0 {
		return nil
	}
	var t0 float64
	have := false
	for _, r := range records {
		t := float64(r.StartTime.UnixMilli())
		if !have || t < t0 {
			t0 = t
			have = true
		}
	}
	out := make([]AggRecord, len(records))
	for i, r := range records {
		out[i] = AggRecord{
			T:    float64(r.StartTime.UnixMilli()) - t0,
			TTFT: r.TTFT,
			Resp: r.ResponseMs,
			Err:  r.IsError,
			In:   r.InputTokens,
			Ca:   r.CachedTokens,
			Out:  r.OutputTokens,
		}
	}
	return out
}

// GlobalTimeRange mirrors the request-record portion of the globalTMin/
// globalTMax scan in visualize.go (the report's overlay layers -- cache-mix
// segments and active-dataset samples -- also extend that range there; this
// phase does not port those overlays, so this considers request records
// only). The report's default, unzoomed summary scope is [tMin, tMax] from
// this function. When every series is empty, returns (0, 0). When every
// series (once normalized to its own t=0) has all its mass at one instant,
// mirrors `if (globalTMax === globalTMin) globalTMax = globalTMin + 1000`.
func GlobalTimeRange(allSeries [][]AggRecord) (tMin, tMax float64) {
	tMin, tMax = math.Inf(1), math.Inf(-1)
	for _, records := range allSeries {
		for _, r := range records {
			if r.T < tMin {
				tMin = r.T
			}
			if r.T > tMax {
				tMax = r.T
			}
		}
	}
	if math.IsInf(tMin, 1) {
		return 0, 0
	}
	if tMax == tMin {
		tMax = tMin + 1000
	}
	return tMin, tMax
}

// WindowSize mirrors the seriesConc/winSize/winConcSource computation at the
// top of computeDerived: an arm's own recorded concurrency (seriesConcurrency,
// e.g. runParamsRecord.effectiveConcurrency(); <= 0 means "not recorded")
// wins over the report-wide globalConcurrency (the --concurrency flag),
// which wins over AggDefaultWindowReqs. source takes one of the
// WinConcSource* constants, matching s._winConcSource's string exactly
// (note it is derived independently of the resulting seriesConc value --
// see the JS comment on _winConcSource for why).
func WindowSize(seriesConcurrency, globalConcurrency int) (winSize, winConc int, source string) {
	seriesConc := seriesConcurrency
	if seriesConc <= 0 {
		seriesConc = globalConcurrency
	}
	switch {
	case seriesConcurrency > 0:
		source = WinConcSourceRecorded
	case globalConcurrency > 0:
		source = WinConcSourceFlag
	default:
		source = WinConcSourceDefault
	}
	winConc = seriesConc
	if seriesConc > 0 {
		winSize = seriesConc * 3
	} else {
		winSize = AggDefaultWindowReqs
	}
	return winSize, winConc, source
}

// AggPoint is one x/y sample of a rolling-percentile line: T is elapsed ms
// (same axis as AggRecord.T), V is the metric value at that anchor.
type AggPoint struct {
	T float64
	V float64
}

// AggErrBar is one tumbling-window error-rate sample, mirroring one entry of
// s._errBars.
type AggErrBar struct {
	T       float64
	ErrRate float64
	Errs    int
	Total   int
	RespAvg float64
}

// AggSeries is the full-resolution derived data for one arm: the same
// request-anchored rolling-percentile lines and error bars that
// visualize.go's computeDerived() builds client-side, computed once here
// instead. WinSize/WinConc/WinConcSource mirror s._winSize/_winConc/
// _winConcSource.
type AggSeries struct {
	WinSize       int
	WinConc       int
	WinConcSource string

	RespP50 []AggPoint
	RespP10 []AggPoint
	RespP90 []AggPoint
	TTFTP50 []AggPoint
	TTFTP95 []AggPoint

	ErrBars []AggErrBar
}

// ComputeAggSeries mirrors computeDerived() in visualize.go exactly for one
// arm's records (already time-normalized -- see NormalizeSeriesRecords):
//
//   - byT: every record (errors included), sorted ascending by T -- the
//     population the error bars are drawn from.
//   - sorted: non-error records, sorted ascending by T -- the population the
//     percentile lines are drawn from.
//   - the rolling window is winSize REQUESTS (not a time span), computed by
//     WindowSize; each anchor's window is the trailing winSize records of
//     `sorted` ending at that anchor (inclusive), so early anchors see a
//     shorter, partial window (start clamps to 0).
//   - anchors are placed every `stride` records of `sorted` (stride = max(1,
//     len(sorted)/aggTargetLinePoints)), plus always the final record, so
//     the line reaches the true end of the run even when stride doesn't
//     land on it.
//   - each anchor's T is sorted[i].T; resp values are NOT filtered further
//     (errors are already excluded by `sorted`); TTFT values additionally
//     drop non-reporting requests (ttft <= 0) before taking a percentile,
//     and an anchor with no such requests in its window reports 0, not the
//     percentile of an empty set (which is also 0 by convention, but this
//     keeps the empty-window and the "computed as zero" cases from being
//     silently the same number for the wrong reason downstream).
//   - error bars are TUMBLING (non-overlapping) winSize-wide windows over
//     byT, sampled at i = winSize-1, 2*winSize-1, ...; a window is only
//     emitted when it contains at least one error, and its RespAvg is the
//     resp-p50 line's value at the first anchor whose T is >= the bar's T
//     (a two-pointer merge over both T-ascending sequences, matching the
//     JS's persistent avgIdx across iterations).
func ComputeAggSeries(records []AggRecord, seriesConcurrency, globalConcurrency int) AggSeries {
	winSize, winConc, source := WindowSize(seriesConcurrency, globalConcurrency)

	byT := append([]AggRecord(nil), records...)
	sort.SliceStable(byT, func(i, j int) bool { return byT[i].T < byT[j].T })

	sorted := make([]AggRecord, 0, len(records))
	for _, r := range records {
		if !r.Err {
			sorted = append(sorted, r)
		}
	}
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].T < sorted[j].T })

	out := AggSeries{WinSize: winSize, WinConc: winConc, WinConcSource: source}

	stride := len(sorted) / aggTargetLinePoints
	if stride < 1 {
		stride = 1
	}

	pushAnchor := func(i int) {
		start := i - winSize + 1
		if start < 0 {
			start = 0
		}
		win := sorted[start : i+1]
		resps := make([]float64, len(win))
		var ttfts []float64
		for k, r := range win {
			resps[k] = r.Resp
			if r.TTFT > 0 {
				ttfts = append(ttfts, r.TTFT)
			}
		}
		t := sorted[i].T
		out.RespP50 = append(out.RespP50, AggPoint{T: t, V: Percentile(resps, 0.5)})
		out.RespP10 = append(out.RespP10, AggPoint{T: t, V: Percentile(resps, 0.1)})
		out.RespP90 = append(out.RespP90, AggPoint{T: t, V: Percentile(resps, 0.9)})
		var ttft50, ttft95 float64
		if len(ttfts) > 0 {
			ttft50 = Percentile(ttfts, 0.5)
			ttft95 = Percentile(ttfts, 0.95)
		}
		out.TTFTP50 = append(out.TTFTP50, AggPoint{T: t, V: ttft50})
		out.TTFTP95 = append(out.TTFTP95, AggPoint{T: t, V: ttft95})
	}

	for i := 0; i < len(sorted); i += stride {
		pushAnchor(i)
	}
	if stride > 1 && len(sorted) > 0 && (len(sorted)-1)%stride != 0 {
		pushAnchor(len(sorted) - 1)
	}

	avgIdx := 0
	for i := winSize - 1; i < len(byT); i += winSize {
		start := i - winSize + 1
		if start < 0 {
			start = 0
		}
		errs, total := 0, 0
		for j := start; j <= i; j++ {
			total++
			if byT[j].Err {
				errs++
			}
		}
		if errs > 0 {
			t := byT[i].T
			for avgIdx < len(out.RespP50)-1 && out.RespP50[avgIdx].T < t {
				avgIdx++
			}
			var respAvg float64
			if len(out.RespP50) > 0 {
				idx := avgIdx
				if idx > len(out.RespP50)-1 {
					idx = len(out.RespP50) - 1
				}
				respAvg = out.RespP50[idx].V
			}
			out.ErrBars = append(out.ErrBars, AggErrBar{
				T: t, ErrRate: float64(errs) / float64(total), Errs: errs, Total: total, RespAvg: respAvg,
			})
		}
	}

	return out
}

// DownsampleAggPoints reduces points (sorted ascending by T, as every
// AggSeries line is) to at most one sample per intervalMs, by taking the
// LAST point at-or-before each bucket boundary -- a step-function
// reduction, so a downsampled line never shows a value that wasn't actually
// the most recent rolling-window computation at that time. Bucket
// boundaries are anchored at the first point's T (t0, t0+interval,
// t0+2*interval, ...), and the series' own final point is always kept, so a
// downsampled line still reaches the true end of the run -- the same
// guarantee ComputeAggSeries's own final-anchor rule makes for the
// full-resolution line. This has no JS counterpart: the report's own
// downsampling is the request-count stride inside ComputeAggSeries itself;
// this is the additional TIME-based reduction a static/exported chart needs
// on top of that. intervalMs <= 0 or an empty input returns points
// unchanged.
func DownsampleAggPoints(points []AggPoint, intervalMs float64) []AggPoint {
	if len(points) == 0 || intervalMs <= 0 {
		return points
	}
	t0 := points[0].T
	nextBoundary := t0 + intervalMs
	out := make([]AggPoint, 0, len(points))
	var last AggPoint
	haveLast := false
	for _, p := range points {
		for p.T >= nextBoundary {
			if haveLast {
				out = append(out, last)
			}
			nextBoundary += intervalMs
		}
		last = p
		haveLast = true
	}
	if haveLast {
		out = append(out, last)
	}
	return out
}

// Percentile returns the nearest-rank percentile (0 <= p <= 1) of arr,
// mirroring percentile(arr, p) in visualize.go exactly: sort ascending, take
// index ceil(len*p)-1, clamped to [0, len-1]; 0 for an empty input.
func Percentile(arr []float64, p float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	s := append([]float64(nil), arr...)
	sort.Float64s(s)
	idx := int(math.Ceil(float64(len(s))*p)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx > len(s)-1 {
		idx = len(s) - 1
	}
	return s[idx]
}

// PercentileSorted is Percentile for a slice already sorted ascending -- no
// copy, no re-sort -- mirroring percentileSorted(sorted, p) in visualize.go.
func PercentileSorted(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(len(sorted))*p)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx > len(sorted)-1 {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// AggSummaryStats is windowStats()'s per-arm output: everything the summary
// panel's eight SUMMARY_METRICS derive from, for one scope.
type AggSummaryStats struct {
	OK, Err, Total       int
	InTok, CaTok, OutTok int
	Prompt               int // InTok + CaTok
	SpanSec              float64
	TTFT50, TTFT95       float64
	TTFTN                int
}

// ComputeSummaryStats mirrors windowStats() in visualize.go over the given
// [tMin, tMax] window (both bounds inclusive -- a record is excluded only
// when r.T < tMin or r.T > tMax, matching the JS test exactly). Pass the
// pair from GlobalTimeRange for the report's default, unzoomed full-run
// scope.
//
// Token volumes (InTok/CaTok/OutTok/Prompt) include errored requests -- a
// failed request still cost its prompt -- while OK/TTFT50/TTFT95/TTFTN come
// from non-error requests only, and TTFT percentiles additionally require a
// reported first token (TTFT > 0), matching the plotted rolling-percentile
// lines exactly.
//
// SpanSec is the rate denominator: the window's own first->last RECORD
// extent (not the window's width), so a window that runs past where an arm's
// data ends doesn't dilute its rate with empty time it never ran through.
// Falls back to max((tMax-tMin)/1000, 1) when that extent is degenerate (0
// or 1 record in the window).
func ComputeSummaryStats(records []AggRecord, tMin, tMax float64) AggSummaryStats {
	var ok, errN int
	var inTok, caTok, outTok int
	tFirst, tLast := math.Inf(1), math.Inf(-1)
	var ttfts []float64
	for _, r := range records {
		if r.T < tMin || r.T > tMax {
			continue
		}
		if r.Err {
			errN++
		} else {
			ok++
			if r.TTFT > 0 {
				ttfts = append(ttfts, r.TTFT)
			}
		}
		inTok += r.In
		caTok += r.Ca
		outTok += r.Out
		if r.T < tFirst {
			tFirst = r.T
		}
		if r.T > tLast {
			tLast = r.T
		}
	}
	sort.Float64s(ttfts)
	total := ok + errN
	var spanSec float64
	if total > 1 {
		spanSec = (tLast - tFirst) / 1000
	}
	if spanSec <= 0 {
		w := (tMax - tMin) / 1000
		if w > 1 {
			spanSec = w
		} else {
			spanSec = 1
		}
	}
	return AggSummaryStats{
		OK: ok, Err: errN, Total: total,
		InTok: inTok, CaTok: caTok, OutTok: outTok, Prompt: inTok + caTok,
		SpanSec: spanSec,
		TTFT50:  PercentileSorted(ttfts, 0.5),
		TTFT95:  PercentileSorted(ttfts, 0.95),
		TTFTN:   len(ttfts),
	}
}

// AggMetricKey identifies one SUMMARY_METRICS column; values match the JS
// `key` strings exactly (also the CSV column-name lookup key there).
type AggMetricKey string

const (
	MetricInput   AggMetricKey = "in"
	MetricOutput  AggMetricKey = "out"
	MetricReqs    AggMetricKey = "reqs"
	MetricInRate  AggMetricKey = "inrate"
	MetricOutRate AggMetricKey = "outrate"
	MetricTTFT50  AggMetricKey = "ttft50"
	MetricTTFT95  AggMetricKey = "ttft95"
	MetricErr1k   AggMetricKey = "err1k"
	// MetricCachedInput is the share of prompt tokens served from cache
	// rather than recomputed: caTok / (inTok + caTok). Promoted from a
	// hover-only title on the Input cell (see valEl.title in visualize.go's
	// renderSummary) to a real column so it's visible without hovering, and
	// carried into the public report's precomputed summary too. Deliberately
	// a distinct metric/column from the per-REQUEST "cache hit rate" (share
	// of requests with server_cache_confirmed) -- this is a per-TOKEN share,
	// a different number, and the two must never be presented as the same
	// thing.
	MetricCachedInput AggMetricKey = "cached"
)

// AggMetricBetter says which direction of an AggMetric's value is an
// improvement, mirroring the JS `better: "up"/"down"` field.
type AggMetricBetter string

const (
	BetterUp   AggMetricBetter = "up"
	BetterDown AggMetricBetter = "down"
)

// AggMetric is one column of the summary table, mirroring one entry of
// SUMMARY_METRICS in visualize.go: Val is that entry's `val` function, Fmt
// its `fmt` function (the on-screen cell text), and Label its `label` hover
// tooltip text. Fmt/Label exist so a --public report can render the exact
// same column text and help-tooltip wording as the interactive one without
// re-deriving either client-side (the public build has no JS access to the
// per-request records Fmt/Val are computed from -- they run once, in Go,
// over the precomputed AggSummaryStats).
type AggMetric struct {
	Key    AggMetricKey
	Short  string
	Label  string
	Better AggMetricBetter
	Val    func(AggSummaryStats) float64
	Fmt    func(AggSummaryStats) string
}

// aggFmtTokens mirrors fmtTokens(v) in visualize.go exactly.
func aggFmtTokens(v float64) string {
	switch {
	case v >= 1e9:
		return strconv.FormatFloat(v/1e9, 'f', 1, 64) + "B"
	case v >= 1e6:
		return strconv.FormatFloat(v/1e6, 'f', 1, 64) + "M"
	case v >= 1e3:
		return strconv.FormatFloat(v/1e3, 'f', 1, 64) + "k"
	default:
		return strconv.FormatFloat(math.Round(v), 'f', 0, 64)
	}
}

// aggThousands mirrors n.toLocaleString() as visualize.go's Reqs column uses
// it: thousands-grouped with commas (en-US grouping; wekai has no i18n
// requirement, and this is the only place toLocaleString() is relied on for
// its grouping rather than just numeric formatting).
func aggThousands(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	if neg {
		s = "-" + s
	}
	return s
}

// aggFmtMs mirrors fmtMs(v) in visualize.go exactly: "-" for a non-positive
// value (a window with no completed request), seconds above 1000ms.
func aggFmtMs(v float64) string {
	if !(v > 0) {
		return "-"
	}
	if v >= 1000 {
		return strconv.FormatFloat(v/1000, 'f', 2, 64) + "s"
	}
	return strconv.FormatFloat(math.Round(v), 'f', 0, 64) + "ms"
}

// AggSummaryMetrics mirrors SUMMARY_METRICS in visualize.go: same nine
// columns, same order (the order IS part of the port -- it's the on-screen
// and CSV column order), same value/format formulas and hover-tooltip text.
var AggSummaryMetrics = []AggMetric{
	{MetricInput, "Input", "Prompt tokens processed, cached plus uncached. The metric KV offload improves directly.", BetterUp,
		func(st AggSummaryStats) float64 { return float64(st.Prompt) },
		func(st AggSummaryStats) string { return aggFmtTokens(float64(st.Prompt)) }},
	{MetricOutput, "Output", "Generated tokens. A property of the workload, not something offload changes.", BetterUp,
		func(st AggSummaryStats) float64 { return float64(st.OutTok) },
		func(st AggSummaryStats) string { return aggFmtTokens(float64(st.OutTok)) }},
	{MetricReqs, "Reqs", "Completed non-error requests in this window.", BetterUp,
		func(st AggSummaryStats) float64 { return float64(st.OK) },
		func(st AggSummaryStats) string { return aggThousands(st.OK) }},
	{MetricInRate, "In/s", "Input tokens per second. The headline throughput number.", BetterUp,
		func(st AggSummaryStats) float64 { return float64(st.Prompt) / st.SpanSec },
		func(st AggSummaryStats) string { return aggFmtTokens(float64(st.Prompt) / st.SpanSec) }},
	{MetricOutRate, "Out/s", "Output tokens per second.", BetterUp,
		func(st AggSummaryStats) float64 { return float64(st.OutTok) / st.SpanSec },
		func(st AggSummaryStats) string { return aggFmtTokens(float64(st.OutTok) / st.SpanSec) }},
	{MetricTTFT50, "TTFT50", "Median time to first token. Includes retry and queue wait.", BetterDown,
		func(st AggSummaryStats) float64 { return st.TTFT50 },
		func(st AggSummaryStats) string { return aggFmtMs(st.TTFT50) }},
	{MetricTTFT95, "TTFT95", "95th-percentile time to first token: the tail users notice.", BetterDown,
		func(st AggSummaryStats) float64 { return st.TTFT95 },
		func(st AggSummaryStats) string { return aggFmtMs(st.TTFT95) }},
	{MetricErr1k, "Err/1k", "Errors per 1,000 requests, so arms with different request counts compare.", BetterDown,
		func(st AggSummaryStats) float64 {
			if st.Total == 0 {
				return 0
			}
			return float64(st.Err) / float64(st.Total) * 1000
		},
		func(st AggSummaryStats) string {
			v := 0.0
			if st.Total > 0 {
				v = float64(st.Err) / float64(st.Total) * 1000
			}
			return strconv.FormatFloat(v, 'f', 1, 64)
		}},
	{MetricCachedInput, "Cached input", "Share of prompt tokens served from cache rather than recomputed.", BetterUp,
		func(st AggSummaryStats) float64 {
			if st.Prompt == 0 {
				return 0
			}
			return float64(st.CaTok) / float64(st.Prompt) * 100
		},
		func(st AggSummaryStats) string {
			if st.Prompt == 0 {
				return "-"
			}
			return strconv.FormatFloat(float64(st.CaTok)/float64(st.Prompt)*100, 'f', 1, 64) + "%"
		}},
}

// aggAliasRe mirrors the /alias[_=]([a-zA-Z0-9_-]+)/i regex getAlias() uses
// in visualize.go.
var aggAliasRe = regexp.MustCompile(`(?i)alias[_=]([a-zA-Z0-9_-]+)`)

// GetArmAlias mirrors getAlias(name) in visualize.go: the value of an
// embedded "alias_"/"alias=" parameter, lowercased, or the whole name
// lowercased when there is none.
func GetArmAlias(name string) string {
	if m := aggAliasRe.FindStringSubmatch(name); len(m) == 2 {
		return strings.ToLower(m[1])
	}
	return strings.ToLower(name)
}

// classifyAlias regexes mirror classifyAlias(a) in visualize.go.
var (
	aggHBMRe  = regexp.MustCompile(`(?:^|[-_])hbm(?:$|[-_])`)
	aggHBMPre = regexp.MustCompile(`^hbm`)
	aggGDSRe  = regexp.MustCompile(`(?:^|[-_])gds(?:$|[-_])`)
	aggWekaRe = regexp.MustCompile(`^weka`)
	aggDramRe = regexp.MustCompile(`(?:^|[-_])dram(?:$|[-_])`)
)

// ClassifyArmAlias mirrors classifyAlias(a) in visualize.go: "gpu" for the
// HBM/no-offload baseline convention, "weka"/"dram" for the named offload
// targets, "other" for anything else. Checks run in the same order as the
// JS (gpu-exact, then hbm, then gds, then weka-prefix, then dram).
func ClassifyArmAlias(alias string) string {
	switch {
	case alias == "gpu":
		return "gpu"
	case aggHBMRe.MatchString(alias) || aggHBMPre.MatchString(alias):
		return "gpu"
	case aggGDSRe.MatchString(alias):
		return "weka"
	case aggWekaRe.MatchString(alias):
		return "weka"
	case aggDramRe.MatchString(alias):
		return "dram"
	default:
		return "other"
	}
}

// sortKey mirrors sortKey(name) in visualize.go: gpu arms sort first, dram
// second, weka last, everything else alphabetically in between.
func sortKey(name string) string {
	switch ClassifyArmAlias(GetArmAlias(name)) {
	case "gpu":
		return "0_" + name
	case "dram":
		return "1_" + name
	case "weka":
		return "9_" + name
	default:
		return "5_" + name
	}
}

// SortArmIndices mirrors the sortedIndices computation that orders
// visualize.go's DATA: given arms' display names in caller order, returns
// the permutation (indices into names) that produces the report's display
// order.
func SortArmIndices(names []string) []int {
	idx := make([]int, len(names))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return sortKey(names[idx[a]]) < sortKey(names[idx[b]])
	})
	return idx
}

// FindBaselineIndex mirrors BASELINE_INDEX in visualize.go: with fewer than
// two arms there is nothing to be a ratio of (-1); otherwise the index (in
// the given order -- pass names already permuted by SortArmIndices to match
// the report's own display order exactly) of the first arm that classifies
// as the "gpu" (HBM/no-offload) baseline.
func FindBaselineIndex(names []string) int {
	if len(names) < 2 {
		return -1
	}
	for i, name := range names {
		if ClassifyArmAlias(GetArmAlias(name)) == "gpu" {
			return i
		}
	}
	return -1
}

// RatioPct mirrors summaryRatioPct(v, base) in visualize.go exactly (which
// in turn mirrors fmtRatio's own gating): the numeric percentage v is of
// base, or (0, false) when there is nothing to compare against -- a
// non-positive base, or a negative or non-finite v. Rounded to 2 decimal
// places (Math.round(v/base*10000)/100 in the JS).
func RatioPct(v, base float64) (pct float64, ok bool) {
	if !(base > 0) || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return 0, false
	}
	return math.Round(v/base*10000) / 100, true
}

// FmtRatioDisplay mirrors fmtRatio(v, base) in visualize.go exactly -- the
// ON-SCREEN ratio text (e.g. "300%", "25%", "0%"), which uses coarser
// rounding than RatioPct's CSV-precision percentage: whole percent at or
// above 10%, one decimal below that, "0%" for a zero-or-negative pct that
// still has something to compare against. "" under the same gate RatioPct
// uses (nothing to compare against).
func FmtRatioDisplay(v, base float64) string {
	if !(base > 0) || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return ""
	}
	pct := v / base * 100
	switch {
	case pct >= 10:
		return strconv.FormatFloat(math.Round(pct), 'f', 0, 64) + "%"
	case pct > 0:
		return strconv.FormatFloat(pct, 'f', 1, 64) + "%"
	default:
		return "0%"
	}
}

// RatioIsImprovement reports whether v is an improvement over base for a
// metric whose better direction is `better`, mirroring renderSummary's
// `good = m.better === "up" ? ratio > 1 : ratio < 1`. ok is false under the
// same gate as RatioPct (nothing to compare against); a ratio of exactly 1
// is not an improvement (matching the JS, which reserves the tint for
// ratio !== 1).
func RatioIsImprovement(v, base float64, better AggMetricBetter) (good, ok bool) {
	if _, gated := RatioPct(v, base); !gated {
		return false, false
	}
	ratio := v / base
	if ratio == 1 {
		return false, true
	}
	if better == BetterUp {
		return ratio > 1, true
	}
	return ratio < 1, true
}

// AggRatio is one metric's ratio-to-baseline for one non-baseline arm.
type AggRatio struct {
	Pct float64
	OK  bool
}

// AggSummaryRow is one arm's row in the summary table.
type AggSummaryRow struct {
	Name       string
	IsBaseline bool
	Stats      AggSummaryStats
	// Ratios is parallel to AggSummaryMetrics; nil when this row has nothing
	// to compare against (no baseline arm in the report, or this IS the
	// baseline row).
	Ratios []AggRatio
}

// BuildSummaryTable mirrors renderSummary()/buildSummaryRows() together: one
// row per arm in the given order, each metric's raw value via
// AggSummaryMetrics, and -- when names contains a recognisable HBM/no-offload
// baseline arm (see FindBaselineIndex) -- each non-baseline row's ratio to
// that baseline for every metric (see RatioPct). names and stats must be the
// same length and in the same (already display-sorted, if that matters to
// the caller) order.
func BuildSummaryTable(names []string, stats []AggSummaryStats) []AggSummaryRow {
	baseIdx := FindBaselineIndex(names)
	rows := make([]AggSummaryRow, len(names))
	var baseStats AggSummaryStats
	if baseIdx >= 0 {
		baseStats = stats[baseIdx]
	}
	for i, name := range names {
		row := AggSummaryRow{Name: name, IsBaseline: i == baseIdx, Stats: stats[i]}
		if baseIdx >= 0 && i != baseIdx {
			row.Ratios = make([]AggRatio, len(AggSummaryMetrics))
			for mi, m := range AggSummaryMetrics {
				pct, ok := RatioPct(m.Val(stats[i]), m.Val(baseStats))
				row.Ratios[mi] = AggRatio{Pct: pct, OK: ok}
			}
		}
		rows[i] = row
	}
	return rows
}
