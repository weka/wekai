package benchmark

import (
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// This file is the --public counterpart of visualize.go: it produces a
// self-contained HTML report from the same per-arm .jsonl directories, but
// with the PRINCIPLE that per-request data must be ABSENT from the emitted
// file, not merely hidden by the UI. Concretely:
//
//   - Every number embedded in the public report is either a downsampled
//     aggregate point (rolling-percentile line, error rate, cache-mix
//     segment, active-dataset sample, cumulative-ingest sample) or a
//     whole-run summary statistic -- built via aggregate.go's ported math
//     (NormalizeSeriesRecords / ComputeAggSeries / ComputeSummaryStats /
//     DownsampleAggPoints / BuildSummaryTable), never a per-request row.
//   - There is no series_num, request_num, series_guid, run_id, endpoint
//     URL, or raw model-spec string anywhere in the payload. Arm names come
//     from --labels when given, else the same clean alias resolution
//     generateVisualization already uses (resolveRecordsAlias), which never
//     surfaces the raw "dynamic/http://...,alias=..." spec string.
//   - The summary table is precomputed once, server-side, over the FULL run
//     ([[GlobalTimeRange]]) and rendered as static HTML -- it does not
//     reprice on zoom, unlike the interactive report's panel.
//
// See DefaultPublicInterval, GeneratePublicVisualization and
// GenerateVisualizationMergedPublic for the entry points, and
// benchmark_options.go / benchmark_commands.go for the --public /
// --public-interval CLI surface.

// DefaultPublicInterval is the downsample cadence used when --public-interval
// is not given.
const DefaultPublicInterval = 30 * time.Second

// GeneratePublicVisualization is the --public counterpart of
// GenerateVisualizationWithOptions: same source directory of .jsonl files,
// but the emitted report ("report-public.html", never "report.html" -- see
// nextVersionedPath -- so a public build can never overwrite or be mistaken
// for the internal one) carries no per-request array, only aggregate series
// downsampled to interval and a whole-run summary table. interval <= 0 uses
// DefaultPublicInterval.
func GeneratePublicVisualization(dir string, concurrency int, maxElapsed, interval time.Duration) (string, error) {
	return generateVisualizationPublic(dir, concurrency, false, maxElapsed, interval)
}

// GenerateVisualizationMergedPublic is the --public counterpart of
// GenerateVisualizationMerged: it merges the same source directories (same
// per-source JSONL/CSV outputs on disk, via prepareMergedSources -- those are
// unchanged, separate files, not embedded in the report), then emits the
// public report from the merged directory instead of the interactive one.
func GenerateVisualizationMergedPublic(dirs []string, labels []string, outputDir string, concurrency int, maxElapsed, interval time.Duration) (string, error) {
	outDir, err := prepareMergedSources(dirs, labels, outputDir, maxElapsed)
	if err != nil {
		return "", err
	}
	return generateVisualizationPublic(outDir, concurrency, len(labels) > 0, 0, interval)
}

// publicPoint mirrors AggPoint, emitted as a positional [t,v] pair -- same
// rationale as vizRecord.MarshalJSON: at report scale (thousands of points
// across several lines/arms) repeating "t"/"v" key strings per point would
// dwarf the actual data.
type publicPoint struct {
	T, V float64
}

func (p publicPoint) MarshalJSON() ([]byte, error) {
	return json.Marshal([2]float64{p.T, p.V})
}

// publicErrBar mirrors AggErrBar, emitted as [t, errRate, errs, total, respAvg].
type publicErrBar struct {
	T, ErrRate, RespAvg float64
	Errs, Total         int
}

func (b publicErrBar) MarshalJSON() ([]byte, error) {
	return json.Marshal([5]float64{b.T, b.ErrRate, float64(b.Errs), float64(b.Total), b.RespAvg})
}

// publicMixSeg mirrors vizSampleSegment (downsampled -- see
// downsampleMixSegments), emitted as [t0, t1, compute, localCache, externalCache].
type publicMixSeg struct{ T0, T1, C, Lc, Ec float64 }

func (m publicMixSeg) MarshalJSON() ([]byte, error) {
	return json.Marshal([5]float64{m.T0, m.T1, m.C, m.Lc, m.Ec})
}

// publicAdtPoint mirrors vizAdtPoint (downsampled), emitted as [t, tokens, series].
type publicAdtPoint struct {
	T, V float64
	S    int
}

func (p publicAdtPoint) MarshalJSON() ([]byte, error) {
	return json.Marshal([3]float64{p.T, p.V, float64(p.S)})
}

// publicCumPoint is a cumulative ingest/output sample (downsampled), emitted
// as [t, cumulativeIngestTokens, cumulativeOutputTokens]. Has no per-request
// counterpart in the interactive report -- there, the equivalent curve is
// computed client-side from every record's running sum (computeDerived's
// _cumTokens/_cumOutTokens); here it is precomputed server-side from the same
// aggregate.AggRecord slice everything else in this file uses.
type publicCumPoint struct{ T, CumIn, CumOut float64 }

func (p publicCumPoint) MarshalJSON() ([]byte, error) {
	return json.Marshal([3]float64{p.T, p.CumIn, p.CumOut})
}

// publicSeries is one arm's data as embedded in a report-public.html's
// PUBLIC_DATA. Color is precomputed server-side (see publicSeriesColors) so
// the client script never needs the classify/palette logic the interactive
// report carries -- one less thing to keep in the "no internal specifics"
// payload, and it guarantees the legend dot and the summary table's dot
// agree because both come from the same computed value.
type publicSeries struct {
	Name          string           `json:"name"`
	Color         string           `json:"color"`
	OK            int              `json:"ok"`
	Err           int              `json:"err"`
	ParamsSummary string           `json:"params,omitempty"`
	RespP50       []publicPoint    `json:"respP50"`
	TTFTP50       []publicPoint    `json:"ttftP50"`
	TTFTP95       []publicPoint    `json:"ttftP95"`
	ErrBars       []publicErrBar   `json:"errBars,omitempty"`
	Mix           []publicMixSeg   `json:"mix,omitempty"`
	Adt           []publicAdtPoint `json:"adt,omitempty"`
	Cum           []publicCumPoint `json:"cum,omitempty"`
}

// publicSummaryCell is one metric's rendered cell in the precomputed summary
// table: Val is the on-screen text (AggMetric.Fmt), Ratio the on-screen
// ratio-to-baseline text (FmtRatioDisplay), Tint "up"/"down"/"" (matching
// RatioIsImprovement).
type publicSummaryCell struct {
	Val   string
	Ratio string
	Tint  string
}

// publicSummaryRow is one arm's row: everything buildPublicSummaryRows needs
// to render the static <table> the public report ships instead of the
// interactive panel's JS-built one.
type publicSummaryRow struct {
	Name          string
	Color         string
	IsBaseline    bool
	ParamsSummary string
	WindowInfo    string
	Cells         []publicSummaryCell
}

// publicSummaryColumn is one column header: text plus its help-tooltip.
type publicSummaryColumn struct {
	Short string
	Label string
}

// generateVisualizationPublic is the implementation behind
// GeneratePublicVisualization / GenerateVisualizationMergedPublic.
// keepFileNames mirrors generateVisualization's own parameter: pin each
// arm's displayed name to its .jsonl basename (set by the merged path when
// explicit --labels were given) instead of re-resolving a record alias.
func generateVisualizationPublic(dir string, concurrency int, keepFileNames bool, maxElapsed, interval time.Duration) (string, error) {
	if interval <= 0 {
		interval = DefaultPublicInterval
	}
	intervalMs := float64(interval.Milliseconds())

	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		return "", fmt.Errorf("glob jsonl files: %w", err)
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no .jsonl files found in %s", dir)
	}
	sort.Strings(files)

	type armSource struct {
		name      string
		records   []requestDataRecord
		mix       []vizSampleSegment
		adt       []vizAdtPoint
		params    runParamsRecord
		hasParams bool
	}
	var arms []armSource
	for _, f := range files {
		name := TrimJSONLExt(f)
		records, samples, params, hasParams, err := readJSONLFileWithParams(f)
		if err != nil {
			return "", fmt.Errorf("read %s: %w", f, err)
		}
		records, samples = truncateToElapsed(records, samples, maxElapsed)
		if !keepFileNames {
			if alias := resolveRecordsAlias(records); alias != "" {
				name = alias
			}
		}
		mix, adt := buildSampleViz(samples)
		arms = append(arms, armSource{name: name, records: records, mix: mix, adt: adt, params: params, hasParams: hasParams})
	}

	names := make([]string, len(arms))
	for i, a := range arms {
		names[i] = a.name
	}
	order := SortArmIndices(names) // same gpu-first/dram/weka-last display order as the interactive report

	// One shared origin per arm -- see computeArmOrigin -- computed BEFORE
	// normalizing records, so allNorm's T values (below) can be shifted onto
	// that same axis instead of staying anchored at NormalizeSeriesRecords'
	// own independent per-arm zero.
	origin := make([]float64, len(arms))
	recordsShift := make([]float64, len(arms))
	mixIn := make([][]vizSampleSegment, len(arms))
	adtIn := make([][]vizAdtPoint, len(arms))
	for i, a := range arms {
		origin[i], recordsShift[i], mixIn[i], adtIn[i] = computeArmOrigin(a.name, a.records, a.mix, a.adt)
	}

	allNorm := make([][]AggRecord, len(arms))
	for i, a := range arms {
		norm := NormalizeSeriesRecords(a.records)
		if shift := recordsShift[i]; shift != 0 {
			for j := range norm {
				norm[j].T += shift
			}
		}
		allNorm[i] = norm
	}
	tMin, tMax := GlobalTimeRange(allNorm)

	orderedNames := make([]string, len(order))
	for i, idx := range order {
		orderedNames[i] = arms[idx].name
	}
	colors := publicSeriesColors(orderedNames)

	series := make([]publicSeries, len(order))
	summaryStats := make([]AggSummaryStats, len(order))
	winSizes := make([]int, len(order))
	winSources := make([]string, len(order))
	for i, idx := range order {
		a := arms[idx]
		norm := allNorm[idx]
		conc := 0
		if a.hasParams {
			conc = a.params.effectiveConcurrency()
		}
		agg := ComputeAggSeries(norm, conc, concurrency)
		st := ComputeSummaryStats(norm, tMin, tMax)
		summaryStats[i] = st
		winSizes[i] = agg.WinSize
		winSources[i] = agg.WinConcSource

		var paramsSummary string
		if a.hasParams {
			paramsSummary = a.params.summaryLine()
		}
		series[i] = publicSeries{
			Name:          a.name,
			Color:         colors[i],
			OK:            st.OK,
			Err:           st.Err,
			ParamsSummary: paramsSummary,
			RespP50:       toPublicPoints(DownsampleAggPoints(agg.RespP50, intervalMs)),
			TTFTP50:       toPublicPoints(DownsampleAggPoints(agg.TTFTP50, intervalMs)),
			TTFTP95:       toPublicPoints(DownsampleAggPoints(agg.TTFTP95, intervalMs)),
			ErrBars:       buildIntervalErrorBars(norm, intervalMs, DownsampleAggPoints(agg.RespP50, intervalMs)),
			Mix:           downsampleMixSegments(mixIn[idx], origin[idx], intervalMs),
			Adt:           downsampleAdtPoints(adtIn[idx], origin[idx], intervalMs),
			Cum:           buildCumulativePoints(norm, intervalMs),
		}
	}

	// Extend the time range with mix/adt tails past the last request, same
	// as the interactive report's globalTMin/globalTMax scan.
	for _, s := range series {
		for _, m := range s.Mix {
			if m.T1 > tMax {
				tMax = m.T1
			}
		}
		for _, p := range s.Adt {
			if p.T > tMax {
				tMax = p.T
			}
		}
	}
	if tMax == tMin {
		tMax = tMin + 1000
	}

	hasCacheMix := false
	for _, s := range series {
		if len(s.Mix) > 0 || len(s.Adt) > 0 {
			hasCacheMix = true
		}
	}

	rows := BuildSummaryTable(orderedNames, summaryStats)
	summaryRows := make([]publicSummaryRow, len(rows))
	for i, r := range rows {
		cells := make([]publicSummaryCell, len(AggSummaryMetrics))
		for mi, m := range AggSummaryMetrics {
			cell := publicSummaryCell{Val: m.Fmt(summaryStats[i])}
			if r.Ratios != nil {
				base := summaryStats[FindBaselineIndex(orderedNames)]
				cell.Ratio = FmtRatioDisplay(m.Val(summaryStats[i]), m.Val(base))
				if good, ok := RatioIsImprovement(m.Val(summaryStats[i]), m.Val(base), m.Better); ok {
					if good {
						cell.Tint = "up"
					} else {
						cell.Tint = "down"
					}
				}
			}
			cells[mi] = cell
		}
		summaryRows[i] = publicSummaryRow{
			Name:          r.Name,
			Color:         series[i].Color,
			IsBaseline:    r.IsBaseline,
			ParamsSummary: series[i].ParamsSummary,
			WindowInfo:    fmt.Sprintf("rolling window: %d reqs (%s)", winSizes[i], winSources[i]),
			Cells:         cells,
		}
	}
	summaryColumns := make([]publicSummaryColumn, len(AggSummaryMetrics))
	for i, m := range AggSummaryMetrics {
		summaryColumns[i] = publicSummaryColumn{Short: m.Short, Label: m.Label}
	}
	hasBaseline := FindBaselineIndex(orderedNames) >= 0

	seriesJSON, err := json.Marshal(series)
	if err != nil {
		return "", fmt.Errorf("marshal public series data: %w", err)
	}

	runLengthSec := (tMax - tMin) / 1000
	footer := fmt.Sprintf(
		"Run length %s. Downsampled to %s intervals. Per-request detail is not included in this file. Generated %s.",
		formatDurationLabel(runLengthSec), interval.String(), time.Now().UTC().Format("2006-01-02 15:04 UTC"))

	htmlPath := nextVersionedPath(dir, "report-public", ".html")
	out, err := os.Create(htmlPath)
	if err != nil {
		return "", fmt.Errorf("create html file: %w", err)
	}
	defer out.Close()

	data := struct {
		Data         template.JS
		Concurrency  template.JS
		GlobalTMin   template.JS
		GlobalTMax   template.JS
		IntervalMs   template.JS
		HasCacheMix  bool
		HasBaseline  bool
		SummaryRows  []publicSummaryRow
		SummaryCols  []publicSummaryColumn
		SummaryLabel string
		Footer       string
	}{
		Data:         template.JS(seriesJSON),
		Concurrency:  template.JS(fmt.Sprintf("%d", concurrency)),
		GlobalTMin:   template.JS(fmt.Sprintf("%v", tMin)),
		GlobalTMax:   template.JS(fmt.Sprintf("%v", tMax)),
		IntervalMs:   template.JS(fmt.Sprintf("%v", intervalMs)),
		HasCacheMix:  hasCacheMix,
		HasBaseline:  hasBaseline,
		SummaryRows:  summaryRows,
		SummaryCols:  summaryColumns,
		SummaryLabel: "full run (" + formatDurationLabel(runLengthSec) + ")",
		Footer:       footer,
	}
	if err := vizPublicTemplate.Execute(out, data); err != nil {
		return "", fmt.Errorf("execute public template: %w", err)
	}
	return htmlPath, nil
}

// TrimJSONLExt returns f's basename with the .jsonl extension removed --
// the same fallback display name generateVisualization uses before trying to
// resolve a cleaner alias.
func TrimJSONLExt(f string) string {
	base := filepath.Base(f)
	return base[:len(base)-len(filepath.Ext(base))]
}

// roundN rounds v to the given number of decimal places -- used throughout
// this file to keep the embedded JSON small: a percentile line's value or an
// error rate carries no meaningful information past 1-4 decimals, but
// float64's default JSON encoding (encoding/json prints the shortest string
// that round-trips exactly, which for an irrational-looking division result
// like 43/1290 is 15-17 digits) would otherwise spend most of the public
// file's bytes on precision no chart pixel can show.
func roundN(v float64, decimals int) float64 {
	if !isFiniteFloat(v) {
		return 0
	}
	scale := math.Pow(10, float64(decimals))
	return math.Round(v*scale) / scale
}

func isFiniteFloat(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// toPublicPoints converts an []AggPoint to the positional-JSON []publicPoint
// wrapper. V is rounded to 1 decimal place (see roundN) -- plenty for a
// latency-in-milliseconds line at 30s-interval resolution.
func toPublicPoints(pts []AggPoint) []publicPoint {
	if len(pts) == 0 {
		return nil
	}
	out := make([]publicPoint, len(pts))
	for i, p := range pts {
		out[i] = publicPoint{T: p.T, V: roundN(p.V, 1)}
	}
	return out
}

// computeArmOrigin computes the ONE shared origin (absolute epoch ms) for
// every layer of one arm's chart -- respP50/ttft/errBars/cum (all derived
// from records, see NormalizeSeriesRecords) as well as mix/adt (persisted
// in absolute epoch ms -- see vizSampleSegment/vizAdtPoint) -- plus the
// additive recordsShift that the caller must add to every already-normalized
// AggRecord.T (NormalizeSeriesRecords zeroes each arm at its OWN records'
// first StartTime, independent of mix/adt) so those series land on that
// same shared axis instead of each layer silently keeping its own separate
// zero.
//
// The origin is the EARLIEST timestamp across every source for the arm
// (records AND the mix/adt sampler), chosen so no series' shifted T ever
// goes negative: if the sampler's first read happens to precede the first
// request (its own async startup, not driven by the request path), that
// surfaces as a small positive recordsShift on the request-derived series
// rather than a negative mix/adt timestamp. mix/adt come from a separate
// collector (the vLLM /metrics sampler goroutine, see buildSampleViz) with
// its own clock reading, not the request path's -- in production the two
// are the same process's time.Now() and differ only by tens of
// milliseconds, and that small, real lead/lag should be VISIBLE on the
// shared axis (e.g. the first mix/adt sample landing at t=32ms, not t=0),
// not absorbed away by giving every layer its own independent origin. An
// EARLIER version of this function did exactly that (derived mix/adt's
// origin from mix/adt alone, respP50/ttft/errBars/cum from records alone),
// which fixed a symptom -- a blank chart and a "496039h" run label -- while
// introducing that same misalignment on well-formed input: any real gap
// between the two clocks (32ms here; a sampler that only started 60s into
// the run in a worse case) would silently vanish, so the cache-mix bands
// and the latency lines they are read against would sit on two different
// axes with nothing visibly wrong. The actual root cause of the "496039h"
// symptom was a malformed test fixture (its generator double-counted the
// series t0 when writing vllm_metrics_sample.ts, landing samples ~57 years
// in the future) -- see the corruption guard below, which now catches that
// case directly instead of via a per-layer origin.
//
// A mix/adt timestamp more than one run-window's duration away from that
// arm's own [min,max] request span is corrupt input, not a legitimate
// early or late sample. It is reported to stderr, naming armName and the
// offending timestamp, and DROPPED before the min-scan below, rather than
// being allowed to drag the shared origin (and therefore the whole chart)
// off axis -- silently re-basing around it is what turned corrupt input
// into a plausible-looking chart in the first place. When the arm's own
// request span is degenerate (a single request, or none at all) there is
// no meaningful duration to test against, so the check is skipped rather
// than flagging every mix/adt point as corrupt by default.
func computeArmOrigin(armName string, records []requestDataRecord, mix []vizSampleSegment, adt []vizAdtPoint) (origin, recordsShift float64, filteredMix []vizSampleSegment, filteredAdt []vizAdtPoint) {
	var recMin, recMax float64
	haveRec := false
	for _, r := range records {
		t := float64(r.StartTime.UnixMilli())
		if !haveRec || t < recMin {
			recMin = t
		}
		if !haveRec || t > recMax {
			recMax = t
		}
		haveRec = true
	}

	duration := recMax - recMin
	checkCorruption := haveRec && duration > 0
	inWindow := func(t float64) bool {
		if !checkCorruption {
			return true
		}
		return t >= recMin-duration && t <= recMax+duration
	}

	for _, m := range mix {
		if inWindow(m.T0) && inWindow(m.T1) {
			filteredMix = append(filteredMix, m)
		} else {
			fmt.Fprintf(os.Stderr,
				"warning: public report: arm %q cache-mix sample at t=%.0fms is outside the run's request window [%.0f,%.0f]ms (+/- %.0fms) -- dropping as corrupt input\n",
				armName, m.T0, recMin, recMax, duration)
		}
	}
	for _, p := range adt {
		if inWindow(p.T) {
			filteredAdt = append(filteredAdt, p)
		} else {
			fmt.Fprintf(os.Stderr,
				"warning: public report: arm %q active-dataset sample at t=%.0fms is outside the run's request window [%.0f,%.0f]ms (+/- %.0fms) -- dropping as corrupt input\n",
				armName, p.T, recMin, recMax, duration)
		}
	}

	haveOrigin := false
	if haveRec {
		origin, haveOrigin = recMin, true
	}
	for _, m := range filteredMix {
		if !haveOrigin || m.T0 < origin {
			origin, haveOrigin = m.T0, true
		}
	}
	for _, p := range filteredAdt {
		if !haveOrigin || p.T < origin {
			origin, haveOrigin = p.T, true
		}
	}
	if !haveOrigin {
		return 0, 0, filteredMix, filteredAdt
	}
	if haveRec {
		recordsShift = recMin - origin
	}
	return origin, recordsShift, filteredMix, filteredAdt
}

// buildIntervalErrorBars computes a wall-clock-interval error rate directly
// from every record (errors included) -- deliberately NOT the request-count
// TUMBLING window AggErrBar uses (that window is sized from the run's
// concurrency, which would make the public report's error-rate cadence
// depend on a number the file no longer carries any other trace of); a
// bucket with zero errors is omitted, matching AggErrBar's own "only emit a
// window that saw at least one error" rule. respP50Down anchors each bar's
// vertical position at the response line's value at or after the bucket's
// time, mirroring ComputeAggSeries's avgIdx two-pointer merge.
func buildIntervalErrorBars(records []AggRecord, intervalMs float64, respP50Down []AggPoint) []publicErrBar {
	if intervalMs <= 0 || len(records) == 0 {
		return nil
	}
	type bucket struct{ errs, total int }
	buckets := map[int64]*bucket{}
	for _, r := range records {
		idx := int64(math.Floor(r.T / intervalMs))
		b := buckets[idx]
		if b == nil {
			b = &bucket{}
			buckets[idx] = b
		}
		b.total++
		if r.Err {
			b.errs++
		}
	}
	idxs := make([]int64, 0, len(buckets))
	for idx := range buckets {
		idxs = append(idxs, idx)
	}
	sort.Slice(idxs, func(i, j int) bool { return idxs[i] < idxs[j] })
	var out []publicErrBar
	pi := 0
	for _, idx := range idxs {
		b := buckets[idx]
		if b.errs == 0 {
			continue
		}
		t := float64(idx) * intervalMs
		for pi < len(respP50Down)-1 && respP50Down[pi].T < t {
			pi++
		}
		var respAvg float64
		if len(respP50Down) > 0 {
			j := pi
			if j > len(respP50Down)-1 {
				j = len(respP50Down) - 1
			}
			respAvg = respP50Down[j].V
		}
		out = append(out, publicErrBar{T: t, ErrRate: roundN(float64(b.errs)/float64(b.total), 4), Errs: b.errs, Total: b.total, RespAvg: roundN(respAvg, 1)})
	}
	return out
}

// downsampleMixSegments coalesces raw per-sample-interval segments (typically
// spaced at the sampler's own cadence, often finer than intervalMs) into
// buckets whose START is at least intervalMs apart, summing each source's
// token delta and extending the bucket's end -- so total ingested volume is
// preserved exactly (every raw segment counted once) while the segment count
// drops to the public cadence. t0 shifts from absolute epoch ms (as stored in
// vizSampleSegment) to the same per-arm-relative axis as everything else here
// (see computeArmOrigin).
func downsampleMixSegments(segs []vizSampleSegment, t0, intervalMs float64) []publicMixSeg {
	if len(segs) == 0 {
		return nil
	}
	sorted := append([]vizSampleSegment(nil), segs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].T0 < sorted[j].T0 })
	var out []publicMixSeg
	var cur *publicMixSeg
	for _, s := range sorted {
		rt0, rt1 := s.T0-t0, s.T1-t0
		if cur == nil || (intervalMs > 0 && rt0-cur.T0 >= intervalMs) {
			if cur != nil {
				out = append(out, *cur)
			}
			cur = &publicMixSeg{T0: rt0, T1: rt1, C: s.Compute, Lc: s.LocalCache, Ec: s.ExternalCache}
		} else {
			cur.T1 = rt1
			cur.C += s.Compute
			cur.Lc += s.LocalCache
			cur.Ec += s.ExternalCache
		}
	}
	if cur != nil {
		out = append(out, *cur)
	}
	return out
}

// downsampleAdtPoints applies the same step-function reduction
// DownsampleAggPoints uses (last point at-or-before each boundary, final
// point always kept), carrying the extra Series field along and shifting T
// onto the per-arm-relative axis.
func downsampleAdtPoints(pts []vizAdtPoint, t0, intervalMs float64) []publicAdtPoint {
	if len(pts) == 0 {
		return nil
	}
	sorted := append([]vizAdtPoint(nil), pts...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].T < sorted[j].T })
	toPub := func(p vizAdtPoint) publicAdtPoint {
		return publicAdtPoint{T: p.T - t0, V: p.Tokens, S: p.Series}
	}
	if intervalMs <= 0 {
		out := make([]publicAdtPoint, len(sorted))
		for i, p := range sorted {
			out[i] = toPub(p)
		}
		return out
	}
	boundary := sorted[0].T + intervalMs
	var out []publicAdtPoint
	var last vizAdtPoint
	have := false
	for _, p := range sorted {
		for p.T >= boundary {
			if have {
				out = append(out, toPub(last))
			}
			boundary += intervalMs
		}
		last = p
		have = true
	}
	if have {
		out = append(out, toPub(last))
	}
	return out
}

// buildCumulativePoints computes the running (input+cached) and output token
// sums over records sorted by T, then downsamples with the same
// step-function rule as DownsampleAggPoints (applied here directly, over the
// paired {cumIn,cumOut} value rather than DownsampleAggPoints' single V, so
// both cumulative curves are guaranteed to share the exact same downsampled T
// set instead of risking two independent reductions drifting apart by a
// point).
func buildCumulativePoints(records []AggRecord, intervalMs float64) []publicCumPoint {
	if len(records) == 0 {
		return nil
	}
	sorted := append([]AggRecord(nil), records...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].T < sorted[j].T })
	full := make([]publicCumPoint, len(sorted))
	var cumIn, cumOut float64
	for i, r := range sorted {
		cumIn += float64(r.In + r.Ca)
		cumOut += float64(r.Out)
		full[i] = publicCumPoint{T: r.T, CumIn: cumIn, CumOut: cumOut}
	}
	if intervalMs <= 0 {
		return full
	}
	boundary := full[0].T + intervalMs
	var out []publicCumPoint
	var last publicCumPoint
	have := false
	for _, p := range full {
		for p.T >= boundary {
			if have {
				out = append(out, last)
			}
			boundary += intervalMs
		}
		last = p
		have = true
	}
	if have {
		out = append(out, last)
	}
	return out
}

// --- Color assignment, ported from visualize.go's client-side palette logic
// (OTHER_PALETTE/GPU_VARIANTS/DRAM_VARIANTS/WEKA_VARIANTS + the seriesColors
// assignment loop) so the public report's colors are computed once, in Go,
// and simply carried in the JSON -- the client script never needs the
// classify/palette machinery at all. Values and iteration order match
// visualize.go exactly; keep the two lockstep if either palette changes.

var pubOtherPalette = []string{
	"#ab9a64", "#8a95a3", "#a678e8", "#5f987d", "#b98f83", "#c2b078",
	"#8d5fd6", "#707c8a", "#75b094", "#d0b3f5", "#9aa5b1", "#c9b8e8",
}
var pubGPUVariants = []string{"#8094b5", "#68809f", "#9cb0cb", "#54687f", "#aec3da"}
var pubDRAMVariants = []string{"#5f987d", "#4d8069", "#75b094", "#3f6a57", "#8cc4a9"}
var pubWekaVariants = []string{"#a678e8", "#C79FF1", "#8d5fd6", "#b58ff0", "#7745c0", "#d0b3f5"}

// publicSeriesColors returns one hex color per name, in the given (already
// display-sorted) order, mirroring visualize.go's seriesColors assignment.
func publicSeriesColors(names []string) []string {
	colors := make([]string, len(names))
	gpuIdx, dramIdx, wekaIdx, otherIdx := 0, 0, 0, 0
	for i, name := range names {
		switch ClassifyArmAlias(GetArmAlias(name)) {
		case "gpu":
			colors[i] = pubGPUVariants[gpuIdx%len(pubGPUVariants)]
			gpuIdx++
		case "dram":
			colors[i] = pubDRAMVariants[dramIdx%len(pubDRAMVariants)]
			dramIdx++
		case "weka":
			colors[i] = pubWekaVariants[wekaIdx%len(pubWekaVariants)]
			wekaIdx++
		default:
			colors[i] = pubOtherPalette[otherIdx%len(pubOtherPalette)]
			otherIdx++
		}
	}
	return colors
}

// vizPublicTemplate is the --public report: same brand look as vizTemplate
// (visualize.go) but a much smaller, self-contained script -- there is no
// per-request array to compute anything FROM client-side, so most of the
// interactive report's client-side math (computeDerived, windowStats/
// seriesStats, the context-filter modal, CSV export) simply has nothing to
// operate on and is not ported at all. What IS ported verbatim where the
// data shape matches (mixAt/adtAt/adtWindow*/niceSteps/formatTickLabel/
// computeXStepSec/percentileDash/fmtTokens/fmtMs/placeTooltip/
// computeMixReserveH/totalsY) is called out in comments below; keep those in
// lockstep with visualize.go if either changes. mixTotalMax/mixStackHeight
// and cacheMixLayout/drawCacheMix are DELIBERATE DEPARTURES from
// visualize.go, not verbatim ports: this report's bands are re-aggregated
// to a fixed 5-minute bucket (see reaggregateMix/bucketMeanAdt below),
// which visualize.go's fixed-cadence bands never do -- see the comments on
// each for what differs and why.
var vizPublicTemplate = template.Must(template.New("viz-public").Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark Results (Public Summary)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Onest:wght@400;500&display=swap">
<style>
  /* Same official WEKA brand tokens as the interactive report -- see
     visualize.go's vizTemplate for the palette rationale. */
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Onest", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 400; background: #0D1013; color: #C9C9C9; padding: 0 16px 16px; }
  .brandbar { height: 3px; margin: 0 -16px 12px; background: linear-gradient(90deg, #7C03EC, #C91FF8, #FF3FD5); }
  .badge { display: inline-block; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.07em; color: #C79FF1; border: 1px solid #7C03EC; border-radius: 4px; padding: 1px 6px; margin-left: 8px; vertical-align: middle; }
  .topgrid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 780px); gap: 12px; align-items: start; margin-bottom: 12px; }
  @media (max-width: 1100px) { .topgrid { grid-template-columns: minmax(0, 1fr); } }
  .panel { background: #171C20; border: 1px solid #42464A; border-radius: 8px; padding: 10px 12px; min-width: 0; }
  .panel-title { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.07em; color: #8a9096; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
  .panel-title .range { color: #C79FF1; text-transform: none; letter-spacing: 0; }
  .summary-wrap { overflow: auto; max-height: 260px; }
  #summaryTable { border-collapse: separate; border-spacing: 0; font-size: 0.8em; white-space: nowrap; width: 100%; }
  #summaryTable th, #summaryTable td { padding: 3px 0 3px 13px; text-align: right; }
  #summaryTable thead th { position: sticky; top: 0; z-index: 2; background: #171C20; color: #F2F2EB; font-weight: 500; border-bottom: 1px solid #42464A; padding-bottom: 6px; }
  #summaryTable .vcol { position: sticky; left: 0; z-index: 1; background: #171C20; text-align: left; padding-left: 0; padding-right: 14px; }
  #summaryTable thead .vcol { z-index: 3; }
  .summary-name { display: block; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #summaryTable tbody td { color: #F2F2EB; font-variant-numeric: tabular-nums; padding-top: 4px; padding-bottom: 4px; }
  #summaryTable tbody tr.row-hidden { opacity: 0.32; }
  #summaryTable .err-hot { color: #FF6B6B; }
  .help-label { border-bottom: 1px dotted #8a9096; cursor: help; padding-bottom: 2px; }
  .sum-ratio { display: none; }
  #summaryTable.has-ratios .sum-ratio { display: block; min-height: 1.05em; line-height: 1.05; color: #8a9096; font-size: 0.8em; }
  #summaryTable.has-ratios .sum-ratio.up { color: #6BE0A0; }
  #summaryTable.has-ratios .sum-ratio.down { color: #FF8569; }
  .sum-baseline { color: #8a9096; font-size: 0.8em; margin-left: 5px; }
  .summary-head { display: inline-flex; align-items: center; gap: 5px; max-width: 100%; }
  .summary-head .legend-dot { flex: 0 0 auto; }
  .controls { margin-bottom: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .controls label { font-size: 0.85em; cursor: pointer; }
  .controls input[type=checkbox] { margin-right: 4px; }
  #legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 12px; font-size: 0.8em; }
  .legend-item { display: flex; align-items: center; gap: 4px; cursor: pointer; opacity: 1; }
  .legend-item.hidden { opacity: 0.35; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  .legend-count { color: #8a9096; font-size: 0.9em; }
  canvas { background: #171C20; border-radius: 8px; display: block; cursor: crosshair; }
  #tooltip { position: fixed; background: #1E2429; border: 1px solid #42464A; border-radius: 6px; padding: 8px 10px; font-size: 0.8em; pointer-events: none; display: none; z-index: 100; max-width: 300px; line-height: 1.5; }
  .help-tip {
    position: fixed; z-index: 150; background: #1E2429; border: 1px solid #42464A;
    border-radius: 6px; padding: 8px 10px; font-size: 0.8em; line-height: 1.45;
    max-width: 280px; color: #C9C9C9; box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    opacity: 0; visibility: hidden; transform: translateY(4px);
    transition: opacity 120ms ease-out, transform 120ms ease-out, visibility 120ms;
    pointer-events: none;
  }
  .help-tip.visible { opacity: 1; visibility: visible; transform: translateY(0); }
  .help-tip::after {
    content: ""; position: absolute; width: 8px; height: 8px; background: #1E2429;
    left: var(--caret-x, 16px); margin-left: -4px; transform: rotate(45deg);
  }
  .help-tip.caret-top::after { top: -5px; border-left: 1px solid #42464A; border-top: 1px solid #42464A; }
  .help-tip.caret-bottom::after { bottom: -5px; border-right: 1px solid #42464A; border-bottom: 1px solid #42464A; }
  #footer { margin-top: 14px; padding-top: 10px; border-top: 1px solid #42464A; font-size: 0.78em; color: #8a9096; line-height: 1.5; }
</style>
</head>
<body>
<div class="brandbar"></div>
<div id="header">
<div class="info">Aggregate-only report<span class="badge">PUBLIC</span></div>
<div class="topgrid">
  <div class="panel" id="controlsPanel">
    <div class="panel-title">Controls</div>
    <div class="controls">
      <label><input type="checkbox" id="showTTFT" checked> <span class="help-label" data-tip="Median time to first token, rolling window. The prefill cost.">TTFT p50</span></label>
      <label><input type="checkbox" id="showTTFTP95"> <span class="help-label" data-tip="Tail time to first token, rolling window.">TTFT p95</span></label>
      <label><input type="checkbox" id="showResp" checked> <span class="help-label" data-tip="Time to last token: the whole request, prefill plus all decode.">Response Time / TTLT</span></label>
      <label><input type="checkbox" id="showErrors"> <span class="help-label" data-tip="Error-rate bars anchored on the response line.">Errors</span></label>
      <label><input type="checkbox" id="showTotals"> <span class="help-label" data-tip="Cumulative ingest tokens, stacked. Normalized to the run's final total.">Totals (ingest)</span></label>
      {{if .HasCacheMix}}<label><input type="checkbox" id="showCacheMix"> <span class="help-label" data-tip="Where prompt tokens came from: recompute, local cache, or external KV.">Cache Mix</span></label>{{end}}
      <span id="zoomInfo" style="font-size:0.8em;color:#8a9096;"></span>
    </div>
  </div>
  <div class="panel" id="summaryPanel">
    <div class="panel-title">Summary<span class="range">&nbsp;{{.SummaryLabel}} (does not change on zoom)</span></div>
    <div class="summary-wrap"><table id="summaryTable"{{if .HasBaseline}} class="has-ratios"{{end}}>
      <thead><tr>
        <th class="vcol">Variant</th>
        {{range .SummaryCols}}<th><span class="help-label" tabindex="0" data-tip="{{.Label}}">{{.Short}}</span></th>{{end}}
      </tr></thead>
      <tbody>
      {{range $ri, $row := .SummaryRows}}
        <tr data-si="{{$ri}}">
          <td class="vcol"><span class="summary-head"><span class="legend-dot" style="background:{{$row.Color}}"></span><span class="summary-name" title="{{$row.Name}}&#10;{{if $row.ParamsSummary}}params: {{$row.ParamsSummary}}&#10;{{end}}{{$row.WindowInfo}}">{{$row.Name}}</span>{{if $row.IsBaseline}}<span class="sum-baseline">(baseline)</span>{{end}}</span></td>
          {{range $ci, $cell := $row.Cells}}<td><span class="sum-val">{{$cell.Val}}</span>{{if $cell.Ratio}}<span class="sum-ratio help-label {{$cell.Tint}}" tabindex="0" data-tip="Share of the baseline arm. Green is better, orange is worse.">{{$cell.Ratio}}</span>{{else}}<span class="sum-ratio"></span>{{end}}</td>{{end}}
        </tr>
      {{end}}
      </tbody>
    </table></div>
  </div>
</div>
<div id="legend"></div>
</div>
<canvas id="chart"></canvas>
<div id="tooltip"></div>
<div id="helpTip" class="help-tip" role="tooltip"></div>
<div id="footer">{{.Footer}}<span id="footerDisclosure"></span></div>

<script>
const PUBLIC_DATA = {{.Data}};

// Rehydrate positional-array points into objects, same encoding-only shim
// visualize.go uses for RAW_DATA (see REC_FIELDS there): keeps the JSON
// payload small (no repeated key strings per point) without complicating
// anything below this point.
PUBLIC_DATA.forEach(s => {
  s.respP50 = (s.respP50 || []).map(p => ({ t: p[0], v: p[1] }));
  s.ttftP50 = (s.ttftP50 || []).map(p => ({ t: p[0], v: p[1] }));
  s.ttftP95 = (s.ttftP95 || []).map(p => ({ t: p[0], v: p[1] }));
  s.errBars = (s.errBars || []).map(p => ({ t: p[0], errRate: p[1], errs: p[2], total: p[3], respAvg: p[4] }));
  s.mix = (s.mix || []).map(p => ({ t0: p[0], t1: p[1], c: p[2], lc: p[3], ec: p[4] }));
  s.adt = (s.adt || []).map(p => ({ t: p[0], v: p[1], s: p[2] }));
  s.cum = (s.cum || []).map(p => ({ t: p[0], cumIn: p[1], cumOut: p[2] }));
  // s._mixNativeMs: this arm's own cache-mix cadence as emitted -- see
  // mixNativeMs below (function declarations hoist, so it's fine to call
  // before its textual definition). Precomputed once here rather than
  // inside computeSmoothed so it never itself gets re-derived from
  // already-re-aggregated (and therefore wider) buckets.
  s._mixNativeMs = mixNativeMs(s.mix);
});
// Already server-sorted into display order (gpu first, dram, weka last --
// see SortArmIndices in aggregate.go); the client never re-sorts.
const DATA = PUBLIC_DATA;
const seriesColors = DATA.map(s => s.color);
const CONCURRENCY = {{.Concurrency}};
const HAS_CACHE_MIX = {{.HasCacheMix}};
// INTERVAL_MS is the actual server-side downsample cadence (--public-interval,
// default 30s) every emitted point is already spaced at. The fixed 5-minute
// smoothing below is a client-side, DISPLAY-ONLY moving average over those
// already-downsampled points -- see the block below for why it must never be
// described as a coarser percentile.
const INTERVAL_MS = {{.IntervalMs}};

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const helpTip = document.getElementById("helpTip");

const margin = { top: 30, right: 20, bottom: 20, left: 70 };
let W, H, plotW, plotH;
let mixReserveH = 0;
let hiddenSeries = new Set();
let dragStart = null, dragCurrent = null;

// Zoom range. Deliberately affects ONLY the chart -- the summary panel above
// is precomputed over the full run and never repriced (see the header's
// "does not change on zoom" label); nothing in this script writes to the
// summary table after initial render.
let globalTMin = {{.GlobalTMin}}, globalTMax = {{.GlobalTMax}};
let viewTMin = globalTMin, viewTMax = globalTMax, viewYMax;

function isZoomed() { return viewTMin !== globalTMin || viewTMax !== globalTMax; }

// --- DOM-free helpers, ported verbatim from visualize.go where the data
// shape matches (mix segments as {t0,t1,c,lc,ec}, adt points as {t,v,s},
// percentile points as {t,v}) -- see visualize.go for the original
// commentary; kept in lockstep if either changes. ---
function fmtTokens(v) {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return "" + Math.round(v);
}
function fmtMs(v) {
  if (!(v > 0)) return "-";
  if (v >= 1000) return (v / 1000).toFixed(2) + "s";
  return Math.round(v) + "ms";
}
function mixAt(mix, t) {
  if (!mix || !mix.length || t < mix[0].t0) return null;
  let found = null;
  for (let i = 0; i < mix.length; i++) {
    const m = mix[i];
    if (m.t0 <= t && t <= m.t1) return m;
    if (m.t1 <= t) found = m; else break;
  }
  return found;
}
function adtAt(adt, t) {
  if (!adt || !adt.length || t < adt[0].t) return null;
  let found = adt[0];
  for (let i = 1; i < adt.length; i++) {
    if (adt[i].t <= t) found = adt[i]; else break;
  }
  return found;
}
function adtWindow(adt, tMin, tMax) {
  if (!adt || !adt.length) return [];
  const out = [];
  let carry = null;
  for (let i = 0; i < adt.length; i++) {
    const p = adt[i];
    if (p.t < tMin) { carry = p; continue; }
    if (p.t > tMax) break;
    out.push(p);
  }
  if (carry) out.unshift(carry);
  return out;
}
function adtWindowRange(ptsPerBand) {
  let lo = Infinity, hi = -Infinity;
  (ptsPerBand || []).forEach(pts => (pts || []).forEach(p => {
    if (p.v < lo) lo = p.v;
    if (p.v > hi) hi = p.v;
  }));
  if (!isFinite(lo) || !isFinite(hi)) return null;
  return { lo, hi };
}
// mixRate: the interval's ingest rate in input tokens/s, over the ACTUAL
// sample interval (missed ticks widen it; never assume 60s). Ported from
// visualize.go, unit-tested there -- kept in lockstep.
function mixRate(seg) {
  const secs = (seg.t1 - seg.t0) / 1000;
  if (secs <= 0) return 0;
  return (seg.c + seg.lc + seg.ec) / secs;
}
// mixTotalMax returns the maximum per-minute ingest RATE (compute+local+
// external, scaled by mixRate to tokens/min) across every series in the
// report -- the single shared scale for all bands, deliberately NOT
// per-series (a series peaking at 50k tok/min next to one peaking at 1M
// renders mostly unfilled). UNLIKE visualize.go's version of this function,
// this uses a RATE rather than the raw per-bucket total: the public
// report's bands are re-aggregated to a wider (fixed 5-minute) bucket (see
// reaggregateMix), and a bucket's raw total grows roughly
// proportional to its width even when the underlying activity hasn't
// changed -- only the rate is comparable across bucket widths, so only the
// rate is a legitimate "peak" figure to label and scale against.
function mixTotalMax(seriesArr) {
  let mx = 0;
  (seriesArr || []).forEach(s => (s.mix || []).forEach(m => {
    const r = mixRate(m) * 60;
    if (r > mx) mx = r;
  }));
  return mx;
}
// mixStackHeight: absolute stack height for one interval -- this interval's
// ingest RATE (see mixTotalMax) as a fraction of the shared max rate, of the
// band height. The inner per-source split below still uses the interval's
// absolute c/lc/ec totals (that ratio is unaffected by dividing all three by
// the same interval width).
function mixStackHeight(seg, globalMaxRatePerMin, bandH) {
  const total = seg.c + seg.lc + seg.ec;
  if (total <= 0 || globalMaxRatePerMin <= 0) return 0;
  const ratePerMin = mixRate(seg) * 60;
  return bandH * (ratePerMin / globalMaxRatePerMin);
}
function placeTooltip(cx, cy, tipW, tipH, vw, vh) {
  const off = 12, pad = 4;
  let x = cx + off;
  if (x + tipW > vw - pad) x = cx - off - tipW;
  x = Math.min(Math.max(x, pad), Math.max(pad, vw - pad - tipW));
  let y = cy - 10;
  if (y + tipH > vh - pad) y = cy - off - tipH;
  y = Math.min(Math.max(y, pad), Math.max(pad, vh - pad - tipH));
  return { x: x, y: y };
}
function niceSteps(maxVal, targetSteps) {
  if (maxVal <= 0) return [0];
  const rough = maxVal / targetSteps;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / pow;
  let step;
  if (norm < 1.5) step = 1 * pow;
  else if (norm < 3.5) step = 2 * pow;
  else if (norm < 7.5) step = 5 * pow;
  else step = 10 * pow;
  const steps = [];
  for (let v = 0; v <= maxVal; v += step) steps.push(v);
  return steps;
}
function computeXStepSec(duration) {
  if (duration <= 10) return 1;
  if (duration <= 30) return 2;
  if (duration <= 60) return 5;
  if (duration <= 300) return 30;
  if (duration <= 600) return 60;
  if (duration <= 1800) return 120;
  if (duration <= 3600) return 300;
  if (duration <= 7200) return 600;
  if (duration <= 14400) return 900;
  if (duration <= 28800) return 1800;
  return 3600;
}
function formatTickLabel(s) {
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m" + (s % 60 ? (s % 60) + "s" : "");
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h + "h" + (m ? m + "m" : "");
}
function percentileDash(kind) {
  if (kind === "ttft50") return [2, 3];
  if (kind === "ttft95") return [10, 7];
  return [];
}
function totalsY(frac, plotTop, plotHeight, ceilingY) {
  const bottom = plotTop + plotHeight;
  return bottom - frac * (bottom - ceilingY);
}
// cumAt: cumulative {in,out} tokens at time t, from a series' downsampled
// cum array (binary search, same idea as cumCountAt/cumTokensAt in
// visualize.go, collapsed into one lookup since both curves share one t).
function cumAt(cum, t) {
  if (!cum || !cum.length) return { in: 0, out: 0 };
  let lo = 0, hi = cum.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid].t <= t) lo = mid + 1; else hi = mid; }
  const idx = lo - 1;
  return idx >= 0 ? { in: cum[idx].cumIn, out: cum[idx].cumOut } : { in: 0, out: 0 };
}

// --- Display smoothing: latency lines, cache-mix bands, and the
// active-dataset line are each re-derived from the raw emitted samples at a
// FIXED 5-minute window (there is no user control), chosen per layer by
// whether the underlying quantity composes:
//
//   - respP50/ttftP50/ttftP95 (latency): a CENTERED MOVING AVERAGE over the
//     already-emitted points. This is deliberately NOT a coarser percentile
//     -- the embedded lines are already rolling percentiles over a
//     request-count window, subsequently downsampled to INTERVAL_MS;
//     percentiles do not compose, so there is no way to derive a true
//     5-minute p95 from 30s-resolution p95 samples. Averaging the samples we
//     already have is honest about what it is (see the footer disclosure --
//     footerDisclosureText below) where re-deriving a "5 min p95" label
//     would not be. APPROXIMATE.
//   - cache-mix bands (compute/local_cache/external_cache): RE-AGGREGATED by
//     SUMMING constituent buckets into wider ones (reaggregateMix). These
//     are token SUMS per interval, and sums compose exactly -- combining N
//     consecutive buckets into one is exact arithmetic, not an
//     approximation, so this is NOT a moving average. EXACT.
//   - active-dataset line (tokens/series): RE-AGGREGATED by AVERAGING
//     (bucketMeanAdt), never summing -- this is a GAUGE (an instantaneous
//     level at sample time), and summing a gauge across a wider bucket would
//     scale it up by however many samples happened to land in that bucket,
//     which is meaningless. The mean is the correct reduction for a gauge.
//
// The error-rate bars and cumulative-ingest curve are left alone: cumulative
// ingest is already a running total (re-aggregating it is a no-op by
// construction) and a smoothed error rate would misrepresent when errors
// actually happened (see recommendation in the PR description).
//
// Ordering: SMOOTH is computed exactly ONCE, at load (computeSmoothed
// below), never inside draw()/recalcYMax() and never recomputed afterwards
// -- there is nothing left to change it. Zoom (viewTMin/viewTMax) only
// selects a window out of that fixed series -- it never triggers a
// recompute -- so panning/zooming is cheap and, more importantly, a
// zoomed-in view can't re-derive a window with a different (misleadingly
// cleaner) result than what the unzoomed chart showed for the same points.
const SMOOTH_WINDOW_MS = 300000; // 5 min, fixed -- see block comment above

// windowPointsFor converts the smoothing window (wall-clock ms) into a point
// count against THIS report's actual emitted cadence (INTERVAL_MS) -- so 5
// minutes means 10 points at the default 30s interval, but only 5 points if
// --public-interval was set to 1m, matching the emitted data instead of a
// hardcoded point count. The active-dataset line is emitted at this same
// INTERVAL_MS cadence (see downsampleAdtPoints server-side), so this count
// also drives its bucket-mean re-aggregation below.
function windowPointsFor(ms) {
  if (!(ms > 0) || !(INTERVAL_MS > 0)) return 1;
  return Math.max(1, Math.round(ms / INTERVAL_MS));
}

// smoothPts is a CENTERED moving average of windowPoints points. The window
// shrinks (never pads with zeros, never drops a point) at both ends: index i
// averages over [i-half, i+halfHi] clamped to the array bounds, so the first
// and last output points still equal a real local average of the points that
// actually exist there, not a value dragged toward zero by an assumed
// out-of-range neighbor. Every input point produces exactly one output
// point, in order, so a moving average never changes the series' length.
function smoothPts(pts, windowPoints) {
  if (!pts || pts.length === 0 || windowPoints <= 1) return pts;
  const n = pts.length;
  const half = Math.floor((windowPoints - 1) / 2);
  const halfHi = (windowPoints - 1) - half;
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + pts[i].v;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n - 1, i + halfHi);
    const cnt = hi - lo + 1;
    out[i] = { t: pts[i].t, v: (prefix[hi + 1] - prefix[lo]) / cnt };
  }
  return out;
}

// mixNativeMs estimates the arm's own cache-mix cadence as already emitted
// (the median bucket width) rather than threading a new field through
// PUBLIC_DATA for it. downsampleMixSegments (Go, server-side) only merges
// raw sampler buckets when the sampler's OWN cadence is finer than
// --public-interval; when the sampler is coarser (60s sampler, 30s
// --public-interval, the reference fixture), the emitted mix bucket width is
// the sampler's native cadence, not INTERVAL_MS -- so the two layers' native
// cadences can legitimately differ, and re-aggregating the bands has to
// snap to ITS OWN cadence, not the latency lines' one.
function mixNativeMs(mix) {
  if (!mix || !mix.length) return INTERVAL_MS;
  const widths = mix.map(m => m.t1 - m.t0).filter(w => w > 0).sort((a, b) => a - b);
  if (!widths.length) return INTERVAL_MS;
  return widths[Math.floor(widths.length / 2)];
}

// mixBucketWidthMs snaps the selected smoothing window to a whole multiple
// of this arm's native cache-mix cadence -- never finer than the data
// actually is, and rounded to the NEAREST multiple (minimum 1x) so e.g. a
// 2-minute selection against a 90s native cadence becomes 2x = 180s, not a
// fractional 1.33x. ms<=0 ("None") means the native cadence itself, i.e. the
// bands render exactly as emitted, with no further re-aggregation.
function mixBucketWidthMs(nativeMs, windowMs) {
  if (!(nativeMs > 0)) return windowMs > 0 ? windowMs : INTERVAL_MS;
  if (!(windowMs > 0)) return nativeMs;
  const mult = Math.max(1, Math.round(windowMs / nativeMs));
  return mult * nativeMs;
}

// reaggregateMix re-buckets already-downsampled cache-mix segments to
// bucketMs by SUMMING each source's token delta -- the exact same
// "bucket start >= bucketMs apart" grouping rule downsampleMixSegments (Go)
// used to build these segments in the first place, applied again
// client-side. Summing sums is still exact: every raw token is counted
// exactly once, in exactly one bucket, so the combined totals are identical
// to the totals in the original (finer) segments -- only the bucket
// boundaries move. The final, possibly short, bucket is kept as-is (never
// merged into its neighbor, never dropped), matching the Go original.
function reaggregateMix(segs, bucketMs) {
  if (!segs || !segs.length || !(bucketMs > 0)) return segs;
  const out = [];
  let cur = null;
  segs.forEach(s => {
    if (cur === null || s.t0 - cur.t0 >= bucketMs) {
      if (cur) out.push(cur);
      cur = { t0: s.t0, t1: s.t1, c: s.c, lc: s.lc, ec: s.ec };
    } else {
      cur.t1 = s.t1;
      cur.c += s.c;
      cur.lc += s.lc;
      cur.ec += s.ec;
    }
  });
  if (cur) out.push(cur);
  return out;
}

// bucketMeanAdt re-buckets the active-dataset line by the MEAN of
// windowPoints consecutive (already INTERVAL_MS-spaced) points -- never the
// sum, since active_dataset_tokens/active_series are GAUGES (an
// instantaneous level at sample time, not an accumulating count), and
// summing a gauge across a wider bucket would scale it by however many
// samples happened to land there instead of describing the level. Buckets
// are non-overlapping and tumbling (unlike the latency lines' centered
// moving average) so every input point contributes to exactly one output
// point; a trailing partial bucket is still averaged over the points it
// actually has, never padded. windowPoints<=1 ("None") is a pure passthrough.
function bucketMeanAdt(pts, windowPoints) {
  if (!pts || !pts.length || windowPoints <= 1) return pts;
  const out = [];
  for (let i = 0; i < pts.length; i += windowPoints) {
    const slice = pts.slice(i, Math.min(i + windowPoints, pts.length));
    let sumT = 0, sumV = 0, sumS = 0;
    slice.forEach(p => { sumT += p.t; sumV += p.v; sumS += p.s; });
    const n = slice.length;
    out.push({ t: sumT / n, v: sumV / n, s: sumS / n });
  }
  return out;
}

// computeSmoothed builds SMOOTH at the fixed SMOOTH_WINDOW_MS -- the moving
// average for the three latency lines, the exact re-aggregation (summed)
// for the cache-mix bands, and the re-aggregation (averaged) for the
// active-dataset line. Called exactly once, at load (see the bottom of this
// script) -- see the ordering note above.
function computeSmoothed() {
  const wp = windowPointsFor(SMOOTH_WINDOW_MS);
  return DATA.map(s => ({
    respP50: smoothPts(s.respP50, wp),
    ttftP50: smoothPts(s.ttftP50, wp),
    ttftP95: smoothPts(s.ttftP95, wp),
    mix: reaggregateMix(s.mix, mixBucketWidthMs(s._mixNativeMs, SMOOTH_WINDOW_MS)),
    adt: bucketMeanAdt(s.adt, wp),
  }));
}
const SMOOTH = computeSmoothed();

// footerDisclosureText states, once, how the latency lines and the
// cache-mix bands/dataset line are derived from the raw samples -- see the
// block comment above for why the treatment (and its honesty about being an
// approximation vs. an exact re-aggregation) differs per layer.
// formatTickLabel keeps the sample-interval wording in sync with the
// server's actual --public-interval instead of hardcoding "30s".
function footerDisclosureText() {
  const sampleLabel = formatTickLabel(Math.round(INTERVAL_MS / 1000));
  return " Latency lines: 5 min moving average of " + sampleLabel + " samples (approximate). Cache mix re-aggregated to 5 min (exact); dataset line uses the mean.";
}

const MIX_COMPUTE_COLOR = "#a86853";
const MIX_LOCAL_COLOR = "#756a99";
const MIX_EXTERNAL_COLOR = "#7C03EC";
const ADT_LINE_COLOR = "#F2F2EB";
const MIX_BAND_H = 64;
const MIX_FILL_ALPHA = 0.78;
// MIX_TOTAL_MAX is computed once against the fixed, already re-aggregated
// SMOOTH bands (computeSmoothed above) -- see mixTotalMax below for why it
// is a RATE, not a raw per-bucket total.
const MIX_TOTAL_MAX = mixTotalMax(SMOOTH);
const TOTALS_AXIS_TARGET_STEPS = 5;

function cacheMixEnabled() {
  const cb = document.getElementById("showCacheMix");
  return HAS_CACHE_MIX && cb && cb.checked;
}
// anyPlotLayerVisible: the public build's smaller layer set (no Requests
// dots) -- see visualize.go's version for why this must cover every layer
// that shares plot/mapY space with the cache-mix bands, not just the latency
// lines recalcYMax scales to.
function anyPlotLayerVisible() {
  return document.getElementById("showTTFT").checked ||
    document.getElementById("showTTFTP95").checked ||
    document.getElementById("showResp").checked ||
    document.getElementById("showErrors").checked ||
    document.getElementById("showTotals").checked;
}
// cacheMixLayout reads SMOOTH[si].mix/.adt -- the fixed, already
// re-aggregated bands/dataset-line (see computeSmoothed) -- not the raw
// DATA[si].mix/.adt. s is kept on each band purely for display (name,
// color); mix/adt are carried alongside it so every caller below reads the
// smoothed view, never the raw one.
function cacheMixLayout() {
  if (!cacheMixEnabled()) return null;
  const bands = [];
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    const mix = SMOOTH[si].mix, adt = SMOOTH[si].adt;
    if ((mix && mix.length) || (adt && adt.length)) bands.push({ s, si, mix, adt });
  });
  if (!bands.length) return null;
  let bandH;
  if (!anyPlotLayerVisible()) {
    bandH = Math.max(24, Math.floor(plotH / bands.length));
  } else {
    bandH = MIX_BAND_H;
    if (bands.length * bandH > plotH * 0.6) bandH = Math.max(24, Math.floor(plotH * 0.6 / bands.length));
  }
  bands.forEach((b, bi) => { b.yTop = margin.top + bi * bandH; b.bandH = bandH; });
  bands.forEach(b => {
    b.adtPts = adtWindow(b.adt, viewTMin, viewTMax);
    b.adtRange = adtWindowRange([b.adtPts]);
  });
  return { bands };
}
function computeMixReserveH() {
  if (!anyPlotLayerVisible()) return 0;
  const layout = cacheMixLayout();
  if (!layout || !layout.bands.length) return 0;
  const last = layout.bands[layout.bands.length - 1];
  return last.yTop + last.bandH - margin.top;
}
function adtY(v, range, yTop, bandH) {
  const inset = 2;
  const h = Math.max(bandH - inset * 2, 1);
  if (!range || range.hi <= range.lo) return yTop + bandH / 2;
  const frac = (v - range.lo) / (range.hi - range.lo);
  return yTop + inset + (1 - Math.min(Math.max(frac, 0), 1)) * h;
}
function drawCacheMix() {
  const layout = cacheMixLayout();
  if (!layout) return;
  layout.bands.forEach(({ s, mix, yTop, bandH, adtPts, adtRange }) => {
    ctx.fillStyle = "#000";
    ctx.fillRect(margin.left, yTop, plotW, bandH);
    (mix || []).forEach(seg => {
      if (seg.t1 < viewTMin || seg.t0 > viewTMax) return;
      const total = seg.c + seg.lc + seg.ec;
      const stackH = mixStackHeight(seg, MIX_TOTAL_MAX, bandH);
      if (stackH <= 0) return;
      const x1 = mapX(seg.t0), x2 = mapX(seg.t1);
      let y = yTop + bandH - stackH;
      [[seg.c, MIX_COMPUTE_COLOR, MIX_FILL_ALPHA],
       [seg.lc, MIX_LOCAL_COLOR, MIX_FILL_ALPHA],
       [seg.ec, MIX_EXTERNAL_COLOR, 0.95]].forEach(([v, col, alpha]) => {
        if (v <= 0) return;
        const h = stackH * (v / total);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = col;
        ctx.fillRect(x1, y, x2 - x1, h);
        y += h;
      });
    });
    ctx.globalAlpha = 1;
    if (adtRange && adtPts && adtPts.length) {
      ctx.strokeStyle = ADT_LINE_COLOR;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      let started = false;
      adtPts.forEach(p => {
        const x = Math.min(Math.max(mapX(p.t), margin.left), margin.left + plotW);
        const y = adtY(p.v, adtRange, yTop, bandH);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      if (adtPts.length === 1) {
        ctx.lineTo(margin.left + plotW, adtY(adtPts[0].v, adtRange, yTop, bandH));
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = "#42464A";
    ctx.lineWidth = 0.5;
    ctx.strokeRect(margin.left, yTop, plotW, bandH);
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#C79FF1";
    ctx.fillText(s.name + " cache mix (peak " + fmtTokens(MIX_TOTAL_MAX) + " tok/min)", margin.left + 4, yTop + 3);
    if (adtPts && adtPts.length) {
      const last = adtPts[adtPts.length - 1];
      ctx.fillStyle = ADT_LINE_COLOR;
      ctx.textAlign = "right";
      ctx.fillText("last dataset size (tokens): " + fmtTokens(last.v) +
        " | " + Math.round(last.s) + " series | scale " + fmtTokens(adtRange.lo) + "-" + fmtTokens(adtRange.hi),
        margin.left + plotW - 4, yTop + 3);
      ctx.textAlign = "left";
    }
  });
}

// --- Totals (cumulative ingest) layer, adapted from drawTotals/
// drawTotalsAxis/volumeGeometry in visualize.go: same geometry and stacking
// rule, reading each series' precomputed s.cum instead of a live per-request
// running sum. ---
function volumeGeometry() {
  const cb = document.getElementById("showTotals");
  if (!cb || !cb.checked) return null;
  const visible = [];
  DATA.forEach((s, si) => { if (!hiddenSeries.has(si) && s.cum && s.cum.length) visible.push({ s, si }); });
  if (!visible.length) return null;
  let finalTotal = 0;
  visible.forEach(({ s }) => { finalTotal += s.cum[s.cum.length - 1].cumIn; });
  if (finalTotal <= 0) return null;
  const layout = cacheMixLayout();
  let ceilingY = margin.top;
  if (layout && layout.bands.length) {
    const last = layout.bands[layout.bands.length - 1];
    ceilingY = last.yTop + last.bandH;
  }
  return { visible, finalTotal, ceilingY };
}
function drawTotals() {
  const geo = volumeGeometry();
  if (!geo) return;
  const { visible, finalTotal, ceilingY } = geo;
  const stepPx = 2;
  const n = Math.max(2, Math.floor(plotW / stepPx) + 1);
  const xs = new Array(n), stacks = new Array(n);
  for (let k = 0; k < n; k++) {
    const px = Math.min(k * stepPx, plotW);
    xs[k] = margin.left + px;
    const t = unmapX(margin.left + px);
    let acc = 0; const row = [];
    visible.forEach(({ s }) => { if (finalTotal > 0) acc += cumAt(s.cum, t).in / finalTotal; row.push(acc); });
    stacks[k] = row;
  }
  ctx.globalAlpha = 0.3;
  visible.forEach(({ si }, li) => {
    ctx.fillStyle = seriesColors[si];
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const y = totalsY(stacks[k][li], margin.top, plotH, ceilingY);
      if (k === 0) ctx.moveTo(xs[k], y); else ctx.lineTo(xs[k], y);
    }
    for (let k = n - 1; k >= 0; k--) {
      const below = li === 0 ? 0 : stacks[k][li - 1];
      ctx.lineTo(xs[k], totalsY(below, margin.top, plotH, ceilingY));
    }
    ctx.closePath();
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}
function calcRightMargin() {
  const geo = volumeGeometry();
  if (!geo) return 20;
  ctx.font = "11px monospace";
  let maxW = 0;
  niceSteps(geo.finalTotal, TOTALS_AXIS_TARGET_STEPS).forEach(v => {
    maxW = Math.max(maxW, ctx.measureText(fmtTokens(v)).width);
  });
  return Math.ceil(maxW) + 4 + 8 + 6;
}
function drawTotalsAxis() {
  const geo = volumeGeometry();
  if (!geo) return;
  const { finalTotal, ceilingY } = geo;
  const bottom = margin.top + plotH;
  const xEdge = margin.left + plotW;
  ctx.save();
  ctx.strokeStyle = "#42464A";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#8a9096";
  ctx.font = "11px monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  niceSteps(finalTotal, TOTALS_AXIS_TARGET_STEPS).forEach(v => {
    const y = totalsY(v / finalTotal, margin.top, plotH, ceilingY);
    if (y < ceilingY - 0.5 || y > bottom + 0.5) return;
    ctx.beginPath();
    ctx.moveTo(xEdge, y);
    ctx.lineTo(xEdge + 4, y);
    ctx.stroke();
    ctx.fillText(fmtTokens(v), xEdge + 8, y);
  });
  ctx.restore();
}

function mapX(t) { return margin.left + ((t - viewTMin) / (viewTMax - viewTMin)) * plotW; }
function unmapX(px) { return viewTMin + ((px - margin.left) / plotW) * (viewTMax - viewTMin); }
function mapY(v) {
  const top = margin.top + mixReserveH;
  const h = Math.max(plotH - mixReserveH, 1);
  return top + h - (v / viewYMax) * h;
}
function calcBottomMargin() { return 20; }

function resize() {
  W = Math.min(window.innerWidth - 32, 1800);
  const headEl = document.getElementById("header");
  const headH = headEl ? headEl.getBoundingClientRect().height : 160;
  H = Math.max(420, Math.min(window.innerHeight - headH - 40, 800));
  margin.bottom = calcBottomMargin();
  margin.right = calcRightMargin();
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  plotW = W - margin.left - margin.right;
  plotH = H - margin.top - margin.bottom;
}

// recalcYMax: same rule as visualize.go -- scaled from the rolling
// percentile lines actually drawn, regardless of which layer checkboxes are
// on, and never from raw per-request values (this build has none anyway).
function recalcYMax() {
  viewYMax = 0;
  const bump = v => { if (v > viewYMax) viewYMax = v; };
  const scan = pts => { (pts || []).forEach(p => { if (p.t < viewTMin || p.t > viewTMax) return; bump(p.v); }); };
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    scan(SMOOTH[si].respP50);
    scan(SMOOTH[si].ttftP50);
    scan(SMOOTH[si].ttftP95);
  });
  viewYMax = Math.max(viewYMax * 1.1, 1);
}
function resetZoomView() {
  viewTMin = globalTMin;
  viewTMax = globalTMax;
  recalcYMax();
  draw();
}

function draw() {
  const newRight = calcRightMargin();
  if (margin.right !== newRight) { margin.right = newRight; resize(); }
  mixReserveH = computeMixReserveH();

  const showTTFT = document.getElementById("showTTFT").checked;
  const showTTFTP95 = document.getElementById("showTTFTP95").checked;
  const showResp = document.getElementById("showResp").checked;
  const showErrors = document.getElementById("showErrors").checked;

  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = "#21262b";
  ctx.lineWidth = 0.5;
  ctx.fillStyle = "#8a9096";
  ctx.font = "11px monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  const ySteps = niceSteps(viewYMax, 8);
  ySteps.forEach(v => {
    const y = mapY(v);
    if (y >= margin.top + mixReserveH && y <= margin.top + plotH) {
      ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
      let label = v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "s" : v.toFixed(0) + "ms";
      ctx.fillText(label, margin.left - 6, y);
    }
  });

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const duration = (viewTMax - viewTMin) / 1000;
  const xStepSec = computeXStepSec(duration);
  const startSec = Math.ceil(((viewTMin - globalTMin) / 1000) / xStepSec) * xStepSec;
  for (let s = startSec; s <= (viewTMax - globalTMin) / 1000; s += xStepSec) {
    const x = mapX(globalTMin + s * 1000);
    if (x < margin.left || x > margin.left + plotW) continue;
    ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
    ctx.fillText(formatTickLabel(s), x, margin.top + plotH + 6);
  }

  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#C9C9C9";
  ctx.font = "12px sans-serif";
  ctx.fillText("Latency", 0, 0);
  ctx.restore();

  drawTotalsAxis();

  ctx.strokeStyle = "#42464A";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.clip();

  drawTotals();
  drawCacheMix();

  const plotLine = (pts, kind, color, alpha, width, skipZero) => {
    if (!pts || pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = width;
    ctx.setLineDash(percentileDash(kind));
    ctx.beginPath();
    let started = false;
    pts.forEach(p => {
      if (p.t < viewTMin || p.t > viewTMax || (skipZero && p.v <= 0)) return;
      const x = mapX(p.t), y = mapY(p.v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  };
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    const color = seriesColors[si];
    if (showResp) plotLine(SMOOTH[si].respP50, "resp50", color, 0.8, 2, false);
    if (showTTFT) plotLine(SMOOTH[si].ttftP50, "ttft50", color, 0.8, 2, true);
    if (showTTFTP95) plotLine(SMOOTH[si].ttftP95, "ttft95", color, 0.55, 1.5, true);
  });

  if (showErrors) {
    const maxBarH = plotH * 0.15;
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      const color = seriesColors[si];
      (s.errBars || []).forEach(b => {
        if (b.t < viewTMin || b.t > viewTMax) return;
        const x = mapX(b.t);
        const barH = maxBarH * b.errRate;
        const yTop = mapY(b.respAvg) - barH;
        const yBot = mapY(b.respAvg);
        ctx.fillStyle = "#FF6B6B";
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x - 2, yTop, 4, barH);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 4, yBot); ctx.lineTo(x + 4, yBot); ctx.stroke();
      });
    });
  }

  ctx.restore();

  if (dragStart !== null && dragCurrent !== null) {
    const x1 = Math.max(margin.left, Math.min(dragStart, dragCurrent));
    const x2 = Math.min(margin.left + plotW, Math.max(dragStart, dragCurrent));
    ctx.fillStyle = "rgba(124, 3, 236, 0.14)";
    ctx.fillRect(x1, margin.top, x2 - x1, plotH);
    ctx.strokeStyle = "#C91FF8";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x1, margin.top, x2 - x1, plotH);
    ctx.setLineDash([]);
  }

  document.getElementById("zoomInfo").textContent = isZoomed() ? "Esc or double-click to exit zoom" : "Drag on chart to zoom into timeframe";
}

// --- Legend: click-to-hide only -- no counts to reprice on zoom, so the
// (ok, err) counts shown are the arm's whole-run totals from the precomputed
// summary, same numbers the static summary panel's Reqs/Err columns show. ---
const legendEl = document.getElementById("legend");
const legendItems = [];
DATA.forEach((s, i) => {
  const item = document.createElement("div");
  item.className = "legend-item";
  const dot = document.createElement("div");
  dot.className = "legend-dot";
  dot.style.background = seriesColors[i];
  item.appendChild(dot);
  const label = document.createElement("span");
  label.textContent = s.name;
  item.appendChild(label);
  const count = document.createElement("span");
  count.className = "legend-count";
  count.textContent = "(" + s.ok + (s.err > 0 ? ", " + s.err + " err" : "") + ")";
  item.appendChild(count);
  item.onclick = () => {
    if (hiddenSeries.has(i)) hiddenSeries.delete(i); else hiddenSeries.add(i);
    item.classList.toggle("hidden", hiddenSeries.has(i));
    recalcYMax();
    draw();
  };
  legendEl.appendChild(item);
  legendItems.push(item);
});

["showTTFT", "showTTFTP95", "showResp"].forEach(id => {
  document.getElementById(id).addEventListener("change", () => { recalcYMax(); draw(); });
});
document.getElementById("showErrors").addEventListener("change", draw);
document.getElementById("showTotals").addEventListener("change", draw);
if (HAS_CACHE_MIX) {
  const cb = document.getElementById("showCacheMix");
  if (cb) { cb.checked = DATA.length <= 4; cb.addEventListener("change", draw); }
}

// --- Drag-to-zoom (chart only; see isZoomed()'s doc comment above) ---
canvas.addEventListener("mousedown", e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  if (mx >= margin.left && mx <= margin.left + plotW) {
    dragStart = mx; dragCurrent = mx;
    tooltip.style.display = "none";
  }
});
canvas.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  if (dragStart !== null) {
    dragCurrent = Math.max(margin.left, Math.min(margin.left + plotW, mx));
    draw();
    return;
  }
  // Simple hover: nearest visible line point, else a cache-mix band, else
  // nothing. No per-request tooltip -- there is no per-request data.
  let best = null, bestDist = 15;
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    const checkLine = (pts, type, label) => {
      (pts || []).forEach(p => {
        if (p.t < viewTMin || p.t > viewTMax) return;
        const d = Math.hypot(mx - mapX(p.t), my - mapY(p.v));
        if (d < bestDist) { bestDist = d; best = { s, type, label, p }; }
      });
    };
    if (document.getElementById("showResp").checked) checkLine(SMOOTH[si].respP50, "resp50", "Resp/TTLT p50");
    if (document.getElementById("showTTFT").checked) checkLine(SMOOTH[si].ttftP50, "ttft50", "TTFT p50");
    if (document.getElementById("showTTFTP95").checked) checkLine(SMOOTH[si].ttftP95, "ttft95", "TTFT p95");
  });
  let mixHover = null;
  if (!best && mx >= margin.left && mx <= margin.left + plotW) {
    const layout = cacheMixLayout();
    if (layout) {
      const band = layout.bands.find(b => my >= b.yTop && my <= b.yTop + b.bandH);
      if (band) {
        const t = unmapX(mx);
        if (mixAt(band.mix, t) || adtAt(band.adt, t)) mixHover = { band, t };
      }
    }
  }
  if (best) {
    const fmt = v => v >= 1000 ? (v / 1000).toFixed(2) + "s" : v.toFixed(0) + "ms";
    tooltip.innerHTML = "<b>" + best.s.name + "</b> — " + best.label + "<br>" +
      "t = " + formatTickLabel(Math.round((best.p.t - globalTMin) / 1000)) + "<br>" +
      "value: " + fmt(best.p.v);
    tooltip.style.display = "block";
    tooltip.style.left = "0px"; tooltip.style.top = "0px";
    const r = tooltip.getBoundingClientRect();
    const pos = placeTooltip(e.clientX, e.clientY, r.width, r.height, window.innerWidth, window.innerHeight);
    tooltip.style.left = pos.x + "px"; tooltip.style.top = pos.y + "px";
  } else if (mixHover) {
    const seg = mixAt(mixHover.band.mix, mixHover.t);
    const p = adtAt(mixHover.band.adt, mixHover.t);
    const lines = [];
    if (seg) {
      const total = seg.c + seg.lc + seg.ec;
      const pct = v => total > 0 ? " (" + (100 * v / total).toFixed(0) + "%)" : "";
      lines.push("<span style='color:" + MIX_COMPUTE_COLOR + "'>compute: " + fmtTokens(seg.c) + pct(seg.c) + "</span>");
      lines.push("<span style='color:" + MIX_LOCAL_COLOR + "'>local cache: " + fmtTokens(seg.lc) + pct(seg.lc) + "</span>");
      lines.push("<span style='color:" + MIX_EXTERNAL_COLOR + "'>external KV: " + fmtTokens(seg.ec) + pct(seg.ec) + "</span>");
    }
    if (p) lines.push("<span style='color:" + ADT_LINE_COLOR + "'>active dataset: " + fmtTokens(p.v) + " tok, " + Math.round(p.s) + " series</span>");
    tooltip.innerHTML = "<b>" + mixHover.band.s.name + "</b> — cache mix<br>" + lines.join("<br>");
    tooltip.style.display = "block";
    tooltip.style.left = "0px"; tooltip.style.top = "0px";
    const r = tooltip.getBoundingClientRect();
    const pos = placeTooltip(e.clientX, e.clientY, r.width, r.height, window.innerWidth, window.innerHeight);
    tooltip.style.left = pos.x + "px"; tooltip.style.top = pos.y + "px";
  } else {
    tooltip.style.display = "none";
  }
});
canvas.addEventListener("mouseup", e => {
  if (dragStart === null) return;
  const rect = canvas.getBoundingClientRect();
  const mx = Math.max(margin.left, Math.min(margin.left + plotW, e.clientX - rect.left));
  const minPx = Math.min(dragStart, mx);
  const maxPx = Math.max(dragStart, mx);
  dragStart = null; dragCurrent = null;
  if (maxPx - minPx < 5) { draw(); return; }
  const tMin = unmapX(minPx);
  const tMax = unmapX(maxPx);
  viewTMin = tMin; viewTMax = tMax;
  recalcYMax();
  draw();
});
canvas.addEventListener("mouseleave", () => {
  if (dragStart !== null) { dragStart = null; dragCurrent = null; draw(); }
  tooltip.style.display = "none";
});
window.addEventListener("mouseup", () => {
  if (dragStart !== null) { dragStart = null; dragCurrent = null; draw(); }
});
canvas.addEventListener("dblclick", resetZoomView);

// --- Help tooltip: shared by every .help-label element, whether from the
// static markup (controls, summary headers/ratios) or -- there is none here,
// but kept generic via querySelectorAll rather than a hand-maintained list,
// unlike visualize.go's helpTriggers array, because the public build's
// triggers are all present in the initial static HTML. ---
let helpShowTimer = null, helpHideTimer = null;
function placeHelpTip(trigger) {
  const r = trigger.getBoundingClientRect();
  const gap = 8, pad = 4;
  const vw = window.innerWidth, vh = window.innerHeight;
  const tw = helpTip.offsetWidth, th = helpTip.offsetHeight;
  let top = r.bottom + gap;
  let below = true;
  if (top + th > vh - pad) { top = r.top - gap - th; below = false; }
  top = Math.min(Math.max(top, pad), Math.max(pad, vh - pad - th));
  let left = r.left + r.width / 2 - tw / 2;
  left = Math.min(Math.max(left, pad), Math.max(pad, vw - pad - tw));
  helpTip.style.left = left + "px";
  helpTip.style.top = top + "px";
  helpTip.classList.toggle("caret-top", below);
  helpTip.classList.toggle("caret-bottom", !below);
  const caretX = Math.min(Math.max(r.left + r.width / 2 - left, 10), Math.max(10, tw - 10));
  helpTip.style.setProperty("--caret-x", caretX + "px");
}
function showHelpTip(trigger) {
  clearTimeout(helpShowTimer);
  clearTimeout(helpHideTimer);
  const text = trigger.dataset.tip || "";
  if (!text) return;
  if (helpTip.classList.contains("visible")) {
    helpTip.textContent = text;
    placeHelpTip(trigger);
    return;
  }
  helpShowTimer = setTimeout(() => {
    helpTip.textContent = text;
    placeHelpTip(trigger);
    helpTip.classList.add("visible");
  }, 120);
}
function hideHelpTip() {
  clearTimeout(helpShowTimer);
  helpHideTimer = setTimeout(() => helpTip.classList.remove("visible"), 60);
}
function hideHelpTipNow() {
  clearTimeout(helpShowTimer);
  clearTimeout(helpHideTimer);
  helpTip.classList.remove("visible");
}
document.querySelectorAll(".help-label").forEach(el => {
  el.addEventListener("mouseenter", () => showHelpTip(el));
  el.addEventListener("mouseleave", hideHelpTip);
  el.addEventListener("focus", () => showHelpTip(el));
  el.addEventListener("blur", hideHelpTip);
});

window.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  if (helpTip.classList.contains("visible")) { hideHelpTipNow(); return; }
  if (isZoomed()) resetZoomView();
});

window.addEventListener("resize", () => { resize(); draw(); });
document.getElementById("footerDisclosure").textContent = footerDisclosureText();
recalcYMax();
resize();
draw();
</script>
</body>
</html>
`))

// formatDurationLabel mirrors formatTickLabel(s) in visualize.go exactly
// (compact "45m"/"1h15m"/"2h" form), for the footer and the summary panel's
// "full run (...)" label -- both server-rendered text in the public report,
// where the interactive report's client-side formatTickLabel would normally
// run.
func formatDurationLabel(seconds float64) string {
	s := int(math.Round(seconds))
	if s < 60 {
		return fmt.Sprintf("%ds", s)
	}
	if s < 3600 {
		m, rem := s/60, s%60
		if rem == 0 {
			return fmt.Sprintf("%dm", m)
		}
		return fmt.Sprintf("%dm%ds", m, rem)
	}
	h, m := s/3600, (s%3600)/60
	if m == 0 {
		return fmt.Sprintf("%dh", h)
	}
	return fmt.Sprintf("%dh%dm", h, m)
}
