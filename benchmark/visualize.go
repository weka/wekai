package benchmark

import (
	"bufio"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/weka/wekai/llm"
)

// aliasParamRe matches an "alias=<value>" parameter embedded in a raw model
// spec string (e.g. "dynamic/http://host:port/v1,type=openai_vllm,alias=DS3H_weka-64r8w").
// Used as a permissive fallback when llm.ParseDynamicModel doesn't recognize
// the spec format, so a real alias is never lost to a strict-parser mismatch.
var aliasParamRe = regexp.MustCompile(`alias=([a-zA-Z0-9._-]+)`)

// extractAlias returns the alias embedded in a raw (unsanitized) model spec
// string, or "" if none is present. Unlike GetModelDisplayName (which falls
// back to returning the full model string for display purposes, so it never
// signals "no alias"), this distinguishes "has a real alias" from "doesn't" —
// required so callers can fall through to a different label source instead
// of using the full raw spec string as a label.
func extractAlias(modelStr string) string {
	if llm.IsDynamicModel(modelStr) {
		if cfg, err := llm.ParseDynamicModel(modelStr); err == nil && cfg.Alias != "" {
			return cfg.Alias
		}
	}
	if m := aliasParamRe.FindStringSubmatch(modelStr); len(m) == 2 {
		return m[1]
	}
	return ""
}

// GenerateVisualization reads all .jsonl files from dir and produces an
// interactive HTML scatter-plot in the same directory. Returns the path
// to the generated HTML file.
func GenerateVisualization(dir string, concurrency int) (string, error) {
	return generateVisualization(dir, concurrency, false, 0)
}

// GenerateVisualizationWithOptions is GenerateVisualization with a
// time-truncation cutoff: when maxElapsed > 0, records (and metrics
// samples) past that elapsed time from each FILE's own run start are
// dropped — see truncateToElapsed.
func GenerateVisualizationWithOptions(dir string, concurrency int, maxElapsed time.Duration) (string, error) {
	return generateVisualization(dir, concurrency, false, maxElapsed)
}

// truncateToElapsed drops request records and metrics samples whose elapsed
// time from the run's own start exceeds maxElapsed, so a crashed run's
// terminal error-storm can be excluded from a report. t0 is the earliest
// request StartTime in the set (falling back to the earliest sample when a
// file somehow has no request rows); the boundary is inclusive — a record
// exactly AT the cutoff is kept. Samples truncate against the same t0 so
// the cache-mix overlay and ingest volume all stop at the
// cutoff consistently with the request data. maxElapsed <= 0 is a no-op.
func truncateToElapsed(records []requestDataRecord, samples []vllmMetricsSample, maxElapsed time.Duration) ([]requestDataRecord, []vllmMetricsSample) {
	if maxElapsed <= 0 || (len(records) == 0 && len(samples) == 0) {
		return records, samples
	}
	var t0 time.Time
	for _, r := range records {
		if t0.IsZero() || r.StartTime.Before(t0) {
			t0 = r.StartTime
		}
	}
	if t0.IsZero() {
		for _, s := range samples {
			if t0.IsZero() || s.TS.Before(t0) {
				t0 = s.TS
			}
		}
	}
	cutoff := t0.Add(maxElapsed)
	var outR []requestDataRecord
	for _, r := range records {
		if !r.StartTime.After(cutoff) {
			outR = append(outR, r)
		}
	}
	var outS []vllmMetricsSample
	for _, s := range samples {
		if !s.TS.After(cutoff) {
			outS = append(outS, s)
		}
	}
	return outR, outS
}

// vizRecord is one request's data as embedded in a report.html's RAW_DATA.
type vizRecord struct {
	// T is delta-encoded against the owning seriesData's T0 (min StartTime
	// of the series), NOT an absolute epoch — see seriesData.T0 and
	// vizRecord.MarshalJSON.
	T          float64
	TTFT       float64
	ResponseMs float64
	IsError    bool
	SeriesNum  int
	RequestNum int
	CacheHit   bool
	// Token counts for the ingest volume layer and its hover rates.
	InputTokens  int // net-of-cache input tokens
	CachedTokens int // server-cached prompt tokens
	OutputTokens int // completion tokens
}

// MarshalJSON emits vizRecord as a positional array —
// [t,ttft,resp,err,sn,rn,ch,in,ca,out], matching REC_FIELDS in the report
// template — instead of a JSON object. The object shape repeats all 10 key
// strings per record; across the ~137k records a typical merged report
// carries, that's several MB of pure key-name bytes. The report's load-time
// rehydration shim (immediately after `const RAW_DATA = {{.Data}}` in the
// template) converts each row back into a
// {t,ttft,resp,err,sn,rn,ch,in,ca,out} object with an absolute t, so every
// downstream render/filter/compute function is unchanged. err/ch are emitted
// as 0/1 and rehydrated back to real booleans by the shim. Field order here
// MUST match REC_FIELDS in the template exactly.
func (r vizRecord) MarshalJSON() ([]byte, error) {
	errV, chV := 0, 0
	if r.IsError {
		errV = 1
	}
	if r.CacheHit {
		chV = 1
	}
	return json.Marshal([10]float64{
		r.T, r.TTFT, r.ResponseMs, float64(errV), float64(r.SeriesNum), float64(r.RequestNum),
		float64(chV), float64(r.InputTokens), float64(r.CachedTokens), float64(r.OutputTokens),
	})
}

// seriesData is one variant's data as embedded in a report.html's RAW_DATA.
type seriesData struct {
	Name string `json:"name"`
	// T0 is the base epoch-ms timestamp (min over Records) that each
	// Records[i].T is delta-encoded against — see vizRecord.T. Rehydrated
	// client-side as t0 + delta to recover the absolute epoch value.
	T0      float64            `json:"t0"`
	Records []vizRecord        `json:"records"`
	Mix     []vizSampleSegment `json:"mix,omitempty"`
	Adt     []vizAdtPoint      `json:"adt,omitempty"`
	// Conc is the request concurrency this arm ran at, taken from its
	// run_params header. 0 for a file written before run_params existed (or a
	// hill-climber run that never pinned one), in which case the report falls
	// back to the report-wide CONCURRENCY and then to deriving a window from
	// the observed series count — exactly the pre-run_params behaviour.
	Conc int `json:"conc,omitempty"`
	// Params is the recorded run configuration, rendered in the report so a
	// reader can see the workload shape without a side file. Omitted entirely
	// for legacy data.
	Params *vizRunParams `json:"params,omitempty"`
}

// vizRunParams is the subset of runParamsRecord the report displays. Kept
// separate from runParamsRecord so adding a recorded field doesn't silently
// grow every embedded report by a column nobody asked for.
type vizRunParams struct {
	Summary     string `json:"summary"`
	Concurrency int    `json:"concurrency,omitempty"`
	HotConc     int    `json:"hot,omitempty"`
	MaxSeries   int    `json:"maxSeries,omitempty"`
	StartSeries int    `json:"startSeries,omitempty"`
	Timeout     string `json:"timeout,omitempty"`
	Total       int    `json:"total,omitempty"`
	Dataset     string `json:"dataset,omitempty"`
	ReplayFile  string `json:"replayFile,omitempty"`
	RunID       string `json:"runId,omitempty"`
}

// buildVizRunParams projects a recorded header into its report form.
func buildVizRunParams(p runParamsRecord) *vizRunParams {
	return &vizRunParams{
		Summary:     p.summaryLine(),
		Concurrency: p.Concurrency,
		HotConc:     p.HotSeriesConcurrency,
		MaxSeries:   p.MaxSeries,
		StartSeries: p.StartSeries,
		Timeout:     p.Timeout,
		Total:       p.TotalRequests,
		Dataset:     p.FromDataset,
		ReplayFile:  p.RouterReplayFile,
		RunID:       p.RunID,
	}
}

// generateVisualization is the implementation behind GenerateVisualization.
// keepFileNames pins each series' DISPLAYED name (legend, cache-mix band
// label, tooltips — all render seriesData.Name) to the .jsonl
// basename instead of re-resolving the record alias. The merged path sets it
// when explicit --labels were given, so labels win end-to-end: two arms
// sharing one alias would otherwise render indistinguishably. maxElapsed > 0
// truncates each file to its own elapsed window (per-file t0) — the merged
// path passes 0 here because it truncates per source directory instead.
func generateVisualization(dir string, concurrency int, keepFileNames bool, maxElapsed time.Duration) (string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		return "", fmt.Errorf("glob jsonl files: %w", err)
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no .jsonl files found in %s", dir)
	}
	sort.Strings(files)

	var allSeries []seriesData
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		records, samples, params, hasParams, err := readJSONLFileWithParams(f)
		if err != nil {
			return "", fmt.Errorf("read %s: %w", f, err)
		}
		records, samples = truncateToElapsed(records, samples, maxElapsed)
		// Prefer the clean model alias (e.g. "DS3H_weka-64r8w") over the raw
		// sanitized filename (e.g. "dynamic_http___..._alias_DS3H_weka-64r8w")
		// when the file's records unambiguously identify one model — unless
		// the caller pinned names to the (label-derived) filenames.
		if !keepFileNames {
			if alias := resolveRecordsAlias(records); alias != "" {
				name = alias
			}
		}
		// t0 = min StartTime across this series' records, used to
		// delta-encode each record's T (see vizRecord.T / seriesData.T0).
		var t0 float64
		haveT0 := false
		for _, r := range records {
			t := float64(r.StartTime.UnixMilli())
			if !haveT0 || t < t0 {
				t0 = t
				haveT0 = true
			}
		}
		var vr []vizRecord
		for _, r := range records {
			vr = append(vr, vizRecord{
				T:            float64(r.StartTime.UnixMilli()) - t0,
				TTFT:         r.TTFT,
				ResponseMs:   r.ResponseMs,
				IsError:      r.IsError,
				SeriesNum:    r.SeriesNum,
				RequestNum:   r.RequestNum,
				CacheHit:     r.CacheHit,
				InputTokens:  r.InputTokens,
				CachedTokens: r.CachedTokens,
				OutputTokens: r.OutputTokens,
			})
		}
		mix, adt := buildSampleViz(samples)
		sd := seriesData{Name: name, T0: t0, Records: vr, Mix: mix, Adt: adt}
		if hasParams {
			sd.Conc = params.effectiveConcurrency()
			sd.Params = buildVizRunParams(params)
		}
		allSeries = append(allSeries, sd)
	}

	seriesJSON, err := json.Marshal(allSeries)
	if err != nil {
		return "", fmt.Errorf("marshal series data: %w", err)
	}

	concStr := "0"
	if concurrency > 0 {
		concStr = fmt.Sprintf("%d", concurrency)
	}

	htmlPath := nextVersionedPath(dir, "report", ".html")
	out, err := os.Create(htmlPath)
	if err != nil {
		return "", fmt.Errorf("create html file: %w", err)
	}
	defer out.Close()

	if err := vizTemplate.Execute(out, map[string]template.JS{
		"Data":        template.JS(seriesJSON),
		"Concurrency": template.JS(concStr),
	}); err != nil {
		return "", fmt.Errorf("execute template: %w", err)
	}
	return htmlPath, nil
}

// resolveRecordsAlias returns the single distinct model alias found across
// records (via extractAlias), or "" when records is empty, no record carries
// a real alias, or records span more than one distinct alias (ambiguous) —
// in all "" cases the caller should fall back to another label source.
func resolveRecordsAlias(records []requestDataRecord) string {
	aliases := map[string]bool{}
	for _, r := range records {
		if a := extractAlias(r.Model); a != "" {
			aliases[a] = true
		}
	}
	if len(aliases) == 1 {
		for a := range aliases {
			return a
		}
	}
	return ""
}

// readJSONLFile reads a request-data JSONL file, discarding any run-params
// header. Callers that can make use of the recorded run parameters should use
// readJSONLFileWithParams instead.
func readJSONLFile(path string) ([]requestDataRecord, []vllmMetricsSample, error) {
	records, samples, _, _, err := readJSONLFileWithParams(path)
	return records, samples, err
}

// readJSONLFileWithParams reads a request-data JSONL file, routing lines by
// their record_type: absent/empty = a request row (legacy files predate the
// field), "vllm_metrics_sample" = a metrics sample, "run_params" = the header
// describing the run. Unknown record types and malformed lines are skipped — a
// new record type must never corrupt request parsing (unmarshalling a sample
// into requestDataRecord would otherwise "succeed" as an all-zero phantom
// request), which is also what lets a file written by a NEWER wekai stay
// readable by an older one.
//
// hasParams is false for every file written before run_params existed; callers
// must keep working in that case rather than treating the zero record as a run
// that was configured with zeroes.
func readJSONLFileWithParams(path string) ([]requestDataRecord, []vllmMetricsSample, runParamsRecord, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, runParamsRecord{}, false, err
	}
	defer f.Close()
	var records []requestDataRecord
	var samples []vllmMetricsSample
	var params runParamsRecord
	var hasParams bool
	sc := bufio.NewScanner(f)
	// 64 MiB cap: reqdata rows embed full prompts; 300k-token contexts exceed 1 MiB.
	sc.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	for sc.Scan() {
		var probe struct {
			RecordType string `json:"record_type"`
		}
		if err := json.Unmarshal(sc.Bytes(), &probe); err != nil {
			continue // skip malformed lines
		}
		switch probe.RecordType {
		case "":
			var r requestDataRecord
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				continue
			}
			records = append(records, r)
		case recordTypeVLLMMetricsSample:
			var s vllmMetricsSample
			if err := json.Unmarshal(sc.Bytes(), &s); err != nil {
				continue
			}
			samples = append(samples, s)
		case recordTypeRunParams:
			// First header wins: a merged per-source JSONL can concatenate
			// files, and the first one describes the run the rows came from.
			if p, ok := parseRunParams(sc.Bytes()); ok && !hasParams {
				params, hasParams = p, true
			}
		}
	}
	return records, samples, params, hasParams, sc.Err()
}

// nextVersionedPath returns a path like dir/report.html, dir/report_v2.html, etc.
// It never overwrites an existing file.
func nextVersionedPath(dir, base, ext string) string {
	p := filepath.Join(dir, base+ext)
	if _, err := os.Stat(p); os.IsNotExist(err) {
		return p
	}
	for v := 2; ; v++ {
		p = filepath.Join(dir, fmt.Sprintf("%s_v%d%s", base, v, ext))
		if _, err := os.Stat(p); os.IsNotExist(err) {
			return p
		}
	}
}

var vizTemplate = template.Must(template.New("viz").Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Onest:wght@400;500&display=swap">
<style>
  /* Official WEKA brand guidance: surfaces #0D1013/#171C20/#1E2429, border
     #42464A, hover #2A3038; primary purple #7C03EC with gradient accents
     #C91FF8/#FF3FD5; text #F2F2EB primary / #C9C9C9 muted / #C79FF1
     purple-accented; status #6BE0A0/#FF6B6B/#FF8569/#FFD600. Neutral-first:
     the palette is the vocabulary, restraint is the grammar. Onest loads
     from Google Fonts when online and falls back to system sans offline —
     reports must render self-contained. */
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Onest", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 400; background: #0D1013; color: #C9C9C9; padding: 0 16px 16px; }
  .brandbar { height: 3px; margin: 0 -16px 12px; background: linear-gradient(90deg, #7C03EC, #C91FF8, #FF3FD5); }
  .info { font-size: 0.85em; color: #8a9096; margin-bottom: 12px; }
  /* Upper view is two panels side by side: toggles on the left, a per-variant
     summary of the SELECTED timeframe on the right. Below ~1100px the grid
     collapses to one column so the summary stacks under the controls rather
     than squeezing both. The summary column is capped (not auto-sized) so a
     10-variant report can never squeeze the controls out. */
  .topgrid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 780px); gap: 12px; align-items: start; margin-bottom: 12px; }
  @media (max-width: 1100px) { .topgrid { grid-template-columns: minmax(0, 1fr); } }
  .panel { background: #171C20; border: 1px solid #42464A; border-radius: 8px; padding: 10px 12px; min-width: 0; }
  .panel-title { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.07em; color: #8a9096; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
  .panel-title .range { color: #C79FF1; text-transform: none; letter-spacing: 0; }
  .panel-title .spacer { margin-left: auto; }
  /* Collapse control: both panels can be shrunk vertically to hand their
     height back to the chart, which is the only element that benefits from
     it. resize() re-measures the header on toggle, so the canvas grows. */
  .panel-toggle { font-size: 1.1em; line-height: 1; padding: 0 5px; background: transparent; color: #8a9096; border: 1px solid #42464A; border-radius: 4px; cursor: pointer; }
  .panel-toggle:hover { background: #2A3038; color: #F2F2EB; }
  .panel.collapsed .panel-title { margin-bottom: 0; }
  .panel.collapsed .panel-body { display: none; }
  /* The summary scrolls in BOTH axes inside a capped box: vertically because
     variants are rows (10 arms must stay navigable), horizontally for narrow
     viewports. The header row and the variant-name column are sticky so
     neither scroll direction can leave a number unlabelled. */
  .summary-wrap { overflow: auto; max-height: 200px; }
  #summaryTable { border-collapse: separate; border-spacing: 0; font-size: 0.8em; white-space: nowrap; width: 100%; }
  #summaryTable th, #summaryTable td { padding: 3px 0 3px 13px; text-align: right; }
  #summaryTable thead th { position: sticky; top: 0; z-index: 2; background: #171C20; color: #F2F2EB; font-weight: 500; border-bottom: 1px solid #42464A; padding-bottom: 6px; }
  #summaryTable .vcol { position: sticky; left: 0; z-index: 1; background: #171C20; text-align: left; padding-left: 0; padding-right: 14px; }
  #summaryTable thead .vcol { z-index: 3; }
  /* Variant names run long (45+ chars on router/variant-suffixed arms). Cap
     the name column and ellipsize so the six metric columns always fit rather
     than being pushed off the right edge — the full name stays on hover. */
  .summary-name { display: block; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #summaryTable tbody td { color: #F2F2EB; font-variant-numeric: tabular-nums; padding-top: 4px; padding-bottom: 4px; }
  #summaryTable tbody tr { cursor: pointer; }
  #summaryTable tbody tr:hover td, #summaryTable tbody tr:hover .vcol { background: #1E2429; }
  #summaryTable tbody tr.row-hidden { opacity: 0.32; }
  #summaryTable .err-hot { color: #FF6B6B; }
  /* Hover-help affordance, shared by summary headers and control labels: the
     label TEXT itself is the hover target -- a dotted underline in a muted
     colour plus a help cursor, the conventional "more info on hover"
     affordance -- rather than a separate "?" glyph, which costs zero
     horizontal space (the summary table has 8 columns to fit). Opens the
     shared #helpTip custom tooltip; see its rules next to #tooltip below and
     the wiring near the bottom of this script. */
  .help-label { border-bottom: 1px dotted #8a9096; cursor: help; padding-bottom: 2px; }
  /* Right-axis title for the Totals (ingest) layer: a real DOM node
     positioned over the canvas (see totalsAxisLabel / drawTotalsAxis in the
     script) rather than ctx.fillText, purely so it can carry the same
     .help-label hover-tooltip affordance as everything else here. Centered
     at (left, top) via the translate(-50%,-50%) trick, then rotated about
     that same center -- see drawTotalsAxis for how left/top are computed. */
  .totals-axis-label { position: fixed; font: 12px sans-serif; color: #C9C9C9; white-space: nowrap; z-index: 2; }
  /* Ratio-to-baseline sits BELOW its value, right-aligned under it, so the
     value column stays in one straight line under its header — inline, the
     ratio pushed each value left by its own width and the numbers no longer
     lined up. The row only exists when there IS a baseline (.has-ratios):
     without one the spans are empty, and an unconditional reserved line would
     spend a row of height per variant to show nothing. min-height keeps the
     baseline's own (empty) row the same height as the rest. */
  .sum-ratio { display: none; }
  #summaryTable.has-ratios .sum-ratio { display: block; min-height: 1.05em; line-height: 1.05; color: #8a9096; font-size: 0.8em; }
  /* Tint by DIRECTION of improvement, not size of number: 284% more tokens is
     good, 19% of the baseline's TTFT is also good, and fewer errors is good —
     so err/1k and both TTFTs are "down is better". These must out-specify the
     .has-ratios rule above (1 id + 2 classes), or its grey wins and the tint
     silently disappears. */
  #summaryTable.has-ratios .sum-ratio.up { color: #6BE0A0; }
  #summaryTable.has-ratios .sum-ratio.down { color: #FF8569; }
  .sum-baseline { color: #8a9096; font-size: 0.8em; margin-left: 5px; }
  .summary-head { display: inline-flex; align-items: center; gap: 5px; max-width: 100%; }
  .summary-head .legend-dot { flex: 0 0 auto; }
  .controls { margin-bottom: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .controls:last-child { margin-bottom: 0; }
  .controls label { font-size: 0.85em; cursor: pointer; }
  .controls input[type=checkbox] { margin-right: 4px; }
  .controls button { font-size: 0.8em; padding: 3px 10px; background: #1E2429; color: #F2F2EB; border: 1px solid #42464A; border-radius: 4px; cursor: pointer; }
  .controls button:hover { background: #2A3038; }
  .controls button:disabled { opacity: 0.3; cursor: default; }
  #legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 12px; font-size: 0.8em; }
  .legend-item { display: flex; align-items: center; gap: 4px; cursor: pointer; opacity: 1; }
  .legend-item.hidden { opacity: 0.35; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  .legend-count { color: #8a9096; font-size: 0.9em; }
  .legend-ctx { color: #C79FF1; font-size: 0.85em; }
  canvas { background: #171C20; border-radius: 8px; display: block; cursor: crosshair; }
  #tooltip { position: fixed; background: #1E2429; border: 1px solid #42464A; border-radius: 6px; padding: 8px 10px; font-size: 0.8em; pointer-events: none; display: none; z-index: 100; max-width: 300px; line-height: 1.5; }
  /* Help tooltip for the .help-label affordance above: a second, independent
     tooltip deliberately matching #tooltip's look (surface, border, radius,
     padding, font) so the report reads as one system, but its own element
     with its own lifecycle so the two can never fight over position or
     visibility. opacity+visibility (not display) so the fade/translate
     transition has something to animate. Always fixed + appended at the
     body level (see the markup near #tooltip) so the summary panel's own
     scroll container (.summary-wrap, overflow: auto) can never clip it --
     an ancestor with overflow set COULD clip an absolutely-positioned
     descendant, so fixed positioning sidesteps the question entirely
     (matching #tooltip's own position: fixed). A small caret
     (::after, rotated square) points at whichever trigger is active; its
     side flips with .caret-top/.caret-bottom depending on whether the tip
     landed below or above the trigger. */
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
  .modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 200; display: none; }
  .modal { background: #1E2429; border: 1px solid #42464A; border-radius: 8px; padding: 16px; width: 440px; max-height: 72vh; overflow-y: auto; margin: 10vh auto 0; }
  .modal h2 { font-size: 1em; font-weight: 500; color: #F2F2EB; margin-bottom: 10px; }
  .modal-series-row { display: flex; align-items: center; gap: 6px; padding: 3px 0; font-size: 0.85em; }
  .modal-series-row .legend-ctx { margin-left: auto; }
  .modal-band { margin: 12px 0; font-size: 0.85em; }
  .modal-band-row { display: flex; gap: 8px; align-items: center; margin-top: 6px; flex-wrap: nowrap; }
  .modal-band input[type=number] { width: 110px; font-size: 0.9em; padding: 3px 6px; background: #171C20; color: #F2F2EB; border: 1px solid #42464A; border-radius: 4px; }
  .modal-band-sep { color: #8a9096; }
  .modal-section-title { font-size: 0.85em; color: #F2F2EB; margin: 12px 0 4px; }
  .modal-sn-row { display: flex; gap: 8px; align-items: center; margin-top: 6px; flex-wrap: nowrap; font-size: 0.85em; }
  .modal-sn-row input[type=text] { flex: 1; font-size: 0.9em; padding: 3px 6px; background: #171C20; color: #F2F2EB; border: 1px solid #42464A; border-radius: 4px; }
  .modal-sn-row button { font-size: 0.8em; padding: 3px 10px; background: #171C20; color: #F2F2EB; border: 1px solid #42464A; border-radius: 4px; cursor: pointer; }
  .modal-sn-list { max-height: 180px; overflow-y: auto; border: 1px solid #42464A; border-radius: 4px; margin-top: 6px; padding: 4px 6px; }
  .modal-sn-item { display: flex; gap: 6px; align-items: center; padding: 2px 0; font-size: 0.8em; }
  .modal-sn-item .legend-ctx { margin-left: auto; }
  .modal-sn-note { font-size: 0.75em; color: #8a9096; margin-top: 4px; }
  .modal-actions { display: flex; gap: 8px; margin-top: 8px; }
  .modal-actions button { font-size: 0.8em; padding: 4px 12px; background: #171C20; color: #F2F2EB; border: 1px solid #42464A; border-radius: 4px; cursor: pointer; }
  .modal-actions button:hover { background: #2A3038; }
  #ctxApply { border-color: #7C03EC; }
</style>
</head>
<body>
<div class="brandbar"></div>
<div id="header">
<div class="info" id="runparams"></div>
<div class="topgrid">
  <div class="panel" id="controlsPanel">
    <div class="panel-title">Controls<span class="spacer"></span><button class="panel-toggle" id="controlsToggle" title="Collapse">&minus;</button></div>
    <div class="panel-body">
    <div class="controls">
      <label><input type="checkbox" id="showTTFT" checked> <span class="help-label" id="hlpTtft50" tabindex="0" aria-describedby="helpTip" data-tip="Median time to first token, rolling window. The prefill cost.">TTFT p50</span></label>
      <label><input type="checkbox" id="showTTFTP95"> <span class="help-label" id="hlpTtft95" tabindex="0" aria-describedby="helpTip" data-tip="Tail time to first token, rolling window.">TTFT p95</span></label>
      <label><input type="checkbox" id="showResp" checked> <span class="help-label" id="hlpResp" tabindex="0" aria-describedby="helpTip" data-tip="Time to last token: the whole request, prefill plus all decode.">Response Time / TTLT</span></label>
      <label><input type="checkbox" id="showDots"> <span class="help-label" id="hlpReqs" tabindex="0" aria-describedby="helpTip" data-tip="One dot per request. Shows spread the percentile lines hide.">Requests</span></label>
      <label><input type="checkbox" id="showErrors"> <span class="help-label" id="hlpErrors" tabindex="0" aria-describedby="helpTip" data-tip="Error-rate bars anchored on the response line.">Errors</span></label>
      <label><input type="checkbox" id="showTotals"> <span class="help-label" id="hlpTotals" tabindex="0" aria-describedby="helpTip" data-tip="Cumulative ingest tokens, stacked. Normalized to the run's final total.">Totals (ingest)</span></label>
      <span id="zoomInfo" style="font-size:0.8em;color:#8a9096;"></span>
    </div>
    <div class="controls">
      <button id="ctxFilterBtn" style="border-color:#7C03EC;">Context Filter</button>
      <button id="selectAll">Select All</button>
      <button id="deselectAll">Deselect All</button>
      <button id="downloadRequestsBtn"><span class="help-label" id="hlpDlReqs" tabindex="0" aria-describedby="helpTip" data-tip="CSV of per-request rows (arm, start time, TTFT, response time, tokens, cache hit, error) for the CURRENT view — zoom window, hidden arms, and the context/series filters all apply. Generated in the browser; nothing leaves the page.">Download Requests CSV</span></button>
      <button id="downloadSummaryBtn"><span class="help-label" id="hlpDlSummary" tabindex="0" aria-describedby="helpTip" data-tip="CSV of the summary panel's numbers (same metrics, same baseline ratios) for the CURRENT view.">Download Summary CSV</span></button>
    </div>
    </div>
  </div>
  <div class="panel" id="summaryPanel">
    <div class="panel-title">Summary<span class="range" id="sumRange"></span><span class="spacer"></span><button class="panel-toggle" id="summaryToggle" title="Collapse">&minus;</button></div>
    <div class="panel-body">
      <div class="summary-wrap"><table id="summaryTable"><thead><tr id="sumHead"></tr></thead><tbody id="sumBody"></tbody></table></div>
    </div>
  </div>
</div>
<div id="legend"></div>
</div>
<canvas id="chart"></canvas>
<div id="tooltip"></div>
<div id="helpTip" class="help-tip" role="tooltip"></div>
<div id="ctxModal" class="modal-backdrop">
  <div class="modal">
    <h2>Variants &amp; context filter</h2>
    <div class="modal-actions" style="margin-top:0;">
      <button id="variantSelectAll">Select All</button>
      <button id="variantDeselectAll">Deselect All</button>
    </div>
    <div id="ctxModalSeries"></div>
    <div class="modal-section-title">In-dataset series (empty selection = all)</div>
    <div class="modal-sn-row">
      <input id="snInput" type="text" placeholder="series indices, e.g. 3,7,12-15">
      <button id="snAdd">Add</button>
      <button id="snClear">Clear</button>
      <span id="snCount"></span>
    </div>
    <div id="snList" class="modal-sn-list"></div>
    <div class="modal-sn-note" id="snNote"></div>
    <div class="modal-band">
      <span>Context band (k tokens, inclusive)</span>
      <div class="modal-band-row">
        <span>&ge;</span>
        <input id="ctxMin" type="number" min="0" step="10" placeholder="min">
        <span class="modal-band-sep">&ndash;</span>
        <span>&le;</span>
        <input id="ctxMax" type="number" min="0" step="10" placeholder="max">
        <span>k tok</span>
      </div>
    </div>
    <div class="modal-section-title">Export current view</div>
    <div class="modal-actions" style="margin-top:0;">
      <button id="modalDownloadRequestsBtn"><span class="help-label" id="hlpDlReqsModal" tabindex="0" aria-describedby="helpTip" data-tip="CSV of per-request rows for the view as currently APPLIED — zoom window, hidden arms, and the applied context/series filters. Click Apply first if you just changed the band above.">Download Requests CSV</span></button>
      <button id="modalDownloadSummaryBtn"><span class="help-label" id="hlpDlSummaryModal" tabindex="0" aria-describedby="helpTip" data-tip="CSV of the summary panel's numbers for the view as currently applied.">Download Summary CSV</span></button>
    </div>
    <div class="modal-actions">
      <button id="ctxApply">Apply</button>
      <button id="ctxReset">Reset</button>
      <button id="ctxClose">Close</button>
    </div>
  </div>
</div>

<script>
const RAW_DATA = {{.Data}};

// --- Rehydrate positional records back into the object shape the rest of
// this script expects (everything below reads r.t, r.ttft, r.resp, r.err,
// r.sn, r.rn, r.ch, r.in, r.ca, r.out as object properties). The Go emitter
// (benchmark/visualize.go) writes each record as a positional array
// [t,ttft,resp,err,sn,rn,ch,in,ca,out] (REC_FIELDS order, must match
// vizRecord.MarshalJSON there) with t delta-encoded against a per-series t0,
// instead of repeating 10 JSON key strings per record — across ~137k records
// that was several MB of pure key-name bytes. This is encoding-only: after
// this loop, RAW_DATA[i].records is structurally identical to the
// pre-optimization array-of-objects shape (absolute epoch t, real booleans),
// so nothing below this point needs to change.
const REC_FIELDS = ["t", "ttft", "resp", "err", "sn", "rn", "ch", "in", "ca", "out"];
const REC_BOOL_FIELDS = ["err", "ch"];
RAW_DATA.forEach(s => {
  const t0 = s.t0 || 0;
  s.records = (s.records || []).map(row => {
    const o = {};
    REC_FIELDS.forEach((k, i) => { o[k] = row[i]; });
    o.t += t0;
    REC_BOOL_FIELDS.forEach(k => { o[k] = !!o[k]; });
    return o;
  });
});

const CONCURRENCY = {{.Concurrency}};

// DEFAULT_WINDOW_REQS is the rolling-percentile window used when neither the
// data nor the caller says what concurrency the run held — 96 requests, i.e.
// the 3x window of a c28 + 4-hot run, the shape most of these reports have.
//
// This replaces deriving the window from the observed series count. That
// inference was catastrophically wrong for router-replay runs, where series
// NUMBERS keep climbing as sessions recycle: an 8h replay reached max series
// num ~1100-1800, so the window came out at 3396 and 5358 requests — a
// percentile line smoothed over a 35x-too-wide window, silently. A fixed
// default that is merely approximate beats an inference that is off by orders
// of magnitude, and recorded run params make it moot for new data.
const DEFAULT_WINDOW_REQS = 96;

// --- Sorting: gpu first, dram second, weka last, others alphabetically ---
function getAlias(name) {
  const m = name.match(/alias[_=]([a-zA-Z0-9_-]+)/i);
  return m ? m[1].toLowerCase() : name.toLowerCase();
}
function classifyAlias(a) {
  if (a === "gpu") return "gpu";
  if (/(?:^|[-_])hbm(?:$|[-_])/.test(a) || /^hbm/.test(a)) return "gpu"; // no-offload arms are named "hbm"
  if (/(?:^|[-_])gds(?:$|[-_])/.test(a)) return "weka";
  if (/^weka/.test(a)) return "weka";
  if (/(?:^|[-_])dram(?:$|[-_])/.test(a)) return "dram";
  return "other";
}
function sortKey(name) {
  const c = classifyAlias(getAlias(name));
  if (c === "gpu") return "0_" + name;
  if (c === "dram") return "1_" + name;
  if (c === "weka") return "9_" + name;  // weka last
  return "5_" + name; // others in between, alphabetical
}
const sortedIndices = RAW_DATA.map((_, i) => i);
sortedIndices.sort((a, b) => sortKey(RAW_DATA[a].name).localeCompare(sortKey(RAW_DATA[b].name)));
const DATA = sortedIndices.map(i => RAW_DATA[i]);

// Normalize timestamps: align each series to start at t=0. Metrics-sample
// overlays (mix segments / active-dataset points) shift by the same offset so
// they stay aligned with the series' requests.
DATA.forEach(s => {
  let tMin = Infinity;
  s.records.forEach(r => { if (r.t < tMin) tMin = r.t; });
  if (!isFinite(tMin)) {
    (s.mix || []).forEach(m => { if (m.t0 < tMin) tMin = m.t0; });
    (s.adt || []).forEach(p => { if (p.t < tMin) tMin = p.t; });
  }
  if (!isFinite(tMin)) return;
  s.records.forEach(r => { r.t -= tMin; });
  (s.mix || []).forEach(m => { m.t0 -= tMin; m.t1 -= tMin; });
  (s.adt || []).forEach(p => { p.t -= tMin; });
});

const HAS_CACHE_MIX = DATA.some(s => (s.mix && s.mix.length) || (s.adt && s.adt.length));
// Report-wide band scale: max total delta over ALL series and the whole
// period (function declarations hoist). Independent of series visibility so
// toggling a series never rescales the others.
const MIX_TOTAL_MAX = mixTotalMax(DATA);

// Context-band filter state: when active, every latency/volume computation
// reads the per-series filtered view (s._view) instead of the full record
// set. min/max of 0 mean unbounded. The time axis and the cache-mix
// overlay (server-side aggregates) stay unfiltered by design.
let ctxFilter = { min: 0, max: 0 };
function ctxFilterActive() { return ctxFilter.min > 0 || ctxFilter.max > 0; }
// In-dataset series selection (by series index r.sn, global across
// variants so arm A's session N compares against arm B's session N).
// Empty set = all series. Composes with the context band: kept rows must
// satisfy BOTH.
let snFilter = new Set();

// computeDerived rebuilds every derived structure of a series from its
// current view: cumulative ingest/output (volume layer + hover rates),
// rolling-percentile lines, and error bars. Called once per series at load
// and again on every context-filter change — never per frame, so 90k-row
// datasets stay responsive.
function computeDerived(s) {
  const view = s._view;
  const byT = view.slice().sort((a, b) => a.t - b.t);
  s._byT = byT;
  s._cumTimes = byT.map(r => r.t);
  s._cumTokens = [];
  s._cumOutTokens = []; // cumulative OUTPUT tokens, aligned with _cumTimes
  let ingestAcc = 0, outAcc = 0;
  byT.forEach(r => {
    ingestAcc += (r.in || 0) + (r.ca || 0);
    outAcc += (r.out || 0);
    s._cumTokens.push(ingestAcc);
    s._cumOutTokens.push(outAcc);
  });
  const sorted = view.filter(r => !r.err).slice().sort((a, b) => a.t - b.t);
  s._sorted = sorted;
  // Rolling-percentile window, in requests. Precedence, most trustworthy
  // first: the concurrency this arm RECORDED in its run_params header, then
  // the report-wide --concurrency the caller passed, then DEFAULT_WINDOW_REQS.
  // The per-arm value matters in a merged report whose arms ran at different
  // concurrency — one global number smooths one arm correctly and the other
  // wrongly, with nothing on screen saying so.
  const seriesConc = s.conc > 0 ? s.conc : CONCURRENCY;
  const winSize = seriesConc > 0 ? seriesConc * 3 : DEFAULT_WINDOW_REQS;
  s._winConcSource = s.conc > 0 ? "recorded" : (CONCURRENCY > 0 ? "--concurrency" : "default");
  s._winConc = seriesConc;
  s._winSize = winSize;
  // Plotted lines: rolling-window percentiles. Response = p50 (plus p10/p90
  // for the "ribbon" Requests render mode -- a spread envelope around the
  // same p50 line, same rolling window, computed here alongside it so it can
  // never drift out of sync); TTFT = p50 and p95 (dash pattern encodes the
  // percentile, color the series). recalcYMax deliberately never reads
  // _respP10/_respP90 -- the axis must stay independent of which Requests
  // mode is selected.
  s._respP50 = [];
  s._respP10 = [];
  s._respP90 = [];
  s._ttftP50 = [];
  s._ttftP95 = [];
  // Anchor the rolling-percentile line at ~TARGET_LINE_POINTS x-positions
  // rather than one per request. A per-record anchor makes this loop
  // O(n*winSize) with a fresh winSize-wide slice + 3 sorts allocated every
  // iteration; at 60k+ records/series (137k across the merge) that is billions
  // of ops + heavy GC and was the dominant page-load hang. A 60k-point line on
  // a ~1400px canvas is visually identical to a ~2000-point one, so we stride
  // the ANCHOR while keeping the window content unchanged (still the trailing
  // winSize records ending at i) — ~30x fewer iterations/allocations, no visual
  // change. computeDerived also runs on every context-filter, so filtering
  // stays responsive too.
  const TARGET_LINE_POINTS = 2000;
  const stride = Math.max(1, Math.floor(sorted.length / TARGET_LINE_POINTS));
  const pushAnchor = (i) => {
    const start = Math.max(0, i - winSize + 1);
    const win = sorted.slice(start, i + 1);
    const ttfts = win.map(r => r.ttft).filter(v => v > 0);
    const resps = win.map(r => r.resp);
    const t = sorted[i].t;
    s._respP50.push({ t: t, v: percentile(resps, 0.5) });
    s._respP10.push({ t: t, v: percentile(resps, 0.1) });
    s._respP90.push({ t: t, v: percentile(resps, 0.9) });
    s._ttftP50.push({ t: t, v: ttfts.length ? percentile(ttfts, 0.5) : 0 });
    s._ttftP95.push({ t: t, v: ttfts.length ? percentile(ttfts, 0.95) : 0 });
  };
  for (let i = 0; i < sorted.length; i += stride) pushAnchor(i);
  // Always anchor the final record so the line reaches the true end of the run
  // even when stride does not land on it.
  if (stride > 1 && sorted.length > 0 && (sorted.length - 1) % stride !== 0) {
    pushAnchor(sorted.length - 1);
  }
  // Error bars: sample every winSize points from the view (including
  // errors), each bar anchored at the response p50 line at that time.
  s._errBars = [];
  let avgIdx = 0;
  for (let i = winSize - 1; i < byT.length; i += winSize) {
    const start = Math.max(0, i - winSize + 1);
    let errs = 0, total = 0;
    for (let j = start; j <= i; j++) { total++; if (byT[j].err) errs++; }
    if (errs > 0) {
      const t = byT[i].t;
      while (avgIdx < s._respP50.length - 1 && s._respP50[avgIdx].t < t) avgIdx++;
      const respAvg = s._respP50.length > 0 ? s._respP50[Math.min(avgIdx, s._respP50.length - 1)].v : 0;
      s._errBars.push({ t: t, errRate: errs / total, errs: errs, total: total, respAvg: respAvg });
    }
  }
}

DATA.forEach(s => {
  // Largest single-request context (input+cached) over the FULL session —
  // the legend/chooser hint; deliberately not filter-dependent.
  s._maxCtx = 0;
  s.records.forEach(r => {
    const ctx = (r.in || 0) + (r.ca || 0);
    if (ctx > s._maxCtx) s._maxCtx = ctx;
  });
  s._view = s.records;
  computeDerived(s);
});

// applyCtxFilter re-derives every series against a context band (input+
// cached tokens per request) and redraws. 0 = unbounded on either side.
function applyCtxFilter(minTok, maxTok) {
  ctxFilter = { min: minTok > 0 ? minTok : 0, max: maxTok > 0 ? maxTok : 0 };
  const anyFilter = ctxFilterActive() || snFilter.size > 0;
  DATA.forEach(s => {
    s._view = anyFilter
      ? s.records.filter(r => ctxInBand(r, ctxFilter.min, ctxFilter.max) &&
          (snFilter.size === 0 || snFilter.has(r.sn)))
      : s.records;
    computeDerived(s);
  });
  statsEpoch++; // every series' view changed — invalidate the summary cache
  const btn = document.getElementById("ctxFilterBtn");
  if (btn) {
    let label = "Context Filter";
    const parts = [];
    if (ctxFilter.min > 0) parts.push(">=" + fmtTokens(ctxFilter.min));
    if (ctxFilter.max > 0) parts.push("<=" + fmtTokens(ctxFilter.max));
    if (snFilter.size > 0) parts.push(snFilter.size + " series");
    if (parts.length) label += " (" + parts.join(", ") + ")";
    btn.textContent = label;
  }
  recalcYMax();
  draw();
}

// --- Color assignment (official WEKA brand, neutral-first) ---
// Multi-series palettes are restrained tints DERIVED from the brand set so
// a 2-8 series report reads as one family: weka arms a purple ramp
// (#7C03EC/#C79FF1 family), hbm/gpu neutral slate (the brand has no blue,
// so it stays desaturated), dram a dimmed #6BE0A0 green, the fallback
// dimmed yellow/green/slate tints. Position and the legend carry identity;
// full saturation belongs to the external-KV accent alone, and #FF6B6B is
// reserved for errors.
const OTHER_PALETTE = [
  "#ab9a64","#8a95a3","#a678e8","#5f987d","#b98f83","#c2b078",
  "#8d5fd6","#707c8a","#75b094","#d0b3f5","#9aa5b1","#c9b8e8",
];
const GPU_VARIANTS = ["#8094b5","#68809f","#9cb0cb","#54687f","#aec3da"]; // muted slate-BLUE (perceptible cool lean, restraint kept)
const DRAM_VARIANTS = ["#5f987d","#4d8069","#75b094","#3f6a57","#8cc4a9"];
const WEKA_VARIANTS = [
  "#a678e8","#C79FF1","#8d5fd6","#b58ff0","#7745c0","#d0b3f5",
];

const seriesColors = [];
{
  let gpuIdx = 0, dramIdx = 0, wekaIdx = 0, otherIdx = 0;
  DATA.forEach(s => {
    const c = classifyAlias(getAlias(s.name));
    if (c === "gpu") {
      seriesColors.push(GPU_VARIANTS[gpuIdx % GPU_VARIANTS.length]);
      gpuIdx++;
    } else if (c === "dram") {
      seriesColors.push(DRAM_VARIANTS[dramIdx % DRAM_VARIANTS.length]);
      dramIdx++;
    } else if (c === "weka") {
      seriesColors.push(WEKA_VARIANTS[wekaIdx % WEKA_VARIANTS.length]);
      wekaIdx++;
    } else {
      seriesColors.push(OTHER_PALETTE[otherIdx % OTHER_PALETTE.length]);
      otherIdx++;
    }
  });
}

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const helpTip = document.getElementById("helpTip");
// helpTriggers collects every ".help-label" hover-help element so the
// tooltip wiring near the bottom of this script can attach to all of them
// the same way, whether they came from the static markup (control-layer
// labels, grabbed here by id) or are built later in JS (summary column
// headers, the Cache Mix toggle -- each pushes itself in as it's created).
const helpTriggers = ["hlpTtft50", "hlpTtft95", "hlpResp", "hlpReqs", "hlpErrors", "hlpTotals",
    "hlpDlReqs", "hlpDlSummary", "hlpDlReqsModal", "hlpDlSummaryModal"]
  .map(id => document.getElementById(id)).filter(Boolean);

// totalsAxisLabel: the rotated title for the Totals-layer right axis (see
// drawTotalsAxis, later in this script). Canvas text can't itself be a hover
// target, so this is a real DOM node -- created once, positioned every
// draw() to sit over the canvas at the axis's location, hidden (display:none)
// whenever that axis isn't drawn. Pushed into helpTriggers (BEFORE the
// listener-attach loop near the bottom of this script) so it gets the exact
// same hover/focus tooltip wiring as every other .help-label here.
const totalsAxisLabel = document.createElement("span");
totalsAxisLabel.id = "hlpTotalsAxis";
totalsAxisLabel.className = "help-label totals-axis-label";
totalsAxisLabel.tabIndex = 0;
totalsAxisLabel.ariaDescribedBy = "helpTip";
totalsAxisLabel.dataset.tip = "Cumulative input tokens processed, stacked across visible arms. Right axis.";
totalsAxisLabel.textContent = "Cumulative ingest tokens";
totalsAxisLabel.style.display = "none";
// Parented under .controls purely so it exists in the DOM somewhere real
// (matching how the Cache Mix toggle label is appended below) -- position:
// fixed takes it out of that flex layout entirely, so it never affects the
// controls row; drawTotalsAxis() places it by absolute viewport coordinates.
document.querySelector(".controls").appendChild(totalsAxisLabel);
helpTriggers.push(totalsAxisLabel);

const margin = { top: 30, right: 20, bottom: 50, left: 70 };

// Per-tick cumulative request/error counts are no longer printed under the
// axis (that was the "X-axis values" layer, removed). Hovering near an
// x-axis gridline on a long view still surfaces the same per-series
// breakdown via tooltip (see tickHover below) -- ANNOTATION_ROWS_MAX_DURATION
// is that fallback's activation threshold.
const ANNOTATION_ROWS_MAX_DURATION = 3600;
let W, H, plotW, plotH;
// mixReserveH is the vertical px the cache-mix band block claims at the TOP
// of the plot, when at least one latency layer is visible -- see mapY() and
// computeMixReserveH(). Recomputed once per draw() into this module-level
// var rather than inside mapY itself: mapY runs per plotted point (tens of
// thousands of calls per draw) and cacheMixLayout() does real layout work,
// so it must not run per point. 0 keeps mapY's old full-plot-height mapping
// (cache mix off, or claiming the whole plot because no latency layer is on).
let mixReserveH = 0;
let hiddenSeries = new Set();
// { cb, i } pairs for the context-filter modal's per-variant checkboxes,
// (re)populated by rebuildList() on each modal open. renderState() writes
// through this array so a Select All/Deselect All click (or any other
// hiddenSeries mutation) is reflected the moment the modal is next drawn,
// even though the rows themselves are rebuilt from scratch each open.
let modalRowCheckboxes = [];
// Positions of the x-axis ticks drawn in the most recent draw() call, so
// mousemove can hover-match a tick even when its annotation row isn't
// printed (long views -- see ANNOTATION_ROWS_MAX_DURATION).
let currentTicks = [];

// Zoom state
let globalTMin = Infinity, globalTMax = -Infinity;
let viewTMin, viewTMax, viewYMax;
let dragStart = null; // pixel X where drag began
let dragCurrent = null;

// calcBottomMargin: fixed room for the time-axis tick labels ("0s", "30m",
// "1h", ...) drawn at margin.top + plotH + 6 in 11px monospace (see draw()'s
// X-axis section) -- no per-arm content lives down here anymore, so this is
// just enough vertical space for that one line of text.
function calcBottomMargin() {
  return 20;
}

// TOTALS_AXIS_TARGET_STEPS: tick count target shared by calcRightMargin
// (sizing the margin) and drawTotalsAxis (drawing into it) -- one constant so
// the two can never disagree about how many ticks the axis actually gets.
const TOTALS_AXIS_TARGET_STEPS = 5;

// calcRightMargin mirrors calcBottomMargin above: the right axis (the Totals
// ingest-token scale added by drawTotalsAxis) only claims plot width when
// that layer is actually contributing pixels -- volumeGeometry() is the
// EXACT SAME gate drawTotals() itself uses (checkbox on AND at least one
// visible series carries ingest data), so the axis, its margin, and the
// stack it labels all appear and disappear together. Sized to the widest
// tick label this run will actually draw (same niceSteps/fmtTokens pair
// drawTotalsAxis uses) plus room for the tick mark, a small gap, and the
// rotated title -- otherwise 20, the original always-on right gutter.
function calcRightMargin() {
  const geo = volumeGeometry();
  if (!geo) return 20;
  ctx.font = "11px monospace";
  let maxW = 0;
  niceSteps(geo.finalTotal, TOTALS_AXIS_TARGET_STEPS).forEach(v => {
    maxW = Math.max(maxW, ctx.measureText(fmtTokens(v)).width);
  });
  return Math.ceil(maxW) + 4 /* tick mark */ + 8 /* label gap */ + 6 /* breathing room */ + 18 /* rotated title */;
}

function resize() {
  W = Math.min(window.innerWidth - 32, 1800);
  // Measure the header instead of assuming a fixed chrome allowance: the
  // controls/summary grid reflows with variant count and viewport width (it
  // stacks to one column under 1100px), so a hardcoded reserve would either
  // clip the canvas or leave a gap.
  const headEl = document.getElementById("header");
  const headH = headEl ? headEl.getBoundingClientRect().height : 160;
  H = Math.max(420, Math.min(window.innerHeight - headH - 40, 800));
  margin.bottom = calcBottomMargin();
  margin.right = calcRightMargin();
  // Re-read devicePixelRatio on every resize() rather than closing over a
  // module-level constant: dragging the window to a display with a
  // different pixel density fires the window "resize" listener but a
  // stale dpr would leave canvas.width/height (and the ctx transform)
  // matched to the OLD display until a full page reload.
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  plotW = W - margin.left - margin.right;
  plotH = H - margin.top - margin.bottom;
}

// Compute global ranges. The latency y-axis is NOT derived here -- it comes
// from the rolling percentile lines, which only exist after computeDerived(),
// so it's left to recalcYMax() below once the view bounds are set.
DATA.forEach(s => {
  s.records.forEach(r => {
    if (r.t < globalTMin) globalTMin = r.t;
    if (r.t > globalTMax) globalTMax = r.t;
  });
  // Cache-mix samples extend the time range (a final sample can land after
  // the last request completes) but never the latency axis.
  (s.mix || []).forEach(m => {
    if (m.t0 < globalTMin) globalTMin = m.t0;
    if (m.t1 > globalTMax) globalTMax = m.t1;
  });
  (s.adt || []).forEach(p => {
    if (p.t < globalTMin) globalTMin = p.t;
    if (p.t > globalTMax) globalTMax = p.t;
  });
});
if (globalTMax === globalTMin) globalTMax = globalTMin + 1000;
viewTMin = globalTMin; viewTMax = globalTMax;
recalcYMax();

function isZoomed() { return viewTMin !== globalTMin || viewTMax !== globalTMax; }

// windowStats reduces a series' current view to everything the header needs,
// in ONE pass: request counts, token volumes, and the record-time extent used
// as the rate denominator. Called once per series per draw() (via updateInfo),
// including during a zoom drag, so it must stay single-pass — the old
// countRecords did the same walk for counts alone.
//
// Token semantics, per the net-of-cache contract in
// benchmark/replay_router_post.go (buildReplayUsage): r.in EXCLUDES cached
// tokens and r.ca is the cached subset, so total prompt tokens the server
// processed = in + ca. That matches the "ingest" volume layer, which sums the
// same pair. Volumes include errored requests (a failed request still cost
// its prompt); "completed" counts only non-errors.
function windowStats(records) {
  let ok = 0, err = 0, inTok = 0, caTok = 0, outTok = 0;
  let tFirst = Infinity, tLast = -Infinity;
  // TTFT percentiles come from non-error requests that actually reported a
  // first token, matching the plotted rolling-percentile lines exactly — so a
  // zoomed summary agrees with the curve it sits above.
  const ttfts = [];
  records.forEach(r => {
    if (r.t < viewTMin || r.t > viewTMax) return;
    if (r.err) err++; else {
      ok++;
      if (r.ttft > 0) ttfts.push(r.ttft);
    }
    inTok += (r.in || 0);
    caTok += (r.ca || 0);
    outTok += (r.out || 0);
    if (r.t < tFirst) tFirst = r.t;
    if (r.t > tLast) tLast = r.t;
  });
  // One numeric sort serves both percentiles. Float64Array.sort is numeric by
  // default and materially faster than a comparator on a plain array, which
  // matters at 55k+ rows per series.
  const sortedTtft = Float64Array.from(ttfts).sort();
  const total = ok + err;
  // Rate denominator: the variant's own first->last record inside the window,
  // not the window width. Zooming past where an arm's data ends would
  // otherwise dilute its rate with empty time it never ran through. Falls
  // back to the window width when the extent is degenerate (0 or 1 record).
  let spanSec = total > 1 ? (tLast - tFirst) / 1000 : 0;
  if (spanSec <= 0) spanSec = Math.max((viewTMax - viewTMin) / 1000, 1);
  return {
    ok, err, total, inTok, caTok, outTok, prompt: inTok + caTok, spanSec,
    ttft50: percentileSorted(sortedTtft, 0.5),
    ttft95: percentileSorted(sortedTtft, 0.95),
    ttftN: sortedTtft.length,
  };
}

function countRecords(records) {
  const s = windowStats(records);
  return { ok: s.ok, err: s.err, total: s.total };
}

function formatCount(ok, err) {
  if (err === 0) return '<span style="color:#C9C9C9">' + ok + '</span>';
  return '<span style="color:#C9C9C9">' + ok + '</span>, <span style="color:#FF6B6B">' + err + '</span>';
}

// windowStats is a full pass over every series (plus a sort for the TTFT
// percentiles), while draw() runs on every mousemove of a zoom drag — during
// which the VIEW does not move, only the selection rectangle. Cache the pass
// against the view bounds and a filter epoch so dragging repaints for free.
let statsCache = null, statsCacheKey = "";
let statsEpoch = 0;
function seriesStats() {
  const key = viewTMin + "|" + viewTMax + "|" + statsEpoch;
  if (statsCache && statsCacheKey === key) return statsCache;
  statsCache = DATA.map(s => windowStats(s._view));
  statsCacheKey = key;
  return statsCache;
}

function updateInfo() {
  const perSeries = seriesStats();
  renderSummary(perSeries);

  // Update legend counts
  DATA.forEach((s, i) => {
    const countEl = document.getElementById("legend-count-" + i);
    if (countEl) countEl.innerHTML = "(" + formatCount(perSeries[i].ok, perSeries[i].err) + ")";
  });

  // No Reset button: Esc and double-click are the exits, so the hint has to
  // carry the affordance the button used to provide.
  document.getElementById("zoomInfo").textContent = isZoomed() ? "Esc or double-click to exit zoom" : "Drag on chart to zoom into timeframe";
}

// Legend
const legendEl = document.getElementById("legend");
const legendItems = [];
DATA.forEach((s, i) => {
  const ok = s.records.filter(r => !r.err).length;
  const err = s.records.filter(r => r.err).length;
  const item = document.createElement("div");
  item.className = "legend-item";
  item.dataset.index = i;
  item.dataset.name = s.name.toLowerCase();
  const ctxHint = s._maxCtx > 0 ? ' <span class="legend-ctx">max ctx ' + fmtTokens(s._maxCtx) + '</span>' : '';
  item.innerHTML = '<div class="legend-dot" style="background:' + seriesColors[i] + '"></div>' +
    s.name + ' <span class="legend-count" id="legend-count-' + i + '">(' + formatCount(ok, err) + ')</span>' + ctxHint;
  item.onclick = () => {
    if (hiddenSeries.has(i)) hiddenSeries.delete(i); else hiddenSeries.add(i);
    renderState();
  };
  legendEl.appendChild(item);
  legendItems.push(item);
});

// renderState is the single authoritative sync from hiddenSeries (the one
// piece of state a variant's visibility actually lives in) to every UI
// surface that depends on it -- legend chips, the context-filter modal's
// per-variant checkboxes (if the modal has been opened at least once), and
// the summary panel's row dimming -- followed by a full recompute/redraw
// (the latency axis must exclude hidden series, so it needs a recalc too).
// Call this from every path that mutates hiddenSeries instead of hand-
// rolling a partial update: previously three different call sites each
// nudged a subset of these (syncLegendVisuals only touched the legend, the
// modal checkboxes were written once on open and then went stale, and the
// top-level buttons only ever drove the legend), which is exactly how the
// modal and the legend drifted out of sync with each other. Uses the
// two-arg classList.toggle(cls, force) form throughout so every element is
// SET to the current state rather than flipped relative to its own history.
function renderState() {
  legendItems.forEach((item, i) => {
    item.classList.toggle("hidden", hiddenSeries.has(i));
  });
  modalRowCheckboxes.forEach(({ cb, i }) => {
    cb.checked = !hiddenSeries.has(i);
  });
  sumRows.forEach((tr, i) => {
    tr.classList.toggle("row-hidden", hiddenSeries.has(i));
  });
  recalcYMax();
  draw();
}

// --- Summary panel: per-variant totals over the SELECTED timeframe ---
// One ROW per variant, one column per metric: a report can carry ~10 arms but
// the metric set is fixed, so growth is vertical and the pane scrolls
// down a stable set of columns rather than sideways past the labels.
// Everything here honours the current view: the zoom window, the context
// band, and the in-dataset series selection (via s._view) — so zooming into
// an interesting stretch reprices every number. The skeleton is built once
// and only cell text is rewritten per draw, so a zoom drag never rebuilds DOM.
// Headers are abbreviated to keep all eight columns inside the pane; each
// carries the full wording as a hover title.
// val is the raw number behind the formatted cell, used for the
// ratio-to-baseline; better says which direction is an improvement, which
// decides the ratio tint (more tokens is good, more latency is not).
const SUMMARY_METRICS = [
  { key: "in",      short: "Input",  label: "Prompt tokens processed, cached plus uncached. The metric KV offload improves directly.", better: "up",
    val: st => st.prompt,               fmt: st => fmtTokens(st.prompt) },
  { key: "out",     short: "Output", label: "Generated tokens. A property of the workload, not something offload changes.", better: "up",
    val: st => st.outTok,               fmt: st => fmtTokens(st.outTok) },
  { key: "reqs",    short: "Reqs",   label: "Completed non-error requests in this window.", better: "up",
    val: st => st.ok,                   fmt: st => st.ok.toLocaleString() },
  { key: "inrate",  short: "In/s",   label: "Input tokens per second. The headline throughput number.", better: "up",
    val: st => st.prompt / st.spanSec,  fmt: st => fmtTokens(st.prompt / st.spanSec) },
  { key: "outrate", short: "Out/s",  label: "Output tokens per second.", better: "up",
    val: st => st.outTok / st.spanSec,  fmt: st => fmtTokens(st.outTok / st.spanSec) },
  { key: "ttft50",  short: "TTFT50", label: "Median time to first token. Includes retry and queue wait.", better: "down",
    val: st => st.ttft50,               fmt: st => fmtMs(st.ttft50) },
  { key: "ttft95",  short: "TTFT95", label: "95th-percentile time to first token: the tail users notice.", better: "down",
    val: st => st.ttft95,               fmt: st => fmtMs(st.ttft95) },
  { key: "err1k",   short: "Err/1k", label: "Errors per 1,000 requests, so arms with different request counts compare.", better: "down",
    val: st => st.total ? st.err / st.total * 1000 : 0,
    fmt: st => (st.total ? st.err / st.total * 1000 : 0).toFixed(1) },
];

// --- Ratio to the HBM baseline ---
// These reports almost always compare an offload arm against a no-offload
// "hbm" control, and the question asked of them is "how much better/worse than
// hbm?" -- previously answered by dividing two numbers by hand. When the report
// contains an hbm arm AND at least one other, every other row carries each
// metric as a percentage of hbm's. classifyAlias already recognises hbm arms
// (it is what sorts them first), so the naming rule stays in one place.
const BASELINE_INDEX = (function () {
  if (DATA.length < 2) return -1;
  for (let i = 0; i < DATA.length; i++) {
    if (classifyAlias(getAlias(DATA[i].name)) === "gpu") return i;
  }
  return -1;
})();

// fmtRatio renders v as a percentage of base. "" when there is nothing to
// compare against -- a zero baseline (no requests in the window, or a metric
// the baseline never recorded) would otherwise divide to Infinity and read as
// a real measurement.
function fmtRatio(v, base) {
  if (!(base > 0) || !isFinite(v) || v < 0) return "";
  const pct = (v / base) * 100;
  if (pct >= 10) return Math.round(pct) + "%";
  if (pct > 0) return pct.toFixed(1) + "%";
  return "0%";
}
// sumCells[seriesIndex][metricIndex] holds the VALUE span, not the <td>, so a
// cell's text stays the datum alone; the ratio-to-baseline lives in its own
// span alongside it.
const sumCells = [];
const sumRatios = [];
const sumRows = [];
(function buildSummary() {
  const table = document.getElementById("summaryTable");
  // Only reserve the second line per cell when there is something to compare
  // against — see the .has-ratios rule.
  if (table && BASELINE_INDEX >= 0) table.className = "has-ratios";
  const head = document.getElementById("sumHead");
  const vth = document.createElement("th");
  vth.className = "vcol";
  vth.textContent = "Variant";
  head.appendChild(vth);
  SUMMARY_METRICS.forEach(m => {
    const th = document.createElement("th");
    // The abbreviated header text itself is the hover target (dotted
    // underline via .help-label, wired up near the bottom of this script) --
    // an abbreviation alone doesn't visibly invite a hover, so the
    // underline gives it that affordance without spending a "?" glyph's
    // worth of the column's width.
    const span = document.createElement("span");
    span.className = "help-label";
    span.tabIndex = 0;
    span.ariaDescribedBy = "helpTip";
    span.dataset.tip = m.label;
    span.textContent = m.short;
    helpTriggers.push(span);
    th.appendChild(span);
    head.appendChild(th);
  });
  const body = document.getElementById("sumBody");
  DATA.forEach((s, i) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.className = "vcol";
    const wrap = document.createElement("span");
    wrap.className = "summary-head";
    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.background = seriesColors[i];
    const nameText = document.createElement("span");
    nameText.className = "summary-name";
    nameText.textContent = s.name;
    wrap.appendChild(dot);
    wrap.appendChild(nameText);
    if (i === BASELINE_INDEX) {
      const chip = document.createElement("span");
      chip.className = "sum-baseline";
      chip.textContent = "(baseline)";
      wrap.appendChild(chip);
    }
    name.appendChild(wrap);
    // Hover carries the full name plus this arm's recorded workload shape and
    // where its smoothing window came from — the two things you need to know
    // before comparing this row against another.
    const bits = [s.name];
    const ps = paramsSummaryFor(s);
    if (ps) bits.push("params: " + ps);
    bits.push("rolling window: " + (s._winSize || 0) + " reqs (" + s._winConcSource + ")");
    name.title = bits.join("\n");
    tr.appendChild(name);
    const cells = [], ratios = [];
    SUMMARY_METRICS.forEach(() => {
      const td = document.createElement("td");
      const val = document.createElement("span");
      val.className = "sum-val";
      const ratio = document.createElement("span");
      ratio.className = "sum-ratio";
      // Hover-help trigger, same as the column headers above -- wired up
      // (and left inert until renderSummary populates a tip) near the
      // bottom of this script. Only carries the .help-label affordance
      // while it actually holds a ratio; see renderSummary.
      helpTriggers.push(ratio);
      td.appendChild(val);
      td.appendChild(ratio);
      tr.appendChild(td);
      cells.push(val);
      ratios.push(ratio);
    });
    // A row IS a variant here, so it carries the same show/hide affordance as
    // the legend entry — with 10 arms the summary is where you're looking.
    tr.onclick = () => {
      if (hiddenSeries.has(i)) hiddenSeries.delete(i); else hiddenSeries.add(i);
      renderState();
    };
    sumCells.push(cells);
    sumRatios.push(ratios);
    sumRows.push(tr);
    body.appendChild(tr);
  });
})();

// renderSummary repaints the panel from the per-series windowStats already
// computed by updateInfo — no extra pass over the records.
function renderSummary(perSeries) {
  const baseStats = BASELINE_INDEX >= 0 ? perSeries[BASELINE_INDEX] : null;
  DATA.forEach((s, si) => {
    const st = perSeries[si];
    SUMMARY_METRICS.forEach((m, mi) => {
      // sumCells holds the VALUE span (see buildSummary), not the <td>.
      const valEl = sumCells[si][mi];
      valEl.textContent = st ? m.fmt(st) : "-";
      // Ratio to the hbm baseline, on every row but the baseline's own.
      const rEl = sumRatios[si][mi];
      let rText = "";
      if (baseStats && st && si !== BASELINE_INDEX) {
        rText = fmtRatio(m.val(st), m.val(baseStats));
      }
      rEl.textContent = rText;
      if (rText) {
        const ratio = m.val(st) / m.val(baseStats);
        const good = m.better === "up" ? ratio > 1 : ratio < 1;
        // help-label only while there IS a ratio to explain -- an empty
        // cell (the baseline's own row) gets no dotted-underline/help-cursor
        // affordance and no tab stop, since there's nothing to hover.
        rEl.className = "sum-ratio help-label " + (ratio === 1 ? "" : (good ? "up" : "down"));
        rEl.tabIndex = 0;
        rEl.ariaDescribedBy = "helpTip";
        rEl.dataset.tip = "Share of the HBM baseline. Green is better, orange is worse. " +
          m.short + " is " + rText + " of " + DATA[BASELINE_INDEX].name;
      } else {
        rEl.className = "sum-ratio";
        rEl.tabIndex = -1;
        rEl.dataset.tip = "";
      }
      // The cached share is what a KV-offload arm actually buys, so it rides
      // along as a hover on the input cell instead of costing a column.
      if (m.key === "in" && st) {
        valEl.title = "prompt tokens = " + fmtTokens(st.inTok) + " uncached + " +
          fmtTokens(st.caTok) + " server-cached" +
          (st.prompt > 0 ? " (" + (st.caTok / st.prompt * 100).toFixed(1) + "% cached)" : "");
      }
      // Percentiles over a thin window are noise — say how many requests back
      // them rather than presenting 3 samples as a p95.
      if ((m.key === "ttft50" || m.key === "ttft95") && st) {
        valEl.title = st.ttftN + " non-error requests reported a first token in this window";
      }
      if (m.key === "err1k") valEl.classList.toggle("err-hot", !!st && st.err > 0);
    });
    if (sumRows[si]) sumRows[si].classList.toggle("row-hidden", hiddenSeries.has(si));
  });
  // A summary without its timeframe stated is a trap — always label the range.
  const rangeEl = document.getElementById("sumRange");
  if (rangeEl) {
    const off = t => formatTickLabel(Math.round((t - globalTMin) / 1000));
    rangeEl.textContent = isZoomed()
      ? off(viewTMin) + " – " + off(viewTMax)
      : "full run (" + off(globalTMax) + ")";
  }
}

// --- Recorded run parameters ---
// Runs written by a wekai that records run_params carry their own workload
// shape (concurrency, hot pool, series, timeout, source). Reports built from
// older data simply have none, and everything below degrades to the previous
// behaviour rather than inventing values.
function paramsSummaryFor(s) {
  return s.params && s.params.summary ? s.params.summary : "";
}
(function renderRunParams() {
  const el = document.getElementById("runparams");
  if (!el) return;
  const summaries = DATA.map(paramsSummaryFor);
  const withParams = summaries.filter(x => x);
  if (!withParams.length) {
    // Nothing recorded: spend no vertical space saying so. The line would
    // carry no information beyond its own absence, and the chart wants the
    // pixels. Provenance isn't lost — every summary row's hover still states
    // the window size and where it came from.
    el.style.display = "none";
    return;
  }
  const distinct = Array.from(new Set(withParams));
  if (distinct.length === 1 && withParams.length === DATA.length) {
    el.textContent = "run params: " + distinct[0];
  } else {
    // Arms that don't share a workload shape aren't directly comparable; say
    // it here rather than letting the reader assume they are.
    el.textContent = "run params differ per variant (" + withParams.length + "/" + DATA.length +
      " recorded) — hover a summary row";
  }
})();

// Panel collapse: hand the panel's vertical space back to the chart. resize()
// measures #header, so hiding a body immediately grows the canvas.
function wirePanelToggle(btnId, panelId) {
  const btn = document.getElementById(btnId);
  const panel = document.getElementById(panelId);
  if (!btn || !panel) return;
  btn.addEventListener("click", () => {
    const collapsed = panel.className.indexOf("collapsed") >= 0;
    panel.className = collapsed ? "panel" : "panel collapsed";
    btn.textContent = collapsed ? "−" : "+";
    btn.title = collapsed ? "Collapse" : "Expand";
    resize();
    draw();
  });
}
wirePanelToggle("controlsToggle", "controlsPanel");
wirePanelToggle("summaryToggle", "summaryPanel");

// Top-level Select All/Deselect All are a master on/off for the PLOTTED
// LAYER checkboxes (not variants -- that control moved into the
// context-filter modal, next to the per-variant checkboxes it actually
// governs; see the ctxModal block below). "X-axis values" is an annotation
// toggle, not a plotted layer, and is deliberately excluded. Cache Mix only
// exists in the DOM when the dataset carries samples, hence the guard.
const LAYER_CHECKBOX_IDS = ["showTTFT", "showTTFTP95", "showResp", "showDots", "showErrors", "showTotals"];
function setAllLayers(on) {
  LAYER_CHECKBOX_IDS.concat(HAS_CACHE_MIX ? ["showCacheMix"] : []).forEach(id => {
    const cb = document.getElementById(id);
    if (cb) cb.checked = on;
  });
  recalcYMax();
  draw();
}
document.getElementById("selectAll").addEventListener("click", () => setAllLayers(true));
document.getElementById("deselectAll").addEventListener("click", () => setAllLayers(false));

function mapX(t) { return margin.left + ((t - viewTMin) / (viewTMax - viewTMin)) * plotW; }
function unmapX(px) { return viewTMin + ((px - margin.left) / plotW) * (viewTMax - viewTMin); }

function mapY(v) {
  const top = margin.top + mixReserveH;
  const h = Math.max(plotH - mixReserveH, 1);
  return top + h - (v / viewYMax) * h;
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

// recalcYMax scales the latency y-axis. Must be called after every change
// that can move the dominant series: zoom, context/series filter, or
// hidden-series toggle -- otherwise the axis keeps scaling to a series
// that's no longer (or wasn't yet) on screen, or mapY runs against a stale
// max.
//
// The max is taken from the ROLLING LINES actually drawn -- resp/TTLT p50,
// TTFT p50, TTFT p95 -- DELIBERATELY IGNORING which of those layers is
// currently checked (ticking a layer must reveal a line, never rescale the
// chart from under you) and DELIBERATELY EXCLUDING raw per-request values
// and the p10-p90 ribbon. On a long run the tallest raw value tends to be a
// request that hit the timeout, well above anything the percentile lines
// reach, so including it stands the axis far above the readable range and
// pins TTFT p50 flat against the bottom. Totals and cache-mix are normalized on their own
// scales and never reach this axis.
function recalcYMax() {
  viewYMax = 0;
  const bump = v => { if (v > viewYMax) viewYMax = v; };
  const scan = pts => {
    (pts || []).forEach(p => {
      if (p.t < viewTMin || p.t > viewTMax) return;
      bump(p.v);
    });
  };
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    scan(s._respP50);
    scan(s._ttftP50);
    scan(s._ttftP95);
  });
  viewYMax = Math.max(viewYMax * 1.1, 1);
}

// resetZoomView is the shared "exit zoom" action -- back to the full time
// range, latency axis rescaled to the widest window (the axis follows the
// time window, never the layer checkboxes -- see recalcYMax).
// Used by the Reset Zoom button, double-click, and ESC.
function resetZoomView() {
  viewTMin = globalTMin;
  viewTMax = globalTMax;
  recalcYMax();
  draw();
}

// computeXStepSec picks the x-axis tick interval (seconds) for a given view
// duration, targeting ~12-20 visible ticks regardless of run length. A flat
// step beyond 10 minutes used to emit a tick every 5m no matter how long the
// run was (48 mashed-together labels on a 4h/14400s span) -- this scales the
// step with the span instead.
function computeXStepSec(duration) {
  if (duration <= 10) return 1;
  if (duration <= 30) return 2;
  if (duration <= 60) return 5;
  if (duration <= 300) return 30;
  if (duration <= 600) return 60;
  if (duration <= 1800) return 120;   // <=30m: 2m ticks
  if (duration <= 3600) return 300;   // <=1h: 5m ticks
  if (duration <= 7200) return 600;   // <=2h: 10m ticks
  if (duration <= 14400) return 900;  // <=4h: 15m ticks
  if (duration <= 28800) return 1800; // <=8h: 30m ticks
  return 3600;                        // beyond 8h: 1h ticks
}

// formatTickLabel renders a compact label for a tick offset of s seconds
// (e.g. "45m", "1h15m", "2h" -- never a redundant "1h0m").
function formatTickLabel(s) {
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m" + (s % 60 ? (s % 60) + "s" : "");
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h + "h" + (m ? m + "m" : "");
}

// tickStats computes, for each visible series, the cumulative request/error
// count and max series-num as of tickTime. Shared by the always-printed
// annotation rows (short/zoomed views) and the hover tooltip (long views).
function tickStats(tickTime) {
  const out = [];
  DATA.forEach((ds, si) => {
    if (hiddenSeries.has(si)) return;
    let cumReqs = 0, cumErrs = 0, maxSn = 0;
    ds._view.forEach(r => { if (r.t <= tickTime) { cumReqs++; if (r.err) cumErrs++; if (r.sn > maxSn) maxSn = r.sn; } });
    out.push({ si, cumReqs, cumErrs, maxSn });
  });
  return out;
}

// --- Cache-mix overlay (vLLM prompt-source metrics samples) ---
// Per-minute source mix rendered as one horizontal band per sampled series,
// stacked from the top of the plot area at 30% opacity. Colors are constant:
// compute=red, local prefix cache=green, external KV transfer=purple. The
// active-dataset token line is drawn inside each band against a shared scale.
// Neutral-first source triad on the black band backdrop, mapped to the
// official brand: external KV = #7C03EC primary purple (THE accent — the
// one vivid element in the report), local cache = dimmed #C79FF1 family
// (calm dominant mass), compute = dimmed #FF8569 warning orange (the cost
// signal). Passes lightness/CVD/contrast checks on #000; the two muted
// fills sit below the chroma floor by design.
const MIX_COMPUTE_COLOR = "#a86853";
const MIX_LOCAL_COLOR = "#756a99";
const MIX_EXTERNAL_COLOR = "#7C03EC"; // official primary purple — the single high-chroma accent
const ADT_LINE_COLOR = "#F2F2EB";
const MIX_BAND_H = 64;
// Band fills sit on a solid-black backdrop; slightly translucent so the
// (muted) band mass sits visually behind the latency lines — the plot reads
// first, the band second.
const MIX_FILL_ALPHA = 0.78;

// DOM-free helpers (fmtTokens/mixAt/adtAt/mixTotalMax/mixStackHeight/
// mixRate/placeTooltip): unit-tested under node by
// TestCacheMixLookupHelpersJS, which slices the emitted script from
// "function fmtTokens(" to "function cacheMixEnabled(" — keep them
// contiguous and ahead of cacheMixEnabled (html/template strips these
// comments from the output, so the test can't anchor on markers).
function fmtTokens(v) {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return "" + Math.round(v);
}

// fmtMs renders a latency compactly for the summary table; 0/absent reads as
// "-" rather than "0ms", so a window with no completed request can't be
// mistaken for an instant one.
function fmtMs(v) {
  if (!(v > 0)) return "-";
  if (v >= 1000) return (v / 1000).toFixed(2) + "s";
  return Math.round(v) + "ms";
}

// mixAt returns the sample interval covering t, or — when t is past the last
// interval — the latest interval at or before t. null when t precedes the
// first interval (no data yet at that point in the run).
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

// percentile of an unsorted array (nearest-rank); 0 for empty input.
// (Declared here, in the DOM-free helper block, so node tests cover it;
// hoisting makes it available to the precompute pass above.)
function percentile(arr, p) {
  if (!arr || arr.length === 0) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  const idx = Math.ceil(s.length * p) - 1;
  return s[Math.max(0, idx)];
}

// percentileSorted is percentile() for an ALREADY-sorted sequence — same
// nearest-rank definition, no copy and no re-sort. The summary panel sorts a
// window's TTFTs once and reads several percentiles off that one sort.
function percentileSorted(sorted, p) {
  if (!sorted || sorted.length === 0) return 0;
  const idx = Math.ceil(sorted.length * p) - 1;
  return sorted[Math.min(Math.max(idx, 0), sorted.length - 1)];
}

// percentileDash: canvas dash pattern per plotted line kind — the pattern
// encodes the percentile, the color encodes the series; the two channels
// never mix. Response p50 solid; TTFT p50 dense dots (primary); TTFT p95
// long sparse dashes (secondary).
function percentileDash(kind) {
  if (kind === "ttft50") return [2, 3];
  if (kind === "ttft95") return [10, 7];
  return []; // resp50: solid
}

// ctxInBand: whether a request row's context size (input+cached tokens)
// falls inside the [minTok, maxTok] band; 0 means unbounded on that side,
// and both bounds are inclusive.
function ctxInBand(rec, minTok, maxTok) {
  const c = ((rec && rec.in) || 0) + ((rec && rec.ca) || 0);
  if (minTok > 0 && c < minTok) return false;
  if (maxTok > 0 && c > maxTok) return false;
  return true;
}

// parseSnList parses a series-index list like "3, 7, 12-15" into sorted
// unique indices; malformed tokens are ignored, ranges are inclusive.
function parseSnList(text) {
  const out = new Set();
  String(text || "").split(",").forEach(tok => {
    tok = tok.trim();
    if (!tok) return;
    const m = tok.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) {
      let a = parseInt(m[1], 10), b = parseInt(m[2], 10);
      if (a > b) { const t = a; a = b; b = t; }
      if (b - a > 100000) return; // reject absurd ranges
      for (let i = a; i <= b; i++) out.add(i);
      return;
    }
    if (/^\d+$/.test(tok)) out.add(parseInt(tok, 10));
  });
  return Array.from(out).sort((a, b) => a - b);
}

// cumCountAt returns how many sorted timestamps are <= t (the cumulative
// completed-request count of a series at time t).
function cumCountAt(times, t) {
  if (!times || !times.length) return 0;
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// totalsY maps a stacked cumulative fraction onto the canvas: the stack
// grows from the plot bottom and fraction 1.0 lands EXACTLY on ceilingY —
// the cache-mix strip's lower edge when the overlay is on (the two layers
// tile the vertical space, never overlap), else the plot top.
function totalsY(frac, plotTop, plotHeight, ceilingY) {
  const bottom = plotTop + plotHeight;
  return bottom - frac * (bottom - ceilingY);
}

// cumTokensAt: cumulative ingest tokens of a series at time t, from the
// sorted times and the aligned cumulative-token array (cached tokens
// included by construction).
function cumTokensAt(times, cumTokens, t) {
  const idx = cumCountAt(times, t);
  return idx > 0 ? cumTokens[idx - 1] : 0;
}

// totalsStack: stacked cumulative ingest-token fractions at time t for
// series given in stacking (legend) order — each entry {times, cum} — and
// normalized against finalTotal (the combined FINAL ingest), so the stack
// reaches exactly 1.0 (full available height) at the end of the run and the
// shape is stable under zoom. Callers pass only visible series (and their
// recomputed finalTotal) so hidden series contribute nothing.
function totalsStack(seriesCums, t, finalTotal) {
  const out = [];
  let acc = 0;
  (seriesCums || []).forEach(sc => {
    if (finalTotal > 0) acc += cumTokensAt(sc.times, sc.cum, t) / finalTotal;
    out.push(acc);
  });
  return out;
}

// closestIndex: index of the timestamp nearest to t (ties to the earlier
// one); -1 for empty input.
function closestIndex(times, t) {
  if (!times || !times.length) return -1;
  const idx = cumCountAt(times, t);
  if (idx <= 0) return 0;
  if (idx >= times.length) return times.length - 1;
  return (t - times[idx - 1]) <= (times[idx] - t) ? idx - 1 : idx;
}

// shareOfBest: the hovered series' cumulative ingest as a fraction of the
// LARGEST series' cumulative at the same time point — cross-implementation
// totals have no combined meaning, so shares are expressed against the
// best: the biggest series reads 100%, others their fraction of it (ties
// all read 100%).
function shareOfBest(seriesCums, t, hoveredTok) {
  let best = 0;
  (seriesCums || []).forEach(sc => {
    const v = cumTokensAt(sc.times, sc.cum, t);
    if (v > best) best = v;
  });
  return best > 0 ? hoveredTok / best : 0;
}

// windowRates: requests/s, ingest tokens/s (input+cached) and output
// tokens/s over the trailing windowMs ending at t, from time-ordered
// records; the span clamps to the run start so early-run rates stay honest.
function windowRates(byT, t, windowMs) {
  if (!byT || !byT.length) return null;
  const t0 = Math.max(t - windowMs, byT[0].t);
  let n = 0, inTok = 0, outTok = 0;
  for (let i = byT.length - 1; i >= 0; i--) {
    const r = byT[i];
    if (r.t > t) continue;
    if (r.t < t0) break;
    n++;
    inTok += (r.in || 0) + (r.ca || 0);
    outTok += (r.out || 0);
  }
  // Same degenerate-span problem windowStats() floors above: near a
  // series' first record, t0 clamps to byT[0].t and (t - t0) collapses
  // toward 0, which would blow the /spanS divisions up toward billions of
  // tok/s. Unlike windowStats() (which has a whole-view width to fall back
  // to for its rate denominator), there's no meaningful trailing window to
  // substitute here, so suppress the rate outright rather than floor it to
  // a tiny-but-nonzero span that just produces a smaller huge number — a
  // missing figure beats a wrong one. Callers already treat a null return
  // as "nothing to show" (see the volHover tooltip).
  const spanS = (t - t0) / 1000;
  if (spanS < 1) return null;
  return { n: n, rps: n / spanS, inPerSec: inTok / spanS, outPerSec: outTok / spanS, spanS: spanS };
}

// adtAt returns the latest active-dataset observation at or before t, or
// null when t precedes the first sample.
function adtAt(adt, t) {
  if (!adt || !adt.length || t < adt[0].t) return null;
  let found = adt[0];
  for (let i = 1; i < adt.length; i++) {
    if (adt[i].t <= t) found = adt[i]; else break;
  }
  return found;
}

// adtWindow returns the active-dataset samples that describe [tMin, tMax]:
// every sample inside the window, PLUS the last one at or before tMin. That
// carry-in point is what makes a zoomed band correct — the dataset size is a
// level that persists between samples, so a window whose first sample lands
// well inside it (or which contains no sample at all) would otherwise start
// from the wrong value, or render nothing. The carry-in keeps its real t;
// callers clamp x to the band edge when drawing.
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

// adtWindowMax is the largest dataset size observed over adtWindow's points.
function adtWindowMax(pts) {
  let mx = 0;
  (pts || []).forEach(p => { if (p.v > mx) mx = p.v; });
  return mx;
}

// adtWindowRange is the [min, max] the active-dataset line is drawn against —
// BOTH ends taken from the window, which is what makes a zoom legible. The
// dataset is a slowly-growing level, so over a narrow window it might move
// 33.2M -> 35.3M: against a 0-anchored axis that is a flat trace glued to the
// band top, saying nothing about the shape the zoom was meant to reveal.
// Anchoring at the window minimum spends the whole band on the variation
// actually present. Over the full run the minimum is near 0 anyway, so the
// familiar 0-to-peak reading is unchanged. null when there is nothing to draw.
function adtWindowRange(ptsPerBand) {
  let lo = Infinity, hi = -Infinity;
  (ptsPerBand || []).forEach(pts => (pts || []).forEach(p => {
    if (p.v < lo) lo = p.v;
    if (p.v > hi) hi = p.v;
  }));
  if (!isFinite(lo) || !isFinite(hi)) return null;
  return { lo, hi };
}

// mixTotalMax returns the maximum per-interval TOTAL ingested delta
// (compute+local+external) across every series in the report — the single
// shared scale for all bands, deliberately NOT per-series: a series peaking
// at 50k tok/min next to one peaking at 1M renders mostly unfilled.
function mixTotalMax(seriesArr) {
  let mx = 0;
  (seriesArr || []).forEach(s => (s.mix || []).forEach(m => {
    const t = m.c + m.lc + m.ec;
    if (t > mx) mx = t;
  }));
  return mx;
}

// mixStackHeight: absolute stack height for one interval — this interval's
// total delta as a fraction of the global max, of the band height.
function mixStackHeight(seg, globalMax, bandH) {
  const total = seg.c + seg.lc + seg.ec;
  if (total <= 0 || globalMax <= 0) return 0;
  return bandH * (total / globalMax);
}

// mixRate: the interval's ingest rate in input tokens/s, over the ACTUAL
// sample interval (missed ticks widen it; never assume 60s).
function mixRate(seg) {
  const secs = (seg.t1 - seg.t0) / 1000;
  if (secs <= 0) return 0;
  return (seg.c + seg.lc + seg.ec) / secs;
}

// placeTooltip: viewport-aware tooltip position for a cursor at (cx,cy).
// Default is right-of/below-ish the cursor (+12,-10); when the tip would
// cross the right or bottom viewport edge it flips to the other side of the
// cursor, then clamps into [pad, viewport-pad-size] on both axes.
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


function cacheMixEnabled() {
  const cb = document.getElementById("showCacheMix");
  return HAS_CACHE_MIX && cb && cb.checked;
}

// anyPlotLayerVisible: whether at least one layer OTHER than cache mix needs
// room in the plot -- used by cacheMixLayout/computeMixReserveH to decide
// whether the cache-mix bands should keep sharing the plot, or -- when cache
// mix is the only thing left on screen -- expand to claim the whole plot
// height. This is NOT just "the four latency lines" (the ones recalcYMax
// scales the axis to): it must also cover every layer that occupies plot
// space or the mapY() coordinate space, or that layer draws straight through
// (or gets crushed by) a full-height band block:
//   - showErrors: error bars are drawn at mapY(b.respAvg) (see the error-rate
//     bars block in draw()) -- that's the latency coordinate space, so bands
//     claiming the whole plot means bars draw straight through them.
//   - showTotals: the totals/ingest layer needs the region BELOW the bands
//     (volumeGeometry() derives ceilingY from the band block's bottom edge)
//     -- if bands claim the whole plot, that region is crushed to nothing.
// A future reader narrowing this back to "just the latency lines" will
// silently reintroduce both bugs -- don't re-narrow without re-checking both.
function anyPlotLayerVisible() {
  return document.getElementById("showTTFT").checked ||
    document.getElementById("showTTFTP95").checked ||
    document.getElementById("showResp").checked ||
    document.getElementById("showDots").checked ||
    document.getElementById("showErrors").checked ||
    document.getElementById("showTotals").checked;
}

// volumeGeometry: shared state for the ingest volume layer — the visible
// contributing series, the combined final ingest-token total, and the
// stack's ceiling (the cache-mix strip's lower edge when the overlay is on,
// else the plot top). Used by drawTotals and the volume hover so they can't
// disagree. null when the layer is off or empty.
function volumeGeometry() {
  const cb = document.getElementById("showTotals");
  if (!cb || !cb.checked) return null;
  const visible = [];
  DATA.forEach((s, si) => {
    if (!hiddenSeries.has(si) && s._cumTimes && s._cumTimes.length) visible.push({ s, si });
  });
  if (!visible.length) return null;
  let finalTotal = 0;
  visible.forEach(({ s }) => { finalTotal += s._cumTokens[s._cumTokens.length - 1]; });
  if (finalTotal <= 0) return null;
  const layout = cacheMixLayout();
  let ceilingY = margin.top;
  if (layout && layout.bands.length) {
    const last = layout.bands[layout.bands.length - 1];
    ceilingY = last.yTop + last.bandH;
  }
  return { visible: visible, finalTotal: finalTotal, ceilingY: ceilingY };
}

// drawTotals renders the cumulative INGEST-TOKEN "volume" layer: stacked
// translucent areas (one per visible series, legend order, series colors,
// alpha 0.3) behind the latency lines. Normalized to the combined FINAL
// ingest of the visible series = full available height, so the stack fills
// the chart exactly at the end of the run and the biggest contributor
// visibly owns the top of the right edge.
function drawTotals() {
  const geo = volumeGeometry();
  if (!geo) return;
  const { visible, finalTotal, ceilingY } = geo;

  const stepPx = 2;
  const n = Math.max(2, Math.floor(plotW / stepPx) + 1);
  const xs = new Array(n), stacks = new Array(n);
  const seriesCums = visible.map(({ s }) => ({ times: s._cumTimes, cum: s._cumTokens }));
  for (let k = 0; k < n; k++) {
    const px = Math.min(k * stepPx, plotW);
    xs[k] = margin.left + px;
    stacks[k] = totalsStack(seriesCums, unmapX(margin.left + px), finalTotal);
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

// drawTotalsAxis renders the right-hand y-axis for the Totals (ingest) layer,
// in absolute tokens -- ticks/labels MUST use exactly the same geometry as
// drawTotals()'s stack (volumeGeometry() for finalTotal/ceilingY, totalsY()
// for the fraction->pixel mapping), so a tick always lines up with the stack
// it's labelling rather than being re-derived and risking drift. Gated on
// the identical volumeGeometry() check drawTotals() and calcRightMargin()
// use, so the axis, its title, and its reserved margin all appear/disappear
// together. Deliberately does NOT draw new horizontal gridlines across the
// plot (that would double the existing latency grid) -- just short tick
// marks and labels on the right edge, styled to match the left latency axis.
function drawTotalsAxis() {
  const geo = volumeGeometry();
  if (!geo) { totalsAxisLabel.style.display = "none"; return; }
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
    if (y < ceilingY - 0.5 || y > bottom + 0.5) return; // guard float slop only
    ctx.beginPath();
    ctx.moveTo(xEdge, y);
    ctx.lineTo(xEdge + 4, y);
    ctx.stroke();
    ctx.fillText(fmtTokens(v), xEdge + 8, y);
  });
  ctx.restore();

  // Rotated axis title: a real DOM node (totalsAxisLabel, created once near
  // the top of this script) so it can carry the standard .help-label
  // tooltip -- centered on the vertical span the totals layer actually
  // occupies (ceilingY..bottom), not the whole plot, so it stays next to its
  // own ticks even when cache-mix bands claim the top of the chart. Canvas
  // draw coordinates (W, margin, plotH, ...) are in the same CSS-pixel space
  // as the canvas element's own box (see resize()'s dpr handling), so the
  // trigger's screen position is just the canvas's own top-left plus that
  // coordinate -- the same rect-plus-offset approach the mouse handlers
  // below already use. translate(-50%,-50%) centers the label on that point
  // before rotate(90deg) spins it about the same center, so the rotation
  // itself can't shift the label off the point.
  const rect = canvas.getBoundingClientRect();
  const cx = W - 14;
  const cy = (ceilingY + bottom) / 2;
  totalsAxisLabel.style.display = "block";
  totalsAxisLabel.style.left = (rect.left + cx) + "px";
  totalsAxisLabel.style.top = (rect.top + cy) + "px";
  totalsAxisLabel.style.transform = "translate(-50%, -50%) rotate(90deg)";
}

// cacheMixLayout computes the band geometry shared by drawCacheMix and the
// overlay hover lookup in mousemove. null when the overlay is off/empty.
function cacheMixLayout() {
  if (!cacheMixEnabled()) return null;
  const bands = [];
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    if ((s.mix && s.mix.length) || (s.adt && s.adt.length)) bands.push({ s, si });
  });
  if (!bands.length) return null;
  // With every other layer off (latency lines, dots, errors, totals), cache
  // mix is the only thing left to look at: let the bands claim the whole plot
  // instead of sharing it under the usual 60% cap. Re-evaluated on every
  // draw, so re-enabling any of those layers reverts to the banded layout
  // immediately. The 24px floor (shared with the banded case) keeps the
  // per-band labels legible either way.
  let bandH;
  if (!anyPlotLayerVisible()) {
    bandH = Math.max(24, Math.floor(plotH / bands.length));
  } else {
    bandH = MIX_BAND_H;
    if (bands.length * bandH > plotH * 0.6) {
      bandH = Math.max(24, Math.floor(plotH * 0.6 / bands.length));
    }
  }
  bands.forEach((b, bi) => { b.yTop = margin.top + bi * bandH; b.bandH = bandH; });
  // Active-dataset scale is re-framed to the CURRENT view, not the whole run,
  // and stays shared across bands so series remain comparable to each other
  // within that view. Each band caches its own windowed points so drawCacheMix
  // and the label don't re-walk the samples.
  // Each band gets its OWN window range. Unlike the cache-mix stack (which
  // shares MIX_TOTAL_MAX so a quiet arm renders visibly quiet), the bands are
  // stacked rows with no common axis line drawn between them — you can't read
  // relative height across them by eye anyway, so a shared scale bought no
  // comparability while costing every band most of its height: two arms whose
  // datasets sit at different levels each get squeezed into their own slice of
  // the union. Per-band, each line spends the full band on its own variation,
  // and the printed "scale lo-hi" carries the absolute levels.
  bands.forEach(b => {
    b.adtPts = adtWindow(b.s.adt, viewTMin, viewTMax);
    b.adtRange = adtWindowRange([b.adtPts]);
  });
  return { bands };
}

// computeMixReserveH returns the vertical space (px, measured down from
// margin.top) the cache-mix band block occupies when it shares the plot with
// at least one other layer -- the two must never draw into the same pixels
// (a tall TTFT p95 line reaches the top of the axis and used to draw
// straight through the bands; error bars, drawn at
// mapY(b.respAvg), have the identical problem; and the totals layer needs
// the region below the bands, via volumeGeometry()'s ceilingY, so it must
// not be told the bands own the whole plot). 0 when cache mix is off/empty,
// or when it claims the WHOLE plot because nothing else is visible (the
// !anyPlotLayerVisible() branch inside cacheMixLayout) -- that case is
// unchanged existing behavior: nothing else is on screen to reserve space
// for. draw() writes the result into the module-level mixReserveH once per
// draw; mapY reads that var instead of calling this (or cacheMixLayout)
// itself, since mapY runs per plotted point.
function computeMixReserveH() {
  if (!anyPlotLayerVisible()) return 0;
  const layout = cacheMixLayout();
  if (!layout || !layout.bands.length) return 0;
  const last = layout.bands[layout.bands.length - 1];
  return last.yTop + last.bandH - margin.top;
}

// adtY maps a dataset size onto its band. A 2px inset keeps the extremes off
// the band border, and a flat window (lo === hi, e.g. a single carried-in
// sample) draws down the middle rather than dividing by zero.
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
  layout.bands.forEach(({ s, yTop, bandH, adtPts, adtRange }) => {
    // Solid-black backdrop, band-area only: fills pop against it, and
    // unfilled (black) band space still reads as "low ingest".
    ctx.fillStyle = "#000";
    ctx.fillRect(margin.left, yTop, plotW, bandH);

    // Source-mix band: stack height is ABSOLUTE — this interval's total
    // ingested delta against the report-wide MIX_TOTAL_MAX (shared across
    // all series), anchored at the band bottom so quiet minutes render
    // mostly empty. Within the stack the split stays proportional by
    // source. The muted fills stay translucent; the external-KV accent —
    // the one vivid class — renders near-opaque so it keeps its punch.
    (s.mix || []).forEach(seg => {
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

    // Active-dataset line, drawn against this band's own windowed range: the
    // window minimum at the band floor, its maximum at the band top. Only the
    // windowed points are drawn, and x is clamped to the band so the carry-in
    // sample (which sits before viewTMin) anchors the line at the left edge
    // instead of painting across the y-axis margin.
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
      // A single windowed sample is a flat level across the view, not a dot.
      if (adtPts.length === 1) {
        ctx.lineTo(margin.left + plotW, adtY(adtPts[0].v, adtRange, yTop, bandH));
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Band border + labels.
    ctx.strokeStyle = "#42464A";
    ctx.lineWidth = 0.5;
    ctx.strokeRect(margin.left, yTop, plotW, bandH);
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#C79FF1"; // purple-accented label per brand guidance
    ctx.fillText(s.name + " cache mix (peak " + fmtTokens(MIX_TOTAL_MAX) + " tok/min)", margin.left + 4, yTop + 3);
    // Label reports the LAST dataset size within the view and the view's own
    // scale — both re-framed by zoom, so the band always describes what is
    // actually on screen rather than the end state of the whole run.
    if (adtPts && adtPts.length) {
      const last = adtPts[adtPts.length - 1];
      ctx.fillStyle = ADT_LINE_COLOR;
      ctx.textAlign = "right";
      ctx.fillText("last dataset size (tokens): " + fmtTokens(last.v) +
        " | " + last.s + " series | scale " + fmtTokens(adtRange.lo) + "-" + fmtTokens(adtRange.hi),
        margin.left + plotW - 4, yTop + 3);
      ctx.textAlign = "left";
    }
  });
}

// --- Requests layer: density-gated render mode ---
// Zoomed out over a long run there can be dozens of requests per pixel
// column; two 2.5px dots plus a connector per request saturates into a
// solid smear. The
// p10-p90 ribbon (drawRequestRibbon) trades per-request detail for a spread
// signal while the view is dense, and falls back to plain individual dots
// (drawRequestDotsCurrent, connectors included) once the view is zoomed in
// far enough that individual requests are legible again -- see
// effectiveReqMode.
const REQ_DENSITY_GATE = 2; // requests per pixel column of the current view

// requestDensityPerPx counts series s's records inside the current view and
// divides by plotW -- the same "requests per pixel column" measure used to
// describe the smear in the first place.
function requestDensityPerPx(s) {
  if (!plotW) return 0;
  let n = 0;
  s._view.forEach(r => { if (r.t >= viewTMin && r.t <= viewTMax) n++; });
  return n / plotW;
}

// effectiveReqMode resolves the mode actually rendered THIS draw: the ribbon
// when the chart is dense, individual dots (drawRequestDotsCurrent) once it's
// sparse enough to read them. The gate is decided ONCE for the whole chart,
// from the DENSEST visible series -- not per series. Gating per series meant
// two arms with different request counts crossed the threshold at different
// zoom levels, so one arm would flip to dots while the other stayed a
// ribbon: the same picture drawn two different ways, which reads as a bug
// and makes the arms incomparable. Whatever the busiest arm needs, every arm
// gets.
function maxRequestDensityPerPx() {
  let d = 0;
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    const v = requestDensityPerPx(s);
    if (v > d) d = v;
  });
  return d;
}

function effectiveReqMode() {
  return maxRequestDensityPerPx() < REQ_DENSITY_GATE ? "dots-current" : "ribbon";
}

// dotsAreDrawn: true exactly when drawRequestDotsCurrent's per-request dots
// are what's actually on screen this frame, as opposed to the density-gated
// ribbon (drawRequestRibbon) or nothing at all. This is the SAME test draw()
// uses to decide whether to call drawRequestDotsCurrent (see the "Data
// points" block below) -- shared here rather than duplicated so the
// mousemove hover hit-test can never drift from what's actually drawn. Before
// this existed, mousemove checked showDots alone, so hovering in ribbon mode
// hit-tested phantom dot positions that were never rendered and pre-empted
// the line/percentile tooltip underneath.
function dotsAreDrawn() {
  return document.getElementById("showDots").checked && effectiveReqMode() !== "ribbon";
}

// drawRequestDotsCurrent is today's Requests rendering, unchanged: a
// TTFT/response dot pair per request joined by a faint connector, error
// requests drawn larger and red. This is the "dots-current" baseline and
// also what the ribbon falls back to under the density gate.
function drawRequestDotsCurrent(s, color, showErrors) {
  s._view.forEach(r => {
    if (r.t < viewTMin || r.t > viewTMax) return;
    if (r.err && !showErrors) return;
    const x = mapX(r.t);

    {
      const y1 = mapY(r.ttft);
      const y2 = mapY(r.resp);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (r.ttft > 0) {
      const y = mapY(r.ttft);
      ctx.beginPath();
      ctx.arc(x, y, r.err ? 4 : 2.5, 0, Math.PI * 2);
      if (r.err) {
        ctx.fillStyle = "#FF6B6B";
        ctx.fill();
      } else {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    {
      const y = mapY(r.resp);
      ctx.beginPath();
      ctx.arc(x, y, r.err ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = r.err ? "#FF6B6B" : color;
      ctx.globalAlpha = r.err ? 0.8 : 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  });
}

// drawRequestRibbon fills the rolling p10-p90 response/TTLT spread envelope
// for series s (same rolling window as _respP50 -- built alongside it in
// computeDerived, so it can't drift out of sync) as a translucent band in
// the arm's own color. Deliberately drawn by draw() BEFORE the
// rolling-percentile lines section, so the existing p50 line paints on top
// of the ribbon rather than the ribbon covering it.
function drawRequestRibbon(s, color) {
  const p10 = s._respP10, p90 = s._respP90;
  if (!p10 || !p90 || p10.length < 2) return;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < p10.length; i++) {
    const p = p10[i];
    if (p.t < viewTMin || p.t > viewTMax) continue;
    const x = mapX(p.t), y = mapY(p.v);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  if (!started) return; // nothing in view
  // p10/p90 share the same anchor t's by construction (computeDerived pushes
  // both in the same loop), so walking p90 in reverse over the same view
  // filter closes the envelope correctly.
  for (let i = p90.length - 1; i >= 0; i--) {
    const p = p90[i];
    if (p.t < viewTMin || p.t > viewTMax) continue;
    ctx.lineTo(mapX(p.t), mapY(p.v));
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.17;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function draw() {
  // Recalc the right margin: whether the Totals layer's ingest-token axis is
  // currently contributing pixels (see calcRightMargin) can change between
  // frames from a checkbox toggle or a series being hidden/shown. The bottom
  // margin (calcBottomMargin) is a fixed constant and never changes, so it's
  // only set once, in resize().
  const newRight = calcRightMargin();
  if (margin.right !== newRight) {
    margin.right = newRight;
    resize();
  }

  // Reserve the cache-mix band strip (if any) at the top of the plot BEFORE
  // anything below reads mapY, so every line/dot/gridline/hover this frame
  // (and until the next draw()) agrees on the same compressed latency
  // region. See mapY() and computeMixReserveH().
  mixReserveH = computeMixReserveH();

  const showTTFT = document.getElementById("showTTFT").checked;
  const showResp = document.getElementById("showResp").checked;
  const showDots = document.getElementById("showDots").checked;
  const showErrors = document.getElementById("showErrors").checked;

  ctx.clearRect(0, 0, W, H);

  // Grid and axes
  ctx.strokeStyle = "#21262b";
  ctx.lineWidth = 0.5;
  ctx.fillStyle = "#8a9096";
  ctx.font = "11px monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  // Y axis
  const ySteps = niceSteps(viewYMax, 8);
  ySteps.forEach(v => {
    const y = mapY(v);
    // Bound against the RESERVED top (margin.top + mixReserveH), not the
    // plot's raw top: mapY already compresses every v into that region, so
    // this just keeps the check honest about where ticks can land -- the
    // topmost tick (v === viewYMax) must still pass at y === margin.top +
    // mixReserveH exactly.
    if (y >= margin.top + mixReserveH && y <= margin.top + plotH) {
      ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
      let label = v >= 1000 ? (v/1000).toFixed(v >= 10000 ? 0 : 1) + "s" : v.toFixed(0) + "ms";
      ctx.fillText(label, margin.left - 6, y);
    }
  });

  // X axis (time ticks)
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const duration = (viewTMax - viewTMin) / 1000;
  const xStepSec = computeXStepSec(duration);
  const startSec = Math.ceil(((viewTMin - globalTMin) / 1000) / xStepSec) * xStepSec;
  currentTicks = []; // rebuilt every draw; consumed by mousemove hover lookup
  for (let s = startSec; s <= (viewTMax - globalTMin) / 1000; s += xStepSec) {
    const x = mapX(globalTMin + s * 1000);
    if (x < margin.left || x > margin.left + plotW) continue;
    const tickTime = globalTMin + s * 1000;
    currentTicks.push({ x, tickTime });
    ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
    ctx.fillText(formatTickLabel(s), x, margin.top + plotH + 6);
  }

  // Y axis label
  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#C9C9C9";
  ctx.font = "12px sans-serif";
  ctx.fillText("Latency", 0, 0);
  ctx.restore();

  // Right axis (Totals/ingest layer, absolute tokens) -- no-ops and hides
  // its title when the layer isn't contributing pixels. Drawn here,
  // unclipped, alongside the rest of the axis chrome -- see drawTotalsAxis.
  drawTotalsAxis();

  // X axis label
  ctx.fillStyle = "#C9C9C9";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";

  // Plot border
  ctx.strokeStyle = "#42464A";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Clip to plot area for data points
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.clip();

  // Totals volume layer first (bottom of the z-order), then the cache-mix
  // overlay (its black backdrop deliberately sits over the volume in the
  // band region), then latency lines on top of both.
  drawTotals();
  drawCacheMix();

  // Requests layer, ribbon mode: fill the p10-p90 spread envelope HERE,
  // before the percentile lines below, so the existing p50 line paints on
  // top of the ribbon rather than the ribbon covering it. The dots-current
  // fallback keeps its historical position later, drawn OVER the lines --
  // see the "Data points" block.
  if (showDots) {
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      if (effectiveReqMode() === "ribbon") drawRequestRibbon(s, seriesColors[si]);
    });
  }

  // Rolling-percentile lines: pattern encodes the percentile, color the
  // series. Response p50 solid; TTFT p50 dense dots; TTFT p95 sparse
  // lighter dashes (independently togglable).
  const showTTFTP95 = document.getElementById("showTTFTP95").checked;
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
    if (showResp) plotLine(s._respP50, "resp50", color, 0.8, 2, false);
    if (showTTFT) plotLine(s._ttftP50, "ttft50", color, 0.8, 2, true);
    if (showTTFTP95) plotLine(s._ttftP95, "ttft95", color, 0.55, 1.5, true);
  });

  // Error rate bars
  if (showErrors) {
    const maxBarH = plotH * 0.15;
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      const color = seriesColors[si];
      s._errBars.forEach(b => {
        if (b.t < viewTMin || b.t > viewTMax) return;
        const x = mapX(b.t);
        const barH = maxBarH * b.errRate;
        const yTop = mapY(b.respAvg) - barH;
        const yBot = mapY(b.respAvg);
        // Red core
        ctx.fillStyle = "#FF6B6B";
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x - 2, yTop, 4, barH);
        ctx.globalAlpha = 1;
        // Series-colored cap lines (top and bottom)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 4, yBot); ctx.lineTo(x + 4, yBot); ctx.stroke();
      });
    });
  }

  // Data points (only when showDots is enabled). The ribbon is drawn
  // earlier, under the percentile lines, and is skipped here; the
  // dots-current fallback keeps its original z-order (drawn over the lines
  // and error bars) -- see effectiveReqMode.
  if (dotsAreDrawn()) {
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      drawRequestDotsCurrent(s, seriesColors[si], showErrors);
    });
  }

  ctx.restore(); // remove clip

  // Draw drag selection overlay
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

  updateInfo();
}

// volumeHoverAt: the ingest-volume layer under the cursor, or null. Finds
// the layer whose stacked band contains the pointer's vertical fraction,
// then snaps to that series' closest data point in time. Lowest hover
// priority: dot/line > cache-mix band > volume.
function volumeHoverAt(mx, my) {
  const geo = volumeGeometry();
  if (!geo) return null;
  const bottom = margin.top + plotH;
  if (my > bottom || my < geo.ceilingY) return null;
  const t = unmapX(mx);
  const fracY = (bottom - my) / Math.max(bottom - geo.ceilingY, 1e-9);
  let acc = 0, hit = null;
  for (const vs of geo.visible) {
    const frac = cumTokensAt(vs.s._cumTimes, vs.s._cumTokens, t) / geo.finalTotal;
    if (fracY <= acc + frac) { hit = vs; break; }
    acc += frac;
  }
  if (!hit) return null;
  const ci = closestIndex(hit.s._cumTimes, t);
  if (ci < 0) return null;
  const tc = hit.s._cumTimes[ci];
  const cumTok = hit.s._cumTokens[ci];
  const cumOut = hit.s._cumOutTokens[ci];
  const cums = geo.visible.map(({ s }) => ({ times: s._cumTimes, cum: s._cumTokens }));
  return { s: hit.s, si: hit.si, tc: tc, cumTok: cumTok, cumOut: cumOut, share: shareOfBest(cums, tc, cumTok) };
}

// mixTooltipHTML renders the cache-mix breakdown lines for series s at time
// t: the covering (or latest preceding) per-minute source deltas plus the
// active dataset size/series count. "" when no sample data exists at t.
function mixTooltipHTML(s, t) {
  const seg = mixAt(s.mix, t);
  const p = adtAt(s.adt, t);
  if (!seg && !p) return "";
  const lines = [];
  if (seg) {
    const total = seg.c + seg.lc + seg.ec;
    const pct = v => total > 0 ? " (" + (100 * v / total).toFixed(0) + "%)" : "";
    lines.push("<span style='color:" + MIX_COMPUTE_COLOR + "'>compute: " + fmtTokens(seg.c) + pct(seg.c) + "</span>");
    lines.push("<span style='color:" + MIX_LOCAL_COLOR + "'>local cache: " + fmtTokens(seg.lc) + pct(seg.lc) + "</span>");
    lines.push("<span style='color:" + MIX_EXTERNAL_COLOR + "'>external KV: " + fmtTokens(seg.ec) + pct(seg.ec) + "</span>");
    lines.push("<span style='color:#8a9096'>ingest: " + fmtTokens(mixRate(seg)) + " tok/s (" +
      fmtTokens(total) + " tok / " + Math.round((seg.t1 - seg.t0) / 1000) + "s)</span>");
  }
  if (p) {
    lines.push("<span style='color:" + ADT_LINE_COLOR + "'>active dataset: " + fmtTokens(p.v) + " tok, " + p.s + " series</span>");
  }
  return lines.join("<br>");
}

// showTooltip: single positioning path for every tooltip (dot, line,
// overlay, tick hover). Content is set FIRST, measured at a neutral
// position (so the right viewport edge can't squeeze the measured width),
// then placed viewport-aware via placeTooltip.
function showTooltip(e, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  tooltip.style.left = "0px";
  tooltip.style.top = "0px";
  const r = tooltip.getBoundingClientRect();
  const pos = placeTooltip(e.clientX, e.clientY, r.width, r.height, window.innerWidth, window.innerHeight);
  tooltip.style.left = pos.x + "px";
  tooltip.style.top = pos.y + "px";
}

// --- Drag-to-zoom ---
canvas.addEventListener("mousedown", e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  if (mx >= margin.left && mx <= margin.left + plotW) {
    dragStart = mx;
    dragCurrent = mx;
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

  // Tooltip
  let best = null, bestDist = 20;

  // Check proximity to dots -- ONLY when dots are actually the thing drawn
  // this frame (see dotsAreDrawn(): showDots checked AND effectiveReqMode()
  // isn't "ribbon"). At high density the Requests layer renders a p10-p90
  // ribbon instead, with no dots on screen; testing showDots alone here used
  // to hit-test phantom dot positions in that case, and since dots were
  // checked before lines with a wider radius, a phantom hit could pre-empt
  // the p50/TTFT line tooltip the user was actually pointing at. Falling
  // through leaves best null so the "proximity to average lines" check
  // below runs instead, exactly as if dots were off.
  if (dotsAreDrawn()) {
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      s._view.forEach(r => {
        if (r.t < viewTMin || r.t > viewTMax) return;
        const x = mapX(r.t);
        const yr = mapY(r.resp);
        const yt = mapY(r.ttft);
        for (const [y, type] of [[yr, "resp"], [yt, "ttft"]]) {
          const d = Math.hypot(mx - x, my - y);
          if (d < bestDist) { bestDist = d; best = { s, si, r, type }; }
        }
      });
    });
  }

  // Also check proximity to average lines
  if (!best) {
    let bestLineDist = 15;
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      const checkLine = (avgData, type) => {
        for (let i = 0; i < avgData.length; i++) {
          const p = avgData[i];
          if (p.t < viewTMin || p.t > viewTMax) continue;
          const x = mapX(p.t), y = mapY(p.v);
          const d = Math.hypot(mx - x, my - y);
          if (d < bestLineDist) {
            bestLineDist = d;
            // Find window for p50/p95
            const idx = Math.min(i, s._sorted.length - 1);
            const start = Math.max(0, idx - s._winSize + 1);
            const win = s._sorted.slice(start, idx + 1);
            best = { s, si, type, idx, win, avgVal: p.v, t: p.t, isLine: true };
          }
        }
      };
      if (document.getElementById("showResp").checked) checkLine(s._respP50, "resp50");
      if (document.getElementById("showTTFT").checked) checkLine(s._ttftP50, "ttft50");
      if (document.getElementById("showTTFTP95").checked) checkLine(s._ttftP95, "ttft95");
    });
  }

  // Cache-mix overlay hover: pointer inside a band (and not on a dot/line)
  // surfaces that minute's source breakdown + active dataset size.
  let mixHover = null;
  if (!best && mx >= margin.left && mx <= margin.left + plotW) {
    const layout = cacheMixLayout();
    if (layout) {
      const band = layout.bands.find(b => my >= b.yTop && my <= b.yTop + b.bandH);
      if (band) {
        const t = unmapX(mx);
        if (mixAt(band.s.mix, t) || adtAt(band.s.adt, t)) mixHover = { band, t };
      }
    }
  }

  // On long views (see ANNOTATION_ROWS_MAX_DURATION), hovering near any
  // x-axis gridline surfaces a per-series cumulative request/error breakdown
  // via tooltip (tickHover below).
  // Ingest-volume hover: lowest priority of the shaped hovers — never
  // steals from dot/line or band tooltips.
  let volHover = null;
  if (!best && !mixHover && mx >= margin.left && mx <= margin.left + plotW) {
    volHover = volumeHoverAt(mx, my);
  }

  let tickHover = null;
  if (!best && !mixHover && !volHover && (viewTMax - viewTMin) / 1000 > ANNOTATION_ROWS_MAX_DURATION && my >= margin.top) {
    let bestTickDist = 10;
    currentTicks.forEach(tk => {
      const d = Math.abs(mx - tk.x);
      if (d < bestTickDist) { bestTickDist = d; tickHover = tk; }
    });
  }

  if (best) {
    if (best.isLine) {
      const win = best.win;
      const isTTFT = best.type === "ttft50" || best.type === "ttft95";
      const vals = isTTFT ? win.map(r => r.ttft).filter(v => v > 0) : win.map(r => r.resp);
      const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      const p50 = percentile(vals, 0.5);
      const p95 = percentile(vals, 0.95);
      const fmt = v => v >= 1000 ? (v/1000).toFixed(2) + "s" : v.toFixed(0) + "ms";
      const lineLabel = { resp50: "Resp/TTLT p50", ttft50: "TTFT p50", ttft95: "TTFT p95" }[best.type] || best.type;
      // Count total requests, errors, and max series index up to this point
      let totalUpTo = 0, errUpTo = 0, maxSnSeen = 0;
      best.s._view.forEach(r => { if (r.t <= best.t) { totalUpTo++; if (r.err) errUpTo++; if (r.sn > maxSnSeen) maxSnSeen = r.sn; } });
      const mixInfo = mixTooltipHTML(best.s, best.t);
      showTooltip(e,
        "<b>" + best.s.name + "</b> \u2014 " + lineLabel + "<br>" +
        "Window: " + win.length + " requests<br>" +
        "Avg: " + fmt(avg) + "<br>" +
        "p50: " + fmt(p50) + "<br>" +
        "p95: " + fmt(p95) + "<br>" +
        "Series: " + maxSnSeen + "<br>" +
        "Total: " + totalUpTo + (errUpTo > 0 ? ", <span style='color:#FF6B6B'>errors: " + errUpTo + "</span>" : "") +
        (mixInfo ? "<br>" + mixInfo : ""));
    } else {
      const r = best.r;
      const mixInfo = mixTooltipHTML(best.s, r.t);
      // Token breakdown for the hovered request. This is what makes a latency
      // outlier diagnosable: a slow request is a different problem depending on
      // whether it carried an enormous prompt, missed the cache, or generated a
      // long completion. r.in is net-of-cache and r.ca is the cached subset
      // (see buildReplayUsage), so the prompt the server saw is the sum.
      const prompt = (r.in || 0) + (r.ca || 0);
      const cachedPct = prompt > 0 ? (r.ca || 0) / prompt * 100 : 0;
      // How this request's context compares with its series' median tells you
      // whether "large" means large for this run or just large in absolute
      // terms — an outlier at the typical size is a different story.
      const ctxs = best.s._view.map(x => (x.in || 0) + (x.ca || 0));
      const medCtx = percentile(ctxs, 0.5);
      const vsMed = medCtx > 0 ? (prompt / medCtx).toFixed(1) + "x median ctx" : "";
      showTooltip(e,
        "<b>" + best.s.name + "</b> (series " + r.sn + ", req " + r.rn + ")<br>" +
        "TTFT: " + r.ttft.toFixed(1) + " ms<br>" +
        "Resp/TTLT: " + r.resp.toFixed(1) + " ms<br>" +
        "<span style='color:#8a9096'>—</span><br>" +
        "Input: " + fmtTokens(prompt) + " tok" +
        (vsMed ? " <span style='color:#8a9096'>(" + vsMed + ")</span>" : "") + "<br>" +
        "&nbsp;&nbsp;cached: " + fmtTokens(r.ca || 0) +
        " <span style='color:#8a9096'>(" + cachedPct.toFixed(1) + "%)</span><br>" +
        "&nbsp;&nbsp;uncached: " + fmtTokens(r.in || 0) + "<br>" +
        "Output: " + fmtTokens(r.out || 0) + " tok<br>" +
        (r.ch ? "Cache hit<br>" : "") +
        (r.err ? "<span style='color:#FF6B6B'>ERROR</span>" : "") +
        (mixInfo ? "<br>" + mixInfo : ""));
    }
  } else if (mixHover) {
    const seg = mixAt(mixHover.band.s.mix, mixHover.t);
    let win = "";
    if (seg) {
      win = "window: " + formatTickLabel(Math.round(seg.t0 / 1000)) + " \u2192 " +
        formatTickLabel(Math.round(seg.t1 / 1000)) + "<br>";
    }
    showTooltip(e, "<b>" + mixHover.band.s.name + "</b> \u2014 cache mix<br>" +
      win + mixTooltipHTML(mixHover.band.s, mixHover.t));
  } else if (volHover) {
    const rates = windowRates(volHover.s._byT, volHover.tc, 60000);
    showTooltip(e,
      "<b>" + volHover.s.name + "</b> \u2014 ingest volume<br>" +
      "at " + formatTickLabel(Math.round(volHover.tc / 1000)) + ": " +
      fmtTokens(volHover.cumTok) + " tok cumulative ingest (" + (100 * volHover.share).toFixed(0) + "% of best), " +
      fmtTokens(volHover.cumOut) + " tok output total<br>" +
      (rates ? "<span style='color:#8a9096'>last " + Math.round(rates.spanS) + "s: " +
        rates.n + " req (" + rates.rps.toFixed(1) + "/s), in " + fmtTokens(rates.inPerSec) + " tok/s, out " +
        fmtTokens(rates.outPerSec) + " tok/s</span>" : ""));
  } else if (tickHover) {
    const elapsedSec = Math.round((tickHover.tickTime - globalTMin) / 1000);
    const rows = tickStats(tickHover.tickTime).map(({ si, cumReqs, cumErrs, maxSn }) => {
      const errPart = cumErrs > 0 ? ", <span style='color:#FF6B6B'>errors: " + cumErrs + "</span>" : "";
      return "<span style='color:" + seriesColors[si] + "'>" + DATA[si].name + "</span>: " +
        cumReqs + " reqs" + errPart + ", series @" + maxSn;
    });
    showTooltip(e, "<b>t = " + formatTickLabel(elapsedSec) + "</b><br>" + rows.join("<br>"));
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
  dragStart = null;
  dragCurrent = null;

  // Only zoom if dragged at least 5 pixels
  if (maxPx - minPx < 5) { draw(); return; }

  // Resolve BOTH edges against the pre-zoom view before mutating either.
  // unmapX is relative to the current viewTMin/viewTMax, so assigning
  // viewTMin first made the second call resolve against the new view: the
  // start was right but the end landed far too late, leaving the dragged
  // region at the front of a much wider window (a narrow drag at position f
  // yielded a span of ~f*(1-f) of the run instead of the width dragged).
  const tMin = unmapX(minPx);
  const tMax = unmapX(maxPx);
  viewTMin = tMin;
  viewTMax = tMax;
  recalcYMax();
  draw();
});

canvas.addEventListener("mouseleave", () => {
  if (dragStart !== null) { dragStart = null; dragCurrent = null; draw(); }
  tooltip.style.display = "none";
});

// Defensive: a drag released OUTSIDE the browser window never delivers
// mouseup to the canvas; a stuck dragStart would then suppress every
// tooltip until the next canvas interaction. Window-level mouseup clears it.
window.addEventListener("mouseup", () => {
  if (dragStart !== null) { dragStart = null; dragCurrent = null; draw(); }
});

// Double-click to reset zoom
canvas.addEventListener("dblclick", resetZoomView);

// Reset button

// Wire up controls. The four latency-plotting layers can change which
// series is now tallest, so they recompute the y-axis; the annotation-only
// and normalized-elsewhere layers (errors, totals) never affect that axis
// and just redraw.
["showTTFT", "showTTFTP95", "showResp", "showDots"].forEach(id => {
  document.getElementById(id).addEventListener("change", () => { recalcYMax(); draw(); });
});
["showErrors", "showTotals"].forEach(id => {
  document.getElementById(id).addEventListener("change", draw);
});

// Cache-mix toggle only exists when the dataset actually carries metrics
// samples; old datasets render exactly as before.
//
// It starts ON only for a report of CACHE_MIX_DEFAULT_MAX_SERIES arms or
// fewer. Every series claims its own band off the top of the plot (up to 60%
// of plot height, split between them), so past a handful of arms the bands
// squeeze the latency chart the report is actually named after, and each band
// gets too thin to read. Small comparisons — the common case — still open with
// the overlay visible; wide sweeps open on the chart and let you switch it on.
const CACHE_MIX_DEFAULT_MAX_SERIES = 4;
function cacheMixDefaultOn(seriesCount) { return seriesCount <= CACHE_MIX_DEFAULT_MAX_SERIES; }
if (HAS_CACHE_MIX) {
  const lbl = document.createElement("label");
  lbl.innerHTML = '<input type="checkbox" id="showCacheMix"> ' +
    '<span class="help-label" id="hlpCacheMix" tabindex="0" aria-describedby="helpTip" ' +
    'data-tip="Where prompt tokens came from: recompute, local cache, or external KV.">Cache Mix</span> ' +
    '<span style="color:' + MIX_COMPUTE_COLOR + '">&#9632;</span>compute ' +
    '<span style="color:' + MIX_LOCAL_COLOR + '">&#9632;</span>local cache ' +
    '<span style="color:' + MIX_EXTERNAL_COLOR + '">&#9632;</span>external KV ' +
    '<span style="color:' + ADT_LINE_COLOR + '">&#8213;</span>active dataset (tokens)';
  document.querySelector(".controls").appendChild(lbl);
  helpTriggers.push(document.getElementById("hlpCacheMix"));
  const cb = document.getElementById("showCacheMix");
  // Set the property rather than baking "checked" into the markup: one source
  // of truth for the default, and it survives the element already existing.
  cb.checked = cacheMixDefaultOn(DATA.length);
  cb.addEventListener("change", draw);
}

// --- Help tooltip: shared by every .help-label trigger (control-layer
// labels, summary column headers, the Cache Mix toggle -- collected into
// helpTriggers as each is created, see above) ---
// Replaces the old native "title" tooltip (browser-controlled ~1-2s delay,
// OS-styled box, no styling control) with one custom element, reused by
// moving/toggling it rather than one div per trigger. Deliberately matches
// #tooltip's visual language (surface #1E2429, border #42464A, 6px radius,
// 0.8em font) -- see the .help-tip rules -- but is otherwise fully
// independent: its own show/hide state, so it can never fight the chart
// hover tooltip over position or visibility.
let helpShowTimer = null, helpHideTimer = null;

// placeHelpTip positions #helpTip near trigger, viewport-aware: prefers
// below the trigger, flips above when that would clip the bottom edge, and
// clamps horizontally so it never renders off-screen. Always fixed/page
// coordinates (matching placeTooltip's approach for the chart tooltip) so
// the summary panel's own scroll container can't clip it. The small caret
// tracks the trigger's horizontal center, clamped inside the box, so it
// still roughly points at the trigger even when the box itself got clamped.
function placeHelpTip(trigger) {
  const r = trigger.getBoundingClientRect();
  const gap = 8, pad = 4;
  const vw = window.innerWidth, vh = window.innerHeight;
  const tw = helpTip.offsetWidth, th = helpTip.offsetHeight;
  let top = r.bottom + gap;
  let below = true;
  if (top + th > vh - pad) {
    top = r.top - gap - th;
    below = false;
  }
  top = Math.min(Math.max(top, pad), Math.max(pad, vh - pad - th));
  let left = r.left + r.width / 2 - tw / 2;
  left = Math.min(Math.max(left, pad), Math.max(pad, vw - pad - tw));
  helpTip.style.left = left + "px";
  helpTip.style.top = top + "px";
  // below === true means the tip sits BELOW the trigger, so the caret goes
  // on the tip's TOP edge, pointing up at it (and vice versa).
  helpTip.classList.toggle("caret-top", below);
  helpTip.classList.toggle("caret-bottom", !below);
  const caretX = Math.min(Math.max(r.left + r.width / 2 - left, 10), Math.max(10, tw - 10));
  helpTip.style.setProperty("--caret-x", caretX + "px");
}

// showHelpTip: ~120ms show delay (the whole point is being faster than the
// native tooltip's ~1-2s), except when the tip is ALREADY visible for a
// neighboring trigger -- then content and position swap immediately, so
// moving across adjacent triggers reads as one continuous tooltip instead of
// re-delaying every time. A trigger with no tip text (e.g. a summary row's
// ratio cell before it holds a baseline comparison) shows nothing rather
// than an empty floating box.
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

// hideHelpTip: a short (~60ms) grace before hiding, so moving the pointer
// between adjacent triggers doesn't flicker the tip closed and back open.
function hideHelpTip() {
  clearTimeout(helpShowTimer);
  helpHideTimer = setTimeout(() => helpTip.classList.remove("visible"), 60);
}

// hideHelpTipNow: immediate hide, no grace period -- used by the Escape
// handler below, where a delayed hide would read as unresponsive.
function hideHelpTipNow() {
  clearTimeout(helpShowTimer);
  clearTimeout(helpHideTimer);
  helpTip.classList.remove("visible");
}

// Keyboard/hover parity: every trigger is a tabbable element (tabindex="0"
// in the markup) carrying aria-describedby="helpTip", and role="tooltip" is
// set on #helpTip itself in the markup -- so the tooltip is reachable and
// announced the same way for a keyboard/screen-reader user as for a mouse
// hover. Escape hides it; see the keydown handler in the context-filter
// modal block below, which checks helpTip's visibility FIRST so this can
// never be shadowed by (or shadow) the modal-close / exit-zoom precedence
// chain it already implements.
helpTriggers.forEach(el => {
  el.addEventListener("mouseenter", () => showHelpTip(el));
  el.addEventListener("mouseleave", hideHelpTip);
  el.addEventListener("focus", () => showHelpTip(el));
  el.addEventListener("blur", hideHelpTip);
});

// Context-filter modal: the series selector — per-series rows with color
// dot, visibility checkbox, and max-context hint — plus the numeric
// context band (>= min / <= max tokens) that re-derives all curves,
// volume, and rates from the filtered requests on Apply. Dismissable
// (Close / backdrop / Esc) and resettable.
{
  const modal = document.getElementById("ctxModal");
  const openBtn = document.getElementById("ctxFilterBtn");
  const listEl = document.getElementById("ctxModalSeries");

  // Global per-series-index stats for the in-dataset selector: max context
  // and request count across all variants, sorted heaviest-context first.
  // Computed once; the rendered list is capped so replay runs with
  // thousands of sessions never lag the modal.
  const SN_LIST_CAP = 300;
  const SN_INFO = (() => {
    const m = new Map();
    DATA.forEach(s => s.records.forEach(r => {
      const ctx = (r.in || 0) + (r.ca || 0);
      let e = m.get(r.sn);
      if (!e) { e = { sn: r.sn, maxCtx: 0, reqs: 0 }; m.set(r.sn, e); }
      if (ctx > e.maxCtx) e.maxCtx = ctx;
      e.reqs++;
    }));
    return Array.from(m.values()).sort((a, b) => b.maxCtx - a.maxCtx);
  })();

  const snCountEl = document.getElementById("snCount");
  const snListEl = document.getElementById("snList");
  const snNoteEl = document.getElementById("snNote");
  const refreshSnCount = () => {
    snCountEl.textContent = snFilter.size > 0 ? snFilter.size + " selected" : "all";
  };
  const rebuildSnList = () => {
    snListEl.innerHTML = "";
    SN_INFO.slice(0, SN_LIST_CAP).forEach(info => {
      const row = document.createElement("div");
      row.className = "modal-sn-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = snFilter.has(info.sn);
      cb.addEventListener("change", () => {
        if (cb.checked) snFilter.add(info.sn); else snFilter.delete(info.sn);
        refreshSnCount();
      });
      row.appendChild(cb);
      const name = document.createElement("span");
      name.textContent = "#" + info.sn + " (" + info.reqs + " reqs)";
      row.appendChild(name);
      const ctxEl = document.createElement("span");
      ctxEl.className = "legend-ctx";
      ctxEl.textContent = "max ctx " + fmtTokens(info.maxCtx);
      row.appendChild(ctxEl);
      snListEl.appendChild(row);
    });
    snNoteEl.textContent = SN_INFO.length > SN_LIST_CAP
      ? "showing top " + SN_LIST_CAP + " of " + SN_INFO.length + " series by max context — use the index field for the rest"
      : SN_INFO.length + " series, heaviest context first";
    refreshSnCount();
  };
  document.getElementById("snAdd").addEventListener("click", () => {
    parseSnList(document.getElementById("snInput").value).forEach(sn => snFilter.add(sn));
    document.getElementById("snInput").value = "";
    rebuildSnList();
  });
  document.getElementById("snClear").addEventListener("click", () => {
    snFilter.clear();
    rebuildSnList();
  });
  // Variant select-all/deselect-all live here, beside the per-variant
  // checkboxes they actually control (the top-level buttons of the same
  // name now govern the plotted-layer checkboxes instead -- see
  // setAllLayers above). renderState() is what makes clicking one of these
  // immediately update every checkbox below plus the legend and the chart.
  document.getElementById("variantSelectAll").addEventListener("click", () => {
    DATA.forEach((_, i) => hiddenSeries.delete(i));
    renderState();
  });
  document.getElementById("variantDeselectAll").addEventListener("click", () => {
    DATA.forEach((_, i) => hiddenSeries.add(i));
    renderState();
  });
  const rebuildList = () => {
    listEl.innerHTML = "";
    modalRowCheckboxes = [];
    DATA.forEach((s, i) => {
      const row = document.createElement("div");
      row.className = "modal-series-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !hiddenSeries.has(i);
      cb.addEventListener("change", () => {
        if (cb.checked) hiddenSeries.delete(i); else hiddenSeries.add(i);
        renderState();
      });
      modalRowCheckboxes.push({ cb, i });
      row.appendChild(cb);
      const dot = document.createElement("div");
      dot.className = "legend-dot";
      dot.style.background = seriesColors[i];
      row.appendChild(dot);
      const name = document.createElement("span");
      name.textContent = s.name;
      row.appendChild(name);
      const ctxHintEl = document.createElement("span");
      ctxHintEl.className = "legend-ctx";
      ctxHintEl.textContent = s._maxCtx > 0 ? "max ctx " + fmtTokens(s._maxCtx) : "";
      row.appendChild(ctxHintEl);
      listEl.appendChild(row);
    });
  };
  const closeModal = () => { modal.style.display = "none"; };
  openBtn.addEventListener("click", () => { rebuildList(); rebuildSnList(); modal.style.display = "block"; });
  document.getElementById("ctxClose").addEventListener("click", closeModal);
  modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
  // Single ordered ESC handler (do not add a second competing listener):
  // the help tooltip takes precedence when visible (dismiss it without
  // touching the modal/zoom underneath), then the modal when open, else a
  // single-step zoom exit (same effect as Reset Zoom), else no-op. No
  // multi-level zoom history.
  window.addEventListener("keydown", e => {
    if (e.key !== "Escape") return;
    if (helpTip.classList.contains("visible")) { hideHelpTipNow(); return; }
    if (modal.style.display === "block") { closeModal(); return; }
    if (isZoomed()) resetZoomView();
  });
  // Band inputs are in K TOKENS: "300" means 300,000 tokens.
  document.getElementById("ctxApply").addEventListener("click", () => {
    const minK = parseFloat(document.getElementById("ctxMin").value) || 0;
    const maxK = parseFloat(document.getElementById("ctxMax").value) || 0;
    applyCtxFilter(minK * 1000, maxK * 1000);
  });
  document.getElementById("ctxReset").addEventListener("click", () => {
    document.getElementById("ctxMin").value = "";
    document.getElementById("ctxMax").value = "";
    snFilter.clear();
    rebuildSnList();
    applyCtxFilter(0, 0);
  });
}

// --- Data export: per-request rows CSV + summary panel CSV -------------
// A copied report is often read somewhere the run's own JSONL is not
// reachable, so this is the only way it yields its numbers back out. Pure client side —
// Blob + a throwaway <a download>, no network, no library — so it works
// offline exactly like the rest of the report. Both exports honor the
// CURRENT view precisely the way the chart and summary panel already do, by
// reusing their own state instead of recomputing scope independently:
// viewTMin/viewTMax for the zoom window, hiddenSeries for deselected arms,
// and s._view (already shaped by applyCtxFilter) for the context-band and
// in-dataset series filters. Scope can't drift from the display because
// there is no second source of truth for it.

// csvField/csvRow: RFC-4180-ish escaping — a field is quoted (with embedded
// quotes doubled) only when it contains a comma, quote, or newline, so plain
// arm names/numbers stay unquoted and readable.
function csvField(v) {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function csvRow(fields) { return fields.map(csvField).join(","); }

// visibleIndices: series indices NOT currently hidden via the legend or the
// context-filter modal's per-variant checkboxes — "what the user is looking
// at" for both exports.
function visibleIndices() { return DATA.map((_, i) => i).filter(i => !hiddenSeries.has(i)); }

// scopeElapsedLabel mirrors renderSummary's #sumRange text exactly (same
// off() computation) so an export's stated scope always agrees with what the
// summary panel prints on screen.
function scopeElapsedLabel() {
  const off = t => formatTickLabel(Math.round((t - globalTMin) / 1000));
  return isZoomed() ? (off(viewTMin) + " - " + off(viewTMax)) : ("full run (" + off(globalTMax) + ")");
}
// scopeToken: a filesystem-safe scope token for filenames — "full", or
// "zoom-<start>s-<end>s" in elapsed seconds from each arm's own run start
// (the common relative axis every arm is time-aligned to, see the
// per-series t-normalization above DATA.forEach), so one token names the
// window unambiguously for every included arm.
function scopeToken() {
  if (!isZoomed()) return "full";
  const s0 = Math.round((viewTMin - globalTMin) / 1000);
  const s1 = Math.round((viewTMax - globalTMin) / 1000);
  return "zoom-" + s0 + "s-" + s1 + "s";
}

// provenanceHeader builds the "#"-prefixed comment lines shared by both
// exports: included/excluded arms, the baseline, the active scope (zoomed
// window with each included arm's own absolute start/end, or "full run"),
// the context band and in-dataset series filters when active, recorded run
// params per arm (gracefully absent on legacy reports), and a generation
// timestamp. extraLines is appended last, for an export-specific caveat.
function provenanceHeader(kind, extraLines) {
  const lines = [];
  lines.push("# wekai benchmark report -- " + kind + " export");
  lines.push("# generated " + new Date().toISOString());
  const visible = visibleIndices();
  const hidden = DATA.map((_, i) => i).filter(i => hiddenSeries.has(i));
  lines.push("# arms included (" + visible.length + "): " + visible.map(i => DATA[i].name).join(", "));
  if (hidden.length) {
    lines.push("# arms hidden/deselected, excluded from this export (" + hidden.length + "): " +
      hidden.map(i => DATA[i].name).join(", "));
  }
  if (BASELINE_INDEX >= 0) {
    lines.push("# baseline arm (ratio columns are % of this arm): " + DATA[BASELINE_INDEX].name +
      (hiddenSeries.has(BASELINE_INDEX)
        ? " (hidden/excluded above, but ratios below still use its recorded numbers, same as the on-screen panel)"
        : ""));
  }
  if (isZoomed()) {
    lines.push("# scope: zoomed window, elapsed " + scopeElapsedLabel() + " (from each arm's own run start)");
    visible.forEach(i => {
      const s = DATA[i];
      lines.push("#   " + s.name + " absolute window: " + new Date(s.t0 + viewTMin).toISOString() +
        " to " + new Date(s.t0 + viewTMax).toISOString());
    });
  } else {
    lines.push("# scope: full run (" + scopeElapsedLabel() + ")");
  }
  if (ctxFilterActive()) {
    lines.push("# context-band filter (input+cached tokens per request): >= " +
      (ctxFilter.min > 0 ? ctxFilter.min : "0") + ", <= " + (ctxFilter.max > 0 ? ctxFilter.max : "unbounded"));
  }
  if (snFilter.size > 0) {
    lines.push("# in-dataset series filter (sn): " + Array.from(snFilter).sort((a, b) => a - b).join(","));
  }
  const paramLines = visible.map(i => ({ name: DATA[i].name, ps: paramsSummaryFor(DATA[i]) })).filter(x => x.ps);
  if (paramLines.length) {
    paramLines.forEach(x => lines.push("# run params [" + x.name + "]: " + x.ps));
  } else {
    lines.push("# run params: not recorded for any included arm (predates the run_params header)");
  }
  (extraLines || []).forEach(l => lines.push("# " + l));
  lines.push("#");
  return lines;
}

// buildRequestsRows: one row per request, from every visible arm's CURRENT
// view (context band + series filter already applied via s._view), clipped
// to the current zoom window -- exactly the rows windowStats() sums for the
// summary panel over the same scope.
function buildRequestsRows() {
  const header = ["arm", "start_time", "ttft_ms", "response_time_ms", "series_num",
    "request_num", "cache_hit", "input_tokens", "cached_tokens", "output_tokens", "is_error"];
  const rows = [csvRow(header)];
  visibleIndices().forEach(i => {
    const s = DATA[i];
    s._view.forEach(r => {
      if (r.t < viewTMin || r.t > viewTMax) return;
      rows.push(csvRow([
        s.name,
        new Date(s.t0 + r.t).toISOString(),
        r.ttft, r.resp, r.sn, r.rn,
        r.ch ? "true" : "false",
        r.in, r.ca, r.out,
        r.err ? "true" : "false",
      ]));
    });
  });
  return rows;
}

// SUMMARY_CSV_COLUMNS names each SUMMARY_METRICS key for the CSV header --
// the on-screen abbreviations ("In/s", "TTFT50"...) aren't clear column
// names on their own, so the mapping lives in exactly one place.
const SUMMARY_CSV_COLUMNS = {
  in: "input_tokens", out: "output_tokens", reqs: "requests",
  inrate: "input_tokens_per_sec", outrate: "output_tokens_per_sec",
  ttft50: "ttft_p50_ms", ttft95: "ttft_p95_ms", err1k: "errors_per_1k",
};

// summaryRatioPct mirrors fmtRatio's own gating exactly (a zero/negative/
// non-finite baseline yields no ratio, just as on screen) but returns the
// numeric percentage instead of a formatted string, so the CSV's ratio
// column is a plain number a spreadsheet can compute with.
function summaryRatioPct(v, base) {
  if (!(base > 0) || !isFinite(v) || v < 0) return "";
  return Math.round((v / base) * 10000) / 100;
}

// buildSummaryRows reuses the exact per-series stats seriesStats() already
// computed for the on-screen panel (the same call renderSummary() makes), so
// the CSV can never disagree with what's displayed for this scope. Rows
// cover only the currently visible arms; when the baseline arm itself is
// hidden its numbers still back every other row's ratio column, exactly as
// the on-screen panel does (renderSummary reads perSeries[BASELINE_INDEX]
// unconditionally, regardless of hiddenSeries).
function buildSummaryRows() {
  const perSeries = seriesStats();
  const header = ["arm", "is_baseline"];
  SUMMARY_METRICS.forEach(m => {
    header.push(SUMMARY_CSV_COLUMNS[m.key]);
    if (BASELINE_INDEX >= 0) header.push(SUMMARY_CSV_COLUMNS[m.key] + "_pct_of_baseline");
  });
  const rows = [csvRow(header)];
  const baseStats = BASELINE_INDEX >= 0 ? perSeries[BASELINE_INDEX] : null;
  visibleIndices().forEach(i => {
    const s = DATA[i];
    const st = perSeries[i];
    const row = [s.name, i === BASELINE_INDEX ? "true" : "false"];
    SUMMARY_METRICS.forEach(m => {
      row.push(st ? m.val(st) : "");
      if (BASELINE_INDEX >= 0) {
        row.push(baseStats && st && i !== BASELINE_INDEX ? summaryRatioPct(m.val(st), m.val(baseStats)) : "");
      }
    });
    rows.push(csvRow(row));
  });
  return rows;
}

// triggerDownload builds a Blob from the given lines and clicks a throwaway
// <a download> at its object URL -- no network, no external library. The
// object URL is revoked right after the synthetic click so it doesn't leak.
function triggerDownload(filename, lines) {
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadRequestsCsv() {
  const filename = "wekai-requests-" + visibleIndices().length + "arms-" + scopeToken() + ".csv";
  const notes = [
    "NOTE: this is the 10-field JSONL subset (t, ttft, resp, err, sn, rn, ch, in, ca, out) -- the",
    "full per-request record (prompts, full timestamps, and more) lives in the run's .jsonl files",
    "on the results volume.",
  ];
  triggerDownload(filename, provenanceHeader("per-request rows", notes).concat(buildRequestsRows()));
}
function downloadSummaryCsv() {
  const filename = "wekai-summary-" + visibleIndices().length + "arms-" + scopeToken() + ".csv";
  triggerDownload(filename, provenanceHeader("summary panel", []).concat(buildSummaryRows()));
}
["downloadRequestsBtn", "modalDownloadRequestsBtn"].forEach(id => {
  const btn = document.getElementById(id);
  if (btn) btn.addEventListener("click", downloadRequestsCsv);
});
["downloadSummaryBtn", "modalDownloadSummaryBtn"].forEach(id => {
  const btn = document.getElementById(id);
  if (btn) btn.addEventListener("click", downloadSummaryCsv);
});

window.addEventListener("resize", () => { resize(); draw(); });
resize();
draw();
</script>
</body>
</html>
`))
