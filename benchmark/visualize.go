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
  h1 { font-size: 1.4em; font-weight: 500; margin-bottom: 8px; color: #F2F2EB; }
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
  #summaryTable td.err-hot { color: #FF6B6B; }
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
<h1>Benchmark Request Timeline</h1>
<div class="info" id="info"></div>
<div class="info" id="runparams"></div>
<div class="topgrid">
  <div class="panel" id="controlsPanel">
    <div class="panel-title">Controls<span class="spacer"></span><button class="panel-toggle" id="controlsToggle" title="Collapse">&minus;</button></div>
    <div class="panel-body">
    <div class="controls">
      <label><input type="checkbox" id="showTTFT" checked> TTFT p50</label>
      <label><input type="checkbox" id="showTTFTP95"> TTFT p95</label>
      <label><input type="checkbox" id="showResp" checked> Show Response Time</label>
      <label><input type="checkbox" id="showDots"> Show Requests</label>
      <label><input type="checkbox" id="showErrors" checked> Show Errors</label>
      <label><input type="checkbox" id="showTotals" checked> Show Totals (ingest)</label>
      <label><input type="checkbox" id="showXAxisValues"> Show X-axis values</label>
      <button id="resetZoom" disabled>Reset Zoom</button>
      <span id="zoomInfo" style="font-size:0.8em;color:#8a9096;"></span>
    </div>
    <div class="controls">
      <button id="ctxFilterBtn" style="border-color:#7C03EC;">Context Filter</button>
      <button id="selectAll">Select All</button>
      <button id="deselectAll">Deselect All</button>
      <input id="variantFilter" type="text" placeholder="Filter variants..." style="font-size:0.8em;padding:3px 8px;background:#1E2429;color:#F2F2EB;border:1px solid #42464A;border-radius:4px;width:200px;">
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
<div id="ctxModal" class="modal-backdrop">
  <div class="modal">
    <h2>Variants &amp; context filter</h2>
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
  // Plotted lines: rolling-window percentiles. Response = p50 only; TTFT =
  // p50 and p95 (dash pattern encodes the percentile, color the series).
  s._respP50 = [];
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
const dpr = window.devicePixelRatio || 1;

const margin = { top: 30, right: 20, bottom: 50, left: 70 };

// Annotation rows (cumulative request/error counts per series) are always
// printed under the axis: computeXStepSec keeps tick count at ~11-17 for any
// span, so the columns no longer overlap (the old flat 5m step crammed dozens
// of columns on multi-hour runs). Hover on a gridline still shows the same
// per-tick breakdown as a tooltip. Constant retained for the hover fallback
// threshold only.
const ANNOTATION_ROWS_MAX_DURATION = 3600;
let W, H, plotW, plotH;
let hiddenSeries = new Set();
// Positions of the x-axis ticks drawn in the most recent draw() call, so
// mousemove can hover-match a tick even when its annotation row isn't
// printed (long views -- see ANNOTATION_ROWS_MAX_DURATION).
let currentTicks = [];

// Zoom state
let globalTMin = Infinity, globalTMax = -Infinity, globalYMax = 0;
let viewTMin, viewTMax, viewYMax;
let dragStart = null; // pixel X where drag began
let dragCurrent = null;

function calcBottomMargin() {
  const duration = (viewTMax - viewTMin) / 1000;
  // Per-tick request-count rows only when "Show X-axis values" is on (the
  // totals volume layer carries the same story by default).
  let reqRows = 0;
  if (typeof xAxisValuesEnabled === "function" && xAxisValuesEnabled()) {
    reqRows = DATA.filter((_, i) => !hiddenSeries.has(i)).length;
  }
  // Dataset size gets NO axis rows: it's a slow-moving level, so a per-tick
  // column of near-identical "ds:13.5M" readings cost one row per series
  // (140px on a 10-variant report) to repeat what the band label and the band
  // hover already say exactly. The band is the place for it.
  // 20px for the time label + 14px per printed row (single line each);
  // adaptive ticks (11-17 columns) leave ample width.
  return 20 + reqRows * 14;
}

function resize() {
  margin.bottom = calcBottomMargin();
  W = Math.min(window.innerWidth - 32, 1800);
  // Measure the header instead of assuming a fixed chrome allowance: the
  // controls/summary grid reflows with variant count and viewport width (it
  // stacks to one column under 1100px), so a hardcoded reserve would either
  // clip the canvas or leave a gap.
  const headEl = document.getElementById("header");
  const headH = headEl ? headEl.getBoundingClientRect().height : 160;
  H = Math.max(420, Math.min(window.innerHeight - headH - 40, 800));
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  plotW = W - margin.left - margin.right;
  plotH = H - margin.top - margin.bottom;
}

// Compute global ranges
let totalRequests = 0;
DATA.forEach(s => {
  s.records.forEach(r => {
    if (r.t < globalTMin) globalTMin = r.t;
    if (r.t > globalTMax) globalTMax = r.t;
    if (r.resp > globalYMax) globalYMax = r.resp;
    if (r.ttft > globalYMax) globalYMax = r.ttft;
    totalRequests++;
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
globalYMax *= 1.1;
viewTMin = globalTMin; viewTMax = globalTMax; viewYMax = globalYMax;

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
  let totalOk = 0, totalErr = 0;
  const perSeries = seriesStats();
  perSeries.forEach(c => {
    totalOk += c.ok;
    totalErr += c.err;
  });
  renderSummary(perSeries);

  const span = ((viewTMax - viewTMin) / 1000).toFixed(1);
  const totalStr = formatCount(totalOk, totalErr);
  document.getElementById("info").innerHTML =
    DATA.length + " variants, " + totalStr + " requests" +
    (isZoomed() ? " (of " + totalRequests + " total)" : "") +
    ", time span: " + span + "s";

  // Update legend counts
  DATA.forEach((s, i) => {
    const countEl = document.getElementById("legend-count-" + i);
    if (countEl) countEl.innerHTML = "(" + formatCount(perSeries[i].ok, perSeries[i].err) + ")";
  });

  document.getElementById("resetZoom").disabled = !isZoomed();
  document.getElementById("zoomInfo").textContent = isZoomed() ? "Drag to zoom, click Reset or double-click to reset" : "Drag on chart to zoom into timeframe";
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
    item.classList.toggle("hidden");
    draw();
  };
  legendEl.appendChild(item);
  legendItems.push(item);
});

function syncLegendVisuals() {
  legendItems.forEach((item, i) => {
    item.classList.toggle("hidden", hiddenSeries.has(i));
  });
}

// --- Summary panel: per-variant totals over the SELECTED timeframe ---
// One ROW per variant, one column per metric: a report can carry ~10 arms but
// the metric set is fixed at six, so growth is vertical and the pane scrolls
// down a stable set of columns rather than sideways past the labels.
// Everything here honours the current view: the zoom window, the context
// band, and the in-dataset series selection (via s._view) — so zooming into
// an interesting stretch reprices every number. The skeleton is built once
// and only cell text is rewritten per draw, so a zoom drag never rebuilds DOM.
// Headers are abbreviated to keep all eight columns inside the pane; each
// carries the full wording as a hover title.
const SUMMARY_METRICS = [
  { key: "in",      short: "Input",  label: "Input tokens (prompt = uncached + server-cached)", fmt: st => fmtTokens(st.prompt) },
  { key: "out",     short: "Output", label: "Output tokens",            fmt: st => fmtTokens(st.outTok) },
  { key: "reqs",    short: "Reqs",   label: "Completed (non-error) requests", fmt: st => st.ok.toLocaleString() },
  { key: "inrate",  short: "In/s",   label: "Avg input tokens/s",       fmt: st => fmtTokens(st.prompt / st.spanSec) },
  { key: "outrate", short: "Out/s",  label: "Avg output tokens/s",      fmt: st => fmtTokens(st.outTok / st.spanSec) },
  { key: "ttft50",  short: "TTFT50", label: "TTFT p50 over the selected window (non-error requests)", fmt: st => fmtMs(st.ttft50) },
  { key: "ttft95",  short: "TTFT95", label: "TTFT p95 over the selected window (non-error requests)", fmt: st => fmtMs(st.ttft95) },
  { key: "err1k",   short: "Err/1k", label: "Errors per 1000 requests",  fmt: st => (st.total ? st.err / st.total * 1000 : 0).toFixed(1) },
];
const sumCells = []; // sumCells[seriesIndex][metricIndex] — mirrors the layout
const sumRows = [];
(function buildSummary() {
  const head = document.getElementById("sumHead");
  const vth = document.createElement("th");
  vth.className = "vcol";
  vth.textContent = "Variant";
  head.appendChild(vth);
  SUMMARY_METRICS.forEach(m => {
    const th = document.createElement("th");
    th.textContent = m.short;
    th.title = m.label;
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
    const cells = [];
    SUMMARY_METRICS.forEach(() => {
      const td = document.createElement("td");
      tr.appendChild(td);
      cells.push(td);
    });
    // A row IS a variant here, so it carries the same show/hide affordance as
    // the legend entry — with 10 arms the summary is where you're looking.
    tr.onclick = () => {
      if (hiddenSeries.has(i)) hiddenSeries.delete(i); else hiddenSeries.add(i);
      syncLegendVisuals();
      draw();
    };
    sumCells.push(cells);
    sumRows.push(tr);
    body.appendChild(tr);
  });
})();

// renderSummary repaints the panel from the per-series windowStats already
// computed by updateInfo — no extra pass over the records.
function renderSummary(perSeries) {
  DATA.forEach((s, si) => {
    const st = perSeries[si];
    SUMMARY_METRICS.forEach((m, mi) => {
      const td = sumCells[si][mi];
      td.textContent = st ? m.fmt(st) : "-";
      // The cached share is what a KV-offload arm actually buys, so it rides
      // along as a hover on the input cell instead of costing a column.
      if (m.key === "in" && st) {
        td.title = "prompt tokens = " + fmtTokens(st.inTok) + " uncached + " +
          fmtTokens(st.caTok) + " server-cached" +
          (st.prompt > 0 ? " (" + (st.caTok / st.prompt * 100).toFixed(1) + "% cached)" : "");
      }
      // Percentiles over a thin window are noise — say how many requests back
      // them rather than presenting 3 samples as a p95.
      if ((m.key === "ttft50" || m.key === "ttft95") && st) {
        td.title = st.ttftN + " non-error requests reported a first token in this window";
      }
      if (m.key === "err1k") td.classList.toggle("err-hot", !!st && st.err > 0);
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

document.getElementById("selectAll").addEventListener("click", () => {
  const filter = document.getElementById("variantFilter").value.toLowerCase();
  DATA.forEach((s, i) => {
    if (!filter || s.name.toLowerCase().includes(filter)) hiddenSeries.delete(i);
  });
  syncLegendVisuals();
  draw();
});

document.getElementById("deselectAll").addEventListener("click", () => {
  const filter = document.getElementById("variantFilter").value.toLowerCase();
  DATA.forEach((s, i) => {
    if (!filter || s.name.toLowerCase().includes(filter)) hiddenSeries.add(i);
  });
  syncLegendVisuals();
  draw();
});

document.getElementById("variantFilter").addEventListener("input", (e) => {
  const filter = e.target.value.toLowerCase();
  legendItems.forEach((item, i) => {
    const matches = !filter || DATA[i].name.toLowerCase().includes(filter);
    item.style.display = matches ? "" : "none";
    if (filter) {
      if (matches) hiddenSeries.delete(i); else hiddenSeries.add(i);
      item.classList.toggle("hidden", hiddenSeries.has(i));
    }
  });
  if (filter) draw();
});

function mapX(t) { return margin.left + ((t - viewTMin) / (viewTMax - viewTMin)) * plotW; }
function unmapX(px) { return viewTMin + ((px - margin.left) / plotW) * (viewTMax - viewTMin); }

function mapY(v) {
  return margin.top + plotH - (v / viewYMax) * plotH;
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

function recalcYMax() {
  viewYMax = 0;
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    s._view.forEach(r => {
      if (r.t < viewTMin || r.t > viewTMax) return;
      if (r.resp > viewYMax) viewYMax = r.resp;
      if (r.ttft > viewYMax) viewYMax = r.ttft;
    });
  });
  viewYMax = Math.max(viewYMax * 1.1, 1);
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
  const spanS = Math.max((t - t0) / 1000, 1e-9);
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

function xAxisValuesEnabled() {
  const cb = document.getElementById("showXAxisValues");
  return cb ? cb.checked : true;
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
  let bandH = MIX_BAND_H;
  if (bands.length * bandH > plotH * 0.6) {
    bandH = Math.max(24, Math.floor(plotH * 0.6 / bands.length));
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

function draw() {
  // Recalc bottom margin for visible series count
  const newBottom = calcBottomMargin();
  if (margin.bottom !== newBottom) { margin.bottom = newBottom; resize(); }

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
    if (y >= margin.top && y <= margin.top + plotH) {
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
  const showAnnotationRows = true; // adaptive tick density keeps columns readable at any span
  const startSec = Math.ceil(((viewTMin - globalTMin) / 1000) / xStepSec) * xStepSec;
  currentTicks = []; // rebuilt every draw; consumed by mousemove hover lookup
  for (let s = startSec; s <= (viewTMax - globalTMin) / 1000; s += xStepSec) {
    const x = mapX(globalTMin + s * 1000);
    if (x < margin.left || x > margin.left + plotW) continue;
    const tickTime = globalTMin + s * 1000;
    currentTicks.push({ x, tickTime });
    ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
    ctx.fillText(formatTickLabel(s), x, margin.top + plotH + 6);
    if (!showAnnotationRows) continue;
    ctx.font = "9px monospace";
    let row = 0;
    if (xAxisValuesEnabled()) tickStats(tickTime).forEach(({ si, cumReqs, cumErrs, maxSn }) => {
      const yBase = margin.top + plotH + 20 + row * 14;
      const snPart = "@" + maxSn;
      ctx.textAlign = "left";
      let cx;
      if (cumErrs > 0) {
        const p1 = "" + cumReqs + "(E";
        const p2 = "" + cumErrs;
        const p3 = ")";
        const fullW = ctx.measureText(p1 + p2 + p3 + snPart).width;
        cx = x - fullW / 2;
        ctx.fillStyle = "#C9C9C9";
        ctx.fillText(p1, cx, yBase);
        cx += ctx.measureText(p1).width;
        ctx.fillStyle = "#FF6B6B";
        ctx.fillText(p2, cx, yBase);
        cx += ctx.measureText(p2).width;
        ctx.fillStyle = "#C9C9C9";
        ctx.fillText(p3, cx, yBase);
        cx += ctx.measureText(p3).width;
      } else {
        const fullW = ctx.measureText("" + cumReqs + snPart).width;
        cx = x - fullW / 2;
        ctx.fillStyle = "#C9C9C9";
        ctx.fillText("" + cumReqs, cx, yBase);
        cx += ctx.measureText("" + cumReqs).width;
      }
      ctx.fillStyle = "#8a9096";
      ctx.fillText(snPart, cx, yBase);
      ctx.textAlign = "center";
      row++;
    });
    ctx.font = "11px monospace";
    ctx.fillStyle = "#8a9096";
  }

  // Row-identity chips: annotation text is neutral, so a small colored chip
  // at the left edge of each request-count row names its series.
  {
    let row = 0;
    const chip = si => {
      ctx.fillStyle = seriesColors[si];
      ctx.fillRect(margin.left - 13, margin.top + plotH + 20 + row * 14 + 1, 7, 7);
      row++;
    };
    if (xAxisValuesEnabled()) {
      DATA.forEach((ds, si) => { if (!hiddenSeries.has(si)) chip(si); });
    }
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

  // Data points (only when showDots is enabled)
  if (showDots) {
    DATA.forEach((s, si) => {
      if (hiddenSeries.has(si)) return;
      const color = seriesColors[si];

      s._view.forEach(r => {
        if (r.t < viewTMin || r.t > viewTMax) return;
        if (r.err && !showErrors) return;
        const x = mapX(r.t);

        // Connector line between TTFT and response
        if (showTTFT && showResp) {
          const y1 = mapY(r.ttft);
          const y2 = mapY(r.resp);
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.2;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke();
          ctx.globalAlpha = 1;
        }

        // TTFT dot (hollow circle)
        if (showTTFT && r.ttft > 0) {
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

        // Response time dot (solid circle)
        if (showResp) {
          const y = mapY(r.resp);
          ctx.beginPath();
          ctx.arc(x, y, r.err ? 4 : 2.5, 0, Math.PI * 2);
          ctx.fillStyle = r.err ? "#FF6B6B" : color;
          ctx.globalAlpha = r.err ? 0.8 : 0.7;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      });
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

  // Check proximity to dots (only when showDots is enabled)
  if (document.getElementById("showDots").checked) {
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

  // On long views the per-tick annotation rows aren't printed (see
  // ANNOTATION_ROWS_MAX_DURATION) to avoid an overlapping label band --
  // hovering near any x-axis gridline surfaces the same per-series
  // cumulative breakdown via tooltip instead.
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
      const lineLabel = { resp50: "Response p50", ttft50: "TTFT p50", ttft95: "TTFT p95" }[best.type] || best.type;
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
      showTooltip(e,
        "<b>" + best.s.name + "</b> (series " + r.sn + ", req " + r.rn + ")<br>" +
        "TTFT: " + r.ttft.toFixed(1) + " ms<br>" +
        "Response: " + r.resp.toFixed(1) + " ms<br>" +
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
canvas.addEventListener("dblclick", () => {
  viewTMin = globalTMin; viewTMax = globalTMax; viewYMax = globalYMax;
  draw();
});

// Reset button
document.getElementById("resetZoom").addEventListener("click", () => {
  viewTMin = globalTMin; viewTMax = globalTMax; viewYMax = globalYMax;
  draw();
});

// Wire up controls
["showTTFT","showTTFTP95","showResp","showDots","showErrors","showTotals","showXAxisValues"].forEach(id => {
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
  lbl.innerHTML = '<input type="checkbox" id="showCacheMix"> Show Cache Mix ' +
    '<span style="color:' + MIX_COMPUTE_COLOR + '">&#9632;</span>compute ' +
    '<span style="color:' + MIX_LOCAL_COLOR + '">&#9632;</span>local cache ' +
    '<span style="color:' + MIX_EXTERNAL_COLOR + '">&#9632;</span>external KV ' +
    '<span style="color:' + ADT_LINE_COLOR + '">&#8213;</span>active dataset (tokens)';
  document.querySelector(".controls").appendChild(lbl);
  const cb = document.getElementById("showCacheMix");
  // Set the property rather than baking "checked" into the markup: one source
  // of truth for the default, and it survives the element already existing.
  cb.checked = cacheMixDefaultOn(DATA.length);
  cb.addEventListener("change", draw);
}

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
  const rebuildList = () => {
    listEl.innerHTML = "";
    DATA.forEach((s, i) => {
      const row = document.createElement("div");
      row.className = "modal-series-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !hiddenSeries.has(i);
      cb.addEventListener("change", () => {
        if (cb.checked) hiddenSeries.delete(i); else hiddenSeries.add(i);
        syncLegendVisuals();
        draw();
      });
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
  window.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });
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

window.addEventListener("resize", () => { resize(); draw(); });
resize();
draw();
</script>
</body>
</html>
`))
