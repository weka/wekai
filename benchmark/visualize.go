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
	return generateVisualization(dir, concurrency, false)
}

// generateVisualization is the implementation behind GenerateVisualization.
// keepFileNames pins each series' DISPLAYED name (legend, cache-mix band
// label, tooltips, ds: axis rows — all render seriesData.Name) to the .jsonl
// basename instead of re-resolving the record alias. The merged path sets it
// when explicit --labels were given, so labels win end-to-end: two arms
// sharing one alias would otherwise render indistinguishably.
func generateVisualization(dir string, concurrency int, keepFileNames bool) (string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		return "", fmt.Errorf("glob jsonl files: %w", err)
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no .jsonl files found in %s", dir)
	}
	sort.Strings(files)

	type vizRecord struct {
		StartTimeUnix float64 `json:"t"`
		TTFT          float64 `json:"ttft"`
		ResponseMs    float64 `json:"resp"`
		IsError       bool    `json:"err"`
		SeriesNum     int     `json:"sn"`
		RequestNum    int     `json:"rn"`
		CacheHit      bool    `json:"ch"`
		// Token counts for the ingest volume layer and its hover rates.
		InputTokens  int `json:"in"`  // net-of-cache input tokens
		CachedTokens int `json:"ca"`  // server-cached prompt tokens
		OutputTokens int `json:"out"` // completion tokens
	}

	type seriesData struct {
		Name    string             `json:"name"`
		Records []vizRecord        `json:"records"`
		Mix     []vizSampleSegment `json:"mix,omitempty"`
		Adt     []vizAdtPoint      `json:"adt,omitempty"`
	}

	var allSeries []seriesData
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		records, samples, err := readJSONLFile(f)
		if err != nil {
			return "", fmt.Errorf("read %s: %w", f, err)
		}
		// Prefer the clean model alias (e.g. "DS3H_weka-64r8w") over the raw
		// sanitized filename (e.g. "dynamic_http___..._alias_DS3H_weka-64r8w")
		// when the file's records unambiguously identify one model — unless
		// the caller pinned names to the (label-derived) filenames.
		if !keepFileNames {
			if alias := resolveRecordsAlias(records); alias != "" {
				name = alias
			}
		}
		var vr []vizRecord
		for _, r := range records {
			vr = append(vr, vizRecord{
				StartTimeUnix: float64(r.StartTime.UnixMilli()),
				TTFT:          r.TTFT,
				ResponseMs:    r.ResponseMs,
				IsError:       r.IsError,
				SeriesNum:     r.SeriesNum,
				RequestNum:    r.RequestNum,
				CacheHit:      r.CacheHit,
				InputTokens:   r.InputTokens,
				CachedTokens:  r.CachedTokens,
				OutputTokens:  r.OutputTokens,
			})
		}
		mix, adt := buildSampleViz(samples)
		allSeries = append(allSeries, seriesData{Name: name, Records: vr, Mix: mix, Adt: adt})
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

// readJSONLFile reads a request-data JSONL file, routing lines by their
// record_type: absent/empty = a request row (legacy files predate the field),
// "vllm_metrics_sample" = a metrics sample. Unknown record types and
// malformed lines are skipped — a new record type must never corrupt request
// parsing (unmarshalling a sample into requestDataRecord would otherwise
// "succeed" as an all-zero phantom request).
func readJSONLFile(path string) ([]requestDataRecord, []vllmMetricsSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var records []requestDataRecord
	var samples []vllmMetricsSample
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
		}
	}
	return records, samples, sc.Err()
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
  .controls { margin-bottom: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
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
  canvas { background: #171C20; border-radius: 8px; display: block; cursor: crosshair; }
  #tooltip { position: fixed; background: #1E2429; border: 1px solid #42464A; border-radius: 6px; padding: 8px 10px; font-size: 0.8em; pointer-events: none; display: none; z-index: 100; max-width: 300px; line-height: 1.5; }
</style>
</head>
<body>
<div class="brandbar"></div>
<h1>Benchmark Request Timeline</h1>
<div class="info" id="info"></div>
<div class="controls">
  <label><input type="checkbox" id="showTTFT" checked> Show TTFT</label>
  <label><input type="checkbox" id="showTTFTP95" checked> TTFT p95</label>
  <label><input type="checkbox" id="showResp" checked> Show Response Time</label>
  <label><input type="checkbox" id="showDots"> Show Requests</label>
  <label><input type="checkbox" id="showErrors" checked> Show Errors</label>
  <label><input type="checkbox" id="showTotals" checked> Show Totals (ingest)</label>
  <label><input type="checkbox" id="showXAxisValues"> Show X-axis values</label>
  <button id="resetZoom" disabled>Reset Zoom</button>
  <span id="zoomInfo" style="font-size:0.8em;color:#8a9096;"></span>
</div>
<div style="margin-bottom:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
  <button id="selectAll" style="font-size:0.8em;padding:3px 10px;background:#1E2429;color:#F2F2EB;border:1px solid #42464A;border-radius:4px;cursor:pointer;">Select All</button>
  <button id="deselectAll" style="font-size:0.8em;padding:3px 10px;background:#1E2429;color:#F2F2EB;border:1px solid #42464A;border-radius:4px;cursor:pointer;">Deselect All</button>
  <input id="seriesFilter" type="text" placeholder="Filter series..." style="font-size:0.8em;padding:3px 8px;background:#1E2429;color:#F2F2EB;border:1px solid #42464A;border-radius:4px;width:200px;">
</div>
<div id="legend"></div>
<canvas id="chart"></canvas>
<div id="tooltip"></div>

<script>
const RAW_DATA = {{.Data}};
const CONCURRENCY = {{.Concurrency}};

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

// Precompute moving averages and error bars per series
DATA.forEach(s => {
  // Time-ordered records with cumulative INGEST tokens (input + cached —
  // the full context volume, same quantity as wekai's in= counter) for the
  // totals volume layer and its hover rates.
  const byT = s.records.slice().sort((a, b) => a.t - b.t);
  s._byT = byT;
  s._cumTimes = byT.map(r => r.t);
  s._cumTokens = [];
  let ingestAcc = 0;
  byT.forEach(r => { ingestAcc += (r.in || 0) + (r.ca || 0); s._cumTokens.push(ingestAcc); });
  const sorted = s.records.filter(r => !r.err).slice().sort((a, b) => a.t - b.t);
  s._sorted = sorted;
  const maxSn = sorted.reduce((mx, r) => Math.max(mx, r.sn), 1);
  const winSize = CONCURRENCY > 0 ? CONCURRENCY * 3 : Math.max(maxSn * 3, 10);
  s._winSize = winSize;
  // Plotted lines: rolling-window percentiles. Response = p50 only; TTFT =
  // p50 and p95 (dash pattern encodes the percentile, color the series).
  s._respP50 = [];
  s._ttftP50 = [];
  s._ttftP95 = [];
  for (let i = 0; i < sorted.length; i++) {
    const start = Math.max(0, i - winSize + 1);
    const win = sorted.slice(start, i + 1);
    const ttfts = win.map(r => r.ttft).filter(v => v > 0);
    const resps = win.map(r => r.resp);
    const t = sorted[i].t;
    s._respP50.push({ t: t, v: percentile(resps, 0.5) });
    s._ttftP50.push({ t: t, v: ttfts.length ? percentile(ttfts, 0.5) : 0 });
    s._ttftP95.push({ t: t, v: ttfts.length ? percentile(ttfts, 0.95) : 0 });
  }
  // Precompute error bars: sample every winSize points from ALL records (including errors)
  // Each bar is anchored at the response p50 line at that time
  const allSorted = s.records.slice().sort((a, b) => a.t - b.t);
  s._errBars = [];
  let avgIdx = 0;
  for (let i = winSize - 1; i < allSorted.length; i += winSize) {
    const start = Math.max(0, i - winSize + 1);
    let errs = 0, total = 0;
    for (let j = start; j <= i; j++) { total++; if (allSorted[j].err) errs++; }
    if (errs > 0) {
      // Find nearest response-p50 value for this timestamp
      const t = allSorted[i].t;
      while (avgIdx < s._respP50.length - 1 && s._respP50[avgIdx].t < t) avgIdx++;
      const respAvg = s._respP50.length > 0 ? s._respP50[Math.min(avgIdx, s._respP50.length - 1)].v : 0;
      s._errBars.push({ t: t, errRate: errs / total, errs: errs, total: total, respAvg: respAvg });
    }
  }
});

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
  // Dataset-size rows: one extra row per visible sampled series when the
  // cache-mix overlay is enabled.
  let dsRows = 0;
  if (typeof cacheMixEnabled === "function" && cacheMixEnabled()) {
    dsRows = DATA.filter((s, i) => !hiddenSeries.has(i) && s.adt && s.adt.length).length;
  }
  // 20px for the time label + 14px per printed row (single line each);
  // adaptive ticks (11-17 columns) leave ample width.
  return 20 + (reqRows + dsRows) * 14;
}

function resize() {
  margin.bottom = calcBottomMargin();
  W = Math.min(window.innerWidth - 32, 1800);
  H = Math.max(500, Math.min(window.innerHeight - 200, 800));
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

function countRecords(records) {
  let ok = 0, err = 0;
  records.forEach(r => {
    if (r.t >= viewTMin && r.t <= viewTMax) { if (r.err) err++; else ok++; }
  });
  return { ok, err, total: ok + err };
}

function formatCount(ok, err) {
  if (err === 0) return '<span style="color:#C9C9C9">' + ok + '</span>';
  return '<span style="color:#C9C9C9">' + ok + '</span>, <span style="color:#FF6B6B">' + err + '</span>';
}

function updateInfo() {
  let totalOk = 0, totalErr = 0;
  const perSeries = [];
  DATA.forEach((s, i) => {
    const c = countRecords(s.records);
    perSeries.push(c);
    totalOk += c.ok;
    totalErr += c.err;
  });

  const span = ((viewTMax - viewTMin) / 1000).toFixed(1);
  const totalStr = formatCount(totalOk, totalErr);
  document.getElementById("info").innerHTML =
    DATA.length + " series, " + totalStr + " requests" +
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
  item.innerHTML = '<div class="legend-dot" style="background:' + seriesColors[i] + '"></div>' +
    s.name + ' <span class="legend-count" id="legend-count-' + i + '">(' + formatCount(ok, err) + ')</span>';
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

document.getElementById("selectAll").addEventListener("click", () => {
  const filter = document.getElementById("seriesFilter").value.toLowerCase();
  DATA.forEach((s, i) => {
    if (!filter || s.name.toLowerCase().includes(filter)) hiddenSeries.delete(i);
  });
  syncLegendVisuals();
  draw();
});

document.getElementById("deselectAll").addEventListener("click", () => {
  const filter = document.getElementById("seriesFilter").value.toLowerCase();
  DATA.forEach((s, i) => {
    if (!filter || s.name.toLowerCase().includes(filter)) hiddenSeries.add(i);
  });
  syncLegendVisuals();
  draw();
});

document.getElementById("seriesFilter").addEventListener("input", (e) => {
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
    s.records.forEach(r => {
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
    ds.records.forEach(r => { if (r.t <= tickTime) { cumReqs++; if (r.err) cumErrs++; if (r.sn > maxSn) maxSn = r.sn; } });
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

// percentileDash: canvas dash pattern per plotted line kind — the pattern
// encodes the percentile, the color encodes the series; the two channels
// never mix. Response p50 solid; TTFT p50 dense dots (primary); TTFT p95
// long sparse dashes (secondary).
function percentileDash(kind) {
  if (kind === "ttft50") return [2, 3];
  if (kind === "ttft95") return [10, 7];
  return []; // resp50: solid
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
  let adtMax = 0;
  bands.forEach(b => (b.s.adt || []).forEach(p => { if (p.v > adtMax) adtMax = p.v; }));
  return { bands, adtMax };
}

function drawCacheMix() {
  const layout = cacheMixLayout();
  if (!layout) return;
  const adtMax = layout.adtMax;

  layout.bands.forEach(({ s, yTop, bandH }) => {
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

    // Active-dataset line: 0 at band bottom, shared adtMax at band top.
    // Thin off-white — the muted fills leave it plenty of contrast.
    if (adtMax > 0 && s.adt && s.adt.length > 1) {
      ctx.strokeStyle = ADT_LINE_COLOR;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      let started = false;
      s.adt.forEach(p => {
        const x = mapX(p.t);
        const y = yTop + bandH - (p.v / adtMax) * bandH;
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
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
    if (s.adt && s.adt.length) {
      const last = s.adt[s.adt.length - 1];
      ctx.fillStyle = ADT_LINE_COLOR;
      ctx.textAlign = "right";
      ctx.fillText("active dataset (tokens): " + fmtTokens(last.v) +
        " | " + last.s + " series | scale 0-" + fmtTokens(adtMax),
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
    // Dataset-size rows: active-dataset tokens as of each tick, one row per
    // visible sampled series, sharing the sparse adaptive tick columns.
    // Text stays neutral; the row's left-edge chip names the series.
    if (cacheMixEnabled()) {
      DATA.forEach((ds, si) => {
        if (hiddenSeries.has(si) || !ds.adt || !ds.adt.length) return;
        const p = adtAt(ds.adt, tickTime);
        const label = "ds:" + (p ? fmtTokens(p.v) : "-");
        const yBase = margin.top + plotH + 20 + row * 14;
        ctx.textAlign = "left";
        ctx.fillStyle = "#C9C9C9";
        ctx.fillText(label, x - ctx.measureText(label).width / 2, yBase);
        ctx.textAlign = "center";
        row++;
      });
    }
    ctx.font = "11px monospace";
    ctx.fillStyle = "#8a9096";
  }

  // Row-identity chips: annotation text is neutral, so a small colored chip
  // at the left edge of each row names its series (request rows first, then
  // ds: rows, matching the per-tick row order).
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
    if (cacheMixEnabled()) {
      DATA.forEach((ds, si) => {
        if (!hiddenSeries.has(si) && ds.adt && ds.adt.length) chip(si);
      });
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

      s.records.forEach(r => {
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
  let stackAll = 0;
  geo.visible.forEach(({ s }) => { stackAll += cumTokensAt(s._cumTimes, s._cumTokens, tc); });
  return { s: hit.s, si: hit.si, tc: tc, cumTok: cumTok, share: stackAll > 0 ? cumTok / stackAll : 0 };
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
      s.records.forEach(r => {
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
      best.s.records.forEach(r => { if (r.t <= best.t) { totalUpTo++; if (r.err) errUpTo++; if (r.sn > maxSnSeen) maxSnSeen = r.sn; } });
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
      fmtTokens(volHover.cumTok) + " tok cumulative (" + (100 * volHover.share).toFixed(0) + "% of stack)<br>" +
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

  viewTMin = unmapX(minPx);
  viewTMax = unmapX(maxPx);
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
if (HAS_CACHE_MIX) {
  const lbl = document.createElement("label");
  lbl.innerHTML = '<input type="checkbox" id="showCacheMix" checked> Show Cache Mix ' +
    '<span style="color:' + MIX_COMPUTE_COLOR + '">&#9632;</span>compute ' +
    '<span style="color:' + MIX_LOCAL_COLOR + '">&#9632;</span>local cache ' +
    '<span style="color:' + MIX_EXTERNAL_COLOR + '">&#9632;</span>external KV ' +
    '<span style="color:' + ADT_LINE_COLOR + '">&#8213;</span>active dataset (tokens)';
  document.querySelector(".controls").appendChild(lbl);
  document.getElementById("showCacheMix").addEventListener("change", draw);
}

window.addEventListener("resize", () => { resize(); draw(); });
resize();
draw();
</script>
</body>
</html>
`))
