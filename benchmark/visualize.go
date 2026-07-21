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
	}

	type seriesData struct {
		Name    string      `json:"name"`
		Records []vizRecord `json:"records"`
	}

	var allSeries []seriesData
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		records, err := readJSONLFile(f)
		if err != nil {
			return "", fmt.Errorf("read %s: %w", f, err)
		}
		// Prefer the clean model alias (e.g. "DS3H_weka-64r8w") over the raw
		// sanitized filename (e.g. "dynamic_http___..._alias_DS3H_weka-64r8w")
		// when the file's records unambiguously identify one model.
		if alias := resolveRecordsAlias(records); alias != "" {
			name = alias
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
			})
		}
		allSeries = append(allSeries, seriesData{Name: name, Records: vr})
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

func readJSONLFile(path string) ([]requestDataRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var records []requestDataRecord
	sc := bufio.NewScanner(f)
	// 64 MiB cap: reqdata rows embed full prompts; 300k-token contexts exceed 1 MiB.
	sc.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	for sc.Scan() {
		var r requestDataRecord
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			continue // skip malformed lines
		}
		records = append(records, r)
	}
	return records, sc.Err()
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
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 16px; }
  h1 { font-size: 1.4em; margin-bottom: 8px; color: #e0e0e0; }
  .info { font-size: 0.85em; color: #888; margin-bottom: 12px; }
  .controls { margin-bottom: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .controls label { font-size: 0.85em; cursor: pointer; }
  .controls input[type=checkbox] { margin-right: 4px; }
  .controls button { font-size: 0.8em; padding: 3px 10px; background: #0f3460; color: #e0e0e0; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
  .controls button:hover { background: #1a4a7a; }
  .controls button:disabled { opacity: 0.3; cursor: default; }
  #legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 12px; font-size: 0.8em; }
  .legend-item { display: flex; align-items: center; gap: 4px; cursor: pointer; opacity: 1; }
  .legend-item.hidden { opacity: 0.35; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  .legend-count { color: #888; font-size: 0.9em; }
  canvas { background: #16213e; border-radius: 8px; display: block; cursor: crosshair; }
  #tooltip { position: fixed; background: #0f3460; border: 1px solid #555; border-radius: 6px; padding: 8px 10px; font-size: 0.8em; pointer-events: none; display: none; z-index: 100; max-width: 300px; line-height: 1.5; }
</style>
</head>
<body>
<h1>Benchmark Request Timeline</h1>
<div class="info" id="info"></div>
<div class="controls">
  <label><input type="checkbox" id="showTTFT" checked> Show TTFT</label>
  <label><input type="checkbox" id="showResp" checked> Show Response Time</label>
  <label><input type="checkbox" id="showDots"> Show Dots</label>
  <label><input type="checkbox" id="showErrors" checked> Show Errors</label>
  <button id="resetZoom" disabled>Reset Zoom</button>
  <span id="zoomInfo" style="font-size:0.8em;color:#888;"></span>
</div>
<div style="margin-bottom:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
  <button id="selectAll" style="font-size:0.8em;padding:3px 10px;background:#0f3460;color:#e0e0e0;border:1px solid #555;border-radius:4px;cursor:pointer;">Select All</button>
  <button id="deselectAll" style="font-size:0.8em;padding:3px 10px;background:#0f3460;color:#e0e0e0;border:1px solid #555;border-radius:4px;cursor:pointer;">Deselect All</button>
  <input id="seriesFilter" type="text" placeholder="Filter series..." style="font-size:0.8em;padding:3px 8px;background:#16213e;color:#e0e0e0;border:1px solid #555;border-radius:4px;width:200px;">
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

// Normalize timestamps: align each series to start at t=0
DATA.forEach(s => {
  if (s.records.length === 0) return;
  let tMin = Infinity;
  s.records.forEach(r => { if (r.t < tMin) tMin = r.t; });
  s.records.forEach(r => { r.t -= tMin; });
});

// Precompute moving averages and error bars per series
DATA.forEach(s => {
  const sorted = s.records.filter(r => !r.err).slice().sort((a, b) => a.t - b.t);
  s._sorted = sorted;
  const maxSn = sorted.reduce((mx, r) => Math.max(mx, r.sn), 1);
  const winSize = CONCURRENCY > 0 ? CONCURRENCY * 3 : Math.max(maxSn * 3, 10);
  s._winSize = winSize;
  s._avgTTFT = [];
  s._avgResp = [];
  for (let i = 0; i < sorted.length; i++) {
    const start = Math.max(0, i - winSize + 1);
    const win = sorted.slice(start, i + 1);
    const ttfts = win.map(r => r.ttft).filter(v => v > 0);
    const resps = win.map(r => r.resp);
    s._avgTTFT.push({ t: sorted[i].t, v: ttfts.length ? ttfts.reduce((a,b)=>a+b,0)/ttfts.length : 0 });
    s._avgResp.push({ t: sorted[i].t, v: resps.reduce((a,b)=>a+b,0)/resps.length });
  }
  // Precompute error bars: sample every winSize points from ALL records (including errors)
  // Each bar is anchored at the response avg line at that time
  const allSorted = s.records.slice().sort((a, b) => a.t - b.t);
  s._errBars = [];
  let avgIdx = 0;
  for (let i = winSize - 1; i < allSorted.length; i += winSize) {
    const start = Math.max(0, i - winSize + 1);
    let errs = 0, total = 0;
    for (let j = start; j <= i; j++) { total++; if (allSorted[j].err) errs++; }
    if (errs > 0) {
      // Find nearest avgResp value for this timestamp
      const t = allSorted[i].t;
      while (avgIdx < s._avgResp.length - 1 && s._avgResp[avgIdx].t < t) avgIdx++;
      const respAvg = s._avgResp.length > 0 ? s._avgResp[Math.min(avgIdx, s._avgResp.length - 1)].v : 0;
      s._errBars.push({ t: t, errRate: errs / total, errs: errs, total: total, respAvg: respAvg });
    }
  }
});

function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  const idx = Math.ceil(s.length * p) - 1;
  return s[Math.max(0, idx)];
}

// --- Color assignment ---
// No reds — red is reserved for errors
const OTHER_PALETTE = [
  "#f39c12","#1abc9c","#e67e22","#00bcd4","#4bc0c0",
  "#ff9f40","#536dfe","#ffea00","#76ff03","#ff9100",
  "#00e676","#c9cbcf","#26a69a",
];
const GREEN_VARIANTS = ["#2ecc71","#27ae60","#00e676","#4caf50","#66bb6a"];
const BLUE_VARIANTS  = ["#29b6f6","#64b5f6","#00acc1","#0277bd","#4fc3f7","#0097a7"];
const PURPLE_VARIANTS = [
  "#e040fb","#7c4dff","#ff4081","#aa00ff","#ce93d8","#d500f9",
];

const seriesColors = [];
{
  let gpuIdx = 0, dramIdx = 0, wekaIdx = 0, otherIdx = 0;
  DATA.forEach(s => {
    const c = classifyAlias(getAlias(s.name));
    if (c === "gpu") {
      seriesColors.push(GREEN_VARIANTS[gpuIdx % GREEN_VARIANTS.length]);
      gpuIdx++;
    } else if (c === "dram") {
      seriesColors.push(BLUE_VARIANTS[dramIdx % BLUE_VARIANTS.length]);
      dramIdx++;
    } else if (c === "weka") {
      seriesColors.push(PURPLE_VARIANTS[wekaIdx % PURPLE_VARIANTS.length]);
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
  const visibleCount = DATA.filter((_, i) => !hiddenSeries.has(i)).length;
  const duration = (viewTMax - viewTMin) / 1000;
  // Rows are always printed: adaptive ticks (11-17 columns) leave ample width.
  // 20px for time label + 14px per visible series (single line each)
  return 20 + visibleCount * 14;
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
  if (err === 0) return '<span style="color:#2ecc71">' + ok + '</span>';
  return '<span style="color:#2ecc71">' + ok + '</span>, <span style="color:#ff4444">' + err + '</span>';
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
  ctx.strokeStyle = "#2a2a4a";
  ctx.lineWidth = 0.5;
  ctx.fillStyle = "#888";
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
    tickStats(tickTime).forEach(({ si, cumReqs, cumErrs, maxSn }) => {
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
        ctx.fillStyle = seriesColors[si];
        ctx.fillText(p1, cx, yBase);
        cx += ctx.measureText(p1).width;
        ctx.fillStyle = "#ff4444";
        ctx.fillText(p2, cx, yBase);
        cx += ctx.measureText(p2).width;
        ctx.fillStyle = seriesColors[si];
        ctx.fillText(p3, cx, yBase);
        cx += ctx.measureText(p3).width;
      } else {
        const fullW = ctx.measureText("" + cumReqs + snPart).width;
        cx = x - fullW / 2;
        ctx.fillStyle = seriesColors[si];
        ctx.fillText("" + cumReqs, cx, yBase);
        cx += ctx.measureText("" + cumReqs).width;
      }
      ctx.fillStyle = "#2ecc71";
      ctx.fillText(snPart, cx, yBase);
      ctx.textAlign = "center";
      row++;
    });
    ctx.font = "11px monospace";
    ctx.fillStyle = "#888";
  }

  // Y axis label
  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#aaa";
  ctx.font = "12px sans-serif";
  ctx.fillText("Latency", 0, 0);
  ctx.restore();

  // X axis label
  ctx.fillStyle = "#aaa";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";

  // Plot border
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Clip to plot area for data points
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.clip();

  // Moving average lines
  DATA.forEach((s, si) => {
    if (hiddenSeries.has(si)) return;
    const color = seriesColors[si];

    // Response time average line (solid)
    if (showResp && s._avgResp.length > 1) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.8;
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      s._avgResp.forEach(p => {
        if (p.t < viewTMin || p.t > viewTMax) return;
        const x = mapX(p.t), y = mapY(p.v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // TTFT average line (dashed)
    if (showTTFT && s._avgTTFT.length > 1) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.8;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      let started = false;
      s._avgTTFT.forEach(p => {
        if (p.t < viewTMin || p.t > viewTMax || p.v <= 0) return;
        const x = mapX(p.t), y = mapY(p.v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
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
        ctx.fillStyle = "#ff4444";
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
            ctx.fillStyle = "#ff4444";
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
          ctx.fillStyle = r.err ? "#ff4444" : color;
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
    ctx.fillStyle = "rgba(52, 152, 219, 0.15)";
    ctx.fillRect(x1, margin.top, x2 - x1, plotH);
    ctx.strokeStyle = "#3498db";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x1, margin.top, x2 - x1, plotH);
    ctx.setLineDash([]);
  }

  updateInfo();
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
      if (document.getElementById("showResp").checked) checkLine(s._avgResp, "resp");
      if (document.getElementById("showTTFT").checked) checkLine(s._avgTTFT, "ttft");
    });
  }

  // On long views the per-tick annotation rows aren't printed (see
  // ANNOTATION_ROWS_MAX_DURATION) to avoid an overlapping label band --
  // hovering near any x-axis gridline surfaces the same per-series
  // cumulative breakdown via tooltip instead.
  let tickHover = null;
  if (!best && (viewTMax - viewTMin) / 1000 > ANNOTATION_ROWS_MAX_DURATION && my >= margin.top) {
    let bestTickDist = 10;
    currentTicks.forEach(tk => {
      const d = Math.abs(mx - tk.x);
      if (d < bestTickDist) { bestTickDist = d; tickHover = tk; }
    });
  }

  if (best) {
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top = (e.clientY - 10) + "px";
    if (best.isLine) {
      const win = best.win;
      const vals = best.type === "ttft" ? win.map(r => r.ttft).filter(v => v > 0) : win.map(r => r.resp);
      const p50 = percentile(vals, 0.5);
      const p95 = percentile(vals, 0.95);
      const fmt = v => v >= 1000 ? (v/1000).toFixed(2) + "s" : v.toFixed(0) + "ms";
      // Count total requests, errors, and max series index up to this point
      let totalUpTo = 0, errUpTo = 0, maxSnSeen = 0;
      best.s.records.forEach(r => { if (r.t <= best.t) { totalUpTo++; if (r.err) errUpTo++; if (r.sn > maxSnSeen) maxSnSeen = r.sn; } });
      tooltip.innerHTML =
        "<b>" + best.s.name + "</b> \u2014 " + (best.type === "ttft" ? "TTFT" : "Response") + " avg<br>" +
        "Window: " + win.length + " requests<br>" +
        "Avg: " + fmt(best.avgVal) + "<br>" +
        "p50: " + fmt(p50) + "<br>" +
        "p95: " + fmt(p95) + "<br>" +
        "Series: " + maxSnSeen + "<br>" +
        "Total: " + totalUpTo + (errUpTo > 0 ? ", <span style='color:#ff4444'>errors: " + errUpTo + "</span>" : "");
    } else {
      const r = best.r;
      tooltip.innerHTML =
        "<b>" + best.s.name + "</b> (series " + r.sn + ", req " + r.rn + ")<br>" +
        "TTFT: " + r.ttft.toFixed(1) + " ms<br>" +
        "Response: " + r.resp.toFixed(1) + " ms<br>" +
        (r.ch ? "Cache hit<br>" : "") +
        (r.err ? "<span style='color:#ff4444'>ERROR</span>" : "");
    }
  } else if (tickHover) {
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top = (e.clientY - 10) + "px";
    const elapsedSec = Math.round((tickHover.tickTime - globalTMin) / 1000);
    const rows = tickStats(tickHover.tickTime).map(({ si, cumReqs, cumErrs, maxSn }) => {
      const errPart = cumErrs > 0 ? ", <span style='color:#ff4444'>errors: " + cumErrs + "</span>" : "";
      return "<span style='color:" + seriesColors[si] + "'>" + DATA[si].name + "</span>: " +
        cumReqs + " reqs" + errPart + ", series @" + maxSn;
    });
    tooltip.innerHTML = "<b>t = " + formatTickLabel(elapsedSec) + "</b><br>" + rows.join("<br>");
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
["showTTFT","showResp","showDots","showErrors"].forEach(id => {
  document.getElementById(id).addEventListener("change", draw);
});

window.addEventListener("resize", () => { resize(); draw(); });
resize();
draw();
</script>
</body>
</html>
`))
