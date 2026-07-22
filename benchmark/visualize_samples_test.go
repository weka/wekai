package benchmark

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeMixedJSONL(t *testing.T, dir, name string, records []requestDataRecord, samples []vllmMetricsSample) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, name+".jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range samples {
		if err := enc.Encode(s); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func benchFixtureData(alias string, base time.Time) ([]requestDataRecord, []vllmMetricsSample) {
	model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=" + alias
	var records []requestDataRecord
	for i := 0; i < 5; i++ {
		st := base.Add(time.Duration(i) * 20 * time.Second)
		records = append(records, requestDataRecord{
			StartTime:    st,
			EndTime:      st.Add(2 * time.Second),
			TTFT:         150,
			ResponseMs:   2000,
			Model:        model,
			SeriesNum:    1,
			RequestNum:   i + 1,
			InputTokens:  100,
			CachedTokens: 400,
			OutputTokens: 50,
		})
	}
	var samples []vllmMetricsSample
	for i := 0; i < 3; i++ {
		samples = append(samples, vllmMetricsSample{
			RecordType:          recordTypeVLLMMetricsSample,
			TS:                  base.Add(time.Duration(i) * 60 * time.Second),
			Model:               model,
			Sources:             vllmSourceCounters{Compute: int64(1000 * (i + 1)), LocalCache: int64(300 * i), ExternalCache: int64(100 * i)},
			ActiveDatasetTokens: int64(4000 * (i + 1)),
			ActiveSeries:        i + 1,
		})
	}
	return records, samples
}

func TestGenerateVisualizationWithCacheMixSamples(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	recA, smpA := benchFixtureData("mixA_weka", base)
	recB, smpB := benchFixtureData("mixB_hbm", base)
	writeMixedJSONL(t, dir, "a", recA, smpA)
	writeMixedJSONL(t, dir, "b", recB, smpB)

	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	for _, want := range []string{
		`"mix":`, `"adt":`, // per-series overlay data embedded
		"drawCacheMix", "showCacheMix", "HAS_CACHE_MIX", // overlay code paths
		"MIX_COMPUTE_COLOR", "active dataset (tokens)",
		"MIX_TOTAL_MAX", "mixStackHeight", "mixRate", // absolute band scaling
		"drawTotals", "totalsStack", "showTotals", "showXAxisValues", // totals volume layer + axis toggle
		"volumeGeometry", "volumeHoverAt", "windowRates", "Show Totals (ingest)", // ingest volume + hover
	} {
		if !strings.Contains(html, want) {
			t.Errorf("generated HTML missing %q", want)
		}
	}
	// Deltas, not cumulatives, are embedded: seg0 compute delta is 1000.
	if !strings.Contains(html, `"t0":`) {
		t.Errorf("mix segments not embedded")
	}
}

func TestGenerateVisualizationWithoutSamplesUnchanged(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, _ := benchFixtureData("plain_run", base)
	writeMixedJSONL(t, dir, "a", rec, nil)

	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	// No samples => no overlay data keys in the embedded series JSON; the
	// runtime HAS_CACHE_MIX gate keeps the checkbox absent.
	if strings.Contains(html, `"mix":`) || strings.Contains(html, `"adt":`) {
		t.Errorf("sample-less dataset must not embed overlay data")
	}
}

// TestCacheMixLookupHelpersJS executes the DOM-free tooltip-lookup helpers
// (mixAt / adtAt, between the __CACHEMIX_PURE_HELPERS__ markers in the
// report template) under node, covering interval containment and the
// before-first / after-last edges. Skipped when node isn't installed.
func TestCacheMixLookupHelpersJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS helper test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("helpers", base)
	writeMixedJSONL(t, dir, "a", rec, smp)
	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	// html/template strips JS comments from the emitted script, so marker
	// comments don't survive — slice on the function boundaries instead:
	// fmtTokens/mixAt/adtAt are contiguous, followed by cacheMixEnabled.
	html := string(b)
	start := strings.Index(html, "function fmtTokens(")
	end := strings.Index(html, "function cacheMixEnabled(")
	if start < 0 || end < 0 || end <= start {
		t.Fatalf("pure helper functions not found in generated HTML (start=%d end=%d)", start, end)
	}
	helpers := html[start:end]

	script := helpers + `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
const mix = [
  {t0: 0,     t1: 60000,  c: 10, lc: 20, ec: 30},
  {t0: 60000, t1: 120000, c: 1,  lc: 2,  ec: 3},
];
const adt = [
  {t: 0,      v: 100, s: 1},
  {t: 60000,  v: 200, s: 2},
  {t: 120000, v: 300, s: 3},
];
assert(mixAt(mix, -1) === null, "before first interval => null");
assert(mixAt([], 5) === null, "empty mix => null");
assert(mixAt(null, 5) === null, "missing mix => null");
assert(mixAt(mix, 0).c === 10, "start boundary covered");
assert(mixAt(mix, 59999).c === 10, "interior covered");
assert(mixAt(mix, 60000).c === 10, "shared boundary belongs to earlier interval");
assert(mixAt(mix, 90000).c === 1, "second interval covered");
assert(mixAt(mix, 120000).c === 1, "end boundary covered");
assert(mixAt(mix, 999999).c === 1, "after last => latest at-or-before");
assert(adtAt(adt, -5) === null, "before first sample => null");
assert(adtAt(null, 5) === null, "missing adt => null");
assert(adtAt(adt, 0).v === 100, "exact first sample");
assert(adtAt(adt, 59999).v === 100, "holds until next sample");
assert(adtAt(adt, 60001).v === 200 && adtAt(adt, 60001).s === 2, "latest at-or-before");
assert(adtAt(adt, 1e12).v === 300, "after last => last");
assert(fmtTokens(1234567) === "1.2M" && fmtTokens(999) === "999", "fmtTokens");

// Absolute band scaling: the max is shared ACROSS series in a report (a
// 50k-peak series next to a 1M-peak series must NOT be per-series
// normalized).
const strong = { mix: [ {t0:0, t1:60000, c:900000, lc:80000, ec:20000} ] };  // total 1M
const weak   = { mix: [ {t0:0, t1:60000, c:10000,  lc:30000, ec:10000} ] };  // total 50k
const gm = mixTotalMax([strong, weak]);
assert(gm === 1000000, "cross-series shared max = 1M, got " + gm);
assert(mixTotalMax([weak]) === 50000, "single-series max");
assert(mixTotalMax([]) === 0 && mixTotalMax(null) === 0, "empty/missing series => 0");
// Partial-fill math: 50k total against the 1M shared max => 5% of the band.
const h = mixStackHeight(weak.mix[0], gm, 64);
assert(Math.abs(h - 3.2) < 1e-9, "50k vs 1M => 5% of 64px = 3.2, got " + h);
assert(mixStackHeight(strong.mix[0], gm, 64) === 64, "max interval fills the band");
assert(mixStackHeight({t0:0,t1:60000,c:0,lc:0,ec:0}, gm, 64) === 0, "zero total => empty");
assert(mixStackHeight(weak.mix[0], 0, 64) === 0, "zero global max => empty");

// Ingest rate uses the ACTUAL interval, not a hardcoded 60s.
assert(mixRate(weak.mix[0]) === 50000 / 60, "60s interval rate");
const wide = {t0:0, t1:120000, c:60000, lc:0, ec:0}; // missed tick: 120s interval
assert(mixRate(wide) === 500, "120s interval => total/120, got " + mixRate(wide));
assert(mixRate({t0:5, t1:5, c:9, lc:0, ec:0}) === 0, "zero-width interval => 0");

// Totals volume layer math — INGEST TOKENS (input+cached), not requests.
const ta = [0, 10, 20, 30];          // series A completion times
const ca = [500, 1500, 1600, 2000];  // A cumulative ingest (incl cached)
const tb = [5, 15];                  // series B
const cb2 = [700, 1000];             // B cumulative ingest
const A = { times: ta, cum: ca }, B = { times: tb, cum: cb2 };
assert(cumCountAt(ta, -1) === 0, "before first completion => 0");
assert(cumCountAt(ta, 10) === 2, "boundary timestamp inclusive");
assert(cumCountAt(ta, 1e9) === 4, "after last => all");
assert(cumCountAt([], 5) === 0 && cumCountAt(null, 5) === 0, "empty/missing => 0");
assert(cumTokensAt(ta, ca, -1) === 0, "tokens before first => 0");
assert(cumTokensAt(ta, ca, 10) === 1500, "cumulative tokens at boundary (cached included in cum)");
assert(cumTokensAt(ta, ca, 1e9) === 2000, "tokens after last => final total");
// Cumulative OUTPUT tokens use the same lookup against the same timestamps,
// so ingest and output totals are boundary-consistent at every point.
const oa = [10, 30, 60, 100]; // A cumulative output, aligned with ta
assert(cumTokensAt(ta, oa, -1) === 0, "output before first => 0");
assert(cumTokensAt(ta, oa, 10) === 30, "output boundary matches ingest boundary index");
assert(cumTokensAt(ta, oa, 15) === 30 && cumTokensAt(ta, ca, 15) === 1500,
  "same idx for ingest and output at interior t");
assert(cumTokensAt(ta, oa, 1e9) === 100, "output after last => final total");
// Normalization: at end-of-run the stack top is exactly 1.0 = full height.
const FINAL = 2000 + 1000;
let st = totalsStack([A, B], 1e9, FINAL);
assert(Math.abs(st[st.length - 1] - 1.0) < 1e-12, "final combined ingest => full height, got " + st);
// Stacking order stable (input = legend order): layer tops are cumulative.
st = totalsStack([A, B], 15, FINAL);
assert(Math.abs(st[0] - 1500 / FINAL) < 1e-12 && Math.abs(st[1] - 2500 / FINAL) < 1e-12,
  "stack order/token cumulation, got " + st);
// Zero-ingest series contributes nothing (zero-thickness layer).
st = totalsStack([A, { times: [], cum: [] }, B], 1e9, FINAL);
assert(st[1] === st[0], "zero-ingest series adds no thickness");
// Hide/show: caller drops hidden series and renormalizes => remaining stack
// still tops out at 1.0.
st = totalsStack([B], 1e9, 1000);
assert(Math.abs(st[0] - 1.0) < 1e-12, "renormalized visible-only stack fills fully");

// Closest-point lookup for the volume hover.
assert(closestIndex(null, 5) === -1 && closestIndex([], 5) === -1, "empty => -1");
assert(closestIndex(ta, -100) === 0, "before first => first");
assert(closestIndex(ta, 1e9) === 3, "after last => last");
assert(closestIndex(ta, 14) === 1, "nearest below wins");
assert(closestIndex(ta, 16) === 2, "nearest above wins");
assert(closestIndex(ta, 15) === 1, "tie goes to the earlier point");

// Trailing-window rates (requests/s, ingest tok/s, output tok/s).
const byT = [
  { t: 0,     in: 100, ca: 400, out: 10 },
  { t: 30000, in: 100, ca: 400, out: 20 },
  { t: 60000, in: 200, ca: 300, out: 30 },
];
let wr = windowRates(byT, 60000, 60000);
assert(wr.n === 3, "window includes boundary records, got n=" + wr.n);
assert(Math.abs(wr.rps - 3 / 60) < 1e-12, "requests/s over 60s");
assert(Math.abs(wr.inPerSec - 1500 / 60) < 1e-12, "ingest tok/s includes cached, got " + wr.inPerSec);
assert(Math.abs(wr.outPerSec - 60 / 60) < 1e-12, "output tok/s");
// Early-run clamp: window start clamps to the first record.
wr = windowRates(byT, 30000, 60000);
assert(Math.abs(wr.spanS - 30) < 1e-12 && wr.n === 2, "span clamps to run start");
assert(windowRates([], 5, 60000) === null && windowRates(null, 5, 60000) === null, "empty => null");

// Volume hover share is vs the BEST series at the same time point, not the
// stack total (cross-implementation totals are meaningless).
// At t=1e9: A=2000 (largest), B=1000.
assert(Math.abs(shareOfBest([A, B], 1e9, 2000) - 1.0) < 1e-12, "hovered==largest => 100%");
assert(Math.abs(shareOfBest([A, B], 1e9, 1000) - 0.5) < 1e-12, "smaller => ratio of best");
// Tie: both series at the same cumulative => both read 100%.
const T1 = { times: [0], cum: [500] }, T2 = { times: [0], cum: [500] };
assert(shareOfBest([T1, T2], 10, 500) === 1.0, "ties => 100% for both");
assert(shareOfBest([], 10, 5) === 0 && shareOfBest([T1], -1, 0) === 0, "no data / zero best => 0");

// Volume ceiling: with cache-mix bands on, fraction 1.0 lands EXACTLY on
// the band strip's lower edge (no overlap); with bands off, on the plot
// top. plotTop=30, plotH=600 => bottom=630; strip bottom e.g. 30+2*64=158.
assert(totalsY(1.0, 30, 600, 158) === 158, "full stack tops at band strip bottom edge");
assert(totalsY(0, 30, 600, 158) === 630, "empty stack sits on the plot bottom");
assert(totalsY(1.0, 30, 600, 30) === 30, "bands off => full stack reaches plot top");
assert(totalsY(0.5, 30, 600, 158) === 630 - 0.5 * (630 - 158), "linear in between");
// Toggling re-targets: same fraction, different ceiling => different y.
assert(totalsY(0.8, 30, 600, 158) !== totalsY(0.8, 30, 600, 30), "ceiling change moves the stack");

// Rolling-window percentile math (nearest-rank).
assert(percentile([], 0.5) === 0 && percentile(null, 0.5) === 0, "empty => 0");
assert(percentile([7], 0.5) === 7 && percentile([7], 0.95) === 7, "singleton");
assert(percentile([4, 1, 3, 2], 0.5) === 2, "p50 of 1..4 = 2 (nearest-rank)");
assert(percentile([4, 1, 3, 2], 0.95) === 4, "p95 of 1..4 = 4");
const hundred = Array.from({length: 100}, (_, i) => i + 1);
assert(percentile(hundred, 0.5) === 50 && percentile(hundred, 0.95) === 95, "p50/p95 of 1..100");

// Line-style mapping: pattern encodes percentile; the three kinds must be
// mutually distinct, and resp50 must be solid.
const d50 = JSON.stringify(percentileDash("ttft50"));
const d95 = JSON.stringify(percentileDash("ttft95"));
const dresp = JSON.stringify(percentileDash("resp50"));
assert(dresp === "[]", "response p50 is solid");
assert(d50 !== d95 && d50 !== dresp && d95 !== dresp, "three distinct patterns: " + d50 + " " + d95 + " " + dresp);

// Viewport-aware tooltip placement (vw=1000, vh=800; tip 200x150).
let p = placeTooltip(100, 100, 200, 150, 1000, 800);
assert(p.x === 112 && p.y === 90, "fits => right of cursor (+12,-10), got " + JSON.stringify(p));
// Right edge: flips to the left of the cursor.
p = placeTooltip(950, 100, 200, 150, 1000, 800);
assert(p.x === 950 - 12 - 200 && p.y === 90, "right edge => flip left, got " + JSON.stringify(p));
// Bottom edge (the reported bug): flips above the cursor.
p = placeTooltip(100, 780, 200, 150, 1000, 800);
assert(p.y === 780 - 12 - 150 && p.x === 112, "bottom edge => flip above, got " + JSON.stringify(p));
// Corner: flips on both axes.
p = placeTooltip(980, 790, 200, 150, 1000, 800);
assert(p.x === 980 - 12 - 200 && p.y === 790 - 12 - 150, "corner => flip both, got " + JSON.stringify(p));
// Clamps: cursor at origin with a flip that would go negative stays >= pad,
// and an oversized tip is pinned inside the viewport.
p = placeTooltip(2, 2, 200, 900, 1000, 800);
assert(p.x === 14 && p.y === 4, "oversized tip clamps to pad, got " + JSON.stringify(p));
p = placeTooltip(500, 500, 2000, 150, 1000, 800);
assert(p.x === 4, "tip wider than viewport pins to left pad, got " + JSON.stringify(p));
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "helpers_test.js")
	if err := os.WriteFile(jsPath, []byte(script), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node helper test failed: %v\n%s", err, out)
	}
}

// reportDOMStub is a minimal DOM/window stub that lets the emitted report
// script run to completion under node and lets tests capture and invoke the
// canvas event listeners. Canvas 2D calls are absorbed by a Proxy.
const reportDOMStub = `
const __listeners = {};
function __makeCtx() {
  return new Proxy({}, {
    get(t, prop) {
      if (prop === "measureText") return () => ({ width: 10 });
      if (typeof prop === "string") return () => undefined;
      return undefined;
    },
    set() { return true; }
  });
}
const __elements = {};
function __el(id, extra) {
  if (!__elements[id]) {
    __elements[id] = Object.assign({
      id, style: {}, innerHTML: "", textContent: "", value: "", checked: false,
      disabled: false, className: "", dataset: {},
      appendChild() {}, addEventListener(type, fn) { (__listeners[id + ":" + type] = __listeners[id + ":" + type] || []).push(fn); },
      getBoundingClientRect() { return { left: 0, top: 0, width: this._w || 0, height: this._h || 0 }; },
      classList: { toggle() {}, add() {}, remove() {} },
      getContext: __makeCtx,
    }, extra || {});
  }
  return __elements[id];
}
__el("chart"); __el("tooltip", { _w: 220, _h: 180 });
__el("showTTFT", { checked: true }); __el("showTTFTP95", { checked: false });
__el("showResp", { checked: true });
__el("showDots", { checked: false }); __el("showErrors", { checked: true });
__el("showCacheMix", { checked: true });
__el("showTotals", { checked: true }); __el("showXAxisValues", { checked: false });
__el("resetZoom"); __el("zoomInfo"); __el("info"); __el("legend");
__el("selectAll"); __el("deselectAll"); __el("seriesFilter");
const document = {
  getElementById: id => __el(id),
  createElement: () => __el("dyn" + Math.random()),
  querySelector: () => __el("controls"),
};
const window = {
  innerWidth: 1600, innerHeight: 900, devicePixelRatio: 1,
  addEventListener() {},
};
`

// TestTooltipShowsOnHoverJS is an end-to-end hover regression test: it runs
// the FULL emitted report script under node with a DOM stub, simulates a
// mousemove over the response moving-average line and over a cache-mix band,
// and asserts the tooltip element ends up displayed, with content, at
// in-viewport coordinates. Guards against a refactor of the tooltip show
// path silently killing hover (positioning-only unit tests can't catch
// that). Skipped when node isn't installed.
func TestTooltipShowsOnHoverJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS hover test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("hover", base)
	writeMixedJSONL(t, dir, "a", rec, smp)
	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	start := strings.Index(html, "<script>")
	end := strings.Index(html, "</script>")
	if start < 0 || end < 0 {
		t.Fatal("script block not found")
	}
	script := html[start+len("<script>") : end]

	probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
function hoverAt(px, py, label) {
  const tip = document.getElementById("tooltip");
  tip.style.display = "none"; tip.innerHTML = "";
  const mm = (__listeners["chart:mousemove"] || [])[0];
  assert(mm, "mousemove listener registered");
  mm({ clientX: px, clientY: py });
  assert(tip.style.display === "block", label + ": tooltip displayed (got " + JSON.stringify(tip.style.display) + ")");
  assert((tip.innerHTML || "").length > 0, label + ": tooltip has content");
  const x = parseFloat(tip.style.left), y = parseFloat(tip.style.top);
  assert(isFinite(x) && isFinite(y), label + ": numeric coords (" + tip.style.left + "," + tip.style.top + ")");
  assert(x >= 0 && x <= 1600 && y >= 0 && y <= 900, label + ": in viewport (" + x + "," + y + ")");
}
const s0 = DATA[0];
assert(s0._respP50.length > 1, "fixture has plotted percentile points");
const p = s0._respP50[Math.floor(s0._respP50.length / 2)];
hoverAt(mapX(p.t), mapY(p.v), "line-hover");
hoverAt(margin.left + plotW / 2, margin.top + 10, "band-hover");
// Priority chain: line hover wins over the volume layer even where the
// volume is underneath...
{
  const tip = document.getElementById("tooltip");
  assert(!(tip.innerHTML || "").includes("ingest volume"), "band hover must not be a volume tooltip");
  hoverAt(mapX(p.t), mapY(p.v), "line-over-volume");
  assert(!(tip.innerHTML || "").includes("ingest volume"), "line hover wins over volume");
  // ...and pointing at empty plot area over the stack yields the volume
  // tooltip with cumulative tokens + window rates.
  hoverAt(mapX(p.t) + 40, margin.top + plotH - 5, "volume-hover");
  assert((tip.innerHTML || "").includes("ingest volume"), "volume hover fires in the stack area: " + (tip.innerHTML || "").slice(0, 80));
  assert((tip.innerHTML || "").includes("% of best"), "volume tooltip carries share-of-best");
  assert((tip.innerHTML || "").includes("output total"), "volume tooltip carries cumulative output tokens");
  assert((tip.innerHTML || "").includes("out ") && (tip.innerHTML || "").includes("tok/s"), "volume tooltip keeps the out/s window rate");
}
// Toggle independence: every combination of the two TTFT checkboxes must
// render without throwing (p95 stays available with p50 off and vice versa).
const t50 = document.getElementById("showTTFT"), t95 = document.getElementById("showTTFTP95");
[[true, true], [true, false], [false, true], [false, false]].forEach(([a, b]) => {
  t50.checked = a; t95.checked = b;
  draw();
});
t50.checked = true; t95.checked = true;
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "hover_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node hover test failed: %v\n%s", err, out)
	}
}

// TestMergedLabelsWinDisplayNames: two source dirs whose records share the
// SAME embedded alias, merged with distinct explicit --labels, must show
// distinct display names. seriesData.Name is the single source for the
// legend, the cache-mix band label, the tooltip headers, and the ds: axis
// rows, so asserting the embedded names covers "everywhere".
func TestMergedLabelsWinDisplayNames(t *testing.T) {
	root := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	dirA := filepath.Join(root, "armA")
	dirB := filepath.Join(root, "armB")
	// benchFixtureData embeds alias=<arg> in every record's model spec; use
	// the same alias for both dirs to reproduce the same-alias collision.
	recA, smpA := benchFixtureData("SHARED_alias", base)
	recB, smpB := benchFixtureData("SHARED_alias", base)
	writeMixedJSONL(t, dirA, "reqs", recA, smpA)
	writeMixedJSONL(t, dirB, "reqs", recB, smpB)

	outDir := filepath.Join(root, "merged")
	htmlPath, err := GenerateVisualizationMerged([]string{dirA, dirB}, []string{"label-arm-a", "label-arm-b"}, outDir, 4, 0)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	for _, want := range []string{`"name":"label-arm-a"`, `"name":"label-arm-b"`} {
		if !strings.Contains(html, want) {
			t.Errorf("merged HTML missing display name %s", want)
		}
	}
	if strings.Contains(html, `"name":"SHARED_alias"`) {
		t.Errorf("record alias overrode explicit --labels as a display name")
	}

	// Without labels the alias still wins (fallback chain unchanged): both
	// dirs derive the same alias and collide into SHARED_alias/_2 filenames,
	// but display names come from the records' alias.
	outDir2 := filepath.Join(root, "merged-nolabels")
	htmlPath2, err := GenerateVisualizationMerged([]string{dirA, dirB}, nil, outDir2, 4, 0)
	if err != nil {
		t.Fatalf("merge without labels: %v", err)
	}
	b2, err := os.ReadFile(htmlPath2)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b2), `"name":"SHARED_alias"`) {
		t.Errorf("alias fallback broken when no labels are given")
	}
}

// TestMaxElapsedTruncation covers the --max-elapsed contract: per-arm t0,
// inclusive boundary, sample rows truncated alongside requests, and runs
// shorter than the cutoff untouched.
func TestMaxElapsedTruncation(t *testing.T) {
	t.Run("boundary keep/drop + samples", func(t *testing.T) {
		t0 := time.Date(2026, 7, 22, 0, 22, 53, 0, time.UTC)
		mkRec := func(offset time.Duration) requestDataRecord {
			return requestDataRecord{StartTime: t0.Add(offset), Model: "m", RequestNum: 1, InputTokens: 10}
		}
		mkSmp := func(offset time.Duration) vllmMetricsSample {
			return vllmMetricsSample{RecordType: recordTypeVLLMMetricsSample, TS: t0.Add(offset), Model: "m"}
		}
		records := []requestDataRecord{mkRec(0), mkRec(7*time.Hour + 45*time.Minute), mkRec(7*time.Hour + 45*time.Minute + time.Second), mkRec(7*time.Hour + 51*time.Minute)}
		samples := []vllmMetricsSample{mkSmp(time.Minute), mkSmp(7*time.Hour + 45*time.Minute), mkSmp(7*time.Hour + 46*time.Minute)}
		gotR, gotS := truncateToElapsed(records, samples, 7*time.Hour+45*time.Minute)
		if len(gotR) != 2 {
			t.Fatalf("records kept = %d, want 2 (t0 + exactly-at-cutoff; past-cutoff dropped)", len(gotR))
		}
		if !gotR[1].StartTime.Equal(t0.Add(7*time.Hour + 45*time.Minute)) {
			t.Errorf("boundary record (exactly at cutoff) must be KEPT")
		}
		if len(gotS) != 2 {
			t.Fatalf("samples kept = %d, want 2 (sample rows truncate against the same t0)", len(gotS))
		}
	})

	t.Run("short run unaffected; zero cutoff is a no-op", func(t *testing.T) {
		t0 := time.Now()
		records := []requestDataRecord{{StartTime: t0}, {StartTime: t0.Add(time.Hour)}}
		if r, _ := truncateToElapsed(records, nil, 8*time.Hour); len(r) != 2 {
			t.Fatal("run shorter than cutoff must be unaffected")
		}
		if r, _ := truncateToElapsed(records, nil, 0); len(r) != 2 {
			t.Fatal("maxElapsed 0 must be a no-op")
		}
	})

	t.Run("merged: per-alias independent t0", func(t *testing.T) {
		root := t.TempDir()
		dirA := filepath.Join(root, "armA")
		dirB := filepath.Join(root, "armB")
		// Arm A starts at 10:00, arm B at 11:00 — each truncates to its OWN
		// 30m elapsed window, not a shared wall-clock cutoff.
		mk := func(dir, alias string, start time.Time) {
			model := "dynamic/http://h:8000/v1,type=openai_vllm,alias=" + alias
			var recs []requestDataRecord
			for _, off := range []time.Duration{0, 20 * time.Minute, 30 * time.Minute, 40 * time.Minute} {
				recs = append(recs, requestDataRecord{StartTime: start.Add(off), Model: model, RequestNum: 1, InputTokens: 5})
			}
			smps := []vllmMetricsSample{
				{RecordType: recordTypeVLLMMetricsSample, TS: start.Add(10 * time.Minute), Model: model},
				{RecordType: recordTypeVLLMMetricsSample, TS: start.Add(35 * time.Minute), Model: model},
			}
			writeMixedJSONL(t, dir, "reqs", recs, smps)
		}
		mk(dirA, "trunc_a", time.Date(2026, 7, 22, 10, 0, 0, 0, time.UTC))
		mk(dirB, "trunc_b", time.Date(2026, 7, 22, 11, 0, 0, 0, time.UTC))

		outDir := filepath.Join(root, "merged")
		if _, err := GenerateVisualizationMerged([]string{dirA, dirB}, []string{"arm-a", "arm-b"}, outDir, 0, 30*time.Minute); err != nil {
			t.Fatalf("merge: %v", err)
		}
		for _, name := range []string{"arm-a", "arm-b"} {
			records, samples, err := readJSONLFile(filepath.Join(outDir, name+".jsonl"))
			if err != nil {
				t.Fatal(err)
			}
			// 0/20/30 kept (30m boundary inclusive), 40m dropped; sample at
			// 10m kept, 35m dropped — for EACH arm against its own t0.
			if len(records) != 3 || len(samples) != 1 {
				t.Errorf("%s: kept %d records / %d samples, want 3/1 (per-alias t0)", name, len(records), len(samples))
			}
		}
	})
}

func TestGenerateVisualizationMergedCarriesSamples(t *testing.T) {
	root := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	dirA := filepath.Join(root, "armA")
	dirB := filepath.Join(root, "armB")
	recA, smpA := benchFixtureData("merged_weka", base)
	recB, smpB := benchFixtureData("merged_hbm", base)
	writeMixedJSONL(t, dirA, "reqs", recA, smpA)
	writeMixedJSONL(t, dirB, "reqs", recB, smpB)

	outDir := filepath.Join(root, "merged")
	htmlPath, err := GenerateVisualizationMerged([]string{dirA, dirB}, nil, outDir, 4, 0)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}

	// Merged per-source JSONL must carry the sample rows through.
	mergedFiles, err := filepath.Glob(filepath.Join(outDir, "*.jsonl"))
	if err != nil || len(mergedFiles) != 2 {
		t.Fatalf("merged jsonl files: %v (%v)", mergedFiles, err)
	}
	for _, f := range mergedFiles {
		records, samples, err := readJSONLFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(records) != 5 || len(samples) != 3 {
			t.Errorf("%s: got %d records / %d samples, want 5/3", f, len(records), len(samples))
		}
	}

	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"mix":`) {
		t.Errorf("merged report missing cache-mix overlay data")
	}
}
