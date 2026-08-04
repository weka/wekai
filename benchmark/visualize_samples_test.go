package benchmark

import (
	"encoding/json"
	"fmt"
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
		"ctxModal", "ctxInBand", "applyCtxFilter", "Context Filter", "max ctx", // context-size filter modal
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
assert(fmtMs(0) === "-" && fmtMs(-1) === "-", "no latency reads as '-', never 0ms");
assert(fmtMs(150) === "150ms" && fmtMs(999) === "999ms", "sub-second stays in ms");
assert(fmtMs(1000) === "1.00s" && fmtMs(2940) === "2.94s", "second and over switches to s");
// percentileSorted matches percentile()'s nearest-rank definition, without
// copying or re-sorting -- the summary sorts once and reads both p50 and p95.
{
  const raw = [5, 1, 4, 2, 3];
  const srt = Float64Array.from(raw).sort();
  [0.5, 0.95, 0.0, 1.0].forEach(p => {
    assert(percentileSorted(srt, p) === percentile(raw, p),
      "percentileSorted agrees with percentile at p=" + p);
  });
  assert(percentileSorted([], 0.5) === 0 && percentileSorted(null, 0.5) === 0, "empty => 0");
  assert(percentileSorted(Float64Array.from([7]), 0.95) === 7, "single sample");
}

// adtWindow / adtWindowMax: the active-dataset band is re-framed by zoom, so
// the scale and the "last dataset size" label must come from the window, not
// the whole run.
assert(adtWindow(null, 0, 1e9).length === 0 && adtWindow([], 0, 1e9).length === 0, "empty adt => no points");
{
  const all = adtWindow(adt, 0, 1e9);
  assert(all.length === 3 && adtWindowMax(all) === 300, "full range keeps every sample, max 300");
}
{
  // Window strictly inside the run: the sample at-or-before tMin carries in,
  // so the line starts at the level actually in force at tMin.
  const w = adtWindow(adt, 70000, 130000);
  assert(w.length === 2, "carry-in + in-window sample, got " + w.length);
  assert(w[0].t === 60000 && w[0].v === 200, "carry-in is the last sample at/before tMin");
  assert(w[1].v === 300, "in-window sample follows");
  assert(adtWindowMax(w) === 300, "window max is 300, not the run max");
}
{
  // A window entirely between two samples still describes a level.
  const w = adtWindow(adt, 70000, 90000);
  assert(w.length === 1 && w[0].v === 200, "gap window carries the standing level");
  assert(adtWindowMax(w) === 200, "gap window scale is the carried level");
}
{
  // A window that ends before the peak must NOT be scaled by the peak --
  // this is the reported bug: zooming early flattened the line against 300.
  const w = adtWindow(adt, 0, 60000);
  assert(w.length === 2 && adtWindowMax(w) === 200, "early window scale is 200, got " + adtWindowMax(w));
  assert(w[w.length - 1].v === 200, "last-in-window is 200, not the run's final 300");
}
assert(adtWindow(adt, -1e9, -1000).length === 0, "window entirely before the run => no points");

// adtWindowRange: BOTH ends come from the window. A 0-anchored axis renders a
// narrow window (a level that moved 200 -> 300) as a flat line glued to the
// band top; anchoring at the window minimum spends the band on the variation.
assert(adtWindowRange([]) === null && adtWindowRange([[]]) === null, "no points => no range");
{
  const r = adtWindowRange([adtWindow(adt, 0, 1e9)]);
  assert(r.lo === 100 && r.hi === 300, "full range is 100-300, got " + JSON.stringify(r));
}
{
  const r = adtWindowRange([adtWindow(adt, 70000, 130000)]);
  assert(r.lo === 200 && r.hi === 300, "zoomed range floor lifts to 200, got " + JSON.stringify(r));
}
{
  // Shared across bands so arms stay comparable to each other.
  const other = [{t: 0, v: 50, s: 1}, {t: 60000, v: 900, s: 2}];
  const r = adtWindowRange([adtWindow(adt, 0, 1e9), adtWindow(other, 0, 1e9)]);
  assert(r.lo === 50 && r.hi === 900, "range spans every band, got " + JSON.stringify(r));
}

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

// Context-band membership (input+cached, inclusive bounds, 0 = unbounded).
const rec500 = { in: 100, ca: 400 };
assert(ctxInBand(rec500, 0, 0), "unbounded band keeps everything");
assert(ctxInBand(rec500, 500, 0) && ctxInBand(rec500, 0, 500), "band boundaries are inclusive");
assert(!ctxInBand(rec500, 501, 0), "below min drops");
assert(!ctxInBand(rec500, 0, 499), "above max drops");
assert(ctxInBand({ in: 500 }, 500, 500), "rows without cached tokens still counted");
assert(!ctxInBand({}, 1, 0) && ctxInBand({}, 0, 0), "empty record = ctx 0");

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
  createTextNode: text => ({ nodeType: 3, textContent: String(text) }),
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
// Context-band filter end-to-end: fixture rows all have ctx=500 (in=100 +
// ca=400). Inclusive boundary keeps them at max=500; min=501 empties every
// view and the chart still renders; reset restores the full view.
applyCtxFilter(0, 500);
assert(DATA[0]._view.length === DATA[0].records.length, "ctx<=500 keeps all rows (inclusive boundary)");
draw();
applyCtxFilter(501, 0);
assert(DATA[0]._view.length === 0, "min 501 filters out all ctx=500 rows");
draw();
applyCtxFilter(0, 0);
assert(DATA[0]._view === DATA[0].records, "reset restores the unfiltered view");
draw();
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

// TestSummaryPanelJS runs the emitted report under the DOM stub and asserts
// every cell of the per-variant summary panel against hand-computed values.
// The panel is the header's headline claim about a run, so its arithmetic
// gets pinned exactly rather than smoke-tested:
//
//	4 requests at t = 0/10/20/30s, the last one an error.
//	prompt tokens (in + ca, net-of-cache contract) = 3*500 + 1000 = 2500
//	output tokens = 3*50 + 100 = 250
//	completed (non-error) = 3
//	rate denominator = first->last record extent = 30s
//	  => in/s = 2500/30 = 83.3 -> "83"; out/s = 250/30 = 8.3 -> "8"
//	errors per 1k = 1/4 * 1000 = "250.0"
//
// Volumes deliberately include the errored request (a failed request still
// cost its prompt), which is why its 1000/100 tokens are in the totals.
func TestSummaryPanelJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS summary test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=sumfix"
	var records []requestDataRecord
	for i := 0; i < 4; i++ {
		st := base.Add(time.Duration(i) * 10 * time.Second)
		r := requestDataRecord{
			StartTime: st, EndTime: st.Add(2 * time.Second),
			TTFT: 150, ResponseMs: 2000, Model: model,
			SeriesNum: 1, RequestNum: i + 1,
			InputTokens: 100, CachedTokens: 400, OutputTokens: 50,
		}
		if i == 3 {
			r.IsError = true
			r.InputTokens, r.CachedTokens, r.OutputTokens = 200, 800, 100
		}
		records = append(records, r)
	}
	writeMixedJSONL(t, dir, "a", records, nil)
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
// sumCells is [variantIndex][metricIndex] -- variants are rows.
function expect(si, mi, want, label) {
  const got = sumCells[si][mi].textContent;
  assert(got === want, label + ": want " + JSON.stringify(want) + ", got " + JSON.stringify(got));
}
assert(SUMMARY_METRICS.length === 8, "eight metric columns, got " + SUMMARY_METRICS.length);
assert(sumCells.length === DATA.length && sumCells[0].length === 8, "one row per variant, eight cells each");
assert(sumRows.length === DATA.length, "one <tr> per variant");
expect(0, 0, "2.5k", "input tokens");
expect(0, 1, "250",  "output tokens");
expect(0, 2, "3",    "completed requests");
expect(0, 3, "83",   "avg input/s");
expect(0, 4, "8",    "avg output/s");
expect(0, 5, "150ms","ttft p50");
expect(0, 6, "150ms","ttft p95");
expect(0, 7, "250.0","errors per 1k");
// TTFT percentiles are backed by the non-error requests only: the errored
// 4th row must not count even though it carries a ttft.
assert(sumCells[0][5].title.indexOf("3 non-error requests") === 0, "ttft sample count, got " + sumCells[0][5].title);
assert(document.getElementById("sumRange").textContent === "full run (30s)",
  "range label, got " + JSON.stringify(document.getElementById("sumRange").textContent));

// The cached split rides along as a hover on the input row:
// in = 3*100 + 200 = 500 uncached, ca = 3*400 + 800 = 2000 server-cached,
// 2000/2500 = 80% cached.
const tip = sumCells[0][0].title || "";
assert(tip.indexOf("500 uncached") >= 0 && tip.indexOf("2.0k server-cached") >= 0 && tip.indexOf("80.0% cached") >= 0,
  "input-row cached breakdown, got " + JSON.stringify(tip));

// Zooming reprices every cell: clip to [0, 10s] and only the first two
// requests remain (2 x 500 prompt, 2 x 50 out, 0 errors, 10s extent).
viewTMin = globalTMin; viewTMax = globalTMin + 10000;
draw();
expect(0, 0, "1.0k", "zoomed input tokens");
expect(0, 1, "100",  "zoomed output tokens");
expect(0, 2, "2",    "zoomed completed requests");
expect(0, 3, "100",  "zoomed avg input/s");
expect(0, 4, "10",   "zoomed avg output/s");
expect(0, 5, "150ms","zoomed ttft p50");
expect(0, 6, "150ms","zoomed ttft p95");
expect(0, 7, "0.0",  "zoomed errors per 1k");
assert(document.getElementById("sumRange").textContent === "0s – 10s",
  "zoomed range label, got " + JSON.stringify(document.getElementById("sumRange").textContent));

// The context filter feeds the panel too: min above the fixture's ctx keeps
// only the errored row (ctx 1000), so completed drops to 0 and the error
// rate saturates.
viewTMin = globalTMin; viewTMax = globalTMax;
applyCtxFilter(600, 0);
expect(0, 2, "0",      "ctx-filtered completed requests");
expect(0, 7, "1000.0", "ctx-filtered errors per 1k");
// No completed request in the view => the percentiles read "-", never "0ms".
expect(0, 5, "-", "ctx-filtered ttft p50 has no sample");
expect(0, 6, "-", "ctx-filtered ttft p95 has no sample");
applyCtxFilter(0, 0);
expect(0, 2, "3", "reset restores completed requests");

// Collapsing a panel hands its height to the chart and must not disturb the
// numbers: the toggle re-runs resize() + draw().
const hBefore = H;
(__listeners["controlsToggle:click"] || []).forEach(fn => fn());
assert(document.getElementById("controlsPanel").className.indexOf("collapsed") >= 0, "controls panel collapses");
assert(H >= hBefore, "collapsing controls does not shrink the canvas");
expect(0, 2, "3", "numbers survive a collapse");
(__listeners["controlsToggle:click"] || []).forEach(fn => fn());
assert(document.getElementById("controlsPanel").className.indexOf("collapsed") < 0, "controls panel expands again");

// Clicking a summary row toggles that variant, same as its legend entry.
sumRows[0].onclick();
assert(hiddenSeries.has(0), "row click hides the variant");
sumRows[0].onclick();
assert(!hiddenSeries.has(0), "row click shows it again");
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "summary_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node summary test failed: %v\n%s", err, out)
	}
}

// TestActiveDatasetReframesOnZoomJS pins the fix for a band that described the
// whole run regardless of zoom: the active-dataset line was scaled against the
// run's peak and labelled with the run's final sample, so zooming into a
// stretch where the dataset was small flattened the line onto the band floor
// and reported a size that was never in force there.
//
// benchFixtureData samples the dataset at t = 0/60/120s with 4k/8k/12k tokens.
func TestActiveDatasetReframesOnZoomJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS active-dataset test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("adtzoom", base)
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
document.getElementById("showCacheMix").checked = true;
const band = () => cacheMixLayout().bands[0];

// Whole run: scale spans the run and the last sample is the final one.
viewTMin = globalTMin; viewTMax = globalTMax;
let L = cacheMixLayout();
assert(band().adtRange.lo === 4000 && band().adtRange.hi === 12000,
  "full view scale is 4000-12000, got " + JSON.stringify(band().adtRange));
let pts = band().adtPts;
assert(pts[pts.length - 1].v === 12000, "full view last dataset size is 12000");

// Zoom to the first minute: the 12k peak is outside the view, so the scale,
// its floor, and the reported size all drop to what the window holds.
viewTMin = globalTMin; viewTMax = globalTMin + 60000;
L = cacheMixLayout();
assert(band().adtRange.lo === 4000 && band().adtRange.hi === 8000,
  "zoomed scale re-frames to 4000-8000, got " + JSON.stringify(band().adtRange));
pts = band().adtPts;
assert(pts[pts.length - 1].v === 8000, "zoomed last dataset size is 8000, got " + pts[pts.length - 1].v);
assert(pts[0].v === 4000, "zoomed window starts at the 4000 sample");

// The band's full height is spent on the window: its extremes land on the
// band edges (modulo the 2px inset), NOT squashed against a 0-anchored top.
{
  const b = band();
  const yLo = adtY(4000, b.adtRange, b.yTop, b.bandH);
  const yHi = adtY(8000, b.adtRange, b.yTop, b.bandH);
  assert(Math.abs(yLo - (b.yTop + b.bandH - 2)) < 0.01, "window min sits at the band floor, got " + yLo);
  assert(Math.abs(yHi - (b.yTop + 2)) < 0.01, "window max sits at the band top, got " + yHi);
  // Against the old 0-anchored axis both would have crowded the upper third.
  assert(yLo - yHi > b.bandH * 0.8, "the window spans nearly the whole band");
}

// A window falling between two samples still describes the standing level
// (carry-in), rather than emptying the band. A flat range draws mid-band
// instead of dividing by zero.
viewTMin = globalTMin + 70000; viewTMax = globalTMin + 90000;
L = cacheMixLayout();
pts = band().adtPts;
assert(pts.length === 1 && pts[0].v === 8000, "gap window carries the standing 8000 level");
assert(band().adtRange.lo === 8000 && band().adtRange.hi === 8000, "gap window range is flat at the carried level");
{
  const b = band();
  const y = adtY(8000, b.adtRange, b.yTop, b.bandH);
  assert(Math.abs(y - (b.yTop + b.bandH / 2)) < 0.01, "flat range draws mid-band, got " + y);
  assert(isFinite(y), "flat range must not divide by zero");
}

// Bands are scaled INDEPENDENTLY: an arm sitting at a different level must
// still spend its full band on its own variation, not a slice of the union.
viewTMin = globalTMin; viewTMax = globalTMax;
{
  const L2 = cacheMixLayout();
  L2.bands.forEach(b => {
    const vs = b.adtPts.map(p => p.v);
    assert(b.adtRange.lo === Math.min.apply(null, vs) && b.adtRange.hi === Math.max.apply(null, vs),
      b.s.name + ": range comes from its OWN points, got " + JSON.stringify(b.adtRange));
    const ys = b.adtPts.map(p => adtY(p.v, b.adtRange, b.yTop, b.bandH));
    const spread = Math.max.apply(null, ys) - Math.min.apply(null, ys);
    assert(spread > b.bandH * 0.9, b.s.name + ": line spans its full band, got " + spread + "/" + b.bandH);
  });
}

// The whole thing still renders at every zoom without throwing.
[[globalTMin, globalTMax], [globalTMin, globalTMin + 60000], [globalTMin + 70000, globalTMin + 90000]].forEach(([a, c]) => {
  viewTMin = a; viewTMax = c;
  draw();
});
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "adt_zoom_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node active-dataset test failed: %v\n%s", err, out)
	}
}

// TestNoDatasetAxisRowsJS pins the removal of the per-tick "ds:<size>" axis
// rows. They cost one row of bottom margin per visible sampled series — 140px
// of chart height on a 10-variant report — to repeat, in near-identical
// numbers across every tick, what the band label and band hover already state
// exactly. Enabling the cache-mix overlay must no longer grow the bottom
// margin at all; only "Show X-axis values" may.
func TestNoDatasetAxisRowsJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS axis-row test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	for _, alias := range []string{"armA", "armB", "armC"} {
		rec, smp := benchFixtureData(alias, base)
		writeMixedJSONL(t, dir, alias, rec, smp)
	}
	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	if strings.Contains(html, `"ds:"`) {
		t.Errorf("generated report still emits the ds: axis-row label")
	}
	start := strings.Index(html, "<script>")
	end := strings.Index(html, "</script>")
	if start < 0 || end < 0 {
		t.Fatal("script block not found")
	}
	script := html[start+len("<script>") : end]

	probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
assert(DATA.length === 3 && DATA.every(s => s.adt && s.adt.length), "3 sampled series in the fixture");
const mix = document.getElementById("showCacheMix"), xax = document.getElementById("showXAxisValues");

// The cache-mix overlay must not buy any bottom margin.
mix.checked = false; xax.checked = false;
const bare = calcBottomMargin();
mix.checked = true;
assert(calcBottomMargin() === bare, "cache-mix overlay must not grow the bottom margin (got " +
  calcBottomMargin() + " vs " + bare + ")");

// The x-axis values toggle still buys one row per visible series, and that
// remains the ONLY thing that does.
xax.checked = true;
const withRows = calcBottomMargin();
assert(withRows === bare + 3 * 14, "x-axis values add one row per series, got " + withRows);
mix.checked = false;
assert(calcBottomMargin() === withRows, "rows come from the x-axis toggle alone");

// Chart height actually reclaims the space: overlay on, axis values off.
mix.checked = true; xax.checked = false;
resize(); draw();
const tallPlotH = plotH;
xax.checked = true;
resize(); draw();
assert(plotH < tallPlotH, "axis rows still cost plot height when explicitly enabled");
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "axis_rows_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node axis-row test failed: %v\n%s", err, out)
	}
}

// TestCacheMixDefaultBySeriesCountJS pins the overlay's default state to the
// report's series count. Every sampled series claims its own band off the top
// of the plot (collectively up to 60% of plot height), so on a wide sweep the
// bands both squeeze the latency chart and get too thin to read individually.
// Four arms or fewer -- the common A/B/C/D comparison -- still open with the
// overlay on; anything wider opens on the chart, one click from the overlay.
func TestCacheMixDefaultBySeriesCountJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS cache-mix default test skipped")
	}
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)

	for _, tc := range []struct {
		name   string
		series int
		wantOn bool
	}{
		{"one series", 1, true},
		{"at the threshold", 4, true},
		{"one past the threshold", 5, false},
		{"wide sweep", 10, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for i := 0; i < tc.series; i++ {
				alias := fmt.Sprintf("arm%02d", i)
				rec, smp := benchFixtureData(alias, base)
				writeMixedJSONL(t, dir, alias, rec, smp)
			}
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

			// The stub pre-creates #showCacheMix as checked, so a report that
			// never touched the property would read as "on" -- clear it first
			// so the assertion can only pass if the report set it itself.
			probe := fmt.Sprintf(`
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
assert(DATA.length === %d, "fixture has %d series, got " + DATA.length);
assert(HAS_CACHE_MIX, "fixture carries metrics samples");
const want = %t;
assert(document.getElementById("showCacheMix").checked === want,
  "%d series => overlay default " + want + ", got " + document.getElementById("showCacheMix").checked);
assert(cacheMixEnabled() === want, "cacheMixEnabled agrees with the checkbox");
// Whatever the default, the toggle still works in both directions.
document.getElementById("showCacheMix").checked = !want;
assert(cacheMixEnabled() === !want, "toggle flips the overlay");
draw();
document.getElementById("showCacheMix").checked = want;
draw();
console.log("ALL_OK");
`, tc.series, tc.series, tc.wantOn, tc.series)

			stub := strings.Replace(reportDOMStub,
				`__el("showCacheMix", { checked: true });`,
				`__el("showCacheMix", { checked: false });`, 1)
			if stub == reportDOMStub {
				t.Fatal("could not neutralise the stub's pre-checked showCacheMix")
			}
			jsPath := filepath.Join(dir, "cachemix_default_test.js")
			if err := os.WriteFile(jsPath, []byte(stub+"\n"+script+"\n"+probe), 0o644); err != nil {
				t.Fatal(err)
			}
			out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
			if err != nil || !strings.Contains(string(out), "ALL_OK") {
				t.Fatalf("node cache-mix default test failed: %v\n%s", err, out)
			}
		})
	}
}

// TestDragZoomPrecisionJS pins the drag-to-zoom selection to the pixels
// actually dragged. unmapX is relative to the CURRENT view, so assigning
// viewTMin before resolving the far edge made the second unmapX resolve
// against the half-updated view: the start was right, the end landed far too
// late, and the dragged region ended up at the front of a much wider window.
// A narrow drag at position f produced a span of ~f*(1-f) of the run instead
// of the width dragged -- e.g. a 2.5% drag mid-run opened a ~25% window, so
// the selection filled only the first tenth of it.
//
// The test drags several windows, including two successive zooms, and asserts
// the resulting view matches the pixel range to within a pixel's worth of time.
func TestDragZoomPrecisionJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS drag-zoom test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("dragzoom", base)
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
const down = (__listeners["chart:mousedown"] || [])[0];
const move = (__listeners["chart:mousemove"] || [])[0];
const up   = (__listeners["chart:mouseup"] || [])[0];
assert(down && move && up, "drag listeners registered");

// Drag between two fractions of the plot width and assert the resulting view
// is exactly the time range those pixels denoted in the PRE-drag view.
function dragFrac(f1, f2, label) {
  const t0 = viewTMin, t1 = viewTMax, span = t1 - t0;
  const px = f => margin.left + f * plotW;
  const wantMin = t0 + f1 * span;
  const wantMax = t0 + f2 * span;
  down({ clientX: px(f1), clientY: margin.top + 10 });
  move({ clientX: px((f1 + f2) / 2), clientY: margin.top + 10 });
  move({ clientX: px(f2), clientY: margin.top + 10 });
  up({ clientX: px(f2), clientY: margin.top + 10 });
  // One pixel's worth of the pre-drag span is the achievable precision.
  const tol = span / plotW;
  assert(Math.abs(viewTMin - wantMin) <= tol,
    label + ": start off by " + (viewTMin - wantMin) + "ms (tol " + tol + ")");
  assert(Math.abs(viewTMax - wantMax) <= tol,
    label + ": end off by " + (viewTMax - wantMax) + "ms (tol " + tol + ")");
  // The dragged region must FILL the new view, not sit at the front of it.
  const got = viewTMax - viewTMin, want = wantMax - wantMin;
  assert(Math.abs(got - want) <= 2 * tol,
    label + ": span " + got + "ms, want " + want + "ms (" + (100 * want / got).toFixed(1) + "% of view)");
}
function reset() { viewTMin = globalTMin; viewTMax = globalTMax; recalcYMax(); draw(); }

reset(); dragFrac(0.40, 0.60, "mid-run half-width");
// The reported case: a narrow drag mid-run. Previously opened a ~25% window
// with the selection filling only its first tenth.
reset(); dragFrac(0.475, 0.500, "narrow drag mid-run");
reset(); dragFrac(0.05, 0.10, "narrow drag near the start");
reset(); dragFrac(0.90, 0.95, "narrow drag near the end");
reset(); dragFrac(0.0, 1.0, "full-width drag is a no-op zoom");
// Dragging right-to-left selects the same window.
reset();
{
  const t0 = viewTMin, span = viewTMax - viewTMin, px = f => margin.left + f * plotW;
  down({ clientX: px(0.7), clientY: margin.top + 10 });
  move({ clientX: px(0.3), clientY: margin.top + 10 });
  up({ clientX: px(0.3), clientY: margin.top + 10 });
  const tol = span / plotW;
  assert(Math.abs(viewTMin - (t0 + 0.3 * span)) <= tol, "backwards drag start");
  assert(Math.abs(viewTMax - (t0 + 0.7 * span)) <= tol, "backwards drag end");
}
// Zooming twice compounds correctly -- the second drag is relative to the
// first result, which is where the half-updated view did the most damage.
reset();
dragFrac(0.20, 0.80, "first zoom");
dragFrac(0.25, 0.50, "second zoom inside the first");
// A click (under the 5px threshold) must not zoom at all.
reset();
{
  const before = [viewTMin, viewTMax];
  down({ clientX: margin.left + 100, clientY: margin.top + 10 });
  up({ clientX: margin.left + 102, clientY: margin.top + 10 });
  assert(viewTMin === before[0] && viewTMax === before[1], "a 2px click must not zoom");
}
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "drag_zoom_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node drag-zoom test failed: %v\n%s", err, out)
	}
}

// TestRollingWindowPrecedenceJS pins where the rolling-percentile window size
// comes from, most trustworthy first: the concurrency each arm RECORDED in its
// run_params header, then the report-wide --concurrency the caller passed, then
// a guess from the observed series count.
//
// The per-arm step is the point of recording params at all. A merged report
// whose arms ran at different concurrency previously got one global number,
// which smooths one arm correctly and the other wrongly with nothing on screen
// saying so -- and that number had to be typed on the command line, where
// forgetting it silently fell through to the guess.
func TestRollingWindowPrecedenceJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS window-precedence test skipped")
	}
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)

	// armA records conc 28 + hot 4 => 32; armB records nothing (legacy).
	writeArm := func(t *testing.T, dir, name string, params *AutoBenchmarkConfig) {
		t.Helper()
		rec, smp := benchFixtureData(name, base)
		if params == nil {
			writeMixedJSONL(t, dir, name, rec, smp)
			return
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		f, err := os.Create(filepath.Join(dir, name+".jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		enc := json.NewEncoder(f)
		if err := enc.Encode(buildRunParams(*params, base)); err != nil {
			t.Fatal(err)
		}
		for _, r := range rec {
			if err := enc.Encode(r); err != nil {
				t.Fatal(err)
			}
		}
		for _, s := range smp {
			if err := enc.Encode(s); err != nil {
				t.Fatal(err)
			}
		}
		f.Close()
	}

	run := func(t *testing.T, dir string, flagConc int, probe string) {
		t.Helper()
		htmlPath, err := GenerateVisualization(dir, flagConc)
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
		jsPath := filepath.Join(dir, "window_test.js")
		full := reportDOMStub + "\n" + script + "\n" +
			"function assert(c,m){if(!c){console.error(\"FAIL: \"+m);process.exit(1);}}\n" +
			probe + "\nconsole.log(\"ALL_OK\");\n"
		if err := os.WriteFile(jsPath, []byte(full), 0o644); err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
		if err != nil || !strings.Contains(string(out), "ALL_OK") {
			t.Fatalf("node window-precedence test failed: %v\n%s", err, out)
		}
	}

	t.Run("recorded params win over the flag", func(t *testing.T) {
		dir := t.TempDir()
		writeArm(t, dir, "recorded", &AutoBenchmarkConfig{Concurrency: 28, HotSeriesConcurrency: 4, StartSeries: 512, MaxSeries: 512})
		// A deliberately WRONG --concurrency: the data knows better.
		run(t, dir, 7, `
const s = DATA[0];
assert(s.conc === 32, "recorded conc 28+hot 4 = 32, got " + s.conc);
assert(s._winConc === 32, "window uses the recorded value, got " + s._winConc);
assert(s._winSize === 96, "window = 32*3 = 96, got " + s._winSize);
assert(s._winConcSource === "recorded", "source = recorded, got " + s._winConcSource);
assert(document.getElementById("runparams").textContent.indexOf("conc=28") >= 0,
  "header shows the recorded params, got " + document.getElementById("runparams").textContent);
`)
	})

	t.Run("flag used when nothing recorded", func(t *testing.T) {
		dir := t.TempDir()
		writeArm(t, dir, "legacy", nil)
		run(t, dir, 7, `
const s = DATA[0];
assert(!s.conc, "legacy arm carries no recorded conc, got " + s.conc);
assert(s._winSize === 21, "window falls back to --concurrency 7*3 = 21, got " + s._winSize);
assert(s._winConcSource === "--concurrency", "source = --concurrency, got " + s._winConcSource);
// Nothing recorded => the line is hidden entirely rather than spending a
// row of vertical space to say so; the per-row hover still carries provenance.
assert(document.getElementById("runparams").style.display === "none",
  "params line hidden when nothing was recorded, got " + document.getElementById("runparams").style.display);
assert(!document.getElementById("runparams").textContent, "and carries no text");
`)
	})

	t.Run("default survives huge series numbers", func(t *testing.T) {
		// The case that motivated dropping the inference: a replay-shaped run
		// whose series numbers climb far past the real in-flight concurrency.
		dir := t.TempDir()
		base2 := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
		var rec []requestDataRecord
		for i := 0; i < 40; i++ {
			st := base2.Add(time.Duration(i) * time.Second)
			rec = append(rec, requestDataRecord{
				StartTime: st, EndTime: st.Add(time.Second), TTFT: 150, ResponseMs: 900,
				Model:     "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=bigsn",
				SeriesNum: 1000 + i*40, RequestNum: i + 1,
				InputTokens: 100, CachedTokens: 400, OutputTokens: 50,
			})
		}
		writeMixedJSONL(t, dir, "bigsn", rec, nil)
		run(t, dir, 0, `
const s = DATA[0];
const maxSn = s.records.reduce((m, r) => Math.max(m, r.sn), 0);
assert(maxSn >= 2500, "fixture really does have huge series numbers, got " + maxSn);
assert(s._winSize === 96, "window stays at the default despite sn=" + maxSn + ", got " + s._winSize);
`)
	})

	t.Run("fixed default when neither", func(t *testing.T) {
		dir := t.TempDir()
		writeArm(t, dir, "legacy", nil)
		run(t, dir, 0, `
const s = DATA[0];
assert(s._winConcSource === "default", "source = default, got " + s._winConcSource);
assert(DEFAULT_WINDOW_REQS === 96, "default window is 96 reqs, got " + DEFAULT_WINDOW_REQS);
assert(s._winSize === 96, "window = the fixed default, got " + s._winSize);
// The old series-count inference is gone: it scaled with series NUMBER, which
// on a router replay climbs into the thousands as sessions recycle, yielding
// windows of 3396/5358 requests on real 8h data.
assert(s._winSize < 200, "window must not scale with series count, got " + s._winSize);
`)
	})

	t.Run("per-arm windows in one report", func(t *testing.T) {
		dir := t.TempDir()
		writeArm(t, dir, "a_fast", &AutoBenchmarkConfig{Concurrency: 60, StartSeries: 512, MaxSeries: 512})
		writeArm(t, dir, "b_slow", &AutoBenchmarkConfig{Concurrency: 28, HotSeriesConcurrency: 4, StartSeries: 512, MaxSeries: 512})
		run(t, dir, 0, `
const by = {};
DATA.forEach(s => { by[s.name] = s; });
const names = Object.keys(by);
assert(names.length === 2, "two arms, got " + names);
const fast = DATA.find(s => s.conc === 60), slow = DATA.find(s => s.conc === 32);
assert(fast && slow, "each arm keeps its OWN recorded concurrency: " + DATA.map(s => s.conc).join(","));
assert(fast._winSize === 180 && slow._winSize === 96,
  "windows sized per arm, got " + fast._winSize + " / " + slow._winSize);
assert(document.getElementById("runparams").textContent.indexOf("differ per variant") >= 0,
  "header flags that the arms are not the same shape, got " + document.getElementById("runparams").textContent);
`)
	})
}

// TestBaselineRatiosJS pins the ratio-to-hbm column behaviour. These reports
// almost always compare an offload arm against a no-offload "hbm" control, and
// the question asked of them -- how much better or worse than hbm? -- was
// previously answered by dividing two numbers by hand.
//
// Baseline detection reuses classifyAlias, the same rule that sorts hbm arms
// first, so the naming convention lives in one place.
func TestBaselineRatiosJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS baseline-ratio test skipped")
	}
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)

	// Two arms whose first->last extent is the SAME 36s -- the rate columns
	// divide by each arm's own extent, so unequal extents would make the rate
	// ratio disagree with the volume ratio for reasons unrelated to this
	// feature. hbm does 4 requests of 500 prompt tokens, weka 12 of the same:
	// exactly 300%, the figure in the report this was asked for. TTFT is
	// deliberately inverted (weka faster) so the lower-is-better direction
	// gets exercised too.
	arm := func(alias string, n int, ttft float64) []requestDataRecord {
		model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=" + alias
		var out []requestDataRecord
		for i := 0; i < n; i++ {
			st := base.Add(time.Duration(i) * (36 * time.Second / time.Duration(n-1)))
			out = append(out, requestDataRecord{
				StartTime: st, EndTime: st.Add(time.Second),
				TTFT: ttft, ResponseMs: 900, Model: model,
				SeriesNum: 1, RequestNum: i + 1,
				InputTokens: 100, CachedTokens: 400, OutputTokens: 50,
			})
		}
		return out
	}

	dir := t.TempDir()
	writeMixedJSONL(t, dir, "hbm-c28", arm("hbm-c28", 4, 4000), nil)
	writeMixedJSONL(t, dir, "weka-c28", arm("weka-c28", 12, 1000), nil)

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
// classifyAlias sorts hbm first, so the baseline is row 0 here -- but the
// index is resolved by classification, not by position.
assert(BASELINE_INDEX === 0, "hbm arm is the baseline, got index " + BASELINE_INDEX);
assert(classifyAlias(getAlias(DATA[BASELINE_INDEX].name)) === "gpu", "baseline classifies as an hbm/gpu arm");

const other = 1 - BASELINE_INDEX;
function ratio(si, key) {
  const mi = SUMMARY_METRICS.findIndex(m => m.key === key);
  return sumRatios[si][mi].textContent;
}
function value(si, key) {
  const mi = SUMMARY_METRICS.findIndex(m => m.key === key);
  return sumCells[si][mi].textContent;
}

// The baseline row itself carries no ratios -- it IS 100% by definition.
SUMMARY_METRICS.forEach(m => {
  assert(ratio(BASELINE_INDEX, m.key) === "", "baseline row shows no ratio for " + m.key);
});

// 12 requests vs 4, same tokens each, same 40s span => 300% on every volume
// and rate metric.
["in", "out", "reqs", "inrate", "outrate"].forEach(k => {
  assert(ratio(other, k) === "300%", k + " ratio = 300%, got " + ratio(other, k) + " (value " + value(other, k) + ")");
});
// Lower-is-better metrics ratio the same way: 1000ms against 4000ms = 25%.
assert(ratio(other, "ttft50") === "25%", "ttft50 ratio = 25%, got " + ratio(other, "ttft50"));

// Tint follows the DIRECTION of improvement, not the size of the number:
// 300% more tokens is good, 300% more latency would not be.
const mIn = SUMMARY_METRICS.findIndex(m => m.key === "in");
const mT = SUMMARY_METRICS.findIndex(m => m.key === "ttft50");
assert(sumRatios[other][mIn].className.indexOf("up") >= 0, "more input tokens tints as better");
assert(sumRatios[other][mT].className.indexOf("up") >= 0, "less latency tints as better, got " + sumRatios[other][mT].className);

// The value text stays the datum alone -- the ratio lives in its own span, so
// nothing downstream has to parse "3.6B300%".
assert(value(other, "in") === "6.0k", "value cell holds only the value, got " + value(other, "in"));

// The ratio renders on its own line beneath the value, so the value column
// stays aligned under its header. The second line is reserved only when a
// baseline exists, and the (empty) baseline row reserves it too, keeping every
// row the same height.
assert(document.getElementById("summaryTable").className.indexOf("has-ratios") >= 0,
  "table opts into the two-line cell layout, got " + document.getElementById("summaryTable").className);

// Ratios track the selected window like every other figure. The two arms
// pace differently, so the zoomed ratio is NOT 300% -- compute what the window
// actually holds and require the cell to agree with it.
viewTMin = globalTMin; viewTMax = globalTMin + 20000;
draw();
const inWin = si => DATA[si].records.filter(r => r.t >= viewTMin && r.t <= viewTMax && !r.err).length;
const want = Math.round(inWin(other) / inWin(BASELINE_INDEX) * 100) + "%";
assert(want !== "300%", "the zoomed window really does change the ratio");
assert(ratio(other, "reqs") === want, "zoomed reqs ratio = " + want + ", got " + ratio(other, "reqs"));
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "baseline_ratio_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node baseline-ratio test failed: %v\n%s", err, out)
	}
}

// TestNoBaselineNoRatiosJS: without an hbm arm — or with only one variant —
// there is nothing to be a percentage OF, and the report must not invent one.
func TestNoBaselineNoRatiosJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS no-baseline test skipped")
	}
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)

	check := func(t *testing.T, dir string) {
		t.Helper()
		htmlPath, err := GenerateVisualization(dir, 4)
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			t.Fatal(err)
		}
		html := string(b)
		s, e := strings.Index(html, "<script>"), strings.Index(html, "</script>")
		probe := `
function assert(c, m) { if (!c) { console.error("FAIL: " + m); process.exit(1); } }
assert(BASELINE_INDEX === -1, "no baseline, got index " + BASELINE_INDEX);
sumRatios.forEach((row, si) => row.forEach((el, mi) =>
  assert(el.textContent === "", "no ratio anywhere (row " + si + " col " + mi + " = " + el.textContent + ")")));
// ...and no reserved second line either: without a baseline that would spend a
// row of height per variant to display nothing.
assert(document.getElementById("summaryTable").className.indexOf("has-ratios") < 0,
  "no baseline => no reserved ratio line, got " + document.getElementById("summaryTable").className);
console.log("ALL_OK");
`
		jsPath := filepath.Join(dir, "nobaseline_test.js")
		if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+html[s+len("<script>"):e]+"\n"+probe), 0o644); err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
		if err != nil || !strings.Contains(string(out), "ALL_OK") {
			t.Fatalf("node no-baseline test failed: %v\n%s", err, out)
		}
	}

	t.Run("two arms, neither is hbm", func(t *testing.T) {
		dir := t.TempDir()
		for _, a := range []string{"weka-rdma", "dram1t"} {
			rec, _ := benchFixtureData(a, base)
			writeMixedJSONL(t, dir, a, rec, nil)
		}
		check(t, dir)
	})

	t.Run("single hbm arm has nothing to compare", func(t *testing.T) {
		dir := t.TempDir()
		rec, _ := benchFixtureData("hbm-solo", base)
		writeMixedJSONL(t, dir, "hbm-solo", rec, nil)
		check(t, dir)
	})
}

// TestRequestHoverTokensJS pins the per-request token breakdown. Toggling "Show
// Requests" exposes latency outliers, but a slow request is a different problem
// depending on whether it carried an enormous prompt, missed the cache, or
// generated a long completion — none of which the tooltip used to say.
func TestRequestHoverTokensJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS request-hover test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=hovertok"
	var rec []requestDataRecord
	// Nine ordinary requests at 500-token context, then one outlier: 20x the
	// context, almost none of it cached, and slow — the shape being diagnosed.
	for i := 0; i < 9; i++ {
		st := base.Add(time.Duration(i) * 10 * time.Second)
		rec = append(rec, requestDataRecord{
			StartTime: st, EndTime: st.Add(time.Second),
			TTFT: 150, ResponseMs: 900, Model: model,
			SeriesNum: 1, RequestNum: i + 1,
			InputTokens: 100, CachedTokens: 400, OutputTokens: 50,
		})
	}
	outlier := base.Add(90 * time.Second)
	rec = append(rec, requestDataRecord{
		StartTime: outlier, EndTime: outlier.Add(40 * time.Second),
		TTFT: 30000, ResponseMs: 40000, Model: model,
		SeriesNum: 1, RequestNum: 10,
		InputTokens: 9000, CachedTokens: 1000, OutputTokens: 2500,
	})
	writeMixedJSONL(t, dir, "hovertok", rec, nil)

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
// The breakdown only reaches the user with request dots drawn, which is the
// mode the outliers are spotted in.
document.getElementById("showDots").checked = true;
draw();
const mm = (__listeners["chart:mousemove"] || [])[0];
const tip = document.getElementById("tooltip");
const s = DATA[0];

function hoverRequest(rn) {
  const r = s._view.find(x => x.rn === rn);
  assert(r, "fixture has request " + rn);
  tip.innerHTML = ""; tip.style.display = "none";
  mm({ clientX: mapX(r.t), clientY: mapY(r.resp) });
  assert(tip.style.display === "block", "req " + rn + ": tooltip shown");
  return tip.innerHTML;
}

// The outlier: 9k uncached + 1k cached = 10k prompt, only 10% cached, and 20x
// the 500-token median context — every clause of "why was this slow".
const out = hoverRequest(10);
assert(out.indexOf("Input: 10.0k tok") >= 0, "prompt = uncached + cached = 10k: " + out);
assert(out.indexOf("cached: 1.0k") >= 0, "cached input shown: " + out);
assert(out.indexOf("(10.0%)") >= 0, "cached share shown: " + out);
assert(out.indexOf("uncached: 9.0k") >= 0, "uncached input shown: " + out);
assert(out.indexOf("Output: 2.5k tok") >= 0, "output shown: " + out);
assert(out.indexOf("20.0x median ctx") >= 0, "context vs the series median shown: " + out);
// The identity and latency lines it already carried must survive.
assert(out.indexOf("series 1, req 10") >= 0, "identity kept: " + out);
assert(out.indexOf("Response: 40000.0 ms") >= 0, "response time kept: " + out);

// An ordinary request reads as ordinary: 80% cached, 1.0x the median.
const ord = hoverRequest(3);
assert(ord.indexOf("Input: 500 tok") >= 0, "ordinary prompt: " + ord);
assert(ord.indexOf("(80.0%)") >= 0, "ordinary cached share: " + ord);
assert(ord.indexOf("1.0x median ctx") >= 0, "ordinary ctx vs median: " + ord);
console.log("ALL_OK");
`
	jsPath := filepath.Join(dir, "hover_tokens_test.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node request-hover test failed: %v\n%s", err, out)
	}
}
