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
			StartTime:  st,
			EndTime:    st.Add(2 * time.Second),
			TTFT:       150,
			ResponseMs: 2000,
			Model:      model,
			SeriesNum:  1,
			RequestNum: i + 1,
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
	htmlPath, err := GenerateVisualizationMerged([]string{dirA, dirB}, nil, outDir, 4)
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
