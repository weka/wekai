package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Fixtures shared by the tests below. All of them plant DISTINCTIVE, easily
// greppable sensitive values (an endpoint host, a series_guid, a run_id) so
// a leak test can catch the data escaping under a renamed key, not just
// under its usual field name.
// ---------------------------------------------------------------------------

const (
	pubSecretEndpointHost = "internal-vllm-9k3x.example.com"
	pubSecretAlias        = "weka-secret-arm"
	pubSecretGUIDFrag     = "DEADBEEF-9f3d2b1a"
	pubSecretRunID        = "RUN-CAFEBABE-77321"
)

func pubDynamicModel(alias string) string {
	return "dynamic/http://" + pubSecretEndpointHost + ":8000/v1,type=openai_vllm,alias=" + alias
}

// buildPublicFixtureArm builds one arm's records + cache-mix samples. The
// first errCount of nReq records are marked as errors, so OK/Err aggregate
// counts are independently checkable per arm. Every record carries a
// distinct series_guid built from pubSecretGUIDFrag.
func buildPublicFixtureArm(alias string, base time.Time, nReq, errCount int) ([]requestDataRecord, []vllmMetricsSample) {
	model := pubDynamicModel(alias)
	var records []requestDataRecord
	for i := 0; i < nReq; i++ {
		st := base.Add(time.Duration(i) * 20 * time.Second)
		rec := requestDataRecord{
			StartTime:    st,
			EndTime:      st.Add(2 * time.Second),
			TTFT:         120 + float64(i)*5,
			ResponseMs:   1800 + float64(i)*10,
			Model:        model,
			SeriesGUID:   "sess-" + pubSecretGUIDFrag + ":inst-" + strconv.Itoa(i),
			SeriesNum:    i/2 + 1,
			RequestNum:   i + 1,
			InputTokens:  100,
			CachedTokens: 400,
			OutputTokens: 50,
		}
		if i < errCount {
			rec.IsError = true
			rec.ErrorMessage = "boom"
		}
		records = append(records, rec)
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

// writePublicFixtureFile writes one arm's JSONL file with a run_params
// header carrying runID and the arm's (sensitive) endpoint+alias model
// string -- exactly what a real `benchmark auto --save-request-data` run
// writes -- followed by its records and samples.
func writePublicFixtureFile(t *testing.T, dir, name string, records []requestDataRecord, samples []vllmMetricsSample, runID string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, name+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	cfg := AutoBenchmarkConfig{Model: records[0].Model, RunID: runID, Concurrency: 8}
	if err := enc.Encode(buildRunParams(cfg, records[0].StartTime)); err != nil {
		t.Fatal(err)
	}
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
}

// rawPublicSeries mirrors publicSeries' JSON shape for test-side decoding.
// It cannot reuse publicSeries directly: publicPoint/publicErrBar/
// publicMixSeg/publicAdtPoint/publicCumPoint only implement MarshalJSON (the
// production code never needs to read its own report back), so every point
// array here is decoded generically as [][]float64 -- the point count and
// values are exactly what a real browser would receive either way.
type rawPublicSeries struct {
	Name    string      `json:"name"`
	OK      int         `json:"ok"`
	Err     int         `json:"err"`
	RespP50 [][]float64 `json:"respP50"`
	TTFTP50 [][]float64 `json:"ttftP50"`
	TTFTP95 [][]float64 `json:"ttftP95"`
	ErrBars [][]float64 `json:"errBars,omitempty"`
	Mix     [][]float64 `json:"mix,omitempty"`
	Adt     [][]float64 `json:"adt,omitempty"`
	Cum     [][]float64 `json:"cum,omitempty"`
}

// extractPublicData pulls PUBLIC_DATA's JSON literal out of a generated
// report-public.html and decodes it. json.Marshal produces the payload with
// no embedded newlines, so the literal is exactly the rest of the
// `const PUBLIC_DATA = ...;` source line.
func extractPublicData(t *testing.T, html string) []rawPublicSeries {
	t.Helper()
	const marker = "const PUBLIC_DATA = "
	i := strings.Index(html, marker)
	if i < 0 {
		t.Fatal("PUBLIC_DATA declaration not found in report")
	}
	rest := html[i+len(marker):]
	end := strings.IndexByte(rest, '\n')
	if end < 0 {
		t.Fatal("PUBLIC_DATA declaration has no line terminator")
	}
	line := strings.TrimSuffix(strings.TrimSpace(rest[:end]), ";")
	var data []rawPublicSeries
	if err := json.Unmarshal([]byte(line), &data); err != nil {
		t.Fatalf("parse PUBLIC_DATA: %v\n%s", err, line)
	}
	return data
}

// ---------------------------------------------------------------------------
// TEST 1 (the important one): --public's entire purpose is that per-request
// data is ABSENT from the emitted file, not merely hidden from the UI --
// hiding controls is defeated by opening devtools, but the data cannot leak
// if it was never written. Before this test, that property had only ever
// been checked by hand-grepping a generated file once, which catches
// nothing when the emit path is later widened (as the internal report's own
// per-request record was, in the commit right before this one on this
// branch) to embed one more field.
//
// The fixture plants both structural markers (series_guid, series_num, ...)
// AND distinctive literal VALUES (a fake endpoint host, a fake series_guid,
// a fake run_id) so this test catches a leak under any key, not only a
// leak that kept its original field name.
// ---------------------------------------------------------------------------

func TestPublicReportOmitsPerRequestData(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 8, 10, 9, 0, 0, 0, time.UTC)

	// "hbm-gpu" classifies as the gpu/baseline arm (ClassifyArmAlias), so
	// FindBaselineIndex picks it and the summary table's ratio columns (the
	// "aggregate values are present" positive check below) are populated.
	baseRecs, baseSamples := buildPublicFixtureArm("hbm-gpu", base, 5, 0)
	wekaRecs, wekaSamples := buildPublicFixtureArm(pubSecretAlias, base, 5, 1)
	writePublicFixtureFile(t, dir, "a", baseRecs, baseSamples, pubSecretRunID+"-a")
	writePublicFixtureFile(t, dir, "b", wekaRecs, wekaSamples, pubSecretRunID+"-b")

	htmlPath, err := GeneratePublicVisualization(dir, 8, 0, 0)
	if err != nil {
		t.Fatalf("generate public visualization: %v", err)
	}
	if filepath.Base(htmlPath) != "report-public.html" {
		t.Fatalf("public generation wrote %q, want report-public.html", filepath.Base(htmlPath))
	}
	if _, err := os.Stat(filepath.Join(dir, "report.html")); err == nil {
		t.Fatalf("--public must never also produce/overwrite report.html")
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)

	// --- ABSENCE assertions -------------------------------------------------
	forbidden := []struct{ substr, why string }{
		{"RAW_DATA", "RAW_DATA is the internal report's per-request array constant; its presence means the internal template leaked into the public build"},
		{"series_num", "per-request series index -- appears in the internal report only as a CSV/export column name, never needed for an aggregate-only view"},
		{"request_num", "per-request ordinal -- same reasoning as series_num"},
		{"series_guid", "per-instance GUID -- a joinable key back to a specific captured session/request in internal logs"},
		{"guidTable", "the internal report's per-series GUID interning table; its presence implies per-request GUIDs are reachable"},
		{"run_id", "the run identifier -- looks harmless in isolation but is a joinable key back to internal run logs/results directories"},
		{pubSecretEndpointHost, "the literal endpoint host planted in the fixture: this is the strong check -- it catches the value leaking under ANY key, not just under a field literally named series_guid/run_id/etc."},
		{"dynamic/http://", "the raw dynamic-model spec prefix; if present, the endpoint URL and alias= are reachable verbatim even under a renamed field"},
		{"alias=" + pubSecretAlias, "the alias= query parameter as it appears inside the raw model-spec string (distinct from the resolved short alias name, which IS expected to appear as the arm's display name)"},
		{"sess-" + pubSecretGUIDFrag, "the literal series_guid value planted in the fixture"},
		{pubSecretRunID, "the literal run_id value planted in the fixture"},
	}
	for _, f := range forbidden {
		if strings.Contains(html, f.substr) {
			t.Errorf("public report contains forbidden string %q (%s)", f.substr, f.why)
		}
	}

	// --- POSITIVE assertions -------------------------------------------------
	// Guards against the absence checks above passing vacuously because the
	// report is empty or broken.
	data := extractPublicData(t, html)
	if len(data) != 2 {
		t.Fatalf("PUBLIC_DATA has %d series, want 2", len(data))
	}
	byName := map[string]rawPublicSeries{}
	for _, s := range data {
		byName[s.Name] = s
	}
	baseArm, ok := byName["hbm-gpu"]
	if !ok {
		t.Fatalf("baseline arm %q missing from PUBLIC_DATA (got names %v)", "hbm-gpu", names(data))
	}
	wekaArm, ok := byName[pubSecretAlias]
	if !ok {
		t.Fatalf("weka arm %q missing from PUBLIC_DATA (got names %v)", pubSecretAlias, names(data))
	}
	if baseArm.OK != 5 || baseArm.Err != 0 {
		t.Errorf("baseline arm ok/err = %d/%d, want 5/0", baseArm.OK, baseArm.Err)
	}
	if wekaArm.OK != 4 || wekaArm.Err != 1 {
		t.Errorf("weka arm ok/err = %d/%d, want 4/1", wekaArm.OK, wekaArm.Err)
	}
	for _, chk := range []struct {
		label string
		n     int
	}{
		{"baseline respP50", len(baseArm.RespP50)},
		{"baseline ttftP50", len(baseArm.TTFTP50)},
		{"baseline ttftP95", len(baseArm.TTFTP95)},
		{"baseline cum", len(baseArm.Cum)},
		{"weka respP50", len(wekaArm.RespP50)},
		{"weka cum", len(wekaArm.Cum)},
	} {
		if chk.n == 0 {
			t.Errorf("%s is empty -- aggregate series data must be present", chk.label)
		}
	}
	if !strings.Contains(html, "summaryTable") {
		t.Errorf("static summary table markup missing")
	}
	if !strings.Contains(html, "(baseline)") {
		t.Errorf("baseline marker missing from summary table")
	}
}

func names(data []rawPublicSeries) []string {
	out := make([]string, len(data))
	for i, s := range data {
		out[i] = s.Name
	}
	return out
}

// ---------------------------------------------------------------------------
// TEST 2: --labels must suppress the internal alias entirely.
//
// When --labels is given, generateVisualizationPublic is called with
// keepFileNames=true (see GenerateVisualizationMergedPublic), which pins
// each arm's display name to the merged source file's basename (the label)
// instead of calling resolveRecordsAlias -- so the internal alias never
// becomes the display name, and (per this test) never appears anywhere else
// in the file either.
//
// When --labels is NOT given, keepFileNames is false and
// generateVisualizationPublic falls back to resolveRecordsAlias(records),
// i.e. the SAME clean alias extracted from the model spec (e.g.
// "weka-secret-arm2"), which becomes the arm's display name. That alias is
// often an internal naming convention (offload target, SKU, run tag) that a
// report handed to an external party should probably not carry -- but that
// is exactly what happens today. Documented here, not changed here, at the
// task's explicit instruction: whoever wants that closed off needs to name
// the arms (finally pass --labels) rather than the code silently guessing a
// "safe" name from the model spec.
// ---------------------------------------------------------------------------

func TestPublicReportLabelsSuppressAlias(t *testing.T) {
	const secretAlias2 = "weka-secret-arm2"
	base := time.Date(2026, 8, 11, 9, 0, 0, 0, time.UTC)
	baseRecs, baseSamples := buildPublicFixtureArm("hbm-gpu", base, 5, 0)
	wekaRecs, wekaSamples := buildPublicFixtureArm(secretAlias2, base, 5, 0)

	buildDirs := func(t *testing.T) (baseDir, wekaDir string) {
		t.Helper()
		root := t.TempDir()
		baseDir = filepath.Join(root, "base")
		wekaDir = filepath.Join(root, "weka")
		writePublicFixtureFile(t, baseDir, "a", baseRecs, baseSamples, pubSecretRunID+"-base")
		writePublicFixtureFile(t, wekaDir, "b", wekaRecs, wekaSamples, pubSecretRunID+"-weka")
		return baseDir, wekaDir
	}

	t.Run("with --labels: internal alias must not appear anywhere", func(t *testing.T) {
		baseDir, wekaDir := buildDirs(t)
		outDir := filepath.Join(filepath.Dir(baseDir), "merged-labeled")
		// Labels double as the merged-source file's basename (see
		// prepareMergedSources' safeFileBase call), which strips anything
		// outside [a-zA-Z0-9._-] -- a space would come back as "_", so these
		// are picked filename-safe already rather than testing that
		// unrelated sanitization.
		htmlPath, err := GenerateVisualizationMergedPublic(
			[]string{baseDir, wekaDir}, []string{"Baseline", "Weka-Offload"}, outDir, 8, 0, 0)
		if err != nil {
			t.Fatalf("generate merged public with labels: %v", err)
		}
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			t.Fatal(err)
		}
		html := string(b)
		if strings.Contains(html, secretAlias2) {
			t.Errorf("report contains internal alias %q even though --labels was given", secretAlias2)
		}
		for _, want := range []string{"Baseline", "Weka-Offload"} {
			if !strings.Contains(html, want) {
				t.Errorf("report missing supplied label %q", want)
			}
		}
	})

	t.Run("without --labels: the internal alias comes through as the arm name (documented, not fixed)", func(t *testing.T) {
		baseDir, wekaDir := buildDirs(t)
		outDir := filepath.Join(filepath.Dir(baseDir), "merged-unlabeled")
		htmlPath, err := GenerateVisualizationMergedPublic(
			[]string{baseDir, wekaDir}, nil, outDir, 8, 0, 0)
		if err != nil {
			t.Fatalf("generate merged public without labels: %v", err)
		}
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			t.Fatal(err)
		}
		html := string(b)
		if !strings.Contains(html, secretAlias2) {
			t.Errorf("expected the internal alias %q to appear as the arm's display name when --labels is omitted -- "+
				"if this now fails because the alias is suppressed, update this test's comment, don't just relax it",
				secretAlias2)
		}
	})
}

// ---------------------------------------------------------------------------
// TEST 3 (cheap): the public report's honesty affordances survive.
//
//   - The footer must state the run length, the downsample interval, and
//     that per-request detail is not included -- these are what let a
//     recipient without access to the raw data judge how much resolution
//     they're looking at, rather than mistaking a smoothed aggregate for
//     the real thing.
//   - Both TTFT p50 AND p95 must be emitted (not just p50): p95 is the
//     deliberate guardrail against a heavy tail being hidden by only ever
//     publishing the median -- a median-only report could show a "fast"
//     run while 1 in 20 requests stalled for seconds.
// ---------------------------------------------------------------------------

func TestPublicReportHonestyAffordances(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 8, 12, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("hbm-gpu", base, 6, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, pubSecretRunID+"-c")

	interval := 45 * time.Second
	htmlPath, err := GeneratePublicVisualization(dir, 8, 0, interval)
	if err != nil {
		t.Fatalf("generate public visualization: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)

	for _, want := range []string{
		"Run length", "Downsampled to " + interval.String() + " intervals",
		"Per-request detail is not included in this file",
	} {
		if !strings.Contains(html, want) {
			t.Errorf("footer missing %q", want)
		}
	}

	data := extractPublicData(t, html)
	if len(data) != 1 {
		t.Fatalf("PUBLIC_DATA has %d series, want 1", len(data))
	}
	s := data[0]
	if len(s.TTFTP50) == 0 {
		t.Errorf("ttftP50 is empty -- TTFT p50 series must be emitted")
	}
	if len(s.TTFTP95) == 0 {
		t.Errorf("ttftP95 is empty -- TTFT p95 must be emitted alongside p50 so a heavy tail cannot be hidden behind the median alone")
	}
}
