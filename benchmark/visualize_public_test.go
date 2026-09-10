package benchmark

import (
	"encoding/json"
	"os"
	"os/exec"
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

// rawPublicSeries mirrors the exported PUBLIC_DATA's JSON shape for
// test-side decoding: every point array is decoded generically as
// [][]float64 -- the point count and values are exactly what a real browser
// would receive either way.
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
// public export and decodes it. JSON.stringify inside the export produces
// the payload with no embedded newlines, so the literal is exactly the rest
// of the `const PUBLIC_DATA = ...;` source line.
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

func names(data []rawPublicSeries) []string {
	out := make([]string, len(data))
	for i, s := range data {
		out[i] = s.Name
	}
	return out
}

// ---------------------------------------------------------------------------
// Shared node harness: generate a full interactive report from a fixture
// directory, extract its embedded <script> body, and call
// buildPublicReportHtml(resolutionMs, smoothingMs) directly under node --
// no DOM/Blob/URL stubbing needed for the call itself (buildPublicReportHtml
// is a pure function per its own doc comment in visualize.go), only the
// reportDOMStub the OUTER script's own top-level setup code needs to load at
// all. The result is JSON-encoded and printed between unique markers so the
// Go test can capture and decode it robustly regardless of what characters
// the exported HTML contains.
// ---------------------------------------------------------------------------

// generateInteractiveScript builds dir's interactive report.html and returns
// its embedded <script>...</script> body.
func generateInteractiveScript(t *testing.T, dir string, concurrency int) string {
	t.Helper()
	htmlPath, err := GenerateVisualization(dir, concurrency)
	if err != nil {
		t.Fatalf("GenerateVisualization: %v", err)
	}
	return extractOuterScript(t, htmlPath)
}

func extractOuterScript(t *testing.T, htmlPath string) string {
	t.Helper()
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	start := strings.Index(html, "<script>")
	end := strings.Index(html, "</script>")
	if start < 0 || end < 0 {
		t.Fatal("script block not found in interactive report")
	}
	// The embedded public template contains no literal </script> once
	// JSON-escaped (json.Marshal \u-escapes it) -- so this simple
	// first-occurrence extraction is safe; verified explicitly by
	// TestPublicTemplateEscapingSurvivesEmbedding below rather than assumed.
	return html[start+len("<script>") : end]
}

// nodeOrSkip returns the node binary path, skipping the test when node is
// not installed (matching every other JS-harness test in this package).
func nodeOrSkip(t *testing.T) string {
	t.Helper()
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS export test skipped")
	}
	return nodeBin
}

// buildPublicExport runs script (an interactive report's own extracted
// <script> body) under node with reportDOMStub, calls
// buildPublicReportHtml(resolutionMs, smoothingMs), and returns the
// resulting exported HTML document as a string.
func buildPublicExport(t *testing.T, script string, resolutionMs, smoothingMs int) string {
	t.Helper()
	nodeBin := nodeOrSkip(t)
	probe := `
const __result = buildPublicReportHtml(` + strconv.Itoa(resolutionMs) + `, ` + strconv.Itoa(smoothingMs) + `);
console.log("===PUBLIC_EXPORT_START===");
console.log(JSON.stringify(__result));
console.log("===PUBLIC_EXPORT_END===");
`
	dir := t.TempDir()
	jsPath := filepath.Join(dir, "export_probe.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil {
		t.Fatalf("node export probe failed: %v\n%s", err, out)
	}
	s := string(out)
	const startMarker = "===PUBLIC_EXPORT_START===\n"
	const endMarker = "\n===PUBLIC_EXPORT_END==="
	si := strings.Index(s, startMarker)
	ei := strings.Index(s, endMarker)
	if si < 0 || ei < 0 || ei <= si {
		t.Fatalf("could not locate export markers in node output:\n%s", s)
	}
	si += len(startMarker)
	var html string
	if err := json.Unmarshal([]byte(s[si:ei]), &html); err != nil {
		t.Fatalf("decode exported html json: %v\nraw: %s", err, s[si:ei])
	}
	return html
}

// ---------------------------------------------------------------------------
// TEST 1 (the important one): the exported public report's entire purpose is
// that per-request data is ABSENT from the emitted file, not merely hidden
// from the UI -- hiding controls is defeated by opening devtools, but the
// data cannot leak if it was never written. The fixture plants both
// structural markers (series_guid, series_num, ...) AND distinctive literal
// VALUES (a fake endpoint host, a fake series_guid, a fake run_id) so this
// test catches a leak under any key, not only a leak that kept its original
// field name.
//
// The forbidden table also covers the export pipeline's OWN failure modes,
// specific to it being built from the interactive report's own already
// -loaded page rather than a separate Go build: the outer report's raw
// per-request field-list constant, the outer report's top-level functions
// appearing verbatim (which would mean the wrong script was captured
// instead of its result), and a literal "model" JSON key (which would mean
// the payload whitelist in buildPublicReportHtml was bypassed by a spread).
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

	script := generateInteractiveScript(t, dir, 8)
	html := buildPublicExport(t, script, 30000, 300000)

	// --- ABSENCE assertions -------------------------------------------------
	forbidden := []struct{ substr, why string }{
		{"RAW_DATA", "RAW_DATA is the interactive report's per-request array constant; its presence means the interactive script leaked into the exported file"},
		{"REC_FIELDS", "the interactive report's raw per-request field-list constant; its presence implies per-request rows are reachable"},
		{"buildPublicReportHtml", "the export's OWN source function appearing verbatim in its output would mean the interactive report's whole script was captured instead of the export function's RESULT"},
		{"series_num", "per-request series index -- appears in the interactive report only as a CSV/export column name, never needed for an aggregate-only view"},
		{"request_num", "per-request ordinal -- same reasoning as series_num"},
		{"series_guid", "per-instance GUID -- a joinable key back to a specific captured session/request in internal logs"},
		{"guidTable", "the interactive report's per-series GUID interning table; its presence implies per-request GUIDs are reachable"},
		{"run_id", "the run identifier -- looks harmless in isolation but is a joinable key back to internal run logs/results directories"},
		{`"model":`, "a literal model JSON key in the export would mean buildPublicReportHtml's payload whitelist was bypassed (e.g. by spreading the live series/params object instead of listing fields explicitly)"},
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
	// export is empty or broken.
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

	// --- node --check on the exported file's OWN inner script --------------
	// New coverage with no counterpart in the pre-rewrite test: this is what
	// actually catches an escaping/substitution bug in the ported template
	// (node --check on the OUTER report's script, exercised by
	// TestPublicTemplateEscapingSurvivesEmbedding below, only proves the
	// outer script parses -- it says nothing about the embedded template
	// once substituted and extracted from a REAL generated export).
	nodeBin := nodeOrSkip(t)
	innerScript := extractInnerScript(t, html)
	innerPath := filepath.Join(t.TempDir(), "inner_check.js")
	if err := os.WriteFile(innerPath, []byte(innerScript), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nodeBin, "--check", innerPath).CombinedOutput(); err != nil {
		t.Fatalf("node --check failed on the exported file's own inner script: %v\n%s", err, out)
	}
}

// extractInnerScript pulls the <script>...</script> body out of a GENERATED
// public export (the string buildPublicExport returns), for both syntax
// checking (node --check) and full execution (TestPublicExportInnerScriptRuns
// below).
func extractInnerScript(t *testing.T, html string) string {
	t.Helper()
	start := strings.Index(html, "<script>")
	end := strings.Index(html, "</script>")
	if start < 0 || end < 0 {
		t.Fatal("script block not found in exported public report")
	}
	return html[start+len("<script>") : end]
}

// ---------------------------------------------------------------------------
// TEST 2: the embedded template's escaping actually solves the nesting
// hazard, not just by convention. json.Marshal HTML-\u-escapes "<"/">"/"&"
// by default, so the ported template's own literal </script> must appear in
// the outer page ONLY as the escaped \u003c/script\u003e sequence, never
// literally -- and the outer generated HTML must therefore contain exactly
// ONE literal </script> (the outer report's own).
// ---------------------------------------------------------------------------

func TestPublicTemplateEscapingSurvivesEmbedding(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 8, 20, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("hbm-gpu", base, 4, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, pubSecretRunID+"-esc")

	htmlPath, err := GenerateVisualization(dir, 8)
	if err != nil {
		t.Fatalf("GenerateVisualization: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)

	if n := strings.Count(html, "</script>"); n != 1 {
		t.Fatalf("expected exactly 1 literal </script> in the outer report, got %d", n)
	}
	marker := "const PUBLIC_TEMPLATE = "
	i := strings.Index(html, marker)
	if i < 0 {
		t.Fatal("PUBLIC_TEMPLATE declaration not found")
	}
	rest := html[i+len(marker):]
	end := strings.Index(rest, ";\n")
	if end < 0 {
		t.Fatal("PUBLIC_TEMPLATE declaration has no terminator")
	}
	literal := rest[:end]
	if len(literal) < 10000 {
		t.Errorf("PUBLIC_TEMPLATE literal looks too short (%d bytes) to be the whole ported template", len(literal))
	}
	if strings.Contains(literal, "</script>") {
		t.Errorf("PUBLIC_TEMPLATE literal contains an UNESCAPED </script> -- the nesting hazard is not solved")
	}
	if !strings.Contains(literal, `\u003c/script\u003e`) {
		t.Errorf("PUBLIC_TEMPLATE literal does not contain the expected escaped \\u003c/script\\u003e sequence")
	}

	nodeBin := nodeOrSkip(t)
	start := strings.Index(html, "<script>")
	scriptEnd := strings.Index(html, "</script>")
	outerScript := html[start+len("<script>") : scriptEnd]
	jsPath := filepath.Join(t.TempDir(), "outer_check.js")
	if err := os.WriteFile(jsPath, []byte(outerScript), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nodeBin, "--check", jsPath).CombinedOutput(); err != nil {
		t.Fatalf("node --check failed on the outer report's script: %v\n%s", err, out)
	}
}

// ---------------------------------------------------------------------------
// TEST 3: --labels must suppress the internal alias entirely, and the
// export must read s.name (already honoring --labels/alias-suppression)
// and nothing else for arm labels.
//
// When --labels is given, the interactive report is built with
// keepFileNames=true (see GenerateVisualizationMerged), which pins each
// arm's display name to the merged source file's basename (the label)
// instead of resolving a record alias -- so the internal alias never
// becomes the display name, and (per this test) never appears anywhere else
// in the exported file either.
//
// When --labels is NOT given, the internal alias comes through as the
// arm's display name (the same clean alias resolution the interactive
// report always used) -- documented, intentional behavior: whoever wants
// that closed off needs to pass --labels rather than relying on the
// exporter to guess a "safe" name from the model spec.
// ---------------------------------------------------------------------------

func TestPublicExportLabelsSuppressAlias(t *testing.T) {
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
		htmlPath, err := GenerateVisualizationMerged(
			[]string{baseDir, wekaDir}, []string{"Baseline", "Weka-Offload"}, outDir, 8, 0)
		if err != nil {
			t.Fatalf("generate merged interactive report with labels: %v", err)
		}
		script := extractOuterScript(t, htmlPath)
		html := buildPublicExport(t, script, 30000, 300000)
		if strings.Contains(html, secretAlias2) {
			t.Errorf("export contains internal alias %q even though --labels was given", secretAlias2)
		}
		for _, want := range []string{"Baseline", "Weka-Offload"} {
			if !strings.Contains(html, want) {
				t.Errorf("export missing supplied label %q", want)
			}
		}
	})

	t.Run("without --labels: the internal alias comes through as the arm name (documented, not fixed)", func(t *testing.T) {
		baseDir, wekaDir := buildDirs(t)
		outDir := filepath.Join(filepath.Dir(baseDir), "merged-unlabeled")
		htmlPath, err := GenerateVisualizationMerged([]string{baseDir, wekaDir}, nil, outDir, 8, 0)
		if err != nil {
			t.Fatalf("generate merged interactive report without labels: %v", err)
		}
		script := extractOuterScript(t, htmlPath)
		html := buildPublicExport(t, script, 30000, 300000)
		if !strings.Contains(html, secretAlias2) {
			t.Errorf("expected the internal alias %q to appear as the arm's display name when --labels is omitted -- "+
				"if this now fails because the alias is suppressed, update this test's comment, don't just relax it",
				secretAlias2)
		}
	})
}

// ---------------------------------------------------------------------------
// TEST 4 (cheap): the exported public report's honesty affordances survive.
//
//   - The footer must state the run length, the selected downsample
//     interval, and that per-request detail is not included -- these are
//     what let a recipient without access to the raw data judge how much
//     resolution they're looking at, rather than mistaking a smoothed
//     aggregate for the real thing.
//   - The exported file's OWN script states, once, how the latency lines/
//     cache-mix bands/dataset line are derived: a smoothing disclosure
//     naming the actual window selected (or "none" when None is picked),
//     the cache-mix "exact" wording, and the dataset-line "mean" wording.
//   - Both TTFT p50 AND p95 must be emitted (not just p50): p95 is the
//     deliberate guardrail against a heavy tail being hidden by only ever
//     publishing the median -- a median-only report could show a "fast"
//     run while 1 in 20 requests stalled for seconds.
// ---------------------------------------------------------------------------

func TestPublicExportHonestyAffordances(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 8, 12, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("hbm-gpu", base, 6, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, pubSecretRunID+"-c")

	script := generateInteractiveScript(t, dir, 8)

	t.Run("smoothing selected", func(t *testing.T) {
		html := buildPublicExport(t, script, 45000, 60000)
		for _, want := range []string{
			"Run length", "Downsampled to",
			"Per-request detail is not included in this file",
		} {
			if !strings.Contains(html, want) {
				t.Errorf("footer missing %q", want)
			}
		}
		// The smoothing disclosure is built at LOAD time inside the exported
		// file's own script (footerDisclosureText, written into
		// #footerDisclosure via textContent) -- it is never baked into the
		// static HTML, so it must be read by actually running the inner
		// script, not by grepping the raw document.
		disclosure := runInnerScriptAndGetFooterDisclosure(t, html)
		for _, want := range []string{"1m moving average", "Cache mix re-aggregated to 1m (exact)", "dataset line uses the mean"} {
			if !strings.Contains(disclosure, want) {
				t.Errorf("footer disclosure %q missing %q", disclosure, want)
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
	})

	t.Run("smoothing None", func(t *testing.T) {
		html := buildPublicExport(t, script, 30000, 0)
		disclosure := runInnerScriptAndGetFooterDisclosure(t, html)
		if !strings.Contains(disclosure, "Smoothing: none") {
			t.Errorf("footer disclosure %q missing the smoothing-none wording", disclosure)
		}
	})
}

// runInnerScriptAndGetFooterDisclosure executes the exported file's own
// <script> under node with reportDOMStub and returns the #footerDisclosure
// element's resulting textContent -- the disclosure text is written by the
// script at load time, not present in the static HTML.
func runInnerScriptAndGetFooterDisclosure(t *testing.T, html string) string {
	t.Helper()
	nodeBin := nodeOrSkip(t)
	innerScript := extractInnerScript(t, html)
	probe := `
console.log("===DISCLOSURE_START===");
console.log(document.getElementById("footerDisclosure").textContent);
console.log("===DISCLOSURE_END===");
`
	jsPath := filepath.Join(t.TempDir(), "disclosure_probe.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+innerScript+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil {
		t.Fatalf("node disclosure probe failed: %v\n%s", err, out)
	}
	s := string(out)
	si := strings.Index(s, "===DISCLOSURE_START===\n")
	ei := strings.Index(s, "\n===DISCLOSURE_END===")
	if si < 0 || ei < 0 {
		t.Fatalf("could not find disclosure markers in node output:\n%s", s)
	}
	return s[si+len("===DISCLOSURE_START===\n") : ei]
}

// ---------------------------------------------------------------------------
// TEST 5: resolution and smoothing selections actually change the export.
// ---------------------------------------------------------------------------

func TestPublicExportResolutionAndSmoothing(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 8, 13, 9, 0, 0, 0, time.UTC)
	// 40 requests over ~13 minutes gives enough points to see resolution
	// differences across 15s..5m.
	recs, samples := buildPublicFixtureArm("hbm-gpu", base, 40, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, pubSecretRunID+"-res")
	script := generateInteractiveScript(t, dir, 8)

	pointCountAt := func(resolutionMs int) int {
		html := buildPublicExport(t, script, resolutionMs, 300000)
		data := extractPublicData(t, html)
		if len(data) != 1 {
			t.Fatalf("PUBLIC_DATA has %d series, want 1", len(data))
		}
		return len(data[0].RespP50)
	}

	resolutions := []int{15000, 30000, 60000, 300000}
	var prev int
	for i, ms := range resolutions {
		n := pointCountAt(ms)
		if n == 0 {
			t.Fatalf("resolution %dms produced zero points", ms)
		}
		if i > 0 && n > prev {
			t.Errorf("resolution %dms produced MORE points (%d) than the previous, finer resolution (%d) -- widening the interval must never increase point count", ms, n, prev)
		}
		prev = n
	}

	// Smoothing must never change how many points exist, only their values.
	htmlNone := buildPublicExport(t, script, 30000, 0)
	htmlSmooth := buildPublicExport(t, script, 30000, 300000)
	dataNone := extractPublicData(t, htmlNone)
	dataSmooth := extractPublicData(t, htmlSmooth)
	if len(dataNone) != 1 || len(dataSmooth) != 1 {
		t.Fatalf("expected 1 series in both exports")
	}
	// Neither the exported PUBLIC_DATA points nor their point count are
	// affected by smoothing -- smoothing is a LOAD-TIME-ONLY transform
	// inside the exported file's own script (SMOOTH_WINDOW_MS), applied to
	// PUBLIC_DATA after rehydration, never baked into the emitted arrays.
	if len(dataNone[0].RespP50) != len(dataSmooth[0].RespP50) {
		t.Errorf("smoothing changed PUBLIC_DATA point count: none=%d smooth=%d (smoothing must only affect the exported file's OWN rendering at load, never the embedded arrays)",
			len(dataNone[0].RespP50), len(dataSmooth[0].RespP50))
	}

	// Footer text changes to match each resolution selection.
	for _, tc := range []struct {
		ms    int
		label string
	}{
		{15000, "15s"}, {30000, "30s"}, {60000, "1m"}, {300000, "5m"},
	} {
		html := buildPublicExport(t, script, tc.ms, 300000)
		want := "Downsampled to " + tc.label + " intervals"
		if !strings.Contains(html, want) {
			t.Errorf("resolution %dms: footer missing %q", tc.ms, want)
		}
	}
}

// ---------------------------------------------------------------------------
// TEST 6: the exported inner script actually EXECUTES under a DOM stub, not
// just parses. This is the strongest replacement for the coverage lost by
// deleting the Go aggregation pipeline's own crosscheck test: it proves the
// ported template's runtime code path (rehydration, computeSmoothed,
// recalcYMax, draw, legend wiring) runs to completion against a real
// generated export.
// ---------------------------------------------------------------------------

func TestPublicExportInnerScriptRuns(t *testing.T) {
	nodeBin := nodeOrSkip(t)
	dir := t.TempDir()
	base := time.Date(2026, 8, 14, 9, 0, 0, 0, time.UTC)
	baseRecs, baseSamples := buildPublicFixtureArm("hbm-gpu", base, 10, 1)
	wekaRecs, wekaSamples := buildPublicFixtureArm("weka-run", base, 10, 0)
	writePublicFixtureFile(t, dir, "a", baseRecs, baseSamples, pubSecretRunID+"-run-a")
	writePublicFixtureFile(t, dir, "b", wekaRecs, wekaSamples, pubSecretRunID+"-run-b")
	script := generateInteractiveScript(t, dir, 8)
	html := buildPublicExport(t, script, 30000, 300000)
	innerScript := extractInnerScript(t, html)

	probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
assert(typeof DATA !== "undefined" && DATA.length === 2, "DATA has 2 series, got " + (typeof DATA !== "undefined" ? DATA.length : "undefined"));
assert(typeof SMOOTH !== "undefined" && SMOOTH.length === 2, "SMOOTH computed for 2 series");
draw();
const cb = document.getElementById("showCacheMix");
if (cb) { cb.checked = true; draw(); }
document.getElementById("showTotals").checked = true;
draw();
const mm = (__listeners["chart:mousemove"] || [])[0];
if (mm) mm({ clientX: margin.left + 20, clientY: margin.top + 20 });
console.log("ALL_OK");
`
	jsPath := filepath.Join(t.TempDir(), "inner_run.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+innerScript+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("exported inner script failed to run: %v\n%s", err, out)
	}
}

// ---------------------------------------------------------------------------
// TEST 7: the resolution downsample is a STEP FUNCTION (last point
// at-or-before each boundary, final point always kept), never a mean --
// see §0.3 of the implementation plan. This is a narrow, easy-to-regress
// detail: a "fix" that switches to averaging would still produce the same
// POINT COUNT (TestPublicExportResolutionAndSmoothing's own check would
// stay green), so only a VALUE-level assertion like this one catches it.
// Extracts pubDownsamplePts/pubDownsampleMix/pubDownsampleAdt as a
// contiguous block (same technique as TestCacheMixLookupHelpersJS) and
// exercises them directly with synthetic points -- no report generation
// needed.
// ---------------------------------------------------------------------------

func TestPublicExportDownsampleIsStepFunctionNotMean(t *testing.T) {
	nodeBin := nodeOrSkip(t)
	dir := t.TempDir()
	base := time.Date(2026, 8, 15, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("hbm-gpu", base, 4, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, pubSecretRunID+"-step")
	script := generateInteractiveScript(t, dir, 8)

	start := strings.Index(script, "function pubDownsamplePts(")
	end := strings.Index(script, "function pubBuildCum(")
	if start < 0 || end < 0 || end <= start {
		t.Fatalf("pubDownsamplePts..pubDownsampleAdt block not found (start=%d end=%d)", start, end)
	}
	helpers := script[start:end]

	probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
// A sharp spike (100) lands just BEFORE the 30s boundary, then the value
// drops back to 4 right after. Step-function downsampling at a 30s
// interval keeps the LAST point at-or-before each boundary -- the point at
// t=29999 (v=100) -- so the emitted value at that boundary is exactly 100,
// never the bucket's mean (which would average in the neighboring 1/2/3
// values and land far from 100).
const pts = [
  { t: 0,     v: 1 },
  { t: 10000, v: 2 },
  { t: 20000, v: 3 },
  { t: 29999, v: 100 },
  { t: 30001, v: 4 },
  { t: 45000, v: 5 },
];
const down = pubDownsamplePts(pts, 30000);
assert(down.length >= 2, "expected at least 2 downsampled points, got " + down.length);
const first = down[0];
assert(first.t === 29999, "step-function boundary point should be the LAST point at-or-before the 30s boundary (t=29999), got t=" + first.t);
assert(first.v === 100, "step-function boundary VALUE must be the spike itself (100), a mean-per-bucket reduction would NOT equal 100 (bug), got " + first.v);
// intervalMs<=0 must be a pure passthrough.
const passthrough = pubDownsamplePts(pts, 0);
assert(passthrough === pts, "intervalMs<=0 must return the input unchanged");
console.log("ALL_OK");
`
	jsPath := filepath.Join(t.TempDir(), "downsample_check.js")
	if err := os.WriteFile(jsPath, []byte(helpers+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("step-function downsample check failed: %v\n%s", err, out)
	}
}
