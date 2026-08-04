package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestRunParamsRoundTrip: what a run records is what a reader gets back.
func TestRunParamsRoundTrip(t *testing.T) {
	cfg := AutoBenchmarkConfig{
		Model:                "dynamic/http://localhost:8000,type=openai_vllm,alias=SOAK8H512_pod0",
		RunID:                "run-abc",
		Concurrency:          28,
		HotSeriesConcurrency: 4,
		MaxSeries:            512,
		StartSeries:          512,
		Timeout:              8 * time.Hour,
		RequestTimeout:       5 * time.Minute,
		RouterReplayFile:     "/mnt/weka/replay.jsonl",
		Total:                1000,
	}
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	line, err := json.Marshal(buildRunParams(cfg, now))
	if err != nil {
		t.Fatal(err)
	}

	got, ok := parseRunParams(line)
	if !ok {
		t.Fatal("params row did not parse back")
	}
	if got.Version != runParamsSchemaVersion {
		t.Errorf("version = %d, want %d", got.Version, runParamsSchemaVersion)
	}
	if got.Concurrency != 28 || got.HotSeriesConcurrency != 4 || got.MaxSeries != 512 {
		t.Errorf("workload shape lost: conc=%d hot=%d series=%d", got.Concurrency, got.HotSeriesConcurrency, got.MaxSeries)
	}
	if got.Alias != "SOAK8H512_pod0" {
		t.Errorf("alias = %q, want SOAK8H512_pod0", got.Alias)
	}
	if got.Timeout != "8h0m0s" || got.TimeoutSec != 28800 {
		t.Errorf("timeout = %q / %v", got.Timeout, got.TimeoutSec)
	}
	if got.RouterReplayFile != "/mnt/weka/replay.jsonl" {
		t.Errorf("replay file = %q", got.RouterReplayFile)
	}
	// The hot pool runs on its own gate on top of the normal budget.
	if c := got.effectiveConcurrency(); c != 32 {
		t.Errorf("effectiveConcurrency = %d, want 28+4=32", c)
	}
	if s := got.summaryLine(); !strings.Contains(s, "conc=28") || !strings.Contains(s, "hot=4") ||
		!strings.Contains(s, "series=512") || !strings.Contains(s, "router-replay") {
		t.Errorf("summaryLine = %q", s)
	}
}

// TestRunParamsUnpinnedConcurrency: a hill-climber run pins no concurrency, and
// must report 0 ("no single number") rather than a fabricated one, so readers
// fall back to inferring a window instead of trusting a zero.
func TestRunParamsUnpinnedConcurrency(t *testing.T) {
	p := buildRunParams(AutoBenchmarkConfig{MaxSeries: 64}, time.Now())
	if c := p.effectiveConcurrency(); c != 0 {
		t.Errorf("effectiveConcurrency = %d, want 0 for an unpinned run", c)
	}
	if _, ok := parseRunParams([]byte(`{"record_type":"vllm_metrics_sample"}`)); ok {
		t.Error("a metrics sample must not parse as run params")
	}
	if _, ok := parseRunParams([]byte(`not json`)); ok {
		t.Error("malformed line must not parse as run params")
	}
}

// TestReadJSONLCompatibility pins both directions of compatibility across the
// record_type boundary: a file WITHOUT a params header (everything recorded
// before this existed) still reads, and a file carrying record types this
// build has never heard of still yields its request rows.
func TestReadJSONLCompatibility(t *testing.T) {
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("compat", base)

	t.Run("legacy file without params", func(t *testing.T) {
		dir := t.TempDir()
		path := writeMixedJSONL(t, dir, "legacy", rec, smp)
		records, samples, params, hasParams, err := readJSONLFileWithParams(path)
		if err != nil {
			t.Fatal(err)
		}
		if hasParams {
			t.Error("legacy file must not report params")
		}
		if params.Concurrency != 0 {
			t.Errorf("zero record expected, got concurrency %d", params.Concurrency)
		}
		if len(records) != len(rec) || len(samples) != len(smp) {
			t.Errorf("got %d/%d records/samples, want %d/%d", len(records), len(samples), len(rec), len(smp))
		}
	})

	t.Run("file with params", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "withparams.jsonl")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		enc := json.NewEncoder(f)
		if err := enc.Encode(buildRunParams(AutoBenchmarkConfig{
			Concurrency: 28, HotSeriesConcurrency: 4, MaxSeries: 512, StartSeries: 512,
		}, base)); err != nil {
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

		records, samples, params, hasParams, err := readJSONLFileWithParams(path)
		if err != nil {
			t.Fatal(err)
		}
		if !hasParams {
			t.Fatal("params header not seen")
		}
		if params.Concurrency != 28 || params.HotSeriesConcurrency != 4 {
			t.Errorf("params = conc %d hot %d", params.Concurrency, params.HotSeriesConcurrency)
		}
		// The header must not be mistaken for a request row: unmarshalling it
		// into requestDataRecord would otherwise succeed as an all-zero phantom.
		if len(records) != len(rec) {
			t.Errorf("got %d request rows, want %d — header leaked into records", len(records), len(rec))
		}
		if len(samples) != len(smp) {
			t.Errorf("got %d samples, want %d", len(samples), len(smp))
		}
	})

	t.Run("file from a newer wekai", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "future.jsonl")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		enc := json.NewEncoder(f)
		// A record type this build has never heard of, plus a params row
		// carrying fields it doesn't know: neither may disturb the request rows.
		if err := enc.Encode(map[string]any{"record_type": "gpu_power_sample", "watts": 700}); err != nil {
			t.Fatal(err)
		}
		if err := enc.Encode(map[string]any{
			"record_type": recordTypeRunParams, "params_version": 99,
			"concurrency": 28, "some_future_knob": "yes",
		}); err != nil {
			t.Fatal(err)
		}
		for _, r := range rec {
			if err := enc.Encode(r); err != nil {
				t.Fatal(err)
			}
		}
		f.Close()

		records, _, params, hasParams, err := readJSONLFileWithParams(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(records) != len(rec) {
			t.Errorf("got %d request rows, want %d — unknown types corrupted parsing", len(records), len(rec))
		}
		if !hasParams || params.Concurrency != 28 {
			t.Errorf("known params fields must still be read: hasParams=%v conc=%d", hasParams, params.Concurrency)
		}
	})
}

// TestVisualizationEmbedsRunParams: a report built from params-carrying data
// embeds the per-arm concurrency and workload shape, and one built from legacy
// data embeds neither — the report must not invent them.
func TestVisualizationEmbedsRunParams(t *testing.T) {
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("embed", base)

	t.Run("new data", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "arm.jsonl")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		enc := json.NewEncoder(f)
		if err := enc.Encode(buildRunParams(AutoBenchmarkConfig{
			Concurrency: 28, HotSeriesConcurrency: 4, MaxSeries: 512, StartSeries: 512,
			Timeout: 8 * time.Hour, RouterReplayFile: "/mnt/weka/replay.jsonl",
		}, base)); err != nil {
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

		// No --concurrency passed: the report must get it from the data.
		htmlPath, err := GenerateVisualization(dir, 0)
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			t.Fatal(err)
		}
		html := string(b)
		for _, want := range []string{
			`"conc":32`, // 28 + hot 4, embedded per series
			`"params":`, // workload shape embedded
			"conc=28",   // rendered summary line
			"hot=4",
			"series=512",
			"router-replay",
			"runparams", // the header element that displays it
		} {
			if !strings.Contains(html, want) {
				t.Errorf("report missing %q", want)
			}
		}
	})

	t.Run("legacy data", func(t *testing.T) {
		dir := t.TempDir()
		writeMixedJSONL(t, dir, "legacy", rec, smp)
		htmlPath, err := GenerateVisualization(dir, 0)
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			t.Fatal(err)
		}
		html := string(b)
		if strings.Contains(html, `"params":`) || strings.Contains(html, `"conc":`) {
			t.Error("legacy report must not embed run params it never had")
		}
		// The report must not spend a line saying it knows nothing — the
		// params row hides itself when there is nothing to show.
		if !strings.Contains(html, `el.style.display = "none"`) {
			t.Error("legacy report should hide the params line entirely")
		}
	})
}

// TestMergeCarriesRunParams: the merged per-source JSONL leads with each arm's
// own params header, so the merged report keeps per-arm workload shape instead
// of falling back to legacy inference.
func TestMergeCarriesRunParams(t *testing.T) {
	root := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)

	writeArm := func(name string, conc, hot int) string {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		rec, smp := benchFixtureData(name, base)
		f, err := os.Create(filepath.Join(dir, "reqs.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		enc := json.NewEncoder(f)
		if err := enc.Encode(buildRunParams(AutoBenchmarkConfig{
			Concurrency: conc, HotSeriesConcurrency: hot, MaxSeries: 512, StartSeries: 512,
		}, base)); err != nil {
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
		return dir
	}

	// Two arms at DIFFERENT concurrency: the merged report must keep both,
	// since one global number would smooth one arm correctly and the other not.
	dirA := writeArm("armA", 28, 4)
	dirB := writeArm("armB", 60, 0)

	outDir := filepath.Join(root, "merged")
	htmlPath, err := GenerateVisualizationMerged([]string{dirA, dirB}, []string{"a28", "b60"}, outDir, 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Each merged per-source JSONL leads with its own header.
	for name, wantConc := range map[string]int{"a28": 28, "b60": 60} {
		_, _, params, hasParams, err := readJSONLFileWithParams(filepath.Join(outDir, name+".jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		if !hasParams {
			t.Errorf("%s: merged file lost its params header", name)
			continue
		}
		if params.Concurrency != wantConc {
			t.Errorf("%s: concurrency = %d, want %d", name, params.Concurrency, wantConc)
		}
	}

	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	if !strings.Contains(html, `"conc":32`) || !strings.Contains(html, `"conc":60`) {
		t.Error("merged report must carry BOTH arms' concurrency, not one global value")
	}
}

// TestRatioTintSpecificity guards a regression the JS tests structurally cannot
// see: the up/down tint is applied as a CLASS, so a node DOM stub happily
// reports it present while the rendered page shows grey. Moving the ratio onto
// its own line introduced `#summaryTable.has-ratios .sum-ratio { color: ... }`
// (1 id + 2 classes), which out-specified the bare `.sum-ratio.up` (2 classes)
// and silently killed every tint.
//
// So assert on the stylesheet: both tint rules must carry the same
// id-qualified prefix as the rule that sets the default colour.
func TestRatioTintSpecificity(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, _ := benchFixtureData("tint", base)
	writeMixedJSONL(t, dir, "tint", rec, nil)
	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	css := string(b)
	for _, want := range []string{
		"#summaryTable.has-ratios .sum-ratio.up { color: #6BE0A0; }",
		"#summaryTable.has-ratios .sum-ratio.down { color: #FF8569; }",
	} {
		if !strings.Contains(css, want) {
			t.Errorf("tint rule missing or under-specified; want exactly:\n  %s", want)
		}
	}
	// A bare `.sum-ratio.up { ... }` would lose to the .has-ratios default.
	for _, bad := range []string{"\n  .sum-ratio.up {", "\n  .sum-ratio.down {"} {
		if strings.Contains(css, bad) {
			t.Errorf("found an under-specified tint rule %q — it will be overridden by the .has-ratios colour", strings.TrimSpace(bad))
		}
	}
}
