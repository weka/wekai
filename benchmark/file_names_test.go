package benchmark

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// theFailingSpec is the model spec from the six-endpoint failover run whose
// request data was lost: os.Create returned ENAMETOOLONG because the sanitized
// name was 260 bytes. The t=1.5 variant of the same run wrote fine — its alias
// was four characters shorter.
const theFailingSpec = "dynamic/http://10.20.48.111:8000,http://10.20.54.163:8000," +
	"http://10.20.54.72:8000,http://10.20.56.32:8000,http://10.20.56.84:8000," +
	"http://10.20.62.64:8000,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4," +
	"alias=awsp6b200-weka-v23-scale6-failover-t12"

// TestSafeFileBaseFitsNameMax: no identifier, however long, may produce a name
// the filesystem rejects.
func TestSafeFileBaseFitsNameMax(t *testing.T) {
	cases := []struct{ name, id string }{
		{"the failing six-endpoint spec", theFailingSpec},
		{"its shorter sibling that worked", strings.Replace(theFailingSpec, "-t12", "", 1)},
		{"absurdly long alias", "dynamic/http://h:8000,type=openai_vllm,alias=" + strings.Repeat("x", 900)},
		{"long spec with no alias at all", "dynamic/" + strings.Repeat("http://10.20.48.111:8000,", 40)},
		{"already short", "anthropic/claude-sonnet-5"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := safeFileBase(tc.id, len(".jsonl"))
			if n := len(got) + len(".jsonl"); n > maxFileNameLen {
				t.Errorf("name is %d bytes, over the %d-byte limit: %q", n, maxFileNameLen, got)
			}
			if got == "" {
				t.Error("empty base name")
			}
			if strings.ContainsAny(got, "/\x00") {
				t.Errorf("name is not a single path component: %q", got)
			}
		})
	}
}

// TestSafeFileBaseKeepsShortNames: names that already fit are untouched, so
// existing runs keep the filenames their reports and scripts refer to.
func TestSafeFileBaseKeepsShortNames(t *testing.T) {
	id := "dynamic/http://localhost:8000,type=openai_vllm,alias=SOAK8H512_pod0"
	want := sanitizeModelRe.ReplaceAllString(id, "_")
	if got := safeFileBase(id, len(".jsonl")); got != want {
		t.Errorf("short name was rewritten:\n got %q\nwant %q", got, want)
	}
}

// TestSafeFileBaseKeepsAlias: when shortening IS needed, the alias survives —
// it is how a human named the run and how every reader identifies the arm.
// Dropping it in favour of the endpoint list would leave a directory of files
// distinguishable only by hash.
func TestSafeFileBaseKeepsAlias(t *testing.T) {
	got := safeFileBase(theFailingSpec, len(".jsonl"))
	if !strings.Contains(got, "awsp6b200-weka-v23-scale6-failover-t12") {
		t.Errorf("alias lost when shortening: %q", got)
	}
	if strings.Contains(got, "10.20.48.111") {
		t.Errorf("endpoint list should have been dropped, not the alias: %q", got)
	}
}

// TestSafeFileBaseDisambiguates: two specs that differ only PAST the truncation
// point must not collide — silently overwriting one arm's data with another's
// would be worse than the crash this replaces.
func TestSafeFileBaseDisambiguates(t *testing.T) {
	long := strings.Repeat("x", 400)
	a := safeFileBase("dynamic/"+long+"aaa,type=openai_vllm", len(".jsonl"))
	b := safeFileBase("dynamic/"+long+"bbb,type=openai_vllm", len(".jsonl"))
	if a == b {
		t.Errorf("distinct specs collided on one file name: %q", a)
	}

	// Same for two aliases sharing a long prefix.
	pre := strings.Repeat("y", 300)
	c := safeFileBase("dynamic/h,alias="+pre+"-arm1", len(".jsonl"))
	d := safeFileBase("dynamic/h,alias="+pre+"-arm2", len(".jsonl"))
	if c == d {
		t.Errorf("distinct aliases collided on one file name: %q", c)
	}
}

// TestSafeFileBaseIsStable: the same spec must map to the same name on every
// run, or a resumed/re-run benchmark would scatter across files.
func TestSafeFileBaseIsStable(t *testing.T) {
	for i := 0; i < 3; i++ {
		if got := safeFileBase(theFailingSpec, len(".jsonl")); got != safeFileBase(theFailingSpec, len(".jsonl")) {
			t.Fatal("safeFileBase is not deterministic")
		}
	}
}

// TestRequestDataWriterCreatesLongNamedFile is the end-to-end regression: the
// writer that failed on this spec must now create, write and reopen its file.
func TestRequestDataWriterCreatesLongNamedFile(t *testing.T) {
	dir := t.TempDir()
	w, err := newRequestDataWriter(dir, theFailingSpec, time.Time{})
	if err != nil {
		t.Fatalf("writer creation failed for a 260-byte spec: %v", err)
	}
	if err := w.write(requestDataRecord{Model: theFailingSpec, SeriesNum: 1, RequestNum: 1, InputTokens: 7}); err != nil {
		t.Fatal(err)
	}
	if err := w.close(); err != nil {
		t.Fatal(err)
	}

	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("expected exactly one JSONL, got %v (%v)", files, err)
	}
	if n := len(filepath.Base(files[0])); n > maxFileNameLen {
		t.Errorf("wrote a %d-byte file name, over the limit", n)
	}
	records, _, err := readJSONLFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || records[0].InputTokens != 7 {
		t.Errorf("round trip lost the record: %+v", records)
	}
	// A reader still recovers the arm's identity from the record, not the name.
	if got := extractAlias(records[0].Model); got != "awsp6b200-weka-v23-scale6-failover-t12" {
		t.Errorf("alias not recoverable from records: %q", got)
	}

	if _, err := os.Stat(files[0]); err != nil {
		t.Fatal(err)
	}
}

// TestSaveRequestDataFailsFast: --save-request-data means the data IS the
// deliverable, so an unwritable destination must abort the run immediately
// rather than warn and proceed. The previous behaviour printed one line and
// ran anyway; a 12-hour benchmark finished having written nothing, and the
// loss was only discovered afterwards.
func TestSaveRequestDataFailsFast(t *testing.T) {
	dir := t.TempDir()
	// A regular file where the output directory should be: MkdirAll fails, the
	// same class of failure as a read-only mount or a bad path.
	notADir := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(notADir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A deliberately unreachable endpoint: if the guard did NOT fire, the run
	// would proceed to issue requests and fail differently (or hang), so
	// returning promptly with this error is itself the assertion.
	cfg := AutoBenchmarkConfig{
		Model:              "dynamic/http://127.0.0.1:1/v1,type=openai_vllm,alias=failfast",
		SaveRequestDataDir: notADir,
		Timeout:            time.Hour,
		Concurrency:        1,
		MaxSeries:          1,
		StartSeries:        1,
		Total:              1,
	}

	start := time.Now()
	err := RunAutoBenchmark(context.Background(), cfg)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("run succeeded despite being unable to save request data")
	}
	if !strings.Contains(err.Error(), "save-request-data") {
		t.Errorf("error should name the flag that was honoured; got: %v", err)
	}
	// The point is that it aborts BEFORE any benchmarking work.
	if elapsed > 10*time.Second {
		t.Errorf("took %s to fail — the guard should fire before any work starts", elapsed)
	}
}

// TestNoSaveRequestDataDirSkipsWriters: without the flag there is nothing to
// open, and the fail-fast guard must not invent a reason to abort.
func TestNoSaveRequestDataDirSkipsWriters(t *testing.T) {
	// Model resolution happens before the writer guard, so an empty model is a
	// cheap way to prove the guard did not fire first.
	err := RunAutoBenchmark(context.Background(), AutoBenchmarkConfig{})
	if err == nil || !strings.Contains(err.Error(), "no model specified") {
		t.Errorf("want the pre-existing no-model error, got: %v", err)
	}
}

// TestRequestDataLandsInRunDir pins WHERE the JSONL is written. Opening the
// writers before the per-run subdirectory was created left every file at the
// --save-request-data root while the run directory the log announced stayed
// empty — and the auto-generated report reads that same (empty) directory, so
// a run produced neither data in the announced place nor a report.
func TestRequestDataLandsInRunDir(t *testing.T) {
	base := t.TempDir()
	cfg := AutoBenchmarkConfig{
		// Unreachable on purpose: this test is about file placement, which is
		// decided before any request is issued.
		Model:              "dynamic/http://127.0.0.1:1/v1,type=openai_vllm,alias=rundir",
		SaveRequestDataDir: base,
		Timeout:            2 * time.Second,
		RequestTimeout:     time.Second,
		Concurrency:        1,
		MaxSeries:          1,
		StartSeries:        1,
		Total:              1,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = RunAutoBenchmark(ctx, cfg) // the run itself is expected to get nowhere

	// Nothing may be written at the base: that is the bug this guards.
	if stray, _ := filepath.Glob(filepath.Join(base, "*.jsonl")); len(stray) > 0 {
		t.Errorf("JSONL written to the --save-request-data root instead of the run dir: %v", stray)
	}

	entries, err := os.ReadDir(base)
	if err != nil {
		t.Fatal(err)
	}
	var runDirs []string
	for _, e := range entries {
		if e.IsDir() {
			runDirs = append(runDirs, e.Name())
		}
	}
	if len(runDirs) != 1 {
		t.Fatalf("want exactly one timestamped run dir, got %v", runDirs)
	}
	if !autoRunDirRe.MatchString(runDirs[0]) {
		t.Errorf("run dir %q is not the expected timestamp form", runDirs[0])
	}
	// The JSONL belongs inside it. It exists even for a run that never got a
	// response, because the writer is opened (and its run_params header
	// written) before any request is issued.
	inside, err := filepath.Glob(filepath.Join(base, runDirs[0], "*.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inside) != 1 {
		t.Fatalf("want one JSONL inside the run dir, got %v", inside)
	}
	if !strings.Contains(filepath.Base(inside[0]), "rundir") {
		t.Errorf("file should still be identifiable by alias: %q", filepath.Base(inside[0]))
	}
}
