package benchmark

import (
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
