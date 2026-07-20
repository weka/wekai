package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeRecordJSONL writes records as a JSONL file at dir/name.jsonl, creating
// dir if needed. Used to set up fixture source directories for
// GenerateVisualizationMerged integration tests below.
func writeRecordJSONL(t *testing.T, dir, name string, records []requestDataRecord) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	f, err := os.Create(filepath.Join(dir, name+".jsonl"))
	if err != nil {
		t.Fatalf("create jsonl: %v", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatalf("encode record: %v", err)
		}
	}
}

// TestGenerateVisualizationMergedLabelPriority locks in the exact priority
// chain requested for visualize-merge source labeling:
// --labels override > model alias > parent run-dir name > timestamp (last
// resort). This is the scenario reported in practice: two `benchmark auto
// --save-request-data <dir>` runs of the SAME model id but distinct
// alias=... arms, each landing in an opaque auto-generated timestamp
// subdirectory that a user naturally points visualize-merge at.
func TestGenerateVisualizationMergedLabelPriority(t *testing.T) {
	root := t.TempDir()

	t.Run("alias wins over opaque timestamp source dirs", func(t *testing.T) {
		wekaDir := filepath.Join(root, "alias", "DS3H_weka-64r8w_reqdata", "2026-07-19T12-00-00Z")
		hbmDir := filepath.Join(root, "alias", "DS3H_hbm_reqdata", "2026-07-19T12-05-00Z")
		writeRecordJSONL(t, wekaDir, "reqs", []requestDataRecord{
			{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4,alias=DS3H_weka-64r8w"},
		})
		writeRecordJSONL(t, hbmDir, "reqs", []requestDataRecord{
			{Model: "dynamic/http://localhost:8001/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4,alias=DS3H_hbm"},
		})

		out := filepath.Join(root, "alias", "merged")
		if _, err := GenerateVisualizationMerged([]string{wekaDir, hbmDir}, nil, out, 0); err != nil {
			t.Fatalf("GenerateVisualizationMerged: %v", err)
		}
		assertExists(t, filepath.Join(out, "DS3H_weka-64r8w.jsonl"))
		assertExists(t, filepath.Join(out, "DS3H_hbm.jsonl"))
		assertNotExists(t, filepath.Join(out, "2026-07-19T12-00-00Z.jsonl"))
		assertNotExists(t, filepath.Join(out, "2026-07-19T12-05-00Z.jsonl"))
	})

	t.Run("labels override wins even when an alias is present", func(t *testing.T) {
		wekaDir := filepath.Join(root, "override", "DS3H_weka-64r8w_reqdata", "2026-07-19T12-00-00Z")
		hbmDir := filepath.Join(root, "override", "DS3H_hbm_reqdata", "2026-07-19T12-05-00Z")
		writeRecordJSONL(t, wekaDir, "reqs", []requestDataRecord{
			{Model: "dynamic/http://localhost:8000/v1,alias=DS3H_weka-64r8w"},
		})
		writeRecordJSONL(t, hbmDir, "reqs", []requestDataRecord{
			{Model: "dynamic/http://localhost:8001/v1,alias=DS3H_hbm"},
		})

		out := filepath.Join(root, "override", "merged")
		if _, err := GenerateVisualizationMerged([]string{wekaDir, hbmDir}, []string{"custom-a", "custom-b"}, out, 0); err != nil {
			t.Fatalf("GenerateVisualizationMerged: %v", err)
		}
		assertExists(t, filepath.Join(out, "custom-a.jsonl"))
		assertExists(t, filepath.Join(out, "custom-b.jsonl"))
		assertNotExists(t, filepath.Join(out, "DS3H_weka-64r8w.jsonl"))
	})

	t.Run("no alias: falls back to parent run-directory name, not the timestamp leaf", func(t *testing.T) {
		dir := filepath.Join(root, "noalias", "DS3H_weka-64r8w_reqdata", "2026-07-19T12-00-00Z")
		writeRecordJSONL(t, dir, "reqs", []requestDataRecord{
			{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm"}, // no alias=
		})

		out := filepath.Join(root, "noalias", "merged")
		if _, err := GenerateVisualizationMerged([]string{dir}, nil, out, 0); err != nil {
			t.Fatalf("GenerateVisualizationMerged: %v", err)
		}
		assertExists(t, filepath.Join(out, "DS3H_weka-64r8w_reqdata.jsonl"))
		assertNotExists(t, filepath.Join(out, "2026-07-19T12-00-00Z.jsonl"))
	})

	t.Run("labels count mismatch errors", func(t *testing.T) {
		dirA := filepath.Join(root, "mismatch", "a")
		dirB := filepath.Join(root, "mismatch", "b")
		writeRecordJSONL(t, dirA, "reqs", []requestDataRecord{{Model: "dynamic/http://localhost:8000/v1,alias=a"}})
		writeRecordJSONL(t, dirB, "reqs", []requestDataRecord{{Model: "dynamic/http://localhost:8001/v1,alias=b"}})

		_, err := GenerateVisualizationMerged([]string{dirA, dirB}, []string{"only-one"}, filepath.Join(root, "mismatch", "merged"), 0)
		if err == nil {
			t.Fatal("expected error on --labels count mismatch, got nil")
		}
	})
}

func assertExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected %s to exist: %v", path, err)
	}
}

func assertNotExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err == nil {
		t.Errorf("expected %s to NOT exist, but it does", path)
	}
}

func TestResolveRecordsAlias(t *testing.T) {
	tests := []struct {
		name    string
		records []requestDataRecord
		want    string
	}{
		{
			name:    "empty",
			records: nil,
			want:    "",
		},
		{
			name: "single dynamic model with alias",
			records: []requestDataRecord{
				{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4,alias=DS3H_weka-64r8w"},
				{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4,alias=DS3H_weka-64r8w"},
			},
			want: "DS3H_weka-64r8w",
		},
		{
			// A model spec with no alias= param must yield "" here, NOT the
			// full raw spec string -- otherwise deriveSourceLabel would use
			// that ugly full string as a label instead of falling through to
			// the parent-dir-name / timestamp-last-resort chain.
			name: "dynamic model without alias yields empty, not the full spec string",
			records: []requestDataRecord{
				{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4"},
			},
			want: "",
		},
		{
			name: "two distinct aliases is ambiguous",
			records: []requestDataRecord{
				{Model: "dynamic/http://localhost:8000/v1,alias=DS3H_weka-64r8w"},
				{Model: "dynamic/http://localhost:8001/v1,alias=DS3H_hbm"},
			},
			want: "",
		},
		{
			name: "non-dynamic model with no alias= substring yields empty",
			records: []requestDataRecord{
				{Model: "gpt-4"},
				{Model: "gpt-4"},
			},
			want: "",
		},
		{
			name: "permissive fallback finds alias= even outside strict dynamic/ syntax",
			records: []requestDataRecord{
				{Model: "some-future-spec-format,alias=weird-but-valid"},
			},
			want: "weird-but-valid",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveRecordsAlias(tt.records)
			if got != tt.want {
				t.Errorf("resolveRecordsAlias() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDeriveSourceLabel(t *testing.T) {
	wekaRecords := []requestDataRecord{
		{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4,alias=DS3H_weka-64r8w"},
	}
	hbmRecords := []requestDataRecord{
		{Model: "dynamic/http://localhost:8001/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4,alias=DS3H_hbm"},
	}
	ambiguousRecords := []requestDataRecord{
		{Model: "dynamic/http://localhost:8000/v1,alias=a"},
		{Model: "dynamic/http://localhost:8001/v1,alias=b"},
	}

	tests := []struct {
		name    string
		dir     string
		records []requestDataRecord
		want    string
	}{
		{
			name:    "alias wins over opaque timestamp dir (the reported bug case)",
			dir:     "/mnt/weka/wekai-runs-logs/DS3H_weka-64r8w_reqdata/2026-07-19T12-00-00Z",
			records: wekaRecords,
			want:    "DS3H_weka-64r8w",
		},
		{
			name:    "alias wins over opaque timestamp dir, second arm",
			dir:     "/mnt/weka/wekai-runs-logs/DS3H_hbm_reqdata/2026-07-19T12-05-00Z",
			records: hbmRecords,
			want:    "DS3H_hbm",
		},
		{
			name:    "no alias, timestamp dir falls back to parent run-directory name",
			dir:     "/mnt/weka/wekai-runs-logs/DS3H_weka-64r8w_reqdata/2026-07-19T12-00-00Z",
			records: ambiguousRecords,
			want:    "DS3H_weka-64r8w_reqdata",
		},
		{
			name:    "no alias, non-timestamp dir keeps its own basename",
			dir:     "/mnt/weka/wekai-runs-logs/my-custom-dir",
			records: ambiguousRecords,
			want:    "my-custom-dir",
		},
		{
			name:    "no records at all falls back to dir basename",
			dir:     "/mnt/weka/wekai-runs-logs/my-custom-dir",
			records: nil,
			want:    "my-custom-dir",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := deriveSourceLabel(tt.dir, tt.records)
			if got != tt.want {
				t.Errorf("deriveSourceLabel(%q) = %q, want %q", tt.dir, got, tt.want)
			}
		})
	}
}

func TestAutoRunDirRe(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"2026-07-19T12-00-00Z", true},
		{"2026-07-19T12-00-00", false}, // missing trailing Z
		{"DS3H_weka-64r8w_reqdata", false},
		{"my-custom-dir", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := autoRunDirRe.MatchString(tt.name); got != tt.want {
			t.Errorf("autoRunDirRe.MatchString(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}
