package benchmark

import "testing"

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
			name: "dynamic model without alias falls back to full string, ambiguous when multiple",
			records: []requestDataRecord{
				{Model: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4"},
			},
			want: "dynamic/http://localhost:8000/v1,type=openai_vllm,model=nvidia/DeepSeek-V4-Pro-NVFP4",
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
			name: "non-dynamic model returns full string",
			records: []requestDataRecord{
				{Model: "gpt-4"},
				{Model: "gpt-4"},
			},
			want: "gpt-4",
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
