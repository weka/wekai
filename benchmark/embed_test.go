package benchmark

import (
	"testing"
)

func TestGetModelDisplayName(t *testing.T) {
	tests := []struct {
		name     string
		modelStr string
		want     string
	}{
		{
			name:     "dynamic model with alias",
			modelStr: "dynamic/http://localhost:8000/v1,type=openai,max_tokens=1024,model=glm-4.5-air,alias=amg_no_chunk_param",
			want:     "amg_no_chunk_param",
		},
		{
			name:     "dynamic model without alias",
			modelStr: "dynamic/http://localhost:8000/v1,type=openai,max_tokens=1024,model=glm-4.5-air",
			want:     "dynamic/http://localhost:8000/v1,type=openai,max_tokens=1024,model=glm-4.5-air",
		},
		{
			name:     "static model",
			modelStr: "openai/gpt-4o-mini",
			want:     "openai/gpt-4o-mini",
		},
		{
			name:     "static model with params",
			modelStr: "zai/glm-4.6,max_tokens=2048",
			want:     "zai/glm-4.6,max_tokens=2048",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getModelDisplayName(tt.modelStr)
			if got != tt.want {
				t.Errorf("getModelDisplayName(%q) = %q, want %q", tt.modelStr, got, tt.want)
			}
		})
	}
}
