package serve

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPickAutoModel(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mode   string
		models []string
		want   string
	}{
		{"auto adopts a sole model", autoModelAuto, []string{"zai-org/GLM-5.2-FP8"}, "zai-org/GLM-5.2-FP8"},
		{"auto leaves a multi-model pool alone", autoModelAuto, []string{"a", "b"}, ""},
		{"auto has nothing to adopt", autoModelAuto, nil, ""},
		{"force takes the first of many", autoModelForce, []string{"a", "b"}, "a"},
		{"force has nothing to take", autoModelForce, nil, ""},
		{"off never picks", autoModelOff, []string{"only"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := pickAutoModel(tc.mode, tc.models); got != tc.want {
				t.Errorf("pickAutoModel(%q, %v) = %q, want %q", tc.mode, tc.models, got, tc.want)
			}
		})
	}
}

// A backend's base path has to reach the listing, and it composes by the same
// rule as everything else the router asks a backend: the base stands in for the
// version segment.
func TestListModelsUsesTheBackendBasePath(t *testing.T) {
	for _, tc := range []struct{ suffix, wantPath string }{
		{"", "/v1/models"},
		{"/v1beta/openai", "/v1beta/openai/models"},
		{"/v1", "/v1/models"},
	} {
		t.Run(tc.suffix, func(t *testing.T) {
			var got string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got = r.URL.Path
				_, _ = w.Write([]byte(`{"data":[{"id":"m"}]}`))
			}))
			defer srv.Close()

			models, err := listModels(context.Background(), srv.URL+tc.suffix)
			if err != nil {
				t.Fatalf("listModels: %v", err)
			}
			if got != tc.wantPath {
				t.Errorf("listed at %q, want %q", got, tc.wantPath)
			}
			if len(models) != 1 || models[0] != "m" {
				t.Errorf("parsed %v, want [m]", models)
			}
		})
	}
}

func TestIsRedundantV1(t *testing.T) {
	for _, tc := range []struct {
		url  string
		want bool
	}{
		{"http://vllm:8000/v1", true},
		{"http://vllm:8000/v1/", true},
		{"http://vllm:8000", false},
		// A sub-mounted API base really does end in a version segment, and it
		// composes correctly — warning about it would be noise.
		{"http://host/inference/v1", false},
		{"https://generativelanguage.googleapis.com/v1beta/openai", false},
	} {
		if got := isRedundantV1(tc.url); got != tc.want {
			t.Errorf("isRedundantV1(%q) = %v, want %v", tc.url, got, tc.want)
		}
	}
}
