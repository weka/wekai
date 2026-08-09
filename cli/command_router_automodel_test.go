package cli

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestPickAutoModel(t *testing.T) {
	cases := []struct {
		name   string
		mode   string
		models []string
		want   string
	}{
		{"auto single", autoModelAuto, []string{"zai-org/GLM-5.2-FP8"}, "zai-org/GLM-5.2-FP8"},
		{"auto multi leaves alone", autoModelAuto, []string{"a", "b"}, ""},
		{"auto none", autoModelAuto, nil, ""},
		{"force takes first", autoModelForce, []string{"a", "b"}, "a"},
		{"force none", autoModelForce, nil, ""},
		{"off never picks", autoModelOff, []string{"only"}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := pickAutoModel(tc.mode, tc.models); got != tc.want {
				t.Errorf("pickAutoModel(%q, %v) = %q, want %q", tc.mode, tc.models, got, tc.want)
			}
		})
	}
}

func TestDiscoverUpstreamModels(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		want    []string
		wantErr bool
	}{
		{
			// vLLM / llama.cpp / SGLang shape.
			name:   "openai listing",
			status: http.StatusOK,
			body:   `{"object":"list","data":[{"id":"zai-org/GLM-5.2-FP8","object":"model","owned_by":"vllm"}]}`,
			want:   []string{"zai-org/GLM-5.2-FP8"},
		},
		{
			// Anthropic's own listing shares data[].id, so one parse covers both.
			name:   "anthropic listing",
			status: http.StatusOK,
			body:   `{"data":[{"type":"model","id":"claude-opus-4-8"},{"type":"model","id":"claude-haiku-4-5"}],"has_more":false}`,
			want:   []string{"claude-opus-4-8", "claude-haiku-4-5"},
		},
		{"unauthorized", http.StatusUnauthorized, `{"error":"nope"}`, nil, true},
		{"not found", http.StatusNotFound, `{"detail":"Not Found"}`, nil, true},
		{"empty listing", http.StatusOK, `{"object":"list","data":[]}`, nil, true},
		{"garbage body", http.StatusOK, `not json`, nil, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			u, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatalf("parse test server URL: %v", err)
			}
			got, err := discoverUpstreamModels(context.Background(), u)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got models %v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("discoverUpstreamModels: %v", err)
			}
			if gotPath != "/v1/models" {
				t.Errorf("probed %q, want /v1/models", gotPath)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			}
		})
	}
}

// An upstream mounted under a base path must be probed under that base path,
// mirroring how request paths are appended when forwarding.
func TestDiscoverUpstreamModelsBasePath(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"data":[{"id":"m"}]}`))
	}))
	defer srv.Close()

	u, err := url.Parse(srv.URL + "/inference")
	if err != nil {
		t.Fatalf("parse test server URL: %v", err)
	}
	if _, err := discoverUpstreamModels(context.Background(), u); err != nil {
		t.Fatalf("discoverUpstreamModels: %v", err)
	}
	if gotPath != "/inference/v1/models" {
		t.Errorf("probed %q, want /inference/v1/models", gotPath)
	}
}

func TestResolveAutoModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"id":"local-model"}]}`))
	}))
	defer srv.Close()

	newRule := func(rewrite string) *routeRule {
		u, err := url.Parse(srv.URL)
		if err != nil {
			t.Fatalf("parse test server URL: %v", err)
		}
		return &routeRule{endpoints: []string{u.String()}, rewriteModel: rewrite}
	}

	t.Run("adopts upstream model", func(t *testing.T) {
		r := newRule("")
		resolveAutoModels([]*routeRule{r}, autoModelAuto)
		if got := r.effectiveRewrite(); got != "local-model" {
			t.Errorf("effectiveRewrite() = %q, want local-model", got)
		}
	})

	t.Run("explicit as wins", func(t *testing.T) {
		r := newRule("pinned")
		resolveAutoModels([]*routeRule{r}, autoModelAuto)
		if got := r.effectiveRewrite(); got != "pinned" {
			t.Errorf("effectiveRewrite() = %q, want pinned", got)
		}
		if r.autoModel.Load() != nil {
			t.Error("explicit rule should not be probed at all")
		}
	})

	t.Run("off disables discovery", func(t *testing.T) {
		r := newRule("")
		resolveAutoModels([]*routeRule{r}, autoModelOff)
		if got := r.effectiveRewrite(); got != "" {
			t.Errorf("effectiveRewrite() = %q, want empty", got)
		}
	})
}
