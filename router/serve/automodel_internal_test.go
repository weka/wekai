package serve

import (
	"context"
	"fmt"
	"io"
	"log/slog"
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

			models, err := listModels(context.Background(), srv.URL+tc.suffix, "")
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

// A pool the router authenticates to itself must be probed with THAT key. Left
// unauthenticated the listing is a 401, discovery gives up, and the route's model
// reaches a vLLM that answers 404 for a name it does not serve.
func TestListModelsPresentsThePoolCredential(t *testing.T) {
	var auth, xapi string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, xapi = r.Header.Get("Authorization"), r.Header.Get("X-Api-Key")
		if auth != "Bearer inner-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"Qwen/Qwen3-32B"}]}`))
	}))
	defer srv.Close()

	models, err := listModels(context.Background(), srv.URL, "inner-key")
	if err != nil {
		t.Fatalf("listModels: %v", err)
	}
	if auth != "Bearer inner-key" {
		t.Errorf("Authorization %q, want the pool's credential", auth)
	}
	if xapi != "inner-key" {
		t.Errorf("X-Api-Key %q; both styles are sent because a pool is configured "+
			"by URL, not by which convention it speaks", xapi)
	}
	if len(models) != 1 || models[0] != "Qwen/Qwen3-32B" {
		t.Errorf("parsed %v", models)
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

// probeIsEngine accepts either engine by its own metric family, and falls back
// to the model listing for anything else that serves one.
func TestProbeIsEngineAcceptsSGLangAndVLLM(t *testing.T) {
	mk := func(metrics string, models bool) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.URL.Path == "/metrics" && metrics != "":
				fmt.Fprint(w, metrics)
			case r.URL.Path == "/v1/models" && models:
				fmt.Fprint(w, `{"data":[]}`)
			default:
				http.NotFound(w, r)
			}
		}))
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	cases := []struct {
		name string
		srv  *httptest.Server
		want bool
	}{
		{"vllm metrics", mk("vllm:num_requests_running 0\n", false), true},
		{"sglang metrics", mk("sglang:cache_hit_rate{model_name=\"m\"} 0.1\n", false), true},
		{"sglang metrics and models", mk("sglang:num_running_reqs 0\n", true), true},
		{"models only", mk("", true), true},
		{"neither", mk("", false), false},
	}
	for _, c := range cases {
		defer c.srv.Close()
		if got := probeIsEngine(context.Background(), c.srv.URL, log); got != c.want {
			t.Errorf("%s: probeIsEngine = %v, want %v", c.name, got, c.want)
		}
	}
}
