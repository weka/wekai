package serve_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/weka/wekai/router/serve"
)

// A provider whose API does not live at the root can only be reached if the
// prefix comes from configuration: Google's OpenAI-compatible surface is at
// /v1beta/openai/chat/completions, and a request sent to /v1/chat/completions
// instead lands on Google's native API, which speaks a different protocol.
//
// The base has to stand in for the client's version segment, not stack on top
// of it, which is also what makes the redundant ".../v1" people write out of
// base_url habit compose back to the very same URL.
func TestBackendBasePathComposesWithClientPath(t *testing.T) {
	for _, tc := range []struct{ name, suffix, want string }{
		{"root-hosted backend is untouched", "", "/v1/chat/completions"},
		{"compat surface keeps one version segment", "/v1beta/openai", "/v1beta/openai/chat/completions"},
		{"redundant /v1 composes back to the same path", "/v1", "/v1/chat/completions"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var seen []string
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				seen = append(seen, r.URL.Path)
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
			}))
			defer up.Close()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			h, err := serve.Handler(ctx, serve.Options{
				Routes:         []serve.Route{{Patterns: "*", Endpoints: []string{up.URL + tc.suffix}, Passive: true}},
				HealthInterval: time.Hour,
			})
			if err != nil {
				t.Fatalf("Handler: %v", err)
			}
			rt := httptest.NewServer(h)
			defer rt.Close()

			if code := chat(t, rt); code != http.StatusOK {
				t.Fatalf("chat returned %d, want 200", code)
			}

			mu.Lock()
			defer mu.Unlock()
			var got string
			for _, p := range seen {
				if strings.HasSuffix(p, "/chat/completions") {
					got = p
				}
			}
			if got != tc.want {
				t.Errorf("backend received %q, want %q; a base path in the configuration has "+
					"to compose with the client's own path, or the request reaches the wrong API",
					got, tc.want)
			}
		})
	}
}

// Auto-model discovery used to probe the FIRST endpoint in a route's list at
// startup, before the health checker had an opinion about any of them. That was
// wrong twice: a pool whose first backend is dead never resolved even though
// another backend could answer, and the answer it did get arrived too late to
// matter, because the rewrite was snapshotted when the routing table was built.
//
// It now asks the POOL, once a backend is healthy enough to route to, and the
// answer takes effect when it lands.
func TestAutoModelAsksAHealthyBackendNotTheFirstOne(t *testing.T) {
	// First endpoint: dead. Nothing here can answer anything.
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer dead.Close()

	// Second endpoint: healthy, and serves exactly one model.
	var mu sync.Mutex
	var gotModel string
	live := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			return
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"Qwen/Qwen3-32B"}]}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		mu.Lock()
		gotModel = req.Model
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	}))
	defer live.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes:          []serve.Route{{Patterns: "*", Endpoints: []string{dead.URL, live.URL}}},
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 20 * time.Millisecond,
		HealthTimeout:   200 * time.Millisecond,
		MaxAttempts:     3,
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	// The client asks for a name the backend does not serve; discovery has to
	// replace it with the one the pool actually loaded.
	eventually(t, func() bool {
		if chat(t, rt) != http.StatusOK {
			return false
		}
		mu.Lock()
		defer mu.Unlock()
		return gotModel == "Qwen/Qwen3-32B"
	})

	mu.Lock()
	defer mu.Unlock()
	if gotModel != "Qwen/Qwen3-32B" {
		t.Errorf("backend received model %q, want the discovered %q; discovery either "+
			"never got past the dead first endpoint or landed too late to take effect",
			gotModel, "Qwen/Qwen3-32B")
	}
}
