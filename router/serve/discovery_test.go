package serve_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/serve"
)

// A vLLM-style backend answers GET /v1/models, and is probed actively from
// then on. Anything else falls back to passive health — served, with health
// inferred from real traffic rather than from probes it would always fail.
func TestVLLMIsDiscoveredAndOthersFallBackToPassive(t *testing.T) {
	var probes int
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			probes++
			// A vLLM instance is identified by its own metric names, whatever
			// request format it is fronted with.
			fmt.Fprint(w, "# HELP vllm:num_requests_running\nvllm:num_requests_running 3\n")
			return
		}
		if r.URL.Path == "/v1/models" {
			fmt.Fprint(w, `{"object":"list","data":[{"id":"m"}]}`)
			return
		}
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer vllm.Close()

	// A hosted API: no /v1/models, no /health. Probing it forever would be the
	// benchmark sampler's retry-forever bug, one layer down.
	var hostedProbes int
	hosted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" || r.URL.Path == "/v1/models" || r.URL.Path == "/health" {
			hostedProbes++
			w.WriteHeader(http.StatusNotFound)
			return
		}
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer hosted.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes: []serve.Route{
			{Patterns: "local", Endpoints: []string{vllm.URL}},
			{Patterns: "*", Endpoints: []string{hosted.URL}},
		},
		HealthInterval: 50 * time.Millisecond,
		HealthTimeout:  20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	srv := httptest.NewServer(h)
	defer srv.Close()

	// The hosted endpoint must serve immediately: passive health means eligible
	// from the start, with no probe to wait for.
	resp, err := srv.Client().Post(srv.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"claude","messages":[]}`))
	if err != nil {
		t.Fatalf("hosted request: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("hosted endpoint status %d; a non-vLLM upstream must still be "+
			"served, not held unhealthy waiting for a probe it cannot pass", resp.StatusCode)
	}

	// The discovery probe runs ONCE per endpoint. A retry loop against an
	// endpoint that will never answer is the shape being avoided.
	before := hostedProbes
	time.Sleep(200 * time.Millisecond)
	if hostedProbes > before {
		t.Errorf("hosted endpoint was probed %d more times after discovery; "+
			"a failed probe must be latched, not retried", hostedProbes-before)
	}
	if probes != 1 {
		t.Errorf("vLLM /metrics was probed %d times, want exactly 1", probes)
	}
}
