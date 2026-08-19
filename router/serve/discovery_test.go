package serve_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/weka/wekai/router/serve"
)

// A vLLM-style backend answers GET /v1/models, and is probed actively from
// then on. Anything else falls back to passive health — served, with health
// inferred from real traffic rather than from probes it would always fail.
func TestVLLMIsDiscoveredAndOthersFallBackToPassive(t *testing.T) {
	var probes atomic.Int64
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			probes.Add(1)
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
	var hostedProbes atomic.Int64
	hosted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" || r.URL.Path == "/v1/models" || r.URL.Path == "/health" {
			hostedProbes.Add(1)
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
		// Upstream aggregation is on by default and scrapes the same /metrics
		// path this counts, so leaving it on would make `probes` the sum of two
		// unrelated activities. What is under test is that DISCOVERY asks once;
		// that the aggregator asks repeatedly is correct and covered elsewhere.
		DisableVLLMMetrics: true,
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
	//
	// Discovery is asynchronous, and the request served above orders nothing
	// about it: on a slow runner the single discovery sequence can still be
	// in flight here, its remaining probes landing during any fixed sleep and
	// misreading as retries. So the assertion is stability, which subsumes
	// the old take-a-baseline-and-sleep form: a latched probe goes quiet
	// within a few health intervals and stays quiet; a retry loop fires every
	// interval and can never hold still.
	deadline := time.Now().Add(2 * time.Second)
	stableFor, last := 0, hostedProbes.Load()
	for time.Now().Before(deadline) && stableFor < 4 {
		time.Sleep(50 * time.Millisecond)
		if cur := hostedProbes.Load(); cur == last {
			stableFor++
		} else {
			stableFor, last = 0, cur
		}
	}
	if stableFor < 4 {
		t.Errorf("hosted endpoint was still being probed after 2s (%d probes and counting); "+
			"a failed probe must be latched, not retried", hostedProbes.Load())
	}
	if got := probes.Load(); got != 1 {
		t.Errorf("vLLM /metrics was probed %d times by DISCOVERY, want exactly 1", got)
	}
}
