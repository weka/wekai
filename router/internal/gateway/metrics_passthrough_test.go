package gateway_test

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/weka/wekai/router/internal/dialect/openai"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

// The bug, as reported from an 8-instance b300 pool:
//
//	"wekai-router does not aggregate upstream vLLM counters — it round-robin
//	 proxies a single backend per scrape. A router scrape was byte-identical to
//	 instance i7 on all three sources simultaneously; the preceding scrape
//	 tracked i0. Consecutive probes jitter across the single-backend range
//	 (78M / 100M / 116M) while the true 8-way sum was 868M — i.e. router ≈ 1/N.
//	 Note: :29000 is unaffected — it carries only router-native router_* metrics,
//	 which look correct."
//
// That last sentence is the whole diagnosis. The aggregator on the metrics
// listener was never involved: the scrapes were going to the INFERENCE
// listener, where /metrics is not a route the dialect claims, so it fell
// through to the passthrough tier and was PROXIED to a backend — chosen by
// least-outstanding, hence a different one per scrape. Every symptom follows:
// one instance's numbers, byte-identical, jittering between instances, ~1/N of
// the fleet, and going backwards whenever consecutive scrapes landed on
// different backends.
//
// These tests use mock backends serving distinct counters, so "which backend
// answered" is decidable from the body.

// vllmMetricsBackend serves a /metrics body carrying one identifiable value,
// and counts how often it was asked.
type vllmMetricsBackend struct {
	srv  *httptest.Server
	name string

	mu   sync.Mutex
	hits int
}

func newMetricsBackend(t *testing.T, name string, value float64) *vllmMetricsBackend {
	t.Helper()
	b := &vllmMetricsBackend{name: name}
	b.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			b.mu.Lock()
			b.hits++
			b.mu.Unlock()
			fmt.Fprintf(w, "# TYPE vllm:prompt_tokens_by_source_total counter\n"+
				"vllm:prompt_tokens_by_source_total{instance=%q,source=\"local_cache_hit\"} %g\n",
				b.name, value)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","choices":[]}`))
	}))
	t.Cleanup(b.srv.Close)
	return b
}

func (b *vllmMetricsBackend) scrapes() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.hits
}

// inferenceOnly builds a router's INFERENCE listener over the given backends —
// the port a client posts completions to, and the port the reporter was
// scraping.
func inferenceOnly(t *testing.T, backends []*vllmMetricsBackend) *httptest.Server {
	t.Helper()
	reg := registry.New(registry.Options{})
	for _, b := range backends {
		be, err := reg.Add(registry.Spec{URL: b.srv.URL, Prov: registry.ProvStatic, Capacity: 8})
		if err != nil {
			t.Fatal(err)
		}
		be.SetHealth(registry.Healthy)
	}
	d := openai.New()
	px := proxy.New(proxy.Config{MaxAttempts: 2, StreamBufferBytes: 64 << 10})
	gw := gateway.New(gateway.Config{MaxBodyBytes: 64 << 20, DefaultCapacity: 1},
		mustTable(t, reg, mustFlow(t, affinity.Config{})), px, d)
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	return srv
}

// TestMetricsIsNotProxiedFromTheServingPort is the regression test. Scraping
// the inference listener must never return a backend's counters, because a
// single backend's counters look exactly like a fleet total that is wrong by a
// factor of N — and wrong in a way that moves backwards between scrapes, which
// silently breaks rate() and increase().
func TestMetricsIsNotProxiedFromTheServingPort(t *testing.T) {
	fleet := []*vllmMetricsBackend{
		newMetricsBackend(t, "i0", 100),
		newMetricsBackend(t, "i1", 200),
		newMetricsBackend(t, "i7", 800),
	}
	srv := inferenceOnly(t, fleet)

	for i := range 12 {
		resp, err := http.Get(srv.URL + "/metrics")
		if err != nil {
			t.Fatalf("scrape %d: %v", i, err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if strings.Contains(string(body), "vllm:prompt_tokens_by_source_total") {
			t.Fatalf("scrape %d returned a BACKEND's counters from the inference listener:\n%s\n"+
				"One backend's numbers are indistinguishable from a fleet total that is wrong by "+
				"a factor of the fleet size, and they move backwards whenever consecutive scrapes "+
				"land on different backends.", i, body)
		}
		if resp.StatusCode == http.StatusOK {
			t.Fatalf("scrape %d answered 200 on the inference listener; metrics live on "+
				"--metrics-listen and this path must say so rather than serve something", i)
		}
	}

	for _, b := range fleet {
		if n := b.scrapes(); n != 0 {
			t.Errorf("backend %s was scraped %d times by a client asking the ROUTER for metrics; "+
				"the router must not forward /metrics to a backend at all", b.name, n)
		}
	}
}

// TestMetricsRefusalSaysWhereToLook. The reporter's workaround was to scrape
// all N instances directly and sum client-side — reached only after paired
// snapshots and a negative delta proved the numbers wrong. The response should
// have told them in one request.
func TestMetricsRefusalSaysWhereToLook(t *testing.T) {
	srv := inferenceOnly(t, []*vllmMetricsBackend{newMetricsBackend(t, "i0", 1)})
	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "metrics-listen") {
		t.Errorf("the refusal does not name --metrics-listen, so a scraper getting it has no way "+
			"to know where the aggregated counters actually are:\n%s", body)
	}
}

// TestPassthroughStillWorksForEverythingElse: the passthrough tier is what lets
// one router front a hosted API on paths this dialect never claims, and it must
// keep doing that. Only /metrics is special.
func TestPassthroughStillWorksForEverythingElse(t *testing.T) {
	b := newMetricsBackend(t, "i0", 1)
	srv := inferenceOnly(t, []*vllmMetricsBackend{b})

	resp, err := http.Post(srv.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"m","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("passthrough to an unclaimed path answered %d, want 200: blocking /metrics must "+
			"not have blocked the tier it lives in", resp.StatusCode)
	}
}
