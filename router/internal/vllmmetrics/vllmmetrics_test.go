package vllmmetrics

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func body(v float64) string {
	return fmt.Sprintf("# HELP vllm:prompt_tokens_by_source_total x\n"+
		"# TYPE vllm:prompt_tokens_by_source_total counter\n"+
		"vllm:prompt_tokens_by_source_total{source=\"cache\"} %g\n", v)
}

func total(t *testing.T, a *Aggregator) float64 {
	t.Helper()
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, v := range a.totals {
		return v
	}
	return 0
}

// The whole point of accumulating deltas: an upstream restart must not make the
// router's total go backwards. A decreasing counter breaks rate() and
// increase() silently, producing plausible graphs that are wrong.
func TestUpstreamRestartDoesNotRewindTheTotal(t *testing.T) {
	var value float64 = 100
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body(value))
	}))
	defer srv.Close()

	a := New(Config{})
	ctx := context.Background()

	// First sighting counts in full, so a pod already serving before the router
	// started still contributes its history.
	a.ScrapeAll(ctx, []string{srv.URL})
	if got := total(t, a); got != 100 {
		t.Fatalf("after first scrape total = %v, want 100", got)
	}

	value = 150 // ordinary progress
	a.ScrapeAll(ctx, []string{srv.URL})
	if got := total(t, a); got != 150 {
		t.Fatalf("after progress total = %v, want 150 (delta 50 added)", got)
	}

	value = 20 // the pod was replaced and its counter restarted
	a.ScrapeAll(ctx, []string{srv.URL})
	if got := total(t, a); got != 170 {
		t.Errorf("after upstream restart total = %v, want 170: a restart must add "+
			"the new value as a delta from zero, never rewind the total", got)
	}
}

// A departing endpoint must not take its contribution with it: the work
// happened, and a counter that drops on scale-down is the same bug as one that
// drops on restart.
func TestDepartedEndpointKeepsItsContribution(t *testing.T) {
	a := New(Config{})
	ctx := context.Background()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body(42))
	}))
	a.ScrapeAll(ctx, []string{srv.URL})
	srv.Close()

	a.ScrapeAll(ctx, nil) // endpoint gone from the fleet
	if got := total(t, a); got != 42 {
		t.Errorf("total = %v after the endpoint left, want 42 retained", got)
	}
}

// An unreachable endpoint contributes nothing and must not be read as a reset
// to zero, which would corrupt every later delta from it.
func TestFailedScrapeIsNotAReset(t *testing.T) {
	var fail bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		fmt.Fprint(w, body(80))
	}))
	defer srv.Close()

	a := New(Config{})
	ctx := context.Background()
	a.ScrapeAll(ctx, []string{srv.URL})
	fail = true
	a.ScrapeAll(ctx, []string{srv.URL})
	fail = false
	a.ScrapeAll(ctx, []string{srv.URL}) // same value 80, no real progress
	if got := total(t, a); got != 80 {
		t.Errorf("total = %v, want 80: a failed scrape must contribute nothing "+
			"and must not be mistaken for the counter restarting", got)
	}
}

// Two endpoints sum, which is the aggregation the router exists to provide.
func TestTotalsSumAcrossEndpoints(t *testing.T) {
	mk := func(v float64) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, body(v))
		}))
	}
	a, b := mk(10), mk(5)
	defer a.Close()
	defer b.Close()

	agg := New(Config{})
	agg.ScrapeAll(context.Background(), []string{a.URL, b.URL})
	if got := total(t, agg); got != 15 {
		t.Errorf("total = %v, want 15 summed across both endpoints", got)
	}

	var sb strings.Builder
	if err := agg.Render(&sb); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	out := sb.String()
	for _, want := range []string{"# TYPE vllm:prompt_tokens_by_source_total counter",
		`vllm:prompt_tokens_by_source_total{source="cache"} 15`} {
		if !strings.Contains(out, want) {
			t.Errorf("exposition missing %q:\n%s", want, out)
		}
	}
}

// Only the configured names are kept: re-exporting every vLLM series would
// multiply cardinality by the fleet size for no one's benefit.
func TestUntrackedSeriesAreIgnored(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "vllm:num_requests_running 7\n"+body(3))
	}))
	defer srv.Close()

	a := New(Config{})
	a.ScrapeAll(context.Background(), []string{srv.URL})
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.totals) != 1 {
		t.Errorf("tracked %d series, want only the configured one: %v", len(a.totals), a.totals)
	}
}

// sglangBody is a live-like SGLang exposition: tier series appear only once
// >0, labels are engine_type/model_name, and a per-rank gauge sits alongside
// the counters.
func sglangBody(device, host, prompt float64) string {
	return fmt.Sprintf("# TYPE sglang:cached_tokens_total counter\n"+
		"sglang:cached_tokens_total{cache_source=\"device\",engine_type=\"unified\",model_name=\"m\"} %g\n"+
		"sglang:cached_tokens_total{cache_source=\"host\",engine_type=\"unified\",model_name=\"m\"} %g\n"+
		"# TYPE sglang:prompt_tokens_total counter\n"+
		"sglang:prompt_tokens_total{engine_type=\"unified\",model_name=\"m\"} %g\n"+
		"# TYPE sglang:cache_hit_rate gauge\n"+
		"sglang:cache_hit_rate{engine_type=\"unified\",model_name=\"m\"} 0.7\n", device, host, prompt)
}

// SGLang's request-level counters are summed across backends and re-emitted
// under their own names; per-rank gauges are never aggregated.
func TestSGLangCountersAggregateAcrossBackendsAndRender(t *testing.T) {
	mk := func(d, h, p float64) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, sglangBody(d, h, p))
		}))
	}
	a, b := mk(100, 10, 400), mk(50, 5, 200)
	defer a.Close()
	defer b.Close()

	agg := New(Config{})
	agg.ScrapeAll(context.Background(), []string{a.URL, b.URL})

	var out strings.Builder
	if err := agg.Render(&out); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	for _, want := range []string{
		`sglang:cached_tokens_total{cache_source="device",engine_type="unified",model_name="m"} 150`,
		`sglang:cached_tokens_total{cache_source="host",engine_type="unified",model_name="m"} 15`,
		`sglang:prompt_tokens_total{engine_type="unified",model_name="m"} 600`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("exposition missing %q\n%s", want, got)
		}
	}
	if strings.Contains(got, "cache_hit_rate") {
		t.Errorf("a per-rank gauge must never be aggregated:\n%s", got)
	}
}

// A mixed fleet carries both engines' families, and each backend contributes
// only the series it serves without being retired as barren.
func TestMixedVLLMAndSGLangFleet(t *testing.T) {
	v := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, body(70)) }))
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, sglangBody(1, 2, 9)) }))
	defer v.Close()
	defer s.Close()
	agg := New(Config{})
	for i := 0; i < 3; i++ {
		agg.ScrapeAll(context.Background(), []string{v.URL, s.URL})
	}
	if ok, asked := agg.Coverage(); ok != 2 || asked != 2 {
		t.Errorf("coverage = %d/%d, want 2/2", ok, asked)
	}
	var out strings.Builder
	_ = agg.Render(&out)
	if !strings.Contains(out.String(), "vllm:prompt_tokens_by_source_total") || !strings.Contains(out.String(), "sglang:prompt_tokens_total") {
		t.Errorf("mixed fleet exposition incomplete:\n%s", out.String())
	}
}
