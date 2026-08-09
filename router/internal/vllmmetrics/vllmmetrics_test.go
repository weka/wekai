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
	if err := agg.WriteTo(&sb); err != nil {
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
