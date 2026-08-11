package vllmmetrics

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
)

// A mock fleet, because the bug this file exists for is invisible with one
// endpoint and one series.
//
// What was reported from production: the router's aggregated
// vllm:prompt_tokens_by_source_total was byte-identical to ONE instance's raw
// counters, roughly an eighth of the fleet's true total, and its
// local_cache_hit delta between two snapshots was NEGATIVE. A counter summed
// over eight backends cannot decrease, so the number being exported was not a
// sum at all.
//
// The shape matters and is reproduced exactly here: every vLLM in a fleet emits
// the SAME label set for a given source (engine, model_name, source), so all
// eight instances collapse onto one series key. That is what makes "is this the
// fleet's sum or one instance's value" a question a test can ask — with
// distinct labels per instance they would never be confusable.

// fakeVLLM is one instance's /metrics: a counter per source that only goes up,
// until it is restarted and starts from zero like a replaced pod.
type fakeVLLM struct {
	srv *httptest.Server

	mu      sync.Mutex
	vals    map[string]float64
	scrapes int
	// serveNothing makes the instance answer 200 with no tracked series at all,
	// which is what a hosted API or a sidecar behind the same address looks like.
	serveNothing bool
}

func newFakeVLLM(t *testing.T, start map[string]float64) *fakeVLLM {
	t.Helper()
	f := &fakeVLLM{vals: map[string]float64{}}
	for k, v := range start {
		f.vals[k] = v
	}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.scrapes++
		nothing := f.serveNothing
		sources := make([]string, 0, len(f.vals))
		for s := range f.vals {
			sources = append(sources, s)
		}
		sort.Strings(sources)
		var sb strings.Builder
		if !nothing {
			sb.WriteString("# HELP vllm:prompt_tokens_by_source_total x\n")
			sb.WriteString("# TYPE vllm:prompt_tokens_by_source_total counter\n")
			for _, s := range sources {
				// The real label set. Identical across every instance in a fleet,
				// which is the whole point — see the file comment.
				fmt.Fprintf(&sb, "vllm:prompt_tokens_by_source_total"+
					"{engine=\"0\",model_name=\"m\",source=%q} %g\n", s, f.vals[s])
			}
		}
		// An untracked series, always present, so the parser's filtering is
		// exercised by every test here rather than only by its own.
		sb.WriteString("vllm:num_requests_running 3\n")
		f.mu.Unlock()
		fmt.Fprint(w, sb.String())
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeVLLM) add(source string, delta float64) {
	f.mu.Lock()
	f.vals[source] += delta
	f.mu.Unlock()
}

// restart replaces the pod: every counter starts from zero again.
func (f *fakeVLLM) restart() {
	f.mu.Lock()
	for s := range f.vals {
		f.vals[s] = 0
	}
	f.mu.Unlock()
}

func (f *fakeVLLM) value(source string) float64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.vals[source]
}

func (f *fakeVLLM) hits() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.scrapes
}

func urls(fleet []*fakeVLLM) []string {
	out := make([]string, len(fleet))
	for i, f := range fleet {
		out[i] = f.srv.URL
	}
	return out
}

// seriesTotal reads one source's accumulated total out of the aggregator.
func seriesTotal(t *testing.T, a *Aggregator, source string) float64 {
	t.Helper()
	a.mu.Lock()
	defer a.mu.Unlock()
	want := fmt.Sprintf("source=%q", source)
	for k, v := range a.totals {
		if strings.Contains(k, want) {
			return v
		}
	}
	return 0
}

const cacheHit = "local_cache_hit"

// TestFleetTotalIsTheSumNotOneInstance is the direct regression test for the
// production report: eight backends, and the router must export their sum.
func TestFleetTotalIsTheSumNotOneInstance(t *testing.T) {
	const n = 8
	var fleet []*fakeVLLM
	var want float64
	for i := range n {
		v := float64(100 * (i + 1))
		fleet = append(fleet, newFakeVLLM(t, map[string]float64{cacheHit: v}))
		want += v
	}

	a := New(Config{})
	a.ScrapeAll(context.Background(), urls(fleet))

	got := seriesTotal(t, a, cacheHit)
	if got != want {
		t.Errorf("aggregated %s = %v, want %v — the sum over all %d backends", cacheHit, got, want, n)
	}
	for i, f := range fleet {
		if got == f.value(cacheHit) {
			t.Errorf("aggregated total %v is byte-identical to instance %d's own counter: the "+
				"router is mirroring one backend rather than aggregating the fleet, which is "+
				"exactly what was seen in production", got, i)
		}
	}
}

// TestFleetTotalNeverDecreases walks a fleet through ordinary progress, one
// pod restarting, and one going unreachable, checking the exported total after
// every step. A decreasing aggregate is the reliable tell that something is
// wrong, so it is asserted at every step rather than only at the end.
func TestFleetTotalNeverDecreases(t *testing.T) {
	const n = 8
	var fleet []*fakeVLLM
	for i := range n {
		fleet = append(fleet, newFakeVLLM(t, map[string]float64{cacheHit: float64(100 * (i + 1))}))
	}
	a := New(Config{})
	ctx := context.Background()

	// Scrape 1: first sighting of everything. 100+200+...+800.
	a.ScrapeAll(ctx, urls(fleet))
	last := seriesTotal(t, a, cacheHit)
	if want := 3600.0; last != want {
		t.Fatalf("after first scrape total = %v, want %v", last, want)
	}

	// Scrape 2: every instance advances by 10.
	for _, f := range fleet {
		f.add(cacheHit, 10)
	}
	a.ScrapeAll(ctx, urls(fleet))
	if got, want := seriesTotal(t, a, cacheHit), 3680.0; got != want {
		t.Fatalf("after fleet-wide progress total = %v, want %v (8 deltas of 10)", got, want)
	}
	last = 3680

	// Scrape 3: instance 3 is replaced — its counter restarts from zero and it
	// serves 5 tokens' worth before the scrape. The other seven advance by 10.
	//
	// The router must remember the history it already accumulated and add the
	// restarted instance's new value as a delta from zero: 3680 + 70 + 5.
	fleet[3].restart()
	fleet[3].add(cacheHit, 5)
	for i, f := range fleet {
		if i != 3 {
			f.add(cacheHit, 10)
		}
	}
	a.ScrapeAll(ctx, urls(fleet))
	got := seriesTotal(t, a, cacheHit)
	if want := 3755.0; got != want {
		t.Errorf("after one pod restarted total = %v, want %v: the accumulated history must be "+
			"kept and the restarted pod's counter added as a delta from zero", got, want)
	}
	if got < last {
		t.Errorf("total went backwards, %v -> %v, on a pod restart", last, got)
	}
	last = got

	// Scrape 4: instance 5 is unreachable. It contributes nothing, and nothing
	// it already contributed is lost.
	fleet[5].srv.Close()
	for i, f := range fleet {
		if i != 5 {
			f.add(cacheHit, 10)
		}
	}
	a.ScrapeAll(ctx, urls(fleet))
	got = seriesTotal(t, a, cacheHit)
	if want := last + 70; got != want {
		t.Errorf("with one backend unreachable total = %v, want %v (seven deltas of 10, and "+
			"nothing lost from the eighth's history)", got, want)
	}
	if got < last {
		t.Errorf("total went backwards, %v -> %v, when a backend became unreachable", last, got)
	}
}

// TestRestartIsPerInstance: a restart on one backend must not be read as a
// restart of the series. Two instances, one restarting, and the other's steady
// progress must still be counted as a delta rather than re-added whole.
func TestRestartIsPerInstance(t *testing.T) {
	a := New(Config{})
	ctx := context.Background()
	x := newFakeVLLM(t, map[string]float64{cacheHit: 1000})
	y := newFakeVLLM(t, map[string]float64{cacheHit: 2000})

	a.ScrapeAll(ctx, urls([]*fakeVLLM{x, y})) // 3000
	x.restart()
	x.add(cacheHit, 7)
	y.add(cacheHit, 50)
	a.ScrapeAll(ctx, urls([]*fakeVLLM{x, y}))

	if got, want := seriesTotal(t, a, cacheHit), 3057.0; got != want {
		t.Errorf("total = %v, want %v: x restarted and contributes its whole new value (7), "+
			"y did not and contributes its delta (50), not its whole counter (2050)", got, want)
	}
}

// TestEveryTrackedSourceIsAggregatedSeparately. The production report was a
// three-row table — local_compute, local_cache_hit, external_kv_transfer — and
// one of the three looked plausible while another was impossible. They are
// distinct series and must not be blended.
func TestEveryTrackedSourceIsAggregatedSeparately(t *testing.T) {
	sources := map[string]float64{"local_compute": 11, "local_cache_hit": 22, "external_kv_transfer": 33}
	var fleet []*fakeVLLM
	for range 4 {
		fleet = append(fleet, newFakeVLLM(t, sources))
	}
	a := New(Config{})
	a.ScrapeAll(context.Background(), urls(fleet))

	for src, per := range sources {
		if got, want := seriesTotal(t, a, src), per*4; got != want {
			t.Errorf("%s = %v, want %v across four backends", src, got, want)
		}
	}
	a.mu.Lock()
	n := len(a.totals)
	a.mu.Unlock()
	if n != len(sources) {
		t.Errorf("tracked %d series, want %d — one per source, summed over the fleet", n, len(sources))
	}
}

// TestBackendsJoiningLaterAreScraped is the half of the production bug that
// lives above this package: the endpoint set used to be captured at startup, so
// a pool populated by Kubernetes pod discovery contributed nothing at all.
// ScrapeAll takes the set per cycle, and a backend that appears later must be
// picked up with its full history.
func TestBackendsJoiningLaterAreScraped(t *testing.T) {
	a := New(Config{})
	ctx := context.Background()
	first := newFakeVLLM(t, map[string]float64{cacheHit: 100})
	a.ScrapeAll(ctx, urls([]*fakeVLLM{first}))

	// A pod arrives, already carrying history from before the router saw it.
	later := newFakeVLLM(t, map[string]float64{cacheHit: 250})
	a.ScrapeAll(ctx, urls([]*fakeVLLM{first, later}))

	if got, want := seriesTotal(t, a, cacheHit), 350.0; got != want {
		t.Errorf("total = %v, want %v: a backend discovered after startup must be scraped, and "+
			"its first sighting counts in full", got, want)
	}
}

// TestBarrenEndpointIsRetired: the endpoint set is now the live registry, which
// may hold things that are not vLLMs at all. Those must stop being asked rather
// than being polled forever.
func TestBarrenEndpointIsRetired(t *testing.T) {
	a := New(Config{})
	ctx := context.Background()
	notVLLM := newFakeVLLM(t, map[string]float64{cacheHit: 1})
	notVLLM.mu.Lock()
	notVLLM.serveNothing = true
	notVLLM.mu.Unlock()

	for range deadAfter + 3 {
		a.ScrapeAll(ctx, urls([]*fakeVLLM{notVLLM}))
	}
	if hits := notVLLM.hits(); hits > deadAfter {
		t.Errorf("endpoint serving none of the tracked counters was scraped %d times, want at "+
			"most %d: it must be latched off, not polled forever", hits, deadAfter)
	}
	if got := seriesTotal(t, a, cacheHit); got != 0 {
		t.Errorf("total = %v from an endpoint that served no tracked series, want 0", got)
	}
}

// TestRetireSkipsAKnownNonVLLM: what the caller already knows should not cost
// two cycles to rediscover.
func TestRetireSkipsAKnownNonVLLM(t *testing.T) {
	a := New(Config{})
	hosted := newFakeVLLM(t, map[string]float64{cacheHit: 5})
	a.Retire(hosted.srv.URL)
	a.ScrapeAll(context.Background(), urls([]*fakeVLLM{hosted}))
	if hits := hosted.hits(); hits != 0 {
		t.Errorf("a retired endpoint was scraped %d times, want 0", hits)
	}
}

// TestLoadBalancedAddressIsCalledOut. One address that answers as a different
// process each time is the configuration that makes aggregation meaningless —
// there is no "what this address said last time" when the address is eight
// pods. The totals cannot be salvaged; the point is that the router says so
// rather than exporting a plausible-looking number in silence.
func TestLoadBalancedAddressIsCalledOut(t *testing.T) {
	// One server, cycling its counters as though a Service were routing each
	// scrape to a different pod.
	vals := []float64{900, 100, 800, 200, 700, 300, 600}
	i := 0
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		v := vals[i%len(vals)]
		i++
		mu.Unlock()
		fmt.Fprintf(w, "vllm:prompt_tokens_by_source_total"+
			"{engine=\"0\",model_name=\"m\",source=%q} %g\n", cacheHit, v)
	}))
	defer srv.Close()

	a := New(Config{})
	ctx := context.Background()
	// Twice round the cycle: only the downward transitions look like restarts,
	// so roughly half of each pass counts, and the warning is deliberately not
	// hair-triggered — real pods do restart.
	for range 2 * len(vals) {
		a.ScrapeAll(ctx, []string{srv.URL})
	}

	a.mu.Lock()
	resets, warned := a.resets[srv.URL], a.warned[srv.URL]
	a.mu.Unlock()
	if resets < resetsBeforeWarning {
		t.Fatalf("counted %d resets from an address answering as a different process every "+
			"scrape, want at least %d", resets, resetsBeforeWarning)
	}
	if !warned {
		t.Errorf("an address whose counters restarted %d times was never called out. Aggregation "+
			"through a Service or load balancer cannot work, and it produces a number that looks "+
			"entirely reasonable — silence is the worst outcome here", resets)
	}
}
