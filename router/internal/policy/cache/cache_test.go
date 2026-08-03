package cache_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/policy"
	cachepolicy "github.com/weka/wekai/router/internal/policy/cache"
	"github.com/weka/wekai/router/internal/registry"
)

func backends(t *testing.T, n int) []*registry.Backend {
	t.Helper()
	r := registry.New(registry.Options{})
	out := make([]*registry.Backend, 0, n)
	for i := 0; i < n; i++ {
		b, err := r.Add(registry.Spec{URL: fmt.Sprintf("http://w%d:8000", i), Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		out = append(out, b)
	}
	return out
}

func units(hashes ...uint64) []kvcache.Unit {
	out := make([]kvcache.Unit, len(hashes))
	for i, h := range hashes {
		out[i] = kvcache.Unit{Hash: h, Tokens: 100}
	}
	return out
}

func req(u []kvcache.Unit) *policy.RoutingRequest {
	return &policy.RoutingRequest{RouteClass: "chat", DialectID: "openai", Units: u}
}

func TestRoutesToTheBackendHoldingThePrefix(t *testing.T) {
	bs := backends(t, 4)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}

	u := units(1, 2, 3, 4)
	p.Commit(bs[2], req(u)) // only w2 has served this prefix

	for i := 0; i < 20; i++ {
		got, err := p.Select(context.Background(), bs, req(u))
		if err != nil {
			t.Fatal(err)
		}
		if got != bs[2] {
			t.Fatalf("selected %s, want the backend holding the prefix (%s)", got.URL, bs[2].URL)
		}
	}
}

// FR-RTR-01: route to the node holding the slice "even if that server is already
// heavily loaded". Affinity wins until the fleet is measurably imbalanced.
func TestAffinityWinsUnderModerateLoad(t *testing.T) {
	bs := backends(t, 4)
	cfg := cachepolicy.DefaultConfig() // abs 32, rel 1.5
	p := cachepolicy.New(cfg, policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}

	u := units(1, 2, 3, 4)
	p.Commit(bs[0], req(u))

	// w0 is busier than its idle peers, but not past the absolute threshold.
	for i := 0; i < 20; i++ {
		bs[0].AddInflight(1)
	}
	got, err := p.Select(context.Background(), bs, req(u))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[0] {
		t.Errorf("selected %s at inflight=20; affinity should still win below the "+
			"spill threshold (FR-RTR-01)", got.URL)
	}
}

// ...but not without limit. Past the threshold it spills to the load policy.
func TestSpillsOnceImbalanced(t *testing.T) {
	bs := backends(t, 4)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	u := units(1, 2, 3, 4)
	p.Commit(bs[0], req(u))

	for i := 0; i < 40; i++ { // (40-0) > 32 and 40 > 0*1.5
		bs[0].AddInflight(1)
	}
	got, err := p.Select(context.Background(), bs, req(u))
	if err != nil {
		t.Fatal(err)
	}
	if got == bs[0] {
		t.Errorf("still selected the saturated cache owner at inflight=40; " +
			"the spill guard did not fire")
	}
}

// CACHE-N4, the v1 bug: the guard must be computed over CANDIDATES only. v1
// folded over the full worker list, so one dead worker holding stale load kept
// max_load high forever and silently disabled cache routing for good.
func TestSpillGuardIgnoresNonCandidates(t *testing.T) {
	bs := backends(t, 4)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	u := units(1, 2, 3, 4)
	p.Commit(bs[0], req(u))

	// A backend that has gone unhealthy while holding a large stale load.
	bs[3].SetHealth(registry.Unhealthy)
	for i := 0; i < 500; i++ {
		bs[3].AddInflight(1)
	}
	candidates := []*registry.Backend{bs[0], bs[1], bs[2]} // filtered, as the gateway does

	got, err := p.Select(context.Background(), candidates, req(u))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[0] {
		t.Errorf("selected %s: a non-candidate's stale load latched the spill guard "+
			"on and disabled cache routing", got.URL)
	}
}

// CU-11: the policy must never fail a request. No units means decline, not guess.
func TestNoUnitsFallsBack(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	got, err := p.Select(context.Background(), bs, req(nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("returned nil backend")
	}
}

func TestNoCandidates(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	if _, err := p.Select(context.Background(), nil, req(units(1))); err != policy.ErrNoCandidates {
		t.Fatalf("err = %v, want ErrNoCandidates", err)
	}
}

// Below the match threshold, affinity is not worth overriding load balance — and
// the fallback must be least-outstanding, not "first candidate". v1's min_by_key
// returned the first minimum, so a cold fleet funnelled every request onto
// candidate 0 until it crossed the imbalance threshold (LB-N4, CACHE-N5).
func TestBelowThresholdSpreadsRatherThanPilingOnIndexZero(t *testing.T) {
	bs := backends(t, 4)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	counts := map[string]int{}
	for i := 0; i < 400; i++ {
		// A cold trie: nothing matches, so every selection is a fallback.
		got, err := p.Select(context.Background(), bs, req(units(uint64(i)*7+1, uint64(i)*7+2)))
		if err != nil {
			t.Fatal(err)
		}
		counts[got.URL]++
	}
	if len(counts) != len(bs) {
		t.Fatalf("only %d of %d backends ever selected on a cold fleet: %v",
			len(counts), len(bs), counts)
	}
}

// A partial match below the threshold must not win.
func TestPartialMatchBelowThresholdDoesNotWin(t *testing.T) {
	bs := backends(t, 2)
	cfg := cachepolicy.DefaultConfig()
	cfg.CacheThreshold = 0.5
	p := cachepolicy.New(cfg, policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	// w0 knows 1 of the 10 units the request carries: 10% matched.
	p.Commit(bs[0], req(units(1)))
	// Make w0 the busier one so least-outstanding would avoid it.
	bs[0].AddInflight(5)

	got, err := p.Select(context.Background(), bs, req(units(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[1] {
		t.Errorf("selected %s on a 10%% match with threshold 0.5; should have "+
			"deferred to load", got.URL)
	}
}

// CACHE-10: a removed backend's model is dropped, and its prefixes are not
// inherited by anyone.
func TestDropBackendDiscardsItsModel(t *testing.T) {
	bs := backends(t, 2)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(bs[0], req(u))
	p.DropBackend(bs[0])

	if s := p.Stats(); len(s) != 1 {
		t.Errorf("stats has %d models after dropping one of two: %v", len(s), s)
	}
	// Re-added, it must start cold rather than inheriting anything.
	p.AddBackend(bs[0])
	got, err := p.Select(context.Background(), bs, req(u))
	if err != nil {
		t.Fatal(err)
	}
	_ = got // either is acceptable; what matters is no stale affinity
	for url, st := range p.Stats() {
		if url == bs[0].URL && st[0] != 0 {
			t.Errorf("re-added backend has %d nodes, want 0", st[0])
		}
	}
}

func TestFlushClearsAllModels(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
		p.Commit(b, req(units(1, 2, 3)))
	}
	p.Flush()
	for url, st := range p.Stats() {
		if st[0] != 0 {
			t.Errorf("%s still holds %d nodes after Flush", url, st[0])
		}
	}
}

// Select must not teach any backend anything: only Commit does.
func TestSelectDoesNotCommit(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	for i := 0; i < 100; i++ {
		if _, err := p.Select(context.Background(), bs, req(units(1, 2, 3))); err != nil {
			t.Fatal(err)
		}
	}
	for url, st := range p.Stats() {
		if st[0] != 0 {
			t.Errorf("%s learned %d nodes from Select alone; only Commit may write",
				url, st[0])
		}
	}
}

func BenchmarkSelect64Backends32Units(b *testing.B) {
	r := registry.New(registry.Options{})
	var bs []*registry.Backend
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	for i := 0; i < 64; i++ {
		be, _ := r.Add(registry.Spec{URL: fmt.Sprintf("http://w%02d:8000", i), Capacity: 1})
		be.SetHealth(registry.Healthy)
		bs = append(bs, be)
		p.AddBackend(be)
	}
	u := make([]kvcache.Unit, 32)
	for i := range u {
		u[i] = kvcache.Unit{Hash: uint64(i), Tokens: 256}
	}
	rr := req(u)
	p.Commit(bs[7], rr)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Select(ctx, bs, rr); err != nil {
			b.Fatal(err)
		}
	}
}

// Among equally-warm backends the choice must go to the least loaded, not to the
// first in candidate order. Snapshot order is sorted by URL, so a strict `>`
// funnels a fleet warm on a shared system prompt onto the lexicographically
// smallest backend — the v1 index-0 herd, reintroduced in the affinity path.
func TestEquallyWarmBackendsPickTheLeastLoaded(t *testing.T) {
	bs := backends(t, 4)
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	u := units(1, 2, 3, 4)
	for _, b := range bs {
		p.AddBackend(b)
		p.Commit(b, req(u)) // every backend is equally warm
	}
	// w0 is the lexicographically first AND the busiest.
	for i := 0; i < 10; i++ {
		bs[0].AddInflight(1)
	}
	for i := 0; i < 50; i++ {
		got, err := p.Select(context.Background(), bs, req(u))
		if err != nil {
			t.Fatal(err)
		}
		if got == bs[0] {
			t.Fatalf("selected the busiest of several equally-warm backends (%s); "+
				"ties must break on load", got.URL)
		}
	}
}
