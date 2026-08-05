package cache_test

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	cachepolicy "github.com/weka/wekai/router/internal/policy/cache"
)

func TestThresholdNoCandidates(t *testing.T) {
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	if _, err := p.Select(context.Background(), nil, req(units(1))); err != policy.ErrNoCandidates {
		t.Fatalf("err = %v, want ErrNoCandidates", err)
	}
}

// CU-11: no routable prefix means decline, not guess.
func TestThresholdNoUnitsFallsBack(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	got, err := p.Select(context.Background(), bs, req(nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("returned nil backend")
	}
}

// A brand new prompt (nothing clears the threshold) must route to the
// least-loaded of ALL candidates, ignoring cache entirely.
func TestThresholdNoCandidateClearsThresholdUsesLeastLoadedOfAll(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	bs[1].AddInflight(5)
	bs[2].AddInflight(9)

	got, err := p.Select(context.Background(), bs, req(units(1, 2, 3, 4)))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[0] {
		t.Errorf("selected %s on a cold fleet, want the least-loaded %s", got.URL, bs[0].URL)
	}
}

// Exactly one candidate clears the threshold and it is not too busy: use it
// even though another candidate is less loaded overall — cache wins in the
// single-candidate case.
func TestThresholdSoleCandidateUnderPendingLimitWins(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	u := units(1, 2, 3, 4)
	p.Commit(bs[1], req(u)) // only w1 knows this prefix
	bs[1].AddInflight(5)    // busier than w0 and w2, but well under MaxPending (32)

	got, err := p.Select(context.Background(), bs, req(u))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[1] {
		t.Errorf("selected %s, want the sole cache candidate %s", got.URL, bs[1].URL)
	}
}

// Exactly one candidate clears the threshold but it is too busy: ignore cache
// and route to the least-loaded of ALL candidates, not the hot one.
func TestThresholdSoleCandidateOverPendingLimitIgnoresCache(t *testing.T) {
	bs := backends(t, 3)
	cfg := cachepolicy.DefaultThresholdConfig()
	cfg.MaxPending = 4
	p := cachepolicy.NewThreshold(cfg, policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	u := units(1, 2, 3, 4)
	p.Commit(bs[1], req(u))
	bs[1].AddInflight(4) // == MaxPending, not below it
	bs[2].AddInflight(9) // busiest, so it must not be picked either

	got, err := p.Select(context.Background(), bs, req(u))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[0] {
		t.Errorf("selected %s, want the least-loaded of all candidates %s (cache ignored)", got.URL, bs[0].URL)
	}
}

// More than one candidate clears the threshold: route to the least-loaded
// AMONG THOSE, even though a non-candidate is less loaded than any of them.
func TestThresholdMultipleCandidatesUsesLeastLoadedAmongThem(t *testing.T) {
	bs := backends(t, 4)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	u := units(1, 2, 3, 4)
	p.Commit(bs[1], req(u))
	p.Commit(bs[2], req(u))
	// bs[0] never served this prefix, but is the least loaded overall.
	bs[1].AddInflight(6)
	bs[2].AddInflight(3)

	got, err := p.Select(context.Background(), bs, req(u))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[2] {
		t.Errorf("selected %s, want the least-loaded cache candidate %s (not the idle non-candidate)", got.URL, bs[2].URL)
	}
}

// A partial match at or below the threshold must not count as a candidate.
func TestThresholdPartialMatchBelowThresholdIsNotACandidate(t *testing.T) {
	bs := backends(t, 2)
	cfg := cachepolicy.DefaultThresholdConfig()
	cfg.CacheThreshold = 0.5
	p := cachepolicy.NewThreshold(cfg, policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	// w0 knows 1 of the 10 units the request carries: 10% matched.
	p.Commit(bs[0], req(units(1)))
	bs[0].AddInflight(5) // busier, so least-outstanding alone would avoid it anyway

	got, err := p.Select(context.Background(), bs, req(units(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[1] {
		t.Errorf("selected %s on a 10%% match with threshold 0.5; should have fallen back", got.URL)
	}
}

// Select must not teach any backend anything: only Commit does.
func TestThresholdSelectDoesNotCommit(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
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
			t.Errorf("%s learned %d nodes from Select alone; only Commit may write", url, st[0])
		}
	}
}

// Commit only teaches the winning backend's model.
func TestThresholdCommitTeachesOnlyTheWinner(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	p.Commit(bs[0], req(units(1, 2, 3)))
	for url, st := range p.Stats() {
		if url == bs[0].URL {
			if st[0] == 0 {
				t.Errorf("%s has 0 nodes after Commit", url)
			}
			continue
		}
		if st[0] != 0 {
			t.Errorf("%s learned %d nodes without being committed to", url, st[0])
		}
	}
}

// CACHE-10: a removed backend's model is dropped and not inherited.
func TestThresholdDropBackendDiscardsItsModel(t *testing.T) {
	bs := backends(t, 2)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	for _, b := range bs {
		p.AddBackend(b)
	}
	p.Commit(bs[0], req(units(1, 2, 3)))
	p.DropBackend(bs[0])
	if s := p.Stats(); len(s) != 1 {
		t.Errorf("stats has %d models after dropping one of two: %v", len(s), s)
	}
}

func TestThresholdFlushClearsAllModels(t *testing.T) {
	bs := backends(t, 3)
	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
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

// router_route_decisions_total must reflect which branch of the decision tree
// actually chose the backend, not just that a selection happened.
func TestThresholdRouteDecisionClassification(t *testing.T) {
	cacheBefore := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("cache"))
	loadBefore := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("load"))

	p := cachepolicy.NewThreshold(cachepolicy.DefaultThresholdConfig(), policy.LeastOutstanding{})
	bs := backends(t, 3)
	for _, b := range bs {
		p.AddBackend(b)
	}

	// Zero candidates clear the threshold: "load".
	if _, err := p.Select(context.Background(), bs, req(units(1, 2, 3, 4))); err != nil {
		t.Fatal(err)
	}
	if got := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("load")); got != loadBefore+1 {
		t.Errorf("load = %v, want %v after a no-candidate selection", got, loadBefore+1)
	}

	// Exactly one candidate, under the pending limit: "cache".
	u := units(5, 6, 7, 8)
	p.Commit(bs[0], req(u))
	if _, err := p.Select(context.Background(), bs, req(u)); err != nil {
		t.Fatal(err)
	}
	if got := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("cache")); got != cacheBefore+1 {
		t.Errorf("cache = %v, want %v after a sole-candidate cache hit", got, cacheBefore+1)
	}

	// Exactly one candidate, over the pending limit: falls back to "load".
	cfg := cachepolicy.DefaultThresholdConfig()
	cfg.MaxPending = 1
	p2 := cachepolicy.NewThreshold(cfg, policy.LeastOutstanding{})
	for _, b := range bs {
		p2.AddBackend(b)
	}
	u2 := units(9, 10, 11, 12)
	p2.Commit(bs[0], req(u2))
	bs[0].AddInflight(1)
	loadBefore2 := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("load"))
	if _, err := p2.Select(context.Background(), bs, req(u2)); err != nil {
		t.Fatal(err)
	}
	if got := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("load")); got != loadBefore2+1 {
		t.Errorf("load = %v, want %v after a pending-exceeded selection", got, loadBefore2+1)
	}

	// Multiple candidates: still "cache", even though the tie-break mechanism
	// is least-outstanding.
	cacheBefore2 := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("cache"))
	u3 := units(13, 14, 15, 16)
	p.Commit(bs[1], req(u3))
	p.Commit(bs[2], req(u3))
	if _, err := p.Select(context.Background(), bs, req(u3)); err != nil {
		t.Fatal(err)
	}
	if got := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues("cache")); got != cacheBefore2+1 {
		t.Errorf("cache = %v, want %v after a multi-candidate selection", got, cacheBefore2+1)
	}
}
