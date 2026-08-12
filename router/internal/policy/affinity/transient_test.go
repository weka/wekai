package affinity

import (
	"context"
	"testing"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// Transient fallback serves a request the split guard refused, WITHOUT
// recording the backend as a holder.
//
// The guard protects the tree rather than capacity: a split adds a holder for
// good, and a fleet that splits freely converges on everyone holding
// everything. Serving without a mark costs nothing permanent, so it is allowed
// much closer to the holders' own load — the same arithmetic against the same
// reference, a smaller margin.

func newTransientPolicy(t *testing.T, transient float64) *Policy {
	t.Helper()
	p, err := New(Config{
		NodeConcurrency:   testConcurrency,
		SplitGuard:        DefaultSplitGuard, // 0.20 -> split below 80% of ref
		TransientFallback: transient,
		TailTTL:           testTTL,
		Clock:             clock.NewFake(time.Time{}),
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// guardBlockedFleet builds the state tier 3 exists for: the prefix's only
// holder is saturated, and every other backend sits between the guard's
// threshold and the reference — idle enough to serve, too loaded to be worth a
// permanent copy.
func guardBlockedFleet(t *testing.T, p *Policy, transientOK bool) ([]*registry.Backend, []kvcache.Unit) {
	t.Helper()
	f := fleet(t, 3)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	load(t, f[0], testConcurrency) // the holder is at the limit: ref = 32

	// Guard threshold is 32*0.8 = 25.6; transient threshold at 0.05 is 32*0.95
	// = 30.4. Park the others between the two, or above both.
	other := int64(28)
	if !transientOK {
		other = 31
	}
	load(t, f[1], other)
	load(t, f[2], other)
	return f, u
}

func TestTransientFallbackServesWithoutMarking(t *testing.T) {
	p := newTransientPolicy(t, 0.05)
	f, u := guardBlockedFleet(t, p, true)

	rr := req(u)
	got, err := p.Select(context.Background(), f, rr)
	if err != nil {
		t.Fatalf("Select refused with two backends at 28 of a 32 reference and a transient "+
			"threshold of 0.05 (limit 30.4): %v", err)
	}
	if got == f[0] {
		t.Fatalf("routed to the saturated holder %s", got.URL)
	}

	// The decision must be explicitly non-marking, and Commit must honour it.
	d, ok := rr.PolicyState.(*decision)
	if !ok || d.mark {
		t.Fatalf("PolicyState = %+v, want a non-marking decision: a transiently-served backend "+
			"must not be recorded as holding the prefix, or it is a split by another name",
			rr.PolicyState)
	}

	before := p.tree.stats()
	p.Commit(got, rr)
	after := p.tree.stats()
	if after.AvgCopies > before.AvgCopies {
		t.Errorf("mean holders per block rose %.3f -> %.3f after a transient fallback; the whole "+
			"point is that it leaves no trace in the tree", before.AvgCopies, after.AvgCopies)
	}

	// And the next request for the same prefix must still see only the original
	// holder, not the backend that transiently served it.
	a := p.tree.walk(path{units: u, modelKey: modelKey("m")}, allSlots(p, f))
	if a.held.Count() != 1 {
		t.Errorf("the prefix now has %d holders, want 1 — the transient serve was recorded",
			a.held.Count())
	}
}

func TestTransientFallbackStillRefusesWhenTrulyLoaded(t *testing.T) {
	p := newTransientPolicy(t, 0.05)
	f, u := guardBlockedFleet(t, p, false) // peers at 31, above the 30.4 limit

	if got, err := p.Select(context.Background(), f, req(u)); err == nil {
		t.Errorf("routed to %s at 31 of a 32 reference: past the transient limit of 30.4 a "+
			"backend is genuinely as loaded as what it would relieve, and the request must be "+
			"refused rather than served anywhere", got.URL)
	} else if err != policy.ErrSplitGuardBlocked {
		t.Errorf("err = %v, want ErrSplitGuardBlocked", err)
	}
}

func TestTransientFallbackOffByDefault(t *testing.T) {
	p := newTransientPolicy(t, 0) // the shipped default
	f, u := guardBlockedFleet(t, p, true)

	if got, err := p.Select(context.Background(), f, req(u)); err == nil {
		t.Errorf("routed to %s with transient fallback off; the guard's rejection is final "+
			"unless the feature is asked for", got.URL)
	} else if err != policy.ErrSplitGuardBlocked {
		t.Errorf("err = %v, want ErrSplitGuardBlocked", err)
	}
}

// TestTransientFallbackAboveGuardIsRejectedAsConfig: a threshold at or above
// the guard is a STRICTER bar than the guard it relaxes, so it could never
// admit anything. Silently accepting it would leave an operator who asked for a
// fallback getting rejections with no way to tell why.
func TestTransientFallbackAboveGuardIsRejectedAsConfig(t *testing.T) {
	for _, v := range []float64{0.20, 0.5, 1.5, -0.1} {
		cfg := Config{SplitGuard: 0.20, TransientFallback: v}.withDefaults()
		if cfg.TransientFallback != 0 {
			t.Errorf("TransientFallback %v against a guard of 0.20 survived as %v; it is not a "+
				"looser bar and must be normalised to off", v, cfg.TransientFallback)
		}
	}
	cfg := Config{SplitGuard: 0.20, TransientFallback: 0.05}.withDefaults()
	if cfg.TransientFallback != 0.05 {
		t.Errorf("a genuinely looser threshold was discarded: %v", cfg.TransientFallback)
	}
}
