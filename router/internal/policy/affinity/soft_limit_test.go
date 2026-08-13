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

// The soft limit turns --max-node-concurrency into a band. Below soft a holder
// is a plain cache hit; inside the band the router would rather spread, and if
// the guard refuses the spread the request comes BACK to the holder and queues
// there rather than spilling onto a backend with none of its KV.
//
// That last step is the whole feature. The alternative already exists —
// TransientFallback serves a non-holder without marking it — and the two pay
// opposite prices for the same relief: a stretch pays queueing and keeps the
// cache hit, a transient serve keeps the queue short and pays a full prefill.

const (
	testSoft = 16 // soft limit for these tests; testConcurrency (32) is hard
)

func newSoftPolicy(t *testing.T, soft int64, transient float64) *Policy {
	t.Helper()
	p, err := New(Config{
		NodeConcurrency:     testConcurrency,
		SoftNodeConcurrency: soft,
		SplitGuard:          DefaultSplitGuard,
		TransientFallback:   transient,
		TailTTL:             testTTL,
		Clock:               clock.NewFake(time.Time{}),
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// softFleet gives f[0] the prefix and parks it at `holder` in-flight. The other
// two sit at `others`.
func softFleet(t *testing.T, p *Policy, holder, others int64) ([]*registry.Backend, []kvcache.Unit) {
	t.Helper()
	f := fleet(t, 3)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	load(t, f[0], holder)
	load(t, f[1], others)
	load(t, f[2], others)
	return f, u
}

func TestBelowSoftIsStillAPlainCacheHit(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	f, u := softFleet(t, p, testSoft-1, 0) // holder just under soft

	decisionsBefore := decisions(t, "cache")
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] {
		t.Errorf("routed to %s, want the holder: below the soft limit nothing has changed", got.URL)
	}
	if d := decisions(t, "cache") - decisionsBefore; d != 1 {
		t.Errorf("decision=\"cache\" moved by %v, want 1", d)
	}
}

// TestPastSoftPrefersToSplit: inside the band, a backend far enough below the
// SOFT limit is worth a copy, and the request spreads rather than piling on.
func TestPastSoftPrefersToSplit(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	// Holder at soft; peers idle, so they clear soft*(1-0.20) = 12.8.
	f, u := softFleet(t, p, testSoft, 0)

	splitsBefore := counter(t, testPoolMetrics().Splits)
	softBefore := counter(t, testPoolMetrics().SoftBlocked)

	rr := req(u)
	got, err := p.Select(context.Background(), f, rr)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got == f[0] {
		t.Errorf("piled onto the holder at the soft limit with two idle backends available; the " +
			"soft limit exists to prefer spreading at exactly this point")
	}
	if d := counter(t, testPoolMetrics().Splits) - splitsBefore; d != 1 {
		t.Errorf("splits moved by %v, want 1", d)
	}
	if d := counter(t, testPoolMetrics().SoftBlocked) - softBefore; d != 1 {
		t.Errorf("router_cache_soft_blocked_total moved by %v, want 1: the trigger must be counted "+
			"whichever way the decision goes, or the gap to stretches means nothing", d)
	}
	// A split is a permanent copy, so it must mark.
	if d, ok := rr.PolicyState.(*decision); !ok || !d.mark {
		t.Errorf("PolicyState = %+v, want a marking decision", rr.PolicyState)
	}
}

// TestGuardRefusesTheSpreadAndTheRequestStretches is the feature: nothing is far
// enough below SOFT to be worth a copy, so the request queues on the holder.
func TestGuardRefusesTheSpreadAndTheRequestStretches(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	// Holder at soft (16), peers at 14: above soft*(1-0.20) = 12.8, so no copy
	// is worth making, but all three are far below the hard limit of 32.
	f, u := softFleet(t, p, testSoft, 14)

	stretchBefore := counter(t, testPoolMetrics().Stretches)
	decisionsBefore := decisions(t, "stretch")

	rr := req(u)
	got, err := p.Select(context.Background(), f, rr)
	if err != nil {
		t.Fatalf("Select refused with the holder at %d and a hard limit of %d: past soft the holder "+
			"can still take the request, and taking it is the point: %v", testSoft, testConcurrency, err)
	}
	if got != f[0] {
		t.Errorf("routed to %s, want the holder: when the guard refuses the spread the request must "+
			"come back to a backend that already has the KV, not spill onto one that does not", got.URL)
	}
	if d := counter(t, testPoolMetrics().Stretches) - stretchBefore; d != 1 {
		t.Errorf("router_cache_stretches_total moved by %v, want 1", d)
	}
	if d := decisions(t, "stretch") - decisionsBefore; d != 1 {
		t.Errorf("decision=\"stretch\" moved by %v, want 1: it is neither a cache hit nor a split, "+
			"and folding it into either breaks the reconciliation", d)
	}

	// It marks — but the backend already held the prefix, so nothing duplicates.
	before := p.tree.stats()
	p.Commit(got, rr)
	after := p.tree.stats()
	if after.AvgCopies > before.AvgCopies {
		t.Errorf("mean holders per block rose %.3f -> %.3f; a stretch routes to an EXISTING holder "+
			"and must not create a copy", before.AvgCopies, after.AvgCopies)
	}
}

// TestStretchPicksTheLeastLoadedHolder — and does not apply the guard among
// them. The guard prevents duplication, and there is none to prevent here.
func TestStretchPicksTheLeastLoadedHolder(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	f := fleet(t, 3)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	p.Commit(f[1], req(u)) // two holders
	load(t, f[0], 30)      // both past soft, both under the hard limit of 32
	load(t, f[1], 20)
	load(t, f[2], 20) // the non-holder is no further below soft than f[1]

	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the holder on 20 rather than the one on 30", got.URL)
	}
}

// TestAllHoldersAtHardIsNotAStretch. At the hard limit a backend is saturated
// and leaves the usable set, so there is nothing to stretch onto — the request
// takes the ordinary guarded path and, if that refuses, the ordinary 429. The
// answer is ErrSplitGuardBlocked and NOT ErrAllBackendsSaturated: capacity
// existed on the non-holders and the guard declined to spend it on a copy.
func TestAllHoldersAtHardIsNotAStretch(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	f, u := softFleet(t, p, testConcurrency, 28) // holder saturated; peers above the guard

	stretchBefore := counter(t, testPoolMetrics().Stretches)
	got, err := p.Select(context.Background(), f, req(u))
	if err == nil {
		t.Fatalf("routed to %s with the only holder at the hard limit and every peer above the "+
			"guard threshold", got.URL)
	}
	if err != policy.ErrSplitGuardBlocked {
		t.Errorf("err = %v, want ErrSplitGuardBlocked: the fleet was not saturated, the guard "+
			"refused to duplicate", err)
	}
	if d := counter(t, testPoolMetrics().Stretches) - stretchBefore; d != 0 {
		t.Errorf("stretches moved by %v; a backend at the hard limit is saturated and must never "+
			"be stretched onto", d)
	}
}

// TestSoftTriggeredSplitMeasuresAgainstSoft. The reference decides where a copy
// may land, and getting it wrong here is not a near miss: measured against the
// HARD limit, a split inside the band could land on a backend BUSIER than the
// holder it was made to relieve — a permanent duplicate bought for nothing.
func TestSoftTriggeredSplitMeasuresAgainstSoft(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	// Holder at soft (16); peers at 15. Against soft the threshold is 12.8 and
	// nothing qualifies. Against the hard limit it would be 25.6 and a peer on
	// 15 — busier than nothing, but no relief at all — would take a copy.
	f, u := softFleet(t, p, testSoft, 15)

	splitsBefore := counter(t, testPoolMetrics().Splits)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] {
		t.Errorf("split onto %s at 15 against a holder at %d: measured against the soft limit "+
			"nothing here is worth a copy, and the request belongs back on the holder", got.URL, testSoft)
	}
	if d := counter(t, testPoolMetrics().Splits) - splitsBefore; d != 0 {
		t.Errorf("splits moved by %v, want 0: the guard reference must be the soft limit, not the "+
			"hard one", d)
	}
}

// TestStretchBeatsTransientFallback pins the precedence. Both resolve the same
// guard block; the stretch keeps the KV and the transient serve throws it away,
// so when both are configured the cache-preserving one wins.
func TestStretchBeatsTransientFallback(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0.05)
	f, u := softFleet(t, p, testSoft, 14) // inside the band, guard refuses

	overflowsBefore := counter(t, testPoolMetrics().Overflows)
	stretchBefore := counter(t, testPoolMetrics().Stretches)

	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] {
		t.Errorf("routed to %s; with both mechanisms on, the one that keeps the request on a "+
			"backend already holding the prefix must decide", got.URL)
	}
	if d := counter(t, testPoolMetrics().Overflows) - overflowsBefore; d != 0 {
		t.Errorf("overflows moved by %v; the transient fallback must not fire when a stretch was "+
			"available, or the two confound each other in every measurement", d)
	}
	if d := counter(t, testPoolMetrics().Stretches) - stretchBefore; d != 1 {
		t.Errorf("stretches moved by %v, want 1", d)
	}
}

func TestSoftLimitOffByDefault(t *testing.T) {
	p := newSoftPolicy(t, 0, 0) // the shipped default
	f, u := softFleet(t, p, testSoft, 0)

	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] {
		t.Errorf("routed to %s with no soft limit set; a holder below the hard limit is a cache "+
			"hit and the fleet's behaviour must be byte-identical to before the flag existed", got.URL)
	}
}

// TestSoftLimitNormalisedWhenItCannotBind: a soft limit at or above the hard
// one names a state a backend can never be in, and with no hard limit there is
// no band for it to floor. Accepting either silently would leave an operator
// reading a flag they set as a flag that is working.
func TestSoftLimitNormalisedWhenItCannotBind(t *testing.T) {
	for _, tc := range []struct {
		name       string
		hard, soft int64
	}{
		{"soft equals hard", 32, 32},
		{"soft above hard", 32, 48},
		{"negative", 32, -1},
		{"no hard limit", 0, 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{NodeConcurrency: tc.hard, SoftNodeConcurrency: tc.soft}.withDefaults()
			if cfg.SoftNodeConcurrency != 0 {
				t.Errorf("soft %d against hard %d survived as %d, want 0",
					tc.soft, tc.hard, cfg.SoftNodeConcurrency)
			}
		})
	}
	cfg := Config{NodeConcurrency: 32, SoftNodeConcurrency: 16}.withDefaults()
	if cfg.SoftNodeConcurrency != 16 {
		t.Errorf("a usable soft limit was discarded: %d", cfg.SoftNodeConcurrency)
	}
}

// TestStretchDoesNotAnchorOnAShallowPrefix. A shallow anchor means the
// request's own holders are gone and only a shared ancestor is left; serving
// there and marking is what puts a session's private tail on every backend. The
// soft limit must not become a new route onto that path.
func TestStretchDoesNotAnchorOnAShallowPrefix(t *testing.T) {
	p := newSoftPolicy(t, testSoft, 0)
	f := fleet(t, 3)
	for _, b := range f {
		p.AddBackend(b)
	}
	// f[0] holds the shared ancestor only; nobody holds the deeper path.
	p.Commit(f[0], req(units(1, 2)))
	p.Commit(f[1], req(units(1, 2, 3, 4)))
	load(t, f[0], testSoft) // past soft
	load(t, f[1], testConcurrency)
	load(t, f[2], testConcurrency)

	stretchBefore := counter(t, testPoolMetrics().Stretches)
	_, _ = p.Select(context.Background(), f, req(units(1, 2, 3, 4)))
	if d := counter(t, testPoolMetrics().Stretches) - stretchBefore; d != 0 {
		t.Errorf("stretches moved by %v on a shallow anchor; under LadderStrict a shared ancestor "+
			"is not a cache hit, and it must not become one by way of the soft limit", d)
	}
}
