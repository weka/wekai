package affinity

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// loadKnob drives a backend's in-flight count to an exact value, up OR down.
// The lease primitive is the only writer of in-flight (LB-1), so a test cannot
// assign the number; it has to hold the leases and give them back. The shared
// load() helper only ever adds, which is enough for the ladder tests but not
// for these — the refused signal is keyed to load falling.
type loadKnob struct {
	b    *registry.Backend
	held []*lease.Lease
}

func (k *loadKnob) set(t *testing.T, n int64) {
	t.Helper()
	for int64(len(k.held)) < n {
		k.held = append(k.held, lease.Acquire(k.b))
	}
	for int64(len(k.held)) > n {
		k.held[len(k.held)-1].Release()
		k.held = k.held[:len(k.held)-1]
	}
	if got := k.b.Inflight(); got != n {
		t.Fatalf("backend %s in-flight = %d, want %d", k.b.URL, got, n)
	}
}

// The router has one routing flow. What varies between deployments is which
// signals are enabled, so these tests cover the seam rather than the ladder:
// what each signal decides, and that the flow honours it the same way whichever
// one fired.

// TestRefusedSignalIsTheUltimateOne covers the only signal that is always on,
// and the only one that is ground truth rather than a guess. Nothing else in
// the router knows a vLLM's real capacity: a concurrency limit is a guess at
// --max-num-seqs, an imbalance is a heuristic, but a 429 is the engine
// declining the work.
func TestRefusedSignalIsTheUltimateOne(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RefusalTTL: time.Second, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cands := fleet(t, 2)
	b := cands[0]
	load(t, b, 7)

	if hit, _ := p.refused.saturated(b, loadView{}); hit {
		t.Fatal("a backend that has not refused anything is saturated")
	}

	p.OnRefused(b)
	hit, ref := p.refused.saturated(b, loadView{})
	if !hit {
		t.Fatal("a backend that answered 429 is not treated as saturated")
	}
	// The guard's reference is what the backend was CARRYING when it refused —
	// observed, not configured. This is the number Anton's rule is written
	// against ("we already have 32 in the air... if candidate has 30, we are
	// not splitting onto it"), and no config value can supply it.
	if ref != 7 {
		t.Errorf("guard reference = %d, want the 7 in flight at the moment of refusal", ref)
	}

	// A success means it is taking work again; waiting out the TTL after that
	// would keep a healthy backend out of its own prefixes for no reason.
	p.OnAccepted(b)
	if hit, _ := p.refused.saturated(b, loadView{}); hit {
		t.Error("a backend that served a request is still latched as refused")
	}

	// And absent any success, the latch expires on its own, so a backend that
	// goes quiet cannot be excluded forever.
	p.OnRefused(b)
	clk.Advance(2 * time.Second)
	if hit, _ := p.refused.saturated(b, loadView{}); hit {
		t.Error("the refusal latch outlived its TTL")
	}
}

// TestRefusalSplitsRatherThanReturningToTheFullBackend is the ultimate signal
// end to end, with no concurrency limit configured at all: the flow learns the
// holder is full only because the backend said so, and answers by splitting.
func TestRefusalSplitsRatherThanReturningToTheFullBackend(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RefusalTTL: time.Second, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cands := fleet(t, 2)

	// The prefix lands on one backend and is committed there.
	holder := route(t, p, cands, req(units(1, 2, 3)))
	other := without(cands, holder)[0]

	// Same prefix again: affinity keeps it on the holder.
	if got := route(t, p, cands, req(units(1, 2, 3))); got != holder {
		t.Fatalf("second request went to %s, want the holder %s", got.URL, holder.URL)
	}

	// The holder now refuses. Nothing else changed — no limit is configured,
	// its in-flight count is unremarkable — so only the 429 can teach the flow
	// anything.
	load(t, holder, 4)
	p.OnRefused(holder)

	beforeSplits := counter(t, metrics.CacheSplits)
	rr := req(units(1, 2, 3))
	got, err := p.Select(context.Background(), cands, rr)
	if err != nil {
		t.Fatalf("Select after a refusal: %v", err)
	}
	if got != other {
		t.Errorf("Select returned %s, want the split target %s: a refused holder must not "+
			"receive the request again", got.URL, other.URL)
	}
	if d := counter(t, metrics.CacheSplits) - beforeSplits; d != 1 {
		t.Errorf("%v splits recorded, want 1: growing the holder set is the only response to a "+
			"refusal that keeps affinity", d)
	}
}

// TestImbalanceRatioArithmetic pins the one number the imbalance signal takes.
// It is expressed against the HIGHER side — (load-min)/load > ratio — which is
// what lets a single value work at any fleet size, unlike the absolute-plus-
// relative pair it replaces from the retired prefix-cache-aware policy.
func TestImbalanceRatioArithmetic(t *testing.T) {
	sig := imbalanceSignal{ratio: 0.5}
	for _, tc := range []struct {
		load, min int64
		want      bool
	}{
		{load: 10, min: 4, want: true},   // gap 6 of 10 = 60% > 50%
		{load: 10, min: 5, want: false},  // gap 5 of 10 = exactly 50%, not over
		{load: 10, min: 6, want: false},  // gap 4 of 10 = 40%
		{load: 2, min: 0, want: true},    // gap 2 of 2 = 100%
		{load: 0, min: 0, want: false},   // an idle backend is never "too loaded"
		{load: 4, min: 10, want: false},  // below the minimum cannot happen, but must not panic
		{load: 100, min: 49, want: true}, // scale-free: same shape at 10x
	} {
		b := fleet(t, 1)[0]
		load(t, b, tc.load)
		got, ref := sig.saturated(b, loadView{minInflight: tc.min})
		if got != tc.want {
			t.Errorf("load=%d min=%d: saturated=%v, want %v", tc.load, tc.min, got, tc.want)
		}
		if got && ref != tc.load {
			t.Errorf("load=%d: guard reference = %d, want the backend's own load", tc.load, ref)
		}
	}
}

// TestEveryBackendSaturatedRejectsDistinctly: when no backend is usable there
// is no capacity at all, which is a different answer from "capacity exists and
// the guard will not spend it on a duplicate". The gateway maps them to
// different error codes, so the flow must not conflate them.
func TestEveryBackendSaturatedRejectsDistinctly(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 2)
	for _, b := range cands {
		load(t, b, testConcurrency)
	}

	_, err := p.Select(context.Background(), cands, req(units(1, 2)))
	if !errors.Is(err, policy.ErrAllBackendsSaturated) {
		t.Fatalf("Select = %v, want ErrAllBackendsSaturated", err)
	}
}

// TestSignalsApplyToRoutesWithNoPrefix is a regression test.
//
// A request with no routable prefix — an embeddings call, an unparseable body —
// has nothing to be affine to and goes straight to the selector. That early
// return used to happen BEFORE the signals were consulted, so such a request
// was routed over the unfiltered candidate list and could be sent to a backend
// that had already refused work. Having no prefix is not a reason to ignore
// capacity.
func TestSignalsApplyToRoutesWithNoPrefix(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RefusalTTL: time.Second, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cands := fleet(t, 2)
	full, spare := cands[0], cands[1]
	p.OnRefused(full)

	rr := &policy.RoutingRequest{Model: "m"} // no Units
	for range 20 {
		got, err := p.Select(context.Background(), cands, rr)
		if err != nil {
			t.Fatalf("Select with no units: %v", err)
		}
		if got != spare {
			t.Fatalf("a prefix-less request was routed to %s, which had refused work; "+
				"want the usable backend %s", got.URL, spare.URL)
		}
	}

	// With every backend refused, it must reject rather than pick one anyway.
	p.OnRefused(spare)
	if _, err := p.Select(context.Background(), cands, rr); !errors.Is(err, policy.ErrAllBackendsSaturated) {
		t.Errorf("Select = %v, want ErrAllBackendsSaturated", err)
	}
}

// TestSaturatedBackendIsNotAnchoredOn: a holder that a signal has excluded must
// not anchor the request, or the flow would route to it anyway through tier 1.
func TestSaturatedBackendIsNotAnchoredOn(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 3)

	holder := route(t, p, cands, req(units(9, 8, 7)))
	load(t, holder, testConcurrency) // at the configured limit

	rr := req(units(9, 8, 7))
	got, err := p.Select(context.Background(), cands, rr)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got == holder {
		t.Error("routed to the saturated holder: a signal-excluded backend must not anchor")
	}
	var b *registry.Backend = got
	if b == nil {
		t.Fatal("Select returned no backend and no error")
	}
}

// TestAllHoldersTriedBeforeSplitting pins the ordering Anton specified: a
// refusal from one holder must send the request to ANOTHER HOLDER if there is
// one, not straight to a split.
//
// The distinction is the whole point of the tree. Splitting marks a backend
// that does not hold this prefix, so it costs a permanent extra copy of the KV;
// failing over to a backend that already holds it costs nothing. A split is
// only correct once no holder can serve.
func TestAllHoldersTriedBeforeSplitting(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RefusalTTL: time.Minute, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cands := fleet(t, 3)

	// Two of the three become holders of the same prefix: one by first touch,
	// the second by a split once the first refuses.
	first := route(t, p, cands, req(units(4, 5, 6)))
	firstLoad := &loadKnob{b: first}
	firstLoad.set(t, 5)
	p.OnRefused(first)
	second, err := p.Select(context.Background(), cands, req(units(4, 5, 6)))
	if err != nil {
		t.Fatalf("Select after the first refusal: %v", err)
	}
	if second == first {
		t.Fatal("a refused holder was chosen again")
	}
	p.Commit(second, req(units(4, 5, 6)))
	third := without(without(cands, first), second)[0]

	// The second holder now refuses too, at a HIGHER in-flight than the first.
	// The first has meanwhile dropped below the level it refused at, so its
	// refusal no longer describes it and it is a holder that can serve again.
	(&loadKnob{b: second}).set(t, 10)
	p.OnRefused(second)
	// The first has since finished work and dropped below the level it refused
	// at, so its refusal no longer describes it.
	firstLoad.set(t, 2)

	beforeSplits := counter(t, metrics.CacheSplits)
	got, err := p.Select(context.Background(), cands, req(units(4, 5, 6)))
	if err != nil {
		t.Fatalf("Select with one usable holder left: %v", err)
	}
	if got == third {
		t.Error("split onto a non-holder while a holder was usable: that is a permanent " +
			"extra copy bought for nothing")
	}
	if got != first {
		t.Errorf("Select returned %s, want the recovered holder %s", got.URL, first.URL)
	}
	if d := counter(t, metrics.CacheSplits) - beforeSplits; d != 0 {
		t.Errorf("%v splits recorded while a holder was still usable, want 0", d)
	}
}

// TestRefusalClearsWhenLoadDrops covers the mechanism that bounds retry
// multiplication under saturation, and the reason the latch is keyed to
// in-flight rather than to a timer.
//
// A backend that refuses at N is skipped by every subsequent failover while it
// is still carrying N or more — it has already answered that question, and
// asking again is a wasted round trip at exactly the moment the fleet cannot
// afford one. The instant its load falls below N it is usable again, with
// nothing needing to expire.
func TestRefusalClearsWhenLoadDrops(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RefusalTTL: time.Hour, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	b := fleet(t, 1)[0]
	knob := &loadKnob{b: b}

	knob.set(t, 12)
	p.OnRefused(b)
	if hit, _ := p.refused.saturated(b, loadView{}); !hit {
		t.Fatal("a backend still carrying what it refused at is not treated as saturated")
	}

	// Still at the refusal level: unchanged.
	knob.set(t, 12)
	if hit, _ := p.refused.saturated(b, loadView{}); !hit {
		t.Error("a backend back at the level it refused at should still be skipped")
	}

	// One slot freed. The refusal described 12 in flight; it no longer applies,
	// and the TTL (an hour here) is irrelevant to that.
	knob.set(t, 11)
	if hit, _ := p.refused.saturated(b, loadView{}); hit {
		t.Error("a backend that has freed a slot since refusing is still being skipped; " +
			"recovery must not wait on a timer")
	}
}
