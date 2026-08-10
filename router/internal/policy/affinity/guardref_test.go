package affinity

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// newGuardPolicy builds a policy with NO configured concurrency limit, which is
// how a fleet runs when the operator has not set --max-node-concurrency. The
// refusal signal is then the only source of a capacity reference, which is the
// configuration these tests are about.
func newGuardPolicy(t *testing.T) (*Policy, *clock.Fake) {
	t.Helper()
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{
		NodeConcurrency: 0,
		SplitGuard:      DefaultSplitGuard,
		TailTTL:         testTTL,
		Clock:           clk,
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p, clk
}

// A holder that refuses while carrying 48 must be relieved onto an idle
// sibling. This is the reported outage: eight backends, roughly a third of
// capacity in flight, and the router answering 429 split_guard_blocked while
// idle backends sat there.
//
// The guard's job is to refuse a copy that buys nothing, not to refuse every
// copy. A candidate at 16 against a holder that refused at 48 is exactly the
// case a split exists for.
func TestSplitsOntoIdleBackendWhenTheHolderRefuses(t *testing.T) {
	p, _ := newGuardPolicy(t)
	be := fleet(t, 8)

	// Backend 0 holds the prefix.
	rr := req(units(1, 2, 3))
	holder := route(t, p, be, rr)

	// It fills up and refuses at 48, while the rest of the fleet idles at 16 —
	// well inside a real vLLM's 48-slot ceiling.
	load(t, holder, 48)
	p.OnRefused(holder)
	for _, b := range be {
		if b != holder {
			load(t, b, 16)
		}
	}

	got, err := p.Select(context.Background(), be, req(units(1, 2, 3)))
	if err != nil {
		t.Fatalf("Select rejected with the fleet at a third of capacity: %v\n"+
			"the holder refused at 48 and every sibling is at 16; a split is what "+
			"the guard is supposed to allow here", err)
	}
	if got == holder {
		t.Fatal("routed back to the backend that just refused")
	}
}

// A backend that refuses at a LOW in-flight — a vLLM out of KV cache rather
// than out of sequence slots — must not set the split threshold for prefixes it
// does not hold.
//
// This was the bug. The guard reference was reduced to one fleet-wide minimum
// across every saturated backend, so a single node latched at 2 gave a
// threshold of 1.6 for every request in the pool. Nothing could clear it, and
// the flow rejected with capacity everywhere.
func TestUnrelatedLowRefusalDoesNotBlockSplitting(t *testing.T) {
	p, _ := newGuardPolicy(t)
	be := fleet(t, 8)

	// A prefix held by backend 0, which then fills and refuses at 48.
	rr := req(units(1, 2, 3))
	holder := route(t, p, be, rr)
	load(t, holder, 48)
	p.OnRefused(holder)

	// Somewhere else in the pool, an unrelated backend refuses while carrying
	// almost nothing. It holds none of this prefix.
	var stranger *registry.Backend
	for _, b := range be {
		if b != holder {
			stranger = b
			break
		}
	}
	load(t, stranger, 2)
	p.OnRefused(stranger)

	// Everyone else is comfortably loaded but far from full.
	for _, b := range be {
		if b != holder && b != stranger {
			load(t, b, 16)
		}
	}

	got, err := p.Select(context.Background(), be, req(units(1, 2, 3)))
	if errors.Is(err, policy.ErrSplitGuardBlocked) {
		t.Fatal("one unrelated backend refusing at 2 in-flight blocked the split for a " +
			"prefix it does not hold; the guard must measure against this prefix's own " +
			"saturated holders, not the pool minimum")
	}
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got == holder || got == stranger {
		t.Fatalf("routed to a backend that had refused: %s", got.URL)
	}
}

// The guard still has to say no when a copy genuinely buys nothing: a candidate
// as loaded as the holder that refused is not relief, it is a second copy of
// the prefix at the same load.
func TestGuardStillRefusesACopyThatBuysNothing(t *testing.T) {
	p, _ := newGuardPolicy(t)
	be := fleet(t, 4)

	rr := req(units(1, 2, 3))
	holder := route(t, p, be, rr)

	// The holder refuses at 48 and every sibling is at 44 — inside the 20%
	// guard band, so a copy would land somewhere just as busy.
	load(t, holder, 48)
	p.OnRefused(holder)
	for _, b := range be {
		if b != holder {
			load(t, b, 44)
		}
	}

	_, err := p.Select(context.Background(), be, req(units(1, 2, 3)))
	if !errors.Is(err, policy.ErrSplitGuardBlocked) {
		t.Fatalf("expected the guard to refuse a copy onto a backend at 44 against a "+
			"holder that refused at 48, got err=%v", err)
	}
}

// When the holders are ABSENT rather than full — they left the fleet — there is
// nothing to relieve, so placing the prefix somewhere is relocation, not
// duplication, and the guard has nothing to protect.
func TestAbsentHoldersDoNotTriggerTheGuard(t *testing.T) {
	p, _ := newGuardPolicy(t)
	be := fleet(t, 4)

	rr := req(units(1, 2, 3))
	holder := route(t, p, be, rr)

	// The holder is gone. The survivors are evenly loaded, which is precisely
	// the shape the old minInflight+1 fallback rejected.
	var rest []*registry.Backend
	for _, b := range be {
		if b != holder {
			load(t, b, 16)
			rest = append(rest, b)
		}
	}

	if _, err := p.Select(context.Background(), rest, req(units(1, 2, 3))); err != nil {
		t.Fatalf("Select rejected a prefix whose holders have left the fleet: %v\n"+
			"nothing is being relieved, so there is no copy for the guard to prevent", err)
	}
}
