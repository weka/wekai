package affinity

import (
	"context"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/registry"
)

// The imbalance signal has to distinguish fleets that are RATIO-IDENTICAL.
//
// 20,20,20,0 and 1,1,1,0 have the same minimum, the same proportions and the
// same mean-to-load relationship. A purely relative test calls them equally
// imbalanced, so it fires on both — and firing on 1,1,1,0 moves a prefix to
// relieve a gap of one request, which costs more than the gap does. Only an
// absolute term separates them.
func TestImbalanceNeedsBothAProportionAndARealGap(t *testing.T) {
	for _, tc := range []struct {
		name    string
		loads   []int64
		subject int // index of the backend being judged
		want    bool
	}{
		{"a busy fleet beside an idle backend rebalances",
			[]int64{20, 20, 20, 0}, 0, true},
		{"the same shape at trivial load does not",
			[]int64{1, 1, 1, 0}, 0, false},
		{"nor does it just below the in-flight floor",
			[]int64{7, 7, 7, 0}, 0, false},
		{"and it does at the floor",
			[]int64{8, 8, 8, 0}, 0, true},
		{"an idle backend is never itself saturated",
			[]int64{20, 20, 20, 0}, 3, false},
		{"an evenly loaded fleet is left alone",
			[]int64{8, 8, 8, 8}, 0, false},
		{"a backend under the floor is never rebalanced from, however lopsided",
			[]int64{7, 0, 0, 0}, 0, false},
		{"a gap too small in proportion is left alone",
			[]int64{10, 9, 8, 7}, 0, false},
		{"a large absolute gap still needs the proportion",
			[]int64{30, 30, 30, 20}, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sig := imbalanceSignal{ratio: 0.5}
			be := fleet(t, len(tc.loads))
			view := loadView{minInflight: tc.loads[0]}
			for i, n := range tc.loads {
				load(t, be[i], n)
				if n < view.minInflight {
					view.minInflight = n
				}
			}
			got, _ := sig.saturated(be[tc.subject], view)
			if got != tc.want {
				t.Errorf("loads %v, judging index %d (inflight %d, fleet min %d): "+
					"saturated=%v, want %v", tc.loads, tc.subject,
					be[tc.subject].Inflight(), view.minInflight, got, tc.want)
			}
		})
	}
}

// The point of rebalancing is that idle capacity gets used. A backend at zero
// is never flagged, so it stays in the usable set and takes the split — which
// is how work actually reaches it.
func TestImbalanceRoutesToTheIdleBackend(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RebalanceRatio: 0.5, SplitGuard: DefaultSplitGuard,
		TailTTL: testTTL, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	be := fleet(t, 4)

	holder := route(t, p, be, req(units(1, 2, 3)))
	var idle *registry.Backend
	for _, b := range be {
		switch {
		case b == holder:
			load(t, b, 20)
		case idle == nil:
			idle = b // left at zero
		default:
			load(t, b, 20)
		}
	}

	got, err := p.Select(context.Background(), be, req(units(1, 2, 3)))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != idle {
		t.Errorf("selected %s (inflight %d), want the idle backend %s; rebalancing "+
			"exists so that idle capacity is used", got.URL, got.Inflight(), idle.URL)
	}
}

// And the inverse: at trivial load the holder keeps its own prefix, because
// there is nothing to relieve and a copy would be pure loss.
func TestNoRebalanceAtTrivialLoad(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{RebalanceRatio: 0.5, SplitGuard: DefaultSplitGuard,
		TailTTL: testTTL, Clock: clk}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	be := fleet(t, 4)

	holder := route(t, p, be, req(units(1, 2, 3)))
	for _, b := range be {
		if b == holder {
			load(t, b, 2) // two in flight, three idle siblings
		}
	}

	got, err := p.Select(context.Background(), be, req(units(1, 2, 3)))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != holder {
		t.Errorf("selected %s, want the holder %s; at two in-flight nothing is under "+
			"pressure, so copying the prefix costs more than the imbalance does",
			got.URL, holder.URL)
	}
}
