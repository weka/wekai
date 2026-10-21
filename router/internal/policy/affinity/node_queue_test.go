package affinity

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// histogramSampleCount reads a resolved HistogramVec child's own sample count,
// the way gateway_test.go's observedFractionStats does — testutil.CollectAndCount
// would count SERIES, not observations, and the series here already exists
// from ForPool's own resolution at construction time regardless of whether
// anything was ever observed into it.
func histogramSampleCount(t *testing.T, o prometheus.Observer) uint64 {
	t.Helper()
	var pb dto.Metric
	if err := o.(prometheus.Metric).Write(&pb); err != nil {
		t.Fatalf("write histogram: %v", err)
	}
	return pb.GetHistogram().GetSampleCount()
}

// saturate pushes a backend's in-flight to exactly n via the lease primitive —
// the only legitimate mutator (LB-6) — so the concurrency signal sees it as
// saturated once n reaches the configured limit.
func saturate(t *testing.T, b *registry.Backend, n int64) {
	t.Helper()
	load(t, b, n)
}

// selectSaturated runs Select and requires it to fail, returning the error for
// the caller to inspect with errors.As.
func selectSaturated(t *testing.T, p *Policy, cands []*registry.Backend, rr *policy.RoutingRequest) error {
	t.Helper()
	b, err := p.Select(context.Background(), cands, rr)
	if err == nil {
		t.Fatalf("Select succeeded with backend %v, want a saturation error", b)
	}
	if !errors.Is(err, policy.ErrAllBackendsSaturated) {
		t.Fatalf("Select error = %v, want it to satisfy errors.Is(ErrAllBackendsSaturated)", err)
	}
	return err
}

// TestNodeQueuePicksTheRightHolderUnderAffinity is the central case: two
// backends hold the same deepest run, one is latched refused (ground truth:
// the backend itself said no) and the other is saturated only by the
// concurrency signal (a router-side guess a release can undo). The queue
// target must be the concurrency-saturated holder, never the refused one —
// waiting on a refused backend would not be fixed by any release.
func TestNodeQueuePicksTheRightHolderUnderAffinity(t *testing.T) {
	p, _ := newTestPolicy(t)
	bs := fleet(t, 2)
	bConc, bRefused := bs[0], bs[1]

	rr := req(units(1, 2, 3))
	p.Commit(bConc, rr)
	p.Commit(bRefused, rr)

	saturate(t, bConc, testConcurrency)
	p.OnRefused(bRefused)

	err := selectSaturated(t, p, bs, req(units(1, 2, 3)))
	var sat *policy.SaturatedError
	if !errors.As(err, &sat) {
		t.Fatalf("error = %v, want a *policy.SaturatedError", err)
	}
	if sat.Preferred != bConc {
		t.Errorf("Preferred = %v, want the concurrency-saturated holder %v", sat.Preferred, bConc)
	}
	if sat.Limit != testConcurrency {
		t.Errorf("Limit = %d, want %d", sat.Limit, testConcurrency)
	}
}

// TestNodeQueueWithoutUnitsPicksAnyWaitableBackend covers the no-routable-
// prefix branch (an embeddings call, say): with nothing to be affine to, the
// queue target is just the fallback selector's pick among the waitable set —
// the same rule Select itself uses for "no units" among the usable set.
func TestNodeQueueWithoutUnitsPicksAnyWaitableBackend(t *testing.T) {
	p, _ := newTestPolicy(t)
	bs := fleet(t, 1)
	b := bs[0]
	saturate(t, b, testConcurrency)

	err := selectSaturated(t, p, bs, &policy.RoutingRequest{Model: "m"}) // no Units
	var sat *policy.SaturatedError
	if !errors.As(err, &sat) {
		t.Fatalf("error = %v, want a *policy.SaturatedError", err)
	}
	if sat.Preferred != b {
		t.Errorf("Preferred = %v, want %v", sat.Preferred, b)
	}
}

// TestNodeQueueOffWhenNodeConcurrencyUnset guards the "default off" contract:
// with NodeConcurrency == 0 the concurrency signal is never constructed, so no
// backend can ever be waitable and a plain ErrAllBackendsSaturated must come
// back even though every backend IS saturated (by the refused signal here).
func TestNodeQueueOffWhenNodeConcurrencyUnset(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{Clock: clk, TailTTL: testTTL}, nil) // NodeConcurrency left at 0
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	bs := fleet(t, 1)
	p.OnRefused(bs[0])

	gotErr := selectSaturated(t, p, bs, req(units(1)))
	var sat *policy.SaturatedError
	if errors.As(gotErr, &sat) {
		t.Fatalf("error = %v (Preferred=%v), want a plain ErrAllBackendsSaturated with "+
			"NodeConcurrency unset", gotErr, sat.Preferred)
	}
}

// TestNodeQueueNilWhenHolderOnlyRefusedLatched: the request's one candidate is
// saturated purely by the refused signal (ground truth from the backend
// itself) and the concurrency signal never fires on it at all, so it is not
// waitable and there is nothing to queue on.
func TestNodeQueueNilWhenHolderOnlyRefusedLatched(t *testing.T) {
	p, _ := newTestPolicy(t)
	bs := fleet(t, 1)
	p.Commit(bs[0], req(units(1, 2)))
	p.OnRefused(bs[0])

	gotErr := selectSaturated(t, p, bs, req(units(1, 2)))
	var sat *policy.SaturatedError
	if errors.As(gotErr, &sat) {
		t.Fatalf("error = %v (Preferred=%v), want a plain ErrAllBackendsSaturated: the only "+
			"holder is refused-latched, not concurrency-saturated", gotErr, sat.Preferred)
	}
}

// TestNodeQueueNilWhenOnlyShallowEligible: the deepest marked run (all 3
// blocks) is held only by a refused-latched backend; a waitable
// (concurrency-saturated) backend holds just a shallower ancestor (the first
// 2 blocks). Queueing on the shallow holder would not be a cache hit for the
// deepest run — Select's own ladder treats exactly this as a shallow anchor,
// not tier 1 — so queueTarget must decline it too, the same way it would hit
// ErrSplitGuardBlocked on wake rather than a clean cache hit.
func TestNodeQueueNilWhenOnlyShallowEligible(t *testing.T) {
	p, _ := newTestPolicy(t)
	bs := fleet(t, 2)
	bDeep, bShallow := bs[0], bs[1]

	p.Commit(bDeep, req(units(1, 2, 3))) // full 3-block run, bDeep only
	p.Commit(bShallow, req(units(1, 2))) // forces a split: bShallow marks only the 2-block ancestor

	saturate(t, bShallow, testConcurrency)
	p.OnRefused(bDeep)

	gotErr := selectSaturated(t, p, bs, req(units(1, 2, 3)))
	var sat *policy.SaturatedError
	if errors.As(gotErr, &sat) {
		t.Fatalf("error = %v (Preferred=%v), want a plain ErrAllBackendsSaturated: the only "+
			"waitable backend holds a shallower ancestor, not the deepest run", gotErr, sat.Preferred)
	}
}

// TestNodeQueueNoSideEffects asserts queueTarget is purely a forecast: it must
// not touch PolicyState (Select's own later re-run owns that) or any pool
// decision/anchor metric, which belong to a real routing decision and would
// otherwise be double-counted or counted for a request that never actually
// routed anywhere.
func TestNodeQueueNoSideEffects(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	const pool = "node-queue-side-effect-test"
	p, err := New(Config{
		NodeConcurrency: testConcurrency,
		SplitGuard:      DefaultSplitGuard,
		TailTTL:         testTTL,
		Clock:           clk,
		PoolName:        pool,
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	bs := fleet(t, 2)
	bConc, bRefused := bs[0], bs[1]
	rr := req(units(10, 11, 12))
	p.Commit(bConc, rr)
	p.Commit(bRefused, rr)
	saturate(t, bConc, testConcurrency)
	p.OnRefused(bRefused)

	anchorChild := metrics.CacheAnchorBlocks.WithLabelValues(pool)
	decisionsBefore := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues(pool, "cache"))
	anchorCountBefore := histogramSampleCount(t, anchorChild)

	fresh := req(units(10, 11, 12))
	_, gotErr := p.Select(context.Background(), bs, fresh)
	var sat *policy.SaturatedError
	if !errors.As(gotErr, &sat) || sat.Preferred == nil {
		t.Fatalf("error = %v, want a *policy.SaturatedError with a Preferred backend", gotErr)
	}

	if fresh.PolicyState != nil {
		t.Errorf("PolicyState = %#v, want nil: queueTarget must not set it, only a real decision may", fresh.PolicyState)
	}
	if got := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues(pool, "cache")); got != decisionsBefore {
		t.Errorf("router_route_decisions_total{decision=cache} = %v, want unchanged at %v", got, decisionsBefore)
	}
	if got := histogramSampleCount(t, anchorChild); got != anchorCountBefore {
		t.Errorf("router_cache_anchor_blocks sample count = %d, want unchanged at %d", got, anchorCountBefore)
	}
}
