package affinity

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

const testConcurrency = 32

// fleet builds n real backends with no HTTP server, mirroring the helper the
// older cache policy's tests use. URLs sort lexicographically in index order.
func fleet(t *testing.T, n int) []*registry.Backend {
	t.Helper()
	r := registry.New(registry.Options{})
	out := make([]*registry.Backend, 0, n)
	for i := range n {
		b, err := r.Add(registry.Spec{URL: fmt.Sprintf("http://w%d:8000", i), Capacity: testConcurrency})
		if err != nil {
			t.Fatalf("add backend %d: %v", i, err)
		}
		b.SetHealth(registry.Healthy)
		out = append(out, b)
	}
	return out
}

func newTestPolicy(t *testing.T) (*Policy, *clock.Fake) {
	t.Helper()
	clk := clock.NewFake(time.Time{})
	p, err := New(Config{
		NodeConcurrency: testConcurrency,
		SplitGuard:      DefaultSplitGuard,
		TailTTL:         testTTL,
		Clock:           clk,
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p, clk
}

// units fabricates a request's blocks. Fixed token counts, as in tree_test.
func units(hashes ...uint64) []kvcache.Unit {
	u := make([]kvcache.Unit, len(hashes))
	for i, h := range hashes {
		u[i] = kvcache.Unit{Hash: h, Tokens: 256}
	}
	return u
}

func req(u []kvcache.Unit) *policy.RoutingRequest {
	return &policy.RoutingRequest{Units: u, Model: "m"}
}

// route runs one full Select/Commit cycle and reports the winner.
func route(t *testing.T, p *Policy, cands []*registry.Backend, rr *policy.RoutingRequest) *registry.Backend {
	t.Helper()
	b, err := p.Select(context.Background(), cands, rr)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	p.Commit(b, rr)
	return b
}

// load pins a backend at a given in-flight count via the lease primitive, which
// is the only thing permitted to move the counter (LB-6, enforced by
// router/hack's fence over test files too).
func load(t *testing.T, b *registry.Backend, n int64) {
	t.Helper()
	for range n {
		lease.Acquire(b)
	}
	if got := b.Inflight(); got != n {
		t.Fatalf("backend %s in-flight = %d, want %d", b.URL, got, n)
	}
}

// counter reads a counter's current value, for delta assertions. Collectors are
// package-level and shared across the whole test binary, so absolutes are
// meaningless here.
func counter(t *testing.T, c interface{ Write(*dto.Metric) error }) float64 {
	t.Helper()
	var m dto.Metric
	if err := c.Write(&m); err != nil {
		t.Fatalf("read counter: %v", err)
	}
	if m.Counter != nil {
		return m.Counter.GetValue()
	}
	return m.Gauge.GetValue()
}

func decisions(t *testing.T, label string) float64 {
	t.Helper()
	return counter(t, metrics.RouteDecisions.WithLabelValues(label))
}

// TestNewRequiresAConcurrencyLimit is the loud failure that replaces a silent
// one. With the shipped defaults every capacity source reads 1 (--backends
// carries no capacity, MaxInflightPerBackend defaults to 1, and
// Backend.Capacity clamps up to 1), so a policy that quietly accepted the
// default would compute its split guard against nonsense and nobody would know.
func TestNewRequiresAConcurrencyLimit(t *testing.T) {
	if _, err := New(Config{}, nil); !errors.Is(err, ErrNoConcurrencyLimit) {
		t.Fatalf("New with no concurrency limit: err = %v, want ErrNoConcurrencyLimit", err)
	}
	if _, err := New(Config{NodeConcurrency: 32}, nil); err != nil {
		t.Fatalf("New with a limit: %v", err)
	}
}

// TestSelectReturnsErrNoCandidatesOnEmptyInput: this policy never rejects.
// Admission belongs to the gateway, which 429s exactly when no backend is under
// its limit — so the only error here is the structural one every policy shares.
func TestSelectReturnsErrNoCandidatesOnEmptyInput(t *testing.T) {
	p, _ := newTestPolicy(t)
	if _, err := p.Select(context.Background(), nil, req(units(1, 2))); !errors.Is(err, policy.ErrNoCandidates) {
		t.Fatalf("err = %v, want ErrNoCandidates", err)
	}
}

// TestFirstRequestGoesByLoadAndEveryFollowingOneByCache is Anton's headline
// expectation: "a single request going via load-split, but then every following
// request will have cache path split based routing... so i would expect
// absolute majority to be routed by cache and not by load".
func TestFirstRequestGoesByLoadAndEveryFollowingOneByCache(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 4)

	beforeLoad := decisions(t, "load")
	beforeCache := decisions(t, "cache")

	first := route(t, p, cands, req(units(1, 2, 3)))
	if got := decisions(t, "load") - beforeLoad; got != 1 {
		t.Errorf("first request produced %v load decisions, want 1", got)
	}

	// Ten follow-up turns, each growing the conversation. Every one must go by
	// cache, to the same backend, no matter how large the request gets
	// relative to its shared prefix.
	u := units(1, 2, 3)
	for turn := range 10 {
		u = append(u, kvcache.Unit{Hash: uint64(500 + turn), Tokens: 256})
		got := route(t, p, cands, req(u))
		if got != first {
			t.Fatalf("turn %d routed to %s, want the pinned %s", turn, got.URL, first.URL)
		}
	}
	if got := decisions(t, "cache") - beforeCache; got != 10 {
		t.Errorf("%v cache decisions across 10 follow-up turns, want 10", got)
	}
	if got := decisions(t, "load") - beforeLoad; got != 1 {
		t.Errorf("%v load decisions total, want only the first request's", got)
	}
}

// TestAffinityHoldsWhenTheSharedPrefixIsATinyFractionOfTheRequest is the
// regression this whole policy exists to prevent. prefix-cache-candidates
// requires cached/total > 0.5 over the WHOLE current request, so this case —
// a small fixed shared prefix inside a large prompt — falls back to load, which
// is what produced the ~15% average predicted fraction in production.
func TestAffinityHoldsWhenTheSharedPrefixIsATinyFractionOfTheRequest(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 4)

	pinned := route(t, p, cands, req(units(1, 2)))

	long := units(1, 2)
	for i := range 200 {
		long = append(long, kvcache.Unit{Hash: uint64(9000 + i), Tokens: 256})
	}
	before := decisions(t, "cache")
	got := route(t, p, cands, req(long))

	if got != pinned {
		t.Errorf("a 2-of-202-block match routed to %s, want the holder %s", got.URL, pinned.URL)
	}
	if d := decisions(t, "cache") - before; d != 1 {
		t.Errorf("%v cache decisions, want 1 — a 1%% match is still a real cache signal", d)
	}
}

// TestSplitExtendsTheHolderSetWhenEveryHolderIsSaturated covers tier 2. The
// deployed policy abandons affinity entirely here and falls back to
// least-loaded of everything; this one grows the holder set so the next request
// has somewhere warm to go.
func TestSplitExtendsTheHolderSetWhenEveryHolderIsSaturated(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 4)

	holder := route(t, p, cands, req(units(1, 2, 3)))

	// The holder saturates and is therefore filtered out by the gateway, the
	// way candidates() does before any policy runs.
	load(t, holder, testConcurrency)
	available := without(cands, holder)

	beforeSplits := counter(t, metrics.CacheSplits)
	got := route(t, p, available, req(units(1, 2, 3)))

	if got == holder {
		t.Fatal("routed to the saturated holder, which was not even a candidate")
	}
	if d := counter(t, metrics.CacheSplits) - beforeSplits; d != 1 {
		t.Errorf("%v splits recorded, want 1", d)
	}

	// The split must have been recorded: with both backends available again,
	// the new one is now a legitimate holder.
	a := p.tree.walk(path{units: units(1, 2, 3), modelKey: modelKey("m")}, allSlots(p, cands))
	if !a.pool.Has(p.tree.slotOrCreate(got.URL)) {
		t.Error("the split target was not marked as a holder, so the split cannot converge")
	}
	if !a.pool.Has(p.tree.slotOrCreate(holder.URL)) {
		t.Error("the original holder lost its mark")
	}
}

// TestOverflowRoutesToIdleCapacityWithoutMarkingIt covers tier 3, the one
// deliberate divergence from the reference simulator.
//
// Every holder is saturated and every remaining backend sits inside the guard
// band — between 80% and 100% of the limit at a 20% guard. The simulator
// rejects here, which is exactly why its own verdict function reports MARGINAL:
// real idle slots go unused. This policy serves the request and, crucially,
// does NOT record a new holder, so the guard still does the job Anton wanted it
// for.
func TestOverflowRoutesToIdleCapacityWithoutMarkingIt(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 3)

	holder := route(t, p, cands, req(units(1, 2, 3)))
	load(t, holder, testConcurrency)
	available := without(cands, holder)

	// Both remaining backends are inside the guard band: loaded enough to fail
	// the split guard (>= 25.6 at limit 32, guard 20%) but still under the
	// limit, so the gateway hands them over as candidates.
	for _, b := range available {
		load(t, b, 30)
	}

	beforeOverflow := counter(t, metrics.CacheOverflows)
	beforeSplits := counter(t, metrics.CacheSplits)

	rr := req(units(1, 2, 3))
	got, err := p.Select(context.Background(), available, rr)
	if err != nil {
		t.Fatalf("Select rejected while idle capacity existed: %v", err)
	}
	p.Commit(got, rr)

	if d := counter(t, metrics.CacheOverflows) - beforeOverflow; d != 1 {
		t.Errorf("%v overflows recorded, want 1", d)
	}
	if d := counter(t, metrics.CacheSplits) - beforeSplits; d != 0 {
		t.Errorf("%v splits recorded during an overflow, want 0", d)
	}

	// The tree must be unchanged: the overflow target is not a holder.
	a := p.tree.walk(path{units: units(1, 2, 3), modelKey: modelKey("m")}, allSlots(p, cands))
	if a.pool.Has(p.tree.slotOrCreate(got.URL)) {
		t.Error("an overflow marked its target as a holder, defeating the guard")
	}
	if a.pool.Count() != 1 {
		t.Errorf("holder count = %d after an overflow, want the original 1", a.pool.Count())
	}
}

// TestSplitGuardBoundary pins the arithmetic exactly. The guard is the only
// thing standing between "grow the holder set under pressure" and "every
// backend eventually holds every prefix", so off-by-one here is not cosmetic.
func TestSplitGuardBoundary(t *testing.T) {
	// limit 32, guard 20% => threshold 25.6; a candidate qualifies while its
	// in-flight is strictly below that.
	cases := []struct {
		guard     float64
		inflight  int64
		wantSplit bool
	}{
		{DefaultSplitGuard, 25, true},  // just under 25.6
		{DefaultSplitGuard, 26, false}, // just over
		{DefaultSplitGuard, 0, true},
		{DefaultSplitGuard, 31, false}, // under the limit but deep in the guard band
		{0.5, 15, true},                // threshold 16
		{0.5, 16, false},
		{0.99, 0, true},  // threshold 0.32: only a fully idle node qualifies
		{0.99, 1, false}, //
	}

	for _, tc := range cases {
		clk := clock.NewFake(time.Time{})
		p, err := New(Config{
			NodeConcurrency: testConcurrency,
			SplitGuard:      tc.guard,
			TailTTL:         testTTL,
			Clock:           clk,
		}, nil)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		cands := fleet(t, 2)
		holder := route(t, p, cands, req(units(1, 2)))
		load(t, holder, testConcurrency)
		other := without(cands, holder)[0]
		load(t, other, tc.inflight)

		beforeSplits := counter(t, metrics.CacheSplits)
		beforeOverflow := counter(t, metrics.CacheOverflows)
		rr := req(units(1, 2))
		if _, err := p.Select(context.Background(), []*registry.Backend{other}, rr); err != nil {
			t.Fatalf("guard=%v inflight=%d: %v", tc.guard, tc.inflight, err)
		}
		gotSplit := counter(t, metrics.CacheSplits)-beforeSplits == 1
		gotOverflow := counter(t, metrics.CacheOverflows)-beforeOverflow == 1

		if gotSplit != tc.wantSplit {
			t.Errorf("guard=%v inflight=%d: split=%v, want %v", tc.guard, tc.inflight, gotSplit, tc.wantSplit)
		}
		if gotOverflow == tc.wantSplit {
			t.Errorf("guard=%v inflight=%d: exactly one of split/overflow must fire (split=%v overflow=%v)",
				tc.guard, tc.inflight, gotSplit, gotOverflow)
		}
	}
}

// TestCommitDoesNotMarkAfterAnOverflowEvenIfCalled guards the Select/Commit
// contract directly, independent of the tier that produced it: the no-mark bit
// travels on the request, and Commit must honour it.
func TestCommitDoesNotMarkAfterAnOverflowEvenIfCalled(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 2)

	rr := req(units(7, 8))
	rr.PolicyState = noMarkDecision
	p.Commit(cands[0], rr)

	if a := p.tree.walk(path{units: units(7, 8), modelKey: modelKey("m")}, allSlots(p, cands)); !a.pool.Empty() {
		t.Fatal("Commit marked a backend for a request flagged as an overflow")
	}

	rr.PolicyState = markDecision
	p.Commit(cands[0], rr)
	if a := p.tree.walk(path{units: units(7, 8), modelKey: modelKey("m")}, allSlots(p, cands)); a.pool.Empty() {
		t.Fatal("Commit failed to mark a normal request")
	}
}

// TestCommitMarksWhenSelectNeverRan: a nil PolicyState must mark, because
// losing affinity silently is worse than an extra holder.
func TestCommitMarksWhenSelectNeverRan(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 2)

	rr := req(units(11, 12))
	p.Commit(cands[0], rr)

	if a := p.tree.walk(path{units: units(11, 12), modelKey: modelKey("m")}, allSlots(p, cands)); a.pool.Empty() {
		t.Fatal("Commit with no recorded decision did not mark")
	}
}

// TestRequestWithNoUnitsDeclines mirrors CU-11: a route with no routable prefix
// falls back rather than guessing, and must not touch the tree.
func TestRequestWithNoUnitsDeclines(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 3)

	before := decisions(t, "load")
	rr := &policy.RoutingRequest{Model: "m"}
	b, err := p.Select(context.Background(), cands, rr)
	if err != nil || b == nil {
		t.Fatalf("Select: %v", err)
	}
	p.Commit(b, rr)

	if d := decisions(t, "load") - before; d != 1 {
		t.Errorf("%v load decisions, want 1", d)
	}
	if st := p.tree.stats(); st.Blocks != 0 {
		t.Errorf("a prefix-less request put %d blocks in the tree", st.Blocks)
	}
}

// TestDropBackendRemovesItsAffinity: a backend that leaves must stop attracting
// its old traffic, or every request pinned to it retries into a gap.
func TestDropBackendRemovesItsAffinity(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 3)
	for _, b := range cands {
		p.AddBackend(b)
	}

	holder := route(t, p, cands, req(units(1, 2, 3)))
	p.DropBackend(holder)

	remaining := without(cands, holder)
	before := decisions(t, "load")
	got := route(t, p, remaining, req(units(1, 2, 3)))

	if got == holder {
		t.Fatal("routed to a dropped backend")
	}
	if d := decisions(t, "load") - before; d != 1 {
		t.Errorf("%v load decisions after the only holder left, want 1 (no stale affinity)", d)
	}
}

// TestPublishGaugesReportsPerBackendAndFleetTotals covers the metrics the
// dashboard reads, in particular AvgCopies — the tripwire for holder sets that
// only ever grow.
func TestPublishGaugesReportsPerBackendAndFleetTotals(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 2)
	for _, b := range cands {
		p.AddBackend(b)
	}

	route(t, p, cands, req(units(1, 2, 3, 4)))
	p.PublishGauges()

	if got := counter(t, metrics.CacheAvgCopies); got != 1 {
		t.Errorf("AvgCopies = %v after one backend served one prefix, want 1", got)
	}
	if got := counter(t, metrics.CacheTreeRuns); got != 1 {
		t.Errorf("TreeRuns = %v, want 1", got)
	}

	per := p.tree.perBackend()
	var held int
	for _, bs := range per {
		if bs.Blocks > 0 {
			held++
			if bs.Blocks != 4 || bs.Tokens != 4*256 {
				t.Errorf("holder stats = %+v, want 4 blocks / 1024 tokens", bs)
			}
		}
	}
	if held != 1 {
		t.Errorf("%d backends report holding blocks, want 1", held)
	}
}

// TestSweepReleasesIdleSessionsAndReportsThem ties the TTL to the counter the
// dashboard uses to decide whether the TTL is set right.
func TestSweepReleasesIdleSessionsAndReportsThem(t *testing.T) {
	p, clk := newTestPolicy(t)
	cands := fleet(t, 2)

	route(t, p, cands, req(units(1, 2, 3)))
	before := counter(t, metrics.CacheBlocksExpired)

	clk.Advance(testTTL + time.Minute)
	if freed := p.Sweep(); freed != 3 {
		t.Errorf("swept %d blocks, want 3", freed)
	}
	if d := counter(t, metrics.CacheBlocksExpired) - before; d != 3 {
		t.Errorf("counter moved by %v, want 3", d)
	}
	if st := p.tree.stats(); st.Runs != 0 {
		t.Errorf("%d runs survived a full-idle sweep", st.Runs)
	}
}

// allSlots is the candidate mask covering every backend in the fleet, for
// assertions that want to see the tree regardless of load.
func allSlots(p *Policy, cands []*registry.Backend) markSet {
	var m markSet
	for _, c := range cands {
		m.Add(p.tree.slotOrCreate(c.URL))
	}
	return m
}

// without mirrors what the gateway's candidate filter does when a backend hits
// its concurrency cap: it simply is not in the candidate set.
func without(cands []*registry.Backend, drop *registry.Backend) []*registry.Backend {
	out := make([]*registry.Backend, 0, len(cands))
	for _, c := range cands {
		if c != drop {
			out = append(out, c)
		}
	}
	return out
}
