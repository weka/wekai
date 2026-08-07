package affinity

import (
	"math/rand/v2"
	"testing"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
)

const testTTL = 5 * time.Minute

func newTestTree(t *testing.T) (*tree, *clock.Fake) {
	t.Helper()
	clk := clock.NewFake(time.Time{})
	return newTree(clk, testTTL), clk
}

// blocks builds a path from raw block hashes. Tokens are fixed so that a token
// count never accidentally carries the signal a hash is supposed to.
func blocks(hashes ...uint64) path {
	u := make([]kvcache.Unit, len(hashes))
	for i, h := range hashes {
		u[i] = kvcache.Unit{Hash: h, Tokens: 256}
	}
	return path{units: u}
}

// mask builds a candidate set from slot numbers.
func mask(slots ...int) markSet {
	var m markSet
	for _, s := range slots {
		m.Add(s)
	}
	return m
}

// all is the candidate mask that admits every backend the tests use.
func all() markSet { return mask(0, 1, 2, 3, 4, 5, 6, 7) }

// walkRuns visits every real run in the forest, parents before children.
func (t *tree) walkRuns(fn func(sh *shard, r *run)) {
	var visit func(sh *shard, r *run)
	visit = func(sh *shard, r *run) {
		if r.parent != nil {
			fn(sh, r)
		}
		for _, k := range r.kids {
			visit(sh, k)
		}
	}
	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.RLock()
		for _, root := range sh.roots {
			visit(sh, root)
		}
		sh.mu.RUnlock()
	}
}

// checkInvariants asserts the two structural properties every other test
// depends on. Called at the end of anything that mutates the tree.
func checkInvariants(t *testing.T, tr *tree) {
	t.Helper()

	// 1. The tail set is exactly the childless runs. If it drifts, TTL
	//    eviction either misses dead chains or corrupts live ones.
	childless := map[*run]*shard{}
	tr.walkRuns(func(sh *shard, r *run) {
		if len(r.kids) == 0 {
			childless[r] = sh
		}
	})
	for i := range tr.shards {
		sh := &tr.shards[i]
		sh.mu.RLock()
		for r := range sh.tails {
			if _, ok := childless[r]; !ok {
				t.Errorf("run %p is in the tail set but has %d children", r, len(r.kids))
			}
		}
		n := len(sh.tails)
		sh.mu.RUnlock()
		_ = n
	}
	total := 0
	for i := range tr.shards {
		tr.shards[i].mu.RLock()
		total += len(tr.shards[i].tails)
		tr.shards[i].mu.RUnlock()
	}
	if total != len(childless) {
		t.Errorf("tail set has %d runs, but %d runs are childless", total, len(childless))
	}

	// 2. A descendant's holders are always a subset of its parent's. This is
	//    what makes "deepest marked run" also mean "smallest, most specific
	//    pool", and it is the reason a widely-held shared prefix does not
	//    degrade routing: a request walks past it to its own narrower run.
	tr.walkRuns(func(_ *shard, r *run) {
		if r.parent == nil || r.parent.parent == nil {
			return // parent is the root sentinel, which holds nothing
		}
		if !r.marks.Subset(r.parent.marks) {
			t.Errorf("run %p holders are not a subset of its parent's", r)
		}
	})
}

// countRuns is the true run count, for checking the maintained counter.
func countRuns(tr *tree) int64 {
	var n int64
	tr.walkRuns(func(_ *shard, _ *run) { n++ })
	return n
}

// TestWalkFindsDeepestMarkedRunRegardlessOfRequestFraction is the core fix this
// policy exists for. The deployed prefix-cache-candidates policy gates affinity
// on cached/total over the WHOLE current request, so a long session whose fixed
// shared prefix is a shrinking share of an ever-growing prompt stops being
// routed by cache — the direct cause of the ~15% average predicted fraction
// observed in production. Here a two-block match inside a 200-block request is
// a perfectly good anchor.
func TestWalkFindsDeepestMarkedRunRegardlessOfRequestFraction(t *testing.T) {
	tr, _ := newTestTree(t)

	short := blocks(1, 2)
	tr.commit(short, 3)

	long := make([]uint64, 200)
	long[0], long[1] = 1, 2
	for i := 2; i < len(long); i++ {
		long[i] = uint64(1000 + i)
	}

	a := tr.walk(blocks(long...), all())
	if a.pool.Empty() {
		t.Fatal("no anchor for a request sharing only 2 of 200 blocks")
	}
	if !a.pool.Has(3) {
		t.Error("anchor is not held by the backend that committed the prefix")
	}
	if a.matched != 2 {
		t.Errorf("matched = %d blocks, want 2", a.matched)
	}
}

// TestWalkSkipsRunsWhoseHoldersAreNotCandidates covers the case the simulator
// has no analogue for: its nodes never go unhealthy, drain, get removed, or hit
// their cap. Here the deepest run's only holder is absent from the candidate
// set, so the walk must fall back to a shallower run that a candidate does
// hold, rather than anchoring on a backend that cannot be routed to.
func TestWalkSkipsRunsWhoseHoldersAreNotCandidates(t *testing.T) {
	tr, _ := newTestTree(t)

	// Backend 0 holds the shared prefix; backend 1 holds the deeper extension.
	tr.commit(blocks(1, 2), 0)
	tr.commit(blocks(1, 2, 3, 4), 1)

	req := blocks(1, 2, 3, 4)

	// With both available, the deepest run wins.
	a := tr.walk(req, mask(0, 1))
	if a.pool.Empty() || !a.pool.Has(1) {
		t.Fatal("expected the deeper run held by backend 1")
	}

	// With backend 1 out of the candidate set, the anchor must retreat to the
	// shared prefix that backend 0 still holds — but `marked` must still see
	// the deeper run, since that is the signal a split answers.
	a = tr.walk(req, mask(0))
	if a.pool.Empty() {
		t.Fatal("no available anchor when the deep holder is not a candidate")
	}
	if a.pool.Has(1) && !a.pool.Has(0) {
		t.Fatal("anchored on a run no candidate holds")
	}
	if !a.pool.Has(0) {
		t.Error("available anchor is not held by the one candidate")
	}
	if a.held.Empty() || !a.held.Has(1) {
		t.Error("marked anchor should still report the saturated/absent holder")
	}
}

// TestWalkReportsNoAnchorForAnEntirelyNewPrompt is the fourth routing tier's
// precondition.
func TestWalkReportsNoAnchorForAnEntirelyNewPrompt(t *testing.T) {
	tr, _ := newTestTree(t)
	tr.commit(blocks(1, 2, 3), 0)

	a := tr.walk(blocks(90, 91), all())
	if !a.pool.Empty() || !a.held.Empty() {
		t.Fatal("a prompt sharing no prefix must produce no anchor")
	}
	if a.matched != 0 {
		t.Errorf("matched = %d, want 0", a.matched)
	}
}

// TestCommitMarksEveryRunOnThePathNotOnlyTheDeepest is Anton's explicit
// requirement ("make sure every node in prefix tree path marked on split, not
// only the last hit") and the single most important behaviour in this file.
// Marking only the terminus would mean a split never converges: the next
// request anchoring at a shallower depth would not see the new holder.
func TestCommitMarksEveryRunOnThePathNotOnlyTheDeepest(t *testing.T) {
	tr, _ := newTestTree(t)

	// Backend 0 lays down a chain that branches, so the path is several runs.
	tr.commit(blocks(1, 2, 3), 0)
	tr.commit(blocks(1, 2, 9), 0)    // forces a split after block 2
	tr.commit(blocks(1, 2, 3, 4), 0) // extends past the branch

	// Backend 1 now serves the deep request: every run it traverses must
	// record it, not just the last.
	tr.commit(blocks(1, 2, 3, 4), 1)

	var runsOnPath, marked int
	tr.walkRuns(func(_ *shard, r *run) {
		// Every run whose first block is on the 1,2,3,4 path.
		switch r.hashes[0] {
		case 1, 2, 3, 4:
			runsOnPath++
			if r.marks.Has(1) {
				marked++
			}
		}
	})
	if runsOnPath < 2 {
		t.Fatalf("expected the path to span several runs, got %d", runsOnPath)
	}
	if marked != runsOnPath {
		t.Errorf("backend 1 marked on %d of %d runs along its path; every one must be marked",
			marked, runsOnPath)
	}
	checkInvariants(t, tr)
}

// TestCommitReportsBlocksTheBackendAlreadyHeld covers the predicted-hit number
// the policy reports, and specifically that credit stops at the first run the
// backend does not hold rather than resuming further down.
func TestCommitReportsBlocksTheBackendAlreadyHeld(t *testing.T) {
	tr, _ := newTestTree(t)

	tr.commit(blocks(1, 2, 3), 0)
	if hit := tr.commit(blocks(1, 2, 3), 0); hit != 3 {
		t.Errorf("re-serving an identical request: hit = %d, want 3", hit)
	}
	if hit := tr.commit(blocks(1, 2, 3), 1); hit != 0 {
		t.Errorf("a backend that has never served this prefix: hit = %d, want 0", hit)
	}
	if hit := tr.commit(blocks(1, 2, 3, 4, 5), 0); hit != 3 {
		t.Errorf("extending a held prefix: hit = %d, want 3", hit)
	}
}

// TestSplitRunPreservesMarksOnBothHalves: when a request diverges mid-run the
// run is cut in two, and the holders of the original hold BOTH halves. Losing
// the marks on either half would silently drop affinity for every session
// already pinned to that prefix.
func TestSplitRunPreservesMarksOnBothHalves(t *testing.T) {
	tr, _ := newTestTree(t)

	tr.commit(blocks(1, 2, 3, 4), 0) // one run of four blocks
	tr.commit(blocks(1, 2, 9), 1)    // diverges after block 2, splitting it

	var upper, lower *run
	tr.walkRuns(func(_ *shard, r *run) {
		switch r.hashes[0] {
		case 1:
			upper = r
		case 3:
			lower = r
		}
	})
	if upper == nil || lower == nil {
		t.Fatal("expected the run to be split into an upper and lower half")
	}
	if !upper.marks.Has(0) || !lower.marks.Has(0) {
		t.Error("the original holder lost a half in the split")
	}
	if !upper.marks.Has(1) {
		t.Error("the diverging backend was not marked on the shared upper half")
	}
	if lower.marks.Has(1) {
		t.Error("the diverging backend was marked on a run it never served")
	}
	checkInvariants(t, tr)
}

// TestCommitExtendsAPrivateTailInPlace keeps a long conversation from costing
// one run per turn: a 200-turn session that only ever grows at the end must
// stay a single run while it has exactly one holder and no branches.
func TestCommitExtendsAPrivateTailInPlace(t *testing.T) {
	tr, _ := newTestTree(t)

	seq := []uint64{1}
	tr.commit(blocks(seq...), 0)
	for turn := range 50 {
		seq = append(seq, uint64(100+turn))
		tr.commit(blocks(seq...), 0)
	}

	if n := countRuns(tr); n != 1 {
		t.Errorf("a growing single-holder session produced %d runs, want 1", n)
	}
	st := tr.stats()
	if st.Blocks != int64(len(seq)) {
		t.Errorf("block count = %d, want %d", st.Blocks, len(seq))
	}
	if st.AvgCopies != 1 {
		t.Errorf("AvgCopies = %v, want exactly 1 for a single-holder tree", st.AvgCopies)
	}
	checkInvariants(t, tr)
}

// TestCommitDoesNotExtendATailHeldByAnotherBackend guards the in-place
// extension's precondition. Appending to a run held by someone else would
// silently claim that backend holds blocks it has never seen.
func TestCommitDoesNotExtendATailHeldByAnotherBackend(t *testing.T) {
	tr, _ := newTestTree(t)

	tr.commit(blocks(1, 2), 0)
	tr.commit(blocks(1, 2, 3), 1)

	var tail *run
	tr.walkRuns(func(_ *shard, r *run) {
		if r.hashes[0] == 3 {
			tail = r
		}
	})
	if tail == nil {
		t.Fatal("expected block 3 to become its own run, not to extend backend 0's tail")
	}
	if tail.marks.Has(0) {
		t.Error("backend 0 was credited with a block it never served")
	}
	checkInvariants(t, tr)
}

// TestSweepEvictsOnlyIdleTails: the TTL is idle time, not age. A prefix under
// continuous traffic must survive indefinitely — that is the whole point of
// keeping a hot shared prefix resident.
func TestSweepEvictsOnlyIdleTails(t *testing.T) {
	tr, clk := newTestTree(t)

	tr.commit(blocks(1, 2), 0)   // will stay hot
	tr.commit(blocks(50, 51), 0) // will go idle

	clk.Advance(4 * time.Minute)
	tr.commit(blocks(1, 2), 0) // refresh the hot one
	clk.Advance(2 * time.Minute)

	if freed := tr.sweep(); freed != 2 {
		t.Errorf("freed %d blocks, want the 2 of the idle tail", freed)
	}
	if a := tr.walk(blocks(1, 2), all()); a.pool.Empty() {
		t.Error("the continuously-used prefix was evicted")
	}
	if a := tr.walk(blocks(50, 51), all()); !a.pool.Empty() {
		t.Error("the idle prefix survived its TTL")
	}
	checkInvariants(t, tr)
}

// TestEvictNeverRemovesRunWithChildren is the "never evict from the middle"
// rule. A shared prefix goes idle as a RUN long before its branches do, and
// removing it would orphan every session below it.
func TestEvictNeverRemovesRunWithChildren(t *testing.T) {
	tr, clk := newTestTree(t)

	tr.commit(blocks(1, 2), 0)
	tr.commit(blocks(1, 2, 3), 0)
	tr.commit(blocks(1, 2, 9), 0) // two branches under the shared 1,2

	// Let the shared prefix go idle while one branch stays hot.
	clk.Advance(4 * time.Minute)
	tr.commit(blocks(1, 2, 3), 0)
	clk.Advance(2 * time.Minute)
	tr.sweep()

	if a := tr.walk(blocks(1, 2, 3), all()); a.pool.Empty() {
		t.Fatal("the hot branch was lost when its idle parent was swept")
	}
	checkInvariants(t, tr)
}

// TestEvictTailCascadesThroughDeadChain: when a whole session dies, its entire
// chain must unwind in one sweep rather than one run per sweep interval, or a
// deep dead chain would take minutes per level to release.
func TestEvictTailCascadesThroughDeadChain(t *testing.T) {
	tr, clk := newTestTree(t)

	// A chain of runs, forced apart by branching at each level.
	tr.commit(blocks(1, 2), 0)
	tr.commit(blocks(1, 2, 3, 4), 0)
	tr.commit(blocks(1, 2, 3, 9), 0)
	tr.commit(blocks(1, 2, 3, 4, 5), 0)
	before := countRuns(tr)
	if before < 3 {
		t.Fatalf("expected a multi-run chain, got %d runs", before)
	}

	clk.Advance(testTTL + time.Second)
	tr.sweep()

	if n := countRuns(tr); n != 0 {
		t.Errorf("%d runs survived a sweep in which everything was idle", n)
	}
	if st := tr.stats(); st.Blocks != 0 || st.Tails != 0 {
		t.Errorf("stats not drained after full eviction: %+v", st)
	}
	checkInvariants(t, tr)
}

// TestDropBackendClearsMarksFleetWide and its sibling below are the reason
// slot reuse is ordered the way it is. Prefixes are never reassigned when a
// backend leaves (CACHE-10 in the older policies).
func TestDropBackendClearsMarksFleetWide(t *testing.T) {
	tr, _ := newTestTree(t)
	slot := tr.slotFor("http://w0:8000")
	other := tr.slotFor("http://w1:8000")

	tr.commit(blocks(1, 2, 3), slot)
	tr.commit(blocks(1, 2, 3), other)
	tr.commit(blocks(40, 41), slot)

	tr.dropBackend("http://w0:8000")

	tr.walkRuns(func(_ *shard, r *run) {
		if r.marks.Has(slot) {
			t.Errorf("run %p still holds the dropped backend's slot", r)
		}
	})
	if a := tr.walk(blocks(1, 2, 3), mask(other)); a.pool.Empty() {
		t.Error("dropping one backend removed another's marks")
	}
	if st := tr.stats(); st.AvgCopies != 1 {
		t.Errorf("AvgCopies = %v after the drop, want 1", st.AvgCopies)
	}
	checkInvariants(t, tr)
}

// TestReusedSlotDoesNotInheritDeadBackendMarks: slots are a scarce index space
// and get reused, so clearing must complete before the slot is handed out or a
// new backend silently inherits every prefix its predecessor held.
func TestReusedSlotDoesNotInheritDeadBackendMarks(t *testing.T) {
	tr, _ := newTestTree(t)

	old := tr.slotFor("http://old:8000")
	// A survivor keeps the run alive, so this tests slot hygiene rather than
	// the orphan sweep that would otherwise delete the evidence.
	keep := tr.slotFor("http://keep:8000")
	tr.commit(blocks(1, 2, 3), old)
	tr.commit(blocks(1, 2, 3), keep)

	tr.dropBackend("http://old:8000")

	fresh := tr.slotFor("http://new:8000")
	if fresh != old {
		t.Fatalf("expected the freed slot %d to be reused, got %d", old, fresh)
	}
	if a := tr.walk(blocks(1, 2, 3), mask(fresh)); !a.pool.Empty() {
		t.Fatal("a newly-added backend inherited the marks of the backend whose slot it reused")
	}
	if a := tr.walk(blocks(1, 2, 3), mask(keep)); a.pool.Empty() {
		t.Fatal("the surviving backend lost the run")
	}
}

// TestConcurrentWalkAndCommit exists because the router is not sequential: the
// gateway is a plain http.Server, so every in-flight request is its own
// goroutine calling Select and later Commit. Under the standard evaluation
// recipe that is 128 concurrent callers against one shared tree, where the
// older design had one independently-locked trie per backend. Run with -race.
func TestConcurrentWalkAndCommit(t *testing.T) {
	tr, _ := newTestTree(t)

	const workers = 16
	done := make(chan struct{})
	for w := range workers {
		go func(w int) {
			defer func() { done <- struct{}{} }()
			rng := rand.New(rand.NewPCG(uint64(w), 99))
			for range 2000 {
				n := 1 + rng.IntN(5)
				hs := make([]uint64, n)
				for i := range hs {
					hs[i] = uint64(rng.IntN(12))
				}
				p := blocks(hs...)
				tr.walk(p, all())
				tr.commit(p, rng.IntN(4))
				if rng.IntN(200) == 0 {
					tr.sweep()
				}
				tr.stats()
			}
		}(w)
	}
	for range workers {
		<-done
	}
	checkInvariants(t, tr)
}

// TestConcurrentDropBackendIsSafe covers the other writer: a rollout removes a
// backend while requests are still being routed.
func TestConcurrentDropBackendIsSafe(t *testing.T) {
	tr, _ := newTestTree(t)
	urls := []string{"http://w0:8000", "http://w1:8000", "http://w2:8000"}
	for _, u := range urls {
		tr.slotFor(u)
	}

	done := make(chan struct{})
	go func() {
		defer func() { done <- struct{}{} }()
		rng := rand.New(rand.NewPCG(7, 8))
		for range 5000 {
			hs := []uint64{uint64(rng.IntN(6)), uint64(rng.IntN(6))}
			slot, _ := tr.slot(urls[rng.IntN(len(urls))])
			tr.commit(blocks(hs...), slot)
		}
	}()
	go func() {
		defer func() { done <- struct{}{} }()
		for range 20 {
			tr.dropBackend("http://w2:8000")
			tr.slotFor("http://w2:8000")
		}
	}()
	<-done
	<-done
	checkInvariants(t, tr)
}

// BenchmarkWalk is the check on the NFR-2 p99 routing budget of 250us. A single
// shared tree concentrates what used to be N independently-locked tries, so the
// read path has to stay cheap under contention.
func BenchmarkWalk(b *testing.B) {
	tr := newTree(clock.NewFake(time.Time{}), testTTL)
	rng := rand.New(rand.NewPCG(3, 4))

	// A realistic shape: a shared 44-block system prompt, then per-session
	// divergence, spread over 64 backends.
	shared := make([]uint64, 44)
	for i := range shared {
		shared[i] = uint64(i + 1)
	}
	var reqs []path
	for s := range 500 {
		hs := append(append([]uint64{}, shared...), uint64(1_000_000+s))
		for range 30 {
			hs = append(hs, rng.Uint64())
		}
		p := blocks(hs...)
		tr.commit(p, s%64)
		reqs = append(reqs, p)
	}
	cands := markSet{}
	for i := range 64 {
		cands.Add(i)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			tr.walk(reqs[i%len(reqs)], cands)
			i++
		}
	})
}

// BenchmarkCommit measures the write path, which serializes per shard.
func BenchmarkCommit(b *testing.B) {
	tr := newTree(clock.NewFake(time.Time{}), testTTL)
	shared := make([]uint64, 44)
	for i := range shared {
		shared[i] = uint64(i + 1)
	}
	var reqs []path
	for s := range 500 {
		hs := append(append([]uint64{}, shared...), uint64(1_000_000+s))
		reqs = append(reqs, blocks(hs...))
	}
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		tr.commit(reqs[i%len(reqs)], i%64)
	}
}

// TestModelKeyPreventsCrossModelPrefixSharing: the gateway filters candidates
// by DialectID and never by Model, so a router fronting two models on one
// dialect would otherwise credit one model's KV cache for the other's prompt.
func TestModelKeyPreventsCrossModelPrefixSharing(t *testing.T) {
	tr, _ := newTestTree(t)

	a := blocks(1, 2, 3)
	a.modelKey = 0xAAAA
	b := blocks(1, 2, 3)
	b.modelKey = 0xBBBB

	tr.commit(a, 0)

	if got := tr.walk(b, all()); !got.pool.Empty() {
		t.Fatal("a different model matched a prefix it has never served")
	}
	if got := tr.walk(a, all()); got.pool.Empty() {
		t.Fatal("the committing model lost its own prefix")
	}
}

// TestTreeInvariantsHoldUnderRandomOperations is the backstop for everything
// above: tails and the subset property are asserted after a long, seeded mix of
// commits, divergences and sweeps, including cases no hand-written test
// enumerates.
func TestTreeInvariantsHoldUnderRandomOperations(t *testing.T) {
	tr, clk := newTestTree(t)
	rng := rand.New(rand.NewPCG(1, 2))

	for step := range 3000 {
		n := 1 + rng.IntN(6)
		hs := make([]uint64, n)
		for i := range hs {
			// A small alphabet so paths collide, branch and split constantly.
			hs[i] = uint64(rng.IntN(8))
		}
		tr.commit(blocks(hs...), rng.IntN(4))

		if step%100 == 0 {
			clk.Advance(time.Duration(rng.IntN(120)) * time.Second)
			tr.sweep()
		}
	}
	checkInvariants(t, tr)

	if got, want := countRuns(tr), tr.stats().Runs; got != want {
		t.Errorf("maintained run counter = %d, actual runs = %d", want, got)
	}

	// Everything goes idle: the forest must drain completely.
	clk.Advance(time.Hour)
	for range 10 {
		tr.sweep()
	}
	if n := countRuns(tr); n != 0 {
		t.Errorf("%d runs survived after every run went idle", n)
	}
	checkInvariants(t, tr)
}

// TestStatsCountBlocksAndCopies backs router_cache_avg_copies, the tripwire for
// the one hazard this design knowingly accepts: holder sets that only ever grow
// on continuously-hot runs. A wrong denominator would hide exactly the drift it
// exists to catch.
func TestStatsCountBlocksAndCopies(t *testing.T) {
	tr, _ := newTestTree(t)

	tr.commit(blocks(1, 2, 3, 4), 0)
	if st := tr.stats(); st.Blocks != 4 || st.AvgCopies != 1 {
		t.Fatalf("single holder: blocks=%d avgCopies=%v, want 4 and 1", st.Blocks, st.AvgCopies)
	}

	tr.commit(blocks(1, 2, 3, 4), 1) // a second holder for all four blocks
	if st := tr.stats(); st.Blocks != 4 || st.AvgCopies != 2 {
		t.Fatalf("two holders: blocks=%d avgCopies=%v, want 4 and 2", st.Blocks, st.AvgCopies)
	}

	tr.commit(blocks(1, 2, 9), 2) // splits after block 2; holds 3 of 5 blocks
	st := tr.stats()
	if st.Blocks != 5 {
		t.Fatalf("blocks = %d, want 5", st.Blocks)
	}
	// blocks 1,2 held by three backends; 3,4 by two; 9 by one => 6+4+1 = 11.
	if want := 11.0 / 5.0; st.AvgCopies != want {
		t.Errorf("AvgCopies = %v, want %v", st.AvgCopies, want)
	}
	checkInvariants(t, tr)
}
