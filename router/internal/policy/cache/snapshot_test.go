package cache_test

import (
	"sync"
	"testing"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/policy"
	cachepolicy "github.com/weka/wekai/router/internal/policy/cache"
	"github.com/weka/wekai/router/internal/viz"
)

// concatUnits joins two Unit chains for a Commit call that shares a and then
// diverges into b.
func concatUnits(a, b []kvcache.Unit) []kvcache.Unit {
	out := make([]kvcache.Unit, 0, len(a)+len(b))
	out = append(out, a...)
	out = append(out, b...)
	return out
}

func backendIndex(backends []viz.BackendMeta, url string) int {
	for i, b := range backends {
		if b.URL == url {
			return i
		}
	}
	return -1
}

func treeRoot(tree []viz.TreeNode) *viz.TreeNode {
	for i := range tree {
		if tree[i].Parent == -1 {
			return &tree[i]
		}
	}
	return nil
}

// TestSnapshot_Empty is the "no traffic yet" state /router-viz must render
// sensibly for: an active policy with backends added but nothing committed.
func TestSnapshot_Empty(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2)
	for _, b := range bs {
		p.AddBackend(b)
	}

	snap := p.Snapshot(viz.SnapshotOptions{})
	if !snap.PolicyActive {
		t.Fatalf("PolicyActive = false, want true")
	}
	if len(snap.Backends) != 2 {
		t.Fatalf("len(Backends) = %d, want 2", len(snap.Backends))
	}
	if len(snap.Tree) != 0 {
		t.Fatalf("len(Tree) = %d, want 0 (nothing committed yet)", len(snap.Tree))
	}
	if snap.AvgCopies != 0 {
		t.Fatalf("AvgCopies = %v, want 0", snap.AvgCopies)
	}
	if snap.NodesShown != 0 || snap.NodesTotal != 0 || snap.Truncated {
		t.Fatalf("unexpected counters on an empty snapshot: %+v", snap)
	}
}

// TestSnapshot_DefaultIsUnlimited is anton's explicit requirement: a zero
// SnapshotOptions (what DataHandler passes when the page's UI controls are
// all left empty) must show the ENTIRE tree, not some hidden default cap.
func TestSnapshot_DefaultIsUnlimited(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 1)
	p.AddBackend(bs[0])
	// 200 independent single-block sessions -> 200 separate root runs, well
	// past any of the old hardcoded defaults (60/80/6).
	for i := 0; i < 200; i++ {
		p.Commit(bs[0], req(units(uint64(1000+i))))
	}

	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Tree) != 200 {
		t.Fatalf("len(Tree) = %d, want 200 (unlimited by default)", len(snap.Tree))
	}
	if snap.NodesShown != 200 || snap.NodesTotal != 200 {
		t.Fatalf("NodesShown/NodesTotal = %d/%d, want 200/200", snap.NodesShown, snap.NodesTotal)
	}
	if snap.Truncated {
		t.Fatalf("Truncated = true, want false (nothing was capped)")
	}
}

// TestSnapshot_SharedPrefixIsOneCommonAncestor is the core tree property
// anton asked for: a prefix shared by two backends must appear ONCE, as a
// single common-ancestor row present on both, with each backend's own
// divergent tail as a separate child row present on only that backend.
func TestSnapshot_SharedPrefixIsOneCommonAncestor(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2) // bs[0].URL < bs[1].URL by construction (w0, w1)
	for _, b := range bs {
		p.AddBackend(b)
	}
	bs[0].AddInflight(3)

	shared := units(100)
	p.Commit(bs[0], req(concatUnits(shared, units(200))))
	p.Commit(bs[1], req(concatUnits(shared, units(300))))

	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Tree) != 3 {
		t.Fatalf("len(Tree) = %d, want 3 (1 shared ancestor + 2 divergent children)", len(snap.Tree))
	}
	if snap.NodesShown != 3 || snap.NodesTotal != 3 || snap.Truncated {
		t.Fatalf("unexpected counters: shown=%d total=%d truncated=%v", snap.NodesShown, snap.NodesTotal, snap.Truncated)
	}

	root := treeRoot(snap.Tree)
	if root == nil {
		t.Fatalf("no root in Tree: %+v", snap.Tree)
	}
	if root.Hash != kvcache.HexHash(100) {
		t.Fatalf("root.Hash = %q, want the shared block's hash", root.Hash)
	}
	if root.RunLen != 1 {
		t.Fatalf("root.RunLen = %d, want 1", root.RunLen)
	}
	if len(root.Children) != 2 {
		t.Fatalf("root should have exactly 2 children (the two divergent tails), got %d: %+v", len(root.Children), root)
	}
	for i, on := range root.Present {
		if !on {
			t.Fatalf("shared root should be present on backend %s (index %d): %v", snap.Backends[i].URL, i, root.Present)
		}
	}

	i0, i1 := backendIndex(snap.Backends, bs[0].URL), backendIndex(snap.Backends, bs[1].URL)
	if i0 < 0 || i1 < 0 {
		t.Fatalf("expected both backend URLs in Backends: %+v", snap.Backends)
	}
	hex200, hex300 := kvcache.HexHash(200), kvcache.HexHash(300)
	var child200, child300 *viz.TreeNode
	for _, ci := range root.Children {
		n := snap.Tree[ci]
		switch n.Hash {
		case hex200:
			c := n
			child200 = &c
		case hex300:
			c := n
			child300 = &c
		}
	}
	if child200 == nil || child300 == nil {
		t.Fatalf("expected both divergent children (200 and 300) under the shared root")
	}
	if !child200.Present[i0] || child200.Present[i1] {
		t.Fatalf("child200 presence wrong: %v (want present on w0 only)", child200.Present)
	}
	if !child300.Present[i1] || child300.Present[i0] {
		t.Fatalf("child300 presence wrong: %v (want present on w1 only)", child300.Present)
	}

	// AvgCopies is block-level: shared block held by 2, the two divergent
	// blocks by 1 each -> mean = (2+1+1)/3.
	wantAvg := 4.0 / 3.0
	if diff := snap.AvgCopies - wantAvg; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("AvgCopies = %v, want %v", snap.AvgCopies, wantAvg)
	}

	if snap.Backends[i0].Healthy == nil || !*snap.Backends[i0].Healthy {
		t.Fatalf("w0.Healthy = %v, want true (added via AddBackend and marked Healthy)", snap.Backends[i0].Healthy)
	}
	if snap.Backends[i0].Inflight != 3 {
		t.Fatalf("w0.Inflight = %d, want 3", snap.Backends[i0].Inflight)
	}
}

// TestSnapshot_SubtreeBlocksCountsRealBlocksNotRows is the per-node "⊂N"
// badge anton asked for: a known 3-row tree where the ROOT compresses
// multiple blocks into one row (RunLen=3), and each child is itself a
// multi-block compressed run (2 and 4 blocks) — so root.SubtreeBlocks (9)
// must differ from a naive row count (3) and from the root's own RunLen
// (3), proving the badge counts real underlying blocks, not tree rows.
func TestSnapshot_SubtreeBlocksCountsRealBlocksNotRows(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2)
	for _, b := range bs {
		p.AddBackend(b)
	}

	shared := units(100, 101, 102)                                       // present on both -> compresses to ONE 3-block row
	p.Commit(bs[0], req(concatUnits(shared, units(200, 201))))           // A's 2-block divergent tail
	p.Commit(bs[1], req(concatUnits(shared, units(300, 301, 302, 303)))) // B's 4-block divergent tail

	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Tree) != 3 {
		t.Fatalf("len(Tree) = %d, want 3 (1 shared root + 2 divergent children)", len(snap.Tree))
	}

	root := treeRoot(snap.Tree)
	if root == nil {
		t.Fatalf("no root in Tree: %+v", snap.Tree)
	}
	if root.RunLen != 3 {
		t.Fatalf("root.RunLen = %d, want 3", root.RunLen)
	}
	if len(root.Children) != 2 {
		t.Fatalf("root should have 2 children, got %d", len(root.Children))
	}
	if root.SubtreeBlocks != 9 {
		t.Fatalf("root.SubtreeBlocks = %d, want 9 (3 own + 2 + 4 descendants) — must count real blocks, not rows", root.SubtreeBlocks)
	}

	for _, ci := range root.Children {
		child := snap.Tree[ci]
		// Each child here is a leaf: its subtree is only itself, so
		// SubtreeBlocks must equal its own RunLen exactly.
		if child.SubtreeBlocks != child.RunLen {
			t.Fatalf("leaf child SubtreeBlocks = %d, want == its own RunLen %d", child.SubtreeBlocks, child.RunLen)
		}
	}
}

// TestSnapshot_BlockDepthAccountsForCompressedRuns is the per-node "d N"
// badge anton asked for: block depth must advance by the FULL RunLen of a
// compressed row in one step, not by 1 per row, and must accumulate
// correctly down each branch. Reuses the same 3-row tree as the
// SubtreeBlocks test above (root compresses 3 blocks into one row; each
// child is itself a multi-block run of 2 and 4 blocks) so the two badges
// can be cross-checked against the same known structure.
func TestSnapshot_BlockDepthAccountsForCompressedRuns(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2)
	for _, b := range bs {
		p.AddBackend(b)
	}

	shared := units(100, 101, 102)                                       // present on both -> compresses to ONE 3-block row
	p.Commit(bs[0], req(concatUnits(shared, units(200, 201))))           // A's 2-block divergent tail
	p.Commit(bs[1], req(concatUnits(shared, units(300, 301, 302, 303)))) // B's 4-block divergent tail

	snap := p.Snapshot(viz.SnapshotOptions{})
	root := treeRoot(snap.Tree)
	if root == nil {
		t.Fatalf("no root in Tree: %+v", snap.Tree)
	}

	// Root's own run is 3 blocks starting at the root of the tree: its
	// end-of-run depth is exactly 3, not 1 (a row =/= a block).
	if root.BlockDepth != 3 {
		t.Fatalf("root.BlockDepth = %d, want 3 (its own 3-block run, counted through the end)", root.BlockDepth)
	}

	hex200, hex300 := kvcache.HexHash(200), kvcache.HexHash(300)
	for _, ci := range root.Children {
		child := snap.Tree[ci]
		switch child.Hash {
		case hex200: // 2-block divergent tail: depth = root's 3 + its own 2
			if child.BlockDepth != 5 {
				t.Fatalf("200-tail BlockDepth = %d, want 5 (3 root blocks + 2 of its own)", child.BlockDepth)
			}
		case hex300: // 4-block divergent tail: depth = root's 3 + its own 4
			if child.BlockDepth != 7 {
				t.Fatalf("300-tail BlockDepth = %d, want 7 (3 root blocks + 4 of its own)", child.BlockDepth)
			}
		default:
			t.Fatalf("unexpected child hash %q", child.Hash)
		}
	}
}

// TestSnapshot_LongSharedChainCompressesToOneRow is the radix-compression
// property: a straight, non-branching chain must collapse into a single
// tree row (one box per RUN, not one per block), the same behavior the
// reference kv-router-sim.html's tree view relies on for readability.
func TestSnapshot_LongSharedChainCompressesToOneRow(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 1)
	p.AddBackend(bs[0])
	p.Commit(bs[0], req(units(1, 2, 3, 4, 5)))

	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Tree) != 1 {
		t.Fatalf("len(Tree) = %d, want 1 (a single-backend straight 5-block chain must fully compress)", len(snap.Tree))
	}
	if snap.Tree[0].RunLen != 5 {
		t.Fatalf("RunLen = %d, want 5", snap.Tree[0].RunLen)
	}
}

// TestSnapshot_CompressionBreaksOnPresenceChange is the correctness
// counterpart to the compression test above: even with NO structural
// branch, a run must stop the moment the backend-presence set changes,
// or the row would show a wrong/blended presence pattern.
func TestSnapshot_CompressionBreaksOnPresenceChange(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2)
	for _, b := range bs {
		p.AddBackend(b)
	}

	// Both backends serve blocks 1,2 (same present-set: compress together);
	// only bs[0] goes on to block 3 (present-set changes: new run).
	p.Commit(bs[0], req(units(1, 2, 3)))
	p.Commit(bs[1], req(units(1, 2)))

	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Tree) != 2 {
		t.Fatalf("len(Tree) = %d, want 2 (compressed run [1,2] + divergent run [3])", len(snap.Tree))
	}
	root := treeRoot(snap.Tree)
	if root == nil {
		t.Fatalf("no root in Tree: %+v", snap.Tree)
	}
	if root.RunLen != 2 {
		t.Fatalf("root.RunLen = %d, want 2 (blocks 1 and 2 share a present-set and must compress together)", root.RunLen)
	}
	if len(root.Children) != 1 {
		t.Fatalf("root should have exactly 1 child (block 3, where the present-set changes), got %d", len(root.Children))
	}
	child := snap.Tree[root.Children[0]]
	if child.RunLen != 1 {
		t.Fatalf("child.RunLen = %d, want 1", child.RunLen)
	}
}

// TestSnapshot_MaxRowsTruncatesTreeRows checks the "no silent truncation"
// contract end to end when the UI explicitly asks for a row cap: NodesTotal
// must reflect the TRUE total regardless of the requested cap (kvcache.Trie.Chains
// always walks its whole trie internally; only the caller's own fetch cap,
// decoupled from the display option, could distort this — this test is what
// would have caught that bug).
func TestSnapshot_MaxRowsTruncatesTreeRows(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 1)
	p.AddBackend(bs[0])
	// 10 independent single-block sessions -> 10 separate root runs.
	for i := 0; i < 10; i++ {
		p.Commit(bs[0], req(units(uint64(1000+i))))
	}

	snap := p.Snapshot(viz.SnapshotOptions{MaxRows: 4})
	if len(snap.Tree) > 4 {
		t.Fatalf("len(Tree) = %d, want <= 4 (MaxRows)", len(snap.Tree))
	}
	if snap.NodesTotal != 10 {
		t.Fatalf("NodesTotal = %d, want 10 (the true total, not capped by the small display option)", snap.NodesTotal)
	}
	if !snap.Truncated {
		t.Fatalf("Truncated = false, want true (10 sessions, MaxRows 4)")
	}
	if snap.NodesShown != len(snap.Tree) {
		t.Fatalf("NodesShown = %d, want %d (== len(Tree))", snap.NodesShown, len(snap.Tree))
	}
}

// TestSnapshot_MaxChildrenLimitsPerNode is the per-node child cap, another
// UI-settable option: unlimited by default, applied only when explicitly
// requested.
func TestSnapshot_MaxChildrenLimitsPerNode(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 1)
	p.AddBackend(bs[0])
	shared := units(100)
	for i := 0; i < 5; i++ {
		p.Commit(bs[0], req(concatUnits(shared, units(uint64(200+i)))))
	}

	full := p.Snapshot(viz.SnapshotOptions{})
	if len(full.Tree) != 6 { // 1 shared root + 5 divergent children
		t.Fatalf("full tree len = %d, want 6", len(full.Tree))
	}

	limited := p.Snapshot(viz.SnapshotOptions{MaxChildren: 2})
	root := treeRoot(limited.Tree)
	if root == nil {
		t.Fatalf("no root in limited tree: %+v", limited.Tree)
	}
	if len(root.Children) != 2 {
		t.Fatalf("root.Children = %d, want 2 (MaxChildren=2)", len(root.Children))
	}
	if !limited.Truncated {
		t.Fatalf("Truncated = false, want true")
	}
	if limited.NodesTotal != 6 {
		t.Fatalf("NodesTotal = %d, want 6 (true total unaffected by MaxChildren)", limited.NodesTotal)
	}
}

// TestSnapshot_MaxDepthLimitsDepth is the depth cap. Uses a chain that
// changes present-set at every step (no structural branch) so each block is
// forced into its own row purely by the compression rule, giving a clean
// 3-level tree to cap.
func TestSnapshot_MaxDepthLimitsDepth(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 3)
	for _, b := range bs {
		p.AddBackend(b)
	}
	p.Commit(bs[0], req(units(1, 2, 3))) // present on block1,2,3
	p.Commit(bs[1], req(units(1, 2)))    // present on block1,2 only
	p.Commit(bs[2], req(units(1)))       // present on block1 only

	full := p.Snapshot(viz.SnapshotOptions{})
	if len(full.Tree) != 3 {
		t.Fatalf("full tree len = %d, want 3 (depths 0,1,2)", len(full.Tree))
	}

	limited := p.Snapshot(viz.SnapshotOptions{MaxDepth: 2})
	if len(limited.Tree) != 2 {
		t.Fatalf("MaxDepth=2 tree len = %d, want 2 (depths 0,1 only)", len(limited.Tree))
	}
	for _, n := range limited.Tree {
		if n.Depth >= 2 {
			t.Fatalf("node at depth %d should have been excluded by MaxDepth=2", n.Depth)
		}
	}
	if !limited.Truncated {
		t.Fatalf("Truncated = false, want true")
	}
	if limited.NodesTotal != 3 {
		t.Fatalf("NodesTotal = %d, want 3 (true total unaffected by MaxDepth)", limited.NodesTotal)
	}
}

// TestSnapshot_BackendNeverAddedIsBestEffort exercises the degraded path: a
// backend that was selected/committed before the registry's AddBackend hook
// ran (trieStore.get's lazy-create). Its blocks must still show up in the
// tree — only Healthy/Inflight, which need the *registry.Backend reference
// AddBackend supplies, are best-effort and come back unset.
func TestSnapshot_BackendNeverAddedIsBestEffort(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 1)
	// Deliberately skip p.AddBackend(bs[0]) — Commit alone must still work
	// via the lazy get() path.
	p.Commit(bs[0], req(units(42)))

	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Backends) != 1 {
		t.Fatalf("len(Backends) = %d, want 1", len(snap.Backends))
	}
	b := snap.Backends[0]
	if b.URL != bs[0].URL {
		t.Fatalf("URL = %q, want %q", b.URL, bs[0].URL)
	}
	if b.Healthy != nil {
		t.Fatalf("Healthy = %v, want nil (backend was never registered via AddBackend)", *b.Healthy)
	}
	if len(snap.Tree) != 1 || !snap.Tree[0].Present[0] {
		t.Fatalf("expected the committed block present in the tree: %+v", snap.Tree)
	}
}

// TestSnapshot_ConcurrentWithCommit is the concurrent-commit-safety case:
// Snapshot polled (as /router-viz/data would, roughly once a second, here
// hammered far harder) while Commit keeps mutating the same backends' tries.
// Uses the unlimited default so it also stresses the now-uncapped fetch
// path. Run with -race for a meaningful check.
func TestSnapshot_ConcurrentWithCommit(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 3)
	for _, b := range bs {
		p.AddBackend(b)
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for _, b := range bs {
		b := b
		wg.Add(1)
		go func() {
			defer wg.Done()
			i := 0
			for {
				select {
				case <-stop:
					return
				default:
				}
				p.Commit(b, req(units(uint64(i), uint64(i)+500_000)))
				i++
			}
		}()
	}

	for i := 0; i < 200; i++ {
		snap := p.Snapshot(viz.SnapshotOptions{})
		if len(snap.Backends) != 3 {
			t.Errorf("iteration %d: len(Backends) = %d, want 3", i, len(snap.Backends))
		}
	}
	close(stop)
	wg.Wait()
}
