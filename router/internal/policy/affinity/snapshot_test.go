package affinity

import (
	"testing"

	"github.com/weka/wekai/router/internal/viz"
)

// TestSnapshotImplementsDataSource is a compile-time-ish guard: main.go's
// cacheLifecycle embeds viz.DataSource, so a signature drift here shows up as a
// wiring failure rather than a type error at the call site.
func TestSnapshotImplementsDataSource(t *testing.T) {
	p, _ := newTestPolicy(t)
	var _ viz.DataSource = p
}

// TestSnapshotSharedPrefixAppearsOnceWithEveryHolder is the whole point of the
// page: a system prompt served by several backends is ONE common ancestor row
// with several holders, not one subtree per backend.
func TestSnapshotSharedPrefixAppearsOnceWithEveryHolder(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 3)
	for _, b := range cands {
		p.AddBackend(b)
	}

	// Three backends serve the same prefix and then diverge.
	for i, b := range cands {
		rr := req(units(1, 2, 3, uint64(100+i)))
		p.Commit(b, rr)
	}

	snap := p.Snapshot(viz.SnapshotOptions{})
	if !snap.PolicyActive {
		t.Fatal("PolicyActive false for a running cache policy")
	}
	if len(snap.Backends) != 3 {
		t.Fatalf("%d backends in the snapshot, want 3", len(snap.Backends))
	}
	for i := 1; i < len(snap.Backends); i++ {
		if snap.Backends[i-1].URL >= snap.Backends[i].URL {
			t.Fatal("backends are not sorted by URL, so Present alignment is unstable")
		}
	}

	// The shared prefix: exactly one row held by all three.
	var shared int
	for _, n := range snap.Tree {
		held := 0
		for _, ok := range n.Present {
			if ok {
				held++
			}
		}
		if held == 3 {
			shared++
			if n.RunLen != 3 {
				t.Errorf("shared row RunLen = %d, want the 3 shared blocks", n.RunLen)
			}
		}
		if len(n.Present) != len(snap.Backends) {
			t.Errorf("Present has %d entries, want %d", len(n.Present), len(snap.Backends))
		}
	}
	if shared != 1 {
		t.Errorf("%d rows are held by all three backends, want exactly 1", shared)
	}

	if snap.AvgCopies <= 1 {
		t.Errorf("AvgCopies = %v; three backends sharing a prefix must exceed 1", snap.AvgCopies)
	}
	if snap.NodesShown != snap.NodesTotal || snap.Truncated {
		t.Errorf("an uncapped snapshot reported truncation: shown=%d total=%d",
			snap.NodesShown, snap.NodesTotal)
	}
}

// TestSnapshotParentIndicesAreConsistent: the page rebuilds the tree from a
// flat array, so a Children entry pointing at a row that was never emitted
// renders a broken graph.
func TestSnapshotParentIndicesAreConsistent(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 2)
	for _, b := range cands {
		p.AddBackend(b)
	}
	for i := range 20 {
		p.Commit(cands[i%2], req(units(1, 2, uint64(200+i), uint64(300+i))))
	}

	for _, opts := range []viz.SnapshotOptions{
		{},
		{MaxRows: 5},
		{MaxDepth: 2},
		{MaxChildren: 1},
		{MaxRows: 3, MaxChildren: 1, MaxDepth: 2},
	} {
		snap := p.Snapshot(opts)
		for i, n := range snap.Tree {
			if n.Parent < -1 || n.Parent >= len(snap.Tree) {
				t.Fatalf("opts %+v: row %d has parent %d out of range", opts, i, n.Parent)
			}
			for _, c := range n.Children {
				if c <= i || c >= len(snap.Tree) {
					t.Fatalf("opts %+v: row %d has child index %d out of range", opts, i, c)
				}
				if snap.Tree[c].Parent != i {
					t.Fatalf("opts %+v: row %d claims child %d, which points at parent %d",
						opts, i, c, snap.Tree[c].Parent)
				}
			}
		}
		if opts.MaxRows > 0 && len(snap.Tree) > opts.MaxRows {
			t.Errorf("opts %+v produced %d rows", opts, len(snap.Tree))
		}
		if len(snap.Tree) < snap.NodesTotal && !snap.Truncated {
			t.Errorf("opts %+v: capped to %d of %d rows without setting Truncated",
				opts, len(snap.Tree), snap.NodesTotal)
		}
	}
}

// TestSnapshotMergesAdjacentRowsWithIdenticalHolders: the tree legitimately
// holds a parent and its only child with the same holders — splitting a run
// marks both halves identically, and nothing rejoins them if the branch that
// forced the split later expires. Two rows for one uninterrupted chain is noise
// on the page.
func TestSnapshotMergesAdjacentRowsWithIdenticalHolders(t *testing.T) {
	p, _ := newTestPolicy(t)
	cands := fleet(t, 1)
	p.AddBackend(cands[0])

	// Committing a strict prefix of an existing run splits it, leaving two
	// runs with identical holders.
	p.Commit(cands[0], req(units(1, 2, 3, 4)))
	p.Commit(cands[0], req(units(1, 2)))

	if got := countRuns(p.tree); got != 2 {
		t.Fatalf("expected the split to leave 2 runs in the tree, got %d", got)
	}
	snap := p.Snapshot(viz.SnapshotOptions{})
	if len(snap.Tree) != 1 {
		t.Fatalf("snapshot shows %d rows for one uninterrupted chain, want 1", len(snap.Tree))
	}
	if snap.Tree[0].RunLen != 4 {
		t.Errorf("merged row RunLen = %d, want 4", snap.Tree[0].RunLen)
	}
}

// TestSnapshotOnAnEmptyTree: a freshly-started router still has to render.
func TestSnapshotOnAnEmptyTree(t *testing.T) {
	p, _ := newTestPolicy(t)
	snap := p.Snapshot(viz.SnapshotOptions{})
	if !snap.PolicyActive || len(snap.Tree) != 0 || snap.NodesTotal != 0 || snap.Truncated {
		t.Fatalf("empty snapshot is wrong: %+v", snap)
	}
}
