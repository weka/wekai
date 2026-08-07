package affinity

import "testing"

// TestMarkSetGrowsBeyondOneWord is the reason markSet is a word SLICE and not a
// single uint64: an earlier draft of this design capped the fleet at 64
// backends and proposed refusing to start above it. A router fronting more
// backends than that must simply work.
func TestMarkSetGrowsBeyondOneWord(t *testing.T) {
	var m markSet
	m.Add(0)
	m.Add(63)
	m.Add(64) // first slot of the second word
	m.Add(200)

	for _, slot := range []int{0, 63, 64, 200} {
		if !m.Has(slot) {
			t.Errorf("slot %d not held after Add", slot)
		}
	}
	for _, slot := range []int{1, 62, 65, 199, 201} {
		if m.Has(slot) {
			t.Errorf("slot %d held but was never added", slot)
		}
	}
	if got := m.Count(); got != 4 {
		t.Errorf("Count = %d, want 4", got)
	}
}

// TestMarkSetHasIsSafeOutsideAllocatedWords guards the read path: walk asks
// about candidate slots on runs that may predate those backends entirely, so a
// query past the end must answer false rather than panic.
func TestMarkSetHasIsSafeOutsideAllocatedWords(t *testing.T) {
	var m markSet
	if m.Has(0) || m.Has(5000) || m.Has(-1) {
		t.Fatal("empty markSet reported a holder")
	}
	m.Add(1)
	if m.Has(5000) {
		t.Fatal("Has reported a holder past the allocated words")
	}
	if m.Empty() {
		t.Fatal("Empty reported true after Add")
	}
}

// TestMarkSetIntersectsAcrossDifferentWidths matters because the candidate mask
// and a run's marks are built independently: a run marked long ago may have one
// word while the current candidate mask has three, or the reverse.
func TestMarkSetIntersectsAcrossDifferentWidths(t *testing.T) {
	var narrow, wide markSet
	narrow.Add(3)
	wide.Add(3)
	wide.Add(150)
	if !narrow.Intersects(wide) || !wide.Intersects(narrow) {
		t.Fatal("overlapping sets of different widths did not intersect")
	}

	var disjoint markSet
	disjoint.Add(150)
	if narrow.Intersects(disjoint) || disjoint.Intersects(narrow) {
		t.Fatal("disjoint sets intersected")
	}
}

// TestMarkSetCloneIsIndependent is what splitRun relies on: both halves of a
// split start from the same holders and must then diverge without aliasing.
func TestMarkSetCloneIsIndependent(t *testing.T) {
	var orig markSet
	orig.Add(1)
	orig.Add(70)

	cp := orig.Clone()
	cp.Add(2)
	orig.Remove(70)

	if !cp.Has(1) || !cp.Has(2) || !cp.Has(70) {
		t.Error("clone lost or failed to gain a slot")
	}
	if orig.Has(2) {
		t.Error("mutating the clone changed the original")
	}
	if orig.Has(70) {
		t.Error("Remove on the original did not take effect")
	}
}

// TestMarkSetRemoveIsIdempotentAndEmptyTracks covers dropBackend, which removes
// a slot from every run whether or not that run held it.
func TestMarkSetRemoveIsIdempotentAndEmptyTracks(t *testing.T) {
	var m markSet
	m.Add(5)
	m.Remove(5)
	m.Remove(5)
	m.Remove(9999)
	if !m.Empty() {
		t.Fatal("markSet not empty after removing its only slot")
	}
	if m.Count() != 0 {
		t.Fatal("Count non-zero on an empty markSet")
	}
}

// TestMarkSetEachVisitsEveryHolderAscending is the iteration viz uses to fill
// TreeNode.Present, which must align with the sorted backend list.
func TestMarkSetEachVisitsEveryHolderAscending(t *testing.T) {
	var m markSet
	want := []int{0, 7, 64, 129}
	for _, s := range want {
		m.Add(s)
	}
	var got []int
	m.Each(func(slot int) { got = append(got, slot) })

	if len(got) != len(want) {
		t.Fatalf("Each visited %d slots, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Each visited %v, want %v", got, want)
		}
	}
}

// TestMarkSetSubset backs the descendant-subset-of-ancestor invariant assertion
// in the tree tests, so it needs to be right on differing widths too.
func TestMarkSetSubset(t *testing.T) {
	var child, parent markSet
	child.Add(2)
	parent.Add(2)
	parent.Add(100)

	if !child.Subset(parent) {
		t.Error("child should be a subset of parent")
	}
	if parent.Subset(child) {
		t.Error("parent is wider and must not be a subset of child")
	}

	var empty markSet
	if !empty.Subset(child) {
		t.Error("the empty set is a subset of everything")
	}
}
