// Package affinity routes by prefix-cache affinity over a single shared tree
// whose nodes record WHICH backends hold them, rather than one private trie
// per backend.
//
// The design is Anton's, implemented concretely in router/docs/kv-router-sim.html
// and specified in router/docs/cache-affinity-redesign.md. Three properties
// distinguish it from router/internal/policy/cache:
//
//   - There is no threshold. A request anchors on the DEEPEST run of its block
//     path that any candidate holds, however small a fraction of the request
//     that run represents. The older policies gate on cached/total over the
//     whole current request, which structurally penalizes exactly the
//     long-running sessions most worth pinning: their shared prefix is fixed
//     while their total keeps growing.
//
//   - Holding is a property of the tree, not something reconstructed by
//     querying N tries and comparing independently-computed fractions.
//
//   - Under saturation the holder set GROWS (a "split") instead of affinity
//     being abandoned for the request.
//
// This package never rejects. Admission is the gateway's: it filters
// candidates to backends under --max-node-concurrency and returns 429
// all_backends_at_capacity when none are left, so a rejection means zero idle
// slots fleet-wide, which is exactly the acceptance criterion.
package affinity

import "math/bits"

// markSet is the set of backends holding one run, addressed by slot index.
//
// Backed by a growable word slice rather than a map or a []bool because there
// is one of these per tree run and runs are the dominant allocation: 8 bytes
// per 64 backends, against ~250 bytes for a map[int]struct{} or one byte per
// backend for a []bool. There is deliberately NO ceiling on fleet size — a
// 200-backend router costs four words per run.
//
// The bitwise work is confined here. Callers say marks.Intersects(cands), not
// marks.words[0]&cands.words[0] != 0; nothing outside this file shifts a bit.
type markSet struct{ words []uint64 }

// grow extends words so that slot is addressable.
func (m *markSet) grow(slot int) {
	need := slot/64 + 1
	for len(m.words) < need {
		m.words = append(m.words, 0)
	}
}

// Has reports whether the backend in slot holds this run.
func (m markSet) Has(slot int) bool {
	w := slot / 64
	if slot < 0 || w >= len(m.words) {
		return false
	}
	return m.words[w]&(1<<(uint(slot)%64)) != 0
}

// Add records that the backend in slot holds this run. Idempotent.
func (m *markSet) Add(slot int) {
	if slot < 0 {
		return
	}
	m.grow(slot)
	m.words[slot/64] |= 1 << (uint(slot) % 64)
}

// Remove drops the backend in slot. Idempotent.
//
// Words are never shrunk: a slot is reused only after DropBackend has cleared
// it everywhere, so a run that keeps a zero high word costs 8 bytes and saves
// re-growing on the next Add.
func (m *markSet) Remove(slot int) {
	w := slot / 64
	if slot < 0 || w >= len(m.words) {
		return
	}
	m.words[w] &^= 1 << (uint(slot) % 64)
}

// Empty reports whether no backend holds this run.
func (m markSet) Empty() bool {
	for _, w := range m.words {
		if w != 0 {
			return false
		}
	}
	return true
}

// Count is the number of holders. Feeds AvgCopies, the KV-duplication metric
// whose target is ~1.0.
func (m markSet) Count() int {
	n := 0
	for _, w := range m.words {
		n += bits.OnesCount64(w)
	}
	return n
}

// Intersects reports whether any backend holds this run AND is in other.
//
// This is the availability test behind the first routing tier: a run may be
// marked by a backend that has since gone unhealthy, started draining, been
// removed, or hit its concurrency cap, and such a run must not anchor a
// request. The simulator has no case for this — its nodes never leave.
func (m markSet) Intersects(other markSet) bool {
	n := min(len(m.words), len(other.words))
	for i := range n {
		if m.words[i]&other.words[i] != 0 {
			return true
		}
	}
	return false
}

// Clone returns an independent copy. Used when a run is split mid-way: both
// halves start from the same holder set and then diverge.
func (m markSet) Clone() markSet {
	if len(m.words) == 0 {
		return markSet{}
	}
	w := make([]uint64, len(m.words))
	copy(w, m.words)
	return markSet{words: w}
}

// Each calls fn for every holder, in ascending slot order.
func (m markSet) Each(fn func(slot int)) {
	for i, w := range m.words {
		for w != 0 {
			b := bits.TrailingZeros64(w)
			fn(i*64 + b)
			w &^= 1 << uint(b)
		}
	}
}

// Subset reports whether every holder of m also holds other. Not used in
// routing: it exists so tests can assert the invariant that a descendant's
// holder set is always a subset of its ancestor's, which is what makes
// "deepest marked run" also mean "smallest, most specific candidate pool".
func (m markSet) Subset(other markSet) bool {
	for i, w := range m.words {
		var o uint64
		if i < len(other.words) {
			o = other.words[i]
		}
		if w&^o != 0 {
			return false
		}
	}
	return true
}
