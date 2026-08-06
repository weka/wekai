package kvcache_test

import (
	"testing"

	"github.com/weka/wekai/kvcache"
)

// These tests pin down RecordAndPin/Unpin — the piece a live server
// (router/internal/mockvllm) needs that neither existing consumer did: a
// block belonging to an in-flight request must survive eviction pressure
// that would otherwise reclaim it, for as long as ANY in-flight request
// still references it (a refcount, not a bool — two overlapping requests
// sharing a prefix must both release before that prefix becomes evictable).

// pressure admits and immediately releases a single-block, never-repeated
// request — i.e. traffic that is NEVER pinned by the time this function
// returns, so it genuinely competes for eviction.
func pressure(t *testing.T, tr *kvcache.Trie, id int) {
	t.Helper()
	_, _, p := tr.RecordAndPin([]kvcache.Unit{{Hash: uint64(1_000_000 + id), Tokens: 5}})
	tr.Unpin(p)
}

func TestPinnedChainSurvivesEvictionPressureUntilUnpin(t *testing.T) {
	tr := kvcache.New(kvcache.Config{MaxNodes: 10})
	chain := []kvcache.Unit{{Hash: 10, Tokens: 5}, {Hash: 20, Tokens: 5}, {Hash: 30, Tokens: 5}}

	cached, total, pin := tr.RecordAndPin(chain)
	if cached != 0 {
		t.Fatalf("expected a cold miss, cached=%d", cached)
	}
	if total != 15 {
		t.Fatalf("total = %d, want 15", total)
	}

	// The WHOLE chain must be pinned, not just the leaf — an ancestor of a
	// pinned leaf must be just as protected, since real vLLM sequences
	// reference every block they hold, not only the newest one.
	if n := tr.PinnedNodes(); n != 3 {
		t.Fatalf("PinnedNodes = %d, want 3 (leaf + both ancestors)", n)
	}

	for i := 0; i < 200; i++ {
		pressure(t, tr, i)
	}

	if c, tot := tr.Query(chain); c != tot || tot != total {
		t.Fatalf("pinned chain evicted while in flight: cached=%d total=%d (want fully hit, total=%d)", c, tot, total)
	}

	tr.Unpin(pin)
	if n := tr.PinnedNodes(); n != 0 {
		t.Fatalf("PinnedNodes = %d after Unpin, want 0", n)
	}

	for i := 200; i < 400; i++ {
		pressure(t, tr, i)
	}
	if c, tot := tr.Query(chain); c == tot {
		t.Fatalf("expected the chain to eventually be evicted after Unpin, but it still fully hits (cached=%d total=%d)", c, tot)
	}

	if _, _, anomalies := tr.Stats(); anomalies != 0 {
		t.Fatalf("eviction invariant violated: anomalies=%d", anomalies)
	}
}

// TestPinRefcountProtectsSharedNodeUntilAllPinsReleased is the refcount
// case: B's entire request is exactly the shared prefix (so its own leaf IS
// the shared node), and A extends one more block past that same prefix.
// Once A releases, its own extra block becomes a genuine, structural leaf
// and is evicted under pressure — which turns the shared node into a bare
// leaf too. It must NOT be evicted at that point: B's pin on it is still
// live. Only once B also releases does the shared node's refcount reach
// zero and become reclaimable.
func TestPinRefcountProtectsSharedNodeUntilAllPinsReleased(t *testing.T) {
	tr := kvcache.New(kvcache.Config{MaxNodes: 10})

	shared := []kvcache.Unit{{Hash: 1, Tokens: 10}, {Hash: 2, Tokens: 10}}
	aChain := []kvcache.Unit{{Hash: 1, Tokens: 10}, {Hash: 2, Tokens: 10}, {Hash: 3, Tokens: 10}}

	_, _, pinB := tr.RecordAndPin(shared) // B's whole request IS the shared prefix
	_, _, pinA := tr.RecordAndPin(aChain) // A extends one more block past it

	if n := tr.PinnedNodes(); n != 3 {
		t.Fatalf("PinnedNodes = %d, want 3 (2 shared + A's own extra block)", n)
	}

	for i := 0; i < 200; i++ {
		pressure(t, tr, i)
	}
	if cached, total := tr.Query(aChain); cached != total {
		t.Fatalf("chain evicted while both A and B still in flight: cached=%d total=%d", cached, total)
	}

	tr.Unpin(pinA) // A finishes; its own block becomes evictable...
	for i := 200; i < 400; i++ {
		pressure(t, tr, i)
	}
	// ...but the SHARED portion must survive: B still holds it, by refcount,
	// even though A's departure just turned it into a bare structural leaf.
	if cached, total := tr.Query(shared); cached != total {
		t.Fatalf("shared prefix evicted while B still in flight (only A released): cached=%d total=%d", cached, total)
	}
	if _, _, anomalies := tr.Stats(); anomalies != 0 {
		t.Fatalf("eviction invariant violated after A's release: anomalies=%d", anomalies)
	}

	tr.Unpin(pinB) // B finishes too; nothing pins the shared prefix anymore
	if n := tr.PinnedNodes(); n != 0 {
		t.Fatalf("PinnedNodes = %d after both released, want 0", n)
	}
	for i := 400; i < 600; i++ {
		pressure(t, tr, i)
	}
	if cached, total := tr.Query(shared); cached == total {
		t.Fatalf("expected the shared prefix to eventually be evicted once both A and B released")
	}
	if _, _, anomalies := tr.Stats(); anomalies != 0 {
		t.Fatalf("eviction invariant violated after both released: anomalies=%d", anomalies)
	}
}

// TestUnpinIsIdempotentAgainstDoubleRelease guards the defensive branch in
// unpinChain: releasing the same Pin twice must not drive a node's refcount
// negative or corrupt PinnedNodes.
func TestUnpinIsIdempotentAgainstDoubleRelease(t *testing.T) {
	tr := kvcache.New(kvcache.Config{MaxNodes: 10})
	units := []kvcache.Unit{{Hash: 1, Tokens: 5}}
	_, _, pin := tr.RecordAndPin(units)

	tr.Unpin(pin)
	if n := tr.PinnedNodes(); n != 0 {
		t.Fatalf("PinnedNodes = %d after first Unpin, want 0", n)
	}
	tr.Unpin(pin) // double release
	if n := tr.PinnedNodes(); n != 0 {
		t.Fatalf("PinnedNodes = %d after a double Unpin, want 0 (must not go negative)", n)
	}
}

func TestUnpinNilAndEmptyPinsAreNoOps(t *testing.T) {
	tr := kvcache.New(kvcache.Config{})
	tr.Unpin(nil)
	_, _, emptyPin := tr.RecordAndPin(nil) // zero units -> empty Pin{}
	tr.Unpin(emptyPin)
	if n := tr.PinnedNodes(); n != 0 {
		t.Fatalf("PinnedNodes = %d, want 0", n)
	}
}

// TestRecordAndPinDoesNotChangeRouterBehavior is a sanity check that the new
// pinning entry points are purely additive: a Trie driven ONLY by
// Query/Commit (the router's actual usage, see router/internal/policy/cache)
// never touches PinnedNodes at all.
func TestRecordAndPinDoesNotChangeRouterBehavior(t *testing.T) {
	tr := kvcache.New(kvcache.RouterConfig())
	for i := 0; i < 50; i++ {
		units := []kvcache.Unit{{Hash: uint64(i), Tokens: 10}, {Hash: uint64(i) + 1e6, Tokens: 10}}
		if cached, total := tr.Query(units); cached != 0 || total != 20 {
			t.Fatalf("iteration %d: unexpected Query result cached=%d total=%d", i, cached, total)
		}
		tr.Commit(units)
	}
	if n := tr.PinnedNodes(); n != 0 {
		t.Fatalf("PinnedNodes = %d, want 0: Query/Commit must never pin anything", n)
	}
}
