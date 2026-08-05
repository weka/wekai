package kvcache_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/weka/wekai/kvcache"
)

// The engine serves two consumers with different needs. These tests pin the
// properties each one depends on, in one place, so neither can be broken for the
// other's benefit.

// The simulator's model: unbounded, so a workload's true reuse is measured rather
// than clipped by an eviction policy.
func TestUnboundedNeverEvicts(t *testing.T) {
	tr := kvcache.New(kvcache.Config{}) // zero config = simulator semantics
	for i := 0; i < 5000; i++ {
		tr.RecordAndCount([]kvcache.Unit{
			{Hash: uint64(i), Tokens: 100}, {Hash: uint64(i) + 1e6, Tokens: 100},
		})
	}
	nodes, _, anomalies := tr.Stats()
	if nodes != 10000 {
		t.Errorf("nodes = %d, want 10000: an unbounded trie must not evict", nodes)
	}
	if anomalies != 0 {
		t.Errorf("anomalies = %d", anomalies)
	}
}

// The router's model: bounded, because a real vLLM node evicts under memory
// pressure and an unbounded model drifts optimistic without limit.
func TestRouterConfigIsBounded(t *testing.T) {
	cfg := kvcache.RouterConfig()
	if !cfg.Bounded() {
		t.Fatal("RouterConfig must be bounded")
	}
	tr := kvcache.New(kvcache.Config{MaxNodes: 100, EvictBudget: 8})
	for i := 0; i < 5000; i++ {
		tr.Commit([]kvcache.Unit{{Hash: uint64(i), Tokens: 10}})
	}
	nodes, _, anomalies := tr.Stats()
	if nodes > 100 {
		t.Errorf("nodes = %d exceeds the cap: eviction is not keeping up", nodes)
	}
	if anomalies != 0 {
		t.Errorf("anomalies = %d: the leaves-only LRU invariant is broken", anomalies)
	}
}

// Query must be pure and RecordAndCount must not be. The router relies on the
// first (it queries every candidate) and the simulator on the second.
func TestQueryPureRecordAndCountNot(t *testing.T) {
	u := []kvcache.Unit{{Hash: 1, Tokens: 10}, {Hash: 2, Tokens: 10}}

	pure := kvcache.New(kvcache.Config{})
	for i := 0; i < 100; i++ {
		pure.Query(u)
	}
	if n, _, _ := pure.Stats(); n != 0 {
		t.Errorf("Query inserted %d nodes; it must be read-only", n)
	}

	mut := kvcache.New(kvcache.Config{})
	mut.RecordAndCount(u)
	if n, _, _ := mut.Stats(); n != 2 {
		t.Errorf("RecordAndCount inserted %d nodes, want 2", n)
	}
}

// Both consumers depend on a shared leading prefix producing shared leading
// units. This is the property the whole idea rests on.
func TestSharedPrefixYieldsSharedUnits(t *testing.T) {
	sys := strings.Repeat("you are a helpful assistant. ", 100)
	a := kvcache.ChunkContent("system", []byte(sys+"question one"), kvcache.DefaultChunkBytes)
	b := kvcache.ChunkContent("system", []byte(sys+"question two"), kvcache.DefaultChunkBytes)

	shared := 0
	for i := 0; i < len(a) && i < len(b) && a[i].Hash == b[i].Hash; i++ {
		shared++
	}
	if shared < 2 {
		t.Errorf("only %d shared units for an identical ~2.9KB prefix", shared)
	}
	if shared == len(a) && len(a) == len(b) {
		t.Error("the differing tails produced identical unit streams")
	}
}

// HashLabel must parse wekai's capture format losslessly: those hashes come from
// redacted captures and are the only link between an offline analysis and the
// live prediction.
func TestHashLabelParsesCaptureFormatLosslessly(t *testing.T) {
	for _, hexPart := range []string{"0123456789abcdef", "ffffffffffffffff", "0000000000000001"} {
		withPrefix := kvcache.HashLabel("sha256:" + hexPart)
		bare := kvcache.HashLabel(hexPart)
		if withPrefix != bare {
			t.Errorf("%q: prefixed %d != bare %d", hexPart, withPrefix, bare)
		}
		if got := kvcache.HexHash(withPrefix); got != hexPart {
			t.Errorf("round trip: %q -> %d -> %q", hexPart, withPrefix, got)
		}
	}
	// A non-16-hex label must still produce a stable key rather than colliding.
	x, y := kvcache.HashLabel("not-a-hash"), kvcache.HashLabel("also-not-a-hash")
	if x == y {
		t.Error("distinct opaque labels collided")
	}
}

// The tag is length-prefixed, so (tag, content) is unambiguous. Without that,
// tag="user\x00"+content="X" and tag="user"+content="\x00X" collide, and both are
// caller-controlled — a client could craft content that steals another's affinity.
func TestTagAndContentCannotBeConfused(t *testing.T) {
	a := kvcache.HashContent("user\x00", []byte("X"))
	b := kvcache.HashContent("user", []byte("\x00X"))
	if a == b {
		t.Error("tag/content boundary is ambiguous: a crafted role can forge a content hash")
	}
}

// A continuation window must not be confusable with a first window of the same
// bytes, or a multi-segment message could be forged by splitting it differently.
func TestContinuationWindowsAreDistinct(t *testing.T) {
	long := []byte(strings.Repeat("a", 2048))
	units := kvcache.ChunkContent("user", long, 1024)
	if len(units) != 2 {
		t.Fatalf("got %d units, want 2", len(units))
	}
	first := kvcache.ChunkContent("user", long[:1024], 1024)
	if units[1].Hash == first[0].Hash {
		t.Error("a continuation window hashes the same as a first window of equal bytes")
	}
}

func TestEstimateTokens(t *testing.T) {
	cases := map[int]int32{0: 0, 1: 1, 3: 1, 4: 1, 400: 100, 1024: 256}
	for in, want := range cases {
		if got := kvcache.EstimateTokens(in); got != want {
			t.Errorf("EstimateTokens(%d) = %d, want %d", in, got, want)
		}
	}
}

func BenchmarkQuery32Units(b *testing.B) {
	tr := kvcache.New(kvcache.RouterConfig())
	u := make([]kvcache.Unit, 32)
	for i := range u {
		u[i] = kvcache.Unit{Hash: uint64(i), Tokens: 256}
	}
	tr.Commit(u)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr.Query(u)
	}
}

func ExampleTrie_Query() {
	tr := kvcache.New(kvcache.RouterConfig())
	units := kvcache.ChunkContent("system", []byte("shared instructions"), 1024)
	tr.Commit(units)
	cached, total := tr.Query(units)
	fmt.Println(cached == total, total > 0)
	// Output: true true
}
