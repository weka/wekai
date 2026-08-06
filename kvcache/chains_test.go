package kvcache_test

import (
	"sync"
	"testing"

	"github.com/weka/wekai/kvcache"
)

func TestChains_EmptyTrie(t *testing.T) {
	tr := kvcache.New(kvcache.Config{})
	chains, total := tr.Chains(0)
	if len(chains) != 0 || total != 0 {
		t.Fatalf("empty trie: chains=%v total=%d, want none", chains, total)
	}
}

func TestChains_SingleChain(t *testing.T) {
	tr := kvcache.New(kvcache.Config{})
	units := []kvcache.Unit{{Hash: 1, Tokens: 5}, {Hash: 2, Tokens: 5}, {Hash: 3, Tokens: 5}}
	tr.Commit(units)

	chains, total := tr.Chains(0)
	if total != 1 || len(chains) != 1 {
		t.Fatalf("total=%d len(chains)=%d, want 1 and 1", total, len(chains))
	}
	got := chains[0]
	want := []uint64{1, 2, 3}
	if len(got.Hashes) != len(want) {
		t.Fatalf("chain hashes = %v, want %v", got.Hashes, want)
	}
	for i, h := range want {
		if got.Hashes[i] != h {
			t.Fatalf("chain hashes = %v, want %v", got.Hashes, want)
		}
	}
	if len(got.Tokens) != 3 || got.Tokens[0] != 5 || got.Tokens[1] != 5 || got.Tokens[2] != 5 {
		t.Fatalf("chain tokens = %v, want [5 5 5]", got.Tokens)
	}
}

// TestChains_SharedPrefixBranchesIntoTwoChains proves the enumeration
// correctly reflects a shared prefix diverging into two sessions: both
// chains share their leading hash but diverge after it.
func TestChains_SharedPrefixBranchesIntoTwoChains(t *testing.T) {
	tr := kvcache.New(kvcache.Config{})
	tr.Commit([]kvcache.Unit{{Hash: 1, Tokens: 5}, {Hash: 2, Tokens: 5}})
	tr.Commit([]kvcache.Unit{{Hash: 1, Tokens: 5}, {Hash: 3, Tokens: 5}})

	chains, total := tr.Chains(0)
	if total != 2 || len(chains) != 2 {
		t.Fatalf("total=%d len(chains)=%d, want 2 and 2", total, len(chains))
	}
	seen := map[uint64]bool{}
	for _, c := range chains {
		if len(c.Hashes) != 2 || c.Hashes[0] != 1 {
			t.Fatalf("unexpected chain %v, want a 2-hash chain starting with 1", c.Hashes)
		}
		seen[c.Hashes[1]] = true
	}
	if !seen[2] || !seen[3] {
		t.Fatalf("expected both divergent tails (2 and 3) to appear, got %v", seen)
	}
}

// TestChains_LimitCapsButReportsTrueTotal is the "no silent truncation"
// contract: capping the returned slice must not lose track of how much was
// actually there.
func TestChains_LimitCapsButReportsTrueTotal(t *testing.T) {
	tr := kvcache.New(kvcache.Config{})
	for i := 0; i < 10; i++ {
		tr.Commit([]kvcache.Unit{{Hash: uint64(1000 + i), Tokens: 1}})
	}

	chains, total := tr.Chains(3)
	if total != 10 {
		t.Fatalf("totalLeaves = %d, want 10 (true total regardless of cap)", total)
	}
	if len(chains) != 3 {
		t.Fatalf("len(chains) = %d, want 3 (capped)", len(chains))
	}
}

func TestChains_UnlimitedReturnsEverything(t *testing.T) {
	tr := kvcache.New(kvcache.Config{})
	for i := 0; i < 25; i++ {
		tr.Commit([]kvcache.Unit{{Hash: uint64(2000 + i), Tokens: 1}})
	}
	chains, total := tr.Chains(0)
	if total != 25 || len(chains) != 25 {
		t.Fatalf("total=%d len(chains)=%d, want 25 and 25", total, len(chains))
	}
	chains, total = tr.Chains(-5) // negative also means unlimited
	if total != 25 || len(chains) != 25 {
		t.Fatalf("negative limit: total=%d len(chains)=%d, want 25 and 25", total, len(chains))
	}
}

// TestChains_ConcurrentWithCommit runs Chains() concurrently with ongoing
// Commit() traffic — the scenario a live-polled viz endpoint actually faces
// — and asserts it neither panics nor corrupts the trie (checked via
// Stats().anomalies, the eviction-invariant canary). Run with -race for a
// meaningful check of the RLock/Lock interaction.
func TestChains_ConcurrentWithCommit(t *testing.T) {
	tr := kvcache.New(kvcache.Config{MaxNodes: 200, EvictBudget: 16})

	var wg sync.WaitGroup
	stop := make(chan struct{})

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
			tr.Commit([]kvcache.Unit{
				{Hash: uint64(i), Tokens: 3},
				{Hash: uint64(i) + 1_000_000, Tokens: 3},
			})
			i++
		}
	}()

	for i := 0; i < 200; i++ {
		chains, total := tr.Chains(20)
		if total < 0 || len(chains) > 20 {
			t.Errorf("iteration %d: implausible result chains=%d total=%d", i, len(chains), total)
		}
	}
	close(stop)
	wg.Wait()

	if _, _, anomalies := tr.Stats(); anomalies != 0 {
		t.Fatalf("eviction invariant violated under concurrent Chains()+Commit(): anomalies=%d", anomalies)
	}
}
