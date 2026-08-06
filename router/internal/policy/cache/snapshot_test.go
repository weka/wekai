package cache_test

import (
	"sort"
	"sync"
	"testing"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/policy"
	cachepolicy "github.com/weka/wekai/router/internal/policy/cache"
)

// concatUnits joins two Unit chains for a Commit call that shares a and then
// diverges into b.
func concatUnits(a, b []kvcache.Unit) []kvcache.Unit {
	out := make([]kvcache.Unit, 0, len(a)+len(b))
	out = append(out, a...)
	out = append(out, b...)
	return out
}

// TestSnapshot_Empty is the "no traffic yet" state /router-viz must render
// sensibly for: an active policy with backends added but nothing committed.
func TestSnapshot_Empty(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2)
	for _, b := range bs {
		p.AddBackend(b)
	}

	snap := p.Snapshot(50)
	if !snap.PolicyActive {
		t.Fatalf("PolicyActive = false, want true")
	}
	if len(snap.Backends) != 2 {
		t.Fatalf("len(Backends) = %d, want 2", len(snap.Backends))
	}
	if len(snap.Blocks) != 0 {
		t.Fatalf("len(Blocks) = %d, want 0 (nothing committed yet)", len(snap.Blocks))
	}
	if snap.AvgCopies != 0 {
		t.Fatalf("AvgCopies = %v, want 0", snap.AvgCopies)
	}
	if snap.ChainsShown != 0 || snap.ChainsTotal != 0 || snap.Truncated {
		t.Fatalf("unexpected chain counters on an empty snapshot: %+v", snap)
	}
	for _, b := range snap.Backends {
		if len(b.Present) != 0 {
			t.Fatalf("backend %s has %d Present entries, want 0", b.URL, len(b.Present))
		}
	}
}

// TestSnapshot_Populated commits a shared prefix plus a diverging tail from
// two backends and checks every field the /router-viz map depends on:
// column alignment (the shared block lands in the SAME column on both
// rows), per-backend presence bits, health/inflight, and AvgCopies
// reflecting that one block is duplicated and the rest are not.
func TestSnapshot_Populated(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 2) // bs[0].URL < bs[1].URL by construction (w0, w1)
	for _, b := range bs {
		p.AddBackend(b)
	}
	bs[0].AddInflight(3)

	shared := units(100)
	p.Commit(bs[0], req(concatUnits(shared, units(200))))
	p.Commit(bs[1], req(concatUnits(shared, units(300))))

	snap := p.Snapshot(50)
	if !snap.PolicyActive {
		t.Fatalf("PolicyActive = false, want true")
	}
	if len(snap.Backends) != 2 {
		t.Fatalf("len(Backends) = %d, want 2", len(snap.Backends))
	}
	if !sort.SliceIsSorted(snap.Backends, func(i, j int) bool { return snap.Backends[i].URL < snap.Backends[j].URL }) {
		t.Fatalf("Backends not sorted by URL: %+v", snap.Backends)
	}
	if len(snap.Blocks) != 3 {
		t.Fatalf("len(Blocks) = %d, want 3 (1 shared + 2 divergent)", len(snap.Blocks))
	}

	hex100, hex200, hex300 := kvcache.HexHash(100), kvcache.HexHash(200), kvcache.HexHash(300)

	// Find the shared block's column and confirm BOTH backends show present
	// there, aligned at the identical index.
	sharedCol := -1
	for i, blk := range snap.Blocks {
		if blk.Hash == hex100 {
			sharedCol = i
		}
	}
	if sharedCol < 0 {
		t.Fatalf("shared block (hash 100) missing from Blocks: %+v", snap.Blocks)
	}
	for _, b := range snap.Backends {
		if !b.Present[sharedCol] {
			t.Fatalf("backend %s should show the shared block present at column %d: %+v", b.URL, sharedCol, b.Present)
		}
	}

	var w0, w1 int = -1, -1
	for i, b := range snap.Backends {
		switch b.URL {
		case bs[0].URL:
			w0 = i
		case bs[1].URL:
			w1 = i
		}
	}
	if w0 < 0 || w1 < 0 {
		t.Fatalf("expected both backend URLs in the snapshot: %+v", snap.Backends)
	}

	// Each backend's own divergent tail must NOT show present on the other.
	col200, col300 := -1, -1
	for i, blk := range snap.Blocks {
		if blk.Hash == hex200 {
			col200 = i
		}
		if blk.Hash == hex300 {
			col300 = i
		}
	}
	if col200 < 0 || col300 < 0 {
		t.Fatalf("divergent blocks missing from Blocks: %+v", snap.Blocks)
	}
	if !snap.Backends[w0].Present[col200] || snap.Backends[w0].Present[col300] {
		t.Fatalf("backend w0 presence wrong: col200=%v col300=%v, want true/false",
			snap.Backends[w0].Present[col200], snap.Backends[w0].Present[col300])
	}
	if !snap.Backends[w1].Present[col300] || snap.Backends[w1].Present[col200] {
		t.Fatalf("backend w1 presence wrong: col300=%v col200=%v, want true/false",
			snap.Backends[w1].Present[col300], snap.Backends[w1].Present[col200])
	}

	// AvgCopies: shared block held by 2, the two divergent blocks by 1 each
	// -> mean copies = (2+1+1)/3.
	wantAvg := 4.0 / 3.0
	if diff := snap.AvgCopies - wantAvg; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("AvgCopies = %v, want %v", snap.AvgCopies, wantAvg)
	}

	if snap.Backends[w0].Healthy == nil || !*snap.Backends[w0].Healthy {
		t.Fatalf("w0.Healthy = %v, want true (added via AddBackend and marked Healthy)", snap.Backends[w0].Healthy)
	}
	if snap.Backends[w0].Inflight != 3 {
		t.Fatalf("w0.Inflight = %d, want 3", snap.Backends[w0].Inflight)
	}
}

// TestSnapshot_BackendNeverAddedIsBestEffort exercises the degraded path: a
// backend that was selected/committed before the registry's AddBackend hook
// ran (trieStore.get's lazy-create). Its trie and blocks must still show up
// — only Healthy/Inflight, which need the *registry.Backend reference
// AddBackend supplies, are best-effort and come back unset.
func TestSnapshot_BackendNeverAddedIsBestEffort(t *testing.T) {
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	bs := backends(t, 1)
	// Deliberately skip p.AddBackend(bs[0]) — Commit alone must still work
	// via the lazy get() path.
	p.Commit(bs[0], req(units(42)))

	snap := p.Snapshot(50)
	if len(snap.Backends) != 1 {
		t.Fatalf("len(Backends) = %d, want 1", len(snap.Backends))
	}
	b := snap.Backends[0]
	if b.URL != bs[0].URL {
		t.Fatalf("URL = %q, want %q", b.URL, bs[0].URL)
	}
	if len(b.Present) != 1 || !b.Present[0] {
		t.Fatalf("Present = %v, want [true]", b.Present)
	}
	if b.Healthy != nil {
		t.Fatalf("Healthy = %v, want nil (backend was never registered via AddBackend)", *b.Healthy)
	}
}

// TestSnapshot_ConcurrentWithCommit is the concurrent-commit-safety case:
// Snapshot polled (as /router-viz/data would, roughly once a second, here
// hammered far harder) while Commit keeps mutating the same backends' tries.
// Run with -race for a meaningful check.
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
		snap := p.Snapshot(30)
		if len(snap.Backends) != 3 {
			t.Errorf("iteration %d: len(Backends) = %d, want 3", i, len(snap.Backends))
		}
	}
	close(stop)
	wg.Wait()
}
