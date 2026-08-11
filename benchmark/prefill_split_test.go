package benchmark

import (
	"strings"
	"testing"

	"github.com/weka/wekai/kvcache"
)

// testDocs is a small, fixed corpus for synthText — big enough that
// hash-derived offsets exercise real slicing/wraparound without paying for
// the 1.2MB embedded corpus in every test.
var testDocs = strings.Repeat("lorem ipsum dolor sit amet consectetur adipiscing elit ", 64)

func TestSimulatePrefillBlocks_BlockCount(t *testing.T) {
	// One message per request, no system/tools — TotalBlocks should be
	// ceil(bytes/kvcache.DefaultChunkBytes), floored at 1 block for bytes<=0,
	// exactly mirroring kvcache.ChunkContent's own windowing.
	cases := []struct {
		name   string
		bytes  int
		blocks int
	}{
		{"empty", 0, 1},
		{"one byte", 1, 1},
		{"exactly one chunk", kvcache.DefaultChunkBytes, 1},
		{"one byte over a chunk", kvcache.DefaultChunkBytes + 1, 2},
		{"exactly two chunks", 2 * kvcache.DefaultChunkBytes, 2},
		{"two chunks plus one", 2*kvcache.DefaultChunkBytes + 1, 3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := RouterReplayRequest{
				Ts: "2025-01-01T00:00:00Z",
				Messages: []RouterReplayMessage{
					{Role: "user", Hash: "m1", Bytes: c.bytes, Tokens: c.bytes},
				},
				InputTokens: c.bytes,
			}
			stats := SimulatePrefillBlocks([]RouterReplayRequest{req}, testDocs, kvcache.Config{}, nil)
			if len(stats) != 1 {
				t.Fatalf("expected 1 stat, got %d", len(stats))
			}
			if stats[0].TotalBlocks != c.blocks {
				t.Errorf("TotalBlocks = %d, want %d (bytes=%d)", stats[0].TotalBlocks, c.blocks, c.bytes)
			}
			// A single, first-ever request has nothing to match against.
			if stats[0].MissingBlocks() != c.blocks {
				t.Errorf("MissingBlocks() = %d, want %d (cold start)", stats[0].MissingBlocks(), c.blocks)
			}
		})
	}
}

func TestSimulatePrefillBlocks_MissingBlocksAcrossRequests(t *testing.T) {
	// Each message is well under one chunk (100 bytes < 1024), so every
	// message maps to exactly one router block — makes the expected
	// missing-block counts easy to reason about by hand.
	msg := func(hash string) RouterReplayMessage {
		return RouterReplayMessage{Role: "user", Hash: hash, Bytes: 100, Tokens: 25}
	}

	reqs := []RouterReplayRequest{
		// req0: cold start, two fresh blocks (A, B).
		{Ts: "2025-01-01T00:00:00Z", Messages: []RouterReplayMessage{msg("A"), msg("B")}, InputTokens: 200},
		// req1: shares leading block A, diverges at the second (C instead of B).
		{Ts: "2025-01-01T00:00:01Z", Messages: []RouterReplayMessage{msg("A"), msg("C")}, InputTokens: 200},
		// req2: identical to req0, fully warm by now.
		{Ts: "2025-01-01T00:00:02Z", Messages: []RouterReplayMessage{msg("A"), msg("B")}, InputTokens: 200},
	}

	stats := SimulatePrefillBlocks(reqs, testDocs, kvcache.Config{}, nil)
	if len(stats) != 3 {
		t.Fatalf("expected 3 stats, got %d", len(stats))
	}

	if got := stats[0]; got.TotalBlocks != 2 || got.MissingBlocks() != 2 {
		t.Errorf("req0 = %+v, want TotalBlocks=2 MissingBlocks()=2 (cold)", got)
	}
	if got := stats[1]; got.TotalBlocks != 2 || got.MissingBlocks() != 1 {
		t.Errorf("req1 = %+v, want TotalBlocks=2 MissingBlocks()=1 (shared leading block A)", got)
	}
	if got := stats[2]; got.TotalBlocks != 2 || got.MissingBlocks() != 0 {
		t.Errorf("req2 = %+v, want TotalBlocks=2 MissingBlocks()=0 (fully warm repeat of req0)", got)
	}
}

func TestSimulatePrefillBlocks_SameHashSameBytes_Memoizes(t *testing.T) {
	// A hash repeated across requests (the common case: a growing agentic
	// conversation resends earlier turns in full) must synthesize to
	// byte-identical, and therefore hash-identical, router blocks whether or
	// not the unit cache is shared — the cache is a memory/CPU optimization,
	// not a semantic one.
	msg := func(hash string) RouterReplayMessage {
		return RouterReplayMessage{Role: "user", Hash: hash, Bytes: 3000, Tokens: 750}
	}
	reqs := []RouterReplayRequest{
		{Ts: "2025-01-01T00:00:00Z", Messages: []RouterReplayMessage{msg("A")}, InputTokens: 750},
		{Ts: "2025-01-01T00:00:01Z", Messages: []RouterReplayMessage{msg("A")}, InputTokens: 750},
	}

	withCache := SimulatePrefillBlocks(reqs, testDocs, kvcache.Config{}, NewPrefillUnitCache())
	withoutCache := SimulatePrefillBlocks(reqs, testDocs, kvcache.Config{}, nil)

	for i := range reqs {
		if withCache[i] != withoutCache[i] {
			t.Errorf("req%d: with cache %+v != without cache %+v", i, withCache[i], withoutCache[i])
		}
	}
	// Second request repeats the first message verbatim: fully warm.
	if withCache[1].MissingBlocks() != 0 {
		t.Errorf("req1.MissingBlocks() = %d, want 0 (identical repeat)", withCache[1].MissingBlocks())
	}
}

func TestPrefillUnitCache_BoundedLRU(t *testing.T) {
	// Memory must not scale with input size (a real replay file runs to
	// double-digit GB) — verify the cache actually enforces its bound rather
	// than growing without limit, and that recently-touched entries survive
	// eviction (the access pattern this cache exists for: a session's own
	// history, re-referenced every turn).
	c := NewPrefillUnitCacheSize(3)
	put := func(k string) { c.put(k, []kvcache.Unit{{Hash: 1, Tokens: 1}}) }

	put("a")
	put("b")
	put("c")
	if c.ll.Len() != 3 {
		t.Fatalf("Len = %d, want 3", c.ll.Len())
	}

	// Touching "a" makes it most-recently-used, so the NEXT insert should
	// evict "b" (now the least-recently-used), not "a".
	if _, ok := c.get("a"); !ok {
		t.Fatal("expected a hit for \"a\"")
	}
	put("d")
	if c.ll.Len() != 3 {
		t.Fatalf("Len = %d, want 3 (bound must hold)", c.ll.Len())
	}
	if _, ok := c.get("a"); !ok {
		t.Error("\"a\" was evicted, want it protected by its recent touch")
	}
	if _, ok := c.get("b"); ok {
		t.Error("\"b\" should have been evicted (least recently used), but is still present")
	}
	if _, ok := c.get("c"); !ok {
		t.Error("\"c\" should still be present")
	}
	if _, ok := c.get("d"); !ok {
		t.Error("\"d\" should still be present")
	}
}

func TestAggregatePrefillSplit(t *testing.T) {
	// Coverage values chosen so MissingBlocks()/MissingTokens() reproduce the
	// table this test was originally written against: (5, 200), (4, 100),
	// (0, 0) — see the comments on kvcache.Coverage for why those are derived
	// (TotalBlocks-MatchedBlocks, TotalTokens-MatchedTokens) rather than
	// stored directly.
	stats := []PrefillBlockStat{
		{Coverage: kvcache.Coverage{MatchedBlocks: 5, TotalBlocks: 10, MatchedTokens: 800, TotalTokens: 1000}, InputTokens: 1000},
		{Coverage: kvcache.Coverage{MatchedBlocks: 6, TotalBlocks: 10, MatchedTokens: 400, TotalTokens: 500}, InputTokens: 500},
		{Coverage: kvcache.Coverage{MatchedBlocks: 3, TotalBlocks: 3, MatchedTokens: 300, TotalTokens: 300}, InputTokens: 300},
	}
	if got := stats[0].MissingBlocks(); got != 5 {
		t.Fatalf("sanity check: stats[0].MissingBlocks() = %d, want 5", got)
	}
	if got := stats[0].MissingTokens(); got != 200 {
		t.Fatalf("sanity check: stats[0].MissingTokens() = %d, want 200", got)
	}

	t.Run("threshold 4 — strictly greater than routes to prefill", func(t *testing.T) {
		r := AggregatePrefillSplit(stats, 4)
		if r.Requests != 3 {
			t.Errorf("Requests = %d, want 3", r.Requests)
		}
		// Only the first stat (5 > 4) qualifies; the second (4 > 4 is false)
		// does not — this is the "at or below routes normally" boundary.
		if r.PrefillRequests != 1 {
			t.Errorf("PrefillRequests = %d, want 1", r.PrefillRequests)
		}
		if r.TotalInputTokens != 1800 {
			t.Errorf("TotalInputTokens = %d, want 1800", r.TotalInputTokens)
		}
		if r.PrefillInputTokens != 1000 {
			t.Errorf("PrefillInputTokens = %d, want 1000", r.PrefillInputTokens)
		}
		if r.TotalMissingTokens != 300 {
			t.Errorf("TotalMissingTokens = %d, want 300", r.TotalMissingTokens)
		}
		if r.PrefillMissingTokens != 200 {
			t.Errorf("PrefillMissingTokens = %d, want 200", r.PrefillMissingTokens)
		}

		if got, want := r.RequestShare(), 1.0/3.0; !floatNear(got, want, 1e-9) {
			t.Errorf("RequestShare() = %f, want %f", got, want)
		}
		if got, want := r.InputTokenShare(), 1000.0/1800.0; !floatNear(got, want, 1e-9) {
			t.Errorf("InputTokenShare() = %f, want %f", got, want)
		}
		// Deliberately over TotalInputTokens (1800), not TotalMissingTokens
		// (300), so it stays comparable to InputTokenShare.
		if got, want := r.MissingTokenShare(), 200.0/1800.0; !floatNear(got, want, 1e-9) {
			t.Errorf("MissingTokenShare() = %f, want %f", got, want)
		}
	})

	t.Run("threshold 3 — two requests qualify", func(t *testing.T) {
		r := AggregatePrefillSplit(stats, 3)
		if r.PrefillRequests != 2 {
			t.Errorf("PrefillRequests = %d, want 2", r.PrefillRequests)
		}
		if r.PrefillInputTokens != 1500 {
			t.Errorf("PrefillInputTokens = %d, want 1500", r.PrefillInputTokens)
		}
	})

	t.Run("threshold 5 — nothing exceeds it", func(t *testing.T) {
		r := AggregatePrefillSplit(stats, 5)
		if r.PrefillRequests != 0 {
			t.Errorf("PrefillRequests = %d, want 0", r.PrefillRequests)
		}
		if r.PrefillInputTokens != 0 || r.PrefillMissingTokens != 0 {
			t.Errorf("expected zero prefill volume at threshold 5, got %+v", r)
		}
	})

	t.Run("empty input — no panic, all shares zero", func(t *testing.T) {
		r := AggregatePrefillSplit(nil, 4)
		if r.Requests != 0 || r.PrefillRequests != 0 {
			t.Errorf("expected zero report, got %+v", r)
		}
		if r.RequestShare() != 0 || r.InputTokenShare() != 0 || r.MissingTokenShare() != 0 {
			t.Errorf("expected all shares 0 on empty input, got req=%f tok=%f miss=%f",
				r.RequestShare(), r.InputTokenShare(), r.MissingTokenShare())
		}
	})
}

func floatNear(a, b, eps float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}
