package kvcache

import (
	"math"
	"testing"
)

// Coverage is the shared definition of "how much of this request is already
// held" — the router's affinity flow, the offline replay analyzer and the
// dashboards all score through it, so the arithmetic is pinned here rather than
// left to whichever caller reads it first.
//
// The distinction the tests below exist for: blocks are VARIABLE-SIZED. A
// 180-byte conversational turn and a 1024-byte system chunk are both one block,
// so a block share and a token share are different numbers, and mixing them is
// what made router_cache_predicted_fraction incomparable with the observed
// fraction it is plotted against.

func units(tokens ...int32) []Unit {
	u := make([]Unit, len(tokens))
	for i, t := range tokens {
		u[i] = Unit{Hash: uint64(i + 1), Tokens: t}
	}
	return u
}

func TestCoverSplitsBlocksAndTokens(t *testing.T) {
	for _, tc := range []struct {
		name          string
		tokens        []int32
		matched       int
		wantMBlk      int
		wantMTok      int
		wantMissBlk   int
		wantMissTok   int
		wantTokenFrac float64
		wantBlockFrac float64
	}{
		{
			// The case the units bug lived in: one huge leading block and three
			// small ones. A quarter of the BLOCKS is 97% of the TOKENS.
			name:   "one fat block dwarfs three thin ones",
			tokens: []int32{1000, 10, 10, 10}, matched: 1,
			wantMBlk: 1, wantMTok: 1000, wantMissBlk: 3, wantMissTok: 30,
			wantTokenFrac: 1000.0 / 1030.0, wantBlockFrac: 0.25,
		},
		{
			// And the reverse: three quarters of the blocks, a twentieth of the
			// tokens. An unweighted number would call this a 75% cache hit.
			name:   "three thin blocks against one fat tail",
			tokens: []int32{10, 10, 10, 1000}, matched: 3,
			wantMBlk: 3, wantMTok: 30, wantMissBlk: 1, wantMissTok: 1000,
			wantTokenFrac: 30.0 / 1030.0, wantBlockFrac: 0.75,
		},
		{
			name:   "uniform blocks make the two agree",
			tokens: []int32{256, 256, 256, 256}, matched: 2,
			wantMBlk: 2, wantMTok: 512, wantMissBlk: 2, wantMissTok: 512,
			wantTokenFrac: 0.5, wantBlockFrac: 0.5,
		},
		{
			name: "nothing matched", tokens: []int32{100, 200}, matched: 0,
			wantMBlk: 0, wantMTok: 0, wantMissBlk: 2, wantMissTok: 300,
			wantTokenFrac: 0, wantBlockFrac: 0,
		},
		{
			name: "everything matched", tokens: []int32{100, 200}, matched: 2,
			wantMBlk: 2, wantMTok: 300, wantMissBlk: 0, wantMissTok: 0,
			wantTokenFrac: 1, wantBlockFrac: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := Cover(units(tc.tokens...), tc.matched)
			if c.MatchedBlocks != tc.wantMBlk || c.MatchedTokens != tc.wantMTok {
				t.Errorf("matched = %d blocks / %d tokens, want %d / %d",
					c.MatchedBlocks, c.MatchedTokens, tc.wantMBlk, tc.wantMTok)
			}
			if got := c.MissingBlocks(); got != tc.wantMissBlk {
				t.Errorf("MissingBlocks = %d, want %d", got, tc.wantMissBlk)
			}
			if got := c.MissingTokens(); got != tc.wantMissTok {
				t.Errorf("MissingTokens = %d, want %d", got, tc.wantMissTok)
			}
			if got := c.TokenFraction(); math.Abs(got-tc.wantTokenFrac) > 1e-9 {
				t.Errorf("TokenFraction = %.6f, want %.6f", got, tc.wantTokenFrac)
			}
			if got := c.BlockFraction(); math.Abs(got-tc.wantBlockFrac) > 1e-9 {
				t.Errorf("BlockFraction = %.6f, want %.6f", got, tc.wantBlockFrac)
			}
		})
	}
}

// TestCoverTokenAndBlockFractionsDiverge states the property directly, because
// it is the reason both accessors exist and the reason picking the wrong one is
// a bug rather than a rounding difference.
func TestCoverTokenAndBlockFractionsDiverge(t *testing.T) {
	c := Cover(units(1000, 10, 10, 10), 1)
	if c.BlockFraction() >= c.TokenFraction() {
		t.Fatalf("block fraction %.3f is not below token fraction %.3f for a fat leading block",
			c.BlockFraction(), c.TokenFraction())
	}
	if ratio := c.TokenFraction() / c.BlockFraction(); ratio < 3 {
		t.Errorf("the two fractions differ by only %.2fx here; this fixture is supposed to make "+
			"the units mistake obvious", ratio)
	}
}

// TestCoverClampsMatched: a caller that reports a match longer than the request
// gets a coherent answer rather than a negative one. walk() cannot currently do
// this, which is exactly why it is worth pinning.
func TestCoverClampsMatched(t *testing.T) {
	u := units(100, 100)
	for _, matched := range []int{-5, 0, 2, 7} {
		c := Cover(u, matched)
		if c.MatchedBlocks < 0 || c.MatchedBlocks > len(u) {
			t.Errorf("matched=%d: MatchedBlocks = %d, out of range", matched, c.MatchedBlocks)
		}
		if c.MissingBlocks() < 0 || c.MissingTokens() < 0 {
			t.Errorf("matched=%d: negative missing (%d blocks, %d tokens)",
				matched, c.MissingBlocks(), c.MissingTokens())
		}
		if f := c.TokenFraction(); f < 0 || f > 1 {
			t.Errorf("matched=%d: TokenFraction = %v, outside [0,1]", matched, f)
		}
	}
}

func TestCoverEmptyRequest(t *testing.T) {
	c := Cover(nil, 3)
	if c.TotalBlocks != 0 || c.TotalTokens != 0 {
		t.Fatalf("empty request scored %+v", c)
	}
	if c.TokenFraction() != 0 || c.BlockFraction() != 0 {
		t.Errorf("empty request must report 0 rather than dividing by zero: %+v", c)
	}
	if c.MissingBlocks() != 0 || c.MissingTokens() != 0 {
		t.Errorf("empty request has nothing missing: %+v", c)
	}
}
