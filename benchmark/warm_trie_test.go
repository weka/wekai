package benchmark

import (
	"fmt"
	"strings"
	"testing"
)

// TestChunkPromptPrefixSharing verifies that two prompts where long = short +
// extra tail produce identical leading chunk hashes (prefix-sharing property).
// short is sized to an exact multiple of promptChunkBytes so all its chunks
// are full and match long's leading chunks without boundary ambiguity.
func TestChunkPromptPrefixSharing(t *testing.T) {
	// Make short exactly 4 chunks, long = short + 3 extra full chunks.
	shortLen := 4 * promptChunkBytes
	longLen := 7 * promptChunkBytes
	short := strings.Repeat("A", shortLen)
	long := short + strings.Repeat("B", longLen-shortLen)

	shortHashes, _ := chunkPromptPrefix(short)
	longHashes, _ := chunkPromptPrefix(long)

	if len(shortHashes) != 4 {
		t.Fatalf("expected 4 chunks for short, got %d", len(shortHashes))
	}
	if len(longHashes) != 7 {
		t.Fatalf("expected 7 chunks for long, got %d", len(longHashes))
	}

	// All short hashes must match the corresponding leading hashes in long.
	for i, h := range shortHashes {
		if longHashes[i] != h {
			t.Errorf("chunk[%d] mismatch: short=%s long=%s", i, h, longHashes[i])
		}
	}
	// The tail chunks of long must differ from any of short's hashes (different bytes).
	for i := len(shortHashes); i < len(longHashes); i++ {
		for j, sh := range shortHashes {
			if longHashes[i] == sh {
				t.Errorf("tail chunk[%d] of long unexpectedly matches short chunk[%d]", i, j)
			}
		}
	}
}

// TestStepPartialReuse simulates a --step series where the prompt grows by
// several full chunks each step. Verifies that the trie credits the shared
// prefix as cached and the increment as novel, and that the final Ratio is
// strictly between 0 and 1.
func TestStepPartialReuse(t *testing.T) {
	trie := newPrefixTrie()

	// Each step adds chunkStep full chunks on top of the previous prompt.
	const chunkStep = 3
	const steps = 5
	base := strings.Repeat("X", promptChunkBytes) // one chunk of repeating content

	var prevLen int
	for step := 0; step < steps; step++ {
		promptLen := (step + 1) * chunkStep * promptChunkBytes
		prompt := strings.Repeat(base, promptLen/len(base))

		hashes, tokens := chunkPromptPrefix(prompt)
		cached, total := trie.RecordAndCount(hashes, tokens)
		trie.ObserveRequest(cached, total)

		expectedChunks := promptLen / promptChunkBytes
		if len(hashes) != expectedChunks {
			t.Errorf("step %d: expected %d chunks, got %d", step, expectedChunks, len(hashes))
		}

		if step == 0 {
			// First request: all novel, nothing cached.
			if cached != 0 {
				t.Errorf("step 0: expected cached=0, got %d", cached)
			}
		} else {
			// Later requests: cached ~ previous prompt size (shared prefix).
			// The trie should credit the entire previous prompt as cached.
			expectedCachedChunks := prevLen / promptChunkBytes
			expectedCachedTokens := expectedCachedChunks * (promptChunkBytes / 4)
			// Allow ±10% tolerance.
			lo := int(float64(expectedCachedTokens) * 0.9)
			hi := int(float64(expectedCachedTokens) * 1.1)
			if cached < lo || cached > hi {
				t.Errorf("step %d: expected cached≈%d (±10%%), got %d", step, expectedCachedTokens, cached)
			}
			// Novel tokens ≈ increment size.
			novel := total - cached
			expectedNovelTokens := (promptLen - prevLen) / 4
			loN := int(float64(expectedNovelTokens) * 0.9)
			hiN := int(float64(expectedNovelTokens) * 1.1)
			if novel < loN || novel > hiN {
				t.Errorf("step %d: expected novel≈%d (±10%%), got %d", step, expectedNovelTokens, novel)
			}
		}
		prevLen = promptLen
	}

	ratio := trie.Ratio()
	if ratio <= 0 || ratio >= 1 {
		t.Errorf("expected 0 < Ratio() < 1 for growing-step series, got %f", ratio)
	}
	t.Logf("step-growth Ratio()=%.3f", ratio)
}

// TestIdenticalPromptRepeated feeds the same prompt N times. The first request
// is novel; subsequent requests are fully cached. Ratio should be ≈ (N-1)/N.
func TestIdenticalPromptRepeated(t *testing.T) {
	const N = 6
	trie := newPrefixTrie()
	prompt := strings.Repeat("Z", 5*promptChunkBytes)
	hashes, tokens := chunkPromptPrefix(prompt)

	var totalTokensSum int
	for _, tk := range tokens {
		totalTokensSum += tk
	}

	for i := 0; i < N; i++ {
		cached, total := trie.RecordAndCount(hashes, tokens)
		trie.ObserveRequest(cached, total)
		if i == 0 {
			if cached != 0 {
				t.Errorf("first request: expected cached=0, got %d", cached)
			}
		} else {
			// Should be fully cached (all hashes already in trie).
			if cached != total {
				t.Errorf("request %d: expected fully cached (cached==total==%d), got cached=%d", i, total, cached)
			}
		}
	}

	ratio := trie.Ratio()
	expected := float64(N-1) / float64(N)
	if ratio < expected-0.05 || ratio > expected+0.05 {
		t.Errorf("expected Ratio()≈%.3f, got %.3f", expected, ratio)
	}
	if N >= 4 && ratio <= 0.5 {
		t.Errorf("expected Ratio() > 0.5 for N=%d, got %.3f", N, ratio)
	}
	t.Logf("identical-repeat Ratio()=%.3f (expected≈%.3f)", ratio, expected)
}

// TestCrossSeriesIsolation verifies that two series with different leading
// bytes (different GUID prefix at chunk 0) do not share trie nodes. SeriesB's
// first request must be fully novel.
func TestCrossSeriesIsolation(t *testing.T) {
	trie := newPrefixTrie()

	// Series A: starts with GUID-A repeated, then filler.
	guidA := strings.Repeat("A", 36) // 36-byte GUID-like prefix
	seriesAPrompt := guidA + strings.Repeat("X", 5*promptChunkBytes-36)

	// Series B: starts with GUID-B (different first bytes).
	guidB := strings.Repeat("B", 36)
	seriesBPrompt := guidB + strings.Repeat("X", 5*promptChunkBytes-36)

	// Feed all of series A.
	for i := 0; i < 3; i++ {
		hA, tA := chunkPromptPrefix(seriesAPrompt)
		cA, totA := trie.RecordAndCount(hA, tA)
		trie.ObserveRequest(cA, totA)
		_ = fmt.Sprintf("A[%d] cached=%d total=%d", i, cA, totA) // suppress unused
	}

	// First request of series B must be ~all novel (different chunk[0] hash).
	hB, tB := chunkPromptPrefix(seriesBPrompt)
	cachedB, _ := trie.RecordAndCount(hB, tB)
	if cachedB != 0 {
		t.Errorf("cross-series isolation: seriesB first request got cached=%d, expected 0", cachedB)
	}
}

// TestCacheEstimatorIdenticalRepeated: Observe the same content N times;
// Ratio should be ≈ (N-1)/N.
func TestCacheEstimatorIdenticalRepeated(t *testing.T) {
	const N = 6
	e := newCacheEstimator(0) // use default chunk size
	content := strings.Repeat("Z", 5*promptChunkBytes)
	for i := 0; i < N; i++ {
		ratio := e.Observe(content)
		if i == 0 {
			if ratio != 0 {
				t.Errorf("first Observe: expected ratio=0, got %f", ratio)
			}
		} else {
			if ratio < 0.95 {
				t.Errorf("request %d: expected ratio≈1.0, got %f", i, ratio)
			}
		}
	}
	globalRatio := e.Ratio()
	expected := float64(N-1) / float64(N)
	if globalRatio < expected-0.05 || globalRatio > expected+0.05 {
		t.Errorf("Ratio()=%.3f, expected ≈%.3f", globalRatio, expected)
	}
}

// TestCacheEstimatorGrowingPrefix: a step-growth series where content grows
// by several chunks each step. Each later Observe returns partial ratio in (0,1).
func TestCacheEstimatorGrowingPrefix(t *testing.T) {
	e := newCacheEstimator(0)
	const chunkStep = 3
	const steps = 5
	base := strings.Repeat("X", promptChunkBytes)
	for step := 0; step < steps; step++ {
		promptLen := (step + 1) * chunkStep * promptChunkBytes
		content := strings.Repeat(base, promptLen/len(base))
		ratio := e.Observe(content)
		if step == 0 {
			if ratio != 0 {
				t.Errorf("step 0: expected ratio=0, got %f", ratio)
			}
		} else {
			if ratio <= 0 || ratio >= 1 {
				t.Errorf("step %d: expected 0 < ratio < 1, got %f", step, ratio)
			}
		}
	}
	globalRatio := e.Ratio()
	if globalRatio <= 0 || globalRatio >= 1 {
		t.Errorf("Ratio()=%f: expected strictly in (0,1)", globalRatio)
	}
}

// TestCacheEstimatorCrossSeriesIsolation: divergent-prefix content gets ~0 ratio.
func TestCacheEstimatorCrossSeriesIsolation(t *testing.T) {
	e := newCacheEstimator(0)
	guidA := strings.Repeat("A", 36)
	contentA := guidA + strings.Repeat("X", 5*promptChunkBytes-36)
	for i := 0; i < 3; i++ {
		e.Observe(contentA)
	}
	guidB := strings.Repeat("B", 36)
	contentB := guidB + strings.Repeat("X", 5*promptChunkBytes-36)
	ratioB := e.Observe(contentB)
	if ratioB != 0 {
		t.Errorf("cross-series isolation: expected ratio=0 for divergent prefix, got %f", ratioB)
	}
}

// TestCacheEstimatorInsertCreditsResponse: Observe(req), Insert(req+resp),
// then Observe(req+resp+u2) should credit the response as cached.
func TestCacheEstimatorInsertCreditsResponse(t *testing.T) {
	e := newCacheEstimator(0)
	req1 := strings.Repeat("R", 3*promptChunkBytes)
	resp1 := strings.Repeat("S", 2*promptChunkBytes)
	u2 := strings.Repeat("U", promptChunkBytes)

	// First observe: all novel.
	r0 := e.Observe(req1)
	if r0 != 0 {
		t.Errorf("first observe: want 0, got %f", r0)
	}

	// Insert the extended context (req1+resp1).
	e.Insert(req1 + "\x00" + resp1)

	// Now observe the continuation: req1+resp1+u2. The req1+resp1 prefix should be cached.
	fullContent := req1 + "\x00" + resp1 + "\x00" + u2
	r2 := e.Observe(fullContent)
	// Cached fraction should be > 0.5 (at least req1 is cached from first Observe,
	// and resp1 from Insert).
	if r2 < 0.5 {
		t.Errorf("continuation: expected ratio > 0.5 (resp credited), got %f", r2)
	}
}

// TestCacheEstimatorChunkSizeVariants: different chunk sizes should both work
// and produce non-zero ratios for repeated content.
func TestCacheEstimatorChunkSizeVariants(t *testing.T) {
	for _, chunkBytes := range []int{1024, 4096} {
		e := newCacheEstimator(chunkBytes)
		content := strings.Repeat("T", 10*chunkBytes)
		e.Observe(content)
		ratio := e.Observe(content)
		if ratio < 0.95 {
			t.Errorf("chunkBytes=%d: expected ratio≈1 on repeat, got %f", chunkBytes, ratio)
		}
	}
}
