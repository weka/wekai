package mockvllm

import (
	"strings"
	"testing"
)

// repeatWord returns a prompt of n space-separated copies of word, long
// enough to span several 16-token (64-byte) blocks for a short word.
func repeatWord(word string, n int) string {
	words := make([]string, n)
	for i := range words {
		words[i] = word
	}
	return strings.Join(words, " ")
}

func TestEngine_ColdRequestHasZeroCache(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16})
	prompt := repeatWord("alpha", 200) // several blocks
	cached, total := e.Serve(e.Tokenize(prompt))
	if cached != 0 {
		t.Fatalf("first-ever request should be a full miss, got cached=%d", cached)
	}
	if total <= 0 {
		t.Fatalf("expected positive estimated token total, got %d", total)
	}
}

func TestEngine_RepeatedPromptIsFullyCached(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16})
	prompt := repeatWord("bravo", 200)

	_, total1 := e.Serve(e.Tokenize(prompt))
	cached2, total2 := e.Serve(e.Tokenize(prompt))

	if total1 != total2 {
		t.Fatalf("token estimate should be stable across identical requests: %d vs %d", total1, total2)
	}
	if cached2 != total2 {
		t.Fatalf("second identical request should hit the whole prompt: cached=%d total=%d", cached2, total2)
	}
}

func TestEngine_DivergingPrefixOnlyCreditsSharedBlocks(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16})
	shared := repeatWord("shared", 200)

	e.Serve(e.Tokenize(shared + " charlie"))
	cached, total := e.Serve(e.Tokenize(shared + " delta"))

	if cached <= 0 {
		t.Fatalf("expected the long shared prefix to be credited, got cached=%d", cached)
	}
	if cached >= total {
		t.Fatalf("the diverging tail must NOT be credited: cached=%d total=%d", cached, total)
	}
}

// TestEngine_ChainHashingRequiresFullAncestorMatch is the "chain hash" case
// from the task: two prompts whose SECOND block is byte-identical but whose
// FIRST block differs must not cross-credit that second block, because a
// real vLLM block hash chains the parent hash in — matching content at the
// same position with a different history is not the same cache entry.
func TestEngine_ChainHashingRequiresFullAncestorMatch(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16})
	commonTail := repeatWord("tail", 200)

	e.Serve(e.Tokenize(repeatWord("alpha-prefix", 200) + " " + commonTail))
	cached, total := e.Serve(e.Tokenize(repeatWord("bravo-prefix", 200) + " " + commonTail))

	if cached != 0 {
		t.Fatalf("a shared tail behind a DIFFERENT prefix must not be credited (chain hashing), got cached=%d of %d", cached, total)
	}
}

func TestEngine_LRUEvictionForgetsOldBlocks(t *testing.T) {
	// A tiny capacity so a second, unrelated prompt evicts the first entirely.
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: 4})

	first := repeatWord("evictme", 200)
	e.Serve(e.Tokenize(first))

	// Push enough distinct content through to overrun the 4-block capacity
	// several times over, so the first prompt's blocks are certainly reclaimed.
	for i := 0; i < 20; i++ {
		e.Serve(e.Tokenize(repeatWord("filler"+string(rune('a'+i)), 200)))
	}

	cached, total := e.Serve(e.Tokenize(first))
	if cached == total {
		t.Fatalf("expected the original prompt to have been evicted under a 4-block cap, but it fully hit (cached=%d total=%d)", cached, total)
	}

	nodes, _, anomalies := e.trie.Stats()
	if nodes > 4 {
		t.Fatalf("node count exceeded configured BlockCapacity: nodes=%d cap=4", nodes)
	}
	if anomalies != 0 {
		t.Fatalf("eviction invariant violated: anomalies=%d", anomalies)
	}
}

func TestEngine_UnboundedCapacityNeverEvicts(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: 0})
	first := repeatWord("neverforget", 200)
	e.Serve(e.Tokenize(first))

	for i := 0; i < 50; i++ {
		e.Serve(e.Tokenize(repeatWord("noise"+string(rune('a'+i%26)), 200)))
	}

	cached, total := e.Serve(e.Tokenize(first))
	if cached != total {
		t.Fatalf("unbounded cache should never forget: cached=%d total=%d", cached, total)
	}
}

func TestEngine_AdmitRejectsPastMaxConcurrency(t *testing.T) {
	e := NewEngine(Config{MaxConcurrency: 2})

	rel1, ok1 := e.Admit()
	rel2, ok2 := e.Admit()
	if !ok1 || !ok2 {
		t.Fatalf("expected the first two admissions to succeed: ok1=%v ok2=%v", ok1, ok2)
	}

	_, ok3 := e.Admit()
	if ok3 {
		t.Fatalf("third admission at MaxConcurrency=2 should have been rejected")
	}
	if got := e.Stats().Rejected; got != 1 {
		t.Fatalf("expected Rejected=1, got %d", got)
	}

	rel1()
	rel4, ok4 := e.Admit()
	if !ok4 {
		t.Fatalf("admission should succeed again after a slot is released")
	}

	rel2()
	rel4()
}

func TestEngine_AdmitUnboundedNeverRejects(t *testing.T) {
	e := NewEngine(Config{MaxConcurrency: 0})
	var releases []func()
	for i := 0; i < 500; i++ {
		rel, ok := e.Admit()
		if !ok {
			t.Fatalf("MaxConcurrency=0 must never reject (request %d)", i)
		}
		releases = append(releases, rel)
	}
	for _, r := range releases {
		r()
	}
	if got := e.Stats().Inflight; got != 0 {
		t.Fatalf("expected Inflight=0 after releasing everything, got %d", got)
	}
}

func TestEngine_MaxTokensOrDefault(t *testing.T) {
	e := NewEngine(Config{DefaultMaxTokens: 64})
	if got := e.MaxTokensOrDefault(0); got != 64 {
		t.Fatalf("expected default 64 for an absent max_tokens, got %d", got)
	}
	if got := e.MaxTokensOrDefault(-5); got != 64 {
		t.Fatalf("expected default 64 for a negative max_tokens, got %d", got)
	}
	if got := e.MaxTokensOrDefault(10); got != 10 {
		t.Fatalf("expected the requested value to pass through, got %d", got)
	}
}

func TestEngine_LatencyCountsOnlyUncachedPrefill(t *testing.T) {
	e := NewEngine(Config{
		BaseLatency:     0,
		PrefillPerToken: 1_000_000, // 1ms/token in nanoseconds
		DecodePerToken:  2_000_000, // 2ms/token
	})
	ttft, total := e.Latency(40, 100, 10) // 60 uncached, 10 output
	if want := 60 * 1_000_000; int64(ttft) != int64(want) {
		t.Fatalf("ttft = %v, want %dns", ttft, want)
	}
	if want := int64(ttft) + 10*2_000_000; int64(total) != want {
		t.Fatalf("total = %v, want %dns", total, want)
	}
}
