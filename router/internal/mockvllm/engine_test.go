package mockvllm

import (
	"fmt"
	"strings"
	"testing"
	"time"
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

// serveOnce simulates a request that is admitted and immediately released —
// i.e. it never overlaps with anything else in flight — for tests that only
// care about steady-state cache accounting (hit counting, chain hashing,
// eviction of UNPINNED content), not the in-flight pinning window itself.
// This is the direct successor to the old Engine.Serve, which this test file
// used before Admit absorbed pinning into admission.
func serveOnce(e *Engine, prompt string) (cached, total int) {
	release, cached, total, ok := e.Admit(e.Tokenize(prompt))
	if !ok {
		return 0, 0
	}
	release()
	return cached, total
}

func TestEngine_ColdRequestHasZeroCache(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16})
	prompt := repeatWord("alpha", 200) // several blocks
	cached, total := serveOnce(e, prompt)
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

	_, total1 := serveOnce(e, prompt)
	cached2, total2 := serveOnce(e, prompt)

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

	serveOnce(e, shared+" charlie")
	cached, total := serveOnce(e, shared+" delta")

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

	serveOnce(e, repeatWord("alpha-prefix", 200)+" "+commonTail)
	cached, total := serveOnce(e, repeatWord("bravo-prefix", 200)+" "+commonTail)

	if cached != 0 {
		t.Fatalf("a shared tail behind a DIFFERENT prefix must not be credited (chain hashing), got cached=%d of %d", cached, total)
	}
}

func TestEngine_LRUEvictionForgetsOldBlocks(t *testing.T) {
	// A tiny capacity so a second, unrelated prompt evicts the first entirely.
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: 4})

	first := repeatWord("evictme", 200)
	serveOnce(e, first)

	// Push enough distinct content through to overrun the 4-block capacity
	// several times over, so the first prompt's blocks are certainly reclaimed.
	for i := 0; i < 20; i++ {
		serveOnce(e, repeatWord("filler"+string(rune('a'+i)), 200))
	}

	cached, total := serveOnce(e, first)
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
	serveOnce(e, first)

	for i := 0; i < 50; i++ {
		serveOnce(e, repeatWord("noise"+string(rune('a'+i%26)), 200))
	}

	cached, total := serveOnce(e, first)
	if cached != total {
		t.Fatalf("unbounded cache should never forget: cached=%d total=%d", cached, total)
	}
}

func TestEngine_AdmitRejectsPastMaxConcurrency(t *testing.T) {
	e := NewEngine(Config{MaxConcurrency: 2})
	units := e.Tokenize("occupy a slot")

	rel1, _, _, ok1 := e.Admit(units)
	rel2, _, _, ok2 := e.Admit(units)
	if !ok1 || !ok2 {
		t.Fatalf("expected the first two admissions to succeed: ok1=%v ok2=%v", ok1, ok2)
	}

	_, _, _, ok3 := e.Admit(units)
	if ok3 {
		t.Fatalf("third admission at MaxConcurrency=2 should have been rejected")
	}
	if got := e.Stats().Rejected; got != 1 {
		t.Fatalf("expected Rejected=1, got %d", got)
	}

	rel1()
	rel4, _, _, ok4 := e.Admit(units)
	if !ok4 {
		t.Fatalf("admission should succeed again after a slot is released")
	}

	rel2()
	rel4()
}

func TestEngine_AdmitUnboundedNeverRejects(t *testing.T) {
	e := NewEngine(Config{MaxConcurrency: 0})
	units := e.Tokenize("unbounded concurrency")
	var releases []func()
	for i := 0; i < 500; i++ {
		rel, _, _, ok := e.Admit(units)
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

// TestEngine_RejectedAdmissionTouchesNothing verifies the "429 never commits
// cache credit" contract: a request that never got a concurrency slot must
// leave zero trace in the cache model, exactly like a real vLLM node that
// refused the request never did any prefill.
func TestEngine_RejectedAdmissionTouchesNothing(t *testing.T) {
	e := NewEngine(Config{MaxConcurrency: 1})
	units := e.Tokenize(repeatWord("neverserved", 200))

	rel1, _, total1, ok1 := e.Admit(units)
	if !ok1 {
		t.Fatalf("first admission should succeed")
	}
	_, _, _, ok2 := e.Admit(units)
	if ok2 {
		t.Fatalf("second admission at MaxConcurrency=1 should have been rejected")
	}

	// PromptTokens must reflect exactly the ONE admitted request, not a
	// second credit for the rejected duplicate — a 429 must leave zero trace
	// in the cache model, just like a real vLLM node that refused the
	// request never did any prefill.
	if got := e.Stats().PromptTokens; got != int64(total1) {
		t.Fatalf("PromptTokens = %d, want %d (only the admitted request)", got, total1)
	}
	if got := e.Stats().Rejected; got != 1 {
		t.Fatalf("expected Rejected=1, got %d", got)
	}

	rel1()
}

// TestEngine_PinnedChainSurvivesEvictionPressureUntilRelease is the mock's
// analog of vLLM's real behavior: an admitted request's blocks — the whole
// chain, not just its own new tail — cannot be evicted while it is still
// "generating" (holding its concurrency slot / pin), even under heavy
// unrelated eviction pressure. Only after release does the same content
// become reclaimable again.
func TestEngine_PinnedChainSurvivesEvictionPressureUntilRelease(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: 6})
	pinnedPrompt := strings.Repeat("A", 140) // 3 blocks at 64 bytes/block: 64+64+12
	units := e.Tokenize(pinnedPrompt)

	release, cached, total, ok := e.Admit(units)
	if !ok {
		t.Fatalf("admit unexpectedly rejected")
	}
	if cached != 0 {
		t.Fatalf("expected a cold miss on first admission, got cached=%d", cached)
	}
	if total <= 0 {
		t.Fatalf("expected a positive token estimate, got %d", total)
	}

	// Heavy pressure from many OTHER, already-completed requests — enough to
	// cycle through the tiny 6-block cap many times over.
	for i := 0; i < 300; i++ {
		serveOnce(e, fmt.Sprintf("pressure-%d", i))
	}

	// The IN-FLIGHT request's whole chain — leaf and ancestors alike — must
	// still be fully resident.
	if c, tot := e.Query(units); c != tot {
		t.Fatalf("pinned request's blocks were evicted while still in flight: cached=%d total=%d", c, tot)
	}

	release() // the request "finishes"

	// More distinct pressure must now eventually be able to reclaim it.
	for i := 300; i < 900; i++ {
		serveOnce(e, fmt.Sprintf("pressure-%d", i))
	}
	if c, _ := e.Query(units); c == total {
		t.Fatalf("expected the released request's blocks to eventually be evicted, but they still fully hit")
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

// TestEngine_LatencyChargesColdCachedAndOutputSeparately is the rate-based
// replacement for the old per-token-duration model: cold (uncached) prompt
// tokens, cached prompt tokens, and output tokens are each charged at their
// own configurable rate, calibratable against a real fleet's measured
// throughput. "want" mirrors tokensAtRate's exact float64 formula and
// Duration-then-sum order so this stays an exact equality, not an epsilon
// comparison.
func TestEngine_LatencyChargesColdCachedAndOutputSeparately(t *testing.T) {
	e := NewEngine(Config{
		BaseLatency:    0,
		ColdInputTPS:   1000, // 1ms/token
		CachedInputTPS: 2000, // 0.5ms/token
		OutputTPS:      500,  // 2ms/token
	})
	ttft, total := e.Latency(40, 100, 10) // 40 cached, 60 uncached, 10 output

	wantTTFT := time.Duration(60.0/1000*float64(time.Second)) + time.Duration(40.0/2000*float64(time.Second))
	if ttft != wantTTFT {
		t.Fatalf("ttft = %v, want %v", ttft, wantTTFT)
	}
	wantTotal := wantTTFT + time.Duration(10.0/500*float64(time.Second))
	if total != wantTotal {
		t.Fatalf("total = %v, want %v", total, wantTotal)
	}
}

// TestEngine_CachedTokensAreNotFree guards the model change anton asked for:
// previously a cache hit contributed zero latency; now it's charged at
// CachedInputTPS (still normally far cheaper than ColdInputTPS, but never
// literally free), matching a real cache read having some cost.
func TestEngine_CachedTokensAreNotFree(t *testing.T) {
	e := NewEngine(Config{CachedInputTPS: 1000})
	ttft, _ := e.Latency(100, 100, 0) // fully cached, zero uncached
	if ttft <= 0 {
		t.Fatalf("a fully cached request should still cost time at CachedInputTPS, got ttft=%v", ttft)
	}
}

// TestEngine_ZeroRateIsInstant confirms the escape hatch a fast test suite
// needs: an unset (zero) rate contributes no time for that term, so
// Config{} stays instant by default.
func TestEngine_ZeroRateIsInstant(t *testing.T) {
	e := NewEngine(Config{})
	ttft, total := e.Latency(0, 100, 50)
	if ttft != 0 || total != 0 {
		t.Fatalf("zero-rate Config should be instant, got ttft=%v total=%v", ttft, total)
	}
}
