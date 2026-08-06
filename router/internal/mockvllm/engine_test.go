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

// TestEngine_PrefillWorkChargesColdAndCachedSeparately is the rate-based
// replacement for the old per-token-duration model: cold (uncached) prompt
// tokens and cached prompt tokens are each charged at their own
// configurable rate, calibratable against a real fleet's measured
// throughput, and decode is charged separately at OutputTPS. "want" mirrors
// tokensAtRate's exact float64 formula so this stays an exact equality, not
// an epsilon comparison — PrefillWork/DecodeDuration are pure functions (no
// waiting), unlike AwaitTTFT below, which is why this can assert exact
// values rather than a wall-clock tolerance.
func TestEngine_PrefillWorkChargesColdAndCachedSeparately(t *testing.T) {
	e := NewEngine(Config{
		BaseLatency:    0,
		ColdInputTPS:   1000, // 1ms/token
		CachedInputTPS: 2000, // 0.5ms/token
		OutputTPS:      500,  // 2ms/token
	})
	defer e.Close()
	work := e.PrefillWork(40, 100) // 40 cached, 60 uncached

	wantWork := time.Duration(60.0/1000*float64(time.Second)) + time.Duration(40.0/2000*float64(time.Second))
	if work != wantWork {
		t.Fatalf("work = %v, want %v", work, wantWork)
	}
	decode := e.DecodeDuration(10)
	wantDecode := time.Duration(10.0 / 500 * float64(time.Second))
	if decode != wantDecode {
		t.Fatalf("decode = %v, want %v", decode, wantDecode)
	}
}

// TestEngine_CachedTokensAreNotFree guards the model change anton asked for:
// previously a cache hit contributed zero latency; now it's charged at
// CachedInputTPS (still normally far cheaper than ColdInputTPS, but never
// literally free), matching a real cache read having some cost.
func TestEngine_CachedTokensAreNotFree(t *testing.T) {
	e := NewEngine(Config{CachedInputTPS: 1000})
	defer e.Close()
	work := e.PrefillWork(100, 100) // fully cached, zero uncached
	if work <= 0 {
		t.Fatalf("a fully cached request should still cost time at CachedInputTPS, got work=%v", work)
	}
}

// TestEngine_ZeroRateIsInstant confirms the escape hatch a fast test suite
// needs: an unset (zero) rate contributes no time for that term, so
// Config{} stays instant by default.
func TestEngine_ZeroRateIsInstant(t *testing.T) {
	e := NewEngine(Config{})
	defer e.Close()
	work := e.PrefillWork(0, 100)
	decode := e.DecodeDuration(50)
	if work != 0 || decode != 0 {
		t.Fatalf("zero-rate Config should be instant, got work=%v decode=%v", work, decode)
	}
}

// TestEngine_CharsPerTokenAffectsTokenAndBlockCounts is the calibration
// property anton needs: the SAME byte content tokenized at a lower
// chars-per-token ratio (closer to real vLLM's actual ~2.9-3.4 for dense
// agentic text) must yield proportionally MORE tokens (and, since the byte
// window per block shrinks too, more blocks) than the historical flat 4.0.
func TestEngine_CharsPerTokenAffectsTokenAndBlockCounts(t *testing.T) {
	body := repeatWord("x", 400) // ~799 bytes, fixed, used identically on both engines

	e40 := NewEngine(Config{BlockSizeTokens: 16, CharsPerToken: 4.0})
	_, total40 := serveOnce(e40, body)
	nodes40, _, _ := e40.trie.Stats()

	e32 := NewEngine(Config{BlockSizeTokens: 16, CharsPerToken: 3.2})
	_, total32 := serveOnce(e32, body)
	nodes32, _, _ := e32.trie.Stats()

	if total32 <= total40 {
		t.Fatalf("chars-per-token=3.2 should yield MORE tokens than 4.0 for identical content: got %d vs %d", total32, total40)
	}
	if ratio := float64(total32) / float64(total40); ratio < 1.15 || ratio > 1.35 {
		t.Fatalf("token ratio = %v, want close to 4.0/3.2=1.25", ratio)
	}
	if nodes32 <= nodes40 {
		t.Fatalf("chars-per-token=3.2 should yield MORE blocks than 4.0 for identical content (smaller byte window per block): got %d vs %d", nodes32, nodes40)
	}
}

// TestEngine_CharsPerTokenBelowZeroFallsBackToDefault is the "invalid input,
// not a meaningful off-switch" contract for CharsPerToken — unlike
// OutputKVMultiplier, 0 or negative here isn't a deliberate escape hatch, it
// falls back to the historical 4.0 ratio, same as an omitted CLI flag.
func TestEngine_CharsPerTokenBelowZeroFallsBackToDefault(t *testing.T) {
	body := repeatWord("x", 400)

	eUnset := NewEngine(Config{BlockSizeTokens: 16})
	eNegative := NewEngine(Config{BlockSizeTokens: 16, CharsPerToken: -1})
	eExplicit4 := NewEngine(Config{BlockSizeTokens: 16, CharsPerToken: 4.0})

	_, totalUnset := serveOnce(eUnset, body)
	_, totalNegative := serveOnce(eNegative, body)
	_, totalExplicit := serveOnce(eExplicit4, body)

	if totalUnset != totalExplicit || totalNegative != totalExplicit {
		t.Fatalf("unset/negative CharsPerToken should behave exactly like an explicit 4.0: unset=%d negative=%d explicit=%d",
			totalUnset, totalNegative, totalExplicit)
	}
}

// TestEngine_OutputKVMultiplierZeroDisablesModeling is the escape hatch:
// today's/historical behavior, outputs stay entirely invisible to the cache.
func TestEngine_OutputKVMultiplierZeroDisablesModeling(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 0})
	promptUnits := e.Tokenize("prompt text for the disabled case")
	before, _, _ := e.trie.Stats()
	e.AppendOutputBlocks(promptUnits, 100, "some reply content that would otherwise become blocks")
	after, _, _ := e.trie.Stats()
	if after != before {
		t.Fatalf("OutputKVMultiplier=0 must be a no-op: nodes before=%d after=%d", before, after)
	}
}

// TestEngine_OutputKVMultiplierScalesBlockCount checks the exact formula:
// ceil(outputTokens * multiplier / BlockSizeTokens).
func TestEngine_OutputKVMultiplierScalesBlockCount(t *testing.T) {
	reply := "short reply" // count is formula-driven, not content-length-driven -- see AppendOutputBlocks's filler fallback

	e1 := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 1.0})
	e1.AppendOutputBlocks(nil, 64, reply)
	if n, _, _ := e1.trie.Stats(); n != 4 { // ceil(64*1.0/16)
		t.Fatalf("multiplier=1.0: nodes = %d, want 4", n)
	}

	e25 := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 2.5})
	e25.AppendOutputBlocks(nil, 64, reply)
	if n, _, _ := e25.trie.Stats(); n != 10 { // ceil(64*2.5/16)
		t.Fatalf("multiplier=2.5: nodes = %d, want 10", n)
	}
}

// TestEngine_OutputBlocksOccupyCapacityAndAreEvictable is GAP 2's primary
// claim: appended output blocks are real trie nodes (occupy capacity) and
// are NOT permanently pinned just because they came from output — once the
// request completes and releases, heavy unrelated pressure against a tiny
// cap must eventually be able to reclaim them, exactly like prompt blocks.
func TestEngine_OutputBlocksOccupyCapacityAndAreEvictable(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: 10, OutputKVMultiplier: 1.0})
	promptUnits := e.Tokenize("short prompt for the output-kv capacity test")

	release, _, _, ok := e.Admit(promptUnits)
	if !ok {
		t.Fatalf("admit failed")
	}

	before, _, _ := e.trie.Stats()
	e.AppendOutputBlocks(promptUnits, 64, strings.Repeat("reply filler text ", 10))
	after, _, _ := e.trie.Stats()
	if after <= before {
		t.Fatalf("AppendOutputBlocks should add nodes to the trie: before=%d after=%d", before, after)
	}

	// Immediately after completion, the request's own prefix must still be
	// fully resident (it's the very content just inserted).
	if c, tot := e.Query(promptUnits); c != tot {
		t.Fatalf("expected the prompt prefix to be fully cached immediately after completion: cached=%d total=%d", c, tot)
	}

	release()

	for i := 0; i < 500; i++ {
		serveOnce(e, fmt.Sprintf("pressure-%d", i))
	}
	if c, tot := e.Query(promptUnits); c == tot {
		t.Fatalf("expected this request's chain (including its output blocks) to eventually be evicted under a tiny cap, but it's still fully cached")
	}
}

// TestEngine_OutputBlocksHitByFollowUpTurn is the achievable-alignment case
// documented on AppendOutputBlocks: when the prompt ends exactly on a block
// boundary and OutputKVMultiplier=1 with a response long enough to cover
// every target block, a follow-up turn that embeds the exact same response
// text (as chatCompletionRequest.promptBytes renders an assistant message)
// hits MORE than just the original prompt — the output blocks aligned.
func TestEngine_OutputBlocksHitByFollowUpTurn(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 1.0})
	blockBytes := e.cfg.blockSizeBytes()

	// Build turn 1's prompt to land EXACTLY on a block boundary (4 blocks),
	// so AppendOutputBlocks' fresh-continuation assumption holds exactly —
	// the documented achievable case, not the documented limitation.
	const prefix, suffix = "user:", "\n"
	targetLen := blockBytes * 4
	fillLen := targetLen - len(prefix) - len(suffix)
	turn1Prompt := prefix + strings.Repeat("A", fillLen) + suffix
	if len(turn1Prompt) != targetLen {
		t.Fatalf("test setup: turn1Prompt length = %d, want exactly %d", len(turn1Prompt), targetLen)
	}

	promptUnits := chunkContent("prompt", []byte(turn1Prompt), blockBytes, e.cfg.CharsPerToken)
	outputTokens := 80
	reply := strings.Join(syntheticTokens(outputTokens), " ") // exactly what a real handler would generate

	e.AppendOutputBlocks(promptUnits, outputTokens, reply)

	turn2Prompt := turn1Prompt + "assistant:" + reply + "\n" + "user:" + "a new question\n"
	turn2Units := chunkContent("prompt", []byte(turn2Prompt), blockBytes, e.cfg.CharsPerToken)

	cached, total := e.Query(turn2Units)
	if cached == 0 {
		t.Fatalf("expected the follow-up turn to hit at least the turn-1 prompt prefix, cached=0 of %d", total)
	}
	_, promptOnlyTotal := e.Query(promptUnits)
	if cached <= promptOnlyTotal {
		t.Fatalf("expected the follow-up to hit MORE than just the original prompt (cached=%d, promptOnlyTotal=%d) — "+
			"the output blocks should have aligned and been credited too", cached, promptOnlyTotal)
	}
}
