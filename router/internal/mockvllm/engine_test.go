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
	release := e.PinOutputBlocks(promptUnits, 100, "some reply content that would otherwise become blocks")
	after, _, _ := e.trie.Stats()
	if after != before {
		t.Fatalf("OutputKVMultiplier=0 must be a no-op: nodes before=%d after=%d", before, after)
	}
	if pinned := e.Stats().PinnedNodes; pinned != 0 {
		t.Fatalf("OutputKVMultiplier=0 must pin nothing, got PinnedNodes=%d", pinned)
	}
	release() // must be safe to call even though nothing was pinned
}

// TestEngine_OutputKVMultiplierScalesBlockCount checks the exact formula:
// ceil(outputTokens * multiplier / BlockSizeTokens).
func TestEngine_OutputKVMultiplierScalesBlockCount(t *testing.T) {
	reply := "short reply" // count is formula-driven, not content-length-driven -- see outputChain's filler fallback

	e1 := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 1.0})
	release1 := e1.PinOutputBlocks(nil, 64, reply)
	if n, _, _ := e1.trie.Stats(); n != 4 { // ceil(64*1.0/16)
		t.Fatalf("multiplier=1.0: nodes = %d, want 4", n)
	}
	release1()

	e25 := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 2.5})
	release25 := e25.PinOutputBlocks(nil, 64, reply)
	if n, _, _ := e25.trie.Stats(); n != 10 { // ceil(64*2.5/16)
		t.Fatalf("multiplier=2.5: nodes = %d, want 10", n)
	}
	release25()
}

// TestEngine_PinOutputBlocks_PinnedNodesReflectsPromptAndOutput is explicit
// ask (a): while a request is in flight — nothing released yet — PinnedNodes
// must count BOTH its prompt blocks and its output-KV blocks, not just the
// prompt. This is the "peak allocation held over the request window" anton
// asked for: output blocks exert in-flight pressure for the request's whole
// simulated duration, mirroring real vLLM's decode growing a running
// request's allocation throughout generation rather than only at the end.
func TestEngine_PinOutputBlocks_PinnedNodesReflectsPromptAndOutput(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 1.0})
	promptUnits := e.Tokenize(repeatWord("prompt", 200)) // several distinct blocks

	release, _, _, ok := e.Admit(promptUnits)
	if !ok {
		t.Fatalf("admit failed")
	}
	promptOnlyPinned := e.Stats().PinnedNodes
	if promptOnlyPinned == 0 {
		t.Fatalf("expected the prompt's own blocks to be pinned after Admit, got PinnedNodes=0")
	}

	outputTokens := 64
	reply := strings.Join(syntheticTokens(outputTokens), " ") // deterministic, as a real handler computes upfront
	releaseOutput := e.PinOutputBlocks(promptUnits, outputTokens, reply)

	withOutputPinned := e.Stats().PinnedNodes
	if withOutputPinned <= promptOnlyPinned {
		t.Fatalf("expected PinnedNodes to grow once output blocks are pinned: prompt-only=%d, with-output=%d",
			promptOnlyPinned, withOutputPinned)
	}

	// The full prompt+output chain must be resident and hit entirely while
	// still "in flight" (mirrors a slow request mid-generation: nothing
	// released yet).
	full := e.outputChain(promptUnits, outputTokens, reply)
	if c, tot := e.Query(full); c != tot {
		t.Fatalf("expected the full in-flight prompt+output chain to be fully cached: cached=%d total=%d", c, tot)
	}

	releaseOutput()
	release()
}

// TestEngine_ConcurrentInFlightOutputBlocksEvictUnpinnedCache is explicit ask
// (b): output blocks pinned by OTHER, still-in-flight requests must exert
// real capacity pressure — enough, under a tight cap, to evict unrelated
// UNPINNED cache — while every in-flight request's own chain (prompt AND
// output) stays fully resident throughout, because it's pinned.
func TestEngine_ConcurrentInFlightOutputBlocksEvictUnpinnedCache(t *testing.T) {
	const (
		requests               = 5
		promptBlocksPerRequest = 3 // strings.Repeat(letter, 140) at 64 bytes/block (64+64+12)
		outputTokens           = 32
		outputBlocksPerRequest = 2 // ceil(32*1.0/16)
		backgroundBlocks       = 3 // same 140-byte shape as each request's prompt
	)
	// blockCapacity MUST be >= the total blocks all requests end up pinning
	// (requests*(prompt+output)): pinned content is never evicted, but a
	// single request's own freshly inserted blocks are briefly UNPINNED
	// between insertion and Engine.Admit/PinOutputBlocks's own pin (see
	// kvcache.Trie.insertFrom, which evicts before the caller pins) — if the
	// cap were tighter than the eventual pinned total, a later request could
	// transiently evict its own not-yet-pinned tail when nothing else is left
	// to reclaim. One block of headroom above that total (less than adding
	// the full backgroundBlocks) is enough for background to be forced out
	// without ever touching a request's own blocks.
	blockCapacity := int64(requests*(promptBlocksPerRequest+outputBlocksPerRequest)) + 1
	if blockCapacity >= int64(requests*(promptBlocksPerRequest+outputBlocksPerRequest)+backgroundBlocks) {
		t.Fatalf("test setup: blockCapacity=%d leaves no pressure to evict background (%d blocks)", blockCapacity, backgroundBlocks)
	}
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: blockCapacity, OutputKVMultiplier: 1.0})

	// Seed unrelated, never-pinned "background" cache — comfortably under cap
	// on its own.
	background := strings.Repeat("B", 140)
	serveOnce(e, background)
	if c, tot := e.Query(e.Tokenize(background)); c != tot {
		t.Fatalf("test setup: background prompt should be fully cached before any pressure, cached=%d total=%d", c, tot)
	}

	// Several concurrent, still-in-flight requests, each pinning its own
	// prompt and output blocks — by the last one, cumulative pinned content
	// plus background exceeds cap, so eviction has no unpinned target left
	// except background.
	var releases []func()
	for i := 0; i < requests; i++ {
		prompt := strings.Repeat(string(rune('C'+i)), 140) // distinct content per request
		units := e.Tokenize(prompt)
		release, _, _, ok := e.Admit(units)
		if !ok {
			t.Fatalf("admit %d unexpectedly rejected", i)
		}
		reply := strings.Join(syntheticTokens(outputTokens), " ")
		releaseOutput := e.PinOutputBlocks(units, outputTokens, reply)
		releases = append(releases, release, releaseOutput)

		// Each in-flight request's own chain must survive its own admission,
		// regardless of how much pressure came before it.
		full := e.outputChain(units, outputTokens, reply)
		if c, tot := e.Query(full); c != tot {
			t.Fatalf("in-flight request %d's own chain was evicted while pinned: cached=%d total=%d", i, c, tot)
		}
	}

	if c, tot := e.Query(e.Tokenize(background)); c == tot {
		t.Fatalf("expected unpinned background cache to be evicted under pressure from concurrent in-flight output blocks, but it's still fully cached")
	}

	for _, r := range releases {
		r()
	}
}

// TestEngine_OutputBlocksAreEvictableAfterRelease is explicit ask (c): once a
// request's output blocks are unpinned at release, they become ordinary
// evictable cache — not permanently pinned just because they came from
// output — exactly like prompt blocks.
func TestEngine_OutputBlocksAreEvictableAfterRelease(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, BlockCapacity: 10, OutputKVMultiplier: 1.0})
	promptUnits := e.Tokenize("short prompt for the output-kv capacity test")

	release, _, _, ok := e.Admit(promptUnits)
	if !ok {
		t.Fatalf("admit failed")
	}

	outputTokens := 64
	reply := strings.Repeat("reply filler text ", 10)
	before, _, _ := e.trie.Stats()
	releaseOutput := e.PinOutputBlocks(promptUnits, outputTokens, reply)
	after, _, _ := e.trie.Stats()
	if after <= before {
		t.Fatalf("PinOutputBlocks should add nodes to the trie: before=%d after=%d", before, after)
	}

	// While still in flight, the request's whole chain — prompt AND output —
	// must be fully resident.
	full := e.outputChain(promptUnits, outputTokens, reply)
	if c, tot := e.Query(full); c != tot {
		t.Fatalf("expected the full prompt+output chain to be fully cached while in flight: cached=%d total=%d", c, tot)
	}

	release()
	releaseOutput()

	for i := 0; i < 500; i++ {
		serveOnce(e, fmt.Sprintf("pressure-%d", i))
	}
	if c, tot := e.Query(full); c == tot {
		t.Fatalf("expected this request's chain (including its output blocks) to eventually be evicted under a tiny cap after release, but it's still fully cached")
	}
}

// TestEngine_OutputBlocksHitByFollowUpTurn is the achievable-alignment case
// documented on outputChain: when the prompt ends exactly on a block
// boundary and OutputKVMultiplier=1 with a response long enough to cover
// every target block, a follow-up turn that embeds the exact same response
// text (as chatCompletionRequest.promptBytes renders an assistant message)
// hits MORE than just the original prompt — the output blocks aligned.
func TestEngine_OutputBlocksHitByFollowUpTurn(t *testing.T) {
	e := NewEngine(Config{BlockSizeTokens: 16, OutputKVMultiplier: 1.0})
	blockBytes := e.cfg.blockSizeBytes()

	// Build turn 1's prompt to land EXACTLY on a block boundary (4 blocks),
	// so outputChain's fresh-continuation assumption holds exactly — the
	// documented achievable case, not the documented limitation.
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

	release := e.PinOutputBlocks(promptUnits, outputTokens, reply)
	release() // request completes; output blocks remain as ordinary cache

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
