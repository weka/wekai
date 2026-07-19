package benchmark

// Prefix-cache simulation and content-level warm estimation.
//
// Two uses:
//
//  1. Offline replay analysis (SimulateReplayCache): feeds a sorted slice of
//     RouterReplayRequest entries into a fresh prefixTrie using per-block
//     hashes and token counts. Mirrors the approach used by `wekai router
//     analyze` (simulateCache) so simulated and ground-truth ratios are
//     directly comparable.
//
//  2. Live content-level estimation (cacheEstimator): used across all
//     benchmark modes (synthetic, --step growth, router-replay). Chunks raw
//     prompt bytes into fixed-size windows, hashes each window, and walks the
//     same trie machinery to estimate what fraction of each request's prompt
//     was already seen — driving the cold/warm token split without relying on
//     the structurally-biased per-series first-request bucketing.

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"sync/atomic"
)

// prefixTrie is a concurrent trie of message-hash sequences. Each node stores
// the estimated token count of the message that created it. Insertions and
// lookups run under a single mutex — lock contention is negligible next to HTTP
// round-trips so a sharded design is not worth the complexity.
type prefixTrie struct {
	mu   sync.Mutex
	root *trieNode

	// Aggregate totals (atomic so snapshot reads don't need the trie mutex).
	cachedTokens atomic.Int64
	totalTokens  atomic.Int64

	// nodeCount is incremented on each novel tail node insertion (atomic).
	nodeCount atomic.Int64
}

type trieNode struct {
	children map[string]*trieNode
	// tokens is the estimated token count for the MESSAGE whose hash produced
	// this node — i.e., the delta this turn's message adds to the prefix. Used
	// when the walking LCP treats this node as "cached".
	tokens int
}

func newPrefixTrie() *prefixTrie {
	return &prefixTrie{root: &trieNode{children: map[string]*trieNode{}}}
}

// RecordAndCount walks the trie with `hashes` to find the longest matching
// prefix, credits `tokens[i]` for matched messages as cached, inserts any
// novel tail, and returns (cachedThisCall, totalThisCall).
//
// Callers pass the full sequence of messages that went into the request —
// system + every prior turn + the current user turn. After the server
// responds, the caller should call RecordAndCount AGAIN with the extended
// sequence (…, server-assistant-message) so future requests can LCP-match
// against it. That second call will normally credit all prior messages as
// cached plus add one novel tail node; its returned "cached" value is not
// meaningful for the current request (which already paid for prefill) and
// can be ignored, but the insertion side-effect matters.
func (t *prefixTrie) RecordAndCount(hashes []string, tokens []int) (int, int) {
	if len(hashes) == 0 {
		return 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	node := t.root
	cached := 0
	i := 0
	// Walk matching prefix.
	for ; i < len(hashes); i++ {
		child, ok := node.children[hashes[i]]
		if !ok {
			break
		}
		cached += child.tokens
		node = child
	}
	// Insert novel tail.
	for ; i < len(hashes); i++ {
		child := &trieNode{children: map[string]*trieNode{}, tokens: tokens[i]}
		node.children[hashes[i]] = child
		t.nodeCount.Add(1)
		node = child
	}

	total := 0
	for _, tk := range tokens {
		total += tk
	}
	return cached, total
}

// ObserveRequest records what a single request would have paid for against
// an infinite perfect prefix cache. Updates aggregate counters used by the
// global ratio; does not insert anything new into the trie (call
// RecordAndCount for that).
func (t *prefixTrie) ObserveRequest(cached, total int) {
	if total <= 0 {
		return
	}
	t.cachedTokens.Add(int64(cached))
	t.totalTokens.Add(int64(total))
}

// Ratio returns cached/total over all observed requests. 0 if nothing observed.
func (t *prefixTrie) Ratio() float64 {
	total := t.totalTokens.Load()
	if total <= 0 {
		return 0
	}
	return float64(t.cachedTokens.Load()) / float64(total)
}

// hashMessage produces a stable short hash for a (role, content) pair. 16
// hex chars = 64 bits, plenty of separation for the trie key and keeps the
// node map memory footprint small.
func hashMessage(role, content string) string {
	h := sha256.New()
	h.Write([]byte(role))
	h.Write([]byte{0})
	h.Write([]byte(content))
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:8])
}

// estimateTokens is the standard 4-bytes-per-token heuristic, clamped to at
// least 1 for non-empty content. This is deliberately crude: the simulator
// measures relative prefix-cache efficiency, not exact billing numbers.
func estimateTokens(content string) int {
	if len(content) == 0 {
		return 0
	}
	n := len(content) / 4
	if n < 1 {
		n = 1
	}
	return n
}

// promptChunkBytes is the granularity at which synthetic prompts are segmented
// for the prefix-trie warm estimate (~256 tokens per chunk). Smaller = finer
// reuse detection at higher trie memory/CPU cost.
const promptChunkBytes = 1024

// chunkPromptPrefixN splits a raw prompt string into fixed-size byte windows of
// chunkBytes and returns per-chunk hashes plus token estimates, suitable for
// prefixTrie.RecordAndCount. Two prompts that share a leading byte-prefix yield
// identical leading chunk hashes, so the trie credits the shared prefix as
// cached and the divergent tail as novel — i.e. the client-side estimate of
// "what fraction of this prompt was repeated", including the partial reuse that
// --step prompt growth produces.
func chunkPromptPrefixN(prompt string, chunkBytes int) (hashes []string, tokens []int) {
	for i := 0; i < len(prompt); i += chunkBytes {
		end := i + chunkBytes
		if end > len(prompt) {
			end = len(prompt)
		}
		chunk := prompt[i:end]
		hashes = append(hashes, hashMessage("chunk", chunk))
		tokens = append(tokens, estimateTokens(chunk))
	}
	return
}

// chunkPromptPrefix is a backward-compatible wrapper that uses the default
// promptChunkBytes chunk size.
func chunkPromptPrefix(prompt string) (hashes []string, tokens []int) {
	return chunkPromptPrefixN(prompt, promptChunkBytes)
}

// cacheEstimator wraps a prefixTrie with a configurable chunk size and provides
// a series-agnostic interface for estimating prompt cache reuse from raw content bytes.
type cacheEstimator struct {
	trie       *prefixTrie
	chunkBytes int
}

func newCacheEstimator(chunkBytes int) *cacheEstimator {
	if chunkBytes <= 0 {
		chunkBytes = promptChunkBytes
	}
	return &cacheEstimator{trie: newPrefixTrie(), chunkBytes: chunkBytes}
}

// Observe hashes content into byte windows, walks/extends the trie, updates
// aggregate counters, and returns this request's cached/total byte ratio in
// [0,1]. Series-agnostic: it sees only content bytes.
func (e *cacheEstimator) Observe(content string) float64 {
	h, tk := chunkPromptPrefixN(content, e.chunkBytes)
	cached, total := e.trie.RecordAndCount(h, tk)
	e.trie.ObserveRequest(cached, total)
	if total <= 0 {
		return 0
	}
	return float64(cached) / float64(total)
}

// Insert extends the trie with content (e.g. request+response) so future
// requests can match against it; the per-request ratio is not meaningful here.
func (e *cacheEstimator) Insert(content string) {
	h, tk := chunkPromptPrefixN(content, e.chunkBytes)
	e.trie.RecordAndCount(h, tk)
}

func (e *cacheEstimator) Ratio() float64 {
	return e.trie.Ratio()
}

// Nodes returns the total number of trie nodes created (O(1)).
func (e *cacheEstimator) Nodes() int {
	return int(e.trie.nodeCount.Load())
}

// ReplayCacheReport is the output of SimulateReplayCache — simulated and
// ground-truth cache-hit ratios for a set of replay requests.
type ReplayCacheReport struct {
	Requests        int
	SimCachedTokens int
	SimTotalTokens  int
	SimRatio        float64 // prefixTrie prediction
	GTCachedTokens  int     // sum of cache_read_tokens
	GTTotalTokens   int     // sum of input_tokens (already total in replay-v3)
	GTRatio         float64 // ground-truth from server usage
	// BlockTokensAllZero is true when every per-block Tokens field was zero,
	// meaning the simulated ratio is hash-count-based only (no token data).
	BlockTokensAllZero bool
}

// SimulateReplayCache feeds reqs into a fresh prefixTrie in slice order and
// returns simulated + ground-truth ratios. Callers should sort the slice by
// timestamp before invoking so simulation order matches chronological order.
func SimulateReplayCache(reqs []RouterReplayRequest) ReplayCacheReport {
	r := ReplayCacheReport{Requests: len(reqs)}
	if len(reqs) == 0 {
		return r
	}

	trie := newPrefixTrie()
	anyBlockToken := false
	hadPrefix := false

	for _, req := range reqs {
		hashes, tokens := BuildReplayRequestPrefix(req)
		if len(hashes) == 0 {
			// No prefix to simulate — still accumulate ground-truth.
		} else {
			hadPrefix = true
			cached, total := trie.RecordAndCount(hashes, tokens)
			trie.ObserveRequest(cached, total)
			r.SimCachedTokens += cached
			r.SimTotalTokens += total
			for _, t := range tokens {
				if t > 0 {
					anyBlockToken = true
					break
				}
			}
		}

		r.GTCachedTokens += req.CacheReadTokens
		r.GTTotalTokens += req.InputTokens
	}

	if r.SimTotalTokens > 0 {
		r.SimRatio = float64(r.SimCachedTokens) / float64(r.SimTotalTokens)
	}
	if r.GTTotalTokens > 0 {
		r.GTRatio = float64(r.GTCachedTokens) / float64(r.GTTotalTokens)
	}
	r.BlockTokensAllZero = hadPrefix && !anyBlockToken

	return r
}
