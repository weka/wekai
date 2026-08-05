package benchmark

// Prefix-cache simulation and content-level warm estimation.
//
// The engine itself now lives in github.com/weka/wekai/kvcache, shared with the
// router. What remains here are thin adapters that keep this package's existing
// signatures — []string hashes and []int tokens — so every call site and test in
// benchmark is unchanged.
//
// Extracting it was the point: the router needs the same prefix-matching and
// token-crediting logic, and a second copy would have drifted. The router's needs
// differ in two ways the shared type absorbs — it queries without mutating, and
// it is bounded because a real vLLM node evicts. A zero kvcache.Config means
// unbounded, which is exactly the infinite-cache model this simulator wants, so
// behaviour here is unchanged by the move.
//
// Two uses remain:
//
//  1. Offline replay analysis (SimulateReplayCache), mirroring
//     `wekai router analyze` so simulated and ground-truth ratios stay
//     comparable.
//  2. Live content-level estimation (cacheEstimator) across all benchmark modes.

import (
	"github.com/weka/wekai/kvcache"
)

// promptChunkBytes is the granularity at which prompts are segmented for the
// prefix estimate (~256 estimated tokens per chunk).
const promptChunkBytes = kvcache.DefaultChunkBytes

// prefixTrie adapts kvcache.Trie to this package's string-hash interface.
type prefixTrie struct{ t *kvcache.Trie }

// newPrefixTrie builds an UNBOUNDED trie: the simulator measures how much reuse
// a workload contains, which is a property of an infinite cache. A router uses
// kvcache.RouterConfig() instead.
func newPrefixTrie() *prefixTrie {
	return &prefixTrie{t: kvcache.New(kvcache.Config{})}
}

// RecordAndCount walks the matching prefix, credits matched messages as cached,
// inserts the novel tail, and returns (cachedThisCall, totalThisCall).
//
// Callers pass the full message sequence — system plus every prior turn plus the
// current one. After the server responds, call again with the extended sequence
// so later requests can match against it; that second call's "cached" value is
// not meaningful for the current request but its insertion side-effect is.
func (p *prefixTrie) RecordAndCount(hashes []string, tokens []int) (int, int) {
	return p.t.RecordAndCount(kvcache.UnitsFromLabels(hashes, tokens))
}

// ObserveRequest accumulates one request into the aggregate ratio without
// inserting anything.
func (p *prefixTrie) ObserveRequest(cached, total int) { p.t.Observe(cached, total) }

// Ratio is cached/total over all observed requests, 0 if none.
func (p *prefixTrie) Ratio() float64 { return p.t.Ratio() }

// hashMessage produces a stable short hash for a (role, content) pair.
func hashMessage(role, content string) string {
	return kvcache.HexHash(kvcache.HashContent(role, []byte(content)))
}

// estimateTokens is the 4-bytes-per-token heuristic, clamped to 1 for non-empty
// content. Deliberately crude: this measures relative prefix efficiency, not
// billing.
func estimateTokens(content string) int { return int(kvcache.EstimateTokens(len(content))) }

// chunkPromptPrefixN splits a prompt into fixed byte windows, returning per-chunk
// hashes and token estimates. Two prompts sharing a leading byte prefix yield
// identical leading hashes, so the trie credits the shared prefix as cached and
// the divergent tail as novel.
func chunkPromptPrefixN(prompt string, chunkBytes int) (hashes []string, tokens []int) {
	for _, u := range kvcache.ChunkContent("chunk", []byte(prompt), chunkBytes) {
		hashes = append(hashes, kvcache.HexHash(u.Hash))
		tokens = append(tokens, int(u.Tokens))
	}
	return
}

// chunkPromptPrefix uses the default chunk size.
func chunkPromptPrefix(prompt string) (hashes []string, tokens []int) {
	return chunkPromptPrefixN(prompt, promptChunkBytes)
}

// cacheEstimator estimates prompt reuse from raw content bytes.
type cacheEstimator struct{ e *kvcache.Estimator }

func newCacheEstimator(chunkBytes int) *cacheEstimator {
	return &cacheEstimator{e: kvcache.NewEstimator(chunkBytes, kvcache.Config{})}
}

// Observe returns this request's cached fraction in [0,1] and extends the trie.
func (c *cacheEstimator) Observe(content string) float64 { return c.e.Observe(content) }

// Insert extends the trie so later requests can match; the ratio is not
// meaningful for this call.
func (c *cacheEstimator) Insert(content string) { c.e.Insert(content) }

func (c *cacheEstimator) Ratio() float64 { return c.e.Ratio() }

// Nodes is the total trie nodes created, O(1).
func (c *cacheEstimator) Nodes() int { return c.e.Nodes() }

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
