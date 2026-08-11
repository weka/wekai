package benchmark

// Offline estimate for dynamic prefill/decode segregation (see
// router/internal/policy/affinity and `wekai router serve --prefill-split`):
// of a replayed workload, what share of requests and prompt tokens would be
// diverted to a dedicated prefill instance?
//
// The router's rule (mirrored here, not reimplemented): a request's prompt
// is chunked into ROUTER BLOCKS via kvcache.ChunkContent at
// kvcache.DefaultChunkBytes (~256 estimated tokens/block — NOT vLLM's
// 16-token block). "Missing blocks" is kvcache.Coverage.MissingBlocks() — the
// block count minus the longest leading run the router predicts a backend
// already holds. Strictly above --prefill-min-missing-blocks, the request is
// first sent whole to a prefill-only instance (max_tokens=1, reply
// discarded), then routed normally for decode.
//
// # Which decomposition this scores, and why it's the comparable one
//
// There are two candidate decompositions of a request's prefix, and they are
// NOT interchangeable:
//
//   - capture-level blocks: one hash per system-block/tools-spec/message, as
//     BuildReplayRequestPrefix and SimulateReplayCache already use. Coarser
//     or finer than a router block depending on message size, and NOT
//     comparable with the live router's numbers — router_cache_predicted_
//     fraction vs. this would repeat exactly the Grafana mistake
//     kvcache.Coverage's doc describes.
//   - router blocks: a re-chunk of the reconstructed body via
//     kvcache.ChunkContent(tag, content, kvcache.DefaultChunkBytes), walked
//     in the router's own segment order (system, then tools, then messages —
//     see router/internal/dialect/openai.ExtractUnits).
//
// This package scores the SECOND: prefillBlockUnits rebuilds that same
// per-segment kvcache.ChunkContent sequence, over content synthesized
// deterministically from each block's stored hash (synthText — the replay
// file carries no raw bytes). That is what makes this estimate comparable
// with what a live --prefill-split router would decide, rather than a
// differently-grained number that happens to share a name.
//
// # Residency model: single fleet-wide trie, and how it differs from the router
//
// A replay file has no fleet to ask "which backend holds this", so this
// models a SINGLE FLEET-WIDE CACHE: a block is resident once anything has
// ever seen it and it has not been evicted from one unbounded (by default)
// kvcache.Trie — matchedBlocks fed to kvcache.Cover comes from walking that
// one trie, not from N independent per-backend tries. That is optimistic
// relative to a real fleet, which splits prefixes across nodes — see
// SimulatePrefillBlocks's doc for the concrete consequence.
//
// It is optimistic in the OTHER direction too, and differently than a naive
// per-backend model would be: the router's real residency is a marked
// forest, where a backend is only ever marked as holding a prefix after a
// guarded split placed it there (affinity.Policy.Commit) — duplication is
// deliberately rationed. A hypothetical estimator that ran N independent
// per-backend kvcache.Trie instances, each learning from every request it
// happened to serve, would OVER-count duplication relative to that guarded
// discipline (every backend a request ever touches "remembers" it, with no
// guard limiting how many backends end up holding the same prefix). This
// package does not do that — it deliberately stays at ONE fleet-wide trie —
// but a future --prefill-nodes-style addition that models several tries
// needs to reckon with this before its numbers can be trusted.

import (
	"container/list"
	"strconv"

	"github.com/weka/wekai/kvcache"
)

// PrefillBlockStat is one request's outcome from the router-block-level
// prefix simulation. It embeds kvcache.Coverage — the one written definition
// of "how cold is this request" — rather than copying MissingBlocks/
// MissingTokens into locally-named fields, so this estimator and the
// router's own PlanPrefill read the same vocabulary and cannot drift apart.
type PrefillBlockStat struct {
	kvcache.Coverage
	InputTokens int // ground truth (req.InputTokens) — for token-volume shares; not part of Coverage, which is block-derived
}

// PrefillUnitCache memoizes the router-block Unit sequence synthesized for
// one (tag, hash, bytes) prefix block. A replay-v3 capture resends the full
// conversation on every turn, so the same early block reappears in every
// later request; without memoizing, a T-turn session would re-synthesize and
// re-hash it T times.
//
// Bounded by LRU eviction: an unbounded map here defeats the "memory must
// not scale with file size" requirement just as surely as loading the file
// whole would — a large enough replay simply has enough DISTINCT content for
// an unbounded memo table to grow without limit (observed: >30GB RSS partway
// through a 13GB real capture before this bound was added). The bound bites
// hardest on cross-session reuse (a system prompt shared by many unrelated
// sessions may need re-synthesizing); the case this cache exists for — one
// session's own history, resent every turn in close succession — stays warm,
// since recently touched entries are exactly what LRU protects.
//
// Use NewPrefillUnitCache; the zero value is not ready to use. Share one
// across every request (even across models — content synthesis doesn't
// depend on model) for the full benefit.
type PrefillUnitCache struct {
	cap   int
	ll    *list.List
	items map[string]*list.Element
}

type prefillCacheEntry struct {
	key   string
	units []kvcache.Unit
}

// DefaultPrefillUnitCacheCap bounds a PrefillUnitCache at a fixed entry
// count, independent of input size — comparable in spirit to
// kvcache.RouterConfig's fixed per-backend bound (2,000,000 tokens, ~7,800
// nodes at router-block granularity), just expressed as distinct blocks
// rather than tokens since a block's synthesized size varies.
const DefaultPrefillUnitCacheCap = 50_000

// NewPrefillUnitCache builds a cache bounded at DefaultPrefillUnitCacheCap
// entries.
func NewPrefillUnitCache() *PrefillUnitCache {
	return NewPrefillUnitCacheSize(DefaultPrefillUnitCacheCap)
}

// NewPrefillUnitCacheSize builds a cache bounded at cap entries (cap<=0
// means DefaultPrefillUnitCacheCap).
func NewPrefillUnitCacheSize(cap int) *PrefillUnitCache {
	if cap <= 0 {
		cap = DefaultPrefillUnitCacheCap
	}
	return &PrefillUnitCache{cap: cap, ll: list.New(), items: make(map[string]*list.Element)}
}

// get returns the cached units for key, promoting it to most-recently-used.
// A nil cache always misses, so callers may pass nil for "no memoization".
func (c *PrefillUnitCache) get(key string) ([]kvcache.Unit, bool) {
	if c == nil {
		return nil, false
	}
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.ll.MoveToFront(el)
	return el.Value.(*prefillCacheEntry).units, true
}

// put stores units for key, evicting the least-recently-used entry if the
// cache is now over its bound. A nil cache is a no-op.
func (c *PrefillUnitCache) put(key string, units []kvcache.Unit) {
	if c == nil {
		return
	}
	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		el.Value.(*prefillCacheEntry).units = units
		return
	}
	el := c.ll.PushFront(&prefillCacheEntry{key: key, units: units})
	c.items[key] = el
	if c.ll.Len() > c.cap {
		oldest := c.ll.Back()
		if oldest != nil {
			c.ll.Remove(oldest)
			delete(c.items, oldest.Value.(*prefillCacheEntry).key)
		}
	}
}

// SimulatePrefillBlocks runs reqs (already chronologically sorted — same
// requirement as SimulateReplayCache) through a single trie at router-block
// granularity and returns one PrefillBlockStat per request, in order.
//
// The replay-v3 file never carries raw prompt bytes (redacted at capture
// time), so each stored block is re-materialized deterministically from its
// own hash via synthText — the identical byte-faithful reconstruction the
// live replayer (buildAnthropicMessagesBody) already uses. Two requests
// referencing the same original block therefore synthesize byte-identical
// content and chunk to identical router-block hashes here; this is what lets
// an unbounded kvcache.Trie recognize the shared prefix at all, without ever
// touching the real (redacted-away) prompt text.
//
// cfg bounds the trie: kvcache.Config{} (the zero value) is unbounded —
// infinite retention, the default lower-bound model — while a non-zero
// MaxTokens models eviction under a finite fleet-wide cache budget instead.
//
// cache memoizes content synthesis across calls (see PrefillUnitCache); pass
// nil if the caller has nothing to share (synthesis then runs unmemoized),
// or NewPrefillUnitCache() for a fresh bounded one.
//
// Each request is scored through kvcache.Cover, same as the router's own
// PlanPrefill — see the package doc for why that, and not a hand-rolled
// subtraction, is what must run here.
func SimulatePrefillBlocks(reqs []RouterReplayRequest, docs string, cfg kvcache.Config, cache *PrefillUnitCache) []PrefillBlockStat {
	trie := kvcache.New(cfg)
	out := make([]PrefillBlockStat, len(reqs))
	for i, req := range reqs {
		units := prefillBlockUnits(req, docs, cache)
		if len(units) == 0 {
			out[i] = PrefillBlockStat{InputTokens: req.InputTokens}
			continue
		}
		cachedTokens, _ := trie.RecordAndCount(units)
		// RecordAndCount reports the matched prefix as a TOKEN sum, not a
		// block count, and kvcache.Cover wants a block count. Units carry
		// Tokens>=1 always (kvcache.ChunkContent's invariant), so the matched
		// token sum lands EXACTLY on a unit boundary — walking our own unit
		// list to that boundary recovers the matched block count without
		// re-walking the trie or duplicating its matching logic.
		matchedBlocks, acc := 0, 0
		for _, u := range units {
			if acc >= cachedTokens {
				break
			}
			acc += int(u.Tokens)
			matchedBlocks++
		}
		out[i] = PrefillBlockStat{
			Coverage:    kvcache.Cover(units, matchedBlocks),
			InputTokens: req.InputTokens,
		}
	}
	return out
}

// prefillBlockUnits reconstructs one request's router-block sequence. Prefix
// selection and order matches BuildReplayRequestPrefix exactly — system
// blocks after the Anthropic billing-header skip (via effectiveSystemBlocks),
// then tools, then messages, this package's one definition of "prefix" — but
// each segment is further rechunked at router-block granularity instead of
// staying one hash per message, mirroring how the router's dialect layer
// builds rr.Units (router/internal/dialect/openai.ExtractUnits): one
// kvcache.ChunkContent call per segment, concatenated in cache order.
func prefillBlockUnits(req RouterReplayRequest, docs string, cache *PrefillUnitCache) []kvcache.Unit {
	var units []kvcache.Unit
	add := func(tag, hash string, n int) {
		key := tag + ":" + hash + ":" + strconv.Itoa(n)
		if u, ok := cache.get(key); ok {
			units = append(units, u...)
			return
		}
		var u []kvcache.Unit
		if n <= 0 {
			// No bytes to rechunk — fall back to the block's own hash as a
			// single unit, the same identity the message-level model
			// (BuildReplayRequestPrefix) already uses for it.
			u = []kvcache.Unit{{Hash: kvcache.HashLabel(hash), Tokens: 1}}
		} else {
			content := []byte(synthText(hash, n, docs))
			u = kvcache.ChunkContent(tag, content, kvcache.DefaultChunkBytes)
		}
		cache.put(key, u)
		units = append(units, u...)
	}
	for _, sb := range effectiveSystemBlocks(req.SystemBlocks) {
		add("system", sb.Hash, sb.Bytes)
	}
	if req.Tools != nil && req.Tools.Hash != "" {
		add("tools", req.Tools.Hash, req.Tools.Bytes)
	}
	for _, m := range req.Messages {
		add("message", m.Hash, m.Bytes)
	}
	return units
}

// PrefillSplitReport summarizes PrefillBlockStat outcomes at one
// missing-blocks threshold: how many requests, and how many tokens, a
// prefill-split router would send to a dedicated prefill instance.
type PrefillSplitReport struct {
	MinMissingBlocks int

	Requests        int
	PrefillRequests int

	// TotalInputTokens/PrefillInputTokens are ground truth (ReplayRequest.
	// InputTokens): a prefill pass sends the WHOLE prompt, so PrefillInputTokens
	// is the sum of prompt tokens over requests routed to prefill.
	TotalInputTokens   int64
	PrefillInputTokens int64

	// TotalMissingTokens/PrefillMissingTokens are the block-level ESTIMATE
	// (not ground truth) of tokens in blocks NOT predicted resident — the
	// portion of a prefill pass that is genuinely new work, as opposed to
	// tokens that would have been a cache hit on the decode fleet anyway.
	TotalMissingTokens   int64
	PrefillMissingTokens int64
}

// RequestShare is PrefillRequests/Requests, 0 if Requests is 0.
func (r PrefillSplitReport) RequestShare() float64 {
	if r.Requests == 0 {
		return 0
	}
	return float64(r.PrefillRequests) / float64(r.Requests)
}

// InputTokenShare is PrefillInputTokens/TotalInputTokens, 0 if TotalInputTokens is 0.
func (r PrefillSplitReport) InputTokenShare() float64 {
	if r.TotalInputTokens == 0 {
		return 0
	}
	return float64(r.PrefillInputTokens) / float64(r.TotalInputTokens)
}

// MissingTokenShare is PrefillMissingTokens/TotalInputTokens — deliberately
// over TotalInputTokens, not TotalMissingTokens, so it stays directly
// comparable to InputTokenShare: how much of ALL prompt volume is genuinely
// new work moved off the decode fleet, versus how much prompt volume merely
// passes through a prefill instance. 0 if TotalInputTokens is 0.
func (r PrefillSplitReport) MissingTokenShare() float64 {
	if r.TotalInputTokens == 0 {
		return 0
	}
	return float64(r.PrefillMissingTokens) / float64(r.TotalInputTokens)
}

// AggregatePrefillSplit classifies each stat against minMissingBlocks
// (strictly above routes to prefill, matching --prefill-min-missing-blocks
// on `wekai router serve`) and totals requests/tokens. Pure and cheap, so a
// caller can sweep many thresholds over one SimulatePrefillBlocks pass
// without re-simulating: the trie's residency predictions don't depend on
// the threshold, only this post-hoc classification does.
func AggregatePrefillSplit(stats []PrefillBlockStat, minMissingBlocks int) PrefillSplitReport {
	r := PrefillSplitReport{MinMissingBlocks: minMissingBlocks, Requests: len(stats)}
	for _, s := range stats {
		missingTokens := s.MissingTokens()
		r.TotalInputTokens += int64(s.InputTokens)
		r.TotalMissingTokens += int64(missingTokens)
		if s.MissingBlocks() > minMissingBlocks {
			r.PrefillRequests++
			r.PrefillInputTokens += int64(s.InputTokens)
			r.PrefillMissingTokens += int64(missingTokens)
		}
	}
	return r
}
