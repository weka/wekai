// Package mockvllm is a standalone, GPU-less stand-in for a vLLM OpenAI
// server, built so router cache-aware policies can be developed and
// evaluated against a fleet of these instead of real hardware.
//
// It is deliberately NOT the same thing as router/internal/testutil/mockvllm:
// that package is a Script-driven httptest double for router unit tests — a
// human sets CachedTokens per test case. This package instead OWNS a live
// prefix-cache model and decides cached_tokens itself, from what it has
// actually been sent, the same way a real vLLM worker's KV cache would. The
// two serve different purposes and are kept separate on purpose.
//
// The cache model is github.com/weka/wekai/kvcache — the same trie the
// router's prefix-cache-aware policy (router/internal/policy/cache) uses to
// PREDICT residency. Using it here too, as the thing being predicted, means a
// run against this fleet produces a real, comparable
// predicted-vs-observed signal (router_cache_predicted_fraction vs.
// router_cache_observed_fraction) instead of a hand-rolled second model that
// could drift from the router's own assumptions.
//
// Block/prefix accounting (splitting, chain hashing, prefix-hit counting,
// the parent-linked tree) lives entirely in kvcache and is shared verbatim —
// nothing here reimplements it. The one place this engine's needs diverge
// from the router's is eviction: a live server actually admits requests, so
// its blocks must be PINNED for as long as a request holding them is in
// flight (kvcache.Trie.RecordAndPin/Unpin), with LRU eviction applying only
// to unpinned blocks. The router's own trie never pins (Query/Commit only)
// and is expected to move from its current LRU to TTL/tail eviction — a
// different discipline for a different consumer of the same core, which is
// exactly what pinning was designed to stay orthogonal to (see the "Eviction
// is pluggable in spirit" note in kvcache's package doc).
package mockvllm

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/kvcache"
)

// Config parameterizes one Engine instance.
type Config struct {
	// ModelID is the id this server reports from /v1/models and echoes into
	// every response's "model" field.
	ModelID string

	// BlockSizeTokens is vLLM's block_size analog: the number of tokens per
	// hashed cache block. Real vLLM defaults to 16. Converted internally to
	// bytes via the repo-wide 4-bytes-per-token estimate (kvcache.EstimateTokens,
	// also used by the benchmark's cache estimator) since nothing here runs a
	// real tokenizer.
	BlockSizeTokens int

	// BlockCapacity bounds the store in blocks (kvcache trie nodes — one node
	// per block, so this maps directly to kvcache.Config.MaxNodes). 0 means
	// unbounded, matching a cache that never evicts.
	BlockCapacity int64

	// MaxConcurrency caps requests admitted at once. 0 means unbounded.
	// Exceeding it returns HTTP 429 — the load signal the router's circuit
	// breaker and retry logic already treat as authoritative (see
	// router/internal/circuit and router/internal/proxy).
	MaxConcurrency int

	// Latency model, expressed as TOKEN RATES rather than per-token
	// durations — this is what lets a run be calibrated against a real vLLM
	// fleet: read the fleet's actual prefill/decode throughput (or a
	// reasonable estimate of it) and set these directly, in the same units.
	//
	//	ttft   = BaseLatency + uncachedPromptTokens/ColdInputTPS + cachedPromptTokens/CachedInputTPS
	//	total  = ttft + outputTokens/OutputTPS
	//
	// Cached tokens are NOT free: a cache hit still costs something (a KV
	// read, a network hop for an offloaded tier, etc.) — only recompute is
	// skipped — so CachedInputTPS is normally set far higher than
	// ColdInputTPS rather than left at zero. Any rate left at 0 contributes
	// no time for that term (division only happens when the rate is
	// positive), so the default Config is instant except for BaseLatency.
	BaseLatency    time.Duration
	ColdInputTPS   float64 // tokens/sec, UNCACHED prompt tokens (prefill)
	CachedInputTPS float64 // tokens/sec, CACHED prompt tokens (cache read)
	OutputTPS      float64 // tokens/sec, decode — also paces SSE chunk spacing

	// DefaultMaxTokens is the completion length used when a request omits
	// max_tokens (or sends <= 0).
	DefaultMaxTokens int
}

// DefaultConfig returns sane, non-instant defaults: 16-token blocks, 100k
// blocks (matches kvcache.RouterConfig's node bound), 64-way concurrency, and
// token rates anton specified as giving a good-enough comparison against a
// real fleet (50k tok/s cold prefill, 1M tok/s cached-token read, 500 tok/s
// decode per request) — order-of-magnitude starting points, meant to be
// overridden with rates read off (or estimated from) the real fleet being
// compared against.
func DefaultConfig() Config {
	return Config{
		ModelID:          "mock-vllm",
		BlockSizeTokens:  16,
		BlockCapacity:    100_000,
		MaxConcurrency:   64,
		BaseLatency:      20 * time.Millisecond,
		ColdInputTPS:     50_000,
		CachedInputTPS:   1_000_000,
		OutputTPS:        500,
		DefaultMaxTokens: 128,
	}
}

func (c Config) normalize() Config {
	d := DefaultConfig()
	if c.ModelID == "" {
		c.ModelID = d.ModelID
	}
	if c.BlockSizeTokens <= 0 {
		c.BlockSizeTokens = d.BlockSizeTokens
	}
	if c.DefaultMaxTokens <= 0 {
		c.DefaultMaxTokens = d.DefaultMaxTokens
	}
	// BlockCapacity, MaxConcurrency, and the latency knobs are legitimately
	// zero-able (unbounded / instant), so they are NOT defaulted here.
	return c
}

// blockSizeBytes converts BlockSizeTokens to the byte window kvcache.ChunkContent
// expects, via the same 4-bytes-per-token heuristic used everywhere else in
// this module (kvcache.EstimateTokens).
func (c Config) blockSizeBytes() int {
	n := c.BlockSizeTokens * 4
	if n <= 0 {
		n = 4
	}
	return n
}

// Engine holds one server's live cache model, admission state, and counters.
// Safe for concurrent use.
type Engine struct {
	cfg  Config
	trie *kvcache.Trie

	inflight   atomic.Int64
	admitted   atomic.Int64
	rejected   atomic.Int64
	promptToks atomic.Int64
	cachedToks atomic.Int64
	genToks    atomic.Int64
}

// NewEngine builds an Engine. A zero BlockCapacity means an unbounded cache
// (kvcache.Config{} semantics), matching an infinite-KV-memory worker.
func NewEngine(cfg Config) *Engine {
	cfg = cfg.normalize()
	kvcfg := kvcache.Config{}
	if cfg.BlockCapacity > 0 {
		kvcfg = kvcache.Config{
			MaxNodes:  cfg.BlockCapacity,
			MaxTokens: cfg.BlockCapacity * int64(cfg.BlockSizeTokens),
		}
	}
	return &Engine{cfg: cfg, trie: kvcache.New(kvcfg)}
}

func (e *Engine) Config() Config { return e.cfg }

// Tokenize chunks raw prompt bytes into vLLM-block-sized Units. Two prompts
// sharing a leading byte run produce identical leading Units, which is what
// lets the trie credit a shared prefix as cached — the same mechanism
// kvcache's own doc describes as standing in for vLLM's parent-hash chaining:
// a match requires the full ancestor chain, not just an equal leaf hash,
// because the trie only walks into a child of the node that already matched.
func (e *Engine) Tokenize(prompt string) []kvcache.Unit {
	return kvcache.ChunkContent("prompt", []byte(prompt), e.cfg.blockSizeBytes())
}

// Query reports how much of units this cache currently holds, WITHOUT
// admitting, pinning, inserting, or otherwise mutating anything — a pure
// peek, for callers (tests, observability) that want to check residency
// without pretending to serve a request. Real request handling must go
// through Admit, not this.
func (e *Engine) Query(units []kvcache.Unit) (cached, total int) {
	return e.trie.Query(units)
}

// Admit is the single admission decision for a request: it reserves a
// concurrency slot AND pins the request's blocks in the same step, mirroring
// what a real vLLM scheduler does — a sequence is admitted and its KV blocks
// (both the reused warm prefix and the newly needed tail) are reference-
// counted together, not as two independent steps. ok is false past
// MaxConcurrency, and in that case NOTHING else happened: no cache credit,
// no insertion, no pin — a 429'd request never touched KV state, exactly
// like a request vLLM never scheduled. On success, release MUST be called
// exactly once (success, error, or client disconnect) to free both the
// concurrency slot and the block pin — see kvcache.Trie.Unpin.
//
// Eviction (when bounded) only ever removes UNPINNED blocks: an admitted
// request's blocks cannot be evicted out from under it while it's in flight,
// whether they were already warm or were just inserted for this request.
// That guarantee lives in kvcache (pinChain/unpinChain via RecordAndPin/
// Unpin) precisely so this engine and a future TTL-evicting router share the
// exact same block-splitting/chain-hashing/prefix-hit-counting core; only the
// eviction discipline differs (LRU-among-unpinned here, tail-TTL there).
func (e *Engine) Admit(units []kvcache.Unit) (release func(), cached, total int, ok bool) {
	if e.cfg.MaxConcurrency > 0 {
		n := e.inflight.Add(1)
		if n > int64(e.cfg.MaxConcurrency) {
			e.inflight.Add(-1)
			e.rejected.Add(1)
			return nil, 0, 0, false
		}
	} else {
		e.inflight.Add(1)
	}
	e.admitted.Add(1)

	cached, total, pin := e.trie.RecordAndPin(units)
	e.trie.Observe(cached, total)
	e.promptToks.Add(int64(total))
	e.cachedToks.Add(int64(cached))

	return func() {
		e.trie.Unpin(pin)
		e.inflight.Add(-1)
	}, cached, total, true
}

// Latency computes TTFT (BaseLatency plus the uncached portion charged at
// ColdInputTPS and the cached portion charged at CachedInputTPS — a cache hit
// still costs a KV read, it just isn't full recompute) and total request
// duration (TTFT plus the output charged at OutputTPS).
func (e *Engine) Latency(cachedTokens, totalTokens, outputTokens int) (ttft, total time.Duration) {
	uncached := totalTokens - cachedTokens
	if uncached < 0 {
		uncached = 0
	}
	ttft = e.cfg.BaseLatency + tokensAtRate(uncached, e.cfg.ColdInputTPS) + tokensAtRate(cachedTokens, e.cfg.CachedInputTPS)
	total = ttft + tokensAtRate(outputTokens, e.cfg.OutputTPS)
	return ttft, total
}

// OutputTokenInterval is the per-token duration OutputTPS implies, for a
// caller (SSE streaming) that needs to space out individual chunks rather
// than compute one lump duration.
func (e *Engine) OutputTokenInterval() time.Duration {
	return tokensAtRate(1, e.cfg.OutputTPS)
}

// tokensAtRate is how long n tokens take at tps tokens/sec. A non-positive
// rate or count contributes zero duration, so an unset rate is "instant" for
// that term rather than a division by zero.
func tokensAtRate(n int, tps float64) time.Duration {
	if n <= 0 || tps <= 0 {
		return 0
	}
	return time.Duration(float64(n) / tps * float64(time.Second))
}

// RecordOutput accounts generated tokens into the aggregate counters. Called
// once the (synthetic) completion length for a request is known.
func (e *Engine) RecordOutput(n int) { e.genToks.Add(int64(n)) }

// MaxTokensOrDefault resolves a request's requested output length, falling
// back to DefaultMaxTokens for an absent or non-positive value.
func (e *Engine) MaxTokensOrDefault(requested int) int {
	if requested > 0 {
		return requested
	}
	return e.cfg.DefaultMaxTokens
}

// Stats snapshots the live cache model and counters, for /metrics and /health.
type Stats struct {
	Nodes, Tokens, Anomalies     int64
	PinnedNodes                  int64 // blocks currently held by an in-flight request's Pin
	Inflight, Admitted, Rejected int64
	PromptTokens, CachedTokens   int64
	GeneratedTokens              int64
	Capacity                     int64 // BlockCapacity, 0 = unbounded
}

func (e *Engine) Stats() Stats {
	nodes, tokens, anomalies := e.trie.Stats()
	return Stats{
		Nodes:           nodes,
		Tokens:          tokens,
		Anomalies:       anomalies,
		PinnedNodes:     e.trie.PinnedNodes(),
		Inflight:        e.inflight.Load(),
		Admitted:        e.admitted.Load(),
		Rejected:        e.rejected.Load(),
		PromptTokens:    e.promptToks.Load(),
		CachedTokens:    e.cachedToks.Load(),
		GeneratedTokens: e.genToks.Load(),
		Capacity:        e.cfg.BlockCapacity,
	}
}

// FillFraction is the cache's node count over its configured capacity, the
// mock's analog of vLLM's vllm:gpu_cache_usage_perc. 0 for an unbounded cache
// (there is no capacity to be a fraction of).
func (e *Engine) FillFraction() float64 {
	if e.cfg.BlockCapacity <= 0 {
		return 0
	}
	nodes, _, _ := e.trie.Stats()
	return float64(nodes) / float64(e.cfg.BlockCapacity)
}

// sleepCtx sleeps for d or returns early (false) if ctx is canceled first, so
// a client disconnect during simulated latency frees the concurrency slot
// promptly instead of holding it for the full synthetic duration.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		select {
		case <-ctx.Done():
			return false
		default:
			return true
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}
