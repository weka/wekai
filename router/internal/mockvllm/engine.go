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

	// Latency model: ttft = BaseLatency + uncachedPromptTokens*PrefillPerToken;
	// total = ttft + outputTokens*DecodePerToken. Any knob left at 0 contributes
	// nothing, so the default Config is instant.
	BaseLatency     time.Duration
	PrefillPerToken time.Duration
	DecodePerToken  time.Duration

	// DefaultMaxTokens is the completion length used when a request omits
	// max_tokens (or sends <= 0).
	DefaultMaxTokens int
}

// DefaultConfig returns sane, non-instant defaults: 16-token blocks, 100k
// blocks (matches kvcache.RouterConfig's node bound), 64-way concurrency, and
// a latency model in the same ballpark as a mid-size dense model on modern
// GPUs (order-of-magnitude only — tune via flags for a specific scenario).
func DefaultConfig() Config {
	return Config{
		ModelID:          "mock-vllm",
		BlockSizeTokens:  16,
		BlockCapacity:    100_000,
		MaxConcurrency:   64,
		BaseLatency:      20 * time.Millisecond,
		PrefillPerToken:  200 * time.Microsecond,
		DecodePerToken:   20 * time.Millisecond,
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

// Admit reserves one concurrency slot. ok is false when the server is at
// MaxConcurrency, and the caller must respond 429 without calling the
// returned (nil) release. When ok is true, release MUST be called exactly
// once to free the slot.
func (e *Engine) Admit() (release func(), ok bool) {
	if e.cfg.MaxConcurrency <= 0 {
		e.inflight.Add(1)
		e.admitted.Add(1)
		return func() { e.inflight.Add(-1) }, true
	}
	n := e.inflight.Add(1)
	if n > int64(e.cfg.MaxConcurrency) {
		e.inflight.Add(-1)
		e.rejected.Add(1)
		return nil, false
	}
	e.admitted.Add(1)
	return func() { e.inflight.Add(-1) }, true
}

// Serve is the ground-truth cache lookup+insert for one request: it walks the
// matched prefix, credits it, inserts the novel tail (evicting LRU blocks if
// bounded), and returns (cached, total) estimated tokens — exactly the
// numbers a real vLLM worker would report via
// usage.prompt_tokens_details.cached_tokens and usage.prompt_tokens.
func (e *Engine) Serve(units []kvcache.Unit) (cached, total int) {
	cached, total = e.trie.RecordAndCount(units)
	e.trie.Observe(cached, total)
	e.promptToks.Add(int64(total))
	e.cachedToks.Add(int64(cached))
	return cached, total
}

// Latency computes TTFT (prefill of the UNCACHED portion only, matching real
// vLLM where cached blocks skip recompute) and total request duration
// (TTFT + decode of the output tokens).
func (e *Engine) Latency(cachedTokens, totalTokens, outputTokens int) (ttft, total time.Duration) {
	uncached := totalTokens - cachedTokens
	if uncached < 0 {
		uncached = 0
	}
	ttft = e.cfg.BaseLatency + time.Duration(uncached)*e.cfg.PrefillPerToken
	total = ttft + time.Duration(outputTokens)*e.cfg.DecodePerToken
	return ttft, total
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
