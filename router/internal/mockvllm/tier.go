package mockvllm

import (
	"sync"

	"github.com/weka/wekai/kvcache"
)

// Tier is a KV cache SHARED BY THE WHOLE FLEET, standing in for LMCache backed
// by WEKA: blocks one instance computes are readable by every other instance,
// at a cost between a local HBM hit and a full recompute.
//
// Why the mock needs one at all. Every instance here owns an independent trie,
// which is a faithful model of plain vLLM — and a fleet of plain vLLMs is a
// fleet where prefilling a prompt on node A does nothing whatsoever for node B.
// Under that model the router's prefill/decode segregation cannot be anything
// but pure double work, and a simulator run would "prove" it worthless by
// assuming away the mechanism it depends on. The real deployment this repo
// exists to measure has exactly such a shared tier; vLLM reports what it reads
// from one as source="external_kv_transfer" (see benchmark/vllm_metrics.go),
// distinct from its own local_cache_hit.
//
// It is OFF by default (a nil Tier, --external-kv-tps 0) so every existing
// calibration run against this mock is unchanged. Turning it on is what lets a
// run A/B the router's segregation against the same fleet without it.
//
// Deliberately NOT modelled here: capacity, eviction, transfer concurrency, and
// failure. A real shared tier is bounded, has a health monitor, and falls back
// to recompute (LMCache's FallbackPolicy.RECOMPUTE). Adding those would make
// this a second-rate LMCache simulator; what is being measured is whether the
// ROUTER puts the right requests in the right place, and for that the tier only
// has to answer "is this prefix available elsewhere, and what does reading it
// cost".
type Tier struct {
	// One trie for the fleet. kvcache.Trie is internally locked, but the
	// query-then-publish sequence in a request is two calls, and nothing here
	// requires them to be atomic with respect to each other: a block published
	// between them simply reads as cold to a request that just missed it, which
	// is exactly what a real race against a shared store would produce.
	mu   sync.Mutex
	trie *kvcache.Trie
}

// NewTier builds a shared tier. Unbounded: see the note above on what is
// deliberately not modelled.
func NewTier() *Tier { return &Tier{trie: kvcache.New(kvcache.Config{})} }

// Query reports how many LEADING blocks of units the tier holds. Prefix
// semantics, matching the local trie and vLLM itself: a gap ends the run,
// because a KV block is only usable if every block before it is too.
func (t *Tier) Query(units []kvcache.Unit) int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	cached, _ := t.trie.Query(units)
	return cached
}

// Publish records that an instance has now computed these blocks and written
// them out, so the rest of the fleet can read them.
//
// Called once a request has actually been served, never at admission: a request
// that was refused or that died mid-prefill computed nothing, and publishing on
// intent rather than completion is how a shared cache ends up advertising
// blocks nobody ever wrote.
func (t *Tier) Publish(units []kvcache.Unit) {
	if t == nil || len(units) == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.trie.Commit(units)
}

// Stats reports the tier's size, for the fleet-level view a test or an operator
// wants when checking that anything is being shared at all.
func (t *Tier) Stats() (nodes, tokens int64) {
	if t == nil {
		return 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	n, tk, _ := t.trie.Stats()
	return n, tk
}

// Residency is what an admitted request found already computed, split by where
// it was found, because the three cost wildly different amounts of time.
//
//	Local    — this instance's own HBM prefix cache. A KV read.
//	External — the shared tier: another instance computed it. A network/storage
//	           read, so slower than local and enormously faster than recompute.
//	Cold     — nobody has it. Full prefill.
//
// Prefix semantics throughout, so these are nested runs rather than sets:
// Local is a prefix of External is a prefix of Total.
type Residency struct {
	Local    int
	External int
	Total    int
}

// Cached is what the OpenAI wire calls cached_tokens. Local and external are
// summed deliberately: real vLLM reports them combined in
// prompt_tokens_details.cached_tokens and only separates them in its
// prompt_tokens_by_source metric, so a mock that split them here would be
// handing consumers a number no real server produces.
func (r Residency) Cached() int { return r.Local + r.External }

// Cold is what has to be computed from scratch.
func (r Residency) Cold() int {
	n := r.Total - r.Cached()
	if n < 0 {
		return 0
	}
	return n
}
