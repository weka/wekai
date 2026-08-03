// Package cache implements prefix-cache-aware routing.
//
// The prefix engine itself is github.com/weka/wekai/kvcache, shared with the
// benchmark tooling so there is one implementation rather than two that drift.
// The router uses its Query/Commit pair rather than RecordAndCount: querying every
// candidate with a call that also inserts would teach each backend every request
// until every model converged to the union, at which point the prediction carries
// no information at all.
//
// It prefers the backend most likely to already hold the request's prefix, and
// spills to a load-based policy only once the fleet is measurably imbalanced.
// That shape is deliberate and comes from the product requirement: route to the
// node holding the KV slice "even if that server is already heavily loaded"
// (FR-RTR-01). A continuous queueing penalty would abandon an 8k-token cache hit
// for a single queued request, which for a cache benchmark hides the effect being
// measured.
//
// The threshold is v1's, so operators keep the model they know:
//
//	spill  ⟺  (max_load − min_load) > balance_abs_threshold
//	      AND   max_load > min_load × balance_rel_threshold
//
// v1's version was broken in four ways, and all four are fixed here — see the
// notes on Select. The most important is not in this file at all: the load it
// reads now comes from the lease primitive, so the guard is evaluating a real
// signal rather than a counter that was incremented on one path, decremented on
// three, and zeroed every ten health cycles.
package cache

import (
	"context"
	"sync"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// tieEpsilon is the band within which two matched fractions count as equal.
// Chosen well above float noise and well below a meaningful difference in
// affinity: at 1024-byte units it is roughly one unit out of a hundred.
const tieEpsilon = 0.01

type Config struct {
	// CacheThreshold is the matched fraction below which affinity is not worth
	// overriding load balance.
	CacheThreshold float64
	// BalanceAbsThreshold and BalanceRelThreshold define the spill guard. Defaults
	// are 32 and 1.5. v1 shipped 32/1.1 in the policy and 64/1.5 on the CLI —
	// two different defaults for the same knob.
	BalanceAbsThreshold int64
	BalanceRelThreshold float64
	Trie                kvcache.Config
}

func DefaultConfig() Config {
	return Config{
		CacheThreshold:      0.5,
		BalanceAbsThreshold: 32,
		BalanceRelThreshold: 1.5,
		Trie:                kvcache.RouterConfig(),
	}
}

// Policy routes by predicted prefix residency.
type Policy struct {
	cfg      Config
	fallback policy.Policy
	store    *trieStore
}

func New(cfg Config, fallback policy.Policy) *Policy {
	if cfg.CacheThreshold <= 0 {
		cfg.CacheThreshold = DefaultConfig().CacheThreshold
	}
	if cfg.BalanceAbsThreshold <= 0 {
		cfg.BalanceAbsThreshold = DefaultConfig().BalanceAbsThreshold
	}
	if cfg.BalanceRelThreshold <= 0 {
		cfg.BalanceRelThreshold = DefaultConfig().BalanceRelThreshold
	}
	if fallback == nil {
		fallback = policy.LeastOutstanding{}
	}
	return &Policy{cfg: cfg, fallback: fallback, store: newTrieStore(cfg.Trie)}
}

func (p *Policy) Name() string { return "prefix-cache-aware" }

// AddBackend creates a model for a backend. Wired to the registry's add hook.
func (p *Policy) AddBackend(b *registry.Backend) { p.store.add(b.URL) }

// DropBackend discards a backend's model. Its prefixes are NOT reassigned to any
// other backend: nothing else has served them (CACHE-10).
func (p *Policy) DropBackend(b *registry.Backend) { p.store.drop(b.URL) }

// Flush clears every model. Backs POST /flush_cache; touches nothing else.
func (p *Policy) Flush() { p.store.flush() }

func (p *Policy) trieFor(url string) *kvcache.Trie { return p.store.get(url) }

// Select picks the backend most likely to hold the request's prefix.
func (p *Policy) Select(ctx context.Context, cands []*registry.Backend, rr *policy.RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, policy.ErrNoCandidates
	}
	// No routable prefix (an embeddings call, an unparseable body): decline rather
	// than guess. The policy must never fail a request (CU-11).
	if len(rr.Units) == 0 {
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_units").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}

	// The spill guard, computed over CANDIDATES ONLY.
	//
	// v1 folded over the full worker list including unhealthy ones, so a single
	// dead worker holding stale load kept max_load high forever, latched the guard
	// permanently on, and silently disabled cache routing for the life of the
	// process. Candidates are already filtered to healthy, non-draining,
	// closed-circuit backends, so that cannot happen here.
	minLoad, maxLoad := cands[0].Inflight(), cands[0].Inflight()
	for _, c := range cands[1:] {
		l := c.Inflight()
		if l < minLoad {
			minLoad = l
		}
		if l > maxLoad {
			maxLoad = l
		}
	}
	if maxLoad-minLoad > p.cfg.BalanceAbsThreshold &&
		float64(maxLoad) > float64(minLoad)*p.cfg.BalanceRelThreshold {
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "imbalanced").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}

	// Query every candidate. Read-only: considering a backend must not teach it
	// anything, or every model converges to every request.
	var best *registry.Backend
	bestFrac := 0.0
	for _, c := range cands {
		cached, total := p.trieFor(c.URL).Query(rr.Units)
		if total == 0 {
			continue
		}
		frac := float64(cached) / float64(total)
		switch {
		case frac > bestFrac+tieEpsilon:
			best, bestFrac = c, frac
		case best != nil && frac > bestFrac-tieEpsilon:
			// Within the tie band: prefer the less loaded of the two.
			//
			// A strict `>` would keep the FIRST of several equally-warm backends,
			// and snapshot order is sorted by URL — so a fleet all warm on a shared
			// system prompt sends every request to the lexicographically smallest
			// backend until it exceeds the spill threshold. That is the v1 index-0
			// thundering herd, reintroduced in the affinity path after being fixed
			// in the fallback path. Warmth is genuinely equal here, so load is the
			// right discriminator.
			if c.NormalizedLoad() < best.NormalizedLoad() {
				best = c
				if frac > bestFrac {
					bestFrac = frac
				}
			}
		}
	}

	if best == nil || bestFrac < p.cfg.CacheThreshold {
		// Not enough affinity to be worth overriding load balance. Falling back to
		// least-outstanding rather than "first candidate" matters: v1's min_by_key
		// returned the first minimum, so on a cold fleet every request piled onto
		// candidate 0 until it crossed the imbalance threshold.
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "below_threshold").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}
	metrics.CachePredictedFraction.Observe(bestFrac)
	return best, nil
}

// Commit records that a backend served the request.
//
// Must be called only after the upstream has accepted it — never before dispatch.
// An attempt that fails at connect would otherwise leave that backend looking warm
// for a prefix it never received, permanently and self-reinforcingly (R3).
func (p *Policy) Commit(b *registry.Backend, rr *policy.RoutingRequest) {
	if b == nil || len(rr.Units) == 0 {
		return
	}
	p.trieFor(b.URL).Commit(rr.Units)
}

// PublishGauges exports per-backend model size. Called on the health interval so
// cache memory is observable — the caps are only trustworthy if someone can see
// them being approached.
func (p *Policy) PublishGauges() {
	for url, st := range p.Stats() {
		metrics.CacheEntries.WithLabelValues(url).Set(float64(st[0]))
		metrics.CacheTokens.WithLabelValues(url).Set(float64(st[1]))
	}
}

// Stats reports per-backend model size, for metrics.
func (p *Policy) Stats() map[string][2]int64 { return p.store.stats() }

// trieStore is the per-backend-trie bookkeeping shared by every cache-aware
// policy in this package: creation on add, discard on drop (CACHE-10), reset
// on flush, lazy get, and size reporting. The scoring/selection algorithm on
// top of it is what actually differs between policies.
type trieStore struct {
	mu    sync.RWMutex
	cfg   kvcache.Config
	tries map[string]*kvcache.Trie // by canonical backend URL
}

func newTrieStore(cfg kvcache.Config) *trieStore {
	return &trieStore{cfg: cfg, tries: map[string]*kvcache.Trie{}}
}

func (s *trieStore) add(url string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.tries[url]; !ok {
		s.tries[url] = kvcache.New(s.cfg)
	}
}

func (s *trieStore) drop(url string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.tries, url)
}

func (s *trieStore) flush() {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, t := range s.tries {
		t.Reset()
	}
}

func (s *trieStore) get(url string) *kvcache.Trie {
	s.mu.RLock()
	t, ok := s.tries[url]
	s.mu.RUnlock()
	if ok {
		return t
	}
	// A backend can be selected before the add hook has run.
	s.mu.Lock()
	defer s.mu.Unlock()
	if t, ok := s.tries[url]; ok {
		return t
	}
	t = kvcache.New(s.cfg)
	s.tries[url] = t
	return t
}

func (s *trieStore) stats() map[string][2]int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string][2]int64, len(s.tries))
	for url, t := range s.tries {
		n, tok, _ := t.Stats()
		out[url] = [2]int64{n, tok}
	}
	return out
}
