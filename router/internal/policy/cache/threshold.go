package cache

import (
	"context"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/viz"
)

// ThresholdConfig configures ThresholdPolicy.
type ThresholdConfig struct {
	// CacheThreshold is the predicted-hit-rate fraction a candidate must
	// exceed to count as holding the request's prefix.
	CacheThreshold float64
	// MaxPending is the in-flight ceiling for the single-candidate case: a
	// lone cache candidate is used only while its in-flight count stays
	// below this, otherwise cache is ignored entirely for this request.
	MaxPending int64
	Trie       kvcache.Config
}

func DefaultThresholdConfig() ThresholdConfig {
	return ThresholdConfig{
		CacheThreshold: 0.5,
		MaxPending:     32,
		Trie:           kvcache.RouterConfig(),
	}
}

// ThresholdPolicy routes by filtering candidates to those predicted to hold
// the request's prefix above CacheThreshold, then choosing among that filtered
// set rather than picking a single best-scoring backend:
//
//   - no candidate clears the threshold (a new prompt): least-loaded of all.
//   - exactly one clears it: use it while its in-flight count is below
//     MaxPending, otherwise ignore cache and fall back to least-loaded of all.
//   - more than one clears it: least-loaded among that filtered set.
//
// This is a different shape from Policy's single-best-plus-spill-guard model
// in cache.go — it is not a bug that the two disagree on ties or on when load
// overrides affinity; they encode two distinct routing strategies.
type ThresholdPolicy struct {
	cfg      ThresholdConfig
	fallback policy.Policy
	store    *trieStore
}

func NewThreshold(cfg ThresholdConfig, fallback policy.Policy) *ThresholdPolicy {
	if cfg.CacheThreshold <= 0 {
		cfg.CacheThreshold = DefaultThresholdConfig().CacheThreshold
	}
	if cfg.MaxPending <= 0 {
		cfg.MaxPending = DefaultThresholdConfig().MaxPending
	}
	if fallback == nil {
		fallback = policy.LeastOutstanding{}
	}
	return &ThresholdPolicy{cfg: cfg, fallback: fallback, store: newTrieStore(cfg.Trie)}
}

func (p *ThresholdPolicy) Name() string { return "prefix-cache-candidates" }

// AddBackend creates a model for a backend. Wired to the registry's add hook.
func (p *ThresholdPolicy) AddBackend(b *registry.Backend) { p.store.add(b) }

// DropBackend discards a backend's model (CACHE-10).
func (p *ThresholdPolicy) DropBackend(b *registry.Backend) { p.store.drop(b) }

// Flush clears every model. Backs POST /flush_cache; touches nothing else.
func (p *ThresholdPolicy) Flush() { p.store.flush() }

// Stats reports per-backend model size, for metrics.
func (p *ThresholdPolicy) Stats() map[string][2]int64 { return p.store.stats() }

// Snapshot implements viz.DataSource for the live KV block map at
// /router-viz — see snapshot.go for the implementation shared with Policy.
func (p *ThresholdPolicy) Snapshot(limit int) viz.Snapshot { return p.store.snapshot(limit) }

// PublishGauges exports per-backend model size on the health interval.
func (p *ThresholdPolicy) PublishGauges() {
	for url, st := range p.Stats() {
		metrics.CacheEntries.WithLabelValues(url).Set(float64(st[0]))
		metrics.CacheTokens.WithLabelValues(url).Set(float64(st[1]))
	}
}

// Select implements the candidate-filter decision tree described on
// ThresholdPolicy.
func (p *ThresholdPolicy) Select(ctx context.Context, cands []*registry.Backend, rr *policy.RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, policy.ErrNoCandidates
	}
	// No routable prefix: decline rather than guess (CU-11).
	if len(rr.Units) == 0 {
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_units").Inc()
		metrics.RouteDecisions.WithLabelValues("load").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}

	// Query every candidate. Read-only: considering a backend must not teach
	// it anything, or every model converges to every request. hotFrac is kept
	// aligned with hot by index so that whichever candidate is eventually
	// selected, its own predicted fraction (not some other candidate's, and
	// not an average across ones that were never going to be picked) is what
	// gets recorded into CachePredictedFraction below.
	var hot []*registry.Backend
	var hotFrac []float64
	fracs := make([]float64, 0, len(cands))
	for _, c := range cands {
		cached, total := p.store.get(c.URL).Query(rr.Units)
		if total == 0 {
			continue
		}
		frac := float64(cached) / float64(total)
		fracs = append(fracs, frac)
		if frac > p.cfg.CacheThreshold {
			hot = append(hot, c)
			hotFrac = append(hotFrac, frac)
		}
	}
	publishPredictionStats(fracs)

	switch len(hot) {
	case 0:
		// A completely new prompt: no affinity signal at all, defer to load.
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_candidates").Inc()
		metrics.RouteDecisions.WithLabelValues("load").Inc()
		return p.fallback.Select(ctx, cands, rr)
	case 1:
		if hot[0].Inflight() < p.cfg.MaxPending {
			metrics.RouteDecisions.WithLabelValues("cache").Inc()
			metrics.CachePredictedFraction.Observe(hotFrac[0])
			return hot[0], nil
		}
		// The lone candidate is too busy: ignore cache for this request
		// entirely rather than pile more work onto it.
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "pending_exceeded").Inc()
		metrics.RouteDecisions.WithLabelValues("load").Inc()
		return p.fallback.Select(ctx, cands, rr)
	default:
		// Still a cache decision: membership in `hot` is what cache affinity
		// decided, and least-outstanding here is only the tie-break among
		// those candidates, not a fallback (no PolicyFallbacks — this is the
		// designed multi-candidate branch, not a decline).
		metrics.RouteDecisions.WithLabelValues("cache").Inc()
		winner, err := policy.LeastOutstanding{}.Select(ctx, hot, rr)
		if err == nil && winner != nil {
			for i, c := range hot {
				if c == winner {
					metrics.CachePredictedFraction.Observe(hotFrac[i])
					break
				}
			}
		}
		return winner, err
	}
}

// Commit records that a backend served the request.
//
// Must be called only after the upstream has accepted it — never before
// dispatch (CACHE-9, R3).
func (p *ThresholdPolicy) Commit(b *registry.Backend, rr *policy.RoutingRequest) {
	if b == nil || len(rr.Units) == 0 {
		return
	}
	p.store.get(b.URL).Commit(rr.Units)
}
