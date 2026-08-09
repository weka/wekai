package affinity

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// PolicyName is the --policy value that selects this policy.
const PolicyName = "prefix-cache-split"

// Defaults.
const (
	DefaultSplitGuard = 0.20
	DefaultTailTTL    = 5 * time.Minute
)

// ErrNoConcurrencyLimit reports the one configuration this policy cannot run
// without. See Config.NodeConcurrency.
var ErrNoConcurrencyLimit = errors.New(
	"affinity: --max-node-concurrency must be set for policy " + PolicyName +
		"; it is the per-backend concurrency limit the split guard is measured against, " +
		"and it should match the backends' vLLM --max-num-seqs")

// Config configures Policy.
type Config struct {
	// NodeConcurrency is each backend's concurrency limit — the signal Anton
	// calls ultimate: "if we throw out everything else, system behaves well".
	//
	// It MUST be set. Every other capacity source in the shipped configuration
	// degrades to 1 (--backends carries no capacity field,
	// MaxInflightPerBackend defaults to 1, and Backend.Capacity clamps
	// anything below 1 up to 1), which would make the guard arithmetic
	// meaningless without saying so. New returns ErrNoConcurrencyLimit rather
	// than inventing a second default that could disagree with the gateway's
	// own admission filter — one number has to mean one thing.
	NodeConcurrency int64

	// SplitGuard keeps a split from landing on a backend that is nearly as
	// loaded as the ones it is relieving, which is what stops every backend
	// from eventually being marked as holding every prefix. A candidate
	// qualifies while its in-flight count is below
	// NodeConcurrency * (1 - SplitGuard).
	SplitGuard float64

	// TailTTL is how long a leaf run may go untouched before eviction.
	TailTTL time.Duration

	// Ladder selects the decision ladder. Default (zero value) is LadderStrict,
	// the reference design. See ladderMode.
	Ladder ladderMode

	Clock clock.Clock
}

// ladderMode selects what happens when a request's prefix has holders and none
// of them is available.
type ladderMode int

const (
	// LadderStrict is the reference design and the default: the guard is an
	// absolute rule. A request either lands on a backend that holds its
	// deepest marked prefix, or splits onto one far enough below the limit to
	// be worth a copy, or is rejected. There is no serve-anyway path, so the
	// only way a backend is ever recorded as holding a prefix is a guarded
	// split.
	//
	// This deliberately rejects while idle capacity exists — a backend between
	// 80% and 100% of the limit is idle-but-unusable — which is Anton's call:
	// duplication near 1.05 is worth 429s and worth nodes not always reaching
	// full concurrency.
	LadderStrict ladderMode = iota

	// LadderServeAnyway is the previous behavior, kept only so the trade can be
	// re-measured (copies_sweep_test.go). Tier 1 anchors on ANY run a candidate
	// holds rather than the deepest marked one, and a request that clears no
	// guard is served anyway without a mark. Measured over a 4x32 fleet it
	// produces ~50% shallow anchors and avg copies 2.5 at full utilisation,
	// against 1.0x for LadderStrict.
	LadderServeAnyway
)

func (c Config) withDefaults() Config {
	if c.SplitGuard <= 0 || c.SplitGuard >= 1 {
		c.SplitGuard = DefaultSplitGuard
	}
	if c.TailTTL <= 0 {
		c.TailTTL = DefaultTailTTL
	}
	if c.Clock == nil {
		c.Clock = clock.Real{}
	}
	return c
}

// Policy routes by prefix affinity over the shared marked tree.
//
// The ladder, in order. limit is Config.NodeConcurrency; candidates arrive
// already filtered by the gateway to healthy backends BELOW that limit.
//
//  1. cache  — the DEEPEST marked run on the request's path is held by a
//     candidate. Route to the least-loaded holder. No threshold: the
//     deepest run wins however small a share of the request it is.
//     The anchor must be the deepest marked run, not merely some
//     ancestor a candidate happens to hold; anchoring on an ancestor
//     is how a backend ends up marked as holding a session tail it
//     never had, which is the whole of the duplication problem.
//  2. split  — holders exist but none is available, i.e. they are all at the
//     limit or gone. Route to the least-loaded backend outside the
//     holder set whose in-flight is under limit*(1-guard), and record
//     it as a new holder. The holder set GROWS under pressure instead
//     of affinity being thrown away.
//  3. reject — holders exist and nothing clears the guard. Return
//     ErrSplitGuardBlocked; the gateway answers 429. Idle capacity
//     may exist and is deliberately left unused: spending it would
//     mean a duplicate copy of this prefix, and the guard says that
//     copy is not worth it. This is Anton's call, taken with the
//     numbers in front of him — mean holders per block near 1.05 is
//     worth 429s and worth nodes not always reaching full
//     concurrency.
//  4. load   — nothing is marked anywhere: a genuinely new prompt, so there is
//     no holder set to split from and nothing to duplicate. Route by
//     least-outstanding and record the result.
//
// The gateway's own admission is unchanged and sits outside this: it filters
// candidates to healthy backends under the limit and returns 429
// all_backends_at_capacity when none are left. That rejection means zero idle
// slots fleet-wide; tier 3's means idle slots existed but every one of them
// was too close to the limit to be worth a copy.
type Policy struct {
	cfg      Config
	tree     *tree
	fallback policy.Policy

	// backends is the set the registry has told us about, kept only so
	// /router-viz can render identity, health and load alongside the tree. Not
	// consulted by routing, which uses the candidate set it is handed.
	mu       sync.RWMutex
	backends map[string]*registry.Backend
}

// New builds the policy. It returns ErrNoConcurrencyLimit when
// cfg.NodeConcurrency is unset; see that field.
func New(cfg Config, fallback policy.Policy) (*Policy, error) {
	if cfg.NodeConcurrency <= 0 {
		return nil, ErrNoConcurrencyLimit
	}
	cfg = cfg.withDefaults()
	if fallback == nil {
		fallback = policy.LeastOutstanding{}
	}
	return &Policy{
		cfg:      cfg,
		tree:     newTree(cfg.Clock, cfg.TailTTL),
		fallback: fallback,
		backends: map[string]*registry.Backend{},
	}, nil
}

func (p *Policy) Name() string { return PolicyName }

// decision is what Select tells Commit. Stored in RoutingRequest.PolicyState.
type decision struct {
	// mark is false only for the overflow tier: the backend served the request
	// but must not be recorded as holding the prefix, or the guard's whole
	// purpose is defeated one overflow at a time.
	mark bool
}

var markDecision = &decision{mark: true}
var noMarkDecision = &decision{mark: false}

// modelKey isolates one model's tree from another's. The gateway filters
// candidates by DialectID and never by Model, so a router fronting two models
// on one dialect would otherwise credit one model's KV cache for the other's
// prompt.
func modelKey(model string) uint64 {
	if model == "" {
		return 0
	}
	return kvcache.HashContent("model", []byte(model))
}

// Select implements the ladder documented on Policy.
func (p *Policy) Select(ctx context.Context, cands []*registry.Backend, rr *policy.RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, policy.ErrNoCandidates
	}
	// No routable prefix: decline rather than guess (CU-11).
	if len(rr.Units) == 0 {
		rr.PolicyState = noMarkDecision
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_units").Inc()
		metrics.RouteDecisions.WithLabelValues("load").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}

	pth := path{units: rr.Units, modelKey: modelKey(rr.Model)}

	// Slots are resolved once and reused, so the candidate mask and the
	// per-candidate membership test cost one map read each rather than two.
	var slotBuf [64]int
	slots := slotBuf[:0]
	if len(cands) > cap(slots) {
		slots = make([]int, 0, len(cands))
	}
	var candMask markSet
	for _, c := range cands {
		s := p.tree.slotOrCreate(c.URL)
		slots = append(slots, s)
		candMask.Add(s)
	}

	a := p.tree.walk(pth, candMask)

	threshold := float64(p.cfg.NodeConcurrency) * (1 - p.cfg.SplitGuard)

	// shallow means the anchor is NOT the deepest marked run: this request's own
	// holders are unavailable and only a shared ancestor is left. Serving it
	// there and marking the whole path is what put a session's private tail on
	// every backend in the fleet, so under LadderStrict it is not a cache hit —
	// it falls through to the split, which is guarded.
	shallow := !a.pool.Empty() && !a.availDeepest
	if shallow {
		metrics.CacheShallowAnchors.Inc()
		metrics.CacheShallowAnchorBlocks.Add(float64(a.heldBlocks - a.anchorBlocks))
	}

	// Tier 1: a candidate holds the deepest marked run.
	if !a.pool.Empty() && !(shallow && p.cfg.Ladder == LadderStrict) {
		pool := make([]*registry.Backend, 0, len(cands))
		for i, c := range cands {
			if a.pool.Has(slots[i]) {
				pool = append(pool, c)
			}
		}
		if len(pool) > 0 {
			rr.PolicyState = markDecision
			metrics.RouteDecisions.WithLabelValues("cache").Inc()
			metrics.CacheAnchorBlocks.Observe(float64(a.anchorBlocks))
			metrics.CachePoolSize.Observe(float64(len(pool)))
			metrics.CachePredictedFraction.Observe(fraction(a.anchorBlocks, pth.len()))
			return p.fallback.Select(ctx, pool, rr)
		}
	}

	// Tiers 2 and 3 both mean: this prefix has holders, and not one of them is
	// available. Either they are all at their concurrency limit or they have
	// left the fleet.
	if !a.held.Empty() {
		split := make([]*registry.Backend, 0, len(cands))
		for i, c := range cands {
			if a.held.Has(slots[i]) {
				continue // already a holder, and unavailable
			}
			if float64(c.Inflight()) < threshold {
				split = append(split, c)
			}
		}
		if len(split) > 0 {
			rr.PolicyState = markDecision
			metrics.RouteDecisions.WithLabelValues("split").Inc()
			metrics.CacheSplits.Inc()
			metrics.CacheAnchorBlocks.Observe(float64(a.matched))
			return p.fallback.Select(ctx, split, rr)
		}

		// Tier 3. Nothing cleared the guard. Idle capacity may exist — a
		// backend between limit*(1-guard) and the limit is idle and refused on
		// purpose — because using it means another copy of this prefix, and the
		// guard is the rule that says the copy is not worth it. Splitting is the
		// only way a backend is ever added to a holder set; there is no
		// serve-anyway path.
		if p.cfg.Ladder == LadderStrict {
			metrics.CacheGuardRejects.Inc()
			metrics.PolicyFallbacks.WithLabelValues(p.Name(), "guard_blocked").Inc()
			return nil, policy.ErrSplitGuardBlocked
		}
		rr.PolicyState = noMarkDecision
		metrics.RouteDecisions.WithLabelValues("overflow").Inc()
		metrics.CacheOverflows.Inc()
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "guard_blocked").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}

	// Tier 4: a genuinely new prompt. Anton's "a single request going via
	// load-split, but then every following request will have cache path split
	// based routing".
	rr.PolicyState = markDecision
	metrics.RouteDecisions.WithLabelValues("load").Inc()
	metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_holders").Inc()
	return p.fallback.Select(ctx, cands, rr)
}

func fraction(part, whole int) float64 {
	if whole <= 0 {
		return 0
	}
	return float64(part) / float64(whole)
}

// Commit records that b served the request, marking it on EVERY run along the
// path rather than only the deepest — which is what makes a split converge.
//
// Called only after the upstream accepted the request, never before dispatch
// (CACHE-9, R3), so a refused attempt leaves no trace.
func (p *Policy) Commit(b *registry.Backend, rr *policy.RoutingRequest) {
	if b == nil || len(rr.Units) == 0 {
		return
	}
	// A nil state means Commit ran without a Select from this policy. Marking
	// is the safe default: it can only make a future request stickier, whereas
	// silently not marking would lose affinity with no signal.
	if d, ok := rr.PolicyState.(*decision); ok && !d.mark {
		return
	}
	pth := path{units: rr.Units, modelKey: modelKey(rr.Model)}
	// The backend is read from b rather than from the recorded decision: the
	// proxy retries onto a different backend without re-running Commit's
	// caller, so b is the authority on who actually served it.
	p.tree.commit(pth, p.tree.slotOrCreate(b.URL))
}

// AddBackend allocates a backend's slot. Wired to the registry's add hook.
func (p *Policy) AddBackend(b *registry.Backend) {
	p.tree.slotFor(b.URL)
	p.mu.Lock()
	p.backends[b.URL] = b
	p.mu.Unlock()
}

// DropBackend clears a backend's marks fleet-wide and frees its slot, in that
// order, so a later backend cannot inherit prefixes it never served (CACHE-10).
func (p *Policy) DropBackend(b *registry.Backend) {
	p.mu.Lock()
	delete(p.backends, b.URL)
	p.mu.Unlock()
	p.tree.dropBackend(b.URL)
}

// Flush discards the whole tree. Backs the same operational reset the older
// cache policies expose.
func (p *Policy) Flush() { p.tree.flush() }

// Sweep evicts idle tails once. Driven on a ticker by the router's main loop
// rather than from the request path, so a routing decision never pays for
// eviction. Returns the blocks freed.
func (p *Policy) Sweep() int64 {
	freed := p.tree.sweep()
	if freed > 0 {
		metrics.CacheBlocksExpired.Add(float64(freed))
	}
	return freed
}

// TailTTL reports the configured eviction TTL, so the caller can size the
// sweep interval against it.
func (p *Policy) TailTTL() time.Duration { return p.cfg.TailTTL }

// PublishGauges exports tree size and duplication on the health interval.
func (p *Policy) PublishGauges() {
	st := p.tree.stats()
	metrics.CacheTreeRuns.Set(float64(st.Runs))
	metrics.CacheTailSet.Set(float64(st.Tails))
	metrics.CacheAvgCopies.Set(st.AvgCopies)
	for url, bs := range p.tree.perBackend() {
		metrics.CacheEntries.WithLabelValues(url).Set(float64(bs.Blocks))
		metrics.CacheTokens.WithLabelValues(url).Set(float64(bs.Tokens))
	}
}
