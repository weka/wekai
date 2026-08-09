package affinity

import (
	"context"
	"sync"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// FlowName identifies the router's one routing flow in metrics and logs. It is
// no longer selectable: there is nothing to select between.
const FlowName = "prefix-cache-split"

// Defaults.
const (
	DefaultSplitGuard = 0.20
	DefaultTailTTL    = 5 * time.Minute
	// DefaultRefusalTTL is how long a backend's 429 keeps it out of its own
	// prefixes. Short on purpose: it is a hint that saves the next request a
	// wasted round trip, not a health verdict, and a success clears it
	// immediately anyway.
	DefaultRefusalTTL = 2 * time.Second
	// DefaultPoolName labels the implicit whole-router pool — the shape a
	// router has when it fronts one fleet and routes every model to it.
	DefaultPoolName = "default"
)

// Config configures the flow. Every field beyond the guard and the TTLs turns
// an optional signal on by being set.
type Config struct {
	// NodeConcurrency enables the concurrency signal: the router's own guess at
	// the backend's vLLM --max-num-seqs, used to predict saturation instead of
	// discovering it one wasted round trip at a time. Zero (the default) leaves
	// the signal off and the refused signal alone decides.
	//
	// It is no longer mandatory and no longer an admission filter. The backend's
	// own 429 is the ultimate signal; this one is an early warning.
	NodeConcurrency int64

	// RebalanceRatio enables the imbalance signal: a backend is unusable while
	// (inflight - fleetMin) / inflight exceeds this. 0.5 means "rebalance once
	// the gap is more than half the busier side". Zero leaves it off, which is
	// the default because a fleet where affinity is working is supposed to look
	// imbalanced.
	RebalanceRatio float64

	// SplitGuard keeps a split from landing on a backend that is nearly as
	// loaded as the ones it is relieving, which is what stops every backend
	// from eventually being marked as holding every prefix. A candidate
	// qualifies while its in-flight count is below ref*(1-SplitGuard), where
	// ref is supplied by whichever signal made the holders unusable.
	SplitGuard float64

	// TailTTL is how long a leaf run may go untouched before eviction.
	TailTTL time.Duration

	// RefusalTTL is how long a 429 latches a backend out of its own prefixes.
	RefusalTTL time.Duration

	// PoolName labels this flow's metrics. One router may front several pools —
	// different models, different fleets — whose trees are unrelated, and whose
	// duplication and decision counts are meaningless summed together.
	PoolName string

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
	if c.RefusalTTL <= 0 {
		c.RefusalTTL = DefaultRefusalTTL
	}
	if c.Clock == nil {
		c.Clock = clock.Real{}
	}
	if c.PoolName == "" {
		c.PoolName = DefaultPoolName
	}
	return c
}

// Policy routes by prefix affinity over the shared marked tree.
//
// The ladder, in order. limit is Config.NodeConcurrency; candidates arrive
// already filtered by the gateway to healthy backends BELOW that limit.
//
// This is the router's ONE flow. There is no policy to choose; what varies is
// which signals are enabled, and a signal only shrinks the usable set and sets
// the guard's reference — it never routes. The ladder below is the same
// whichever are on.
//
// First, `usable` = candidates that no enabled signal calls saturated. Then:
//
//  1. cache  — the DEEPEST marked run on the request's path is held by a usable
//     backend. Route to the least-loaded holder. No threshold: the
//     deepest run wins however small a share of the request it is.
//     The anchor must be the deepest marked run, not merely some
//     ancestor a candidate happens to hold; anchoring on an ancestor
//     is how a backend ends up marked as holding a session tail it
//     never had, which is the whole of the duplication problem.
//  2. split  — holders exist but none is usable. Route to the least-loaded
//     backend outside the holder set whose in-flight is under
//     ref*(1-guard), where ref comes from the signal that made the
//     holders unusable, and record it as a new holder. The holder set
//     GROWS under pressure instead of affinity being thrown away.
//  3. reject — holders exist and nothing clears the guard:
//     ErrSplitGuardBlocked, answered 429. Idle capacity may exist and
//     is deliberately left unused, because spending it would mean a
//     duplicate copy of this prefix and the guard says that copy is
//     not worth it. Anton's call, taken with the numbers in front of
//     him: mean holders per block near 1.05 is worth 429s and worth
//     nodes not always reaching full concurrency.
//  4. load   — nothing is marked anywhere: a genuinely new prompt, so there is
//     no holder set to split from and nothing to duplicate. Route by
//     least-outstanding over the usable set and record the result.
// Before all of them: if no backend is usable at all, ErrAllBackendsSaturated,
// answered 429. Distinct from tier 3, which means capacity existed and the
// guard refused to spend it on a copy.
//
// Admission is no longer the gateway's. It filters for health only; every
// capacity judgement is a signal here, so that "is this backend full" has one
// answer in one place rather than a router-side guess in front of a policy that
// re-derives it.
type Policy struct {
	cfg      Config
	tree     *tree
	fallback policy.Policy

	// refused is the ultimate signal and always present; signals holds it
	// alongside whatever else is enabled, so the flow iterates one list.
	refused *refusedSignal
	signals []signal

	// m is this pool's metrics, resolved once here rather than per request.
	m *metrics.PoolMetrics

	// backends is the set the registry has told us about, kept only so
	// /router-viz can render identity, health and load alongside the tree. Not
	// consulted by routing, which uses the candidate set it is handed.
	mu       sync.RWMutex
	backends map[string]*registry.Backend
}

// New builds the flow. The refused signal is always on; NodeConcurrency and
// RebalanceRatio each enable one more by being set.
func New(cfg Config, fallback policy.Policy) (*Policy, error) {
	cfg = cfg.withDefaults()
	if fallback == nil {
		fallback = policy.LeastOutstanding{}
	}
	refused := newRefusedSignal(cfg.Clock, cfg.RefusalTTL)
	p := &Policy{
		cfg:      cfg,
		tree:     newTree(cfg.Clock, cfg.TailTTL),
		fallback: fallback,
		refused:  refused,
		signals:  []signal{refused},
		backends: map[string]*registry.Backend{},
		m:        metrics.ForPool(cfg.PoolName),
	}
	if cfg.NodeConcurrency > 0 {
		p.signals = append(p.signals, concurrencySignal{limit: cfg.NodeConcurrency})
	}
	if cfg.RebalanceRatio > 0 {
		p.signals = append(p.signals, imbalanceSignal{ratio: cfg.RebalanceRatio})
	}
	return p, nil
}

// OnRefused latches a backend's 429 as the ultimate signal. Called by the proxy
// once the upstream status is known.
func (p *Policy) OnRefused(b *registry.Backend) { p.refused.record(b) }

// OnAccepted clears a backend's refusal latch. Called by the proxy on a
// successful response, so recovery costs no waiting.
func (p *Policy) OnAccepted(b *registry.Backend) { p.refused.clear(b) }

func (p *Policy) Name() string { return FlowName }

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

	// Ask the signals once, before anything else, and route only over what they
	// leave usable. Doing it here rather than in the gateway is what collapses
	// the old policy zoo into one flow: "is this backend full" gets one answer
	// in one place, and every tier below inherits it.
	//
	// guardRef is the reference the split guard is measured against — the
	// in-flight level of the thing being relieved. When several signals fire,
	// the SMALLEST wins: it is the strictest, and a guard that is too strict
	// costs a 429 while one that is too loose costs a permanent extra copy of
	// the prefix.
	view := loadView{minInflight: cands[0].Inflight()}
	for _, c := range cands[1:] {
		if l := c.Inflight(); l < view.minInflight {
			view.minInflight = l
		}
	}
	usable := make([]*registry.Backend, 0, len(cands))
	guardRef := int64(-1)
	for _, c := range cands {
		blocked := false
		for _, sg := range p.signals {
			hit, ref := sg.saturated(c, view)
			if !hit {
				continue
			}
			blocked = true
			p.m.Signal(sg.name())
			if ref > 0 && (guardRef < 0 || ref < guardRef) {
				guardRef = ref
			}
		}
		if !blocked {
			usable = append(usable, c)
		}
	}

	// Slots are resolved once and reused, so the candidate mask and the
	// per-candidate membership test cost one map read each rather than two.
	var slotBuf [64]int
	slots := slotBuf[:0]
	if len(usable) > cap(slots) {
		slots = make([]int, 0, len(usable))
	}
	var candMask markSet
	for _, c := range usable {
		s := p.tree.slotOrCreate(c.URL)
		slots = append(slots, s)
		candMask.Add(s)
	}

	// Nothing is usable: every backend is saturated by some signal. This is the
	// rejection the gateway used to issue as all_backends_at_capacity, moved
	// here so one component decides capacity.
	if len(usable) == 0 {
		// Counted by the gateway when it writes the 429, not here: one
		// rejection must be one increment, and Select can run twice for one
		// request when the proxy retries.
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "all_saturated").Inc()
		return nil, policy.ErrAllBackendsSaturated
	}

	// No routable prefix — an embeddings call, an unparseable body: there is
	// nothing to be affine to, so the selector decides (CU-11). It decides over
	// the USABLE set, not every candidate: a request without a prefix still may
	// not be sent to a backend that has already said it is full.
	if len(rr.Units) == 0 {
		rr.PolicyState = noMarkDecision
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_units").Inc()
		p.m.Decision("load")
		return p.fallback.Select(ctx, usable, rr)
	}

	pth := path{units: rr.Units, modelKey: modelKey(rr.Model)}
	a := p.tree.walk(pth, candMask)

	// No signal fired, so nothing is being relieved and the guard has no
	// observed reference. Fall back to the configured limit when there is one;
	// otherwise the fleet's own busiest backend is the only scale available.
	if guardRef <= 0 {
		guardRef = p.cfg.NodeConcurrency
	}
	if guardRef <= 0 {
		guardRef = view.minInflight + 1
	}
	threshold := float64(guardRef) * (1 - p.cfg.SplitGuard)

	// shallow means the anchor is NOT the deepest marked run: this request's own
	// holders are unavailable and only a shared ancestor is left. Serving it
	// there and marking the whole path is what put a session's private tail on
	// every backend in the fleet, so under LadderStrict it is not a cache hit —
	// it falls through to the split, which is guarded.
	shallow := !a.pool.Empty() && !a.availDeepest
	if shallow {
		p.m.ShallowAnchors.Inc()
		p.m.ShallowAnchorBlocks.Add(float64(a.heldBlocks - a.anchorBlocks))
	}

	// Tier 1: a usable backend holds the deepest marked run.
	if !a.pool.Empty() && !(shallow && p.cfg.Ladder == LadderStrict) {
		pool := make([]*registry.Backend, 0, len(usable))
		for i, c := range usable {
			if a.pool.Has(slots[i]) {
				pool = append(pool, c)
			}
		}
		if len(pool) > 0 {
			rr.PolicyState = markDecision
			p.m.Decision("cache")
			p.m.AnchorBlocks.Observe(float64(a.anchorBlocks))
			p.m.PoolSize.Observe(float64(len(pool)))
			p.m.PredictedFraction.Observe(fraction(a.anchorBlocks, pth.len()))
			return p.fallback.Select(ctx, pool, rr)
		}
	}

	// Tiers 2 and 3 both mean: this prefix has holders, and not one of them is
	// available. Either they are all at their concurrency limit or they have
	// left the fleet.
	if !a.held.Empty() {
		split := make([]*registry.Backend, 0, len(usable))
		for i, c := range usable {
			if a.held.Has(slots[i]) {
				continue // already a holder, and unavailable
			}
			if float64(c.Inflight()) < threshold {
				split = append(split, c)
			}
		}
		if len(split) > 0 {
			rr.PolicyState = markDecision
			p.m.Decision("split")
			p.m.Splits.Inc()
			p.m.AnchorBlocks.Observe(float64(a.matched))
			return p.fallback.Select(ctx, split, rr)
		}

		// Tier 3. Nothing cleared the guard. Idle capacity may exist — a
		// backend between limit*(1-guard) and the limit is idle and refused on
		// purpose — because using it means another copy of this prefix, and the
		// guard is the rule that says the copy is not worth it. Splitting is the
		// only way a backend is ever added to a holder set; there is no
		// serve-anyway path.
		if p.cfg.Ladder == LadderStrict {
			p.m.GuardRejects.Inc()
			metrics.PolicyFallbacks.WithLabelValues(p.Name(), "guard_blocked").Inc()
			return nil, policy.ErrSplitGuardBlocked
		}
		rr.PolicyState = noMarkDecision
		p.m.Decision("overflow")
		p.m.Overflows.Inc()
		metrics.PolicyFallbacks.WithLabelValues(p.Name(), "guard_blocked").Inc()
		return p.fallback.Select(ctx, cands, rr)
	}


	// Tier 4: a genuinely new prompt. Anton's "a single request going via
	// load-split, but then every following request will have cache path split
	// based routing".
	rr.PolicyState = markDecision
	p.m.Decision("load")
	metrics.PolicyFallbacks.WithLabelValues(p.Name(), "no_holders").Inc()
	return p.fallback.Select(ctx, usable, rr)
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
		p.m.BlocksExpired.Add(float64(freed))
	}
	return freed
}

// TailTTL reports the configured eviction TTL, so the caller can size the
// sweep interval against it.
func (p *Policy) TailTTL() time.Duration { return p.cfg.TailTTL }

// PublishGauges exports tree size and duplication on the health interval.
func (p *Policy) PublishGauges() {
	st := p.tree.stats()
	p.m.TreeRuns.Set(float64(st.Runs))
	p.m.TailSet.Set(float64(st.Tails))
	p.m.AvgCopies.Set(st.AvgCopies)
	for url, bs := range p.tree.perBackend() {
		metrics.CacheEntries.WithLabelValues(url).Set(float64(bs.Blocks))
		metrics.CacheTokens.WithLabelValues(url).Set(float64(bs.Tokens))
	}
}
