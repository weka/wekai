package affinity

import (
	"context"
	"math"
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

	// SoftNodeConcurrency splits NodeConcurrency into a BAND. Below it a holder
	// is a plain cache hit; between it and the hard limit the router would
	// rather spread the request than pile on, so tier 1 is skipped and the
	// guarded split gets first refusal; at the hard limit the backend is
	// saturated and leaves the usable set entirely, exactly as before.
	//
	// What makes it more than an earlier cliff is where it lands when spreading
	// is refused. A request the guard blocks does NOT go to a backend holding
	// none of its prefix — it goes back to the least-loaded holder and queues
	// there. Both choices relieve the same moment and pay opposite prices:
	// TransientFallback keeps the holder's queue short and pays a full prefill
	// on a cold backend, this keeps the cache hit and pays the queue. On a fleet
	// where prefix placement is already worth a 27% spread in prompt-token
	// throughput between the best and worst backend, that is the trade worth
	// having a knob for.
	//
	// Zero (the default) leaves NodeConcurrency a single hard limit.
	SoftNodeConcurrency int64

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

	// TransientFallback enables serving a request on a backend the split guard
	// refused, WITHOUT recording it as a holder. A candidate qualifies while its
	// in-flight is below ref*(1-TransientFallback), against the same reference
	// the guard uses — so a value BELOW SplitGuard is a looser bar, which is the
	// only setting that does anything. Zero (the default) leaves the guard's
	// rejection final.
	//
	// The guard protects the tree, not capacity: a split adds a holder
	// permanently, and a fleet that splits freely converges on everyone holding
	// everything. A request served without a mark costs nothing permanent, so it
	// can be allowed much closer to the holders' own load.
	TransientFallback float64

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
	// A transient threshold at or above the guard is a STRICTER bar than the
	// guard it exists to relax, so it could never admit a request the guard had
	// already refused. Quietly doing nothing is the wrong failure here: the
	// operator asked for a fallback and would get rejections instead, with no
	// way to tell the difference from the outside.
	if c.TransientFallback < 0 || c.TransientFallback >= c.SplitGuard {
		c.TransientFallback = 0
	}
	// A soft limit only means something as the lower edge of a band. At or above
	// the hard limit it names a state the backend can never be in, and with no
	// hard limit at all there is no band for it to be the floor of — the
	// concurrency signal is off, so nothing bounds the stretch.
	if c.SoftNodeConcurrency < 0 || c.NodeConcurrency <= 0 || c.SoftNodeConcurrency >= c.NodeConcurrency {
		c.SoftNodeConcurrency = 0
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
//
//  2. split  — holders exist but none is usable, or --soft-node-concurrency is
//     set and every available holder is past it. Route to the
//     least-loaded backend outside the holder set whose in-flight is
//     under ref*(1-guard), where ref comes from the signal that made
//     the holders unusable (or from the soft limit, when that is what
//     pushed the request out of tier 1), and record it as a new
//     holder. The holder set GROWS under pressure instead of affinity
//     being thrown away.
//
//     2b. stretch — only with --soft-node-concurrency. Nothing cleared the
//     guard, but the holders are merely past soft rather than
//     saturated, so the request goes back to the least loaded of
//     them and queues. Keeps the cache hit, pays the queue.
//
//     2c. overflow — only with --transient-fallback-threshold. Serves a
//     non-holder inside a looser margin WITHOUT marking it. Keeps
//     the queue short, pays a full prefill.
//
//  3. reject — holders exist and nothing clears the guard:
//     ErrSplitGuardBlocked, answered 429. Idle capacity may exist and
//     is deliberately left unused, because spending it would mean a
//     duplicate copy of this prefix and the guard says that copy is
//     not worth it. Anton's call, taken with the numbers in front of
//     him: mean holders per block near 1.05 is worth 429s and worth
//     nodes not always reaching full concurrency.
//
//  4. load   — nothing is marked anywhere: a genuinely new prompt, so there is
//     no holder set to split from and nothing to duplicate. Route by
//     least-outstanding over the usable set and record the result.
//
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
	// Saturated backends and the reference each supplies, kept PER BACKEND.
	// Reducing them to one fleet-wide number here was the bug: the guard would
	// then measure this prefix against whichever backend anywhere in the pool
	// happened to refuse at the lowest in-flight, including backends that hold
	// nothing of this prefix. One node that refused while carrying 2 — a vLLM
	// out of KV cache rather than out of sequence slots — set a threshold of
	// 1.6 for every request in the pool, so no candidate could ever clear it
	// and the flow rejected with the fleet nearly idle.
	saturated := make(map[string]int64, len(cands))
	for _, c := range cands {
		ref := int64(-1)
		for _, sg := range p.signals {
			hit, r := sg.saturated(c, view)
			if !hit {
				continue
			}
			p.m.Signal(sg.name())
			// Smallest across the SIGNALS on one backend: refused says 48,
			// concurrency says 32, and the strictest of them describes it.
			if r > 0 && (ref < 0 || r < ref) {
				ref = r
			}
			if ref < 0 {
				ref = 0 // saturated, but this signal named no level
			}
		}
		if ref < 0 {
			usable = append(usable, c)
			continue
		}
		saturated[c.URL] = ref
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

	// The available holders of the anchor, resolved once. Tier 1 routes to them,
	// and the soft limit decides between tier 1 and a split before the guard has
	// even chosen its reference.
	var pool []*registry.Backend
	if !a.pool.Empty() && !(shallow && p.cfg.Ladder == LadderStrict) {
		pool = make([]*registry.Backend, 0, len(usable))
		for i, c := range usable {
			if a.pool.Has(slots[i]) {
				pool = append(pool, c)
			}
		}
	}

	// stretch is non-empty only when the SOFT limit is what pushed this request
	// out of tier 1: every available holder is at or above it, yet all are below
	// the hard limit and could still take the request. Tier 2b brings it back to
	// them if nothing clears the guard.
	//
	// Deliberately not a signal. A signal answers "can this backend take more
	// work", and the answer here is yes — what the soft limit expresses is that
	// the router would rather it did not have to. Routing them through the same
	// mechanism would remove these backends from the usable set, and then a
	// split could land on a backend the soft limit had also passed, which is the
	// opposite of the point.
	var stretch []*registry.Backend
	if p.cfg.SoftNodeConcurrency > 0 && len(pool) > 0 {
		under := make([]*registry.Backend, 0, len(pool))
		for _, c := range pool {
			if c.Inflight() < p.cfg.SoftNodeConcurrency {
				under = append(under, c)
			}
		}
		if len(under) == 0 {
			p.m.SoftBlocked.Inc()
			stretch, pool = pool, nil
		} else {
			pool = under
		}
	}

	// The guard measures a candidate against THE THING BEING RELIEVED — the
	// saturated holders of this prefix — and nothing else. Among them the
	// busiest defines "as loaded as it": that is the load a copy is being made
	// to escape.
	guardRef := int64(-1)
	for url, ref := range saturated {
		if ref > guardRef && a.held.Has(p.tree.slotOrCreate(url)) {
			guardRef = ref
		}
	}
	if len(stretch) > 0 {
		// A soft-triggered split is escaping the SOFT limit, not the hard one:
		// the holders it would relieve are sitting just above soft and are not
		// saturated at all. Measuring against the hard limit would let the copy
		// land on a backend BUSIER than the holder it was made to escape, which
		// is a permanent duplicate bought for nothing.
		//
		// Deliberately the limit rather than the busiest holder's actual load,
		// matching how the concurrency signal names its own ceiling. It is the
		// conservative choice — it makes splits harder the further the holders
		// run past soft — and conservative is the intent here: this whole
		// mechanism exists to keep requests on backends that already have the KV.
		guardRef = p.cfg.SoftNodeConcurrency
	} else if guardRef <= 0 {
		// No saturated holder named a level. Either the configured limit is the
		// only capacity model available, or there is none.
		guardRef = p.cfg.NodeConcurrency
	}

	// No reference at all means the holders are ABSENT rather than full — they
	// left the fleet, or they are unhealthy. Nothing is being relieved, so
	// placing this prefix somewhere is relocation, not duplication, and there is
	// nothing for the guard to protect. It does not apply.
	//
	// The previous fallback here was the fleet's own minimum in-flight plus one,
	// which manufactured a threshold just under the least-loaded backend. In any
	// evenly-loaded fleet that is below every candidate, so the guard rejected
	// everything except a completely idle node — a 429 to the client with the
	// fleet at a third of capacity.
	threshold := math.Inf(1)
	if guardRef > 0 {
		threshold = float64(guardRef) * (1 - p.cfg.SplitGuard)
	}

	// Tier 1: a usable backend holds the deepest marked run, and is below the
	// soft limit if one is set.
	if len(pool) > 0 {
		rr.PolicyState = markDecision
		p.m.Decision("cache")
		p.m.AnchorBlocks.Observe(float64(a.anchorBlocks))
		p.m.PoolSize.Observe(float64(len(pool)))
		// A TOKEN fraction, weighted by each block's size, because the number it
		// is read against — CacheObservedFraction, from the backend's own
		// usage.prompt_tokens_details.cached_tokens — is one too. Blocks here
		// are variable-sized: a 180-byte conversational turn and a 1024-byte
		// system chunk are both one block, so on agentic traffic an unweighted
		// count and a token share differ severalfold and the pair says nothing.
		p.m.PredictedFraction.Observe(kvcache.Cover(rr.Units, a.anchorBlocks).TokenFraction())
		return p.fallback.Select(ctx, pool, rr)
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

		// Tier 2b: STRETCH. The soft limit said "rather not" and the guard said
		// "not worth a copy", and both are true at once — so the request goes
		// back to the holders it was being steered away from, onto the least
		// loaded of them, and queues there until the hard limit.
		//
		// It runs BEFORE the transient fallback deliberately. Both resolve the
		// same guard block and pay opposite prices: a stretch pays queueing and
		// keeps the cache hit, a transient serve keeps the queue short and pays
		// a full prefill on a backend holding none of this prefix. Anton's
		// intent for this mechanism is the cache side of that trade, so when
		// both are configured the one that preserves the KV wins. The two also
		// confound each other in measurement, which is why the evaluation runs
		// with transient off.
		//
		// The mark is the ordinary one: this backend already holds the prefix,
		// so recording the request extends a run that exists rather than
		// creating a copy. Nothing is duplicated, which is also why the guard
		// does not apply among holders — there is nothing here for it to
		// protect against.
		if len(stretch) > 0 {
			rr.PolicyState = markDecision
			p.m.Decision("stretch")
			p.m.Stretches.Inc()
			p.m.AnchorBlocks.Observe(float64(a.anchorBlocks))
			b, err := p.fallback.Select(ctx, stretch, rr)
			if err == nil && b != nil {
				// Read before the lease is taken, so this is the queue the
				// request is about to join rather than the one it created.
				p.m.StretchInflight.Observe(float64(b.Inflight()))
			}
			return b, err
		}

		// Tier 2c: TRANSIENT FALLBACK. Nothing was far enough below the holders
		// to be worth a permanent copy — but a backend that is merely close to
		// them can still serve this one request, provided we do not pretend
		// afterwards that it holds the prefix.
		//
		// The distinction is the whole point. What the split guard is protecting
		// is not capacity, it is the TREE: every split adds a holder for good,
		// and a fleet that splits freely converges on everyone holding
		// everything. Serving without marking costs nothing permanent, so it can
		// be allowed on a much looser threshold — the same arithmetic against
		// the same reference, just a smaller margin.
		//
		// At the shipped pair (guard 0.20, transient 0.05) a backend under 80%
		// of the holders' load takes the prefix and keeps it; one between 80%
		// and 95% serves the request and is forgotten; above 95% it is genuinely
		// as loaded as what it would relieve, and the request is refused.
		if p.cfg.TransientFallback > 0 && guardRef > 0 {
			limit := float64(guardRef) * (1 - p.cfg.TransientFallback)
			transient := make([]*registry.Backend, 0, len(usable))
			for i, c := range usable {
				if a.held.Has(slots[i]) {
					continue // a holder, and unavailable — that is why we are here
				}
				if float64(c.Inflight()) < limit {
					transient = append(transient, c)
				}
			}
			if len(transient) > 0 {
				rr.PolicyState = noMarkDecision
				p.m.Decision("overflow")
				p.m.Overflows.Inc()
				return p.fallback.Select(ctx, transient, rr)
			}
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
