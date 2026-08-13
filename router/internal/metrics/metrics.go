// Package metrics declares every Prometheus collector in the router, in one
// file, registered explicitly.
//
// One file and explicit registration are deliberate: it is what makes the
// dead-metric check in hack/ possible. v1 registered an entire `vllm_tokenizer_*`
// family for a subsystem the routing path never called, and fed its
// worker_load / max_load / min_load gauges from a counter that was structurally
// corrupt — so operators were shown numbers that looked authoritative and meant
// nothing (OBS-N1, OBS-N2).
//
// promauto is deliberately NOT used; it registers as a side effect of variable
// initialization, which defeats the check.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Label sets are closed enums so cardinality stays bounded (API-14).
var (
	// Requests.
	RequestsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_requests_total",
		Help: "Requests handled, by route class, dialect and status class.",
	}, []string{"route", "dialect", "status"})

	RequestDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_request_duration_seconds",
		Help:    "End-to-end request duration.",
		Buckets: durationBuckets,
	}, []string{"route"})

	TimeToFirstByte = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_ttft_seconds",
		Help:    "Time from request receipt to first response byte.",
		Buckets: durationBuckets,
	}, []string{"route"})

	// Backends.
	BackendInflight = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_backend_inflight",
		Help: "In-flight requests per backend, from the lease primitive — the only trusted load signal.",
	}, []string{"backend"})

	BackendHealth = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_backend_health",
		Help: "Backend health: 0 unknown, 1 healthy, 2 unhealthy.",
	}, []string{"backend"})

	BackendsTotal = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_backends_total",
		Help: "Backend count by state.",
	}, []string{"state"})

	CircuitState = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_circuit_state",
		Help: "Circuit breaker state per backend: 0 closed, 1 open, 2 half-open.",
	}, []string{"backend"})

	CircuitTransitions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_circuit_transitions_total",
		Help: "Circuit breaker state transitions.",
	}, []string{"backend", "from", "to"})

	// Routing.
	RoutingDecisionDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "router_routing_decision_duration_seconds",
		Help: "Time spent in policy Select only, so the NFR-2 budget is directly observable.",
		// Microsecond-scale: the budget is p99 <= 250us for load policies.
		Buckets: []float64{1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3},
	}, []string{"policy"})

	PolicySelections = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_policy_selections_total",
		Help: "Selections made, by policy and chosen backend.",
	}, []string{"policy", "backend"})

	PolicyFallbacks = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_policy_fallback_total",
		Help: "Times a policy declined and delegated to the fallback policy.",
	}, []string{"policy", "reason"})

	// RouteDecisions classifies every selection that reached an attempt by the
	// mechanism that actually chose the backend:
	//
	//   cache    — prefix affinity decided, whether alone or as the tie-break
	//              among several cache candidates.
	//   split    — every backend holding the prefix was saturated, so affinity
	//              was EXTENDED onto a backend outside the holder set, which
	//              then becomes a holder too (prefix-cache-split only).
	//   overflow — the split guard refused, and --transient-fallback-threshold
	//              served the request anyway on a backend it did NOT record as a
	//              holder. Reads 0 unless that flag is set. Watch it against
	//              router_cache_guard_rejects_total: the two are the same
	//              situation resolved and unresolved, and their ratio is what
	//              the threshold actually buys.
	//   load     — no prefix was marked anywhere, so the selector decided: a
	//              genuinely new prompt, or a route with no routable prefix.
	//   other    — unused; kept so the enum stays closed.
	//
	// A closed enum, so this stays cheap to keep forever (API-14). Consumers
	// must aggregate by label rather than enumerating members: the dashboard
	// panel that named cache/load/other individually silently dropped traffic
	// the day split and overflow were added.
	RouteDecisions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_route_decisions_total",
		Help: "Selections by decision mechanism: cache, split, overflow, load, or other.",
	}, []string{"pool", "decision"})

	// Fleet load. Cache policies already expose per-backend inflight via
	// BackendInflight; these three summarize it so a dashboard doesn't need a
	// PromQL aggregation just to see whether the fleet is balanced. Computed
	// over Available() backends only — the same set policies actually choose
	// among — so one dead/draining backend holding stale load can't skew it
	// (the same v1 defect class as CACHE-N4/LB-N7, just for a gauge instead of
	// a routing guard).
	WorkerLoadAvg = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "router_worker_load_avg",
		Help: "Average NormalizedLoad (in-flight / capacity) across available backends.",
	})

	WorkerLoadMax = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "router_worker_load_max",
		Help: "Maximum NormalizedLoad across available backends.",
	})

	WorkerLoadMin = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "router_worker_load_min",
		Help: "Minimum NormalizedLoad across available backends.",
	})

	// Upstream and reliability.
	UpstreamErrors = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_upstream_errors_total",
		Help: "Upstream failures by backend and kind.",
	}, []string{"backend", "kind"})

	RetriesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_retries_total",
		Help: "Retry ATTEMPTS by reason and outcome. Per attempt, not per request — a request that waits four times adds four. Use router_retry_wait_seconds_count for a per-request figure.",
	}, []string{"reason", "outcome"})

	// RetryWaitSeconds is the latency --retry-time-limit added, observed once
	// per request that waited at all, spanning the first capacity refusal to
	// the moment the request left the retry path.
	//
	// A histogram rather than a counter because its _count answers the question
	// RetriesTotal cannot: how many REQUESTS entered the retry path and how
	// many the budget rescued. Attempts and requests differ by however many
	// times each request went round, so "retried minus exhausted" is not a
	// count of anything. With this, "the budget saved N requests at a p50 cost
	// of X ms" is two queries against one series.
	//
	// It measures the whole path, not the sum of the sleeps: the re-decisions
	// between waits are latency the caller pays too, and a closed-loop
	// benchmark reads any added latency as lost throughput.
	RetryWaitSeconds = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_retry_wait_seconds",
		Help:    "Added latency per request that waited out a capacity refusal, by the refusal it last waited on and how the wait ended.",
		Buckets: prometheus.ExponentialBuckets(0.005, 2, 12), // 5ms .. ~10s, the shipped budget's range
	}, []string{"reason", "outcome"})

	StreamAborted = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_stream_aborted_total",
		Help: "Streams terminated before their dialect's terminal marker.",
	}, []string{"reason"})

	PanicsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_panics_total",
		Help: "Handler panics converted to a 500 by the recover boundary.",
	})

	// ClientDisconnects counts clients that hung up mid-response. Routine at any
	// scale — a load driver hitting its timeout produces a burst — and kept out of
	// PanicsTotal so that metric stays an alarm rather than background noise.
	ClientDisconnects = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_client_disconnects_total",
		Help: "Responses aborted because the client went away mid-stream.",
	})

	// LoadAccountingErrors must stay at zero. Any non-zero value means the
	// in-flight invariant is broken, and it is the canary release gate (LB-5).
	LoadAccountingErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_load_accounting_errors_total",
		Help: "Detected in-flight underflows. Non-zero is a bug, not a condition.",
	})

	// Cache-affinity routing. PredictedFraction is what the router believed;
	// ObservedFraction is what vLLM reported via
	// usage.prompt_tokens_details.cached_tokens. Emitting both is the only way to
	// know whether prediction is worth anything, since residency is approximated
	// rather than observed (RES-3).
	CachePredictedFraction = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_cache_predicted_fraction",
		Help:    "Predicted fraction of a request's prompt TOKENS already resident on the chosen backend. Token-weighted so it is comparable with router_cache_observed_fraction.",
		Buckets: fractionBuckets,
	}, []string{"pool"})

	// CachePrediction{Avg,Max,Min} summarize the predicted-hit-fraction spread
	// across a single request's queried candidates — not the chosen backend's
	// fraction alone (that's CachePredictedFraction above), but every
	// candidate that had a computable prediction. Read alongside
	// WorkerLoad{Avg,Max,Min}: correlating a high prediction spread against a
	// spike in worker_load_max is exactly how you'd tell whether cache
	// affinity is buying anything or just riding along with load.
	//
	CacheObservedFraction = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_observed_fraction",
		Help:    "Observed cached fraction from the worker's usage.prompt_tokens_details.cached_tokens.",
		Buckets: fractionBuckets,
	})

	CacheEntries = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_cache_entries",
		Help: "Nodes held in a backend's prefix model.",
	}, []string{"backend"})

	CacheTokens = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_cache_tokens",
		Help: "Estimated tokens held in a backend's prefix model.",
	}, []string{"backend"})

	// --- prefix-cache-split only, from here to CacheBlocksExpired ---

	// CacheSplits counts affinity being EXTENDED under saturation rather than
	// abandoned: every backend holding the prefix was at its cap, so a backend
	// outside the holder set was chosen and recorded as a new holder. A healthy
	// system splits during load peaks and then stops.
	CacheSplits = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_splits_total",
		Help: "Times the holder set for a prefix was extended onto a new backend under saturation.",
	}, []string{"pool"})

	// CacheOverflows counts idle capacity being used WITHOUT recording a
	// holder, because nothing cleared the split guard.
	//
	// Zero unless --transient-fallback-threshold is set. When it is, this and
	// CacheGuardRejects are the same situation with opposite outcomes — the
	// guard refused, and the fallback either found a backend inside its looser
	// margin or did not — so the pair reads as one number: how often the
	// threshold rescued a request that would otherwise have been a 429.
	CacheOverflows = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_overflows_total",
		Help: "Requests the split guard refused and --transient-fallback-threshold served anyway, without recording the backend as a prefix holder. The resolved half of the pair whose other half is router_cache_guard_rejects_total.",
	}, []string{"pool"})

	// CacheSoftBlocked counts decisions where every available holder of the
	// prefix was at or above --soft-node-concurrency: the moment the soft limit
	// exists to create, where the router would rather spread than pile on.
	//
	// The TRIGGER, not the outcome. What happens next is a split if one clears
	// the guard, a stretch if none does, and occasionally a transient serve, so
	// this reconciles as soft_blocked >= stretches and the gap is how often
	// spreading actually worked. Flat at zero means the soft limit is set too
	// high to bind; equal to the decision count means it is set too low and the
	// fleet lives above it.
	CacheSoftBlocked = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_soft_blocked_total",
		Help: "Decisions where every available holder of the prefix was at or above the soft concurrency limit. The trigger for a split or a stretch, not an outcome.",
	}, []string{"pool"})

	// CacheStretches counts requests loaded onto a holder ALREADY past the soft
	// limit because no backend cleared the split guard.
	//
	// The opposite trade from CacheOverflows, at the same moment. A transient
	// serve keeps the holder's queue short and pays a full prefill on a backend
	// with none of the KV; a stretch keeps the cache hit and pays queueing. Both
	// resolve a guard block, so overflows + stretches + guard_rejects accounts
	// for every one of them.
	CacheStretches = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_stretches_total",
		Help: "Requests kept on an existing prefix holder that was already past the soft concurrency limit, because no backend cleared the split guard.",
	}, []string{"pool"})

	// CacheStretchInflight is the chosen holder's in-flight at selection time,
	// on the stretch path only.
	//
	// It answers the question the counter cannot: whether the soft-to-hard band
	// is being entered lightly or is where the fleet actually lives. Piled
	// against the hard limit means soft is set too low — the router is paying
	// the queueing cost continuously rather than as relief.
	CacheStretchInflight = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_cache_stretch_inflight",
		Help:    "In-flight on the holder chosen by a stretch, at selection time. Shows how far into the soft-to-hard band the fleet is running.",
		Buckets: prometheus.LinearBuckets(0, 8, 16), // 0..120 requests, the observed per-backend range
	}, []string{"pool"})

	// SignalFired counts, per signal, how often it called a backend saturated.
	//
	// The router has one routing flow; what varies between deployments is which
	// signals are enabled. This is how you tell which one is actually driving
	// decisions. `refused` is the ultimate signal and always on — a backend's
	// own 429. `concurrency` and `imbalance` are opt-in early warnings, enabled
	// by --max-node-concurrency and --rebalance-ratio respectively; each firing
	// far more often than `refused` means it is predicting saturation the
	// backends do not actually have.
	SignalFired = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_signal_fired_total",
		Help: "Times a split signal judged a backend unable to take more work.",
	}, []string{"pool", "signal"})

	// CacheGuardRejects counts the 429s the split guard causes: every backend
	// holding the prefix was at its limit and no other backend was far enough
	// below it to be worth a duplicate copy. Idle capacity existed and was
	// deliberately left unused.
	//
	// This is the price of keeping router_cache_avg_copies near 1.0, and the
	// pair to watch together — a rise here with no fall there means the guard
	// is costing throughput without buying locality. Distinct from
	// SaturationRejects, which means zero idle slots fleet-wide.
	CacheGuardRejects = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_guard_rejects_total",
		Help: "Requests rejected with 429 because no backend cleared the split guard, though idle capacity existed.",
	}, []string{"pool"})

	// CacheShallowAnchors counts tier-1 decisions whose anchor was NOT the
	// deepest marked run: the request's own holders were all unavailable, so
	// affinity fell back to a shared ancestor and served a backend that did not
	// hold the specific prefix. The commit that follows marks that backend on
	// the WHOLE path, so this is the unguarded path by which holder sets grow —
	// the split guard never sees these, because tier 1 answered first.
	//
	// Read it against router_cache_splits_total: splits are guarded growth,
	// these are not.
	CacheShallowAnchors = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_shallow_anchors_total",
		Help: "Affinity decisions anchored on a shared ancestor because the deepest holders were unavailable.",
	}, []string{"pool"})

	// CacheShallowAnchorBlocks is the block depth those decisions gave up
	// (deepest marked depth minus anchor depth), i.e. the blocks each one
	// duplicates onto a new holder. Volume, where CacheShallowAnchors is count.
	CacheShallowAnchorBlocks = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_shallow_anchor_blocks_total",
		Help: "Blocks of prefix depth forgone by shallow-anchored affinity decisions, and thereby duplicated.",
	}, []string{"pool"})

	// CacheAvgCopies is the mean number of backends holding each block, and the
	// tripwire for the one hazard this design knowingly accepts: a run under
	// continuous traffic never reaches its idle TTL, and TTL is the only thing
	// that removes a holder, so holder sets on hot runs can only grow. Target
	// is ~1.0. Sustained drift upward means the same context is being
	// duplicated across GPUs.
	CacheAvgCopies = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_cache_avg_copies",
		Help: "Mean number of backends holding each cached block; 1.0 means no duplication.",
	}, []string{"pool"})

	// CacheAnchorBlocks is how deep, in blocks, the affinity match ran. It
	// replaces CachePredictedFraction as the affinity-strength signal for this
	// policy: a fraction of the whole request shrinks as a session grows even
	// though the backend still holds everything it has ever seen, which is
	// precisely the defect this policy exists to fix.
	CacheAnchorBlocks = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_cache_anchor_blocks",
		Help:    "Blocks matched at the anchor run when affinity chose the backend.",
		Buckets: blockBuckets,
	}, []string{"pool"})

	// CachePoolSize is how many backends held the anchor. Persistent large
	// values mean the fleet is converging on "everyone holds everything", which
	// is the observable form of the CacheAvgCopies hazard.
	CachePoolSize = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "router_cache_pool_size",
		Help:    "Number of backends holding the anchor run at selection time.",
		Buckets: poolBuckets,
	}, []string{"pool"})

	CacheTreeRuns = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_cache_tree_runs",
		Help: "Compressed runs in the shared prefix tree.",
	}, []string{"pool"})

	// CacheTailSet is the size of the eviction candidate set. Eviction is
	// tail-only, so this bounds the sweep's cost and is the evidence for
	// whether the TTL is set right.
	CacheTailSet = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "router_cache_tail_set",
		Help: "Runs currently eligible for TTL eviction (leaves of the shared prefix tree).",
	}, []string{"pool"})

	CacheBlocksExpired = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_cache_blocks_expired_total",
		Help: "Blocks released by TTL eviction of idle tails.",
	}, []string{"pool"})

	RequestsShed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_requests_shed_total",
		Help: "Requests rejected with 503 because the router was at its concurrency cap.",
	})

	// SaturationRejects counts the DISTINCT 429 shed: every healthy backend was
	// at its router-enforced --max-node-concurrency cap. Kept separate from
	// RequestsShed (the router's OWN body-limit 503) because they are different
	// conditions with different fixes — one says "the router itself is full",
	// the other says "every backend is".
	SaturationRejects = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_saturation_rejects_total",
		Help: "Requests rejected with 429 because every healthy backend was at its router-enforced max-node-concurrency cap.",
	})

	// BackendCapExceeded is the per-backend occurrence counter for the same
	// condition: incremented once per candidate-selection pass for each
	// backend excluded for being at or above max-node-concurrency, so an
	// operator can see WHICH backend is saturating, not just that the fleet as
	// a whole is.
	BackendCapExceeded = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_backend_cap_exceeded_total",
		Help: "Times a backend was excluded from candidate selection for being at or above --max-node-concurrency.",
	}, []string{"backend"})

	// Discovery.
	DiscoveryConflicts = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_discovery_conflicts_total",
		Help: "Discovered endpoints ignored because a static backend already claims the URL.",
	}, []string{"backend"})
)

var fractionBuckets = []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1}

var durationBuckets = []float64{
	0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 240,
}

// blockBuckets spans a shared system prompt (tens of blocks) through a
// long-running agentic session (hundreds), at ~256 estimated tokens per block.
var blockBuckets = []float64{0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}

// poolBuckets counts backends, so it is small and linear at the low end where
// the interesting difference between "pinned to one" and "spread over four"
// lives.
var poolBuckets = []float64{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}

// All returns every collector this package declares. The dead-metric check in
// hack/ walks this list, so a new collector must be added here to be registered
// — and must be referenced somewhere to pass.
func All() []prometheus.Collector {
	return []prometheus.Collector{
		RequestsTotal, RequestDuration, TimeToFirstByte,
		BackendInflight, BackendHealth, BackendsTotal,
		CircuitState, CircuitTransitions,
		RoutingDecisionDuration, PolicySelections, PolicyFallbacks, RouteDecisions,
		WorkerLoadAvg, WorkerLoadMax, WorkerLoadMin,
		UpstreamErrors, RetriesTotal, RetryWaitSeconds, StreamAborted, PanicsTotal, ClientDisconnects,
		LoadAccountingErrors, DiscoveryConflicts,
		CachePredictedFraction, CacheObservedFraction, CacheEntries, CacheTokens,
		CacheSplits, CacheOverflows, CacheSoftBlocked, CacheStretches, CacheStretchInflight,
		CacheAvgCopies, CacheAnchorBlocks,
		CacheShallowAnchors, CacheShallowAnchorBlocks, CacheGuardRejects, SignalFired,
		CachePoolSize, CacheTreeRuns, CacheTailSet, CacheBlocksExpired,
		RequestsShed, SaturationRejects, BackendCapExceeded, observedShadow,
	}
}

// Registry builds a registry containing exactly our collectors plus the Go and
// process defaults.
func Registry() *prometheus.Registry {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors()...)
	reg.MustRegister(All()...)
	warm()
	return reg
}

// The refusals --retry-time-limit waits out, named here rather than at the
// increment so the set that is warmed and the set that is emitted cannot drift.
const (
	// ReasonSaturated: every backend was called full by some signal. There was
	// no candidate, so --transient-fallback-threshold could not have applied
	// however it was set, and waiting is the only move left.
	ReasonSaturated = "capacity_saturated"
	// ReasonGuardBlocked: capacity existed and the split guard refused a
	// duplicate. The transient fallback runs inside that same decision, before
	// the error is returned, so this reason means the fallback already looked
	// and found nobody inside its margin — the threshold is too tight, or off.
	ReasonGuardBlocked = "capacity_guard_blocked"
)

// CapacityRetryReasons is that set as an enum, for warming and for tests that
// check the two stay in step.
var CapacityRetryReasons = []string{ReasonSaturated, ReasonGuardBlocked}

// warm creates every label combination a *Vec can produce from a closed enum,
// so the series exists at 0 from startup.
//
// A lazily-created series is absent until its first event, and absent is not
// zero — it is "this router does not report that". An operator reading a scrape
// with no router_retries_total cannot distinguish a retry budget that was never
// needed from one that was never wired up, and both readings are consistent
// with every other number on the page. That ambiguity has already cost a fleet
// experiment: three arms concluded "--retry-time-limit eliminated the guard
// storm" from a series that was missing rather than flat.
//
// Only closed enums are warmed. Open label sets — backend URLs, error kinds —
// stay lazy: there is no complete list to enumerate, and inventing one would
// fill the scrape with series for backends that do not exist.
func warm() {
	for _, r := range CapacityRetryReasons {
		for _, o := range []string{"retried", "exhausted"} {
			RetriesTotal.WithLabelValues(r, o)
		}
		for _, o := range []string{"satisfied", "expired", "abandoned"} {
			RetryWaitSeconds.WithLabelValues(r, o)
		}
	}
}

// Handler serves the metrics endpoint. It is mounted on its own listener, never
// on the inference mux (GW-13).
func Handler(reg *prometheus.Registry) http.Handler {
	return promhttp.HandlerFor(reg, promhttp.HandlerOpts{Registry: reg})
}

// StatusClass buckets a status code so the `status` label stays a closed enum
// rather than one series per code.
func StatusClass(code int) string {
	switch {
	case code >= 500:
		return "5xx"
	case code >= 400:
		return "4xx"
	case code >= 300:
		return "3xx"
	case code >= 200:
		return "2xx"
	default:
		return "1xx"
	}
}

// PoolMetrics is one pool's view of the per-pool collectors, with the `pool`
// label already applied.
//
// Resolved once when a pool is built, never per request — the same reason
// backend gauges are resolved at add time (R5). A label lookup on the routing
// path would put a map access and a mutex inside the NFR-2 budget for no
// reason.
//
// It exists because these numbers are meaningless summed across pools. Two
// pools front different models with unrelated KV caches; adding their
// avg_copies together produces a figure describing nothing, which is the same
// defect class as the dashboard panel that silently dropped decision tiers.
type PoolMetrics struct {
	Splits              prometheus.Counter
	Overflows           prometheus.Counter
	SoftBlocked         prometheus.Counter
	Stretches           prometheus.Counter
	GuardRejects        prometheus.Counter
	ShallowAnchors      prometheus.Counter
	ShallowAnchorBlocks prometheus.Counter
	BlocksExpired       prometheus.Counter

	AvgCopies prometheus.Gauge
	TreeRuns  prometheus.Gauge
	TailSet   prometheus.Gauge

	AnchorBlocks      prometheus.Observer
	PoolSize          prometheus.Observer
	PredictedFraction prometheus.Observer
	StretchInflight   prometheus.Observer

	// decisions and signals stay vectors: their second label varies per event.
	decisions *prometheus.CounterVec
	signals   *prometheus.CounterVec
}

// DecisionTiers is the closed enum of routing decisions, and SplitSignals of
// the signals that judge a backend full. Exported so a dashboard or a scrape
// check can enumerate what it should find.
var (
	DecisionTiers = []string{"cache", "split", "overflow", "stretch", "load"}
	SplitSignals  = []string{"refused", "concurrency", "imbalance"}
)

// ForPool resolves every per-pool collector for name.
//
// Every per-pool series is created here, at 0, including the two that carry a
// second label — see warm() for why absent and zero must not be confused. A
// signal that is configured off still reports 0 rather than vanishing: "it
// never fired" is the true statement either way, and a reader chasing a missing
// series learns nothing from its absence.
func ForPool(name string) *PoolMetrics {
	lbl := prometheus.Labels{"pool": name}
	for _, t := range DecisionTiers {
		RouteDecisions.With(prometheus.Labels{"pool": name, "decision": t})
	}
	for _, s := range SplitSignals {
		SignalFired.With(prometheus.Labels{"pool": name, "signal": s})
	}
	return &PoolMetrics{
		Splits:              CacheSplits.With(lbl),
		Overflows:           CacheOverflows.With(lbl),
		SoftBlocked:         CacheSoftBlocked.With(lbl),
		Stretches:           CacheStretches.With(lbl),
		GuardRejects:        CacheGuardRejects.With(lbl),
		ShallowAnchors:      CacheShallowAnchors.With(lbl),
		ShallowAnchorBlocks: CacheShallowAnchorBlocks.With(lbl),
		BlocksExpired:       CacheBlocksExpired.With(lbl),
		AvgCopies:           CacheAvgCopies.With(lbl),
		TreeRuns:            CacheTreeRuns.With(lbl),
		TailSet:             CacheTailSet.With(lbl),
		AnchorBlocks:        CacheAnchorBlocks.With(lbl),
		PoolSize:            CachePoolSize.With(lbl),
		PredictedFraction:   CachePredictedFraction.With(lbl),
		StretchInflight:     CacheStretchInflight.With(lbl),
		decisions:           RouteDecisions.MustCurryWith(lbl),
		signals:             SignalFired.MustCurryWith(lbl),
	}
}

// Decision counts one routing decision by the tier that made it.
func (m *PoolMetrics) Decision(tier string) { m.decisions.WithLabelValues(tier).Inc() }

// Signal counts one signal judging a backend unable to take more work.
func (m *PoolMetrics) Signal(name string) { m.signals.WithLabelValues(name).Inc() }
