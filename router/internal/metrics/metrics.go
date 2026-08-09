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
	//   overflow — retired with the serve-anyway ladder; reads 0.
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
	}, []string{"decision"})

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
		Help: "Retry attempts by reason and outcome.",
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
	CachePredictedFraction = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_predicted_fraction",
		Help:    "Predicted fraction of a request's prefix already resident on the chosen backend.",
		Buckets: fractionBuckets,
	})

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
	CacheSplits = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_cache_splits_total",
		Help: "Times the holder set for a prefix was extended onto a new backend under saturation.",
	})

	// CacheOverflows counts idle capacity being used WITHOUT recording a
	// holder, because nothing cleared the split guard. This is the capacity the
	// reference simulator would instead have rejected with a 429 while nodes
	// sat idle; the difference between this counter and zero is the reason
	// premature rejections do not happen here.
	CacheOverflows = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_cache_overflows_total",
		Help: "Requests routed to idle capacity without marking the backend as a prefix holder.",
	})

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
	}, []string{"signal"})

	// CacheGuardRejects counts the 429s the split guard causes: every backend
	// holding the prefix was at its limit and no other backend was far enough
	// below it to be worth a duplicate copy. Idle capacity existed and was
	// deliberately left unused.
	//
	// This is the price of keeping router_cache_avg_copies near 1.0, and the
	// pair to watch together — a rise here with no fall there means the guard
	// is costing throughput without buying locality. Distinct from
	// SaturationRejects, which means zero idle slots fleet-wide.
	CacheGuardRejects = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_cache_guard_rejects_total",
		Help: "Requests rejected with 429 because no backend cleared the split guard, though idle capacity existed.",
	})

	// CacheShallowAnchors counts tier-1 decisions whose anchor was NOT the
	// deepest marked run: the request's own holders were all unavailable, so
	// affinity fell back to a shared ancestor and served a backend that did not
	// hold the specific prefix. The commit that follows marks that backend on
	// the WHOLE path, so this is the unguarded path by which holder sets grow —
	// the split guard never sees these, because tier 1 answered first.
	//
	// Read it against router_cache_splits_total: splits are guarded growth,
	// these are not.
	CacheShallowAnchors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_cache_shallow_anchors_total",
		Help: "Affinity decisions anchored on a shared ancestor because the deepest holders were unavailable.",
	})

	// CacheShallowAnchorBlocks is the block depth those decisions gave up
	// (deepest marked depth minus anchor depth), i.e. the blocks each one
	// duplicates onto a new holder. Volume, where CacheShallowAnchors is count.
	CacheShallowAnchorBlocks = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_cache_shallow_anchor_blocks_total",
		Help: "Blocks of prefix depth forgone by shallow-anchored affinity decisions, and thereby duplicated.",
	})

	// CacheAvgCopies is the mean number of backends holding each block, and the
	// tripwire for the one hazard this design knowingly accepts: a run under
	// continuous traffic never reaches its idle TTL, and TTL is the only thing
	// that removes a holder, so holder sets on hot runs can only grow. Target
	// is ~1.0. Sustained drift upward means the same context is being
	// duplicated across GPUs.
	CacheAvgCopies = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "router_cache_avg_copies",
		Help: "Mean number of backends holding each cached block; 1.0 means no duplication.",
	})

	// CacheAnchorBlocks is how deep, in blocks, the affinity match ran. It
	// replaces CachePredictedFraction as the affinity-strength signal for this
	// policy: a fraction of the whole request shrinks as a session grows even
	// though the backend still holds everything it has ever seen, which is
	// precisely the defect this policy exists to fix.
	CacheAnchorBlocks = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_anchor_blocks",
		Help:    "Blocks matched at the anchor run when affinity chose the backend.",
		Buckets: blockBuckets,
	})

	// CachePoolSize is how many backends held the anchor. Persistent large
	// values mean the fleet is converging on "everyone holds everything", which
	// is the observable form of the CacheAvgCopies hazard.
	CachePoolSize = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_pool_size",
		Help:    "Number of backends holding the anchor run at selection time.",
		Buckets: poolBuckets,
	})

	CacheTreeRuns = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "router_cache_tree_runs",
		Help: "Compressed runs in the shared prefix tree.",
	})

	// CacheTailSet is the size of the eviction candidate set. Eviction is
	// tail-only, so this bounds the sweep's cost and is the evidence for
	// whether the TTL is set right.
	CacheTailSet = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "router_cache_tail_set",
		Help: "Runs currently eligible for TTL eviction (leaves of the shared prefix tree).",
	})

	CacheBlocksExpired = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "router_cache_blocks_expired_total",
		Help: "Blocks released by TTL eviction of idle tails.",
	})

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
		UpstreamErrors, RetriesTotal, StreamAborted, PanicsTotal, ClientDisconnects,
		LoadAccountingErrors, DiscoveryConflicts,
		CachePredictedFraction, CacheObservedFraction, CacheEntries, CacheTokens,
		CacheSplits, CacheOverflows, CacheAvgCopies, CacheAnchorBlocks,
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
	return reg
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
