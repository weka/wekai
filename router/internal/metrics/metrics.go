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
	// mechanism that actually chose the backend: "cache" (prefix affinity
	// decided, whether alone or as the tie-break among several cache
	// candidates), "load" (least-outstanding decided — either because that is
	// the configured policy, or because a cache policy declined and fell back),
	// or "other" (round-robin/random). A closed three-value enum, so this is
	// cheap to keep forever (API-14).
	RouteDecisions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "router_route_decisions_total",
		Help: "Selections classified by decision mechanism: cache, load, or other.",
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
	// Histograms, not gauges: a gauge here would be overwritten by every
	// concurrent request's Observe, so a scrape would sample whichever
	// request happened to finish last — one random data point per scrape
	// interval, not a trend (this is exactly what v1's gauge version did,
	// and it read as noise at any real concurrency). _sum/_count let a query
	// compute a genuine rate()-windowed mean across every request in the
	// window instead of stroboscopically sampling one of them.
	CachePredictionAvg = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_prediction_avg",
		Help:    "Average predicted-hit fraction across a request's queried candidates.",
		Buckets: fractionBuckets,
	})

	CachePredictionMax = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_prediction_max",
		Help:    "Maximum predicted-hit fraction across a request's queried candidates.",
		Buckets: fractionBuckets,
	})

	CachePredictionMin = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "router_cache_prediction_min",
		Help:    "Minimum predicted-hit fraction across a request's queried candidates.",
		Buckets: fractionBuckets,
	})

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
		CachePredictionAvg, CachePredictionMax, CachePredictionMin,
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
