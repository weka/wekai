package cli

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/weka/wekai/llm"
)

// Prometheus metrics for the router. Counters are user/session/model-tagged so
// operators can answer "who spent what on which model?" without scraping
// captures.
//
// Cardinality caveat: `session_id` is high-cardinality — each Claude Code
// session creates a fresh UUID. Long-running deployments with many sessions
// per day will accumulate many distinct label sets. Mitigations available
// later if it becomes a problem: drop the label, hash it, or aggregate at
// the Prometheus side via `sum by (user, model)`. For now we keep it
// because the user explicitly asked for per-session visibility.
var (
	routerRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_requests_total",
		Help: "Total proxied requests, labeled by user, session, model, status class and whether this was an inference endpoint.",
	}, []string{"user", "session_id", "model", "status", "inference"})

	routerInputTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_input_tokens_total",
		Help: "Total uncached input tokens consumed (does not include cache reads or cache writes; see the *_cache_* metrics).",
	}, []string{"user", "session_id", "model"})

	routerCacheReadTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_cache_read_tokens_total",
		Help: "Input tokens served from cache (billed at the cache-read rate).",
	}, []string{"user", "session_id", "model"})

	routerCacheCreationTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_cache_creation_tokens_total",
		Help: "Input tokens written to cache (billed at the cache-write rate).",
	}, []string{"user", "session_id", "model"})

	routerOutputTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_output_tokens_total",
		Help: "Output tokens produced by the upstream model.",
	}, []string{"user", "session_id", "model"})

	// Estimated USD cost via llm.LookupModelByIdentifier + llm.CalculateCost.
	// Increments by 0 (no-op) for models we don't have prices for — see
	// wekai_router_unknown_model_total for that signal.
	routerCostUSDTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_cost_usd_total",
		Help: "Total estimated USD cost (per llm.CalculateCost). 0 for unpriced models.",
	}, []string{"user", "session_id", "model"})

	// Counts requests with usage where the model id wasn't in the llm
	// registry, so cost is underreported. Use this to identify registry
	// gaps. No session_id/user labels — the value is per-model.
	routerUnknownModelTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "wekai_router_unknown_model_total",
		Help: "Requests whose model id wasn't in the llm registry (cost underreported).",
	}, []string{"model"})
)

// recordRequestMetric emits the request counter. Always called per proxied
// request, regardless of capture mode or response status. Empty user / session
// id labels are normalized to "unknown" so the metric is always emittable —
// otherwise a missing label would silently drop the sample.
func recordRequestMetric(user, sessionID, model string, status int, inference bool) {
	routerRequestsTotal.WithLabelValues(
		labelOrUnknown(user),
		labelOrUnknown(sessionID),
		labelOrUnknown(model),
		strconv.Itoa(status),
		strconv.FormatBool(inference),
	).Inc()
}

// recordTokenMetrics emits the four token counters from one parsed usage
// record plus the cost counter. Called only when usage was successfully
// extracted (i.e., for inference requests with a 2xx response and a
// parseable SSE/JSON body).
func recordTokenMetrics(user, sessionID, model string, inputTokens, cacheReadTokens, cacheCreationTokens, outputTokens int) {
	u := labelOrUnknown(user)
	s := labelOrUnknown(sessionID)
	m := labelOrUnknown(model)
	if inputTokens > 0 {
		routerInputTokensTotal.WithLabelValues(u, s, m).Add(float64(inputTokens))
	}
	if cacheReadTokens > 0 {
		routerCacheReadTokensTotal.WithLabelValues(u, s, m).Add(float64(cacheReadTokens))
	}
	if cacheCreationTokens > 0 {
		routerCacheCreationTokensTotal.WithLabelValues(u, s, m).Add(float64(cacheCreationTokens))
	}
	if outputTokens > 0 {
		routerOutputTokensTotal.WithLabelValues(u, s, m).Add(float64(outputTokens))
	}

	// Cost. Lookup by raw API id (handles [1m] flag and date suffix). If
	// not found, bump the unknown-model counter so operators can see which
	// model entries are missing from the registry.
	if model != "" && llm.LookupModelByIdentifier != nil {
		if info, ok := llm.LookupModelByIdentifier(model); ok {
			cost := llm.CalculateCost(info, llm.Usage{
				InputTokens:              inputTokens,
				CacheReadInputTokens:     cacheReadTokens,
				CacheCreationInputTokens: cacheCreationTokens,
				OutputTokens:             outputTokens,
			})
			if cost > 0 {
				routerCostUSDTotal.WithLabelValues(u, s, m).Add(cost)
			}
		} else {
			routerUnknownModelTotal.WithLabelValues(m).Inc()
		}
	}
}

func labelOrUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}
