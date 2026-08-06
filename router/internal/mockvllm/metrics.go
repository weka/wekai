package mockvllm

import (
	"github.com/prometheus/client_golang/prometheus"
)

// collectors declares this server's Prometheus surface, named to match real
// vLLM's own metric names where a direct analog exists
// (vllm:num_requests_running, vllm:gpu_cache_usage_perc,
// vllm:prompt_tokens_total, vllm:generation_tokens_total) plus two cache
// counters in vLLM's naming style that recent vLLM versions also expose
// (vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total).
//
// NOTE for anyone wiring a consumer against this: as of this writing nothing
// in this repo scrapes a backend's own /metrics — the router's load signal is
// its own in-flight lease count (router/internal/registry) and its cache
// prediction comes from its own kvcache trie, not from asking the worker. This
// endpoint exists for operator visibility and for a future consumer, not
// because one exists today. Real vLLM's exact metric names/labels have also
// drifted across versions, so treat these as "close enough for a mock", not a
// pinned contract.
type collectors struct {
	requestsTotal     *prometheus.CounterVec
	promptTokensTotal prometheus.Counter
	genTokensTotal    prometheus.Counter
	cacheQueriesTotal prometheus.Counter
	cacheHitsTotal    prometheus.Counter
	// numRequestsRunning and cacheUsagePerc are GaugeFuncs, not Gauges: they
	// must reflect the engine's state AT SCRAPE TIME, not a value frozen from
	// whichever request last called observe(). A plain Gauge.Set() here would
	// go stale the instant the request that set it releases its slot — which,
	// for a fast request, can be before the scrape that reads it.
	numRequestsRunning prometheus.GaugeFunc
	cacheUsagePerc     prometheus.GaugeFunc
}

func newCollectors(e *Engine) *collectors {
	return &collectors{
		requestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vllm:request_success_total",
			Help: "Requests completed, by outcome.",
		}, []string{"status"}),
		numRequestsRunning: prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name: "vllm:num_requests_running",
			Help: "Requests currently admitted (occupying a concurrency slot), read live at scrape time.",
		}, func() float64 { return float64(e.Stats().Inflight) }),
		cacheUsagePerc: prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name: "vllm:gpu_cache_usage_perc",
			Help: "Fraction of the configured block capacity currently in use, read live at scrape time.",
		}, e.FillFraction),
		promptTokensTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vllm:prompt_tokens_total",
			Help: "Estimated prompt tokens processed (cached + uncached).",
		}),
		genTokensTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vllm:generation_tokens_total",
			Help: "Synthetic completion tokens generated.",
		}),
		cacheQueriesTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vllm:prefix_cache_queries_total",
			Help: "Estimated prompt tokens queried against the prefix cache.",
		}),
		cacheHitsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vllm:prefix_cache_hits_total",
			Help: "Estimated prompt tokens served from the prefix cache.",
		}),
	}
}

func (c *collectors) register(reg *prometheus.Registry) {
	reg.MustRegister(
		c.requestsTotal, c.numRequestsRunning, c.cacheUsagePerc,
		c.promptTokensTotal, c.genTokensTotal,
		c.cacheQueriesTotal, c.cacheHitsTotal,
	)
}

// observe folds one completed request's outcome into the counters. status is
// "success" or "rejected" (429). The two live gauges are NOT touched here —
// see the GaugeFunc comment above.
func (c *collectors) observe(status string, cached, total, generated int) {
	c.requestsTotal.WithLabelValues(status).Inc()
	if status == "success" {
		c.promptTokensTotal.Add(float64(total))
		c.genTokensTotal.Add(float64(generated))
		c.cacheQueriesTotal.Add(float64(total))
		c.cacheHitsTotal.Add(float64(cached))
	}
}
