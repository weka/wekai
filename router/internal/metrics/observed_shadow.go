package metrics

import "github.com/prometheus/client_golang/prometheus"

// CacheObservedFractionSumForTest exposes the histogram as a Collector whose sum
// tests can read. Histograms have no direct value accessor, so a tiny gauge shadow
// is kept in step for assertion purposes only.
func CacheObservedFractionSumForTest() prometheus.Gauge { return observedShadow }

var observedShadow = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "router_cache_observed_fraction_last",
	Help: "Most recent observed cached fraction. Mirrors the histogram for assertions and dashboards.",
})

// SetObservedShadow mirrors the latest observation.
func SetObservedShadow(v float64) { observedShadow.Set(v) }
