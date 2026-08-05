package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	promcollectors "github.com/prometheus/client_golang/prometheus/collectors"
)

// collectors returns the Go runtime and process collectors. Kept separate from
// All() so the dead-metric check only reasons about collectors we declare.
func collectors() []prometheus.Collector {
	return []prometheus.Collector{
		promcollectors.NewGoCollector(),
		promcollectors.NewProcessCollector(promcollectors.ProcessCollectorOpts{}),
	}
}
