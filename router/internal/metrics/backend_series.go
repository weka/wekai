package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/weka/wekai/router/internal/clock"
)

// BackendSeriesRetention is how long a backend may stay unhealthy or removed
// before its per-backend series stop being exported.
//
// Per-backend series used to live forever: a GaugeVec child is never deleted
// on its own, so every pod discovery ever churned through left a router_backend_*
// series behind, and a dashboard summing or listing backends showed ghosts.
const BackendSeriesRetention = 15 * time.Minute

// SeriesBackend is what the per-backend collector needs from a backend.
// Satisfied by *registry.Backend; an interface so this package stays free of
// the registry, which already imports it indirectly through lease and health.
type SeriesBackend interface {
	Label() string
	Inflight() int64
	Requests() uint64
	// Down reports unhealthy or being removed.
	Down() bool
}

// BackendSeries exports router_backend_inflight and
// router_backend_requests_total, and decides when a backend's series end.
//
// It is a Collector rather than a GaugeVec/CounterVec pair because deletion is
// the hard part. With vectors, something has to call DeleteLabelValues at the
// right moment, and the request path holds a resolved child (R5): an Inc after
// the delete resurrects the series, and a Dec after a Set(0) drives the gauge
// negative. Here the request path only touches the Backend's own atomics, and
// the series are derived from them at scrape time, so neither can happen.
//
// While a backend is down (unhealthy, or removed from discovery) in-flight is
// reported as 0 — whatever it was holding is not going to complete usefully —
// and the request counter keeps its last value. After BackendSeriesRetention
// of continuous downtime both series are omitted. If the backend comes back
// they resume with its real values; a backend re-added after it was fully
// dropped is a new object and its counter restarts from 0, which Prometheus
// reads as an ordinary counter reset.
//
// "Continuous downtime" is measured from the first scrape that saw the backend
// down, so it is only as precise as the scrape interval.
type BackendSeries struct {
	clk       clock.Clock
	retention time.Duration

	inflightDesc, requestsDesc *prometheus.Desc

	mu      sync.Mutex
	sources []func() []SeriesBackend
	tracked map[trackKey]*tracked
}

// trackKey is per source as well as per URL: two pools can front the same
// endpoint, and each owns its own Backend (and counters).
type trackKey struct {
	src   int
	label string
}

type tracked struct {
	b SeriesBackend // last seen; retained after removal so its counter can still be read
	// downSince is when this backend was first seen down; zero while it is up.
	downSince time.Time
}

// NewBackendSeries builds the collector. A non-positive retention means
// BackendSeriesRetention.
func NewBackendSeries(clk clock.Clock, retention time.Duration) *BackendSeries {
	if clk == nil {
		clk = clock.Real{}
	}
	if retention <= 0 {
		retention = BackendSeriesRetention
	}
	return &BackendSeries{
		clk: clk, retention: retention,
		inflightDesc: prometheus.NewDesc("router_backend_inflight",
			"In-flight requests per backend, from the lease primitive — the only trusted load signal. "+
				"Reported as 0 while the backend is unhealthy or removed; the series is dropped after 15 minutes of that.",
			[]string{"backend"}, nil),
		requestsDesc: prometheus.NewDesc("router_backend_requests_total",
			"Requests routed to a backend (one per lease acquired; a retry counts on the backend it went to). "+
				"Kept while the backend is unhealthy or removed and dropped after 15 minutes of that; "+
				"a backend that returns after being dropped may restart from 0, a normal counter reset.",
			[]string{"backend"}, nil),
		tracked: map[trackKey]*tracked{},
	}
}

// BackendSeriesCollector is the process-wide instance on the metrics endpoint.
// Registry() is rebuilt per scrape, so the state it keeps has to live here.
var BackendSeriesCollector = NewBackendSeries(clock.Real{}, BackendSeriesRetention)

// SetSources replaces the live-backend sources, one per pool. Each is called
// once per scrape and must be cheap and safe for concurrent use.
func (c *BackendSeries) SetSources(srcs ...func() []SeriesBackend) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sources = srcs
	c.tracked = map[trackKey]*tracked{}
}

func (c *BackendSeries) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.inflightDesc
	ch <- c.requestsDesc
}

func (c *BackendSeries) Collect(ch chan<- prometheus.Metric) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.clk.Now()

	live := map[trackKey]SeriesBackend{}
	for i, src := range c.sources {
		for _, b := range src() {
			live[trackKey{i, b.Label()}] = b
		}
	}

	// Aggregated by label: a series is one (name, backend) pair however many
	// pools carry the endpoint.
	type sample struct {
		inflight int64
		requests uint64
	}
	out := map[string]*sample{}
	report := func(t *tracked, down bool) {
		if !t.downSince.IsZero() && now.Sub(t.downSince) >= c.retention {
			return
		}
		s := out[t.b.Label()]
		if s == nil {
			s = &sample{}
			out[t.b.Label()] = s
		}
		s.requests += t.b.Requests()
		if !down {
			s.inflight += t.b.Inflight()
		}
	}

	for k, b := range live {
		t := c.tracked[k]
		if t == nil || t.b != b {
			// New, or re-added as a fresh object after being dropped.
			t = &tracked{b: b}
			c.tracked[k] = t
		}
		down := b.Down()
		switch {
		case !down:
			t.downSince = time.Time{}
		case t.downSince.IsZero():
			t.downSince = now
		}
		report(t, down)
	}
	for k, t := range c.tracked {
		if _, ok := live[k]; ok {
			continue
		}
		// Removed from discovery: in no snapshot any more, kept only so its
		// last values can be reported for the retention window.
		if t.downSince.IsZero() {
			t.downSince = now
		}
		if now.Sub(t.downSince) >= c.retention {
			delete(c.tracked, k)
			continue
		}
		report(t, true)
	}

	for label, s := range out {
		ch <- prometheus.MustNewConstMetric(c.inflightDesc, prometheus.GaugeValue, float64(s.inflight), label)
		ch <- prometheus.MustNewConstMetric(c.requestsDesc, prometheus.CounterValue, float64(s.requests), label)
	}
}
