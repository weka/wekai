package metrics_test

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

type seriesRig struct {
	t       *testing.T
	clk     *clock.Fake
	reg     *registry.Registry
	col     *metrics.BackendSeries
	dropped chan string
}

func newSeriesRig(t *testing.T) *seriesRig {
	t.Helper()
	clk := clock.NewFake(time.Time{})
	r := &seriesRig{t: t, clk: clk, dropped: make(chan string, 4)}
	r.reg = registry.New(registry.Options{
		Clock:  clk,
		OnDrop: func(b *registry.Backend) { r.dropped <- b.URL },
	})
	r.col = metrics.NewBackendSeries(clk, metrics.BackendSeriesRetention)
	r.col.SetSources(func() []metrics.SeriesBackend {
		bs := r.reg.Snapshot().Backends
		out := make([]metrics.SeriesBackend, len(bs))
		for i, b := range bs {
			out[i] = b
		}
		return out
	})
	return r
}

func (r *seriesRig) add(url string) *registry.Backend {
	r.t.Helper()
	b, err := r.reg.Add(registry.Spec{URL: url})
	if err != nil {
		r.t.Fatal(err)
	}
	b.SetHealth(registry.Healthy)
	return b
}

// scrape returns inflight and requests_total for backend, and whether each
// series is present.
func (r *seriesRig) scrape(backend string) (inflight, requests float64, hasIn, hasReq bool) {
	r.t.Helper()
	reg := prometheus.NewPedanticRegistry()
	reg.MustRegister(r.col)
	fams, err := reg.Gather()
	if err != nil {
		r.t.Fatalf("gather: %v", err)
	}
	for _, f := range fams {
		for _, m := range f.Metric {
			if m.Label[0].GetValue() != backend {
				continue
			}
			switch f.GetName() {
			case "router_backend_inflight":
				inflight, hasIn = m.GetGauge().GetValue(), true
			case "router_backend_requests_total":
				requests, hasReq = m.GetCounter().GetValue(), true
			}
		}
	}
	return
}

func (r *seriesRig) want(backend string, inflight, requests float64, present bool) {
	r.t.Helper()
	in, req, hasIn, hasReq := r.scrape(backend)
	if hasIn != present || hasReq != present {
		r.t.Fatalf("%s: present inflight=%v requests=%v, want %v", backend, hasIn, hasReq, present)
	}
	if present && (in != inflight || req != requests) {
		r.t.Fatalf("%s: inflight=%v requests=%v, want %v/%v", backend, in, req, inflight, requests)
	}
}

const bk = "http://w0:8000"

func TestBackendSeriesHealthyReportsRealValues(t *testing.T) {
	r := newSeriesRig(t)
	b := r.add(bk)
	l1, l2 := lease.Acquire(b), lease.Acquire(b)
	r.want(bk, 2, 2, true)
	l1.Release()
	l2.Release()
	l2.Release() // idempotent: no second effect
	r.want(bk, 0, 2, true)
}

func TestBackendSeriesRetryCountsOnEachBackend(t *testing.T) {
	r := newSeriesRig(t)
	a, b := r.add(bk), r.add("http://w1:8000")
	lease.Acquire(a).Release()
	lease.Acquire(b).Release()
	r.want(bk, 0, 1, true)
	r.want("http://w1:8000", 0, 1, true)
}

func TestBackendSeriesUnhealthyZeroesInflightThenDrops(t *testing.T) {
	r := newSeriesRig(t)
	b := r.add(bk)
	l := lease.Acquire(b)
	r.want(bk, 1, 1, true)

	b.SetHealth(registry.Unhealthy)
	r.want(bk, 0, 1, true) // inflight forced to 0, requests kept
	r.clk.Advance(metrics.BackendSeriesRetention - time.Second)
	r.want(bk, 0, 1, true)

	// A release after the zeroing must not push anything negative: the series
	// is derived from the counter, not a gauge that was Set.
	l.Release()
	r.want(bk, 0, 1, true)

	r.clk.Advance(time.Second)
	r.want(bk, 0, 0, false)
}

func TestBackendSeriesRecoveryResumes(t *testing.T) {
	r := newSeriesRig(t)
	b := r.add(bk)
	lease.Acquire(b).Release()
	b.SetHealth(registry.Unhealthy)
	r.want(bk, 0, 1, true)
	r.clk.Advance(metrics.BackendSeriesRetention)
	r.want(bk, 0, 0, false)

	b.SetHealth(registry.Healthy)
	l := lease.Acquire(b)
	r.want(bk, 1, 2, true)
	l.Release()

	// The window restarts: a fresh failure is not already expired.
	b.SetHealth(registry.Unhealthy)
	r.want(bk, 0, 2, true)
	r.clk.Advance(metrics.BackendSeriesRetention - time.Second)
	r.want(bk, 0, 2, true)
}

func TestBackendSeriesBriefRecoveryResetsWindow(t *testing.T) {
	r := newSeriesRig(t)
	b := r.add(bk)
	b.SetHealth(registry.Unhealthy)
	r.want(bk, 0, 0, true)
	r.clk.Advance(10 * time.Minute)
	b.SetHealth(registry.Healthy)
	r.want(bk, 0, 0, true)
	b.SetHealth(registry.Unhealthy)
	r.want(bk, 0, 0, true)
	r.clk.Advance(10 * time.Minute) // 20 min since first down, 10 since the last
	r.want(bk, 0, 0, true)
}

func TestBackendSeriesRemovedThenDroppedThenReAdded(t *testing.T) {
	r := newSeriesRig(t)
	b := r.add(bk)
	lease.Acquire(b).Release()
	r.want(bk, 0, 1, true)

	if err := r.reg.Remove(bk); err != nil {
		t.Fatal(err)
	}
	<-r.dropped // unlinked: no longer in any snapshot
	r.want(bk, 0, 1, true)
	r.clk.Advance(metrics.BackendSeriesRetention - time.Second)
	r.want(bk, 0, 1, true)
	r.clk.Advance(time.Second)
	r.want(bk, 0, 0, false)

	// Re-added: a new backend, counter restarts, series resume.
	nb := r.add(bk)
	r.want(bk, 0, 0, true)
	l := lease.Acquire(nb)
	r.want(bk, 1, 1, true)
	l.Release()
}

func TestBackendSeriesRemovedBackendReportsZeroInflight(t *testing.T) {
	r := newSeriesRig(t)
	b := r.add(bk)
	l := lease.Acquire(b)
	if err := r.reg.Remove(bk); err != nil {
		t.Fatal(err)
	}
	// Draining with a request still attached: still in the snapshot, reported 0.
	r.want(bk, 0, 1, true)
	l.Release()
	r.want(bk, 0, 1, true)
}
