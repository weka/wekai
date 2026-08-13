package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// Absent is not zero. A scrape with no router_retries_total is consistent both
// with a retry budget that was never needed and with one that was never wired
// up, and nothing else on the page separates them — which is how three fleet
// arms came to report that --retry-time-limit had eliminated a guard storm it
// had in fact never been asked about.
//
// So every closed-enum series must be present from startup. These tests assert
// presence, not value: zero is what a fresh process reports, but presence is
// the property that makes a zero readable at all.

// seriesLabels collects the label sets a metric family exposes in reg.
func seriesLabels(t *testing.T, reg *prometheus.Registry, name string) []map[string]string {
	t.Helper()
	fams, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	var out []map[string]string
	for _, f := range fams {
		if f.GetName() != name {
			continue
		}
		for _, m := range f.Metric {
			lbl := map[string]string{}
			for _, p := range m.Label {
				lbl[p.GetName()] = p.GetValue()
			}
			out = append(out, lbl)
		}
	}
	return out
}

func hasLabels(got []map[string]string, want map[string]string) bool {
	for _, g := range got {
		ok := true
		for k, v := range want {
			if g[k] != v {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func TestCapacityRetrySeriesExistBeforeAnyRetry(t *testing.T) {
	reg := Registry() // nothing has been served; this is a router at startup

	for _, r := range CapacityRetryReasons {
		for _, o := range []string{"retried", "exhausted"} {
			want := map[string]string{"reason": r, "outcome": o}
			if !hasLabels(seriesLabels(t, reg, "router_retries_total"), want) {
				t.Errorf("router_retries_total{reason=%q,outcome=%q} is absent at startup; an "+
					"operator cannot tell a budget that never fired from one that is not wired up", r, o)
			}
		}
		for _, o := range []string{"satisfied", "expired", "abandoned"} {
			want := map[string]string{"reason": r, "outcome": o}
			if !hasLabels(seriesLabels(t, reg, "router_retry_wait_seconds"), want) {
				t.Errorf("router_retry_wait_seconds{reason=%q,outcome=%q} is absent at startup", r, o)
			}
		}
	}
}

// TestPerPoolEnumSeriesExistBeforeAnyTraffic covers the two per-pool vectors
// that carry a second label. The rest of the pool's collectors are already
// resolved by ForPool's own .With calls, which is what made the asymmetry
// confusing: cache_overflows_total read a real zero while the decision label
// naming the very same event read nothing.
func TestPerPoolEnumSeriesExistBeforeAnyTraffic(t *testing.T) {
	reg := Registry()
	const pool = "warm-test-pool"
	ForPool(pool)

	for _, tier := range DecisionTiers {
		want := map[string]string{"pool": pool, "decision": tier}
		if !hasLabels(seriesLabels(t, reg, "router_route_decisions_total"), want) {
			t.Errorf("router_route_decisions_total{pool=%q,decision=%q} is absent before traffic", pool, tier)
		}
	}
	for _, sig := range SplitSignals {
		want := map[string]string{"pool": pool, "signal": sig}
		if !hasLabels(seriesLabels(t, reg, "router_signal_fired_total"), want) {
			t.Errorf("router_signal_fired_total{pool=%q,signal=%q} is absent before traffic; a signal "+
				"configured off still reports 0 — 'it never fired' is true either way, and a reader "+
				"chasing a missing series learns nothing from its absence", pool, sig)
		}
	}
}

// TestWarmedSeriesStartAtZero: presence is worth nothing if the warm-up itself
// counts as an event.
func TestWarmedSeriesStartAtZero(t *testing.T) {
	reg := Registry()
	const pool = "zero-test-pool"
	ForPool(pool)

	fams, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, f := range fams {
		if f.GetName() != "router_route_decisions_total" {
			continue
		}
		for _, m := range f.Metric {
			if !hasLabels([]map[string]string{labelsOf(m)}, map[string]string{"pool": pool}) {
				continue
			}
			if v := m.GetCounter().GetValue(); v != 0 {
				t.Errorf("a freshly warmed series reads %v, want 0: warming must create the series, "+
					"not record an event", v)
			}
		}
	}
}

func labelsOf(m *dto.Metric) map[string]string {
	lbl := map[string]string{}
	for _, p := range m.Label {
		lbl[p.GetName()] = p.GetValue()
	}
	return lbl
}
