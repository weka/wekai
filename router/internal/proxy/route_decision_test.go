package proxy

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// fakeSelector is a minimal policy.Policy-shaped Selector for testing
// recordRouteDecision's classification without a real registry/candidate set.
type fakeSelector struct {
	name string
}

func (f fakeSelector) Name() string { return f.name }
func (f fakeSelector) Select(context.Context, []*registry.Backend, *policy.RoutingRequest) (*registry.Backend, error) {
	return nil, nil
}

// fakeCacheSelector additionally implements policy.Committer, the same
// structural test recordRouteDecision and gateway.go both use to identify a
// cache-affinity policy.
type fakeCacheSelector struct{ fakeSelector }

func (fakeCacheSelector) Commit(*registry.Backend, *policy.RoutingRequest) {}

func TestRecordRouteDecisionClassifiesByName(t *testing.T) {
	cases := []struct {
		sel  Selector
		want string
	}{
		{fakeSelector{name: "least-outstanding"}, "load"},
		{fakeSelector{name: "round-robin"}, "other"},
		{fakeSelector{name: "random"}, "other"},
	}
	for _, c := range cases {
		before := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues(c.want))
		recordRouteDecision(c.sel)
		after := testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues(c.want))
		if after != before+1 {
			t.Errorf("policy %q: RouteDecisions{%s} = %v, want %v", c.sel.Name(), c.want, after, before+1)
		}
	}
}

// A cache-affinity policy (implements policy.Committer) must NOT be
// classified here — it self-instruments per-decision from inside its own
// Select, since only it knows whether a given call was a cache hit or an
// internal fallback. Double-counting would corrupt the total.
func TestRecordRouteDecisionSkipsCommitterPolicies(t *testing.T) {
	sel := fakeCacheSelector{fakeSelector{name: "prefix-cache-candidates"}}
	before := snapshotAllRouteDecisions(t)
	recordRouteDecision(sel)
	after := snapshotAllRouteDecisions(t)
	if after != before {
		t.Errorf("RouteDecisions total changed from %v to %v; a Committer policy must self-instrument, not be classified by name", before, after)
	}
}

func snapshotAllRouteDecisions(t *testing.T) float64 {
	t.Helper()
	total := 0.0
	for _, kind := range []string{"cache", "load", "other"} {
		total += testutil.ToFloat64(metrics.RouteDecisions.WithLabelValues(kind))
	}
	return total
}
