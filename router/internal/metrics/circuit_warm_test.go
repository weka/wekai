package metrics

import (
	"strings"
	"testing"
)

func TestCircuitTransitionsExistBeforeAnyTransition(t *testing.T) {
	reg := Registry()
	got := seriesLabels(t, reg, "router_circuit_transitions_total")
	if len(got) == 0 {
		t.Fatal("router_circuit_transitions_total is absent at startup: an operator cannot tell " +
			"a breaker that never opened from a metric that was never instrumented, and " +
			"circuit_state is a gauge so a transition between scrapes leaves no other trace")
	}
	var seen []string
	for _, l := range got {
		seen = append(seen, l["from"]+"->"+l["to"])
	}
	for _, want := range []string{"closed->open", "open->half-open", "half-open->closed"} {
		if !strings.Contains(strings.Join(seen, ","), want) {
			t.Errorf("transition %q not warmed; saw %v", want, seen)
		}
	}
}
