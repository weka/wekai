package cli

import (
	"testing"
	"time"

	"github.com/jessevdk/go-flags"
)

// The imbalance signal is OFF by default. A fleet where prefix affinity is
// working is supposed to look imbalanced — that is what affinity does — and
// repeated measurement on the b300 fleet found spreading it away buys no
// throughput while costing cache hits.
//
// Asserted through the real flag parser rather than by reading the struct tag,
// because a `default:` tag only takes effect when the command is parsed — a
// value set anywhere else would leave the help text saying one thing and the
// router doing another.
func TestRebalanceRatioDefaultsToOff(t *testing.T) {
	var c RouterServeCommand
	p := flags.NewParser(&c, flags.Default)
	if _, err := p.ParseArgs([]string{"--backends", "http://a:8000"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.RebalanceRatio != 0 {
		t.Errorf("RebalanceRatio = %v, want 0: the signal must not shrink the usable set unless "+
			"a deployment asks for it", c.RebalanceRatio)
	}

	// And it must still be switchable on, for a fleet that values evenness over
	// locality.
	var on RouterServeCommand
	p2 := flags.NewParser(&on, flags.Default)
	if _, err := p2.ParseArgs([]string{"--backends", "http://a:8000", "--rebalance-ratio", "0.5"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if on.RebalanceRatio != 0.5 {
		t.Errorf("RebalanceRatio = %v with --rebalance-ratio 0.5, want 0.5", on.RebalanceRatio)
	}
}

// The engine-neutral metrics flags are aliases of the vLLM-named ones: the
// disable flag and the interval work either way, and names combine.
func TestRouterBackendMetricsFlagAliases(t *testing.T) {
	var c RouterServeCommand
	p := flags.NewParser(&c, flags.Default)
	if _, err := p.ParseArgs([]string{"--backends", "http://a:8000", "--no-backend-metrics",
		"--backend-metrics-interval", "5s", "--backend-metrics-name", "sglang:x_total", "--vllm-metrics-name", "vllm:y_total"}); err != nil {
		t.Fatal(err)
	}
	if !c.NoBackendMetrics || c.metricsInterval() != 5*time.Second {
		t.Errorf("alias flags not honoured: no=%v interval=%v", c.NoBackendMetrics, c.metricsInterval())
	}
	if len(c.VLLMMetricsNames) != 1 || len(c.BackendMetricsNames) != 1 {
		t.Errorf("names = %v / %v", c.VLLMMetricsNames, c.BackendMetricsNames)
	}
	var d RouterServeCommand
	p2 := flags.NewParser(&d, flags.Default)
	if _, err := p2.ParseArgs([]string{"--backends", "http://a:8000"}); err != nil {
		t.Fatal(err)
	}
	if d.metricsInterval() != 30*time.Second {
		t.Errorf("default interval = %v, want 30s", d.metricsInterval())
	}
}
