package cli

import (
	"testing"

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
