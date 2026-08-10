package cli

import (
	"testing"

	"github.com/jessevdk/go-flags"
)

// The imbalance signal is on by default at 0.5: a backend carrying more than
// twice the fleet minimum stops taking new work.
//
// Asserted through the real flag parser rather than by reading the struct tag,
// because a `default:` tag only takes effect when the command is parsed — a
// value set anywhere else would leave the help text saying one thing and the
// router doing another.
func TestRebalanceRatioDefaultsToHalf(t *testing.T) {
	var c RouterServeCommand
	p := flags.NewParser(&c, flags.Default)
	if _, err := p.ParseArgs([]string{"--backends", "http://a:8000"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.RebalanceRatio != 0.5 {
		t.Errorf("RebalanceRatio = %v, want 0.5", c.RebalanceRatio)
	}

	// And it must still be switchable off, which is what a deployment valuing
	// locality over evenness wants.
	var off RouterServeCommand
	p2 := flags.NewParser(&off, flags.Default)
	if _, err := p2.ParseArgs([]string{"--backends", "http://a:8000", "--rebalance-ratio", "0"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if off.RebalanceRatio != 0 {
		t.Errorf("RebalanceRatio = %v with --rebalance-ratio 0, want 0", off.RebalanceRatio)
	}
}
