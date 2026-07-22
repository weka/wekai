package cli

import (
	"os"
	"testing"

	flags "github.com/jessevdk/go-flags"
)

// TestRandomGateOrderFlagDefault locks the 2026-07-22 default flip: random
// gate order is the default; legacy FIFO stays reachable explicitly via
// --random-gate-order=false.
func TestRandomGateOrderFlagDefault(t *testing.T) {
	// Neutralize the env override so the tag default is what's under test.
	t.Setenv("BENCHMARK_RANDOM_GATE_ORDER", "unused")
	os.Unsetenv("BENCHMARK_RANDOM_GATE_ORDER")

	var def BenchmarkAutoOptions
	if _, err := flags.ParseArgs(&def, []string{}); err != nil {
		t.Fatalf("parse no-args: %v", err)
	}
	if def.RandomGateOrder != "true" {
		t.Fatalf("no flags: RandomGateOrder = %q, must default to true (random)", def.RandomGateOrder)
	}

	var fifo BenchmarkAutoOptions
	if _, err := flags.ParseArgs(&fifo, []string{"--random-gate-order=false"}); err != nil {
		t.Fatalf("parse =false: %v", err)
	}
	if fifo.RandomGateOrder != "false" {
		t.Fatalf("--random-gate-order=false parsed as %q, must select legacy FIFO", fifo.RandomGateOrder)
	}

	var bare BenchmarkAutoOptions
	if _, err := flags.ParseArgs(&bare, []string{"--random-gate-order"}); err != nil {
		t.Fatalf("parse bare flag: %v", err)
	}
	if bare.RandomGateOrder != "true" {
		t.Fatalf("bare --random-gate-order parsed as %q, must select random", bare.RandomGateOrder)
	}
}
