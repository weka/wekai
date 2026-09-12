package cli

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/jessevdk/go-flags"
)

func TestAdmissionBatchFlags(t *testing.T) {
	t.Setenv("BENCHMARK_ADMIT_COUNT", "0")
	t.Setenv("BENCHMARK_ADMIT_EVERY", "0s")
	for _, tc := range []struct {
		name    string
		args    []string
		count   int
		every   time.Duration
		wantErr bool
	}{
		{"default batch", []string{"--admit-every=2m"}, 0, 2 * time.Minute, false},
		{"32 every two minutes", []string{"--admit-count=32", "--admit-every=2m"}, 32, 2 * time.Minute, false},
		{"missing interval", []string{"--admit-count=32"}, 32, 0, true},
		{"negative batch", []string{"--admit-count=-1", "--admit-every=2m"}, -1, 2 * time.Minute, true},
		{"negative interval", []string{"--admit-every=-2m"}, 0, -2 * time.Minute, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var opts BenchmarkAutoOptions
			if _, err := flags.ParseArgs(&opts, tc.args); err != nil {
				t.Fatal(err)
			}
			c := BenchmarkAutoCommand{BenchmarkAutoOptions: &opts}
			if err := c.validateRealtime(&bytes.Buffer{}); (err != nil) != tc.wantErr {
				t.Fatalf("validation error = %v, want error %v", err, tc.wantErr)
			}
			if opts.AdmitCount != tc.count || opts.AdmitEvery != tc.every {
				t.Fatalf("got %d every %v, want %d every %v", opts.AdmitCount, opts.AdmitEvery, tc.count, tc.every)
			}
		})
	}
}

// The real-time flags check each other, because the failure they guard against
// is silent: a pinned pool caps the session count, the run halts there, and the
// number it reports looks exactly like a measured ceiling.

func TestRealtimeRefusesAPinnedSeriesPool(t *testing.T) {
	c := &BenchmarkAutoCommand{BenchmarkAutoOptions: &BenchmarkAutoOptions{}}
	c.ReplayRealtime = true
	c.Series = 256
	c.MaxSeries = 256

	err := c.validateRealtime(&bytes.Buffer{})
	if err == nil {
		t.Fatal("--replay-realtime with --series=256 was accepted; the count would stop climbing " +
			"at 256 and the run would report the pool size as the fleet's ceiling")
	}
	if !strings.Contains(err.Error(), "--series") || !strings.Contains(err.Error(), "--max-series") {
		t.Errorf("the error must name what to drop and what to use instead: %v", err)
	}
}

// TestRealtimeAcceptsASafetyCap: --max-series is the right way to bound the run,
// so it must not be caught by the same rule.
func TestRealtimeAcceptsASafetyCap(t *testing.T) {
	c := &BenchmarkAutoCommand{BenchmarkAutoOptions: &BenchmarkAutoOptions{}}
	c.ReplayRealtime = true
	c.MaxSeries = 100000

	if err := c.validateRealtime(&bytes.Buffer{}); err != nil {
		t.Errorf("--replay-realtime --max-series=100000 was rejected: %v", err)
	}
}

func TestGovernorFlagsWithoutRealtimeWarn(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*BenchmarkAutoCommand)
		want string
	}{
		{"skip-idle", func(c *BenchmarkAutoCommand) { c.ReplaySkipIdle = true }, "--replay-skip-idle"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &BenchmarkAutoCommand{BenchmarkAutoOptions: &BenchmarkAutoOptions{}}
			tc.set(c)
			var w bytes.Buffer
			if err := c.validateRealtime(&w); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !strings.Contains(w.String(), tc.want) {
				t.Errorf("no warning for %s without --replay-realtime; the run would quietly do "+
					"less than was asked for. got: %q", tc.want, w.String())
			}
		})
	}
}

// TestDefaultsAreSilent: a run that asks for none of this must print nothing
// and be rejected by nothing.
func TestDefaultsAreSilent(t *testing.T) {
	c := &BenchmarkAutoCommand{BenchmarkAutoOptions: &BenchmarkAutoOptions{}}
	c.Series = 256
	var w bytes.Buffer
	if err := c.validateRealtime(&w); err != nil {
		t.Errorf("a plain fixed-pool run was rejected: %v", err)
	}
	if w.Len() != 0 {
		t.Errorf("a plain fixed-pool run warned: %q", w.String())
	}
}
