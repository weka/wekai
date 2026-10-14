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

// TestTTFTRetireFlags: --ttft-retire defaults ON with a 15% margin under
// --replay-realtime, can be turned off, and the margin is validated.
func TestTTFTRetireFlags(t *testing.T) {
	t.Setenv("BENCHMARK_ADMIT_COUNT", "0")
	t.Setenv("BENCHMARK_ADMIT_EVERY", "0s")
	for _, tc := range []struct {
		name    string
		args    []string
		retire  bool
		margin  float64
		wantErr bool
	}{
		{"default on", []string{"--replay-realtime"}, true, 0.15, false},
		{"explicit off", []string{"--replay-realtime", "--ttft-retire=false"}, false, 0.15, false},
		{"explicit margin", []string{"--replay-realtime", "--ttft-retire-margin=0.3"}, true, 0.3, false},
		{"negative margin", []string{"--replay-realtime", "--ttft-retire-margin=-0.1"}, true, -0.1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var opts BenchmarkAutoOptions
			if _, err := flags.ParseArgs(&opts, tc.args); err != nil {
				t.Fatal(err)
			}
			c := BenchmarkAutoCommand{BenchmarkAutoOptions: &opts}
			c.applyRealtimeDefaults()
			if err := c.validateRealtime(&bytes.Buffer{}); (err != nil) != tc.wantErr {
				t.Fatalf("validation error = %v, want error %v", err, tc.wantErr)
			}
			if opts.Retire() != tc.retire || opts.TTFTRetireMargin != tc.margin {
				t.Fatalf("retire=%v margin=%v, want %v %v", opts.Retire(), opts.TTFTRetireMargin, tc.retire, tc.margin)
			}
		})
	}
}

// TestRealtimeIsACompleteRecipe: --replay-realtime alone gives the ramp
// (start 32, +16 every minute, uncapped), session reuse, a 10s TTFT limit and
// the timeline; explicit flags still win.
func TestRealtimeIsACompleteRecipe(t *testing.T) {
	t.Setenv("BENCHMARK_ADMIT_COUNT", "0")
	t.Setenv("BENCHMARK_ADMIT_EVERY", "0s")
	t.Setenv("BENCHMARK_START_SERIES", "0")
	t.Setenv("BENCHMARK_MAX_SERIES", "0")
	var opts BenchmarkAutoOptions
	if _, err := flags.ParseArgs(&opts, []string{"--replay-realtime"}); err != nil {
		t.Fatal(err)
	}
	c := BenchmarkAutoCommand{BenchmarkAutoOptions: &opts}
	c.applyRealtimeDefaults()
	if opts.StartSeries != 32 || opts.AdmitCount != 16 || opts.AdmitEvery != time.Minute || opts.MaxSeries != 0 ||
		opts.TTFTLimit != 10*time.Second || !opts.ReuseSessions() || !opts.Retire() || !opts.Timeline {
		t.Fatalf("defaults: start=%d admit=%d/%v max=%d ttft=%v reuse=%v retire=%v timeline=%v", opts.StartSeries, opts.AdmitCount,
			opts.AdmitEvery, opts.MaxSeries, opts.TTFTLimit, opts.ReuseSessions(), opts.Retire(), opts.Timeline)
	}
	var o2 BenchmarkAutoOptions
	if _, err := flags.ParseArgs(&o2, []string{"--replay-realtime", "--start-series=8", "--admit-count=4", "--admit-every=30s", "--max-series=100", "--replay-reuse-sessions=false"}); err != nil {
		t.Fatal(err)
	}
	c2 := BenchmarkAutoCommand{BenchmarkAutoOptions: &o2}
	c2.applyRealtimeDefaults()
	if o2.StartSeries != 8 || o2.AdmitCount != 4 || o2.AdmitEvery != 30*time.Second || o2.MaxSeries != 100 || o2.ReuseSessions() {
		t.Fatalf("explicit flags must win: %+v", o2)
	}
	var o3 BenchmarkAutoOptions
	if _, err := flags.ParseArgs(&o3, []string{"--start-series=8"}); err != nil {
		t.Fatal(err)
	}
	c3 := BenchmarkAutoCommand{BenchmarkAutoOptions: &o3}
	c3.applyRealtimeDefaults()
	if o3.AdmitEvery != 0 || o3.Timeline {
		t.Fatal("non-realtime runs must not pick up realtime defaults")
	}
}

func TestDecodeRateFlags(t *testing.T) {
	t.Setenv("BENCHMARK_ADMIT_EVERY", "0s")
	for _, tc := range []struct {
		args    []string
		wantErr bool
	}{
		{[]string{"--replay-realtime", "--decode-rate-min=20"}, false},
		{[]string{"--decode-rate-min=20"}, true},
		{[]string{"--replay-realtime", "--decode-rate-min=-1"}, true},
		{[]string{"--replay-realtime", "--decode-rate-min=20", "--decode-rate-window=0s"}, true},
	} {
		var opts BenchmarkAutoOptions
		if _, err := flags.ParseArgs(&opts, tc.args); err != nil {
			t.Fatal(err)
		}
		c := BenchmarkAutoCommand{BenchmarkAutoOptions: &opts}
		c.applyRealtimeDefaults()
		if err := c.validateRealtime(&bytes.Buffer{}); (err != nil) != tc.wantErr {
			t.Errorf("%v: err=%v want error %v", tc.args, err, tc.wantErr)
		}
	}
}
