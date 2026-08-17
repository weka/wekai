package benchmark

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// The governor's two moving parts, tested where they can be wrong: the window
// the gate reads, and the pacing that makes a replayed session keep its own
// think time.

func TestTTFTWindowIsTrailingWallTimeNotACount(t *testing.T) {
	base := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	w := newTTFTWindow(30 * time.Second)

	// Ten slow samples, then thirty seconds later ten fast ones. A count window
	// of 32 would still be carrying all twenty; the trailing window must have
	// dropped the first ten entirely.
	for i := range 10 {
		w.Observe(base.Add(time.Duration(i)*time.Second), 9*time.Second)
	}
	if mean, n := w.Mean(base.Add(9 * time.Second)); n != 10 || mean != 9*time.Second {
		t.Fatalf("mean=%v n=%d, want 9s over 10", mean, n)
	}

	later := base.Add(60 * time.Second)
	for i := range 10 {
		w.Observe(later.Add(time.Duration(i)*time.Millisecond), 100*time.Millisecond)
	}
	mean, n := w.Mean(later.Add(10 * time.Millisecond))
	if n != 10 {
		t.Errorf("window holds %d samples, want 10: the slow batch is older than the window and "+
			"must not still be counted", n)
	}
	if mean > 200*time.Millisecond {
		t.Errorf("mean=%v: stale samples are holding the gate shut after the fleet recovered", mean)
	}
}

// TestGateReopensWhenLatencyRecovers is the behaviour Anton specified: the gate
// stops admission above the limit and resumes below it, rather than latching.
func TestGateReopensWhenLatencyRecovers(t *testing.T) {
	base := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	w := newTTFTWindow(30 * time.Second)
	const limit = 5 * time.Second

	if !w.Open(base, limit) {
		t.Error("an empty window must open the gate; a run has to be able to start")
	}
	for i := range 20 {
		w.Observe(base.Add(time.Duration(i)*time.Second), 8*time.Second)
	}
	if w.Open(base.Add(20*time.Second), limit) {
		t.Error("gate open with a windowed mean of 8s against a 5s limit")
	}
	// Recovery: the slow samples age out and fast ones replace them.
	recov := base.Add(90 * time.Second)
	for i := range 20 {
		w.Observe(recov.Add(time.Duration(i)*time.Millisecond), 200*time.Millisecond)
	}
	if !w.Open(recov.Add(20*time.Millisecond), limit) {
		t.Error("gate still shut after latency recovered; it must resume admitting, not latch")
	}
}

// TestZeroTTFTIsNotASample. A request that never produced a first token — an
// error, a timeout — reports 0, and averaging that in would drag the mean down
// and hold the gate OPEN on a fleet that had stopped answering. The failure
// mode is admitting harder the worse things get.
func TestZeroTTFTIsNotASample(t *testing.T) {
	base := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	w := newTTFTWindow(30 * time.Second)
	for range 10 {
		w.Observe(base, 8*time.Second)
	}
	for range 90 {
		w.Observe(base, 0)
	}
	mean, n := w.Mean(base)
	if n != 10 {
		t.Errorf("window holds %d samples, want 10: a request with no first token is not evidence "+
			"about how long a first token takes", n)
	}
	if mean != 8*time.Second {
		t.Errorf("mean=%v, want 8s", mean)
	}
	if w.Open(base, 5*time.Second) {
		t.Error("failures pulled the mean under the limit and reopened the gate — the governor " +
			"would admit hardest exactly when the fleet is worst")
	}
}

func TestPacerKeepsTheCapturedGaps(t *testing.T) {
	origin := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	p := newSessionPacer("2026-05-12T08:50:06Z", origin, true, nil)

	for _, tc := range []struct {
		ts   string
		want time.Duration
	}{
		{"2026-05-12T08:50:06Z", 0},
		{"2026-05-12T08:50:13.5Z", 7500 * time.Millisecond},
		{"2026-05-12T08:55:06Z", 5 * time.Minute},
	} {
		got, ok := p.dueAt(tc.ts)
		if !ok {
			t.Fatalf("%s: not schedulable", tc.ts)
		}
		if d := got.Sub(origin); d != tc.want {
			t.Errorf("%s scheduled at +%v, want +%v", tc.ts, d, tc.want)
		}
	}
}

// TestPacerNeverSchedulesBackwards: a capture whose instances interleave can
// present a timestamp before the session's own start. Sending "in the past" is
// the right answer — send it now — and a negative offset would otherwise put it
// before the run began.
func TestPacerNeverSchedulesBackwards(t *testing.T) {
	origin := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	p := newSessionPacer("2026-05-12T08:50:06Z", origin, true, nil)
	got, ok := p.dueAt("2026-05-12T08:00:00Z")
	if !ok {
		t.Fatal("not schedulable")
	}
	if got.Before(origin) {
		t.Errorf("scheduled %v before the session started", origin.Sub(got))
	}
}

func TestPacerDisabledIsAPassthrough(t *testing.T) {
	p := newSessionPacer("2026-05-12T08:50:06Z", time.Now(), false, nil)
	if _, ok := p.dueAt("2026-05-12T09:50:06Z"); ok {
		t.Error("pacing is opt-in; with it off nothing may be scheduled into the future")
	}
	// And Wait must not block.
	done := make(chan struct{})
	go func() { p.Wait(context.Background(), "2026-05-12T09:50:06Z"); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Wait blocked with pacing disabled")
	}
}

// TestPacerUnparseableTimestampDoesNotStall. A capture with a malformed or
// absent ts must degrade to back-to-back, not to a session that never sends.
func TestPacerUnparseableTimestampDoesNotStall(t *testing.T) {
	p := newSessionPacer("2026-05-12T08:50:06Z", time.Now(), true, nil)
	if _, ok := p.dueAt("not-a-timestamp"); ok {
		t.Error("an unparseable timestamp was treated as schedulable")
	}
	if p2 := newSessionPacer("also-not-a-timestamp", time.Now(), true, nil); p2.enabled {
		t.Error("a session with no parseable start must fall back to unpaced rather than pinning " +
			"every request to a zero origin")
	}
}

// TestSkipClockOnlyJumpsWhenEverythingIsIdle. The skip exists so a replay does
// not spend its wall-clock budget on think time, and it is only safe because
// the condition requires an idle fleet: with nothing in flight, no request's
// timing can be distorted by moving the clock.
func TestSkipClockOnlyJumpsWhenEverythingIsIdle(t *testing.T) {
	c := newSkipClock(true)
	c.AddSession(2)

	future := c.Now().Add(time.Hour)
	c.enterWait(future)
	if got := c.Skew(); got != 0 {
		t.Errorf("skew moved to %v with one of two sessions still working", got)
	}

	// The second session parks too, but a request is still in flight.
	c.AddInflight(1)
	c.enterWait(future)
	if got := c.Skew(); got != 0 {
		t.Errorf("skew moved to %v with a request in flight; that request's latency would be "+
			"measured against a clock that jumped underneath it", got)
	}

	// The request completes: now everything is idle and the jump is safe.
	c.AddInflight(-1)
	if got := c.Skew(); got < 59*time.Minute {
		t.Errorf("skew=%v, want ~1h once the fleet is idle and both sessions are parked", got)
	}
}

func TestSkipClockDisabledNeverJumps(t *testing.T) {
	c := newSkipClock(false)
	c.AddSession(1)
	c.enterWait(c.Now().Add(time.Hour))
	if got := c.Skew(); got != 0 {
		t.Errorf("skew=%v with skipping disabled", got)
	}
}

// TestAdmitEveryOwnsSeriesScaling. Both the cache-target scaler and the
// admission governor call spawnSeries. With both live the run would ramp on two
// unrelated triggers at once — cache hit rate and TTFT — and its session count
// would be neither one's answer. Asserted against the source because the
// alternative is a full benchmark run to observe a double ramp.
func TestAdmitEveryOwnsSeriesScaling(t *testing.T) {
	src, err := os.ReadFile("auto.go")
	if err != nil {
		t.Fatalf("read auto.go: %v", err)
	}
	if !strings.Contains(string(src), "if !seriesDone && cfg.AdmitEvery <= 0 {") {
		t.Error("the cache-target series scaler is not gated on --admit-every being off; with " +
			"--admit-every set, two independent ramps would spawn series against the same run")
	}
}

// TestGovernorDefaultsAreOffAndUnsurprising: a run that does not ask for the
// governor must behave exactly as it did before it existed.
func TestGovernorDefaultsAreOff(t *testing.T) {
	var cfg AutoBenchmarkConfig
	if cfg.AdmitEvery != 0 {
		t.Errorf("AdmitEvery defaults to %v, want 0 (fixed series pool)", cfg.AdmitEvery)
	}
	if cfg.ReplayRealtime {
		t.Error("ReplayRealtime defaults on; pacing must be opt-in")
	}
	if cfg.ReplaySkipIdle {
		t.Error("ReplaySkipIdle defaults on")
	}
	// A zero window must still produce a usable gate rather than dividing by
	// nothing or holding shut forever.
	w := newTTFTWindow(cfg.TTFTWindow)
	if w.window != 30*time.Second {
		t.Errorf("zero window resolved to %v, want the 30s default", w.window)
	}
	if !w.Open(time.Now(), 5*time.Second) {
		t.Error("a fresh window must open the gate so a run can start")
	}
}

// Lateness is what says whether a real-time replay was real-time. A run whose
// sessions all fell an hour behind produces the same request count, token
// totals and cache curve as one that kept up, and differs only in how much
// captured conversation it actually covered — which nothing else records.
func TestPacerReportsHowLateItWas(t *testing.T) {
	origin := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	// Origin an hour in the past: every turn is already overdue.
	p := newSessionPacer("2026-05-12T08:50:06Z", origin.Add(-time.Hour), true, newSkipClock(false))
	p.origin = time.Now().Add(-time.Hour)

	late := p.Wait(context.Background(), "2026-05-12T08:50:06Z")
	if late < 50*time.Minute {
		t.Errorf("reported %v late on a turn due an hour ago; a fleet slower than the capture "+
			"makes every session fall behind and this is the only thing that says so", late)
	}
}

func TestPacerReportsZeroWhenItActuallyWaited(t *testing.T) {
	p := newSessionPacer("2026-05-12T08:50:06Z", time.Now(), true, newSkipClock(false))
	// Due 50ms out: Wait blocks and is therefore not late.
	if late := p.Wait(context.Background(), "2026-05-12T08:50:06.05Z"); late != 0 {
		t.Errorf("reported %v late on a turn it waited for", late)
	}
}

func TestPacingLagSummarySeparatesLateFromOnTime(t *testing.T) {
	var l pacingLag
	for range 7 {
		l.observe(0) // on time
	}
	l.observe(2 * time.Second)
	l.observe(30 * time.Second)
	l.observe(20 * time.Minute)

	s := l.summary()
	for _, want := range []string{"paced=10", "late=3", ">1s=3", ">10s=2", ">1m=1", ">10m=1"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary %q is missing %q", s, want)
		}
	}
	// The mean must be over LATE requests only — (2s + 30s + 20m) / 3 — not over
	// all ten. Averaging in the on-time zeros would report this run as 2m03s
	// behind when the requests that were late averaged 6m51s, which flatters
	// exactly the runs worth catching.
	if !strings.Contains(s, "mean_late=6m50.667s") {
		t.Errorf("summary %q: mean_late must average the late requests only, giving 6m50.667s", s)
	}
}

// TestPacingLagSaysNothingWhenNothingWasPaced: a flat-out run must not emit a
// line implying it was paced.
func TestPacingLagSaysNothingWhenNothingWasPaced(t *testing.T) {
	var l pacingLag
	if s := l.summary(); s != "" {
		t.Errorf("summary = %q on a run that paced nothing", s)
	}
}
