package proxy

import (
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/circuit"
	"github.com/weka/wekai/router/internal/clock"
)

// halfOpenBreaker returns a breaker with exactly one half-open probe token
// admitted, mirroring the state Serve sees right after b.CB.Allow() succeeds
// during recovery.
func halfOpenBreaker(clk *clock.Fake) (b *circuit.Breaker, token bool) {
	cfg := circuit.DefaultConfig()
	cfg.MinRequests = 1
	cfg.FailureRate = 0.5
	cfg.OpenFor = 5 * time.Second
	cfg.HalfOpenMax = 1
	b = circuit.New(cfg, clk)
	b.Record(circuit.Failure, false)
	b.Record(circuit.Failure, false) // opens
	clk.Advance(6 * time.Second)     // past OpenFor: probes admissible again

	ok, token := b.Allow()
	if !ok || !token {
		panic("probe not admitted while setting up the test breaker")
	}
	return b, token
}

// recoverCall runs fn and returns whatever it panicked with, or nil.
func recoverCall(fn func()) (v any) {
	defer func() { v = recover() }()
	fn()
	return nil
}

// TestRunAttemptReleasesTokenOnAbortPanic is the regression test for the
// leak: rp.ServeHTTP raises http.ErrAbortHandler as a real panic whenever a
// client disconnects mid-stream, and Serve has no defer between Allow and
// Record. Without runAttempt's own recover, that panic would skip Record
// entirely and the one HalfOpenMax=1 token would never come back, denying
// every future probe.
func TestRunAttemptReleasesTokenOnAbortPanic(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b, token := halfOpenBreaker(clk)

	panicked := recoverCall(func() {
		runAttempt(b, token, func() attemptOut {
			panic(http.ErrAbortHandler)
		})
	})

	err, ok := panicked.(error)
	if !ok || !errors.Is(err, http.ErrAbortHandler) {
		t.Fatalf("panic value = %v, want http.ErrAbortHandler to propagate unchanged", panicked)
	}
	// The token must be back: a subsequent probe can be admitted again.
	if ok, tok := b.Allow(); !ok || !tok {
		t.Fatalf("Allow() after abort panic = (%v, %v), want (true, true): token leaked", ok, tok)
	}
	// A client abort is not the backend's fault: it must not have tripped the
	// breaker back into Open.
	if got := b.State(); got != circuit.HalfOpen {
		t.Fatalf("state after abort panic = %v, want half_open (Ignored, not Failure)", got)
	}
}

// TestRunAttemptReleasesTokenOnOtherPanic covers a panic that is NOT a client
// abort. Unlike an abort it is classified Failure, so the probe reopens the
// breaker exactly like any other failed probe — but the underlying token
// still must not leak, or the NEXT recovery attempt would find the semaphore
// still held and refuse every probe forever.
func TestRunAttemptReleasesTokenOnOtherPanic(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b, token := halfOpenBreaker(clk)

	panicked := recoverCall(func() {
		runAttempt(b, token, func() attemptOut {
			panic(errors.New("boom"))
		})
	})
	if panicked == nil {
		t.Fatal("panic did not propagate out of runAttempt")
	}
	if got := b.State(); got != circuit.Open {
		t.Fatalf("state after non-abort panic = %v, want open (Failure, not Ignored)", got)
	}

	clk.Advance(6 * time.Second) // past OpenFor again
	if ok, tok := b.Allow(); !ok || !tok {
		t.Fatalf("Allow() on the next recovery attempt = (%v, %v), want (true, true): token leaked", ok, tok)
	}
}

// TestRunAttemptNormalPathRecordsClassifiedOutcome pins the unchanged
// behavior when do returns normally: two successful probes routed through
// runAttempt close the breaker exactly as calling Record directly would.
func TestRunAttemptNormalPathRecordsClassifiedOutcome(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b, token := halfOpenBreaker(clk)

	out := runAttempt(b, token, func() attemptOut {
		return attemptOut{status: 200}
	})
	if out.status != 200 {
		t.Fatalf("out.status = %d, want 200", out.status)
	}
	if got := b.State(); got != circuit.HalfOpen {
		t.Fatalf("state after 1 of 2 required successes = %v, want half_open", got)
	}

	ok, tok := b.Allow()
	if !ok || !tok {
		t.Fatalf("Allow() for the second probe = (%v, %v), want (true, true)", ok, tok)
	}
	runAttempt(b, tok, func() attemptOut { return attemptOut{status: 200} })
	if got := b.State(); got != circuit.Closed {
		t.Fatalf("state after 2 successful probes via runAttempt = %v, want closed", got)
	}
}
