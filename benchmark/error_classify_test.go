package benchmark

import (
	"context"
	"strings"
	"testing"
	"time"
)

// "context deadline exceeded" is the same sentence for failures that mean
// opposite things. Across two realtime arms it was 100% of client-visible
// errors, and classifying them retrospectively showed the two fleets were not
// failing the same way: one died at the configured cap having streamed for five
// minutes, the other died before a first token in a cluster nowhere near any
// cap. The two error rates were being compared as though they counted one
// event.

func TestDeadlineErrorNamesTheLimitPhaseAndElapsed(t *testing.T) {
	for _, tc := range []struct {
		name        string
		ttft        time.Duration
		gotResponse bool
		wantPhase   string
	}{
		{"hung before any bytes", 0, false, "before the response headers arrived"},
		{"headers then nothing", 0, true, "after headers, before any token"},
		{"cut mid-generation", 30 * time.Second, true, "mid-stream"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := RequestMetrics{TimeToFirstToken: tc.ttft}
			err := classifyDeadline(context.DeadlineExceeded, 5*time.Minute,
				4*time.Minute+59*time.Second, tc.gotResponse, &m)
			if err == nil {
				t.Fatal("classification dropped the error")
			}
			got := err.Error()
			if !strings.Contains(got, "5m0s") {
				t.Errorf("%q does not name the limit that fired", got)
			}
			if !strings.Contains(got, "4m59s") {
				t.Errorf("%q does not say how long it actually ran — the gap between elapsed and "+
					"the limit is what shows a DIFFERENT deadline cut it", got)
			}
			if !strings.Contains(got, tc.wantPhase) {
				t.Errorf("%q does not say it died %q", got, tc.wantPhase)
			}
		})
	}
}

// TestNonDeadlineErrorsPassThroughUntouched: a 429, a 404 or a broken pipe
// already says what it is, and rewriting it as a timeout would be worse than
// the bare string this replaces.
func TestNonDeadlineErrorsPassThroughUntouched(t *testing.T) {
	orig := context.Canceled
	if got := classifyDeadline(orig, time.Minute, time.Second, true, &RequestMetrics{}); got != orig {
		t.Errorf("a non-deadline error was rewritten as %v", got)
	}
	if got := classifyDeadline(nil, time.Minute, time.Second, true, &RequestMetrics{}); got != nil {
		t.Errorf("a successful request was given an error: %v", got)
	}
}

// TestMidStreamReportsTimeSinceFirstToken: "died 4m29s after its first token"
// separates a slow generation from a stalled one, which is the distinction
// between the two arms' failure modes.
func TestMidStreamReportsTimeSinceFirstToken(t *testing.T) {
	m := RequestMetrics{TimeToFirstToken: 30 * time.Second}
	err := classifyDeadline(context.DeadlineExceeded, 5*time.Minute, 5*time.Minute, true, &m)
	if !strings.Contains(err.Error(), "4m30s after its first token") {
		t.Errorf("%q should say how long it streamed before being cut", err.Error())
	}
}

// The 429 retry budget bounds WALL TIME, not the sum of the sleeps.
//
// Bounding sleeping only let a second cost compound invisibly: each attempt also
// sat inside the router for up to --retry-time-limit before being refused, so
// total = budget + (retries+1) x hold. On hardware that put 2,514 failures on
// exactly 200/210/220/230/240s out of a 30s budget, variance under a second.
// Nothing summed to those numbers because the second term is a PRODUCT.
func TestRetryBudgetBoundsWallTimeNotSleep(t *testing.T) {
	const budget = 30 * time.Second

	// Well inside the budget by either measure: keep going.
	if _, ok := backoff429(time.Second, 5*time.Second, budget); !ok {
		t.Error("gave up 5s into a 30s budget")
	}
	// Elapsed has reached the budget even though little of it was spent
	// sleeping — which is exactly the case that used to keep retrying.
	if _, ok := backoff429(time.Second, budget, budget); ok {
		t.Error("kept retrying at the budget; the bound must be on elapsed time, or a server that " +
			"holds each attempt multiplies the budget by the attempt count")
	}
	// And the last wait never overruns what is left.
	w, ok := backoff429(10*time.Second, budget-time.Second, budget)
	if !ok {
		t.Fatal("refused a retry with a second still left")
	}
	if w > time.Second {
		t.Errorf("waited %v with 1s of budget left", w)
	}
}
