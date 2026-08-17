package circuit

import (
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
)

// The sequence a backend produced on hardware when its inference engine died:
// it stayed up, answered 500s, and its breaker never opened.
//
// Measured from the router's own counters, over the breaker's own 30s window
// and with the backend saturated at --max-node-concurrency: 61 attempts
// admitted, 53 of them 500s. Rate 0.87 against a threshold of 0.5, and
// MinRequests met three times over. Four explanations were ruled out from the
// outside — the floor, the window, Record not being reached, and cap refusals
// starving the denominator — so this reproduces the arithmetic directly to say
// whether the breaker itself is the defect or something upstream of it is.
func TestBreakerOpensOnTheCrashSequence(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := New(DefaultConfig(), clk)

	// Interleaved rather than batched: 53 failures among 61 attempts, spread
	// through the window as real traffic was, so the result cannot depend on
	// ordering.
	fails, total := 0, 61
	for i := range total {
		permitted, token := b.Allow()
		if !permitted {
			// Already open, which is the outcome under test.
			t.Logf("breaker refused admission at attempt %d — it opened", i+1)
			break
		}
		out := Success
		if i%8 != 7 && fails < 53 { // 53 of 61
			out = Failure
			fails++
		}
		b.Record(out, token)
		clk.Advance(500 * time.Millisecond) // 61 attempts across ~30s
	}

	if got := b.State(); got != Open {
		t.Errorf("breaker is %v after %d failures in %d attempts (rate %.2f) against "+
			"MinRequests=%d FailureRate=%.2f — a backend answering 500s to a live traffic share "+
			"must be ejected, and on hardware this one was not",
			got, fails, total, float64(fails)/float64(total),
			DefaultConfig().MinRequests, DefaultConfig().FailureRate)
	}
}

// TestBreakerOpensOnAPureFailureStreak is the simplest possible statement of the
// same requirement, so a failure above can be attributed to the interleaving or
// the clock rather than to the threshold logic.
func TestBreakerOpensOnAPureFailureStreak(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := New(DefaultConfig(), clk)
	for range DefaultConfig().MinRequests + 5 {
		permitted, token := b.Allow()
		if !permitted {
			break
		}
		b.Record(Failure, token)
		clk.Advance(100 * time.Millisecond)
	}
	if got := b.State(); got != Open {
		t.Errorf("breaker is %v after %d consecutive failures with MinRequests=%d",
			got, DefaultConfig().MinRequests+5, DefaultConfig().MinRequests)
	}
}

// TestA500IsAFailure pins the classification the whole path depends on.
func TestA500IsAFailure(t *testing.T) {
	if got := Classify(500, nil); got != Failure {
		t.Errorf("Classify(500, nil) = %v, want Failure", got)
	}
	if got := Classify(503, nil); got != Failure {
		t.Errorf("Classify(503, nil) = %v, want Failure", got)
	}
	if got := Classify(429, nil); got != Ignored {
		t.Errorf("Classify(429, nil) = %v, want Ignored — shedding is not a fault", got)
	}
}
