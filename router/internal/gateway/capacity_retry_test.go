package gateway

import (
	"testing"
	"time"
)

// The backoff ladder and its jitter, asserted directly. The wiring above it —
// that a capacity refusal retries and a broken backend does not — is covered
// through the real handler in capacity_retry_wire_test.go.

func TestBackoffFollowsTheLadderAndFlattens(t *testing.T) {
	// Every delay must land inside [step/2, step], the equal-jitter band, and
	// the steps themselves must be the ladder Anton specified.
	want := []time.Duration{
		10 * time.Millisecond, 20 * time.Millisecond, 50 * time.Millisecond,
		100 * time.Millisecond, 200 * time.Millisecond, 500 * time.Millisecond,
		time.Second, 2 * time.Second, 3 * time.Second,
	}
	for n, step := range want {
		for range 200 {
			d := backoffAt(n, time.Hour)
			if d < step/2 || d > step {
				t.Fatalf("attempt %d: delay %v outside the equal-jitter band [%v, %v] for step %v",
					n, d, step/2, step, step)
			}
		}
	}

	// Past the ladder it flattens at 3s rather than growing without bound: a
	// 10s budget spent in 30s naps would make the limit a lie.
	for _, n := range []int{len(want), len(want) + 5, 100} {
		d := backoffAt(n, time.Hour)
		if d > 3*time.Second {
			t.Errorf("attempt %d: delay %v exceeds the 3s cap", n, d)
		}
	}
}

// TestBackoffJitterActuallyVaries. Without jitter every request refused by the
// same fleet at the same instant wakes together, re-saturates it, and
// re-synchronises on the next rung.
func TestBackoffJitterActuallyVaries(t *testing.T) {
	seen := map[time.Duration]bool{}
	for range 500 {
		seen[backoffAt(5, time.Hour)] = true // the 500ms step, wide enough to see spread
	}
	if len(seen) < 50 {
		t.Errorf("500 draws produced only %d distinct delays; the jitter is not spreading "+
			"retries across the interval", len(seen))
	}
}

// TestBackoffNeverOverrunsTheBudget: the last wait before the deadline must not
// sail past it. A 10s limit that sleeps 3s at 9.5s elapsed is a 12.5s limit.
func TestBackoffNeverOverrunsTheBudget(t *testing.T) {
	for _, remaining := range []time.Duration{0, time.Millisecond, 7 * time.Millisecond, time.Second} {
		for n := range 12 {
			if d := backoffAt(n, remaining); d > remaining {
				t.Errorf("attempt %d with %v left slept %v", n, remaining, d)
			}
		}
	}
}
