package benchmark

import "testing"

// The error ceilings have to be reachable during the ramp, because the ramp is
// where a misconfigured fleet is at its most obviously broken.
//
// The evaluator's warmup gate is StartSeries*2 completions before anything
// below it is looked at. That is 10 on a small run and 16,000 at --series 8000,
// and a ceiling of 512 consecutive failures that cannot be consulted until
// 16,000 requests have landed is not the ceiling its flag says it is. Errors do
// count as completions, so such a run does eventually abort — after thirty
// times the failures it was configured to tolerate.
//
// errorAbortEarned cannot see totalCompleted at all, which is what makes the
// ordering a property of the code rather than of where a block happens to sit.

func abortState(totalErrors, consecutive int64) *autoState {
	st := &autoState{}
	st.totalErrors.Store(totalErrors)
	st.consecutiveFailures.Store(consecutive)
	return st
}

func TestConsecutiveFailureCeilingIsReachableBeforeWarmup(t *testing.T) {
	// A run at --series 8000: the warmup gate is 16,000 completions away, and
	// nothing here is allowed to care.
	cfg := AutoBenchmarkConfig{MaxConsecutiveFailures: 512, MinEvalRequests: 16000, StartSeries: 8000}

	if abortState(0, 511).errorAbortEarned(cfg) {
		t.Error("aborted at 511 consecutive failures with the ceiling at 512")
	}
	if !abortState(0, 512).errorAbortEarned(cfg) {
		t.Error("512 consecutive failures did not earn an abort at a ceiling of 512; a fleet failing " +
			"every request must stop the run when the flag says it should, not after the warmup " +
			"gate's thousands of completions have gone by")
	}
}

func TestTotalErrorCeilingStillFiresAndStaysOptIn(t *testing.T) {
	off := AutoBenchmarkConfig{MaxConsecutiveFailures: 512}
	if abortState(9999, 0).errorAbortEarned(off) {
		t.Error("total errors aborted the run with --max-total-errors unset; it is opt-in, and a long " +
			"run on a fleet with a steady low error rate is not a failed run")
	}

	on := AutoBenchmarkConfig{MaxConsecutiveFailures: 512, MaxTotalErrors: 3}
	if !abortState(3, 0).errorAbortEarned(on) {
		t.Error("3 total errors did not earn an abort at a ceiling of 3")
	}
	// Monotonic: it catches errors that successes keep resetting the
	// consecutive counter between.
	if !abortState(3, 1).errorAbortEarned(on) {
		t.Error("the total ceiling must fire on scattered errors, which is the case the consecutive " +
			"counter is blind to by design")
	}
}

func TestNeitherCeilingFiresOnACleanRun(t *testing.T) {
	cfg := AutoBenchmarkConfig{MaxConsecutiveFailures: 512, MaxTotalErrors: 100}
	if abortState(0, 0).errorAbortEarned(cfg) {
		t.Error("a run with no errors earned an abort")
	}
	if abortState(99, 511).errorAbortEarned(cfg) {
		t.Error("a run just under both ceilings earned an abort")
	}
}

// Both ceilings default off when zero, so a caller that leaves them unset gets
// a run that no counter can stop. (The run's own defaulting sets 512 — this is
// about the predicate not inventing a ceiling of its own.)
func TestZeroCeilingsDisableTheAbort(t *testing.T) {
	if abortState(1e6, 1e6).errorAbortEarned(AutoBenchmarkConfig{}) {
		t.Error("an unset ceiling aborted the run")
	}
}
