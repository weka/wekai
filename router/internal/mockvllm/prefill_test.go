package mockvllm

import (
	"context"
	"testing"
	"time"
)

// prefillTolerance is the wall-clock slack these tests allow — generous
// enough to absorb goroutine-scheduling jitter on a loaded CI box without
// hiding a real correctness bug: the processor-sharing MATH itself
// (settle/reschedule) is exact float64 arithmetic, only the wall-clock
// MEASUREMENT of it needs slack. Per anton's spec: ±10%.
const prefillTolerance = 0.10

func withinTolerance(got, want time.Duration) bool {
	if want <= 0 {
		return got >= 0 && got < 5*time.Millisecond
	}
	lo := time.Duration(float64(want) * (1 - prefillTolerance))
	hi := time.Duration(float64(want) * (1 + prefillTolerance))
	return got >= lo && got <= hi
}

// TestPrefillScheduler_SoloJobCompletesInItsOwnWork is spec test (a): one
// request with W seconds of prefill work, nothing else contending,
// completes in ~W.
func TestPrefillScheduler_SoloJobCompletesInItsOwnWork(t *testing.T) {
	s := newPrefillScheduler(nil)
	defer s.Close()

	const w = 60 * time.Millisecond
	start := time.Now()
	if !s.submit(context.Background(), w.Seconds()) {
		t.Fatal("submit returned false unexpectedly")
	}
	elapsed := time.Since(start)
	if !withinTolerance(elapsed, w) {
		t.Fatalf("elapsed = %v, want ~%v (±%.0f%%)", elapsed, w, prefillTolerance*100)
	}
}

// TestPrefillScheduler_FourConcurrentJobsShareTheRate is spec test (b): 4
// identical concurrent jobs of solo-size W must each take ~4W wall-clock —
// the instance's AGGREGATE throughput is conserved regardless of how many
// requests are concurrently prefilling, which is the entire point of
// processor sharing over independent per-request rates (with independent
// rates, all 4 would complete in ~W, and the instance would appear to serve
// 4x its real aggregate throughput).
func TestPrefillScheduler_FourConcurrentJobsShareTheRate(t *testing.T) {
	s := newPrefillScheduler(nil)
	defer s.Close()

	const w = 40 * time.Millisecond
	const n = 4

	start := time.Now()
	done := make(chan time.Duration, n)
	for i := 0; i < n; i++ {
		go func() {
			if !s.submit(context.Background(), w.Seconds()) {
				t.Error("submit returned false unexpectedly")
			}
			done <- time.Since(start)
		}()
	}

	want := n * w
	for i := 0; i < n; i++ {
		elapsed := <-done
		if !withinTolerance(elapsed, want) {
			t.Errorf("job %d elapsed = %v, want ~%v (±%.0f%%) — aggregate rate not conserved",
				i, elapsed, want, prefillTolerance*100)
		}
	}
}

// TestPrefillScheduler_StaggeredJoinDelaysTheLongJobProportionally is spec
// test (c): a short job joining mid-way through a long job delays the long
// job proportionally.
//
// The exact prediction follows from PS's work-conservation property: the
// resource always serves exactly 1 solo-second of aggregate work per
// wall-clock second, split however many ways are active, so any solo-time S
// the short job consumes while both are active is time the long job did
// NOT get — regardless of exactly when the overlap happened. The long job's
// total completion time is therefore its own solo size PLUS the short job's
// entire solo size: long+short. (Worked out per-phase: the long job runs
// solo for `mid`, both share 1/2-rate for 2*short — during which the long
// job's remaining drops by exactly `short` solo-seconds — then the long job
// finishes solo again. mid + 2*short + (long-mid-short) = long+short,
// independent of `mid`.)
func TestPrefillScheduler_StaggeredJoinDelaysTheLongJobProportionally(t *testing.T) {
	s := newPrefillScheduler(nil)
	defer s.Close()

	const long = 100 * time.Millisecond
	const mid = 40 * time.Millisecond // short job joins this far into the long job
	const short = 20 * time.Millisecond

	start := time.Now()
	longDone := make(chan time.Duration, 1)
	go func() {
		if !s.submit(context.Background(), long.Seconds()) {
			t.Error("long job's submit returned false unexpectedly")
		}
		longDone <- time.Since(start)
	}()

	time.Sleep(mid)

	shortStart := time.Now()
	if !s.submit(context.Background(), short.Seconds()) {
		t.Fatal("short job's submit returned false unexpectedly")
	}
	shortElapsed := time.Since(shortStart)
	longElapsed := <-longDone

	wantShort := 2 * short // both jobs share 1/2 rate for the short job's entire lifetime
	wantLong := long + short

	if !withinTolerance(shortElapsed, wantShort) {
		t.Errorf("short job elapsed = %v, want ~%v (±%.0f%%)", shortElapsed, wantShort, prefillTolerance*100)
	}
	if !withinTolerance(longElapsed, wantLong) {
		t.Errorf("long job elapsed = %v, want ~%v (±%.0f%%) — proportional delay from the short job's contention",
			longElapsed, wantLong, prefillTolerance*100)
	}
}

// TestEngine_AwaitTTFTReportsBasePlusPrefillCompletion is spec test (d):
// TTFT (as measured by how long AwaitTTFT actually blocks) is still
// BaseLatency + this request's prefill completion time — for a SOLO
// request with no contention, that's exactly base+work, matching the old
// pure-duration Latency()'s formula, just now genuinely elapsed rather than
// precomputed.
func TestEngine_AwaitTTFTReportsBasePlusPrefillCompletion(t *testing.T) {
	const base = 30 * time.Millisecond
	e := NewEngine(Config{BaseLatency: base, ColdInputTPS: 1000}) // 1ms/token
	defer e.Close()

	work := e.PrefillWork(Residency{Total: 50}) // 50 uncached tokens @ 1000 tok/s = 50ms
	start := time.Now()
	if !e.AwaitTTFT(context.Background(), work) {
		t.Fatal("AwaitTTFT returned false unexpectedly")
	}
	elapsed := time.Since(start)

	want := base + work
	if !withinTolerance(elapsed, want) {
		t.Fatalf("elapsed = %v, want ~%v (±%.0f%%) = base(%v) + prefill work(%v)",
			elapsed, want, prefillTolerance*100, base, work)
	}
}

// TestPrefillScheduler_CancelledJobStopsConsumingItsShare is not one of
// anton's four spec tests, but it exercises the leave/cancellation path
// (submit's ctx.Done branch, the leaveCh handling in run) that the other
// four never touch: a client disconnect must remove the job from the
// active set, or it would permanently steal a share of the instance's rate
// from every other request for the rest of the process's life — the
// simulator equivalent of a leaked lease.
func TestPrefillScheduler_CancelledJobStopsConsumingItsShare(t *testing.T) {
	s := newPrefillScheduler(nil)
	defer s.Close()

	// A long job that will be cancelled long before it could complete.
	longCtx, cancelLong := context.WithCancel(context.Background())
	longDone := make(chan bool, 1)
	go func() {
		longDone <- s.submit(longCtx, (200 * time.Millisecond).Seconds())
	}()

	// Give it time to actually join (become an active, contending job).
	time.Sleep(20 * time.Millisecond)
	cancelLong()
	if ok := <-longDone; ok {
		t.Fatal("cancelled job's submit returned true, want false")
	}

	// A fresh solo job submitted AFTER the cancellation must see NO
	// contention — if the cancelled job's share leaked, this would take
	// roughly 2x as long as its own solo size instead of ~1x.
	const w = 40 * time.Millisecond
	start := time.Now()
	if !s.submit(context.Background(), w.Seconds()) {
		t.Fatal("submit returned false unexpectedly")
	}
	elapsed := time.Since(start)
	if !withinTolerance(elapsed, w) {
		t.Fatalf("elapsed = %v, want ~%v (±%.0f%%) — a cancelled job is still consuming a rate share",
			elapsed, w, prefillTolerance*100)
	}
}
