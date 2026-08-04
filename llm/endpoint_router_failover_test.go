package llm

import (
	"sync"
	"testing"
)

func eps(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = string(rune('a'+i)) + "://x"
	}
	return out
}

// A series must keep its home endpoint as long as that endpoint is not
// overloaded — stickiness is the default, failover is the exception.
func TestSticksToHomeWhenNotOverloaded(t *testing.T) {
	r := NewEndpointRouterWithFailover(eps(6), 192, 1.5) // fair share 32, limit 48
	home := r.PickIndex(1)
	for i := 0; i < 100; i++ {
		if got := r.PickIndex(1); got != home {
			t.Fatalf("series drifted off home: got %d want %d", got, home)
		}
	}
	// Load it to exactly the limit — still not "over".
	for i := int64(0); i < 48; i++ {
		r.AcquireIndex(home)
	}
	if got := r.PickIndex(1); got != home {
		t.Fatalf("failed over at the limit rather than above it: got %d want %d", got, home)
	}
}

func TestFailsOverAboveThreshold(t *testing.T) {
	r := NewEndpointRouterWithFailover(eps(6), 192, 1.5) // limit 48
	home := r.PickIndex(1)
	for i := int64(0); i < 49; i++ { // one past the limit
		r.AcquireIndex(home)
	}
	got := r.PickIndex(1)
	if got == home {
		t.Fatalf("did not fail over: still on overloaded home %d", home)
	}
}

// The alternative must be deterministic: same rejection set -> same target.
func TestFailoverIsDeterministic(t *testing.T) {
	pick := func() int {
		r := NewEndpointRouterWithFailover(eps(6), 192, 1.5)
		home := r.PickIndex(1)
		for i := int64(0); i < 49; i++ {
			r.AcquireIndex(home)
		}
		return r.PickIndex(1)
	}
	first := pick()
	for i := 0; i < 20; i++ {
		if got := pick(); got != first {
			t.Fatalf("non-deterministic failover: got %d want %d", got, first)
		}
	}
}

// Overloading the failover target too must push the request on to a third
// endpoint -- the hash is taken over the whole rejected set, not just home.
func TestCascadesPastMultipleOverloadedEndpoints(t *testing.T) {
	r := NewEndpointRouterWithFailover(eps(6), 192, 1.5)
	home := r.PickIndex(1)
	saturate := func(idx int) {
		for i := int64(0); i < 49; i++ {
			r.AcquireIndex(idx)
		}
	}
	saturate(home)
	second := r.PickIndex(1)
	if second == home {
		t.Fatal("expected failover off home")
	}
	saturate(second)
	third := r.PickIndex(1)
	if third == home || third == second {
		t.Fatalf("did not cascade: got %d, already-rejected {%d,%d}", third, home, second)
	}
}

// When every endpoint is overloaded there is nowhere better; fall back to home
// rather than spraying requests at equally-saturated peers.
func TestAllOverloadedReturnsHome(t *testing.T) {
	n := 4
	r := NewEndpointRouterWithFailover(eps(n), 4*32, 1.5) // limit 48 each
	home := r.PickIndex(1)
	for i := 0; i < n; i++ {
		for j := int64(0); j < 49; j++ {
			r.AcquireIndex(i)
		}
	}
	if got := r.PickIndex(1); got != home {
		t.Fatalf("expected fallback to home %d, got %d", home, got)
	}
}

// Release must undo Acquire so an endpoint recovers and takes its series back.
func TestRecoversAfterRelease(t *testing.T) {
	r := NewEndpointRouterWithFailover(eps(6), 192, 1.5)
	home := r.PickIndex(1)
	for i := int64(0); i < 49; i++ {
		r.AcquireIndex(home)
	}
	if r.PickIndex(1) == home {
		t.Fatal("expected failover while loaded")
	}
	for i := int64(0); i < 49; i++ {
		r.ReleaseIndex(home)
	}
	if got := r.PickIndex(1); got != home {
		t.Fatalf("did not return home after drain: got %d want %d", got, home)
	}
}

// Zero concurrency disables failover -- pure stickiness, matching the old
// behaviour for callers that never supply a concurrency figure.
func TestZeroConcurrencyDisablesFailover(t *testing.T) {
	r := NewEndpointRouterWithFailover(eps(6), 0, 1.5)
	home := r.PickIndex(1)
	for i := int64(0); i < 10000; i++ {
		r.AcquireIndex(home)
	}
	if got := r.PickIndex(1); got != home {
		t.Fatalf("failover triggered despite being disabled: got %d want %d", got, home)
	}
}

// Home assignment still spreads series evenly -- failover must not disturb it.
func TestHomeAssignmentStaysBalanced(t *testing.T) {
	n := 6
	r := NewEndpointRouterWithFailover(eps(n), 192, 1.5)
	counts := make([]int, n)
	for s := 1; s <= 1536; s++ {
		counts[r.HomeIndexForSeries(s)]++
	}
	for i, c := range counts {
		if c != 1536/n {
			t.Fatalf("endpoint %d owns %d series, want %d", i, c, 1536/n)
		}
	}
}

func TestThresholdIsConfigurable(t *testing.T) {
	// At 1.0 the limit is the fair share itself (32), so 33 trips it.
	r := NewEndpointRouterWithFailover(eps(6), 192, 1.0)
	home := r.PickIndex(1)
	for i := int64(0); i < 33; i++ {
		r.AcquireIndex(home)
	}
	if r.PickIndex(1) == home {
		t.Fatal("threshold 1.0 should have failed over at 33 in flight")
	}
	// At 3.0 the limit is 96, so 49 is fine.
	r2 := NewEndpointRouterWithFailover(eps(6), 192, 3.0)
	home2 := r2.PickIndex(1)
	for i := int64(0); i < 49; i++ {
		r2.AcquireIndex(home2)
	}
	if r2.PickIndex(1) != home2 {
		t.Fatal("threshold 3.0 should not have failed over at 49 in flight")
	}
}

// AcquireForRequest must make check-and-increment indivisible. A fan-out group's
// siblings share a seriesNum, so they share a home endpoint and all pick at once;
// with a non-atomic check they would all read "under the limit" before any of
// them incremented and pile onto the same endpoint. This is the regression test
// for that: no endpoint may end up more than one over its limit.
func TestAcquireForRequestSpreadsConcurrentBurst(t *testing.T) {
	const n = 6
	r := NewEndpointRouterWithFailover(eps(n), 192, 1.5) // fair share 32, limit 48

	var wg sync.WaitGroup
	const burst = 600 // far more than 6 * 48
	for i := 0; i < burst; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.AcquireForRequest(1) // every goroutine uses the SAME series
		}()
	}
	wg.Wait()

	got := r.InFlight()
	var total int64
	for _, v := range got {
		total += v
	}
	if total != burst {
		t.Fatalf("lost or double-counted acquisitions: total=%d want %d (%v)", total, burst, got)
	}
	// Everything saturates eventually (600 >> 6*48) and then falls back to home,
	// but no endpoint should be starved: each must have taken real traffic.
	for i, v := range got {
		if v == 0 {
			t.Fatalf("endpoint %d took no traffic at all: %v", i, got)
		}
	}
}

// Under a burst that fits, no endpoint should exceed limit+1 — one over is the
// unavoidable case where the last acquirer takes it from limit to limit+1.
func TestAcquireForRequestRespectsLimitUnderConcurrency(t *testing.T) {
	const n = 6
	r := NewEndpointRouterWithFailover(eps(n), 192, 1.5) // limit 48
	var wg sync.WaitGroup
	const burst = 200 // < 6*48, so a correct router keeps everyone under
	for i := 0; i < burst; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.AcquireForRequest(1)
		}()
	}
	wg.Wait()
	for i, v := range r.InFlight() {
		if v > 49 {
			t.Fatalf("endpoint %d overshot: %d in flight (limit 48), all=%v", i, v, r.InFlight())
		}
	}
}
