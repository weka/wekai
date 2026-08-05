package circuit_test

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/circuit"
	"github.com/weka/wekai/router/internal/clock"
)

func newBreaker(clk clock.Clock, tweak func(*circuit.Config)) *circuit.Breaker {
	cfg := circuit.DefaultConfig()
	if tweak != nil {
		tweak(&cfg)
	}
	return circuit.New(cfg, clk)
}

// TestClassify429IsFailure guards HLT-N4. v1 recorded `4xx including 429` as
// success, so an overloaded backend was simultaneously retried away from *and*
// recorded as healthy — its breaker could never trip.
func TestClassify429IsFailure(t *testing.T) {
	cases := []struct {
		status int
		err    error
		want   circuit.Outcome
	}{
		{200, nil, circuit.Success},
		{204, nil, circuit.Success},
		{301, nil, circuit.Success},
		{400, nil, circuit.Success}, // client's fault; backend is fine
		{404, nil, circuit.Success},
		{408, nil, circuit.Failure},
		{425, nil, circuit.Failure},
		{429, nil, circuit.Failure}, // the v1 bug
		{500, nil, circuit.Failure},
		{502, nil, circuit.Failure},
		{503, nil, circuit.Failure},
		{504, nil, circuit.Failure},
		{0, errors.New("connection refused"), circuit.Failure},
	}
	for _, c := range cases {
		if got := circuit.Classify(c.status, c.err); got != c.want {
			t.Errorf("Classify(%d, %v) = %v, want %v", c.status, c.err, got, c.want)
		}
	}
}

// TestSlidingWindowOpenTriggerUsesWindowDuration guards HLT-N2: v1 configured
// and documented a window_duration and never read it, counting only consecutive
// failures — so a backend failing intermittently never opened.
func TestSlidingWindowOpenTriggerUsesWindowDuration(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := newBreaker(clk, func(c *circuit.Config) {
		c.Window = 10 * time.Second
		c.Buckets = 10
		c.MinRequests = 10
		c.FailureRate = 0.5
	})

	// 20 alternating outcomes inside the window: 50% failure rate, >= min.
	// Non-consecutive, so a consecutive-only implementation would never open.
	for i := 0; i < 20; i++ {
		o := circuit.Success
		if i%2 == 0 {
			o = circuit.Failure
		}
		b.Record(o, false)
	}
	if got := b.State(); got != circuit.Open {
		t.Fatalf("state = %v, want open (rate 0.5 over %d requests)", got, 20)
	}
}

func TestFailuresOutsideWindowDoNotOpen(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := newBreaker(clk, func(c *circuit.Config) {
		c.Window = 10 * time.Second
		c.Buckets = 10
		c.MinRequests = 10
		c.FailureRate = 0.5
	})

	for i := 0; i < 9; i++ {
		b.Record(circuit.Failure, false)
	}
	// Age those failures out of the window entirely.
	clk.Advance(11 * time.Second)
	for i := 0; i < 9; i++ {
		b.Record(circuit.Failure, false)
	}
	// 9 in-window failures is below MinRequests=10, so still closed.
	if got := b.State(); got != circuit.Closed {
		t.Fatalf("state = %v, want closed (expired failures must not count)", got)
	}
}

func TestBelowMinRequestsNeverOpens(t *testing.T) {
	b := newBreaker(clock.NewFake(time.Time{}), func(c *circuit.Config) {
		c.MinRequests = 20
	})
	for i := 0; i < 19; i++ {
		b.Record(circuit.Failure, false)
	}
	if got := b.State(); got != circuit.Closed {
		t.Fatalf("state = %v, want closed: 19 failures is below MinRequests=20", got)
	}
}

// TestHalfOpenAdmitsExactlyMaxUnder100Probes guards HLT-N3. v1 returned "allow"
// for HalfOpen from a bare state check with no probe cap, despite a comment
// claiming limited admission, so a recovering backend was instantly re-flooded.
func TestHalfOpenAdmitsExactlyMaxUnder100Probes(t *testing.T) {
	for _, max := range []int32{1, 3} {
		clk := clock.NewFake(time.Time{})
		b := newBreaker(clk, func(c *circuit.Config) {
			c.MinRequests = 1
			c.FailureRate = 0.5
			c.OpenFor = 5 * time.Second
			c.HalfOpenMax = max
		})
		b.Record(circuit.Failure, false)
		b.Record(circuit.Failure, false)
		if got := b.State(); got != circuit.Open {
			t.Fatalf("max=%d: state = %v, want open", max, got)
		}
		clk.Advance(6 * time.Second) // OpenFor elapsed: probes now admissible

		var mu sync.Mutex
		admitted := 0
		var wg sync.WaitGroup
		wg.Add(100)
		for i := 0; i < 100; i++ {
			go func() {
				defer wg.Done()
				if ok, _ := b.Allow(); ok {
					mu.Lock()
					admitted++
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		if admitted != int(max) {
			t.Errorf("HalfOpenMax=%d: admitted %d of 100 concurrent probes, want %d",
				max, admitted, max)
		}
	}
}

// TestFilteringNeverConsumesHalfOpenTokens is the R2 regression test.
//
// Candidate filtering reads State(); only the selected backend calls Allow().
// If filtering consumed a probe token — as the original design's
// "HalfOpen-with-token" filter would have — then with HalfOpenMax=1 a single
// filtering pass would exhaust the budget forever: the backend could never be
// probed, so it could never close, so it would never return to rotation.
func TestFilteringNeverConsumesHalfOpenTokens(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := newBreaker(clk, func(c *circuit.Config) {
		c.MinRequests = 1
		c.FailureRate = 0.5
		c.OpenFor = 5 * time.Second
		c.HalfOpenMax = 1
	})
	b.Record(circuit.Failure, false)
	b.Record(circuit.Failure, false)
	clk.Advance(6 * time.Second)

	// Simulate 1000 routing passes that merely *filter* this backend.
	for i := 0; i < 1000; i++ {
		if got := b.State(); got != circuit.HalfOpen {
			t.Fatalf("pass %d: State() = %v, want half_open", i, got)
		}
	}
	// A probe must still be admissible afterwards.
	ok, token := b.Allow()
	if !ok || !token {
		t.Fatal("probe refused after 1000 filtering passes: filtering consumed tokens")
	}
	b.Record(circuit.Success, token)
}

// An Open breaker whose OpenFor has elapsed must report HalfOpen from State(),
// or candidate filtering would exclude it forever and it could never recover.
func TestStateReportsHalfOpenOnceOpenForElapsed(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := newBreaker(clk, func(c *circuit.Config) {
		c.MinRequests = 1
		c.FailureRate = 0.5
		c.OpenFor = 5 * time.Second
	})
	b.Record(circuit.Failure, false)
	b.Record(circuit.Failure, false)
	if got := b.State(); got != circuit.Open {
		t.Fatalf("state = %v, want open", got)
	}
	clk.Advance(4 * time.Second)
	if got := b.State(); got != circuit.Open {
		t.Fatalf("state = %v before OpenFor elapsed, want open", got)
	}
	clk.Advance(2 * time.Second)
	if got := b.State(); got != circuit.HalfOpen {
		t.Fatalf("state = %v after OpenFor elapsed, want half_open", got)
	}
}

func TestHalfOpenSuccessesCloseAndFailureReopens(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	mk := func() *circuit.Breaker {
		b := newBreaker(clk, func(c *circuit.Config) {
			c.MinRequests = 1
			c.FailureRate = 0.5
			c.OpenFor = 5 * time.Second
			c.HalfOpenMax = 1
			c.HalfOpenSuccesses = 2
		})
		b.Record(circuit.Failure, false)
		b.Record(circuit.Failure, false)
		clk.Advance(6 * time.Second)
		return b
	}

	b := mk()
	for i := 0; i < 2; i++ {
		ok, tok := b.Allow()
		if !ok {
			t.Fatalf("probe %d refused", i)
		}
		b.Record(circuit.Success, tok)
	}
	if got := b.State(); got != circuit.Closed {
		t.Fatalf("state = %v after 2 successful probes, want closed", got)
	}

	b2 := mk()
	ok, tok := b2.Allow()
	if !ok {
		t.Fatal("probe refused")
	}
	b2.Record(circuit.Failure, tok)
	if got := b2.State(); got != circuit.Open {
		t.Fatalf("state = %v after failed probe, want open", got)
	}
}

func TestTransitionCallbackFires(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	b := newBreaker(clk, func(c *circuit.Config) {
		c.MinRequests = 1
		c.FailureRate = 0.5
	})
	var got []string
	b.OnTransition(func(from, to circuit.State, ok, fail int) {
		got = append(got, from.String()+"->"+to.String())
	})
	b.Record(circuit.Failure, false)
	b.Record(circuit.Failure, false)
	if len(got) != 1 || got[0] != "closed->open" {
		t.Fatalf("transitions = %v, want [closed->open]", got)
	}
}

func TestClosedAllowTakesNoToken(t *testing.T) {
	b := newBreaker(clock.NewFake(time.Time{}), nil)
	ok, token := b.Allow()
	if !ok || token {
		t.Fatalf("Allow() = (%v, %v), want (true, false) when closed", ok, token)
	}
}
