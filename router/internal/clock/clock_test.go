package clock_test

import (
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
)

func TestFakeNowAdvances(t *testing.T) {
	start := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	f := clock.NewFake(start)
	if got := f.Now(); !got.Equal(start) {
		t.Fatalf("Now() = %v, want %v", got, start)
	}
	f.Advance(90 * time.Second)
	if got, want := f.Now(), start.Add(90*time.Second); !got.Equal(want) {
		t.Fatalf("Now() = %v, want %v", got, want)
	}
}

func TestFakeZeroTimeIsDeterministic(t *testing.T) {
	// A zero start must not fall back to the wall clock, or tests become
	// time-of-day dependent.
	a, b := clock.NewFake(time.Time{}), clock.NewFake(time.Time{})
	if !a.Now().Equal(b.Now()) {
		t.Fatalf("two zero-start fakes disagree: %v vs %v", a.Now(), b.Now())
	}
	if a.Now().IsZero() {
		t.Fatal("zero start should map to a fixed non-zero instant")
	}
}

func TestFakeAfterFiresOnlyWhenDue(t *testing.T) {
	f := clock.NewFake(time.Time{})
	ch := f.After(10 * time.Second)

	f.Advance(9 * time.Second)
	select {
	case v := <-ch:
		t.Fatalf("After fired early at %v", v)
	default:
	}

	f.Advance(1 * time.Second)
	select {
	case <-ch:
	default:
		t.Fatal("After did not fire once due")
	}
}

func TestFakeAfterFiresOnce(t *testing.T) {
	f := clock.NewFake(time.Time{})
	ch := f.After(time.Second)
	f.Advance(10 * time.Second)
	<-ch
	select {
	case v := <-ch:
		t.Fatalf("After fired twice: %v", v)
	default:
	}
}

// The firing time must be the waiter's deadline, not the advance target.
// Otherwise a handler that reads Now() during its own firing sees a time in the
// future, which would make circuit-window and drain-deadline arithmetic wrong.
func TestFakeAfterReportsDeadlineNotAdvanceTarget(t *testing.T) {
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	f := clock.NewFake(start)
	ch := f.After(5 * time.Second)
	f.Advance(time.Hour)

	got := <-ch
	if want := start.Add(5 * time.Second); !got.Equal(want) {
		t.Fatalf("fired at %v, want the deadline %v", got, want)
	}
	// Now() should still reflect the full advance.
	if want := start.Add(time.Hour); !f.Now().Equal(want) {
		t.Fatalf("Now() = %v, want %v", f.Now(), want)
	}
}

// Waiters must fire in deadline order regardless of registration order, so a
// test's assertions do not depend on the order it happened to create timers.
func TestFakeFiresInDeadlineOrderNotRegistrationOrder(t *testing.T) {
	f := clock.NewFake(time.Time{})
	late := f.After(30 * time.Second)
	early := f.After(5 * time.Second)
	mid := f.After(10 * time.Second)

	f.Advance(time.Minute)
	got := []time.Time{<-early, <-mid, <-late}
	for i := 1; i < len(got); i++ {
		if got[i].Before(got[i-1]) {
			t.Fatalf("fired out of order: %v", got)
		}
	}
}

func TestFakeTickerRepeats(t *testing.T) {
	f := clock.NewFake(time.Time{})
	tk := f.NewTicker(time.Second)
	defer tk.Stop()

	for i := 0; i < 5; i++ {
		f.Advance(time.Second)
		select {
		case <-tk.C():
		default:
			t.Fatalf("tick %d did not fire", i)
		}
	}
}

func TestFakeTickerStopHalts(t *testing.T) {
	f := clock.NewFake(time.Time{})
	tk := f.NewTicker(time.Second)

	f.Advance(time.Second)
	<-tk.C()
	tk.Stop()

	f.Advance(10 * time.Second)
	select {
	case v := <-tk.C():
		t.Fatalf("stopped ticker fired at %v", v)
	default:
	}
}

// An unread tick must be dropped rather than blocking Advance, mirroring
// time.Ticker. If Advance blocked, a test that stops reading would deadlock.
func TestFakeTickerDropsUnreadTicks(t *testing.T) {
	f := clock.NewFake(time.Time{})
	tk := f.NewTicker(time.Second)
	defer tk.Stop()

	done := make(chan struct{})
	go func() { f.Advance(100 * time.Second); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Advance blocked on an unread ticker channel")
	}
	// Exactly one tick is buffered.
	<-tk.C()
	select {
	case <-tk.C():
		t.Fatal("more than one tick buffered")
	default:
	}
}

func TestFakeNewTickerRejectsNonPositive(t *testing.T) {
	f := clock.NewFake(time.Time{})
	for _, d := range []time.Duration{0, -time.Second} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("NewTicker(%v) did not panic", d)
				}
			}()
			f.NewTicker(d)
		}()
	}
}

func TestRealClockBasics(t *testing.T) {
	var c clock.Clock = clock.Real{}
	if c.Now().IsZero() {
		t.Fatal("Real.Now() returned zero time")
	}
	select {
	case <-c.After(time.Millisecond):
	case <-time.After(2 * time.Second):
		t.Fatal("Real.After did not fire")
	}
	tk := c.NewTicker(time.Millisecond)
	defer tk.Stop()
	select {
	case <-tk.C():
	case <-time.After(2 * time.Second):
		t.Fatal("Real ticker did not fire")
	}
}
