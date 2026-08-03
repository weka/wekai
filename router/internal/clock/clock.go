// Package clock abstracts time so that every time-dependent decision in the
// router — circuit-breaker windows, health hysteresis, drain deadlines, cache
// TTLs — is unit-testable without sleeps.
//
// Satisfies AC-0.2. Production code MUST take a Clock rather than calling
// time.Now directly; hack/ enforces this with a grep fence.
package clock

import (
	"sync"
	"time"
)

// Clock is the minimal surface the router needs.
type Clock interface {
	Now() time.Time
	// After returns a channel that receives once, after d has elapsed.
	After(d time.Duration) <-chan time.Time
	// NewTicker returns a ticker that fires every d until stopped.
	NewTicker(d time.Duration) Ticker
}

// Ticker mirrors time.Ticker behind an interface.
type Ticker interface {
	C() <-chan time.Time
	Stop()
}

// Real is the production clock.
type Real struct{}

func (Real) Now() time.Time                         { return time.Now() }
func (Real) After(d time.Duration) <-chan time.Time { return time.After(d) }
func (Real) NewTicker(d time.Duration) Ticker       { return &realTicker{t: time.NewTicker(d)} }

type realTicker struct{ t *time.Ticker }

func (r *realTicker) C() <-chan time.Time { return r.t.C }
func (r *realTicker) Stop()               { r.t.Stop() }

// Fake is a manually-advanced clock for tests.
//
// Advance fires every waiter whose deadline has passed, in deadline order, so
// tests are deterministic regardless of registration order.
type Fake struct {
	mu      sync.Mutex
	now     time.Time
	waiters []*waiter
}

type waiter struct {
	at     time.Time
	ch     chan time.Time
	period time.Duration // non-zero for tickers
	closed bool
}

// NewFake returns a Fake clock started at t. A zero t defaults to a fixed,
// arbitrary instant so tests never depend on the wall clock.
func NewFake(t time.Time) *Fake {
	if t.IsZero() {
		t = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	}
	return &Fake{now: t}
}

func (f *Fake) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.now
}

func (f *Fake) After(d time.Duration) <-chan time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	w := &waiter{at: f.now.Add(d), ch: make(chan time.Time, 1)}
	f.waiters = append(f.waiters, w)
	return w.ch
}

func (f *Fake) NewTicker(d time.Duration) Ticker {
	if d <= 0 {
		panic("clock: non-positive ticker interval")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	w := &waiter{at: f.now.Add(d), ch: make(chan time.Time, 1), period: d}
	f.waiters = append(f.waiters, w)
	return &fakeTicker{f: f, w: w}
}

// Advance moves the clock forward by d, firing any waiters that come due.
func (f *Fake) Advance(d time.Duration) {
	f.mu.Lock()
	target := f.now.Add(d)
	// Fire in deadline order, stepping `now` to each deadline so that a
	// handler observing Now() during its own firing sees a consistent time.
	for {
		var next *waiter
		for _, w := range f.waiters {
			if w.closed || w.at.After(target) {
				continue
			}
			if next == nil || w.at.Before(next.at) {
				next = w
			}
		}
		if next == nil {
			break
		}
		f.now = next.at
		if next.period > 0 {
			next.at = next.at.Add(next.period)
		} else {
			next.closed = true
		}
		ch, at := next.ch, f.now
		f.mu.Unlock()
		select {
		case ch <- at:
		default: // a ticker whose previous tick was never read: drop, like time.Ticker
		}
		f.mu.Lock()
	}
	f.now = target
	f.mu.Unlock()
}

type fakeTicker struct {
	f *Fake
	w *waiter
}

func (t *fakeTicker) C() <-chan time.Time { return t.w.ch }
func (t *fakeTicker) Stop() {
	t.f.mu.Lock()
	defer t.f.mu.Unlock()
	t.w.closed = true
}
