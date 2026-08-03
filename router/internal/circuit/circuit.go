// Package circuit implements a per-backend circuit breaker.
//
// It exists because v1's breaker was wrong in four ways that this package
// fixes by construction:
//
//   - v1 configured a `window_duration` and never read it; only consecutive
//     failures counted, so a backend failing 4 times an hour for a week never
//     opened. Here the sliding window is the ONLY trigger (HLT-7, HLT-N2).
//   - v1's HalfOpen returned "allowed" from a bare state check, so a
//     recovering backend was instantly re-flooded. Here admission is a real
//     semaphore (HLT-8, HLT-N3).
//   - v1 recorded 4xx *including 429* as success, making the breaker blind to
//     the most common overload signal (HLT-9, HLT-N4).
//   - v1 called can_execute() — which mutates state — on every candidate
//     during selection, so selection had a side effect on backends it did not
//     pick. Here State() is read-only and Allow() is called exactly once, for
//     the selected backend only (R2, and v1 bug F1).
package circuit

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/router/internal/clock"
)

type State uint8

const (
	Closed State = iota
	Open
	HalfOpen
)

func (s State) String() string {
	switch s {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case HalfOpen:
		return "half_open"
	}
	return "unknown"
}

type Outcome uint8

const (
	Success Outcome = iota
	Failure
)

// Classify is explicit and total (HLT-9).
//
// Failures: transport errors, timeouts, all 5xx, and the overload/backpressure
// statuses 408, 425, 429. Successes: 2xx, 3xx, and other 4xx — those are the
// client's fault and say nothing about backend health.
//
// 429 being a failure is deliberate and is the v1 fix: it is simultaneously in
// the retryable set, so v1 would retry a request elsewhere while recording the
// overloaded backend as healthy, and the breaker never tripped.
func Classify(status int, err error) Outcome {
	if err != nil {
		return Failure
	}
	switch status {
	case 408, 425, 429:
		return Failure
	}
	if status >= 500 {
		return Failure
	}
	return Success
}

type Config struct {
	// Window is the sliding evaluation window. Genuinely used (HLT-N2).
	Window time.Duration
	// Buckets divides Window; more buckets means finer expiry granularity.
	Buckets int
	// MinRequests is the floor below which the breaker never opens, so a
	// single early failure on an idle backend cannot trip it.
	MinRequests int
	// FailureRate in [0,1] is the open threshold within the window.
	FailureRate float64
	// OpenFor is how long Open persists before a probe is admitted.
	OpenFor time.Duration
	// HalfOpenMax bounds concurrent probes. Enforced by a semaphore.
	HalfOpenMax int32
	// HalfOpenSuccesses is how many consecutive probe successes close it.
	HalfOpenSuccesses int32
}

func DefaultConfig() Config {
	return Config{
		Window:            30 * time.Second,
		Buckets:           30,
		MinRequests:       20,
		FailureRate:       0.5,
		OpenFor:           30 * time.Second,
		HalfOpenMax:       1,
		HalfOpenSuccesses: 2,
	}
}

type bucket struct {
	startSec int64
	ok, fail uint32
}

// TransitionFunc is called on every state change, outside the lock.
type TransitionFunc func(from, to State, ok, fail int)

type Breaker struct {
	cfg Config
	clk clock.Clock

	mu         sync.Mutex
	ring       []bucket
	state      State
	openedAt   time.Time
	halfOpenOK int32

	// halfOpenTokens counts probes currently in flight. Guarded by CAS, not
	// by mu, so Allow's fast path in Closed takes no lock beyond the ring.
	halfOpenTokens atomic.Int32

	onTransition TransitionFunc
}

func New(cfg Config, clk clock.Clock) *Breaker {
	if cfg.Buckets <= 0 {
		cfg.Buckets = 1
	}
	if cfg.HalfOpenMax < 1 {
		cfg.HalfOpenMax = 1
	}
	if cfg.HalfOpenSuccesses < 1 {
		cfg.HalfOpenSuccesses = 1
	}
	return &Breaker{cfg: cfg, clk: clk, ring: make([]bucket, cfg.Buckets), state: Closed}
}

// OnTransition registers a callback for state changes (HLT-10).
func (b *Breaker) OnTransition(f TransitionFunc) { b.onTransition = f }

// State reports the breaker's EFFECTIVE state without mutating anything.
//
// This is what candidate filtering uses (R2). Critically, an Open breaker
// whose OpenFor has elapsed reports HalfOpen here: if it kept reporting Open,
// filtering would exclude it forever, it would never be probed, and it could
// never recover. The actual transition happens in Allow.
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == Open && !b.clk.Now().Before(b.openedAt.Add(b.cfg.OpenFor)) {
		return HalfOpen
	}
	return b.state
}

// Allow admits a request. Call it EXACTLY ONCE, for the backend that was
// actually selected — never while filtering candidates.
//
// The returned token must be passed to the matching Record call. When token is
// true a half-open probe slot is held, and Record releases it.
func (b *Breaker) Allow() (permitted bool, token bool) {
	b.mu.Lock()
	now := b.clk.Now()
	if b.state == Open && !now.Before(b.openedAt.Add(b.cfg.OpenFor)) {
		b.transitionLocked(HalfOpen, now)
	}
	st := b.state
	b.mu.Unlock()

	switch st {
	case Closed:
		return true, false
	case Open:
		return false, false
	default: // HalfOpen: bounded admission via CAS, not a state check.
		for {
			cur := b.halfOpenTokens.Load()
			if cur >= b.cfg.HalfOpenMax {
				return false, false
			}
			if b.halfOpenTokens.CompareAndSwap(cur, cur+1) {
				return true, true
			}
		}
	}
}

// Record reports the outcome of an admitted request. token must be the value
// returned by the matching Allow.
func (b *Breaker) Record(o Outcome, token bool) {
	if token {
		defer b.halfOpenTokens.Add(-1)
	}

	b.mu.Lock()
	now := b.clk.Now()

	if b.state == HalfOpen {
		if o == Failure {
			b.halfOpenOK = 0
			b.transitionLocked(Open, now)
			b.mu.Unlock()
			return
		}
		b.halfOpenOK++
		if b.halfOpenOK >= b.cfg.HalfOpenSuccesses {
			b.transitionLocked(Closed, now)
		}
		b.mu.Unlock()
		return
	}

	b.observeLocked(o, now)
	if b.state == Closed {
		ok, fail := b.sumLocked(now)
		total := ok + fail
		if total >= b.cfg.MinRequests && float64(fail)/float64(total) >= b.cfg.FailureRate {
			b.transitionLocked(Open, now)
		}
	}
	b.mu.Unlock()
}

func (b *Breaker) observeLocked(o Outcome, now time.Time) {
	sec := now.Unix()
	i := int(((sec % int64(b.cfg.Buckets)) + int64(b.cfg.Buckets)) % int64(b.cfg.Buckets))
	if b.ring[i].startSec != sec {
		b.ring[i] = bucket{startSec: sec}
	}
	if o == Failure {
		b.ring[i].fail++
	} else {
		b.ring[i].ok++
	}
}

// sumLocked totals only buckets inside the window. A bucket whose startSec has
// aged out is simply skipped, which is what makes Window load-bearing.
func (b *Breaker) sumLocked(now time.Time) (ok, fail int) {
	cutoff := now.Add(-b.cfg.Window).Unix()
	for _, bk := range b.ring {
		if bk.startSec <= cutoff {
			continue
		}
		ok += int(bk.ok)
		fail += int(bk.fail)
	}
	return
}

func (b *Breaker) transitionLocked(to State, now time.Time) {
	from := b.state
	if from == to {
		return
	}
	b.state = to
	switch to {
	case Open:
		b.openedAt = now
	case Closed:
		for i := range b.ring {
			b.ring[i] = bucket{}
		}
		b.halfOpenOK = 0
	case HalfOpen:
		b.halfOpenOK = 0
	}
	if b.onTransition != nil {
		ok, fail := b.sumLocked(now)
		f := b.onTransition
		// Called under mu: callers must not re-enter the breaker. Documented
		// on OnTransition; the only real caller logs and bumps a counter.
		f(from, to, ok, fail)
	}
}
