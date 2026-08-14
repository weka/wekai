package benchmark

import (
	"context"
	"sync"
	"time"
)

// Real-time replay: sessions paced by the timestamps they were captured with,
// and admitted by a governor that watches time-to-first-token.
//
// The existing replay driver runs a fixed pool of series, each firing its
// session's requests back to back. That answers "how does the fleet behave at
// concurrency N". This answers a different question — "what is the most load
// this fleet can carry" — and the two need different machinery, because the
// answer to the second must not be a number the operator chose.
//
// Load grows by adding SESSIONS, not by raising a concurrency limit. A session
// is a conversation with its own think time, so N sessions produce whatever
// concurrency the fleet's own latency implies rather than a number set in
// advance. The governor adds one per tick while windowed TTFT stays under the
// limit and pauses above it, so the fleet's latency is the only throttle.

// ttftWindow is the mean TTFT over a trailing wall-clock window.
//
// A DURATION window, not a count of requests: a count is a different amount of
// history at every load level — at 140 req/s the last 32 requests are a fifth
// of a second, at 5 req/s they are six — so a count-based gate tightens as the
// fleet fills, which is exactly where its behaviour has to stay comparable.
type ttftWindow struct {
	mu     sync.Mutex
	window time.Duration
	at     []time.Time
	val    []time.Duration
	sum    time.Duration
}

func newTTFTWindow(window time.Duration) *ttftWindow {
	if window <= 0 {
		window = 30 * time.Second
	}
	return &ttftWindow{window: window}
}

// Observe records one completed request's TTFT.
func (w *ttftWindow) Observe(now time.Time, d time.Duration) {
	if d <= 0 {
		// A request that never reported a first token says nothing about how
		// long a first token takes. Counting it as zero would drag the mean
		// down and hold the gate open on a fleet that had stopped answering.
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.at = append(w.at, now)
	w.val = append(w.val, d)
	w.sum += d
	w.expireLocked(now)
}

func (w *ttftWindow) expireLocked(now time.Time) {
	cut := now.Add(-w.window)
	i := 0
	for i < len(w.at) && w.at[i].Before(cut) {
		w.sum -= w.val[i]
		i++
	}
	if i > 0 {
		w.at = append(w.at[:0], w.at[i:]...)
		w.val = append(w.val[:0], w.val[i:]...)
	}
}

// Mean returns the windowed mean TTFT and how many samples it rests on.
func (w *ttftWindow) Mean(now time.Time) (time.Duration, int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.expireLocked(now)
	if len(w.val) == 0 {
		return 0, 0
	}
	return w.sum / time.Duration(len(w.val)), len(w.val)
}

// Open reports whether the admission gate should let another session in.
//
// An empty window opens it. On an idle fleet that is plainly right, and on a
// stalled one the sessions already admitted are the evidence — holding the gate
// shut because nothing has COMPLETED would let one slow request freeze the ramp
// for the rest of the run, which reports the stall as a capacity ceiling.
func (w *ttftWindow) Open(now time.Time, limit time.Duration) bool {
	mean, n := w.Mean(now)
	if n == 0 {
		return true
	}
	return mean < limit
}

// ---------------------------------------------------------------- pacing

// sessionPacer turns a session's recorded timestamps into wall-clock targets,
// so the gaps between its requests are the gaps its user actually left.
//
// The captured gap is think time plus the ORIGINAL response time, which this
// run's fleet will not reproduce. Treating it as a floor rather than a schedule
// is the honest reading: a session cannot issue turn i+1 before it has the
// answer to turn i, and if this fleet is slower than the original the session
// simply falls behind rather than firing a backlog.
type sessionPacer struct {
	enabled bool
	origin  time.Time // wall clock when the session was admitted
	base    time.Time // recorded timestamp of the session's first request
	clk     *skipClock
}

func newSessionPacer(startTs string, now time.Time, enabled bool, clk *skipClock) *sessionPacer {
	p := &sessionPacer{enabled: enabled, origin: now, clk: clk}
	if !enabled {
		return p
	}
	if t, err := time.Parse(time.RFC3339Nano, startTs); err == nil {
		p.base = t
	} else {
		p.enabled = false // no origin to measure offsets from
	}
	return p
}

// dueAt is the wall-clock instant a request recorded at ts should be sent.
func (p *sessionPacer) dueAt(ts string) (time.Time, bool) {
	if !p.enabled {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339Nano, ts)
	if err != nil {
		return time.Time{}, false
	}
	off := t.Sub(p.base)
	if off < 0 {
		off = 0
	}
	return p.origin.Add(off), true
}

// Wait blocks until the request is due, or returns immediately if it is already
// late — which is the common case on a fleet slower than the capture.
func (p *sessionPacer) Wait(ctx context.Context, ts string) {
	due, ok := p.dueAt(ts)
	if !ok {
		return
	}
	for {
		now := p.clk.Now()
		if !due.After(now) {
			return
		}
		p.clk.enterWait(due)
		t := time.NewTimer(due.Sub(now))
		select {
		case <-ctx.Done():
			t.Stop()
			p.clk.leaveWait()
			return
		case <-t.C:
			p.clk.leaveWait()
			return
		case <-p.clk.skipped():
			// The clock jumped forward while every session was idle; re-read it
			// rather than trusting the timer we set against the old reading.
			t.Stop()
			p.clk.leaveWait()
		}
	}
}

// ---------------------------------------------------------------- skip clock

// skipClock is wall time plus a skew that only ever moves forward, and only
// when the whole run is idle: nothing in flight and every active session
// waiting on its next turn.
//
// This is what keeps a replay from spending its wall-clock budget on think
// time. The captured traffic has a median gap of about eight seconds and a mean
// of four minutes, so a faithful replay of an hour of capture is mostly a
// replay of nobody typing. Skipping is safe precisely because the condition
// requires an idle fleet — no in-flight request has its timing distorted,
// because there are none.
//
// At the session counts this driver reaches the condition almost never holds,
// which is the intended outcome: the skip matters while the run is ramping and
// stops mattering once it is loaded.
type skipClock struct {
	mu           sync.Mutex
	skew         time.Duration
	inflight     int
	waiting      int // sessions parked on a future due time
	active       int // sessions admitted and not yet finished
	earliest     time.Time
	haveEarliest bool
	ch           chan struct{} // closed and replaced on each jump
	enabled      bool
}

func newSkipClock(enabled bool) *skipClock {
	return &skipClock{ch: make(chan struct{}), enabled: enabled}
}

func (c *skipClock) Now() time.Time {
	if c == nil {
		return time.Now()
	}
	c.mu.Lock()
	skew := c.skew
	c.mu.Unlock()
	//clockexempt: the benchmark's own wall clock; the router's clock rules do not reach here
	return time.Now().Add(skew)
}

func (c *skipClock) skipped() <-chan struct{} {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ch
}

func (c *skipClock) AddSession(n int) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.active += n
	c.mu.Unlock()
	c.maybeSkip()
}

func (c *skipClock) AddInflight(n int) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.inflight += n
	c.mu.Unlock()
	if n < 0 {
		c.maybeSkip()
	}
}

func (c *skipClock) enterWait(due time.Time) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.waiting++
	if !c.haveEarliest || due.Before(c.earliest) {
		c.earliest = due
		c.haveEarliest = true
	}
	c.mu.Unlock()
	c.maybeSkip()
}

func (c *skipClock) leaveWait() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.waiting--
	if c.waiting <= 0 {
		c.waiting = 0
		c.haveEarliest = false
	}
	c.mu.Unlock()
}

// maybeSkip jumps the clock to the earliest pending due time, but only with the
// fleet fully idle and every active session parked.
func (c *skipClock) maybeSkip() {
	if c == nil || !c.enabled {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.inflight != 0 || c.active == 0 || c.waiting < c.active || !c.haveEarliest {
		return
	}
	//clockexempt: the benchmark's own wall clock
	now := time.Now().Add(c.skew)
	if !c.earliest.After(now) {
		return
	}
	c.skew += c.earliest.Sub(now)
	c.haveEarliest = false
	close(c.ch)
	c.ch = make(chan struct{})
}

// Skew is how much dead time the run has skipped, for the summary.
func (c *skipClock) Skew() time.Duration {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.skew
}
