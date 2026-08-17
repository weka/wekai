package benchmark

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
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

// ttftWindow is the ARITHMETIC MEAN TTFT over a trailing wall-clock window.
//
// Mean and not median, deliberately, and it decides the answer rather than
// decorating it. This workload's TTFT is heavily skewed — a median of 4.3s
// against a mean of 8.7s, p99 52s, max 193s — so a mean gate closes at roughly
// half the session count a median gate would, and one 193s request in a window
// of 300 moves it 0.64s on its own. The gate is therefore tail-driven, which is
// the intended reading: the tail is real callers waiting, and a fleet whose p99
// is 52s is not comfortably carrying that load however healthy its median
// looks. Anton's call, taken with the distribution in front of him.
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

// offsetOf is how far into the captured conversation a turn sits, measured from
// the session's own start.
func (p *sessionPacer) offsetOf(ts string) (time.Duration, bool) {
	due, ok := p.dueAt(ts)
	if !ok {
		return 0, false
	}
	return due.Sub(p.origin), true
}

// Wait blocks until the request is due, and reports how LATE it was — zero if it
// waited, positive if the moment had already passed.
//
// Lateness is the mode's defining measurement and nothing else exposes it. A
// fleet slower than the capture makes every session fall behind its own
// schedule, and the run then replays less captured time than it spent: eight
// hours of wall clock covering three hours of conversation is a different
// experiment from the one that was asked for, and the totals look identical
// either way. Only the lag says which happened.
func (p *sessionPacer) Wait(ctx context.Context, ts string) time.Duration {
	due, ok := p.dueAt(ts)
	if !ok {
		return 0
	}
	if late := p.clk.Now().Sub(due); late > 0 {
		return late
	}
	for {
		now := p.clk.Now()
		if !due.After(now) {
			return 0
		}
		p.clk.enterWait(due)
		t := time.NewTimer(due.Sub(now))
		select {
		case <-ctx.Done():
			t.Stop()
			p.clk.leaveWait()
			return 0
		case <-t.C:
			p.clk.leaveWait()
			return 0
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

// ---------------------------------------------------------------- lag

// pacingLag accumulates how far behind their captured schedule requests are
// going out.
//
// Reported rather than inferred, because the alternative is unanswerable after
// the fact: a run whose sessions all fell an hour behind produces the same
// request count, token totals and cache curve as one that kept up, and differs
// only in how much captured conversation it actually covered.
type pacingLag struct {
	mu    sync.Mutex
	n     int64
	late  int64 // requests that went out after their due time
	sum   time.Duration
	max   time.Duration
	overs [4]int64 // > 1s, > 10s, > 1m, > 10m

	// Per-session coverage: how much captured conversation was replayed against
	// how much wall time the session was alive for. 1.0 means an hour of wall
	// clock replayed an hour of capture.
	//
	// What it actually measures is THIS FLEET'S SPEED AGAINST THE CAPTURE
	// FLEET'S, and that is not what it was built to be. The pacer targets
	// absolute offsets, so a session hits every target exactly while its turns
	// finish inside the captured gaps and falls behind proportionally once they
	// do not. Fidelity therefore settles near captured_gap / our_turn_duration —
	// a ratio of two fleets, with load almost absent from it.
	//
	// Measured: flat at 0.36-0.50 across a twenty-fold range of admitted slots,
	// including 60 slots on a nearly idle fleet, and never once above 0.50 in 44
	// samples. If contention drove it, it would fall as load rose. It does not.
	// This corpus has a median start-to-start gap of 7.5s, and turns here take
	// upwards of twice that whatever else is happening.
	//
	// Two consequences worth knowing before anyone builds on it. It is useless
	// as an admission signal — at 0.9 it admits nothing and at 0.5 it is
	// indistinguishable from no gate, because it does not respond to the control
	// variable. And it is a good instrument for COMPARING fleets on one corpus,
	// better than throughput in one respect: it is normalised against the same
	// recorded workload, so two fleets' numbers mean the same thing.
	sessions int64
	covered  time.Duration
	wall     time.Duration

	// Sessions still running, included in the ratio.
	//
	// Computing fidelity over finished sessions ONLY is survivorship bias, and
	// on this workload it is severe rather than subtle: a session that finishes
	// early is a short one, and the long conversations — the ones carrying the
	// four-minute think gaps this mode exists to reproduce — are precisely the
	// ones still running and therefore excluded. The ratio then moves with the
	// mix of what has retired rather than with how the fleet is doing, and it
	// can fall by a third between two samples without the fleet changing at all.
	live   map[int64]*liveSession
	nextID int64
}

// liveSession is a session in progress: how far into its capture it has reached
// so far, and when it started.
type liveSession struct {
	covered *atomic.Int64
	origin  time.Time
}

// beginSession registers a running session and returns its handle.
func (l *pacingLag) beginSession(covered *atomic.Int64, origin time.Time) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.live == nil {
		l.live = map[int64]*liveSession{}
	}
	l.nextID++
	l.live[l.nextID] = &liveSession{covered: covered, origin: origin}
	return l.nextID
}

// observeSession records one finished session's progress through its capture.
func (l *pacingLag) observeSession(id int64, covered, wall time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.live, id)
	l.sessions++
	l.covered += covered
	l.wall += wall
}

func (l *pacingLag) observe(d time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.n++
	if d <= 0 {
		return
	}
	l.late++
	l.sum += d
	if d > l.max {
		l.max = d
	}
	for i, t := range [4]time.Duration{time.Second, 10 * time.Second, time.Minute, 10 * time.Minute} {
		if d > t {
			l.overs[i]++
		}
	}
}

// summary is one line, empty when nothing was paced.
func (l *pacingLag) summary(now time.Time) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.n == 0 {
		return ""
	}
	var mean time.Duration
	if l.late > 0 {
		mean = l.sum / time.Duration(l.late)
	}
	out := fmt.Sprintf(
		"paced=%d late=%d (%.1f%%) mean_late=%s max_late=%s >1s=%d >10s=%d >1m=%d >10m=%d",
		l.n, l.late, 100*float64(l.late)/float64(l.n), mean.Round(time.Millisecond),
		l.max.Round(time.Millisecond), l.overs[0], l.overs[1], l.overs[2], l.overs[3])
	// Live sessions count too. Their coverage so far against the time they have
	// so far been alive is exactly as meaningful as a finished session's, and
	// leaving them out is what made the ratio a statement about which sessions
	// happened to be short.
	// `now` MUST come from the same clock the origins did — the skip clock, not
	// wall time. Sessions are stamped with skipClk.Now(), which runs ahead of
	// the wall by however much idle time was compressed, so a wall-clock `now`
	// makes every session admitted after a skip look as though it started in the
	// future. The guard below then dropped them from the DENOMINATOR while their
	// coverage still counted in the numerator, and fidelity came out above 1.0 —
	// which is not a physically possible number, since a turn cannot fire before
	// its captured offset.
	covered, wall := l.covered, l.wall
	for _, s := range l.live {
		covered += time.Duration(s.covered.Load())
		if d := now.Sub(s.origin); d > 0 {
			wall += d
		}
	}
	if wall > 0 {
		out += fmt.Sprintf(" | sessions=%d(%d done) covered=%s of %s alive (fidelity %.2f)",
			int64(len(l.live))+l.sessions, l.sessions,
			covered.Round(time.Second), wall.Round(time.Second),
			float64(covered)/float64(wall))
	}
	return out
}
