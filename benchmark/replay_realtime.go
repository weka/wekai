package benchmark

import (
	"context"
	"fmt"
	"sort"
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
// advance. The governor adds a batch per tick while windowed TTFT stays under the
// limit and pauses above it, so the fleet's latency is the only throttle.

// resolveReplayTiming decides whether a run schedules by think time.
func resolveReplayTiming(mode string, hdr RouterReplayHeader) (bool, error) {
	has := hdr.Summary.Timing != nil && hdr.Summary.Timing.RequestsWithThink > 0
	switch mode {
	case "", "auto":
		return has, nil
	case "absolute":
		return false, nil
	case "think":
		if !has {
			return false, fmt.Errorf("--replay-timing=think needs a replay-v4 file with think_ms; this file (%s) has none -- use --replay-timing=absolute or regenerate it with 'wekai router replay-prepare'", hdr.Schema)
		}
		return true, nil
	}
	return false, fmt.Errorf("--replay-timing: unknown mode %q (think, absolute)", mode)
}

func describeReplayTiming(think bool, requested string, hdr RouterReplayHeader) string {
	how := "default"
	if requested == "think" || requested == "absolute" {
		how = "requested"
	}
	if think {
		return fmt.Sprintf("think (%s; each turn due at its reference's end on this run's clock + captured think_ms; file %s)", how, hdr.Schema)
	}
	if hdr.Summary.Timing == nil || hdr.Summary.Timing.RequestsWithThink == 0 {
		return fmt.Sprintf("absolute (%s; file %s has no think_ms; each turn due at its captured offset from session start)", how, hdr.Schema)
	}
	return fmt.Sprintf("absolute (%s; each turn due at its captured offset from session start)", how)
}

func (cfg *AutoBenchmarkConfig) setSeriesDefaults() {
	if cfg.MaxSeries <= 0 && cfg.AdmitEvery <= 0 {
		cfg.MaxSeries = 64
	}
	if cfg.StartSeries <= 0 {
		cfg.StartSeries = 1
	}
	if cfg.MaxSeries > 0 && cfg.StartSeries > cfg.MaxSeries {
		cfg.StartSeries = cfg.MaxSeries
	}
}

// retireThreshold is the gate statistic above which --ttft-retire retires a
// finished session's slot: --ttft-limit raised by --ttft-retire-margin.
func retireThreshold(cfg AutoBenchmarkConfig) time.Duration {
	return time.Duration(float64(cfg.TTFTLimit) * (1 + cfg.TTFTRetireMargin))
}

// shouldRetireSlot reports whether a slot whose session just ended should stop
// instead of pulling the next one. Only with --ttft-retire on a realtime run,
// and only while the windowed statistic is above the retire threshold; between
// the limit and the threshold sessions are still replaced (the gate merely
// stops adding), and an empty window never retires.
func shouldRetireSlot(cfg AutoBenchmarkConfig, window *ttftWindow, now time.Time) bool {
	if !cfg.TTFTRetire || !cfg.ReplayRealtime || window == nil {
		return false
	}
	s := window.Stats(now)
	return s.N > 0 && s.pick(cfg.TTFTLimitStat) > retireThreshold(cfg)
}

// admitSeries adds --admit-count slots every --admit-every while the TTFT gate
// is open. spawn gets the new slot's 1-based number. Without --ttft-retire
// slots only ever grow, so --max-series caps the number spawned and admission
// ends there (capped). With it, retired slots free room: active(), the slots
// still running, is what --max-series caps, and admission keeps ticking so the
// run can regrow after TTFT recovers.
func admitSeries(ctx context.Context, cfg AutoBenchmarkConfig, window *ttftWindow, spawn func(int), capped func(), active func() int) {
	if cfg.AdmitEvery <= 0 {
		return
	}
	ticker := time.NewTicker(cfg.AdmitEvery)
	defer ticker.Stop()
	next := cfg.StartSeries
	retire := cfg.TTFTRetire && active != nil
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		running := next
		if retire {
			running = active()
		}
		if cfg.MaxSeries > 0 && running >= cfg.MaxSeries {
			if retire {
				continue
			}
			capped()
			return
		}
		if cfg.ReplayRealtime && !window.Open(time.Now(), cfg.TTFTLimit, cfg.TTFTLimitStat) {
			continue
		}
		count := max(1, cfg.AdmitCount)
		if cfg.MaxSeries > 0 {
			count = min(count, cfg.MaxSeries-running)
		}
		for range count {
			if ctx.Err() != nil {
				return
			}
			next++
			spawn(next)
		}
	}
}

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
	st := w.Stats(now)
	return st.Mean, st.N
}

// ttftStats is every statistic the gate could be asked for, over one window.
//
// All of them are computed whether or not they are the one gating, because the
// cost is a sort of a few hundred samples and the value is that ONE run reports
// what its plateau would have been under each. The alternative is an eight-hour
// arm per statistic, and the choice moves the answer by about a factor of two.
type ttftStats struct {
	Mean, P50, P95, P99 time.Duration
	N                   int
}

// Stats computes them all over the current window.
func (w *ttftWindow) Stats(now time.Time) ttftStats {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.expireLocked(now)
	n := len(w.val)
	if n == 0 {
		return ttftStats{}
	}
	sorted := make([]time.Duration, n)
	copy(sorted, w.val)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	q := func(p float64) time.Duration {
		i := int(float64(n) * p)
		if i >= n {
			i = n - 1
		}
		return sorted[i]
	}
	return ttftStats{
		Mean: w.sum / time.Duration(n),
		P50:  q(0.50), P95: q(0.95), P99: q(0.99),
		N: n,
	}
}

// pick returns the statistic named, defaulting to the mean.
func (s ttftStats) pick(name string) time.Duration {
	switch name {
	case "p50":
		return s.P50
	case "p95":
		return s.P95
	case "p99":
		return s.P99
	default:
		return s.Mean
	}
}

// String renders every statistic with the gating one named, because
// "ttft_win=6.8s" does not say what it is and under a selectable gate that is
// genuinely ambiguous.
func (s ttftStats) String(gating string) string {
	if s.N == 0 {
		return "ttft_win(" + gating + ")=-"
	}
	return fmt.Sprintf("ttft_win(%s)=%s [mean=%s p50=%s p95=%s p99=%s n=%d]",
		gating, s.pick(gating).Round(time.Millisecond),
		s.Mean.Round(time.Millisecond), s.P50.Round(time.Millisecond),
		s.P95.Round(time.Millisecond), s.P99.Round(time.Millisecond), s.N)
}

// Open reports whether the admission gate should let another session in.
//
// An empty window opens it. On an idle fleet that is plainly right, and on a
// stalled one the sessions already admitted are the evidence — holding the gate
// shut because nothing has COMPLETED would let one slow request freeze the ramp
// for the rest of the run, which reports the stall as a capacity ceiling.
func (w *ttftWindow) Open(now time.Time, limit time.Duration, stat string) bool {
	s := w.Stats(now)
	if s.N == 0 {
		return true
	}
	return s.pick(stat) < limit
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
//
// Two schedules exist. The absolute one (above) targets offsets on the captured
// clock. The think one, used when think is set and a request carries think_ms,
// targets the END of the request's reference on THIS run's clock plus the
// captured think time: the user's pause after an answer is preserved whether
// our answer arrived sooner or later than the original, which is what a replay
// against a different fleet has to hold fixed.
type sessionPacer struct {
	think   bool // schedule by think_ms where the request has one
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

// thinkDue is the instant a request is due under the think schedule: ref plus
// the request's captured think time. ok is false when the request has to fall
// back to the absolute schedule -- think mode off, no think_ms on the request
// (a replay-v3 file), or no reference (the referenced request never completed).
func (p *sessionPacer) thinkDue(req RouterReplayRequest, ref time.Time) (time.Time, bool) {
	if !p.enabled || !p.think || req.ThinkMs == nil || ref.IsZero() {
		return time.Time{}, false
	}
	return ref.Add(time.Duration(*req.ThinkMs) * time.Millisecond), true
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
	return p.waitUntil(ctx, due)
}

// waitUntil blocks until due and reports how late it already was on entry.
func (p *sessionPacer) waitUntil(ctx context.Context, due time.Time) time.Duration {
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

	// Think-schedule accounting. A request paced by think_ms is due at the end
	// of its reference plus the captured think time, so it is "late" by however
	// long the concurrency gate (if one is set) held it past that moment; the
	// realized think (send time minus reference end) is reported against the
	// captured one. Response time is ours against the capture's total_ms.
	thinkN       int64
	thinkLate    int64
	thinkLateSum time.Duration
	thinkLateMax time.Duration
	thinkReal    time.Duration
	thinkOrig    time.Duration
	respN        int64
	respOurs     time.Duration
	respOrig     time.Duration
}

// lateTolerance is how far past its due time a turn has to go before it counts
// as late. A think-paced turn is released by a timer, so scheduling jitter of a
// millisecond or two lands on every one of them; counting that made the late
// share 100% on a fleet that was keeping up.
const lateTolerance = 50 * time.Millisecond

// observeThink records one think-paced request: lateness past its due time
// (gate wait included), the think it actually got, and the captured one.
func (l *pacingLag) observeThink(late, realized, original time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.thinkN++
	l.thinkReal += realized
	l.thinkOrig += original
	if late > lateTolerance {
		l.thinkLate++
		l.thinkLateSum += late
		if late > l.thinkLateMax {
			l.thinkLateMax = late
		}
	}
}

// observeResponse records how long this run's fleet took to answer a request
// against how long the capture's did.
func (l *pacingLag) observeResponse(ours, orig time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.respN++
	l.respOurs += ours
	l.respOrig += orig
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
	if d <= lateTolerance {
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
	if l.n == 0 && l.thinkN == 0 {
		return ""
	}
	var mean time.Duration
	if l.late > 0 {
		mean = l.sum / time.Duration(l.late)
	}
	out := ""
	if l.thinkN > 0 {
		var lm time.Duration
		if l.thinkLate > 0 {
			lm = l.thinkLateSum / time.Duration(l.thinkLate)
		}
		out = fmt.Sprintf("think-paced=%d think_real=%s think_orig=%s late=%d (%.1f%%, >50ms gate/queue wait) mean_late=%s max_late=%s",
			l.thinkN, (l.thinkReal / time.Duration(l.thinkN)).Round(time.Millisecond),
			(l.thinkOrig / time.Duration(l.thinkN)).Round(time.Millisecond),
			l.thinkLate, 100*float64(l.thinkLate)/float64(l.thinkN),
			lm.Round(time.Millisecond), l.thinkLateMax.Round(time.Millisecond))
	}
	if l.respN > 0 && l.respOrig > 0 {
		out += fmt.Sprintf(" resp_ratio=%.2f (ours %s / orig %s mean, n=%d)",
			float64(l.respOurs)/float64(l.respOrig),
			(l.respOurs / time.Duration(l.respN)).Round(time.Millisecond),
			(l.respOrig / time.Duration(l.respN)).Round(time.Millisecond), l.respN)
	}
	if l.n > 0 {
		if out != "" {
			out += " | abs: "
		}
		out += fmt.Sprintf(
			"paced=%d late=%d (%.1f%%, >50ms) mean_late=%s max_late=%s >1s=%d >10s=%d >1m=%d >10m=%d",
			l.n, l.late, 100*float64(l.late)/float64(l.n), mean.Round(time.Millisecond),
			l.max.Round(time.Millisecond), l.overs[0], l.overs[1], l.overs[2], l.overs[3])
	}
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

// admissionGate describes the admission governor now: which statistic over
// which window is compared with the limit, whether that opens the gate, and
// when the next admission tick is. nil when no governor runs. admitStart is
// when the admission ticker started.
func admissionGate(cfg AutoBenchmarkConfig, st *autoState, admitStart time.Time) *liveGate {
	if cfg.AdmitEvery <= 0 && !cfg.ReplayRealtime {
		return nil
	}
	now := time.Now()
	stat := cfg.TTFTLimitStat
	if stat == "" {
		stat = "mean"
	}
	g := &liveGate{
		Enabled: cfg.ReplayRealtime, Stat: stat, LimitMs: float64(cfg.TTFTLimit.Milliseconds()),
		WindowS: st.ttft.window.Seconds(), AdmitCount: max(1, cfg.AdmitCount), EveryS: cfg.AdmitEvery.Seconds(),
	}
	s := st.ttft.Stats(now)
	g.N = s.N
	g.MeanMs, g.P50Ms = float64(s.Mean.Milliseconds()), float64(s.P50.Milliseconds())
	g.P95Ms, g.P99Ms = float64(s.P95.Milliseconds()), float64(s.P99.Milliseconds())
	g.ValueMs = float64(s.pick(stat).Milliseconds())
	g.Open = !cfg.ReplayRealtime || s.N == 0 || s.pick(stat) < cfg.TTFTLimit
	st.mu.Lock()
	g.Capped = cfg.MaxSeries > 0 && st.series >= cfg.MaxSeries
	st.mu.Unlock()
	if cfg.TTFTRetire {
		g.RetireMs = float64(retireThreshold(cfg).Milliseconds())
		g.Retired = st.seriesRetired.Load()
		g.Capped = cfg.MaxSeries > 0 && int(st.activeReplayWorkers.Load()) >= cfg.MaxSeries
	}
	if cfg.AdmitEvery > 0 {
		ticks := now.Sub(admitStart) / cfg.AdmitEvery
		g.NextInS = (admitStart.Add((ticks + 1) * cfg.AdmitEvery)).Sub(now).Seconds()
	}
	return g
}
