package benchmark

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// openAIContinuousUsage asks OpenAI-compatible (vLLM) streams for per-chunk
// usage; set once at run start when --decode-rate-min is on.
var openAIContinuousUsage atomic.Bool

// decodeRate is the fleet's per-request decode speed over a trailing window:
// output tokens streamed during the window divided by the request-seconds spent
// decoding in it (time between a request's first token and its end, or now
// while it still runs). It is a time-weighted mean of each streaming request's
// tokens/s, so ten requests at 30 tok/s read 30 whatever their lengths.
//
// Completed requests count with their exact output tokens (final usage),
// spread evenly over their decode span. Running requests count the tokens
// streamed so far: exact when the server sends per-chunk usage
// (stream_options.continuous_usage_stats, requested whenever this gate is on),
// else one per content delta, which undercounts multi-token chunks.
type decodeRate struct {
	mu       sync.Mutex
	window   time.Duration
	done     []decodeSpan // completed, ordered by end
	inflight map[*liveProgress]struct{}
}

type decodeSpan struct {
	first, end int64 // unix ns
	tokens     int64
}

// newDecodeRateFor returns the run's tracker, or nil when --decode-rate-min is
// off (every method is a no-op on nil).
func newDecodeRateFor(cfg AutoBenchmarkConfig) *decodeRate {
	if !cfg.ReplayRealtime || cfg.DecodeRateMin <= 0 {
		return nil
	}
	openAIContinuousUsage.Store(true)
	return newDecodeRate(cfg.DecodeRateWindow)
}

func newDecodeRate(window time.Duration) *decodeRate {
	if window <= 0 {
		window = 30 * time.Second
	}
	return &decodeRate{window: window, inflight: map[*liveProgress]struct{}{}}
}

func (d *decodeRate) begin(p *liveProgress) {
	if d == nil || p == nil {
		return
	}
	d.mu.Lock()
	d.inflight[p] = struct{}{}
	d.mu.Unlock()
}

// end moves a request from running to completed. tokens is its final output
// count (usage), or 0 when unknown, in which case the streamed count is used.
func (d *decodeRate) end(p *liveProgress, endAt time.Time, tokens int64) {
	if d == nil || p == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	delete(d.inflight, p)
	first := p.firstNs.Load()
	if first == 0 {
		return // never streamed a token: no decode phase
	}
	if tokens <= 0 {
		tokens = p.out.Load()
	}
	d.done = append(d.done, decodeSpan{first: first, end: endAt.UnixNano(), tokens: tokens})
	d.pruneLocked(endAt.UnixNano())
}

func (d *decodeRate) pruneLocked(nowNs int64) {
	cut := nowNs - int64(d.window)
	i := 0
	for i < len(d.done) && d.done[i].end < cut {
		i++
	}
	if i > 0 {
		d.done = append(d.done[:0], d.done[i:]...)
	}
}

// rate returns the per-request decode tokens/s over the window ending now and
// the decoding request-seconds it is based on.
func (d *decodeRate) rate(now time.Time) (tokPerSec, reqSeconds float64) {
	if d == nil {
		return 0, 0
	}
	nowNs := now.UnixNano()
	from := nowNs - int64(d.window)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.pruneLocked(nowNs)
	var tok, secs float64
	add := func(first, end int64, tokens float64) {
		if end <= first {
			return
		}
		lo := max(first, from)
		if end <= lo {
			return
		}
		overlap := float64(end - lo)
		tok += tokens * overlap / float64(end-first)
		secs += overlap / 1e9
	}
	for _, s := range d.done {
		add(s.first, s.end, float64(s.tokens))
	}
	for p := range d.inflight {
		if first := p.firstNs.Load(); first != 0 {
			add(first, nowNs, float64(p.out.Load()))
		}
	}
	if secs <= 0 {
		return 0, 0
	}
	return tok / secs, secs
}

// decodeGateMinSeconds is how much decoding the window must hold before the
// gate trusts its rate: with less, nothing is decoding and holding requests
// back could only stall the run.
const decodeGateMinSeconds = 5.0

// open reports whether a request may be sent: the gate is off, there is too
// little decoding in the window to judge, or the rate is at or above min.
func (d *decodeRate) open(now time.Time, min float64) bool {
	if d == nil || min <= 0 {
		return true
	}
	r, secs := d.rate(now)
	return secs < decodeGateMinSeconds || r >= min
}

// hold blocks until the gate is open (or ctx ends) and returns how long it held.
func (d *decodeRate) hold(ctx context.Context, min float64) time.Duration {
	if d == nil || min <= 0 {
		return 0
	}
	start := time.Now()
	for !d.open(time.Now(), min) {
		t := time.NewTimer(250 * time.Millisecond)
		select {
		case <-ctx.Done():
			t.Stop()
			return time.Since(start)
		case <-t.C:
		}
	}
	return time.Since(start)
}
