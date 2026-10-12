package benchmark

import (
	"math"
	"sort"
	"sync/atomic"
	"time"
)

// Token-rate and TTFT time series for the timeline.
//
// Attributing a request's tokens where the request COMPLETES makes a long
// request look like a spike at its end. Instead, prompt tokens land when
// prefill finishes (the first token) and output tokens are spread uniformly
// across the decode interval (first token -> end), so a three-minute decode
// contributes a steady rate for three minutes. In the live view a request still
// running contributes the same way from what the client has seen of it.

// rateAccum accumulates tokens per time bucket (bucket i covers
// [i*bw, (i+1)*bw) ms from the arm's start). It is bounded: past `max` buckets
// the bucket width doubles and pairs merge, so a run of any length costs at
// most max buckets and stays whole at coarser resolution.
type rateAccum struct {
	bw  float64
	max int
	// in is total prompt tokens; ic/ie/ig are its computed / external+host /
	// GPU-cache parts (the remainder is of unknown source).
	in, ic, ie, ig, out []float64
}

func newRateAccum(bw float64, max int) *rateAccum {
	if bw <= 0 {
		bw = 1000
	}
	if max < 64 {
		max = 64
	}
	return &rateAccum{bw: bw, max: max}
}

func (r *rateAccum) series() [5]*[]float64 {
	return [5]*[]float64{&r.in, &r.ic, &r.ie, &r.ig, &r.out}
}

func (r *rateAccum) coarsen() {
	for _, s := range r.series() {
		v := *s
		n := (len(v) + 1) / 2
		for i := 0; i < n; i++ {
			v[i] = v[2*i]
			if 2*i+1 < len(v) {
				v[i] += v[2*i+1]
			}
		}
		*s = v[:n]
	}
	r.bw *= 2
}

// bucket returns the index for time t (ms), growing storage and coarsening as
// needed.
func (r *rateAccum) bucket(t float64) int {
	if t < 0 {
		t = 0
	}
	for int(t/r.bw) >= r.max {
		r.coarsen()
	}
	i := int(t / r.bw)
	for _, s := range r.series() {
		for len(*s) <= i {
			*s = append(*s, 0)
		}
	}
	return i
}

func (r *rateAccum) impulse(t float64, in, ic, ie, ig float64) {
	i := r.bucket(t)
	r.in[i] += in
	r.ic[i] += ic
	r.ie[i] += ie
	r.ig[i] += ig
}

// spread adds tokens uniformly over [t0, t1] ms.
func (r *rateAccum) spread(t0, t1, tokens float64) {
	if tokens <= 0 {
		return
	}
	if t1 <= t0 {
		r.out[r.bucket(t0)] += tokens
		return
	}
	r.bucket(t1) // size (and coarsen) for the whole interval first
	if t0 < 0 {
		t0 = 0
	}
	per := tokens / (t1 - t0)
	for i := int(t0 / r.bw); i <= int(t1/r.bw) && i < len(r.out); i++ {
		lo, hi := math.Max(t0, float64(i)*r.bw), math.Min(t1, float64(i+1)*r.bw)
		if hi > lo {
			r.out[i] += per * (hi - lo)
		}
	}
}

// tokenSplit is a request's prompt tokens by source.
type tokenSplit struct{ total, computed, external, gpu float64 }

func splitOf(q *timelineReq) tokenSplit {
	s := tokenSplit{total: float64(q.In + q.Cached)}
	if len(q.Src) == 4 { // gpu, host, external, computed
		s.gpu, s.external, s.computed = float64(q.Src[0]), float64(q.Src[1]+q.Src[2]), float64(q.Src[3])
	}
	return s
}

// firstTokenAt is when the request's first token arrived, in the arm's ms:
// sent + client TTFT (which includes any 429 backoff), or the end when the
// request never reported one.
func firstTokenAt(q *timelineReq) float64 {
	end := q.Start + q.E2E
	if q.TTFT <= 0 {
		return end
	}
	if f := q.Start + q.CW + q.TTFT; f < end {
		return f
	}
	return end
}

// addRequest attributes a completed request.
func (r *rateAccum) addRequest(q *timelineReq) {
	if q.Flags&flagError != 0 {
		return
	}
	first := firstTokenAt(q)
	s := splitOf(q)
	if s.total > 0 {
		r.impulse(first, s.total, s.computed, s.external, s.gpu)
	}
	r.spread(first, q.Start+q.E2E, float64(q.Out))
}

// windowSum is the tokens inside [from, to] ms (partial buckets pro rata).
func (r *rateAccum) windowSum(from, to float64) (in, out float64) {
	if to <= from {
		return 0, 0
	}
	for i := int(math.Max(0, from) / r.bw); i < len(r.in) && float64(i)*r.bw < to; i++ {
		lo, hi := math.Max(from, float64(i)*r.bw), math.Min(to, float64(i+1)*r.bw)
		if hi <= lo {
			continue
		}
		f := (hi - lo) / r.bw
		in += r.in[i] * f
		out += r.out[i] * f
	}
	return in, out
}

// rateSeries is the wire form: tokens per bucket of width Bw ms.
type rateSeries struct {
	Bw  float64 `json:"bw"`
	In  []int   `json:"in"`
	IC  []int   `json:"ic"`
	IE  []int   `json:"ie"`
	IG  []int   `json:"ig"`
	Out []int   `json:"out"`
}

// export merges buckets by an integer factor so at most maxN remain.
func (r *rateAccum) export(maxN int) *rateSeries {
	f := 1
	if maxN > 0 && len(r.in) > maxN {
		f = (len(r.in) + maxN - 1) / maxN
	}
	n := (len(r.in) + f - 1) / f
	out := &rateSeries{Bw: r.bw * float64(f), In: make([]int, n), IC: make([]int, n), IE: make([]int, n), IG: make([]int, n), Out: make([]int, n)}
	acc := func(dst []int, src []float64) {
		for i, v := range src {
			dst[i/f] += int(math.Round(v))
		}
	}
	acc(out.In, r.in)
	acc(out.IC, r.ic)
	acc(out.IE, r.ie)
	acc(out.IG, r.ig)
	acc(out.Out, r.out)
	return out
}

// ---------------------------------------------------------------- TTFT

// ttftSeries is TTFT statistics over a trailing window, sampled over time (ms
// values, T in ms from the arm's start).
type ttftSeries struct {
	T       []float64 `json:"t"`
	Mean    []float64 `json:"mean"`
	P50     []float64 `json:"p50"`
	P95     []float64 `json:"p95"`
	P99     []float64 `json:"p99"`
	WindowS float64   `json:"window_s"`
	LimitMs float64   `json:"limit_ms,omitempty"`
	Stat    string    `json:"stat,omitempty"`
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(math.Ceil(p*float64(len(sorted)))) - 1
	return sorted[max(0, min(i, len(sorted)-1))]
}

// ttftOverTime computes windowed TTFT stats from completed requests, at about
// `points` evenly spaced instants. window is in ms.
func ttftOverTime(reqs []timelineReq, span, window float64, points int) *ttftSeries {
	type ev struct{ at, ttft float64 }
	var evs []ev
	for i := range reqs {
		q := &reqs[i]
		if q.Flags&flagError != 0 || q.TTFT <= 0 {
			continue
		}
		evs = append(evs, ev{firstTokenAt(q), q.TTFT})
	}
	out := &ttftSeries{WindowS: window / 1000}
	if len(evs) == 0 || span <= 0 {
		return out
	}
	sort.Slice(evs, func(i, j int) bool { return evs[i].at < evs[j].at })
	step := math.Max(span/float64(points), 1000)
	lo, hi := 0, 0
	for t := step; t <= span+step; t += step {
		for hi < len(evs) && evs[hi].at <= t {
			hi++
		}
		for lo < hi && evs[lo].at <= t-window {
			lo++
		}
		if hi == lo {
			continue
		}
		w := make([]float64, 0, hi-lo)
		var sum float64
		for _, e := range evs[lo:hi] {
			w = append(w, e.ttft)
			sum += e.ttft
		}
		sort.Float64s(w)
		out.T = append(out.T, t)
		out.Mean = append(out.Mean, sum/float64(len(w)))
		out.P50 = append(out.P50, percentile(w, .5))
		out.P95 = append(out.P95, percentile(w, .95))
		out.P99 = append(out.P99, percentile(w, .99))
	}
	return out
}

// ---------------------------------------------------------------- live progress

// liveProgress is what the client has seen of a request still running: when its
// first token arrived and how many stream deltas (about one token each) since.
// All methods are safe on nil.
type liveProgress struct {
	firstNs   atomic.Int64
	out       atomic.Int64
	promptEst atomic.Int64
}

func (p *liveProgress) markFirst() {
	if p != nil {
		p.firstNs.CompareAndSwap(0, time.Now().UnixNano())
	}
}

func (p *liveProgress) addOut(n int64) {
	if p != nil {
		p.out.Add(n)
	}
}

func (p *liveProgress) setPromptEst(n int) {
	if p != nil {
		p.promptEst.Store(int64(n))
	}
}
