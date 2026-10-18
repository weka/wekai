package benchmark

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/weka/wekai/llm"
)

// Timeline data model.
//
// One horizontal bar per request on a wall-clock axis. The Go side turns each
// request into a list of coloured segments (wait, prefill split by token
// source, decode); the embedded renderer (timeline/timeline.js) only draws
// them. Keeping the segment arithmetic here makes it testable and keeps the
// report view, the standalone page and the live page byte-for-byte consistent.

// Segment kinds. The numeric values are the wire contract with the renderer;
// timelineKinds below carries the matching legend entries, so the renderer
// never hard-codes a kind.
const (
	segWaitOther      = 0 // engine queue wait that is not KV-blocked (admission limits)
	segWaitKV         = 1 // engine queue wait because KV blocks could not be allocated
	segPrefillCompute = 2 // prefill share for tokens the engine computed
	segPrefillExt     = 3 // prefill share for externally loaded tokens (weka/offload), plus kv_fetch
	segPrefillHost    = 4 // prefill share for host-DRAM tokens (SGLang HiCache L2)
	segPrefillGPU     = 5 // prefill share for tokens served from the GPU prefix cache
	segPrefillUnknown = 6 // prefill with no per-source information
	segPreempted      = 7 // time spent preempted
	segDecode         = 8
	segError          = 9
	segInflight       = 10 // live view: request still running
	segWaitClient     = 11 // client-side wait: due / eligible, but no free concurrency slot yet
	segUnattributed   = 12 // sent, but not yet in the engine's clock: network / frontend
	segWaited         = 13 // renderer only: all wait kinds merged
	segRetry429       = 14 // client backoff after the fleet shed the request with a 429
)

// Per-request flag bits.
const (
	flagError     = 1
	flagCold      = 2
	flagEmpty     = 4
	flagTimings   = 8   // engine timings were reported
	flagSources   = 16  // per-source token counts were reported
	flagInflight  = 32  // live view: not finished yet
	flagTruncated = 64  // segments were scaled down to fit the client-observed duration
	flagSession   = 128 // first request of a new session on its slot: draw a boundary
	flagHot       = 256 // issued by a hot-pool slot
	flagWaiting   = 512 // live view: still waiting for a concurrency slot
)

type timelineKind struct {
	ID    int    `json:"id"`
	Key   string `json:"key"`
	Label string `json:"label"`
	Color string `json:"color"`
}

// timelineKinds is the legend, in draw order. Colours stay inside the WEKA
// report palette: cool slate for waiting, the weka purple for the external
// tier, warm amber for computed work.
var timelineKinds = []timelineKind{
	{segWaitClient, "wait_client", "client wait (concurrency limit)", "#b8c0c8"},
	{segRetry429, "wait_429", "429 shed / backoff", "#d65db1"},
	{segWaitOther, "wait_other", "engine queue: other limits", "#5b646d"},
	{segWaitKV, "wait_kv", "engine queue: no free KV blocks", "#f08a3c"},
	{segUnattributed, "wait_unattributed", "unattributed wait (network / frontend)", "#a08f7a"},
	{segWaited, "waited", "waited (all kinds merged)", "#8a9096"},
	{segPrefillCompute, "prefill_computed", "prefill: computed", "#d9b73a"},
	{segPrefillExt, "prefill_external", "prefill: external (weka / offload)", "#a678e8"},
	{segPrefillHost, "prefill_host", "prefill: host DRAM", "#3fa7b5"},
	{segPrefillGPU, "prefill_gpu", "prefill: GPU cache", "#8094b5"},
	{segPrefillUnknown, "prefill_unknown", "prefill (unknown source)", "#9aa5b1"},
	{segPreempted, "preempted", "preempted", "#c2788a"},
	{segDecode, "decode", "decode", "#5f987d"},
	{segError, "error", "error", "#ff3b3b"},
	{segInflight, "inflight", "in flight", "#d0b3f5"},
}

// timelineReq is one request as shipped to the renderer. Times are
// milliseconds; Start is relative to the arm's T0. Zero-valued optional
// fields are omitted to keep a 70k-request payload small.
type timelineReq struct {
	Row    int       `json:"r"`            // series group, in order of the series' first request
	Sub    int       `json:"u,omitempty"`  // sub-row within the group (overlapping requests of one series)
	Series int       `json:"s"`            // series_num: the replay session this request belongs to
	Slot   int       `json:"sl,omitempty"` // fixed-pool modes: the worker; the row group when present
	Sess   int       `json:"ss,omitempty"` // router replay: the session run; the row group when present
	Turn   int       `json:"n"`
	Start  float64   `json:"t"`
	E2E    float64   `json:"e"`
	TTFT   float64   `json:"f,omitempty"`
	In     int       `json:"i,omitempty"` // net-of-cache input tokens
	Cached int       `json:"c,omitempty"`
	Out    int       `json:"o,omitempty"`
	TP     float64   `json:"tp,omitempty"` // think pause (ms) immediately before the bar's start
	RW     float64   `json:"rw,omitempty"` // 429 backoff (ms), part of E2E, drawn after the client wait
	R4     int       `json:"r4,omitempty"` // number of 429 retries
	CW     float64   `json:"cw,omitempty"` // client-side wait (ready -> sent) in ms; included in E2E
	Flags  int       `json:"x,omitempty"`
	Src    []int     `json:"src,omitempty"` // gpu, host, external, computed
	Rec    []int     `json:"rc,omitempty"`  // preemptions, discarded tokens, load failures, recompute tokens
	Tm     []float64 `json:"tm,omitempty"`  // ms: queue_wait_kv, queue_wait_other, kv_fetch, prefill, decode, preempted; then preemption count
	GI     int       `json:"gi,omitempty"`  // 1-based index into the arm's Guids (session:instance); static pages
	GUID   string    `json:"gu,omitempty"`  // the same inline; live responses
	Segs   []float64 `json:"g"`             // flattened (kind, durationMs) pairs, left to right
	idx    int       // position in the records the arm was built from
}

// rowKey identifies a row group: one session (its run number, or the n-th
// session a worker played when only the worker was recorded), one worker (the
// live view of the fixed-pool modes), or, when neither was recorded, a
// series_num.
type rowKey struct{ session, slot, seg, series int }

func rowKeyOf(slot, series int) rowKey {
	if slot > 0 {
		return rowKey{slot: slot}
	}
	return rowKey{series: series}
}

// timelineArm is one benchmark arm (one reqdata file / model). The layout is
// one row group per series; Reqs[i].Row is the group, Reqs[i].Sub the sub-row.
type timelineArm struct {
	Name string `json:"name"`
	// T0 is the epoch-ms of the earliest request start; every Start is
	// relative to it.
	T0 int64 `json:"t0"`
	// Span is the extent of the data in ms from T0.
	Span float64 `json:"span"`
	// Groups is the number of slot (or series) rows.
	Groups int `json:"groups"`
	// Guids interns the per-request session:instance ids (see timelineReq.GI).
	Guids []string      `json:"guids,omitempty"`
	Reqs  []timelineReq `json:"reqs"`
	// Rates and TTFT are the run's token throughput and trailing-window TTFT
	// over time (see rates.go), for the strips above the rows.
	Rates *rateSeries `json:"rates,omitempty"`
	TTFT  *ttftSeries `json:"ttft,omitempty"`
}

// timelinePayload is everything the renderer needs.
type timelinePayload struct {
	Kinds []timelineKind `json:"kinds"`
	Arms  []timelineArm  `json:"arms"`
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }

type tlSeg struct {
	kind int
	ms   float64
}

// splitPrefill divides a prefill duration across token sources in proportion
// to their token counts, drawing computed, host, external, then GPU left to
// right. kvFetchMs (time waiting for an async external load) is added to the
// external share, which is created even when no external tokens were counted.
// Without any source information the whole span is one unknown segment.
func splitPrefill(prefillMs, kvFetchMs float64, src *llm.TokenSources) []tlSeg {
	if src == nil || src.Total() <= 0 {
		if prefillMs+kvFetchMs <= 0 {
			return nil
		}
		return []tlSeg{{segPrefillUnknown, prefillMs + kvFetchMs}}
	}
	tot := float64(src.Total())
	var out []tlSeg
	add := func(kind, tokens int, extra float64) {
		d := prefillMs*float64(tokens)/tot + extra
		if d > 0 {
			out = append(out, tlSeg{kind, d})
		}
	}
	add(segPrefillCompute, src.Computed, 0)
	add(segPrefillHost, src.Host, 0)
	add(segPrefillExt, src.External, kvFetchMs)
	add(segPrefillGPU, src.GPU, 0)
	return out
}

// timelineSegments computes a request's bar. e2eMs and ttftMs are the
// client-observed durations.
//
// clientWaitMs is time the request spent eligible but not yet sent (no free
// concurrency slot); it leads the bar and is NOT part of e2eMs.
//
// With engine timings the bar is: client wait, engine queue_wait_other,
// queue_wait_kv, unattributed (client time the engine did not see: network,
// frontend), prefill (split by token source, kv_fetch folded into the external
// share), preempted, decode. If the
// engine's phases add up to more than the client saw (clock skew, rounding)
// they are scaled down to fit and the truncated flag is returned.
//
// Without engine timings the client's first-token time is the prefill span
// (split by source when sources are known, e.g. SGLang) and the rest decode;
// whatever wait preceded it cannot be separated from prefill.
func timelineSegments(clientWaitMs, retryMs, e2eMs, ttftMs float64, isErr bool, kv *llm.KVDetails) (segs []tlSeg, truncated bool) {
	if e2eMs < 0 {
		e2eMs = 0
	}
	// The 429 backoff is inside e2e and TTFT (the client waited it), but the
	// engine only ever saw the attempt that was admitted: take it out first so
	// it is not misread as unattributed time.
	if retryMs > e2eMs {
		retryMs = e2eMs
	}
	if retryMs < 0 {
		retryMs = 0
	}
	e2eMs -= retryMs
	if ttftMs > retryMs {
		ttftMs -= retryMs
	}
	if clientWaitMs > 0 || retryMs > 0 {
		defer func() {
			var lead []tlSeg
			if clientWaitMs > 0 {
				lead = append(lead, tlSeg{segWaitClient, clientWaitMs})
			}
			if retryMs > 0 {
				lead = append(lead, tlSeg{segRetry429, retryMs})
			}
			segs = append(lead, segs...)
		}()
	}
	if isErr {
		return []tlSeg{{segError, e2eMs}}, false
	}
	var src *llm.TokenSources
	if kv != nil {
		src = kv.Sources
	}
	if kv != nil && kv.Timings != nil {
		t := kv.Timings
		qkv, qo, kvf := t.QueueWaitKV*1000, t.QueueWaitOther*1000, t.KVFetch*1000
		pre, dec, prm := t.Prefill*1000, t.Decode*1000, t.PreemptedTime*1000
		sum := qkv + qo + kvf + pre + dec + prm
		lead := e2eMs - sum
		if lead < 0 && sum > 0 {
			// Tolerate sub-millisecond rounding; scale anything bigger.
			if -lead > 1 {
				k := e2eMs / sum
				qkv, qo, kvf, pre, dec, prm = qkv*k, qo*k, kvf*k, pre*k, dec*k, prm*k
				truncated = true
			}
			lead = 0
		}
		if qo > 0 {
			segs = append(segs, tlSeg{segWaitOther, qo})
		}
		if qkv > 0 {
			segs = append(segs, tlSeg{segWaitKV, qkv})
		}
		if lead > 0 {
			segs = append(segs, tlSeg{segUnattributed, lead})
		}
		segs = append(segs, splitPrefill(pre, kvf, src)...)
		if prm > 0 {
			segs = append(segs, tlSeg{segPreempted, prm})
		}
		if dec > 0 {
			segs = append(segs, tlSeg{segDecode, dec})
		}
		return segs, truncated
	}
	pre := ttftMs
	if pre <= 0 || pre > e2eMs {
		pre = e2eMs
	}
	segs = append(segs, splitPrefill(pre, 0, src)...)
	if d := e2eMs - pre; d > 0 {
		segs = append(segs, tlSeg{segDecode, d})
	}
	return segs, false
}

// timelineFromRecord converts one reqdata row. Row is assigned by the caller.
func timelineFromRecord(r requestDataRecord, t0 int64) timelineReq {
	e2e := r.ResponseMs
	if e2e <= 0 && !r.EndTime.IsZero() {
		e2e = float64(r.EndTime.Sub(r.StartTime).Milliseconds())
	}
	cw := clientWaitMs(r)
	tp := float64(r.ThinkPauseMs)
	rw := float64(r.RetryWaitMs)
	segs, trunc := timelineSegments(cw, rw, e2e, r.TTFT, r.IsError, r.KV)
	tr := timelineReq{
		Series: r.SeriesNum,
		Turn:   r.Turn,
		Start:  float64(r.StartTime.UnixMilli()-t0) - cw,
		E2E:    e2e + cw,
		CW:     cw,
		RW:     rw,
		R4:     r.Retries429,
		TP:     tp,
		TTFT:   r.TTFT,
		In:     r.InputTokens,
		Cached: r.CachedTokens,
		Out:    r.OutputTokens,
	}
	if r.IsError {
		tr.Flags |= flagError
	}
	if r.IsColdStart {
		tr.Flags |= flagCold
	}
	if r.IsEmpty {
		tr.Flags |= flagEmpty
	}
	if trunc {
		tr.Flags |= flagTruncated
	}
	if kv := r.KV; kv != nil {
		if s := kv.Sources; s != nil {
			tr.Flags |= flagSources
			tr.Src = []int{s.GPU, s.Host, s.External, s.Computed}
		}
		if c := kv.Recompute; c != nil {
			tr.Rec = []int{c.NumPreemptions, c.PreemptDiscardedTokens, c.NumExternalLoadFailures, c.ExternalLoadRecomputeToks}
		}
		if t := kv.Timings; t != nil {
			tr.Flags |= flagTimings
			tr.Tm = []float64{
				round1(t.QueueWaitKV * 1000), round1(t.QueueWaitOther * 1000), round1(t.KVFetch * 1000),
				round1(t.Prefill * 1000), round1(t.Decode * 1000), round1(t.PreemptedTime * 1000),
				float64(t.NumPreemptions),
			}
		}
	}
	tr.Slot, tr.Sess = r.Slot, r.SessionRun
	if r.SlotHot {
		tr.Flags |= flagHot
	}
	tr.Segs = flattenSegs(segs)
	return tr
}

func flattenSegs(segs []tlSeg) []float64 {
	out := make([]float64, 0, len(segs)*2)
	for _, s := range segs {
		if s.ms <= 0 {
			continue
		}
		out = append(out, float64(s.kind), round1(s.ms))
	}
	return out
}

// subRowPacker assigns a series' requests to sub-rows greedily: a request
// takes the first sub-row whose last request has ended by its start, else opens
// a new one. Sequential turns therefore share sub-row 0; only overlapping
// requests (fan-out to sub-agents) add sub-rows.
type subRowPacker struct {
	free []float64 // per sub-row: ms at which it is free (+Inf while a request is running)
}

// place returns the sub-row for a request starting at start; end is when it
// frees the row (math.Inf(1) for a request still running).
func (p *subRowPacker) place(start, end float64) int {
	for i, f := range p.free {
		if f <= start {
			p.free[i] = end
			return i
		}
	}
	p.free = append(p.free, end)
	return len(p.free) - 1
}

// release frees sub-row u at end.
func (p *subRowPacker) release(u int, end float64) {
	if u >= 0 && u < len(p.free) {
		p.free[u] = end
	}
}

// buildTimelineArm builds an arm with one row group per session, ordered by the
// session's first request. A session is its run number when recorded; when only
// the worker was recorded (the fixed-pool modes) it is one stretch of that
// worker's requests under one series_num, since a worker plays its sessions one
// after another; with neither, its series_num. Time zero is the earliest start
// in recs.
func buildTimelineArm(name string, recs []requestDataRecord) timelineArm {
	arm := timelineArm{Name: name, Reqs: make([]timelineReq, 0, len(recs))}
	if len(recs) == 0 {
		return arm
	}
	t0 := recs[0].StartTime.UnixMilli()
	for _, r := range recs {
		if ms := r.StartTime.UnixMilli(); ms < t0 {
			t0 = ms
		}
	}
	arm.T0 = t0
	guidIdx := map[string]int{}
	for i, r := range recs {
		tr := timelineFromRecord(r, t0)
		tr.idx = i
		if r.SeriesGUID != "" {
			i, ok := guidIdx[r.SeriesGUID]
			if !ok {
				arm.Guids = append(arm.Guids, r.SeriesGUID)
				i = len(arm.Guids)
				guidIdx[r.SeriesGUID] = i
			}
			tr.GI = i
		}
		arm.Reqs = append(arm.Reqs, tr)
	}
	sort.SliceStable(arm.Reqs, func(i, j int) bool { return arm.Reqs[i].Start < arm.Reqs[j].Start })
	groups := map[rowKey]int{}
	packers := map[rowKey]*subRowPacker{}
	workerSeries, workerSeg := map[int]int{}, map[int]int{} // per worker: series_num of its current session, sessions so far
	for i := range arm.Reqs {
		q := &arm.Reqs[i]
		k := lineOf(recs[q.idx])
		if k.slot > 0 {
			if s, seen := workerSeries[k.slot]; !seen || s != q.Series {
				workerSeries[k.slot] = q.Series
				workerSeg[k.slot]++
			}
			k.seg = workerSeg[k.slot]
		}
		g, ok := groups[k]
		if !ok {
			g = len(groups)
			groups[k] = g
			packers[k] = &subRowPacker{}
		}
		q.Row = g
		q.Sub = packers[k].place(q.Start, q.Start+q.E2E)
		if end := q.Start + q.E2E; end > arm.Span {
			arm.Span = end
		}
	}
	arm.Groups = len(groups)
	acc := newRateAccum(1000, 4000)
	for i := range arm.Reqs {
		acc.addRequest(&arm.Reqs[i])
	}
	arm.Rates = acc.export(4000)
	arm.TTFT = ttftOverTime(arm.Reqs, arm.Span, 30000, 1500)
	return arm
}

func newTimelinePayload(arms ...timelineArm) timelinePayload {
	return timelinePayload{Kinds: timelineKinds, Arms: arms}
}

// timelineSummary is a one-line description for CLI output.
func (a timelineArm) summary() string {
	return fmt.Sprintf("%s: %d requests over %.0fs", a.Name, len(a.Reqs), a.Span/1000)
}

// clientWaitMs is how long the request sat eligible but unsent (ready -> sent),
// or 0 when either stamp is absent (older data) or they are inconsistent.
func clientWaitMs(r requestDataRecord) float64 {
	if r.ReadyMs <= 0 || r.SentMs < r.ReadyMs {
		return 0
	}
	return float64(r.SentMs - r.ReadyMs)
}

// eligibleAt is when a paced request became eligible to send. The pacer waited
// until the request was due (late == 0, so it is eligible now) or found it
// already late by `late` (it was due that long ago). It cannot have been
// eligible before its predecessor finished (loopTop).
func eligibleAt(loopTop, now time.Time, late time.Duration) time.Time {
	if r := now.Add(-late); r.After(loopTop) {
		return r
	}
	return loopTop
}
