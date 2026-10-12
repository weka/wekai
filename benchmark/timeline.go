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
	{segWaitClient, "wait_client", "client wait (no free slot)", "#b8c0c8"},
	{segWaitOther, "wait_other", "engine queue: other limits", "#5b646d"},
	{segWaitKV, "wait_kv", "engine queue: no free KV blocks", "#FF8569"},
	{segUnattributed, "wait_unattributed", "unattributed wait (network / frontend)", "#a08f7a"},
	{segWaited, "waited", "waited (all kinds merged)", "#8a9096"},
	{segPrefillCompute, "prefill_computed", "prefill: computed", "#d9b73a"},
	{segPrefillExt, "prefill_external", "prefill: external (weka / offload)", "#a678e8"},
	{segPrefillHost, "prefill_host", "prefill: host DRAM", "#3fa7b5"},
	{segPrefillGPU, "prefill_gpu", "prefill: GPU cache", "#8094b5"},
	{segPrefillUnknown, "prefill_unknown", "prefill (unknown source)", "#9aa5b1"},
	{segPreempted, "preempted", "preempted", "#c2788a"},
	{segDecode, "decode", "decode", "#5f987d"},
	{segError, "error", "error", "#FF6B6B"},
	{segInflight, "inflight", "in flight", "#d0b3f5"},
}

// timelineReq is one request as shipped to the renderer. Times are
// milliseconds; Start is relative to the arm's T0. Zero-valued optional
// fields are omitted to keep a 70k-request payload small.
type timelineReq struct {
	Row    int       `json:"r"`            // series group, in order of the series' first request
	Sub    int       `json:"u,omitempty"`  // sub-row within the group (overlapping requests of one series)
	Series int       `json:"s"`            // series_num: the replay session this request belongs to
	Slot   int       `json:"sl,omitempty"` // concurrency slot (worker lane); the row group when present
	Turn   int       `json:"n"`
	Start  float64   `json:"t"`
	E2E    float64   `json:"e"`
	TTFT   float64   `json:"f,omitempty"`
	In     int       `json:"i,omitempty"` // net-of-cache input tokens
	Cached int       `json:"c,omitempty"`
	Out    int       `json:"o,omitempty"`
	CW     float64   `json:"cw,omitempty"` // client-side wait (ready -> sent) in ms; included in E2E
	Flags  int       `json:"x,omitempty"`
	Src    []int     `json:"src,omitempty"` // gpu, host, external, computed
	Rec    []int     `json:"rc,omitempty"`  // preemptions, discarded tokens, load failures, recompute tokens
	Tm     []float64 `json:"tm,omitempty"`  // ms: queue_wait_kv, queue_wait_other, kv_fetch, prefill, decode, preempted; then preemption count
	GI     int       `json:"gi,omitempty"`  // 1-based index into the arm's Guids (session:instance); static pages
	GUID   string    `json:"gu,omitempty"`  // the same inline; live responses
	Segs   []float64 `json:"g"`             // flattened (kind, durationMs) pairs, left to right
}

// rowKey identifies a row group: the concurrency slot when the request carries
// one, else (files written before slots were recorded) its series_num.
type rowKey struct{ slot, series int }

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
func timelineSegments(clientWaitMs, e2eMs, ttftMs float64, isErr bool, kv *llm.KVDetails) (segs []tlSeg, truncated bool) {
	if e2eMs < 0 {
		e2eMs = 0
	}
	if clientWaitMs > 0 {
		defer func() {
			segs = append([]tlSeg{{segWaitClient, clientWaitMs}}, segs...)
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
	segs, trunc := timelineSegments(cw, e2e, r.TTFT, r.IsError, r.KV)
	tr := timelineReq{
		Series: r.SeriesNum,
		Turn:   r.Turn,
		Start:  float64(r.StartTime.UnixMilli()-t0) - cw,
		E2E:    e2e + cw,
		CW:     cw,
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
	tr.Slot = r.Slot
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

// buildTimelineArm builds an arm with one row group per concurrency slot (per
// series_num for data recorded without slots), ordered by the group's first
// request. A slot replays sessions one after another, so the first request of
// each new session is flagged for a boundary marker. Time zero is the earliest start in recs.
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
	for _, r := range recs {
		tr := timelineFromRecord(r, t0)
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
	lastSeries := map[rowKey]int{}
	for i := range arm.Reqs {
		q := &arm.Reqs[i]
		k := rowKeyOf(q.Slot, q.Series)
		g, ok := groups[k]
		if !ok {
			g = len(groups)
			groups[k] = g
			packers[k] = &subRowPacker{}
		}
		if prev, seen := lastSeries[k]; !seen || prev != q.Series {
			q.Flags |= flagSession
			lastSeries[k] = q.Series
		}
		q.Row = g
		q.Sub = packers[k].place(q.Start, q.Start+q.E2E)
		if end := q.Start + q.E2E; end > arm.Span {
			arm.Span = end
		}
	}
	arm.Groups = len(groups)
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
