package cli

import (
	"sort"
	"time"
)

// Think-time references. A request's think_ms is the gap between the moment
// its reference ENDED in the capture and the moment the request STARTED, so a
// replay can schedule it from when the reference ends against the system under
// test, whenever that is, instead of at a fixed offset on the captured clock.
//
//	prev           the previous request of the same instance. The reference
//	               ends at prev.ts + prev.total_ms.
//	spawn          the first request of an instance that has a parent
//	               (parent_spawn_request_id). The reference is that parent
//	               request, which ends at parent.ts + parent.total_ms. Every
//	               sibling of a fan-out has the same parent and so measures
//	               from the same instant.
//	session_start  the first request of an instance with no parent, measured
//	               from the session's start_ts (0 for the session's first
//	               request).
const (
	ThinkRefPrev         = "prev"
	ThinkRefSpawn        = "spawn"
	ThinkRefSessionStart = "session_start"
)

// ReplayTimingSummary is the header's account of the timing fields.
type ReplayTimingSummary struct {
	// RequestsWithTiming counts requests that carry a captured total_ms.
	RequestsWithTiming int `json:"requests_with_timing"`
	// RequestsWithThink counts requests that carry think_ms.
	RequestsWithThink int `json:"requests_with_think"`
	// ThinkClamped counts requests whose reference ended AFTER they started
	// (overlapping requests, or a sub-agent that began before its parent's
	// response finished); their think_ms is 0.
	ThinkClamped        int     `json:"think_clamped"`
	ThinkClampedPercent float64 `json:"think_clamped_percent"`
	// Think percentiles in ms over RequestsWithThink, overall and by reference.
	Think   ThinkPercentiles            `json:"think_ms"`
	ByRef   map[string]ThinkPercentiles `json:"think_ms_by_ref"`
	TotalMs ThinkPercentiles            `json:"total_ms"`
}

// ThinkPercentiles is a count and a set of percentiles in milliseconds.
type ThinkPercentiles struct {
	N    int   `json:"n"`
	P50  int64 `json:"p50"`
	P90  int64 `json:"p90"`
	P99  int64 `json:"p99"`
	Max  int64 `json:"max"`
	Mean int64 `json:"mean"`
}

type thinkAccumulator struct {
	all, total []int64
	byRef      map[string][]int64
	timing     int
	clamped    int
}

func newThinkAccumulator() *thinkAccumulator {
	return &thinkAccumulator{byRef: map[string][]int64{}}
}

// annotateThink fills ThinkMs / ThinkRef on every request of the session, after
// parent repair and cycle breaking so the references it measures from are the
// ones the replayer will actually wait on.
func annotateThink(rs *ReplaySession, acc *thinkAccumulator) {
	endOf := map[uint64]time.Time{}
	for i := range rs.Instances {
		for j := range rs.Instances[i].Requests {
			r := &rs.Instances[i].Requests[j]
			ts, err := time.Parse(time.RFC3339Nano, r.Ts)
			if err != nil || r.RequestID == 0 {
				continue
			}
			endOf[r.RequestID] = ts.Add(time.Duration(r.TotalMs) * time.Millisecond)
		}
	}
	sessStart, sessErr := time.Parse(time.RFC3339Nano, rs.StartTs)

	for i := range rs.Instances {
		inst := &rs.Instances[i]
		var prevEnd time.Time
		for j := range inst.Requests {
			r := &inst.Requests[j]
			if r.TotalMs > 0 {
				acc.timing++
				acc.total = append(acc.total, r.TotalMs)
			}
			ts, err := time.Parse(time.RFC3339Nano, r.Ts)
			if err != nil {
				continue
			}
			var ref string
			var base time.Time
			switch {
			case j > 0:
				ref, base = ThinkRefPrev, prevEnd
			case inst.ParentSpawnRequestID != 0:
				if e, ok := endOf[inst.ParentSpawnRequestID]; ok {
					ref, base = ThinkRefSpawn, e
				}
			}
			if ref == "" {
				if sessErr != nil {
					continue
				}
				ref, base = ThinkRefSessionStart, sessStart
			}
			think := ts.Sub(base).Milliseconds()
			if think < 0 {
				think = 0
				acc.clamped++
			}
			r.ThinkMs = &think
			r.ThinkRef = ref
			acc.all = append(acc.all, think)
			acc.byRef[ref] = append(acc.byRef[ref], think)
			prevEnd = ts.Add(time.Duration(r.TotalMs) * time.Millisecond)
		}
	}
}

func percentiles(v []int64) ThinkPercentiles {
	n := len(v)
	if n == 0 {
		return ThinkPercentiles{}
	}
	s := append([]int64(nil), v...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	var sum int64
	for _, x := range s {
		sum += x
	}
	q := func(p float64) int64 {
		i := int(float64(n) * p)
		if i >= n {
			i = n - 1
		}
		return s[i]
	}
	return ThinkPercentiles{N: n, P50: q(0.5), P90: q(0.9), P99: q(0.99), Max: s[n-1], Mean: sum / int64(n)}
}

func (a *thinkAccumulator) summary() *ReplayTimingSummary {
	out := &ReplayTimingSummary{
		RequestsWithTiming: a.timing,
		RequestsWithThink:  len(a.all),
		ThinkClamped:       a.clamped,
		Think:              percentiles(a.all),
		TotalMs:            percentiles(a.total),
		ByRef:              map[string]ThinkPercentiles{},
	}
	if len(a.all) > 0 {
		out.ThinkClampedPercent = percent(a.clamped, len(a.all))
	}
	for k, v := range a.byRef {
		out.ByRef[k] = percentiles(v)
	}
	return out
}
