package cli

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func rq(id uint64, ts string, totalMs int64) ReplayRequest {
	return ReplayRequest{RequestID: id, Ts: ts, TotalMs: totalMs}
}

func thinkOf(t *testing.T, r ReplayRequest) (int64, string) {
	t.Helper()
	if r.ThinkMs == nil {
		t.Fatalf("request %d has no think_ms", r.RequestID)
	}
	return *r.ThinkMs, r.ThinkRef
}

// The worked example: request 1 starts T0 and ends T10, request 2 starts T20,
// so its think is 10s from "prev".
func TestAnnotateThinkPrevAndClamp(t *testing.T) {
	rs := &ReplaySession{StartTs: "2026-05-12T08:00:00Z", Instances: []ReplayInstance{{
		InstanceID: "a",
		Requests: []ReplayRequest{
			rq(1, "2026-05-12T08:00:00Z", 10_000),
			rq(2, "2026-05-12T08:00:20Z", 5_000),
			// Starts before request 2 finished: overlapping, clamped to 0.
			rq(3, "2026-05-12T08:00:22Z", 1_000),
		},
	}}}
	acc := newThinkAccumulator()
	annotateThink(rs, acc)
	r := rs.Instances[0].Requests

	if ms, ref := thinkOf(t, r[0]); ms != 0 || ref != ThinkRefSessionStart {
		t.Errorf("first request: think=%d ref=%s, want 0 session_start", ms, ref)
	}
	if ms, ref := thinkOf(t, r[1]); ms != 10_000 || ref != ThinkRefPrev {
		t.Errorf("second request: think=%d ref=%s, want 10000 prev", ms, ref)
	}
	if ms, ref := thinkOf(t, r[2]); ms != 0 || ref != ThinkRefPrev {
		t.Errorf("overlapping request: think=%d ref=%s, want 0 prev (clamped)", ms, ref)
	}
	s := acc.summary()
	if s.ThinkClamped != 1 || s.RequestsWithThink != 3 || s.RequestsWithTiming != 3 {
		t.Errorf("summary = %+v, want 1 clamped / 3 with think / 3 with timing", s)
	}
	if s.ThinkClampedPercent < 33 || s.ThinkClampedPercent > 34 {
		t.Errorf("clamped percent = %v, want ~33.3", s.ThinkClampedPercent)
	}
}

// A sub-agent's first request measures from its parent request's end, and every
// sibling of a fan-out from the same instant; later requests chain from prev.
func TestAnnotateThinkSpawnAndFanOut(t *testing.T) {
	rs := &ReplaySession{StartTs: "2026-05-12T08:00:00Z", Instances: []ReplayInstance{
		{InstanceID: "main", Requests: []ReplayRequest{
			rq(1, "2026-05-12T08:00:00Z", 30_000), // ends 08:00:30
		}},
		{InstanceID: "sub1", ParentSpawnRequestID: 1, FanOutGroup: "1", Requests: []ReplayRequest{
			rq(2, "2026-05-12T08:00:32Z", 8_000), // 2s after the spawn ended
			rq(4, "2026-05-12T08:00:50Z", 1_000), // prev ended 08:00:40 -> 10s
		}},
		{InstanceID: "sub2", ParentSpawnRequestID: 1, FanOutGroup: "1", Requests: []ReplayRequest{
			rq(3, "2026-05-12T08:00:35Z", 2_000), // 5s after the spawn ended
		}},
	}}
	annotateThink(rs, newThinkAccumulator())

	for _, tc := range []struct {
		inst, idx int
		ms        int64
		ref       string
	}{
		{1, 0, 2_000, ThinkRefSpawn},
		{1, 1, 10_000, ThinkRefPrev},
		{2, 0, 5_000, ThinkRefSpawn},
	} {
		ms, ref := thinkOf(t, rs.Instances[tc.inst].Requests[tc.idx])
		if ms != tc.ms || ref != tc.ref {
			t.Errorf("instance %d request %d: think=%d ref=%s, want %d %s", tc.inst, tc.idx, ms, ref, tc.ms, tc.ref)
		}
	}
}

// A parent that no request in the session carries (dangling) degrades to the
// session start rather than inventing a reference.
func TestAnnotateThinkDanglingParentUsesSessionStart(t *testing.T) {
	rs := &ReplaySession{StartTs: "2026-05-12T08:00:00Z", Instances: []ReplayInstance{
		{InstanceID: "x", ParentSpawnRequestID: 999, Requests: []ReplayRequest{rq(1, "2026-05-12T08:01:00Z", 0)}},
	}}
	annotateThink(rs, newThinkAccumulator())
	if ms, ref := thinkOf(t, rs.Instances[0].Requests[0]); ms != 60_000 || ref != ThinkRefSessionStart {
		t.Errorf("think=%d ref=%s, want 60000 session_start", ms, ref)
	}
}

func TestThinkSummaryPercentiles(t *testing.T) {
	acc := newThinkAccumulator()
	rs := &ReplaySession{StartTs: "2026-05-12T08:00:00Z", Instances: []ReplayInstance{{InstanceID: "a"}}}
	base := time.Date(2026, 5, 12, 8, 0, 0, 0, time.UTC)
	for i := 0; i < 100; i++ {
		rs.Instances[0].Requests = append(rs.Instances[0].Requests,
			rq(uint64(i+1), base.Add(time.Duration(i)*10*time.Second).Format(time.RFC3339Nano), 9_000))
	}
	annotateThink(rs, acc)
	s := acc.summary()
	if s.Think.P50 != 1_000 || s.ByRef[ThinkRefPrev].N != 99 {
		t.Errorf("think summary = %+v / %+v, want p50 1000 and 99 prev samples", s.Think, s.ByRef[ThinkRefPrev])
	}
}

func TestBuildReplayRequestCarriesTimings(t *testing.T) {
	rr := buildReplayRequest(&treeRecord{ID: 7, Ts: time.Now(), UpstreamLatencyMs: 1601, TotalMs: 39042})
	if rr == nil || rr.UpstreamLatencyMs != 1601 || rr.TotalMs != 39042 {
		t.Fatalf("timings not carried: %+v", rr)
	}
}

func TestParseTreeLineCarriesTimings(t *testing.T) {
	line := `{"id":5,"ts":"2026-05-12T08:00:00Z","method":"POST","status":200,"upstream_latency_ms":1601,"total_ms":39042,"request":{"body":{"_schema":"req-v1"}},"response":{}}`
	var recs []treeRecord
	var a, b, c int
	if err := (&RouterTreeCommand{}).parseLine([]byte(line), &recs, &a, &b, &c); err != nil || len(recs) != 1 {
		t.Fatalf("parse: %v, %d records", err, len(recs))
	}
	if recs[0].UpstreamLatencyMs != 1601 || recs[0].TotalMs != 39042 {
		t.Errorf("record timings = %d/%d, want 1601/39042", recs[0].UpstreamLatencyMs, recs[0].TotalMs)
	}
}

// think_ms absent (v3) must not serialize as a zero think.
func TestReplayRequestOmitsAbsentThink(t *testing.T) {
	b, _ := json.Marshal(ReplayRequest{RequestID: 1})
	if strings.Contains(string(b), "think") {
		t.Errorf("a request without think serialized %s", b)
	}
	z := int64(0)
	b, _ = json.Marshal(ReplayRequest{RequestID: 1, ThinkMs: &z, ThinkRef: ThinkRefPrev})
	if !strings.Contains(string(b), `"think_ms":0`) {
		t.Errorf("a zero think was dropped: %s", b)
	}
}

func TestFilterRecordsByTime(t *testing.T) {
	mk := func(s string) treeRecord { ts, _ := time.Parse(time.RFC3339Nano, s); return treeRecord{Ts: ts} }
	recs := []treeRecord{mk("2026-05-11T00:00:00Z"), mk("2026-05-12T00:00:00Z"), mk("2026-07-07T08:17:18.5Z"), mk("2026-07-08T00:00:00Z")}
	out, err := filterRecordsByTime(append([]treeRecord(nil), recs...), "2026-05-12T00:00:00Z", "2026-07-07T08:17:18.999999999Z")
	if err != nil || len(out) != 2 {
		t.Fatalf("got %d records (err %v), want 2", len(out), err)
	}
	if _, err := filterRecordsByTime(recs, "garbage", ""); err == nil {
		t.Error("bad --since accepted")
	}
}
