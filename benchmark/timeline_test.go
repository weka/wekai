package benchmark

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/llm"
)

func near(a, b float64) bool { return math.Abs(a-b) < 0.01 }

func segMap(segs []tlSeg) map[int]float64 {
	m := map[int]float64{}
	for _, s := range segs {
		m[s.kind] += s.ms
	}
	return m
}

func goldenKV() *llm.KVDetails {
	return &llm.KVDetails{
		Sources: &llm.TokenSources{GPU: 64, External: 16, Computed: 20},
		Timings: &llm.RequestTimings{QueueWaitKV: 0.5, QueueWaitOther: 0.25, KVFetch: 1.5, Prefill: 2.0, Decode: 3.0, PreemptedTime: 0.75, NumPreemptions: 1},
	}
}

// Golden numbers: prefill 2000ms splits 20/16/64 of 100 tokens; kv_fetch joins
// the external share; the 500ms the engine did not see is unattributed wait.
func TestTimelineSegmentsEngineTimings(t *testing.T) {
	segs, trunc := timelineSegments(0, 0, 8500, 3000, false, goldenKV())
	if trunc {
		t.Error("unexpected truncation")
	}
	m := segMap(segs)
	want := map[int]float64{
		segWaitOther:      250, // queue_wait_other
		segWaitKV:         500,
		segUnattributed:   500,  // e2e minus everything the engine accounted for
		segPrefillCompute: 400,  // 2000 * 20/100
		segPrefillExt:     1820, // 2000 * 16/100 + 1500 kv_fetch
		segPrefillGPU:     1280, // 2000 * 64/100
		segPreempted:      750,
		segDecode:         3000,
	}
	for k, v := range want {
		if !near(m[k], v) {
			t.Errorf("kind %d = %.2f, want %.2f", k, m[k], v)
		}
	}
	var total float64
	for _, s := range segs {
		total += s.ms
	}
	if !near(total, 8500) {
		t.Errorf("segments sum to %.2f, want the 8500ms e2e", total)
	}
	// Left to right: engine queue, kv queue, unattributed, computed, external, gpu, preempted, decode.
	order := []int{}
	for _, s := range segs {
		order = append(order, s.kind)
	}
	wantOrder := []int{segWaitOther, segWaitKV, segUnattributed, segPrefillCompute, segPrefillExt, segPrefillGPU, segPreempted, segDecode}
	if len(order) != len(wantOrder) {
		t.Fatalf("order = %v", order)
	}
	for i := range order {
		if order[i] != wantOrder[i] {
			t.Fatalf("order = %v, want %v", order, wantOrder)
		}
	}
}

func TestTimelineSegmentsScaleWhenEngineExceedsClient(t *testing.T) {
	segs, trunc := timelineSegments(0, 0, 4000, 1000, false, goldenKV()) // engine phases sum to 8000
	if !trunc {
		t.Error("expected the truncated flag")
	}
	var total float64
	for _, s := range segs {
		total += s.ms
	}
	if !near(total, 4000) {
		t.Errorf("sum = %.2f, want 4000", total)
	}
	if m := segMap(segs); !near(m[segDecode], 1500) { // 3000 * 0.5
		t.Errorf("decode = %.2f, want proportional 1500", m[segDecode])
	}
}

func TestTimelineSegmentsExternalFromKVFetchWithoutTokens(t *testing.T) {
	kv := &llm.KVDetails{
		Sources: &llm.TokenSources{Computed: 100},
		Timings: &llm.RequestTimings{KVFetch: 0.2, Prefill: 1, Decode: 1},
	}
	m := segMap(mustSegs(timelineSegments(0, 0, 2200, 0, false, kv)))
	if !near(m[segPrefillExt], 200) || !near(m[segPrefillCompute], 1000) {
		t.Errorf("segments = %v", m)
	}
}

func mustSegs(s []tlSeg, _ bool) []tlSeg { return s }

// Sources without engine timings (SGLang): the client's TTFT is the prefill
// span, split by tokens.
func TestTimelineSegmentsSourcesWithoutTimings(t *testing.T) {
	kv := &llm.KVDetails{Sources: &llm.TokenSources{GPU: 600, Host: 300, External: 100, Computed: 200}}
	m := segMap(mustSegs(timelineSegments(0, 0, 5000, 1200, false, kv)))
	if !near(m[segPrefillGPU], 600) || !near(m[segPrefillHost], 300) || !near(m[segPrefillExt], 100) || !near(m[segPrefillCompute], 200) {
		t.Errorf("prefill split = %v", m)
	}
	if !near(m[segDecode], 3800) {
		t.Errorf("decode = %.2f, want 3800", m[segDecode])
	}
}

// No engine details at all: start, first token, end -> one unknown prefill
// colour then decode.
func TestTimelineSegmentsClientOnlyFallback(t *testing.T) {
	segs, _ := timelineSegments(0, 0, 5000, 1200, false, nil)
	if len(segs) != 2 || segs[0] != (tlSeg{segPrefillUnknown, 1200}) || segs[1] != (tlSeg{segDecode, 3800}) {
		t.Errorf("segs = %+v", segs)
	}
	// TTFT missing or beyond e2e: the whole bar is prefill.
	for _, ttft := range []float64{0, 9000} {
		segs, _ = timelineSegments(0, 0, 5000, ttft, false, nil)
		if len(segs) != 1 || segs[0] != (tlSeg{segPrefillUnknown, 5000}) {
			t.Errorf("ttft=%v segs = %+v", ttft, segs)
		}
	}
	if segs, _ = timelineSegments(0, 0, 700, 100, true, goldenKV()); len(segs) != 1 || segs[0].kind != segError {
		t.Errorf("error request segs = %+v", segs)
	}
}

func TestTimelineSegmentsZeroSources(t *testing.T) {
	kv := &llm.KVDetails{Sources: &llm.TokenSources{}, Timings: &llm.RequestTimings{Prefill: 1, KVFetch: 0.5, Decode: 1}}
	m := segMap(mustSegs(timelineSegments(0, 0, 2500, 0, false, kv)))
	if !near(m[segPrefillUnknown], 1500) {
		t.Errorf("segments = %v", m)
	}
}

func TestBuildTimelineArm(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	mk := func(off time.Duration, series int, guid string, kv *llm.KVDetails) requestDataRecord {
		return requestDataRecord{StartTime: base.Add(off), EndTime: base.Add(off + 2*time.Second), ResponseMs: 2000, TTFT: 500,
			SeriesNum: series, SeriesGUID: guid, Turn: 1, InputTokens: 10, CachedTokens: 90, OutputTokens: 5, KV: kv}
	}
	arm := buildTimelineArm("a", []requestDataRecord{
		mk(10*time.Second, 2, "s2:main", nil), mk(0, 1, "s1:main", goldenKV()), mk(5*time.Second, 3, "s3:main", nil),
	})
	if arm.T0 != base.UnixMilli() {
		t.Errorf("t0 = %d", arm.T0)
	}
	// One group per series, ordered by first request: series 1, 3, 2.
	if len(arm.Reqs) != 3 || arm.Groups != 3 {
		t.Fatalf("arm = %+v", arm)
	}
	for i, want := range []struct{ series, row int }{{1, 0}, {3, 1}, {2, 2}} {
		r := arm.Reqs[i]
		if r.Series != want.series || r.Row != want.row || r.Sub != 0 {
			t.Errorf("req %d = series %d row %d sub %d, want series %d row %d", i, r.Series, r.Row, r.Sub, want.series, want.row)
		}
	}
	if len(arm.Guids) != 3 || arm.Guids[arm.Reqs[0].GI-1] != "s1:main" {
		t.Errorf("guid table = %v", arm.Guids)
	}
	if arm.Reqs[0].Flags&flagTimings == 0 || arm.Reqs[0].Flags&flagSources == 0 || len(arm.Reqs[0].Tm) != 7 || len(arm.Reqs[0].Src) != 4 {
		t.Errorf("engine details not carried: %+v", arm.Reqs[0])
	}
	if arm.Reqs[1].Flags&(flagTimings|flagSources) != 0 || arm.Reqs[1].Tm != nil {
		t.Errorf("client-only request has details: %+v", arm.Reqs[1])
	}
	if !near(arm.Span, 12000) {
		t.Errorf("span = %v, want 12000", arm.Span)
	}
	if got := buildTimelineArm("empty", nil); len(got.Reqs) != 0 {
		t.Errorf("empty arm = %+v", got)
	}
}

// Sequential turns of one series tile a single sub-row; overlapping fan-out
// requests take as many sub-rows as are concurrently running, reusing freed ones.
func TestBuildTimelineArmSubRowPacking(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	mk := func(startSec, durSec float64, series int) requestDataRecord {
		return requestDataRecord{StartTime: base.Add(time.Duration(startSec * float64(time.Second))), ResponseMs: durSec * 1000, TTFT: 100, SeriesNum: series}
	}
	seq := buildTimelineArm("seq", []requestDataRecord{mk(0, 2, 1), mk(2, 2, 1), mk(10, 1, 1), mk(20, 1, 1)})
	for _, r := range seq.Reqs {
		if r.Row != 0 || r.Sub != 0 {
			t.Errorf("sequential turn at %.0f on row %d sub %d, want 0/0", r.Start, r.Row, r.Sub)
		}
	}
	// Turn at t=0 (10s) fans out to children at 1s (2s) and 2s (5s); the one at 4s fits
	// the row freed at 3s; the one at 20s is back on row 0.
	fan := buildTimelineArm("fan", []requestDataRecord{mk(0, 10, 1), mk(1, 2, 1), mk(2, 5, 1), mk(4, 1, 1), mk(20, 1, 1), mk(0, 1, 2)})
	got := map[float64]int{}
	for _, r := range fan.Reqs {
		if r.Series == 1 {
			got[r.Start/1000] = r.Sub
		}
		if r.Series == 2 && (r.Row != 1 || r.Sub != 0) {
			t.Errorf("second series = row %d sub %d", r.Row, r.Sub)
		}
	}
	want := map[float64]int{0: 0, 1: 1, 2: 2, 4: 1, 20: 0}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("request at %.0fs on sub-row %d, want %d (all: %v)", k, got[k], v, got)
		}
	}
}

func TestGenerateTimelineStandalone(t *testing.T) {
	dir := t.TempDir()
	w, err := newRequestDataWriter(dir, "dynamic,alias=armA", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	for i := 0; i < 5; i++ {
		rec := requestDataRecord{StartTime: now.Add(time.Duration(i) * time.Second), ResponseMs: 800, TTFT: 100, Model: "dynamic,alias=armA", SeriesNum: i + 1, Turn: 1}
		if i == 0 {
			rec.KV = goldenKV()
		}
		w.write(rec)
	}
	w.close()
	out := filepath.Join(dir, "t.html")
	got, err := GenerateTimeline(dir, out, 0)
	if err != nil || got != out {
		t.Fatalf("GenerateTimeline = %q, %v", got, err)
	}
	b, _ := os.ReadFile(out)
	html := string(b)
	for _, want := range []string{"var WekaiTimeline = (function", `"name":"armA"`, "prefill (unknown source)", "<title>Request timeline</title>"} {
		if !strings.Contains(html, want) {
			t.Errorf("page lacks %q", want)
		}
	}
	if _, err := GenerateTimeline(t.TempDir(), "", 0); err == nil {
		t.Error("an empty directory must be an error")
	}
}

// The report's Timeline view carries every arm's model, only when asked for.
func TestReportTimelineOptIn(t *testing.T) {
	dir := t.TempDir()
	w, _ := newRequestDataWriter(dir, "m", time.Now())
	now := time.Now()
	for i := 0; i < 3; i++ {
		w.write(requestDataRecord{StartTime: now.Add(time.Duration(i) * time.Second), ResponseMs: 500, TTFT: 50, Model: "m", SeriesNum: i + 1})
	}
	w.close()
	p, err := GenerateVisualizationTimeline(dir, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	if !strings.Contains(string(b), `"tl":{`) || !strings.Contains(string(b), `id="viewTimeline"`) {
		t.Error("timeline report lacks the embedded arm or the view switch")
	}
	p, err = GenerateVisualizationWithOverwrite(dir, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(p)
	if strings.Contains(string(b), `"tl":{`) {
		t.Error("default report must not embed timeline data")
	}
}

// Slots are the row groups; the sessions a slot replays follow one another along
// its row, each new session flagged. Data recorded without slots falls back to
// grouping by series_num.
func TestBuildTimelineArmWorkerSessionsGetTheirOwnRows(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	mk := func(startSec float64, slot, series int, hot bool) requestDataRecord {
		return requestDataRecord{StartTime: base.Add(time.Duration(startSec * float64(time.Second))), ResponseMs: 1000, TTFT: 100,
			Slot: slot, SlotHot: hot, SeriesNum: series, Turn: 1}
	}
	// Only the worker was recorded: each stretch of a worker's requests under
	// one series_num is one session and gets its own row.
	arm := buildTimelineArm("workers", []requestDataRecord{
		mk(0, 1, 100, true), mk(1, 1, 100, true), mk(5, 1, 101, true), // worker 1: session 100 then 101
		mk(0, 2, 200, false), mk(7, 2, 205, false), mk(9, 2, 200, false), // worker 2: 200, 205, then 200 again
	})
	if arm.Groups != 5 {
		t.Fatalf("groups = %d, want one per session (5)", arm.Groups)
	}
	rows := map[int]map[int]bool{} // row -> series on it
	for _, r := range arm.Reqs {
		if rows[r.Row] == nil {
			rows[r.Row] = map[int]bool{}
		}
		rows[r.Row][r.Series] = true
		if (r.Series < 200) != (r.Flags&flagHot != 0) { // worker 1 (hot) ran series 100 and 101
			t.Errorf("hot flag wrong on %+v", r)
		}
		if r.Flags&flagSession != 0 {
			t.Errorf("session boundary flagged on a row that holds one session: %+v", r)
		}
	}
	for row, ss := range rows {
		if len(ss) != 1 {
			t.Errorf("row %d mixes sessions %v", row, ss)
		}
	}
	for _, r := range arm.Reqs {
		if r.Slot != 0 || r.Sess != r.Row+1 {
			t.Errorf("derived session carries worker %d / run %d on row %d, want no worker and run = row+1", r.Slot, r.Sess, r.Row)
		}
	}
	// The two turns of session 100 share a row.
	var r0, r1 int
	for _, r := range arm.Reqs {
		if r.Series == 100 {
			if r.Start == 0 {
				r0 = r.Row
			} else {
				r1 = r.Row
			}
		}
	}
	if r0 != r1 {
		t.Error("a continuing session was split across rows")
	}

	// No worker recorded: one group per series_num.
	old := buildTimelineArm("old", []requestDataRecord{mk(0, 0, 7, false), mk(1, 0, 8, false), mk(2, 0, 7, false)})
	if old.Groups != 2 {
		t.Errorf("fallback groups = %d, want 2 series", old.Groups)
	}
}

func TestRequestDataSlotRoundTrip(t *testing.T) {
	dir := t.TempDir()
	w, _ := newRequestDataWriter(dir, "m", time.Now())
	w.write(requestDataRecord{StartTime: time.Now(), Slot: 12, SlotHot: true, SeriesNum: 302})
	w.close()
	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	recs, _, err := readJSONLFile(files[0])
	if err != nil || len(recs) != 1 || recs[0].Slot != 12 || !recs[0].SlotHot || recs[0].SeriesNum != 302 {
		t.Fatalf("recs=%+v err=%v", recs, err)
	}
	// Rows written without a slot do not carry the keys at all.
	w2, _ := newRequestDataWriter(t.TempDir(), "m", time.Now())
	_ = w2
	b, _ := json.Marshal(requestDataRecord{})
	if strings.Contains(string(b), `"slot"`) || strings.Contains(string(b), `"slot_hot"`) {
		t.Errorf("zero slot serialized: %s", b)
	}
}

// Client wait (eligible but unsent) leads the bar and is not part of e2e; engine
// queue waits and the unattributed gap follow, each in its own kind.
func TestTimelineSegmentsWaitsSeparated(t *testing.T) {
	segs, _ := timelineSegments(1200, 0, 8500, 3000, false, goldenKV())
	want := []int{segWaitClient, segWaitOther, segWaitKV, segUnattributed, segPrefillCompute, segPrefillExt, segPrefillGPU, segPreempted, segDecode}
	if len(segs) != len(want) {
		t.Fatalf("segs = %+v", segs)
	}
	for i, k := range want {
		if segs[i].kind != k {
			t.Fatalf("segment %d kind = %d, want %d (%+v)", i, segs[i].kind, k, segs)
		}
	}
	if !near(segs[0].ms, 1200) {
		t.Errorf("client wait = %.1f, want 1200", segs[0].ms)
	}
	var total float64
	for _, s := range segs {
		total += s.ms
	}
	if !near(total, 9700) { // client wait + e2e
		t.Errorf("sum = %.1f, want 9700", total)
	}
	// No gate wait: no client segment at all.
	if segs, _ = timelineSegments(0, 0, 8500, 3000, false, goldenKV()); segs[0].kind == segWaitClient {
		t.Error("client wait segment drawn for a request that did not wait")
	}
	// Without engine timings the client wait still leads, before the unknown prefill.
	segs, _ = timelineSegments(300, 0, 5000, 1200, false, nil)
	if len(segs) != 3 || segs[0] != (tlSeg{segWaitClient, 300}) || segs[1].kind != segPrefillUnknown {
		t.Errorf("client-only segs = %+v", segs)
	}
	// Engine phases exceeding the client e2e never make the unattributed gap negative.
	for _, s := range mustSegs(timelineSegments(0, 0, 4000, 1000, false, goldenKV())) {
		if s.ms < 0 {
			t.Errorf("negative segment %+v", s)
		}
	}
}

// ready/sent stamps become the client wait and shift the bar's start left; old
// rows (no stamps) are unchanged.
func TestTimelineClientWaitFromStamps(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	rec := requestDataRecord{StartTime: base.Add(2 * time.Second), ResponseMs: 1000, TTFT: 200, SeriesNum: 1,
		ReadyMs: base.UnixMilli(), SentMs: base.Add(1500 * time.Millisecond).UnixMilli()}
	q := timelineFromRecord(rec, base.UnixMilli())
	if q.CW != 1500 || q.Start != 500 || q.E2E != 2500 {
		t.Errorf("cw=%v start=%v e2e=%v, want 1500 / 500 / 2500", q.CW, q.Start, q.E2E)
	}
	if q.Segs[0] != segWaitClient || q.Segs[1] != 1500 {
		t.Errorf("segs = %v", q.Segs)
	}
	rec.ReadyMs, rec.SentMs = 0, 0
	if q = timelineFromRecord(rec, base.UnixMilli()); q.CW != 0 || q.Start != 2000 || q.E2E != 1000 {
		t.Errorf("old row: %+v", q)
	}
	// Inconsistent stamps (sent before ready) are ignored rather than trusted.
	rec.ReadyMs, rec.SentMs = base.UnixMilli()+5000, base.UnixMilli()
	if q = timelineFromRecord(rec, base.UnixMilli()); q.CW != 0 {
		t.Errorf("inconsistent stamps gave cw=%v", q.CW)
	}
}

// Gate wait present vs absent in the live in-flight view: a request that has not
// been sent yet is flagged waiting; once sent, its wait is fixed.
func TestLiveInflightClientWait(t *testing.T) {
	a := newLiveArm("a", 10)
	ready := time.Now().Add(-2 * time.Second)
	tok := a.begin(1, 1, "", slotInfo{N: 1}, ready)
	d := a.snapshot(0)
	if d.Inflight[0].Flags&flagWaiting == 0 {
		t.Error("unsent request must be flagged waiting")
	}
	a.sent(tok, ready.Add(1500*time.Millisecond))
	q := a.snapshot(0).Inflight[0]
	if q.Flags&flagWaiting != 0 || q.CW != 1500 {
		t.Errorf("after sent: flags=%d cw=%v", q.Flags, q.CW)
	}
	rec := requestDataRecord{StartTime: ready.Add(1500 * time.Millisecond), ResponseMs: 400, SeriesNum: 1, Slot: 1,
		ReadyMs: ready.UnixMilli(), SentMs: ready.Add(1500 * time.Millisecond).UnixMilli()}
	a.complete(tok, rec)
	done := a.snapshot(0).Reqs[0]
	if done.CW != 1500 || done.E2E != 1900 {
		t.Errorf("completed: cw=%v e=%v", done.CW, done.E2E)
	}
}

// The renderer's merge toggle collapses exactly the wait kinds the model emits.
func TestTimelineKindsMergeContract(t *testing.T) {
	js := timelineJS
	for _, want := range []string{"WAIT_KINDS = { 0: 1, 1: 1, 11: 1, 12: 1, 14: 1, 15: 1 }", "WAITED = 13"} {
		if !strings.Contains(js, want) {
			t.Errorf("renderer lacks %q", want)
		}
	}
	ids := map[int]string{}
	for _, k := range timelineKinds {
		ids[k.ID] = k.Key
	}
	if ids[segWaitClient] != "wait_client" || ids[segWaitOther] != "wait_other" || ids[segWaitKV] != "wait_kv" ||
		ids[segUnattributed] != "wait_unattributed" || ids[segWaited] != "waited" {
		t.Errorf("kinds = %v", ids)
	}
}

// The live page's fetch URLs are built from a base that is absent from the config
// when empty; concatenating it unguarded requested "undefinedapi/arms".
func TestLivePageFetchBaseGuarded(t *testing.T) {
	if strings.Contains(timelinePageHTML, "CFG.base +") {
		t.Error("page concatenates CFG.base without a default")
	}
}

// Realtime replay: a turn the pacer waited for became eligible at its due time
// (now); one it found late was eligible `late` ago, but never before its
// predecessor finished.
func TestEligibleAtDueVersusSent(t *testing.T) {
	top := time.Unix(1000, 0)
	now := top.Add(10 * time.Second)
	if got := eligibleAt(top, now, 0); !got.Equal(now) {
		t.Errorf("waited-for turn: ready = %v, want now", got)
	}
	if got := eligibleAt(top, now, 4*time.Second); !got.Equal(top.Add(6 * time.Second)) {
		t.Errorf("4s-late turn: ready = %v, want due time", got)
	}
	if got := eligibleAt(top, now, time.Hour); !got.Equal(top) {
		t.Errorf("turn due before its predecessor ended: ready = %v, want loop top", got)
	}
	// Closed loop (no pacing): ready is the loop top only when nothing delays it.
	if got := eligibleAt(top, top, 0); !got.Equal(top) {
		t.Errorf("unpaced: ready = %v", got)
	}
}

func TestTimelineCarriesThinkPause(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	q := timelineFromRecord(requestDataRecord{StartTime: base.Add(time.Second), ResponseMs: 500, SeriesNum: 1, ThinkPauseMs: 4000}, base.UnixMilli())
	if q.TP != 4000 {
		t.Errorf("tp = %v, want 4000", q.TP)
	}
	if q = timelineFromRecord(requestDataRecord{StartTime: base, ResponseMs: 500}, base.UnixMilli()); q.TP != 0 {
		t.Errorf("tp = %v without a pause", q.TP)
	}
}

// A 429 backoff leads the engine's own phases, after any client wait, and is
// taken out of the unattributed gap rather than inflating it.
func TestTimelineSegments429Backoff(t *testing.T) {
	// e2e 9700 = 1200 backoff + 8500 engine-side; client waited 300 for a slot first.
	segs, _ := timelineSegments(300, 1200, 9700, 4200, false, goldenKV())
	if segs[0] != (tlSeg{segWaitClient, 300}) || segs[1] != (tlSeg{segRetry429, 1200}) || segs[2].kind != segWaitOther {
		t.Fatalf("segs = %+v", segs)
	}
	m := segMap(segs)
	if !near(m[segUnattributed], 500) { // 8500 - engine phases 8000
		t.Errorf("unattributed = %.1f, want 500 (backoff must not count)", m[segUnattributed])
	}
	var total float64
	for _, s := range segs {
		total += s.ms
	}
	if !near(total, 10000) { // client wait + e2e
		t.Errorf("sum = %.1f, want 10000", total)
	}
	// Without engine timings the backoff is carved out of the first-token span.
	segs, _ = timelineSegments(0, 1000, 5000, 3000, false, nil)
	if segs[0] != (tlSeg{segRetry429, 1000}) || segs[1] != (tlSeg{segPrefillUnknown, 2000}) {
		t.Errorf("client-only segs = %+v", segs)
	}
	// Backoff larger than e2e is clamped, never negative time.
	for _, s := range mustSegs(timelineSegments(0, 99999, 500, 100, false, nil)) {
		if s.ms < 0 {
			t.Errorf("negative segment %+v", s)
		}
	}
}

func TestTimelineCarries429Fields(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	q := timelineFromRecord(requestDataRecord{StartTime: base, ResponseMs: 3000, TTFT: 2500, SeriesNum: 1, Retries429: 3, RetryWaitMs: 1800}, base.UnixMilli())
	if q.R4 != 3 || q.RW != 1800 || q.Segs[0] != segRetry429 || q.Segs[1] != 1800 {
		t.Errorf("q = %+v", q)
	}
}

// The report carries the timeline as extra columns of its request table, and
// the page rebuilds the renderer's requests from them on first open. That
// rebuild must equal buildTimelineArm field for field, across every kind of
// value a request can carry.
func TestReportTimelineMatchesBuilder(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed")
	}
	dir := t.TempDir()
	w, _ := newRequestDataWriter(dir, "m", time.Now())
	now := time.Now().Truncate(time.Millisecond)
	var recs []requestDataRecord
	for i := 0; i < 12; i++ {
		st := now.Add(time.Duration(i*700) * time.Millisecond)
		r := requestDataRecord{StartTime: st, EndTime: st.Add(900 * time.Millisecond), ResponseMs: 900, TTFT: 120, Model: "m",
			SeriesNum: i%3 + 1, Turn: i/3 + 1, InputTokens: 100 + i, CachedTokens: 50 * i, OutputTokens: 10 + i,
			Slot: i%2 + 1, SlotHot: i%4 == 0, SeriesGUID: map[bool]string{true: "s:a", false: "s:b"}[i%2 == 0]}
		switch i % 6 {
		case 0:
			r.KV = goldenKV()
		case 1:
			r.KV = &llm.KVDetails{Recompute: &llm.KVRecompute{NumPreemptions: 1, PreemptDiscardedTokens: 7, NumExternalLoadFailures: 1, ExternalLoadRecomputeToks: 9}}
			r.IsError = true
		case 2:
			r.ReadyMs, r.SentMs = st.UnixMilli()-40, st.UnixMilli() // client wait
			r.ThinkPauseMs = 300
		case 3:
			r.Retries429, r.RetryWaitMs = 2, 150
		case 4:
			r.ResponseMs = 0 // end time stands in for the latency
			r.SeriesGUID = ""
		}
		recs = append(recs, r)
		w.write(r)
	}
	w.close()
	p, err := GenerateVisualizationTimeline(dir, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	script := extractReportScript(t, string(b))
	probe := `console.log("ARM=" + JSON.stringify(timelineArmOf(RAW_DATA[0])));`
	jsPath := filepath.Join(dir, "tl_probe.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil {
		t.Fatalf("node: %v\n%s", err, out)
	}
	line := string(out)[strings.Index(string(out), "ARM=")+4:]
	line = line[:strings.IndexByte(line, '\n')]
	var got struct {
		Reqs []map[string]any `json:"reqs"`
	}
	if err := json.Unmarshal([]byte(line), &got); err != nil {
		t.Fatal(err)
	}
	// The builder reads the same records back from the file, as the report did.
	fileRecs, _, _, _, _, err := readJSONLFileWithParams(filepath.Join(dir, mustOneJSONL(t, dir)))
	if err != nil {
		t.Fatal(err)
	}
	arm := buildTimelineArm("m", fileRecs)
	wb, _ := json.Marshal(arm.Reqs)
	var want []map[string]any
	if err := json.Unmarshal(wb, &want); err != nil {
		t.Fatal(err)
	}
	// Absent, zero and all-zero arrays are the same to the renderer.
	norm := func(m map[string]any) map[string]any {
		out := map[string]any{}
		for k, v := range m {
			switch x := v.(type) {
			case float64:
				if x == 0 {
					continue
				}
			case []any:
				zero := true
				for _, e := range x {
					if f, ok := e.(float64); !ok || f != 0 {
						zero = false
					}
				}
				if zero {
					continue
				}
			case nil:
				continue
			}
			out[k] = v
		}
		return out
	}
	if len(got.Reqs) != len(want) {
		t.Fatalf("%d requests, want %d", len(got.Reqs), len(want))
	}
	for i := range want {
		g, w := norm(got.Reqs[i]), norm(want[i])
		gb, _ := json.Marshal(g)
		wb, _ := json.Marshal(w)
		if string(gb) != string(wb) {
			t.Errorf("request %d:\n got %s\nwant %s", i, gb, wb)
		}
	}
}

func mustOneJSONL(t *testing.T, dir string) string {
	t.Helper()
	m, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if len(m) != 1 {
		t.Fatalf("want one .jsonl in %s, got %v", dir, m)
	}
	return filepath.Base(m[0])
}

func TestTimelineOneRowPerSession(t *testing.T) {
	t0 := time.Now()
	rec := func(run, series int, at time.Duration) requestDataRecord {
		return requestDataRecord{SessionRun: run, SeriesNum: series, StartTime: t0.Add(at), EndTime: t0.Add(at + time.Second), ResponseMs: 1000}
	}
	// Runs 1 and 2 are consecutive sessions; run 3 replays series 1 on a later pass.
	arm := buildTimelineArm("a", []requestDataRecord{
		rec(1, 1, 0), rec(1, 1, 2*time.Second), rec(2, 2, 4*time.Second), rec(3, 1, 6*time.Second),
	})
	if arm.Groups != 3 {
		t.Fatalf("%d rows, want one per session run (3)", arm.Groups)
	}
	for _, q := range arm.Reqs {
		if q.Row != q.Sess-1 {
			t.Errorf("session run %d drawn on row %d", q.Sess, q.Row)
		}
		if q.Flags&flagSession != 0 {
			t.Errorf("session boundary flagged on a row that holds one session: %+v", q)
		}
	}
}

func TestLiveEndedSessionLeavesTheCount(t *testing.T) {
	a := newLiveArm("a", 100)
	s1, s2 := slotInfo{N: 1, Session: true}, slotInfo{N: 2, Session: true}
	t1 := a.begin(1, 1, "s1:i0", s1, time.Now())
	a.sent(t1, time.Now())
	a.begin(2, 1, "s2:i0", s2, time.Now())
	if st := a.state(); st.Sessions != 2 {
		t.Fatalf("active sessions %d, want 2", st.Sessions)
	}
	a.complete(t1, requestDataRecord{SessionRun: 1, SeriesNum: 1, StartTime: time.Now(), EndTime: time.Now()})
	a.endLine(s1)
	if st := a.state(); st.Sessions != 1 || st.Idle != 0 {
		t.Fatalf("after session 1 ended: active %d idle %d, want 1 and 0", st.Sessions, st.Idle)
	}
	a.endLine(slotInfo{N: 2}) // a fixed-pool worker line carries on
	if st := a.state(); st.Sessions != 1 {
		t.Fatalf("a worker line was ended: active %d", st.Sessions)
	}
}
