package benchmark

import (
	"math"
	"os"
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
// the external share; the 500ms the engine did not see leads as wait.
func TestTimelineSegmentsEngineTimings(t *testing.T) {
	segs, trunc := timelineSegments(8500, 3000, false, goldenKV())
	if trunc {
		t.Error("unexpected truncation")
	}
	m := segMap(segs)
	want := map[int]float64{
		segWaitOther:      750,  // 500 client-side lead + 250 queue_wait_other
		segWaitKV:         500,  //
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
	// Left to right: wait, kv wait, computed, external, gpu, preempted, decode.
	order := []int{}
	for _, s := range segs {
		order = append(order, s.kind)
	}
	wantOrder := []int{segWaitOther, segWaitKV, segPrefillCompute, segPrefillExt, segPrefillGPU, segPreempted, segDecode}
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
	segs, trunc := timelineSegments(4000, 1000, false, goldenKV()) // engine phases sum to 8000
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
	m := segMap(mustSegs(timelineSegments(2200, 0, false, kv)))
	if !near(m[segPrefillExt], 200) || !near(m[segPrefillCompute], 1000) {
		t.Errorf("segments = %v", m)
	}
}

func mustSegs(s []tlSeg, _ bool) []tlSeg { return s }

// Sources without engine timings (SGLang): the client's TTFT is the prefill
// span, split by tokens.
func TestTimelineSegmentsSourcesWithoutTimings(t *testing.T) {
	kv := &llm.KVDetails{Sources: &llm.TokenSources{GPU: 600, Host: 300, External: 100, Computed: 200}}
	m := segMap(mustSegs(timelineSegments(5000, 1200, false, kv)))
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
	segs, _ := timelineSegments(5000, 1200, false, nil)
	if len(segs) != 2 || segs[0] != (tlSeg{segPrefillUnknown, 1200}) || segs[1] != (tlSeg{segDecode, 3800}) {
		t.Errorf("segs = %+v", segs)
	}
	// TTFT missing or beyond e2e: the whole bar is prefill.
	for _, ttft := range []float64{0, 9000} {
		segs, _ = timelineSegments(5000, ttft, false, nil)
		if len(segs) != 1 || segs[0] != (tlSeg{segPrefillUnknown, 5000}) {
			t.Errorf("ttft=%v segs = %+v", ttft, segs)
		}
	}
	if segs, _ = timelineSegments(700, 100, true, goldenKV()); len(segs) != 1 || segs[0].kind != segError {
		t.Errorf("error request segs = %+v", segs)
	}
}

func TestTimelineSegmentsZeroSources(t *testing.T) {
	kv := &llm.KVDetails{Sources: &llm.TokenSources{}, Timings: &llm.RequestTimings{Prefill: 1, KVFetch: 0.5, Decode: 1}}
	m := segMap(mustSegs(timelineSegments(2500, 0, false, kv)))
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
	if !strings.Contains(string(b), `"timeline":{`) || !strings.Contains(string(b), `id="viewTimeline"`) {
		t.Error("timeline report lacks the embedded arm or the view switch")
	}
	p, err = GenerateVisualizationWithOverwrite(dir, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(p)
	if strings.Contains(string(b), `"timeline":{`) {
		t.Error("default report must not embed timeline data")
	}
}
