package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// The accumulator is a continuous sum of PER-ENDPOINT DELTAS, which is what
// makes it survive a fleet that moves. Summing raw counters breaks twice over:
// a pod restarting resets its counter and the fleet sum drops, read downstream
// as a counter reset; a pod leaving subtracts its whole history.

// fakeVLLM serves the prompt-tokens family with a settable value, or fails.
type fakeVLLM struct {
	compute  atomic.Int64
	local    atomic.Int64
	external atomic.Int64
	fail     atomic.Bool
	stall    atomic.Bool
	hits     atomic.Int64
	srv      *httptest.Server
}

func newFakeVLLM(t *testing.T) *fakeVLLM {
	t.Helper()
	f := &fakeVLLM{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.hits.Add(1)
		if f.stall.Load() {
			time.Sleep(vllmMetricsFetchTimeout + time.Second) //clockexempt: exercises the fetch timeout
		}
		if f.fail.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, `# TYPE vllm:prompt_tokens_by_source counter
vllm:prompt_tokens_by_source_total{source="local_compute",model_name="m"} %d
vllm:prompt_tokens_by_source_total{source="local_cache_hit",model_name="m"} %d
vllm:prompt_tokens_by_source_total{source="external_kv_transfer",model_name="m"} %d
`, f.compute.Load(), f.local.Load(), f.external.Load())
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeVLLM) url() string { return f.srv.URL + "/metrics" }

// capture reads back what the sampler actually persisted, through the real
// writer and the real JSONL encoding — the same path the report consumes, so a
// field that fails to serialise fails here too.
type capture struct {
	t    *testing.T
	path string
}

func (c *capture) all() []vllmMetricsSample {
	c.t.Helper()
	f, err := os.Open(c.path)
	if err != nil {
		c.t.Fatalf("open jsonl: %v", err)
	}
	defer f.Close()
	var out []vllmMetricsSample
	dec := json.NewDecoder(f)
	for {
		var s vllmMetricsSample
		if err := dec.Decode(&s); err == io.EOF {
			break
		} else if err != nil {
			c.t.Fatalf("decode jsonl: %v", err)
		}
		if s.RecordType == recordTypeVLLMMetricsSample {
			out = append(out, s)
		}
	}
	return out
}

func newTestSampler(t *testing.T, urls []string, cap *capture) *vllmMetricsSampler {
	t.Helper()
	cap.t = t
	cap.path = filepath.Join(t.TempDir(), "samples.jsonl")
	f, err := os.Create(cap.path)
	if err != nil {
		t.Fatalf("create jsonl: %v", err)
	}
	t.Cleanup(func() { f.Close() })
	rdw := &requestDataWriter{f: f, enc: json.NewEncoder(f)}
	return &vllmMetricsSampler{
		model:    "m",
		urls:     urls,
		explicit: true,
		tracker:  newActiveDatasetTracker(),
		rdw:      rdw,
		interval: time.Minute,
		client:   &http.Client{},
		now:      time.Now,
		prev:     map[string]map[string]float64{},
		totals:   map[string]float64{},
		logf:     func(string, ...any) {},
	}
}

func TestTotalsAreDeltasNotRawSums(t *testing.T) {
	a, b := newFakeVLLM(t), newFakeVLLM(t)
	var cap capture
	s := newTestSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	// Baseline: both pods already carry history from whatever ran before. A
	// run wants only what happens DURING it, so the first sighting must count
	// as zero rather than importing that history.
	a.compute.Store(1_000_000)
	b.compute.Store(2_000_000)
	s.sampleOnce(ctx)
	if got := cap.all()[0].Sources.Compute; got != 0 {
		t.Errorf("first sample = %d, want 0: the baseline must not import counters accumulated "+
			"before the run started", got)
	}

	a.compute.Add(100)
	b.compute.Add(200)
	s.sampleOnce(ctx)
	if got := cap.all()[1].Sources.Compute; got != 300 {
		t.Errorf("after +100/+200 the total is %d, want 300", got)
	}
}

// TestRestartCountsFromZeroInsteadOfGoingBackwards is the case that motivated
// the whole scheme: a pod restarts, its counter resets, and a sum of raw
// counters would DROP — which downstream reads as a counter reset and discards.
func TestRestartCountsFromZeroInsteadOfGoingBackwards(t *testing.T) {
	a, b := newFakeVLLM(t), newFakeVLLM(t)
	var cap capture
	s := newTestSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	a.compute.Store(500)
	b.compute.Store(500)
	s.sampleOnce(ctx) // baseline
	a.compute.Store(900)
	b.compute.Store(700)
	s.sampleOnce(ctx) // +400 +200 = 600
	if got := cap.all()[1].Sources.Compute; got != 600 {
		t.Fatalf("total = %d, want 600", got)
	}

	// a restarts: its counter resets to 0 and climbs to 50.
	a.compute.Store(50)
	b.compute.Store(750)
	s.sampleOnce(ctx)

	got := cap.all()[2].Sources.Compute
	if got < cap.all()[1].Sources.Compute {
		t.Errorf("total went backwards %d -> %d across a restart; that is the failure the delta "+
			"scheme exists to prevent", cap.all()[1].Sources.Compute, got)
	}
	// 600 + a's post-restart 50 + b's +50.
	if got != 700 {
		t.Errorf("total = %d, want 700 (600 + 50 from the restarted pod counted from zero + 50)", got)
	}
	if cap.all()[2].Resets != 1 {
		t.Errorf("Resets = %d, want 1", cap.all()[2].Resets)
	}
}

// TestOneEndpointFailingCostsOnlyItsOwnDelta. The old behaviour skipped the
// whole sample when any endpoint failed, so on a loaded fleet most intervals
// vanished — /metrics is served by the same event loop as inference, so the
// backends most worth measuring are the ones most likely to miss the timeout.
func TestOneEndpointFailingCostsOnlyItsOwnDelta(t *testing.T) {
	a, b := newFakeVLLM(t), newFakeVLLM(t)
	var cap capture
	s := newTestSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	s.sampleOnce(ctx) // baseline at 0/0

	a.compute.Add(100)
	b.compute.Add(999)
	b.fail.Store(true)
	s.sampleOnce(ctx)

	smp := cap.all()[1]
	if smp.Sources.Compute != 100 {
		t.Errorf("total = %d, want 100: the healthy endpoint's delta must land even though its "+
			"peer failed", smp.Sources.Compute)
	}
	if smp.EndpointsOK != 1 || smp.EndpointsTotal != 2 {
		t.Errorf("coverage %d/%d, want 1/2 — a partial interval has to say so, or a low bar reads "+
			"as an idle fleet", smp.EndpointsOK, smp.EndpointsTotal)
	}

	// b recovers. Its delta is measured from the last value actually SEEN, so
	// nothing it did while unreachable is lost.
	b.fail.Store(false)
	s.sampleOnce(ctx)
	if got := cap.all()[2].Sources.Compute; got != 1099 {
		t.Errorf("total = %d, want 1099: the recovered endpoint's work during the outage is "+
			"counted on its next successful read, not dropped", got)
	}
}

// TestEndpointThatNeverReturnsKeepsItsContribution: a pod that leaves the fleet
// must not subtract what it already did.
func TestEndpointThatNeverReturnsKeepsItsContribution(t *testing.T) {
	a, b := newFakeVLLM(t), newFakeVLLM(t)
	var cap capture
	s := newTestSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	s.sampleOnce(ctx)
	a.compute.Add(400)
	b.compute.Add(600)
	s.sampleOnce(ctx)
	if got := cap.all()[1].Sources.Compute; got != 1000 {
		t.Fatalf("total = %d, want 1000", got)
	}

	b.fail.Store(true) // gone for good
	for range 3 {
		a.compute.Add(10)
		s.sampleOnce(ctx)
	}
	got := cap.all()
	last := got[len(got)-1]
	if last.Sources.Compute != 1030 {
		t.Errorf("total = %d, want 1030: a departed endpoint stops adding, it does not subtract",
			last.Sources.Compute)
	}
	if last.EndpointsOK != 1 {
		t.Errorf("coverage %d/%d", last.EndpointsOK, last.EndpointsTotal)
	}
}

// TestNoEndpointsAnsweringIsRecordedAsUnobserved. A flat interval and an
// unobserved one are different claims, and the natural reading of a flat bar —
// "the fleet did nothing" — is the wrong one.
func TestNoEndpointsAnsweringIsRecordedAsUnobserved(t *testing.T) {
	a := newFakeVLLM(t)
	var cap capture
	s := newTestSampler(t, []string{a.url()}, &cap)
	ctx := context.Background()

	s.sampleOnce(ctx)
	a.fail.Store(true)
	s.sampleOnce(ctx)

	if len(cap.all()) != 2 {
		t.Fatalf("%d samples written, want 2: an interval nobody could observe is still an "+
			"interval, and omitting it leaves a gap with no explanation", len(cap.all()))
	}
	if got := cap.all()[1].EndpointsOK; got != 0 {
		t.Errorf("EndpointsOK = %d, want 0", got)
	}
	if cap.all()[1].EndpointsTotal != 1 {
		t.Errorf("EndpointsTotal = %d, want 1", cap.all()[1].EndpointsTotal)
	}
}

// TestCoverageReachesTheReport: the numbers are worthless if the chart cannot
// see them.
func TestCoverageReachesTheReport(t *testing.T) {
	base := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	mix, _ := buildSampleViz([]vllmMetricsSample{
		{TS: base, Sources: vllmSourceCounters{Compute: 0}, EndpointsOK: 8, EndpointsTotal: 8},
		{TS: base.Add(time.Minute), Sources: vllmSourceCounters{Compute: 100}, EndpointsOK: 3, EndpointsTotal: 8},
	})
	if len(mix) != 1 {
		t.Fatalf("%d segments, want 1", len(mix))
	}
	if mix[0].EndpointsOK != 3 || mix[0].EndpointsTotal != 8 {
		t.Errorf("segment coverage %d/%d, want 3/8: a partial interval must be marked as one in "+
			"the report, not drawn as a short bar", mix[0].EndpointsOK, mix[0].EndpointsTotal)
	}
}

// TestMetricsURLOverrideReachesTheSampler.
//
// The default derives /metrics from the serving spec, which is right only when
// the thing serving inference is also the thing exposing the counters. Behind a
// router it never is: the router refuses /metrics on its serving port on
// purpose, because proxying it answers with ONE backend's counters — a number
// that looks like a fleet total and is not. Without a way to name the real
// address, an entire campaign records zero samples and the report shows nothing
// where the interesting half of the comparison should be.
func TestMetricsURLOverrideReachesTheSampler(t *testing.T) {
	a, b := newFakeVLLM(t), newFakeVLLM(t)
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "override_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	// A serving spec pointing somewhere with no /metrics at all — the router.
	s := startVLLMMetricsSampler(context.Background(),
		"dynamic/http://router.invalid:9000/v1,type=openai_vllm",
		[]string{a.srv.URL, b.url()}, // bare endpoint AND explicit /metrics
		newActiveDatasetTracker(), rdw)
	if s == nil {
		t.Fatal("no sampler started despite explicit metrics URLs")
	}
	defer s.stop()

	want := map[string]bool{a.url(): true, b.url(): true}
	for _, u := range s.urls {
		if !want[u] {
			t.Errorf("sampler is scraping %q; want exactly %v — a bare endpoint must be normalised "+
				"to /metrics rather than fetched as-is", u, want)
		}
		delete(want, u)
	}
	if len(want) != 0 {
		t.Errorf("sampler never resolved %v", want)
	}
	if !s.explicit {
		t.Error("explicitly named endpoints must poll forever rather than spending a speculative " +
			"budget; the operator said where they are and a backend loading weights must not cost " +
			"the rest of the run")
	}
}

func TestNormalizeMetricsURLs(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"http://h:29000", "http://h:29000/metrics"},
		{"http://h:29000/", "http://h:29000/metrics"},
		{"http://h:29000/metrics", "http://h:29000/metrics"},
		{"http://h:8000/v1", "http://h:8000/metrics"},
		{"http://h:8000/v1/", "http://h:8000/metrics"},
	} {
		got := normalizeMetricsURLs([]string{tc.in})
		if len(got) != 1 || got[0] != tc.want {
			t.Errorf("%q -> %v, want [%s]", tc.in, got, tc.want)
		}
	}
	if got := normalizeMetricsURLs([]string{"", "  "}); len(got) != 0 {
		t.Errorf("blank entries survived as %v", got)
	}
}
