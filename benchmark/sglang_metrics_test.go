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
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Realistic Prometheus text exposition: HELP/TYPE preamble, the four
// cache_source label values, a _created sibling series, and an unrelated
// family sharing a name prefix.
const sglangPromFixture = `# HELP sglang:cached_tokens_total Cached tokens by source.
# TYPE sglang:cached_tokens_total counter
sglang:cached_tokens_total{cache_source="device",model_name="m"} 1000
sglang:cached_tokens_total{cache_source="host",model_name="m"} 200
sglang:cached_tokens_total{cache_source="storage",model_name="m"} 40
sglang:cached_tokens_total{cache_source="total",model_name="m"} 1240
sglang:cached_tokens_total_created{cache_source="device",model_name="m"} 1.7e+09
# HELP sglang:prompt_tokens_total Total prompt tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="m"} 5000
sglang:num_running_reqs{model_name="m"} 3
`

func TestParseSGLangCounters(t *testing.T) {
	vals, err := parseSGLangCounters(strings.NewReader(sglangPromFixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := map[string]float64{
		"device":              1000,
		"host":                200,
		"storage":             40,
		"total":               1240,
		sglangPromptTokensKey: 5000,
	}
	if len(vals) != len(want) {
		t.Fatalf("got %d keys, want %d: %v", len(vals), len(want), vals)
	}
	for k, w := range want {
		if vals[k] != w {
			t.Errorf("key %q = %v, want %v", k, vals[k], w)
		}
	}
}

func TestParseSGLangCountersSumsAcrossOtherLabels(t *testing.T) {
	// Multi-engine deployment: both families sum across labels other than
	// cache_source (engine, model_name).
	in := `sglang:cached_tokens_total{cache_source="device",engine="0"} 100
sglang:cached_tokens_total{cache_source="device",engine="1"} 50
sglang:prompt_tokens_total{engine="0"} 300
sglang:prompt_tokens_total{engine="1"} 200
`
	vals, err := parseSGLangCounters(strings.NewReader(in))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if vals["device"] != 150 {
		t.Errorf("device = %v, want 150", vals["device"])
	}
	if vals[sglangPromptTokensKey] != 500 {
		t.Errorf("%s = %v, want 500", sglangPromptTokensKey, vals[sglangPromptTokensKey])
	}
}

func TestParseSGLangCountersNoLabelsAndTimestamp(t *testing.T) {
	// prompt_tokens_total may carry no labels at all, and a trailing
	// timestamp after the value must still parse.
	in := `sglang:prompt_tokens_total 42 1720000000000` + "\n"
	vals, err := parseSGLangCounters(strings.NewReader(in))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if vals[sglangPromptTokensKey] != 42 {
		t.Fatalf("got %v, want %s=42", vals, sglangPromptTokensKey)
	}
}

func TestParseSGLangCountersMissingFamily(t *testing.T) {
	// A scrape carrying only the (unrelated, gauge) cache_hit_rate family —
	// the shape this sampler no longer reads.
	vals, err := parseSGLangCounters(strings.NewReader("sglang:cache_hit_rate{model_name=\"m\"} 0.75\n"))
	if err != nil {
		t.Fatalf("parse should tolerate an unrelated family: %v", err)
	}
	if len(vals) != 0 {
		t.Fatalf("expected no keys from an unrelated family, got %v", vals)
	}
}

func TestParseSGLangCountersGarbage(t *testing.T) {
	vals, err := parseSGLangCounters(strings.NewReader("<html>not prometheus</html>\n"))
	if err != nil {
		t.Fatalf("parse should tolerate garbage: %v", err)
	}
	if len(vals) != 0 {
		t.Fatalf("expected no keys from garbage, got %v", vals)
	}
}

func TestSGLangMetricsEndpoints(t *testing.T) {
	cases := []struct {
		spec string
		want []string
	}{
		{"dynamic/http://localhost:8000/v1,type=openai_sglang,alias=x", []string{"http://localhost:8000/metrics"}},
		{"dynamic/http://a:8000/v1|http://b:8001/v1,type=openai_sglang", []string{"http://a:8000/metrics", "http://b:8001/metrics"}},
		// Unlike vLLM, SGLang sampling is never speculative: plain "openai" or
		// the default type must NOT start a sampler.
		{"dynamic/http://localhost:8000/v1,type=openai", nil},
		{"dynamic/http://localhost:8000/v1", nil},
		{"dynamic/http://localhost:8000/v1,type=openai_vllm", nil},
		// Not a dynamic model at all.
		{"gpt-4", nil},
	}
	for _, c := range cases {
		got := sglangMetricsEndpoints(c.spec)
		if len(got) != len(c.want) {
			t.Errorf("spec %q: got %v, want %v", c.spec, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("spec %q: got %v, want %v", c.spec, got, c.want)
				break
			}
		}
	}
}

// TestSGLangMetricsSamplerDoesNotStartForNonSGLang confirms
// startSGLangMetricsSampler stays inert for the vLLM/default eligibility
// shapes — SGLang sampling is additive, never a second guess at the same
// endpoint — and for a nil tracker, mirroring startVLLMMetricsSampler's
// nil-dependency guard.
func TestSGLangMetricsSamplerDoesNotStartForNonSGLang(t *testing.T) {
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "no_sglang_model", time.Now())
	if err != nil {
		t.Fatalf("newRequestDataWriter: %v", err)
	}
	defer func() { _ = rdw.close() }()
	for _, spec := range []string{
		"dynamic/http://localhost:8000/v1,type=openai",
		"dynamic/http://localhost:8000/v1,type=openai_vllm",
		"dynamic/http://localhost:8000/v1",
	} {
		if s := startSGLangMetricsSampler(context.Background(), spec, newActiveDatasetTracker(), rdw); s != nil {
			s.stop()
			t.Errorf("spec %q: expected no sampler, got one", spec)
		}
	}
	// A valid spec but no tracker must also stay inert.
	if s := startSGLangMetricsSampler(context.Background(), "dynamic/http://localhost:8000/v1,type=openai_sglang", nil, rdw); s != nil {
		s.stop()
		t.Error("expected no sampler without a tracker")
	}
}

// TestStartSGLangMetricsSamplerLifecycle drives a real sampler (via the
// background goroutine, not sampleOnce directly) against a fake SGLang
// /metrics endpoint and confirms it lands one sample record — mirroring
// TestStartVLLMMetricsSamplerLifecycle.
func TestStartSGLangMetricsSamplerLifecycle(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(sglangPromFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "sglang_lifecycle_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}

	tracker := newActiveDatasetTracker()
	tracker.Update(1, 4000)
	tracker.Update(2, 1000)

	model := "dynamic/" + srv.URL + "/v1,type=openai_sglang"
	s := startSGLangMetricsSampler(context.Background(), model, tracker, rdw)
	if s == nil {
		t.Fatal("sampler did not start for an openai_sglang spec")
	}
	path := filepath.Join(dir, "sglang_lifecycle_model.jsonl")
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil && strings.Contains(string(b), recordTypeSGLangMetricsSample) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	s.stop()
	if err := rdw.close(); err != nil {
		t.Fatal(err)
	}

	samples := readSGLangSamplesFromFile(t, path)
	if len(samples) != 1 {
		t.Fatalf("got %d sglang samples, want 1: %v", len(samples), samples)
	}
	got := samples[0]
	if got.RecordType != recordTypeSGLangMetricsSample {
		t.Errorf("record_type = %q", got.RecordType)
	}
	// First sighting is the baseline and contributes nothing — see fold's doc.
	if got.Sources != (sglangSourceCounters{}) {
		t.Errorf("first sample = %+v, want all zero: it is the baseline, not a measurement", got.Sources)
	}
	if got.EndpointsOK != 1 || got.EndpointsTotal != 1 {
		t.Errorf("coverage = %d/%d, want 1/1", got.EndpointsOK, got.EndpointsTotal)
	}
	if got.ActiveDatasetTokens != 5000 || got.ActiveSeries != 2 {
		t.Errorf("active dataset = %d/%d, want 5000/2", got.ActiveDatasetTokens, got.ActiveSeries)
	}
}

// readSGLangSamplesFromFile reads only the sglang_metrics_sample rows from a
// request-data JSONL file via the shared reader.
func readSGLangSamplesFromFile(t *testing.T, path string) []sglangMetricsSample {
	t.Helper()
	_, _, sglangSamples, _, _, err := readJSONLFileWithParams(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return sglangSamples
}

// ── Delta accumulation (mirrors vllm_metrics_delta_test.go) ────────────────

// fakeSGLang serves the cached_tokens_total/prompt_tokens_total families with
// settable values, or fails.
type fakeSGLang struct {
	device, host, storage, total, prompt atomic.Int64
	fail                                 atomic.Bool
	srv                                  *httptest.Server
}

func newFakeSGLang(t *testing.T) *fakeSGLang {
	t.Helper()
	f := &fakeSGLang{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.fail.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, `# TYPE sglang:cached_tokens_total counter
sglang:cached_tokens_total{cache_source="device",model_name="m"} %d
sglang:cached_tokens_total{cache_source="host",model_name="m"} %d
sglang:cached_tokens_total{cache_source="storage",model_name="m"} %d
sglang:cached_tokens_total{cache_source="total",model_name="m"} %d
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="m"} %d
`, f.device.Load(), f.host.Load(), f.storage.Load(), f.total.Load(), f.prompt.Load())
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeSGLang) url() string { return f.srv.URL + "/metrics" }

// sglangCapture reads back what the sampler actually persisted, through the
// real writer and the real JSONL encoding.
type sglangCapture struct {
	t    *testing.T
	path string
}

func (c *sglangCapture) all() []sglangMetricsSample {
	c.t.Helper()
	f, err := os.Open(c.path)
	if err != nil {
		c.t.Fatalf("open jsonl: %v", err)
	}
	defer f.Close()
	var out []sglangMetricsSample
	dec := json.NewDecoder(f)
	for {
		var s sglangMetricsSample
		if err := dec.Decode(&s); err == io.EOF {
			break
		} else if err != nil {
			c.t.Fatalf("decode jsonl: %v", err)
		}
		if s.RecordType == recordTypeSGLangMetricsSample {
			out = append(out, s)
		}
	}
	return out
}

func newTestSGLangSampler(t *testing.T, urls []string, cap *sglangCapture) *sglangMetricsSampler {
	t.Helper()
	cap.t = t
	cap.path = filepath.Join(t.TempDir(), "sglang_samples.jsonl")
	f, err := os.Create(cap.path)
	if err != nil {
		t.Fatalf("create jsonl: %v", err)
	}
	t.Cleanup(func() { f.Close() })
	rdw := &requestDataWriter{f: f, enc: json.NewEncoder(f)}
	return &sglangMetricsSampler{
		model:    "m",
		urls:     urls,
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

func TestSGLangTotalsAreDeltasNotRawSums(t *testing.T) {
	a, b := newFakeSGLang(t), newFakeSGLang(t)
	var cap sglangCapture
	s := newTestSGLangSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	// Baseline: both pods already carry history from whatever ran before —
	// the first sighting must count as zero.
	a.device.Store(1_000_000)
	b.device.Store(2_000_000)
	s.sampleOnce(ctx)
	if got := cap.all()[0].Sources.Device; got != 0 {
		t.Errorf("first sample = %d, want 0: the baseline must not import counters accumulated "+
			"before the run started", got)
	}

	a.device.Add(100)
	b.device.Add(200)
	s.sampleOnce(ctx)
	if got := cap.all()[1].Sources.Device; got != 300 {
		t.Errorf("after +100/+200 the total is %d, want 300", got)
	}
}

// TestSGLangRestartCountsFromZeroInsteadOfGoingBackwards mirrors vLLM's
// equivalent: a pod restarts, its counter resets, and a sum of raw counters
// would DROP — which downstream reads as a counter reset and discards.
func TestSGLangRestartCountsFromZeroInsteadOfGoingBackwards(t *testing.T) {
	a, b := newFakeSGLang(t), newFakeSGLang(t)
	var cap sglangCapture
	s := newTestSGLangSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	a.device.Store(500)
	b.device.Store(500)
	s.sampleOnce(ctx) // baseline
	a.device.Store(900)
	b.device.Store(700)
	s.sampleOnce(ctx) // +400 +200 = 600
	if got := cap.all()[1].Sources.Device; got != 600 {
		t.Fatalf("device total = %d, want 600", got)
	}

	// a restarts: its device counter resets to 0 and climbs to 50.
	a.device.Store(50)
	b.device.Store(750)
	s.sampleOnce(ctx)

	got := cap.all()[2].Sources.Device
	if got < cap.all()[1].Sources.Device {
		t.Errorf("device total went backwards %d -> %d across a restart; that is the failure the "+
			"delta scheme exists to prevent", cap.all()[1].Sources.Device, got)
	}
	// 600 + a's post-restart 50 + b's +50.
	if got != 700 {
		t.Errorf("device total = %d, want 700 (600 + 50 from the restarted pod counted from zero + 50)", got)
	}
	if cap.all()[2].Resets != 1 {
		t.Errorf("Resets = %d, want 1", cap.all()[2].Resets)
	}
}

// TestSGLangComputeDerivedFromPromptMinusCachedTotal verifies the one
// non-direct band: compute = Δprompt_tokens_total − Δcached_tokens_total{total}.
func TestSGLangComputeDerivedFromPromptMinusCachedTotal(t *testing.T) {
	a := newFakeSGLang(t)
	var cap sglangCapture
	s := newTestSGLangSampler(t, []string{a.url()}, &cap)
	ctx := context.Background()

	a.prompt.Store(1000)
	a.total.Store(200)
	a.device.Store(150)
	a.host.Store(40)
	a.storage.Store(10)
	s.sampleOnce(ctx) // baseline

	a.prompt.Add(500)
	a.total.Add(100)
	a.device.Add(70)
	a.host.Add(20)
	a.storage.Add(10)
	s.sampleOnce(ctx)

	got := cap.all()[1].Sources
	if got.Compute != 400 { // Δprompt(500) - Δtotal(100)
		t.Errorf("compute = %d, want 400", got.Compute)
	}
	if got.Device != 70 || got.Host != 20 || got.Storage != 10 {
		t.Errorf("sources = %+v, want device=70 host=20 storage=10", got)
	}
}

// TestSGLangOneEndpointFailingCostsOnlyItsOwnDelta mirrors vLLM's equivalent:
// one endpoint failing must not zero the whole interval.
func TestSGLangOneEndpointFailingCostsOnlyItsOwnDelta(t *testing.T) {
	a, b := newFakeSGLang(t), newFakeSGLang(t)
	var cap sglangCapture
	s := newTestSGLangSampler(t, []string{a.url(), b.url()}, &cap)
	ctx := context.Background()

	s.sampleOnce(ctx) // baseline at 0/0

	a.device.Add(100)
	b.device.Add(999)
	b.fail.Store(true)
	s.sampleOnce(ctx)

	smp := cap.all()[1]
	if smp.Sources.Device != 100 {
		t.Errorf("device total = %d, want 100: the healthy endpoint's delta must land even though "+
			"its peer failed", smp.Sources.Device)
	}
	if smp.EndpointsOK != 1 || smp.EndpointsTotal != 2 {
		t.Errorf("coverage %d/%d, want 1/2", smp.EndpointsOK, smp.EndpointsTotal)
	}
}

// TestSGLangNoEndpointsAnsweringIsRecordedAsUnobserved: a flat interval and
// an unobserved one are different claims.
func TestSGLangNoEndpointsAnsweringIsRecordedAsUnobserved(t *testing.T) {
	a := newFakeSGLang(t)
	var cap sglangCapture
	s := newTestSGLangSampler(t, []string{a.url()}, &cap)
	ctx := context.Background()

	s.sampleOnce(ctx)
	a.fail.Store(true)
	s.sampleOnce(ctx)

	if len(cap.all()) != 2 {
		t.Fatalf("%d samples written, want 2", len(cap.all()))
	}
	if got := cap.all()[1].EndpointsOK; got != 0 {
		t.Errorf("EndpointsOK = %d, want 0", got)
	}
	if cap.all()[1].EndpointsTotal != 1 {
		t.Errorf("EndpointsTotal = %d, want 1", cap.all()[1].EndpointsTotal)
	}
	// Unlike vLLM's speculative sampler, sglang sampling never gives up —
	// the operator asserted type=openai_sglang.
	if !s.sampleOnce(ctx) {
		t.Error("sglang sampler must keep polling after failures; it is never speculative")
	}
}

// ── buildSampleVizSGLang ─────────────────────────────────────────────────

func TestBuildSampleVizSGLangDeltasAndClamp(t *testing.T) {
	t0 := time.Unix(1000, 0)
	mk := func(offsetSec int, compute, device, host, storage, adt int64, as int) sglangMetricsSample {
		return sglangMetricsSample{
			RecordType:          recordTypeSGLangMetricsSample,
			TS:                  t0.Add(time.Duration(offsetSec) * time.Second),
			Sources:             sglangSourceCounters{Compute: compute, Device: device, Host: host, Storage: storage},
			ActiveDatasetTokens: adt,
			ActiveSeries:        as,
		}
	}
	// Out of order on purpose; sample 3 has a counter reset (values drop).
	samples := []sglangMetricsSample{
		mk(60, 1500, 300, 100, 42, 2000, 2),
		mk(0, 1000, 200, 50, 40, 1000, 1),
		mk(120, 100, 50, 10, 5, 3000, 3), // reset: deltas must clamp to 0
		mk(180, 200, 80, 20, 6, 2500, 2),
	}
	mix, adt := buildSampleVizSGLang(samples)
	if len(mix) != 3 {
		t.Fatalf("got %d segments, want 3", len(mix))
	}
	if len(mix[0].Bands) != 4 {
		t.Fatalf("seg0 has %d bands, want 4 (compute/device/host/storage)", len(mix[0].Bands))
	}
	// Segment 1: 0s -> 60s.
	if bandTokens(t, mix[0], "compute") != 500 || bandTokens(t, mix[0], "device") != 100 ||
		bandTokens(t, mix[0], "host") != 50 || bandTokens(t, mix[0], "storage") != 2 {
		t.Errorf("seg0 deltas = %+v, want compute=500 device=100 host=50 storage=2", mix[0])
	}
	// Segment 2 spans the counter reset: all deltas clamp at 0.
	for _, name := range sglangBandOrder {
		if bandTokens(t, mix[1], name) != 0 {
			t.Errorf("seg1 (reset) band %q = %v, want 0", name, bandTokens(t, mix[1], name))
		}
	}
	// Segment 3 resumes from the post-reset baseline.
	if bandTokens(t, mix[2], "compute") != 100 || bandTokens(t, mix[2], "device") != 30 ||
		bandTokens(t, mix[2], "host") != 10 || bandTokens(t, mix[2], "storage") != 1 {
		t.Errorf("seg2 deltas = %+v, want compute=100 device=30 host=10 storage=1", mix[2])
	}
	if mix[0].T0 != float64(t0.UnixMilli()) || mix[0].T1 != float64(t0.Add(60*time.Second).UnixMilli()) {
		t.Errorf("seg0 span = %v..%v", mix[0].T0, mix[0].T1)
	}
	if len(adt) != 4 || adt[0].Tokens != 1000 || adt[3].Tokens != 2500 || adt[3].Series != 2 {
		t.Errorf("adt points wrong: %+v", adt)
	}
}

func TestBuildSampleVizSGLangEmpty(t *testing.T) {
	mix, adt := buildSampleVizSGLang(nil)
	if mix != nil || adt != nil {
		t.Fatalf("expected nil/nil, got %v %v", mix, adt)
	}
}
