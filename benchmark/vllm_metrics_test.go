package benchmark

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Realistic Prometheus text exposition: HELP/TYPE preamble, two engines,
// all three sources, a _created sibling series, and unrelated families
// sharing the name prefix.
const promFixture = `# HELP vllm:prompt_tokens_by_source Number of prompt tokens by source.
# TYPE vllm:prompt_tokens_by_source counter
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_compute"} 1000
vllm:prompt_tokens_by_source_total{engine="1",model_name="m",source="local_compute"} 500
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_cache_hit"} 200
vllm:prompt_tokens_by_source_total{engine="1",model_name="m",source="local_cache_hit"} 100
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="external_kv_transfer"} 40
vllm:prompt_tokens_by_source_total{engine="1",model_name="m",source="external_kv_transfer"} 2
vllm:prompt_tokens_by_source_created{engine="0",model_name="m",source="local_compute"} 1.7e+09
vllm:prompt_tokens_total{engine="0",model_name="m"} 99999
vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 88888
`

func TestParsePromptTokensBySource(t *testing.T) {
	vals, err := parsePromptTokensBySource(strings.NewReader(promFixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := map[string]float64{
		"local_compute":        1500,
		"local_cache_hit":      300,
		"external_kv_transfer": 42,
	}
	if len(vals) != len(want) {
		t.Fatalf("got %d sources, want %d: %v", len(vals), len(want), vals)
	}
	for k, w := range want {
		if vals[k] != w {
			t.Errorf("source %q = %v, want %v", k, vals[k], w)
		}
	}
}

func TestParsePromptTokensBySourceNoSuffixAndTimestamp(t *testing.T) {
	// Real vLLM always emits the _total suffix (see promFixture); this pins
	// the fallback for an exposition path that doesn't, plus trailing
	// timestamps after the value.
	in := `vllm:prompt_tokens_by_source{engine="0",source="local_compute"} 7 1720000000000` + "\n"
	vals, err := parsePromptTokensBySource(strings.NewReader(in))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if vals["local_compute"] != 7 {
		t.Fatalf("got %v, want local_compute=7", vals)
	}
}

func TestParsePromptTokensBySourceGarbage(t *testing.T) {
	vals, err := parsePromptTokensBySource(strings.NewReader("<html>not prometheus</html>\n"))
	if err != nil {
		t.Fatalf("parse should tolerate garbage: %v", err)
	}
	if len(vals) != 0 {
		t.Fatalf("expected no sources from garbage, got %v", vals)
	}
}

func TestVLLMEngineDerive(t *testing.T) {
	got := vllmEngine.Derive(map[string]float64{
		"local_compute":        500,
		"local_cache_hit":      300,
		"external_kv_transfer": 42,
	})
	want := map[string]float64{"compute": 500, "host": 300, "external": 42}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("band %q = %v, want %v (got %+v)", k, got[k], w, got)
		}
	}
}

func TestVLLMMetricsEndpoints(t *testing.T) {
	cases := []struct {
		spec         string
		want         []string
		wantExplicit bool
	}{
		{"dynamic/http://localhost:8000/v1,type=openai_vllm,alias=x", []string{"http://localhost:8000/metrics"}, true},
		{"dynamic/http://a:8000/v1|http://b:8001/v1,type=openai_vllm", []string{"http://a:8000/metrics", "http://b:8001/metrics"}, true},
		// Every chat/completions spec is eligible, just speculatively: the
		// endpoint may still be vLLM and the only way to know is to ask.
		{"dynamic/http://localhost:8000/v1,type=openai", []string{"http://localhost:8000/metrics"}, false},
		// No type= at all — ParseDynamicModel's "openai" default.
		{"dynamic/http://localhost:8000/v1", []string{"http://localhost:8000/metrics"}, false},
		// Bare host:port, as autodiscovery leaves it before appending /v1.
		{"dynamic/http://localhost:8000,type=openai", []string{"http://localhost:8000/metrics"}, false},
		// Wire formats no vLLM deployment answers on.
		{"dynamic/http://localhost:8000/v1,type=anthropic", nil, false},
		{"dynamic/http://localhost:8000/v1,type=gemini", nil, false},
		{"anthropic/claude-3", nil, false},
	}
	for _, c := range cases {
		got := engineEndpoints(vllmEngine, c.spec)
		if len(got.urls) != len(c.want) {
			t.Errorf("%s: got %v, want %v", c.spec, got.urls, c.want)
			continue
		}
		for i := range got.urls {
			if got.urls[i] != c.want[i] {
				t.Errorf("%s: got %v, want %v", c.spec, got.urls, c.want)
			}
		}
		if got.explicit != c.wantExplicit {
			t.Errorf("%s: explicit = %v, want %v", c.spec, got.explicit, c.wantExplicit)
		}
	}
}

// TestRequestDataWriterWriteAfterCloseIsSilent: during abnormal shutdown
// (error storm + early kill) in-flight workers race the deferred close; a
// write after close must be a silent no-op, not a warning-per-row storm.
func TestRequestDataWriterWriteAfterCloseIsSilent(t *testing.T) {
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "close_test", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := rdw.write(requestDataRecord{RequestNum: 1}); err != nil {
		t.Fatalf("pre-close write: %v", err)
	}
	if err := rdw.close(); err != nil {
		t.Fatal(err)
	}
	if err := rdw.write(requestDataRecord{RequestNum: 2}); err != nil {
		t.Fatalf("write after close must be a silent no-op, got: %v", err)
	}
	if err := rdw.writeAny(map[string]int{"x": 1}); err != nil {
		t.Fatalf("writeAny after close must be a silent no-op, got: %v", err)
	}
	if err := rdw.close(); err != nil {
		t.Fatalf("double close must be a no-op, got: %v", err)
	}
	records, samples, err := readJSONLFile(filepath.Join(dir, "close_test.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || len(samples) != 0 {
		t.Fatalf("post-close writes must not land: got %d records / %d samples", len(records), len(samples))
	}
}

func TestSamplerSampleOnceWritesRecord(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(promFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "sampler_test_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}

	tracker := newActiveDatasetTracker()
	tracker.Update(1, 4000)
	tracker.Update(2, 1000)

	s := &metricsSampler{
		engine:  vllmEngine,
		model:   "dynamic/" + srv.URL + "/v1,type=openai_vllm",
		urls:    []string{srv.URL + "/metrics"},
		tracker: tracker,
		rdw:     rdw,
		client:  srv.Client(),
		now:     func() time.Time { return time.Unix(1721000000, 0) },
	}
	s.sampleOnce(context.Background())
	if err := rdw.close(); err != nil {
		t.Fatal(err)
	}

	_, samples, err := readJSONLFile(filepath.Join(dir, "sampler_test_model.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 {
		t.Fatalf("got %d samples, want 1", len(samples))
	}
	got := samples[0]
	if got.RecordType != recordTypeCacheMetricsSample {
		t.Errorf("record_type = %q", got.RecordType)
	}
	if got.Engine != "vllm" {
		t.Errorf("engine = %q, want vllm", got.Engine)
	}
	// The first sighting of an endpoint sets a BASELINE and contributes nothing.
	for band, v := range got.Bands {
		if v != 0 {
			t.Errorf("first sample band %q = %d, want 0: it is the baseline, not a measurement", band, v)
		}
	}
	if got.EndpointsOK != 1 || got.EndpointsTotal != 1 {
		t.Errorf("coverage = %d/%d, want 1/1", got.EndpointsOK, got.EndpointsTotal)
	}
	if got.ActiveDatasetTokens != 5000 || got.ActiveSeries != 2 {
		t.Errorf("active dataset = %d/%d, want 5000/2", got.ActiveDatasetTokens, got.ActiveSeries)
	}
}

func TestStartVLLMMetricsSamplerLifecycle(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(promFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "lifecycle_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}

	// Non-chat/completions spec or nil writer: sampler must not start.
	if s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=anthropic", nil, newActiveDatasetTracker(), rdw); s != nil {
		t.Fatal("sampler started for non-chat/completions spec")
	}
	if s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=openai_vllm", nil, newActiveDatasetTracker(), nil); s != nil {
		t.Fatal("sampler started without writer")
	}

	tracker := newActiveDatasetTracker()
	tracker.Update(1, 123)
	s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=openai_vllm", nil, tracker, rdw)
	if s == nil {
		t.Fatal("sampler did not start for openai_vllm spec")
	}
	// The immediate first sample is asynchronous — wait for it to land before
	// stopping (stop cancels the context, which would abort an unstarted fetch).
	path := filepath.Join(dir, "lifecycle_model.jsonl")
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil && strings.Contains(string(b), recordTypeCacheMetricsSample) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	s.stop()
	if err := rdw.close(); err != nil {
		t.Fatal(err)
	}
	_, samples, err := readJSONLFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 {
		t.Fatalf("got %d samples after immediate first sample, want 1", len(samples))
	}
	if samples[0].ActiveDatasetTokens != 123 || samples[0].ActiveSeries != 1 {
		t.Errorf("sample = %+v", samples[0])
	}
}

// A plain type=openai spec — what a bare host:port resolves to — must still
// collect, since the endpoint behind it is commonly vLLM.
func TestStartVLLMMetricsSamplerSpeculativeCollects(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(promFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "speculative_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=openai", nil, newActiveDatasetTracker(), rdw)
	if s == nil {
		t.Fatal("sampler did not start for a chat/completions spec")
	}
	if s.explicit {
		t.Error("type=openai sampler should be speculative, not explicit")
	}

	path := filepath.Join(dir, "speculative_model.jsonl")
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil && strings.Contains(string(b), recordTypeCacheMetricsSample) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	s.stop()
	if err := rdw.close(); err != nil {
		t.Fatal(err)
	}
	_, samples, err := readJSONLFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 {
		t.Fatalf("got %d samples, want 1", len(samples))
	}
}

// The give-up rule, exercised through sampleOnce directly so the test doesn't
// have to wait out metricsSampleInterval between attempts.
func TestVLLMMetricsSamplerGiveUp(t *testing.T) {
	// A server that has no /metrics at all — a public API, a gateway, or any
	// non-vLLM OpenAI-compatible endpoint.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	newSampler := func(explicit bool) (*metricsSampler, *[]string) {
		var logs []string
		return &metricsSampler{
			engine:   vllmEngine,
			model:    "m",
			urls:     []string{srv.URL + "/metrics"},
			explicit: explicit,
			tracker:  newActiveDatasetTracker(),
			interval: time.Minute,
			client:   &http.Client{},
			now:      time.Now,
			logf:     func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) },
		}, &logs
	}

	t.Run("speculative gives up after the budget", func(t *testing.T) {
		s, logs := newSampler(false)
		for i := 1; i < metricsSpeculativeBudget; i++ {
			if !s.sampleOnce(context.Background()) {
				t.Fatalf("gave up after %d attempts, budget is %d", i, metricsSpeculativeBudget)
			}
		}
		if s.sampleOnce(context.Background()) {
			t.Fatalf("still polling after %d attempts, budget is %d", metricsSpeculativeBudget, metricsSpeculativeBudget)
		}
		if len(*logs) != 1 {
			t.Errorf("want exactly one log line on give-up, got %v", *logs)
		}
	})

	t.Run("explicit keeps polling", func(t *testing.T) {
		s, logs := newSampler(true)
		for i := 0; i < metricsSpeculativeBudget+3; i++ {
			if !s.sampleOnce(context.Background()) {
				t.Fatalf("explicit sampler gave up on attempt %d", i+1)
			}
		}
		// One line for the first failure, then silence — a down upstream must
		// not spam a line per minute for the whole run.
		if len(*logs) != 1 {
			t.Errorf("want exactly one log line across repeated failures, got %v", *logs)
		}
	})

	t.Run("cancellation is not a failure", func(t *testing.T) {
		s, _ := newSampler(false)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if s.sampleOnce(ctx) {
			t.Error("cancelled sample should stop the loop")
		}
		if s.consecFails != 0 {
			t.Errorf("cancellation spent %d of the speculative budget", s.consecFails)
		}
	})
}

// One good sample retires the speculative budget: later failures are blips to
// ride out, not evidence the guess was wrong.
func TestVLLMMetricsSamplerSuccessRetiresBudget(t *testing.T) {
	var serve atomic.Bool
	serve.Store(true)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !serve.Load() {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(promFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "retire_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rdw.close() }()

	s := &metricsSampler{
		engine:   vllmEngine,
		model:    "m",
		urls:     []string{srv.URL + "/metrics"},
		tracker:  newActiveDatasetTracker(),
		rdw:      rdw,
		interval: time.Minute,
		client:   &http.Client{},
		now:      time.Now,
		logf:     func(string, ...any) {},
	}
	if !s.sampleOnce(context.Background()) {
		t.Fatal("first sample should have succeeded")
	}
	serve.Store(false)
	for i := 0; i < metricsSpeculativeBudget+3; i++ {
		if !s.sampleOnce(context.Background()) {
			t.Fatalf("gave up on attempt %d after a proven-good endpoint went down", i+1)
		}
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
	a := newFakeEngineServer(t, renderVLLMFixture)
	b := newFakeEngineServer(t, renderVLLMFixture)
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "override_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	// A serving spec pointing somewhere with no /metrics at all — the router.
	s := startVLLMMetricsSampler(context.Background(),
		"dynamic/http://router.invalid:9000/v1,type=openai_vllm",
		[]string{strings.TrimSuffix(a.url(), "/metrics"), b.url()}, // bare endpoint AND explicit /metrics
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
