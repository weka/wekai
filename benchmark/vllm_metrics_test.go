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
		got := vllmMetricsEndpoints(c.spec)
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

func TestActiveDatasetTracker(t *testing.T) {
	tr := newActiveDatasetTracker()
	if tok, n := tr.Sum(); tok != 0 || n != 0 {
		t.Fatalf("empty tracker: got %d/%d", tok, n)
	}
	tr.Update(1, 1000)
	tr.Update(2, 500)
	tr.Update(1, 1200) // latest snapshot replaces, not accumulates
	if tok, n := tr.Sum(); tok != 1700 || n != 2 {
		t.Fatalf("got %d tokens / %d series, want 1700/2", tok, n)
	}
	// Recycle: slot 1 resets, contributes nothing until its new session's
	// first response lands.
	tr.Reset(1)
	if tok, n := tr.Sum(); tok != 500 || n != 1 {
		t.Fatalf("after reset: got %d/%d, want 500/1", tok, n)
	}
	tr.Update(1, 64)
	if tok, n := tr.Sum(); tok != 564 || n != 2 {
		t.Fatalf("after re-populate: got %d/%d, want 564/2", tok, n)
	}
}

func TestBuildSampleVizDeltasAndClamp(t *testing.T) {
	t0 := time.Unix(1000, 0)
	mk := func(offsetSec int, c, lc, ec, adt int64, as int) vllmMetricsSample {
		return vllmMetricsSample{
			RecordType:          recordTypeVLLMMetricsSample,
			TS:                  t0.Add(time.Duration(offsetSec) * time.Second),
			Sources:             vllmSourceCounters{Compute: c, LocalCache: lc, ExternalCache: ec},
			ActiveDatasetTokens: adt,
			ActiveSeries:        as,
		}
	}
	// Out of order on purpose; sample 3 has a counter reset (values drop).
	samples := []vllmMetricsSample{
		mk(60, 1500, 300, 42, 2000, 2),
		mk(0, 1000, 200, 40, 1000, 1),
		mk(120, 100, 50, 5, 3000, 3), // reset: deltas must clamp to 0
		mk(180, 200, 80, 6, 2500, 2),
	}
	mix, adt := buildSampleViz(samples)
	if len(mix) != 3 {
		t.Fatalf("got %d segments, want 3", len(mix))
	}
	// Segment 1: 0s -> 60s.
	if mix[0].Compute != 500 || mix[0].LocalCache != 100 || mix[0].ExternalCache != 2 {
		t.Errorf("seg0 deltas = %+v, want c=500 lc=100 ec=2", mix[0])
	}
	// Segment 2 spans the counter reset: all deltas clamp at 0.
	if mix[1].Compute != 0 || mix[1].LocalCache != 0 || mix[1].ExternalCache != 0 {
		t.Errorf("seg1 (reset) deltas = %+v, want all 0", mix[1])
	}
	// Segment 3 resumes from the post-reset baseline.
	if mix[2].Compute != 100 || mix[2].LocalCache != 30 || mix[2].ExternalCache != 1 {
		t.Errorf("seg2 deltas = %+v, want c=100 lc=30 ec=1", mix[2])
	}
	if mix[0].T0 != float64(t0.UnixMilli()) || mix[0].T1 != float64(t0.Add(60*time.Second).UnixMilli()) {
		t.Errorf("seg0 span = %v..%v", mix[0].T0, mix[0].T1)
	}
	if len(adt) != 4 || adt[0].Tokens != 1000 || adt[3].Tokens != 2500 || adt[3].Series != 2 {
		t.Errorf("adt points wrong: %+v", adt)
	}
}

func TestBuildSampleVizEmpty(t *testing.T) {
	mix, adt := buildSampleViz(nil)
	if mix != nil || adt != nil {
		t.Fatalf("expected nil/nil, got %v %v", mix, adt)
	}
}

func TestReadJSONLFileMixedRecordTypes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mixed.jsonl")
	lines := []string{
		`{"start_time":"2026-07-21T10:00:00Z","end_time":"2026-07-21T10:00:01Z","ttft_ms":100,"response_time_ms":1000,"model":"dynamic/http://h:8000/v1,type=openai_vllm,alias=A","series_num":1,"request_num":1,"input_tokens":50,"cached_tokens":0,"cache_hit":false,"server_cache_confirmed":false,"is_cold_start":true,"output_tokens":10,"is_error":false,"is_empty":false,"local_cache_ratio":0}`,
		`{"record_type":"vllm_metrics_sample","ts":"2026-07-21T10:00:30Z","model":"dynamic/http://h:8000/v1,type=openai_vllm,alias=A","sources":{"compute":1000,"local_cache":200,"external_cache":40},"active_dataset_tokens":5000,"active_series":3}`,
		`not json at all`,
		`{"record_type":"future_unknown_type","whatever":1}`,
		`{"start_time":"2026-07-21T10:01:00Z","end_time":"2026-07-21T10:01:01Z","ttft_ms":50,"response_time_ms":500,"model":"dynamic/http://h:8000/v1,type=openai_vllm,alias=A","series_num":1,"request_num":2,"input_tokens":60,"cached_tokens":40,"cache_hit":true,"server_cache_confirmed":true,"is_cold_start":false,"output_tokens":10,"is_error":false,"is_empty":false,"local_cache_ratio":0.5}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	records, samples, err := readJSONLFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("got %d request records, want 2 (samples/unknown must not leak in as phantom requests): %+v", len(records), records)
	}
	if records[0].RequestNum != 1 || records[1].RequestNum != 2 {
		t.Errorf("request rows corrupted: %+v", records)
	}
	if len(samples) != 1 {
		t.Fatalf("got %d samples, want 1", len(samples))
	}
	s := samples[0]
	if s.Sources.Compute != 1000 || s.Sources.LocalCache != 200 || s.Sources.ExternalCache != 40 {
		t.Errorf("sample sources = %+v", s.Sources)
	}
	if s.ActiveDatasetTokens != 5000 || s.ActiveSeries != 3 {
		t.Errorf("sample active dataset = %d/%d", s.ActiveDatasetTokens, s.ActiveSeries)
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

	s := &vllmMetricsSampler{
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
	if got.RecordType != recordTypeVLLMMetricsSample {
		t.Errorf("record_type = %q", got.RecordType)
	}
	if got.Sources.Compute != 1500 || got.Sources.LocalCache != 300 || got.Sources.ExternalCache != 42 {
		t.Errorf("sources = %+v", got.Sources)
	}
	if got.ActiveDatasetTokens != 5000 || got.ActiveSeries != 2 {
		t.Errorf("active dataset = %d/%d, want 5000/2", got.ActiveDatasetTokens, got.ActiveSeries)
	}
}

func TestSamplerGracefulDegradation(t *testing.T) {
	// 404 endpoint: sample skipped, nothing written, no panic.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "degraded_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	s := &vllmMetricsSampler{
		model:   "m",
		urls:    []string{srv.URL + "/metrics"},
		tracker: newActiveDatasetTracker(),
		rdw:     rdw,
		client:  srv.Client(),
		now:     time.Now,
	}
	s.sampleOnce(context.Background())

	// Connection refused: close the server and sample again.
	srv.Close()
	s.client = &http.Client{}
	s.sampleOnce(context.Background())

	if err := rdw.close(); err != nil {
		t.Fatal(err)
	}
	_, samples, err := readJSONLFile(filepath.Join(dir, "degraded_model.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 0 {
		t.Fatalf("degraded endpoint must not produce samples, got %d", len(samples))
	}
	if s.skipped.Load() != 2 {
		t.Errorf("skipped = %d, want 2", s.skipped.Load())
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
	if s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=anthropic", newActiveDatasetTracker(), rdw); s != nil {
		t.Fatal("sampler started for non-chat/completions spec")
	}
	if s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=openai_vllm", newActiveDatasetTracker(), nil); s != nil {
		t.Fatal("sampler started without writer")
	}

	tracker := newActiveDatasetTracker()
	tracker.Update(1, 123)
	s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=openai_vllm", tracker, rdw)
	if s == nil {
		t.Fatal("sampler did not start for openai_vllm spec")
	}
	// The immediate first sample is asynchronous — wait for it to land before
	// stopping (stop cancels the context, which would abort an unstarted fetch).
	path := filepath.Join(dir, "lifecycle_model.jsonl")
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil && strings.Contains(string(b), recordTypeVLLMMetricsSample) {
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
	s := startVLLMMetricsSampler(context.Background(), "dynamic/"+srv.URL+"/v1,type=openai", newActiveDatasetTracker(), rdw)
	if s == nil {
		t.Fatal("sampler did not start for a chat/completions spec")
	}
	if s.explicit {
		t.Error("type=openai sampler should be speculative, not explicit")
	}

	path := filepath.Join(dir, "speculative_model.jsonl")
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(path); err == nil && strings.Contains(string(b), recordTypeVLLMMetricsSample) {
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
// have to wait out vllmMetricsSampleInterval between attempts.
func TestVLLMMetricsSamplerGiveUp(t *testing.T) {
	// A server that has no /metrics at all — a public API, a gateway, or any
	// non-vLLM OpenAI-compatible endpoint.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	newSampler := func(explicit bool) (*vllmMetricsSampler, *[]string) {
		var logs []string
		return &vllmMetricsSampler{
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
		for i := 1; i < vllmMetricsSpeculativeBudget; i++ {
			if !s.sampleOnce(context.Background()) {
				t.Fatalf("gave up after %d attempts, budget is %d", i, vllmMetricsSpeculativeBudget)
			}
		}
		if s.sampleOnce(context.Background()) {
			t.Fatalf("still polling after %d attempts, budget is %d", vllmMetricsSpeculativeBudget, vllmMetricsSpeculativeBudget)
		}
		if len(*logs) != 1 {
			t.Errorf("want exactly one log line on give-up, got %v", *logs)
		}
	})

	t.Run("explicit keeps polling", func(t *testing.T) {
		s, logs := newSampler(true)
		for i := 0; i < vllmMetricsSpeculativeBudget+3; i++ {
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

	s := &vllmMetricsSampler{
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
	for i := 0; i < vllmMetricsSpeculativeBudget+3; i++ {
		if !s.sampleOnce(context.Background()) {
			t.Fatalf("gave up on attempt %d after a proven-good endpoint went down", i+1)
		}
	}
}
