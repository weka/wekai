package benchmark

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const sglangPromFixture = `# HELP sglang:cache_hit_rate The cache hit rate.
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="m",engine_type="unified"} 0.75
sglang:num_running_reqs{model_name="m"} 3
`

func TestParseCacheHitRateGauge(t *testing.T) {
	avg, ok, err := parseCacheHitRateGauge(strings.NewReader(sglangPromFixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if avg != 0.75 {
		t.Fatalf("got %v, want 0.75", avg)
	}
}

func TestParseCacheHitRateGaugeAveragesMultipleInstances(t *testing.T) {
	in := `sglang:cache_hit_rate{model_name="m",engine="0"} 0.6
sglang:cache_hit_rate{model_name="m",engine="1"} 0.4
`
	avg, ok, err := parseCacheHitRateGauge(strings.NewReader(in))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if avg != 0.5 {
		t.Fatalf("got %v, want 0.5", avg)
	}
}

func TestParseCacheHitRateGaugeAbsent(t *testing.T) {
	avg, ok, err := parseCacheHitRateGauge(strings.NewReader("vllm:prompt_tokens_by_source_total{source=\"local_compute\"} 7\n"))
	if err != nil {
		t.Fatalf("parse should tolerate an unrelated family: %v", err)
	}
	if ok {
		t.Fatalf("expected ok=false, got avg=%v", avg)
	}
}

func TestParseCacheHitRateGaugeGarbage(t *testing.T) {
	avg, ok, err := parseCacheHitRateGauge(strings.NewReader("<html>not prometheus</html>\n"))
	if err != nil {
		t.Fatalf("parse should tolerate garbage: %v", err)
	}
	if ok {
		t.Fatalf("expected ok=false, got avg=%v", avg)
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
// endpoint.
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
		if s := startSGLangMetricsSampler(context.Background(), spec, rdw); s != nil {
			s.stop()
			t.Errorf("spec %q: expected no sampler, got one", spec)
		}
	}
}

// TestSGLangMetricsSamplerCollects drives a real sampler against a fake
// SGLang /metrics endpoint and confirms it writes a sample carrying the
// scraped gauge average.
func TestSGLangMetricsSamplerCollects(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(sglangPromFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "sglang_sampler_test_model", time.Now())
	if err != nil {
		t.Fatalf("newRequestDataWriter: %v", err)
	}

	// Drive sampleOnce synchronously (same pattern as
	// TestVLLMMetricsSamplerCollects) rather than starting the background
	// goroutine and mutating fields on it afterward — the latter races with
	// the goroutine under -race.
	s := &sglangMetricsSampler{
		model:  "dynamic/" + srv.URL + "/v1,type=openai_sglang,model=m",
		urls:   []string{srv.URL + "/metrics"},
		rdw:    rdw,
		client: srv.Client(),
		now:    func() time.Time { return time.Unix(1721000000, 0) },
	}
	if !s.sampleOnce(context.Background()) {
		t.Fatal("sampleOnce reported it should stop, want keepPolling=true")
	}
	if err := rdw.close(); err != nil {
		t.Fatalf("close rdw: %v", err)
	}

	// readJSONLFile (used by vllm_metrics_test.go's equivalent) deliberately
	// skips record types it doesn't recognize — sglang_metrics_sample isn't
	// wired into the shared reader/report path yet (see sglang_metrics.go's
	// package comment), so read the raw line directly instead.
	samples := readSGLangSamples(t, filepath.Join(dir, "sglang_sampler_test_model.jsonl"))
	if len(samples) != 1 {
		t.Fatalf("got %d sglang samples, want 1: %v", len(samples), samples)
	}
	got := samples[0]
	if got.RecordType != recordTypeSGLangMetricsSample {
		t.Errorf("record_type = %q", got.RecordType)
	}
	if got.CacheHitRate != 0.75 {
		t.Errorf("cache_hit_rate = %v, want 0.75", got.CacheHitRate)
	}
	if got.EndpointsOK != 1 || got.EndpointsTotal != 1 {
		t.Errorf("coverage = %d/%d, want 1/1", got.EndpointsOK, got.EndpointsTotal)
	}
}

// readSGLangSamples reads only the sglang_metrics_sample rows from a
// request-data JSONL file, mirroring readJSONLFile's record_type dispatch for
// the one record type it doesn't (yet) recognize.
func readSGLangSamples(t *testing.T, path string) []sglangMetricsSample {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	var out []sglangMetricsSample
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var probe struct {
			RecordType string `json:"record_type"`
		}
		if err := json.Unmarshal(sc.Bytes(), &probe); err != nil {
			continue
		}
		if probe.RecordType != recordTypeSGLangMetricsSample {
			continue
		}
		var s sglangMetricsSample
		if err := json.Unmarshal(sc.Bytes(), &s); err != nil {
			t.Fatalf("unmarshal sglang sample: %v", err)
		}
		out = append(out, s)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", path, err)
	}
	return out
}
