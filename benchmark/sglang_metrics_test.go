package benchmark

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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

// ── B1: storage_* prefix fold ───────────────────────────────────────────────

// TestParseSGLangCountersStorageUnderscorePrefixFoldsToStorage: sglang
// versions before upstream PR #37503 emit cache_source=f"storage_{backend}"
// (e.g. "storage_HiCacheFile") instead of plain "storage". Both must fold
// into the one "storage" key.
func TestParseSGLangCountersStorageUnderscorePrefixFoldsToStorage(t *testing.T) {
	for _, label := range []string{"storage_HiCacheFile", "storage_WekaFS", "storage"} {
		in := fmt.Sprintf(`sglang:cached_tokens_total{cache_source=%q,model_name="m"} 777`, label) + "\n"
		vals, err := parseSGLangCounters(strings.NewReader(in))
		if err != nil {
			t.Fatalf("%s: parse: %v", label, err)
		}
		if vals["storage"] != 777 {
			t.Errorf("%s: storage = %v, want 777 (folded regardless of the exact suffix)", label, vals["storage"])
		}
		if _, ok := vals[label]; ok && label != "storage" {
			t.Errorf("%s: must not ALSO appear under its own raw label — got %v", label, vals)
		}
	}
}

// TestParseSGLangCountersStorageContinuityAcrossLabelSpellingChange: a
// cluster upgrade can change the label spelling mid-fleet (one pod reporting
// "storage_HiCacheFile", another already upgraded to "storage"). Because
// parseSGLangCounters folds both into the same "storage" key BEFORE fold()
// ever sees them, the delta-accumulator's per-key continuity is unaffected by
// which spelling a given scrape happened to use.
func TestParseSGLangCountersStorageContinuityAcrossLabelSpellingChange(t *testing.T) {
	s := &metricsSampler{prev: map[string]map[string]float64{}, totals: map[string]float64{}}

	v1, err := parseSGLangCounters(strings.NewReader(`sglang:cached_tokens_total{cache_source="storage_HiCacheFile"} 100` + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	s.fold("u1", v1) // baseline

	v2, err := parseSGLangCounters(strings.NewReader(`sglang:cached_tokens_total{cache_source="storage_HiCacheFile"} 150` + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	s.fold("u1", v2)
	if s.totals["storage"] != 50 {
		t.Fatalf("storage total = %v, want 50", s.totals["storage"])
	}

	// The exposition text now spells the label differently -- continuity must
	// not break just because of that.
	v3, err := parseSGLangCounters(strings.NewReader(`sglang:cached_tokens_total{cache_source="storage"} 200` + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	s.fold("u1", v3)
	if s.totals["storage"] != 100 {
		t.Errorf("storage total after the label spelling changed = %v, want 100 (50 + another 50) — "+
			"a label-spelling change must not look like a brand new series", s.totals["storage"])
	}
}

// TestSGLangUnknownCacheSourceWarnsOnce: a cache_source value outside
// {device, host, storage*, total} must trigger exactly one warning across
// repeated occurrences (dedup is for the process lifetime — this is a
// one-shot CLI, not a long-lived server) — so a FUTURE unrecognized label
// doesn't silently vanish the way storage_HiCacheFile did before B1.
func TestSGLangUnknownCacheSourceWarnsOnce(t *testing.T) {
	var warnings []string
	orig := sglangUnknownCacheSourceWarn
	sglangUnknownCacheSourceWarn = func(label string) { warnings = append(warnings, label) }
	defer func() { sglangUnknownCacheSourceWarn = orig }()

	// A label unique to this test run, so process-lifetime dedup state from
	// other tests can't mask a regression here.
	label := "mystery_tier_for_TestSGLangUnknownCacheSourceWarnsOnce"
	in := fmt.Sprintf(`sglang:cached_tokens_total{cache_source=%q,model_name="m"} 5`, label) + "\n"
	for i := 0; i < 3; i++ {
		if _, err := parseSGLangCounters(strings.NewReader(in)); err != nil {
			t.Fatalf("attempt %d: parse: %v", i, err)
		}
	}
	if len(warnings) != 1 || warnings[0] != label {
		t.Fatalf("warnings = %v, want exactly one firing for %q across 3 occurrences", warnings, label)
	}

	// Its tokens are still parsed and accumulated (just not under a known
	// band) — an unrecognized label must not be silently dropped either.
	vals, err := parseSGLangCounters(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if vals[label] != 5 {
		t.Errorf("unrecognized label's tokens = %v, want still-parsed 5 under %q", vals[label], label)
	}
}

// TestKnownCacheSourceLabelsNeverWarn confirms the known vocabulary (device,
// host, any storage* spelling, and the backward-compat "total") never
// triggers the unknown-label warning.
func TestKnownCacheSourceLabelsNeverWarn(t *testing.T) {
	var warnings []string
	orig := sglangUnknownCacheSourceWarn
	sglangUnknownCacheSourceWarn = func(label string) { warnings = append(warnings, label) }
	defer func() { sglangUnknownCacheSourceWarn = orig }()

	if _, err := parseSGLangCounters(strings.NewReader(sglangPromFixture)); err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Errorf("known labels triggered warnings: %v", warnings)
	}
}

// ── sglangEngine.Derive ──────────────────────────────────────────────────────

// TestSGLangEngineDeriveSubtractsCachedSum verifies the one non-direct band:
// compute = prompt_tokens_total − (device + host + storage + total). sglang
// emits cache_source="total" only as a backward-compat fallback when the
// per-tier breakdown is unavailable — mutually exclusive with device/host/
// storage, never a sibling (see the package comment) — so summing it in
// alongside the tiers handles both deployment shapes without double
// counting: whichever shape is active, the other side of the sum is 0.
func TestSGLangEngineDeriveSubtractsCachedSum(t *testing.T) {
	got := sglangEngine.Derive(map[string]float64{
		sglangPromptTokensKey: 1000,
		"device":              150,
		"host":                40,
		"storage":             10,
		"total":               100,
	})
	want := map[string]float64{"compute": 700, "gpu": 150, "host": 40, "storage": 10}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("band %q = %v, want %v: %+v", k, got[k], w, got)
		}
	}
}

// TestSGLangEngineDeriveWithoutTotalLabel is the regression case for the bug:
// a real sglang deployment reporting the per-tier breakdown (device/host/
// storage) NEVER emits cache_source="total" at all -- it's an either/or
// backward-compat fallback, not a sibling aggregate. compute must still come
// out correctly when "total" is simply absent from the totals map (Go's
// zero-value default), not just when it's present-but-zero.
func TestSGLangEngineDeriveWithoutTotalLabel(t *testing.T) {
	got := sglangEngine.Derive(map[string]float64{
		sglangPromptTokensKey: 328001000,
		"device":              100,
		"host":                200,
		"storage":             50,
		// "total" intentionally absent.
	})
	wantCompute := 328001000.0 - (100 + 200 + 50)
	if got["compute"] != wantCompute {
		t.Errorf("compute = %v, want %v", got["compute"], wantCompute)
	}
}

// TestSGLangEngineDeriveClampsNegativeComputeToZero guards against transient
// scrape skew between the independently-updated prompt/cached counters
// producing a negative compute value.
func TestSGLangEngineDeriveClampsNegativeComputeToZero(t *testing.T) {
	got := sglangEngine.Derive(map[string]float64{
		sglangPromptTokensKey: 100,
		"device":              200, // more cached than total prompt tokens this interval
	})
	if got["compute"] != 0 {
		t.Errorf("compute = %v, want 0 (clamped)", got["compute"])
	}
}

// ── Eligibility ──────────────────────────────────────────────────────────────

func TestSGLangEngineEndpoints(t *testing.T) {
	cases := []struct {
		spec string
		want []string
	}{
		{"dynamic/http://localhost:8000/v1,type=openai_sglang,alias=x", []string{"http://localhost:8000/metrics"}},
		{"dynamic/http://a:8000/v1|http://b:8001/v1,type=openai_sglang", []string{"http://a:8000/metrics", "http://b:8001/metrics"}},
		// Unlike vLLM, sglang sampling is never speculative: plain "openai" or
		// the default type must NOT be eligible.
		{"dynamic/http://localhost:8000/v1,type=openai", nil},
		{"dynamic/http://localhost:8000/v1", nil},
		{"dynamic/http://localhost:8000/v1,type=openai_vllm", nil},
		{"gpt-4", nil},
	}
	for _, c := range cases {
		got := engineEndpoints(sglangEngine, c.spec)
		if len(got.urls) != len(c.want) {
			t.Errorf("spec %q: got %v, want %v", c.spec, got.urls, c.want)
			continue
		}
		for i := range got.urls {
			if got.urls[i] != c.want[i] {
				t.Errorf("spec %q: got %v, want %v", c.spec, got.urls, c.want)
				break
			}
		}
		if len(c.want) > 0 && !got.explicit {
			t.Errorf("spec %q: want explicit=true (sglang only ever starts on an exact type match)", c.spec)
		}
	}
}

// TestSGLangMetricsSamplerDoesNotStartForNonSGLang confirms
// startSGLangMetricsSampler stays inert for the vLLM/default eligibility
// shapes without an override, and for a nil tracker. An override is covered
// separately by TestSGLangMetricsURLOverride — it DOES bypass the
// explicit-type gate (see startMetricsSampler's doc in cache_metrics.go): the
// gate exists to stop a speculative guess from costing failed-probe traffic,
// and an explicit override is the operator naming exactly where to scrape,
// leaving no guess to protect against.
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
		if s := startSGLangMetricsSampler(context.Background(), spec, nil, newActiveDatasetTracker(), rdw); s != nil {
			s.stop()
			t.Errorf("spec %q: expected no sampler, got one", spec)
		}
	}
	// A valid spec but no tracker must also stay inert.
	if s := startSGLangMetricsSampler(context.Background(), "dynamic/http://localhost:8000/v1,type=openai_sglang", nil, nil, rdw); s != nil {
		s.stop()
		t.Error("expected no sampler without a tracker")
	}
}

// TestStartSGLangMetricsSamplerLifecycle drives a real sampler (via the
// background goroutine, not sampleOnce directly) against a fake sglang
// /metrics endpoint and confirms it lands one sample record.
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
	s := startSGLangMetricsSampler(context.Background(), model, nil, tracker, rdw)
	if s == nil {
		t.Fatal("sampler did not start for an openai_sglang spec")
	}
	path := filepath.Join(dir, "sglang_lifecycle_model.jsonl")
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
		t.Fatalf("got %d sglang samples, want 1: %v", len(samples), samples)
	}
	got := samples[0]
	if got.RecordType != recordTypeCacheMetricsSample {
		t.Errorf("record_type = %q", got.RecordType)
	}
	if got.Engine != "sglang" {
		t.Errorf("engine = %q, want sglang", got.Engine)
	}
	// First sighting is the baseline and contributes nothing — see fold's doc.
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

// TestStartSGLangMetricsSamplerUsesOverride confirms the override replaces
// the derived-from-model-URL endpoint once the spec has already qualified as
// type=openai_sglang -- the shape this cluster's router-fronted topology
// needs, since the router refuses to serve real backend counters on its own
// serving port. The spec's own URL deliberately points nowhere reachable, so
// the sampler can only ever land a sample via the override.
func TestStartSGLangMetricsSamplerUsesOverride(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(sglangPromFixture))
	}))
	defer srv.Close()

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "sglang_override_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}

	tracker := newActiveDatasetTracker()
	model := "dynamic/http://127.0.0.1:1,type=openai_sglang"
	s := startSGLangMetricsSampler(context.Background(), model, []string{srv.URL}, tracker, rdw)
	if s == nil {
		t.Fatal("sampler did not start for an openai_sglang spec with an override")
	}
	path := filepath.Join(dir, "sglang_override_model.jsonl")
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
		t.Fatalf("got %d sglang samples, want 1 -- the override endpoint must have been scraped instead "+
			"of the unreachable derived one", len(samples))
	}
	if samples[0].EndpointsOK != 1 || samples[0].EndpointsTotal != 1 {
		t.Errorf("coverage = %d/%d, want 1/1", samples[0].EndpointsOK, samples[0].EndpointsTotal)
	}
}

// TestSGLangSamplerNeverGivesUpRegardlessOfFailureCount: sglang sampling is
// never speculative (sglangEngine.Speculative = false), so unlike vLLM it
// must keep polling forever even past what would be vLLM's speculative
// budget — the operator asserted type=openai_sglang outright.
func TestSGLangSamplerNeverGivesUpRegardlessOfFailureCount(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	s := &metricsSampler{
		engine:   sglangEngine,
		model:    "m",
		urls:     []string{srv.URL + "/metrics"},
		explicit: true, // sglang is always explicit -- see engineEndpoints
		tracker:  newActiveDatasetTracker(),
		interval: time.Minute,
		client:   &http.Client{},
		now:      time.Now,
		logf:     func(string, ...any) {},
	}
	for i := 0; i < metricsSpeculativeBudget+5; i++ {
		if !s.sampleOnce(context.Background()) {
			t.Fatalf("sglang sampler gave up on attempt %d; it must never give up", i+1)
		}
	}
}

// Live-like exposition from an SGLang server: labels engine_type/model_name
// (no tp_rank), tier series present only when >0, and no cache_source="total"
// because per-tier details exist.
const sglangLiveDeviceOnly = `# HELP sglang:cached_tokens_total Number of cached prompt tokens.
# TYPE sglang:cached_tokens_total counter
sglang:cached_tokens_total{cache_source="device",engine_type="unified",model_name="Qwen/Qwen3-8B"} 7000.0
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{engine_type="unified",model_name="Qwen/Qwen3-8B"} 10000.0
`

const sglangLiveDeviceHost = `# TYPE sglang:cached_tokens_total counter
sglang:cached_tokens_total{cache_source="device",engine_type="unified",model_name="m"} 6000.0
sglang:cached_tokens_total{cache_source="host",engine_type="unified",model_name="m"} 1500.0
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{engine_type="unified",model_name="m"} 10000.0
`

func TestSGLangComputeLiveExposition(t *testing.T) {
	cases := []struct {
		name, in string
		want     map[string]float64
		compute  float64
	}{
		{"device only", sglangLiveDeviceOnly, map[string]float64{"device": 7000, sglangPromptTokensKey: 10000}, 3000},
		{"device+host", sglangLiveDeviceHost, map[string]float64{"device": 6000, "host": 1500, sglangPromptTokensKey: 10000}, 2500},
		{"total fallback only", `sglang:cached_tokens_total{cache_source="total",model_name="m"} 4000
sglang:prompt_tokens_total{model_name="m"} 10000
`, map[string]float64{"total": 4000, sglangPromptTokensKey: 10000}, 6000},
		{"missing prompt family", `sglang:cached_tokens_total{cache_source="device",x="y"} 5
`, map[string]float64{"device": 5}, 0},
		{"nothing", "# nothing here\n", map[string]float64{}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseSGLangCounters(strings.NewReader(c.in))
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(c.want) {
				t.Fatalf("parsed = %v, want %v", got, c.want)
			}
			for k, v := range c.want {
				if got[k] != v {
					t.Errorf("%s = %v, want %v", k, got[k], v)
				}
			}
			if cmp := sglangCompute(got); cmp != c.compute {
				t.Errorf("compute = %v, want %v", cmp, c.compute)
			}
		})
	}
}

func TestSGLangComputeClampsAtZero(t *testing.T) {
	if got := sglangCompute(map[string]float64{"device": 10, sglangPromptTokensKey: 5}); got != 0 {
		t.Errorf("compute = %v, want 0", got)
	}
}

func TestSGLangMetricsURLOverride(t *testing.T) {
	tracker := newActiveDatasetTracker()
	rdw, err := newRequestDataWriter(t.TempDir(), "override_model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = rdw.close() }()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// A non-sglang spec samples nothing without an override...
	if s := startSGLangMetricsSampler(ctx, "dynamic/http://h:1/v1,type=openai", nil, tracker, rdw); s != nil {
		s.stop()
		t.Fatal("sampler started without override")
	}
	// ...and with one it targets exactly the normalized override URLs.
	s := startSGLangMetricsSampler(ctx, "dynamic/http://h:1/v1,type=openai", []string{"http://127.0.0.1:1", "http://r:9/metrics/"}, tracker, rdw)
	if s == nil {
		t.Fatal("override did not start a sampler")
	}
	defer s.stop()
	if len(s.urls) != 2 || s.urls[0] != "http://127.0.0.1:1/metrics" || s.urls[1] != "http://r:9/metrics" {
		t.Errorf("urls = %v", s.urls)
	}
}
