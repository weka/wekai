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
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── fold(): the reset-safe, mid-run-aware delta accumulator ────────────────
//
// fold is engine-agnostic (it only ever sees raw label->value maps, never
// engine semantics), so it's tested directly here rather than through a full
// sampler/HTTP round trip.

// TestFoldEndpointsVeryFirstScrapeIsAPureBaseline: nothing in s.prev[url] at
// all means every key in this scrape is a baseline — these pods carry
// counters from whatever ran before the benchmark started, and a run wants
// only what happened DURING it.
func TestFoldEndpointsVeryFirstScrapeIsAPureBaseline(t *testing.T) {
	s := &metricsSampler{prev: map[string]map[string]float64{}, totals: map[string]float64{}}
	s.fold("u1", map[string]float64{"a": 1_000_000, "b": 2_000_000})
	if s.totals["a"] != 0 || s.totals["b"] != 0 {
		t.Fatalf("endpoint's first scrape ever must contribute nothing: totals=%+v", s.totals)
	}
	s.fold("u1", map[string]float64{"a": 1_000_100, "b": 2_000_050})
	if s.totals["a"] != 100 || s.totals["b"] != 50 {
		t.Errorf("totals after one real interval = %+v, want a=100 b=50", s.totals)
	}
}

// TestFoldMidRunBornSeriesCreditsFullDelta is the regression test for the
// bug this refactor fixes: a series that is absent from an endpoint's FIRST
// scrape and appears nonzero on a LATER scrape (the endpoint has already
// scraped successfully before — this is not its first sighting) must be
// credited in full on the scrape where it appears, not discarded as a false
// "baseline". This is exactly sglang's host/storage cache_source labels,
// which are only created on their first nonzero hit: the OLD code's
// `!had` branch didn't distinguish "brand new endpoint" from "brand new key
// on a known endpoint", so it silently treated both as a baseline and
// dropped the series' entire accumulated value — even though
// prompt_tokens_total already counted those tokens, so they landed in
// "compute" instead.
func TestFoldMidRunBornSeriesCreditsFullDelta(t *testing.T) {
	s := &metricsSampler{prev: map[string]map[string]float64{}, totals: map[string]float64{}}
	// Scrape 1: only "a" is present. This endpoint's first scrape ever, so it
	// is a genuine baseline.
	s.fold("u1", map[string]float64{"a": 500})
	if got := s.totals["a"]; got != 0 {
		t.Fatalf("endpoint's very first scrape must be a pure baseline: a=%v, want 0", got)
	}
	if _, ok := s.totals["b"]; ok {
		t.Fatalf("b must not appear in totals before it is ever scraped")
	}
	// Scrape 2: "a" advances, and "b" appears for the FIRST time — born
	// mid-run. The endpoint has already scraped successfully once, so this is
	// NOT a baseline: b's whole value must count.
	s.fold("u1", map[string]float64{"a": 600, "b": 120})
	if got := s.totals["a"]; got != 100 {
		t.Errorf("a's delta = %v, want 100", got)
	}
	if got := s.totals["b"]; got != 120 {
		t.Errorf("b's full value on its first appearance must be credited (mid-run-born series), "+
			"got %v, want 120 — the bug this regresses against silently attributed this to "+
			"\"compute\" by discarding it as a false baseline", got)
	}
	// Scrape 3: "b" keeps advancing normally like any other series.
	s.fold("u1", map[string]float64{"a": 650, "b": 150})
	if got := s.totals["b"]; got != 150 {
		t.Errorf("b's total after +30 = %v, want 150", got)
	}
}

// TestFoldRestartCountsFromZero: a counter going backwards is a restart, and
// the post-restart value counts from zero rather than going negative.
func TestFoldRestartCountsFromZero(t *testing.T) {
	s := &metricsSampler{prev: map[string]map[string]float64{}, totals: map[string]float64{}}
	s.fold("u1", map[string]float64{"a": 500})
	s.fold("u1", map[string]float64{"a": 900}) // +400
	if s.totals["a"] != 400 {
		t.Fatalf("totals = %v, want 400", s.totals["a"])
	}
	s.fold("u1", map[string]float64{"a": 50}) // restart: counts from zero
	if s.totals["a"] != 450 {
		t.Errorf("totals after restart = %v, want 450 (400 + 50 counted from zero)", s.totals["a"])
	}
	if s.resets != 1 {
		t.Errorf("resets = %d, want 1", s.resets)
	}
}

// ── Shared sampler mechanics, table-driven over cacheEngines ───────────────
//
// engineFixture drives the genuinely engine-agnostic parts of metricsSampler
// (poll loop, ADT wiring, graceful degradation, coverage reporting) through a
// minimal per-engine fake /metrics server, without needing each engine's full
// label vocabulary: primaryKey is one raw counter label the engine's Parse
// produces, and bandKey is the canonical band it lands in after Derive.
type engineFixture struct {
	engine     cacheEngine
	render     func(vals map[string]int64) string
	primaryKey string
	bandKey    string
}

func renderVLLMFixture(vals map[string]int64) string {
	return fmt.Sprintf(`# TYPE vllm:prompt_tokens_by_source counter
vllm:prompt_tokens_by_source_total{source="local_compute",model_name="m"} %d
vllm:prompt_tokens_by_source_total{source="local_cache_hit",model_name="m"} %d
vllm:prompt_tokens_by_source_total{source="external_kv_transfer",model_name="m"} %d
`, vals["local_compute"], vals["local_cache_hit"], vals["external_kv_transfer"])
}

func renderSGLangFixture(vals map[string]int64) string {
	return fmt.Sprintf(`# TYPE sglang:cached_tokens_total counter
sglang:cached_tokens_total{cache_source="device",model_name="m"} %d
sglang:cached_tokens_total{cache_source="host",model_name="m"} %d
sglang:cached_tokens_total{cache_source="storage",model_name="m"} %d
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="m"} %d
`, vals["device"], vals["host"], vals["storage"], vals[sglangPromptTokensKey])
}

var engineFixtures = []engineFixture{
	{engine: vllmEngine, render: renderVLLMFixture, primaryKey: "local_compute", bandKey: "compute"},
	{engine: sglangEngine, render: renderSGLangFixture, primaryKey: "device", bandKey: "gpu"},
}

// fakeEngineServer serves one engine's /metrics exposition from settable raw
// totals (in that engine's own label vocabulary), or fails on demand.
type fakeEngineServer struct {
	mu     sync.Mutex
	vals   map[string]int64
	fail   bool
	render func(map[string]int64) string
	srv    *httptest.Server
}

func newFakeEngineServer(t *testing.T, render func(map[string]int64) string) *fakeEngineServer {
	t.Helper()
	f := &fakeEngineServer{vals: map[string]int64{}, render: render}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		fail, body := f.fail, f.render(f.vals)
		f.mu.Unlock()
		if fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeEngineServer) set(key string, v int64) {
	f.mu.Lock()
	f.vals[key] = v
	f.mu.Unlock()
}

func (f *fakeEngineServer) add(key string, d int64) {
	f.mu.Lock()
	f.vals[key] += d
	f.mu.Unlock()
}

func (f *fakeEngineServer) setFail(v bool) {
	f.mu.Lock()
	f.fail = v
	f.mu.Unlock()
}

func (f *fakeEngineServer) url() string { return f.srv.URL + "/metrics" }

// metricsCapture reads back what a sampler actually persisted, through the
// real writer and the real JSONL encoding — the same path the report
// consumes, so a field that fails to serialise fails here too.
type metricsCapture struct {
	t    *testing.T
	path string
}

func (c *metricsCapture) all() []cacheMetricsSample {
	c.t.Helper()
	f, err := os.Open(c.path)
	if err != nil {
		c.t.Fatalf("open jsonl: %v", err)
	}
	defer f.Close()
	var out []cacheMetricsSample
	dec := json.NewDecoder(f)
	for {
		var s cacheMetricsSample
		if err := dec.Decode(&s); err == io.EOF {
			break
		} else if err != nil {
			c.t.Fatalf("decode jsonl: %v", err)
		}
		if s.RecordType == recordTypeCacheMetricsSample {
			out = append(out, s)
		}
	}
	return out
}

func newTestMetricsSampler(t *testing.T, engine cacheEngine, urls []string, cap *metricsCapture) *metricsSampler {
	t.Helper()
	cap.t = t
	cap.path = filepath.Join(t.TempDir(), "samples.jsonl")
	f, err := os.Create(cap.path)
	if err != nil {
		t.Fatalf("create jsonl: %v", err)
	}
	t.Cleanup(func() { f.Close() })
	rdw := &requestDataWriter{f: f, enc: json.NewEncoder(f)}
	return &metricsSampler{
		engine:   engine,
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
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			a, b := newFakeEngineServer(t, fx.render), newFakeEngineServer(t, fx.render)
			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{a.url(), b.url()}, &cap)
			ctx := context.Background()

			a.set(fx.primaryKey, 1_000_000)
			b.set(fx.primaryKey, 2_000_000)
			s.sampleOnce(ctx)
			if got := cap.all()[0].Bands[fx.bandKey]; got != 0 {
				t.Errorf("first sample = %d, want 0: the baseline must not import counters "+
					"accumulated before the run started", got)
			}

			a.add(fx.primaryKey, 100)
			b.add(fx.primaryKey, 200)
			s.sampleOnce(ctx)
			if got := cap.all()[1].Bands[fx.bandKey]; got != 300 {
				t.Errorf("after +100/+200 the total is %d, want 300", got)
			}
		})
	}
}

func TestSamplerRestartCountsFromZeroInsteadOfGoingBackwards(t *testing.T) {
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			a, b := newFakeEngineServer(t, fx.render), newFakeEngineServer(t, fx.render)
			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{a.url(), b.url()}, &cap)
			ctx := context.Background()

			a.set(fx.primaryKey, 500)
			b.set(fx.primaryKey, 500)
			s.sampleOnce(ctx) // baseline
			a.set(fx.primaryKey, 900)
			b.set(fx.primaryKey, 700)
			s.sampleOnce(ctx) // +400 +200 = 600
			if got := cap.all()[1].Bands[fx.bandKey]; got != 600 {
				t.Fatalf("total = %d, want 600", got)
			}

			a.set(fx.primaryKey, 50) // restart: resets to 0, climbs to 50
			b.set(fx.primaryKey, 750)
			s.sampleOnce(ctx)

			got := cap.all()[2].Bands[fx.bandKey]
			if got < cap.all()[1].Bands[fx.bandKey] {
				t.Errorf("total went backwards %d -> %d across a restart", cap.all()[1].Bands[fx.bandKey], got)
			}
			if got != 700 {
				t.Errorf("total = %d, want 700 (600 + 50 from the restarted pod counted from zero + 50)", got)
			}
			if cap.all()[2].Resets != 1 {
				t.Errorf("Resets = %d, want 1", cap.all()[2].Resets)
			}
		})
	}
}

func TestSamplerOneEndpointFailingCostsOnlyItsOwnDelta(t *testing.T) {
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			a, b := newFakeEngineServer(t, fx.render), newFakeEngineServer(t, fx.render)
			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{a.url(), b.url()}, &cap)
			ctx := context.Background()

			s.sampleOnce(ctx) // baseline at 0/0

			a.add(fx.primaryKey, 100)
			b.add(fx.primaryKey, 999)
			b.setFail(true)
			s.sampleOnce(ctx)

			smp := cap.all()[1]
			if smp.Bands[fx.bandKey] != 100 {
				t.Errorf("total = %d, want 100: the healthy endpoint's delta must land even though "+
					"its peer failed", smp.Bands[fx.bandKey])
			}
			if smp.EndpointsOK != 1 || smp.EndpointsTotal != 2 {
				t.Errorf("coverage %d/%d, want 1/2", smp.EndpointsOK, smp.EndpointsTotal)
			}

			b.setFail(false)
			s.sampleOnce(ctx)
			if got := cap.all()[2].Bands[fx.bandKey]; got != 1099 {
				t.Errorf("total = %d, want 1099: the recovered endpoint's work during the outage is "+
					"counted on its next successful read, not dropped", got)
			}
		})
	}
}

func TestSamplerEndpointThatNeverReturnsKeepsItsContribution(t *testing.T) {
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			a, b := newFakeEngineServer(t, fx.render), newFakeEngineServer(t, fx.render)
			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{a.url(), b.url()}, &cap)
			ctx := context.Background()

			s.sampleOnce(ctx)
			a.add(fx.primaryKey, 400)
			b.add(fx.primaryKey, 600)
			s.sampleOnce(ctx)
			if got := cap.all()[1].Bands[fx.bandKey]; got != 1000 {
				t.Fatalf("total = %d, want 1000", got)
			}

			b.setFail(true) // gone for good
			for range 3 {
				a.add(fx.primaryKey, 10)
				s.sampleOnce(ctx)
			}
			last := cap.all()[len(cap.all())-1]
			if last.Bands[fx.bandKey] != 1030 {
				t.Errorf("total = %d, want 1030: a departed endpoint stops adding, it does not subtract",
					last.Bands[fx.bandKey])
			}
			if last.EndpointsOK != 1 {
				t.Errorf("coverage %d/%d", last.EndpointsOK, last.EndpointsTotal)
			}
		})
	}
}

func TestSamplerNoEndpointsAnsweringIsRecordedAsUnobserved(t *testing.T) {
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			a := newFakeEngineServer(t, fx.render)
			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{a.url()}, &cap)
			ctx := context.Background()

			s.sampleOnce(ctx)
			a.setFail(true)
			if !s.sampleOnce(ctx) {
				t.Error("explicit sampler must keep polling after a total outage")
			}

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
		})
	}
}

// TestSamplerGracefulDegradation: an endpoint that cannot be reached at all
// (connection refused, not just a 500) contributes no delta and no panic —
// but the interval is still RECORDED, at zero coverage.
func TestSamplerGracefulDegradation(t *testing.T) {
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.NotFound(w, r)
			}))
			defer srv.Close()

			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{srv.URL + "/metrics"}, &cap)
			s.sampleOnce(context.Background())

			srv.Close()
			s.client = &http.Client{}
			s.sampleOnce(context.Background())

			samples := cap.all()
			if len(samples) != 2 {
				t.Fatalf("got %d samples, want 2", len(samples))
			}
			for i, smp := range samples {
				if smp.EndpointsOK != 0 {
					t.Errorf("sample %d: EndpointsOK = %d, want 0", i, smp.EndpointsOK)
				}
				if smp.EndpointsTotal != 1 {
					t.Errorf("sample %d: EndpointsTotal = %d, want 1", i, smp.EndpointsTotal)
				}
				for band, v := range smp.Bands {
					if v != 0 {
						t.Errorf("sample %d: an unreadable endpoint contributed band %q=%d", i, band, v)
					}
				}
			}
		})
	}
}

// TestActiveDatasetTracker is engine-agnostic: both samplers share the one
// *activeDatasetTracker type.
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
	tr.Reset(1)
	if tok, n := tr.Sum(); tok != 500 || n != 1 {
		t.Fatalf("after reset: got %d/%d, want 500/1", tok, n)
	}
	tr.Update(1, 64)
	if tok, n := tr.Sum(); tok != 564 || n != 2 {
		t.Fatalf("after re-populate: got %d/%d, want 564/2", tok, n)
	}
}

func TestADTWiringReachesSamples(t *testing.T) {
	for _, fx := range engineFixtures {
		t.Run(fx.engine.Name, func(t *testing.T) {
			a := newFakeEngineServer(t, fx.render)
			var cap metricsCapture
			s := newTestMetricsSampler(t, fx.engine, []string{a.url()}, &cap)
			s.tracker.Update(1, 4000)
			s.tracker.Update(2, 1000)
			s.sampleOnce(context.Background())
			got := cap.all()[0]
			if got.ActiveDatasetTokens != 5000 || got.ActiveSeries != 2 {
				t.Errorf("active dataset = %d/%d, want 5000/2", got.ActiveDatasetTokens, got.ActiveSeries)
			}
		})
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

// ── buildSampleViz ───────────────────────────────────────────────────────

// bandTokens looks up one named band's delta within a segment, failing the
// test if the band is absent.
func bandTokens(t *testing.T, seg vizSampleSegment, name string) float64 {
	t.Helper()
	for _, b := range seg.Bands {
		if b.Name == name {
			return b.Tokens
		}
	}
	t.Fatalf("segment %+v has no band %q", seg, name)
	return 0
}

func TestBuildSampleVizDeltasAndClamp(t *testing.T) {
	t0 := time.Unix(1000, 0)
	mk := func(offsetSec int, compute, host, external, adt int64, as int) cacheMetricsSample {
		return cacheMetricsSample{
			RecordType:          recordTypeCacheMetricsSample,
			TS:                  t0.Add(time.Duration(offsetSec) * time.Second),
			Engine:              "vllm",
			Bands:               map[string]int64{"compute": compute, "host": host, "external": external},
			ActiveDatasetTokens: adt,
			ActiveSeries:        as,
		}
	}
	// Out of order on purpose; sample 3 has a counter reset (values drop).
	samples := []cacheMetricsSample{
		mk(60, 1500, 300, 42, 2000, 2),
		mk(0, 1000, 200, 40, 1000, 1),
		mk(120, 100, 50, 5, 3000, 3), // reset: deltas must clamp to 0
		mk(180, 200, 80, 6, 2500, 2),
	}
	mix, adt := buildSampleViz(samples)
	if len(mix) != 3 {
		t.Fatalf("got %d segments, want 3", len(mix))
	}
	if len(mix[0].Bands) != 3 {
		t.Fatalf("seg0 has %d bands, want 3 (compute/host/external)", len(mix[0].Bands))
	}
	if bandTokens(t, mix[0], "compute") != 500 || bandTokens(t, mix[0], "host") != 100 || bandTokens(t, mix[0], "external") != 2 {
		t.Errorf("seg0 deltas = %+v, want compute=500 host=100 external=2", mix[0])
	}
	if bandTokens(t, mix[1], "compute") != 0 || bandTokens(t, mix[1], "host") != 0 || bandTokens(t, mix[1], "external") != 0 {
		t.Errorf("seg1 (reset) deltas = %+v, want all 0", mix[1])
	}
	if bandTokens(t, mix[2], "compute") != 100 || bandTokens(t, mix[2], "host") != 30 || bandTokens(t, mix[2], "external") != 1 {
		t.Errorf("seg2 deltas = %+v, want compute=100 host=30 external=1", mix[2])
	}
	if mix[0].T0 != float64(t0.UnixMilli()) || mix[0].T1 != float64(t0.Add(60*time.Second).UnixMilli()) {
		t.Errorf("seg0 span = %v..%v", mix[0].T0, mix[0].T1)
	}
	if len(adt) != 4 || adt[0].Tokens != 1000 || adt[3].Tokens != 2500 || adt[3].Series != 2 {
		t.Errorf("adt points wrong: %+v", adt)
	}
}

func TestBuildSampleVizSGLangFourBands(t *testing.T) {
	t0 := time.Unix(1000, 0)
	mk := func(offsetSec int, compute, gpu, host, storage int64) cacheMetricsSample {
		return cacheMetricsSample{
			RecordType: recordTypeCacheMetricsSample,
			TS:         t0.Add(time.Duration(offsetSec) * time.Second),
			Engine:     "sglang",
			Bands:      map[string]int64{"compute": compute, "gpu": gpu, "host": host, "storage": storage},
		}
	}
	samples := []cacheMetricsSample{mk(0, 1000, 200, 50, 40), mk(60, 1500, 300, 100, 42)}
	mix, _ := buildSampleViz(samples)
	if len(mix) != 1 || len(mix[0].Bands) != 4 {
		t.Fatalf("got %d segments / %d bands, want 1/4", len(mix), len(mix[0].Bands))
	}
	// Canonical order: compute, gpu, host, storage, external — sglang samples
	// never carry "external", so only the first four appear, in that order.
	wantOrder := []string{"compute", "gpu", "host", "storage"}
	for i, b := range mix[0].Bands {
		if b.Name != wantOrder[i] {
			t.Errorf("band %d = %q, want %q (canonical order)", i, b.Name, wantOrder[i])
		}
	}
	if bandTokens(t, mix[0], "compute") != 500 || bandTokens(t, mix[0], "gpu") != 100 ||
		bandTokens(t, mix[0], "host") != 50 || bandTokens(t, mix[0], "storage") != 2 {
		t.Errorf("deltas = %+v, want compute=500 gpu=100 host=50 storage=2", mix[0])
	}
}

func TestBuildSampleVizEmpty(t *testing.T) {
	mix, adt := buildSampleViz(nil)
	if mix != nil || adt != nil {
		t.Fatalf("expected nil/nil, got %v %v", mix, adt)
	}
}

// TestCoverageReachesTheReport: the numbers are worthless if the chart cannot
// see them.
func TestCoverageReachesTheReport(t *testing.T) {
	base := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	mix, _ := buildSampleViz([]cacheMetricsSample{
		{TS: base, Engine: "vllm", Bands: map[string]int64{"compute": 0}, EndpointsOK: 8, EndpointsTotal: 8},
		{TS: base.Add(time.Minute), Engine: "vllm", Bands: map[string]int64{"compute": 100}, EndpointsOK: 3, EndpointsTotal: 8},
	})
	if len(mix) != 1 {
		t.Fatalf("%d segments, want 1", len(mix))
	}
	if mix[0].EndpointsOK != 3 || mix[0].EndpointsTotal != 8 {
		t.Errorf("segment coverage %d/%d, want 3/8", mix[0].EndpointsOK, mix[0].EndpointsTotal)
	}
}

// ── Legacy on-disk record translation ───────────────────────────────────────

// TestReadJSONLFileTranslatesLegacySampleRecords: a trace directory written
// by a pre-unification wekai (real cluster runs already produced files like
// this) must still read correctly — both legacy record shapes translate into
// the current unified cacheMetricsSample, with the right Engine/Bands
// mapping and every other field carried straight across.
func TestReadJSONLFileTranslatesLegacySampleRecords(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "legacy.jsonl")
	lines := []string{
		// A request row must not be disturbed by samples routing elsewhere.
		`{"start_time":"2026-07-21T10:00:00Z","end_time":"2026-07-21T10:00:01Z","model":"m","series_num":1,"request_num":1,"input_tokens":50}`,
		`{"record_type":"vllm_metrics_sample","ts":"2026-07-21T10:00:30Z","model":"m","sources":{"compute":1000,"local_cache":200,"external_cache":40},"endpoints_ok":1,"endpoints_total":1,"resets":2,"active_dataset_tokens":5000,"active_series":3}`,
		`{"record_type":"sglang_metrics_sample","ts":"2026-07-21T10:01:30Z","model":"m2","sources":{"compute":900,"device":300,"host":150,"storage":60},"endpoints_ok":2,"endpoints_total":2,"resets":1,"active_dataset_tokens":6000,"active_series":4}`,
		`not json at all`,
		`{"record_type":"future_unknown_type","whatever":1}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	records, samples, _, _, err := readJSONLFileWithParams(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("got %d request records, want 1 (samples/unknown must not leak in as phantom requests): %+v", len(records), records)
	}
	if len(samples) != 2 {
		t.Fatalf("got %d samples, want 2", len(samples))
	}

	v := samples[0]
	if v.RecordType != recordTypeCacheMetricsSample {
		t.Errorf("translated record_type = %q, want %q", v.RecordType, recordTypeCacheMetricsSample)
	}
	if v.Engine != "vllm" {
		t.Errorf("engine = %q, want vllm", v.Engine)
	}
	if v.Bands["compute"] != 1000 || v.Bands["host"] != 200 || v.Bands["external"] != 40 {
		t.Errorf("vllm legacy bands = %+v, want compute=1000 host=200 external=40", v.Bands)
	}
	if v.Resets != 2 || v.ActiveDatasetTokens != 5000 || v.ActiveSeries != 3 || v.EndpointsOK != 1 || v.EndpointsTotal != 1 {
		t.Errorf("vllm legacy metadata lost in translation: %+v", v)
	}

	s := samples[1]
	if s.Engine != "sglang" {
		t.Errorf("engine = %q, want sglang", s.Engine)
	}
	if s.Bands["compute"] != 900 || s.Bands["gpu"] != 300 || s.Bands["host"] != 150 || s.Bands["storage"] != 60 {
		t.Errorf("sglang legacy bands = %+v, want compute=900 gpu=300 host=150 storage=60", s.Bands)
	}
	if s.Resets != 1 || s.ActiveDatasetTokens != 6000 || s.ActiveSeries != 4 || s.EndpointsOK != 2 || s.EndpointsTotal != 2 {
		t.Errorf("sglang legacy metadata lost in translation: %+v", s)
	}
}

// ── Style table: external vs storage color collision ───────────────────────

// TestMixBandStyleExternalStorageColorsDiffer is the regression test for the
// flagged bug: MIX_BAND_STYLE used to define external_kv_transfer and
// storage with the SAME color (#7C03EC), so a report comparing a vLLM arm
// against an sglang arm rendered the two as visually identical bands even
// though vLLM's "external" means "any KV connector" (not specifically WEKA
// L3) and sglang's "storage" means WEKA L3 specifically. mixBandStyleJS is
// the one shared Go string both the interactive and public report templates
// embed verbatim, so asserting on it covers both.
func TestMixBandStyleExternalStorageColorsDiffer(t *testing.T) {
	color := func(constName string) string {
		re := regexp.MustCompile(`const ` + constName + `\s*=\s*"(#[0-9a-fA-F]+)"`)
		m := re.FindStringSubmatch(mixBandStyleJS)
		if m == nil {
			t.Fatalf("%s not found in mixBandStyleJS", constName)
		}
		return m[1]
	}
	storage := color("MIX_STORAGE_COLOR")
	external := color("MIX_EXTERNAL_COLOR")
	if storage != "#7C03EC" {
		t.Errorf("MIX_STORAGE_COLOR = %s, want the documented #7C03EC (storage keeps its existing color)", storage)
	}
	if storage == external {
		t.Errorf("MIX_STORAGE_COLOR and MIX_EXTERNAL_COLOR collide at %s — vLLM's external band and "+
			"sglang's storage band would render identically on a merged chart", storage)
	}
	// Both band entries must actually reference their own dedicated color
	// constant (not fall back to a shared/default one).
	if !strings.Contains(mixBandStyleJS, `storage:  { color: MIX_STORAGE_COLOR`) {
		t.Error("storage band style does not reference MIX_STORAGE_COLOR")
	}
	if !strings.Contains(mixBandStyleJS, `external: { color: MIX_EXTERNAL_COLOR`) {
		t.Error("external band style does not reference MIX_EXTERNAL_COLOR")
	}
}
