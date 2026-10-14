package benchmark

// Unified cache-source metrics collector: during `benchmark auto` runs
// against an OpenAI-compatible endpoint, a per-model sampler goroutine polls
// the server's Prometheus /metrics endpoint every metricsSampleInterval and
// persists per-source cumulative prompt-token counters into the same
// --save-request-data JSONL stream as the request rows, as records of their
// own type (recordTypeCacheMetricsSample). Sampling is strictly best-effort
// and can never affect the benchmark itself.
//
// This file holds everything genuinely shared between the vLLM and sglang
// cache-source samplers: the cacheEngine struct-of-functions table that
// parameterizes the one metricsSampler over which engine it is scraping, the
// sampler itself (poll loop, delta-fold accumulator, lifecycle), the unified
// on-disk record type (with read-time translation of the two legacy
// per-engine record shapes), and the report-side transform
// (buildSampleViz/canonicalBandOrder) that turns raw cumulative samples into
// per-interval band deltas for the embedded report JS.
//
// Engine-specific parsing (what a /metrics scrape's text exposition means)
// stays in vllm_metrics.go and sglang_metrics.go, each of which defines its
// own cacheEngine value (vllmEngine / sglangEngine) that plugs into the
// machinery here.
//
// The persisted totals are a CONTINUOUS SUM OF PER-ENDPOINT DELTAS, not a sum
// of the raw counters. Summing raw counters breaks the moment the fleet
// moves: a pod restarting resets its counter to zero and the fleet sum drops,
// which downstream reads as a counter reset and discards the interval; a pod
// leaving subtracts everything it ever did. Deltas survive both — a restart
// contributes its post-restart value from zero, and an endpoint that goes
// away simply stops adding while the work it already did stays counted.
//
// One endpoint failing therefore costs only that endpoint's contribution for
// that interval, not the whole sample. What IS lost is recorded: every sample
// carries how many endpoints answered, so a flat interval can be told apart
// from an unobserved one.

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/llm"
)

const (
	recordTypeCacheMetricsSample = "cache_metrics_sample"

	// Legacy per-engine record_type values written by pre-unification wekai
	// binaries. Real cluster runs already produced files with these — see
	// readJSONLFileWithParams's translation.
	legacyRecordTypeVLLMMetricsSample   = "vllm_metrics_sample"
	legacyRecordTypeSGLangMetricsSample = "sglang_metrics_sample"

	metricsSampleInterval = 60 * time.Second
	metricsFetchTimeout   = 5 * time.Second

	// metricsSpeculativeBudget is how many consecutive failed samples a
	// speculatively-started sampler (one whose spec did not name its engine
	// outright) tolerates before concluding the endpoint simply isn't that
	// engine and shutting itself down. Bounding it matters because the
	// endpoint may be a public API that never had a /metrics: three 404s
	// spread over two minutes is a fair price for the guess, an hourly drip
	// is not. One successful sample retires the budget for good — see
	// metricsSampler.noteSuccess.
	metricsSpeculativeBudget = 3
)

// canonicalBandOrder fixes the global, engine-independent display/diffing
// order for cache-source bands. Both engines' Derive functions emit a subset
// of these keys (vLLM: compute/host/external; sglang: compute/gpu/host/
// storage) into one cacheMetricsSample.Bands map, so a report comparing both
// can iterate this one fixed list instead of needing to know which engine
// produced which sample.
var canonicalBandOrder = []string{"compute", "gpu", "host", "storage", "external"}

// cacheEngine is a struct-of-functions describing one inference engine's
// cache-source metrics: how to tell a model spec is worth polling for it, how
// to parse its /metrics exposition, and how to project its raw counters onto
// the canonical band set. vllm_metrics.go and sglang_metrics.go each define
// one value of this type; cacheEngines lists both for callers that need to
// iterate every known engine (e.g. table-driven tests).
type cacheEngine struct {
	// Name identifies the engine in the on-disk record (cacheMetricsSample.Engine)
	// and in log lines ("[<name>-metrics] ...").
	Name string
	// SpecTypes are the dynamic-model-spec type= values that select this
	// engine EXPLICITLY (e.g. "openai_vllm", "openai_sglang").
	SpecTypes []string
	// Speculative, when true, also probes a spec whose type is "openai" or
	// unset (the chat/completions default) on the theory that the endpoint
	// might be this engine behind the scenes — bounded by
	// metricsSpeculativeBudget. vLLM is speculative; sglang is not (see
	// sglang_metrics.go's package comment for why guessing there would only
	// add failed-probe traffic against every non-sglang endpoint).
	Speculative bool
	// FamilyHint names the Prometheus metric family (or families, joined by
	// "/") this engine's Parse reads, for log/error messages.
	FamilyHint string
	// FlagHint is the type= value to suggest in a give-up log message when a
	// speculative sampler abandons an endpoint.
	FlagHint string
	// Parse scans one /metrics scrape's text exposition and returns
	// cumulative counter values keyed by this engine's own label vocabulary
	// (e.g. vLLM's "local_compute"/"local_cache_hit"/"external_kv_transfer";
	// sglang's "device"/"host"/"storage"/sglangPromptTokensKey). Missing
	// families simply produce no entries; malformed lines are skipped.
	Parse func(r io.Reader) (map[string]float64, error)
	// Bands lists the canonical band names (a subset of canonicalBandOrder)
	// this engine's Derive actually populates, in display order — vLLM:
	// ["compute","host","external"]; sglang: ["compute","gpu","host","storage"].
	Bands []string
	// Derive projects one interval's accumulated raw per-label totals (the
	// fold accumulator's own label vocabulary, NOT canonical band names) onto
	// the canonical band set.
	Derive func(totals map[string]float64) map[string]float64
}

// cacheEngines lists every known engine, for table-driven callers.
var cacheEngines = []cacheEngine{vllmEngine, sglangEngine}

// metricsEligibility says whether a model spec is worth polling for an
// engine's cache metrics, and how sure we are.
type metricsEligibility struct {
	// urls are the Prometheus /metrics URLs, one per endpoint. Empty means
	// the spec doesn't qualify and no sampler should start.
	urls []string
	// explicit is true when the spec named this engine outright (type=
	// matched one of engine.SpecTypes). An explicit sampler never gives up on
	// failures; a speculative one does (see metricsSampler.noteFailure).
	explicit bool
}

// engineEndpoints derives the Prometheus /metrics URLs for a dynamic model
// spec against one engine (the /v1 API suffix is stripped to reach the server
// root, where both engines mount /metrics by convention).
func engineEndpoints(engine cacheEngine, model string) metricsEligibility {
	if !llm.IsDynamicModel(model) {
		return metricsEligibility{}
	}
	dyn, err := llm.ParseDynamicModel(model)
	if err != nil {
		return metricsEligibility{}
	}
	explicit := slices.Contains(engine.SpecTypes, dyn.Type)
	eligible := explicit
	if !eligible && engine.Speculative {
		switch dyn.Type {
		case "openai", "":
			eligible = true
		}
	}
	if !eligible {
		return metricsEligibility{}
	}
	out := metricsEligibility{explicit: explicit}
	for _, u := range dyn.BaseURLs {
		u = strings.TrimRight(u, "/")
		u = strings.TrimSuffix(u, "/v1")
		out.urls = append(out.urls, u+"/metrics")
	}
	return out
}

// normalizeMetricsURLs accepts either a bare endpoint or a full /metrics URL,
// because both are natural to type and guessing wrong costs a whole run's
// cache-source breakdown.
func normalizeMetricsURLs(in []string) []string {
	out := make([]string, 0, len(in))
	for _, u := range in {
		u = strings.TrimSpace(u)
		if u == "" {
			continue
		}
		u = strings.TrimRight(u, "/")
		if !strings.HasSuffix(u, "/metrics") {
			u = strings.TrimSuffix(u, "/v1") + "/metrics"
		}
		out = append(out, u)
	}
	return out
}

// activeDatasetTracker keeps, per live series slot, the total prompt tokens
// (server-reported cached + uncached) of that series' most recent response.
// The per-sample sum over slots approximates the "active dataset": the token
// volume a fully-warm cache would need to hold for the currently-running
// series. A recycled slot (--exhaust-sessions, replay pulling the next
// conversation) is Reset so retired sessions stop counting. Engine-agnostic —
// shared by both samplers.
type activeDatasetTracker struct {
	mu    sync.Mutex
	slots map[int]int64
}

func newActiveDatasetTracker() *activeDatasetTracker {
	return &activeDatasetTracker{slots: map[int]int64{}}
}

// Update records the latest full-prompt token count for a series slot.
func (t *activeDatasetTracker) Update(slot int, promptTokens int64) {
	t.mu.Lock()
	t.slots[slot] = promptTokens
	t.mu.Unlock()
}

// Reset drops a slot on recycle; it contributes nothing until the new
// session's first response lands via Update.
func (t *activeDatasetTracker) Reset(slot int) {
	t.mu.Lock()
	delete(t.slots, slot)
	t.mu.Unlock()
}

// Sum returns the total latest-prompt tokens across active slots and the
// number of slots contributing (completed >=1 request, not recycled).
func (t *activeDatasetTracker) Sum() (tokens int64, series int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, v := range t.slots {
		if v > 0 {
			tokens += v
			series++
		}
	}
	return tokens, series
}

// metricsSampler polls one model's endpoints for one engine's cache metrics
// and writes samples to rdw. Replaces the former vllmMetricsSampler and
// sglangMetricsSampler, which differed only in which engine they scraped.
type metricsSampler struct {
	engine   cacheEngine
	model    string
	urls     []string
	explicit bool
	tracker  *activeDatasetTracker
	rdw      *requestDataWriter
	interval time.Duration
	client   *http.Client
	now      func() time.Time
	logf     func(format string, args ...any)

	// Outcome bookkeeping. Touched only by the run goroutine, so no locking.
	consecFails   int
	everSucceeded bool

	// The delta accumulator. prev is the last raw counter seen per endpoint
	// per label (in the engine's own vocabulary); totals is the running sum
	// of deltas, which is what gets persisted (via engine.Derive).
	prev   map[string]map[string]float64
	totals map[string]float64
	resets int

	cancel context.CancelFunc
	done   chan struct{}
}

// startMetricsSampler launches the sampler goroutine for cfg.Model against
// one engine, when the spec qualifies and request data is being saved.
// Returns nil when sampling doesn't apply. Callers must invoke stop() before
// closing rdw.
//
// override, when non-empty, names the real /metrics address(es) to scrape
// instead of the ones derived from the serving spec — needed behind a router
// topology where the serving URL and the metrics-serving URL are never the
// same host:port. It REPLACES eligibility outright for every engine,
// regardless of the spec's type or the engine's Speculative flag: the type
// gate exists to stop a SPECULATIVE auto-start (guessing an engine behind a
// chat/completions endpoint with no operator confirmation) from costing
// failed-probe traffic against every non-matching endpoint, and that
// rationale is about the no-override path. An explicit override is the
// operator naming exactly where to scrape — there is no guessing left to
// protect against, so it both makes an otherwise-ineligible spec (bare host,
// wrong type) eligible and keeps the sampler polling forever rather than
// spending a speculative budget, the same way a Speculative engine's own
// eligibility already does.
func startMetricsSampler(ctx context.Context, engine cacheEngine, model string, override []string, tracker *activeDatasetTracker, rdw *requestDataWriter) *metricsSampler {
	if rdw == nil || tracker == nil {
		return nil
	}
	elig := engineEndpoints(engine, model)
	if len(override) > 0 {
		elig = metricsEligibility{urls: normalizeMetricsURLs(override), explicit: true}
	}
	if len(elig.urls) == 0 {
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	s := &metricsSampler{
		engine:   engine,
		model:    model,
		urls:     elig.urls,
		explicit: elig.explicit,
		tracker:  tracker,
		rdw:      rdw,
		interval: metricsSampleInterval,
		client:   &http.Client{},
		now:      time.Now,
		prev:     map[string]map[string]float64{},
		totals:   map[string]float64{},
		logf:     func(f string, a ...any) { fmt.Fprintf(os.Stderr, "["+engine.Name+"-metrics] "+f+"\n", a...) },
		cancel:   cancel,
		done:     make(chan struct{}),
	}
	go s.run(runCtx)
	return s
}

// startVLLMMetricsSampler and startSGLangMetricsSampler are thin wrappers
// kept so the two call sites in auto.go don't need to change: they just pin
// startMetricsSampler to one engine.
func startVLLMMetricsSampler(ctx context.Context, model string, override []string, tracker *activeDatasetTracker, rdw *requestDataWriter) *metricsSampler {
	return startMetricsSampler(ctx, vllmEngine, model, override, tracker, rdw)
}

func startSGLangMetricsSampler(ctx context.Context, model string, override []string, tracker *activeDatasetTracker, rdw *requestDataWriter) *metricsSampler {
	return startMetricsSampler(ctx, sglangEngine, model, override, tracker, rdw)
}

// stop terminates the sampler and waits for the goroutine to exit, so no
// write can race the rdw close that follows.
func (s *metricsSampler) stop() {
	s.cancel()
	<-s.done
}

func (s *metricsSampler) run(ctx context.Context) {
	defer close(s.done)
	// Immediate first sample establishes the cumulative baseline for delta
	// computation in the report.
	if !s.sampleOnce(ctx) {
		return
	}
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !s.sampleOnce(ctx) {
				return
			}
		}
	}
}

// sampleOnce fetches every endpoint, folds each one's DELTA into the running
// totals, and writes one sample record.
//
// Per endpoint, and by delta, because the alternative breaks on any fleet
// that moves. Summing raw counters means a pod restart drops the fleet sum —
// read downstream as a counter reset — and a pod leaving subtracts its
// entire history. Requiring every endpoint to answer before recording
// anything means one slow node costs the whole fleet's interval, which on a
// loaded fleet is most of them: /metrics is served by the same event loop as
// inference, so the backends most worth measuring are the ones most likely to
// miss the timeout.
//
// So: an endpoint that fails contributes nothing this round and keeps its
// baseline, an endpoint whose counter went backwards restarted and
// contributes from zero, and an endpoint that never comes back stops
// contributing while what it already did stays counted. The sample is
// written either way, carrying how many endpoints answered so a flat
// interval is distinguishable from an unobserved one.
//
// It reports whether the sampler should keep polling; false means the
// endpoint set has been established as one that will never answer.
func (s *metricsSampler) sampleOnce(ctx context.Context) (keepPolling bool) {
	ok := 0
	var lastErr error
	var lastBadURL string
	for _, u := range s.urls {
		vals, err := s.fetchOne(ctx, u)
		if err != nil {
			// A cancelled run is the benchmark ending, not the endpoint failing.
			if ctx.Err() != nil {
				return false
			}
			lastErr, lastBadURL = err, u
			continue
		}
		ok++
		s.fold(u, vals)
	}

	if ok == 0 {
		// Nothing answered. Still record the interval — with EndpointsOK at 0
		// it reads as unobserved rather than idle, which is the true
		// statement and the one a flat bar cannot make on its own.
		s.write(0)
		return s.noteFailure(lastBadURL, lastErr)
	}
	if ok < len(s.urls) {
		if s.consecFails == 0 {
			s.log("%d of %d endpoints answered (%v: %v) — their deltas are missing from this "+
				"interval; totals stay correct because they are sums of deltas",
				ok, len(s.urls), lastBadURL, lastErr)
		}
		s.consecFails++
	} else {
		s.noteSuccess()
	}
	s.write(ok)
	return true
}

// fold adds one endpoint's delta since its last reading into the totals.
//
// Distinguishes "this endpoint's very first successful scrape ever" (nothing
// in s.prev[url] at all) from "this endpoint has scraped before, but this
// particular series/key is new". Only the former is a true baseline that
// should contribute 0 — these pods carry counters from whatever ran before
// the benchmark started, and a run wants only what happened DURING it. The
// latter is a series born MID-RUN (sglang's host/storage cache_source labels
// are only created on their first nonzero hit — see sglang_metrics.go's
// package comment): treating its missing prior value as a baseline would
// discard its entire accumulated value even though the already-scraped
// prompt_tokens_total counted those tokens — they would silently land in
// compute instead. So a new key on an already-seen endpoint is treated as
// last=0, crediting the full cumulative value as this interval's delta,
// because the series was born during the run and its whole value happened
// during the run.
func (s *metricsSampler) fold(url string, vals map[string]float64) {
	if s.prev == nil {
		s.prev = map[string]map[string]float64{}
	}
	if s.totals == nil {
		s.totals = map[string]float64{}
	}
	prev, endpointSeenBefore := s.prev[url]
	if !endpointSeenBefore {
		prev = map[string]float64{}
		s.prev[url] = prev
	}
	for key, cur := range vals {
		last, had := prev[key]
		switch {
		case !had && !endpointSeenBefore:
			// This endpoint's very first successful scrape ever: every series
			// here is a pure baseline (whatever accumulated before the run
			// started), so it contributes nothing this interval.
		case !had:
			// Endpoint has scraped before, but this series is new — born
			// mid-run. Treat the missing prior value as 0 so the whole
			// cumulative value counts, instead of discarding it as a false
			// "baseline".
			s.totals[key] += cur
		case cur < last:
			// Restarted. Treating this as a decrease is the bug the scheme
			// exists to avoid, so the post-restart value counts from zero.
			s.totals[key] += cur
			s.resets++
		default:
			s.totals[key] += cur - last
		}
		prev[key] = cur
	}
}

// write persists the accumulated totals (projected onto the canonical band
// set via engine.Derive) with this round's coverage.
func (s *metricsSampler) write(endpointsOK int) {
	if s.rdw == nil || s.tracker == nil {
		return
	}
	adt, active := s.tracker.Sum()
	derived := s.engine.Derive(s.totals)
	bands := make(map[string]int64, len(derived))
	for k, v := range derived {
		bands[k] = int64(v)
	}
	rec := cacheMetricsSample{
		RecordType:          recordTypeCacheMetricsSample,
		TS:                  s.now(),
		Model:               s.model,
		Engine:              s.engine.Name,
		Bands:               bands,
		Resets:              s.resets,
		ActiveDatasetTokens: adt,
		ActiveSeries:        active,
		EndpointsOK:         endpointsOK,
		EndpointsTotal:      len(s.urls),
	}
	// Write errors are swallowed: sampling must never affect the benchmark.
	_ = s.rdw.writeAny(rec)
}

// noteSuccess records a landed sample. The first one is announced, and for a
// speculative engine it retires the speculative budget permanently: the
// endpoint has proven it serves the counter family, so any later failure is a
// blip (server restart, transient timeout) to ride out, not evidence of a bad
// guess.
func (s *metricsSampler) noteSuccess() {
	s.consecFails = 0
	if s.everSucceeded {
		return
	}
	s.everSucceeded = true
	s.log("%s serves %s — sampling every %s", strings.Join(s.urls, ", "), s.engine.FamilyHint, s.interval)
}

// noteFailure records a skipped sample and decides whether to keep going.
//
// An explicit spec (or any endpoint that has already answered once) keeps
// polling indefinitely: the operator said what this is, and a server still
// loading weights or restarted mid-run must not cost the rest of the run's
// samples. A speculative sampler instead spends metricsSpeculativeBudget
// attempts and then stops. For a non-speculative engine (sglang), explicit is
// always true (it only ever starts on an exact type= match), so this
// naturally reduces to "always log once, keep polling forever" with no
// special-casing.
func (s *metricsSampler) noteFailure(url string, err error) (keepPolling bool) {
	s.consecFails++
	if s.explicit || s.everSucceeded {
		if s.consecFails == 1 {
			s.log("%s unavailable (%v) — skipping samples, still polling every %s", url, err, s.interval)
		}
		return true
	}
	if s.consecFails < metricsSpeculativeBudget {
		return true
	}
	s.log("%s did not serve %s in %d attempts (%v) — sampling off; pass %s to keep polling",
		url, s.engine.FamilyHint, s.consecFails, err, s.engine.FlagHint)
	return false
}

func (s *metricsSampler) log(format string, args ...any) {
	if s.logf != nil {
		s.logf(format, args...)
	}
}

func (s *metricsSampler) fetchOne(ctx context.Context, url string) (map[string]float64, error) {
	reqCtx, cancel := context.WithTimeout(ctx, metricsFetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics fetch: status %d", resp.StatusCode)
	}
	vals, err := s.engine.Parse(resp.Body)
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		return nil, fmt.Errorf("metrics fetch: family %s not found", s.engine.FamilyHint)
	}
	return vals, nil
}

// ── On-disk record type ─────────────────────────────────────────────────────

// cacheMetricsSample is one periodic sample persisted into the request-data
// JSONL alongside requestDataRecord rows. record_type distinguishes it from
// request rows (which carry no record_type field). Replaces the former
// per-engine vllmMetricsSample/sglangMetricsSample — Bands holds whichever
// canonical bands this sample's engine reports (see cacheEngine.Bands),
// keyed by canonicalBandOrder's names.
type cacheMetricsSample struct {
	RecordType string    `json:"record_type"`
	TS         time.Time `json:"ts"`
	Model      string    `json:"model"`
	// Engine is "vllm" or "sglang" — see cacheEngine.Name.
	Engine string `json:"engine"`
	// Bands maps a canonical band name (see canonicalBandOrder) to its
	// cumulative value for this engine.
	Bands map[string]int64 `json:"bands"`

	// Resets seen so far, cumulative. A pod restart is normal and handled; the
	// same address resetting repeatedly means each scrape reads a DIFFERENT
	// process, which is what a Service or load balancer in place of a pod
	// looks like, and no delta scheme can work through that.
	Resets int `json:"resets"`

	ActiveDatasetTokens int64 `json:"active_dataset_tokens"`
	ActiveSeries        int   `json:"active_series"`

	// EndpointsOK of EndpointsTotal answered this round. Without them a flat
	// interval and an unobserved one look identical, and the natural reading
	// of a flat one — "the fleet did nothing" — is the wrong one.
	EndpointsOK    int `json:"endpoints_ok"`
	EndpointsTotal int `json:"endpoints_total"`
}

// ── Legacy on-disk shapes (read-time translation only) ──────────────────────
//
// Real cluster runs already produced files with the OLD per-engine record
// shapes (record_type "vllm_metrics_sample" / "sglang_metrics_sample").
// These structs exist ONLY to decode those old rows in
// readJSONLFileWithParams; nothing new is ever written in this shape.

type legacyVLLMSourceCounters struct {
	Compute       int64 `json:"compute"`
	LocalCache    int64 `json:"local_cache"`
	ExternalCache int64 `json:"external_cache"`
}

type legacyVLLMMetricsSample struct {
	TS                  time.Time                `json:"ts"`
	Model               string                   `json:"model"`
	Sources             legacyVLLMSourceCounters `json:"sources"`
	EndpointsOK         int                      `json:"endpoints_ok"`
	EndpointsTotal      int                      `json:"endpoints_total"`
	Resets              int                      `json:"resets"`
	ActiveDatasetTokens int64                    `json:"active_dataset_tokens"`
	ActiveSeries        int                      `json:"active_series"`
}

// toUnified maps a legacy vLLM sample onto the unified shape: local_compute
// -> compute, local_cache_hit -> host, external_kv_transfer -> external (see
// vllmEngine.Derive for why these particular bands).
func (l legacyVLLMMetricsSample) toUnified() cacheMetricsSample {
	return cacheMetricsSample{
		RecordType: recordTypeCacheMetricsSample,
		TS:         l.TS,
		Model:      l.Model,
		Engine:     vllmEngine.Name,
		Bands: map[string]int64{
			"compute":  l.Sources.Compute,
			"host":     l.Sources.LocalCache,
			"external": l.Sources.ExternalCache,
		},
		Resets:              l.Resets,
		ActiveDatasetTokens: l.ActiveDatasetTokens,
		ActiveSeries:        l.ActiveSeries,
		EndpointsOK:         l.EndpointsOK,
		EndpointsTotal:      l.EndpointsTotal,
	}
}

type legacySGLangSourceCounters struct {
	Compute int64 `json:"compute"`
	Device  int64 `json:"device"`
	Host    int64 `json:"host"`
	Storage int64 `json:"storage"`
}

type legacySGLangMetricsSample struct {
	TS                  time.Time                  `json:"ts"`
	Model               string                     `json:"model"`
	Sources             legacySGLangSourceCounters `json:"sources"`
	EndpointsOK         int                        `json:"endpoints_ok"`
	EndpointsTotal      int                        `json:"endpoints_total"`
	Resets              int                        `json:"resets"`
	ActiveDatasetTokens int64                      `json:"active_dataset_tokens"`
	ActiveSeries        int                        `json:"active_series"`
}

// toUnified maps a legacy sglang sample onto the unified shape: compute,
// device -> gpu, host -> host, storage -> storage.
func (l legacySGLangMetricsSample) toUnified() cacheMetricsSample {
	return cacheMetricsSample{
		RecordType: recordTypeCacheMetricsSample,
		TS:         l.TS,
		Model:      l.Model,
		Engine:     sglangEngine.Name,
		Bands: map[string]int64{
			"compute": l.Sources.Compute,
			"gpu":     l.Sources.Device,
			"host":    l.Sources.Host,
			"storage": l.Sources.Storage,
		},
		Resets:              l.Resets,
		ActiveDatasetTokens: l.ActiveDatasetTokens,
		ActiveSeries:        l.ActiveSeries,
		EndpointsOK:         l.EndpointsOK,
		EndpointsTotal:      l.EndpointsTotal,
	}
}

// ── Report-side transforms ──────────────────────────────────────────────────

// vizBand is one named token-source band's delta within a vizSampleSegment.
// n/v stay terse because this is embedded, per-segment, in a large JSON blob.
type vizBand struct {
	Name   string  `json:"n"`
	Tokens float64 `json:"v"`
}

// vizSampleSegment is one inter-sample interval with per-band token DELTAS
// (cumulative counter diffs, clamped at 0 so counter resets don't render as
// negative), embedded into the visualization data.
type vizSampleSegment struct {
	T0 float64 `json:"t0"` // interval start, unix ms
	T1 float64 `json:"t1"` // interval end, unix ms
	// EndpointsOK/Total for the sample that CLOSED this interval. A segment
	// whose deltas are zero means the fleet was idle only if the endpoints
	// answered; with none answering it means nobody looked.
	EndpointsOK    int       `json:"endpoints_ok"`
	EndpointsTotal int       `json:"endpoints_total"`
	Bands          []vizBand `json:"bands"`
}

// vizAdtPoint is one active-dataset observation for the overlay line.
type vizAdtPoint struct {
	T      float64 `json:"t"` // unix ms
	Tokens float64 `json:"v"`
	Series int     `json:"s"`
}

// sampleBandNames returns the canonical band names actually carried by any of
// the given samples' Bands maps, in canonicalBandOrder's fixed order. This is
// how buildSampleViz and generateVisualization learn a series' band set
// without needing to know which engine produced it (an engine's Derive always
// sets all of its own bands, even when a value is 0, so this reliably
// recovers e.g. ["compute","host","external"] for a vLLM series and
// ["compute","gpu","host","storage"] for an sglang one).
func sampleBandNames(samples []cacheMetricsSample) []string {
	present := map[string]bool{}
	for _, s := range samples {
		for k := range s.Bands {
			present[k] = true
		}
	}
	var out []string
	for _, b := range canonicalBandOrder {
		if present[b] {
			out = append(out, b)
		}
	}
	return out
}

// buildSampleViz converts raw cumulative cache-metrics samples into
// per-interval band deltas (mix) and active-dataset points (adt) for the
// embedded report JS. Replaces the former buildSampleVizVLLM/
// buildSampleVizSGLang: driven entirely by each sample's Bands map and
// canonicalBandOrder, so it needs no engine-specific knowledge.
func buildSampleViz(samples []cacheMetricsSample) (mix []vizSampleSegment, adt []vizAdtPoint) {
	if len(samples) == 0 {
		return nil, nil
	}
	sorted := make([]cacheMetricsSample, len(samples))
	copy(sorted, samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].TS.Before(sorted[j].TS) })

	bandNames := sampleBandNames(sorted)

	// Totals are sums of per-endpoint deltas and so never decrease. The clamp
	// stays as a floor against a malformed or hand-edited JSONL rather than
	// as the thing that makes restarts survivable — that is the
	// accumulator's job, upstream, where the endpoint identity is still
	// known.
	clamp := func(cur, prev int64) float64 {
		d := cur - prev
		if d < 0 {
			return 0
		}
		return float64(d)
	}
	for i, smp := range sorted {
		adt = append(adt, vizAdtPoint{
			T:      float64(smp.TS.UnixMilli()),
			Tokens: float64(smp.ActiveDatasetTokens),
			Series: smp.ActiveSeries,
		})
		if i == 0 {
			continue
		}
		prev := sorted[i-1]
		bands := make([]vizBand, 0, len(bandNames))
		for _, name := range bandNames {
			bands = append(bands, vizBand{Name: name, Tokens: clamp(smp.Bands[name], prev.Bands[name])})
		}
		mix = append(mix, vizSampleSegment{
			T0:             float64(prev.TS.UnixMilli()),
			T1:             float64(smp.TS.UnixMilli()),
			EndpointsOK:    smp.EndpointsOK,
			EndpointsTotal: smp.EndpointsTotal,
			Bands:          bands,
		})
	}
	return mix, adt
}

// mixBandStyleJS is the shared cache-mix color/label/alpha JS block embedded
// verbatim in both the interactive report (visualize.go's vizTemplate) and
// the public export template (visualize_public.go's publicReportTemplateHTML)
// — kept as ONE Go string so the two can never drift out of lockstep, which
// is exactly how external_kv_transfer and storage ended up sharing one color
// (#7C03EC) before this unification: two independently-maintained copies of
// the same table.
//
// external gets its OWN color, distinct from storage's #7C03EC: vLLM's
// "external" band (external_kv_transfer) means "any KV connector" per the
// vLLM fork's own source comments, not specifically WEKA L3, so a report
// comparing a vLLM arm against an sglang arm must not render the two as the
// same color.
const mixBandStyleJS = `const MIX_COMPUTE_COLOR = "#a86853";
const MIX_GPU_COLOR = "#756a99";
const MIX_HOST_COLOR = "#4a9d8f";
const MIX_STORAGE_COLOR = "#7C03EC";
const MIX_EXTERNAL_COLOR = "#E8A33D";
const ADT_LINE_COLOR = "#F2F2EB";
const MIX_BAND_H = 64;
// Band fills sit on a solid-black backdrop; slightly translucent so the
// (muted) band mass sits visually behind the latency lines — the plot reads
// first, the band second.
const MIX_FILL_ALPHA = 0.78;
// MIX_BAND_STYLE maps a canonical band name (see canonicalBandOrder in
// cache_metrics.go) to its report color, legend/tooltip label, and fill
// alpha, so the drawing and tooltip code can iterate any series' seg.bands
// without knowing ahead of time whether it's a vLLM or sglang series. storage
// and external are each that series' single vivid class, so both render
// near-opaque (alpha 0.95) rather than the muted alpha the other bands use.
const MIX_BAND_STYLE = {
  compute:  { color: MIX_COMPUTE_COLOR,  label: "compute",            alpha: MIX_FILL_ALPHA },
  gpu:      { color: MIX_GPU_COLOR,      label: "device",             alpha: MIX_FILL_ALPHA },
  host:     { color: MIX_HOST_COLOR,     label: "host",               alpha: MIX_FILL_ALPHA },
  storage:  { color: MIX_STORAGE_COLOR,  label: "storage",            alpha: 0.95 },
  external: { color: MIX_EXTERNAL_COLOR, label: "external connector", alpha: 0.95 },
};
`
