package benchmark

// vLLM metrics collector: during `benchmark auto` runs against an
// OpenAI-compatible (chat/completions) endpoint, a per-model sampler goroutine
// polls the server's Prometheus /metrics endpoint every
// vllmMetricsSampleInterval and persists per-source cumulative prompt-token
// counters into the same --save-request-data JSONL stream as the request rows,
// as records of their own type ("vllm_metrics_sample"). Sampling is strictly
// best-effort and can never affect the benchmark itself.
//
// The persisted totals are a CONTINUOUS SUM OF PER-ENDPOINT DELTAS, not a sum
// of the raw counters. Summing raw counters breaks the moment the fleet moves:
// a pod restarting resets its counter to zero and the fleet sum drops, which
// downstream reads as a counter reset and discards the interval; a pod leaving
// subtracts everything it ever did. Deltas survive both — a restart contributes
// its post-restart value from zero, and an endpoint that goes away simply stops
// adding while the work it already did stays counted.
//
// One endpoint failing therefore costs only that endpoint's contribution for
// that interval, not the whole sample. What IS lost is recorded: every sample
// carries how many endpoints answered, so a flat interval can be told apart
// from an unobserved one.
//
// Eligibility is deliberately looser than type=openai_vllm. Any chat/completions
// endpoint — type=openai, the default type, or an autodiscovered bare host:port
// — may well be vLLM behind the scenes, and the only way to find out is to ask.
// The cost of guessing wrong is bounded by vllmMetricsSpeculativeBudget; the
// cost of not guessing is a whole benchmark run with no cache-source breakdown.
// type=openai_vllm still differs in one way: the operator asserted the server
// is vLLM, so a sampler for such a spec keeps retrying forever rather than
// giving up (see noteFailure).
//
// The counter family is the fork's vllm:prompt_tokens_by_source
// (vllm/v1/metrics/loggers.py) with label source in {local_compute,
// local_cache_hit, external_kv_transfer} (vllm/v1/metrics/stats.py
// PromptTokenStats.ALL_SOURCES); the text exposition renders it as
// vllm:prompt_tokens_by_source_total. Values are summed across all other
// labels (model_name, engine index).

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/llm"
)

const (
	recordTypeVLLMMetricsSample = "vllm_metrics_sample"

	vllmPromptTokensBySourceFamily = "vllm:prompt_tokens_by_source"

	vllmMetricsSampleInterval = 60 * time.Second
	vllmMetricsFetchTimeout   = 5 * time.Second

	// vllmMetricsSpeculativeBudget is how many consecutive failed samples a
	// speculatively-started sampler (one whose spec did not say
	// type=openai_vllm) tolerates before concluding the endpoint simply isn't
	// vLLM and shutting itself down. Bounding it matters because the endpoint
	// may be a public API that never had a /metrics: three 404s spread over two
	// minutes is a fair price for the guess, an hourly drip is not. One
	// successful sample retires the budget for good — see noteSuccess.
	vllmMetricsSpeculativeBudget = 3
)

// vllmSourceCounters holds cumulative prompt-token counter values by source
// at one sample instant.
type vllmSourceCounters struct {
	Compute       int64 `json:"compute"`        // source="local_compute"
	LocalCache    int64 `json:"local_cache"`    // source="local_cache_hit"
	ExternalCache int64 `json:"external_cache"` // source="external_kv_transfer"
}

// vllmMetricsSample is one periodic sample persisted into the request-data
// JSONL alongside requestDataRecord rows. record_type distinguishes it from
// request rows (which carry no record_type field).
type vllmMetricsSample struct {
	RecordType string             `json:"record_type"`
	TS         time.Time          `json:"ts"`
	Model      string             `json:"model"`
	Sources    vllmSourceCounters `json:"sources"`

	// EndpointsOK of EndpointsTotal answered this round. Without them a flat
	// interval and an unobserved one look identical, and the natural reading of
	// a flat one — "the fleet did nothing" — is the wrong one.
	EndpointsOK    int `json:"endpoints_ok"`
	EndpointsTotal int `json:"endpoints_total"`

	// Resets seen so far, cumulative. A pod restart is normal and handled; the
	// same address resetting repeatedly means each scrape reads a DIFFERENT
	// process, which is what a Service or load balancer in place of a pod looks
	// like, and no delta scheme can work through that.
	Resets int `json:"resets"`

	ActiveDatasetTokens int64 `json:"active_dataset_tokens"`
	ActiveSeries        int   `json:"active_series"`
}

var promSourceLabelRe = regexp.MustCompile(`\bsource="([^"]*)"`)

// parsePromptTokensBySource scans Prometheus text exposition for the
// vllm:prompt_tokens_by_source family and returns cumulative values keyed by
// the source label, summed across all other label combinations (engine index,
// model_name).
//
// The series to match is vllm:prompt_tokens_by_source_total — prometheus_client
// appends the counter suffix, so that is the only form a real vLLM /metrics
// emits. The bare name is accepted too, purely as tolerance for an exposition
// path that doesn't append it; the two never coexist in one scrape. (The bare
// name does appear on the # HELP/# TYPE lines, but those are skipped as
// comments before any name matching.)
func parsePromptTokensBySource(r io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		rest, ok := strings.CutPrefix(line, vllmPromptTokensBySourceFamily)
		if !ok {
			continue
		}
		rest, _ = strings.CutPrefix(rest, "_total")
		// Reject sibling series like _created and other families sharing the
		// prefix: the next byte must open the label set.
		if len(rest) == 0 || rest[0] != '{' {
			continue
		}
		closeIdx := strings.LastIndex(rest, "}")
		if closeIdx < 0 {
			continue
		}
		fields := strings.Fields(rest[closeIdx+1:])
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			continue
		}
		m := promSourceLabelRe.FindStringSubmatch(rest[1:closeIdx])
		if m == nil {
			continue
		}
		out[m[1]] += v
	}
	return out, sc.Err()
}

// activeDatasetTracker keeps, per live series slot, the total prompt tokens
// (server-reported cached + uncached) of that series' most recent response.
// The per-sample sum over slots approximates the "active dataset": the token
// volume a fully-warm cache would need to hold for the currently-running
// series. A recycled slot (--exhaust-sessions, replay pulling the next
// conversation) is Reset so retired sessions stop counting.
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

// vllmMetricsEligibility says whether a model spec is worth polling for vLLM
// metrics, and how sure we are.
type vllmMetricsEligibility struct {
	// urls are the Prometheus /metrics URLs, one per endpoint. Empty means the
	// spec is not a chat/completions endpoint and no sampler should start.
	urls []string
	// explicit is true when the spec said type=openai_vllm outright. Such a
	// sampler never gives up on failures; a speculative one does.
	explicit bool
}

// vllmMetricsEndpoints derives the Prometheus /metrics URLs for a dynamic
// model spec pointing at an OpenAI-compatible endpoint (the /v1 API suffix is
// stripped to reach the server root, where vLLM mounts /metrics).
//
// Every chat/completions type qualifies, not just openai_vllm: "openai" is
// also what a bare host:port resolves to once autodiscovery has confirmed a
// /v1/models listing, and that shape is exactly what a local vLLM serves.
// type=anthropic and type=gemini are excluded — no vLLM deployment answers on
// those wire formats, so probing them would be pure noise.
func vllmMetricsEndpoints(model string) vllmMetricsEligibility {
	if !llm.IsDynamicModel(model) {
		return vllmMetricsEligibility{}
	}
	dyn, err := llm.ParseDynamicModel(model)
	if err != nil {
		return vllmMetricsEligibility{}
	}
	// "" cannot come out of ParseDynamicModel today (it defaults to "openai"),
	// but treat it as the default anyway so a future zero-valued config here
	// fails open rather than silently dropping sampling.
	switch dyn.Type {
	case "openai", "openai_vllm", "":
	default:
		return vllmMetricsEligibility{}
	}
	out := vllmMetricsEligibility{explicit: dyn.Type == "openai_vllm"}
	for _, u := range dyn.BaseURLs {
		u = strings.TrimRight(u, "/")
		u = strings.TrimSuffix(u, "/v1")
		out.urls = append(out.urls, u+"/metrics")
	}
	return out
}

// vllmMetricsSampler polls one model's endpoints and writes samples to rdw.
type vllmMetricsSampler struct {
	model    string
	urls     []string
	explicit bool
	tracker  *activeDatasetTracker
	rdw      *requestDataWriter
	interval time.Duration
	client   *http.Client
	skipped  atomic.Int64
	now      func() time.Time
	logf     func(format string, args ...any)

	// Outcome bookkeeping. Touched only by the run goroutine, so no locking.
	consecFails   int
	everSucceeded bool

	// The delta accumulator. prev is the last raw counter seen per endpoint per
	// source; totals is the running sum of deltas, which is what gets persisted.
	prev   map[string]map[string]float64
	totals map[string]float64
	resets int

	cancel context.CancelFunc
	done   chan struct{}
}

// startVLLMMetricsSampler launches the sampler goroutine for cfg.Model when
// the spec is an OpenAI-compatible endpoint and request data is being saved.
// Returns nil when sampling doesn't apply. Callers must invoke stop() before
// closing rdw.
//
// Starting is not a promise that samples will land: a speculatively-started
// sampler shuts itself down once it establishes the endpoint has no vLLM
// metrics (see noteFailure). stop() stays safe to call either way.
func startVLLMMetricsSampler(ctx context.Context, model string, tracker *activeDatasetTracker, rdw *requestDataWriter) *vllmMetricsSampler {
	if rdw == nil || tracker == nil {
		return nil
	}
	elig := vllmMetricsEndpoints(model)
	if len(elig.urls) == 0 {
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	s := &vllmMetricsSampler{
		model:    model,
		urls:     elig.urls,
		explicit: elig.explicit,
		tracker:  tracker,
		rdw:      rdw,
		interval: vllmMetricsSampleInterval,
		client:   &http.Client{},
		now:      time.Now,
		prev:     map[string]map[string]float64{},
		totals:   map[string]float64{},
		logf:     func(f string, a ...any) { fmt.Fprintf(os.Stderr, "[vllm-metrics] "+f+"\n", a...) },
		cancel:   cancel,
		done:     make(chan struct{}),
	}
	go s.run(runCtx)
	return s
}

// stop terminates the sampler and waits for the goroutine to exit, so no
// write can race the rdw close that follows.
func (s *vllmMetricsSampler) stop() {
	s.cancel()
	<-s.done
}

func (s *vllmMetricsSampler) run(ctx context.Context) {
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
// Per endpoint, and by delta, because the alternative breaks on any fleet that
// moves. Summing raw counters means a pod restart drops the fleet sum — read
// downstream as a counter reset — and a pod leaving subtracts its entire
// history. Requiring every endpoint to answer before recording anything means
// one slow node costs the whole fleet's interval, which on a loaded fleet is
// most of them: /metrics is served by the same event loop as inference, so the
// backends most worth measuring are the ones most likely to miss the timeout.
//
// So: an endpoint that fails contributes nothing this round and keeps its
// baseline, an endpoint whose counter went backwards restarted and contributes
// from zero, and an endpoint that never comes back stops contributing while
// what it already did stays counted. The sample is written either way, carrying
// how many endpoints answered so a flat interval is distinguishable from an
// unobserved one.
//
// It reports whether the sampler should keep polling; false means the endpoint
// set has been established as one that will never answer.
func (s *vllmMetricsSampler) sampleOnce(ctx context.Context) (keepPolling bool) {
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
		s.skipped.Add(1)
		// Nothing answered. Still record the interval — with EndpointsOK at 0 it
		// reads as unobserved rather than idle, which is the true statement and
		// the one a flat bar cannot make on its own.
		s.write(0)
		return s.noteFailure(lastBadURL, lastErr)
	}
	if ok < len(s.urls) {
		s.skipped.Add(1)
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
func (s *vllmMetricsSampler) fold(url string, vals map[string]float64) {
	prev := s.prev[url]
	if prev == nil {
		prev = map[string]float64{}
		s.prev[url] = prev
	}
	for key, cur := range vals {
		last, had := prev[key]
		switch {
		case !had:
			// First sighting establishes a BASELINE and contributes nothing.
			//
			// This is the one place the benchmark deliberately differs from the
			// router's aggregator, which adds the whole first value so its
			// fleet totals include work done before it started. A run wants only
			// what happened DURING it, and these pods carry counters from
			// whatever ran before. The cost is at most one interval of a pod
			// that joins mid-run, which starts near zero anyway.
		case cur < last:
			// Restarted. Treating this as a decrease is the bug the scheme exists
			// to avoid, so the post-restart value counts from zero.
			s.totals[key] += cur
			s.resets++
		default:
			s.totals[key] += cur - last
		}
		prev[key] = cur
	}
}

// write persists the accumulated totals with this round's coverage.
func (s *vllmMetricsSampler) write(endpointsOK int) {
	adt, active := s.tracker.Sum()
	rec := vllmMetricsSample{
		RecordType: recordTypeVLLMMetricsSample,
		TS:         s.now(),
		Model:      s.model,
		Sources: vllmSourceCounters{
			Compute:       int64(s.totals["local_compute"]),
			LocalCache:    int64(s.totals["local_cache_hit"]),
			ExternalCache: int64(s.totals["external_kv_transfer"]),
		},
		EndpointsOK:         endpointsOK,
		EndpointsTotal:      len(s.urls),
		Resets:              s.resets,
		ActiveDatasetTokens: adt,
		ActiveSeries:        active,
	}
	// Write errors are swallowed: sampling must never affect the benchmark.
	_ = s.rdw.writeAny(rec)
}

// noteSuccess records a landed sample. The first one is announced, and it
// retires the speculative budget permanently: the endpoint has proven it
// serves the counter family, so any later failure is a blip (server restart,
// transient timeout) to ride out, not evidence of a bad guess.
func (s *vllmMetricsSampler) noteSuccess() {
	s.consecFails = 0
	if s.everSucceeded {
		return
	}
	s.everSucceeded = true
	s.log("%s serves %s — sampling every %s", strings.Join(s.urls, ", "), vllmPromptTokensBySourceFamily, s.interval)
}

// noteFailure records a skipped sample and decides whether to keep going.
//
// An explicit type=openai_vllm spec (or any endpoint that has already
// answered once) keeps polling indefinitely: the operator said this is vLLM,
// and a server still loading weights or restarted mid-run must not cost the
// rest of the run's samples. A speculative sampler instead spends
// vllmMetricsSpeculativeBudget attempts and then stops.
func (s *vllmMetricsSampler) noteFailure(url string, err error) (keepPolling bool) {
	s.consecFails++
	if s.explicit || s.everSucceeded {
		if s.consecFails == 1 {
			s.log("%s unavailable (%v) — skipping samples, still polling every %s", url, err, s.interval)
		}
		return true
	}
	if s.consecFails < vllmMetricsSpeculativeBudget {
		return true
	}
	s.log("%s did not serve %s in %d attempts (%v) — sampling off; pass type=openai_vllm to keep polling",
		url, vllmPromptTokensBySourceFamily, s.consecFails, err)
	return false
}

func (s *vllmMetricsSampler) log(format string, args ...any) {
	if s.logf != nil {
		s.logf(format, args...)
	}
}

func (s *vllmMetricsSampler) fetchOne(ctx context.Context, url string) (map[string]float64, error) {
	reqCtx, cancel := context.WithTimeout(ctx, vllmMetricsFetchTimeout)
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
	vals, err := parsePromptTokensBySource(resp.Body)
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		return nil, fmt.Errorf("metrics fetch: family %s not found", vllmPromptTokensBySourceFamily)
	}
	return vals, nil
}

// ── Report-side transforms ──────────────────────────────────────────────────

// vizSampleSegment is one inter-sample interval with per-source token DELTAS
// (cumulative counter diffs, clamped at 0 so counter resets don't render as
// negative), embedded into the visualization data.
type vizSampleSegment struct {
	T0 float64 `json:"t0"` // interval start, unix ms
	T1 float64 `json:"t1"` // interval end, unix ms
	// EndpointsOK/Total for the sample that CLOSED this interval. A segment
	// whose deltas are zero means the fleet was idle only if the endpoints
	// answered; with none answering it means nobody looked.
	EndpointsOK    int     `json:"endpoints_ok"`
	EndpointsTotal int     `json:"endpoints_total"`
	Compute        float64 `json:"c"`
	LocalCache     float64 `json:"lc"`
	ExternalCache  float64 `json:"ec"`
}

// vizAdtPoint is one active-dataset observation for the overlay line.
type vizAdtPoint struct {
	T      float64 `json:"t"` // unix ms
	Tokens float64 `json:"v"`
	Series int     `json:"s"`
}

// buildSampleViz converts raw cumulative samples into per-interval source
// deltas (mix) and active-dataset points (adt) for the embedded report JS.
func buildSampleViz(samples []vllmMetricsSample) (mix []vizSampleSegment, adt []vizAdtPoint) {
	if len(samples) == 0 {
		return nil, nil
	}
	sorted := make([]vllmMetricsSample, len(samples))
	copy(sorted, samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].TS.Before(sorted[j].TS) })

	// Totals are sums of per-endpoint deltas and so never decrease. The clamp
	// stays as a floor against a malformed or hand-edited JSONL rather than as
	// the thing that makes restarts survivable — that is the accumulator's job,
	// upstream, where the endpoint identity is still known.
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
		mix = append(mix, vizSampleSegment{
			T0:             float64(prev.TS.UnixMilli()),
			T1:             float64(smp.TS.UnixMilli()),
			Compute:        clamp(smp.Sources.Compute, prev.Sources.Compute),
			LocalCache:     clamp(smp.Sources.LocalCache, prev.Sources.LocalCache),
			ExternalCache:  clamp(smp.Sources.ExternalCache, prev.Sources.ExternalCache),
			EndpointsOK:    smp.EndpointsOK,
			EndpointsTotal: smp.EndpointsTotal,
		})
	}
	return mix, adt
}
