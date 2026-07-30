package benchmark

// vLLM metrics collector: during `benchmark auto` runs against an
// OpenAI-compatible (chat/completions) endpoint, a per-model sampler goroutine
// polls the server's Prometheus /metrics endpoint every
// vllmMetricsSampleInterval and persists per-source cumulative prompt-token
// counters into the same --save-request-data JSONL stream as the request rows,
// as records of their own type ("vllm_metrics_sample"). Sampling is strictly
// best-effort: any fetch/parse failure skips that sample and can never affect
// the benchmark itself.
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
	RecordType          string             `json:"record_type"`
	TS                  time.Time          `json:"ts"`
	Model               string             `json:"model"`
	Sources             vllmSourceCounters `json:"sources"`
	ActiveDatasetTokens int64              `json:"active_dataset_tokens"`
	ActiveSeries        int                `json:"active_series"`
}

var promSourceLabelRe = regexp.MustCompile(`\bsource="([^"]*)"`)

// parsePromptTokensBySource scans Prometheus text exposition for the
// vllm:prompt_tokens_by_source family (with or without the counter _total
// suffix) and returns cumulative values keyed by the source label, summed
// across all other label combinations (engine index, model_name).
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

// sampleOnce fetches every endpoint and writes one sample record. Any
// endpoint failing (refused/timeout/non-200/parse failure/family absent)
// skips the whole sample so the cumulative sums stay consistent across the
// endpoint set — a partial sum would read as a counter reset downstream.
//
// It reports whether the sampler should keep polling; false means the
// endpoint has been established as one that will never answer.
func (s *vllmMetricsSampler) sampleOnce(ctx context.Context) (keepPolling bool) {
	sums := map[string]float64{}
	for _, u := range s.urls {
		vals, err := s.fetchOne(ctx, u)
		if err != nil {
			s.skipped.Add(1)
			// A cancelled run is the benchmark ending, not the endpoint
			// failing; don't spend the speculative budget on it.
			if ctx.Err() != nil {
				return false
			}
			return s.noteFailure(u, err)
		}
		for k, v := range vals {
			sums[k] += v
		}
	}
	s.noteSuccess()
	adt, active := s.tracker.Sum()
	rec := vllmMetricsSample{
		RecordType: recordTypeVLLMMetricsSample,
		TS:         s.now(),
		Model:      s.model,
		Sources: vllmSourceCounters{
			Compute:       int64(sums["local_compute"]),
			LocalCache:    int64(sums["local_cache_hit"]),
			ExternalCache: int64(sums["external_kv_transfer"]),
		},
		ActiveDatasetTokens: adt,
		ActiveSeries:        active,
	}
	// Write errors are swallowed: sampling must never affect the benchmark.
	_ = s.rdw.writeAny(rec)
	return true
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
	T0            float64 `json:"t0"` // interval start, unix ms
	T1            float64 `json:"t1"` // interval end, unix ms
	Compute       float64 `json:"c"`
	LocalCache    float64 `json:"lc"`
	ExternalCache float64 `json:"ec"`
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
			T0:            float64(prev.TS.UnixMilli()),
			T1:            float64(smp.TS.UnixMilli()),
			Compute:       clamp(smp.Sources.Compute, prev.Sources.Compute),
			LocalCache:    clamp(smp.Sources.LocalCache, prev.Sources.LocalCache),
			ExternalCache: clamp(smp.Sources.ExternalCache, prev.Sources.ExternalCache),
		})
	}
	return mix, adt
}
