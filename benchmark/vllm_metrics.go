package benchmark

// vLLM metrics collector: during `benchmark auto` runs against a
// type=openai_vllm endpoint, a per-model sampler goroutine polls the server's
// Prometheus /metrics endpoint every vllmMetricsSampleInterval and persists
// per-source cumulative prompt-token counters into the same
// --save-request-data JSONL stream as the request rows, as records of their
// own type ("vllm_metrics_sample"). Sampling is strictly best-effort: any
// fetch/parse failure skips that sample silently and can never affect the
// benchmark itself.
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

// vllmMetricsEndpoints derives the Prometheus /metrics URLs for an
// openai_vllm dynamic model spec (the /v1 API suffix is stripped to reach the
// server root). Returns nil when the spec is not a vLLM endpoint — the
// sampler simply doesn't start for other model types.
func vllmMetricsEndpoints(model string) []string {
	if !llm.IsDynamicModel(model) {
		return nil
	}
	dyn, err := llm.ParseDynamicModel(model)
	if err != nil || dyn.Type != "openai_vllm" {
		return nil
	}
	var urls []string
	for _, u := range dyn.BaseURLs {
		u = strings.TrimRight(u, "/")
		u = strings.TrimSuffix(u, "/v1")
		urls = append(urls, u+"/metrics")
	}
	return urls
}

// vllmMetricsSampler polls one model's endpoints and writes samples to rdw.
type vllmMetricsSampler struct {
	model    string
	urls     []string
	tracker  *activeDatasetTracker
	rdw      *requestDataWriter
	interval time.Duration
	client   *http.Client
	skipped  atomic.Int64
	now      func() time.Time

	cancel context.CancelFunc
	done   chan struct{}
}

// startVLLMMetricsSampler launches the sampler goroutine for cfg.Model when
// the spec is a type=openai_vllm endpoint and request data is being saved.
// Returns nil when sampling doesn't apply. Callers must invoke stop() before
// closing rdw.
func startVLLMMetricsSampler(ctx context.Context, model string, tracker *activeDatasetTracker, rdw *requestDataWriter) *vllmMetricsSampler {
	if rdw == nil || tracker == nil {
		return nil
	}
	urls := vllmMetricsEndpoints(model)
	if len(urls) == 0 {
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	s := &vllmMetricsSampler{
		model:    model,
		urls:     urls,
		tracker:  tracker,
		rdw:      rdw,
		interval: vllmMetricsSampleInterval,
		client:   &http.Client{},
		now:      time.Now,
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
	s.sampleOnce(ctx)
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.sampleOnce(ctx)
		}
	}
}

// sampleOnce fetches every endpoint and writes one sample record. Any
// endpoint failing (refused/timeout/non-200/parse failure/family absent)
// skips the whole sample so the cumulative sums stay consistent across the
// endpoint set; failures are silent by design.
func (s *vllmMetricsSampler) sampleOnce(ctx context.Context) {
	sums := map[string]float64{}
	for _, u := range s.urls {
		vals, err := s.fetchOne(ctx, u)
		if err != nil {
			s.skipped.Add(1)
			return
		}
		for k, v := range vals {
			sums[k] += v
		}
	}
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
