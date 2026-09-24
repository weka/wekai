package benchmark

// SGLang metrics collector: during `benchmark auto` runs against a
// type=openai_sglang endpoint, a per-model sampler goroutine polls the
// server's Prometheus /metrics endpoint every sglangMetricsSampleInterval and
// persists per-source cumulative prompt-token counters into the same
// --save-request-data JSONL stream as the request rows, as records of their
// own type ("sglang_metrics_sample"). Sampling is strictly best-effort and
// can never affect the benchmark itself.
//
// This reuses vllmMetricsSampler's delta-accumulation scheme (see
// vllm_metrics.go's package comment for the full rationale): SGLang's
// sglang:cached_tokens_total (labeled by cache_source in
// {device,host,storage,total}) and sglang:prompt_tokens_total are both plain
// monotonic Prometheus counters, so the persisted totals are a continuous sum
// of per-endpoint DELTAS, not a sum of the raw counters — a pod restart
// resets its counter to zero, and only a delta scheme survives that without
// misreading it as the fleet's work disappearing.
//
// The four reported bands are:
//   - device  = Δcached_tokens_total{cache_source="device"}  (GPU-resident
//     radix-cache hit)
//   - host    = Δcached_tokens_total{cache_source="host"}    (CPU/DRAM
//     HiCache L2 hit)
//   - storage = Δcached_tokens_total{cache_source="storage"} (WekaFS L3 hit)
//   - compute = Δprompt_tokens_total − Δcached_tokens_total{cache_source="total"}
//     (recomputed from scratch; not itself a cache_source label)
//
// Unlike vLLM eligibility (which speculatively samples any chat/completions
// endpoint on the theory that it might be vLLM behind the scenes — see
// vllmMetricsEndpoints), SGLang sampling only ever starts when the spec says
// type=openai_sglang outright: there is no bare-host/default shape in this
// codebase that plausibly resolves to SGLang, so guessing would only add
// failed-probe traffic against every non-SGLang endpoint. A sampler that
// starts therefore keeps polling for the life of the run — the operator
// asserted the server is SGLang, so a server still loading weights or
// briefly unreachable must not cost the rest of the run's samples.

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
	"time"

	"github.com/weka/wekai/llm"
)

const (
	recordTypeSGLangMetricsSample = "sglang_metrics_sample"

	sglangCachedTokensTotalFamily = "sglang:cached_tokens_total"
	sglangPromptTokensTotalFamily = "sglang:prompt_tokens_total"

	// sglangPromptTokensKey is the totals/prev map key for the (unlabeled)
	// prompt_tokens_total family, kept distinct from the cache_source label
	// values ("device"/"host"/"storage"/"total") it's differenced against.
	sglangPromptTokensKey = "prompt_tokens_total"

	sglangMetricsSampleInterval = 60 * time.Second
	sglangMetricsFetchTimeout   = 5 * time.Second
)

// sglangSourceCounters holds cumulative prompt-token counter values by
// source at one sample instant. Compute is derived (prompt_tokens_total minus
// the "total" cache_source label), not itself a scraped label.
type sglangSourceCounters struct {
	Compute int64 `json:"compute"`
	Device  int64 `json:"device"`  // cache_source="device"
	Host    int64 `json:"host"`    // cache_source="host"
	Storage int64 `json:"storage"` // cache_source="storage"
}

// sglangMetricsSample is one periodic sample persisted into the request-data
// JSONL alongside requestDataRecord rows. record_type distinguishes it from
// request rows (which carry no record_type field) and from vllmMetricsSample.
type sglangMetricsSample struct {
	RecordType string               `json:"record_type"`
	TS         time.Time            `json:"ts"`
	Model      string               `json:"model"`
	Sources    sglangSourceCounters `json:"sources"`

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

var sglangCacheSourceLabelRe = regexp.MustCompile(`\bcache_source="([^"]*)"`)

// parseSGLangCounters scans Prometheus text exposition for the
// sglang:cached_tokens_total family (labeled by cache_source, summed across
// all other label combinations — model_name, engine/rank) and the
// sglang:prompt_tokens_total family (summed across all its labels into one
// value), returning cumulative values keyed by "device", "host", "storage",
// "total", and sglangPromptTokensKey. Missing families simply produce no
// entries for their keys; malformed lines are skipped.
func parseSGLangCounters(r io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		if rest, ok := strings.CutPrefix(line, sglangCachedTokensTotalFamily); ok {
			// The next byte must open the label set — cached_tokens_total is
			// always labeled by cache_source, so a bare value (or a sibling
			// series like _created sharing the prefix) is rejected.
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
			m := sglangCacheSourceLabelRe.FindStringSubmatch(rest[1:closeIdx])
			if m == nil {
				continue
			}
			out[m[1]] += v
			continue
		}
		if rest, ok := strings.CutPrefix(line, sglangPromptTokensTotalFamily); ok {
			// Unlike cached_tokens_total, this family may or may not carry
			// labels (e.g. model_name) — accept either shape, and sum
			// everything into one key regardless of label values.
			if len(rest) == 0 || (rest[0] != '{' && rest[0] != ' ') {
				continue
			}
			valueField := rest
			if rest[0] == '{' {
				closeIdx := strings.LastIndex(rest, "}")
				if closeIdx < 0 {
					continue
				}
				valueField = rest[closeIdx+1:]
			}
			fields := strings.Fields(valueField)
			if len(fields) == 0 {
				continue
			}
			v, err := strconv.ParseFloat(fields[0], 64)
			if err != nil {
				continue
			}
			out[sglangPromptTokensKey] += v
			continue
		}
	}
	return out, sc.Err()
}

// sglangMetricsEndpoints derives the Prometheus /metrics URLs for a dynamic
// model spec pointing at an SGLang endpoint (the /v1 API suffix is stripped
// to reach the server root, where SGLang mounts /metrics — same convention
// as vLLM). Returns nil unless the spec says type=openai_sglang outright:
// see the package comment for why this is never speculative.
func sglangMetricsEndpoints(model string) []string {
	if !llm.IsDynamicModel(model) {
		return nil
	}
	dyn, err := llm.ParseDynamicModel(model)
	if err != nil || dyn.Type != "openai_sglang" {
		return nil
	}
	out := make([]string, 0, len(dyn.BaseURLs))
	for _, u := range dyn.BaseURLs {
		u = strings.TrimRight(u, "/")
		u = strings.TrimSuffix(u, "/v1")
		out = append(out, u+"/metrics")
	}
	return out
}

// sglangMetricsSampler polls one model's endpoints and writes samples to rdw.
type sglangMetricsSampler struct {
	model    string
	urls     []string
	tracker  *activeDatasetTracker
	rdw      *requestDataWriter
	interval time.Duration
	client   *http.Client
	now      func() time.Time
	logf     func(format string, args ...any)

	// Outcome bookkeeping. Touched only by the run goroutine, so no locking.
	consecFails   int
	everSucceeded bool

	// The delta accumulator. prev is the last raw counter seen per endpoint per
	// key ("device"/"host"/"storage"/"total"/sglangPromptTokensKey); totals is
	// the running sum of deltas, which is what gets persisted. Same reset-safe
	// scheme as vllmMetricsSampler.fold — see vllm_metrics.go.
	prev   map[string]map[string]float64
	totals map[string]float64
	resets int

	cancel context.CancelFunc
	done   chan struct{}
}

// startSGLangMetricsSampler launches the sampler goroutine for cfg.Model when
// the spec is type=openai_sglang and request data is being saved. Returns nil
// when sampling doesn't apply. Callers must invoke stop() before closing rdw.
//
// tracker is the same *activeDatasetTracker instance passed to
// startVLLMMetricsSampler (it is engine-agnostic — see vllm_metrics.go) so the
// active-dataset-tokens overlay populates for SGLang runs the same way it
// does for vLLM.
func startSGLangMetricsSampler(ctx context.Context, model string, tracker *activeDatasetTracker, rdw *requestDataWriter) *sglangMetricsSampler {
	if rdw == nil || tracker == nil {
		return nil
	}
	urls := sglangMetricsEndpoints(model)
	if len(urls) == 0 {
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	s := &sglangMetricsSampler{
		model:    model,
		urls:     urls,
		tracker:  tracker,
		rdw:      rdw,
		interval: sglangMetricsSampleInterval,
		client:   &http.Client{},
		now:      time.Now,
		prev:     map[string]map[string]float64{},
		totals:   map[string]float64{},
		logf:     func(f string, a ...any) { fmt.Fprintf(os.Stderr, "[sglang-metrics] "+f+"\n", a...) },
		cancel:   cancel,
		done:     make(chan struct{}),
	}
	go s.run(runCtx)
	return s
}

// stop terminates the sampler and waits for the goroutine to exit, so no
// write can race the rdw close that follows.
func (s *sglangMetricsSampler) stop() {
	s.cancel()
	<-s.done
}

func (s *sglangMetricsSampler) run(ctx context.Context) {
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
// totals, and writes one sample record. See vllmMetricsSampler.sampleOnce for
// the full rationale (per-endpoint deltas, partial coverage still recorded).
//
// It always reports keepPolling=true unless ctx is cancelled: the operator
// asserted type=openai_sglang, so — unlike vLLM's speculative sampler — there
// is no guess here to give up on.
func (s *sglangMetricsSampler) sampleOnce(ctx context.Context) (keepPolling bool) {
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
		// Nothing answered. Still record the interval — with EndpointsOK at 0 it
		// reads as unobserved rather than idle, which is the true statement and
		// the one a flat bar cannot make on its own.
		s.write(0)
		if lastErr != nil {
			if s.consecFails == 0 {
				s.log("%s unavailable (%v) — skipping this sample, still polling every %s", lastBadURL, lastErr, s.interval)
			}
			s.consecFails++
		}
		return true
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
// Identical reset-safe scheme to vllmMetricsSampler.fold — see its doc.
func (s *sglangMetricsSampler) fold(url string, vals map[string]float64) {
	if s.prev == nil {
		s.prev = map[string]map[string]float64{}
	}
	if s.totals == nil {
		s.totals = map[string]float64{}
	}
	prev := s.prev[url]
	if prev == nil {
		prev = map[string]float64{}
		s.prev[url] = prev
	}
	for key, cur := range vals {
		last, had := prev[key]
		switch {
		case !had:
			// First sighting establishes a BASELINE and contributes nothing — a
			// run wants only what happened DURING it (see vllm_metrics.go's fold
			// for the full rationale).
		case cur < last:
			// Restarted. The post-restart value counts from zero.
			s.totals[key] += cur
			s.resets++
		default:
			s.totals[key] += cur - last
		}
		prev[key] = cur
	}
}

// write persists the accumulated totals with this round's coverage. compute
// is recomputed fresh each time from the two accumulated totals it's derived
// from (prompt_tokens_total and the "total" cache_source label) rather than
// being folded as its own key, and clamped at 0 as a floor against transient
// scrape skew between the two independently-updated counters.
func (s *sglangMetricsSampler) write(endpointsOK int) {
	if s.rdw == nil || s.tracker == nil {
		return
	}
	adt, active := s.tracker.Sum()
	compute := s.totals[sglangPromptTokensKey] - s.totals["total"]
	if compute < 0 {
		compute = 0
	}
	rec := sglangMetricsSample{
		RecordType: recordTypeSGLangMetricsSample,
		TS:         s.now(),
		Model:      s.model,
		Sources: sglangSourceCounters{
			Compute: int64(compute),
			Device:  int64(s.totals["device"]),
			Host:    int64(s.totals["host"]),
			Storage: int64(s.totals["storage"]),
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

// noteSuccess records a landed sample. The first one is announced.
func (s *sglangMetricsSampler) noteSuccess() {
	s.consecFails = 0
	if s.everSucceeded {
		return
	}
	s.everSucceeded = true
	s.log("%s serves %s/%s — sampling every %s", strings.Join(s.urls, ", "), sglangCachedTokensTotalFamily, sglangPromptTokensTotalFamily, s.interval)
}

func (s *sglangMetricsSampler) log(format string, args ...any) {
	if s.logf != nil {
		s.logf(format, args...)
	}
}

func (s *sglangMetricsSampler) fetchOne(ctx context.Context, url string) (map[string]float64, error) {
	reqCtx, cancel := context.WithTimeout(ctx, sglangMetricsFetchTimeout)
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
	vals, err := parseSGLangCounters(resp.Body)
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		return nil, fmt.Errorf("metrics fetch: families %s/%s not found", sglangCachedTokensTotalFamily, sglangPromptTokensTotalFamily)
	}
	return vals, nil
}

// sglangBandOrder is the band order buildSampleVizSGLang emits, and the value
// seriesData.BandOrder carries for an sglang-sampled series. Unlike vLLM's
// flat local/external split, sglang reports device (GPU radix cache), host
// (CPU/DRAM HiCache L2) and storage (WekaFS L3) as three distinct bands
// rather than collapsing device+host into one "local" bucket.
var sglangBandOrder = []string{"compute", "device", "host", "storage"}

// buildSampleVizSGLang converts raw cumulative sglang samples into
// per-interval band deltas (mix) and active-dataset points (adt) for the
// embedded report JS. See buildSampleVizVLLM (vllm_metrics.go) for the vLLM
// equivalent and the shared vizSampleSegment/vizBand shape.
func buildSampleVizSGLang(samples []sglangMetricsSample) (mix []vizSampleSegment, adt []vizAdtPoint) {
	if len(samples) == 0 {
		return nil, nil
	}
	sorted := make([]sglangMetricsSample, len(samples))
	copy(sorted, samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].TS.Before(sorted[j].TS) })

	// Same floor-only clamp as buildSampleVizVLLM — see its doc.
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
			EndpointsOK:    smp.EndpointsOK,
			EndpointsTotal: smp.EndpointsTotal,
			Bands: []vizBand{
				{Name: "compute", Tokens: clamp(smp.Sources.Compute, prev.Sources.Compute)},
				{Name: "device", Tokens: clamp(smp.Sources.Device, prev.Sources.Device)},
				{Name: "host", Tokens: clamp(smp.Sources.Host, prev.Sources.Host)},
				{Name: "storage", Tokens: clamp(smp.Sources.Storage, prev.Sources.Storage)},
			},
		})
	}
	return mix, adt
}
