package benchmark

// SGLang metrics collector: during `benchmark auto` runs against a
// type=openai_sglang endpoint, a per-model sampler goroutine polls the
// server's Prometheus /metrics endpoint every sglangMetricsSampleInterval and
// persists the current cache-hit-rate gauge into the same
// --save-request-data JSONL stream as the request rows, as records of their
// own type ("sglang_metrics_sample"). Sampling is strictly best-effort and
// can never affect the benchmark itself.
//
// This deliberately does NOT reuse vllmMetricsSampler's delta-accumulation
// scheme. vLLM's vllm:prompt_tokens_by_source is a monotonic counter, so a
// fleet total is the sum of per-endpoint DELTAS (see vllm_metrics.go).
// SGLang's sglang:cache_hit_rate is a GAUGE already expressed as a ratio in
// [0,1] — it is a point-in-time reading, not something that accumulates.
// Delta-summing it would produce a number with no meaning; the only correct
// thing to do with a gauge is read its current value.
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
//
// The gauge family is sglang:cache_hit_rate (sglang/srt/metrics/collector.py),
// labeled by model_name (and, on a multi-GPU deployment, engine/rank labels
// this code does not care about). Multiple label instances are averaged, not
// summed — each is already a ratio in [0,1], so averaging is the only
// combination that keeps the result in range.

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/weka/wekai/llm"
)

const (
	recordTypeSGLangMetricsSample = "sglang_metrics_sample"

	sglangCacheHitRateFamily = "sglang:cache_hit_rate"

	sglangMetricsSampleInterval = 60 * time.Second
	sglangMetricsFetchTimeout   = 5 * time.Second
)

// sglangMetricsSample is one periodic sample persisted into the request-data
// JSONL alongside requestDataRecord rows. record_type distinguishes it from
// request rows (which carry no record_type field) and from vllmMetricsSample.
type sglangMetricsSample struct {
	RecordType string    `json:"record_type"`
	TS         time.Time `json:"ts"`
	Model      string    `json:"model"`

	// CacheHitRate is the average of sglang:cache_hit_rate across all
	// endpoints and label instances that answered this round — a current
	// reading, not an accumulated delta (see package comment).
	CacheHitRate float64 `json:"cache_hit_rate"`

	// EndpointsOK of EndpointsTotal answered this round. Without them a flat
	// interval and an unobserved one look identical.
	EndpointsOK    int `json:"endpoints_ok"`
	EndpointsTotal int `json:"endpoints_total"`
}

// parseCacheHitRateGauge scans Prometheus text exposition for the
// sglang:cache_hit_rate gauge and returns the average value across every
// label-set instance found (e.g. one per model_name). Returns ok=false when
// the family is absent from the scrape.
func parseCacheHitRateGauge(r io.Reader) (avg float64, ok bool, err error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var sum float64
	var n int
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		rest, matched := strings.CutPrefix(line, sglangCacheHitRateFamily)
		if !matched {
			continue
		}
		// The next byte must open a label set or separate the name from the
		// value with whitespace — reject sibling series sharing the prefix
		// (there are none known today, but the family-name-not-followed-by-
		// '{'-or-space check mirrors parsePromptTokensBySource's guard).
		if len(rest) == 0 {
			continue
		}
		if rest[0] != '{' && rest[0] != ' ' {
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
		v, perr := strconv.ParseFloat(fields[0], 64)
		if perr != nil {
			continue
		}
		sum += v
		n++
	}
	if err := sc.Err(); err != nil {
		return 0, false, err
	}
	if n == 0 {
		return 0, false, nil
	}
	return sum / float64(n), true, nil
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
	rdw      *requestDataWriter
	interval time.Duration
	client   *http.Client
	now      func() time.Time
	logf     func(format string, args ...any)

	everSucceeded bool

	cancel context.CancelFunc
	done   chan struct{}
}

// startSGLangMetricsSampler launches the sampler goroutine for cfg.Model when
// the spec is type=openai_sglang and request data is being saved. Returns nil
// when sampling doesn't apply. Callers must invoke stop() before closing rdw.
func startSGLangMetricsSampler(ctx context.Context, model string, rdw *requestDataWriter) *sglangMetricsSampler {
	if rdw == nil {
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
		rdw:      rdw,
		interval: sglangMetricsSampleInterval,
		client:   &http.Client{},
		now:      time.Now,
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

// sampleOnce fetches every endpoint and writes one sample of the AVERAGE
// current gauge value across whichever endpoints answered. It reports
// whether the sampler should keep polling; ctx cancellation is the only case
// that stops it — the operator asserted type=openai_sglang, so a transient
// failure (server still loading weights, momentary timeout) is ridden out
// rather than treated as evidence of a bad guess (there is no guess here).
func (s *sglangMetricsSampler) sampleOnce(ctx context.Context) (keepPolling bool) {
	var sum float64
	ok := 0
	var lastErr error
	var lastBadURL string
	for _, u := range s.urls {
		v, err := s.fetchOne(ctx, u)
		if err != nil {
			if ctx.Err() != nil {
				return false
			}
			lastErr, lastBadURL = err, u
			continue
		}
		sum += v
		ok++
	}

	rate := 0.0
	if ok > 0 {
		rate = sum / float64(ok)
		s.noteSuccess()
	} else if lastErr != nil {
		s.log("%s unavailable (%v) — skipping this sample, still polling every %s", lastBadURL, lastErr, s.interval)
	}
	s.write(rate, ok)
	return true
}

func (s *sglangMetricsSampler) noteSuccess() {
	if s.everSucceeded {
		return
	}
	s.everSucceeded = true
	s.log("%s serves %s — sampling every %s", strings.Join(s.urls, ", "), sglangCacheHitRateFamily, s.interval)
}

func (s *sglangMetricsSampler) write(rate float64, endpointsOK int) {
	if s.rdw == nil {
		return
	}
	rec := sglangMetricsSample{
		RecordType:     recordTypeSGLangMetricsSample,
		TS:             s.now(),
		Model:          s.model,
		CacheHitRate:   rate,
		EndpointsOK:    endpointsOK,
		EndpointsTotal: len(s.urls),
	}
	// Write errors are swallowed: sampling must never affect the benchmark.
	_ = s.rdw.writeAny(rec)
}

func (s *sglangMetricsSampler) log(format string, args ...any) {
	if s.logf != nil {
		s.logf(format, args...)
	}
}

func (s *sglangMetricsSampler) fetchOne(ctx context.Context, url string) (float64, error) {
	reqCtx, cancel := context.WithTimeout(ctx, sglangMetricsFetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("metrics fetch: status %d", resp.StatusCode)
	}
	avg, ok, err := parseCacheHitRateGauge(resp.Body)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, fmt.Errorf("metrics fetch: family %s not found", sglangCacheHitRateFamily)
	}
	return avg, nil
}
