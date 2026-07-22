package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/weka/wekai/llm"
)

// hotGateFanoutMultiplier scales the hot gate so router-replay sub-agent fan-out
// doesn't block hot sessions. For synthetic/hermes (linear streams) only H slots
// are ever used — the multiplier is harmless headroom.
const hotGateFanoutMultiplier = 8

// AutoBenchmarkConfig holds configuration for the auto benchmark mode.
type AutoBenchmarkConfig struct {
	DocsDir                  string
	DocsContent              string   // Pre-loaded documentation content (used instead of DocsDir when set)
	Model                    string   // single model (legacy); prefer Models
	Models                   []string // multi-model; takes priority over Model
	Question                 string
	Timeout                  time.Duration
	MaxSeries                int           // safety cap, default 64
	StartSeries              int           // initial series count, default 1
	MaxConcurrency           int           // 0 = unlimited
	MinEvalRequests          int           // minimum total completed before any eval, default 10
	CacheTarget              float64       // cache hit rate threshold at which series scaling begins, default 0.90
	CacheWindowSize          int           // fixed number of recent requests for cache hit measurement, default 20
	ScaleWaitFactor          int           // fast measurement window factor for backoff recovery detection (series × this), default 2
	MinStabilization         int           // minimum requests for any stabilization window, default 20
	ScaleFactor              float64       // fractional scale step (0.20 = 20%), default 0.20
	MaxScaleStep             int           // maximum delta per scale event, default 8
	ErrorRateLimit           float64       // DEPRECATED: no-op, retained for flag compatibility (see MaxConsecutiveFailures)
	MaxConsecutiveFailures   int           // abort after this many failures in a row (any success resets the counter); default 512
	MaxTotalErrors           int           // abort after this many TOTAL errors (monotonic, not reset by success); 0 = disabled
	MinSeries                int           // informational minimum (not used in algo), default 2
	TTFTDegradationFactor    int           // TTFT multiplier above early cold-start baseline that disqualifies the heuristic entirely (default 4)
	TTFTHitThreshold         float64       // fraction of cold-start baseline below which a request is classified as a cache hit (default 0.5 = 50%)
	Concurrency              int           // fixed concurrency; disables hill-climber entirely (0 = auto)
	VerboseCache             bool          // print TTFT baseline/threshold/miss distribution on hit-rate change; forces non-TTY (periodic) display mode
	PrintResponses           bool          // print each request/response to stdout; forces non-TTY display mode
	PrintErrorsThreshold     time.Duration // if >0, print at most one error to stderr per this interval (rate-limited)
	SaveRequestDataDir       string        // if non-empty, write per-request JSONL to this dir
	Total                    int           // stop after N total completed requests (0 = unlimited)
	HotSeriesConcurrency     int           // H of the --series workers run as a 'hot' pool with a dedicated gate; 0 = no hot pool.
	RequestTimeout           time.Duration // per-request timeout (default 5m)
	Step                     int           // 0=disabled; token step size for prefix growth per request
	StepStartingTokens       int           // 0=start at Step; initial prefix token size for each series when Step > 0
	Tokens                   int           // upper bound on per-request prompt tokens (caps Step growth and DocsDir source). 0 = bounded only by len(fullDocs)/4.
	SharedPrefixPerSeries    int           // 0=disabled; N series share same doc prefix
	GlobalCacheHitRateTarget float64       // 0=disabled; block new series until global hit rate >= this
	MaxOutputTokens          int           // 0=use model spec default; override max_tokens for all requests
	// ExhaustSessions: when Step > 0 and a series' grown prefix reaches the
	// token cap, recycle the slot with a fresh session GUID instead of
	// pinning at 100% cache. The same goroutine continues with a new
	// seriesGUID, stepCurrentTokens reset to Step, and isFirstRequest=true
	// (cold). Naturally maintains GlobalCacheHitRateTarget by mixing fresh
	// sessions in as old ones retire. Requires Step > 0.
	ExhaustSessions bool

	// Dataset-replay mode. When FromDataset != "", each series replays one
	// conversation from the dataset instead of running the synthetic request
	// loop. The synthetic --step / --shared-prefix-per-series / --docs-dir
	// knobs are ignored in this mode.
	FromDataset     string // short name, e.g. "hermes-lambda"
	ReplaySeries    int    // number of conversations to replay (0 = whole dataset)
	ReplayNoStamp   bool   // when true, skip the per-run <ignore>RUN_GUID</ignore> prefix injection (default is to stamp so each run starts with a pristine server prefix cache while still permitting within-run cross-series cache hits)
	AbortOnCollapse bool   // when true, abort if windowed cache hit rate < 50% for 2 minutes (legacy collapse detector, off by default — fires spuriously on legitimate low-reuse workloads)
	// ReplayStopAtLowConcurrency terminates the run when the queue is drained
	// AND active worker count < desired concurrency — a long-tail cutoff so
	// throughput numbers reflect steady-state behavior rather than the slow
	// tail of 1-2 surviving long conversations.
	//
	// Note: activeReplayWorkers includes hot-pool workers, so the cutoff is
	// slightly conservative when --hot-series-concurrency > 0 — the target
	// remains the normal budget C (--concurrency).
	ReplayStopAtLowConcurrency bool

	// RunID is populated internally by RunAutoBenchmark at the start of each
	// run. It's the UUID injected into every conversation's system prompt
	// (when ReplayNoStamp is false). Per-run scope — conversations that share
	// a system prompt (e.g. hermes has only ~6 unique system prompts) will
	// hit the server cache for that shared prefix within a single run.
	RunID string

	// replayConversations is the preloaded dataset, populated once by
	// RunAutoBenchmark before any per-model goroutine spawns. Every model's
	// runSingleModelBenchmark builds its own replayQueue from this slice —
	// the slice is read-only after load so aliasing is safe. Loading once
	// here avoids hitting the parquet reader N times in parallel (which
	// serializes behind arrow-go's file handle and starves models 2..N of
	// startup progress).
	replayConversations []Conversation

	// Tree-aware router replay. When set, instead of synthetic prompts
	// or hermes-style flat conversations, the benchmark replays a tree of
	// CLI sessions captured by the router (see `wekai router
	// replay-prepare`). Each series = one session; within a session,
	// sub-agents fan out concurrently and honor parent->child sequencing.
	//
	// The replay file is JSONL (replay-v3): line 1 is a header carrying
	// the summary; subsequent lines are one session each. Sessions are
	// streamed off disk through a small bounded channel — the producer
	// reads only as fast as workers consume, keeping memory bounded
	// regardless of file size.
	RouterReplayFile   string
	RouterReplayRoles  string             // comma list of roles to include; empty = all
	routerReplayHeader RouterReplayHeader // header parsed from line 1, populated at startup
	// RouterReplaySeriesIndices, when non-nil, restricts replay to the given
	// 0-based session line indices (i.e. the Nth line after the header, N=0
	// being the first session). Populated from --replay-series-indices or
	// --replay-series-range by the CLI; nil means "replay all sessions".
	RouterReplaySeriesIndices map[int]bool

	// Dry-run mode (router replay only): skip HTTP, drive with synthetic timing.
	DryRun          bool
	DryRunColdTPS   int
	DryRunWarmTPS   int
	DryRunOutputTPS int

	CacheSimChunkBytes int // chunk size for cacheEstimator (0 = default 1024)

	// FIFOGateOrder: when true, the concurrencyGate wakes normal (non-cold)
	// waiters in strict FIFO order — the LEGACY behavior, kept reachable via
	// --random-gate-order=false. The DEFAULT (zero value) is uniformly
	// random wake order: in the oversubscribed regime strict FIFO enforces
	// exact round-robin over series (each series releases its slot then
	// re-queues behind every other waiting series) — the adversarial worst
	// case for GPU prefix-cache LRU, since it guarantees maximum
	// time-between-revisits for every series. Random order lets some series
	// get consecutive turns, trading strict fairness for cache-friendlier
	// revisit patterns. Default flipped to random on 2026-07-22: historical
	// FIFO runs compare only against explicit --random-gate-order=false runs
	// from here on. Cold-start waiters are always served first, in FIFO
	// order, regardless of this setting.
	FIFOGateOrder bool
}

// requestDataRecord holds per-request data written to JSONL output.
type requestDataRecord struct {
	// Timing
	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	TTFT       float64   `json:"ttft_ms"`          // milliseconds
	ResponseMs float64   `json:"response_time_ms"` // milliseconds

	// Identity
	Model      string `json:"model"`
	SeriesGUID string `json:"series_guid"`
	SeriesNum  int    `json:"series_num"`
	RequestNum int    `json:"request_num"`

	// Cache status
	CacheHit             bool `json:"cache_hit"`              // implicit or explicit
	ServerCacheConfirmed bool `json:"server_cache_confirmed"` // explicit only
	IsColdStart          bool `json:"is_cold_start"`

	// Token usage
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	CachedTokens int `json:"cached_tokens"`

	// Status
	IsError      bool   `json:"is_error"`
	ErrorMessage string `json:"error_message,omitempty"`
	IsEmpty      bool   `json:"is_empty"`

	LocalCacheRatio float64 `json:"local_cache_ratio"`

	// Full prompt/response/raw-tail for failed requests only (omitted on success to avoid bloat)
	PromptText      string `json:"prompt_text,omitempty"`
	Question        string `json:"question,omitempty"`
	ResponseText    string `json:"response_text,omitempty"`
	RawResponseTail string `json:"raw_response_tail,omitempty"`
}

// requestDataWriter writes requestDataRecord entries as JSONL, safe for concurrent use.
type requestDataWriter struct {
	mu     sync.Mutex
	f      *os.File
	enc    *json.Encoder
	closed bool
}

var sanitizeModelRe = regexp.MustCompile(`[^a-zA-Z0-9._-]`)

func newRequestDataWriter(outputDir, model string, _ time.Time) (*requestDataWriter, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("create output dir: %w", err)
	}
	safeModel := sanitizeModelRe.ReplaceAllString(model, "_")
	filename := safeModel + ".jsonl"
	f, err := os.Create(outputDir + "/" + filename)
	if err != nil {
		return nil, fmt.Errorf("create JSONL file: %w", err)
	}
	return &requestDataWriter{
		f:   f,
		enc: json.NewEncoder(f),
	}, nil
}

// write encodes a request record. Writes after close() are silent no-ops:
// during abnormal shutdown (error storm + early kill) in-flight workers can
// race the deferred close, and a warning-per-row "file already closed"
// storm helps nobody.
func (w *requestDataWriter) write(rec requestDataRecord) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	return w.enc.Encode(rec)
}

// writeAny encodes a non-request record (e.g. vllmMetricsSample) into the
// same JSONL stream. Safe for concurrent use with write(); same silent
// no-op after close.
func (w *requestDataWriter) writeAny(v any) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	return w.enc.Encode(v)
}

func (w *requestDataWriter) close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	return w.f.Close()
}

const maxEarlyCold = 48
const maxEarlyHit = 96

// completionRecord is a single completed request.
type completionRecord struct {
	completedAt          time.Time
	ttft                 time.Duration
	isError              bool
	isColdStart          bool // first request of this series goroutine
	cacheHit             bool // explicit (CachedTokens > 0) OR implicit (TTFT ≤ 50% of cold baseline) — used for scaling
	serverCacheConfirmed bool // explicit metadata only (CachedTokens > 0) — used for display
	inputTokens          int
	outputTokens         int
	cachedTokens         int     // server-reported cached prompt tokens
	localCacheRatio      float64 // per-request estimate from cacheEstimator.Observe()
}

// cacheMetrics summarises cache behaviour over a recent window.
type cacheMetrics struct {
	hitRate           float64 // implicit TTFT-heuristic hits / total non-error — drives all scaling decisions
	serverConfirmRate float64 // serverCacheConfirmed (metadata only) / total non-error — display only, never used for decisions
	count             int     // non-error records analysed
	coldStarts        int
	serverTokenRate   float64 // cached/input token ratio from server-reported data
	serverReported    bool    // true when at least one record had cachedTokens > 0
}

// throughputMetrics summarises request throughput over a time window.
type throughputMetrics struct {
	reqPerSec       float64
	ttftP50         time.Duration
	ttftP95         time.Duration
	count           int
	errorRate       float64
	inputTokPerSec  float64
	outputTokPerSec float64
}

// completionStream is an append-only, mutex-protected record store.
// It self-trims to keep the last maxKeep records whenever it doubles.
// global* counters are running totals (never trimmed) for O(1) lookups.
type completionStream struct {
	mu                 sync.Mutex
	records            []completionRecord
	maxKeep            int
	globalNonError     int64 // all non-error completions ever added
	globalNonCold      int64 // non-error, non-cold-start completions ever added
	globalInputCold    int64 // full prompt tokens (non-cached + server-cached) from cold-start requests
	globalInputWarm    int64 // full prompt tokens (non-cached + server-cached) from warm (non-cold) requests
	globalOutputTokens int64 // output tokens from all non-error requests
	globalCachedTokens int64 // server-reported cached prompt tokens (all non-error requests)
}

func newCompletionStream(maxKeep int) *completionStream {
	if maxKeep < 50 {
		maxKeep = 50
	}
	return &completionStream{maxKeep: maxKeep}
}

// Add appends a record, trims when size exceeds 2×maxKeep, and updates global counters.
func (s *completionStream) Add(r completionRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append(s.records, r)
	if len(s.records) > s.maxKeep*2 {
		copy(s.records, s.records[len(s.records)-s.maxKeep:])
		s.records = s.records[:s.maxKeep]
	}
	if !r.isError {
		s.globalNonError++
		// inputTokens is already net-of-cache (providers subtract server-cached
		// tokens out of prompt_tokens — see internal/llm/openai.go). The cold/warm
		// split is driven by the per-request localCacheRatio from the content-level
		// estimator, so warm and cold track the fraction of the full logical prompt
		// volume that was likely cached at this point in the series.
		// Full prompt = non-cached input + server-cached tokens.
		full := int64(r.inputTokens) + int64(r.cachedTokens)
		warm := int64(float64(full)*r.localCacheRatio + 0.5)
		if warm > full {
			warm = full
		}
		if warm < 0 {
			warm = 0
		}
		s.globalNonCold++ // kept for compatibility
		s.globalInputWarm += warm
		s.globalInputCold += full - warm
		s.globalOutputTokens += int64(r.outputTokens)
		s.globalCachedTokens += int64(r.cachedTokens)
	}
}

// UpdateMaxKeep grows maxKeep (never shrinks).
func (s *completionStream) UpdateMaxKeep(n int) {
	s.mu.Lock()
	if n > s.maxKeep {
		s.maxKeep = n
	}
	s.mu.Unlock()
}

// CacheMetrics computes cache hit rates over the last max(windowSize, minCount) non-error records.
func (s *completionStream) CacheMetrics(windowSize, minCount int) cacheMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()

	want := windowSize
	if want < minCount {
		want = minCount
	}

	n := len(s.records)
	start := n - want
	if start < 0 {
		start = 0
	}
	window := s.records[start:]

	var cold, hits, serverConfirmed, total int
	var winInput, winCached int
	for _, r := range window {
		if r.isError {
			continue // skip errors
		}
		total++ // cold starts count as misses in the denominator
		winInput += r.inputTokens
		winCached += r.cachedTokens
		if r.isColdStart {
			cold++ // first request of a series: always a structural miss
			// intentionally not incrementing hits — first request is always a cache miss
			continue
		}
		if r.cacheHit {
			hits++
		}
		if r.serverCacheConfirmed {
			serverConfirmed++
		}
	}

	var hitRate, serverRate, serverTokenRate float64
	serverReported := winCached > 0
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}
	if total > 0 {
		serverRate = float64(serverConfirmed) / float64(total)
	}
	// winInput is net-of-cache; full prompt = winInput + winCached. The server
	// cache hit fraction is cached / full-prompt, which stays in [0,1]. Dividing
	// by winInput alone overshoots 100% whenever the server caches most of the
	// prompt (the net input shrinks toward zero).
	if winInput+winCached > 0 {
		serverTokenRate = float64(winCached) / float64(winInput+winCached)
	}
	return cacheMetrics{
		hitRate:           hitRate,
		serverConfirmRate: serverRate,
		count:             total,
		coldStarts:        cold,
		serverTokenRate:   serverTokenRate,
		serverReported:    serverReported,
	}
}

// DisplayHitRate returns the best available cache hit rate for display.
// Uses the server-reported token ratio when available; falls back to the TTFT heuristic.
func (m cacheMetrics) DisplayHitRate() float64 {
	if m.serverReported {
		return m.serverTokenRate
	}
	return m.hitRate
}

// GlobalLocalCacheRate returns the all-time fraction of warm input tokens among all input tokens.
// O(1): reads two running counters maintained in Add(), never scans history.
// Purely local: a request is "cached" when its series already submitted this prefix before.
// Token-based (not request-based) so it matches the % warm metric in the summary.
func (s *completionStream) GlobalLocalCacheRate() float64 {
	s.mu.Lock()
	warm := s.globalInputWarm
	cold := s.globalInputCold
	s.mu.Unlock()
	total := warm + cold
	if total == 0 {
		return 0
	}
	return float64(warm) / float64(total)
}

// TokenTotals returns all-time token counts split by cold/warm for input and total output.
type tokenTotals struct {
	inputCold int64
	inputWarm int64
	output    int64
	cached    int64
}

func (s *completionStream) TokenTotals() tokenTotals {
	s.mu.Lock()
	t := tokenTotals{
		inputCold: s.globalInputCold,
		inputWarm: s.globalInputWarm,
		output:    s.globalOutputTokens,
		cached:    s.globalCachedTokens,
	}
	s.mu.Unlock()
	return t
}

// ThroughputMetricsByCount computes req/s and TTFT percentiles over the last n records,
// using the actual elapsed time between first and last completion in that window.
// This is adaptive to request latency: slow models get a proportionally longer time span
// without requiring a tuned time window. Returns zero RPS if fewer than 2 records.
func (s *completionStream) ThroughputMetricsByCount(n int) throughputMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()

	start := len(s.records) - n
	if start < 0 {
		start = 0
	}
	window := s.records[start:]
	if len(window) < 2 {
		return throughputMetrics{}
	}

	span := window[len(window)-1].completedAt.Sub(window[0].completedAt).Seconds()

	var errCount, totalIn, totalOut int
	var ttfts []time.Duration
	for _, r := range window {
		if r.isError {
			errCount++
		} else {
			ttfts = append(ttfts, r.ttft)
			totalIn += r.inputTokens
			totalOut += r.outputTokens
		}
	}

	var rps, inTPS, outTPS float64
	if span > 0 {
		rps = float64(len(window)) / span
		inTPS = float64(totalIn) / span
		outTPS = float64(totalOut) / span
	}

	var p50, p95 time.Duration
	if len(ttfts) > 0 {
		sort.Slice(ttfts, func(i, j int) bool { return ttfts[i] < ttfts[j] })
		p50 = ttfts[int(float64(len(ttfts)-1)*0.50)]
		p95 = ttfts[int(float64(len(ttfts)-1)*0.95)]
	}

	var errRate float64
	if len(window) > 0 {
		errRate = float64(errCount) / float64(len(window))
	}

	return throughputMetrics{
		reqPerSec:       rps,
		ttftP50:         p50,
		ttftP95:         p95,
		count:           len(window),
		errorRate:       errRate,
		inputTokPerSec:  inTPS,
		outputTokPerSec: outTPS,
	}
}

// ErrorRate computes the error rate over the last n records.
func (s *completionStream) ErrorRate(n int) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	start := len(s.records) - n
	if start < 0 {
		start = 0
	}
	window := s.records[start:]
	if len(window) == 0 {
		return 0
	}
	var errs int
	for _, r := range window {
		if r.isError {
			errs++
		}
	}
	return float64(errs) / float64(len(window))
}

// RecentHitTTFT returns the average TTFT of cache-hit records in the
// last `window` records (excluding errors, cold-starts, zero-TTFT).
// Returns 0 if no qualifying records exist in the window.
func (s *completionStream) RecentHitTTFT(window int) time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.records)
	start := n - window
	if start < 0 {
		start = 0
	}
	var count int
	var sum time.Duration
	for _, r := range s.records[start:] {
		if r.isError || r.isColdStart || !r.cacheHit || r.ttft <= 0 {
			continue
		}
		sum += r.ttft
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / time.Duration(count)
}

// RecentColdTTFT returns the average TTFT of the last `n` cold-start (non-error) records.
// Used as a rolling baseline for hit-threshold comparison: reflects current system conditions
// rather than the frozen early-run snapshot used for disqualification.
// Returns 0 if fewer than 1 qualifying records exist.
func (s *completionStream) RecentColdTTFT(n int) time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	var ttfts []time.Duration
	for i := len(s.records) - 1; i >= 0 && len(ttfts) < n; i-- {
		r := s.records[i]
		if r.isError || !r.isColdStart || r.ttft <= 0 {
			continue
		}
		ttfts = append(ttfts, r.ttft)
	}
	if len(ttfts) == 0 {
		return 0
	}
	var sum time.Duration
	for _, t := range ttfts {
		sum += t
	}
	return sum / time.Duration(len(ttfts))
}

// MissTTFTStats returns p50 and p95 of TTFT for non-cold, non-error, cache-miss requests
// in the last `window` records. Useful for diagnosing whether misses are borderline or
// genuinely slow (compared to the implicit-hit threshold = coldBaseline/2).
func (s *completionStream) MissTTFTStats(window int) (p50, p95 time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.records)
	start := n - window
	if start < 0 {
		start = 0
	}
	var ttfts []time.Duration
	for _, r := range s.records[start:] {
		if r.isError || r.isColdStart || r.cacheHit || r.ttft <= 0 {
			continue
		}
		ttfts = append(ttfts, r.ttft)
	}
	if len(ttfts) == 0 {
		return 0, 0
	}
	sort.Slice(ttfts, func(i, j int) bool { return ttfts[i] < ttfts[j] })
	p50 = ttfts[int(float64(len(ttfts)-1)*0.50)]
	p95 = ttfts[int(float64(len(ttfts)-1)*0.95)]
	return p50, p95
}

// concurrencyGate controls max parallel in-flight requests with a mutable limit.
// Cold-start waiters are served before normal waiters. When randomOrder is
// set, normal waiters are woken in uniformly random order instead of FIFO
// (see AutoBenchmarkConfig.FIFOGateOrder — random is the default); coldWaiters are always FIFO.
type concurrencyGate struct {
	mu          sync.Mutex
	limit       int
	active      int
	randomOrder bool
	waiters     []chan struct{} // normal waiters; FIFO unless randomOrder
	coldWaiters []chan struct{} // cold-start priority waiters (always FIFO, served first)
}

func newConcurrencyGate(limit int, randomOrder bool) *concurrencyGate {
	return &concurrencyGate{limit: limit, randomOrder: randomOrder}
}

// popNormalWaiter removes and returns the next normal waiter to serve, or nil
// if g.waiters is empty. Caller must hold g.mu. FIFO (index 0) by default;
// with randomOrder set, picks a uniformly random index and swap-removes it
// (O(1), reorders the slice but every element is still served exactly once).
// Uniform random selection is statistically starvation-free: every waiting
// series has the same 1/n chance on each release regardless of queue
// position or how long it has already waited.
func (g *concurrencyGate) popNormalWaiter() chan struct{} {
	n := len(g.waiters)
	if n == 0 {
		return nil
	}
	if !g.randomOrder {
		w := g.waiters[0]
		g.waiters = g.waiters[1:]
		return w
	}
	i := rand.IntN(n)
	w := g.waiters[i]
	g.waiters[i] = g.waiters[n-1]
	g.waiters = g.waiters[:n-1]
	return w
}

// AcquireCold blocks until a slot is available, with priority over normal Acquire calls.
// Used by series goroutines for their first (cold-start) request only.
func (g *concurrencyGate) AcquireCold(ctx context.Context) error {
	g.mu.Lock()
	if g.active < g.limit {
		g.active++
		g.mu.Unlock()
		return nil
	}
	ch := make(chan struct{}, 1)
	g.coldWaiters = append(g.coldWaiters, ch)
	g.mu.Unlock()

	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		g.mu.Lock()
		for i, w := range g.coldWaiters {
			if w == ch {
				g.coldWaiters = append(g.coldWaiters[:i], g.coldWaiters[i+1:]...)
				break
			}
		}
		g.mu.Unlock()
		return ctx.Err()
	}
}

// Acquire blocks until a slot is available (normal priority).
func (g *concurrencyGate) Acquire(ctx context.Context) error {
	g.mu.Lock()
	if g.active < g.limit {
		g.active++
		g.mu.Unlock()
		return nil
	}
	ch := make(chan struct{}, 1)
	g.waiters = append(g.waiters, ch)
	g.mu.Unlock()

	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		g.mu.Lock()
		for i, w := range g.waiters {
			if w == ch {
				g.waiters = append(g.waiters[:i], g.waiters[i+1:]...)
				break
			}
		}
		g.mu.Unlock()
		return ctx.Err()
	}
}

// Release frees a slot. Priority order: cold → normal.
func (g *concurrencyGate) Release() {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(g.coldWaiters) > 0 {
		w := g.coldWaiters[0]
		g.coldWaiters = g.coldWaiters[1:]
		w <- struct{}{}
		return
	}
	if w := g.popNormalWaiter(); w != nil {
		w <- struct{}{}
		return
	}
	g.active--
}

// SetLimit adjusts the concurrency limit and wakes waiters for any new slots.
func (g *concurrencyGate) SetLimit(newLimit int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.limit = newLimit
	for g.active < g.limit {
		if len(g.coldWaiters) > 0 {
			w := g.coldWaiters[0]
			g.coldWaiters = g.coldWaiters[1:]
			w <- struct{}{}
			g.active++
			continue
		}
		if w := g.popNormalWaiter(); w != nil {
			w <- struct{}{}
			g.active++
			continue
		}
		break
	}
}

// GateStats returns current gate counters for diagnostics.
func (g *concurrencyGate) GateStats() (active, coldWaiting, normalWaiting int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.active, len(g.coldWaiters), len(g.waiters)
}

// autoTermReason describes why the auto benchmark terminated.

type autoTermReason int

const (
	termReasonCollapse             autoTermReason = iota // cache hit < 50% sustained for 1 minute
	termReasonError                                      // error rate exceeded
	termReasonSignal                                     // SIGINT/SIGTERM
	termReasonTimeout                                    // context deadline
	termReasonMaxSeries                                  // safety cap hit
	termReasonTotal                                      // total requests reached
	termReasonReplayDone                                 // replay queue drained, all series done
	termReasonReplayLowConcurrency                       // replay queue drained and active workers fell below target concurrency
)

func (r autoTermReason) String() string {
	switch r {
	case termReasonCollapse:
		return "Cache hit sustained below 50% for 2 minutes"
	case termReasonError:
		return "Fatal error (high failure rate)"
	case termReasonSignal:
		return "Interrupted by signal"
	case termReasonTimeout:
		return "Timeout reached"
	case termReasonMaxSeries:
		return "MaxSeries safety cap reached"
	case termReasonTotal:
		return "Total completed requests reached"
	case termReasonReplayDone:
		return "Replay dataset fully processed"
	case termReasonReplayLowConcurrency:
		return "Replay stopped: active workers fell below target concurrency"
	}
	return "Unknown"
}

// autoState holds the mutable benchmark state shared between evaluator and series goroutines.
type autoState struct {
	mu sync.Mutex

	// Scaling state
	series      int
	concurrency int

	// Concurrency hill-climber state
	concEvalAfter   int64   // total completed before next concurrency eval
	seriesEvalAfter int64   // total completed before next series scale eval
	lastEvalRPS     float64 // req/s at last concurrency evaluation (fixed window)
	climbDirection  int     // +1 climbing, -1 retreating, 0 holding
	stallCount      int     // consecutive evals where RPS didn't meaningfully change

	seriesDone   bool
	cacheWarning bool // server doesn't appear to support caching

	allTimePeakRPS    float64 // highest req/s ever seen (never reset)
	allTimePeakConc   int     // concurrency at which allTimePeakRPS was achieved
	allTimePeakSeries int     // series count at which allTimePeakRPS was achieved

	// Atomic counters (no lock needed for reads/writes)
	totalCompleted atomic.Int64
	totalEmitted   atomic.Int64
	totalErrors    atomic.Int64

	// Unix-nano timestamp of last error printed via --print-errors-threshold.
	// 0 means no error printed yet. Used to rate-limit error stderr output.
	lastErrorPrintNs atomic.Int64

	// Replay-mode: number of conversations fully walked (one increment per
	// completed Conversation, regardless of turn count). Zero in synthetic mode.
	seriesReplayCompleted atomic.Int64

	// Running count of consecutive request failures across ALL workers. Reset
	// to 0 on any success, incremented on any error. Used by the simple abort
	// heuristic — if this exceeds cfg.MaxConsecutiveFailures the run is
	// aborted as "Fatal error". Replaces the old windowed error-rate heuristic
	// which spuriously fired on low overall error counts.
	consecutiveFailures atomic.Int64

	// Replay-mode queue (nil in synthetic mode). Set once at startup and read
	// by updateSnap; the queue itself has its own mutex for Pull/Remaining.
	replay *replayQueue

	// Tree-aware router-replay stream (nil unless --router-replay-file).
	// Mutually exclusive with replay above. Producer is a goroutine
	// reading replay-v2 JSONL lines into a small bounded channel; series
	// workers Pull() from it. Backpressure → producer reads ahead by at
	// most chan_capacity sessions.
	routerReplay *routerReplayStream

	// Count of replay-mode worker goroutines currently alive. Bumped at spawn,
	// decremented when the worker exits (queue drained or context cancel).
	// Shown in the live one-liner as active=N so the user can see that
	// effective concurrency has dropped below st.series once workers started
	// running out of conversations to pull.
	activeReplayWorkers atomic.Int64

	// Content-level cache estimator (*cacheEstimator) that estimates repeated
	// (warm) prompt tokens client-side across all modes (synthetic/step/replay).
	// Its Ratio() drives the cold/warm split and gcache display, replacing the
	// coarse per-request bucketing (which trends to 100% warm by construction
	// in replay, and misses partial reuse in --step growth).
	estimator *cacheEstimator

	// Per-series latest-prompt-token snapshots for the active-dataset metric
	// sampled by the vLLM metrics collector (see vllm_metrics.go). Always
	// non-nil; series loops Update on each successful response and Reset on
	// slot recycle.
	datasetTracker *activeDatasetTracker

	// Global cold-start TTFT tracking for implicit cache detection
	coldStartTTFTCount atomic.Int64 // count of cold-start samples (used for series-scaling gate)
	ttftDegradedCount  atomic.Int64 // requests disqualified from cache-hit by TTFT degradation

	// Persistent early-sample buffers — never trimmed, survive stream eviction.
	printMu sync.Mutex // serialises --print-responses output across concurrent series

	earlyColdMu    sync.Mutex
	earlyColdTTFTs []time.Duration // first ≤maxEarlyCold cold-start TTFTs

	earlyHitMu    sync.Mutex
	earlyHitTTFTs []time.Duration // first ≤maxEarlyHit cache-hit TTFTs

	// Stream and gate
	stream *completionStream
	gate   *concurrencyGate

	// Hot-pool gate: nil when no hot series configured; when non-nil, the first
	// H workers use this dedicated gate sized H × hotGateFanoutMultiplier.
	hotGate *concurrencyGate
}

// earlyColdStartTTFT returns the average TTFT of the first min(16, n/3) cold-start samples.
// Uses a persistent buffer — never affected by stream trimming.
func (st *autoState) earlyColdStartTTFT() time.Duration {
	st.earlyColdMu.Lock()
	defer st.earlyColdMu.Unlock()
	total := len(st.earlyColdTTFTs)
	n := total / 3
	if n > 16 {
		n = 16
	}
	if n < 1 {
		return 0
	}
	var sum time.Duration
	for _, t := range st.earlyColdTTFTs[:n] {
		sum += t
	}
	return sum / time.Duration(n)
}

// earlyHitTTFT returns the average TTFT of the first min(32, n/3) cache-hit samples.
// Uses a persistent buffer — never affected by stream trimming.
func (st *autoState) earlyHitTTFT() time.Duration {
	st.earlyHitMu.Lock()
	defer st.earlyHitMu.Unlock()
	total := len(st.earlyHitTTFTs)
	n := total / 3
	if n > 32 {
		n = 32
	}
	if n < 1 {
		return 0
	}
	var sum time.Duration
	for _, t := range st.earlyHitTTFTs[:n] {
		sum += t
	}
	return sum / time.Duration(n)
}

// modelSnapshotUpdate carries a displaySnapshot update from a per-model goroutine.
type modelSnapshotUpdate struct {
	modelIndex int
	snap       *displaySnapshot
	model      string
}

// modelBenchmarkResult holds the outcome of benchmarking a single model.
type modelBenchmarkResult struct {
	model  string
	result autoBenchmarkResult
	err    error
}

// autoBenchmarkResult is the final result of RunAutoBenchmark.
type autoBenchmarkResult struct {
	termReason        autoTermReason
	elapsed           time.Duration
	optimalSeries     int
	optimalConc       int
	allTimePeakRPS    float64
	allTimePeakConc   int
	allTimePeakSeries int
	cacheHitRate      float64
	cacheWarning      bool
	totalCompleted    int64
	totalErrors       int64
	ttftP50           time.Duration
	ttftP95           time.Duration
	inputTokPerSec    float64
	outputTokPerSec   float64
	totalInputCold    int64 // input tokens from cold-start requests (had to prefill from scratch)
	totalInputWarm    int64 // input tokens from warm requests (prefix was cached)
	totalOutput       int64 // output tokens across all requests
	totalCachedTokens int64 // server-reported cached prompt tokens
}

// displaySnapshot is an atomic snapshot of state for the display goroutine.
type displaySnapshot struct {
	modelName            string // set for multi-model display; empty for single-model
	series               int
	concurrency          int
	seriesStatus         string
	concStatus           string
	cacheHitRate         float64
	globalLocalCacheRate float64 // all-time local hit rate (non-cold / total non-error)
	reqPerSec            float64
	inputTokPerSec       float64
	outputTokPerSec      float64
	allTimePeakRPS       float64
	allTimePeakConc      int
	allTimePeakSeries    int
	ttftP50              time.Duration
	ttftP95              time.Duration
	ttftCutoff           time.Duration // degradation cutoff = earlyColdBaseline * factor
	ttftDegradedCount    int64         // requests disqualified by TTFT cutoff
	totalCompleted       int64
	totalErrors          int64
	elapsed              time.Duration
	cacheWarning         bool
	bothDone             bool
	termReason           string // non-empty when model benchmark has finished
	gateActive           int
	gateColdWaiting      int
	gateNormalWait       int
	gateHotActive        int   // active slots in the hot-pool gate (non-zero only with hot-series-concurrency)
	totalInput           int64 // total full-prompt input tokens (cold + warm)
	totalInputWarm       int64 // warm full-prompt input tokens
	totalCached          int64 // server-reported cached prompt tokens (subset of totalInput)
	totalOutput          int64 // total output tokens

	// Replay-mode progress (replayTotal > 0 enables the replay panel in render)
	replayTotal     int   // total conversations queued (fixed at start)
	replayCompleted int64 // conversations fully walked
	replayRemaining int   // still-queued conversations
	replayActive    int64 // worker goroutines still alive (effective concurrency cap)
}

// displayState manages the display snapshot.
type displayState struct {
	snap atomic.Value // stores *displaySnapshot
}

func newDisplayState() *displayState {
	return &displayState{}
}

func (d *displayState) updateSnapshot(snap *displaySnapshot) {
	d.snap.Store(snap)
}

// formatKilo formats a float as "1.2k" when >= 1000, or "123" otherwise.
func formatKilo(f float64) string {
	if f <= 0 {
		return "--"
	}
	if f >= 1000 {
		return fmt.Sprintf("%.1fk", f/1000)
	}
	return fmt.Sprintf("%.0f", f)
}

func formatFloat(f float64) string {
	if f == 0 {
		return "0.0"
	}
	return fmt.Sprintf("%.1f", f)
}

func formatDur(d time.Duration) string {
	if d == 0 {
		return "-"
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func formatDuration(d time.Duration) string {
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60
	if h > 0 {
		return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}

// formatKiloInt formats an int64 token count as a human-readable string (e.g. 1234567 → "1.2M").
func formatKiloInt(n int64) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// printAutoSummary prints the final benchmark summary.
func printAutoSummary(res autoBenchmarkResult, cfg AutoBenchmarkConfig) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 62))
	fmt.Println(" Auto Benchmark Summary")
	fmt.Println(strings.Repeat("=", 62))
	fmt.Printf(" Termination reason : %s\n", res.termReason)
	fmt.Printf(" Elapsed time       : %s\n", res.elapsed.Truncate(time.Second))
	fmt.Println(strings.Repeat("-", 62))
	fmt.Printf(" Optimal series     : %d\n", res.optimalSeries)
	fmt.Printf(" Optimal concurrency: %d\n", res.optimalConc)
	if res.allTimePeakRPS > 0 {
		fmt.Printf(" Peak req/s         : %.2f @ concurrency %d, series %d\n", res.allTimePeakRPS, res.allTimePeakConc, res.allTimePeakSeries)
	} else {
		fmt.Printf(" Peak req/s         : --\n")
	}
	fmt.Println(strings.Repeat("-", 62))
	fmt.Printf(" Cache hit rate     : %.1f%%\n", res.cacheHitRate*100)
	fmt.Printf(" Tok/s in/out       : %s / %s\n", formatKilo(res.inputTokPerSec), formatKilo(res.outputTokPerSec))
	if res.cacheWarning {
		fmt.Println(" ⚠  Server may not support prompt caching")
	}
	fmt.Println(strings.Repeat("-", 62))
	totalInput := res.totalInputCold + res.totalInputWarm
	warmPct := 0.0
	if totalInput > 0 {
		warmPct = float64(res.totalInputWarm) / float64(totalInput) * 100
	}
	fmt.Printf(" Input tokens       : cold=%s  warm=%s  (%.1f%% warm)\n",
		formatKiloInt(res.totalInputCold), formatKiloInt(res.totalInputWarm), warmPct)
	serverTotal := res.totalInputCold + res.totalInputWarm
	serverUncached := serverTotal - res.totalCachedTokens
	if serverUncached < 0 {
		serverUncached = 0
	}
	cachedPct := 0.0
	if serverTotal > 0 {
		cachedPct = float64(res.totalCachedTokens) / float64(serverTotal) * 100
	}
	fmt.Printf(" Server cache       : cached=%s  uncached=%s  (%.1f%% cached)\n",
		formatKiloInt(res.totalCachedTokens), formatKiloInt(serverUncached), cachedPct)
	fmt.Printf(" Output tokens      : %s\n", formatKiloInt(res.totalOutput))
	fmt.Println(strings.Repeat("-", 62))
	fmt.Printf(" TTFT p50           : %s\n", formatDur(res.ttftP50))
	fmt.Printf(" TTFT p95           : %s\n", formatDur(res.ttftP95))
	fmt.Println(strings.Repeat("-", 62))
	fmt.Printf(" Total completed    : %d\n", res.totalCompleted)
	fmt.Printf(" Total errors       : %d\n", res.totalErrors)
	fmt.Println(strings.Repeat("=", 62))
}

// shortModelName returns a display-friendly model label.
// If the model string contains an alias=<name> parameter, that is returned directly.
// For dynamic/http URLs it returns "<host:port><path>" (no scheme, no provider prefix, no params).
// For other models it strips trailing params after the first comma.
func shortModelName(model string) string {
	// Check for alias=<value> in comma-separated params.
	if idx := strings.Index(model, ","); idx >= 0 {
		for _, param := range strings.Split(model[idx+1:], ",") {
			if strings.HasPrefix(param, "alias=") {
				if alias := param[len("alias="):]; alias != "" {
					return alias
				}
			}
		}
	}
	// Strip comma-separated params (e.g. ",type=openai_vllm,max_tokens=100")
	base := model
	if idx := strings.Index(base, ","); idx >= 0 {
		base = base[:idx]
	}
	// If it contains "://", keep host:port + path only (drop scheme and provider prefix like "dynamic/")
	if schemeIdx := strings.Index(base, "://"); schemeIdx >= 0 {
		hostPath := base[schemeIdx+3:] // strip "http://" or "https://"
		return hostPath
	}
	// For plain named models, strip a leading provider prefix (e.g. "anthropic/claude-3")
	if idx := strings.LastIndex(base, "/"); idx >= 0 {
		return base[idx+1:]
	}
	return base
}

// renderModelOneLiner returns a compact one-line status for non-TTY multi-model display.
func renderModelOneLiner(snap *displaySnapshot) string {
	ttftStr := formatDur(snap.ttftP50)
	// Replay progress is prefixed when a replay queue is active, so the user
	// always sees how much of the dataset is left regardless of --total caps.
	replayPrefix := ""
	if snap.replayTotal > 0 {
		replayPrefix = fmt.Sprintf("replay=%d/%d(queue=%d,active=%d) ",
			snap.replayCompleted, snap.replayTotal, snap.replayRemaining, snap.replayActive)
	}
	// in_flight: HTTP requests currently sitting in the gate's "active"
	// counter. Useful for spotting (a) long-tail drops (in_flight < series)
	// and (b) fan-out bursts (in_flight > series when sub-agents fire in
	// parallel and concurrency cap is higher than series count).
	if snap.termReason != "" {
		hotInfo := ""
		if snap.gateHotActive > 0 {
			hotInfo = fmt.Sprintf(" hot=%d", snap.gateHotActive)
		}
		return fmt.Sprintf("DONE(%s) %sseries=%d conc=%d in_flight=%d%s rps=%s cache=%.1f%% gcache=%.1f%% ttft50=%s total=%d errors=%d in=%s warm=%s scached=%s out=%s",
			snap.termReason, replayPrefix, snap.series, snap.concurrency, snap.gateActive, hotInfo,
			formatFloat(snap.reqPerSec), snap.cacheHitRate*100, snap.globalLocalCacheRate*100, ttftStr, snap.totalCompleted, snap.totalErrors,
			formatKiloInt(snap.totalInput), formatKiloInt(snap.totalInputWarm), formatKiloInt(snap.totalCached), formatKiloInt(snap.totalOutput))
	}
	hotInfo := ""
	if snap.gateHotActive > 0 {
		hotInfo = fmt.Sprintf(" hot=%d", snap.gateHotActive)
	}
	return fmt.Sprintf("%sseries=%d conc=%d in_flight=%d%s rps=%s cache=%.1f%% gcache=%.1f%% ttft50=%s total=%d errors=%d elapsed=%s in=%s warm=%s scached=%s out=%s",
		replayPrefix, snap.series, snap.concurrency, snap.gateActive, hotInfo,
		formatFloat(snap.reqPerSec), snap.cacheHitRate*100, snap.globalLocalCacheRate*100, ttftStr, snap.totalCompleted, snap.totalErrors,
		formatDuration(snap.elapsed),
		formatKiloInt(snap.totalInput), formatKiloInt(snap.totalInputWarm), formatKiloInt(snap.totalCached), formatKiloInt(snap.totalOutput))
}

// runMultiModelDisplay drives the live display for one or more models.
func runMultiModelDisplay(ctx context.Context, snapCh <-chan modelSnapshotUpdate, models []string, display *displayState) {
	latestSnaps := make([]*displaySnapshot, len(models))

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	render := func(_ bool) {
		for i, snap := range latestSnaps {
			if snap != nil {
				fmt.Printf("[%s] %s\n", shortModelName(models[i]), renderModelOneLiner(snap))
			}
		}
	}

	for {
		select {
		case upd, ok := <-snapCh:
			if !ok {
				render(true)
				return
			}
			latestSnaps[upd.modelIndex] = upd.snap
		case <-ticker.C:
			render(false)
		case <-ctx.Done():
			render(true)
			return
		}
	}
}

// printMultiModelSummary prints a consolidated summary for a multi-model run.
func printMultiModelSummary(results []modelBenchmarkResult) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println(" Multi-Model Auto Benchmark Summary")
	fmt.Println(strings.Repeat("=", 70))
	for _, mr := range results {
		fmt.Printf("\n  Model: %s\n", mr.model)
		if mr.err != nil {
			fmt.Printf("  ERROR: %v\n", mr.err)
			continue
		}
		res := mr.result
		fmt.Printf("  Termination : %s\n", res.termReason)
		fmt.Printf("  Elapsed     : %s\n", res.elapsed.Truncate(time.Second))
		fmt.Printf("  Peak req/s  : %s", formatFloat(res.allTimePeakRPS))
		if res.allTimePeakRPS > 0 {
			fmt.Printf(" @ c%d s%d", res.allTimePeakConc, res.allTimePeakSeries)
		}
		fmt.Println()
		fmt.Printf("  Cache hit   : %.1f%%  Tok/s in:%s out:%s\n", res.cacheHitRate*100, formatKilo(res.inputTokPerSec), formatKilo(res.outputTokPerSec))
		totalInput := res.totalInputCold + res.totalInputWarm
		warmPct := 0.0
		if totalInput > 0 {
			warmPct = float64(res.totalInputWarm) / float64(totalInput) * 100
		}
		fmt.Printf("  Input toks  : cold=%s  warm=%s  (%.1f%% warm)\n",
			formatKiloInt(res.totalInputCold), formatKiloInt(res.totalInputWarm), warmPct)
		sTotal := res.totalInputCold + res.totalInputWarm
		sUncached := sTotal - res.totalCachedTokens
		if sUncached < 0 {
			sUncached = 0
		}
		sPct := 0.0
		if sTotal > 0 {
			sPct = float64(res.totalCachedTokens) / float64(sTotal) * 100
		}
		fmt.Printf("  Server cache: cached=%s  uncached=%s  (%.1f%% cached)\n",
			formatKiloInt(res.totalCachedTokens), formatKiloInt(sUncached), sPct)
		fmt.Printf("  Output toks : %s\n", formatKiloInt(res.totalOutput))
		fmt.Printf("  TTFT p50/p95: %s / %s\n", formatDur(res.ttftP50), formatDur(res.ttftP95))
		fmt.Printf("  Total/Errors: %d / %d\n", res.totalCompleted, res.totalErrors)
		fmt.Println(strings.Repeat("-", 70))
	}
}

// runSingleModelBenchmark executes the full auto benchmark for one model.
// fullDocs is the pre-loaded documentation string.
// snapCh receives display snapshots; modelIndex identifies this model in the display.
// SIGINT handling is NOT performed here — the orchestrator owns benchCtx cancellation.
func runSingleModelBenchmark(
	benchCtx context.Context,
	cfg AutoBenchmarkConfig,
	fullDocs string,
	snapCh chan<- modelSnapshotUpdate,
	modelIndex int,
) autoBenchmarkResult {
	startTime := time.Now()

	// Set up per-request JSONL writer if configured.
	var rdw *requestDataWriter
	if cfg.SaveRequestDataDir != "" {
		var err error
		rdw, err = newRequestDataWriter(cfg.SaveRequestDataDir, cfg.Model, startTime)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to create request data writer: %v\n", err)
		} else {
			defer rdw.close()
		}
	}

	// Apply defaults.
	if cfg.MaxSeries <= 0 {
		cfg.MaxSeries = 64
	}
	if cfg.StartSeries <= 0 {
		cfg.StartSeries = 1
	}
	if cfg.StartSeries > cfg.MaxSeries {
		cfg.StartSeries = cfg.MaxSeries
	}
	if cfg.MinEvalRequests <= 0 {
		cfg.MinEvalRequests = 10
	}
	// When starting with multiple series, ensure the evaluator waits for all
	// initial cold starts to complete plus at least one warm request each
	// before making any scaling or termination decisions.
	if warmup := cfg.StartSeries * 2; warmup > cfg.MinEvalRequests {
		cfg.MinEvalRequests = warmup
	}
	if cfg.CacheTarget <= 0 {
		cfg.CacheTarget = 0.90
	}
	if cfg.CacheWindowSize <= 0 {
		cfg.CacheWindowSize = 20
	}
	if cfg.ScaleWaitFactor <= 0 {
		cfg.ScaleWaitFactor = 2
	}
	if cfg.MinStabilization <= 0 {
		cfg.MinStabilization = 20
	}
	if cfg.TTFTDegradationFactor <= 0 {
		cfg.TTFTDegradationFactor = 4
	}
	if cfg.TTFTHitThreshold <= 0 {
		cfg.TTFTHitThreshold = 0.5
	}
	if cfg.ScaleFactor <= 0 {
		cfg.ScaleFactor = 0.20
	}
	if cfg.MaxScaleStep <= 0 {
		cfg.MaxScaleStep = 8
	}
	if cfg.ErrorRateLimit <= 0 {
		cfg.ErrorRateLimit = 0.10
	}
	if cfg.MaxConsecutiveFailures <= 0 {
		cfg.MaxConsecutiveFailures = 512
	}
	// Defensive: hot-series count must not exceed the total number of series.
	if cfg.HotSeriesConcurrency > cfg.StartSeries {
		cfg.HotSeriesConcurrency = cfg.StartSeries
	}

	display := newDisplayState()

	// sendSnap forwards a display snapshot to the orchestrator non-blocking.
	sendSnap := func(snap *displaySnapshot) {
		snap.modelName = cfg.Model
		select {
		case snapCh <- modelSnapshotUpdate{modelIndex: modelIndex, snap: snap, model: cfg.Model}:
		default:
		}
	}

	// updateSnap builds a snapshot and sends it to the orchestrator's display goroutine.
	updateSnap := func(st *autoState) {
		st.mu.Lock()
		series := st.series
		concurrency := st.concurrency
		concEvalAfter := st.concEvalAfter
		seriesDone := st.seriesDone
		cacheWarning := st.cacheWarning
		allTimePeakRPS := st.allTimePeakRPS
		allTimePeakConc := st.allTimePeakConc
		allTimePeakSeries := st.allTimePeakSeries
		st.mu.Unlock()

		total := st.totalCompleted.Load()
		errors := st.totalErrors.Load()

		// Use CacheWindowSize for both cache and throughput so all displayed
		// metrics reflect the same recent-request window.
		cm := st.stream.CacheMetrics(cfg.CacheWindowSize, cfg.MinStabilization)
		tm := st.stream.ThroughputMetricsByCount(cfg.CacheWindowSize)
		tt := st.stream.TokenTotals()

		var seriesStatus string
		switch {
		case cacheWarning:
			seriesStatus = "⚠"
		case seriesDone:
			seriesStatus = "✓"
		default:
			seriesStatus = "↑"
		}

		var concStatus string
		if total < concEvalAfter {
			concStatus = fmt.Sprintf("⏳%d/%d", total, concEvalAfter)
		} else {
			concStatus = "↑"
		}

		snap := &displaySnapshot{
			series:               series,
			concurrency:          concurrency,
			seriesStatus:         seriesStatus,
			concStatus:           concStatus,
			cacheHitRate:         cm.DisplayHitRate(),
			globalLocalCacheRate: st.stream.GlobalLocalCacheRate(),
			inputTokPerSec:       tm.inputTokPerSec,
			outputTokPerSec:      tm.outputTokPerSec,
			reqPerSec:            tm.reqPerSec,
			allTimePeakRPS:       allTimePeakRPS,
			allTimePeakConc:      allTimePeakConc,
			allTimePeakSeries:    allTimePeakSeries,
			ttftP50:              tm.ttftP50,
			ttftP95:              tm.ttftP95,
			ttftCutoff:           st.earlyColdStartTTFT() * time.Duration(cfg.TTFTDegradationFactor),
			ttftDegradedCount:    st.ttftDegradedCount.Load(),
			totalCompleted:       total,
			totalErrors:          errors,
			elapsed:              time.Since(startTime),
			cacheWarning:         cacheWarning,
			bothDone:             seriesDone,
			totalInput:           tt.inputCold + tt.inputWarm,
			totalInputWarm:       tt.inputWarm,
			totalCached:          tt.cached,
			totalOutput:          tt.output,
		}
		gateActive, gateColdWaiting, gateNormalWait := st.gate.GateStats()
		gateHotActive := 0
		if st.hotGate != nil {
			ha, _, _ := st.hotGate.GateStats()
			gateHotActive = ha
			gateActive += ha
		}
		snap.gateActive = gateActive
		snap.gateColdWaiting = gateColdWaiting
		snap.gateNormalWait = gateNormalWait
		snap.gateHotActive = gateHotActive
		if st.replay != nil {
			snap.replayTotal = st.replay.Total()
			snap.replayCompleted = st.seriesReplayCompleted.Load()
			snap.replayRemaining = st.replay.Remaining()
			snap.replayActive = st.activeReplayWorkers.Load()
		}
		if st.routerReplay != nil {
			snap.replayTotal = st.routerReplay.Total()
			snap.replayCompleted = st.seriesReplayCompleted.Load()
			snap.replayRemaining = st.routerReplay.Remaining()
			snap.replayActive = st.activeReplayWorkers.Load()
		}
		snap.globalLocalCacheRate = st.stream.GlobalLocalCacheRate()
		display.updateSnapshot(snap)
		sendSnap(snap)
	}

	initMaxKeep := cfg.CacheWindowSize
	if initMaxKeep < cfg.MinStabilization {
		initMaxKeep = cfg.MinStabilization
	}
	if initMaxKeep < 200 {
		initMaxKeep = 200
	}

	initConc := 1
	if cfg.Concurrency > 0 {
		initConc = cfg.Concurrency
	}
	st := &autoState{
		series:            cfg.StartSeries,
		concurrency:       initConc,
		allTimePeakConc:   initConc,
		allTimePeakSeries: cfg.StartSeries,
		concEvalAfter:     int64(cfg.MinEvalRequests),
		seriesEvalAfter:   int64(cfg.MinEvalRequests),
		lastEvalRPS:       0,
		stream:            newCompletionStream(initMaxKeep),
		gate:              newConcurrencyGate(initConc, !cfg.FIFOGateOrder),
		datasetTracker:    newActiveDatasetTracker(),
	}
	if sampler := startVLLMMetricsSampler(benchCtx, cfg.Model, st.datasetTracker, rdw); sampler != nil {
		// stop() is deferred after rdw's close, so it runs first (LIFO) and
		// waits for the goroutine — no sample write can race the file close.
		defer sampler.stop()
	}
	if cfg.HotSeriesConcurrency > 0 {
		st.hotGate = newConcurrencyGate(cfg.HotSeriesConcurrency*hotGateFanoutMultiplier, !cfg.FIFOGateOrder)
	}

	// Single unconditional content-level estimator for all modes (synthetic/step/replay).
	st.estimator = newCacheEstimator(cfg.CacheSimChunkBytes)

	termChan := make(chan autoTermReason, 1)
	var termReason autoTermReason

	var seriesWg sync.WaitGroup

	// Create endpoint router for sticky series-to-endpoint assignment
	var endpointRouter *llm.EndpointRouter
	if llm.IsDynamicModel(cfg.Model) {
		if dynCfg, err := llm.ParseDynamicModel(cfg.Model); err == nil && len(dynCfg.BaseURLs) > 1 {
			endpointRouter = llm.NewEndpointRouter(dynCfg.BaseURLs)
		}
	}

	// Replay-mode: use the preloaded dataset from cfg (populated once in
	// RunAutoBenchmark before any per-model goroutine spawns — see comment on
	// AutoBenchmarkConfig.replayConversations). Each model gets its own queue
	// backed by the same immutable slice; pulls are index-independent.
	if cfg.FromDataset != "" {
		if len(cfg.replayConversations) == 0 {
			fmt.Fprintf(os.Stderr, "[auto][%s] replay mode with no preloaded conversations\n", shortModelName(cfg.Model))
			return autoBenchmarkResult{termReason: termReasonError, elapsed: time.Since(startTime)}
		}
		st.replay = newReplayQueue(cfg.replayConversations)
		fmt.Fprintf(os.Stderr, "[auto][%s] replay %d conversations queued\n",
			shortModelName(cfg.Model), st.replay.Total())
	}

	// Tree-aware router replay. Each model opens its own stream over the
	// replay file — sessions stream through a small bounded channel so
	// memory stays flat regardless of file size. (Two models running
	// concurrently both open the file; that's fine, each gets its own
	// independent producer.)
	if cfg.RouterReplayFile != "" {
		stream, err := openRouterReplayStream(cfg.RouterReplayFile, 8, cfg.ReplaySeries, cfg.RouterReplaySeriesIndices)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[auto][%s] open router-replay file: %v\n",
				shortModelName(cfg.Model), err)
			return autoBenchmarkResult{termReason: termReasonError, elapsed: time.Since(startTime)}
		}
		st.routerReplay = stream
		defer stream.Close()
		fmt.Fprintf(os.Stderr, "[auto][%s] router-replay streaming %d sessions (chan cap=8)\n",
			shortModelName(cfg.Model), stream.Total())
	}

	// Group GUIDs for shared-prefix-per-series: groupIdx → groupGUID
	groupGUIDs := make(map[int]string)
	var groupGUIDsMu sync.Mutex

	// groupPrimed tracks which groups have had their first cold start complete.
	// Once primed, subsequent series in the same group share the cached prefix — not truly cold.
	groupPrimed := make(map[int]bool)
	var groupPrimedMu sync.Mutex

	spawnSeries := func(seriesGUID string, seriesNum int) {
		// Derive hot-pool membership from the 1-based series index.
		isHot := cfg.HotSeriesConcurrency > 0 && seriesNum <= cfg.HotSeriesConcurrency
		workerGate := st.gate
		if isHot {
			workerGate = st.hotGate
		}

		// Replay-mode worker: ignore seriesGUID/seriesNum and synthetic
		// group mechanics — the replay loop re-derives them per conversation.
		if st.replay != nil {
			seriesWg.Add(1)
			st.activeReplayWorkers.Add(1)
			go func() {
				defer seriesWg.Done()
				defer st.activeReplayWorkers.Add(-1)
				var endpointOverride string
				if endpointRouter != nil {
					endpointOverride = endpointRouter.EndpointForSeries(seriesNum)
				}
				runReplaySeriesLoop(benchCtx, cfg, st, rdw, st.replay, endpointOverride, updateSnap, workerGate)
			}()
			return
		}
		if st.routerReplay != nil {
			seriesWg.Add(1)
			st.activeReplayWorkers.Add(1)
			go func() {
				defer seriesWg.Done()
				defer st.activeReplayWorkers.Add(-1)
				var endpointOverride string
				if endpointRouter != nil {
					endpointOverride = endpointRouter.EndpointForSeries(seriesNum)
				}
				runRouterReplaySeriesLoop(benchCtx, cfg, st, rdw, st.routerReplay, endpointOverride, fullDocs, updateSnap, workerGate)
			}()
			return
		}

		// Compute group GUID for shared-prefix feature
		var groupGUID string
		if cfg.SharedPrefixPerSeries > 0 {
			groupIdx := (seriesNum - 1) / cfg.SharedPrefixPerSeries
			groupGUIDsMu.Lock()
			if guid, ok := groupGUIDs[groupIdx]; ok {
				groupGUID = guid
			} else {
				groupGUID = uuid.New().String()
				groupGUIDs[groupIdx] = groupGUID
			}
			groupGUIDsMu.Unlock()
		}

		seriesWg.Add(1)
		go func() {
			defer seriesWg.Done()

			useShared := cfg.SharedPrefixPerSeries > 0
			var cachedPrompt string
			var stepCurrentTokens int
			stepDone := cfg.Step == 0
			if stepDone {
				cachedPrompt = buildSeriesPrompt(fullDocs, seriesGUID, groupGUID, 0, useShared)
			} else {
				stepCurrentTokens = cfg.StepStartingTokens
				if stepCurrentTokens <= 0 {
					stepCurrentTokens = cfg.Step
				}
			}
			isFirstRequest := true
			var coldStartTTFT time.Duration

			var endpointOverride string
			if endpointRouter != nil {
				endpointOverride = endpointRouter.EndpointForSeries(seriesNum)
			}

			for {
				select {
				case <-benchCtx.Done():
					return
				default:
				}

				// Check --total limit before emitting
				if cfg.Total > 0 && st.totalEmitted.Load() >= int64(cfg.Total) {
					return
				}
				st.totalEmitted.Add(1)

				if isFirstRequest {
					if err := workerGate.AcquireCold(benchCtx); err != nil {
						return
					}
				} else {
					if err := workerGate.Acquire(benchCtx); err != nil {
						return
					}
				}

				requestNum := int(st.totalCompleted.Load()) + 1
				reqTimeout := cfg.RequestTimeout
				if reqTimeout == 0 {
					reqTimeout = 5 * time.Minute
				}
				reqCtx, reqCancel := context.WithTimeout(benchCtx, reqTimeout)
				var prompt string
				if stepDone {
					prompt = cachedPrompt
				} else {
					prompt = buildSeriesPrompt(fullDocs, seriesGUID, groupGUID, stepCurrentTokens, useShared)
				}
				ratio := st.estimator.Observe(prompt + "\x00" + cfg.Question)
				metrics := runSingleRequest(reqCtx, cfg.Model, prompt, cfg.Question,
					requestNum, seriesNum, 0, seriesGUID, endpointOverride, cfg.MaxOutputTokens)
				reqCancel()

				workerGate.Release()

				// Active-dataset snapshot: latest full prompt tokens for this
				// slot, BEFORE any recycle below so the retiring session's last
				// response can't repopulate the reset slot.
				if metrics.Error == nil {
					st.datasetTracker.Update(seriesNum,
						int64(metrics.UsageData.InputTokens.Count+metrics.UsageData.CachedTokens.Count))
				}

				if !stepDone {
					maxDocTokens := len(fullDocs) / 4
					if cfg.Tokens > 0 && cfg.Tokens < maxDocTokens {
						maxDocTokens = cfg.Tokens
					}
					stepCurrentTokens += cfg.Step
					if stepCurrentTokens >= maxDocTokens {
						if cfg.ExhaustSessions {
							seriesGUID = uuid.New().String()
							stepCurrentTokens = cfg.StepStartingTokens
							if stepCurrentTokens <= 0 {
								stepCurrentTokens = cfg.Step
							}
							isFirstRequest = true
							coldStartTTFT = 0
							st.datasetTracker.Reset(seriesNum)
						} else {
							cachedPrompt = buildSeriesPrompt(fullDocs, seriesGUID, groupGUID, maxDocTokens, useShared)
							stepDone = true
						}
					}
				}

				if cfg.PrintResponses {
					st.printMu.Lock()
					ttftStr := formatDur(metrics.TimeToFirstToken)
					if metrics.Error != nil {
						fmt.Printf("\n\u2501\u2501\u2501 [%s] s%d r%d TTFT:%s ERROR: %v \u2501\u2501\u2501\n",
							shortModelName(cfg.Model), seriesNum, requestNum, ttftStr, metrics.Error)
					} else {
						fmt.Printf("\n\u2501\u2501\u2501 [%s] s%d r%d TTFT:%s total:%s in:%d out:%d \u2501\u2501\u2501\n%s\n",
							shortModelName(cfg.Model), seriesNum, requestNum, ttftStr,
							formatDur(metrics.TotalResponseTime),
							metrics.UsageData.InputTokens.Count, metrics.UsageData.OutputTokens.Count,
							metrics.Response)
					}
					st.printMu.Unlock()
				}

				isCold := isFirstRequest
				isFirstRequest = false

				// Group mechanics: if shared-prefix is enabled, only the first series in a group
				// to complete its first request is truly cold. Subsequent series in the same group
				// share identical prefix bytes already cached by the group's first series.
				if isCold && cfg.SharedPrefixPerSeries > 0 {
					groupIdx := (seriesNum - 1) / cfg.SharedPrefixPerSeries
					groupPrimedMu.Lock()
					alreadyPrimed := groupPrimed[groupIdx]
					if !alreadyPrimed {
						groupPrimed[groupIdx] = true
					}
					groupPrimedMu.Unlock()
					if alreadyPrimed {
						isCold = false
					}
				}

				if isCold && metrics.Error == nil {
					st.coldStartTTFTCount.Add(1)
					if metrics.TimeToFirstToken > 0 {
						coldStartTTFT = metrics.TimeToFirstToken
						st.earlyColdMu.Lock()
						if len(st.earlyColdTTFTs) < maxEarlyCold {
							st.earlyColdTTFTs = append(st.earlyColdTTFTs, metrics.TimeToFirstToken)
						}
						st.earlyColdMu.Unlock()
					}
				}

				isErr := metrics.Error != nil
				explicitCache := metrics.UsageData.CachedTokens.Count > 0

				earlyColdBaseline := st.earlyColdStartTTFT()

				st.mu.Lock()
				curConc := st.concurrency
				st.mu.Unlock()
				if curConc < 1 {
					curConc = 1
				}
				recentColdBaseline := st.stream.RecentColdTTFT(curConc)
				if recentColdBaseline == 0 {
					recentColdBaseline = coldStartTTFT
				}

				var implicitCache bool
				if recentColdBaseline > 0 {
					hitThresh := time.Duration(float64(recentColdBaseline) * cfg.TTFTHitThreshold)
					implicitCache = !isCold && metrics.TimeToFirstToken > 0 &&
						metrics.TimeToFirstToken <= hitThresh
				}

				ttftDegraded := earlyColdBaseline > 0 && metrics.TimeToFirstToken > 0 &&
					metrics.TimeToFirstToken >= earlyColdBaseline*time.Duration(cfg.TTFTDegradationFactor)
				if ttftDegraded {
					st.ttftDegradedCount.Add(1)
				}

				cacheHit := !isCold && !ttftDegraded && implicitCache
				if cacheHit && metrics.TimeToFirstToken > 0 {
					st.earlyHitMu.Lock()
					if len(st.earlyHitTTFTs) < maxEarlyHit {
						st.earlyHitTTFTs = append(st.earlyHitTTFTs, metrics.TimeToFirstToken)
					}
					st.earlyHitMu.Unlock()
				}
				serverCacheConfirmed := explicitCache

				// Write per-request data if configured.
				if rdw != nil {
					reqEnd := time.Now()
					reqStart := reqEnd.Add(-metrics.TotalResponseTime)
					errMsg := ""
					if metrics.Error != nil {
						errMsg = metrics.Error.Error()
					}
					rec := requestDataRecord{
						StartTime:            reqStart,
						EndTime:              reqEnd,
						TTFT:                 float64(metrics.TimeToFirstToken.Milliseconds()),
						ResponseMs:           float64(metrics.TotalResponseTime.Milliseconds()),
						Model:                cfg.Model,
						SeriesGUID:           seriesGUID,
						SeriesNum:            seriesNum,
						RequestNum:           requestNum,
						CacheHit:             cacheHit,
						ServerCacheConfirmed: serverCacheConfirmed,
						IsColdStart:          isCold,
						InputTokens:          metrics.UsageData.InputTokens.Count,
						OutputTokens:         metrics.UsageData.OutputTokens.Count,
						CachedTokens:         metrics.UsageData.CachedTokens.Count,
						IsError:              isErr,
						ErrorMessage:         errMsg,
						IsEmpty:              metrics.IsEmpty,
						LocalCacheRatio:      ratio,
					}
					if rec.IsError || rec.IsEmpty {
						rec.PromptText = metrics.CachedPrompt
						rec.Question = metrics.Question
						rec.ResponseText = metrics.Response
						rec.RawResponseTail = metrics.RawResponseTail
					}
					if writeErr := rdw.write(rec); writeErr != nil {
						fmt.Fprintf(os.Stderr, "warning: failed to write request data: %v\n", writeErr)
					}
				}

				st.stream.Add(completionRecord{
					completedAt:          time.Now(),
					ttft:                 metrics.TimeToFirstToken,
					isError:              isErr,
					isColdStart:          isCold,
					cacheHit:             cacheHit,
					serverCacheConfirmed: serverCacheConfirmed,
					inputTokens:          metrics.UsageData.InputTokens.Count,
					outputTokens:         metrics.UsageData.OutputTokens.Count,
					cachedTokens:         metrics.UsageData.CachedTokens.Count,
					localCacheRatio:      ratio,
				})

				st.totalCompleted.Add(1)
				if isErr {
					st.totalErrors.Add(1)
					st.consecutiveFailures.Add(1)
					if cfg.PrintErrorsThreshold > 0 && metrics.Error != nil {
						nowNs := time.Now().UnixNano()
						last := st.lastErrorPrintNs.Load()
						if nowNs-last >= int64(cfg.PrintErrorsThreshold) &&
							st.lastErrorPrintNs.CompareAndSwap(last, nowNs) {
							fmt.Fprintf(os.Stderr, "[%s] error: %v\n",
								shortModelName(cfg.Model), metrics.Error)
						}
					}
				} else {
					st.consecutiveFailures.Store(0)
				}
			}
		}()
	}

	for i := 0; i < cfg.StartSeries; i++ {
		spawnSeries(uuid.New().String(), i+1)
	}

	// Replay-mode drain watcher: when every worker has exited (queue drained
	// or context cancelled) signal termChan so the main select unblocks with a
	// clean termReasonReplayDone. Runs independently of the evaluator.
	if st.replay != nil {
		go func() {
			seriesWg.Wait()
			// Only surface a clean "replay done" if we actually finished the
			// queue — if the context was cancelled (SIGINT / timeout) the main
			// select already has the right termReason.
			select {
			case <-benchCtx.Done():
				return
			default:
			}
			if st.replay.Remaining() == 0 {
				select {
				case termChan <- termReasonReplayDone:
				default:
				}
			}
		}()
	}
	// Same drain watcher for tree-aware router replay.
	if st.routerReplay != nil {
		go func() {
			seriesWg.Wait()
			select {
			case <-benchCtx.Done():
				return
			default:
			}
			if st.routerReplay.Remaining() == 0 {
				select {
				case termChan <- termReasonReplayDone:
				default:
				}
			}
		}()
	}

	// ── Evaluator goroutine ──────────────────────────────────────────────────
	evaluatorDone := make(chan struct{})
	var lastVerboseHitRate float64 = -1
	var collapseStart time.Time

	doEvaluate := func() bool {
		total := st.totalCompleted.Load()

		// Always push display updates, even during warmup.
		updateSnap(st)

		// Fail-fast: abort once TOTAL errors reach the configured ceiling.
		// Unlike consecutiveFailures (reset on success), totalErrors is
		// monotonic, so this catches a single intermittent error even when
		// surrounded by successes. Placed before warmup/replay gates so it
		// fires ASAP. Does not touch the LLM server — it stays up for replay.
		if cfg.MaxTotalErrors > 0 && benchCtx.Err() == nil &&
			st.totalErrors.Load() >= int64(cfg.MaxTotalErrors) {
			termReason = termReasonError
			select {
			case termChan <- termReason:
			default:
			}
			return true
		}

		// Replay mode: exit once the queue drains and every worker has reported
		// its last conversation as completed. Without this check the evaluator
		// would keep ticking until --timeout — the drain watcher delivers a
		// termReason but can't close evaluatorDone from the outside.
		if st.replay != nil && st.replay.Remaining() == 0 &&
			st.seriesReplayCompleted.Load() >= int64(st.replay.Total()) {
			termReason = termReasonReplayDone
			select {
			case termChan <- termReason:
			default:
			}
			return true
		}

		// Same drain check for tree-aware router replay: the drain watcher
		// signals termChan but cannot close evaluatorDone, so the evaluator
		// must detect completion itself or it ticks until --timeout.
		if st.routerReplay != nil && st.routerReplay.Remaining() == 0 &&
			st.seriesReplayCompleted.Load() >= int64(st.routerReplay.Total()) {
			termReason = termReasonReplayDone
			select {
			case termChan <- termReason:
			default:
			}
			return true
		}

		// Optional early stop: once the queue is drained and the count of
		// still-active workers has dropped below the target concurrency, we
		// can no longer keep the gate saturated — remaining work is the long
		// tail of a few slow conversations. Cut the run to avoid skewing
		// throughput numbers with under-utilized time.
		if cfg.ReplayStopAtLowConcurrency && st.replay != nil && st.replay.Remaining() == 0 {
			active := st.activeReplayWorkers.Load()
			target := int64(cfg.Concurrency)
			if target <= 0 {
				// Auto-concurrency mode — use the hill-climber's current target.
				st.mu.Lock()
				target = int64(st.concurrency)
				st.mu.Unlock()
			}
			if active > 0 && target > 0 && active < target {
				termReason = termReasonReplayLowConcurrency
				select {
				case termChan <- termReason:
				default:
				}
				return true
			}
		}

		// Check --total limit
		if cfg.Total > 0 && total >= int64(cfg.Total) {
			termReason = termReasonTotal
			select {
			case termChan <- termReason:
			default:
			}
			return true
		}

		if total < int64(cfg.MinEvalRequests) {
			return false
		}

		{
			st.mu.Lock()
			curConc := st.concurrency
			st.mu.Unlock()
			windowCount := maxInt(curConc*4, cfg.MinStabilization)
			tm := st.stream.ThroughputMetricsByCount(windowCount)
			if tm.reqPerSec > 0 {
				st.mu.Lock()
				if tm.reqPerSec > st.allTimePeakRPS {
					st.allTimePeakRPS = tm.reqPerSec
					st.allTimePeakConc = st.concurrency
					st.allTimePeakSeries = st.series
				}
				st.mu.Unlock()
			}
		}

		// Simple abort: N consecutive failures. Counter is reset on any success
		// so transient blips don't accumulate. Default 512 is high enough that
		// a temporarily overloaded server recovers before we bail.
		if cfg.MaxConsecutiveFailures > 0 && benchCtx.Err() == nil &&
			st.consecutiveFailures.Load() >= int64(cfg.MaxConsecutiveFailures) {
			termReason = termReasonError
			select {
			case termChan <- termReason:
			default:
			}
			return true
		}

		st.mu.Lock()
		series := st.series
		concurrency := st.concurrency
		seriesDone := st.seriesDone
		st.mu.Unlock()

		// Count-based window tied to concurrency — stays recent regardless of series count.
		cm2 := st.stream.CacheMetrics(cfg.CacheWindowSize, cfg.MinStabilization)
		hitRate := cm2.hitRate

		if cfg.VerboseCache && cm2.count > 0 && math.Abs(hitRate-lastVerboseHitRate) >= 0.03 {
			lastVerboseHitRate = hitRate
			frozenBaseline := st.earlyColdStartTTFT()
			st.mu.Lock()
			printConc := st.concurrency
			st.mu.Unlock()
			if printConc < 1 {
				printConc = 1
			}
			recentCold := st.stream.RecentColdTTFT(printConc)
			threshold := time.Duration(float64(recentCold) * cfg.TTFTHitThreshold)
			missP50, missP95 := st.stream.MissTTFTStats(cm2.count)
			fmt.Fprintf(os.Stderr, "[CACHE][%s] hit=%.1f%% recent_cold=%s thresh=%s(%.0f%%) frozen=%s miss_p50=%s miss_p95=%s\n",
				shortModelName(cfg.Model),
				hitRate*100, formatDur(recentCold), formatDur(threshold), cfg.TTFTHitThreshold*100,
				formatDur(frozenBaseline), formatDur(missP50), formatDur(missP95))
		}

		if !seriesDone {
			// When --global-cache-hit-rate-target is set it replaces the windowed TTFT hit rate
			// as the series-scaling trigger (purely local, works correctly with --step mode).
			// Otherwise fall back to the windowed TTFT-based rate.
			scaleOK := false
			if cfg.GlobalCacheHitRateTarget > 0 {
				scaleOK = st.stream.GlobalLocalCacheRate() >= cfg.GlobalCacheHitRateTarget
			} else {
				scaleOK = hitRate >= cfg.CacheTarget
			}
			if scaleOK && total >= st.seriesEvalAfter {
				delta := series / 10
				if delta < 1 {
					delta = 1
				}
				// Cap: never add more than half the current concurrency at once.
				if concHalf := concurrency / 2; concHalf > 0 && delta > concHalf {
					delta = concHalf
				}
				newSeries := series + delta
				if newSeries > cfg.MaxSeries {
					newSeries = cfg.MaxSeries
				}
				if newSeries > series {
					st.mu.Lock()
					st.series = newSeries
					// Cooldown: wait for at least max(delta*2, concurrency) more completions
					// before the next series scale. Newly-spawned goroutines need time to
					// start, acquire a gate slot, and complete their cold start. Using
					// completion count (like concEvalAfter) avoids goroutine scheduling races.
					st.seriesEvalAfter = total + int64(maxInt(delta*2, concurrency))
					if newSeries >= cfg.MaxSeries {
						st.seriesDone = true
					}
					st.mu.Unlock()
					for i := 0; i < newSeries-series; i++ {
						spawnSeries(uuid.New().String(), series+i+1)
					}
					st.stream.UpdateMaxKeep(maxInt(cfg.CacheWindowSize*2, 200))
				} else {
					st.mu.Lock()
					st.seriesDone = true
					st.mu.Unlock()
				}
				updateSnap(st)
				return false
			}
		}
		if cfg.Concurrency == 0 && total >= st.concEvalAfter {
			st.mu.Lock()
			concurrency = st.concurrency
			st.mu.Unlock()

			windowCount := maxInt(concurrency*4, cfg.MinStabilization)
			tm := st.stream.ThroughputMetricsByCount(windowCount)
			currentRPS := tm.reqPerSec

			earlyTTFT := st.earlyHitTTFT()
			ttftWindow := maxInt(concurrency*4, cfg.MinStabilization)
			recentTTFT := st.stream.RecentHitTTFT(ttftWindow)
			ttftCeilingHit := earlyTTFT > 0 && recentTTFT > 0 && recentTTFT >= earlyTTFT*3

			delta := int(math.Round(float64(concurrency) * cfg.ScaleFactor))
			if delta < 1 {
				delta = 1
			}
			if cfg.MaxScaleStep > 0 && delta > cfg.MaxScaleStep {
				delta = cfg.MaxScaleStep
			}

			st.mu.Lock()
			newConc := concurrency
			if st.lastEvalRPS == 0 {
				st.lastEvalRPS = currentRPS
				st.mu.Unlock()
				updateSnap(st)
				st.mu.Lock()
				st.concEvalAfter = total + int64(maxInt(concurrency*4, cfg.MinStabilization))
				st.mu.Unlock()
				return false
			}

			improved := currentRPS >= st.lastEvalRPS*0.98
			dropped := currentRPS < st.lastEvalRPS*0.95

			switch {
			case improved && !ttftCeilingHit:
				newConc = concurrency + delta
				st.climbDirection = +1
				st.stallCount = 0
			case dropped:
				newConc = concurrency - delta
				st.climbDirection = -1
				st.stallCount++
			default:
				st.stallCount++
				if st.stallCount >= 3 && st.allTimePeakConc > 0 && st.allTimePeakConc != concurrency {
					gap := st.allTimePeakConc - concurrency
					jump := gap / 2
					if jump == 0 {
						if gap > 0 {
							jump = 1
						} else {
							jump = -1
						}
					}
					newConc = concurrency + jump
					st.climbDirection = jump / abs(jump)
					st.stallCount = 0
				}
			}

			if newConc < 1 {
				newConc = 1
			}
			if cfg.MaxConcurrency > 0 && newConc > cfg.MaxConcurrency {
				newConc = cfg.MaxConcurrency
			}

			st.concurrency = newConc
			st.lastEvalRPS = currentRPS
			st.concEvalAfter = total + int64(maxInt(newConc*4, cfg.MinStabilization))
			st.mu.Unlock()

			if newConc != concurrency {
				st.gate.SetLimit(newConc)
			}
			updateSnap(st)
			return false
		}

		// Collapse detector: abort if windowed cache hit rate stays below 50%
		// for 2 minutes. Off by default — this heuristic fires on legitimate
		// low-reuse workloads (e.g. replay across many distinct conversations
		// or any single-turn traffic) and has no bearing on whether the server
		// is actually misbehaving. Still suppressed when --timeout is set,
		// since an explicit timeout already bounds the run.
		if cfg.AbortOnCollapse && cfg.Timeout == 0 {
			if hitRate < 0.50 {
				if collapseStart.IsZero() {
					collapseStart = time.Now()
				} else if time.Since(collapseStart) >= 2*time.Minute && benchCtx.Err() == nil {
					termReason = termReasonCollapse
					select {
					case termChan <- termReason:
					default:
					}
					return true
				}
			} else {
				collapseStart = time.Time{}
			}
		}

		return false
	}

	go func() {
		defer close(evaluatorDone)
		evalTicker := time.NewTicker(1 * time.Second)
		defer evalTicker.Stop()
		for {
			select {
			case <-benchCtx.Done():
				return
			case <-evalTicker.C:
				if doEvaluate() {
					return
				}
			}
		}
	}()

	// Wait for termination (termChan signal or context cancel).
	select {
	case termReason = <-termChan:
		// termination already set; benchCtx may still be running (other models).
	case <-benchCtx.Done():
		if benchCtx.Err() == context.DeadlineExceeded {
			termReason = termReasonTimeout
		} else {
			termReason = termReasonSignal
		}
	}

	<-evaluatorDone
	if !waitWithTimeout(&seriesWg, 10*time.Second) {
		fmt.Fprintf(os.Stderr, "[auto][%s] series goroutines did not drain within 10s; proceeding\n", shortModelName(cfg.Model))
	}

	cm := st.stream.CacheMetrics(cfg.CacheWindowSize, cfg.MinStabilization)
	tm := st.stream.ThroughputMetricsByCount(cfg.CacheWindowSize)

	st.mu.Lock()
	res := autoBenchmarkResult{
		termReason:        termReason,
		optimalSeries:     st.series,
		optimalConc:       st.concurrency,
		allTimePeakRPS:    st.allTimePeakRPS,
		allTimePeakConc:   st.allTimePeakConc,
		allTimePeakSeries: st.allTimePeakSeries,
		cacheHitRate:      cm.DisplayHitRate(),
		inputTokPerSec:    tm.inputTokPerSec,
		outputTokPerSec:   tm.outputTokPerSec,
		cacheWarning:      st.cacheWarning,
	}
	st.mu.Unlock()

	res.elapsed = time.Since(startTime)
	res.totalCompleted = st.totalCompleted.Load()
	res.totalErrors = st.totalErrors.Load()
	res.ttftP50 = tm.ttftP50
	res.ttftP95 = tm.ttftP95
	tt := st.stream.TokenTotals()
	res.totalInputCold = tt.inputCold
	res.totalInputWarm = tt.inputWarm
	res.totalOutput = tt.output
	res.totalCachedTokens = tt.cached

	// Send a final snapshot with termReason set so multi-model display shows DONE.
	{
		st.mu.Lock()
		series := st.series
		concurrency := st.concurrency
		cacheWarning := st.cacheWarning
		allTimePeakRPS := st.allTimePeakRPS
		allTimePeakConc := st.allTimePeakConc
		allTimePeakSeries := st.allTimePeakSeries
		st.mu.Unlock()
		total := st.totalCompleted.Load()
		errors := st.totalErrors.Load()
		gateActive, gateColdWaiting, gateNormalWait := st.gate.GateStats()
		gateHotActive := 0
		if st.hotGate != nil {
			ha, _, _ := st.hotGate.GateStats()
			gateHotActive = ha
			gateActive += ha
		}
		finalSnap := &displaySnapshot{
			series:            series,
			concurrency:       concurrency,
			seriesStatus:      "✓",
			concStatus:        "",
			cacheHitRate:      cm.DisplayHitRate(),
			inputTokPerSec:    tm.inputTokPerSec,
			outputTokPerSec:   tm.outputTokPerSec,
			reqPerSec:         tm.reqPerSec,
			allTimePeakRPS:    allTimePeakRPS,
			allTimePeakConc:   allTimePeakConc,
			allTimePeakSeries: allTimePeakSeries,
			ttftP50:           tm.ttftP50,
			ttftP95:           tm.ttftP95,
			ttftCutoff:        st.earlyColdStartTTFT() * time.Duration(cfg.TTFTDegradationFactor),
			ttftDegradedCount: st.ttftDegradedCount.Load(),
			totalCompleted:    total,
			totalErrors:       errors,
			elapsed:           res.elapsed,
			cacheWarning:      cacheWarning,
			bothDone:          true,
			termReason:        termReason.String(),
			gateActive:        gateActive,
			gateColdWaiting:   gateColdWaiting,
			gateNormalWait:    gateNormalWait,
			gateHotActive:     gateHotActive,
		}
		if st.replay != nil {
			finalSnap.replayTotal = st.replay.Total()
			finalSnap.replayCompleted = st.seriesReplayCompleted.Load()
			finalSnap.replayRemaining = st.replay.Remaining()
			finalSnap.replayActive = st.activeReplayWorkers.Load()
		}
		if st.routerReplay != nil {
			finalSnap.replayTotal = st.routerReplay.Total()
			finalSnap.replayCompleted = st.seriesReplayCompleted.Load()
			finalSnap.replayRemaining = st.routerReplay.Remaining()
			finalSnap.replayActive = st.activeReplayWorkers.Load()
		}
		// Populate token totals on the DONE snap (previously left zero, which
		// showed up as in=0 warm=0 out=0 on the final rendered line).
		finalSnap.totalInput = res.totalInputCold + res.totalInputWarm
		finalSnap.totalInputWarm = res.totalInputWarm
		finalSnap.totalCached = res.totalCachedTokens
		finalSnap.totalOutput = res.totalOutput
		finalSnap.globalLocalCacheRate = st.stream.GlobalLocalCacheRate()
		sendSnap(finalSnap)
	}

	return res
}

// RunAutoBenchmark executes the auto-scaling benchmark.
//
// Supports one or more models via cfg.Models (or single cfg.Model for backwards compatibility).
// Multiple models run concurrently; each gets its own autoState, evaluator, and series goroutines.
// A single shared display goroutine renders all models side by side.
func RunAutoBenchmark(ctx context.Context, cfg AutoBenchmarkConfig) error {
	// Resolve models.
	models := cfg.Models
	if len(models) == 0 && cfg.Model != "" {
		models = []string{cfg.Model}
	}
	if len(models) == 0 {
		return fmt.Errorf("no model specified")
	}

	// Apply defaults once (passed through to runSingleModelBenchmark via per-model cfg copy).
	if cfg.MaxSeries <= 0 {
		cfg.MaxSeries = 64
	}
	if cfg.StartSeries <= 0 {
		cfg.StartSeries = 1
	}
	if cfg.StartSeries > cfg.MaxSeries {
		cfg.StartSeries = cfg.MaxSeries
	}
	if cfg.MinEvalRequests <= 0 {
		cfg.MinEvalRequests = 10
	}
	// When starting with multiple series, ensure the evaluator waits for all
	// initial cold starts to complete plus at least one warm request each
	// before making any scaling or termination decisions.
	if warmup := cfg.StartSeries * 2; warmup > cfg.MinEvalRequests {
		cfg.MinEvalRequests = warmup
	}
	if cfg.CacheTarget <= 0 {
		cfg.CacheTarget = 0.90
	}
	if cfg.CacheWindowSize <= 0 {
		cfg.CacheWindowSize = 20
	}
	if cfg.ScaleWaitFactor <= 0 {
		cfg.ScaleWaitFactor = 2
	}
	if cfg.MinStabilization <= 0 {
		cfg.MinStabilization = 20
	}
	if cfg.TTFTDegradationFactor <= 0 {
		cfg.TTFTDegradationFactor = 4
	}
	if cfg.TTFTHitThreshold <= 0 {
		cfg.TTFTHitThreshold = 0.5
	}
	if cfg.ScaleFactor <= 0 {
		cfg.ScaleFactor = 0.20
	}
	if cfg.MaxScaleStep <= 0 {
		cfg.MaxScaleStep = 8
	}
	if cfg.ErrorRateLimit <= 0 {
		cfg.ErrorRateLimit = 0.10
	}
	if cfg.MaxConsecutiveFailures <= 0 {
		cfg.MaxConsecutiveFailures = 512
	}

	// Apply optional timeout.
	if cfg.Timeout > 0 {
		var cancelTimeout context.CancelFunc
		ctx, cancelTimeout = context.WithTimeout(ctx, cfg.Timeout)
		defer cancelTimeout()
	}

	// Load documentation once. Used by synthetic mode for prefix-growth and
	// by tree-aware router replay for sizing per-request input. Hermes-style
	// replay (cfg.FromDataset != "") doesn't need it — conversations carry
	// their own content.
	var fullDocs string
	if cfg.FromDataset == "" {
		fmt.Println("Loading documentation...")
		if cfg.DocsContent != "" {
			fullDocs = cfg.DocsContent
		} else {
			var err error
			fullDocs, err = readDirectoryContents(cfg.DocsDir)
			if err != nil {
				return fmt.Errorf("failed to read docs directory: %w", err)
			}
		}
		fmt.Printf("Documentation loaded (%d bytes). Starting auto benchmark...\n\n", len(fullDocs))
	} else {
		if !cfg.ReplayNoStamp && cfg.RunID == "" {
			cfg.RunID = uuid.New().String()
		}
		fmt.Printf("Replay mode: dataset=%s replay-series=%d run-id=%s. Loading dataset...\n",
			cfg.FromDataset, cfg.ReplaySeries, cfg.RunID)
		convs, err := LoadConversationDataset(ctx, cfg.FromDataset, cfg.ReplaySeries)
		if err != nil {
			return fmt.Errorf("failed to load dataset %q: %w", cfg.FromDataset, err)
		}
		cfg.replayConversations = convs
		fmt.Printf("Loaded %d conversations. Starting auto benchmark...\n\n", len(convs))
	}

	// Tree-aware router replay: only the header (line 1) is read here so
	// we can show the summary up front. Sessions are streamed by each
	// model's runSingleModelBenchmark via its own stream over the file.
	if cfg.RouterReplayFile != "" {
		if !cfg.ReplayNoStamp && cfg.RunID == "" {
			cfg.RunID = uuid.New().String()
		}
		hdr, err := readRouterReplayHeader(cfg.RouterReplayFile)
		if err != nil {
			return fmt.Errorf("read router replay header %q: %w", cfg.RouterReplayFile, err)
		}
		cfg.routerReplayHeader = hdr
		fmt.Printf("Router-replay mode: file=%s run-id=%s\n",
			cfg.RouterReplayFile, cfg.RunID)
		fmt.Printf("Header: %d sessions / %d instances / %d requests / %d fan-out turns / max fan-out %d\n\n",
			hdr.Summary.Sessions, hdr.Summary.Instances, hdr.Summary.Requests,
			hdr.Summary.FanOutTurns, hdr.Summary.MaxFanOutInOneTurn)
	}

	// Create per-run subdirectory for request data if configured.
	if cfg.SaveRequestDataDir != "" {
		runDir := fmt.Sprintf("%s/%s", cfg.SaveRequestDataDir, time.Now().UTC().Format("2006-01-02T15-04-05Z"))
		if err := os.MkdirAll(runDir, 0755); err != nil {
			return fmt.Errorf("create run directory: %w", err)
		}
		cfg.SaveRequestDataDir = runDir
		fmt.Printf("Request data will be saved to: %s\n", runDir)
	}

	// Cancellable context for all benchmark goroutines.
	benchCtx, benchCancel := context.WithCancel(ctx)
	defer benchCancel()

	// SIGINT → cancel all models.
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigChan)
	go func() {
		select {
		case <-sigChan:
			benchCancel()
		case <-benchCtx.Done():
		}
	}()

	// Channel for snapshot updates from all model goroutines.
	snapCh := make(chan modelSnapshotUpdate, len(models)*8)

	// Shared display state.
	display := newDisplayState()

	// Launch one goroutine per model.
	results := make([]modelBenchmarkResult, len(models))
	var wg sync.WaitGroup
	for i, model := range models {
		modelCfg := cfg
		modelCfg.Model = model
		wg.Add(1)
		go func(idx int, mcfg AutoBenchmarkConfig) {
			defer wg.Done()
			res := runSingleModelBenchmark(benchCtx, mcfg, fullDocs, snapCh, idx)
			results[idx] = modelBenchmarkResult{model: mcfg.Model, result: res}
		}(i, modelCfg)
	}

	// Display goroutine — runs until snapCh is closed.
	displayDone := make(chan struct{})
	go func() {
		defer close(displayDone)
		runMultiModelDisplay(benchCtx, snapCh, models, display)
	}()

	wg.Wait()
	benchCancel() // display exits via <-ctx.Done(); series workers exit too.
	// NOTE: do not close(snapCh). runSingleModelBenchmark uses waitWithTimeout
	// on its series WaitGroup, so orphan series goroutines can still be alive
	// when we get here (collapse / total / replay-done terminations don't
	// cancel benchCtx by themselves). Closing the channel under them would
	// race with their non-blocking sendSnap → "send on closed channel" panic.
	<-displayDone

	// Print summary.
	if len(models) == 1 {
		printAutoSummary(results[0].result, cfg)
	} else {
		printMultiModelSummary(results)
	}

	// Auto-generate HTML visualization if request data was saved.
	if cfg.SaveRequestDataDir != "" {
		htmlPath, err := GenerateVisualization(cfg.SaveRequestDataDir, cfg.Concurrency)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to generate HTML visualization: %v\n", err)
		} else {
			fmt.Printf("\nVisualization saved to: %s\n", htmlPath)
		}
	}

	return nil
}

// waitWithTimeout waits for wg with a timeout. Returns false if timed out.
func waitWithTimeout(wg *sync.WaitGroup, timeout time.Duration) bool {
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
		return true
	case <-time.After(timeout):
		return false
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
