package benchmark

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
)

// hotGateFanoutMultiplier scales the hot gate so router-replay sub-agent fan-out
// doesn't block hot sessions. For synthetic/hermes (linear streams) only H slots
// are ever used — the multiplier is harmless headroom.
const hotGateFanoutMultiplier = 8

// AutoBenchmarkConfig holds configuration for the auto benchmark mode.
type AutoBenchmarkConfig struct {
	DocsDir                   string
	DocsContent               string   // Pre-loaded documentation content (used instead of DocsDir when set)
	Model                     string   // single model (legacy); prefer Models
	Models                    []string // multi-model; takes priority over Model
	Question                  string
	Timeout                   time.Duration
	MaxSeries                 int           // safety cap, default 64
	StartSeries               int           // initial series count, default 1
	MaxConcurrency            int           // 0 = unlimited
	MinEvalRequests           int           // minimum total completed before any eval, default 10
	CacheTarget               float64       // cache hit rate threshold at which series scaling begins, default 0.90
	CacheWindowSize           int           // fixed number of recent requests for cache hit measurement, default 20
	ScaleWaitFactor           int           // fast measurement window factor for backoff recovery detection (series × this), default 2
	MinStabilization          int           // minimum requests for any stabilization window, default 20
	ScaleFactor               float64       // fractional scale step (0.20 = 20%), default 0.20
	MaxScaleStep              int           // maximum delta per scale event, default 8
	ErrorRateLimit            float64       // DEPRECATED: no-op, retained for flag compatibility (see MaxConsecutiveFailures)
	MaxConsecutiveFailures    int           // abort after this many failures in a row (any success resets the counter); default 512
	MaxTotalErrors            int           // abort after this many TOTAL errors (monotonic, not reset by success); 0 = disabled
	MinSeries                 int           // informational minimum (not used in algo), default 2
	TTFTDegradationFactor     int           // TTFT multiplier above early cold-start baseline that disqualifies the heuristic entirely (default 4)
	TTFTHitThreshold          float64       // fraction of cold-start baseline below which a request is classified as a cache hit (default 0.5 = 50%)
	Concurrency               int           // fixed concurrency; disables hill-climber entirely (0 = auto)
	VerboseCache              bool          // print TTFT baseline/threshold/miss distribution on hit-rate change; forces non-TTY (periodic) display mode
	PrintResponses            bool          // print each request/response to stdout; forces non-TTY display mode
	PrintErrorsThreshold      time.Duration // if >0, print at most one error to stderr per this interval (rate-limited)
	SaveRequestDataDir        string        // if non-empty, write per-request JSONL to this dir
	Total                     int           // stop after N total completed requests (0 = unlimited)
	HotSeriesConcurrency      int           // H of the --series workers run as a 'hot' pool with a dedicated gate; 0 = no hot pool.
	EndpointOverloadThreshold float64       // Multi-endpoint only: fail over off an endpoint once its in-flight exceeds this multiple of its fair share. 0 = default (1.5).
	RequestTimeout            time.Duration // per-request timeout (default 5m)
	Step                      int           // 0=disabled; token step size for prefix growth per request
	StepStartingTokens        int           // 0=start at Step; initial prefix token size for each series when Step > 0
	Tokens                    int           // upper bound on per-request prompt tokens (caps Step growth and DocsDir source). 0 = bounded only by len(fullDocs)/4.
	SharedPrefixPerSeries     int           // 0=disabled; N series share same doc prefix
	GlobalCacheHitRateTarget  float64       // 0=disabled; block new series until global hit rate >= this
	MaxOutputTokens           int           // 0=use model spec default; override max_tokens for all requests
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
	FromDataset  string // short name, e.g. "hermes-lambda"
	ReplaySeries int    // number of conversations to replay (0 = whole dataset)
	// ReplayMaxRequestsPerSession truncates each replayed session to its first
	// N requests in capture order (0 = off). A capture's sessions are wildly
	// uneven, so a run over one spends most of its time on a handful of them;
	// capping normalizes the corpus to the shape of its own median.
	ReplayMaxRequestsPerSession int
	LimitContext                int // --limit-context: skip requests over N tokens (~N*4 chars); 0=off
	// ReplayCharsPerToken: --replay-chars-per-token. When > 0, synthesized
	// replay content is sized off each block's captured Tokens count
	// (tokens * ReplayCharsPerToken chars) instead of its captured Bytes
	// count, so the serving tokenizer's counts land near the original
	// capture's. 0 = byte-faithful sizing (default).
	ReplayCharsPerToken float64
	// VLLMMetricsURLs overrides where upstream counters are scraped from.
	//
	// By default the sampler derives /metrics from the serving spec, which only
	// works when the thing serving inference is also the thing exposing the
	// counters. Behind a router it is not: the router refuses /metrics on the
	// serving port on purpose — proxying it would answer with ONE backend's
	// counters, which is worse than nothing because it looks like a fleet total
	// — and serves the aggregate on its metrics listener instead. Nothing in
	// the model spec names that address, so it has to be given.
	//
	// Either point this at the router's metrics listener (with the router run
	// with --vllm-metrics, which is what produces a real fleet aggregate) or at
	// every backend directly, in which case the per-endpoint delta accumulator
	// sums them correctly across restarts and departures.
	VLLMMetricsURLs []string

	// Real-time replay: sessions keep the think time they were captured with,
	// and load grows by admitting sessions under a TTFT governor rather than by
	// setting a concurrency number.
	ReplayRealtime bool
	AdmitEvery     time.Duration // 0 = off, no session admission governor
	TTFTLimit      time.Duration // gate closes at or above this windowed mean
	TTFTWindow     time.Duration // how far back the gate looks; default 30s
	// TTFTLimitStat is which statistic over that window the gate compares
	// against TTFTLimit: mean (default), p50, p95 or p99.
	//
	// It decides the answer rather than decorating it. TTFT here is heavily
	// skewed, so a mean gate closes at roughly half the session count a p50 gate
	// would. All four are reported whichever is chosen, so one run says what its
	// plateau would have been under each.
	TTFTLimitStat  string
	ReplaySkipIdle bool // compress dead time when the whole run is idle

	ReplayNoStamp   bool // when true, skip the per-run <ignore>RUN_GUID</ignore> prefix injection (default is to stamp so each run starts with a pristine server prefix cache while still permitting within-run cross-series cache hits)
	AbortOnCollapse bool // when true, abort if windowed cache hit rate < 50% for 2 minutes (legacy collapse detector, off by default — fires spuriously on legitimate low-reuse workloads)
	// ReplayStopAtLowConcurrency terminates the run when the queue is drained
	// AND active worker count < desired concurrency — a long-tail cutoff so
	// throughput numbers reflect steady-state behavior rather than the slow
	// tail of 1-2 surviving long conversations.
	//
	// Note: activeReplayWorkers includes hot-pool workers, so the cutoff is
	// slightly conservative when --hot-series-concurrency > 0 — the target
	// remains the normal budget C (--concurrency).
	ReplayStopAtLowConcurrency bool

	// ReplayAllowUnderfill lets a real-time run CONTINUE after the corpus runs
	// out and admitted slots can no longer be filled. Off by default, so the run
	// aborts.
	//
	// Aborting is the point. The governor admits slots and the queue supplies
	// sessions to fill them; when the queue is empty the slots stay admitted and
	// idle, so `slots` keeps reading N while the fleet is doing less than N
	// sessions' worth of work. Nothing else in the output moves — the number
	// stays authoritative while the thing beneath it decays, and totals
	// accumulated over that stretch are diluted by an amount nothing reveals.
	// Neither `late` nor `fidelity` catches it either: sessions that never start
	// are not late.
	ReplayAllowUnderfill bool

	// ReplayReuseSessions replays the corpus again from the top instead of
	// draining, so a run can outlast its own dataset.
	//
	// Each pass gets its OWN stamp, applied uniformly across the whole pass.
	// That is the property the whole thing turns on: within a pass, two sessions
	// that shared a system prompt in the capture still share it, so the sharing
	// topology the benchmark exists to measure is identical to pass one's;
	// across passes the keyspace is disjoint, so pass two cannot trivially hit
	// pass one's cache entries and inflate the hit rate toward 100%.
	//
	// A per-SESSION stamp would satisfy the second property and destroy the
	// first, and it would still produce a cache hit rate — just a meaningless
	// one. That is why the stamp is per pass and never per session.
	ReplayReuseSessions bool
	// UUID-based cache-coherency validation (--verify). ROUTER
	// PATH ONLY (cfg.RouterReplayFile != ""); the CLI rejects this combined
	// with --from-dataset (see cli/benchmark_commands.go). One deterministic
	// UUID is injected per user turn, spread through the conversation
	// (rather than clustered at a session boundary) — see the package doc
	// in replay_router_uuid.go for the full design. Each request recites a
	// bounded WINDOW (first turn + up to 3 most-recent turns, excluding the
	// current turn, capped at 4), keeping the recite cost/response budget
	// constant regardless of session length.
	Verify bool
	// Seed seeds the UUID generator (see newUUIDGenerator); 0 = crypto/rand
	// (non-deterministic across runs).
	Seed int64
	// VerifyReciteEvery: ask the model to recite the first-line UUID
	// window on EVERY request (default true), not just each instance's
	// final request.
	// uuidRegistry is the live marker set for --verify, shared
	// by every poster in the run. Built at run start and populated as
	// sessions are dispatched — there is no precompute phase; see
	// replay_uuid_registry.go for why none is needed and what the registry's
	// detection window is.
	uuidRegistry *uuidRegistry
	// VerifyContinueOnContamination keeps the run going after a leaked
	// marker. Off by default: see the evaluator's contamination check.
	VerifyContinueOnContamination bool
	// DumpDir/DumpLimit drive the verbatim exchange capture; see
	// replay_dump.go. dumper is built once and shared by every poster.
	DumpDir   string
	DumpLimit int
	// DumpGarbage keeps the default-on garbage-only capture (--verify runs
	// only); DumpGarbageDir names its directory instead of an mktemp one.
	// --dump-dir supersedes both: it already writes every exchange, garbage
	// among them, and two captures of the same bytes help nobody.
	DumpGarbage    bool
	DumpGarbageDir string
	dumper         *requestDumper

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

	// ReplayOutputRatio (--replay-output-ratio), when > 0, retargets each
	// router-replay request's max_tokens to round(InputTokens * ReplayOutputRatio)
	// instead of the recorded original output_tokens (which otherwise pins
	// max_tokens to what the model produced in the ORIGINAL capture, so the
	// model stops almost immediately on replay — a ~500:1 input:output run).
	// 0 = off (original precedence: output_tokens, then max_tokens, then 1).
	ReplayOutputRatio float64
	// ReplayMinOutputTokens (--replay-min-output-tokens) floors every replayed
	// request's max_tokens. See pickMaxTokens.
	ReplayMinOutputTokens int
	// VerifyForceEOS keeps ignore_eos on under --verify. See forceVolume for
	// the rule it feeds.
	VerifyForceEOS bool

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
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	// TTFT and ResponseMs are both CALLER-experienced: every 429 the client
	// backed off is inside them. There is no retry-excluded variant to compare
	// against, deliberately — reporting retries as their own dimension invites a
	// reader to subtract them back out, and a request that took seventeen
	// attempts did not have a fast first token.
	TTFT       float64 `json:"ttft_ms"`          // milliseconds
	ResponseMs float64 `json:"response_time_ms"` // milliseconds

	// Identity
	Model      string `json:"model"`
	SeriesGUID string `json:"series_guid"`
	SeriesNum  int    `json:"series_num"`
	RequestNum int    `json:"request_num"`
	// Turn is this request's 1-based position within its own instance, which
	// RequestNum is not — that one counts the whole run. Carried because the
	// summary locates a failure as series:turn:length, and without it that
	// triple cannot be joined back to the rows it came from.
	Turn int `json:"turn"`

	// Cache status
	CacheHit bool `json:"cache_hit"` // server-reported where available, else the TTFT heuristic
	// CacheHitRatio is the share of prompt tokens the SERVER said it reused. A
	// partial prefix hit is the normal case on agentic traffic and is neither a
	// hit nor a miss, so the boolean above loses what this keeps. 1.0 where only
	// the TTFT heuristic was available and it fired.
	CacheHitRatio        float64 `json:"cache_hit_ratio"`
	ServerCacheConfirmed bool    `json:"server_cache_confirmed"` // explicit only
	IsColdStart          bool    `json:"is_cold_start"`

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

	// UUID validation (router-replay --verify only). The three
	// counts are always populated (0 when the feature is off); the raw detail
	// lists are populated ONLY on a miss or a leak (mirrors the
	// failed-request-only policy above — avoid bloating every row).
	UUIDExpected     int      `json:"uuid_expected"`
	UUIDFound        int      `json:"uuid_found"`
	UUIDLeaked       int      `json:"uuid_leaked"`
	UUIDExactMatch   bool     `json:"uuid_exact_match"`
	ExpectedUUIDsRaw []string `json:"expected_uuids_raw,omitempty"`
	FoundMask        []bool   `json:"found_mask,omitempty"`
	LeakedUUIDsRaw   []string `json:"leaked_uuids_raw,omitempty"`
}

// requestDataWriter writes requestDataRecord entries as JSONL, safe for concurrent use.
type requestDataWriter struct {
	mu     sync.Mutex
	f      *os.File
	enc    *json.Encoder
	closed bool
}

var sanitizeModelRe = regexp.MustCompile(`[^a-zA-Z0-9._-]`)

// maxFileNameLen is the per-component file-name limit these results land on.
// ext4, xfs, overlayfs and NFS all cap a single name at 255 BYTES — the limit
// is on the component, not the path, so a shorter output dir does not help.
const maxFileNameLen = 255

// safeFileBase turns an arbitrary identifier into a file-name base guaranteed
// to fit, reserving `reserve` bytes for the extension and any suffix the caller
// appends (e.g. "_part12.csv").
//
// A multi-endpoint dynamic model spec embeds EVERY endpoint URL alongside the
// model name and the alias. A six-endpoint failover run produced a 260-byte
// name, os.Create failed with ENAMETOOLONG, and the run wrote no request data
// at all — a 12-hour benchmark reduced to a single stderr warning. Four
// characters of alias separated the run that worked from the one that didn't.
//
// When shortening is needed the ALIAS is kept in preference to the endpoint
// list — it is the part a human named the run by, and the part every reader
// identifies the arm by — with a hash of the FULL identifier appended so two
// specs differing only past the cut can never collide on one file. Names that
// already fit are returned unchanged, so existing runs keep their filenames.
func safeFileBase(id string, reserve int) string {
	// Sanitizing leaves only [a-zA-Z0-9._-], so the result is pure ASCII and
	// slicing by byte below cannot split a rune.
	safe := sanitizeModelRe.ReplaceAllString(id, "_")
	limit := maxFileNameLen - reserve
	if limit < 24 {
		limit = 24 // always leave room for a usable head plus the hash
	}
	if len(safe) <= limit {
		return safe
	}
	sum := sha256.Sum256([]byte(id))
	suffix := "_" + hex.EncodeToString(sum[:])[:12]
	head := safe
	if alias := extractAlias(id); alias != "" {
		head = sanitizeModelRe.ReplaceAllString(alias, "_")
	}
	if len(head)+len(suffix) > limit {
		head = head[:limit-len(suffix)]
	}
	return head + suffix
}

func newRequestDataWriter(outputDir, model string, _ time.Time) (*requestDataWriter, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("create output dir: %w", err)
	}
	filename := safeFileBase(model, len(".jsonl")) + ".jsonl"
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

// errorAbortEarned reports whether the run's error counters have already
// crossed a configured ceiling. The evaluator calls it before every other gate.
//
// The two ceilings answer different questions. MaxTotalErrors is monotonic, so
// it catches a single intermittent error even when surrounded by successes;
// MaxConsecutiveFailures resets on any success, so transient blips do not
// accumulate and its default of 512 lets a temporarily overloaded fleet
// recover first.
//
// It deliberately cannot see totalCompleted, which is what the evaluator's
// warmup gate counts. That gate is StartSeries*2 completions — thousands of
// requests at a few thousand sessions — and a ceiling of 512 that cannot be
// reached until 16,000 requests have landed is not the ceiling it says it is.
// A fleet failing every request is at its most obviously broken during the
// ramp, which is exactly the window the gate covers.
func (st *autoState) errorAbortEarned(cfg AutoBenchmarkConfig) bool {
	if cfg.MaxTotalErrors > 0 && st.totalErrors.Load() >= int64(cfg.MaxTotalErrors) {
		return true
	}
	return cfg.MaxConsecutiveFailures > 0 &&
		st.consecutiveFailures.Load() >= int64(cfg.MaxConsecutiveFailures)
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
	termReasonReplayUnderfilled                          // the corpus ran out, so admitted slots could not be filled
	termReasonContamination                              // --verify found a leaked marker and the run did not opt out of stopping
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
	case termReasonReplayUnderfilled:
		return "Corpus exhausted: admitted slots could not be filled, so offered load had fallen below the governor's session count"
	case termReasonReplayLowConcurrency:
		return "Replay stopped: active workers fell below target concurrency"
	case termReasonContamination:
		return "Cross-contamination detected (--verify); stopped so the evidence stays fresh"
	}
	return "Unknown"
}

// autoState holds the mutable benchmark state shared between evaluator and series goroutines.
// realtimeUngatedConcurrency is the client gate under --replay-realtime with no
// explicit --concurrency: high enough to be out of the way, finite enough that a
// runaway meets a limit rather than the file-descriptor table.
const realtimeUngatedConcurrency = 100000

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

	// Real-time replay governor. ttft is what the admission gate reads; skipClk
	// is the shared clock the pacers wait against; lag records how far behind
	// their captured schedule requests are actually going out.
	ttft    *ttftWindow
	skipClk *skipClock
	lag     *pacingLag

	allTimePeakRPS    float64 // highest req/s ever seen (never reset)
	allTimePeakConc   int     // concurrency at which allTimePeakRPS was achieved
	allTimePeakSeries int     // series count at which allTimePeakRPS was achieved

	// Atomic counters (no lock needed for reads/writes)
	// dispatched is requests whose HTTP exchange is open: counted from the
	// round trip starting to the response body being closed. Distinct from the
	// gate's active count, which also covers building and uploading a body —
	// see inflight_transport.go.
	dispatched atomic.Int64

	totalCompleted atomic.Int64
	totalEmitted   atomic.Int64
	totalErrors    atomic.Int64

	// Client-side 429 backoff: how many sheds were waited out, and the total
	// time spent waiting. A router with a per-backend concurrency cap sheds by
	// design, so these say how much of the run's wall time was the client
	// queueing rather than the fleet working. Errors count only the sheds that
	// outlasted the whole retry budget.
	totalRetries429  atomic.Int64
	totalRetryWaitNs atomic.Int64

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

	// UUID validation (replay --verify only). All zero when the
	// feature is off — recordReplayRequest only touches these when
	// metrics.ExpectedUUIDs is non-empty.
	valReqs               atomic.Int64 // requests that carried >=1 expected UUID (i.e. validation ran)
	valUUIDChecks         atomic.Int64 // total per-UUID presence checks made
	valUUIDFound          atomic.Int64 // per-UUID presence checks that found the UUID
	valExactMatchReqs     atomic.Int64 // requests whose first line was the exact ordered UUID list (output conformity)
	valPresenceMissUUIDs  atomic.Int64 // per-UUID PRESENCE_MISS count (expected UUID absent)
	valCrossContamUUIDs   atomic.Int64 // per-UUID CROSS_CONTAMINATION count (other-conversation UUID present)
	valLeakCheckedReqs    atomic.Int64 // responses actually scanned for contamination — the denominator that zero belongs to
	valBudgetShortReqs    atomic.Int64 // requests whose captured output budget could not carry one id, so presence was never asked
	valMissSubstituted    atomic.Int64 // misses where the response carried another marker from this same prompt
	valMissAbsent         atomic.Int64 // misses with no such evidence either way
	valEchoedTagsReqs     atomic.Int64 // responses that repeated turn names instead of guids — an ASK defect
	valNoIDsReqs          atomic.Int64 // responses containing no guid at all — an ASK defect
	valGarbageReqs        atomic.Int64 // responses carrying decode-level corruption (U+FFFD, NUL, stray control chars)
	valGarbagePostEOS     atomic.Int64 // a literal stop token precedes the corruption — ignore_eos continuation
	valGarbageTail        atomic.Int64 // corruption runs to the end of the budget, no visible marker
	valGarbageGuidBabble  atomic.Int64 // the tail after corruption is invented uuid shapes
	valGarbageMidResponse atomic.Int64
	// valInstances holds the two signals that live between requests rather than
	// in one — back-to-back garbage and a miss the series never recovers
	// from. See replay_verify_series.go. Zero value is ready to use.
	valInstances instanceVerifyHistory
	// Output-profile conformity: the wire budgets and what actually came
	// back, summed over completed replay requests. actual/target is ~100%
	// under ignore_eos by construction; when prompt-based length control
	// replaces it, this ratio is the score.
	outTargetSum atomic.Int64
	outActualSum atomic.Int64 // none of the above — the class ignore_eos cannot explain
	// contaminationStop is armed by the first leaked marker unless the run
	// opted out; the evaluator turns it into a termination.
	contaminationStop   atomic.Bool
	valPresenceMissReqs atomic.Int64 // requests with >=1 PRESENCE_MISS
	valCrossContamReqs  atomic.Int64 // requests with >=1 CROSS_CONTAMINATION

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
	totalRetries429   int64
	totalRetryWait    time.Duration
	ttftP50           time.Duration
	ttftP95           time.Duration
	inputTokPerSec    float64
	outputTokPerSec   float64
	totalInputCold    int64 // input tokens from cold-start requests (had to prefill from scratch)
	totalInputWarm    int64 // input tokens from warm requests (prefix was cached)
	totalOutput       int64 // output tokens across all requests
	totalCachedTokens int64 // server-reported cached prompt tokens

	// UUID validation (replay --verify only); all zero when the
	// feature is off. See autoState's val* atomics for field meanings.
	valReqs               int64
	valUUIDChecks         int64
	valUUIDFound          int64
	valExactMatchReqs     int64
	valPresenceMissUUIDs  int64
	valCrossContamUUIDs   int64
	valLeakCheckedReqs    int64
	valBudgetShortReqs    int64
	valMissSubstituted    int64
	valMissAbsent         int64
	valEchoedTagsReqs     int64
	valNoIDsReqs          int64
	valGarbageReqs        int64
	valGarbagePostEOS     int64
	valGarbageTail        int64
	valGarbageGuidBabble  int64
	valGarbageMidResponse int64
	valInstances          instanceVerifyTotals
	outTargetSum          int64
	outActualSum          int64
	valWindowSessions     int
	valWindowMarkers      int
	valPresenceMissReqs   int64
	valCrossContamReqs    int64
}

// displaySnapshot is an atomic snapshot of state for the display goroutine.
type displaySnapshot struct {
	modelName            string // set for multi-model display; empty for single-model
	series               int
	concurrency          int
	seriesStatus         string
	concStatus           string
	cacheHitRate         float64
	globalLocalCacheRate float64
	// --verify live counters for the progress line. verifyOn distinguishes
	// "verification disabled" from "enabled but nothing scored yet", which
	// would otherwise both render as nothing.
	verifyOn      bool
	verifyChecks  int64
	verifyFound   int64
	verifyLeaked  int64
	verifyGarbage int64
	verifyAbsent  int64
	// Garbage by class, so a running comparison across fleets does not need
	// the summary or a scroll through per-event lines. Only mid-response is
	// an alarm; the others are the run's own ignore_eos at work.
	verifyGarbagePostEOS int64
	verifyGarbageTail    int64
	verifyGarbageBabble  int64
	verifyGarbageMid     int64
	// Output-profile conformity, replay mode: actual vs requested output
	// tokens. Rendered whenever a target exists — it is a replay property,
	// not a verify one.
	outTargetSum      int64
	outActualSum      int64
	reqPerSec         float64
	inputTokPerSec    float64
	outputTokPerSec   float64
	allTimePeakRPS    float64
	allTimePeakConc   int
	allTimePeakSeries int
	ttftP50           time.Duration
	ttftP95           time.Duration
	ttftCutoff        time.Duration // degradation cutoff = earlyColdBaseline * factor
	ttftDegradedCount int64         // requests disqualified by TTFT cutoff
	totalCompleted    int64
	totalErrors       int64
	totalRetries429   int64
	totalRetryWait    time.Duration
	elapsed           time.Duration
	cacheWarning      bool
	bothDone          bool
	termReason        string // non-empty when model benchmark has finished
	gateActive        int
	gateColdWaiting   int
	gateNormalWait    int
	gateHotActive     int // active slots in the hot-pool gate (non-zero only with hot-series-concurrency)
	// gateDispatched is requests whose HTTP exchange is actually open. gateActive
	// counts gate slots, which are held from before the body is built until after
	// the response is consumed, so the difference is the client's own prep and
	// upload — real work, in a real place, but not concurrency the fleet carried.
	gateDispatched int
	totalInput     int64 // total full-prompt input tokens (cold + warm)
	totalInputWarm int64 // warm full-prompt input tokens
	totalCached    int64 // server-reported cached prompt tokens (subset of totalInput)
	totalOutput    int64 // total output tokens

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
	if res.outTargetSum > 0 {
		// actual vs the budgets the run put on the wire. With ignore_eos on
		// (regular runs, or --verify-force-eos) this is ~100% by
		// construction; under --verify it scores how well the per-request
		// length ask alone reproduces the captured output profile.
		fmt.Printf(" Output conformity  : %s of %s requested (%s)\n",
			formatKiloInt(res.outActualSum), formatKiloInt(res.outTargetSum), pctOf(res.outActualSum, res.outTargetSum))
	}
	fmt.Println(strings.Repeat("-", 62))
	fmt.Printf(" TTFT p50           : %s\n", formatDur(res.ttftP50))
	fmt.Printf(" TTFT p95           : %s\n", formatDur(res.ttftP95))
	fmt.Println(strings.Repeat("-", 62))
	fmt.Printf(" Total completed    : %d\n", res.totalCompleted)
	fmt.Printf(" Total errors       : %d\n", res.totalErrors)
	if res.totalRetries429 > 0 {
		// A router enforcing a per-backend concurrency cap sheds on purpose, so
		// this is the run's cost of that policy: sheds waited out, and the wall
		// time the client spent queueing for them. Errors above count only the
		// sheds that outlasted the entire retry budget.
		mean := res.totalRetryWait / time.Duration(res.totalRetries429)
		fmt.Printf(" 429 backoff        : %d retries, %s total wait (mean %s/retry)\n",
			res.totalRetries429, formatDur(res.totalRetryWait), formatDur(mean))
	}
	if cfg.Verify {
		fmt.Println(strings.Repeat("-", 62))
		fmt.Println(" UUID validation (replay)")
		fmt.Printf("   Requests validated                   : %d\n", res.valReqs)
		// Two tests, mirroring the cache-coherency eval CLI's layout: UUID
		// correctness (per-stamp presence, Contains anywhere in the response)
		// and output conformity (first line is exactly the ordered,
		// comma-joined UUID list — see firstLineConformity).
		fmt.Printf("   UUID correctness (presence)          : %d/%d (%s)\n", res.valUUIDFound, res.valUUIDChecks, pctOf(res.valUUIDFound, res.valUUIDChecks))
		fmt.Printf("   Output conformity (leads with list)  : %d/%d (%s)\n", res.valExactMatchReqs, res.valReqs, pctOf(res.valExactMatchReqs, res.valReqs))
		fmt.Printf("   PRESENCE_MISS (expected UUID absent) : %d across %d requests\n", res.valPresenceMissUUIDs, res.valPresenceMissReqs)
		// Attributed, because the raw count sums unrelated causes. A
		// substitution proves the tagged content was there to read and the
		// model picked the wrong tag; only the unattributed remainder is
		// consistent with a loss, and even that shares its bucket with "the
		// model declined to answer".
		fmt.Printf("     wrong turn, from this prompt       : %d (content was present)\n", res.valMissSubstituted)
		fmt.Printf("     no evidence either way             : %d\n", res.valMissAbsent)
		// These two measure the ASK. Anything but near-zero and the presence
		// ratio above is describing the instruction, not the fleet.
		fmt.Printf("   Ask quality: responses echoing tags  : %d\n", res.valEchoedTagsReqs)
		fmt.Printf("   Ask quality: responses with no ids   : %d\n", res.valNoIDsReqs)
		// A verify run keeps ignore_eos OFF (see forceVolume), so nothing here
		// can be the one innocent thing garbage sometimes is: a model forced
		// to keep generating past its own stop token. The classes say WHERE
		// the corruption sat, which is what separates their causes, and the
		// post-EOS class is printed only if it happened at all — that takes
		// --verify-force-eos, and in any other run its zero was a line about
		// a mode the run was not in.
		fmt.Printf("   Garbage responses (decode corruption): %d\n", res.valGarbageReqs)
		if res.valGarbageReqs > 0 {
			if res.valGarbagePostEOS > 0 {
				fmt.Printf("     post-EOS (ignore_eos continuation) : %d\n", res.valGarbagePostEOS)
			}
			fmt.Printf("     tail garbage (runs to end)         : %d\n", res.valGarbageTail)
			fmt.Printf("     guid-babble tail                   : %d\n", res.valGarbageGuidBabble)
			fmt.Printf("     mid-response                       : %d\n", res.valGarbageMidResponse)
		}
		// Two shapes that exist only BETWEEN requests, printed beside the
		// per-request counts because the counts cannot hold them: 40 corrupt
		// responses one per session and 40 arriving in pairs are the same
		// number and two different fleets. Independent faults land adjacent
		// at the square of their rate, while a corrupted prefix stays
		// corrupted until the session ends — so consecutive garbage in one
		// series is the shape KV corruption makes and a flaky decode does
		// not. The same argument holds for a miss nothing later in the series
		// recovers from: a model that declines one request answers the next,
		// and a session that lost its context does not get it back.
		fmt.Printf("   Instances with back-to-back garbage  : %d of %d that produced any\n",
			res.valInstances.garbageRunInstances, res.valInstances.garbageInstances)
		// "asked to recite" and not "scored": the denominator is the series that
		// carried a marker the model was told to repeat, which is a population
		// a reader cannot infer and is far smaller than the request count. A
		// bare "scored" says neither what was measured nor over what.
		fmt.Printf("   Instances never recovering a miss    : %d of %d that missed\n",
			res.valInstances.missToEndInstances, res.valInstances.missInstances)
		// Named, not just counted. The count says the fleet has the fault; these
		// say which conversation to open, and the first two fields are the same
		// s/t coordinates the per-request error lines carry, so a run can be
		// grepped straight out of the log or the request-data JSONL.
		if runs := formatMissRuns(res.valInstances.missToEnd, maxReportedMissRuns); runs != "" {
			fmt.Printf("     series:turn:length                 : %s\n", runs)
		}
		// The capped list above points at nothing once it says "and N more", so
		// the whole set goes beside the captured exchanges — and the pointer to
		// it sits under the list it completes, not among the capture lines at
		// the foot of the section, where it reads as having replaced them.
		// Written only when there is something to write, which keeps a clean
		// run from creating a directory to announce that nothing went wrong.
		if runs := res.valInstances.missToEnd; len(runs) > 0 {
			// run_id and seed rather than the model: this summary is only
			// printed for a single-model run, and those two are what identify
			// the content on the fleet and reproduce it.
			report := struct {
				Signal    string          `json:"signal"`
				RunID     string          `json:"run_id"`
				Seed      int64           `json:"seed"`
				Instances int             `json:"instances"`
				Runs      []seriesMissRun `json:"runs"`
			}{
				Signal:    "instances never recovering from a presence miss",
				RunID:     cfg.RunID,
				Seed:      cfg.Seed,
				Instances: len(runs),
				Runs:      runs,
			}
			if b, err := json.MarshalIndent(report, "", "  "); err == nil {
				if path := cfg.dumper.WriteReport("never-recovered.json", append(b, '\n')); path != "" {
					if abs, err := filepath.Abs(path); err == nil {
						path = abs
					}
					fmt.Printf("     %-35s: %s\n", "full list", path)
				}
			}
		}
		// Both denominators, because they are different populations: presence
		// is checked only where a recite was asked, contamination on every
		// completed response. The window is printed with the count because a
		// bare zero invites "no session ever saw another's content", and what
		// was actually checked is "no session saw another's content among
		// those live at the same moment" — markers stop being recognisable
		// once every session holding them has finished.
		// Printed even at zero: an absent line would leave a reader to assume
		// every request was asked, and the excluded population is exactly what
		// makes a presence ratio mean less than it appears to.
		fmt.Printf("   Not asked (output budget too small)  : %d\n", res.valBudgetShortReqs)
		fmt.Printf("   Responses scanned for leaks          : %d\n", res.valLeakCheckedReqs)
		fmt.Printf("   CROSS_CONTAMINATION (other-conv)     : %d across %d requests\n", res.valCrossContamUUIDs, res.valCrossContamReqs)
		fmt.Printf("   Detection window (peak concurrent)   : %d session(s), %d marker(s)\n", res.valWindowSessions, res.valWindowMarkers)
		if dir, n, garbageOnly := cfg.dumper.Written(); dir != "" {
			// Absolute, so the line is copy-pastable from any shell regardless
			// of where the run was started.
			if abs, err := filepath.Abs(dir); err == nil {
				dir = abs
			}
			// Named for what is IN it. The garbage-only capture holds a
			// fraction of the run's exchanges by design, and a reader who
			// takes it for the full one reads its count as a request total.
			label := "Captured exchanges"
			if garbageOnly {
				label = "Captured garbage exchanges"
			}
			fmt.Printf("   %-37s: %d in %s\n", label, n, dir)
		}
	}
	// Repeated from the run's first line, because that is thousands of progress
	// lines ago by the time anyone reads a summary, and the seed is what makes
	// a result reproducible at all — an arm nobody can rerun is an anecdote.
	if cfg.Seed != 0 {
		fmt.Println(strings.Repeat("-", 62))
		fmt.Printf(" Run seed           : %d (run-id %s)\n", cfg.Seed, cfg.RunID)
		fmt.Printf(" Reproduce with     : --seed=%d\n", cfg.Seed)
	}
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
	// Two counts, because they answer different questions and were being read
	// as one. in_flight is requests whose HTTP exchange is open — the
	// concurrency the FLEET is carrying, and the number to quote against its own
	// num_requests_running. held is gate slots, taken before the body is built
	// and released after the response is consumed, so held-minus-in_flight is
	// the client synthesising and uploading several hundred kilobytes per
	// request. On this workload that gap ran to about a third of the total, and
	// quoting the gate figure as fleet concurrency overstated it by that much.
	//
	// held is still worth seeing: below the series count it means long-tail
	// drops, above it means fan-out bursts.
	// The verify segment shows the guid hit rate and ONLY the counters whose
	// non-zero value is genuinely bad on its own: a leaked marker, a corrupted
	// response, and a miss with no evidence either way. Substitutions and
	// ask-quality counters stay off the line — they describe the model and
	// the prompt, and belong in the summary where their caveats are printed
	// next to them.
	// outconf: how faithfully the run reproduced the captured output sizes.
	// With ignore_eos on (regular runs) ~100% by construction; under
	// --verify it is the score for prompt-based length control.
	outConf := ""
	if snap.outTargetSum > 0 {
		outConf = fmt.Sprintf(" outconf=%.1f%%", 100*float64(snap.outActualSum)/float64(snap.outTargetSum))
	}
	verifyInfo := ""
	if snap.verifyOn {
		rate := 100.0
		if snap.verifyChecks > 0 {
			rate = 100 * float64(snap.verifyFound) / float64(snap.verifyChecks)
		}
		verifyInfo = fmt.Sprintf(" guid=%.1f%%", rate)
		// Garbage renders as its classes. post-EOS is the only one excluded
		// from BAD — a literal stop token before the corruption is the model
		// having finished and ignore_eos pushing it onward — and it takes
		// width on the line only in a run that can produce it. A default
		// verify run has ignore_eos off, where eos=0 is a fact about the
		// mode rather than about the fleet, and the classes that move are
		// what the line has room for.
		if snap.verifyGarbage > 0 {
			eos := ""
			if snap.verifyGarbagePostEOS > 0 {
				eos = fmt.Sprintf("eos=%d ", snap.verifyGarbagePostEOS)
			}
			verifyInfo += fmt.Sprintf(" gbg(%stail=%d babble=%d mid=%d)",
				eos, snap.verifyGarbageTail, snap.verifyGarbageBabble, snap.verifyGarbageMid)
		}
		badGarbage := snap.verifyGarbage - snap.verifyGarbagePostEOS
		if snap.verifyLeaked > 0 || badGarbage > 0 || snap.verifyAbsent > 0 {
			// garbage here excludes the proven post-EOS class — the one with
			// an explanation. The plain name is safe because a default verify
			// run has no ignore_eos and so CANNOT produce that class: the
			// number equals total garbage. Only under --verify-force-eos can
			// the two diverge, and there the gbg(...) block beside it shows
			// the eos count the arithmetic excludes.
			verifyInfo += fmt.Sprintf(" BAD(leak=%d garbage=%d lost=%d)",
				snap.verifyLeaked, badGarbage, snap.verifyAbsent)
		}
	}
	if snap.termReason != "" {
		hotInfo := ""
		if snap.gateHotActive > 0 {
			hotInfo = fmt.Sprintf(" hot=%d", snap.gateHotActive)
		}
		return fmt.Sprintf("DONE(%s) %sseries=%d conc=%d in_flight=%d held=%d%s rps=%s cache=%.1f%% gcache=%.1f%% ttft50=%s total=%d errors=%d%s in=%s warm=%s scached=%s out=%s%s",
			snap.termReason, replayPrefix, snap.series, snap.concurrency, snap.gateDispatched, snap.gateActive, hotInfo,
			formatFloat(snap.reqPerSec), snap.cacheHitRate*100, snap.globalLocalCacheRate*100, ttftStr, snap.totalCompleted, snap.totalErrors, verifyInfo,
			formatKiloInt(snap.totalInput), formatKiloInt(snap.totalInputWarm), formatKiloInt(snap.totalCached), formatKiloInt(snap.totalOutput), outConf)
	}
	hotInfo := ""
	if snap.gateHotActive > 0 {
		hotInfo = fmt.Sprintf(" hot=%d", snap.gateHotActive)
	}
	return fmt.Sprintf("%sseries=%d conc=%d in_flight=%d held=%d%s rps=%s cache=%.1f%% gcache=%.1f%% ttft50=%s total=%d errors=%d%s elapsed=%s in=%s warm=%s scached=%s out=%s%s",
		replayPrefix, snap.series, snap.concurrency, snap.gateDispatched, snap.gateActive, hotInfo,
		formatFloat(snap.reqPerSec), snap.cacheHitRate*100, snap.globalLocalCacheRate*100, ttftStr, snap.totalCompleted, snap.totalErrors, verifyInfo,
		formatDuration(snap.elapsed),
		formatKiloInt(snap.totalInput), formatKiloInt(snap.totalInputWarm), formatKiloInt(snap.totalCached), formatKiloInt(snap.totalOutput), outConf)
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
	// rdw is the already-open per-request JSONL writer for this model, or nil
	// when --save-request-data wasn't given. RunAutoBenchmark opens it before
	// any model starts so an unwritable destination fails the run immediately
	// rather than after hours, and owns closing it.
	rdw *requestDataWriter,
) autoBenchmarkResult {
	startTime := time.Now()

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

	// Header row describing the run, written once ahead of the request rows.
	// Deliberately emitted HERE rather than where the writer is opened: the
	// block above clamps and defaults cfg, and the file must record what the
	// run actually used, not what the caller typed. Readers key off
	// record_type, so this is additive — older tooling skips it as an unknown
	// type and reads the request rows exactly as before.
	if rdw != nil {
		if err := rdw.writeAny(buildRunParams(cfg, startTime)); err != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to write run params: %v\n", err)
		}
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
		retries429 := st.totalRetries429.Load()
		retryWait := time.Duration(st.totalRetryWaitNs.Load())

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
			totalRetries429:      retries429,
			totalRetryWait:       retryWait,
			elapsed:              time.Since(startTime),
			cacheWarning:         cacheWarning,
			bothDone:             seriesDone,
			totalInput:           tt.inputCold + tt.inputWarm,
			totalInputWarm:       tt.inputWarm,
			totalCached:          tt.cached,
			totalOutput:          tt.output,
		}
		gateActive, gateColdWaiting, gateNormalWait := st.gate.GateStats()
		gateDispatched := int(st.dispatched.Load())
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
		snap.gateDispatched = gateDispatched
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
		if cfg.Verify {
			snap.verifyOn = true
			snap.verifyChecks = st.valUUIDChecks.Load()
			snap.verifyFound = st.valUUIDFound.Load()
			snap.verifyLeaked = st.valCrossContamUUIDs.Load()
			snap.verifyGarbage = st.valGarbageReqs.Load()
			snap.verifyAbsent = st.valMissAbsent.Load()
			snap.verifyGarbagePostEOS = st.valGarbagePostEOS.Load()
			snap.verifyGarbageTail = st.valGarbageTail.Load()
			snap.verifyGarbageBabble = st.valGarbageGuidBabble.Load()
			snap.verifyGarbageMid = st.valGarbageMidResponse.Load()
		}
		snap.outTargetSum = st.outTargetSum.Load()
		snap.outActualSum = st.outActualSum.Load()
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
	} else if cfg.ReplayRealtime {
		// Real-time replay means the FLEET's latency is the throttle, so the
		// client's own gate has to be out of the way. Left at the default it is
		// a second, independent limiter: it starts at 1 and is raised by a
		// hill-climber watching cache hit rate, which has nothing to do with the
		// TTFT governor deciding how many sessions to admit. Two limiters, one
		// of them ignorant of the experiment.
		//
		// Bounded rather than truly unlimited so a runaway still hits something
		// before the process runs out of sockets. Simulation of this workload
		// tops out near 2,500 concurrent at the most aggressive ramp worth
		// using, so this is out of the way by a wide margin without being a
		// blank cheque. An explicit --concurrency still caps, for an operator
		// who wants one.
		initConc = realtimeUngatedConcurrency
		cfg.Concurrency = initConc // fixed, so the hill-climber stays out too
		fmt.Fprintf(os.Stderr,
			"[realtime] client concurrency gate set to %d: --replay-realtime lets the fleet's "+
				"latency decide the load, and a gate below what the session governor reaches "+
				"would silently become the bottleneck. Pass --concurrency to cap it deliberately.\n",
			initConc)
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
		ttft:              newTTFTWindow(cfg.TTFTWindow),
		skipClk:           newSkipClock(cfg.ReplayRealtime && cfg.ReplaySkipIdle),
		lag:               &pacingLag{},
	}
	if sampler := startVLLMMetricsSampler(benchCtx, cfg.Model, cfg.VLLMMetricsURLs, st.datasetTracker, rdw); sampler != nil {
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
			// Fair share = total benchmark concurrency / endpoints. Hot series
			// run on their own gate, so their concurrency adds to the fleet
			// total rather than being carved out of Concurrency.
			totalConc := cfg.Concurrency + cfg.HotSeriesConcurrency
			endpointRouter = llm.NewEndpointRouterWithFailover(
				dynCfg.BaseURLs, totalConc, cfg.EndpointOverloadThreshold)
		}
	}

	// One poster per endpoint, built ONCE and shared by every series worker.
	// Posters are safe for concurrent use (epMu guards the latched URL form),
	// and building them per-worker would mean len(endpoints) x series HTTP
	// clients — 9k+ at our scale, with no connection reuse between them.
	var replayPicker endpointPicker
	if endpointRouter != nil {
		replayRunID := ""
		if !cfg.ReplayNoStamp {
			replayRunID = cfg.RunID
		}
		eps := endpointRouter.Endpoints()
		posters := make([]*replayPoster, len(eps))
		ok := true
		for i, ep := range eps {
			pp, perr := newReplayPoster(cfg.Model, config.GetAPIKeys(), ep, replayRunID,
				cfg.DryRun, cfg.DryRunColdTPS, cfg.DryRunWarmTPS, cfg.DryRunOutputTPS, st.estimator, &st.dispatched)
			if perr != nil {
				ok = false
				break
			}
			pp.outputRatio = cfg.ReplayOutputRatio
			// Package-level floor: set once, identical for every poster, read-only
			// for the rest of the run (see replayMinOutputTokens).
			replayMinOutputTokens = cfg.ReplayMinOutputTokens
			pp.limitContext = cfg.LimitContext
			pp.replayCharsPerToken = cfg.ReplayCharsPerToken
			pp.forceVolume = cfg.forceVolume()
			// UUID cache-coherency injection (--verify). Same
			// global/read-only refs set on the per-instance poster in
			// replay_router.go — every poster in the run, per-instance or
			// pooled-per-endpoint, shares these identical slices/maps, so
			// this poster (potentially shared across many concurrent
			// sessions once picked by the router) can safely serve any
			// session: the caller passes that session's own turn view into
			// do()/dryDo() per request, so nothing session-specific is ever
			// cached on the poster.
			if cfg.Verify {
				pp.uuidEnabled = true
				pp.registry = cfg.uuidRegistry
				pp.continueOnContamination = cfg.VerifyContinueOnContamination
			}
			pp.dumper = cfg.dumper
			posters[i] = pp
		}
		if ok {
			replayPicker = endpointPicker{router: endpointRouter, posters: posters}
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
		stream, err := openRouterReplayStream(cfg.RouterReplayFile, routerReplayStreamOpts{
			ChanCap:               8,
			SessionLimit:          cfg.ReplaySeries,
			AllowedIndices:        cfg.RouterReplaySeriesIndices,
			Reuse:                 cfg.ReplayReuseSessions,
			MaxRequestsPerSession: cfg.ReplayMaxRequestsPerSession,
		})
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
				runRouterReplaySeriesLoop(benchCtx, cfg, st, rdw, st.routerReplay, replayPicker, fullDocs, updateSnap, workerGate)
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

				// The SERVER'S OWN ANSWER WINS wherever it exists.
				//
				// It did not. cacheHit was the TTFT heuristic alone, and explicitCache was
				// recorded in a separate column that nothing consulted — so a request the
				// server said reused 90k of a 100k prompt was written down as a miss because
				// its first token was not fast. Sampled on a live arm, 55% of the requests
				// called misses had a non-zero cached_tokens. The heuristic exists for
				// servers that report no usage at all; where usage exists it is a guess
				// standing in front of a fact.
				//
				// And a hit is not really a boolean. The server reports HOW MANY tokens it
				// reused, and a partial prefix hit — the normal case on agentic traffic — is
				// neither a hit nor a miss. cacheHitRatio carries that; the boolean is kept
				// only because the progress line and the report have always had one. Where
				// the heuristic is all there is, a hit implies the whole prompt, so the ratio
				// is 1.
				usageReported := metrics.UsageData.InputTokens.Count+metrics.UsageData.CachedTokens.Count > 0
				cacheHitRatio := 0.0
				if usageReported {
					cacheHitRatio = float64(metrics.UsageData.CachedTokens.Count) /
						float64(metrics.UsageData.InputTokens.Count+metrics.UsageData.CachedTokens.Count)
				}
				cacheHit := explicitCache
				if !usageReported {
					cacheHit = !isCold && !ttftDegraded && implicitCache
					if cacheHit {
						cacheHitRatio = 1
					}
				}
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
						CacheHitRatio:        cacheHitRatio,
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
				if metrics.Retries429 > 0 {
					st.totalRetries429.Add(int64(metrics.Retries429))
					st.totalRetryWaitNs.Add(int64(metrics.RetryWait))
				}
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

	// Session-admission governor. Replaces the cache-target series scaler: load
	// grows one session at a time while windowed TTFT stays under the limit, so
	// the fleet's own latency decides where the run settles rather than a
	// concurrency number chosen up front.
	//
	// The gate pauses admission but never sheds — a session already admitted
	// runs to the end of its capture. That is safe only because the ramp is slow
	// relative to how fast the fleet answers; overshoot is bounded, not
	// runaway, and simulation puts it around 0.9s past a 5s limit at the most
	// aggressive rate worth using.
	// The mode's own vital signs, on a slow cadence: how far behind the captured
	// schedule the run is, how much dead time was skipped, and how many sessions
	// are live. Without the first of these a run that fell an hour behind is
	// indistinguishable from one that kept up.
	if cfg.ReplayRealtime {
		go func() {
			t := time.NewTicker(time.Minute)
			defer t.Stop()
			for {
				select {
				case <-benchCtx.Done():
					fmt.Fprintf(os.Stderr, "[realtime] final %s skipped=%s\n",
						st.lag.summary(st.skipClk.Now()), st.skipClk.Skew().Round(time.Second))
					return
				case <-t.C:
				}
				st.mu.Lock()
				series := st.series
				st.mu.Unlock()
				tstats := st.ttft.Stats(time.Now())
				// `slots` and not `sessions`: this is the admitted worker count,
				// and a worker runs one session at a time then pulls the next.
				// It is the concurrency of conversations, not a total of them —
				// adding it to the retired count double-counts every worker that
				// has finished one and started another.
				// The pass count is reported, not inferred. A hit rate spanning
				// several passes is a different quantity from a single-pass one,
				// and leaving it to be deduced from the session count exceeding
				// the corpus size is how it reaches a ledger uncaught.
				passInfo := ""
				if st.routerReplay != nil && st.routerReplay.Pass() > 0 {
					passInfo = fmt.Sprintf(" corpus_pass=%d", st.routerReplay.Pass()+1)
				}
				fmt.Fprintf(os.Stderr, "[realtime] slots=%d%s %s %s skipped=%s\n",
					series, passInfo, tstats.String(cfg.TTFTLimitStat),
					st.lag.summary(st.skipClk.Now()), st.skipClk.Skew().Round(time.Second))
			}
		}()
	}

	if cfg.AdmitEvery > 0 {
		go func() {
			t := time.NewTicker(cfg.AdmitEvery)
			defer t.Stop()
			next := cfg.StartSeries
			for {
				select {
				case <-benchCtx.Done():
					return
				case <-t.C:
				}
				if cfg.MaxSeries > 0 && next >= cfg.MaxSeries {
					// The safety cap, not the fleet. Say so: a run that stops
					// here has measured the cap and nothing else.
					st.mu.Lock()
					if !st.seriesDone {
						st.seriesDone = true
						fmt.Fprintf(os.Stderr, "[admit] stopped at --max-series=%d; this is the cap, "+
							"not the fleet's ceiling — raise it or the result is the cap's\n", cfg.MaxSeries)
					}
					st.mu.Unlock()
					return
				}
				if !st.ttft.Open(time.Now(), cfg.TTFTLimit, cfg.TTFTLimitStat) {
					continue // over the limit: hold, and re-check next tick
				}
				next++
				st.mu.Lock()
				st.series = next
				if next > st.allTimePeakSeries {
					st.allTimePeakSeries = next
				}
				st.mu.Unlock()
				spawnSeries(uuid.New().String(), next)
			}
		}()
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
	// Underfill watcher: the corpus ran out and the admitted slots cannot be
	// filled, so the run has stopped measuring the load it reports.
	//
	// Distinguished from a slow fleet ON PURPOSE, because the two look identical
	// in throughput and mean opposite things. This fires only when the producer
	// is drained — an empty queue is a statement about the corpus, not about the
	// backends.
	if st.routerReplay != nil && cfg.ReplayRealtime && !cfg.ReplayAllowUnderfill {
		go func() {
			t := time.NewTicker(15 * time.Second)
			defer t.Stop()
			for {
				select {
				case <-benchCtx.Done():
					return
				case <-t.C:
				}
				if st.routerReplay.Remaining() > 0 {
					continue
				}
				st.mu.Lock()
				admitted := st.series
				st.mu.Unlock()
				running := int(st.activeReplayWorkers.Load())
				if running >= admitted {
					continue // drained but still fully occupied: a clean finish
				}
				fmt.Fprintf(os.Stderr,
					"[realtime] ABORT: the corpus is exhausted and %d of %d admitted slots have no "+
						"session to run. The fleet is not slow — the queue is empty — so offered load "+
						"has fallen below the %d this run reports, and every total from here is "+
						"diluted by an amount nothing in the output shows. Pass "+
						"--replay-allow-underfill to continue anyway.\n",
					admitted-running, admitted, admitted)
				select {
				case termChan <- termReasonReplayUnderfilled:
				default:
				}
				return
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

		// A leaked marker stops the run by default. Placed first: nothing
		// else the evaluator does matters more than freezing the fleet in the
		// state that just produced cross-session content, and every further
		// request churns the caches the investigation needs intact.
		if st.contaminationStop.Load() && benchCtx.Err() == nil {
			termReason = termReasonContamination
			select {
			case termChan <- termReason:
			default:
			}
			return true
		}

		// Both error ceilings, ahead of every gate below. Does not touch the LLM
		// server — it stays up for replay.
		if st.errorAbortEarned(cfg) && benchCtx.Err() == nil {
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

		// --admit-every owns series scaling when it is set. Both this scaler and
		// the admission governor call spawnSeries, so leaving both live would
		// ramp on two unrelated triggers at once — cache hit rate and TTFT —
		// and the run's session count would be neither one's answer.
		if !seriesDone && cfg.AdmitEvery <= 0 {
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
	if st.routerReplay != nil {
		if sessions, requests := st.routerReplay.Truncated(); sessions > 0 {
			fmt.Fprintf(os.Stderr, "[auto][%s] --replay-max-requests-per-session shortened %d session(s), "+
				"dropping %d request(s) they would otherwise have sent\n",
				shortModelName(cfg.Model), sessions, requests)
		}
	}
	res.totalErrors = st.totalErrors.Load()
	res.totalRetries429 = st.totalRetries429.Load()
	res.totalRetryWait = time.Duration(st.totalRetryWaitNs.Load())
	res.ttftP50 = tm.ttftP50
	res.ttftP95 = tm.ttftP95
	tt := st.stream.TokenTotals()
	res.totalInputCold = tt.inputCold
	res.totalInputWarm = tt.inputWarm
	res.totalOutput = tt.output
	res.totalCachedTokens = tt.cached
	res.valReqs = st.valReqs.Load()
	res.valUUIDChecks = st.valUUIDChecks.Load()
	res.valUUIDFound = st.valUUIDFound.Load()
	res.valExactMatchReqs = st.valExactMatchReqs.Load()
	res.valPresenceMissUUIDs = st.valPresenceMissUUIDs.Load()
	res.valCrossContamUUIDs = st.valCrossContamUUIDs.Load()
	res.valLeakCheckedReqs = st.valLeakCheckedReqs.Load()
	res.valBudgetShortReqs = st.valBudgetShortReqs.Load()
	res.valMissSubstituted = st.valMissSubstituted.Load()
	res.valMissAbsent = st.valMissAbsent.Load()
	res.valEchoedTagsReqs = st.valEchoedTagsReqs.Load()
	res.valNoIDsReqs = st.valNoIDsReqs.Load()
	res.valGarbageReqs = st.valGarbageReqs.Load()
	res.valGarbagePostEOS = st.valGarbagePostEOS.Load()
	res.valGarbageTail = st.valGarbageTail.Load()
	res.valGarbageGuidBabble = st.valGarbageGuidBabble.Load()
	res.valGarbageMidResponse = st.valGarbageMidResponse.Load()
	res.valInstances = st.valInstances.totals()
	res.outTargetSum = st.outTargetSum.Load()
	res.outActualSum = st.outActualSum.Load()
	if cfg.uuidRegistry != nil {
		_, res.valWindowMarkers, res.valWindowSessions = cfg.uuidRegistry.Stats()
	}
	res.valPresenceMissReqs = st.valPresenceMissReqs.Load()
	res.valCrossContamReqs = st.valCrossContamReqs.Load()

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
		if cfg.Verify {
			finalSnap.verifyOn = true
			finalSnap.verifyChecks = st.valUUIDChecks.Load()
			finalSnap.verifyFound = st.valUUIDFound.Load()
			finalSnap.verifyLeaked = st.valCrossContamUUIDs.Load()
			finalSnap.verifyGarbage = st.valGarbageReqs.Load()
			finalSnap.verifyAbsent = st.valMissAbsent.Load()
			finalSnap.verifyGarbagePostEOS = st.valGarbagePostEOS.Load()
			finalSnap.verifyGarbageTail = st.valGarbageTail.Load()
			finalSnap.verifyGarbageBabble = st.valGarbageGuidBabble.Load()
			finalSnap.verifyGarbageMid = st.valGarbageMidResponse.Load()
		}
		finalSnap.outTargetSum = st.outTargetSum.Load()
		finalSnap.outActualSum = st.outActualSum.Load()
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

	// Per-run subdirectory, then the per-request JSONL writers — in that order,
	// and both BEFORE any work starts.
	//
	// Order matters: the writers must open against the run directory, not the
	// base path. Creating them first left every file at the PVC root while the
	// announced run directory stayed empty, which also broke the report step
	// below (it reads cfg.SaveRequestDataDir).
	//
	// Opening here at all is what makes --save-request-data fail fast: the data
	// IS the deliverable, so a run that quietly produces none has wasted its
	// entire duration, and the failure is only noticed afterwards.
	writers := make([]*requestDataWriter, len(models))
	if cfg.SaveRequestDataDir != "" {
		runDir := fmt.Sprintf("%s/%s", cfg.SaveRequestDataDir, time.Now().UTC().Format("2006-01-02T15-04-05Z"))
		if err := os.MkdirAll(runDir, 0755); err != nil {
			return fmt.Errorf("--save-request-data: create run directory: %w", err)
		}
		cfg.SaveRequestDataDir = runDir
		fmt.Printf("Request data will be saved to: %s\n", runDir)
		for i, model := range models {
			w, err := newRequestDataWriter(cfg.SaveRequestDataDir, model, time.Now())
			if err != nil {
				for _, prev := range writers[:i] {
					if prev != nil {
						prev.close()
					}
				}
				return fmt.Errorf("--save-request-data: %w", err)
			}
			writers[i] = w
		}
		defer func() {
			for _, w := range writers {
				if w != nil {
					w.close()
				}
			}
		}()
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
			if err := applyRunSeed(&cfg); err != nil {
				return err
			}
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
			if err := applyRunSeed(&cfg); err != nil {
				return err
			}
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

		// Both of these are properties of the run, not of a model, and every
		// model streams the same file — so they belong here, where the header is
		// read once, rather than in the per-model path that would print them
		// once per model interleaved with each other's output.
		sessions := effectiveSessionCount(hdr.Summary.Sessions,
			len(cfg.RouterReplaySeriesIndices), cfg.ReplaySeries, cfg.ReplayReuseSessions)
		// A fleet bigger than the capture is the ordinary reason to ask for more
		// session slots than the corpus holds, and with cycling turned off the
		// surplus slots pull once, find the stream drained and exit. The run
		// still prints the slot count it was asked for while a fraction of it is
		// running, so say it up front rather than leaving it to be read out of a
		// throughput number hours later.
		if !cfg.ReplayReuseSessions && cfg.MaxSeries > sessions {
			fmt.Fprintf(os.Stderr, "[auto] %d session slots over a %d-session corpus with "+
				"--replay-reuse-sessions=false: %d of them will never hold a session, so the offered "+
				"load is not the %d this run reports. Drop the flag to replay the corpus again "+
				"instead of draining it.\n",
				cfg.MaxSeries, sessions, cfg.MaxSeries-sessions, cfg.MaxSeries)
		}
		if cfg.ReplayMaxRequestsPerSession > 0 {
			fmt.Fprintf(os.Stderr, "[auto] truncating each session to its first %d requests: this run "+
				"replays a normalized corpus, and its totals are not comparable with an uncapped run's.\n",
				cfg.ReplayMaxRequestsPerSession)
		}
		// Cycling is the default, and it removes the terminator a replay run
		// used to have: the corpus never drains, so nothing ends the run on its
		// own. Worth one line, because the alternative is finding out by having
		// a run still going the next morning.
		if cfg.ReplayReuseSessions && cfg.Timeout == 0 && cfg.Total == 0 {
			fmt.Fprintf(os.Stderr, "[auto] cycling the corpus with no --timeout and no --total: "+
				"this run ends when you stop it. Pass one of those, or "+
				"--replay-reuse-sessions=false to stop at the end of the corpus.\n")
		}

		// One registry for the whole run, built before any per-model
		// goroutine spawns so every model shares the identical live marker
		// set (same sharing rationale as replayConversations above).
		//
		// Nothing is precomputed. A session's markers are derived from the
		// block hashes it already carries, when it is dispatched — see
		// replay_uuid_registry.go for why the corpus-wide pass this replaced
		// was answering a question the scoring no longer asks.
		// --dump-dir wins where both apply: it is the wider capture, and it
		// already contains every exchange the garbage-only one would have
		// taken (each .meta.json carries the verdict, so they stay findable).
		switch {
		case cfg.DumpDir != "":
			d, err := newRequestDumper(dumpAll, cfg.DumpDir, cfg.DumpLimit)
			if err != nil {
				return err
			}
			cfg.dumper = d
			fmt.Printf("Dumping verbatim exchanges to %s (limit %d)\n", cfg.DumpDir, cfg.DumpLimit)
		case cfg.Verify && cfg.DumpGarbage:
			// On by default, because a corrupt response is the one verdict
			// that cannot be reconstructed once the run is over, and it is
			// rare enough to keep whole. The directory is only created if
			// something goes wrong (see ensureDir), so a clean run pays
			// nothing and leaves nothing behind.
			d, err := newRequestDumper(dumpGarbage, cfg.DumpGarbageDir, cfg.DumpLimit)
			if err != nil {
				return err
			}
			cfg.dumper = d
			if cfg.DumpGarbageDir != "" {
				fmt.Printf("Dumping garbage exchanges to %s (limit %d)\n", cfg.DumpGarbageDir, cfg.DumpLimit)
			}
		}
		if cfg.Verify {
			cfg.uuidRegistry = newUUIDRegistry()
			// The seed is not repeated here: it is printed once with the run
			// stamp it produced, and markers derive from that stamp.
			fmt.Println("Coherency verification enabled: markers derived per block hash, recited on every request")
		}
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
			res := runSingleModelBenchmark(benchCtx, mcfg, fullDocs, snapCh, idx, writers[idx])
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

// forceVolume decides whether the engine enforces the output profile
// (vLLM's ignore_eos).
//
// A regular benchmark run keeps it ON: the tool's primary job is load with a
// deterministic output volume, and what the padded tokens say is irrelevant
// to throughput. --verify turns it OFF, because forcing generation past the
// stop token pads every response with degenerate text — the babble, the
// guid-shaped invention, the repeated stop attempts — which is exactly the
// noise a coherency check exists to notice, not to manufacture. The built-in
// per-request length ask holds ~90% output conformity in that mode.
// --verify-force-eos overrides for verify runs where deterministic volume
// matters more than genuine output.
func (cfg *AutoBenchmarkConfig) forceVolume() bool {
	return !cfg.Verify || cfg.VerifyForceEOS
}

// ForceVolumeForTest exposes the mode-to-mechanism rule to the CLI package's
// tests, which own the flag surface that feeds it.
func (cfg AutoBenchmarkConfig) ForceVolumeForTest() bool { return cfg.forceVolume() }

// pctOf renders a ratio for the summary. "n/a" rather than 100% when nothing
// was measured: a run that asked no questions has no hit rate, and printing a
// perfect score for it invites exactly the cross-run comparison this line
// exists to serve.
func pctOf(num, den int64) string {
	if den == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", 100*float64(num)/float64(den))
}
