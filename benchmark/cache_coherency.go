package benchmark

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/weka/wekai-core/config"
	"github.com/weka/wekai-core/llm"
)

// CacheCoherencyResult holds the results of a cache coherency evaluation
type CacheCoherencyResult struct {
	Model           string
	SeriesCount     int
	Concurrency     int
	GarbageChars    int // total garbage character budget (literal characters, no *4 conversion)
	StampsPerSeries int // number of UUID stamps per series prompt, derived from GarbageChars
	MaxOutputTokens int // auto-computed max output tokens (--max-output-multiplier x expected multi-UUID response size)
	TotalRequests   int
	// SeriesUUIDs holds, per series, the full ordered list of UUID stamps embedded in
	// that series's prompt. Used downstream (CLI layer) to build the global
	// uuid -> owning-series mapping for cross-contamination detection.
	SeriesUUIDs [][]string
	Results     []SeriesRequestResult

	TotalCost         float64
	TotalInputTokens  int
	TotalOutputTokens int
	TotalCachedTokens int

	CacheHits    int     // cycle-2 requests with TTFT ≤ 50% of cycle-1 mean
	CacheTotal   int     // total cycle-2 requests (non-error)
	MeanColdTTFT float64 // mean cycle-1 TTFT in ms
	MeanWarmTTFT float64 // mean cycle-2 TTFT in ms

	ElapsedSeconds float64 // wall-clock time for the entire evaluation

	// AbortedCount is how many requests were intentionally canceled mid-flight by
	// --abort-fraction (HTTP context canceled while vLLM was mid-prefill/mid-load,
	// simulating a client disconnect). Aborted requests carry no usable response, so
	// they are excluded entirely from coherency scoring (see the CLI layer) — this
	// count is the only record of them. Always 0 when --abort-fraction is 0.
	AbortedCount int

	// ResetTriggerCount / ResetErrorCount track --reset-every-n's periodic POSTs to
	// vLLM's reset_prefix_cache dev-mode endpoint: how many were attempted, and of
	// those, how many failed (transport error or non-200, e.g. the server wasn't
	// started with VLLM_SERVER_DEV_MODE=1). Both always 0 when --reset-every-n is 0.
	ResetTriggerCount int
	ResetErrorCount   int
}

// AdversarialOptions bundles the coherency eval's "adversarial" injection flags —
// all default to their zero value (disabled), which preserves the eval's original
// happy-path behavior exactly.
type AdversarialOptions struct {
	// AbortFraction is the fraction (0.0-1.0) of requests to cancel mid-flight, HTTP
	// context canceled, simulating a client disconnect while vLLM is mid-prefill /
	// mid-load (WAITING_FOR_REMOTE_KVS). Exercises the abort-during-load pin path
	// (has_pending_load). 0 = disabled.
	AbortFraction float64
	// AbortDelayMs is how long (ms) to wait after sending an aborted request before
	// canceling it. 0 = pick a random delay in [0, live cold/cycle-1 TTFT estimate)
	// instead of a fixed value.
	AbortDelayMs int
	// ResetEveryN: every N completed requests, POST vLLM's reset_prefix_cache
	// dev-mode endpoint (reset_external=true), injecting a cache reset while other
	// requests may still be mid-load. 0 = disabled.
	ResetEveryN int
}

// SeriesRequestResult holds the result of a single request
type SeriesRequestResult struct {
	SeriesIdx int // index into CacheCoherencyResult.SeriesUUIDs identifying the series this request belongs to
	// ExpectedUUIDs is this series's full ordered UUID stamp list (what the model
	// is expected to return, comma-joined, in order).
	ExpectedUUIDs []string
	Cycle         int
	Response      string // regular output content only (no reasoning)
	Thinking      string // reasoning/thinking content (if any)
	Error         string
	// UUIDFound is parallel to ExpectedUUIDs: UUIDFound[i] reports whether
	// ExpectedUUIDs[i] was found in the response (or thinking). All false when
	// Error != "" (no usable response).
	UUIDFound  []bool
	ExactMatch bool    // response content is the comma-separated ExpectedUUIDs in order (whitespace around separators tolerated), nothing else (output conformity)
	TTFTMs     float64 // time-to-first-token in milliseconds
	// Aborted reports whether this request was intentionally canceled mid-flight by
	// --abort-fraction (confirmed: the request errored AND its context was actually
	// canceled — a request that happens to finish before the abort timer fires is a
	// normal completion, not an abort). Always false when --abort-fraction is 0.
	Aborted bool
}

// stampIntervalChars is the number of garbage filler characters placed between
// consecutive UUID stamps in a series prompt (one UUID stamped every stampIntervalChars
// of garbage). At the default 400000-char budget this yields ~49 stamps.
const stampIntervalChars = 8192

// expectedUUIDStrLen is the canonical hyphenated UUID string length (8-4-4-4-12 + 4 hyphens).
const expectedUUIDStrLen = 36

// computeStampsPerSeries returns how many UUID stamps a series prompt should carry
// for a given total garbage character budget. Stamps are separated by stampIntervalChars-char
// garbage blocks, so (stamps-1) blocks are emitted; stamps = round(garbageChars/stampIntervalChars)
// keeps total garbage volume ≈ garbageChars. Always at least 2 stamps, so there is a
// meaningful "first" and "last" (and the model has more than one UUID to report).
func computeStampsPerSeries(garbageChars int) int {
	stamps := int(math.Round(float64(garbageChars) / float64(stampIntervalChars)))
	if stamps < 2 {
		stamps = 2
	}
	return stamps
}

// ignoreFillerText is the repeating unit used to pad coherency-eval garbage
// blocks. Earlier versions padded with repeated 'A' characters, but reasoning
// models (e.g. Kimi) treated the meaningless 'A' run as content worth
// re-transcribing in their reasoning trace, burning through max_tokens before
// ever emitting the UUID answer. Wrapping the filler in a self-describing
// <ignore>...</ignore> instruction tells the model the block carries no
// information, so it can skip past it instead of re-deriving that itself.
const ignoreFillerText = "<ignore>ignore this text</ignore>"

// ignoreFiller returns a filler string of EXACTLY n characters, built by
// repeating ignoreFillerText and truncating to length n. It is a drop-in
// content replacement for the old strings.Repeat("A", n) padding: every
// call site passes the exact same character budget n it used before — only
// the bytes filling that budget change, not the count. n<=0 returns "".
func ignoreFiller(n int) string {
	if n <= 0 {
		return ""
	}
	reps := n/len(ignoreFillerText) + 1
	return strings.Repeat(ignoreFillerText, reps)[:n]
}

// computeMaxOutputTokens auto-sizes the response token budget for a comma-joined
// list of numUUIDs canonical UUIDs, at multiplier x the expected size so the model
// has headroom to emit every UUID of a multi-UUID answer without truncation.
// multiplier is --max-output-multiplier (default 3.0, preserving the eval's
// original fixed-3x sizing); reasoning models (e.g. Kimi) may need a higher value
// so reasoning tokens don't exhaust the budget before the answer is emitted.
func computeMaxOutputTokens(numUUIDs int, multiplier float64) int {
	if numUUIDs < 1 {
		numUUIDs = 1
	}
	expectedChars := numUUIDs*expectedUUIDStrLen + (numUUIDs - 1) // + separating commas
	expectedTokens := int(math.Ceil(float64(expectedChars) / 4.0))
	maxTokens := int(math.Ceil(float64(expectedTokens) * multiplier))
	if maxTokens < 1 {
		maxTokens = 1
	}
	return maxTokens
}

// buildCoherencySeriesPrompt interleaves uuids with garbageBlock, wrapped in <request>...</request>:
//
//	<request>UUID0 <garbageBlock> UUID1 <garbageBlock> UUID2 ... UUIDlast</request>
func buildCoherencySeriesPrompt(uuids []string, garbageBlock string) string {
	var b strings.Builder
	b.WriteString("<request>")
	for i, u := range uuids {
		if i > 0 {
			b.WriteString(" ")
			b.WriteString(garbageBlock)
			b.WriteString(" ")
		}
		b.WriteString(u)
	}
	b.WriteString("</request>")
	return b.String()
}

// buildCoherencySharedSeriesPrompt is the --shared-prefix-per-series variant: a long
// leading sharedPrefix (byte-identical across every series in the same group, so peers
// co-hit the same prefix-cache blocks) followed by this series' unique UUID stamps in
// the tail:
//
//	<request><sharedPrefix> UUID0 UUID1 UUID2 ... UUIDlast</request>
//
// The per-series UUIDs go in the TAIL, never the shared prefix — mirroring
// constructBenchmarkPromptShared's rule that any per-series bytes in the prefix would
// diverge block 0 of the KV hash chain and destroy sharing. The tail is exactly the
// per-series load/decode region the KV-offload load-pin protects, so any mis-scattered
// block there surfaces as a wrong/leaked UUID. UUIDs are space-separated (no garbage
// between them) so the whole garbage budget lives in the shared, cacheable prefix
// rather than doubling the prompt.
func buildCoherencySharedSeriesPrompt(uuids []string, sharedPrefix string) string {
	var b strings.Builder
	b.WriteString("<request>")
	b.WriteString(sharedPrefix)
	for _, u := range uuids {
		b.WriteString(" ")
		b.WriteString(u)
	}
	b.WriteString("</request>")
	return b.String()
}

// matchesExpectedUUIDList reports whether response is exactly the ordered list of
// expected UUIDs, comma-separated, tolerant ONLY of whitespace around the separators.
// It splits on ",", trims whitespace from each element, and drops a single trailing
// empty element (a response ending in a separator/newline) — interior empties are NOT
// dropped, so a doubled "uuid,,uuid" or a stray "," still fails. It then compares
// element-by-element: match iff the element counts are equal AND every element equals
// the expected UUID at that index. Any extra chatty token, missing/reordered/duplicated
// UUID, or count change makes an element mismatch and fails the check.
func matchesExpectedUUIDList(response string, expected []string) bool {
	parts := strings.Split(response, ",")
	// Drop only a single trailing empty element (trailing separator/newline).
	if n := len(parts); n > 0 && strings.TrimSpace(parts[n-1]) == "" {
		parts = parts[:n-1]
	}
	if len(parts) != len(expected) {
		return false
	}
	for i, p := range parts {
		if strings.TrimSpace(p) != expected[i] {
			return false
		}
	}
	return true
}

// newUUIDGenerator returns a UUID-producing function. When seed != 0 it is backed by a
// deterministic PRNG (rand.NewPCG(seed, seed+1)), so the same seed always yields the same
// UUID sequence regardless of call site — this is what makes --seed reproduce identical
// per-series UUID lists (and therefore identical prompts) across runs. When seed == 0 it
// falls back to crypto/rand (uuid.New()), the pre-existing default behaviour.
func newUUIDGenerator(seed int64) func() string {
	if seed == 0 {
		return func() string { return uuid.New().String() }
	}
	uuidReader := rand.New(rand.NewPCG(uint64(seed), uint64(seed)+1))
	return func() string {
		var buf [16]byte
		for i := range buf {
			buf[i] = byte(uuidReader.UintN(256))
		}
		// Set version 4 and variant bits per RFC 4122.
		buf[6] = (buf[6] & 0x0f) | 0x40
		buf[8] = (buf[8] & 0x3f) | 0x80
		u, _ := uuid.FromBytes(buf[:])
		return u.String()
	}
}

// modelSpecifiesMaxTokens reports whether modelName has an explicit max_tokens=
// (or equivalent) parameter embedded in its spec — the "dynamic/...", "openrouter/...",
// or static "model,max_tokens=N" forms. The cache-coherency eval computes and sets the
// output token budget itself (--max-output-multiplier x the expected multi-UUID
// response size); a user-supplied max_tokens could silently truncate multi-UUID
// answers or be silently overridden, so callers should fail fast instead of running
// with an ambiguous budget.
func modelSpecifiesMaxTokens(modelName string) (bool, error) {
	switch {
	case llm.IsDynamicModel(modelName):
		cfg, err := llm.ParseDynamicModel(modelName)
		if err != nil {
			return false, err
		}
		return cfg.MaxTokens != 0, nil
	case llm.IsOpenRouterModel(modelName):
		cfg, err := llm.ParseOpenRouterModel(modelName)
		if err != nil {
			return false, err
		}
		return cfg.MaxTokens != 0, nil
	default:
		cfg, err := llm.ParseStaticModelParams(modelName)
		if err != nil {
			return false, err
		}
		return cfg.MaxTokens != 0, nil
	}
}

// defaultAbortWindowMs seeds the abort-delay window before any cold (cycle-1) TTFT
// sample has been observed. Large coherency prompts can take seconds to prefill, so
// this is intentionally conservative — once real samples arrive, abortWindowEstimator
// tracks the live mean instead.
const defaultAbortWindowMs = 3000

// abortWindowEstimator maintains a running mean of cold (cycle-1) TTFT, used as the
// upper bound for --abort-delay-ms=0's "random delay in [0, prefill window)"
// behavior — an approximation of "sometime during this request's expected
// prefill/load duration" without knowing that duration up front.
type abortWindowEstimator struct {
	mu sync.Mutex
	ms float64
	n  int
}

func newAbortWindowEstimator() *abortWindowEstimator {
	return &abortWindowEstimator{ms: defaultAbortWindowMs}
}

// observe folds a new cold-cycle TTFT sample into the running mean.
func (e *abortWindowEstimator) observe(ttftMs float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.n++
	e.ms += (ttftMs - e.ms) / float64(e.n)
}

func (e *abortWindowEstimator) windowMs() float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ms
}

// abortDelay resolves how long to wait before canceling an aborted request. A
// positive fixedMs is used verbatim; 0 (the --abort-delay-ms default) draws a
// uniform random delay in [0, window) from the live cold-TTFT estimate.
func abortDelay(fixedMs int, window *abortWindowEstimator) time.Duration {
	if fixedMs > 0 {
		return time.Duration(fixedMs) * time.Millisecond
	}
	w := window.windowMs()
	if w <= 0 {
		w = defaultAbortWindowMs
	}
	return time.Duration(rand.Float64() * w * float64(time.Millisecond))
}

// resetPrefixCacheURL derives vLLM's dev-mode reset_prefix_cache admin URL from an
// OpenAI-compatible dynamic-model base URL. vLLM mounts /reset_prefix_cache at the
// server root (vllm/entrypoints/serve/dev/cache/api_router.py), not under /v1, so the
// conventional "http://host:port/v1" base has any trailing "/v1" suffix stripped
// before appending the admin path (tolerant of a trailing "/" either way — e.g.
// ParseDynamicModel always appends one, but this helper doesn't depend on that).
func resetPrefixCacheURL(baseURL string) string {
	root := strings.TrimRight(baseURL, "/")
	root = strings.TrimSuffix(root, "/v1")
	return root + "/reset_prefix_cache"
}

// resolveResetPrefixCacheURL resolves --reset-every-n's target URL from the eval's
// model spec, or returns an error if modelName isn't a dynamic/ vLLM endpoint (the
// reset_prefix_cache admin endpoint only exists on vLLM's own HTTP server, so hosted
// providers — anthropic/, openai/, openrouter/, etc. — can never support this flag).
func resolveResetPrefixCacheURL(modelName string) (string, error) {
	if !llm.IsDynamicModel(modelName) {
		return "", fmt.Errorf("model %q is not a dynamic/ vLLM endpoint spec — reset_prefix_cache is a vLLM HTTP admin endpoint and has no equivalent on hosted providers", modelName)
	}
	cfg, err := llm.ParseDynamicModel(modelName)
	if err != nil {
		return "", err
	}
	return resetPrefixCacheURL(cfg.BaseURL), nil
}

// triggerPrefixCacheReset POSTs vLLM's dev-mode reset_prefix_cache endpoint
// (vllm/entrypoints/serve/dev/cache/api_router.py) with reset_external=true, so the
// connector-managed (weka) side resets too — the same reset_prefix_cache(reset_
// connector=True) path documented as able to race an in-flight connector load/store
// (vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py). The
// endpoint only exists when the vLLM server was started with VLLM_SERVER_DEV_MODE=1;
// a 404 here almost always means that env var is unset on the server.
func triggerPrefixCacheReset(resetURL string, count, errCount *int32) {
	req, err := http.NewRequest(http.MethodPost, resetURL+"?reset_external=true", nil)
	if err != nil {
		atomic.AddInt32(errCount, 1)
		fmt.Printf("[eval coherency] --reset-every-n: failed to build reset request: %v\n", err)
		return
	}
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		atomic.AddInt32(count, 1)
		atomic.AddInt32(errCount, 1)
		fmt.Printf("[eval coherency] --reset-every-n: POST %s failed: %v\n", resetURL, err)
		return
	}
	defer resp.Body.Close()
	atomic.AddInt32(count, 1)
	if resp.StatusCode != http.StatusOK {
		atomic.AddInt32(errCount, 1)
		fmt.Printf("[eval coherency] --reset-every-n: POST %s returned %s (endpoint requires VLLM_SERVER_DEV_MODE=1 on the vLLM server)\n", resetURL, resp.Status)
	}
}

// RunCacheCoherencyEval runs the cache coherency evaluation.
// It creates `seriesCount` series, each with a LIST of unique UUIDs stamped every
// stampIntervalChars (8192) garbage characters through a large system prompt. All series
// are sent once (cycle 1), then all again (cycle 2), to test whether cached KV
// produces coherent responses. The model is asked to return every UUID it saw, in
// order, comma-separated — testing full-request coherency, not just start/end.
//
// When seed != 0, all UUIDs and prompt randomness are derived from the given seed via a
// deterministic PRNG, making runs fully reproducible. When seed == 0, UUIDs are generated
// from crypto/rand (the default, pre-existing behaviour).
//
// garbageChars is a literal character count (no *4 token approximation) — callers
// resolve the deprecated --garbage-tokens flag (N*4 chars) before calling in.
//
// adv carries the "adversarial" injection flags (--abort-fraction, --abort-delay-ms,
// --reset-every-n). Its zero value disables all of them, which reproduces the eval's
// original happy-path behavior identically (see AdversarialOptions).
//
// sharedPrefixPerSeries (--shared-prefix-per-series) groups series into cohorts of N
// that share one byte-identical leading garbage prefix, so peers concurrently co-hit
// the same prefix-cache blocks (the precondition for a peer-prefix-hit of a block whose
// scatter is still in flight). 0 = every series fully unique (original behavior,
// byte-identical prompts).
//
// maxOutputMultiplier (--max-output-multiplier) scales the auto-computed max_tokens
// budget: multiplier x the expected multi-UUID response size. Default 3.0 reproduces
// the eval's original fixed-3x sizing; reasoning models (e.g. Kimi) may exhaust
// max_tokens on reasoning before emitting the answer, so a higher multiplier gives
// them room to think + answer.
func RunCacheCoherencyEval(ctx context.Context, modelName string, seriesCount, concurrency, garbageChars int, seed int64, totalRequests, sharedPrefixPerSeries int, maxOutputMultiplier float64, adv AdversarialOptions) (*CacheCoherencyResult, error) {
	if seriesCount < 1 {
		seriesCount = 1
	}
	if concurrency < 1 {
		concurrency = 1
	}
	if adv.AbortFraction < 0 || adv.AbortFraction > 1 {
		return nil, fmt.Errorf("--abort-fraction must be in [0,1], got %v", adv.AbortFraction)
	}
	if adv.AbortDelayMs < 0 {
		return nil, fmt.Errorf("--abort-delay-ms must be >= 0, got %d", adv.AbortDelayMs)
	}
	if adv.ResetEveryN < 0 {
		return nil, fmt.Errorf("--reset-every-n must be >= 0, got %d", adv.ResetEveryN)
	}
	if sharedPrefixPerSeries < 0 {
		return nil, fmt.Errorf("--shared-prefix-per-series must be >= 0, got %d", sharedPrefixPerSeries)
	}
	if maxOutputMultiplier <= 0 {
		return nil, fmt.Errorf("--max-output-multiplier must be > 0, got %v", maxOutputMultiplier)
	}

	// The eval owns the output token budget (auto-computed below from the multi-UUID
	// response size); a model spec with an embedded max_tokens would either silently
	// truncate that budget or be silently overridden by it, so fail fast up front.
	if hasMaxTokens, err := modelSpecifiesMaxTokens(modelName); err != nil {
		return nil, fmt.Errorf("invalid model spec %q: %w", modelName, err)
	} else if hasMaxTokens {
		return nil, fmt.Errorf("cache-coherency eval controls the output token budget itself (auto-computed as --max-output-multiplier=%v x the expected multi-UUID response size) — remove max_tokens=... from --model %q", maxOutputMultiplier, modelName)
	}

	// Set up UUID generator: seeded PRNG when --seed is given, crypto/rand otherwise.
	if seed != 0 {
		fmt.Printf("[eval coherency] seeded with --seed %d — UUIDs and prompts deterministic\n", seed)
	}
	newUUID := newUUIDGenerator(seed)

	// --abort-fraction: cancel a fraction of requests' HTTP context mid-flight,
	// simulating a client disconnect while vLLM is mid-prefill/mid-load. abortWindow
	// tracks a live cold-TTFT estimate for --abort-delay-ms=0's random-delay mode.
	abortWindow := newAbortWindowEstimator()
	if adv.AbortFraction > 0 {
		if adv.AbortDelayMs > 0 {
			fmt.Printf("[eval coherency] --abort-fraction %.3f: canceling requests %dms after send\n", adv.AbortFraction, adv.AbortDelayMs)
		} else {
			fmt.Printf("[eval coherency] --abort-fraction %.3f: canceling requests at a random point in [0, live cold-TTFT estimate)\n", adv.AbortFraction)
		}
	}

	// --reset-every-n: resolve the vLLM reset_prefix_cache admin URL once up front.
	// Left empty (no-op) with a warning when the model spec can't support it.
	var resetURL string
	var resetTriggerCount, resetErrorCount int32
	if adv.ResetEveryN > 0 {
		url, err := resolveResetPrefixCacheURL(modelName)
		if err != nil {
			fmt.Printf("[eval coherency] WARNING: --reset-every-n %d disabled: %v\n", adv.ResetEveryN, err)
		} else {
			resetURL = url
			fmt.Printf("[eval coherency] --reset-every-n %d: POSTing %s?reset_external=true periodically (requires VLLM_SERVER_DEV_MODE=1 on the server)\n", adv.ResetEveryN, resetURL)
		}
	}

	// Number of UUID stamps per series, derived from the garbage character budget.
	numStamps := computeStampsPerSeries(garbageChars)
	maxOutputTokens := computeMaxOutputTokens(numStamps, maxOutputMultiplier)

	// Generate the full UUID list for each series, in a fixed (series-major,
	// stamp-minor) order so the same --seed always yields identical lists.
	seriesUUIDs := make([][]string, seriesCount)
	for i := range seriesUUIDs {
		uuids := make([]string, numStamps)
		for j := range uuids {
			uuids[j] = newUUID()
		}
		seriesUUIDs[i] = uuids
	}

	// Build system prompts.
	systemPrompts := make([]string, seriesCount)
	if sharedPrefixPerSeries > 0 {
		// --shared-prefix-per-series: group series into cohorts of N (grouped by
		// seriesIdx/N, the same rule as auto.go). Each group gets ONE shared leading
		// garbage prefix that is byte-identical for every series in the group, so peers
		// concurrently co-hit the same prefix-cache blocks. The group is made distinct
		// from other groups by a NON-UUID "[group G]" marker at the very front (a bare
		// group UUID would be listed by the model as a request UUID and pollute the
		// output-conformity check). The whole garbage budget lives in this shared
		// prefix; each series' unique UUID stamps trail it (see
		// buildCoherencySharedSeriesPrompt).
		//
		// Per-series UUID uniqueness is untouched: seriesUUIDs are generated exactly as
		// in the non-shared path (numStamps distinct draws each, globally unique), and
		// the shared prefix contains NO per-series stamps — so a leaked UUID still maps
		// to exactly one source series and findLeakedUUIDs stays valid.
		// PER-RUN NONCE (2026-07-14 — fixes cross-run KV-key collision).
		//
		// The shared prefix MUST be byte-identical across the series WITHIN a
		// run (that is the whole point: peers co-hit the same prefix-cache
		// blocks). But it must ALSO differ ACROSS runs. The previous
		// `strings.Repeat("A", garbageChars)` was a literal constant, so every
		// run of every build produced the SAME prefix bytes -> the same token
		// block hashes -> the SAME content-addressed weka KV keys, forever.
		//
		// That silently broke the harness: a buggy build writes a poisoned
		// shared-prefix window file, and because the store path dedups on key
		// existence ("key already stored, skipping" / WEKA_WRITE_DEDUP), a
		// LATER, CORRECT build never overwrites it — it just READS the stale
		// poisoned KV and scores as corrupt. Runs on different pods sharing
		// /mnt/weka/kv cross-contaminate the same way. Result: coherency scores
		// that track which stale files happen to be present (and when GC aged
		// them out), NOT the code under test — which is exactly why a long
		// series of fixes produced wandering, non-monotonic, non-converging
		// numbers.
		//
		// Prepending a per-run nonce makes the FIRST block's hash unique, and
		// since vLLM chains block hashes through the parent, EVERY block hash
		// downstream is unique too -> fresh keys every run -> a run can only
		// ever read KV that this same run wrote. Under --seed the nonce is
		// deterministic (newUUID is seeded), so reproducibility is preserved.
		// The nonce MUST be impossible to confuse with a UUID. The eval scores
		// "expected UUID present in response" and lists the UUIDs the model
		// echoes back, so a UUID-shaped token sitting in the prompt gets
		// reported as a request UUID and pollutes the output-conformity check
		// (every series then comes back NOT_EXACT, led by the nonce). This is
		// the same reason the group marker below is "[group N]" and not a UUID.
		//
		// Merely stripping the dashes is NOT enough: a 32-hex-char string is
		// still UUID-shaped, and the model can recognise it and even re-insert
		// the dashes. So emit DECIMAL digits plus letters from a deliberately
		// NON-HEX alphabet (no 0-9a-f), which cannot be reformatted into a
		// canonical UUID and contains no hex run at all.
		//
		// Entropy is derived from newUUID(), so this stays unique per run by
		// default (crypto/rand) AND deterministic under --seed (seeded PRNG).
		nsum := fnv.New64a()
		_, _ = nsum.Write([]byte(newUUID()))
		nval := nsum.Sum64()
		const nonHexAlpha = "GHJKLMNPQRSTVWXYZ" // no 0-9 a-f => never hex-like
		tag := make([]byte, 8)
		for i := range tag {
			tag[i] = nonHexAlpha[(nval>>(uint(i)*6))%uint64(len(nonHexAlpha))]
		}
		runNonce := fmt.Sprintf("%d%s", nval, string(tag))
		noncePrefix := fmt.Sprintf("[run %s] ", runNonce)
		pad := garbageChars - len(noncePrefix)
		if pad < 0 {
			pad = 0
		}
		sharedBlock := noncePrefix + ignoreFiller(pad)
		numGroups := (seriesCount + sharedPrefixPerSeries - 1) / sharedPrefixPerSeries
		groupPrefixes := make([]string, numGroups)
		for g := range groupPrefixes {
			groupPrefixes[g] = fmt.Sprintf("[group %d] ", g) + sharedBlock
		}
		for i, uuids := range seriesUUIDs {
			systemPrompts[i] = buildCoherencySharedSeriesPrompt(uuids, groupPrefixes[i/sharedPrefixPerSeries])
		}
		fmt.Printf("[eval coherency] --shared-prefix-per-series %d: %d series in %d group(s) share a %d-char cacheable prefix; per-series UUID stamps stay unique in the tail\n",
			sharedPrefixPerSeries, seriesCount, numGroups, garbageChars)
		// Log the per-run nonce so cross-run KV-key uniqueness is AUDITABLE from
		// the run log alone: two runs must never print the same nonce (if they
		// did, they would share content-addressed KV keys and a run could read a
		// previous -- possibly poisoned -- run's window files).
		fmt.Printf("[eval coherency] shared-prefix run nonce: %s (unique per run; non-UUID by design so the model cannot echo it back as a request UUID)\n",
			runNonce)
	} else {
		// Default: UUID0 <garbage> UUID1 <garbage> ... UUIDlast, wrapped in <request>.
		// Every series is fully unique (no shared cacheable prefix).
		garbageBlock := ignoreFiller(stampIntervalChars)
		for i, uuids := range seriesUUIDs {
			systemPrompts[i] = buildCoherencySeriesPrompt(uuids, garbageBlock)
		}
	}

	userMessage := "List every UUID shown in the request, in order, separated by commas. Output only the UUIDs and commas, nothing else."

	// Build request list:
	//   - default (totalRequests==0): 2*seriesCount = 1 cold cycle + 1 warm cycle.
	//   - totalRequests>0: round-robin requests across series, looping cycles
	//     until totalRequests are emitted. Cycle = floor(reqIdx/seriesCount)+1.
	//     Useful for cache-warmth tests: SERIES=64 + TOTAL=1024 → 16 cycles
	//     of each series → most reads hit GPU cache, no Weka tier downloads.
	type request struct {
		seriesIdx int
		cycle     int
	}
	var requests []request
	if totalRequests > 0 {
		for i := 0; i < totalRequests; i++ {
			requests = append(requests, request{
				seriesIdx: i % seriesCount,
				cycle:     i/seriesCount + 1,
			})
		}
	} else {
		for cycle := 1; cycle <= 2; cycle++ {
			for i := range seriesCount {
				requests = append(requests, request{seriesIdx: i, cycle: cycle})
			}
		}
	}

	result := &CacheCoherencyResult{
		Model:           modelName,
		SeriesCount:     seriesCount,
		Concurrency:     concurrency,
		GarbageChars:    garbageChars,
		StampsPerSeries: numStamps,
		MaxOutputTokens: maxOutputTokens,
		TotalRequests:   len(requests),
		SeriesUUIDs:     seriesUUIDs,
	}

	// Execute with concurrency control
	sem := make(chan struct{}, concurrency)
	var mu sync.Mutex
	var wg sync.WaitGroup
	var completed int32
	var lastProgressPrint atomic.Value
	lastProgressPrint.Store(time.Now())

	// Pre-allocate results slice to maintain order
	result.Results = make([]SeriesRequestResult, len(requests))

	start := time.Now()
	for idx, req := range requests {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, r request) {
			defer wg.Done()
			defer func() { <-sem }()

			expectedUUIDs := seriesUUIDs[r.seriesIdx]

			var ttftOnce sync.Once
			var ttft time.Duration
			var thinkingBuilder strings.Builder
			reqStart := time.Now()

			chatGetter := config.GetChatGetter(modelName, &llm.ChatParams{
				MaxTokens: maxOutputTokens,
				ResponseCallback: func(s string) {
					ttftOnce.Do(func() {
						ttft = time.Since(reqStart)
					})
				},
				ThinkingCallback: func(s string) {
					ttftOnce.Do(func() {
						ttft = time.Since(reqStart)
					})
					thinkingBuilder.WriteString(s)
				},
				APIKeys: config.GetAPIKeys(),
			})

			// --abort-fraction: optionally cancel this request's HTTP context partway
			// through (context.WithCancel + a timer firing the cancel), simulating a
			// client disconnect while vLLM is mid-prefill/mid-load. reqCtx == ctx
			// (unwrapped) whenever this request isn't selected, so the abort machinery
			// is entirely inert at the --abort-fraction=0 default.
			reqCtx := ctx
			selectedForAbort := adv.AbortFraction > 0 && rand.Float64() < adv.AbortFraction
			if selectedForAbort {
				var reqCancel context.CancelFunc
				reqCtx, reqCancel = context.WithCancel(ctx)
				timer := time.AfterFunc(abortDelay(adv.AbortDelayMs, abortWindow), reqCancel)
				defer timer.Stop()
				defer reqCancel()
			}

			invokeResult, err := InvokeChat(reqCtx, chatGetter, nil, systemPrompts[r.seriesIdx], userMessage)

			// Confirmed abort iff we actually canceled reqCtx (the timer fired) before
			// the request finished on its own — reqCtx.Err() is nil if the request
			// completed first, in which case this is a normal completion, not an abort,
			// regardless of whether it was selectedForAbort.
			aborted := selectedForAbort && err != nil && errors.Is(reqCtx.Err(), context.Canceled)

			var response string
			if invokeResult != nil {
				response = invokeResult.Content
			}
			trimmed := strings.TrimSpace(response)
			thinking := strings.TrimSpace(thinkingBuilder.String())
			sr := SeriesRequestResult{
				SeriesIdx:     r.seriesIdx,
				ExpectedUUIDs: expectedUUIDs,
				Cycle:         r.cycle,
				Response:      trimmed,
				Thinking:      thinking,
				TTFTMs:        float64(ttft.Milliseconds()),
				UUIDFound:     make([]bool, len(expectedUUIDs)),
				Aborted:       aborted,
			}

			if err != nil {
				sr.Error = err.Error()
			} else {
				for j, u := range expectedUUIDs {
					// UUID found in either content or thinking
					sr.UUIDFound[j] = strings.Contains(trimmed, u) || strings.Contains(thinking, u)
				}
				// Conformity: response is the comma-separated expected list (content only, not
				// thinking), tolerant of whitespace around separators but nothing else.
				sr.ExactMatch = matchesExpectedUUIDList(trimmed, expectedUUIDs)
			}

			mu.Lock()
			result.Results[i] = sr
			if aborted {
				result.AbortedCount++
			} else if r.cycle == 1 && sr.TTFTMs > 0 {
				// Feed the live abort-delay window from genuine cold-cycle completions
				// only (aborted/errored requests never got a first token).
				abortWindow.observe(sr.TTFTMs)
			}

			// Accumulate usage (nil on error — no usage was recorded, matching
			// agents.Chatbot's zero-value usageData on a failed request)
			if invokeResult != nil {
				usage := invokeResult.Usage
				result.TotalCost += usage.TotalCost
				result.TotalInputTokens += usage.InputTokens.Count
				result.TotalOutputTokens += usage.OutputTokens.Count
				result.TotalCachedTokens += usage.CachedTokens.Count
			}
			mu.Unlock()

			done := atomic.AddInt32(&completed, 1)

			// --reset-every-n: periodically POST vLLM's reset_prefix_cache endpoint
			// while other requests may still be mid-flight (mid-load). Fire-and-forget
			// on the shared WaitGroup so wg.Wait() below still drains it before the
			// eval returns.
			if resetURL != "" && int(done)%adv.ResetEveryN == 0 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					triggerPrefixCacheReset(resetURL, &resetTriggerCount, &resetErrorCount)
				}()
			}

			last := lastProgressPrint.Load().(time.Time)
			if time.Since(last) >= 10*time.Second || int(done) == len(requests) {
				lastProgressPrint.Store(time.Now())
				fmt.Printf("[progress] %d/%d requests completed (%.0fs elapsed)\n", done, len(requests), time.Since(start).Seconds())
			}
		}(idx, req)
	}

	wg.Wait()
	result.ElapsedSeconds = time.Since(start).Seconds()
	result.ResetTriggerCount = int(resetTriggerCount)
	result.ResetErrorCount = int(resetErrorCount)

	// Compute TTFT-based cache hit rate
	var coldTTFTs, warmTTFTs []float64
	for _, r := range result.Results {
		if r.Error != "" {
			continue
		}
		if r.Cycle == 1 {
			coldTTFTs = append(coldTTFTs, r.TTFTMs)
		} else {
			warmTTFTs = append(warmTTFTs, r.TTFTMs)
		}
	}

	if len(coldTTFTs) > 0 {
		var sum float64
		for _, t := range coldTTFTs {
			sum += t
		}
		result.MeanColdTTFT = sum / float64(len(coldTTFTs))
	}

	hitThresh := result.MeanColdTTFT * 0.5
	for _, t := range warmTTFTs {
		result.CacheTotal++
		if t <= hitThresh {
			result.CacheHits++
		}
	}
	if len(warmTTFTs) > 0 {
		var sum float64
		for _, t := range warmTTFTs {
			sum += t
		}
		result.MeanWarmTTFT = sum / float64(len(warmTTFTs))
	}

	return result, nil
}
