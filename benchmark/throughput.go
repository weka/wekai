package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai-core/config"
	"github.com/weka/wekai-core/llm"
	"golang.org/x/sync/errgroup"
)

// ThroughputRequest holds parameters for a single throughput phase run.
type ThroughputRequest struct {
	Model       string
	DocsContent string
	Question    string
	GUIDs       []string
	Concurrency int
	MaxTokens   int
}

// ThroughputPhaseResult holds the result of one throughput phase.
type ThroughputPhaseResult struct {
	WallDuration      time.Duration
	TotalPromptTokens int
	TotalOutputTokens int
	Requests          []RequestMetrics
	FailedRequests    int
}

// HighSeriesCheckResult holds results for the auxiliary high-series warm prefill measurement.
type HighSeriesCheckResult struct {
	TargetSeries     int
	PoolBefore       int
	NewlyPrimed      int
	Concurrency      int
	WarmPrefillRate  float64 // tok/s
	WarmWall         time.Duration
	WarmPromptTokens int
	Skipped          bool   // true if sweep already reached target
	SkipReason       string // e.g. "final pool=128 >= target=64"
	Failed           int    // any failed requests during the check
}

// ThroughputModelResult holds per-model results for all three phases plus computed rates.
type ThroughputModelResult struct {
	ModelName        string
	ModelDisplayName string

	Phase1 ThroughputPhaseResult // cold prefill
	Phase2 ThroughputPhaseResult // warm prefill
	Phase3 ThroughputPhaseResult // warm prefill + decode

	// Computed rates (tok/s)
	ColdPrefillRate float64
	WarmPrefillRate float64
	DecodeRate      float64

	DecodeTokens          int
	Concurrency           int
	NumSeries             int
	AvgDecodeOutputTokens float64 // average actual output tokens per request in decode phase

	// Sweep is populated when --auto-sweep is used; nil for single-level runs.
	Sweep *ThroughputSweepResult

	// HighSeriesCheck is populated when --high-series-check > 0.
	HighSeriesCheck *HighSeriesCheckResult
}

// ThroughputResult holds results for all models.
type ThroughputResult struct {
	Models               []ThroughputModelResult
	TotalDuration        time.Duration
	Concurrency          int
	SeriesPerConcurrency int
	DecodeTokens         int
	Question             string
}

// ThroughputLevelResult holds results for a single concurrency level in a sweep.
type ThroughputLevelResult struct {
	Concurrency           int
	PoolSize              int
	NewSeriesCount        int
	ColdPrefillRate       float64
	WarmPrefillRate       float64
	DecodeRate            float64
	ColdWall              time.Duration
	WarmWall              time.Duration
	DecodeWall            time.Duration
	ColdPromptTokens      int
	WarmPromptTokens      int
	FailedRequests        int
	DecodeOutputTokens    int     // total actual output tokens in decode phase
	AvgDecodeOutputTokens float64 // DecodeOutputTokens / PoolSize
	DecodeNegativeDT      bool    // true when dt≤0 even after retry
}

// ThroughputSweepResult holds results for a full concurrency sweep for one model.
type ThroughputSweepResult struct {
	Model           string
	Levels          []ThroughputLevelResult
	BestConcurrency int
	StoppedReason   string // "plateau", "max_concurrency", "first_level_only", "cancelled"
}

// effectiveColdPrefillConcurrency returns override when positive,
// otherwise the general concurrency. A non-positive override means "unset".
func effectiveColdPrefillConcurrency(general, override int) int {
	if override > 0 {
		return override
	}
	return general
}

// RunThroughputPhase fires len(req.GUIDs) requests bounded by req.Concurrency.
// Wall clock is measured from just before launch to just after the last goroutine returns.
func RunThroughputPhase(ctx context.Context, req ThroughputRequest) (*ThroughputPhaseResult, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "RunThroughputPhase")
	defer end()

	n := len(req.GUIDs)
	results := make([]RequestMetrics, n)

	sem := make(chan struct{}, req.Concurrency)
	var wg sync.WaitGroup
	wg.Add(n)

	startTime := time.Now()
	for i, guid := range req.GUIDs {
		i, guid := i, guid
		sem <- struct{}{}
		go func() {
			defer func() {
				<-sem
				wg.Done()
			}()
			cachedPrompt := constructBenchmarkPrompt(req.DocsContent, guid)
			metrics := runSingleRequest(ctx, req.Model, cachedPrompt, req.Question,
				i+1, i+1, 0, guid, "", req.MaxTokens)
			results[i] = metrics
		}()
	}
	wg.Wait()
	wallDuration := time.Since(startTime)

	phase := &ThroughputPhaseResult{
		WallDuration: wallDuration,
		Requests:     results,
	}

	missingTokens := false
	for _, m := range results {
		if m.Error != nil {
			phase.FailedRequests++
			continue
		}
		if m.UsageData.InputTokens.Count == 0 {
			missingTokens = true
		}
		phase.TotalPromptTokens += m.UsageData.InputTokens.Count
		// Include reasoning tokens so TotalOutputTokens reflects ALL generated tokens.
		phase.TotalOutputTokens += m.UsageData.OutputTokens.Count + m.UsageData.ReasoningTokens.Count
	}
	if missingTokens {
		logger.Info("throughput: some requests returned zero PromptTokens — provider may not report usage; token rates will be underestimated")
	}

	return phase, nil
}

// RunTimedDecodePhase sends all GUIDs as streaming requests, waits for ALL to receive their
// first token (confirming all are in decode phase), then counts tokens for exactly windowDuration,
// cancels all connections, and returns tokens/second.
//
// If not all requests receive their first token before context cancellation or request completion,
// the function returns 0 (rate could not be measured cleanly).
func RunTimedDecodePhase(
	ctx context.Context,
	model, docsContent, question string,
	GUIDs []string,
	concurrency int,
	maxTokens int,
	windowDuration time.Duration,
) (tokensPerSecond float64, err error) {
	n := len(GUIDs)
	if n == 0 {
		return 0, nil
	}

	// cancelCtx is cancelled once the measurement window closes (or on early exit).
	cancelCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// ttftCount tracks how many requests have fired their first token.
	var ttftCount int64

	// windowStart is set exactly once, when ttftCount reaches n.
	var windowStartOnce sync.Once
	var windowStart time.Time // zero until set

	// totalWindowTokens accumulates tokens received during the measurement window.
	var totalWindowTokens int64

	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	wg.Add(n)

	for i, guid := range GUIDs {
		i, guid := i, guid
		sem <- struct{}{}
		go func() {
			defer func() {
				<-sem
				wg.Done()
			}()

			cachedPrompt := constructBenchmarkPrompt(docsContent, guid)

			// Per-request state: has this request fired its first token yet?
			var firstFired bool

			responseCallback := func(s string) {
				if len(s) == 0 {
					return
				}

				// First-token barrier logic.
				if !firstFired {
					firstFired = true
					newCount := atomic.AddInt64(&ttftCount, 1)
					if newCount == int64(n) {
						// All requests got TTFT — open the measurement window.
						windowStartOnce.Do(func() {
							windowStart = time.Now()
						})
					}
				}

				// Only count tokens while the window is open and within its duration.
				ws := windowStart // read once; zero if not yet set
				if ws.IsZero() {
					return
				}
				elapsed := time.Since(ws)
				if elapsed >= windowDuration {
					// Window closed — cancel all connections.
					cancel()
					return
				}
				atomic.AddInt64(&totalWindowTokens, int64(len([]rune(s))))
			}

			chatParams := &llm.ChatParams{
				ResponseCallback: responseCallback,
				APIKeys:          config.GetAPIKeys(),
				MaxTokens:        maxTokens,
			}
			chatGetter := config.GetChatGetter(model, chatParams)
			chat := chatGetter.GetChat()
			chat.SetSystemBlocks([]llm.SystemBlock{{ID: "root", Content: cachedPrompt, Cache: true}})

			// Ignore per-request errors; the goroutine exits cleanly on context cancel.
			_, _ = chat.Request(cancelCtx, llm.TextParts(question), nil)
			_ = i // suppress unused warning
		}()
	}

	wg.Wait()

	// If the window was never opened (not all TTFTs arrived), return 0.
	if windowStart.IsZero() {
		return 0, nil
	}

	// Determine actual elapsed window (may be shorter than windowDuration if context
	// was already cancelled from the parent before the window expired).
	elapsed := min(time.Since(windowStart), windowDuration)
	if elapsed.Seconds() <= 0 {
		return 0, nil
	}

	tokens := atomic.LoadInt64(&totalWindowTokens)
	return float64(tokens) / elapsed.Seconds(), nil
}

// computeRates derives cold prefill, warm prefill, and decode rates from the three phase results.
// Uses actual output tokens from p2 and p3 rather than the configured max_tokens cap.
// Returns (coldPrefillRate, warmPrefillRate, decodeRate) in tok/s.
// decode is computed via wall-clock delta: (p3.OutputTokens - p2.OutputTokens) / (p3.Wall - p2.Wall).
func computeRates(p1, p2, p3 *ThroughputPhaseResult) (cold, warm, decode float64) {
	t1 := p1.WallDuration.Seconds()
	t2 := p2.WallDuration.Seconds()

	if t1 > 0 {
		cold = float64(p1.TotalPromptTokens) / t1
	}
	if t2 > 0 {
		warm = float64(p2.TotalPromptTokens) / t2
	}
	decode = decodeRateFromDelta(p2.TotalOutputTokens, p3.TotalOutputTokens, p2.WallDuration, p3.WallDuration)
	return
}

// decodeRateFromDelta computes decode throughput from actual output tokens generated
// in the decode phase minus the one-token warm phase baseline.
func decodeRateFromDelta(warmOutputTokens, decodeOutputTokens int, warmWall, decodeWall time.Duration) float64 {
	dt := (decodeWall - warmWall).Seconds()
	netTokens := decodeOutputTokens - warmOutputTokens
	if netTokens <= 0 || dt <= 0 {
		return 0
	}
	return float64(netTokens) / dt
}

// minInt returns the smaller of two ints.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// RunHighSeriesCheck performs an auxiliary warm-prefill-only measurement at a high series count.
// It upfills the existing pool to targetSeries GUIDs, cold-primes any new ones, then measures
// warm prefill on the full pool. Returns the result, the updated pool, and any error.
// If len(pool) >= targetSeries the check is skipped (Skipped=true in result).
func RunHighSeriesCheck(ctx context.Context, model, docsContent, question string,
	pool []string, targetSeries, concurrency, coldPrefillConcurrency int, progress func(string)) (*HighSeriesCheckResult, []string, error) {

	poolBefore := len(pool)

	if poolBefore >= targetSeries {
		reason := fmt.Sprintf("final pool=%d >= target=%d", poolBefore, targetSeries)
		if progress != nil {
			progress(fmt.Sprintf("High-series check (target=%d): skipped — %s\n", targetSeries, reason))
		}
		return &HighSeriesCheckResult{
			TargetSeries: targetSeries,
			PoolBefore:   poolBefore,
			Skipped:      true,
			SkipReason:   reason,
		}, pool, nil
	}

	newCount := targetSeries - poolBefore
	newGUIDs := make([]string, newCount)
	for i := range newGUIDs {
		newGUIDs[i] = uuid.New().String()
	}
	pool = append(pool, newGUIDs...)

	if progress != nil {
		progress(fmt.Sprintf("High-series check (target=%d): upfilling %d new GUIDs at C=%d (cold-prime)...\n",
			targetSeries, newCount, concurrency))
	}

	baseReq := ThroughputRequest{
		Model:       model,
		DocsContent: docsContent,
		Question:    question,
		MaxTokens:   1,
	}

	// Cold-prime new GUIDs only.
	coldReq := baseReq
	coldReq.GUIDs = newGUIDs
	coldReq.Concurrency = minInt(effectiveColdPrefillConcurrency(concurrency, coldPrefillConcurrency), newCount)
	coldResult, err := RunThroughputPhase(ctx, coldReq)
	if err != nil {
		return nil, pool, fmt.Errorf("high-series check cold-prime: %w", err)
	}

	if progress != nil {
		progress(fmt.Sprintf("High-series check: warm prefill on %d-series pool at C=%d...\n",
			targetSeries, concurrency))
	}

	// Warm prefill entire pool.
	warmReq := baseReq
	warmReq.GUIDs = pool
	warmReq.Concurrency = concurrency
	warmResult, err := RunThroughputPhase(ctx, warmReq)
	if err != nil {
		return nil, pool, fmt.Errorf("high-series check warm prefill: %w", err)
	}

	var warmRate float64
	if warmResult.WallDuration.Seconds() > 0 {
		warmRate = float64(warmResult.TotalPromptTokens) / warmResult.WallDuration.Seconds()
	}

	if progress != nil {
		progress(fmt.Sprintf("High-series check done — warm: %.0f tok/s\n", warmRate))
	}

	result := &HighSeriesCheckResult{
		TargetSeries:     targetSeries,
		PoolBefore:       poolBefore,
		NewlyPrimed:      newCount,
		Concurrency:      concurrency,
		WarmPrefillRate:  warmRate,
		WarmWall:         warmResult.WallDuration,
		WarmPromptTokens: warmResult.TotalPromptTokens,
		Failed:           coldResult.FailedRequests + warmResult.FailedRequests,
	}

	return result, pool, nil
}

// RunThroughputSweep runs the concurrency sweep for a single model, doubling C from
// startingConcurrency up to maxConcurrency until the decode rate improvement falls below
// threshold or the context is cancelled.
// If highSeriesCheck > 0, an auxiliary warm-prefill check at that series count is performed
// after the sweep and returned via ThroughputSweepResult (attached to the caller's model result).
func RunThroughputSweep(ctx context.Context, model, docsContent, question string,
	startingConcurrency, coldPrefillConcurrency, seriesPerConcurrency, decodeTokens, tokens, maxConcurrency int,
	threshold float64, highSeriesCheck int, progress func(string)) (*ThroughputSweepResult, *HighSeriesCheckResult, error) {

	ctx, _, end := instrumentation.GetLogSpan(ctx, "RunThroughputSweep")
	defer end()

	result := &ThroughputSweepResult{
		Model: model,
	}

	pool := make([]string, 0)
	prevDecodeRate := 0.0

	for C := startingConcurrency; C <= maxConcurrency; C *= 2 {
		// Check context before each level.
		if err := ctx.Err(); err != nil {
			result.StoppedReason = "cancelled"
			break
		}

		target := C * seriesPerConcurrency
		// Grow pool with fresh GUIDs.
		newCount := target - len(pool)
		for i := 0; i < newCount; i++ {
			pool = append(pool, uuid.New().String())
		}

		if progress != nil {
			progress(fmt.Sprintf("[C=%d] pool=%d new=%d — starting phases...\n", C, len(pool), newCount))
		}

		baseReq := ThroughputRequest{
			Model:       model,
			DocsContent: docsContent,
			Question:    question,
		}

		// Phase 1: cold prefill — new GUIDs only.
		var coldResult *ThroughputPhaseResult
		if newCount > 0 {
			newGUIDs := pool[len(pool)-newCount:]
			req1 := baseReq
			req1.GUIDs = newGUIDs
			req1.Concurrency = minInt(effectiveColdPrefillConcurrency(C, coldPrefillConcurrency), newCount)
			req1.MaxTokens = 1
			var err error
			coldResult, err = RunThroughputPhase(ctx, req1)
			if err != nil {
				return result, nil, fmt.Errorf("sweep C=%d cold prefill: %w", C, err)
			}
		}

		// Phase 2: warm prefill — entire pool at concurrency C.
		req2 := baseReq
		req2.GUIDs = pool
		req2.Concurrency = C
		req2.MaxTokens = 1
		warmResult, err := RunThroughputPhase(ctx, req2)
		if err != nil {
			return result, nil, fmt.Errorf("sweep C=%d warm prefill: %w", C, err)
		}

		// Phase 3: timed decode — 3-second window after all-TTFT barrier.
		const timedDecodeWindow = 3 * time.Second
		decodeRate, err := RunTimedDecodePhase(ctx, model, docsContent, question,
			pool, C, 4000, timedDecodeWindow)
		if err != nil {
			return result, nil, fmt.Errorf("sweep C=%d timed decode: %w", C, err)
		}
		negativeDT := decodeRate == 0

		// Compute rates.
		poolSize := len(pool)

		var coldRate, warmRate float64
		if coldResult != nil && coldResult.WallDuration.Seconds() > 0 {
			coldRate = float64(coldResult.TotalPromptTokens) / coldResult.WallDuration.Seconds()
		}
		if warmResult.WallDuration.Seconds() > 0 {
			warmRate = float64(warmResult.TotalPromptTokens) / warmResult.WallDuration.Seconds()
		}

		var coldWall time.Duration
		var coldPromptTokens int
		var failedRequests int
		if coldResult != nil {
			coldWall = coldResult.WallDuration
			coldPromptTokens = coldResult.TotalPromptTokens
			failedRequests = coldResult.FailedRequests
		}
		failedRequests += warmResult.FailedRequests

		level := ThroughputLevelResult{
			Concurrency:      C,
			PoolSize:         poolSize,
			NewSeriesCount:   newCount,
			ColdPrefillRate:  coldRate,
			WarmPrefillRate:  warmRate,
			DecodeRate:       decodeRate,
			ColdWall:         coldWall,
			WarmWall:         warmResult.WallDuration,
			DecodeWall:       timedDecodeWindow,
			ColdPromptTokens: coldPromptTokens,
			WarmPromptTokens: warmResult.TotalPromptTokens,
			FailedRequests:   failedRequests,
			DecodeNegativeDT: negativeDT,
		}
		result.Levels = append(result.Levels, level)

		if progress != nil {
			progress(fmt.Sprintf("  [C=%d] cold: %.0f tok/s, warm: %.0f tok/s, decode: %.0f tok/s\n",
				C, coldRate, warmRate, decodeRate))
		}

		// Stop condition: plateau check.
		if threshold > 0 && prevDecodeRate > 0 && decodeRate > 0 && decodeRate < prevDecodeRate*(1+threshold) {
			result.StoppedReason = "plateau"
			break
		}
		prevDecodeRate = decodeRate

		// Stop if next level would exceed cap.
		if C*2 > maxConcurrency {
			result.StoppedReason = "max_concurrency"
			break
		}
	}

	// If only one level was run and we didn't set a reason yet, label it.
	if len(result.Levels) == 1 && result.StoppedReason == "" {
		result.StoppedReason = "first_level_only"
	}

	// Find best concurrency (highest decode rate).
	best := 0
	bestRate := -1.0
	for i, l := range result.Levels {
		if l.DecodeRate > bestRate {
			bestRate = l.DecodeRate
			best = i
		}
	}
	if len(result.Levels) > 0 {
		result.BestConcurrency = result.Levels[best].Concurrency
	}

	// Run optional high-series warm prefill check.
	var hsResult *HighSeriesCheckResult
	if highSeriesCheck > 0 {
		finalC := startingConcurrency
		if len(result.Levels) > 0 {
			finalC = result.Levels[len(result.Levels)-1].Concurrency
		}
		var err error
		hsResult, _, err = RunHighSeriesCheck(ctx, model, docsContent, question,
			pool, highSeriesCheck, finalC, coldPrefillConcurrency, progress)
		if err != nil {
			return result, nil, fmt.Errorf("high-series check: %w", err)
		}
	}

	return result, hsResult, nil
}

// RunThroughput orchestrates the three-phase throughput benchmark across models.
// Models are run in parallel; within each model the three phases are sequential.
// If highSeriesCheck > 0, an auxiliary warm-prefill check at that series count is appended per model.
func RunThroughput(ctx context.Context, models []string, docsContent, question string,
	concurrency, coldPrefillConcurrency, seriesPerConcurrency, decodeTokens, tokens, highSeriesCheck int, progress func(string)) (*ThroughputResult, error) {

	ctx, _, end := instrumentation.GetLogSpan(ctx, "RunThroughput")
	defer end()

	numSeries := concurrency * seriesPerConcurrency

	var mu sync.Mutex
	modelResults := make([]ThroughputModelResult, 0, len(models))

	g, gctx := errgroup.WithContext(ctx)

	startTime := time.Now()

	for _, model := range models {
		model := model
		g.Go(func() error {
			if progress != nil {
				progress(fmt.Sprintf("Starting throughput benchmark for %s (series: %d, concurrency: %d)...\n",
					model, numSeries, concurrency))
			}

			// Generate GUIDs once; reuse across all three phases.
			guids := make([]string, numSeries)
			for i := range guids {
				guids[i] = uuid.New().String()
			}

			baseReq := ThroughputRequest{
				Model:       model,
				DocsContent: docsContent,
				Question:    question,
				GUIDs:       guids,
				Concurrency: concurrency,
			}

			// Phase 1: cold prefill (max_tokens=1)
			if progress != nil {
				progress(fmt.Sprintf("  %s: Phase 1 — cold prefill (%d requests, max_tokens=1, concurrency=%d)...\n",
					model, numSeries, effectiveColdPrefillConcurrency(concurrency, coldPrefillConcurrency)))
			}
			req1 := baseReq
			req1.MaxTokens = 1
			req1.Concurrency = effectiveColdPrefillConcurrency(concurrency, coldPrefillConcurrency)
			p1, err := RunThroughputPhase(gctx, req1)
			if err != nil {
				return fmt.Errorf("model %s phase1: %w", model, err)
			}
			if progress != nil {
				progress(fmt.Sprintf("  %s: Phase 1 done — %v, %d prompt tokens, %d failed\n",
					model, p1.WallDuration.Round(time.Millisecond), p1.TotalPromptTokens, p1.FailedRequests))
			}

			// Phase 2: warm prefill (max_tokens=1)
			if progress != nil {
				progress(fmt.Sprintf("  %s: Phase 2 — warm prefill (%d requests, max_tokens=1)...\n", model, numSeries))
			}
			req2 := baseReq
			req2.MaxTokens = 1
			p2, err := RunThroughputPhase(gctx, req2)
			if err != nil {
				return fmt.Errorf("model %s phase2: %w", model, err)
			}
			if progress != nil {
				progress(fmt.Sprintf("  %s: Phase 2 done — %v, %d prompt tokens, %d failed\n",
					model, p2.WallDuration.Round(time.Millisecond), p2.TotalPromptTokens, p2.FailedRequests))
			}

			// Phase 3: warm prefill + decode (max_tokens=decodeTokens)
			if progress != nil {
				progress(fmt.Sprintf("  %s: Phase 3 — warm prefill + decode (%d requests, max_tokens=%d)...\n",
					model, numSeries, decodeTokens))
			}
			req3 := baseReq
			req3.MaxTokens = decodeTokens
			p3, err := RunThroughputPhase(gctx, req3)
			if err != nil {
				return fmt.Errorf("model %s phase3: %w", model, err)
			}
			if progress != nil {
				progress(fmt.Sprintf("  %s: Phase 3 done — %v, %d prompt tokens, %d failed\n",
					model, p3.WallDuration.Round(time.Millisecond), p3.TotalPromptTokens, p3.FailedRequests))
			}

			cold, warm, decode := computeRates(p1, p2, p3)

			var avgDecodeOut float64
			if numSeries > 0 {
				avgDecodeOut = float64(p3.TotalOutputTokens) / float64(numSeries)
			}

			mr := ThroughputModelResult{
				ModelName:             model,
				ModelDisplayName:      getModelDisplayName(model),
				Phase1:                *p1,
				Phase2:                *p2,
				Phase3:                *p3,
				ColdPrefillRate:       cold,
				WarmPrefillRate:       warm,
				DecodeRate:            decode,
				DecodeTokens:          decodeTokens,
				Concurrency:           concurrency,
				NumSeries:             numSeries,
				AvgDecodeOutputTokens: avgDecodeOut,
			}

			// Run optional high-series warm prefill check.
			if highSeriesCheck > 0 {
				// Copy guids slice so RunHighSeriesCheck can append without aliasing.
				poolCopy := make([]string, len(guids))
				copy(poolCopy, guids)
				hsResult, _, err := RunHighSeriesCheck(gctx, model, docsContent, question,
					poolCopy, highSeriesCheck, concurrency, coldPrefillConcurrency, progress)
				if err != nil {
					return fmt.Errorf("model %s high-series check: %w", model, err)
				}
				mr.HighSeriesCheck = hsResult
			}

			mu.Lock()
			modelResults = append(modelResults, mr)
			mu.Unlock()

			if progress != nil {
				progress(fmt.Sprintf("Completed %s — cold prefill: %.0f tok/s, warm prefill: %.0f tok/s, decode: %.0f tok/s\n",
					model, cold, warm, decode))
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return &ThroughputResult{
		Models:               modelResults,
		TotalDuration:        time.Since(startTime),
		Concurrency:          concurrency,
		SeriesPerConcurrency: seriesPerConcurrency,
		DecodeTokens:         decodeTokens,
		Question:             question,
	}, nil
}

// FormatText returns a human-readable summary of the throughput benchmark.
func (r *ThroughputResult) FormatText() string {
	var sb strings.Builder

	sb.WriteString(strings.Repeat("=", 80) + "\n")
	sb.WriteString("Throughput Benchmark Results\n")
	sb.WriteString(strings.Repeat("=", 80) + "\n\n")
	sb.WriteString(fmt.Sprintf("Concurrency: %d  Series/concurrency: %d  Total series: %d\n",
		r.Concurrency, r.SeriesPerConcurrency, r.Concurrency*r.SeriesPerConcurrency))
	sb.WriteString(fmt.Sprintf("Decode tokens (phase 3 max_tokens): %d\n", r.DecodeTokens))
	sb.WriteString(fmt.Sprintf("Question: %s\n", r.Question))
	sb.WriteString(fmt.Sprintf("Total duration: %v\n\n", r.TotalDuration.Round(time.Millisecond)))

	for _, m := range r.Models {
		sb.WriteString(strings.Repeat("-", 80) + "\n")
		sb.WriteString(fmt.Sprintf("Model: %s\n", m.ModelDisplayName))
		sb.WriteString(strings.Repeat("-", 80) + "\n")

		if m.Sweep != nil {
			// Render sweep table.
			sb.WriteString(fmt.Sprintf("  %-6s %-8s %-8s %-14s %-14s %-14s %-12s\n",
				"C", "pool", "new", "cold tok/s", "warm tok/s", "decode tok/s", "avg out/req"))
			sb.WriteString("  " + strings.Repeat("-", 80) + "\n")
			for _, l := range m.Sweep.Levels {
				coldStr := fmt.Sprintf("%.0f", l.ColdPrefillRate)
				if l.NewSeriesCount == 0 {
					coldStr = "-"
				}
				decodeStr := fmt.Sprintf("%.0f", l.DecodeRate)
				if l.DecodeNegativeDT {
					decodeStr = "dt≤0"
				}
				sb.WriteString(fmt.Sprintf("  %-6d %-8d %-8d %-14s %-14.0f %-14s %-12.0f\n",
					l.Concurrency, l.PoolSize, l.NewSeriesCount,
					coldStr, l.WarmPrefillRate, decodeStr, l.AvgDecodeOutputTokens))
			}
			sb.WriteString("\n")
			sb.WriteString(fmt.Sprintf("  Best concurrency: %d   Stopped: %s\n",
				m.Sweep.BestConcurrency, m.Sweep.StoppedReason))
		} else {
			fmtPhase := func(label string, p ThroughputPhaseResult) {
				sb.WriteString(fmt.Sprintf("  %-20s wall: %-12v  prompt_tokens: %-8d  failed: %d\n",
					label+":", p.WallDuration.Round(time.Millisecond), p.TotalPromptTokens, p.FailedRequests))
			}
			fmtPhase("Phase 1 (cold prefill)", m.Phase1)
			fmtPhase("Phase 2 (warm prefill)", m.Phase2)
			fmtPhase("Phase 3 (decode)", m.Phase3)
			sb.WriteString("\n")

			sb.WriteString(fmt.Sprintf("  Cold prefill rate : %.0f tok/s\n", m.ColdPrefillRate))
			sb.WriteString(fmt.Sprintf("  Warm prefill rate : %.0f tok/s\n", m.WarmPrefillRate))
			sb.WriteString(fmt.Sprintf("  Decode rate       : %.0f tok/s\n", m.DecodeRate))
			sb.WriteString(fmt.Sprintf("  Decode output tokens / req (avg): %.0f\n", m.AvgDecodeOutputTokens))
		}
		if m.HighSeriesCheck != nil {
			hs := m.HighSeriesCheck
			if hs.Skipped {
				sb.WriteString(fmt.Sprintf("\n  High-series warm check (target=%d): skipped — %s\n",
					hs.TargetSeries, hs.SkipReason))
			} else {
				sb.WriteString(fmt.Sprintf("\n  High-series warm check (target=%d):\n", hs.TargetSeries))
				sb.WriteString(fmt.Sprintf("    Pool:                %d (upfilled from %d, added %d new)\n",
					hs.TargetSeries, hs.PoolBefore, hs.NewlyPrimed))
				sb.WriteString(fmt.Sprintf("    Concurrency:         %d\n", hs.Concurrency))
				sb.WriteString(fmt.Sprintf("    Warm prefill rate:   %.0f tok/s\n", hs.WarmPrefillRate))
				sb.WriteString(fmt.Sprintf("    Warm wall:           %.1fs\n",
					hs.WarmWall.Seconds()))
				if hs.Failed > 0 {
					sb.WriteString(fmt.Sprintf("    Failed requests:     %d\n", hs.Failed))
				}
			}
		}
		sb.WriteString("\n")
	}

	if len(r.Models) > 1 {
		sb.WriteString(strings.Repeat("=", 80) + "\n")
		sb.WriteString("Summary\n")
		sb.WriteString(strings.Repeat("=", 80) + "\n")
		sb.WriteString(fmt.Sprintf("%-30s %15s %15s %15s\n", "Model", "Cold (tok/s)", "Warm (tok/s)", "Decode (tok/s)"))
		sb.WriteString(strings.Repeat("-", 80) + "\n")
		for _, m := range r.Models {
			sb.WriteString(fmt.Sprintf("%-30s %15.0f %15.0f %15.0f\n",
				truncate(m.ModelDisplayName, 30), m.ColdPrefillRate, m.WarmPrefillRate, m.DecodeRate))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// FormatJSON returns the throughput benchmark results as a JSON string.
func (r *ThroughputResult) FormatJSON() (string, error) {
	type phaseJSON struct {
		WallMS            int64 `json:"wall_ms"`
		TotalPromptTokens int   `json:"total_prompt_tokens"`
		TotalOutputTokens int   `json:"total_output_tokens"`
		FailedRequests    int   `json:"failed_requests"`
	}

	type levelJSON struct {
		Concurrency           int     `json:"concurrency"`
		PoolSize              int     `json:"pool_size"`
		NewSeriesCount        int     `json:"new_series_count"`
		ColdPrefillRate       float64 `json:"cold_prefill_tok_per_sec"`
		WarmPrefillRate       float64 `json:"warm_prefill_tok_per_sec"`
		DecodeRate            float64 `json:"decode_tok_per_sec"`
		ColdWallMS            int64   `json:"cold_wall_ms"`
		WarmWallMS            int64   `json:"warm_wall_ms"`
		DecodeWallMS          int64   `json:"decode_wall_ms"`
		ColdPromptTokens      int     `json:"cold_prompt_tokens"`
		WarmPromptTokens      int     `json:"warm_prompt_tokens"`
		FailedRequests        int     `json:"failed_requests"`
		DecodeOutputTokens    int     `json:"decode_output_tokens"`
		AvgDecodeOutputTokens float64 `json:"avg_decode_output_tokens_per_req"`
	}

	type sweepJSON struct {
		Model           string      `json:"model"`
		Levels          []levelJSON `json:"levels"`
		BestConcurrency int         `json:"best_concurrency"`
		StoppedReason   string      `json:"stopped_reason"`
	}

	type highSeriesCheckJSON struct {
		TargetSeries     int     `json:"target_series"`
		PoolBefore       int     `json:"pool_before"`
		NewlyPrimed      int     `json:"newly_primed"`
		Concurrency      int     `json:"concurrency"`
		WarmPrefillRate  float64 `json:"warm_prefill_tok_per_sec"`
		WarmWallMS       int64   `json:"warm_wall_ms"`
		WarmPromptTokens int     `json:"warm_prompt_tokens"`
		Skipped          bool    `json:"skipped"`
		SkipReason       string  `json:"skip_reason,omitempty"`
		Failed           int     `json:"failed_requests"`
	}

	type modelJSON struct {
		ModelName             string               `json:"model_name"`
		ModelDisplayName      string               `json:"model_display_name"`
		Phase1                phaseJSON            `json:"phase1_cold_prefill"`
		Phase2                phaseJSON            `json:"phase2_warm_prefill"`
		Phase3                phaseJSON            `json:"phase3_decode"`
		ColdPrefillRate       float64              `json:"cold_prefill_tok_per_sec"`
		WarmPrefillRate       float64              `json:"warm_prefill_tok_per_sec"`
		DecodeRate            float64              `json:"decode_tok_per_sec"`
		DecodeTokens          int                  `json:"decode_tokens"`
		Concurrency           int                  `json:"concurrency"`
		NumSeries             int                  `json:"num_series"`
		AvgDecodeOutputTokens float64              `json:"avg_decode_output_tokens_per_req"`
		Sweep                 *sweepJSON           `json:"sweep,omitempty"`
		HighSeriesCheck       *highSeriesCheckJSON `json:"high_series_check,omitempty"`
	}

	type resultJSON struct {
		Models               []modelJSON `json:"models"`
		TotalDurationMS      int64       `json:"total_duration_ms"`
		TotalDuration        string      `json:"total_duration"`
		Concurrency          int         `json:"concurrency"`
		SeriesPerConcurrency int         `json:"series_per_concurrency"`
		DecodeTokens         int         `json:"decode_tokens"`
		Question             string      `json:"question"`
	}

	toPhase := func(p ThroughputPhaseResult) phaseJSON {
		return phaseJSON{
			WallMS:            p.WallDuration.Milliseconds(),
			TotalPromptTokens: p.TotalPromptTokens,
			TotalOutputTokens: p.TotalOutputTokens,
			FailedRequests:    p.FailedRequests,
		}
	}

	toHighSeriesCheck := func(hs *HighSeriesCheckResult) *highSeriesCheckJSON {
		if hs == nil {
			return nil
		}
		return &highSeriesCheckJSON{
			TargetSeries:     hs.TargetSeries,
			PoolBefore:       hs.PoolBefore,
			NewlyPrimed:      hs.NewlyPrimed,
			Concurrency:      hs.Concurrency,
			WarmPrefillRate:  hs.WarmPrefillRate,
			WarmWallMS:       hs.WarmWall.Milliseconds(),
			WarmPromptTokens: hs.WarmPromptTokens,
			Skipped:          hs.Skipped,
			SkipReason:       hs.SkipReason,
			Failed:           hs.Failed,
		}
	}

	toSweep := func(s *ThroughputSweepResult) *sweepJSON {
		if s == nil {
			return nil
		}
		levels := make([]levelJSON, len(s.Levels))
		for i, l := range s.Levels {
			levels[i] = levelJSON{
				Concurrency:           l.Concurrency,
				PoolSize:              l.PoolSize,
				NewSeriesCount:        l.NewSeriesCount,
				ColdPrefillRate:       l.ColdPrefillRate,
				WarmPrefillRate:       l.WarmPrefillRate,
				DecodeRate:            l.DecodeRate,
				ColdWallMS:            l.ColdWall.Milliseconds(),
				WarmWallMS:            l.WarmWall.Milliseconds(),
				DecodeWallMS:          l.DecodeWall.Milliseconds(),
				ColdPromptTokens:      l.ColdPromptTokens,
				WarmPromptTokens:      l.WarmPromptTokens,
				FailedRequests:        l.FailedRequests,
				DecodeOutputTokens:    l.DecodeOutputTokens,
				AvgDecodeOutputTokens: l.AvgDecodeOutputTokens,
			}
		}
		return &sweepJSON{
			Model:           s.Model,
			Levels:          levels,
			BestConcurrency: s.BestConcurrency,
			StoppedReason:   s.StoppedReason,
		}
	}

	out := resultJSON{
		Models:               make([]modelJSON, len(r.Models)),
		TotalDurationMS:      r.TotalDuration.Milliseconds(),
		TotalDuration:        r.TotalDuration.String(),
		Concurrency:          r.Concurrency,
		SeriesPerConcurrency: r.SeriesPerConcurrency,
		DecodeTokens:         r.DecodeTokens,
		Question:             r.Question,
	}
	for i, m := range r.Models {
		out.Models[i] = modelJSON{
			ModelName:             m.ModelName,
			ModelDisplayName:      m.ModelDisplayName,
			Phase1:                toPhase(m.Phase1),
			Phase2:                toPhase(m.Phase2),
			Phase3:                toPhase(m.Phase3),
			ColdPrefillRate:       m.ColdPrefillRate,
			WarmPrefillRate:       m.WarmPrefillRate,
			DecodeRate:            m.DecodeRate,
			DecodeTokens:          m.DecodeTokens,
			Concurrency:           m.Concurrency,
			NumSeries:             m.NumSeries,
			AvgDecodeOutputTokens: m.AvgDecodeOutputTokens,
			Sweep:                 toSweep(m.Sweep),
			HighSeriesCheck:       toHighSeriesCheck(m.HighSeriesCheck),
		}
	}

	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal JSON: %w", err)
	}
	return string(data), nil
}
