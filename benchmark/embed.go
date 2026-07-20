package benchmark

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
	"golang.org/x/sync/errgroup"
)

// readDirectoryContents reads a directory recursively and concatenates every
// file's content into one string, each file/subdirectory marked with a
// "---File: path---" / "---Directory: path---" header. Minimal local
// reimplementation of wekai's pkg/utils/files.ReadDirectoryContents (which
// is part of the agentic file-editing toolkit and stays in wekai) — this
// covers only what --docs-dir needs: no path allow-listing, no symlink
// guards, since the caller is a benchmark CLI reading a user-supplied local
// path, not an agent operating on a sandboxed workspace.
func readDirectoryContents(dirPath string) (string, error) {
	resolved, err := filepath.Abs(dirPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve path %s: %w", dirPath, err)
	}
	return readDirectoryContentsRecursive(resolved)
}

func readDirectoryContentsRecursive(dirPath string) (string, error) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return "", fmt.Errorf("failed to read directory %s: %w", dirPath, err)
	}

	var result strings.Builder
	for _, entry := range entries {
		if entry.Name() == ".DS_Store" {
			continue
		}
		entryPath := filepath.Join(dirPath, entry.Name())

		if entry.IsDir() {
			result.WriteString(fmt.Sprintf("---Directory: %s---\n", entryPath))
			subContent, err := readDirectoryContentsRecursive(entryPath)
			if err != nil {
				result.WriteString(fmt.Sprintf("---Error reading directory: %s ---\n%v\n\n", entryPath, err))
				continue
			}
			result.WriteString(subContent)
			result.WriteString("\n\n")
			continue
		}

		content, err := os.ReadFile(entryPath)
		if err != nil {
			result.WriteString(fmt.Sprintf("---Error reading file: %s ---\n%v\n\n", entryPath, err))
			continue
		}
		result.WriteString(fmt.Sprintf("---File: %s---\n", entryPath))
		result.Write(content)
		result.WriteString("\n\n")
	}

	return result.String(), nil
}

// GetModelDisplayName extracts the display name (alias) from a model string.
// For dynamic models with alias parameter, returns the alias; otherwise returns the full model string.
func GetModelDisplayName(modelStr string) string {
	return getModelDisplayName(modelStr)
}

// getModelDisplayName extracts the display name (alias) from a model string
// For dynamic models with alias parameter, returns the alias; otherwise returns the full model string
func getModelDisplayName(modelStr string) string {
	// Check if this is a dynamic model with alias
	if llm.IsDynamicModel(modelStr) {
		config, err := llm.ParseDynamicModel(modelStr)
		if err == nil && config.Alias != "" {
			return config.Alias
		}
	}
	// For all other cases (non-dynamic, or dynamic without alias), return the full model string
	return modelStr
}

// stampDocsWithGUID replaces len(guid) bytes every 768 bytes throughout the document
// to ensure no cross-series token sharing in KV cache
func stampDocsWithGUID(docs string, guid string) string {
	guidLen := len(guid)
	if guidLen == 0 || len(docs) < guidLen {
		return docs
	}

	b := []byte(docs)
	const interval = 768
	for pos := 0; pos+guidLen <= len(b); pos += interval {
		copy(b[pos:pos+guidLen], guid)
	}
	return string(b)
}

// constructBenchmarkPrompt creates a simple prompt with embedded documentation
// Stamps the series GUID throughout the document every 768 bytes to prevent
// cross-series KV cache sharing at any position in the token sequence
func constructBenchmarkPrompt(fullDocs string, seriesGUID string) string {
	stampedDocs := stampDocsWithGUID(fullDocs, seriesGUID)
	return fmt.Sprintf(`<ignore>SERIES_GUID: %s</ignore>

You are a helpful assistant answering questions based on the provided documentation.

<documentation>
%s
</documentation>

Answer the following question based on the documentation provided above.`, seriesGUID, stampedDocs)
}

// constructBenchmarkPromptShared creates a prompt where the doc is stamped with a GROUP guid
// (shared across N series in the same group) and the session ID is the unique per-series guid.
// This allows KV cache sharing for the shared doc prefix while keeping each series distinct.
//
// SERIES_GUID lives only in the trailing <session> tag — placing it in the
// header would diverge block 0 of the KV cache hash chain per series and
// destroy prefix sharing at depth 0 (verified via /weka/data tree snapshot:
// putting SERIES_GUID in the prefix produced 256 distinct depth-0 hashes
// for 256 series, with zero forks).
func constructBenchmarkPromptShared(docs string, groupGUID string, seriesGUID string) string {
	stampedDocs := stampDocsWithGUID(docs, groupGUID)
	return fmt.Sprintf(`<ignore>GROUP_GUID: %s</ignore>

You are a helpful assistant answering questions based on the provided documentation.

<documentation>
%s
</documentation>

<session>%s</session>

Answer the following question based on the documentation provided above.`, groupGUID, stampedDocs, seriesGUID)
}

// truncateToApproxTokens truncates docs to approximately n tokens using 4 bytes/token ratio.
func truncateToApproxTokens(docs string, tokens int) string {
	byteLimit := tokens * 4
	if byteLimit >= len(docs) {
		return docs
	}
	return docs[:byteLimit]
}

// buildSeriesPrompt constructs the benchmark prompt for a series request.
// If stepTokens > 0, truncates docs to that approximate token count.
// If sharedPrefix is true, stamps with groupGUID and appends seriesGUID as session.
// Otherwise uses normal per-series stamping.
func buildSeriesPrompt(fullDocs, seriesGUID, groupGUID string, stepTokens int, sharedPrefix bool) string {
	docs := fullDocs
	if stepTokens > 0 {
		docs = truncateToApproxTokens(fullDocs, stepTokens)
	}
	if sharedPrefix {
		return constructBenchmarkPromptShared(docs, groupGUID, seriesGUID)
	}
	return constructBenchmarkPrompt(docs, seriesGUID)
}

// runSingleRequest executes a single LLM request and captures timing metrics.
// maxTokens overrides the model default when > 0.
func runSingleRequest(ctx context.Context, model, cachedPrompt, question string, requestNum int, seriesNum int, cycleNum int, seriesGUID string, endpointOverride string, maxTokens int) RequestMetrics {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "runSingleRequest")
	defer end()

	var ttft time.Duration
	var firstTokenReceived bool
	var ttftMu sync.Mutex
	startTime := time.Now()
	// Callbacks to capture TTFT from whichever token arrives first (content or reasoning)
	responseCallback := func(s string) {
		ttftMu.Lock()
		defer ttftMu.Unlock()
		if !firstTokenReceived && len(s) > 0 {
			ttft = time.Since(startTime)
			firstTokenReceived = true
			logger.V(1).Info("First token received", "ttft", ttft, "requestNum", requestNum, "seriesNum", seriesNum, "cycleNum", cycleNum)
		}
	}

	thinkingCallback := func(s string) {
		ttftMu.Lock()
		defer ttftMu.Unlock()
		if !firstTokenReceived && len(s) > 0 {
			ttft = time.Since(startTime)
			firstTokenReceived = true
			logger.V(1).Info("First thinking token received", "ttft", ttft, "requestNum", requestNum, "seriesNum", seriesNum, "cycleNum", cycleNum)
		}
	}

	// Create chat params with callbacks for this specific request
	chatParams := &llm.ChatParams{
		ResponseCallback: responseCallback,
		ThinkingCallback: thinkingCallback,
		APIKeys:          config.GetAPIKeys(),
		EndpointOverride: endpointOverride,
		MaxTokens:        maxTokens,
	}

	// Get a new ChatGetter with these callbacks
	chatGetter := config.GetChatGetter(model, chatParams)
	chat := chatGetter.GetChat()

	// Set the cached intro (this enables caching for subsequent requests)
	chat.SetSystemBlocks([]llm.SystemBlock{{ID: "root", Content: cachedPrompt, Cache: true}})

	// Execute the request
	response, err := chat.Request(ctx, llm.TextParts(question), nil)
	totalTime := time.Since(startTime)

	// Prefer response.Content; fall back to Thinking so thinking-only
	// responses (e.g. reasoning models exhausting max_tokens on reasoning)
	// aren't misclassified as empty.
	finalContent := ""
	if response != nil {
		finalContent = response.Content
		if strings.TrimSpace(finalContent) == "" {
			finalContent = response.Thinking
		}
	}

	// Compute raw SSE tail once; used for Warn log and stored in metrics on any failure.
	rawTail := ""
	if response != nil && len(response.RawResponseBytes) > 0 {
		raw := string(response.RawResponseBytes)
		if len(raw) > 8192 {
			raw = raw[len(raw)-8192:]
		}
		rawTail = raw
	}

	// Detect empty response that returned without error (silent failure)
	isEmpty := false
	if err == nil && strings.TrimSpace(finalContent) == "" {
		isEmpty = true
		logger.Warn("benchmark: empty response detected",
			"model", chatGetter.GetModelName(),
			"series_num", seriesNum,
			"cycle_num", cycleNum,
			"request_num", requestNum,
			"ttft", ttft,
			"total_time", totalTime,
			"input_tokens", response.Usage.PromptTokens,
			"output_tokens", response.Usage.CompletionTokens,
			"cached_tokens", response.Usage.CachedTokens,
			"reasoning_tokens", response.Usage.ReasoningTokens,
			"response_content_len", len(finalContent),
			"raw_response_tail", rawTail,
		)
		err = fmt.Errorf("empty response from model")
	}

	metrics := RequestMetrics{
		RequestNum:        requestNum,
		SeriesNum:         seriesNum,
		CycleNum:          cycleNum,
		SeriesGUID:        seriesGUID,
		TimeToFirstToken:  ttft,
		TotalResponseTime: totalTime,
		Error:             err,
		Response:          finalContent,
		IsEmpty:           isEmpty,
	}

	// Retain the full prompt/question/raw-tail only for FAILED requests.
	// Exclude context.Canceled: those are benign abort-drain cancellations (not
	// interesting failures) and carrying a full ~399KB prompt per record adds
	// MB of JSONL noise during the --max-total-errors shutdown drain.
	// Empties use fmt.Errorf (not Canceled) and DeadlineExceeded is not Canceled,
	// so both are still captured correctly.
	if err != nil && !errors.Is(err, context.Canceled) {
		metrics.CachedPrompt = cachedPrompt
		metrics.Question = question
		metrics.RawResponseTail = rawTail
	}

	if err != nil && !isEmpty {
		logger.Error(err, "Request failed", "requestNum", requestNum, "seriesNum", seriesNum, "cycleNum", cycleNum)
	} else if isEmpty {
		// already logged as Warn above
	} else {
		// Copy usage data from response
		metrics.UsageData = tools.ExecutionUsageData{
			InputTokens: tools.TokenUsage{
				Count: response.Usage.PromptTokens,
			},
			OutputTokens: tools.TokenUsage{
				Count: response.Usage.CompletionTokens,
			},
			CachedTokens: tools.TokenUsage{
				Count: response.Usage.CachedTokens,
			},
			ReasoningTokens: tools.TokenUsage{
				Count: response.Usage.ReasoningTokens,
			},
			RequestCount: 1,
		}

		logger.Info("Request completed",
			"requestNum", requestNum,
			"seriesNum", seriesNum,
			"cycleNum", cycleNum,
			"ttft", ttft,
			"totalTime", totalTime,
			"cachedTokens", response.Usage.CachedTokens)
	}

	return metrics
}

// aggregateMetrics computes aggregate statistics from individual request metrics
func aggregateMetrics(modelName string, requests []RequestMetrics) *ModelBenchmarkResult {
	result := &ModelBenchmarkResult{
		ModelName:        modelName,
		ModelDisplayName: getModelDisplayName(modelName),
		Requests:         requests,
		TotalRequests:    len(requests),
	}

	if len(requests) == 0 {
		return result
	}

	// Initialize min/max values - will be set on first successful request
	var minMaxInitialized bool

	var totalTTFT time.Duration
	var totalResponseTime time.Duration

	for _, req := range requests {
		if req.Error != nil {
			result.FailedRequests++
			continue
		}

		// Initialize min/max on first successful request
		if !minMaxInitialized {
			result.MinTTFT = req.TimeToFirstToken
			result.MaxTTFT = req.TimeToFirstToken
			result.MinResponseTime = req.TotalResponseTime
			result.MaxResponseTime = req.TotalResponseTime
			minMaxInitialized = true
		}

		// TTFT metrics (only count non-zero values)
		if req.TimeToFirstToken > 0 {
			totalTTFT += req.TimeToFirstToken
			if req.TimeToFirstToken < result.MinTTFT {
				result.MinTTFT = req.TimeToFirstToken
			}
			if req.TimeToFirstToken > result.MaxTTFT {
				result.MaxTTFT = req.TimeToFirstToken
			}
		}

		// Response time metrics
		totalResponseTime += req.TotalResponseTime
		if req.TotalResponseTime > 0 {
			if req.TotalResponseTime < result.MinResponseTime {
				result.MinResponseTime = req.TotalResponseTime
			}
			if req.TotalResponseTime > result.MaxResponseTime {
				result.MaxResponseTime = req.TotalResponseTime
			}
		}

		// Check if request used explicit cache (CachedTokens > 0)
		if req.UsageData.CachedTokens.Count > 0 {
			result.CachedRequests++
		}

		// Check if request shows implicit caching (TTFT improvement)
		if req.IsImplicitCached {
			result.ImplicitCachedRequests++
		}

		// Aggregate usage data
		result.TotalInputTokens += req.UsageData.InputTokens.Count
		result.TotalOutputTokens += req.UsageData.OutputTokens.Count
		result.TotalCachedTokens += req.UsageData.CachedTokens.Count
		result.TotalCost += req.UsageData.TotalCost
	}

	successfulRequests := result.TotalRequests - result.FailedRequests
	if successfulRequests > 0 {
		result.AvgTTFT = totalTTFT / time.Duration(successfulRequests)
		result.AvgResponseTime = totalResponseTime / time.Duration(successfulRequests)
	}

	return result
}

// RunEmbedBenchmark executes a benchmark for a single model
// The benchmark runs multiple series, where each series is executed with a unique GUID for cache invalidation
func RunEmbedBenchmark(ctx context.Context, req EmbedBenchmarkRequest) (*ModelBenchmarkResult, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "RunEmbedBenchmark")
	defer end()

	logger.Info("Starting benchmark for model",
		"model", req.Model,
		"docsDir", req.DocsDir,
		"numSeries", req.NumSeries,
		"numRequestsPerSeries", req.NumRequests,
		"seriesGUID", req.SeriesGUID)

	// Read documentation once
	var fullDocs string
	if req.DocsContent != "" {
		fullDocs = req.DocsContent
	} else {
		var err error
		fullDocs, err = readDirectoryContents(req.DocsDir)
		if err != nil {
			return nil, fmt.Errorf("failed to read documentation directory: %w", err)
		}
	}

	logger.Info("Documentation loaded", "size", len(fullDocs))

	// Construct cached prompt with embedded docs and unique GUID for this series
	cachedPrompt := constructBenchmarkPrompt(fullDocs, req.SeriesGUID)

	// Execute requests sequentially (to properly test caching within a series)
	var results []RequestMetrics
	for i := 0; i < req.NumRequests; i++ {
		requestNum := i + 1
		logger.Info("Starting request", "requestNum", requestNum, "model", req.Model, "seriesGUID", req.SeriesGUID)

		// Note: cycleNum is passed from RunBenchmarkParallel context
		// For now, we use 0 as a placeholder since RunEmbedBenchmark doesn't track cycles directly
		metrics := runSingleRequest(ctx, req.Model, cachedPrompt, req.Question, requestNum, 0, 0, req.SeriesGUID, "", 0)
		results = append(results, metrics)

		// Sleep between requests (except after the last one)
		if i < req.NumRequests-1 {
			logger.V(1).Info("Sleeping between requests", "duration", req.SleepBetween)
			time.Sleep(req.SleepBetween)
		}
	}

	// Aggregate metrics
	aggregated := aggregateMetrics(req.Model, results)

	logger.Info("Benchmark completed for model",
		"model", req.Model,
		"totalRequests", aggregated.TotalRequests,
		"cachedRequests", aggregated.CachedRequests,
		"failedRequests", aggregated.FailedRequests,
		"avgTTFT", aggregated.AvgTTFT,
		"avgResponseTime", aggregated.AvgResponseTime)

	return aggregated, nil
}

// RunBenchmarkParallel executes benchmarks for multiple models in parallel
// For each model, it runs NumCycles * NumSeries series, where each series contains NumRequests requests
// Each series gets a unique GUID to invalidate caching between series
// Cycles repeat the same series (with same GUIDs) to test long-term cache hits
func RunBenchmarkParallel(ctx context.Context, models []string, baseReq EmbedBenchmarkRequest, progressCallback func(string)) (*BenchmarkResult, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "RunBenchmarkParallel")
	defer end()

	// Determine the number of series and cycles (default to 1 if not specified)
	numSeries := baseReq.NumSeries
	if numSeries <= 0 {
		numSeries = 1
	}
	numCycles := baseReq.NumCycles
	if numCycles <= 0 {
		numCycles = 1
	}

	// Calculate total requests: numCycles * numSeries * numRequests per model
	totalRequestsPerModel := numCycles * numSeries * baseReq.NumRequests

	logger.Info("Starting parallel benchmark",
		"numModels", len(models),
		"numCycles", numCycles,
		"numSeries", numSeries,
		"requestsPerSeries", baseReq.NumRequests,
		"totalRequestsPerModel", totalRequestsPerModel)
	startTime := time.Now()

	// Use errgroup for parallel execution with proper error handling
	var mu sync.Mutex
	allResults := make([]ModelBenchmarkResult, 0, len(models))
	var timeoutOccurred bool

	g, gctx := errgroup.WithContext(ctx)

	for _, model := range models {
		model := model // Capture for goroutine
		g.Go(func() error {
			// Check if context was already cancelled before starting
			select {
			case <-gctx.Done():
				if progressCallback != nil {
					progressCallback(fmt.Sprintf("Skipping model %s due to timeout\n", model))
				}
				return nil // Don't propagate error, just skip this model
			default:
			}
			if progressCallback != nil {
				progressCallback(fmt.Sprintf("Starting benchmark for %s (cycles: %d, series per cycle: %d, requests per series: %d)...\n",
					model, numCycles, numSeries, baseReq.NumRequests))
			}

			// Collect all request metrics across all cycles and series for this model
			modelStartTime := time.Now()
			var allRequestMetrics []RequestMetrics

			// Generate series GUIDs once - they will be reused across cycles
			seriesGUIDs := make([]string, numSeries)
			for seriesIdx := 0; seriesIdx < numSeries; seriesIdx++ {
				seriesNum := seriesIdx + 1
				seriesGUIDs[seriesIdx] = fmt.Sprintf("%s-series%d", generateGUID(), seriesNum)
			}

			// Track baseline TTFT for each series GUID (first request of first cycle)
			// Key: seriesGUID, Value: baseline TTFT
			baselineTTFT := make(map[string]time.Duration)

			// Per-series mutex: ensures cycle N+1 of a series waits for cycle N to finish.
			// Only blocks the specific semaphore slot, not the whole pipeline.
			seriesLocks := make([]sync.Mutex, numSeries)

			// Determine concurrency level for series (default to 1 if not set)
			concurrency := baseReq.Concurrency
			if concurrency <= 0 {
				concurrency = 1
			}

			// Pipeline all (cycle, series) pairs through a single semaphore.
			// Each cycle's goroutines must all acquire their series locks (be queued
			// for the semaphore) before the next cycle's goroutines are launched.
			// This guarantees cycle N requests are emitted before cycle N+1.
			pipelineGroup, pipelineCtx := errgroup.WithContext(gctx)
			sem := make(chan struct{}, concurrency)
			var resultsMu sync.Mutex

			// Shared counters for periodic progress reporting (protected by resultsMu)
			var completedSeries int
			var totalTTFTSum time.Duration
			var totalCachedReqs int
			var totalReqs int
			var totalEmptyResponses int

			// Periodic progress ticker: reports aggregated stats every 10 seconds
			doneCh := make(chan struct{})
			go func() {
				ticker := time.NewTicker(10 * time.Second)
				defer ticker.Stop()
				lastReportedSeries := 0
				for {
					select {
					case <-doneCh:
						return
					case <-ticker.C:
						if progressCallback == nil {
							continue
						}
						resultsMu.Lock()
						done := completedSeries
						ttftSum := totalTTFTSum
						cached := totalCachedReqs
						reqs := totalReqs
						resultsMu.Unlock()
						if done <= lastReportedSeries {
							continue
						}
						lastReportedSeries = done
						var avgTTFT time.Duration
						if done > 0 {
							avgTTFT = ttftSum / time.Duration(done)
						}
						resultsMu.Lock()
						emptyResp := totalEmptyResponses
						resultsMu.Unlock()
						progressLine := fmt.Sprintf("  %s: progress — %d series done, avg TTFT: %v, cached: %d/%d",
							model, done, avgTTFT.Round(time.Millisecond), cached, reqs)
						if emptyResp > 0 {
							progressLine += fmt.Sprintf(", empty: %d", emptyResp)
						}
						progressCallback(progressLine + "\n")
					}
				}
			}()

			for cycleIdx := 0; cycleIdx < numCycles; cycleIdx++ {
				cycleNum := cycleIdx + 1

				// Shuffle series dispatch order each cycle to avoid fixed ordering bias
				cycleOrder := rand.Perm(numSeries)

				if progressCallback != nil && numCycles > 1 {
					progressCallback(fmt.Sprintf("  %s: Dispatching cycle %d/%d (concurrency: %d)\n",
						model, cycleNum, numCycles, concurrency))
				}

				// Barrier: each goroutine signals after acquiring its series lock,
				// meaning it's queued for the semaphore. We wait for all signals
				// before dispatching the next cycle.
				cycleReady := make(chan struct{}, numSeries)

				for _, seriesIdx := range cycleOrder {
					seriesIdx := seriesIdx // Capture for goroutine
					cycleNum := cycleNum   // Capture for goroutine
					seriesNum := seriesIdx + 1
					seriesGUID := seriesGUIDs[seriesIdx]

					pipelineGroup.Go(func() error {
						// Per-series ordering: wait for previous cycle of this series to finish
						seriesLocks[seriesIdx].Lock()
						defer seriesLocks[seriesIdx].Unlock()

						// Signal: series lock acquired, now queued for semaphore
						cycleReady <- struct{}{}

						// Acquire semaphore slot
						select {
						case sem <- struct{}{}:
						case <-pipelineCtx.Done():
							return nil
						}
						defer func() { <-sem }()

						// Check cancellation after acquiring slot
						select {
						case <-pipelineCtx.Done():
							return nil
						default:
						}

						req := baseReq
						req.Model = model
						req.SeriesGUID = seriesGUID

						result, err := RunEmbedBenchmark(pipelineCtx, req)
						if err != nil {
							if pipelineCtx.Err() == context.DeadlineExceeded || pipelineCtx.Err() == context.Canceled {
								if progressCallback != nil {
									progressCallback(fmt.Sprintf("    %s: Timeout at cycle %d, series %d. Returning partial results.\n", model, cycleNum, seriesNum))
								}
								return nil
							}
							logger.Error(err, "Benchmark failed", "model", model, "cycle", cycleNum, "series", seriesNum)
							if progressCallback != nil {
								progressCallback(fmt.Sprintf("    %s: Cycle %d, Series %d failed: %v\n", model, cycleNum, seriesNum, err))
							}
							return fmt.Errorf("benchmark failed for model %s cycle %d series %d: %w", model, cycleNum, seriesNum, err)
						}

						// Update metadata and detect implicit caching
						for i := range result.Requests {
							result.Requests[i].SeriesNum = seriesNum
							result.Requests[i].CycleNum = cycleNum

							resultsMu.Lock()
							if result.Requests[i].Error == nil {
								if result.Requests[i].RequestNum == 1 {
									// Set baseline only if not already set (first cycle wins)
									if _, exists := baselineTTFT[seriesGUID]; !exists {
										if result.Requests[i].TimeToFirstToken > 0 {
											baselineTTFT[seriesGUID] = result.Requests[i].TimeToFirstToken
										}
									} else {
										// Baseline exists — check for implicit caching
										baseline := baselineTTFT[seriesGUID]
										if baseline > 0 && result.Requests[i].TimeToFirstToken <= baseline/2 {
											result.Requests[i].IsImplicitCached = true
										}
									}
								} else {
									// Non-first request in series — check against baseline
									if baseline, exists := baselineTTFT[seriesGUID]; exists && baseline > 0 {
										if result.Requests[i].TimeToFirstToken <= baseline/2 {
											result.Requests[i].IsImplicitCached = true
										}
									}
								}
							}
							resultsMu.Unlock()
						}

						resultsMu.Lock()
						allRequestMetrics = append(allRequestMetrics, result.Requests...)
						completedSeries++
						totalTTFTSum += result.AvgTTFT
						totalCachedReqs += result.CachedRequests
						totalReqs += result.TotalRequests
						for _, req := range result.Requests {
							if req.IsEmpty {
								totalEmptyResponses++
							}
						}
						resultsMu.Unlock()

						return nil
					})
				}

				// Wait for all goroutines in this cycle to acquire their series locks
				// before dispatching the next cycle
				for range numSeries {
					<-cycleReady
				}
			}

			// Wait for all pipelined jobs
			pipelineErr := pipelineGroup.Wait()
			close(doneCh)
			if pipelineErr != nil {
				if gctx.Err() == context.DeadlineExceeded {
					logger.Info("Benchmark pipeline interrupted by timeout", "model", model)
					goto savePartialResults
				}
				return pipelineErr
			}

		savePartialResults:
			// Aggregate all metrics across all cycles and series for this model
			// This happens even if we hit timeout and have partial results
			if len(allRequestMetrics) > 0 {
				aggregatedResult := aggregateMetrics(model, allRequestMetrics)
				aggregatedResult.WallDuration = time.Since(modelStartTime)
				aggregatedResult.Concurrency = concurrency

				mu.Lock()
				allResults = append(allResults, *aggregatedResult)
				mu.Unlock()

				if progressCallback != nil {
					progressCallback(fmt.Sprintf("Completed benchmark for %s (total requests: %d, avg TTFT: %v, avg response time: %v)\n",
						model, aggregatedResult.TotalRequests, aggregatedResult.AvgTTFT, aggregatedResult.AvgResponseTime))
					resultsMu.Lock()
					emptyCount := totalEmptyResponses
					resultsMu.Unlock()
					if emptyCount > 0 {
						progressCallback(fmt.Sprintf("  ⚠  %s: %d empty responses detected (potential server errors or overload)\n",
							model, emptyCount))
					}
				}
			} else {
				if progressCallback != nil {
					progressCallback(fmt.Sprintf("No results collected for %s\n", model))
				}
			}

			return nil
		})
	}

	// Wait for all goroutines, but don't fail if some were interrupted by timeout
	if err := g.Wait(); err != nil {
		// Check if the error is due to timeout
		if ctx.Err() == context.DeadlineExceeded {
			timeoutOccurred = true
			logger.Info("Benchmark timeout occurred, returning partial results", "modelsCompleted", len(allResults))
		} else {
			return nil, err
		}
	}

	totalDuration := time.Since(startTime)
	logger.Info("Parallel benchmark completed",
		"totalModels", len(allResults),
		"numCycles", numCycles,
		"numSeries", numSeries,
		"totalDuration", totalDuration)

	concurrency := baseReq.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	return &BenchmarkResult{
		Models:        allResults,
		TotalDuration: totalDuration,
		DocsDir:       baseReq.DocsDir,
		Question:      baseReq.Question,
		NumSeries:     numSeries,
		NumCycles:     numCycles,
		Concurrency:   concurrency,
		TimedOut:      timeoutOccurred,
	}, nil
}

// generateGUID generates a unique GUID using google/uuid package
func generateGUID() string {
	// Import is at the top of the file
	return time.Now().Format("20060102-150405") + "-" + fmt.Sprintf("%d", time.Now().UnixNano()%1000000)
}
