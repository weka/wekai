package benchmark

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
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

func generateGUID() string {
	// Import is at the top of the file
	return time.Now().Format("20060102-150405") + "-" + fmt.Sprintf("%d", time.Now().UnixNano()%1000000)
}
