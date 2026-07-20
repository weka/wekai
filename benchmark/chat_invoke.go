package benchmark

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// ChatInvokeResult is the outcome of InvokeChat: the final assistant text
// plus accumulated usage across every LLM round-trip the call needed
// (including any tool-execution rounds).
type ChatInvokeResult struct {
	Content string
	Usage   tools.ExecutionUsageData
}

// InvokeChat runs a minimal single-turn chat exchange against chatGetter: an
// optional cached system prompt, one user message, then a tool-execution
// loop (if toolset is non-nil and the model calls tools) until the model
// returns a final text response.
//
// This replaces agents.Chatbot for benchmark/eval code paths that don't need
// conversation history, message injection, context compaction, dangling
// tool-use recovery, or provider-state persistence — just: send, execute any
// tool calls, return the final text and its cost. Streaming/TTFT callbacks
// are unaffected — they're wired into the Chat client itself via ChatParams
// when chatGetter was created, not here.
func InvokeChat(ctx context.Context, chatGetter *llm.ChatGetter, toolset *tools.ToolSet, cachedSystemPrompt, message string) (*ChatInvokeResult, error) {
	chat := chatGetter.GetChat()
	if cachedSystemPrompt != "" {
		chat.SetSystemBlocks([]llm.SystemBlock{{ID: "root", Content: cachedSystemPrompt, Cache: true}})
	}

	modelInfo := chatGetter.GetModelInfo()
	usage := tools.ExecutionUsageData{}

	response, err := callWithRetry(ctx, func() (*llm.Response, error) {
		return chat.Request(ctx, []llm.ContentPart{&llm.TextContent{Text: message}}, toolset)
	})
	if err != nil {
		return nil, err
	}
	accumulateUsage(&usage, modelInfo, response.Usage)

	for len(response.ToolCalls) > 0 {
		toolResponses := executeToolCallsParallel(ctx, toolset, response.ToolCalls, &usage)

		response, err = callWithRetry(ctx, func() (*llm.Response, error) {
			return chat.Respond(ctx, toolResponses, nil, toolset)
		})
		if err != nil {
			return nil, err
		}
		accumulateUsage(&usage, modelInfo, response.Usage)
	}

	usage.ModelName = modelInfo.Provider + "/" + modelInfo.ModelIdentifier
	usage.TotalCost = usage.InputTokens.Cost + usage.OutputTokens.Cost +
		usage.CachedTokens.Cost + usage.ReasoningTokens.Cost

	return &ChatInvokeResult{Content: response.Content, Usage: usage}, nil
}

// executeToolCallsParallel runs every tool call in response in parallel,
// merges each tool's own usage (if any) into usage as a sub-execution, and
// returns the callID->result map Chat.Respond expects.
func executeToolCallsParallel(ctx context.Context, toolset *tools.ToolSet, calls llm.ToolsCalls, usage *tools.ExecutionUsageData) map[string]string {
	type result struct {
		callID    string
		content   string
		usageData tools.ExecutionUsageData
	}
	results := make([]result, len(calls))

	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Add(1)
		go func(idx int, tc llm.ToolCall) {
			defer wg.Done()
			r := result{callID: tc.CallId}
			tool := toolset.GetToolByName(tc.Name)
			if tool == nil {
				r.content = fmt.Sprintf("Tool '%s' not found", tc.Name)
				results[idx] = r
				return
			}
			toolResult, err := tool.Run(ctx, tc.Args)
			if err != nil {
				r.content = fmt.Sprintf("Error executing tool %s: %v", tc.Name, err)
				results[idx] = r
				return
			}
			r.content = toolResult.Content
			r.usageData = toolResult.UsageData
			results[idx] = r
		}(i, call)
	}
	wg.Wait()

	toolResponses := make(map[string]string, len(results))
	for _, r := range results {
		toolResponses[r.callID] = r.content
		if r.usageData.RequestCount > 0 || r.usageData.TotalCost > 0 || len(r.usageData.SubExecutions) > 0 {
			usage.SubExecutions = append(usage.SubExecutions, r.usageData)
		}
	}
	return toolResponses
}

// accumulateUsage adds one LLM response's token usage into usage, pricing it
// via modelInfo — mirrors agents.Chatbot.addUsage's bucket-by-bucket cost math.
func accumulateUsage(usage *tools.ExecutionUsageData, modelInfo llm.ModelInfo, u llm.UsageData) {
	usage.InputTokens.Count += u.PromptTokens
	usage.OutputTokens.Count += u.CompletionTokens
	usage.CachedTokens.Count += u.CachedTokens
	usage.ReasoningTokens.Count += u.ReasoningTokens

	// Anthropic-specific: cache_creation tokens (5-min TTL write) are billed
	// as input tokens at a dedicated rate.
	if u.CacheTokens5Min > 0 {
		usage.InputTokens.Count += u.CacheTokens5Min
		usage.InputTokens.Cost += float64(u.CacheTokens5Min) * modelInfo.CacheTokens5MinCostPerMillion / 1_000_000
	}

	usage.InputTokens.Cost += float64(u.PromptTokens) * modelInfo.InputCostPerMillion / 1_000_000
	usage.OutputTokens.Cost += float64(u.CompletionTokens) * modelInfo.OutputCostPerMillion / 1_000_000
	usage.CachedTokens.Cost += float64(u.CachedTokens) * modelInfo.CachedCostPerMillion / 1_000_000
	usage.ReasoningTokens.Cost += float64(u.ReasoningTokens) * modelInfo.OutputCostPerMillion / 1_000_000

	usage.RequestCount++
}

// callWithRetry retries transient LLM API errors (rate limits, server
// errors, connection resets) with exponential backoff. Mirrors
// agents.Chatbot.callWithRetry's policy minimally — no dangling-tool-use
// recovery, no raw-exchange capture.
func callWithRetry(ctx context.Context, callFunc func() (*llm.Response, error)) (*llm.Response, error) {
	const maxRetries = 10
	const baseDelay = time.Second
	const maxDelay = time.Minute

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		response, err := callFunc()
		if err == nil {
			return response, nil
		}
		lastErr = err
		if !isRetriableError(err) {
			return nil, err
		}
		delay := baseDelay * time.Duration(1<<attempt)
		if delay > maxDelay {
			delay = maxDelay
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	return nil, fmt.Errorf("failed after %d retries: %w", maxRetries, lastErr)
}

func isRetriableError(err error) bool {
	var apiErr *llm.APIError
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case 429, 500, 502, 503, 504, 529:
			return true
		}
		respBody := strings.ToLower(apiErr.Body)
		return strings.Contains(respBody, "overloaded") || strings.Contains(respBody, "internal_error")
	}

	errMsg := strings.ToLower(err.Error())
	for _, s := range []string{
		"connection reset", "connection termination", "connection refused", "connection closed",
		"upstream connect error", "disconnect/reset", "broken pipe", "eof",
		"timeout", "timed out", "overloaded",
	} {
		if strings.Contains(errMsg, s) {
			return true
		}
	}
	return false
}
