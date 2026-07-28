package benchmark

// Direct-HTTP poster for tree-aware router replay. Bypasses llm.Chat
// because Chat builds the body itself from accumulated chat history; it
// can't accept a pre-built multi-turn conversation with mixed text /
// tool_use / tool_result blocks (which is what the replay needs to send).
//
// Endpoint: attempt-then-fallback — <base>+leaf verbatim first, /v1 inserted
// on 404 (see replayPoster). Body construction is in replay_router_wire.go;
// this file handles HTTP transport, headers, SSE parsing, and metrics.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// replayPoster owns the per-instance HTTP plumbing.
type replayPoster struct {
	model  string
	apiKey string
	// Attempt-then-fallback endpoint resolution: the FIRST attempt honors
	// the spec's base URL exactly (base + endpoint leaf — whatever path
	// prefix the operator gave: /v1, /v2, a proxy prefix, anything). If it
	// 404s, the same request retries once with "/v1" inserted (so bare-root
	// specs work against real vLLM without the operator knowing the API
	// path). The first 2xx latches its form for the rest of the run; only
	// 404 ever triggers the fallback — other 4xx/5xx are real request
	// errors on a valid path.
	epPrimary  string
	epFallback string
	epMu       sync.Mutex
	epResolved string // latched endpoint; "" until the first success
	epFellBack bool
	apiType    string // "anthropic", "openai", or "openai_vllm"
	client     *http.Client
	runID      string
	dryRun     bool
	estimator  *cacheEstimator
	dryRates   struct {
		coldTPS   int
		warmTPS   int
		outputTPS int
	}

	// outputRatio and forceOutput implement --replay-output-ratio /
	// --replay-natural-output. Set directly on the poster after construction
	// (see runRouterReplayInstance) rather than threaded through
	// newReplayPoster, to avoid touching its many existing call sites.
	outputRatio float64
	forceOutput bool
}

func newReplayPoster(modelSpec string, keys llm.APIKeys, endpointOverride string, runID string, dryRun bool, coldTPS, warmTPS, outputTPS int, estimator *cacheEstimator) (*replayPoster, error) {
	if !llm.IsDynamicModel(modelSpec) {
		return nil, fmt.Errorf("router-replay requires a dynamic/ model spec; got %q", modelSpec)
	}
	dyn, err := llm.ParseDynamicModel(modelSpec)
	if err != nil {
		return nil, fmt.Errorf("parse model spec: %w", err)
	}

	// Accepted target types: anthropic (original), openai, openai_vllm.
	// openai_vllm is treated identically to openai in the replay path — both
	// use /v1/chat/completions. The distinction matters for the Chat path
	// (max_tokens vs max_completion_tokens) but replay requests carry their
	// own max_tokens so it's irrelevant here.
	switch dyn.Type {
	case "anthropic":
		// OK — existing behaviour.
	case "openai", "openai_vllm":
		// OK — new path.
	default:
		return nil, fmt.Errorf("router-replay supports type=anthropic, type=openai, or type=openai_vllm (got %q)", dyn.Type)
	}

	base := ""
	if endpointOverride != "" {
		base = endpointOverride
	} else if len(dyn.BaseURLs) > 0 {
		base = dyn.BaseURLs[0]
	} else {
		return nil, fmt.Errorf("no base URL in model spec")
	}
	base = strings.TrimRight(base, "/")

	// API key selection: Anthropic targets use x-api-key (or dummy-key for
	// local endpoints); OpenAI targets use Bearer auth with the OpenAI key
	// (or dummy-key for local endpoints).
	apiKey := keys.Anthropic
	if dyn.Type == "openai" || dyn.Type == "openai_vllm" {
		apiKey = keys.OpenAI
	}
	if apiKey == "" {
		apiKey = "dummy-key"
	}

	// Endpoint leaf: /messages for Anthropic, /chat/completions for OpenAI.
	// The primary attempt appends it to the operator's base verbatim; the
	// fallback inserts /v1 (see the struct comment for the contract).
	leaf := "/messages"
	if dyn.Type == "openai" || dyn.Type == "openai_vllm" {
		leaf = "/chat/completions"
	}
	epPrimary := base + leaf
	epFallback := base + "/v1" + leaf

	model := dyn.Model
	if model == "" && !dryRun {
		discovered, derr := discoverModelName(base)
		if derr != nil {
			return nil, fmt.Errorf("model=... not set in %q and model discovery from %s failed: %w", modelSpec, base, derr)
		}
		model = discovered
		logDiscoveredModelOnce(base, model)
	}
	if model == "" && dryRun {
		model = "dry-run"
	}
	return &replayPoster{
		model:      model,
		epPrimary:  epPrimary,
		epFallback: epFallback,
		apiKey:     apiKey,
		apiType:    dyn.Type,
		runID:      runID,
		dryRun:     dryRun,
		estimator:  estimator,
		dryRates: struct {
			coldTPS   int
			warmTPS   int
			outputTPS int
		}{coldTPS, warmTPS, outputTPS},
		client: &http.Client{},
	}, nil
}

var (
	discoveredOnceMu sync.Mutex
	discoveredOnce   = map[string]bool{} // base -> printed-already
)

// logDiscoveredModelOnce prints the auto-discovered model name once per
// endpoint per process. Each agent instance creates its own replayPoster,
// so the discovery line would otherwise repeat once per instance.
func logDiscoveredModelOnce(base, model string) {
	discoveredOnceMu.Lock()
	defer discoveredOnceMu.Unlock()
	if discoveredOnce[base] {
		return
	}
	discoveredOnce[base] = true
	fmt.Fprintf(os.Stderr, "[router-replay] %s: discovered model %q from the models listing\n", base, model)
}

// discoverModelName returns the first model id from the endpoint's models
// listing. Used when --model 'http://...,type=anthropic' omits model=... so
// the operator doesn't have to remember the exact id served by each port.
// Same attempt-then-fallback contract as the replay endpoint itself: honor
// the operator's path first (<base>/models), insert /v1 only on 404.
func discoverModelName(base string) (string, error) {
	id, status, err := fetchFirstModelID(base + "/models")
	if err != nil && status == http.StatusNotFound {
		id, _, err = fetchFirstModelID(base + "/v1/models")
	}
	return id, err
}

// fetchFirstModelID GETs an OpenAI-style models listing and returns the
// first id, the HTTP status (0 on transport error), and any error.
func fetchFirstModelID(url string) (string, int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return "", resp.StatusCode, fmt.Errorf("status %d from %s", resp.StatusCode, url)
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", resp.StatusCode, err
	}
	if len(body.Data) == 0 {
		return "", resp.StatusCode, fmt.Errorf("no models returned by %s", url)
	}
	return body.Data[0].ID, resp.StatusCode, nil
}

// endpointAttempts returns the URL(s) to try for a request: the latched
// endpoint once resolved, else primary then /v1 fallback.
func (p *replayPoster) endpointAttempts() []string {
	p.epMu.Lock()
	defer p.epMu.Unlock()
	if p.epResolved != "" {
		return []string{p.epResolved}
	}
	return []string{p.epPrimary, p.epFallback}
}

// Endpoint resolutions logged so far, PACKAGE-scoped: replayPoster is
// constructed per replay instance (per series), so a per-poster sync.Once
// printed "[router-replay] endpoint resolved" once per SERIES — thousands
// of lines per soak. Log once per distinct resolved endpoint instead; a
// genuine endpoint change still logs.
var (
	epLogMu   sync.Mutex
	epLogSeen = map[string]bool{}
)

// latchEndpoint records the first endpoint form that returned 2xx and logs
// the resolution once per distinct endpoint (process-wide). Concurrent
// first requests may each probe both forms in parallel — benign duplicate
// probes by design (no single-flight, so a high-concurrency launch is
// never serialized behind one resolver); the first success wins and later
// latches are no-ops.
func (p *replayPoster) latchEndpoint(url string) {
	p.epMu.Lock()
	if p.epResolved == "" {
		p.epResolved = url
		p.epFellBack = url == p.epFallback && p.epFallback != p.epPrimary
	}
	resolved, fellBack := p.epResolved, p.epFellBack
	p.epMu.Unlock()
	epLogMu.Lock()
	seen := epLogSeen[resolved]
	if !seen {
		epLogSeen[resolved] = true
	}
	epLogMu.Unlock()
	if !seen {
		note := ""
		if fellBack {
			note = " (fallback /v1 applied)"
		}
		fmt.Fprintf(os.Stderr, "[router-replay] endpoint resolved: %s%s\n", resolved, note)
	}
}

// sendOnce POSTs bodyBytes to url with the poster's auth/accept headers.
func (p *replayPoster) sendOnce(ctx context.Context, url string, bodyBytes []byte, stream bool) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiType == "anthropic" {
		httpReq.Header.Set("Accept", "application/json")
		httpReq.Header.Set("X-Api-Key", p.apiKey)
		httpReq.Header.Set("Anthropic-Version", "2023-06-01")
		if stream {
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	} else {
		// OpenAI-compatible: Bearer auth, no Anthropic-Version header.
		httpReq.Header.Set("Accept", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		if stream {
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	}
	return p.client.Do(httpReq)
}

// do issues one request and returns its metrics. Honors ctx for
// cancellation / deadline. Dispatches to the appropriate body builder
// and response parser based on p.apiType.
func (p *replayPoster) do(
	ctx context.Context,
	req RouterReplayRequest,
	docs string,
	turnIdx int,
	sessionID string,
	instanceID string,
	seriesNum int,
	st *autoState,
) RequestMetrics {
	startTime := time.Now()

	var bodyBytes []byte
	var canonical string
	var err error
	switch p.apiType {
	case "openai", "openai_vllm":
		bodyBytes, canonical, err = buildOpenAIChatCompletionsBody(req, docs, p.model, p.runID, p.outputRatio, p.forceOutput)
	default:
		bodyBytes, canonical, err = buildAnthropicMessagesBody(req, docs, p.model, p.runID, p.outputRatio, p.forceOutput)
	}
	if err != nil {
		return RequestMetrics{
			RequestNum:        int(st.totalCompleted.Load()) + 1,
			SeriesNum:         seriesNum,
			CycleNum:          turnIdx,
			SeriesGUID:        sessionID + ":" + instanceID,
			Error:             fmt.Errorf("build body: %w", err),
			TotalResponseTime: time.Since(startTime),
		}
	}

	var localCacheRatio float64
	if p.estimator != nil {
		localCacheRatio = p.estimator.Observe(canonical)
	}

	// Attempt-then-fallback: try the operator's path verbatim; ONLY a 404
	// (path-level error) triggers one retry of the same request with /v1
	// inserted. Transport errors and non-404 statuses return as-is.
	var resp *http.Response
	var attemptURL string
	attempts := p.endpointAttempts()
	for i, u := range attempts {
		resp, err = p.sendOnce(ctx, u, bodyBytes, req.Stream)
		if err != nil {
			return RequestMetrics{
				RequestNum:        int(st.totalCompleted.Load()) + 1,
				SeriesNum:         seriesNum,
				CycleNum:          turnIdx,
				SeriesGUID:        sessionID + ":" + instanceID,
				Error:             err,
				TotalResponseTime: time.Since(startTime),
			}
		}
		attemptURL = u
		if resp.StatusCode == http.StatusNotFound && i+1 < len(attempts) {
			io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
			resp.Body.Close()
			continue
		}
		break
	}
	defer resp.Body.Close()

	m := RequestMetrics{
		RequestNum: int(st.totalCompleted.Load()) + 1,
		SeriesNum:  seriesNum,
		CycleNum:   turnIdx,
		SeriesGUID: sessionID + ":" + instanceID,
	}

	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		m.Error = fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		m.TotalResponseTime = time.Since(startTime)
		return m
	}
	p.latchEndpoint(attemptURL)

	if p.apiType == "openai" || p.apiType == "openai_vllm" {
		if req.Stream {
			consumeOpenAISSE(resp.Body, startTime, &m)
		} else {
			consumeOpenAIPlain(resp.Body, startTime, &m)
		}
	} else {
		if req.Stream {
			consumeSSE(resp.Body, startTime, &m)
		} else {
			consumePlain(resp.Body, startTime, &m)
		}
	}

	if m.TotalResponseTime == 0 {
		m.TotalResponseTime = time.Since(startTime)
	}
	if m.Error == nil && strings.TrimSpace(m.Response) == "" && m.UsageData.OutputTokens.Count == 0 {
		m.IsEmpty = true
		m.Error = fmt.Errorf("empty response from model")
	}
	m.LocalCacheRatio = localCacheRatio
	return m
}

// buildReplayUsage builds ExecutionUsageData from a replayed provider response,
// enforcing the net-of-cache contract auto.go:283 depends on: InputTokens MUST
// EXCLUDE server-cached tokens, because auto.go reconstructs the full prompt as
// full = InputTokens + CachedTokens. Storing gross prompt tokens here double-counts
// the cached volume in the `in` denominator (deflating scached/in). ALL replay
// consumers (OpenAI + Anthropic, streaming + plain) MUST go through this one helper
// so the contract can never drift between paths again.
//
//	grossPrompt = total prompt tokens the server processed, INCLUDING cached
//	              (OpenAI: prompt_tokens; Anthropic: input + cache_read + cache_creation)
//	cached      = subset served from cache without recompute
//	              (OpenAI: prompt_tokens_details.cached_tokens; Anthropic: cache_read_input_tokens)
func buildReplayUsage(grossPrompt, cached, output int) tools.ExecutionUsageData {
	net := grossPrompt - cached
	if net < 0 {
		net = 0
	}
	return tools.ExecutionUsageData{
		InputTokens:  tools.TokenUsage{Count: net},
		OutputTokens: tools.TokenUsage{Count: output},
		CachedTokens: tools.TokenUsage{Count: cached},
		RequestCount: 1,
	}
}

// consumeSSE reads an Anthropic streaming response, capturing TTFT on the
// first content delta and usage numbers from message_start /
// message_delta events.
func consumeSSE(body io.Reader, startTime time.Time, m *RequestMetrics) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1<<16), 16<<20)
	var firstToken sync.Once
	var resp strings.Builder
	for scanner.Scan() {
		line := scanner.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data: "):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var ev map[string]json.RawMessage
		if err := json.Unmarshal(data, &ev); err != nil {
			continue
		}
		var evType string
		if t, ok := ev["type"]; ok {
			_ = json.Unmarshal(t, &evType)
		}
		switch evType {
		case "message_start":
			if msg, ok := ev["message"]; ok {
				var ms struct {
					Usage struct {
						InputTokens              int `json:"input_tokens"`
						CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
						CacheReadInputTokens     int `json:"cache_read_input_tokens"`
						OutputTokens             int `json:"output_tokens"`
					} `json:"usage"`
				}
				if err := json.Unmarshal(msg, &ms); err == nil {
					m.UsageData = buildReplayUsage(ms.Usage.InputTokens+ms.Usage.CacheReadInputTokens+ms.Usage.CacheCreationInputTokens, ms.Usage.CacheReadInputTokens, ms.Usage.OutputTokens)
				}
			}
		case "content_block_delta":
			if delta, ok := ev["delta"]; ok {
				var d struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
					Thinking    string `json:"thinking"`
				}
				if err := json.Unmarshal(delta, &d); err == nil {
					if d.Text != "" || d.Thinking != "" || d.PartialJSON != "" {
						firstToken.Do(func() {
							m.TimeToFirstToken = time.Since(startTime)
						})
					}
					resp.WriteString(d.Text)
					resp.WriteString(d.Thinking)
				}
			}
		case "message_delta":
			if usage, ok := ev["usage"]; ok {
				var u struct {
					InputTokens              int `json:"input_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					OutputTokens             int `json:"output_tokens"`
				}
				if err := json.Unmarshal(usage, &u); err == nil {
					if u.InputTokens != 0 {
						m.UsageData.InputTokens.Count = u.InputTokens + u.CacheCreationInputTokens // net of cache_read
						m.UsageData.CachedTokens.Count = u.CacheReadInputTokens
					}
					if u.OutputTokens != 0 {
						m.UsageData.OutputTokens.Count = u.OutputTokens
					}
				}
			}
			if delta, ok := ev["delta"]; ok {
				var d struct {
					StopReason string `json:"stop_reason"`
				}
				_ = json.Unmarshal(delta, &d)
			}
		case "error":
			if data, ok := ev["error"]; ok {
				m.Error = fmt.Errorf("anthropic stream error: %s", string(data))
			}
		}
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		if m.Error == nil {
			m.Error = err
		}
	}
	m.TotalResponseTime = time.Since(startTime)
	m.Response = resp.String()
}

// ---- OpenAI SSE consumption ----

// consumeOpenAISSE reads an OpenAI-compatible streaming chat/completions
// response (SSE with data: {} lines). Captures TTFT on the first content
// delta and usage from the final chunk carrying the "usage" field.
//
// OpenAI SSE format (standard):
//
//	data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
//	data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
//	data: [DONE]
//
// Some servers (sglang) omit the usage field entirely; we tolerate that
// gracefully.
func consumeOpenAISSE(body io.Reader, startTime time.Time, m *RequestMetrics) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1<<16), 16<<20)
	var firstToken sync.Once
	var resp strings.Builder

	for scanner.Scan() {
		line := scanner.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data: "):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"` // vLLM uses "reasoning"
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens        int `json:"prompt_tokens"`
				CompletionTokens    int `json:"completion_tokens"`
				TotalTokens         int `json:"total_tokens"`
				PromptTokensDetails *struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details,omitempty"`
			} `json:"usage,omitempty"`
		}
		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}

		// Capture usage from the final chunk.
		if chunk.Usage != nil {
			cached := 0
			if chunk.Usage.PromptTokensDetails != nil {
				cached = chunk.Usage.PromptTokensDetails.CachedTokens
			}
			m.UsageData = buildReplayUsage(chunk.Usage.PromptTokens, cached, chunk.Usage.CompletionTokens)
		}

		// Capture content delta & TTFT.
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			content := delta.Content
			reasoning := delta.ReasoningContent
			if reasoning == "" {
				reasoning = delta.Reasoning
			}
			if content != "" || reasoning != "" {
				firstToken.Do(func() {
					m.TimeToFirstToken = time.Since(startTime)
				})
			}
			resp.WriteString(content)
			resp.WriteString(reasoning)
		}
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		if m.Error == nil {
			m.Error = err
		}
	}
	m.TotalResponseTime = time.Since(startTime)
	m.Response = resp.String()
}

// consumeOpenAIPlain reads a non-streaming OpenAI chat/completions response.
func consumeOpenAIPlain(body io.Reader, startTime time.Time, m *RequestMetrics) {
	b, err := io.ReadAll(body)
	if err != nil {
		m.Error = err
		return
	}
	m.TimeToFirstToken = time.Since(startTime)
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			TotalTokens         int `json:"total_tokens"`
			PromptTokensDetails *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details,omitempty"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		m.Error = err
		return
	}
	if len(resp.Choices) > 0 {
		m.Response = resp.Choices[0].Message.Content
	}
	cached := 0
	if resp.Usage.PromptTokensDetails != nil {
		cached = resp.Usage.PromptTokensDetails.CachedTokens
	}
	m.UsageData = buildReplayUsage(resp.Usage.PromptTokens, cached, resp.Usage.CompletionTokens)
}

// consumePlain reads a non-streaming Anthropic response.
func consumePlain(body io.Reader, startTime time.Time, m *RequestMetrics) {
	b, err := io.ReadAll(body)
	if err != nil {
		m.Error = err
		return
	}
	m.TimeToFirstToken = time.Since(startTime)
	var resp struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		m.Error = err
		return
	}
	var sb strings.Builder
	for _, c := range resp.Content {
		if c.Type == "text" {
			sb.WriteString(c.Text)
		}
	}
	m.Response = sb.String()
	m.UsageData = buildReplayUsage(resp.Usage.InputTokens+resp.Usage.CacheReadInputTokens+resp.Usage.CacheCreationInputTokens, resp.Usage.CacheReadInputTokens, resp.Usage.OutputTokens)
}

// dryRunDurations returns the synthetic time-to-first-token (prefill) and total
// response time for a request, given its cold/warm input token split and output
// token count, at the configured per-second rates.
// A rate <= 0 OR a token count <= 0 contributes 0 duration.
func dryRunDurations(coldTokens, warmTokens, outputTokens, coldTPS, warmTPS, outputTPS int) (ttft, total time.Duration) {
	if coldTPS > 0 && coldTokens > 0 {
		ttft += time.Duration(float64(coldTokens) / float64(coldTPS) * float64(time.Second))
	}
	if warmTPS > 0 && warmTokens > 0 {
		ttft += time.Duration(float64(warmTokens) / float64(warmTPS) * float64(time.Second))
	}
	total = ttft
	if outputTPS > 0 && outputTokens > 0 {
		total += time.Duration(float64(outputTokens) / float64(outputTPS) * float64(time.Second))
	}
	return
}

// dryDo returns a synthetic RequestMetrics for dry-run mode without making any
// HTTP requests. It computes TTFT and total duration from the cold/warm token
// split (derived from the cache estimator ratio) and output tokens using the
// configured per-second rates, then sleeps for that duration (ctx-aware) to
// simulate real passage of time.
func (p *replayPoster) dryDo(
	ctx context.Context,
	req RouterReplayRequest,
	docs string,
	turnIdx int,
	sessionID string,
	instanceID string,
	seriesNum int,
	st *autoState,
) RequestMetrics {
	// Build canonical string for estimator and compute ratio.
	var canonical string
	switch p.apiType {
	case "openai", "openai_vllm":
		_, canonical, _ = buildOpenAIChatCompletionsBody(req, docs, p.model, p.runID, p.outputRatio, p.forceOutput)
	default:
		_, canonical, _ = buildAnthropicMessagesBody(req, docs, p.model, p.runID, p.outputRatio, p.forceOutput)
	}
	var ratio float64
	if p.estimator != nil {
		ratio = p.estimator.Observe(canonical)
	}

	total := req.InputTokens
	full := int64(total) + int64(req.CacheReadTokens)
	warm := int64(float64(full)*ratio + 0.5)
	if warm > full {
		warm = full
	}
	cold := int(full) - int(warm)
	if cold < 0 {
		cold = 0
	}
	cachedForTiming := int(warm)
	ttft, totalDur := dryRunDurations(cold, cachedForTiming, req.OutputTokens, p.dryRates.coldTPS, p.dryRates.warmTPS, p.dryRates.outputTPS)

	select {
	case <-time.After(totalDur):
	case <-ctx.Done():
	}

	return RequestMetrics{
		RequestNum:        int(st.totalCompleted.Load()) + 1,
		SeriesNum:         seriesNum,
		CycleNum:          turnIdx,
		SeriesGUID:        sessionID + ":" + instanceID,
		TimeToFirstToken:  ttft,
		TotalResponseTime: totalDur,
		LocalCacheRatio:   ratio,
		UsageData: tools.ExecutionUsageData{
			InputTokens:  tools.TokenUsage{Count: total},
			OutputTokens: tools.TokenUsage{Count: req.OutputTokens},
			CachedTokens: tools.TokenUsage{Count: 0},
			RequestCount: 1,
		},
		Error:   nil,
		IsEmpty: false,
	}
}
