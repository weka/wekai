package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai/tools"
)

type ChatMessage struct {
	Role       string `json:"role"`
	ToolCallId string `json:"tool_call_id,omitempty"`
	Content    string `json:"content,omitempty"`
}

type OpenAIFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIFunctionCall struct {
	Id       string         `json:"id"`
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenaiMessage struct {
	Role         string               `json:"role"`
	Content      interface{}          `json:"content,omitempty"`
	ToolCallId   string               `json:"tool_call_id,omitempty"`
	FunctionCall []OpenAIFunctionCall `json:"tool_calls,omitempty"`
}

type OpenAiChat struct {
	client             *http.Client
	config             LLMConfig
	history            []OpenaiMessage
	responseCallback   func(string)
	thinkingCallback   func(string)
	pendingToolResults map[string]string
}

func NewOpenAiLLMClient(llmConfig LLMConfig) Chat {
	if llmConfig.MaxTokens == 0 {
		llmConfig.MaxTokens = 32000
	}
	if llmConfig.BaseURL == "" {
		llmConfig.BaseURL = "https://api.openai.com/v1/"
	}
	// API key must be provided through the constructor
	if llmConfig.APIKey == "" {
		panic("OpenAI API key is required in LLMConfig")
	}

	return &OpenAiChat{
		client:           sharedHTTPClient,
		config:           llmConfig,
		history:          make([]OpenaiMessage, 0),
		responseCallback: llmConfig.StreamResponseCallback,
		thinkingCallback: llmConfig.StreamThinkingCallback,
	}
}

func (l *OpenAiChat) Request(ctx context.Context, content []ContentPart, toolset *tools.ToolSet) (*Response, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "requestLlmMessage")
	defer end()
	logger.Debug("Requesting OpenAI", "content_parts", len(content))

	if len(l.pendingToolResults) > 0 {
		for toolId, result := range l.pendingToolResults {
			l.history = append(l.history, OpenaiMessage{
				Role:       "tool",
				ToolCallId: toolId,
				Content:    result,
			})
		}
		l.pendingToolResults = nil
	}

	// Convert content parts to OpenAI format
	convertedContent := l.toProviderContent(content)

	l.history = append(l.history, OpenaiMessage{
		Role:    "user",
		Content: convertedContent,
	})

	return l.request(ctx, toolset, nil)
}

// toProviderContent converts ContentPart slice to OpenAI Chat format.
// For text-only content, returns a simple string.
// For multi-modal content, returns a JSON-compatible structure.
func (l *OpenAiChat) toProviderContent(parts []ContentPart) interface{} {
	// If only one text part, return as string for backward compatibility
	if len(parts) == 1 {
		if textPart, ok := parts[0].(*TextContent); ok {
			return textPart.Text
		}
	}

	// Multi-modal content: build array of content objects
	contentArray := make([]map[string]interface{}, 0, len(parts))

	for _, part := range parts {
		switch p := part.(type) {
		case *TextContent:
			contentArray = append(contentArray, map[string]interface{}{
				"type": "text",
				"text": p.Text,
			})

		case *ImageContent:
			// OpenAI Chat format: {"type":"image_url","image_url":{"url":"data:mime;base64,..."}}
			var imageURL string
			if p.URL != "" {
				imageURL = p.URL
			} else if len(p.Data) > 0 {
				// Encode as base64 data URL
				imageURL = fmt.Sprintf("data:%s;base64,%s", p.MimeType, encodeBase64(p.Data))
			}
			if imageURL != "" {
				contentArray = append(contentArray, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": imageURL,
					},
				})
			}

		case *FileContent:
			// OpenAI supports PDFs via base64 encoding in the same way as images
			if len(p.Data) > 0 {
				dataURL := fmt.Sprintf("data:%s;base64,%s", p.MimeType, encodeBase64(p.Data))
				contentArray = append(contentArray, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": dataURL,
					},
				})
			}
		}
	}

	return contentArray
}

func (l *OpenAiChat) Respond(ctx context.Context, toolResponses map[string]string, additionalMessages []Message, toolset *tools.ToolSet) (*Response, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "respondLlmMessage")
	defer end()

	// Check if the last message in history is from the assistant and has tool calls
	if len(l.history) == 0 {
		return nil, fmt.Errorf("no history to respond to")
	}

	lastMsg := l.history[len(l.history)-1]
	if lastMsg.Role != "assistant" {
		logger.Info("Last message is not from assistant", "role", lastMsg.Role)
		return nil, fmt.Errorf("last message is not from assistant")
	}

	// Create a new history with tool responses
	newHistory := make([]OpenaiMessage, len(l.history))
	copy(newHistory, l.history)

	// Add tool responses to history
	for toolId, toolResult := range toolResponses {
		newHistory = append(newHistory, OpenaiMessage{
			Role:       "tool",
			ToolCallId: toolId,
			Content:    toolResult,
		})
	}

	// Append additional messages as user messages
	for _, msg := range additionalMessages {
		userText := ""
		switch v := msg.Content.(type) {
		case string:
			userText = v
		case []ContentPart:
			userText = extractTextFromParts(v)
		}
		if userText != "" {
			newHistory = append(newHistory, OpenaiMessage{
				Role:    "user",
				Content: userText,
			})
		}
	}

	ret, err := l.request(ctx, toolset, newHistory)
	return ret, err
}

func (l *OpenAiChat) request(ctx context.Context, toolset *tools.ToolSet, historyOverride []OpenaiMessage) (*Response, error) {
	ctx, _, end := instrumentation.GetLogSpan(ctx, "requestLlmMessage")
	defer end()

	messages := l.history
	if len(historyOverride) != 0 {
		messages = historyOverride
	}

	// Build payload. (Include tools if provided and supported.)
	maxTokensKey := "max_completion_tokens"
	if l.config.UseCompatMaxTokens {
		maxTokensKey = "max_tokens"
	}
	payload := map[string]interface{}{
		"model":      l.config.Model,
		maxTokensKey: l.config.MaxTokens,
		"messages":   messages,
		"stream":     true,
		"stream_options": map[string]interface{}{
			"include_usage": true,
		},
	}
	if toolset != nil {
		payload["tools"] = toolset.AsOpenAi()
	}

	if l.config.ReasoningEffort != "" {
		payload["reasoning_effort"] = string(l.config.ReasoningEffort)
	}

	if l.config.Thinking != "" {
		payload["thinking"] = l.config.Thinking
	}

	// Include extra body parameters if provided (e.g., OpenRouter provider preferences)
	for key, value := range l.config.ExtraBodyParams {
		payload[key] = value
	}

	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", l.config.BaseURL+"chat/completions", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+l.config.APIKey)

	// Log the HTTP request
	LogHTTPRequest(ctx, req, bodyBytes)

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, err
	}

	// Wrap response body with logging wrapper for streaming responses
	resp.Body = NewLoggingResponseBody(ctx, resp.Body, resp.StatusCode, resp.Header)
	defer resp.Body.Close()

	// Wrap body with TeeReader before any read so ALL bytes (error or success) are captured.
	var rawResponseBuf bytes.Buffer
	resp.Body = io.NopCloser(io.TeeReader(resp.Body, &rawResponseBuf))

	if resp.StatusCode != http.StatusOK {
		responseBytes, _ := io.ReadAll(resp.Body)
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(responseBytes), Provider: "OpenAI", RawResponseBytes: rawResponseBuf.Bytes()}
	}

	response, err := l.processStreamingResponse(ctx, resp, messages)
	if err != nil {
		return nil, err
	}
	response.RawRequestJSON = bodyBytes

	// Build a synthetic JSON response from parsed content.
	// The raw SSE stream is not valid JSON, so we synthesize a response
	// matching OpenAI's non-streaming format for reliable state recovery.
	syntheticMsg := map[string]interface{}{
		"role":    "assistant",
		"content": response.Content,
	}
	if len(response.ToolCalls) > 0 {
		tcList := make([]map[string]interface{}, len(response.ToolCalls))
		for i, tc := range response.ToolCalls {
			tcList[i] = map[string]interface{}{
				"id":   tc.CallId,
				"type": "function",
				"function": map[string]interface{}{
					"name":      tc.Name,
					"arguments": tc.Args,
				},
			}
		}
		syntheticMsg["tool_calls"] = tcList
	}
	syntheticResponse, _ := json.Marshal(map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{"message": syntheticMsg},
		},
	})
	response.RawResponseJSON = syntheticResponse
	response.RawResponseBytes = rawResponseBuf.Bytes()
	return response, nil
}

type OpenAIStreamChoice struct {
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
	Delta        struct {
		Role             string                 `json:"role"`
		Content          string                 `json:"content"`
		ReasoningContent string                 `json:"reasoning_content"`
		Reasoning        string                 `json:"reasoning"` // vLLM uses "reasoning" instead of "reasoning_content"
		ToolCalls        []OpenAIStreamToolCall `json:"tool_calls"`
		Refusal          *string                `json:"refusal"`
	} `json:"delta"`
}

type OpenAIStreamToolCall struct {
	Index    int    `json:"index"`
	Id       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type OpenAIStreamResponse struct {
	Id                string               `json:"id"`
	Object            string               `json:"object"`
	Created           int64                `json:"created"`
	Model             string               `json:"model"`
	ServiceTier       string               `json:"service_tier"`
	SystemFingerprint string               `json:"system_fingerprint"`
	Choices           []OpenAIStreamChoice `json:"choices"`
	Usage             *OpenAIUsage         `json:"usage,omitempty"`
}

// OpenAIUsage represents the usage data from OpenAI API responses
type OpenAIUsage struct {
	PromptTokens      int                           `json:"prompt_tokens"`
	CompletionTokens  int                           `json:"completion_tokens"`
	TotalTokens       int                           `json:"total_tokens"`
	PromptDetails     *OpenAIPromptTokenDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionDetails *OpenAICompletionTokenDetails `json:"completion_tokens_details,omitempty"`
}

type OpenAIPromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens"`
}

type OpenAICompletionTokenDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens"`
	AudioTokens              int `json:"audio_tokens"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
}

func (l *OpenAiChat) processStreamingResponse(ctx context.Context, resp *http.Response, messages []OpenaiMessage) (*Response, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "processStreamingResponse")
	defer end()

	reader := bufio.NewReader(resp.Body)
	contentBuilder := strings.Builder{}  // content only (no reasoning)
	thinkingBuilder := strings.Builder{} // reasoning only
	var toolCalls []ToolCall
	var toolCallsMap = make(map[int]*ToolCall)
	var usageData UsageData

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var streamResp OpenAIStreamResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			logger.Error(err, "Error unmarshaling stream response", "data", data)
			return nil, fmt.Errorf("error unmarshaling stream response: %v", err)
		}

		// Process usage data if present
		// PromptTokens = non-cached input only (OpenAI includes cached in prompt_tokens, so subtract)
		if streamResp.Usage != nil {
			usageData.PromptTokens = streamResp.Usage.PromptTokens
			usageData.CompletionTokens = streamResp.Usage.CompletionTokens
			usageData.TotalTokens = streamResp.Usage.TotalTokens

			if streamResp.Usage.PromptDetails != nil {
				usageData.CachedTokens = streamResp.Usage.PromptDetails.CachedTokens
				usageData.PromptTokens -= usageData.CachedTokens
			}

			if streamResp.Usage.CompletionDetails != nil {
				usageData.ReasoningTokens = streamResp.Usage.CompletionDetails.ReasoningTokens
			}
		}

		// If there are no choices, this might be a usage-only chunk, so continue processing
		if len(streamResp.Choices) == 0 {
			continue
		}

		choice := streamResp.Choices[0]
		delta := choice.Delta

		// Process reasoning content — OpenAI uses "reasoning_content", vLLM uses "reasoning"
		reasoningChunk := delta.ReasoningContent
		if reasoningChunk == "" {
			reasoningChunk = delta.Reasoning
		}
		if reasoningChunk != "" {
			if l.thinkingCallback != nil {
				l.thinkingCallback(reasoningChunk)
			}
			thinkingBuilder.WriteString(reasoningChunk)
		}

		// Process content
		if delta.Content != "" {
			if l.responseCallback != nil {
				l.responseCallback(delta.Content)
			}
			contentBuilder.WriteString(delta.Content)
		}

		// Process tool calls
		if len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				toolCall, exists := toolCallsMap[tc.Index]
				if !exists {
					// New tool call
					toolCall = &ToolCall{
						CallId: tc.Id,
						Name:   tc.Function.Name,
						Args:   tc.Function.Arguments,
					}
					toolCallsMap[tc.Index] = toolCall
				} else {
					// Continuation of current tool call
					if tc.Function.Name != "" {
						toolCall.Name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						toolCall.Args += tc.Function.Arguments
					}
				}
			}
		}

		// Don't break on finish_reason anymore, continue until [DONE] or usage data found
		// This allows us to receive the final usage chunk
	}

	logger.SetValues("llm.prompt_tokens", usageData.PromptTokens, "llm.completion_tokens", usageData.CompletionTokens, "llm.cached_tokens", usageData.CachedTokens, "llm.reasoning_tokens", usageData.ReasoningTokens)

	// Convert map to slice
	for _, toolCall := range toolCallsMap {
		toolCalls = append(toolCalls, *toolCall)
	}

	// Create the final message to add to history
	content := contentBuilder.String()
	receivedMsg := OpenaiMessage{
		Role:    "assistant",
		Content: content,
	}

	// If we have tool calls but no content, set content to empty string
	// This matches the behavior in the example where content is null when tool_calls are present
	if len(toolCalls) > 0 && content == "" {
		receivedMsg.Content = ""
	}

	// Add tool calls to the assistant message
	if len(toolCalls) > 0 {
		functionCalls := make([]OpenAIFunctionCall, 0, len(toolCalls))
		for _, tc := range toolCalls {
			functionCalls = append(functionCalls, OpenAIFunctionCall{
				Id:   tc.CallId,
				Type: "function",
				Function: OpenAIFunction{
					Name:      tc.Name,
					Arguments: tc.Args,
				},
			})
		}
		receivedMsg.FunctionCall = functionCalls
	}

	// Add to history
	messages = append(messages, receivedMsg)
	l.history = messages

	// Drain any remaining data from the bufio.Reader AND the underlying response body
	// This is critical for HTTP connection pooling to work properly
	// The bufio.Reader may have buffered data, so we need to drain it first
	io.Copy(io.Discard, reader)
	// Then drain the underlying response body in case there's data that wasn't buffered
	io.Copy(io.Discard, resp.Body)

	// Create response
	response := &Response{
		Content:   content,
		Thinking:  thinkingBuilder.String(),
		ToolCalls: toolCalls,
		Usage:     usageData,
	}

	return response, nil
}

func (l *OpenAiChat) SetSystemBlocks(blocks []SystemBlock) {
	if len(blocks) == 0 {
		return
	}
	const preamble = "Strictly follow instructions set as system prompt\n\n"
	// Remove any existing system messages from the front of history
	start := 0
	for start < len(l.history) {
		if l.history[start].Role == "system" {
			start++
		} else {
			break
		}
	}
	l.history = l.history[start:]
	// Build new system messages: cacheable blocks first (better prefix caching), then non-cacheable
	var cacheable []OpenaiMessage
	var nonCacheable []OpenaiMessage
	prepended := false
	for _, b := range blocks {
		text := b.Content
		if !prepended {
			text = preamble + text
			prepended = true
		}
		msg := OpenaiMessage{Role: "system", Content: text}
		if b.Cache {
			cacheable = append(cacheable, msg)
		} else {
			nonCacheable = append(nonCacheable, msg)
		}
	}
	sysMsgs := append(cacheable, nonCacheable...)
	l.history = append(sysMsgs, l.history...)
}

func (l *OpenAiChat) ClearHistory() {
	l.history = nil
}

func (l *OpenAiChat) SetPendingToolResults(toolResponses map[string]string) {
	l.pendingToolResults = toolResponses
}

func (l *OpenAiChat) GetDanglingToolUseIDs() []string {
	return nil // OpenAI handles tool_use differently
}

// encodeBase64 encodes byte data to base64 string
func encodeBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

// Transcribe sends audio data to OpenAI Whisper API for transcription
func (l *OpenAiChat) Transcribe(ctx context.Context, audioData []byte, format string) (string, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "OpenAiChat.Transcribe")
	defer end()

	logger.Debug("Transcribing audio", "format", format, "size", len(audioData))

	// Create multipart form data
	var requestBody bytes.Buffer
	multipartWriter := multipart.NewWriter(&requestBody)

	// Add the audio file
	filename := fmt.Sprintf("audio.%s", format)
	fileWriter, err := multipartWriter.CreateFormFile("file", filename)
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := fileWriter.Write(audioData); err != nil {
		return "", fmt.Errorf("failed to write audio data: %w", err)
	}

	// Add the model field
	if err := multipartWriter.WriteField("model", "whisper-1"); err != nil {
		return "", fmt.Errorf("failed to write model field: %w", err)
	}

	// Close the multipart writer to finalize the request body
	if err := multipartWriter.Close(); err != nil {
		return "", fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create the HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", l.config.BaseURL+"audio/transcriptions", &requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+l.config.APIKey)

	// Log the HTTP request (without body since it's binary)
	LogHTTPRequest(ctx, req, []byte("[binary audio data]"))

	// Send the request
	resp, err := l.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}

	// Wrap response body with logging wrapper
	resp.Body = NewLoggingResponseBody(ctx, resp.Body, resp.StatusCode, resp.Header)
	defer resp.Body.Close()

	// Read the response body
	responseBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Whisper API error: status %d, body: %s", resp.StatusCode, string(responseBytes))
	}

	// Parse the response
	var result struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(responseBytes, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	logger.Debug("Transcription successful", "text_length", len(result.Text))
	return result.Text, nil
}
