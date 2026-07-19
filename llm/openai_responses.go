package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai-core/tools"
)

// ResponsesMessage represents a message in the Responses API format
type ResponsesMessage struct {
	Role    string                 `json:"role"`
	Content []ResponsesContentPart `json:"content,omitempty"`
}

type ResponsesContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// ResponsesToolCallOutput represents tool call output for the Responses API
type ResponsesToolCallOutput struct {
	Type   string `json:"type"`
	CallId string `json:"call_id"`
	Output string `json:"output"`
}

type OpenAiResponsesChat struct {
	client             *http.Client
	config             LLMConfig
	responseId         string   // Store the current response ID for chaining
	systemMessages     []string // Store system message blocks for automatic caching
	responseCallback   func(string)
	thinkingCallback   func(string)
	pendingToolResults map[string]string
}

func NewOpenAiResponsesLLMClient(llmConfig LLMConfig) Chat {
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

	return &OpenAiResponsesChat{
		client:           sharedHTTPClient,
		config:           llmConfig,
		responseCallback: llmConfig.StreamResponseCallback,
		thinkingCallback: llmConfig.StreamThinkingCallback,
	}
}

func (l *OpenAiResponsesChat) Request(ctx context.Context, content []ContentPart, toolset *tools.ToolSet) (*Response, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "requestLlmMessage")
	defer end()
	logger.Debug("Requesting OpenAI Responses API", "content_parts", len(content))

	// If we have pending tool results AND a response ID, send them along with the user message
	if len(l.pendingToolResults) > 0 && l.responseId != "" {
		var input []interface{}
		for toolId, result := range l.pendingToolResults {
			input = append(input, ResponsesToolCallOutput{
				Type:   "function_call_output",
				CallId: toolId,
				Output: result,
			})
		}
		input = append(input, ResponsesMessage{
			Role:    "user",
			Content: l.toProviderContent(content),
		})
		l.pendingToolResults = nil
		return l.request(ctx, toolset, input, l.responseId)
	}

	// If we have a previous response ID, use it for conversation chaining
	// This enables prompt caching as the API recognizes it as a continuation
	if l.responseId != "" {
		// For conversation continuation, just send the new user message
		input := []ResponsesMessage{
			{
				Role:    "user",
				Content: l.toProviderContent(content),
			},
		}
		return l.request(ctx, toolset, input, l.responseId)
	}

	// First request: include system message for caching
	var input []ResponsesMessage

	// Add system messages if set (each block as a separate system message for caching)
	for _, sysMsg := range l.systemMessages {
		input = append(input, ResponsesMessage{
			Role: "system",
			Content: []ResponsesContentPart{
				{
					Type: "input_text",
					Text: sysMsg,
				},
			},
		})
	}

	// Add user message
	input = append(input, ResponsesMessage{
		Role:    "user",
		Content: l.toProviderContent(content),
	})

	return l.request(ctx, toolset, input, "")
}

// toProviderContent converts ContentPart slice to OpenAI Responses API format.
// Returns []ResponsesContentPart with appropriate types for each content part.
func (l *OpenAiResponsesChat) toProviderContent(parts []ContentPart) []ResponsesContentPart {
	result := make([]ResponsesContentPart, 0, len(parts))

	for _, part := range parts {
		switch p := part.(type) {
		case *TextContent:
			result = append(result, ResponsesContentPart{
				Type: "input_text",
				Text: p.Text,
			})

		case *ImageContent:
			// OpenAI Responses format: {"type":"input_image","image_url":"data:..."}
			// Note: ResponsesContentPart would need an ImageURL field added
			var imageURL string
			if p.URL != "" {
				imageURL = p.URL
			} else if len(p.Data) > 0 {
				imageURL = fmt.Sprintf("data:%s;base64,%s", p.MimeType, encodeBase64(p.Data))
			}
			if imageURL != "" {
				// For now, encode as text-based representation
				// TODO: Extend ResponsesContentPart to support image_url field
				result = append(result, ResponsesContentPart{
					Type: "input_image",
					Text: imageURL, // Using Text field temporarily for image URL
				})
			}

		case *FileContent:
			// OpenAI Responses format: {"type":"input_file",...}
			if len(p.Data) > 0 {
				dataURL := fmt.Sprintf("data:%s;base64,%s", p.MimeType, encodeBase64(p.Data))
				result = append(result, ResponsesContentPart{
					Type: "input_file",
					Text: dataURL, // Using Text field temporarily for file data
				})
			}
		}
	}

	return result
}

func (l *OpenAiResponsesChat) Respond(ctx context.Context, toolResponses map[string]string, additionalMessages []Message, toolset *tools.ToolSet) (*Response, error) {
	ctx, _, end := instrumentation.GetLogSpan(ctx, "respondLlmMessage")
	defer end()

	if l.responseId == "" {
		return nil, fmt.Errorf("no previous response to continue from")
	}

	// Create function call outputs, possibly followed by additional messages
	var input []interface{}
	for toolId, toolResult := range toolResponses {
		input = append(input, ResponsesToolCallOutput{
			Type:   "function_call_output",
			CallId: toolId,
			Output: toolResult,
		})
	}

	// Append additional messages as user input items
	for _, msg := range additionalMessages {
		userText := ""
		switch v := msg.Content.(type) {
		case string:
			userText = v
		case []ContentPart:
			userText = extractTextFromParts(v)
		}
		if userText != "" {
			input = append(input, ResponsesMessage{
				Role: "user",
				Content: []ResponsesContentPart{
					{Type: "input_text", Text: userText},
				},
			})
		}
	}

	return l.request(ctx, toolset, input, l.responseId)
}

func (l *OpenAiResponsesChat) request(ctx context.Context, toolset *tools.ToolSet, input interface{}, previousResponseId string) (*Response, error) {
	ctx, _, end := instrumentation.GetLogSpan(ctx, "requestLlmMessage")
	defer end()

	// Build payload
	payload := map[string]interface{}{
		"model":  l.config.Model,
		"stream": true,
		"input":  input,
	}

	if l.config.MaxTokens > 0 {
		payload["max_output_tokens"] = l.config.MaxTokens
	}

	if previousResponseId != "" {
		payload["previous_response_id"] = previousResponseId
	}

	if toolset != nil {
		payload["tools"] = toolset.AsOpenAiResponses()
	}

	// Force tool use for models that require it (e.g. gpt-5.3-codex)
	if l.config.ForceToolCall && toolset != nil {
		payload["tool_choice"] = "required"
	}

	// Add server-side web search tool if enabled
	if l.config.WebSearchEnabled {
		webSearchTool := map[string]interface{}{"type": "web_search_preview"}
		if existingTools, ok := payload["tools"].([]map[string]interface{}); ok {
			payload["tools"] = append(existingTools, webSearchTool)
		} else {
			payload["tools"] = []map[string]interface{}{webSearchTool}
		}
	}

	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", l.config.BaseURL+"responses", bytes.NewBuffer(bodyBytes))
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
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(responseBytes), Provider: "OpenAI Responses", RawResponseBytes: rawResponseBuf.Bytes()}
	}

	response, err := l.processStreamingResponse(ctx, resp)
	if err != nil {
		return nil, err
	}
	response.RawRequestJSON = bodyBytes
	response.RawResponseBytes = rawResponseBuf.Bytes()

	// Build a synthetic JSON response matching the Responses API output format.
	// The raw SSE stream is not valid JSON, so we synthesize a response
	// for reliable state recovery and logging.
	outputItems := make([]map[string]interface{}, 0)
	if response.Content != "" {
		outputItems = append(outputItems, map[string]interface{}{
			"type": "message",
			"content": []map[string]interface{}{
				{"type": "output_text", "text": response.Content},
			},
		})
	}
	for _, tc := range response.ToolCalls {
		outputItems = append(outputItems, map[string]interface{}{
			"type":      "function_call",
			"call_id":   tc.CallId,
			"name":      tc.Name,
			"arguments": tc.Args,
		})
	}
	syntheticResponse, _ := json.Marshal(map[string]interface{}{
		"output": outputItems,
	})
	response.RawResponseJSON = syntheticResponse
	return response, nil
}

// Responses API streaming event types
type ResponsesStreamEvent struct {
	Type           string                `json:"type"`
	SequenceNumber int                   `json:"sequence_number"`
	Response       *ResponsesResponse    `json:"response,omitempty"`
	OutputIndex    int                   `json:"output_index,omitempty"`
	Item           *ResponsesOutputItem  `json:"item,omitempty"`
	ItemId         string                `json:"item_id,omitempty"`
	ContentIndex   int                   `json:"content_index,omitempty"`
	Part           *ResponsesContentPart `json:"part,omitempty"`
	Delta          string                `json:"delta,omitempty"`
	Text           string                `json:"text,omitempty"`
}

type ResponsesResponse struct {
	Id                 string                `json:"id"`
	Object             string                `json:"object"`
	CreatedAt          int64                 `json:"created_at"`
	Status             string                `json:"status"`
	Model              string                `json:"model"`
	Output             []ResponsesOutputItem `json:"output"`
	Usage              *ResponsesUsage       `json:"usage,omitempty"`
	PreviousResponseId *string               `json:"previous_response_id"`
}

type ResponsesOutputItem struct {
	Id      string                 `json:"id"`
	Type    string                 `json:"type"`
	Status  string                 `json:"status"`
	Content []ResponsesContentPart `json:"content"`
	Role    string                 `json:"role"`
	// Tool call fields
	CallId    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type ResponsesUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	OutputTokens        int                          `json:"output_tokens"`
	TotalTokens         int                          `json:"total_tokens"`
	InputTokensDetails  *ResponsesInputTokenDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *ResponsesOutputTokenDetails `json:"output_tokens_details,omitempty"`
}

type ResponsesInputTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ResponsesOutputTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

func (l *OpenAiResponsesChat) processStreamingResponse(ctx context.Context, resp *http.Response) (*Response, error) {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "processStreamingResponse")
	defer end()

	reader := bufio.NewReader(resp.Body)
	contentBuilder := strings.Builder{}
	var toolCalls []ToolCall
	var usageData UsageData
	var responseId string
	// Map to track item ID to call ID mapping
	itemToCallId := make(map[string]string)

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

		var streamEvent ResponsesStreamEvent
		if err := json.Unmarshal([]byte(data), &streamEvent); err != nil {
			logger.Error(err, "Error unmarshaling stream response", "data", data)
			continue // Continue processing other events
		}

		// Debug logging for events (can be disabled in production)
		//logger.Debug("Received event", "type", streamEvent.Type, "sequence", streamEvent.SequenceNumber)

		// Store response ID for chaining
		if streamEvent.Response != nil && streamEvent.Response.Id != "" {
			responseId = streamEvent.Response.Id
		}

		// Process different event types
		switch streamEvent.Type {
		case "response.output_text.delta":
			if streamEvent.Delta != "" {
				if l.responseCallback != nil {
					l.responseCallback(streamEvent.Delta)
				}
				contentBuilder.WriteString(streamEvent.Delta)
			}

		case "response.output_text.done":
			// Final text content is available in streamEvent.Text
			if streamEvent.Text != "" {
				// Ensure we have the complete text
				finalText := streamEvent.Text
				if contentBuilder.String() != finalText {
					contentBuilder.Reset()
					contentBuilder.WriteString(finalText)
				}
			}

		case "response.output_item.added":
			// Handle function calls when they are added
			if streamEvent.Item != nil && streamEvent.Item.Type == "function_call" {
				// Initialize a new tool call - arguments will be built up from delta events
				// Use the actual call ID if available, otherwise fall back to item ID
				callId := streamEvent.Item.CallId
				if callId == "" {
					callId = streamEvent.Item.Id
				}
				// Store the mapping for later matching
				itemToCallId[streamEvent.Item.Id] = callId
				logger.Debug("Adding function call", "item_id", streamEvent.Item.Id, "call_id", callId)
				toolCalls = append(toolCalls, ToolCall{
					CallId: callId, // Use the proper call ID for tool responses
					Name:   "",     // Will be filled when we get the function name
					Args:   "",     // Will be built up from delta events
				})
			}

		case "response.function_call_arguments.delta":
			// Build up function call arguments from delta events.
			// Use item_id to route the delta to the correct tool call when available;
			// fall back to the last tool call if item_id is absent.
			if len(toolCalls) > 0 && streamEvent.Delta != "" {
				if streamEvent.ItemId != "" {
					// Find the tool call whose CallId matches the item_id→callId mapping
					targetCallId := itemToCallId[streamEvent.ItemId]
					for i := range toolCalls {
						if toolCalls[i].CallId == targetCallId {
							toolCalls[i].Args += streamEvent.Delta
							break
						}
					}
				} else {
					// Fallback: append to the last tool call
					toolCalls[len(toolCalls)-1].Args += streamEvent.Delta
				}
			}

		case "response.function_call_arguments.done":
			// Function call arguments are complete
			// The function name and final arguments should be available in the completed response

		case "response.completed":
			// Process final response data
			if streamEvent.Response != nil {
				// Extract usage data
				// PromptTokens = non-cached input only (OpenAI includes cached in input_tokens, so subtract)
				if streamEvent.Response.Usage != nil {
					usageData.PromptTokens = streamEvent.Response.Usage.InputTokens
					usageData.CompletionTokens = streamEvent.Response.Usage.OutputTokens
					usageData.TotalTokens = streamEvent.Response.Usage.TotalTokens

					if streamEvent.Response.Usage.InputTokensDetails != nil {
						usageData.CachedTokens = streamEvent.Response.Usage.InputTokensDetails.CachedTokens
						usageData.PromptTokens -= usageData.CachedTokens
					}

					if streamEvent.Response.Usage.OutputTokensDetails != nil {
						usageData.ReasoningTokens = streamEvent.Response.Usage.OutputTokensDetails.ReasoningTokens
					}
				}

				// Extract function call details from output items and update existing tool calls
				for _, item := range streamEvent.Response.Output {
					if item.Type == "function_call" {
						logger.Debug("Processing function call in final response", "item_id", item.Id, "call_id", item.CallId, "name", item.Name)
						// Find the corresponding tool call using the item ID mapping
						expectedCallId := itemToCallId[item.Id]
						for i, existingCall := range toolCalls {
							if existingCall.CallId == expectedCallId {
								// Update with function name from the final response
								toolCalls[i].Name = item.Name
								// Arguments should already be built up from delta events
								logger.Debug("Updated function call", "call_id", existingCall.CallId, "name", item.Name)
								break
							}
						}
					}
				}
			}
		}
	}

	logger.SetValues("llm.prompt_tokens", usageData.PromptTokens, "llm.completion_tokens", usageData.CompletionTokens, "llm.cached_tokens", usageData.CachedTokens, "llm.reasoning_tokens", usageData.ReasoningTokens)

	// Store the response ID for potential chaining
	l.responseId = responseId

	finalContent := contentBuilder.String()

	// Create response
	response := &Response{
		Content:   finalContent,
		ToolCalls: toolCalls,
		Usage:     usageData,
	}

	// Debug logging
	logger.Debug("Final response", "content_length", len(finalContent), "tool_calls_count", len(toolCalls))

	return response, nil
}

func (l *OpenAiResponsesChat) SetSystemBlocks(blocks []SystemBlock) {
	if len(blocks) == 0 {
		l.systemMessages = nil
		return
	}
	const preamble = "Strictly follow instructions set as system prompt\n\n"
	// Cacheable blocks first (better prefix caching), then non-cacheable
	var cacheable []string
	var nonCacheable []string
	prepended := false
	for _, b := range blocks {
		text := b.Content
		if !prepended {
			text = preamble + text
			prepended = true
		}
		if b.Cache {
			cacheable = append(cacheable, text)
		} else {
			nonCacheable = append(nonCacheable, text)
		}
	}
	l.systemMessages = append(cacheable, nonCacheable...)
}

func (l *OpenAiResponsesChat) ClearHistory() {
	l.responseId = ""
}

func (l *OpenAiResponsesChat) SetPendingToolResults(toolResponses map[string]string) {
	l.pendingToolResults = toolResponses
}

func (l *OpenAiResponsesChat) GetDanglingToolUseIDs() []string {
	return nil
}
