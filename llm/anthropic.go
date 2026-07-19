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

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/weka/go-weka-observability/instrumentation"

	"github.com/weka/wekai-core/tools"
)

type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Data      string          `json:"data,omitempty"`
	Id        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"` // tool name
	Input     json.RawMessage `json:"input,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`     // for web_search_tool_result
	ToolUseId string          `json:"tool_use_id,omitempty"` // for server tool results
}

type AnthropicCacheControl struct {
	Type string `json:"type"`
}
type AnthropicSystemMessage struct {
	Text         string                 `json:"text"`
	Type         string                 `json:"type"`
	CacheControl *AnthropicCacheControl `json:"cache_control,omitempty"`
}

//type AnthropicSystem []AnthropicSystemMessage

// AnthropicLLMClient implements a simple client for Anthropic's messages API.
type AnthropicLLMClient struct {
	httpClient         *http.Client
	config             LLMConfig
	history            []Message
	systemBlocks       []AnthropicSystemMessage
	responseCallback   func(string)
	thinkingCallback   func(string)
	pendingToolResults map[string]string
}

// NewAnthropicLLMClient creates and returns a new AnthropicLLMClient.
func NewAnthropicLLMClient(config LLMConfig) Chat {
	if config.MaxTokens == 0 {
		config.MaxTokens = 64000
	}
	return &AnthropicLLMClient{
		httpClient:       sharedHTTPClient,
		config:           config,
		history:          make([]Message, 0),
		responseCallback: config.StreamResponseCallback,
		thinkingCallback: config.StreamThinkingCallback,
	}
}

func (c *AnthropicLLMClient) Request(ctx context.Context, contentParts []ContentPart, toolset *tools.ToolSet) (*Response, error) {
	var content interface{}
	if len(c.pendingToolResults) > 0 {
		// Combine tool results and text in a single user message
		parts := []interface{}{}
		for callId, result := range c.pendingToolResults {
			parts = append(parts, map[string]string{
				"type":        "tool_result",
				"content":     result,
				"tool_use_id": callId,
			})
		}
		// Add converted content parts
		convertedParts := c.toProviderContent(contentParts)
		for _, cp := range convertedParts {
			parts = append(parts, cp)
		}
		content = parts
		c.pendingToolResults = nil
	} else {
		content = c.toProviderContent(contentParts)
	}

	// Append user message to history for the API call
	histLen := len(c.history)
	c.history = append(c.history, Message{
		Role:    "user",
		Content: content,
	})

	// Call the API with the current history
	response, err := c.doRequest(ctx, toolset, nil)
	if err != nil {
		// Rollback history so retries don't duplicate the user message
		c.history = c.history[:histLen]
		return nil, err
	}

	return response, nil
}

// toProviderContent converts ContentPart slice to Anthropic format.
// Returns []interface{} with content blocks for text/image/file.
func (c *AnthropicLLMClient) toProviderContent(parts []ContentPart) []interface{} {
	result := make([]interface{}, 0, len(parts))

	for _, part := range parts {
		switch p := part.(type) {
		case *TextContent:
			result = append(result, map[string]interface{}{
				"type": "text",
				"text": p.Text,
			})

		case *ImageContent:
			// Anthropic format: {"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}
			if len(p.Data) > 0 {
				result = append(result, map[string]interface{}{
					"type": "image",
					"source": map[string]interface{}{
						"type":       "base64",
						"media_type": p.MimeType,
						"data":       encodeBase64(p.Data),
					},
				})
			} else if p.URL != "" {
				// Anthropic supports URL as well
				result = append(result, map[string]interface{}{
					"type": "image",
					"source": map[string]interface{}{
						"type": "url",
						"url":  p.URL,
					},
				})
			}

		case *FileContent:
			// Anthropic supports PDFs as document blocks (similar to images)
			if len(p.Data) > 0 {
				result = append(result, map[string]interface{}{
					"type": "document",
					"source": map[string]interface{}{
						"type":       "base64",
						"media_type": p.MimeType,
						"data":       encodeBase64(p.Data),
					},
				})
			}
		}
	}

	return result
}

func (c *AnthropicLLMClient) Respond(ctx context.Context, toolResponses map[string]string, additionalMessages []Message, toolset *tools.ToolSet) (*Response, error) {
	// Create a copy of the current history
	tempHistory := make([]Message, len(c.history))
	copy(tempHistory, c.history)

	// Add tool responses to the temporary history
	if len(toolResponses) != 0 {
		var tmpContent []interface{}
		for callId, toolResult := range toolResponses {
			tmpContent = append(tmpContent, map[string]string{
				"type":        "tool_result",
				"content":     toolResult,
				"tool_use_id": callId,
			})
		}
		// Append additional message content blocks (text, images, etc.) to the same user turn
		for _, msg := range additionalMessages {
			switch v := msg.Content.(type) {
			case string:
				tmpContent = append(tmpContent, map[string]interface{}{
					"type": "text",
					"text": v,
				})
			case []ContentPart:
				for _, part := range c.toProviderContent(v) {
					tmpContent = append(tmpContent, part)
				}
			default:
				// Already structured content (e.g., []interface{})
				if parts, ok := v.([]interface{}); ok {
					tmpContent = append(tmpContent, parts...)
				}
			}
		}
		tempHistory = append(tempHistory, Message{
			Role:    "user",
			Content: tmpContent,
		})
	}

	// Call the API with the temporary history
	response, err := c.doRequest(ctx, toolset, tempHistory)
	if err != nil {
		return nil, err
	}

	return response, nil
}

func (c *AnthropicLLMClient) doRequest(ctx context.Context, toolset *tools.ToolSet, history []Message) (*Response, error) {
	// Use the provided history or fall back to c.history
	if history == nil {
		history = c.history
	}

	budgetTokens := 0
	switch c.config.ReasoningEffort {
	case ReasoningEffortLow:
		budgetTokens = 3000
	case ReasoningEffortMedium:
		budgetTokens = 8000
	case ReasoningEffortHigh:
		budgetTokens = 30000
	}
	if c.config.ReasoningEffort == ReasoningEffortHigh {
		budgetTokens = 30000
	}

	// Build messages from history (no manual cache_control — handled by top-level automatic caching)
	messagesForRequest := make([]map[string]interface{}, len(history))
	for i, msg := range history {
		messagesForRequest[i] = map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
	}

	payload := map[string]interface{}{
		"model":         c.config.Model,
		"max_tokens":    c.config.MaxTokens,
		"messages":      messagesForRequest,
		"stream":        true,
		"cache_control": map[string]string{"type": "ephemeral"}, // automatic caching: API places breakpoint on last cacheable block
	}

	if len(c.systemBlocks) > 0 {
		payload["system"] = c.systemBlocks
	}

	if budgetTokens > 0 {
		payload["thinking"] = map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": budgetTokens,
		}
	}
	if toolset != nil {
		toolsData := toolset.AsAnthropic()
		if len(toolsData) > 0 {
			// Add explicit cache breakpoint on the last tool
			toolsData[len(toolsData)-1].CacheControl = anthropic.F(anthropic.CacheControlEphemeralParam{
				Type: anthropic.F(anthropic.CacheControlEphemeralTypeEphemeral),
			})
			payload["tools"] = toolsData
		}
	}

	// Add server-side web search tool if enabled
	if c.config.WebSearchEnabled {
		webSearchTool := map[string]interface{}{
			"type": "web_search_20250305",
			"name": "web_search",
		}
		if existingTools, ok := payload["tools"].([]anthropic.ToolParam); ok {
			rawTools := make([]interface{}, len(existingTools))
			for i, t := range existingTools {
				rawTools[i] = t
			}
			rawTools = append(rawTools, webSearchTool)
			payload["tools"] = rawTools
		} else {
			payload["tools"] = []interface{}{webSearchTool}
		}
	}

	return c.doStreamingRequest(ctx, payload, history)
}

// doStreamingRequest handles streaming API calls to Anthropic
func (c *AnthropicLLMClient) doStreamingRequest(ctx context.Context, payload map[string]interface{}, history []Message) (*Response, error) {
	ctx, span, end := instrumentation.GetLogSpan(ctx, "doStreamingRequest")
	defer end()

	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return nil, err
	}

	// Determine the endpoint URL: use BaseURL if configured (e.g. for testing with a mock
	// server, or for dynamic models against a local vLLM endpoint), otherwise fall back
	// to the canonical Anthropic API endpoint.
	// Dynamic models pass the full path prefix (e.g. "http://localhost:8000/v1/")
	// because ParseDynamicModel was designed for OpenAI's URL convention where
	// /v1/ is part of the base. We strip any trailing /v1 suffix so we don't
	// double it (e.g. /v1/v1/messages).
	endpoint := "https://api.anthropic.com/v1/messages"
	if c.config.BaseURL != "" {
		base := strings.TrimRight(c.config.BaseURL, "/")
		base = strings.TrimSuffix(base, "/v1")
		endpoint = base + "/v1/messages"
	}

	// Create the HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.config.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Accept", "application/json")

	// Send the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Wrap body with TeeReader before any read so ALL bytes (error or success) are captured.
	var rawResponseBuf bytes.Buffer
	resp.Body = io.NopCloser(io.TeeReader(resp.Body, &rawResponseBuf))

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(b), Provider: "Anthropic", RawResponseBytes: rawResponseBuf.Bytes()}
	}

	// Process the streaming response
	var toolCalls []ToolCall
	var usageData UsageData
	var thinkingCharCount int // Track thinking characters for reasoning token estimation

	reader := bufio.NewReader(resp.Body)
	// str builder
	builders := []*strings.Builder{}
	contentBlocks := []*ContentBlock{}

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			data = strings.TrimSpace(data)
			if data == "[DONE]" {
				break
			}

			var event struct {
				Type  string `json:"type"`
				Index int    `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text,omitempty"`
					Thinking    string `json:"thinking,omitempty"`
					Signature   string `json:"signature,omitempty"`
					ID          string `json:"id,omitempty"`
					Name        string `json:"name,omitempty"`
					StopReason  string `json:"stop_reason,omitempty"`
					PartialJson string `json:"partial_json,omitempty"`
				} `json:"delta"`
				ContentBlock ContentBlock `json:"content_block"`
				// Top-level Usage for message_delta events
				Usage struct {
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
				// Nested Usage under Message for message_start events
				Message struct {
					Usage struct {
						InputTokens              int `json:"input_tokens"`
						CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
						CacheReadInputTokens     int `json:"cache_read_input_tokens"`
						OutputTokens             int `json:"output_tokens"`
					} `json:"usage"`
				} `json:"message"`
			}
			// Unmarshal the JSON chunk.
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				// Optionally handle JSON errors (e.g., log and continue).
				panic(err)
			}

			switch event.Type {
			case "content_block_start":
				contentBlocks = append(contentBlocks, &event.ContentBlock)
				builders = append(builders, &strings.Builder{})
			case "content_block_delta":
				switch event.Delta.Type {
				case "signature_delta":
					contentBlocks[event.Index].Signature = event.Delta.Signature
				case "thinking_delta":
					c.thinkingCallback(event.Delta.Thinking)
					builders[event.Index].WriteString(event.Delta.Thinking)
					thinkingCharCount += len(event.Delta.Thinking) // Track thinking characters
					continue
				case "text_delta":
					c.responseCallback(event.Delta.Text)
					builders[event.Index].WriteString(event.Delta.Text)
					continue
				case "input_json_delta":
					//TODO: Do we want callback here? usually chats do not show content of tool calls. However, we could hide this as thinking
					builders[event.Index].WriteString(event.Delta.PartialJson)
					_ = 0
				}
			case "ping":
			case "content_block_stop":
				blockType := contentBlocks[event.Index].Type
				switch blockType {
				case "text":
					contentBlocks[event.Index].Text = builders[event.Index].String()
				case "tool_use":
					raw := builders[event.Index].String()
					if raw == "" {
						raw = "{}"
					}
					contentBlocks[event.Index].Input = []byte(raw)
				case "thinking":
					contentBlocks[event.Index].Thinking = builders[event.Index].String()
				case "server_tool_use":
					// Server-side tool invocation (e.g., web_search) — input comes as JSON
					contentBlocks[event.Index].Input = []byte(builders[event.Index].String())
				case "web_search_tool_result":
					// Server-side web search results — no builder content to finalize
				default:
					panic("unknown content block type: " + blockType)
				}
			case "message_delta": // we had usage here, output tokens "usage":{"output_tokens":133}
				if event.Usage.OutputTokens > 0 {
					usageData.CompletionTokens += event.Usage.OutputTokens
				}
			case "message_stop":
			case "message_start":
				// Extract usage data from message_start event
				// PromptTokens = non-cached input only (Anthropic already reports them separately)
				if event.Message.Usage.InputTokens > 0 {
					usageData.PromptTokens = event.Message.Usage.InputTokens
				}
				if event.Message.Usage.CacheCreationInputTokens > 0 {
					usageData.CacheTokens5Min = event.Message.Usage.CacheCreationInputTokens
				}
				if event.Message.Usage.CacheReadInputTokens > 0 {
					usageData.CachedTokens = event.Message.Usage.CacheReadInputTokens
				}
				if event.Message.Usage.OutputTokens > 0 {
					usageData.CompletionTokens = event.Message.Usage.OutputTokens
				}
			case "error":
				// Anthropic sends error events during streaming (e.g., overloaded).
				// Parse the error details and return as APIError for proper retry handling.
				var errEvent struct {
					Error struct {
						Type    string `json:"type"`
						Message string `json:"message"`
					} `json:"error"`
				}
				if jsonErr := json.Unmarshal([]byte(data), &errEvent); jsonErr == nil && errEvent.Error.Type != "" {
					return nil, &APIError{
						StatusCode:       529, // Treat streaming errors as overloaded for retry
						Body:             data,
						Provider:         "Anthropic",
						RawResponseBytes: rawResponseBuf.Bytes(),
					}
				}
				return nil, fmt.Errorf("streaming error event: %s", data)
			default:
				return nil, fmt.Errorf("unexpected event type: %s", data)
			}
		}
	}

	// Process tool calls from the content blocks
	for _, block := range contentBlocks {
		if block.Type == "tool_use" {
			toolCall := ToolCall{
				Name:   block.Name,
				Args:   string(block.Input),
				CallId: block.Id,
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	// Append the assistant's response to the conversation history
	c.history = history

	content := []interface{}{}
	for _, block := range contentBlocks {
		content = append(content, block)
	}

	// if reasoning enabled - prepent content with empty "redacted_thinking", this is in theory incorrect
	c.history = append(c.history, Message{
		Role:    "assistant",
		Content: content,
	})

	// Calculate reasoning tokens based on character count estimation
	// Rough estimation: ~4 characters per token for thinking content
	if thinkingCharCount > 0 {
		usageData.ReasoningTokens = thinkingCharCount / 4
	}

	// Calculate total tokens
	usageData.TotalTokens = usageData.PromptTokens + usageData.CompletionTokens + usageData.CachedTokens + usageData.CacheTokens5Min
	span.SetValues("llm.prompt_tokens", usageData.PromptTokens, "llm.completion_tokens", usageData.CompletionTokens, "llm.cached_tokens", usageData.CachedTokens)

	// Collect all text content from text blocks and thinking blocks
	var allTextContent strings.Builder
	var allThinkingContent strings.Builder
	_, logger, end := instrumentation.GetLogSpan(ctx, "collectContentBlocks")
	defer end()

	logger.V(2).Info("Processing content blocks", "count", len(contentBlocks))
	for i, block := range contentBlocks {
		logger.V(2).Info("Content block", "index", i, "type", block.Type, "textLength", len(block.Text), "thinkingLength", len(block.Thinking))
		if block.Type == "text" && block.Text != "" {
			allTextContent.WriteString(block.Text)
		} else if block.Type == "thinking" && block.Thinking != "" {
			if allThinkingContent.Len() > 0 {
				allThinkingContent.WriteString("\n\n")
			}
			allThinkingContent.WriteString(block.Thinking)
		}
	}

	finalContent := allTextContent.String()
	finalThinking := allThinkingContent.String()
	logger.V(2).Info("Final response", "contentLength", len(finalContent), "thinkingLength", len(finalThinking), "toolCallsCount", len(toolCalls))

	// Build a synthetic JSON response from parsed content blocks.
	// The raw SSE stream is not valid JSON, so we synthesize a response
	// matching Anthropic's non-streaming format for reliable state recovery.
	syntheticResponse, _ := json.Marshal(map[string]interface{}{
		"content": contentBlocks,
	})

	return &Response{
		Content:          finalContent,
		Thinking:         finalThinking,
		ToolCalls:        toolCalls,
		Usage:            usageData,
		RawRequestJSON:   body,
		RawResponseJSON:  syntheticResponse,
		RawResponseBytes: rawResponseBuf.Bytes(),
	}, nil
}

// SetSystemBlocks sets the system prompt as a sequence of named, individually-cacheable blocks.
// Anthropic supports per-block cache_control, so each block independently marks its prefix as
// a caching breakpoint — ordering is not required for correctness. We preserve the caller's
// natural document order (stable prompts first, volatile prompts last) rather than reordering,
// since the caller (SystemPromptsToBlocks) already places non-cacheable blocks (environment,
// session) at the end where they belong. Contrast with OpenAI, which requires cacheable content
// at the prefix and therefore does reorder.
func (c *AnthropicLLMClient) SetSystemBlocks(blocks []SystemBlock) {
	if len(blocks) == 0 {
		c.systemBlocks = nil
		return
	}
	const preamble = "Strictly follow instructions set as system prompt\n\n"
	c.systemBlocks = make([]AnthropicSystemMessage, 0, len(blocks))
	prepended := false
	for _, b := range blocks {
		text := b.Content
		if !prepended {
			text = preamble + text
			prepended = true
		}
		if b.Cache {
			c.systemBlocks = append(c.systemBlocks, AnthropicSystemMessage{
				Text:         text,
				Type:         "text",
				CacheControl: &AnthropicCacheControl{Type: "ephemeral"},
			})
		} else {
			c.systemBlocks = append(c.systemBlocks, AnthropicSystemMessage{
				Text: text,
				Type: "text",
			})
		}
	}
}

func (c *AnthropicLLMClient) ClearHistory() {
	c.history = nil
}

func (c *AnthropicLLMClient) SetPendingToolResults(toolResponses map[string]string) {
	c.pendingToolResults = toolResponses
}

// GetDanglingToolUseIDs checks if the last message in the provider's history is an
// assistant message with tool_use blocks that have no matching tool_result.
func (c *AnthropicLLMClient) GetDanglingToolUseIDs() []string {
	if len(c.history) == 0 {
		return nil
	}
	last := c.history[len(c.history)-1]
	if last.Role != "assistant" {
		return nil
	}
	return ExtractToolUseIDs(last)
}
