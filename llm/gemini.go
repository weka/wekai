package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"

	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai-core/tools"
)

// GeminiLLMClient implements a simple client for Google's Gemini API with streaming support
type GeminiLLMClient struct {
	httpClient         *http.Client
	config             LLMConfig
	history            []GeminiMessage
	responseCallback   func(string)
	thinkingCallback   func(string)
	pendingToolResults map[string]string
	systemParts        []string
}

type GeminiStreamMessage struct {
	Candidates    []GeminiCandidate   `json:"candidates"`
	UsageMetadata GeminiUsageMetadata `json:"usageMetadata"`
	ModelVersion  string              `json:"modelVersion"`
}

type GeminiCandidate struct {
	Content      GeminiCandidateContent `json:"content"`
	FinishReason string                 `json:"finishReason"`
}

type GeminiCandidateContent struct {
	Parts []GeminiPart `json:"parts"`
	Role  string       `json:"role"`
}

type GeminiFunctionCall struct {
	Name string      `json:"name"`
	Args interface{} `json:"args"`
}

type GeminiUsageMetadata struct {
	PromptTokenCount        int                 `json:"promptTokenCount"`
	CandidatesTokenCount    int                 `json:"candidatesTokenCount"`
	TotalTokenCount         int                 `json:"totalTokenCount"`
	ThoughtsTokenCount      int                 `json:"thoughtsTokenCount"`
	CachedContentTokenCount int                 `json:"cachedContentTokenCount"`
	PromptTokensDetails     []GeminiTokenDetail `json:"promptTokensDetails"`
	CandidatesTokensDetails []GeminiTokenDetail `json:"candidatesTokensDetails"`
}

type GeminiTokenDetail struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

// GeminiMessage represents a message in the conversation
type GeminiMessage struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

// GeminiPart represents a part of a message
type GeminiPart struct {
	Text             string                `json:"text,omitempty"`
	InlineData       *GeminiInlineData     `json:"inlineData,omitempty"`
	FunctionCall     *GeminiFunctionCall   `json:"functionCall,omitempty"`
	FunctionResult   *GeminiFunctionResult `json:"functionResponse,omitempty"`
	ThoughtSignature string                `json:"thoughtSignature,omitempty"`
}

// GeminiInlineData represents inline data (images, files)
type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64 encoded
}

// GeminiFunctionCall represents a function call
// GeminiFunctionResult represents a function result
type GeminiFunctionResult struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response,omitempty"`
}

// GeminiFunctionDeclaration represents a function declaration
type GeminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// GeminiTool represents a tool
type GeminiTool struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"functionDeclarations"`
}

// NewGeminiLLMClient creates a new GeminiLLMClient with streaming support
func NewGeminiLLMClient(config LLMConfig) Chat {
	if config.MaxTokens == 0 {
		config.MaxTokens = 64000
	}
	return &GeminiLLMClient{
		httpClient:       sharedHTTPClient,
		config:           config,
		history:          make([]GeminiMessage, 0),
		responseCallback: config.StreamResponseCallback,
		thinkingCallback: config.StreamThinkingCallback,
	}
}

// toProviderContent converts ContentPart slice to Gemini Parts format
func (c *GeminiLLMClient) toProviderContent(parts []ContentPart) ([]GeminiPart, error) {
	geminiParts := make([]GeminiPart, 0, len(parts))

	for _, part := range parts {
		switch p := part.(type) {
		case *TextContent:
			geminiParts = append(geminiParts, GeminiPart{Text: p.Text})

		case *ImageContent:
			if len(p.Data) == 0 {
				return nil, fmt.Errorf("image content has no data")
			}
			encoded := base64.StdEncoding.EncodeToString(p.Data)
			geminiParts = append(geminiParts, GeminiPart{
				InlineData: &GeminiInlineData{
					MimeType: p.MimeType,
					Data:     encoded,
				},
			})

		case *FileContent:
			if len(p.Data) == 0 {
				return nil, fmt.Errorf("file content has no data")
			}
			encoded := base64.StdEncoding.EncodeToString(p.Data)
			geminiParts = append(geminiParts, GeminiPart{
				InlineData: &GeminiInlineData{
					MimeType: p.MimeType,
					Data:     encoded,
				},
			})

		default:
			return nil, fmt.Errorf("unsupported content type: %T", part)
		}
	}

	return geminiParts, nil
}

func (c *GeminiLLMClient) Request(ctx context.Context, contentParts []ContentPart, toolset *tools.ToolSet) (*Response, error) {
	parts, err := c.toProviderContent(contentParts)
	if err != nil {
		return nil, fmt.Errorf("failed to convert content: %w", err)
	}

	if len(c.pendingToolResults) > 0 {
		toolResultParts := []GeminiPart{}
		for name, result := range c.pendingToolResults {
			toolResultParts = append(toolResultParts, GeminiPart{
				FunctionResult: &GeminiFunctionResult{
					Name:     name,
					Response: map[string]interface{}{"result": result},
				},
			})
		}
		parts = append(toolResultParts, parts...)
		c.pendingToolResults = nil
	}

	c.history = append(c.history, GeminiMessage{Role: "user", Parts: parts})

	// Call the API with the temporary history
	response, err := c.doRequest(ctx, toolset, nil)
	if err != nil {
		return nil, err
	}

	return response, nil
}

func (c *GeminiLLMClient) Respond(ctx context.Context, toolResponses map[string]string, additionalMessages []Message, toolset *tools.ToolSet) (*Response, error) {
	// Create a copy of the current history
	tempHistory := make([]GeminiMessage, len(c.history))
	copy(tempHistory, c.history)

	// Add tool responses to the temporary history
	// According to Gemini API docs, function responses should have role "user"
	parts := []GeminiPart{}
	for toolName, toolResult := range toolResponses {
		parts = append(parts, GeminiPart{
			FunctionResult: &GeminiFunctionResult{
				Name:     toolName,
				Response: map[string]interface{}{"result": toolResult},
			},
		})
	}

	// Append additional messages as text parts in the same user turn
	for _, msg := range additionalMessages {
		userText := ""
		switch v := msg.Content.(type) {
		case string:
			userText = v
		case []ContentPart:
			userText = extractTextFromParts(v)
		}
		if userText != "" {
			parts = append(parts, GeminiPart{
				Text: userText,
			})
		}
	}

	if len(parts) > 0 {
		tempHistory = append(tempHistory, GeminiMessage{
			Role:  "user",
			Parts: parts,
		})
	}

	// Call the API with the temporary history
	response, err := c.doRequest(ctx, toolset, tempHistory)
	if err != nil {
		return nil, err
	}

	return response, nil
}

func (c *GeminiLLMClient) doRequest(ctx context.Context, toolset *tools.ToolSet, history []GeminiMessage) (*Response, error) {
	ctx, _, end := instrumentation.GetLogSpan(ctx, "requestLlmMessage")
	defer end()

	// Use the provided history or fall back to c.history
	if history == nil {
		history = c.history
	}

	// Build the request payload with the provided history
	generationConfig := map[string]interface{}{
		"maxOutputTokens": c.config.MaxTokens,
	}

	// Add thinkingConfig based on ReasoningEffort
	// Note: Gemini 2.5 Pro and 3 Pro cannot have thinking disabled
	// Gemini 2.5 Flash can have thinking disabled with thinkingBudget: 0
	switch c.config.ReasoningEffort {
	case ReasoningEffortNone:
		// Disable thinking for models that support it (e.g., Gemini 2.5 Flash)
		// For models that don't support disabling (Pro models), this will be ignored
		generationConfig["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget": 0,
		}
	case ReasoningEffortLow:
		generationConfig["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget": 1024,
		}
	case ReasoningEffortMedium:
		generationConfig["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget": 8192,
		}
	case ReasoningEffortHigh:
		generationConfig["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget": 24576,
		}
		// Default: don't set thinkingConfig, let the model use its default behavior
	}

	payload := map[string]interface{}{
		"contents":         history,
		"generationConfig": generationConfig,
	}

	// Add system_instruction if set
	if len(c.systemParts) > 0 {
		parts := make([]map[string]interface{}, len(c.systemParts))
		for i, p := range c.systemParts {
			parts[i] = map[string]interface{}{"text": p}
		}
		payload["system_instruction"] = map[string]interface{}{
			"parts": parts,
		}
	}

	// Add tools if provided
	if toolset != nil && len(toolset.Tools) > 0 {
		tools := c.convertToolset(toolset)
		if len(tools) > 0 {
			payload["tools"] = tools
			// Use AUTO mode instead of ANY to allow the model to decide when to stop
			payload["tool_config"] = map[string]interface{}{
				"function_calling_config": map[string]interface{}{
					"mode": "AUTO",
				},
			}
		}
	}

	// Add server-side Google Search grounding if enabled
	if c.config.WebSearchEnabled {
		googleSearchTool := map[string]interface{}{"google_search": map[string]interface{}{}}
		if existingTools, ok := payload["tools"].([]GeminiTool); ok {
			rawTools := make([]interface{}, len(existingTools))
			for i, t := range existingTools {
				rawTools[i] = t
			}
			rawTools = append(rawTools, googleSearchTool)
			payload["tools"] = rawTools
		} else {
			payload["tools"] = []interface{}{googleSearchTool}
		}
	}

	ret, history, err := c.doStreamingRequest(ctx, payload, history)
	if err != nil {
		return ret, err
	}
	c.history = history
	return ret, err
}

func (c *GeminiLLMClient) doStreamingRequest(ctx context.Context, payload map[string]interface{}, history []GeminiMessage) (*Response, []GeminiMessage, error) {
	ctx, _, end := instrumentation.GetLogSpan(ctx, "doStreamingRequest")
	defer end()

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	if os.Getenv("GEMINI_DEBUG_RESPONSES") == "true" {
		cyan(fmt.Sprintf("[GEMINI DEBUG] Outgoing request body: %s\n", string(body)))
	}

	baseURL := c.config.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	apiUrl := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?key=%s",
		baseURL, c.config.Model, c.config.APIKey)

	req, err := http.NewRequestWithContext(ctx, "POST", apiUrl, bytes.NewBuffer(body))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Wrap body with TeeReader before any read so ALL bytes (error or success) are captured.
	var rawResponseBuf bytes.Buffer
	resp.Body = io.NopCloser(io.TeeReader(resp.Body, &rawResponseBuf))

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, nil, &APIError{StatusCode: resp.StatusCode, Body: string(b), Provider: "Gemini", RawResponseBytes: rawResponseBuf.Bytes()}
	}

	response, newHistory, functionCallParts, err := c.processStreamingResponse(ctx, resp.Body, history)
	if err != nil {
		return nil, nil, err
	}
	response.RawRequestJSON = body

	// Build a synthetic JSON response from parsed content.
	// The raw streaming response is a JSON array of chunks, not a single valid response.
	// Synthesize a response matching Gemini's non-streaming format for reliable state recovery.
	// IMPORTANT: Use functionCallParts directly (not response.ToolCalls) so that thought_signature
	// is preserved. Gemini requires thought_signature on function calls in subsequent turns.
	var parts []map[string]interface{}
	if response.Content != "" {
		parts = append(parts, map[string]interface{}{"text": response.Content})
	}
	for _, part := range functionCallParts {
		syntheticPart := map[string]interface{}{
			"functionCall": part.FunctionCall,
		}
		if part.ThoughtSignature != "" {
			syntheticPart["thoughtSignature"] = part.ThoughtSignature
		}
		parts = append(parts, syntheticPart)
	}
	syntheticResponse, _ := json.Marshal(map[string]interface{}{
		"candidates": []interface{}{
			map[string]interface{}{
				"content": map[string]interface{}{
					"role":  "model",
					"parts": parts,
				},
			},
		},
	})
	response.RawResponseJSON = syntheticResponse
	response.RawResponseBytes = rawResponseBuf.Bytes()
	return response, newHistory, nil
}

func (c *GeminiLLMClient) processStreamingResponse(ctx context.Context, body io.ReadCloser, history []GeminiMessage) (*Response, []GeminiMessage, []GeminiPart, error) {
	ctx, span, end := instrumentation.GetLogSpan(ctx, "processStreamingResponse")
	defer end()

	reader := bufio.NewReader(body)
	var responseContent strings.Builder
	var toolCalls []ToolCall
	var functionCallParts []GeminiPart
	var usageData UsageData
	callIndex := 0 // Counter for generating unique CallIds

	//dumpReader(reader)

	dec := json.NewDecoder(reader)
	// Read the opening '[' token.
	token, err := dec.Token()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to read opening token: %w", err)
	}
	// Verify that token is the beginning of an array.
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return nil, nil, nil, fmt.Errorf("expected '[' at start of array, got %v", token)
	}

	// Loop through the array elements.
	for dec.More() {
		var streamMsg GeminiStreamMessage
		if err := dec.Decode(&streamMsg); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to decode message: %w", err)
		}

		// Debug logging for raw API responses (enabled via GEMINI_DEBUG_RESPONSES env var)
		if os.Getenv("GEMINI_DEBUG_RESPONSES") == "true" {
			msgJSON, _ := json.MarshalIndent(streamMsg, "", "  ")
			cyan(fmt.Sprintf("[GEMINI DEBUG] Stream message: %s\n", string(msgJSON)))
		}

		// Process usage metadata if present
		// PromptTokens = non-cached input only (Gemini includes cached in PromptTokenCount, so subtract)
		if streamMsg.UsageMetadata.TotalTokenCount > 0 {
			usageData.PromptTokens = streamMsg.UsageMetadata.PromptTokenCount
			usageData.CompletionTokens = streamMsg.UsageMetadata.CandidatesTokenCount
			usageData.TotalTokens = streamMsg.UsageMetadata.TotalTokenCount
			usageData.ReasoningTokens = streamMsg.UsageMetadata.ThoughtsTokenCount
			usageData.CachedTokens = streamMsg.UsageMetadata.CachedContentTokenCount
			usageData.PromptTokens -= usageData.CachedTokens

			// Debug logging for usage metadata
			if os.Getenv("GEMINI_DEBUG_RESPONSES") == "true" {
				yellow(fmt.Sprintf("[GEMINI DEBUG] Usage: Prompt=%d, Completion=%d, Total=%d, Reasoning=%d, Cached=%d\n",
					usageData.PromptTokens, usageData.CompletionTokens, usageData.TotalTokens,
					usageData.ReasoningTokens, usageData.CachedTokens))
			}
		}

		// Process candidates if present
		if len(streamMsg.Candidates) > 0 {
			candidate := streamMsg.Candidates[0]
			// Process each part.
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					responseContent.WriteString(part.Text)
					c.responseCallback(part.Text)
				}
				if part.FunctionCall != nil {
					functionCallParts = append(functionCallParts, part)

					argsJSON, err := json.Marshal(part.FunctionCall.Args)
					if err != nil {
						argsJSON = []byte("{}")
					}
					// Generate unique CallId using index to avoid collisions when multiple calls to the same tool are made
					toolCall := ToolCall{
						Name:   part.FunctionCall.Name,
						Args:   string(argsJSON),
						CallId: fmt.Sprintf("call_%d", callIndex),
					}
					callIndex++ // Increment for next tool call
					toolCalls = append(toolCalls, toolCall)
				}
			}
		}
	}

	span.SetValues("llm.prompt_tokens", usageData.PromptTokens, "llm.completion_tokens", usageData.CompletionTokens, "llm.cached_tokens", usageData.CachedTokens, "llm.reasoning_tokens", usageData.ReasoningTokens)

	// Add text response to history if present
	if responseContent.Len() > 0 {
		history = append(history, GeminiMessage{
			Role: "model",
			Parts: []GeminiPart{
				{
					Text: responseContent.String(),
				},
			},
		})
	}

	// Add ALL function calls to history, not just the first one
	if len(functionCallParts) > 0 {
		history = append(history, GeminiMessage{
			Role:  "model",
			Parts: functionCallParts,
		})
	}

	return &Response{
		Content:   responseContent.String(),
		ToolCalls: toolCalls,
		Usage:     usageData,
	}, history, functionCallParts, nil
}

// convertToolset converts a ToolSet to Gemini tools format
func (c *GeminiLLMClient) convertToolset(toolset *tools.ToolSet) []GeminiTool {
	var geminiTools []GeminiTool
	if toolset == nil {
		return geminiTools
	}

	for _, tool := range toolset.SortedTools() {
		// Create function declaration
		functionDeclaration := GeminiFunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": make(map[string]interface{}),
				"required":   tool.Required,
			},
		}

		// Convert properties
		properties := functionDeclaration.Parameters["properties"].(map[string]interface{})
		propKeys := make([]string, 0, len(tool.Properties))
		for k := range tool.Properties {
			propKeys = append(propKeys, k)
		}
		sort.Strings(propKeys)
		for _, propName := range propKeys {
			propDef := tool.Properties[propName]
			propMap := map[string]interface{}{
				"type":        propDef.Type,
				"description": propDef.Description,
			}

			// Handle array items
			if propDef.Type == "array" && propDef.Items != nil {
				propMap["items"] = map[string]interface{}{
					"type": propDef.Items.Type,
				}
				if propDef.Items.Description != "" {
					propMap["items"].(map[string]interface{})["description"] = propDef.Items.Description
				}
			}

			//TODO: WARN: Gemini API requires Objects to be well-typed, i.e describing all keys, additional properties also are not supported
			// meaning, we cannot use it for plan executor which has Object for inputs
			// will either have to hack it to be in different format, or alternatively generate tool per next step. Which is really not fun and bans more dynamic use of steps by agent, like re-writing them on the flight
			// we dont have deep structures now, so assuming used only as a map
			//if propDef.Type == "object" && propDef.Items != nil {
			// delete(propMap, "items")
			// propMap["additionalProperties"] = map[string]interface{}{
			//    "type":        propDef.Items.Type,
			//    "description": propDef.Items.Description,
			// }
			//}

			// Handle enum values
			if len(propDef.Enum) > 0 {
				propMap["enum"] = propDef.Enum
			}

			properties[propName] = propMap
		}

		// Add function declaration to tools
		geminiTools = append(geminiTools, GeminiTool{
			FunctionDeclarations: []GeminiFunctionDeclaration{functionDeclaration},
		})
	}

	return geminiTools
}

func (c *GeminiLLMClient) SetSystemBlocks(blocks []SystemBlock) {
	if len(blocks) == 0 {
		c.systemParts = nil
		return
	}
	const preamble = "Strictly follow instructions set as system prompt\n\n"
	// Cacheable blocks first (better implicit caching), then non-cacheable
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
	c.systemParts = append(cacheable, nonCacheable...)
}

func (c *GeminiLLMClient) ClearHistory() {
	c.history = nil
	c.systemParts = nil
}

func (c *GeminiLLMClient) SetPendingToolResults(toolResponses map[string]string) {
	c.pendingToolResults = toolResponses
}

func (c *GeminiLLMClient) GetDanglingToolUseIDs() []string {
	return nil
}
