package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/weka/wekai/tools"
)

// APIError represents an HTTP API error with status code, allowing retry
// SystemBlock represents one named block of a system prompt.
// Cache hints whether this block is eligible for provider-level caching breakpoints.
type SystemBlock struct {
	ID      string
	Content string
	Cache   bool
}

// decisions based on status code rather than string matching on error text.
type APIError struct {
	StatusCode       int
	Body             string
	Provider         string
	RawResponseBytes []byte // Raw HTTP response body bytes captured before the error was parsed
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s API error (status %d): %s", e.Provider, e.StatusCode, e.Body)
}

// ToolCall represents a single tool invocation request from the LLM
type ToolCall struct {
	Name   string
	Args   string
	CallId string
}

// ToolsCalls is a collection of tool calls
type ToolsCalls []ToolCall

// MessageAttachment stores attachment metadata alongside a message for persistence.
type MessageAttachment struct {
	BlobID   string `json:"blob_id"`
	Name     string `json:"name"`
	MimeType string `json:"mime_type"`
	Size     int64  `json:"size"`
}

// Message represents a single conversation turn.
type Message struct {
	Role        string              `json:"role"`
	Content     interface{}         `json:"content"`
	Thinking    string              `json:"thinking,omitempty"`    // Extended thinking/reasoning content
	IsError     bool                `json:"is_error,omitempty"`    // true when this tool_use_response is an error
	Attachments []MessageAttachment `json:"attachments,omitempty"` // persisted attachment refs for multi-modal messages
	Timestamp   int64               `json:"timestamp,omitempty"`   // Unix ms when recorded
	DurationMs  int64               `json:"duration_ms,omitempty"` // Execution duration in milliseconds
}

// UsageData represents token usage information from LLM responses
type UsageData struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CachedTokens     int `json:"cached_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens"`
	CacheTokens5Min  int `json:"cache_tokens_5min"` // Anthropic-specific: tokens written to 5-minute cache
}

// Response represents an LLM response with tool calls and content
type Response struct {
	ToolCalls        ToolsCalls
	Content          string // Regular output content (no reasoning)
	Thinking         string // Reasoning/thinking content (e.g., from Anthropic thinking blocks, OpenAI reasoning)
	Usage            UsageData
	RawRequestJSON   json.RawMessage // Raw API request JSON for debugging/state recovery
	RawResponseJSON  json.RawMessage // Raw API response JSON for debugging/state recovery
	RawResponseBytes []byte          // Raw wire bytes (SSE stream or response body) for all response paths
}

// Chat represents an LLM conversation interface
type Chat interface {
	Request(ctx context.Context, content []ContentPart, toolset *tools.ToolSet) (*Response, error)
	Respond(ctx context.Context, toolResponses map[string]string, additionalMessages []Message, toolset *tools.ToolSet) (*Response, error)
	SetSystemBlocks(blocks []SystemBlock)
	ClearHistory()                                         // Clears conversation message history, keeps system prompt
	SetPendingToolResults(toolResponses map[string]string) // Store tool results to include in next Request
	GetDanglingToolUseIDs() []string                       // Returns tool_use IDs in the last assistant message that have no tool_result
}

// ChatGetter provides access to Chat instances and model information
type ChatGetter struct {
	modelInfo ModelInfo
	modelName string // User-facing model name (e.g., "openai/gpt-5")
	chatFunc  func() Chat
}

// GetChat returns a new Chat instance
func (cg *ChatGetter) GetChat() Chat {
	return cg.chatFunc()
}

// GetModelInfo returns the ModelInfo used to initialize this ChatGetter
func (cg *ChatGetter) GetModelInfo() ModelInfo {
	return cg.modelInfo
}

// GetModelName returns the user-facing model identifier (e.g., "openai/gpt-5")
func (cg *ChatGetter) GetModelName() string {
	return cg.modelName
}

// extractToolUseID extracts the tool call ID from a message if it represents
// a tool use request. Handles both formats:
//   - Internal format: Role=="tool_use_request", Content is ToolCall or map with CallId
//   - Anthropic format: Role=="assistant", Content is []interface{} with type:"tool_use" blocks
func extractToolUseID(msg Message) []string {
	// Internal format: tool_use_request with ToolCall content
	if msg.Role == "tool_use_request" {
		switch v := msg.Content.(type) {
		case ToolCall:
			if v.CallId != "" {
				return []string{v.CallId}
			}
		case map[string]interface{}:
			if id, _ := v["CallId"].(string); id != "" {
				return []string{id}
			}
		}
		return nil
	}

	// Anthropic format: assistant message with tool_use content blocks
	if msg.Role == "assistant" {
		contentSlice, ok := msg.Content.([]interface{})
		if !ok {
			return nil
		}
		var ids []string
		for _, item := range contentSlice {
			switch v := item.(type) {
			case map[string]interface{}:
				if t, _ := v["type"].(string); t == "tool_use" {
					if id, _ := v["id"].(string); id != "" {
						ids = append(ids, id)
					}
				}
			case ContentBlock:
				if v.Type == "tool_use" && v.Id != "" {
					ids = append(ids, v.Id)
				}
			}
		}
		return ids
	}

	return nil
}

// extractToolResultIDs extracts tool call IDs that have been answered in a message.
// Handles both formats:
//   - Internal format: Role=="tool_use_response" (matched by position, not ID)
//   - Anthropic format: Role=="user", Content has type:"tool_result" blocks with tool_use_id
func extractToolResultIDs(msg Message) []string {
	// Anthropic format: user message with tool_result content blocks
	if msg.Role == "user" {
		contentSlice, ok := msg.Content.([]interface{})
		if !ok {
			return nil
		}
		var ids []string
		for _, item := range contentSlice {
			switch v := item.(type) {
			case map[string]interface{}:
				if t, _ := v["type"].(string); t == "tool_result" {
					if id, _ := v["tool_use_id"].(string); id != "" {
						ids = append(ids, id)
					}
				}
			case ContentBlock:
				if v.Type == "tool_result" && v.ToolUseId != "" {
					ids = append(ids, v.ToolUseId)
				}
			}
		}
		return ids
	}

	return nil
}

// ExtractToolUseIDs returns the tool_use IDs from a message (any format).
func ExtractToolUseIDs(msg Message) []string {
	return extractToolUseID(msg)
}

// FindDanglingToolUseIDs scans a full message history and returns tool_use IDs
// that have no matching tool_result/tool_use_response, along with the index of
// the first message that contains a dangling tool_use.
// Handles both internal format (tool_use_request/tool_use_response roles) and
// Anthropic format (assistant/user roles with content blocks).
// Returns nil, -1 if the history is clean.
func FindDanglingToolUseIDs(history []Message) ([]string, int) {
	// Collect all answered tool call IDs:
	// 1. Anthropic format: tool_result blocks in user messages with tool_use_id
	// 2. Internal format: tool_use_response messages (matched positionally to preceding tool_use_request)
	answered := make(map[string]bool)

	// For internal format, track pending tool_use_request IDs and match them
	// with subsequent tool_use_response messages positionally
	var pendingInternalIDs []string
	for _, msg := range history {
		switch msg.Role {
		case "tool_use_request":
			ids := extractToolUseID(msg)
			pendingInternalIDs = append(pendingInternalIDs, ids...)
		case "tool_use_response":
			// Each tool_use_response answers one pending tool_use_request (FIFO order)
			if len(pendingInternalIDs) > 0 {
				answered[pendingInternalIDs[0]] = true
				pendingInternalIDs = pendingInternalIDs[1:]
			}
		case "user":
			for _, id := range extractToolResultIDs(msg) {
				answered[id] = true
			}
		}
	}

	// Find tool_use IDs that were never answered
	var dangling []string
	firstIdx := -1
	for i, msg := range history {
		for _, id := range extractToolUseID(msg) {
			if !answered[id] {
				dangling = append(dangling, id)
				if firstIdx == -1 {
					firstIdx = i
				}
			}
		}
	}
	return dangling, firstIdx
}

// NewChatGetterForTesting creates a ChatGetter for testing purposes.
// This is primarily used to inject mock Chat implementations with proper model info.
func NewChatGetterForTesting(modelName string, modelInfo ModelInfo, chatFunc func() Chat) *ChatGetter {
	return &ChatGetter{
		modelInfo: modelInfo,
		modelName: modelName,
		chatFunc:  chatFunc,
	}
}
