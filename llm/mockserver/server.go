package mockserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"
)

// ToolCall represents a tool_use block in the assistant response.
type ToolCall struct {
	Name string // tool name
	Args string // JSON string for input
	ID   string // tool_use ID, auto-generated if empty
}

// MockTurn defines what the server returns for a single /v1/messages call.
type MockTurn struct {
	Content    string        // text response
	ToolCalls  []ToolCall    // tool_use blocks to return
	Latency    time.Duration // delay before first event
	TokenDelay time.Duration // delay between SSE content_block_delta events
	Error      *MockError    // return error instead of success
}

// MockError causes the server to return an HTTP error.
type MockError struct {
	StatusCode int
	Type       string
	Message    string
}

// MockServer wraps httptest.Server and serves scripted Anthropic-style SSE responses.
type MockServer struct {
	*httptest.Server
	script         []MockTurn
	mu             sync.Mutex
	callIdx        int
	receivedBodies []json.RawMessage
}

// NewMockServer creates a test server with a scripted sequence of turns.
func NewMockServer(script []MockTurn) *MockServer {
	s := &MockServer{
		script: script,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", s.handleMessages)
	s.Server = httptest.NewServer(mux)
	return s
}

// CallCount returns how many calls have been served so far.
func (s *MockServer) CallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.callIdx
}

// Reset replaces the script and resets the call counter and captured bodies.
func (s *MockServer) Reset(script []MockTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.script = script
	s.callIdx = 0
	s.receivedBodies = nil
}

// GetReceivedBodies returns a copy of the raw JSON bodies received so far.
func (s *MockServer) GetReceivedBodies() []json.RawMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := make([]json.RawMessage, len(s.receivedBodies))
	copy(cp, s.receivedBodies)
	return cp
}

// handleMessages implements POST /v1/messages with SSE streaming.
func (s *MockServer) handleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read and capture the request body for later inspection.
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "failed to read body: "+err.Error())
		return
	}
	s.mu.Lock()
	s.receivedBodies = append(s.receivedBodies, json.RawMessage(bodyBytes))
	s.mu.Unlock()

	// Parse request body to validate required fields.
	var req struct {
		Messages []any  `json:"messages"`
		Model    string `json:"model"`
		Stream   bool   `json:"stream"`
	}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error())
		return
	}
	if req.Messages == nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "messages is required")
		return
	}
	if req.Model == "" {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}

	// Get the next scripted turn.
	s.mu.Lock()
	idx := s.callIdx
	if idx >= len(s.script) {
		s.mu.Unlock()
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "mock script exhausted")
		return
	}
	turn := s.script[idx]
	s.callIdx++
	s.mu.Unlock()

	// Handle error turns.
	if turn.Error != nil {
		writeAnthropicError(w, turn.Error.StatusCode, turn.Error.Type, turn.Error.Message)
		return
	}

	// Apply initial latency.
	if turn.Latency > 0 {
		select {
		case <-time.After(turn.Latency):
		case <-r.Context().Done():
			return
		}
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	msgID := fmt.Sprintf("msg_mock_%03d", idx+1)
	stopReason := "end_turn"
	if len(turn.ToolCalls) > 0 {
		stopReason = "tool_use"
	}

	// Estimate output tokens.
	outputTokens := len(turn.Content)/4 + 1
	for _, tc := range turn.ToolCalls {
		outputTokens += len(tc.Args)/4 + 10
	}

	// message_start
	writeSSE(w, flusher, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":      msgID,
			"type":    "message",
			"role":    "assistant",
			"content": []any{},
			"model":   "mock-model",
			"usage": map[string]any{
				"input_tokens":  100,
				"output_tokens": 0,
			},
		},
	})

	blockIdx := 0

	// Stream text content.
	if turn.Content != "" {
		writeSSE(w, flusher, "content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": blockIdx,
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		})

		chunks := splitIntoChunks(turn.Content)
		for _, chunk := range chunks {
			if turn.TokenDelay > 0 {
				select {
				case <-time.After(turn.TokenDelay):
				case <-r.Context().Done():
					return
				}
			}
			writeSSE(w, flusher, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": blockIdx,
				"delta": map[string]any{
					"type": "text_delta",
					"text": chunk,
				},
			})
		}

		writeSSE(w, flusher, "content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": blockIdx,
		})
		blockIdx++
	}

	// Stream tool calls.
	for i, tc := range turn.ToolCalls {
		callID := tc.ID
		if callID == "" {
			callID = fmt.Sprintf("toolu_mock_%03d", i+1)
		}

		writeSSE(w, flusher, "content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": blockIdx,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    callID,
				"name":  tc.Name,
				"input": map[string]any{},
			},
		})

		args := tc.Args
		if args == "" {
			args = "{}"
		}
		writeSSE(w, flusher, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": blockIdx,
			"delta": map[string]any{
				"type":         "input_json_delta",
				"partial_json": args,
			},
		})

		writeSSE(w, flusher, "content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": blockIdx,
		})
		blockIdx++
	}

	// message_delta with stop_reason.
	writeSSE(w, flusher, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason": stopReason,
		},
		"usage": map[string]any{
			"output_tokens": outputTokens,
		},
	})

	// message_stop
	writeSSE(w, flusher, "message_stop", map[string]any{
		"type": "message_stop",
	})
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, event string, data any) {
	jsonBytes, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(jsonBytes))
	flusher.Flush()
}

func writeAnthropicError(w http.ResponseWriter, statusCode int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    errType,
			"message": message,
		},
	})
}

// splitIntoChunks splits text into word-sized chunks for realistic streaming.
// Chunks concatenated together reproduce the original text exactly.
// Split points are after each space character, so each chunk (except possibly
// the first) starts with a space.
func splitIntoChunks(text string) []string {
	if text == "" {
		return nil
	}
	var chunks []string
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == ' ' && i+1 < len(text) {
			chunks = append(chunks, text[start:i+1])
			start = i + 1
		}
	}
	if start < len(text) {
		chunks = append(chunks, text[start:])
	}
	// If we only got one chunk (no spaces), just return it.
	if len(chunks) == 0 {
		chunks = []string{text}
	}
	return chunks
}
