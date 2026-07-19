package mockserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
)

// OpenAIToolCall represents a tool call in an OpenAI mock response.
type OpenAIToolCall struct {
	Name string
	Args string // JSON string, e.g. `{"key":"value"}`
	ID   string // auto-generated as "call_mock_001" if empty
}

// OpenAIMockError causes the server to return an HTTP error.
type OpenAIMockError struct {
	StatusCode int
	Message    string
}

// OpenAIMockTurn defines what the server returns for a single chat/completions call.
type OpenAIMockTurn struct {
	Content   string
	ToolCalls []OpenAIToolCall
	Error     *OpenAIMockError
	// Token counts
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int // reported in prompt_tokens_details.cached_tokens
	ReasoningTokens  int // reported in completion_tokens_details.reasoning_tokens
}

// OpenAIMockServer wraps httptest.Server and serves scripted OpenAI-style SSE responses.
type OpenAIMockServer struct {
	*httptest.Server
	script           []OpenAIMockTurn
	mu               sync.Mutex
	callIdx          int
	lastRequest      map[string]interface{}
	lastRequestBytes []byte
	BaseURL          string
}

// NewOpenAIMockServer creates a test server handling POST /v1/chat/completions.
func NewOpenAIMockServer(script []OpenAIMockTurn) *OpenAIMockServer {
	s := &OpenAIMockServer{script: script}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	s.Server = httptest.NewServer(mux)
	s.BaseURL = s.Server.URL + "/v1/"
	return s
}

func (s *OpenAIMockServer) CallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.callIdx
}

func (s *OpenAIMockServer) Reset(script []OpenAIMockTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.script = script
	s.callIdx = 0
}

func (s *OpenAIMockServer) LastRequest() map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastRequest
}

func (s *OpenAIMockServer) LastRequestBytes() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lastRequestBytes == nil {
		return nil
	}
	cp := make([]byte, len(s.lastRequestBytes))
	copy(cp, s.lastRequestBytes)
	return cp
}

func (s *OpenAIMockServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("failed to read body: %v", err))
		return
	}

	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	if reqBody["messages"] == nil {
		writeOpenAIError(w, http.StatusBadRequest, "messages is required")
		return
	}
	if reqBody["model"] == nil {
		writeOpenAIError(w, http.StatusBadRequest, "model is required")
		return
	}

	s.mu.Lock()
	s.lastRequest = reqBody
	s.lastRequestBytes = body
	idx := s.callIdx
	if idx >= len(s.script) {
		s.mu.Unlock()
		writeOpenAIError(w, http.StatusInternalServerError, "mock script exhausted")
		return
	}
	turn := s.script[idx]
	s.callIdx++
	s.mu.Unlock()

	if turn.Error != nil {
		writeOpenAIError(w, turn.Error.StatusCode, turn.Error.Message)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	msgID := fmt.Sprintf("chatcmpl-mock-%03d", idx+1)

	// Stream text content as delta chunks
	if turn.Content != "" {
		for _, chunk := range splitIntoChunks(turn.Content) {
			writeOpenAISSE(w, flusher, map[string]interface{}{
				"id":     msgID,
				"object": "chat.completion.chunk",
				"choices": []interface{}{
					map[string]interface{}{
						"index":         0,
						"finish_reason": nil,
						"delta":         map[string]interface{}{"content": chunk},
					},
				},
			})
		}
	}

	// Stream tool calls
	for i, tc := range turn.ToolCalls {
		callID := tc.ID
		if callID == "" {
			callID = fmt.Sprintf("call_mock_%03d", i+1)
		}
		args := tc.Args
		if args == "" {
			args = "{}"
		}
		// First chunk: declare the tool call with name
		writeOpenAISSE(w, flusher, map[string]interface{}{
			"id":     msgID,
			"object": "chat.completion.chunk",
			"choices": []interface{}{
				map[string]interface{}{
					"index":         0,
					"finish_reason": nil,
					"delta": map[string]interface{}{
						"tool_calls": []interface{}{
							map[string]interface{}{
								"index": i,
								"id":    callID,
								"type":  "function",
								"function": map[string]interface{}{
									"name":      tc.Name,
									"arguments": "",
								},
							},
						},
					},
				},
			},
		})
		// Second chunk: arguments
		writeOpenAISSE(w, flusher, map[string]interface{}{
			"id":     msgID,
			"object": "chat.completion.chunk",
			"choices": []interface{}{
				map[string]interface{}{
					"index":         0,
					"finish_reason": nil,
					"delta": map[string]interface{}{
						"tool_calls": []interface{}{
							map[string]interface{}{
								"index": i,
								"function": map[string]interface{}{
									"arguments": args,
								},
							},
						},
					},
				},
			},
		})
	}

	// Finish chunk
	finishReason := "stop"
	if len(turn.ToolCalls) > 0 {
		finishReason = "tool_calls"
	}
	writeOpenAISSE(w, flusher, map[string]interface{}{
		"id":     msgID,
		"object": "chat.completion.chunk",
		"choices": []interface{}{
			map[string]interface{}{
				"index":         0,
				"finish_reason": finishReason,
				"delta":         map[string]interface{}{},
			},
		},
	})

	// Usage chunk (stream_options include_usage)
	promptTokens := turn.PromptTokens
	if promptTokens == 0 {
		promptTokens = 100
	}
	completionTokens := turn.CompletionTokens
	if completionTokens == 0 {
		completionTokens = len(turn.Content)/4 + 1
	}

	writeOpenAISSE(w, flusher, map[string]interface{}{
		"id":      msgID,
		"object":  "chat.completion.chunk",
		"choices": []interface{}{},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
			"prompt_tokens_details": map[string]interface{}{
				"cached_tokens": turn.CachedTokens,
				"audio_tokens":  0,
			},
			"completion_tokens_details": map[string]interface{}{
				"reasoning_tokens": turn.ReasoningTokens,
			},
		},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAISSE(w http.ResponseWriter, flusher http.Flusher, data interface{}) {
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "data: %s\n\n", string(b))
	flusher.Flush()
}

func writeOpenAIError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    statusCode,
		},
	})
}
