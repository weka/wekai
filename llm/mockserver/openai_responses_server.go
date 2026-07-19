package mockserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
)

// OpenAIResponsesToolCall represents a tool call in an OpenAI Responses API mock response.
type OpenAIResponsesToolCall struct {
	Name string
	Args string // JSON string
	ID   string // call ID, auto-generated if empty
}

// OpenAIResponsesMockError causes the server to return an HTTP error.
type OpenAIResponsesMockError struct {
	StatusCode int
	Message    string
}

// OpenAIResponsesMockTurn defines what the server returns for a single /responses call.
type OpenAIResponsesMockTurn struct {
	Content    string
	ToolCalls  []OpenAIResponsesToolCall
	Error      *OpenAIResponsesMockError
	ResponseID string // response ID to return (for chaining tests), auto-generated if empty
	// Token counts
	InputTokens     int
	OutputTokens    int
	CachedTokens    int // in input_tokens_details.cached_tokens
	ReasoningTokens int // in output_tokens_details.reasoning_tokens
}

// OpenAIResponsesMockServer wraps httptest.Server and serves scripted Responses API SSE.
type OpenAIResponsesMockServer struct {
	*httptest.Server
	script           []OpenAIResponsesMockTurn
	mu               sync.Mutex
	callIdx          int
	lastRequest      map[string]interface{}
	lastRequestBytes []byte
	BaseURL          string
}

// NewOpenAIResponsesMockServer creates a test server handling POST /v1/responses.
func NewOpenAIResponsesMockServer(script []OpenAIResponsesMockTurn) *OpenAIResponsesMockServer {
	s := &OpenAIResponsesMockServer{script: script}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/responses", s.handleResponses)
	s.Server = httptest.NewServer(mux)
	s.BaseURL = s.Server.URL + "/v1/"
	return s
}

func (s *OpenAIResponsesMockServer) CallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.callIdx
}

func (s *OpenAIResponsesMockServer) Reset(script []OpenAIResponsesMockTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.script = script
	s.callIdx = 0
}

func (s *OpenAIResponsesMockServer) LastRequest() map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastRequest
}

func (s *OpenAIResponsesMockServer) LastRequestBytes() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lastRequestBytes == nil {
		return nil
	}
	cp := make([]byte, len(s.lastRequestBytes))
	copy(cp, s.lastRequestBytes)
	return cp
}

func (s *OpenAIResponsesMockServer) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIResponsesError(w, http.StatusBadRequest, fmt.Sprintf("failed to read body: %v", err))
		return
	}

	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		writeOpenAIResponsesError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	s.mu.Lock()
	s.lastRequest = reqBody
	s.lastRequestBytes = body
	idx := s.callIdx
	if idx >= len(s.script) {
		s.mu.Unlock()
		writeOpenAIResponsesError(w, http.StatusInternalServerError, "mock script exhausted")
		return
	}
	turn := s.script[idx]
	s.callIdx++
	s.mu.Unlock()

	if turn.Error != nil {
		writeOpenAIResponsesError(w, turn.Error.StatusCode, turn.Error.Message)
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

	respID := turn.ResponseID
	if respID == "" {
		respID = fmt.Sprintf("resp_mock_%03d", idx+1)
	}

	inputTokens := turn.InputTokens
	if inputTokens == 0 {
		inputTokens = 100
	}
	outputTokens := turn.OutputTokens
	if outputTokens == 0 {
		outputTokens = len(turn.Content)/4 + 1
	}

	// Build output items for the completed response
	var outputItems []interface{}

	// Stream text content
	if turn.Content != "" {
		textItemID := fmt.Sprintf("msg_%s_0", respID)
		// output_text.delta events
		for _, chunk := range splitIntoChunks(turn.Content) {
			writeOpenAIResponsesSSE(w, flusher, map[string]interface{}{
				"type":  "response.output_text.delta",
				"delta": chunk,
			})
		}
		// output_text.done
		writeOpenAIResponsesSSE(w, flusher, map[string]interface{}{
			"type": "response.output_text.done",
			"text": turn.Content,
		})
		outputItems = append(outputItems, map[string]interface{}{
			"id":     textItemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []interface{}{
				map[string]interface{}{"type": "output_text", "text": turn.Content},
			},
		})
	}

	// Stream tool calls
	for i, tc := range turn.ToolCalls {
		callID := tc.ID
		if callID == "" {
			callID = fmt.Sprintf("call_mock_resp_%03d", i+1)
		}
		itemID := fmt.Sprintf("fc_%s_%d", respID, i)
		args := tc.Args
		if args == "" {
			args = "{}"
		}

		// output_item.added
		writeOpenAIResponsesSSE(w, flusher, map[string]interface{}{
			"type": "response.output_item.added",
			"item": map[string]interface{}{
				"id":        itemID,
				"type":      "function_call",
				"call_id":   callID,
				"name":      tc.Name,
				"arguments": "",
			},
		})
		// function_call_arguments.delta
		writeOpenAIResponsesSSE(w, flusher, map[string]interface{}{
			"type":    "response.function_call_arguments.delta",
			"item_id": itemID,
			"delta":   args,
		})
		// function_call_arguments.done
		writeOpenAIResponsesSSE(w, flusher, map[string]interface{}{
			"type":    "response.function_call_arguments.done",
			"item_id": itemID,
		})

		outputItems = append(outputItems, map[string]interface{}{
			"id":        itemID,
			"type":      "function_call",
			"call_id":   callID,
			"name":      tc.Name,
			"arguments": args,
		})
	}

	// response.completed with usage and output
	writeOpenAIResponsesSSE(w, flusher, map[string]interface{}{
		"type": "response.completed",
		"response": map[string]interface{}{
			"id":     respID,
			"object": "response",
			"status": "completed",
			"model":  "mock-model",
			"output": outputItems,
			"usage": map[string]interface{}{
				"input_tokens":  inputTokens,
				"output_tokens": outputTokens,
				"total_tokens":  inputTokens + outputTokens,
				"input_tokens_details": map[string]interface{}{
					"cached_tokens": turn.CachedTokens,
				},
				"output_tokens_details": map[string]interface{}{
					"reasoning_tokens": turn.ReasoningTokens,
				},
			},
		},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAIResponsesSSE(w http.ResponseWriter, flusher http.Flusher, data interface{}) {
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "data: %s\n\n", string(b))
	flusher.Flush()
}

func writeOpenAIResponsesError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
		},
	})
}
