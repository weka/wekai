package mockserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
)

// GeminiToolCall represents a function call in a Gemini response.
type GeminiToolCall struct {
	Name             string                 // function name
	Args             map[string]interface{} // function arguments
	ThoughtSignature string                 // optional thought_signature (for thinking models)
}

// GeminiMockTurn defines what the server returns for a single streamGenerateContent call.
type GeminiMockTurn struct {
	Content   string           // text response
	ToolCalls []GeminiToolCall // function_call parts to return
	Error     *GeminiMockError // return HTTP error instead of success
	// UsageMetadata fields
	PromptTokens  int
	OutputTokens  int
	ThoughtTokens int
	CachedTokens  int
}

// GeminiMockError causes the mock to return an HTTP error.
type GeminiMockError struct {
	StatusCode int
	Message    string
}

// GeminiMockServer wraps httptest.Server and serves scripted Gemini-native streaming responses.
// The Gemini streamGenerateContent endpoint returns a JSON array of stream chunks,
// not SSE events.
type GeminiMockServer struct {
	*httptest.Server
	script           []GeminiMockTurn
	mu               sync.Mutex
	callIdx          int
	lastRequest      map[string]interface{} // parsed body of the most recent request
	lastRequestBytes []byte                 // raw body of the most recent request
	// BaseURLForClient returns the base URL to use in LLMConfig.BaseURL.
	// Pattern: "{BaseURLForClient}/models/{model}:streamGenerateContent?key=xxx"
	// Since Gemini appends "/v1beta/models/...", we expose the server URL directly.
	BaseURL string
}

// NewGeminiMockServer creates a test server with a scripted sequence of turns.
// The returned server handles POST requests to /v1beta/models/{model}:streamGenerateContent.
func NewGeminiMockServer(script []GeminiMockTurn) *GeminiMockServer {
	s := &GeminiMockServer{
		script: script,
	}
	mux := http.NewServeMux()
	// Match any path under /v1beta/models/ with streamGenerateContent
	mux.HandleFunc("/v1beta/models/", s.handleStreamGenerateContent)
	s.Server = httptest.NewServer(mux)
	s.BaseURL = s.Server.URL
	return s
}

// CallCount returns how many calls have been served so far.
func (s *GeminiMockServer) CallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.callIdx
}

// Reset replaces the script and resets the call counter.
func (s *GeminiMockServer) Reset(script []GeminiMockTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.script = script
	s.callIdx = 0
}

// LastRequest returns the parsed body of the most recent request.
func (s *GeminiMockServer) LastRequest() map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastRequest
}

// LastRequestBytes returns the raw body bytes of the most recent request.
func (s *GeminiMockServer) LastRequestBytes() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lastRequestBytes == nil {
		return nil
	}
	cp := make([]byte, len(s.lastRequestBytes))
	copy(cp, s.lastRequestBytes)
	return cp
}

// handleStreamGenerateContent handles POST /v1beta/models/{model}:streamGenerateContent
func (s *GeminiMockServer) handleStreamGenerateContent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read and capture the raw request body.
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeGeminiError(w, http.StatusBadRequest, fmt.Sprintf("failed to read body: %v", err))
		return
	}

	// Parse request body to validate required fields.
	var req struct {
		Contents         []interface{} `json:"contents"`
		GenerationConfig interface{}   `json:"generationConfig"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeGeminiError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	if req.Contents == nil {
		writeGeminiError(w, http.StatusBadRequest, "contents is required")
		return
	}

	// Capture the full parsed request body for inspection in tests.
	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err == nil {
		s.mu.Lock()
		s.lastRequest = reqBody
		s.lastRequestBytes = body
		s.mu.Unlock()
	}

	// Get the next scripted turn.
	s.mu.Lock()
	idx := s.callIdx
	if idx >= len(s.script) {
		s.mu.Unlock()
		writeGeminiError(w, http.StatusInternalServerError, "mock script exhausted")
		return
	}
	turn := s.script[idx]
	s.callIdx++
	s.mu.Unlock()

	// Handle error turns.
	if turn.Error != nil {
		writeGeminiError(w, turn.Error.StatusCode, turn.Error.Message)
		return
	}

	// Build Gemini stream response: a JSON array of stream chunks.
	// The real Gemini API streams JSON objects separated by commas in an array.
	// We build a single-chunk response that contains all content.
	response := buildGeminiStreamResponse(turn)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

// buildGeminiStreamResponse creates the JSON array response for the given turn.
// The Gemini streaming format is: [{candidates: [...], usageMetadata: {...}}, ...]
// We produce a two-element array: first chunk has content, last chunk has usage.
func buildGeminiStreamResponse(turn GeminiMockTurn) []interface{} {
	var parts []interface{}

	// Text part
	if turn.Content != "" {
		parts = append(parts, map[string]interface{}{
			"text": turn.Content,
		})
	}

	// Function call parts (with optional thought_signature)
	for _, tc := range turn.ToolCalls {
		part := map[string]interface{}{
			"functionCall": map[string]interface{}{
				"name": tc.Name,
				"args": tc.Args,
			},
		}
		if tc.ThoughtSignature != "" {
			part["thoughtSignature"] = tc.ThoughtSignature
		}
		parts = append(parts, part)
	}

	// Determine finish reason
	finishReason := "STOP"
	if len(turn.ToolCalls) > 0 {
		finishReason = "FUNCTION_CALL"
	}

	// Estimate tokens if not set
	promptTokens := turn.PromptTokens
	if promptTokens == 0 {
		promptTokens = 100
	}
	outputTokens := turn.OutputTokens
	if outputTokens == 0 {
		outputTokens = len(turn.Content)/4 + 1
		for _, tc := range turn.ToolCalls {
			outputTokens += len(tc.Name) + 10
		}
	}
	totalTokens := promptTokens + outputTokens + turn.CachedTokens + turn.ThoughtTokens

	contentChunk := map[string]interface{}{
		"candidates": []interface{}{
			map[string]interface{}{
				"content": map[string]interface{}{
					"parts": parts,
					"role":  "model",
				},
				"finishReason": finishReason,
			},
		},
		"usageMetadata": map[string]interface{}{
			"promptTokenCount":        promptTokens,
			"candidatesTokenCount":    outputTokens,
			"totalTokenCount":         totalTokens,
			"thoughtsTokenCount":      turn.ThoughtTokens,
			"cachedContentTokenCount": turn.CachedTokens,
		},
	}

	return []interface{}{contentChunk}
}

func writeGeminiError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"code":    statusCode,
			"message": message,
			"status":  "INTERNAL",
		},
	})
}
