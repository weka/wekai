package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/weka/wekai/llm"
)

// TestReplayEndpointResolution covers the attempt-then-fallback contract:
// bare-root bases fall back to /v1 on 404 and latch that form; /v1 bases
// work on the first attempt with no doubled probe; non-404 errors never
// trigger the fallback; concurrent first requests race the latch benignly
// (duplicate probes allowed, single winner).
func TestReplayEndpointResolution(t *testing.T) {
	newVLLMStyleServer := func() (*httptest.Server, func() []string) {
		var mu sync.Mutex
		var paths []string
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			paths = append(paths, r.URL.Path)
			mu.Unlock()
			if r.URL.Path != "/v1/chat/completions" {
				w.WriteHeader(404)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
		}))
		return ts, func() []string {
			mu.Lock()
			defer mu.Unlock()
			return append([]string{}, paths...)
		}
	}
	minimalReq := RouterReplayRequest{
		RequestID:    1,
		Stream:       false,
		OutputTokens: 10,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 40, BlockTypes: []string{"text"}},
		},
	}
	docs := strings.Repeat("x", 400)
	keys := llm.APIKeys{OpenAI: "sk-test"}
	newState := func() *autoState { return &autoState{stream: newCompletionStream(200)} }
	mustPoster := func(t *testing.T, spec string) *replayPoster {
		t.Helper()
		p, err := newReplayPoster(spec, keys, "", "", false, 0, 0, 0, nil, nil)
		if err != nil {
			t.Fatalf("newReplayPoster: %v", err)
		}
		return p
	}

	t.Run("bare root falls back to /v1 and latches", func(t *testing.T) {
		ts, seen := newVLLMStyleServer()
		defer ts.Close()
		p := mustPoster(t, fmt.Sprintf("dynamic/%s,type=openai_vllm,model=m", ts.URL))
		if m := p.do(context.Background(), minimalReq, docs, 1, "s", "i", 1, newState()); m.Error != nil {
			t.Fatalf("first request: %v", m.Error)
		}
		got := seen()
		if len(got) != 2 || got[0] != "/chat/completions" || got[1] != "/v1/chat/completions" {
			t.Fatalf("probe sequence = %v, want [/chat/completions /v1/chat/completions]", got)
		}
		if p.epResolved != ts.URL+"/v1/chat/completions" || !p.epFellBack {
			t.Errorf("latch = %q fellBack=%v, want fallback latched", p.epResolved, p.epFellBack)
		}
		// Latched: the second request goes straight to /v1, one wire call.
		if m := p.do(context.Background(), minimalReq, docs, 2, "s", "i", 1, newState()); m.Error != nil {
			t.Fatalf("second request: %v", m.Error)
		}
		if got = seen(); len(got) != 3 || got[2] != "/v1/chat/completions" {
			t.Fatalf("post-latch paths = %v, want exactly one /v1 request appended", got)
		}
	})

	t.Run("/v1 base works first try, no doubled probe", func(t *testing.T) {
		ts, seen := newVLLMStyleServer()
		defer ts.Close()
		p := mustPoster(t, fmt.Sprintf("dynamic/%s/v1,type=openai_vllm,model=m", ts.URL))
		if m := p.do(context.Background(), minimalReq, docs, 1, "s", "i", 1, newState()); m.Error != nil {
			t.Fatalf("request: %v", m.Error)
		}
		got := seen()
		if len(got) != 1 || got[0] != "/v1/chat/completions" {
			t.Fatalf("paths = %v, want exactly [/v1/chat/completions] (no /v1/v1 probe)", got)
		}
		if p.epResolved != ts.URL+"/v1/chat/completions" || p.epFellBack {
			t.Errorf("latch = %q fellBack=%v, want primary latched without fallback", p.epResolved, p.epFellBack)
		}
	})

	t.Run("non-404 does not trigger fallback", func(t *testing.T) {
		var mu sync.Mutex
		hits := 0
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			hits++
			mu.Unlock()
			w.WriteHeader(500)
		}))
		defer ts.Close()
		p := mustPoster(t, fmt.Sprintf("dynamic/%s,type=openai_vllm,model=m", ts.URL))
		m := p.do(context.Background(), minimalReq, docs, 1, "s", "i", 1, newState())
		if m.Error == nil || !strings.Contains(m.Error.Error(), "status 500") {
			t.Fatalf("expected status-500 error, got %v", m.Error)
		}
		mu.Lock()
		n := hits
		mu.Unlock()
		if n != 1 {
			t.Errorf("server saw %d requests, want 1 (500 must not trigger the /v1 fallback)", n)
		}
		// The 500 DOES latch, and that is the point of the rule: a 404 means the
		// path is wrong, anything else means the path is right and something
		// else went wrong. This assertion used to require the opposite — no
		// latch on any failure — and was left behind when the latch rule
		// changed, so the suite has been red since. What this subtest is
		// actually for is the `hits != 1` check above: a 500 must not send the
		// client probing the /v1 form.
		if got := p.endpointAttempts(); len(got) != 1 {
			t.Errorf("endpoint attempts = %v, want the primary latched: a non-404 answer means "+
				"the path was right, so the form is resolved and must not be probed again", got)
		}
	})

	t.Run("concurrent first requests latch benignly", func(t *testing.T) {
		ts, seen := newVLLMStyleServer()
		defer ts.Close()
		p := mustPoster(t, fmt.Sprintf("dynamic/%s,type=openai_vllm,model=m", ts.URL))
		const n = 8
		errs := make([]error, n)
		var wg sync.WaitGroup
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				m := p.do(context.Background(), minimalReq, docs, 1, "s", fmt.Sprintf("i%d", i), 1, newState())
				errs[i] = m.Error
			}(i)
		}
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("concurrent request %d: %v", i, err)
			}
		}
		if p.epResolved != ts.URL+"/v1/chat/completions" || !p.epFellBack {
			t.Fatalf("latch after concurrent start = %q fellBack=%v", p.epResolved, p.epFellBack)
		}
		// Duplicate probes during the race are allowed; once latched, a new
		// request adds exactly one wire call.
		before := len(seen())
		if m := p.do(context.Background(), minimalReq, docs, 2, "s", "i", 1, newState()); m.Error != nil {
			t.Fatalf("post-latch request: %v", m.Error)
		}
		after := seen()
		if len(after) != before+1 || after[len(after)-1] != "/v1/chat/completions" {
			t.Fatalf("post-latch added %d calls (last %q), want exactly one /v1 call",
				len(after)-before, after[len(after)-1])
		}
	})
}

func TestNewReplayPoster_OpenAI(t *testing.T) {
	tests := []struct {
		name         string
		modelSpec    string
		wantType     string
		wantPrimary  string
		wantFallback string
		wantErr      bool
	}{
		{
			// The primary attempt honors the operator's /v1 base verbatim;
			// the fallback candidate exists but only fires on 404.
			name:         "openai type, /v1 base",
			modelSpec:    "dynamic/http://127.0.0.1:8000/v1,type=openai,model=gpt-4",
			wantType:     "openai",
			wantPrimary:  "http://127.0.0.1:8000/v1/chat/completions",
			wantFallback: "http://127.0.0.1:8000/v1/v1/chat/completions",
		},
		{
			name:         "openai_vllm type, /v1 base",
			modelSpec:    "dynamic/http://127.0.0.1:8000/v1,type=openai_vllm,model=my-model",
			wantType:     "openai_vllm",
			wantPrimary:  "http://127.0.0.1:8000/v1/chat/completions",
			wantFallback: "http://127.0.0.1:8000/v1/v1/chat/completions",
		},
		{
			name:         "anthropic type, /v1 base",
			modelSpec:    "dynamic/http://127.0.0.1:8000/v1,type=anthropic,model=claude",
			wantType:     "anthropic",
			wantPrimary:  "http://127.0.0.1:8000/v1/messages",
			wantFallback: "http://127.0.0.1:8000/v1/v1/messages",
		},
		{
			name:         "bare URL defaults to openai_vllm via NormalizeModelSpec",
			modelSpec:    llm.NormalizeModelSpec("http://127.0.0.1:8000/v1") + ",model=test-model", // add explicit model to avoid discovery
			wantType:     "openai_vllm",
			wantPrimary:  "http://127.0.0.1:8000/v1/chat/completions",
			wantFallback: "http://127.0.0.1:8000/v1/v1/chat/completions",
		},
		{
			// Bare-root base: the primary honors it verbatim (would 404 on
			// real vLLM), the fallback inserts the /v1 that vLLM serves.
			name:         "openai_vllm, bare-root base",
			modelSpec:    "dynamic/http://127.0.0.1:8000,type=openai_vllm,model=my-model",
			wantType:     "openai_vllm",
			wantPrimary:  "http://127.0.0.1:8000/chat/completions",
			wantFallback: "http://127.0.0.1:8000/v1/chat/completions",
		},
		{
			name:         "anthropic, bare-root base",
			modelSpec:    "dynamic/http://127.0.0.1:8000,type=anthropic,model=claude",
			wantType:     "anthropic",
			wantPrimary:  "http://127.0.0.1:8000/messages",
			wantFallback: "http://127.0.0.1:8000/v1/messages",
		},
		{
			// A /v2 base is honored verbatim — the old TrimSuffix design
			// would have mangled only /v1; the fallback still inserts /v1
			// after the prefix (and only fires on 404).
			name:         "openai_vllm, /v2 base",
			modelSpec:    "dynamic/http://127.0.0.1:8000/v2,type=openai_vllm,model=my-model",
			wantType:     "openai_vllm",
			wantPrimary:  "http://127.0.0.1:8000/v2/chat/completions",
			wantFallback: "http://127.0.0.1:8000/v2/v1/chat/completions",
		},
		{
			// Arbitrary path prefixes (proxies) are honored on the first try.
			name:         "openai_vllm, proxy prefix base",
			modelSpec:    "dynamic/http://127.0.0.1:8000/proxy/llm,type=openai_vllm,model=my-model",
			wantType:     "openai_vllm",
			wantPrimary:  "http://127.0.0.1:8000/proxy/llm/chat/completions",
			wantFallback: "http://127.0.0.1:8000/proxy/llm/v1/chat/completions",
		},
		{
			name:      "unsupported type (rejected)",
			modelSpec: "dynamic/http://127.0.0.1:8000/v1,type=gemini_native,model=gemini",
			wantErr:   true,
		},
	}

	keys := llm.APIKeys{OpenAI: "sk-test", Anthropic: "ak-test"}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := newReplayPoster(tt.modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if p.apiType != tt.wantType {
				t.Errorf("apiType = %q, want %q", p.apiType, tt.wantType)
			}
			if p.epPrimary != tt.wantPrimary {
				t.Errorf("epPrimary = %q, want %q", p.epPrimary, tt.wantPrimary)
			}
			if p.epFallback != tt.wantFallback {
				t.Errorf("epFallback = %q, want %q", p.epFallback, tt.wantFallback)
			}
			if got := p.endpointAttempts(); len(got) != 2 || got[0] != tt.wantPrimary || got[1] != tt.wantFallback {
				t.Errorf("endpointAttempts pre-latch = %v, want [primary fallback]", got)
			}
		})
	}
}

// TestOpenAIReplayEndToEnd spawns an httptest server that speaks the OpenAI
// SSE chat/completions protocol and verifies that the poster correctly
// builds the request body, sends it, and parses the response.
func TestOpenAIReplayEndToEnd(t *testing.T) {
	docs := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	if len(docs) < 200 {
		t.Fatalf("docs too short: %d bytes", len(docs))
	}

	// Track what was received so we can verify the translation.
	var receivedBody map[string]interface{}
	var receivedPath string
	var receivedAuth string

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedAuth = r.Header.Get("Authorization")

		// vLLM-shaped: only /v1/chat/completions exists. With a bare-root
		// base the poster's first attempt (/chat/completions) 404s here and
		// the /v1 fallback carries the request — this e2e exercises that.
		if r.URL.Path != "/v1/chat/completions" {
			w.WriteHeader(404)
			return
		}

		if err := json.NewDecoder(r.Body).Decode(&receivedBody); err != nil {
			t.Errorf("failed to decode request body: %v", err)
			w.WriteHeader(400)
			return
		}

		// Return a valid OpenAI SSE stream.
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)

		// First content chunk.
		fmt.Fprintf(w, `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`+"\n\n")
		// Flusher required for streaming.
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		time.Sleep(5 * time.Millisecond)

		// More content.
		fmt.Fprintf(w, `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`+"\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		time.Sleep(5 * time.Millisecond)

		// Final chunk with usage.
		fmt.Fprintf(w, `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":2,"total_tokens":44,"prompt_tokens_details":{"cached_tokens":20}}}`+"\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	modelSpec := fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL)
	keys := llm.APIKeys{OpenAI: "sk-test-123"}
	p, err := newReplayPoster(modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}

	// Build a minimal replay request that simulates a real captured session.
	req := RouterReplayRequest{
		RequestID:    1,
		Stream:       true,
		OutputTokens: 100,
		SystemBlocks: []RouterReplaySystemBlock{
			// >=200B so it is kept (not treated as the droppable per-request
			// header block; see effectiveSystemBlocks).
			{Hash: "syshash", Bytes: 250},
		},
		Messages: []RouterReplayMessage{
			{
				Role:       "user",
				Hash:       "msghash1",
				Bytes:      60,
				BlockTypes: []string{"text"},
			},
			{
				Role:       "assistant",
				Hash:       "msghash2",
				Bytes:      40,
				BlockTypes: []string{"text"},
			},
			{
				Role:       "user",
				Hash:       "msghash3",
				Bytes:      50,
				BlockTypes: []string{"text"},
			},
		},
	}

	st := &autoState{
		stream: newCompletionStream(200),
	}

	ctx := context.Background()
	metrics := p.do(ctx, req, docs, 1, "session-1", "instance-1", 1, st)

	// Verify no error.
	if metrics.Error != nil {
		t.Fatalf("unexpected error: %v", metrics.Error)
	}

	// Verify path.
	if receivedPath != "/v1/chat/completions" {
		t.Errorf("received path = %q, want /v1/chat/completions", receivedPath)
	}

	// Verify auth header.
	if receivedAuth != "Bearer sk-test-123" {
		t.Errorf("received Authorization = %q, want Bearer sk-test-123", receivedAuth)
	}

	// Verify the translated body has OpenAI-format messages.
	messages, ok := receivedBody["messages"].([]interface{})
	if !ok {
		t.Fatal("body.messages is not an array")
	}
	// Expected: system(1 stamp + 1 sys block) + user + assistant + user = 5 messages.
	if len(messages) < 3 {
		t.Fatalf("expected >= 3 messages, got %d", len(messages))
	}
	// First message should be system (the runID stamp — no runID in this test so it should be absent);
	// actually runID is "", so no stamp. Let's verify the first system block.
	if msg0, ok := messages[0].(map[string]interface{}); ok {
		if msg0["role"] != "system" {
			t.Errorf("first message role = %q, want system", msg0["role"])
		}
	}
	// Content should be a string, not an array.
	if msg1, ok := messages[1].(map[string]interface{}); ok {
		if _, isStr := msg1["content"].(string); !isStr {
			t.Errorf("content is not a plain string for text-only message: %T", msg1["content"])
		}
	}

	// Verify response content.
	if metrics.Response != "Hello world" {
		t.Errorf("response = %q, want %q", metrics.Response, "Hello world")
	}

	// Verify TTFT was captured.
	if metrics.TimeToFirstToken <= 0 {
		t.Error("TTFT was not captured")
	}

	// Verify usage data.
	// prompt_tokens=42, cached_tokens=20 → net InputTokens = 42-20 = 22.
	if metrics.UsageData.InputTokens.Count != 22 {
		t.Errorf("input tokens = %d, want 22 (net: prompt 42 - cached 20)", metrics.UsageData.InputTokens.Count)
	}
	if metrics.UsageData.OutputTokens.Count != 2 {
		t.Errorf("output tokens = %d, want 2", metrics.UsageData.OutputTokens.Count)
	}
	if metrics.UsageData.CachedTokens.Count != 20 {
		t.Errorf("cached tokens = %d, want 20", metrics.UsageData.CachedTokens.Count)
	}
}

// TestOpenAIReplayToolTranslation verifies that assistant messages with
// tool_use blocks are translated into proper OpenAI tool_calls (not flattened).
func TestOpenAIReplayToolTranslation(t *testing.T) {
	docs := strings.Repeat("x", 300)
	req := RouterReplayRequest{
		Stream:       true,
		OutputTokens: 100,
		Messages: []RouterReplayMessage{
			{
				Role:       "assistant",
				Hash:       "toolmsg",
				Bytes:      80,
				BlockTypes: []string{"text", "tool_use"},
				ToolUseIDs: []string{"toolu_001"},
			},
		},
	}

	body, _, err := buildOpenAIChatCompletionsBody(req, docs, "test-model", "", 0, false, 0)
	if err != nil {
		t.Fatalf("buildOpenAIChatCompletionsBody: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	messages := parsed["messages"].([]interface{})
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	msg := messages[0].(map[string]interface{})
	if msg["role"] != "assistant" {
		t.Errorf("role = %q, want assistant", msg["role"])
	}
	// tool_calls must be present and properly structured.
	toolCallsRaw, ok := msg["tool_calls"]
	if !ok {
		t.Fatal("assistant message missing tool_calls")
	}
	toolCalls, ok := toolCallsRaw.([]interface{})
	if !ok || len(toolCalls) == 0 {
		t.Fatal("tool_calls is empty or wrong type")
	}
	tc := toolCalls[0].(map[string]interface{})
	if tc["id"] != "toolu_001" {
		t.Errorf("tool_call id = %v, want toolu_001", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_call type = %v, want function", tc["type"])
	}
	fn, _ := tc["function"].(map[string]interface{})
	if fn["name"] == "" {
		t.Error("function.name is empty")
	}
	// Content must not contain the old placeholder text.
	content, _ := msg["content"].(string)
	if strings.Contains(content, "[tool_use") {
		t.Errorf("content still contains placeholder: %s", content)
	}
	t.Logf("tool_call id=%v name=%v arguments=%v", tc["id"], fn["name"], fn["arguments"])
}

// TestOpenAIBodyBuilderExtra verifies edge cases of the OpenAI body builder:
// zero-length system blocks (just runID stamp), non-streaming mode (no
// stream_options), and missing messages.
func TestOpenAIBodyBuilderExtra(t *testing.T) {
	docs := strings.Repeat("x", 300)

	t.Run("runID stamp prepended", func(t *testing.T) {
		req := RouterReplayRequest{
			Stream:       true,
			OutputTokens: 100,
			Messages: []RouterReplayMessage{
				{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
			},
		}
		body, _, err := buildOpenAIChatCompletionsBody(req, docs, "test-model", "run-42", 0, false, 0)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		msgs := parsed["messages"].([]interface{})
		if len(msgs) != 2 {
			t.Fatalf("expected 2 messages (stamp + user), got %d", len(msgs))
		}
		msg0 := msgs[0].(map[string]interface{})
		if msg0["role"] != "system" {
			t.Errorf("first msg role = %q, want system", msg0["role"])
		}
		if c, ok := msg0["content"].(string); !ok || !strings.Contains(c, "RUN_GUID: run-42") {
			t.Errorf("stamp content missing RUN_GUID: %v", msg0["content"])
		}
	})

	t.Run("non-streaming omits stream_options", func(t *testing.T) {
		req := RouterReplayRequest{
			Stream:       false,
			OutputTokens: 100,
			Messages: []RouterReplayMessage{
				{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
			},
		}
		body, _, err := buildOpenAIChatCompletionsBody(req, docs, "test-model", "", 0, false, 0)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, exists := parsed["stream_options"]; exists {
			t.Error("stream_options present in non-streaming request")
		}
		if st, ok := parsed["stream"].(bool); !ok || st {
			t.Errorf("stream = %v, want false", parsed["stream"])
		}
	})

	t.Run("temperature and top_p forwarded", func(t *testing.T) {
		temp := 0.7
		topp := 0.9
		req := RouterReplayRequest{
			Stream:       true,
			OutputTokens: 100,
			Temperature:  &temp,
			TopP:         &topp,
			Messages: []RouterReplayMessage{
				{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
			},
		}
		body, _, err := buildOpenAIChatCompletionsBody(req, docs, "test-model", "", 0, false, 0)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		var parsed map[string]interface{}
		json.Unmarshal(body, &parsed)
		if tVal, ok := parsed["temperature"].(float64); !ok || tVal != 0.7 {
			t.Errorf("temperature = %v, want 0.7", parsed["temperature"])
		}
		if pVal, ok := parsed["top_p"].(float64); !ok || pVal != 0.9 {
			t.Errorf("top_p = %v, want 0.9", parsed["top_p"])
		}
		// top_k and thinking must NOT appear (Anthropic-only).
		if _, exists := parsed["top_k"]; exists {
			t.Error("top_k should not appear in OpenAI body")
		}
		if _, exists := parsed["thinking"]; exists {
			t.Error("thinking should not appear in OpenAI body")
		}
	})

	t.Run("empty messages and no system blocks", func(t *testing.T) {
		req := RouterReplayRequest{
			Stream:       true,
			OutputTokens: 100,
		}
		body, _, err := buildOpenAIChatCompletionsBody(req, docs, "test-model", "", 0, false, 0)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		msgs := parsed["messages"].([]interface{})
		if len(msgs) != 0 {
			t.Errorf("expected 0 messages, got %d", len(msgs))
		}
	})
}

// TestOpenAINonStreamingEndToEnd verifies the non-streaming (plain JSON) response
// parsing path through the full HTTP round-trip.
func TestOpenAINonStreamingEndToEnd(t *testing.T) {
	docs := strings.Repeat("x", 300)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		fmt.Fprintf(w, `{
			"id": "chatcmpl-1",
			"object": "chat.completion",
			"model": "test-model",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32, "prompt_tokens_details": {"cached_tokens": 10}}
		}`)
	}))
	defer ts.Close()

	modelSpec := fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL)
	keys := llm.APIKeys{OpenAI: "sk-test"}
	p, err := newReplayPoster(modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}

	req := RouterReplayRequest{
		RequestID:    1,
		Stream:       false,
		OutputTokens: 100,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
		},
	}

	st := &autoState{stream: newCompletionStream(200)}
	metrics := p.do(context.Background(), req, docs, 1, "s1", "i1", 1, st)

	if metrics.Error != nil {
		t.Fatalf("unexpected error: %v", metrics.Error)
	}
	if metrics.Response != "Hello world" {
		t.Errorf("response = %q, want %q", metrics.Response, "Hello world")
	}
	// prompt_tokens=30, cached_tokens=10 → net InputTokens = 30-10 = 20.
	if metrics.UsageData.InputTokens.Count != 20 {
		t.Errorf("input tokens = %d, want 20 (net: prompt 30 - cached 10)", metrics.UsageData.InputTokens.Count)
	}
	if metrics.UsageData.OutputTokens.Count != 2 {
		t.Errorf("output tokens = %d, want 2", metrics.UsageData.OutputTokens.Count)
	}
	if metrics.UsageData.CachedTokens.Count != 10 {
		t.Errorf("cached tokens = %d, want 10", metrics.UsageData.CachedTokens.Count)
	}
}

// TestOpenAIErrorResponse verifies that server errors are propagated correctly.
func TestOpenAIErrorResponse(t *testing.T) {
	docs := strings.Repeat("x", 300)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		fmt.Fprintln(w, `{"error": "internal server error"}`)
	}))
	defer ts.Close()

	modelSpec := fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL)
	keys := llm.APIKeys{OpenAI: "sk-test"}
	p, err := newReplayPoster(modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}

	req := RouterReplayRequest{
		Stream:       false,
		OutputTokens: 100,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
		},
	}

	st := &autoState{stream: newCompletionStream(200)}
	metrics := p.do(context.Background(), req, docs, 1, "s1", "i1", 1, st)

	if metrics.Error == nil {
		t.Fatal("expected error for 500 response, got nil")
	}
	if !strings.Contains(metrics.Error.Error(), "500") {
		t.Errorf("error should mention status 500: %v", metrics.Error)
	}
}

// TestOpenAISSEWithoutUsage verifies that servers that omit the usage field
// (like sglang sometimes does) don't crash the parser.
func TestOpenAISSEWithoutUsage(t *testing.T) {
	docs := strings.Repeat("x", 300)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fmt.Fprintf(w, `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	modelSpec := fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL)
	keys := llm.APIKeys{OpenAI: "sk-test"}
	p, err := newReplayPoster(modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}

	req := RouterReplayRequest{
		Stream:       true,
		OutputTokens: 100,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
		},
	}

	st := &autoState{stream: newCompletionStream(200)}
	metrics := p.do(context.Background(), req, docs, 1, "s1", "i1", 1, st)

	if metrics.Error != nil {
		// "empty response from model" is acceptable for no-usage streams.
		t.Logf("error (expected for no-usage stream): %v", metrics.Error)
	}
	if metrics.Response != "ok" {
		t.Errorf("response = %q, want %q", metrics.Response, "ok")
	}
}
