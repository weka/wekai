package mockvllm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// newTestServer starts a real listener (per the repo's testing policy:
// "httptest servers for HTTP utilities") so concurrency behavior is genuine,
// not simulated via ResponseRecorder.
func newTestServer(t *testing.T, cfg Config) (*httptest.Server, *Server) {
	t.Helper()
	srv := NewServer(NewEngine(cfg))
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts, srv
}

func chatBody(t *testing.T, prompt string, stream bool, includeUsage bool) []byte {
	t.Helper()
	body := map[string]any{
		"model": "whatever",
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
		"stream":     stream,
		"max_tokens": 4,
	}
	if stream && includeUsage {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	return b
}

func TestHandleModels(t *testing.T) {
	ts, _ := newTestServer(t, Config{ModelID: "my-model"})
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var out modelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Data) != 1 || out.Data[0].ID != "my-model" {
		t.Fatalf("unexpected models response: %+v", out)
	}
}

func TestHandleHealth(t *testing.T) {
	ts, _ := newTestServer(t, Config{})
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// TestChatCompletions_CachedTokensGrowOnRepeat is the smoke scenario from the
// task spec at the unit-test layer: the SAME long prompt posted twice must
// report cached_tokens 0 on the first call and the full prompt on the second,
// via the exact field the benchmark poster parses
// (usage.prompt_tokens_details.cached_tokens).
func TestChatCompletions_CachedTokensGrowOnRepeat(t *testing.T) {
	ts, _ := newTestServer(t, Config{BlockSizeTokens: 16})
	prompt := strings.Repeat("the quick brown fox jumps over the lazy dog ", 100)

	post := func() struct {
		Usage usage `json:"usage"`
	} {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
			bytes.NewReader(chatBody(t, prompt, false, false)))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		var out struct {
			Usage usage `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return out
	}

	first := post()
	if first.Usage.PromptTokensDetails == nil || first.Usage.PromptTokensDetails.CachedTokens != 0 {
		t.Fatalf("first request should be a cold miss, got usage=%+v", first.Usage)
	}
	if first.Usage.PromptTokens <= 0 {
		t.Fatalf("expected positive prompt_tokens, got %+v", first.Usage)
	}

	second := post()
	if second.Usage.PromptTokensDetails == nil {
		t.Fatalf("second response missing prompt_tokens_details")
	}
	if second.Usage.PromptTokensDetails.CachedTokens != second.Usage.PromptTokens {
		t.Fatalf("repeating the identical prompt should fully hit: got cached=%d of %d",
			second.Usage.PromptTokensDetails.CachedTokens, second.Usage.PromptTokens)
	}
	if second.Usage.PromptTokens != first.Usage.PromptTokens {
		t.Fatalf("token estimate should be stable across identical requests: %d vs %d",
			first.Usage.PromptTokens, second.Usage.PromptTokens)
	}
}

func TestChatCompletions_MalformedBodyIs400(t *testing.T) {
	ts, _ := newTestServer(t, Config{})
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

// TestChatCompletions_429PastMaxConcurrency holds the single admitted slot
// open for the duration of BaseLatency and asserts a second, concurrent
// request is refused with 429 — the load signal
// router/internal/circuit.go and router/internal/proxy.go treat as
// authoritative.
func TestChatCompletions_429PastMaxConcurrency(t *testing.T) {
	ts, _ := newTestServer(t, Config{
		MaxConcurrency:   1,
		BaseLatency:      200 * time.Millisecond,
		DefaultMaxTokens: 1,
	})

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
			bytes.NewReader(chatBody(t, "occupy the only slot", false, false)))
		if err != nil {
			t.Errorf("first POST: %v", err)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("first request status = %d, want 200", resp.StatusCode)
		}
	}()

	// Give the first request time to enter the handler and reserve the slot
	// (Admit happens before any decode work, so this margin is generous).
	time.Sleep(30 * time.Millisecond)

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
		bytes.NewReader(chatBody(t, "second request should be shed", false, false)))
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("second request status = %d, want 429", resp.StatusCode)
	}

	wg.Wait()
}

// TestChatCompletions_Streaming exercises the SSE path exactly as
// benchmark/replay_router_post.go:consumeOpenAISSE reads it: data: chunks,
// content deltas, a final chunk carrying usage (because
// stream_options.include_usage was set), and a terminal [DONE] marker.
func TestChatCompletions_Streaming(t *testing.T) {
	ts, _ := newTestServer(t, Config{DefaultMaxTokens: 3, DecodePerToken: time.Millisecond})
	prompt := strings.Repeat("stream me please ", 50)

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
		bytes.NewReader(chatBody(t, prompt, true, true)))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	var dataLines []string
	sawDone := false
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			continue
		}
		dataLines = append(dataLines, payload)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if !sawDone {
		t.Fatalf("stream never sent the [DONE] terminal marker")
	}
	if len(dataLines) < 2 {
		t.Fatalf("expected at least a role-open chunk and a final chunk, got %d chunks", len(dataLines))
	}

	var last chatChunk
	if err := json.Unmarshal([]byte(dataLines[len(dataLines)-1]), &last); err != nil {
		t.Fatalf("decode final chunk: %v", err)
	}
	if last.Usage == nil {
		t.Fatalf("final chunk missing usage even though stream_options.include_usage was set")
	}
	if last.Usage.PromptTokensDetails == nil || last.Usage.PromptTokensDetails.CachedTokens != 0 {
		t.Fatalf("first-ever streamed prompt should be a cold miss: %+v", last.Usage)
	}

	var content strings.Builder
	for _, dl := range dataLines {
		var c chatChunk
		if err := json.Unmarshal([]byte(dl), &c); err != nil {
			continue
		}
		if len(c.Choices) > 0 && c.Choices[0].Delta != nil {
			content.WriteString(c.Choices[0].Delta.Content)
		}
	}
	if !strings.Contains(content.String(), "tok0") {
		t.Fatalf("expected synthetic token content in the stream, got %q", content.String())
	}
}

func TestMetricsEndpointServesPrometheusFormat(t *testing.T) {
	ts, _ := newTestServer(t, Config{})
	// Generate at least one data point.
	http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatBody(t, "warm the metrics", false, false)))

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	out := buf.String()
	for _, want := range []string{"vllm:request_success_total", "vllm:prompt_tokens_total", "vllm:num_requests_running"} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics output missing %q:\n%s", want, out)
		}
	}
}
