package benchmark

// Regression tests: replay consumers must enforce the net-of-cache contract
// that auto.go:283 depends on.
//
//   auto.go:283  full := int64(r.inputTokens) + int64(r.cachedTokens)
//
// The comment at auto.go:277-281 REQUIRES that inputTokens be net-of-cache
// (i.e. gross prompt_tokens MINUS cached_tokens), so that:
//
//   full = (gross - cached) + cached = gross   ← correct denominator
//
// Previously, the Anthropic paths stored GROSS InputTokens, and the OpenAI
// paths had inline "- cached" fixes.  All paths now route through
// buildReplayUsage(), which enforces the net contract uniformly.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/weka/wekai/llm"
)

// Known server response scenario used throughout.
// Server reports: prompt_tokens=1000, cached_tokens=600.
// Therefore net (non-cached) = 400.
const (
	knownGross  = 1000                     // prompt_tokens as reported by server
	knownCached = 600                      // cached_tokens as reported by server
	knownNet    = knownGross - knownCached // 400 — what inputTokens MUST be after subtract
)

// buildReplayPosterForDoubleCount constructs an httptest server returning the
// given body and a replayPoster pointed at it.
func buildReplayPosterForDoubleCount(t *testing.T, responseBody string, streaming bool) (*httptest.Server, *replayPoster) {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if streaming {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			fmt.Fprint(w, responseBody)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(200)
			fmt.Fprint(w, responseBody)
		}
	}))
	modelSpec := fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL)
	keys := llm.APIKeys{OpenAI: "sk-test"}
	p, err := newReplayPoster(modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		ts.Close()
		t.Fatalf("newReplayPoster: %v", err)
	}
	return ts, p
}

// buildAnthropicReplayPosterForDoubleCount constructs an httptest server
// returning the given body and a replayPoster pointed at it using the
// Anthropic API type.
func buildAnthropicReplayPosterForDoubleCount(t *testing.T, responseBody string, streaming bool) (*httptest.Server, *replayPoster) {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if streaming {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			fmt.Fprint(w, responseBody)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(200)
			fmt.Fprint(w, responseBody)
		}
	}))
	modelSpec := fmt.Sprintf("dynamic/%s,type=anthropic,model=claude-test", ts.URL)
	keys := llm.APIKeys{Anthropic: "ak-test"}
	p, err := newReplayPoster(modelSpec, keys, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		ts.Close()
		t.Fatalf("newReplayPoster: %v", err)
	}
	return ts, p
}

// minReplayReq returns the smallest valid RouterReplayRequest.
func minReplayReq(streaming bool) RouterReplayRequest {
	return RouterReplayRequest{
		RequestID:    1,
		Stream:       streaming,
		OutputTokens: 10,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
		},
	}
}

// TestReplayConsumerInputTokensAreNetOfCache_Streaming asserts that
// consumeOpenAISSE stores NET (not gross) InputTokens.
// Regression for the double-count bug: previously stored gross=1000 instead of net=400.
func TestReplayConsumerInputTokensAreNetOfCache_Streaming(t *testing.T) {
	// Server reports: prompt_tokens=1000, cached_tokens=600 → net=400.
	streamBody := strings.Join([]string{
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
		``,
		fmt.Sprintf(
			`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":5,"total_tokens":%d,"prompt_tokens_details":{"cached_tokens":%d}}}`,
			knownGross, knownGross+5, knownCached),
		``,
		`data: [DONE]`,
		``,
		``,
	}, "\n")

	ts, p := buildReplayPosterForDoubleCount(t, streamBody, true)
	defer ts.Close()

	st := &autoState{stream: newCompletionStream(200)}
	metrics := p.do(context.Background(), minReplayReq(true), strings.Repeat("x", 300), 1, "s1", "i1", 1, st, nil)
	if metrics.Error != nil {
		t.Fatalf("unexpected error: %v", metrics.Error)
	}

	gotInput := metrics.UsageData.InputTokens.Count
	gotCached := metrics.UsageData.CachedTokens.Count

	// InputTokens must be NET (gross - cached).
	if gotInput != knownNet {
		t.Errorf("[STREAMING] InputTokens.Count = %d, want NET=%d (gross=%d - cached=%d)",
			gotInput, knownNet, knownGross, knownCached)
	}
	if gotCached != knownCached {
		t.Fatalf("[STREAMING] CachedTokens.Count = %d, want %d", gotCached, knownCached)
	}

	// full = InputTokens + CachedTokens must equal gross (auto.go:283 invariant).
	full := int64(gotInput) + int64(gotCached)
	if full != int64(knownGross) {
		t.Errorf("[STREAMING] full = %d, want %d (= gross)", full, knownGross)
	}

	// cached/full ratio must be correct (0.60), not deflated.
	ratio := float64(gotCached) / float64(full)
	const eps = 0.001
	if absf(ratio-0.60) > eps {
		t.Errorf("[STREAMING] cached/full ratio = %.4f, want 0.60", ratio)
	}

	t.Logf("[STREAMING] net=%d, cached=%d, full=%d (=gross), ratio=%.4f — CORRECT",
		gotInput, gotCached, full, ratio)
}

// TestReplayConsumerInputTokensAreNetOfCache_Plain asserts that
// consumeOpenAIPlain stores NET (not gross) InputTokens.
// Regression for the double-count bug: previously stored gross=1000 instead of net=400.
func TestReplayConsumerInputTokensAreNetOfCache_Plain(t *testing.T) {
	plainBody := fmt.Sprintf(`{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"model": "test-model",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
		"usage": {
			"prompt_tokens": %d,
			"completion_tokens": 5,
			"total_tokens": %d,
			"prompt_tokens_details": {"cached_tokens": %d}
		}
	}`, knownGross, knownGross+5, knownCached)

	ts, p := buildReplayPosterForDoubleCount(t, plainBody, false)
	defer ts.Close()

	st := &autoState{stream: newCompletionStream(200)}
	metrics := p.do(context.Background(), minReplayReq(false), strings.Repeat("x", 300), 1, "s1", "i1", 1, st, nil)
	if metrics.Error != nil {
		t.Fatalf("unexpected error: %v", metrics.Error)
	}

	gotInput := metrics.UsageData.InputTokens.Count
	gotCached := metrics.UsageData.CachedTokens.Count

	// InputTokens must be NET.
	if gotInput != knownNet {
		t.Errorf("[PLAIN] InputTokens.Count = %d, want NET=%d (gross=%d - cached=%d)",
			gotInput, knownNet, knownGross, knownCached)
	}
	if gotCached != knownCached {
		t.Fatalf("[PLAIN] CachedTokens.Count = %d, want %d", gotCached, knownCached)
	}

	// full = InputTokens + CachedTokens must equal gross.
	full := int64(gotInput) + int64(gotCached)
	if full != int64(knownGross) {
		t.Errorf("[PLAIN] full = %d, want %d (= gross)", full, knownGross)
	}

	ratio := float64(gotCached) / float64(full)
	const eps = 0.001
	if absf(ratio-0.60) > eps {
		t.Errorf("[PLAIN] cached/full ratio = %.4f, want 0.60", ratio)
	}

	t.Logf("[PLAIN] net=%d, cached=%d, full=%d (=gross), ratio=%.4f — CORRECT",
		gotInput, gotCached, full, ratio)
}

// TestReplayConsumerInputTokensAreNetOfCache_AnthropicPlain asserts that
// consumePlain (Anthropic) stores NET InputTokens.
// input_tokens=400 is already net; cache_read=600; gross=400+600=1000.
func TestReplayConsumerInputTokensAreNetOfCache_AnthropicPlain(t *testing.T) {
	// Anthropic reports input_tokens=net (400), cache_read_input_tokens=600.
	// grossPrompt = input + cache_read + cache_creation = 400+600+0 = 1000.
	// buildReplayUsage must store net=400, cached=600.
	plainBody := `{
		"id": "msg_test",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "text", "text": "hi"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 400,
			"cache_read_input_tokens": 600,
			"cache_creation_input_tokens": 0,
			"output_tokens": 5
		}
	}`

	ts, p := buildAnthropicReplayPosterForDoubleCount(t, plainBody, false)
	defer ts.Close()

	st := &autoState{stream: newCompletionStream(200)}
	// Anthropic non-streaming does not set Stream=true.
	req := RouterReplayRequest{
		RequestID:    1,
		Stream:       false,
		OutputTokens: 5,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 50, BlockTypes: []string{"text"}},
		},
	}
	metrics := p.do(context.Background(), req, strings.Repeat("x", 300), 1, "s1", "i1", 1, st, nil)
	if metrics.Error != nil {
		t.Fatalf("unexpected error: %v", metrics.Error)
	}

	gotInput := metrics.UsageData.InputTokens.Count
	gotCached := metrics.UsageData.CachedTokens.Count

	// Anthropic: grossPrompt = input_tokens(400) + cache_read(600) + cache_creation(0) = 1000.
	// buildReplayUsage(1000, 600, 5) → net=400, cached=600.
	if gotInput != 400 {
		t.Errorf("[ANTHROPIC PLAIN] InputTokens.Count = %d, want 400 (net)", gotInput)
	}
	if gotCached != 600 {
		t.Errorf("[ANTHROPIC PLAIN] CachedTokens.Count = %d, want 600", gotCached)
	}

	full := int64(gotInput) + int64(gotCached)
	if full != 1000 {
		t.Errorf("[ANTHROPIC PLAIN] full = %d, want 1000 (= gross)", full)
	}

	ratio := float64(gotCached) / float64(full)
	const eps = 0.001
	if absf(ratio-0.60) > eps {
		t.Errorf("[ANTHROPIC PLAIN] cached/full ratio = %.4f, want 0.60", ratio)
	}

	t.Logf("[ANTHROPIC PLAIN] net=%d, cached=%d, full=%d, ratio=%.4f — CORRECT",
		gotInput, gotCached, full, ratio)
}

// TestBuildReplayUsage_DirectUnit is a direct unit test of the helper,
// covering the net-of-cache arithmetic and the clamp-to-zero guard.
func TestBuildReplayUsage_DirectUnit(t *testing.T) {
	cases := []struct {
		name        string
		grossPrompt int
		cached      int
		output      int
		wantNet     int
		wantCached  int
	}{
		{"normal 60%", 1000, 600, 10, 400, 600},
		{"no cache", 1000, 0, 10, 1000, 0},
		{"full cache", 1000, 1000, 5, 0, 1000},
		{"clamp negative", 500, 600, 5, 0, 600}, // cached > gross → net clamped to 0
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildReplayUsage(tc.grossPrompt, tc.cached, tc.output)
			if got.InputTokens.Count != tc.wantNet {
				t.Errorf("InputTokens.Count = %d, want %d", got.InputTokens.Count, tc.wantNet)
			}
			if got.CachedTokens.Count != tc.wantCached {
				t.Errorf("CachedTokens.Count = %d, want %d", got.CachedTokens.Count, tc.wantCached)
			}
			if got.OutputTokens.Count != tc.output {
				t.Errorf("OutputTokens.Count = %d, want %d", got.OutputTokens.Count, tc.output)
			}
			if got.RequestCount != 1 {
				t.Errorf("RequestCount = %d, want 1", got.RequestCount)
			}
		})
	}
}

// TestReplayVsSyntheticContrastSubtraction documents the CONTRAST with the
// synthetic path.  internal/llm/openai.go:418-421 subtracts CachedTokens
// from PromptTokens before storing, producing net=400 so that
// full = net + cached = gross = 1000 and ratio = 0.60.
//
// We replicate the exact arithmetic from llm/openai.go:420 here because
// that path is inside a streaming closure that is not separately exported.
func TestReplayVsSyntheticContrastSubtraction(t *testing.T) {
	// Starting values from server.
	promptTokens := knownGross  // 1000
	cachedTokens := knownCached // 600

	// llm/openai.go:418-421:
	//   if streamResp.Usage.PromptDetails != nil {
	//       usageData.CachedTokens = streamResp.Usage.PromptDetails.CachedTokens
	//       usageData.PromptTokens -= usageData.CachedTokens   // ← subtract
	//   }
	promptTokens -= cachedTokens // the one line that the replay path is missing

	// ── (d) synthetic path yields net → correct full ────────────────────────
	if promptTokens != knownNet {
		t.Errorf("[SYNTHETIC] net prompt tokens = %d, want %d", promptTokens, knownNet)
	}

	syntheticFull := int64(promptTokens) + int64(cachedTokens) // auto.go:283
	if syntheticFull != int64(knownGross) {
		t.Errorf("[SYNTHETIC] full = %d, want %d (= gross)", syntheticFull, knownGross)
	}

	syntheticRatio := float64(cachedTokens) / float64(syntheticFull)
	const eps = 0.001
	if absf(syntheticRatio-0.60) > eps {
		t.Errorf("[SYNTHETIC] ratio = %.4f, want 0.60", syntheticRatio)
	}

	t.Logf("[SYNTHETIC] net=%d, full=%d (=gross), ratio=%.4f — correct (llm/openai.go:420 subtract present)",
		promptTokens, syntheticFull, syntheticRatio)

	// buildReplayUsage produces the same correct result.
	got := buildReplayUsage(knownGross, knownCached, 5)
	replayFull := int64(got.InputTokens.Count) + int64(got.CachedTokens.Count)
	replayRatio := float64(got.CachedTokens.Count) / float64(replayFull)
	t.Logf("SUMMARY: replay (via helper) full=%d ratio=%.4f; synthetic full=%d ratio=%.4f — BOTH CORRECT",
		replayFull, replayRatio, syntheticFull, syntheticRatio)

	if replayFull != syntheticFull {
		t.Errorf("replay full (%d) != synthetic full (%d)", replayFull, syntheticFull)
	}
}

func absf(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Ensure the test file uses encoding/json (for the Anthropic body helper).
var _ = json.Marshal
