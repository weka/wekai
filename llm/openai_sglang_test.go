package llm

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

// TestOpenAISGLangDynamicModelConfig confirms type=openai_sglang picks up the
// two behaviors it exists for: max_tokens (not max_completion_tokens, same as
// vLLM) and return_cached_tokens_details, SGLang's per-request opt-in for
// usage.prompt_tokens_details.cached_tokens (vLLM's equivalent is the
// server-launch flag --enable-prompt-tokens-details). Without the latter,
// cache hit rate reads as 0 against every SGLang response — see
// benchmark/replay_router_wire.go for the replay-path equivalent.
func TestOpenAISGLangDynamicModelConfig(t *testing.T) {
	cg := GetChatGetter("dynamic/http://127.0.0.1:1/v1,type=openai_sglang,model=m", &ChatParams{APIKeys: APIKeys{OpenAI: "k"}})
	if cg == nil {
		t.Fatal("expected a ChatGetter")
	}
	chat := cg.GetChat()
	oa, ok := chat.(*OpenAiChat)
	if !ok {
		t.Fatalf("expected *OpenAiChat, got %T", chat)
	}
	if !oa.config.UseCompatMaxTokens {
		t.Error("expected UseCompatMaxTokens=true for type=openai_sglang")
	}
	if v, ok := oa.config.ExtraBodyParams["return_cached_tokens_details"]; !ok || v != true {
		t.Errorf("expected ExtraBodyParams[\"return_cached_tokens_details\"]=true, got %v (present=%v)", v, ok)
	}
}

// TestOpenAIVLLMDynamicModelConfigUnaffected pins the existing openai_vllm
// behavior: it still gets UseCompatMaxTokens, but never
// return_cached_tokens_details (vLLM has its own server-launch flag for
// this, and does not recognize the SGLang request field).
func TestOpenAIVLLMDynamicModelConfigUnaffected(t *testing.T) {
	cg := GetChatGetter("dynamic/http://127.0.0.1:1/v1,type=openai_vllm,model=m", &ChatParams{APIKeys: APIKeys{OpenAI: "k"}})
	oa, ok := cg.GetChat().(*OpenAiChat)
	if !ok {
		t.Fatalf("expected *OpenAiChat, got %T", cg.GetChat())
	}
	if !oa.config.UseCompatMaxTokens {
		t.Error("expected UseCompatMaxTokens=true for type=openai_vllm")
	}
	if _, ok := oa.config.ExtraBodyParams["return_cached_tokens_details"]; ok {
		t.Error("type=openai_vllm must not set return_cached_tokens_details")
	}
}

// TestOpenAIStreamParsesSGLangCachedTokensDetails feeds a recorded SGLang
// stream, where the tier breakdown arrives in an empty-choices chunk after the
// content, and checks it lands in UsageData.CachedDetails next to the usual
// cached_tokens.
func TestOpenAIStreamParsesSGLangCachedTokensDetails(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`,
		``,
		`data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[],"sglext":{"cached_tokens_details":{"device":600,"host":300,"storage":100,"storage_backend":"weka"}}}`,
		``,
		`data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":5,"total_tokens":1205,"prompt_tokens_details":{"cached_tokens":1000}}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	l := &OpenAiChat{}
	resp, err := l.processStreamingResponse(context.Background(), &http.Response{Body: io.NopCloser(strings.NewReader(sse))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	d := resp.Usage.CachedDetails
	if d == nil {
		t.Fatal("CachedDetails not parsed")
	}
	if d.Device != 600 || d.Host != 300 || d.Storage != 100 || d.StorageBackend != "weka" {
		t.Errorf("details = %+v", *d)
	}
	if resp.Usage.CachedTokens != 1000 || resp.Usage.PromptTokens != 200 {
		t.Errorf("usage = %+v, want cached 1000 / net prompt 200", resp.Usage)
	}
}

func TestOpenAIStreamWithoutSGLangDetails(t *testing.T) {
	sse := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\ndata: [DONE]\n\n"
	l := &OpenAiChat{}
	resp, err := l.processStreamingResponse(context.Background(), &http.Response{Body: io.NopCloser(strings.NewReader(sse))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.CachedDetails != nil {
		t.Errorf("CachedDetails = %+v, want nil", resp.Usage.CachedDetails)
	}
}
