package llm

import "testing"

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
