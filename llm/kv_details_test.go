package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

// goldenUsage builds a usage object from vLLM's golden usage extension (the
// spec both of its frontends are tested against), adding the standard fields.
func goldenUsage(t *testing.T) (usage string, details map[string]any) {
	t.Helper()
	raw, err := os.ReadFile("testdata/request_kv_details_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Details        map[string]any `json:"details"`
		UsageExtension map[string]any `json:"usage_extension"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	g.UsageExtension["prompt_tokens"] = 100
	g.UsageExtension["completion_tokens"] = 8
	g.UsageExtension["total_tokens"] = 108
	b, _ := json.Marshal(g.UsageExtension)
	return string(b), g.Details
}

func TestOpenAIUsageKVGolden(t *testing.T) {
	usage, _ := goldenUsage(t)
	var u OpenAIUsage
	if err := json.Unmarshal([]byte(usage), &u); err != nil {
		t.Fatal(err)
	}
	kv := u.KV()
	if kv == nil {
		t.Fatal("KV not parsed")
	}
	want := TokenSources{GPU: 64, External: 16, Computed: 20}
	if *kv.Sources != want {
		t.Errorf("sources = %+v, want %+v", *kv.Sources, want)
	}
	if kv.Sources.Total() != 100 {
		t.Errorf("sources total = %d, want the 100-token prompt", kv.Sources.Total())
	}
	rc := kv.Recompute
	if rc == nil || rc.NumPreemptions != 1 || rc.PreemptDiscardedTokens != 40 || rc.NumExternalLoadFailures != 2 || rc.ExternalLoadRecomputeToks != 32 {
		t.Errorf("recompute = %+v", rc)
	}
	tm := kv.Timings
	if tm == nil || tm.QueueWaitKV != 0.5 || tm.QueueWaitOther != 0.25 || tm.KVFetch != 1.5 ||
		tm.Prefill != 2.0 || tm.Decode != 3.0 || tm.PreemptedTime != 0.75 || tm.NumPreemptions != 1 {
		t.Errorf("timings = %+v", tm)
	}
	if u.PromptDetails.CachedTokens != 80 {
		t.Errorf("cached_tokens = %d, want 80", u.PromptDetails.CachedTokens)
	}
}

func TestOpenAIStreamParsesKVDetails(t *testing.T) {
	usage, _ := goldenUsage(t)
	sse := strings.Join([]string{
		`data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}`, ``,
		`data: {"id":"c","choices":[],"usage":` + usage + `}`, ``,
		`data: [DONE]`, ``,
	}, "\n")
	l := &OpenAiChat{}
	resp, err := l.processStreamingResponse(context.Background(), &http.Response{Body: io.NopCloser(strings.NewReader(sse))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	kv := resp.Usage.KV
	if kv == nil || kv.Sources == nil || kv.Sources.External != 16 || kv.Timings == nil || kv.Timings.Prefill != 2.0 {
		t.Fatalf("KV = %+v", kv)
	}
	if resp.Usage.CachedTokens != 80 || resp.Usage.PromptTokens != 20 {
		t.Errorf("usage = %+v, want cached 80 / net prompt 20", resp.Usage)
	}
}

func TestOpenAIUsageWithoutKVDetails(t *testing.T) {
	var u OpenAIUsage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":4}}`), &u); err != nil {
		t.Fatal(err)
	}
	if kv := u.KV(); kv != nil {
		t.Errorf("KV = %+v, want nil when the engine reported none", kv)
	}
	if (*OpenAIUsage)(nil).KV() != nil {
		t.Error("nil usage must yield nil KV")
	}
}

func TestTokenSourcesFromSGLang(t *testing.T) {
	// 1200-token prompt: 600 device + 300 host + 100 storage cached, rest computed.
	s := TokenSourcesFromSGLang(&CachedTokensDetails{Device: 600, Host: 300, Storage: 100}, 1200)
	if *s != (TokenSources{GPU: 600, Host: 300, External: 100, Computed: 200}) {
		t.Errorf("sources = %+v", *s)
	}
	if TokenSourcesFromSGLang(nil, 100) != nil {
		t.Error("nil details must map to nil sources")
	}
	// A cached total above the prompt must not produce negative computed.
	if s := TokenSourcesFromSGLang(&CachedTokensDetails{Device: 50}, 40); s.Computed != 0 {
		t.Errorf("computed = %d, want 0", s.Computed)
	}
}

func TestWithSGLangKeepsVLLMSources(t *testing.T) {
	vllm := &KVDetails{Sources: &TokenSources{GPU: 1, Computed: 2}}
	if got := vllm.WithSGLang(&CachedTokensDetails{Device: 9}, 10); got.Sources.GPU != 1 {
		t.Errorf("vLLM sources were overwritten: %+v", got.Sources)
	}
	var none *KVDetails
	if got := none.WithSGLang(nil, 10); got != nil {
		t.Errorf("got %+v, want nil", got)
	}
	if got := none.WithSGLang(&CachedTokensDetails{Device: 4}, 10); got == nil || got.Sources.Computed != 6 {
		t.Errorf("got %+v", got)
	}
}
