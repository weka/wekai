package benchmark

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// Recorded SGLang responses (return_cached_tokens_details=true).

const sglangPlainResponse = `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
"usage":{"prompt_tokens":1200,"completion_tokens":5,"total_tokens":1205,"prompt_tokens_details":{"cached_tokens":1000}},
"sglext":{"cached_tokens_details":{"device":600,"host":300,"storage":100,"storage_backend":"weka"}}}`

const sglangStreamEmptyChoicesChunk = `data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}

data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"x","object":"chat.completion.chunk","choices":[],"sglext":{"cached_tokens_details":{"device":700,"host":300}}}

data: {"id":"x","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":5,"total_tokens":1205,"prompt_tokens_details":{"cached_tokens":1000}}}

data: [DONE]

`

// Details alongside the final usage chunk, and no storage tier configured.
const sglangStreamFinalUsageChunk = `data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}

data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":2,"total_tokens":52,"prompt_tokens_details":{"cached_tokens":20}},"sglext":{"cached_tokens_details":{"device":20,"host":0}}}

data: [DONE]

`

func TestConsumeOpenAIPlainSGLangTiers(t *testing.T) {
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(sglangPlainResponse), time.Now(), &m)
	d := m.CachedDetails
	if d == nil || d.Device != 600 || d.Host != 300 || d.Storage != 100 || d.StorageBackend != "weka" {
		t.Fatalf("details = %+v", d)
	}
	if m.UsageData.CachedTokens.Count != 1000 || m.UsageData.InputTokens.Count != 200 {
		t.Errorf("usage cached=%d net=%d", m.UsageData.CachedTokens.Count, m.UsageData.InputTokens.Count)
	}
}

func TestConsumeOpenAISSESGLangTiers(t *testing.T) {
	var m RequestMetrics
	consumeOpenAISSE(strings.NewReader(sglangStreamEmptyChoicesChunk), time.Now(), &m)
	d := m.CachedDetails
	if d == nil || d.Device != 700 || d.Host != 300 || d.Storage != 0 {
		t.Fatalf("details = %+v", d)
	}
	if m.Response != "ok" || m.UsageData.CachedTokens.Count != 1000 {
		t.Errorf("response=%q cached=%d", m.Response, m.UsageData.CachedTokens.Count)
	}

	m = RequestMetrics{}
	consumeOpenAISSE(strings.NewReader(sglangStreamFinalUsageChunk), time.Now(), &m)
	if d := m.CachedDetails; d == nil || d.Device != 20 || d.Host != 0 {
		t.Fatalf("details = %+v", d)
	}
}

func TestSGLangTiersAbsentForOtherEngines(t *testing.T) {
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(`{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`), time.Now(), &m)
	if m.CachedDetails != nil {
		t.Errorf("details = %+v, want nil", m.CachedDetails)
	}
	var rec requestDataRecord
	rec.applyCachedDetails(m.CachedDetails)
	b, _ := json.Marshal(rec)
	for _, k := range []string{"cached_device", "cached_host", "cached_storage"} {
		if strings.Contains(string(b), k) {
			t.Errorf("record carries %s without a breakdown: %s", k, b)
		}
	}
}

func TestSGLangTiersRecordAndSummary(t *testing.T) {
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(sglangPlainResponse), time.Now(), &m)
	var rec requestDataRecord
	rec.applyCachedDetails(m.CachedDetails)
	b, _ := json.Marshal(rec)
	var back map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back["cached_device"] != float64(600) || back["cached_host"] != float64(300) ||
		back["cached_storage"] != float64(100) || back["cached_storage_backend"] != "weka" {
		t.Errorf("record = %s", b)
	}

	var tot sglangTierTotals
	tot.add(m.CachedDetails, m.UsageData.InputTokens.Count)
	tot.add(nil, 999) // a request with no breakdown is not counted
	line := tot.snapshot().summaryLine()
	if got, want := line, "sglang cache tiers: device=600 host=300 storage=100 computed=200 (1 reqs)"; got != want {
		t.Errorf("summary = %q, want %q", got, want)
	}
	if (sglangTierSnapshot{}).summaryLine() != "" {
		t.Error("empty snapshot must print nothing")
	}
}
