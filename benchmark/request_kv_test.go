package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/llm"
)

// goldenUsageJSON returns a usage object built from vLLM's golden usage
// extension (the spec its frontends are tested against).
func goldenUsageJSON(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("testdata/request_kv_details_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		UsageExtension map[string]any `json:"usage_extension"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	g.UsageExtension["prompt_tokens"] = 100
	g.UsageExtension["completion_tokens"] = 8
	g.UsageExtension["total_tokens"] = 108
	b, _ := json.Marshal(g.UsageExtension)
	return string(b)
}

func assertGoldenKV(t *testing.T, kv *llm.KVDetails) {
	t.Helper()
	if kv == nil || kv.Sources == nil || kv.Recompute == nil || kv.Timings == nil {
		t.Fatalf("KV incomplete: %+v", kv)
	}
	if *kv.Sources != (llm.TokenSources{GPU: 64, External: 16, Computed: 20}) {
		t.Errorf("sources = %+v", *kv.Sources)
	}
	if kv.Recompute.ExternalLoadRecomputeToks != 32 || kv.Timings.KVFetch != 1.5 || kv.Timings.NumPreemptions != 1 {
		t.Errorf("recompute=%+v timings=%+v", kv.Recompute, kv.Timings)
	}
}

func TestConsumeOpenAIPlainKVGolden(t *testing.T) {
	body := `{"choices":[{"message":{"content":"ok"}}],"usage":` + goldenUsageJSON(t) + `}`
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(body), time.Now(), &m)
	assertGoldenKV(t, m.KV)
	if m.UsageData.CachedTokens.Count != 80 || m.UsageData.InputTokens.Count != 20 {
		t.Errorf("usage cached=%d net=%d", m.UsageData.CachedTokens.Count, m.UsageData.InputTokens.Count)
	}
}

func TestConsumeOpenAISSEKVGolden(t *testing.T) {
	sse := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":" + goldenUsageJSON(t) + "}\n\ndata: [DONE]\n\n"
	var m RequestMetrics
	consumeOpenAISSE(strings.NewReader(sse), time.Now(), &m)
	assertGoldenKV(t, m.KV)
	if m.Response != "ok" {
		t.Errorf("response = %q", m.Response)
	}
}

func TestConsumeOpenAIWithoutKVFields(t *testing.T) {
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(`{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":2}}}`), time.Now(), &m)
	if m.KV != nil {
		t.Errorf("KV = %+v, want nil", m.KV)
	}
	m = RequestMetrics{}
	consumeOpenAISSE(strings.NewReader("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n"), time.Now(), &m)
	if m.KV != nil {
		t.Errorf("KV = %+v, want nil", m.KV)
	}
}

// Timings alone (no sources) are still recorded: each part is independent.
func TestConsumeOpenAIPartialKVFields(t *testing.T) {
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(`{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":5,"completion_tokens":1,"vllm_request_timings":{"prefill":0.1,"decode":0.2}}}`), time.Now(), &m)
	if m.KV == nil || m.KV.Sources != nil || m.KV.Timings == nil || m.KV.Timings.Decode != 0.2 {
		t.Fatalf("KV = %+v", m.KV)
	}
}

// SGLang's tier split lands in the same engine-neutral sources shape as vLLM's.
func TestApplyKVSGLangSources(t *testing.T) {
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(sglangPlainResponse), time.Now(), &m)
	rec := requestDataRecord{InputTokens: m.UsageData.InputTokens.Count, CachedTokens: m.UsageData.CachedTokens.Count}
	rec.applyKV(&m)
	if rec.KV == nil || rec.KV.Sources == nil {
		t.Fatalf("KV = %+v", rec.KV)
	}
	if *rec.KV.Sources != (llm.TokenSources{GPU: 600, Host: 300, External: 100, Computed: 200}) {
		t.Errorf("sources = %+v", *rec.KV.Sources)
	}
	if rec.KV.Timings != nil {
		t.Error("sglang must not invent timings")
	}
	// The pre-existing flat tier fields stay.
	if rec.CachedDevice == nil || *rec.CachedDevice != 600 {
		t.Errorf("cached_device = %v", rec.CachedDevice)
	}
}

func TestApplyKVAbsentRecordsNothing(t *testing.T) {
	var m RequestMetrics
	rec := requestDataRecord{InputTokens: 5}
	rec.applyKV(&m)
	b, _ := json.Marshal(rec)
	if strings.Contains(string(b), `"kv"`) {
		t.Errorf("record carries kv without engine details: %s", b)
	}
}

// Rows with kv survive the JSONL writer/reader; rows written before the field
// existed still load, with KV nil.
func TestRequestDataKVRoundTripAndOldFormat(t *testing.T) {
	dir := t.TempDir()
	w, err := newRequestDataWriter(dir, "m", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	start := time.Unix(1_700_000_000, 0)
	var m RequestMetrics
	consumeOpenAIPlain(strings.NewReader(`{"choices":[{"message":{"content":"ok"}}],"usage":`+goldenUsageJSON(t)+`}`), start, &m)
	rec := requestDataRecord{StartTime: start, EndTime: start.Add(8500 * time.Millisecond), ResponseMs: 8500, TTFT: 3000,
		Model: "m", SeriesNum: 3, Turn: 2, InputTokens: 20, CachedTokens: 80, OutputTokens: 8}
	rec.applyKV(&m)
	if err := w.write(rec); err != nil {
		t.Fatal(err)
	}
	w.close()

	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	// Append an old-format row: no kv key at all, as written by earlier builds.
	f, _ := os.OpenFile(files[0], os.O_APPEND|os.O_WRONLY, 0o644)
	f.WriteString(`{"start_time":"2026-01-01T00:00:00Z","end_time":"2026-01-01T00:00:02Z","ttft_ms":500,"response_time_ms":2000,"model":"m","series_guid":"g","series_num":1,"request_num":1,"turn":1,"cache_hit":false,"cache_hit_ratio":0,"server_cache_confirmed":false,"is_cold_start":true,"input_tokens":10,"output_tokens":4,"cached_tokens":0,"is_error":false,"is_empty":false,"local_cache_ratio":0,"uuid_expected":0,"uuid_found":0,"uuid_leaked":0,"uuid_exact_match":false}` + "\n")
	f.Close()

	recs, _, err := readJSONLFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 2 {
		t.Fatalf("read %d rows, want 2", len(recs))
	}
	assertGoldenKV(t, recs[0].KV)
	if recs[1].KV != nil || recs[1].InputTokens != 10 {
		t.Errorf("old-format row: %+v", recs[1])
	}
	// A report is still generated from the mixed file, with and without the timeline.
	for _, withTL := range []bool{false, true} {
		path, err := renderVisualization(dir, 4, false, 0, "", true, withTL)
		if err != nil {
			t.Fatalf("timeline=%v: %v", withTL, err)
		}
		html, _ := os.ReadFile(path)
		if got := strings.Contains(string(html), "var WekaiTimeline = (function"); got != withTL {
			t.Errorf("timeline=%v: page embeds renderer = %v", withTL, got)
		}
	}
}
