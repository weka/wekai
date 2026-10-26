package mockvllm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The engine's WEKA_PREFILL_REJECT_TOKENS contract: a request whose recompute
// exceeds the limit is refused with a 503 naming the recompute (header and body), unless it
// carries vllm_xargs.weka_prefill — and what the shared tier already holds does
// not count as recompute.

func rejectServer(t *testing.T, limit int, tier *Tier) *httptest.Server {
	t.Helper()
	eng := NewEngine(Config{PrefillRejectTokens: limit, Tier: tier, DefaultMaxTokens: 2})
	t.Cleanup(eng.Close)
	ts := httptest.NewServer(NewServer(eng).Handler())
	t.Cleanup(ts.Close)
	return ts
}

func postChat(t *testing.T, base, content, xargs string) (int, map[string]any) {
	t.Helper()
	c, _ := json.Marshal(content)
	body := `{"model":"m","max_tokens":2,"messages":[{"role":"user","content":` + string(c) + `}]` + xargs + `}`
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func TestHeavyRequestIsRejectedWithItsRecomputeUnlessBypassed(t *testing.T) {
	ts := rejectServer(t, 1000, nil)
	prompt := strings.Repeat("lorem ipsum ", 2000)

	status, body := postChat(t, ts.URL, prompt, "")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", status)
	}
	e, _ := body["error"].(map[string]any)
	if e["type"] != LargePrefillRejectedType {
		t.Errorf("error type = %v, want %s", e["type"], LargePrefillRejectedType)
	}
	if p, _ := e["param"].(string); p == "" || p == "0" {
		t.Errorf("param = %q, want the recompute token count", p)
	}

	if status, _ := postChat(t, ts.URL, prompt, `,"vllm_xargs":{"weka_prefill":1}`); status != http.StatusOK {
		t.Errorf("bypassed request status = %d, want 200", status)
	}
	if status, _ := postChat(t, ts.URL, "a short prompt", ""); status != http.StatusOK {
		t.Errorf("light request status = %d, want 200", status)
	}
}

func TestRejectionIsOffByDefault(t *testing.T) {
	ts := rejectServer(t, 0, nil)
	if status, _ := postChat(t, ts.URL, strings.Repeat("lorem ipsum ", 5000), ""); status != http.StatusOK {
		t.Errorf("status = %d, want 200 with no limit configured", status)
	}
}

func TestSharedTierContentIsNotRecompute(t *testing.T) {
	tier := NewTier()
	prefiller := rejectServer(t, 1000, tier)
	other := rejectServer(t, 1000, tier)
	prompt := strings.Repeat("lorem ipsum ", 2000)

	if status, _ := postChat(t, other.URL, prompt, ""); status != http.StatusServiceUnavailable {
		t.Fatalf("cold fleet status = %d, want 503", status)
	}
	if status, _ := postChat(t, prefiller.URL, prompt, `,"vllm_xargs":{"weka_prefill":1}`); status != http.StatusOK {
		t.Fatalf("prefill job status = %d", status)
	}
	if status, body := postChat(t, other.URL, prompt, ""); status != http.StatusOK {
		t.Errorf("after the prefill published to the tier, a peer still refused it: %d %v", status, body)
	}
}
