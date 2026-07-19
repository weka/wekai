package benchmark

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildAnthropicMessagesBodyRunGUIDStamp(t *testing.T) {
	// docs must be long enough that synthText has material — ~200 chars.
	docs := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	if len(docs) < 200 {
		t.Fatalf("docs too short: %d bytes", len(docs))
	}
	modelName := "test-model"

	// Shared request with 2 system blocks. Block 0 is >=200B so it is NOT
	// treated as the droppable per-request header (see effectiveSystemBlocks);
	// this test exercises RUN_GUID stamping, not the header-skip.
	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{
			{Hash: "hash-1", Bytes: 250, CacheControl: "ephemeral"},
			{Hash: "hash-2", Bytes: 250},
		},
	}

	// --- runID set ---
	runID := "test-run-id"
	body, _, err := buildAnthropicMessagesBody(req, docs, modelName, runID)
	if err != nil {
		t.Fatalf("buildAnthropicMessagesBody with runID: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal with runID: %v", err)
	}
	sys, ok := parsed["system"].([]interface{})
	if !ok {
		t.Fatal("system is not an array")
	}
	if len(sys) != 3 {
		t.Fatalf("expected 3 system blocks with runID, got %d", len(sys))
	}
	sys0 := sys[0].(map[string]interface{})
	if sys0["type"] != "text" {
		t.Errorf("stamp block type = %v, want text", sys0["type"])
	}
	if sys0["text"] != "<ignore>RUN_GUID: test-run-id</ignore>" {
		t.Errorf("stamp text = %q, want %q", sys0["text"], "<ignore>RUN_GUID: test-run-id</ignore>")
	}
	if _, hasCC := sys0["cache_control"]; hasCC {
		t.Error("stamp block must NOT have cache_control")
	}

	// --- runID empty ---
	body, _, err = buildAnthropicMessagesBody(req, docs, modelName, "")
	if err != nil {
		t.Fatalf("buildAnthropicMessagesBody empty runID: %v", err)
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal empty runID: %v", err)
	}
	sys, ok = parsed["system"].([]interface{})
	if !ok {
		t.Fatal("system is not an array (empty runID)")
	}
	if len(sys) != 2 {
		t.Fatalf("expected 2 system blocks with empty runID, got %d", len(sys))
	}
	sys0Text, _ := sys[0].(map[string]interface{})["text"].(string)
	if sys0Text == "<ignore>RUN_GUID: test-run-id</ignore>" {
		t.Error("stamp should NOT be present when runID is empty")
	}

	// --- runID set, 0 system blocks ---
	reqNoSys := RouterReplayRequest{} // no SystemBlocks
	body, _, err = buildAnthropicMessagesBody(reqNoSys, docs, modelName, runID)
	if err != nil {
		t.Fatalf("buildAnthropicMessagesBody no system blocks: %v", err)
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal no system blocks: %v", err)
	}
	sys, ok = parsed["system"].([]interface{})
	if !ok {
		t.Fatal("system is not an array (0 system blocks)")
	}
	if len(sys) != 1 {
		t.Fatalf("expected 1 system block (stamp only) with 0 system blocks, got %d", len(sys))
	}
	sys0 = sys[0].(map[string]interface{})
	if sys0["text"] != "<ignore>RUN_GUID: test-run-id</ignore>" {
		t.Errorf("stamp text with 0 blocks = %q", sys0["text"])
	}
	if _, hasCC := sys0["cache_control"]; hasCC {
		t.Error("stamp block must NOT have cache_control (0 system blocks)")
	}
}

// TestBuildAnthropicMessagesBodyCanonicalDeterminism verifies:
// identical request → identical canonical.
func TestBuildAnthropicMessagesBodyCanonicalDeterminism(t *testing.T) {
	docs := strings.Repeat("doc", 200)
	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "h1", Bytes: 50}},
		Messages: []RouterReplayMessage{
			{Hash: "m1", Role: "user", BlockTypes: []string{"text"}, Bytes: 30},
		},
	}
	_, c1, err1 := buildAnthropicMessagesBody(req, docs, "model", "run1")
	_, c2, err2 := buildAnthropicMessagesBody(req, docs, "model", "run1")
	if err1 != nil || err2 != nil {
		t.Fatalf("errors: %v %v", err1, err2)
	}
	if c1 != c2 {
		t.Error("canonical not deterministic for identical request")
	}
	if c1 == "" {
		t.Error("canonical must not be empty")
	}
}

// TestBuildAnthropicMessagesBodyCanonicalContainsAllBlocks verifies
// canonical contains text from all block types.
func TestBuildAnthropicMessagesBodyCanonicalContainsAllBlocks(t *testing.T) {
	docs := strings.Repeat("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 100)
	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 100}},
		Tools:        &RouterReplayToolsSpec{Hash: "tools1", Bytes: 500, Count: 2},
		Messages: []RouterReplayMessage{
			{Hash: "msg1", Role: "user", BlockTypes: []string{"text"}, Bytes: 100},
		},
	}
	_, canonical, err := buildAnthropicMessagesBody(req, docs, "model", "runX")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(canonical) < 100 {
		t.Errorf("canonical too short: %d bytes — must include system+tools+messages", len(canonical))
	}
}

// TestEffectiveSystemBlocksSkipsHeader verifies the per-request <200B header
// block (SystemBlocks[0]) is dropped from the wire body and the cache estimator,
// while the shared system blocks behind it are preserved. This is the replay
// hit-rate fix: the unique-per-request header poisons vLLM's sequential prefix
// cache; dropping it lets the GPU/weka cache key on the stable shared prompt.
func TestEffectiveSystemBlocksSkipsHeader(t *testing.T) {
	// effectiveSystemBlocks unit behavior.
	cases := []struct {
		name string
		in   []RouterReplaySystemBlock
		want int
	}{
		{"drops <200B block0", []RouterReplaySystemBlock{{Hash: "hdr", Bytes: 106}, {Hash: "sys1", Bytes: 5000}, {Hash: "sys2", Bytes: 3000}}, 2},
		{"keeps >=200B block0", []RouterReplaySystemBlock{{Hash: "sys0", Bytes: 5000}, {Hash: "sys1", Bytes: 3000}}, 2},
		{"single small block dropped", []RouterReplaySystemBlock{{Hash: "hdr", Bytes: 106}}, 0},
		{"empty", nil, 0},
	}
	for _, c := range cases {
		if got := len(effectiveSystemBlocks(c.in)); got != c.want {
			t.Errorf("%s: effectiveSystemBlocks len = %d, want %d", c.name, got, c.want)
		}
	}

	// Wire-level: the unique header must NOT appear in body or canonical, so the
	// shared sys1 block leads the system content (stable key across requests).
	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{
			{Hash: "uniq-header-per-req", Bytes: 106},
			{Hash: "shared-sys1", Bytes: 5000},
		},
	}
	hdrText := synthText("uniq-header-per-req", 106, "")
	for _, builder := range []struct {
		name string
		fn   func(RouterReplayRequest, string, string, string) ([]byte, string, error)
	}{
		{"openai", buildOpenAIChatCompletionsBody},
		{"anthropic", buildAnthropicMessagesBody},
	} {
		body, canonical, err := builder.fn(req, "", "m", "")
		if err != nil {
			t.Fatalf("%s build: %v", builder.name, err)
		}
		if strings.Contains(string(body), hdrText) {
			t.Errorf("%s: body still contains the dropped header text", builder.name)
		}
		if strings.Contains(canonical, hdrText) {
			t.Errorf("%s: canonical still contains the dropped header text", builder.name)
		}
	}
}
