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
	body, _, err := buildAnthropicMessagesBody(req, docs, modelName, runID, 0, false, replaySizer{})
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
	body, _, err = buildAnthropicMessagesBody(req, docs, modelName, "", 0, false, replaySizer{})
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
	body, _, err = buildAnthropicMessagesBody(reqNoSys, docs, modelName, runID, 0, false, replaySizer{})
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
	_, c1, err1 := buildAnthropicMessagesBody(req, docs, "model", "run1", 0, false, replaySizer{})
	_, c2, err2 := buildAnthropicMessagesBody(req, docs, "model", "run1", 0, false, replaySizer{})
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
	_, canonical, err := buildAnthropicMessagesBody(req, docs, "model", "runX", 0, false, replaySizer{})
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
		fn   func(RouterReplayRequest, string, string, string, float64, bool, replaySizer) ([]byte, string, error)
	}{
		{"openai", buildOpenAIChatCompletionsBody},
		{"anthropic", buildAnthropicMessagesBody},
	} {
		body, canonical, err := builder.fn(req, "", "m", "", 0, false, replaySizer{})
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

// TestPickMaxTokensOutputRatio verifies --replay-output-ratio retargeting:
// when outputRatio > 0 and InputTokens is known, max_tokens is retargeted to
// round(InputTokens * outputRatio), overriding the recorded output_tokens.
// With outputRatio == 0, the original output_tokens/max_tokens precedence is
// unchanged.
func TestPickMaxTokensOutputRatio(t *testing.T) {
	cases := []struct {
		name        string
		req         RouterReplayRequest
		outputRatio float64
		want        int
	}{
		{
			name:        "ratio retargets above original output_tokens",
			req:         RouterReplayRequest{InputTokens: 1000, OutputTokens: 20},
			outputRatio: 0.5,
			want:        500,
		},
		{
			name:        "ratio zero falls back to original output_tokens",
			req:         RouterReplayRequest{InputTokens: 1000, OutputTokens: 20},
			outputRatio: 0,
			want:        20,
		},
		{
			name:        "ratio rounds to nearest",
			req:         RouterReplayRequest{InputTokens: 999},
			outputRatio: 0.3333,
			want:        333, // round(999*0.3333) = round(332.9667) = 333
		},
		{
			name:        "ratio floors at 1, never 0",
			req:         RouterReplayRequest{InputTokens: 1},
			outputRatio: 0.1,
			want:        1, // round(1*0.1) = round(0.1) = 0, floored to 1
		},
		{
			name:        "ratio set but InputTokens unknown falls back",
			req:         RouterReplayRequest{OutputTokens: 42},
			outputRatio: 0.5,
			want:        42,
		},
		{
			name:        "ratio sizes off full input (gross), cache_read ignored",
			req:         RouterReplayRequest{InputTokens: 2000, CacheReadTokens: 1500, OutputTokens: 10},
			outputRatio: 0.5,
			want:        1000, // round(2000 * 0.5); full prompt (gross) basis
		},
		{
			name:        "no ratio, no output_tokens, falls back to max_tokens",
			req:         RouterReplayRequest{MaxTokens: 77},
			outputRatio: 0,
			want:        77,
		},
		{
			name:        "no ratio, nothing set, guard of 1",
			req:         RouterReplayRequest{},
			outputRatio: 0,
			want:        1,
		},
	}
	for _, c := range cases {
		if got := pickMaxTokens(c.req, c.outputRatio); got != c.want {
			t.Errorf("%s: pickMaxTokens() = %d, want %d", c.name, got, c.want)
		}
	}
}

// TestBuildAnthropicMessagesBodyForceOutput verifies the continue-generating
// instruction is injected as an appended system block when forceOutput is
// true, and absent when false. Anthropic has no ignore_eos equivalent, so
// forceOutput only affects the system block here.
func TestBuildAnthropicMessagesBodyForceOutput(t *testing.T) {
	docs := strings.Repeat("doc content ", 50)
	req := RouterReplayRequest{
		InputTokens: 100,
		SystemBlocks: []RouterReplaySystemBlock{
			{Hash: "sys1", Bytes: 5000}, // >=200B so not dropped as the per-request header
		},
	}

	// --- force-output off (--replay-natural-output) ---
	body, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, replaySizer{})
	if err != nil {
		t.Fatalf("build (force-output off): %v", err)
	}
	if strings.Contains(string(body), verboseOutputInstruction) {
		t.Error("continue-generating instruction present in body when forceOutput=false")
	}

	// --- force-output on (default) ---
	body, _, err = buildAnthropicMessagesBody(req, docs, "model", "", 0, true, replaySizer{})
	if err != nil {
		t.Fatalf("build (force-output on): %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal (force-output on): %v", err)
	}
	sys, ok := parsed["system"].([]interface{})
	if !ok {
		t.Fatal("system is not an array (force-output on)")
	}
	last := sys[len(sys)-1].(map[string]interface{})
	if last["text"] != verboseOutputInstruction {
		t.Errorf("last system block = %q, want continue-generating instruction", last["text"])
	}
	if !strings.Contains(string(body), verboseOutputInstruction) {
		t.Error("continue-generating instruction missing from body when forceOutput=true")
	}

	// --- force-output on, no system blocks at all: instruction must still be injected ---
	reqNoSys := RouterReplayRequest{InputTokens: 100}
	body, _, err = buildAnthropicMessagesBody(reqNoSys, docs, "model", "", 0, true, replaySizer{})
	if err != nil {
		t.Fatalf("build (force-output on, no system blocks): %v", err)
	}
	if !strings.Contains(string(body), verboseOutputInstruction) {
		t.Error("continue-generating instruction missing when there were no other system blocks")
	}
}

// TestBuildOpenAIChatCompletionsBodyForceOutput verifies that for the OpenAI
// (vLLM) body builder, forceOutput forces full output via BOTH the appended
// continue-generating system message AND the vLLM ignore_eos sampling flag —
// the model generates up to max_tokens instead of stopping early.
func TestBuildOpenAIChatCompletionsBodyForceOutput(t *testing.T) {
	docs := strings.Repeat("doc content ", 50)
	req := RouterReplayRequest{
		InputTokens: 100,
		SystemBlocks: []RouterReplaySystemBlock{
			{Hash: "sys1", Bytes: 5000},
		},
	}

	// force-output off: neither ignore_eos nor the instruction in the body.
	body, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, false, replaySizer{})
	if err != nil {
		t.Fatalf("build (force-output off): %v", err)
	}
	var off map[string]interface{}
	if err := json.Unmarshal(body, &off); err != nil {
		t.Fatalf("unmarshal (force-output off): %v", err)
	}
	if _, ok := off["ignore_eos"]; ok {
		t.Error("ignore_eos present in body when forceOutput=false")
	}
	if strings.Contains(string(body), verboseOutputInstruction) {
		t.Error("continue-generating instruction present in body when forceOutput=false")
	}

	// force-output on (default): ignore_eos=true AND the instruction is present.
	body, _, err = buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, true, replaySizer{})
	if err != nil {
		t.Fatalf("build (force-output on): %v", err)
	}
	var on map[string]interface{}
	if err := json.Unmarshal(body, &on); err != nil {
		t.Fatalf("unmarshal (force-output on): %v", err)
	}
	if on["ignore_eos"] != true {
		t.Errorf("ignore_eos = %v, want true when forceOutput=true", on["ignore_eos"])
	}
	if !strings.Contains(string(body), verboseOutputInstruction) {
		t.Error("continue-generating instruction missing from body when forceOutput=true")
	}
}

// TestBuildAnthropicMessagesBodyOutputRatioMaxTokens verifies the emitted
// max_tokens reflects the retargeted value when --replay-output-ratio is set,
// for both the Anthropic and OpenAI body builders.
func TestBuildAnthropicMessagesBodyOutputRatioMaxTokens(t *testing.T) {
	docs := strings.Repeat("doc content ", 50)
	req := RouterReplayRequest{InputTokens: 2000, OutputTokens: 10}

	anthBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0.25, false, replaySizer{})
	if err != nil {
		t.Fatalf("anthropic build: %v", err)
	}
	var anthParsed map[string]interface{}
	if err := json.Unmarshal(anthBody, &anthParsed); err != nil {
		t.Fatalf("anthropic unmarshal: %v", err)
	}
	if got, want := anthParsed["max_tokens"].(float64), 500.0; got != want {
		t.Errorf("anthropic max_tokens = %v, want %v", got, want)
	}

	openaiBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0.25, false, replaySizer{})
	if err != nil {
		t.Fatalf("openai build: %v", err)
	}
	var openaiParsed map[string]interface{}
	if err := json.Unmarshal(openaiBody, &openaiParsed); err != nil {
		t.Fatalf("openai unmarshal: %v", err)
	}
	if got, want := openaiParsed["max_tokens"].(float64), 500.0; got != want {
		t.Errorf("openai max_tokens = %v, want %v", got, want)
	}

	// Without a ratio, max_tokens falls back to the original output_tokens.
	anthBodyNoRatio, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, replaySizer{})
	if err != nil {
		t.Fatalf("anthropic build (no ratio): %v", err)
	}
	if err := json.Unmarshal(anthBodyNoRatio, &anthParsed); err != nil {
		t.Fatalf("anthropic unmarshal (no ratio): %v", err)
	}
	if got, want := anthParsed["max_tokens"].(float64), 10.0; got != want {
		t.Errorf("anthropic max_tokens (no ratio) = %v, want %v", got, want)
	}
}
