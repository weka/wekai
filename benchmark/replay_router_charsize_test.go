package benchmark

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// TestSizeBudget unit-tests the sizeBudget helper that every replay
// content-sizing site now routes through: token-based sizing when
// charsPerToken > 0 AND the block carries a captured Tokens count,
// byte-faithful sizing (the historical behavior) otherwise.
func TestSizeBudget(t *testing.T) {
	cases := []struct {
		name          string
		bytes         int
		tokens        int
		charsPerToken float64
		want          int
	}{
		{"charsPerToken off falls back to bytes", 50, 100, 0, 50},
		{"charsPerToken negative falls back to bytes", 50, 100, -1, 50},
		{"tokens unset (0) falls back to bytes even with charsPerToken set", 50, 0, 3.4, 50},
		{"token-based sizing", 50, 100, 3.4, 340},
		{"token-based sizing rounds to nearest", 1, 1, 3.4, 3}, // round(3.4) = 3
		{"token-based sizing rounds half up", 1, 5, 0.5, 3},    // round(2.5) = 3 (round-half-away-from-zero)
	}
	for _, c := range cases {
		if got := sizeBudget(c.bytes, c.tokens, c.charsPerToken); got != c.want {
			t.Errorf("%s: sizeBudget(%d, %d, %v) = %d, want %d", c.name, c.bytes, c.tokens, c.charsPerToken, got, c.want)
		}
	}
}

// TestBuildAnthropicMessagesBodyCharsPerToken verifies --replay-chars-per-token
// end to end: with charsPerToken == 0 (default) a system block's synthesized
// text is sized off its captured Bytes (byte-faithful, the pre-existing
// behavior); with charsPerToken > 0 it is instead sized off Tokens *
// charsPerToken, landing near the ORIGINAL capture's token count once the
// serving tokenizer re-counts the synthetic text. Also verifies the
// synthesized content stays deterministic (hash-seeded) across repeated
// builds at the same charsPerToken.
func TestBuildAnthropicMessagesBodyCharsPerToken(t *testing.T) {
	docs := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)

	// Block 0 must be >=200B so effectiveSystemBlocks does not drop it as the
	// per-request header block.
	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{
			{Hash: "sys-known-tokens", Bytes: 500, Tokens: 200},
		},
	}

	getSystemText := func(body []byte) string {
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		sys, ok := parsed["system"].([]interface{})
		if !ok || len(sys) == 0 {
			t.Fatalf("system array missing or empty: %v", parsed["system"])
		}
		block := sys[0].(map[string]interface{})
		return block["text"].(string)
	}

	// --- byte-faithful (default, charsPerToken=0): text sized off Bytes=500 ---
	byteBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, 0, nil)
	if err != nil {
		t.Fatalf("build (charsPerToken=0): %v", err)
	}
	byteText := getSystemText(byteBody)
	if len(byteText) != 500 {
		t.Errorf("byte-faithful system text len = %d, want 500 (Bytes)", len(byteText))
	}

	// --- token-faithful (charsPerToken=3.4): text sized off Tokens=200 ---
	const charsPerToken = 3.4
	wantLen := int(math.Round(200 * charsPerToken))
	tokBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, charsPerToken, nil)
	if err != nil {
		t.Fatalf("build (charsPerToken=3.4): %v", err)
	}
	tokText := getSystemText(tokBody)
	if len(tokText) != wantLen {
		t.Errorf("token-faithful system text len = %d, want %d (tokens*charsPerToken)", len(tokText), wantLen)
	}
	if len(tokText) == len(byteText) {
		t.Error("token-faithful sizing produced the same length as byte-faithful sizing; charsPerToken not applied")
	}

	// --- determinism: same seed + same charsPerToken -> byte-identical content ---
	tokBody2, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, charsPerToken, nil)
	if err != nil {
		t.Fatalf("build (charsPerToken=3.4, repeat): %v", err)
	}
	tokText2 := getSystemText(tokBody2)
	if tokText != tokText2 {
		t.Error("hash-derived content differs across repeated builds at the same charsPerToken; determinism broken")
	}
}

// TestBuildOpenAIChatCompletionsBodyCharsPerToken mirrors the Anthropic test
// above for the OpenAI-side builder: system blocks become the first
// system-role message, sized the same way.
func TestBuildOpenAIChatCompletionsBodyCharsPerToken(t *testing.T) {
	docs := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)

	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{
			{Hash: "sys-known-tokens-openai", Bytes: 400, Tokens: 150},
		},
	}

	getFirstSystemContent := func(body []byte) string {
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		msgs, ok := parsed["messages"].([]interface{})
		if !ok || len(msgs) == 0 {
			t.Fatalf("messages array missing or empty: %v", parsed["messages"])
		}
		msg := msgs[0].(map[string]interface{})
		if msg["role"] != "system" {
			t.Fatalf("first message role = %v, want system", msg["role"])
		}
		return msg["content"].(string)
	}

	byteBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, false, 0, nil, "", "")
	if err != nil {
		t.Fatalf("build (charsPerToken=0): %v", err)
	}
	byteText := getFirstSystemContent(byteBody)
	if len(byteText) != 400 {
		t.Errorf("byte-faithful system content len = %d, want 400 (Bytes)", len(byteText))
	}

	const charsPerToken = 3.4
	wantLen := int(math.Round(150 * charsPerToken))
	tokBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, false, charsPerToken, nil, "", "")
	if err != nil {
		t.Fatalf("build (charsPerToken=3.4): %v", err)
	}
	tokText := getFirstSystemContent(tokBody)
	if len(tokText) != wantLen {
		t.Errorf("token-faithful system content len = %d, want %d (tokens*charsPerToken)", len(tokText), wantLen)
	}
}
