package benchmark

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestOpenAIVsAnthropicBodySize measures the token-count gap between the two
// wire builders before and after the tools-array fix.
//
// Part (a): synthetic fixtures.
// Part (b): real replay file (skipped if absent).
func TestOpenAIVsAnthropicBodySize(t *testing.T) {
	docs := embeddedBenchDoc // deterministic corpus; same as production path

	// ---- (a) synthetic fixtures ----

	toolsSpec := &RouterReplayToolsSpec{
		Count: 8,
		Bytes: 6000,
		Hash:  "synth-tools-hash",
	}
	sysBlocks := []RouterReplaySystemBlock{
		{Hash: "sys-1", Bytes: 800, CacheControl: "ephemeral"},
		{Hash: "sys-2", Bytes: 400},
	}
	msgs := []RouterReplayMessage{
		{Role: "user", Hash: "msg-1", BlockTypes: []string{"text"}, Bytes: 200},
		{Role: "assistant", Hash: "msg-2", BlockTypes: []string{"text", "tool_use"}, Bytes: 300,
			ToolUseIDs: []string{"tu_abc123"}},
		{Role: "user", Hash: "msg-3", BlockTypes: []string{"tool_result", "text"}, Bytes: 300,
			ToolResultIDs: []string{"tu_abc123"}},
	}

	req := RouterReplayRequest{
		Stream:       false,
		MaxTokens:    2048,
		SystemBlocks: sysBlocks,
		Tools:        toolsSpec,
		Messages:     msgs,
	}

	anthBody, _, err := buildAnthropicMessagesBody(req, docs, "model-a", "", 0, false, 0)
	if err != nil {
		t.Fatalf("buildAnthropicMessagesBody: %v", err)
	}
	openaiBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model-a", "", 0, false, 0)
	if err != nil {
		t.Fatalf("buildOpenAIChatCompletionsBody: %v", err)
	}

	anthTok := len(anthBody) / 4
	openaiTok := len(openaiBody) / 4
	ratio := float64(openaiTok) / float64(anthTok)

	t.Logf("SYNTHETIC — Anthropic tokens(est): %d  OpenAI tokens(est): %d  ratio: %.3f",
		anthTok, openaiTok, ratio)

	// Parse OpenAI body and assert tools array presence + shape.
	var parsed map[string]interface{}
	if err := json.Unmarshal(openaiBody, &parsed); err != nil {
		t.Fatalf("unmarshal openai body: %v", err)
	}
	toolsRaw, ok := parsed["tools"]
	if !ok {
		t.Fatal("OpenAI body is missing 'tools' array")
	}
	toolsArr, ok := toolsRaw.([]interface{})
	if !ok {
		t.Fatalf("OpenAI body 'tools' is not an array, got %T", toolsRaw)
	}
	if len(toolsArr) != toolsSpec.Count {
		t.Errorf("OpenAI tools count = %d, want %d", len(toolsArr), toolsSpec.Count)
	}
	for i, raw := range toolsArr {
		entry, ok := raw.(map[string]interface{})
		if !ok {
			t.Errorf("tools[%d] is not an object", i)
			continue
		}
		if entry["type"] != "function" {
			t.Errorf("tools[%d].type = %v, want function", i, entry["type"])
		}
		fn, ok := entry["function"].(map[string]interface{})
		if !ok {
			t.Errorf("tools[%d].function is not an object", i)
			continue
		}
		name, _ := fn["name"].(string)
		if name == "" {
			t.Errorf("tools[%d].function.name is empty", i)
		}
	}

	// Body size should be within 15% of Anthropic for this tools-heavy fixture.
	if ratio < 0.85 {
		t.Errorf("OpenAI/Anthropic size ratio %.3f < 0.85 — tools array is too small", ratio)
	}
	if ratio > 1.15 {
		t.Errorf("OpenAI/Anthropic size ratio %.3f > 1.15 — tools array is unexpectedly large", ratio)
	}

	// ---- (b) real replay file ----
	replayPath := os.ExpandEnv("$HOME/.wekai/router/capture/replays/replay-20-may-2027-20h-weka-v2.jsonl")
	f, err := os.Open(replayPath)
	if err != nil {
		t.Logf("real replay file not found (%v); skipping part (b)", err)
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 128*1024*1024), 128*1024*1024)

	// Line 1: header (consume but ignore for our purposes).
	if !scanner.Scan() {
		t.Log("replay file is empty; skipping part (b)")
		return
	}

	var totalAnthTok, totalOpenaiTok int
	sessCount := 0
	const maxSessions = 300

	for sessCount < maxSessions && scanner.Scan() {
		line := scanner.Bytes()
		var sess RouterReplaySession
		if err := json.Unmarshal(line, &sess); err != nil {
			continue
		}
		sessCount++
		for _, inst := range sess.Instances {
			for _, r := range inst.Requests {
				ab, _, err := buildAnthropicMessagesBody(r, docs, "model-a", "", 0, false, 0)
				if err != nil {
					continue
				}
				ob, _, err := buildOpenAIChatCompletionsBody(r, docs, "model-a", "", 0, false, 0)
				if err != nil {
					continue
				}
				totalAnthTok += len(ab) / 4
				totalOpenaiTok += len(ob) / 4
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Logf("scanner error: %v", err)
	}

	realRatio := float64(0)
	if totalAnthTok > 0 {
		realRatio = float64(totalOpenaiTok) / float64(totalAnthTok)
	}
	t.Logf("REAL FILE (%d sessions) — Anthropic tokens(est): %d  OpenAI tokens(est): %d  ratio: %.3f",
		sessCount, totalAnthTok, totalOpenaiTok, realRatio)
}

// TestOpenAIToolUseConversion verifies that tool_use/tool_result blocks are
// translated into proper OpenAI tool_calls and role="tool" messages, and that
// orphaned tool_result blocks are folded into user text instead.
func TestOpenAIToolUseConversion(t *testing.T) {
	docs := strings.Repeat("abcdefghijklmnopqrstuvwxyz", 20) // ~520 bytes

	req := RouterReplayRequest{
		Stream:    false,
		MaxTokens: 256,
		Messages: []RouterReplayMessage{
			{
				Role:       "assistant",
				Hash:       "asst-msg",
				BlockTypes: []string{"text", "tool_use"},
				ToolUseIDs: []string{"call_1"},
				Bytes:      300,
			},
			{
				Role:          "user",
				Hash:          "user-msg",
				BlockTypes:    []string{"tool_result", "text"},
				ToolResultIDs: []string{"call_1"},
				Bytes:         300,
			},
		},
	}

	body, _, err := buildOpenAIChatCompletionsBody(req, docs, "model-x", "", 0, false, 0)
	if err != nil {
		t.Fatalf("buildOpenAIChatCompletionsBody: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	msgsRaw, ok := parsed["messages"].([]interface{})
	if !ok {
		t.Fatal("messages is not an array")
	}

	// Find assistant message with tool_calls.
	var asstMsg map[string]interface{}
	var toolMsg map[string]interface{}
	for _, raw := range msgsRaw {
		m, _ := raw.(map[string]interface{})
		if m["role"] == "assistant" {
			asstMsg = m
		}
		if m["role"] == "tool" {
			toolMsg = m
		}
		// Verify no placeholder text anywhere.
		content, _ := m["content"].(string)
		if strings.Contains(content, "[tool_use") {
			t.Errorf("message content still contains [tool_use placeholder: %s", content)
		}
		if strings.Contains(content, "[tool_result") {
			t.Errorf("message content still contains [tool_result placeholder: %s", content)
		}
	}

	if asstMsg == nil {
		t.Fatal("no assistant message found")
	}

	// Assert tool_calls on assistant message.
	toolCallsRaw, ok := asstMsg["tool_calls"]
	if !ok {
		t.Fatal("assistant message has no tool_calls")
	}
	toolCalls, ok := toolCallsRaw.([]interface{})
	if !ok || len(toolCalls) == 0 {
		t.Fatal("tool_calls is empty or wrong type")
	}
	tc, _ := toolCalls[0].(map[string]interface{})
	if tc["id"] != "call_1" {
		t.Errorf("tool_call[0].id = %v, want call_1", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_call[0].type = %v, want function", tc["type"])
	}
	fn, _ := tc["function"].(map[string]interface{})
	if fn["name"] == "" {
		t.Error("function.name is empty")
	}
	args, _ := fn["arguments"].(string)
	if args == "" {
		t.Error("function.arguments is empty")
	}
	// arguments must be valid JSON string.
	var argsMap interface{}
	if err := json.Unmarshal([]byte(args), &argsMap); err != nil {
		t.Errorf("function.arguments is not valid JSON: %v (got: %s)", err, args)
	}

	// Assert role="tool" message with matching tool_call_id.
	if toolMsg == nil {
		t.Fatal("no role=tool message found")
	}
	if toolMsg["tool_call_id"] != "call_1" {
		t.Errorf("tool message tool_call_id = %v, want call_1", toolMsg["tool_call_id"])
	}

	// Orphan case: tool_result with no matching prior tool_call must NOT
	// produce a role="tool" message; it must be folded into user text.
	reqOrphan := RouterReplayRequest{
		Stream:    false,
		MaxTokens: 256,
		Messages: []RouterReplayMessage{
			{
				Role:          "user",
				Hash:          "orphan-msg",
				BlockTypes:    []string{"tool_result"},
				ToolResultIDs: []string{"nope_unseen"},
				Bytes:         100,
			},
		},
	}
	orphanBody, _, err := buildOpenAIChatCompletionsBody(reqOrphan, docs, "model-x", "", 0, false, 0)
	if err != nil {
		t.Fatalf("buildOpenAIChatCompletionsBody orphan: %v", err)
	}
	var orphanParsed map[string]interface{}
	if err := json.Unmarshal(orphanBody, &orphanParsed); err != nil {
		t.Fatalf("unmarshal orphan: %v", err)
	}
	orphanMsgs, _ := orphanParsed["messages"].([]interface{})
	for _, raw := range orphanMsgs {
		m, _ := raw.(map[string]interface{})
		if m["role"] == "tool" && m["tool_call_id"] == "nope_unseen" {
			t.Error("orphan tool_result produced a role=tool message — should be folded into user text")
		}
	}
}
