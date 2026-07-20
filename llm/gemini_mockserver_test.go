package llm

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/weka/wekai/llm/mockserver"
	"github.com/weka/wekai/tools"
)

// newGeminiTestClient creates a GeminiLLMClient pointing at the given base URL.
// Bypasses GetChatGetter API-key validation; suitable for unit tests with a mock server.
func newGeminiTestClient(baseURL string) *GeminiLLMClient {
	config := LLMConfig{
		Model:                  "gemini-2.5-flash-test",
		APIKey:                 "test-api-key",
		BaseURL:                baseURL,
		MaxTokens:              1024,
		StreamResponseCallback: func(string) {},
		StreamThinkingCallback: func(string) {},
	}
	return NewGeminiLLMClient(config).(*GeminiLLMClient)
}

// TestGeminiMockServer_SimpleTextResponse verifies that a plain text response
// is correctly parsed from the Gemini streaming JSON array format.
func TestGeminiMockServer_SimpleTextResponse(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{Content: "Hello from Gemini!"},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	if resp.Content != "Hello from Gemini!" {
		t.Errorf("expected 'Hello from Gemini!', got %q", resp.Content)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(resp.ToolCalls))
	}
	if srv.CallCount() != 1 {
		t.Errorf("expected 1 call, got %d", srv.CallCount())
	}
}

// TestGeminiMockServer_ToolCall verifies that function_call parts are correctly
// parsed and returned as ToolCalls.
func TestGeminiMockServer_ToolCall(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			ToolCalls: []mockserver.GeminiToolCall{
				{
					Name: "increment_by_one",
					Args: map[string]interface{}{"current_value": 1},
				},
			},
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	toolset := buildTestToolset()
	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "start"}}, toolset)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "increment_by_one" {
		t.Errorf("expected tool name 'increment_by_one', got %q", resp.ToolCalls[0].Name)
	}
	if !strings.Contains(resp.ToolCalls[0].Args, "current_value") {
		t.Errorf("expected args to contain 'current_value', got %q", resp.ToolCalls[0].Args)
	}
}

// TestGeminiMockServer_ThoughtSignaturePreservedInHistory is the key regression test
// for the thought_signature bug fix. It verifies that when the model returns a
// function call with a thought_signature, that signature is preserved in the
// conversation history sent in subsequent requests.
//
// Without the fix (storing just GeminiFunctionCall instead of the full GeminiPart),
// the thought_signature would be lost, causing a 400 error on the next request:
// "Function call is missing a thought_signature in functionCall parts."
func TestGeminiMockServer_ThoughtSignaturePreservedInHistory(t *testing.T) {
	const thoughtSig = "abc123thoughtsig"

	// Turn 1: model returns a tool call WITH thought_signature
	// Turn 2: model returns final text answer
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			ToolCalls: []mockserver.GeminiToolCall{
				{
					Name:             "increment_by_one",
					Args:             map[string]interface{}{"current_value": 1},
					ThoughtSignature: thoughtSig,
				},
			},
		},
		{
			Content: "The answer is 2",
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	toolset := buildTestToolset()

	// Turn 1: first request triggers tool call
	resp1, err := client.Request(ctx, []ContentPart{&TextContent{Text: "increment from 1"}}, toolset)
	if err != nil {
		t.Fatalf("Turn 1 request failed: %v", err)
	}
	if len(resp1.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call in turn 1, got %d", len(resp1.ToolCalls))
	}

	// Turn 2: provide tool result; this is where thought_signature matters —
	// the mock server (or real Gemini) would reject the request if the
	// function call in history is missing its thought_signature.
	resp2, err := client.Respond(ctx, map[string]string{
		"increment_by_one": "2",
	}, nil, toolset)
	if err != nil {
		t.Fatalf("Turn 2 request failed (thought_signature bug would manifest here): %v", err)
	}
	if resp2.Content != "The answer is 2" {
		t.Errorf("expected 'The answer is 2', got %q", resp2.Content)
	}

	if srv.CallCount() != 2 {
		t.Errorf("expected 2 API calls, got %d", srv.CallCount())
	}

	// Verify the thought_signature was preserved in the history by inspecting
	// the client's internal history.
	found := false
	for _, msg := range client.history {
		for _, part := range msg.Parts {
			if part.ThoughtSignature == thoughtSig {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("thought_signature %q was not preserved in conversation history", thoughtSig)
	}
}

// TestGeminiMockServer_MultiTurnToolChain verifies a multi-turn tool-use conversation
// works end-to-end with the mock server: tool call → result → tool call → result → final answer.
func TestGeminiMockServer_MultiTurnToolChain(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			// Turn 1: call tool with value 1
			ToolCalls: []mockserver.GeminiToolCall{
				{Name: "increment_by_one", Args: map[string]interface{}{"current_value": 1}},
			},
		},
		{
			// Turn 2: call tool again with value 2
			ToolCalls: []mockserver.GeminiToolCall{
				{Name: "increment_by_one", Args: map[string]interface{}{"current_value": 2}},
			},
		},
		{
			// Turn 3: final text answer
			Content: "3",
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()
	toolset := buildTestToolset()

	// Turn 1
	resp1, err := client.Request(ctx, []ContentPart{&TextContent{Text: "increment to 3"}}, toolset)
	if err != nil {
		t.Fatalf("Turn 1 failed: %v", err)
	}
	if len(resp1.ToolCalls) != 1 {
		t.Fatalf("Turn 1: expected 1 tool call, got %d", len(resp1.ToolCalls))
	}

	// Turn 2: send result of first tool call
	resp2, err := client.Respond(ctx, map[string]string{"increment_by_one": "2"}, nil, toolset)
	if err != nil {
		t.Fatalf("Turn 2 failed: %v", err)
	}
	if len(resp2.ToolCalls) != 1 {
		t.Fatalf("Turn 2: expected 1 tool call, got %d", len(resp2.ToolCalls))
	}

	// Turn 3: send result of second tool call
	resp3, err := client.Respond(ctx, map[string]string{"increment_by_one": "3"}, nil, toolset)
	if err != nil {
		t.Fatalf("Turn 3 failed: %v", err)
	}
	if resp3.Content != "3" {
		t.Errorf("Turn 3: expected '3', got %q", resp3.Content)
	}
	if len(resp3.ToolCalls) != 0 {
		t.Errorf("Turn 3: expected no tool calls, got %d", len(resp3.ToolCalls))
	}

	if srv.CallCount() != 3 {
		t.Errorf("expected 3 API calls, got %d", srv.CallCount())
	}
}

// TestGeminiMockServer_UsageMetadata verifies that usage tokens from the response
// are correctly parsed and returned in Response.Usage.
func TestGeminiMockServer_UsageMetadata(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			Content:       "test response",
			PromptTokens:  150,
			OutputTokens:  25,
			ThoughtTokens: 50,
			CachedTokens:  30,
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "test"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	// Gemini client subtracts CachedTokens from PromptTokens in usage accounting
	expectedPrompt := 150 - 30 // 120
	if resp.Usage.PromptTokens != expectedPrompt {
		t.Errorf("expected PromptTokens=%d, got %d", expectedPrompt, resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 25 {
		t.Errorf("expected CompletionTokens=25, got %d", resp.Usage.CompletionTokens)
	}
	if resp.Usage.ReasoningTokens != 50 {
		t.Errorf("expected ReasoningTokens=50, got %d", resp.Usage.ReasoningTokens)
	}
	if resp.Usage.CachedTokens != 30 {
		t.Errorf("expected CachedTokens=30, got %d", resp.Usage.CachedTokens)
	}
}

// TestGeminiMockServer_HTTPError verifies that HTTP errors from the API are
// returned as APIError with the correct status code.
func TestGeminiMockServer_HTTPError(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			Error: &mockserver.GeminiMockError{
				StatusCode: 400,
				Message:    "Function call is missing a thought_signature in functionCall parts.",
			},
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	_, err := client.Request(ctx, []ContentPart{&TextContent{Text: "test"}}, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("expected status 400, got %d", apiErr.StatusCode)
	}
	if !strings.Contains(apiErr.Body, "thought_signature") {
		t.Errorf("expected error body to mention thought_signature, got %q", apiErr.Body)
	}
}

// TestGeminiMockServer_ScriptExhausted verifies that calling beyond the scripted
// turns returns an error.
func TestGeminiMockServer_ScriptExhausted(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{Content: "only one"},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	// First call succeeds
	_, err := client.Request(ctx, []ContentPart{&TextContent{Text: "hi"}}, nil)
	if err != nil {
		t.Fatalf("First call failed: %v", err)
	}

	// Second call should fail — script exhausted
	client2 := newGeminiTestClient(srv.BaseURL)
	_, err = client2.Request(ctx, []ContentPart{&TextContent{Text: "hi again"}}, nil)
	if err == nil {
		t.Fatal("expected error on exhausted script, got nil")
	}
}

// TestGeminiMockServer_Reset verifies that Reset clears the call counter
// and replaces the script.
func TestGeminiMockServer_Reset(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{Content: "first"},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	_, err := client.Request(ctx, []ContentPart{&TextContent{Text: "hi"}}, nil)
	if err != nil {
		t.Fatalf("First call failed: %v", err)
	}
	if srv.CallCount() != 1 {
		t.Fatalf("expected 1 call, got %d", srv.CallCount())
	}

	srv.Reset([]mockserver.GeminiMockTurn{{Content: "second"}})
	if srv.CallCount() != 0 {
		t.Fatalf("expected 0 after reset, got %d", srv.CallCount())
	}

	client2 := newGeminiTestClient(srv.BaseURL)
	resp, err := client2.Request(ctx, []ContentPart{&TextContent{Text: "hi again"}}, nil)
	if err != nil {
		t.Fatalf("Second call failed: %v", err)
	}
	if resp.Content != "second" {
		t.Errorf("expected 'second', got %q", resp.Content)
	}
}

// TestGeminiMockServer_SetSystemBlocks verifies that SetSystemBlocks sets the system
// parts (not fake history turns), and they are sent as system_instruction in requests.
func TestGeminiMockServer_SetSystemBlocks(t *testing.T) {
	const systemPrompt = "You are a helpful assistant."

	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{Content: "response after intro", PromptTokens: 10, OutputTokens: 3},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	client.SetSystemBlocks([]SystemBlock{{ID: "root", Content: systemPrompt, Cache: true}})

	// History should be EMPTY — no fake turns added
	if len(client.history) != 0 {
		t.Fatalf("expected 0 history entries after SetSystemBlocks, got %d", len(client.history))
	}
	// systemParts should be set (with preamble prepended)
	if len(client.systemParts) != 1 {
		t.Errorf("expected 1 systemPart, got %d", len(client.systemParts))
	}

	ctx := context.Background()
	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "hello"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.Content != "response after intro" {
		t.Errorf("expected 'response after intro', got %q", resp.Content)
	}

	// Verify the request contained system_instruction
	req := srv.LastRequest()
	if req == nil {
		t.Fatal("expected captured request, got nil")
	}
	sysInstr, ok := req["system_instruction"]
	if !ok {
		t.Fatalf("expected system_instruction in request, keys: %v", req)
	}
	sysInstrMap, ok := sysInstr.(map[string]interface{})
	if !ok {
		t.Fatalf("expected system_instruction to be a map, got %T", sysInstr)
	}
	parts, ok := sysInstrMap["parts"].([]interface{})
	if !ok || len(parts) == 0 {
		t.Fatalf("expected system_instruction.parts to be a non-empty slice, got %v", sysInstrMap["parts"])
	}
	partMap, ok := parts[0].(map[string]interface{})
	if !ok {
		t.Fatalf("expected first part to be a map, got %T", parts[0])
	}
	// The preamble is prepended to the first block
	const preamble = "Strictly follow instructions set as system prompt\n\n"
	expectedText := preamble + systemPrompt
	if partMap["text"] != expectedText {
		t.Errorf("expected system_instruction text %q, got %q", expectedText, partMap["text"])
	}

	// Verify ClearHistory also clears systemParts
	client.ClearHistory()
	if len(client.systemParts) != 0 {
		t.Errorf("expected systemParts cleared after ClearHistory, got %v", client.systemParts)
	}
}

// TestGeminiMockServer_MultipleToolCallsInOneTurn verifies that multiple function_call
// parts in a single response are all correctly parsed.
func TestGeminiMockServer_MultipleToolCallsInOneTurn(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			ToolCalls: []mockserver.GeminiToolCall{
				{Name: "tool_a", Args: map[string]interface{}{"x": 1}},
				{Name: "tool_b", Args: map[string]interface{}{"y": 2}},
			},
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "call two tools"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	if len(resp.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "tool_a" {
		t.Errorf("expected first tool 'tool_a', got %q", resp.ToolCalls[0].Name)
	}
	if resp.ToolCalls[1].Name != "tool_b" {
		t.Errorf("expected second tool 'tool_b', got %q", resp.ToolCalls[1].Name)
	}
	// Unique call IDs
	if resp.ToolCalls[0].CallId == resp.ToolCalls[1].CallId {
		t.Errorf("expected unique CallIds, both are %q", resp.ToolCalls[0].CallId)
	}
}

// TestGeminiMockServer_ThoughtSignatureInSyntheticJSON verifies that the thought_signature
// is present in the RawResponseJSON (the synthetic JSON used for state recovery).
// This is the core regression test for the bug where thought_signature was lost in
// the synthetic response builder, causing 400 errors on state recovery.
func TestGeminiMockServer_ThoughtSignatureInSyntheticJSON(t *testing.T) {
	const thoughtSig = "real_base64_thought_signature"

	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{
			ToolCalls: []mockserver.GeminiToolCall{
				{
					Name:             "increment_by_one",
					Args:             map[string]interface{}{"current_value": 1},
					ThoughtSignature: thoughtSig,
				},
			},
		},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)
	ctx := context.Background()

	toolset := buildTestToolset()
	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "increment from 1"}}, toolset)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}

	// The RawResponseJSON is the synthetic JSON used for state recovery.
	// It MUST contain thought_signature so that replaying it to Gemini won't produce a 400.
	if len(resp.RawResponseJSON) == 0 {
		t.Fatal("RawResponseJSON is empty")
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(resp.RawResponseJSON, &parsed); err != nil {
		t.Fatalf("failed to parse RawResponseJSON: %v", err)
	}

	candidates, ok := parsed["candidates"].([]interface{})
	if !ok || len(candidates) == 0 {
		t.Fatal("RawResponseJSON missing candidates")
	}
	content, ok := candidates[0].(map[string]interface{})["content"].(map[string]interface{})
	if !ok {
		t.Fatal("RawResponseJSON missing content")
	}
	parts, ok := content["parts"].([]interface{})
	if !ok || len(parts) == 0 {
		t.Fatal("RawResponseJSON missing parts")
	}

	// Find the function call part and verify thought_signature is present.
	found := false
	for _, p := range parts {
		part, ok := p.(map[string]interface{})
		if !ok {
			continue
		}
		if _, hasFuncCall := part["functionCall"]; hasFuncCall {
			sig, hasSig := part["thoughtSignature"]
			if !hasSig {
				t.Error("functionCall part in RawResponseJSON is missing thoughtSignature")
			}
			if sig != thoughtSig {
				t.Errorf("thoughtSignature in RawResponseJSON: expected %q, got %q", thoughtSig, sig)
			}
			found = true
		}
	}
	if !found {
		t.Error("no functionCall part found in RawResponseJSON")
	}
}

// TestGeminiMockServer_ToolOrdering verifies that tools are sent in alphabetical
// order regardless of Go map iteration order, ensuring stable LLM cache keys.
func TestGeminiMockServer_ToolOrdering(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{Content: "done", PromptTokens: 10, OutputTokens: 3},
	})
	defer srv.Close()

	client := newGeminiTestClient(srv.BaseURL)

	// Build toolset with tools in non-alphabetical order
	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"zebra_tool":  {Name: "zebra_tool", Description: "Z tool", Properties: map[string]jsonschema.Definition{}},
			"alpha_tool":  {Name: "alpha_tool", Description: "A tool", Properties: map[string]jsonschema.Definition{}},
			"middle_tool": {Name: "middle_tool", Description: "M tool", Properties: map[string]jsonschema.Definition{}},
		},
	}

	_, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}

	req := srv.LastRequest()
	if req == nil {
		t.Fatal("expected captured request, got nil")
	}
	toolsRaw, ok := req["tools"].([]interface{})
	if !ok || len(toolsRaw) == 0 {
		t.Fatalf("expected tools in request, got: %v", req["tools"])
	}

	// Extract tool names in the order they were sent
	var names []string
	for _, toolRaw := range toolsRaw {
		toolMap, ok := toolRaw.(map[string]interface{})
		if !ok {
			continue
		}
		decls, ok := toolMap["functionDeclarations"].([]interface{})
		if !ok {
			continue
		}
		for _, decl := range decls {
			declMap, ok := decl.(map[string]interface{})
			if !ok {
				continue
			}
			if name, ok := declMap["name"].(string); ok {
				names = append(names, name)
			}
		}
	}

	expected := []string{"alpha_tool", "middle_tool", "zebra_tool"}
	if !reflect.DeepEqual(names, expected) {
		t.Errorf("tools not in alphabetical order: got %v, want %v", names, expected)
	}
}

// TestGeminiMockServer_PropertyOrdering verifies that tool properties are sent
// in deterministic alphabetical order across multiple requests (stable cache key).
func TestGeminiMockServer_PropertyOrdering(t *testing.T) {
	srv := mockserver.NewGeminiMockServer([]mockserver.GeminiMockTurn{
		{Content: "done1", PromptTokens: 10, OutputTokens: 3},
		{Content: "done2", PromptTokens: 10, OutputTokens: 3},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"my_tool": {
				Name:        "my_tool",
				Description: "test tool",
				Properties: map[string]jsonschema.Definition{
					"zebra_param":  {Type: "string", Description: "z"},
					"alpha_param":  {Type: "string", Description: "a"},
					"middle_param": {Type: "string", Description: "m"},
				},
			},
		},
	}

	client1 := newGeminiTestClient(srv.BaseURL)
	_, err := client1.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	bytes1 := srv.LastRequestBytes()

	client2 := newGeminiTestClient(srv.BaseURL)
	_, err = client2.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	bytes2 := srv.LastRequestBytes()

	// Both requests must produce identical JSON (deterministic property ordering)
	if string(bytes1) != string(bytes2) {
		t.Errorf("tool definitions not deterministic across requests:\nrequest1: %s\nrequest2: %s", bytes1, bytes2)
	}

	// Verify all 3 properties are present in the request
	req := srv.LastRequest()
	toolsRaw := req["tools"].([]interface{})
	toolMap := toolsRaw[0].(map[string]interface{})
	decls := toolMap["functionDeclarations"].([]interface{})
	declMap := decls[0].(map[string]interface{})
	params := declMap["parameters"].(map[string]interface{})
	props := params["properties"].(map[string]interface{})

	for _, name := range []string{"alpha_param", "middle_param", "zebra_param"} {
		if _, ok := props[name]; !ok {
			t.Errorf("missing property %q in request", name)
		}
	}
}

// buildTestToolset creates a minimal toolset for use in tests.
func buildTestToolset() *tools.ToolSet {
	return nil // nil is accepted by GeminiLLMClient.Request (no tools sent)
}
