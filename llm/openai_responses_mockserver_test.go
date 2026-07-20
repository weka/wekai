package llm

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/weka/wekai/llm/mockserver"
	"github.com/weka/wekai/tools"
)

func newOpenAIResponsesTestClient(baseURL string) *OpenAiResponsesChat {
	config := LLMConfig{
		Model:                  "gpt-4o-test",
		APIKey:                 "test-api-key",
		BaseURL:                baseURL,
		MaxTokens:              1024,
		StreamResponseCallback: func(string) {},
		StreamThinkingCallback: func(string) {},
	}
	return NewOpenAiResponsesLLMClient(config).(*OpenAiResponsesChat)
}

func TestOpenAIResponsesMockServer_SimpleTextResponse(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{Content: "Hello from Responses API!"},
	})
	defer srv.Close()

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.Content != "Hello from Responses API!" {
		t.Errorf("expected 'Hello from Responses API!', got %q", resp.Content)
	}
	if srv.CallCount() != 1 {
		t.Errorf("expected 1 call, got %d", srv.CallCount())
	}
}

func TestOpenAIResponsesMockServer_ToolCall(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{
			ToolCalls: []mockserver.OpenAIResponsesToolCall{
				{Name: "my_tool", Args: `{"param":"value"}`},
			},
		},
		{Content: "done"},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"my_tool": {Name: "my_tool", Description: "test", Properties: map[string]jsonschema.Definition{}},
		},
	}

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "call"}}, toolset)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "my_tool" {
		t.Errorf("expected 'my_tool', got %q", resp.ToolCalls[0].Name)
	}
	if !strings.Contains(resp.ToolCalls[0].Args, "param") {
		t.Errorf("expected args to contain 'param', got %q", resp.ToolCalls[0].Args)
	}
}

func TestOpenAIResponsesMockServer_UsageWithCachedTokens(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{
			Content:         "response",
			InputTokens:     300,
			OutputTokens:    40,
			CachedTokens:    200,
			ReasoningTokens: 15,
		},
	})
	defer srv.Close()

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "test"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	// PromptTokens = InputTokens - CachedTokens
	if resp.Usage.PromptTokens != 300-200 {
		t.Errorf("expected PromptTokens=%d, got %d", 300-200, resp.Usage.PromptTokens)
	}
	if resp.Usage.CachedTokens != 200 {
		t.Errorf("expected CachedTokens=200, got %d", resp.Usage.CachedTokens)
	}
	if resp.Usage.CompletionTokens != 40 {
		t.Errorf("expected CompletionTokens=40, got %d", resp.Usage.CompletionTokens)
	}
	if resp.Usage.ReasoningTokens != 15 {
		t.Errorf("expected ReasoningTokens=15, got %d", resp.Usage.ReasoningTokens)
	}
}

func TestOpenAIResponsesMockServer_SetSystemBlocks(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{Content: "response with system"},
	})
	defer srv.Close()

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	client.SetSystemBlocks([]SystemBlock{{ID: "root", Content: "You are a helpful assistant.", Cache: true}})

	if len(client.systemMessages) != 1 {
		t.Errorf("expected 1 systemMessage, got %d", len(client.systemMessages))
	}

	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "hello"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.Content != "response with system" {
		t.Errorf("expected correct response, got %q", resp.Content)
	}

	// Verify system message appears in the request input
	req := srv.LastRequest()
	inputRaw, ok := req["input"].([]interface{})
	if !ok || len(inputRaw) == 0 {
		t.Fatalf("expected input in request, got: %v", req["input"])
	}
	firstMsg := inputRaw[0].(map[string]interface{})
	if firstMsg["role"] != "system" {
		t.Errorf("expected first input message role 'system', got %q", firstMsg["role"])
	}
}

func TestOpenAIResponsesMockServer_ResponseIDChaining(t *testing.T) {
	// Verify that after first response, subsequent requests use previous_response_id
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{Content: "first response", ResponseID: "resp_chain_001"},
		{Content: "second response"},
	})
	defer srv.Close()

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	ctx := context.Background()

	_, err := client.Request(ctx, []ContentPart{&TextContent{Text: "first"}}, nil)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}

	// After first response, client should have stored the response ID
	if client.responseId != "resp_chain_001" {
		t.Errorf("expected responseId='resp_chain_001', got %q", client.responseId)
	}

	_, err = client.Request(ctx, []ContentPart{&TextContent{Text: "second"}}, nil)
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}

	// Second request should have included previous_response_id
	req := srv.LastRequest()
	prevID, ok := req["previous_response_id"]
	if !ok {
		t.Fatalf("expected previous_response_id in second request, keys: %v", req)
	}
	if prevID != "resp_chain_001" {
		t.Errorf("expected previous_response_id='resp_chain_001', got %q", prevID)
	}
}

func TestOpenAIResponsesMockServer_ToolOrdering(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{Content: "done", InputTokens: 10, OutputTokens: 3},
		{Content: "done2", InputTokens: 10, OutputTokens: 3},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"zebra_tool":  {Name: "zebra_tool", Description: "Z", Properties: map[string]jsonschema.Definition{}},
			"alpha_tool":  {Name: "alpha_tool", Description: "A", Properties: map[string]jsonschema.Definition{}},
			"middle_tool": {Name: "middle_tool", Description: "M", Properties: map[string]jsonschema.Definition{}},
		},
	}

	client1 := newOpenAIResponsesTestClient(srv.BaseURL)
	_, err := client1.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	bytes1 := srv.LastRequestBytes()

	client2 := newOpenAIResponsesTestClient(srv.BaseURL)
	_, err = client2.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	bytes2 := srv.LastRequestBytes()

	if string(bytes1) != string(bytes2) {
		t.Errorf("tool definitions not deterministic:\nrequest1: %s\nrequest2: %s", bytes1, bytes2)
	}

	// Verify alphabetical ordering
	req := srv.LastRequest()
	toolsRaw, ok := req["tools"].([]interface{})
	if !ok || len(toolsRaw) == 0 {
		t.Fatalf("expected tools in request")
	}
	var names []string
	for _, toolRaw := range toolsRaw {
		tm := toolRaw.(map[string]interface{})
		if name, ok := tm["name"].(string); ok {
			names = append(names, name)
		}
	}
	expected := []string{"alpha_tool", "middle_tool", "zebra_tool"}
	for i, name := range names {
		if i >= len(expected) || name != expected[i] {
			t.Errorf("tools not in alphabetical order: got %v, want %v", names, expected)
			break
		}
	}
}

func TestOpenAIResponsesMockServer_MultiTurnToolChain(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{ToolCalls: []mockserver.OpenAIResponsesToolCall{{Name: "increment", Args: `{"value":1}`}}, ResponseID: "resp_001"},
		{ToolCalls: []mockserver.OpenAIResponsesToolCall{{Name: "increment", Args: `{"value":2}`}}, ResponseID: "resp_002"},
		{Content: "Final: 3", ResponseID: "resp_003"},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"increment": {Name: "increment", Description: "increments", Properties: map[string]jsonschema.Definition{
				"value": {Type: "integer"},
			}},
		},
	}

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	ctx := context.Background()

	resp1, err := client.Request(ctx, []ContentPart{&TextContent{Text: "start"}}, toolset)
	if err != nil {
		t.Fatalf("turn 1 failed: %v", err)
	}
	if len(resp1.ToolCalls) != 1 {
		t.Fatalf("turn 1: expected 1 tool call, got %d", len(resp1.ToolCalls))
	}

	resp2, err := client.Respond(ctx, map[string]string{resp1.ToolCalls[0].CallId: "2"}, nil, toolset)
	if err != nil {
		t.Fatalf("turn 2 failed: %v", err)
	}
	if len(resp2.ToolCalls) != 1 {
		t.Fatalf("turn 2: expected 1 tool call, got %d", len(resp2.ToolCalls))
	}

	resp3, err := client.Respond(ctx, map[string]string{resp2.ToolCalls[0].CallId: "3"}, nil, toolset)
	if err != nil {
		t.Fatalf("turn 3 failed: %v", err)
	}
	if resp3.Content != "Final: 3" {
		t.Errorf("expected 'Final: 3', got %q", resp3.Content)
	}

	if srv.CallCount() != 3 {
		t.Errorf("expected 3 calls, got %d", srv.CallCount())
	}
}

func TestOpenAIResponsesMockServer_ClearHistory(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{Content: "first", ResponseID: "resp_001"},
		{Content: "after clear"},
	})
	defer srv.Close()

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	ctx := context.Background()

	_, err := client.Request(ctx, []ContentPart{&TextContent{Text: "first"}}, nil)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	if client.responseId != "resp_001" {
		t.Fatalf("expected responseId after first request, got %q", client.responseId)
	}

	client.ClearHistory()
	if client.responseId != "" {
		t.Errorf("expected empty responseId after ClearHistory, got %q", client.responseId)
	}

	_, err = client.Request(ctx, []ContentPart{&TextContent{Text: "second"}}, nil)
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}

	// After ClearHistory, no previous_response_id should be sent
	req := srv.LastRequest()
	if _, ok := req["previous_response_id"]; ok {
		t.Errorf("expected no previous_response_id after ClearHistory, but it was present")
	}
}

func TestOpenAIResponsesMockServer_TextAndToolCallSameTurn(t *testing.T) {
	// Codex models commonly return explanatory text AND a tool call in the same response.
	// This verifies both Content and ToolCalls are populated correctly.
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{
			Content: "I will call the tool now.",
			ToolCalls: []mockserver.OpenAIResponsesToolCall{
				{Name: "do_thing", Args: `{"x":42}`},
			},
		},
		{Content: "Done."},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"do_thing": {Name: "do_thing", Description: "does a thing", Properties: map[string]jsonschema.Definition{
				"x": {Type: "integer"},
			}},
		},
	}

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	ctx := context.Background()

	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "go"}}, toolset)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.Content != "I will call the tool now." {
		t.Errorf("expected text content, got %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %v", len(resp.ToolCalls), resp.ToolCalls)
	}
	if resp.ToolCalls[0].Name != "do_thing" {
		t.Errorf("expected tool name 'do_thing', got %q", resp.ToolCalls[0].Name)
	}
	if resp.ToolCalls[0].Args != `{"x":42}` {
		t.Errorf("expected args '{\"x\":42}', got %q", resp.ToolCalls[0].Args)
	}
}

func TestOpenAIResponsesMockServer_MultipleToolCallsSameTurn(t *testing.T) {
	// Verifies that when multiple tool calls are returned in the same turn, each
	// call receives its own arguments (not corrupted by lastIndex logic).
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{
			ToolCalls: []mockserver.OpenAIResponsesToolCall{
				{Name: "tool_a", Args: `{"n":1}`, ID: "call_aaa"},
				{Name: "tool_b", Args: `{"n":2}`, ID: "call_bbb"},
			},
		},
		{Content: "both done"},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"tool_a": {Name: "tool_a", Description: "a", Properties: map[string]jsonschema.Definition{
				"n": {Type: "integer"},
			}},
			"tool_b": {Name: "tool_b", Description: "b", Properties: map[string]jsonschema.Definition{
				"n": {Type: "integer"},
			}},
		},
	}

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	ctx := context.Background()

	resp, err := client.Request(ctx, []ContentPart{&TextContent{Text: "run both"}}, toolset)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(resp.ToolCalls))
	}

	// Build a map for easier assertion (order not guaranteed)
	byName := map[string]ToolCall{}
	for _, tc := range resp.ToolCalls {
		byName[tc.Name] = tc
	}

	if tc, ok := byName["tool_a"]; !ok {
		t.Error("missing tool_a")
	} else {
		if tc.Args != `{"n":1}` {
			t.Errorf("tool_a: expected args '{\"n\":1}', got %q", tc.Args)
		}
		if tc.CallId != "call_aaa" {
			t.Errorf("tool_a: expected CallId 'call_aaa', got %q", tc.CallId)
		}
	}
	if tc, ok := byName["tool_b"]; !ok {
		t.Error("missing tool_b")
	} else {
		if tc.Args != `{"n":2}` {
			t.Errorf("tool_b: expected args '{\"n\":2}', got %q", tc.Args)
		}
		if tc.CallId != "call_bbb" {
			t.Errorf("tool_b: expected CallId 'call_bbb', got %q", tc.CallId)
		}
	}
}

func TestOpenAIResponsesMockServer_TextThenToolCallMultiTurn(t *testing.T) {
	// 3-turn conversation: turn1=text+toolcall, turn2=tool result→more tool calls, turn3=final text.
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{
			Content: "Let me search for that.",
			ToolCalls: []mockserver.OpenAIResponsesToolCall{
				{Name: "search", Args: `{"query":"foo"}`, ID: "call_s1"},
			},
			ResponseID: "resp_t1",
		},
		{
			Content: "Found it, now fetching details.",
			ToolCalls: []mockserver.OpenAIResponsesToolCall{
				{Name: "fetch", Args: `{"url":"http://example.com"}`, ID: "call_f1"},
			},
			ResponseID: "resp_t2",
		},
		{
			Content:    "The answer is 42.",
			ResponseID: "resp_t3",
		},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"search": {Name: "search", Description: "search", Properties: map[string]jsonschema.Definition{
				"query": {Type: "string"},
			}},
			"fetch": {Name: "fetch", Description: "fetch url", Properties: map[string]jsonschema.Definition{
				"url": {Type: "string"},
			}},
		},
	}

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	ctx := context.Background()

	// Turn 1: text + tool call
	resp1, err := client.Request(ctx, []ContentPart{&TextContent{Text: "find foo"}}, toolset)
	if err != nil {
		t.Fatalf("turn 1 failed: %v", err)
	}
	if resp1.Content != "Let me search for that." {
		t.Errorf("turn 1 content: got %q", resp1.Content)
	}
	if len(resp1.ToolCalls) != 1 || resp1.ToolCalls[0].Name != "search" {
		t.Fatalf("turn 1: expected search tool call, got %v", resp1.ToolCalls)
	}

	// Turn 2: provide tool result, get more tool calls
	resp2, err := client.Respond(ctx, map[string]string{resp1.ToolCalls[0].CallId: "result: foo data"}, nil, toolset)
	if err != nil {
		t.Fatalf("turn 2 failed: %v", err)
	}
	if resp2.Content != "Found it, now fetching details." {
		t.Errorf("turn 2 content: got %q", resp2.Content)
	}
	if len(resp2.ToolCalls) != 1 || resp2.ToolCalls[0].Name != "fetch" {
		t.Fatalf("turn 2: expected fetch tool call, got %v", resp2.ToolCalls)
	}

	// Turn 3: provide tool result, get final text
	resp3, err := client.Respond(ctx, map[string]string{resp2.ToolCalls[0].CallId: "page content"}, nil, toolset)
	if err != nil {
		t.Fatalf("turn 3 failed: %v", err)
	}
	if resp3.Content != "The answer is 42." {
		t.Errorf("turn 3 content: expected 'The answer is 42.', got %q", resp3.Content)
	}
	if len(resp3.ToolCalls) != 0 {
		t.Errorf("turn 3: expected no tool calls, got %d", len(resp3.ToolCalls))
	}
	if srv.CallCount() != 3 {
		t.Errorf("expected 3 server calls, got %d", srv.CallCount())
	}
}

func TestOpenAIResponsesMockServer_HTTPError(t *testing.T) {
	srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
		{Error: &mockserver.OpenAIResponsesMockError{StatusCode: 500, Message: "server error"}},
	})
	defer srv.Close()

	client := newOpenAIResponsesTestClient(srv.BaseURL)
	_, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "test"}}, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 500 {
		t.Errorf("expected status 500, got %d", apiErr.StatusCode)
	}
}

// TestOpenAIResponsesMockServer_ForceToolCall verifies that ForceToolCall=true causes
// tool_choice: "required" to be sent in the request payload when tools are present.
func TestOpenAIResponsesMockServer_ForceToolCall(t *testing.T) {
	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"my_tool": {Name: "my_tool", Description: "test tool", Properties: map[string]jsonschema.Definition{}},
		},
	}

	t.Run("ForceToolCall=true sets tool_choice required", func(t *testing.T) {
		srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
			{
				ToolCalls: []mockserver.OpenAIResponsesToolCall{
					{Name: "my_tool", Args: `{}`},
				},
			},
			{Content: "done"},
		})
		defer srv.Close()

		configWithForce := LLMConfig{
			Model:                  "gpt-5.3-codex",
			APIKey:                 "test-api-key",
			BaseURL:                srv.BaseURL,
			MaxTokens:              1024,
			ForceToolCall:          true,
			StreamResponseCallback: func(string) {},
			StreamThinkingCallback: func(string) {},
		}
		client := NewOpenAiResponsesLLMClient(configWithForce).(*OpenAiResponsesChat)

		_, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "call tool"}}, toolset)
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}

		reqBytes := srv.LastRequestBytes()
		if len(reqBytes) == 0 {
			t.Fatal("no request captured by mock server")
		}

		var payload map[string]interface{}
		if err := json.Unmarshal(reqBytes, &payload); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		toolChoice, ok := payload["tool_choice"]
		if !ok {
			t.Fatal("expected tool_choice in request payload, but not found")
		}
		if toolChoice != "required" {
			t.Errorf("expected tool_choice=\"required\", got %q", toolChoice)
		}
	})

	t.Run("ForceToolCall=false omits tool_choice", func(t *testing.T) {
		srv := mockserver.NewOpenAIResponsesMockServer([]mockserver.OpenAIResponsesMockTurn{
			{
				ToolCalls: []mockserver.OpenAIResponsesToolCall{
					{Name: "my_tool", Args: `{}`},
				},
			},
			{Content: "done"},
		})
		defer srv.Close()

		configNoForce := LLMConfig{
			Model:                  "gpt-4o-test",
			APIKey:                 "test-api-key",
			BaseURL:                srv.BaseURL,
			MaxTokens:              1024,
			ForceToolCall:          false,
			StreamResponseCallback: func(string) {},
			StreamThinkingCallback: func(string) {},
		}
		client := NewOpenAiResponsesLLMClient(configNoForce).(*OpenAiResponsesChat)

		_, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "call tool"}}, toolset)
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}

		reqBytes := srv.LastRequestBytes()
		if len(reqBytes) == 0 {
			t.Fatal("no request captured by mock server")
		}

		var payload map[string]interface{}
		if err := json.Unmarshal(reqBytes, &payload); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		if _, ok := payload["tool_choice"]; ok {
			t.Errorf("expected tool_choice to be absent from request, but it was present: %v", payload["tool_choice"])
		}
	})
}
