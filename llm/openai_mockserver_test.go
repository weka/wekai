package llm

import (
	"context"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/weka/wekai-core/llm/mockserver"
	"github.com/weka/wekai-core/tools"
)

func newOpenAITestClient(baseURL string) *OpenAiChat {
	config := LLMConfig{
		Model:                  "gpt-4o-test",
		APIKey:                 "test-api-key",
		BaseURL:                baseURL,
		MaxTokens:              1024,
		StreamResponseCallback: func(string) {},
		StreamThinkingCallback: func(string) {},
	}
	return NewOpenAiLLMClient(config).(*OpenAiChat)
}

func TestOpenAIMockServer_SimpleTextResponse(t *testing.T) {
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{Content: "Hello from OpenAI!"},
	})
	defer srv.Close()

	client := newOpenAITestClient(srv.BaseURL)
	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.Content != "Hello from OpenAI!" {
		t.Errorf("expected 'Hello from OpenAI!', got %q", resp.Content)
	}
	if srv.CallCount() != 1 {
		t.Errorf("expected 1 call, got %d", srv.CallCount())
	}
}

func TestOpenAIMockServer_ToolCall(t *testing.T) {
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{
			ToolCalls: []mockserver.OpenAIToolCall{
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

	client := newOpenAITestClient(srv.BaseURL)
	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "call tool"}}, toolset)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "my_tool" {
		t.Errorf("expected tool name 'my_tool', got %q", resp.ToolCalls[0].Name)
	}
	if !strings.Contains(resp.ToolCalls[0].Args, "param") {
		t.Errorf("expected args to contain 'param', got %q", resp.ToolCalls[0].Args)
	}
}

func TestOpenAIMockServer_UsageWithCachedTokens(t *testing.T) {
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{
			Content:          "response",
			PromptTokens:     200,
			CompletionTokens: 30,
			CachedTokens:     150,
			ReasoningTokens:  10,
		},
	})
	defer srv.Close()

	client := newOpenAITestClient(srv.BaseURL)
	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "test"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	// PromptTokens should be total minus cached (client subtracts cached)
	if resp.Usage.PromptTokens != 200-150 {
		t.Errorf("expected PromptTokens=%d, got %d", 200-150, resp.Usage.PromptTokens)
	}
	if resp.Usage.CachedTokens != 150 {
		t.Errorf("expected CachedTokens=150, got %d", resp.Usage.CachedTokens)
	}
	if resp.Usage.CompletionTokens != 30 {
		t.Errorf("expected CompletionTokens=30, got %d", resp.Usage.CompletionTokens)
	}
	if resp.Usage.ReasoningTokens != 10 {
		t.Errorf("expected ReasoningTokens=10, got %d", resp.Usage.ReasoningTokens)
	}
}

func TestOpenAIMockServer_SetSystemBlocks(t *testing.T) {
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{Content: "response after system prompt"},
	})
	defer srv.Close()

	client := newOpenAITestClient(srv.BaseURL)
	client.SetSystemBlocks([]SystemBlock{{ID: "root", Content: "You are a helpful assistant.", Cache: true}})

	// History should have 1 system message (with preamble prepended)
	if len(client.history) != 1 {
		t.Fatalf("expected 1 history entry after SetSystemBlocks, got %d", len(client.history))
	}
	if client.history[0].Role != "system" {
		t.Errorf("expected role 'system', got %q", client.history[0].Role)
	}

	resp, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "hello"}}, nil)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	if resp.Content != "response after system prompt" {
		t.Errorf("expected correct response, got %q", resp.Content)
	}

	// Verify system message appears in the request
	req := srv.LastRequest()
	messages, ok := req["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		t.Fatalf("expected messages in request")
	}
	firstMsg := messages[0].(map[string]interface{})
	if firstMsg["role"] != "system" {
		t.Errorf("expected first message role 'system', got %q", firstMsg["role"])
	}
}

func TestOpenAIMockServer_ToolOrdering(t *testing.T) {
	// Tools should be sent in alphabetical order (AsOpenAi uses sortedToolNames)
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{Content: "done", PromptTokens: 10, CompletionTokens: 3},
		{Content: "done2", PromptTokens: 10, CompletionTokens: 3},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"zebra_tool":  {Name: "zebra_tool", Description: "Z", Properties: map[string]jsonschema.Definition{}},
			"alpha_tool":  {Name: "alpha_tool", Description: "A", Properties: map[string]jsonschema.Definition{}},
			"middle_tool": {Name: "middle_tool", Description: "M", Properties: map[string]jsonschema.Definition{}},
		},
	}

	client1 := newOpenAITestClient(srv.BaseURL)
	_, err := client1.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	bytes1 := srv.LastRequestBytes()

	client2 := newOpenAITestClient(srv.BaseURL)
	_, err = client2.Request(context.Background(), []ContentPart{&TextContent{Text: "hi"}}, toolset)
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	bytes2 := srv.LastRequestBytes()

	if string(bytes1) != string(bytes2) {
		t.Errorf("tool definitions not deterministic:\nrequest1: %s\nrequest2: %s", bytes1, bytes2)
	}

	// Also verify alphabetical order
	req := srv.LastRequest()
	toolsRaw, ok := req["tools"].([]interface{})
	if !ok || len(toolsRaw) == 0 {
		t.Fatalf("expected tools in request")
	}
	var names []string
	for _, toolRaw := range toolsRaw {
		tm := toolRaw.(map[string]interface{})
		fn := tm["function"].(map[string]interface{})
		names = append(names, fn["name"].(string))
	}
	expected := []string{"alpha_tool", "middle_tool", "zebra_tool"}
	for i, name := range names {
		if i >= len(expected) || name != expected[i] {
			t.Errorf("tools not in alphabetical order: got %v, want %v", names, expected)
			break
		}
	}
}

func TestOpenAIMockServer_MultiTurnToolChain(t *testing.T) {
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{ToolCalls: []mockserver.OpenAIToolCall{{Name: "increment", Args: `{"value":1}`}}},
		{ToolCalls: []mockserver.OpenAIToolCall{{Name: "increment", Args: `{"value":2}`}}},
		{Content: "Final answer: 3"},
	})
	defer srv.Close()

	toolset := &tools.ToolSet{
		Tools: map[string]*tools.Tool{
			"increment": {Name: "increment", Description: "increments", Properties: map[string]jsonschema.Definition{
				"value": {Type: "integer"},
			}},
		},
	}

	client := newOpenAITestClient(srv.BaseURL)
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
	if resp3.Content != "Final answer: 3" {
		t.Errorf("expected 'Final answer: 3', got %q", resp3.Content)
	}

	if srv.CallCount() != 3 {
		t.Errorf("expected 3 calls, got %d", srv.CallCount())
	}
}

func TestOpenAIMockServer_HTTPError(t *testing.T) {
	srv := mockserver.NewOpenAIMockServer([]mockserver.OpenAIMockTurn{
		{Error: &mockserver.OpenAIMockError{StatusCode: 429, Message: "rate limited"}},
	})
	defer srv.Close()

	client := newOpenAITestClient(srv.BaseURL)
	_, err := client.Request(context.Background(), []ContentPart{&TextContent{Text: "test"}}, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", apiErr.StatusCode)
	}
}
