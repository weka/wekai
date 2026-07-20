package llm

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/weka/wekai/tools"
)

// ---- helpers ----

// buildState constructs a ProviderRecoveryState from raw req/resp JSON.
func buildState(provider, clientType, modelID string, reqJSON, respJSON []byte) *ProviderRecoveryState {
	return NewProviderRecoveryState(provider, clientType, modelID,
		json.RawMessage(reqJSON), json.RawMessage(respJSON))
}

// ---- nil / empty ----

func TestRecoverProviderHistory_NilState(t *testing.T) {
	err := RecoverProviderHistory(context.Background(), &mockChatForRecovery{}, nil)
	if err != nil {
		t.Errorf("expected no error for nil state: %v", err)
	}
}

func TestRecoverProviderHistory_EmptyState(t *testing.T) {
	state := NewProviderRecoveryState("openai", "openai", "gpt-4", nil, nil)
	err := RecoverProviderHistory(context.Background(), &mockChatForRecovery{}, state)
	if err != nil {
		t.Errorf("expected no error for empty state: %v", err)
	}
}

// ---- Anthropic ----

func TestRecoverAnthropicHistory(t *testing.T) {
	ctx := context.Background()
	client := &AnthropicLLMClient{history: make([]Message, 0)}

	// Simulate the last exchange: request has full 3-message history.
	req, _ := json.Marshal(map[string]interface{}{
		"model": "claude-3-5-sonnet-20241022",
		"system": []interface{}{
			map[string]interface{}{"type": "text", "text": "You are helpful."},
		},
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": []interface{}{
				map[string]interface{}{"type": "text", "text": "What is 2+2?"},
			}},
			map[string]interface{}{"role": "assistant", "content": []interface{}{
				map[string]interface{}{"type": "text", "text": "4."},
			}},
			map[string]interface{}{"role": "user", "content": []interface{}{
				map[string]interface{}{"type": "text", "text": "What is 3+3?"},
			}},
		},
	})
	resp, _ := json.Marshal(map[string]interface{}{
		"content": []interface{}{
			map[string]interface{}{"type": "text", "text": "6."},
		},
	})
	state := buildState("anthropic", "anthropic", "claude-3-5-sonnet-20241022", req, resp)

	if err := RecoverProviderHistory(ctx, client, state); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 3 msgs from request + 1 assistant from response = 4
	if len(client.history) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(client.history))
	}
	for i, role := range []string{"user", "assistant", "user", "assistant"} {
		if client.history[i].Role != role {
			t.Errorf("history[%d].Role = %q, want %q", i, client.history[i].Role, role)
		}
	}
	if len(client.systemBlocks) == 0 || client.systemBlocks[0].Text != "You are helpful." {
		got := ""
		if len(client.systemBlocks) > 0 {
			got = client.systemBlocks[0].Text
		}
		t.Errorf("system prompt = %q, want %q", got, "You are helpful.")
	}
}

func TestRecoverAnthropicHistory_ToolUse(t *testing.T) {
	client := &AnthropicLLMClient{history: make([]Message, 0)}

	req, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Use the calculator tool"},
		},
	})
	resp, _ := json.Marshal(map[string]interface{}{
		"content": []interface{}{
			map[string]interface{}{"type": "text", "text": "I'll use the calculator."},
			map[string]interface{}{"type": "tool_use", "id": "toolu_01ABC", "name": "calculator", "input": map[string]interface{}{"expr": "2+2"}},
		},
	})
	state := buildState("anthropic", "anthropic", "claude-3", req, resp)

	if err := RecoverProviderHistory(context.Background(), client, state); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(client.history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(client.history))
	}
	assistantContent, ok := client.history[1].Content.([]interface{})
	if !ok || len(assistantContent) != 2 {
		t.Errorf("expected 2 content blocks in assistant message")
	}
}

// ---- OpenAI ----

func TestRecoverOpenAIHistory(t *testing.T) {
	client := &OpenAiChat{history: make([]OpenaiMessage, 0)}

	req, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "capital of France?"},
			map[string]interface{}{"role": "assistant", "content": "Paris."},
			map[string]interface{}{"role": "user", "content": "population?"},
		},
	})
	resp, _ := json.Marshal(map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{"role": "assistant", "content": "~2.2M."},
			},
		},
	})
	state := buildState("openai", "openai", "gpt-4o-mini", req, resp)

	if err := RecoverProviderHistory(context.Background(), client, state); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(client.history) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(client.history))
	}
	for i, role := range []string{"user", "assistant", "user", "assistant"} {
		if client.history[i].Role != role {
			t.Errorf("history[%d].Role = %q, want %q", i, client.history[i].Role, role)
		}
	}
}

func TestRecoverOpenAIHistory_ToolCalls(t *testing.T) {
	client := &OpenAiChat{history: make([]OpenaiMessage, 0)}

	req, _ := json.Marshal(map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Read file.txt"},
		},
	})
	resp, _ := json.Marshal(map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "",
					"tool_calls": []interface{}{
						map[string]interface{}{
							"id": "call_abc", "type": "function",
							"function": map[string]interface{}{"name": "read_file", "arguments": `{"path":"file.txt"}`},
						},
					},
				},
			},
		},
	})
	state := buildState("openai", "openai", "gpt-4o", req, resp)

	if err := RecoverProviderHistory(context.Background(), client, state); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(client.history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(client.history))
	}
	if len(client.history[1].FunctionCall) != 1 {
		t.Fatalf("expected 1 tool call on assistant message")
	}
	if client.history[1].FunctionCall[0].Function.Name != "read_file" {
		t.Errorf("tool call name = %q, want read_file", client.history[1].FunctionCall[0].Function.Name)
	}
}

// ---- Gemini ----

func TestRecoverGeminiHistory(t *testing.T) {
	client := &GeminiLLMClient{history: make([]GeminiMessage, 0)}

	req, _ := json.Marshal(map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "Explain qubits."}}},
			map[string]interface{}{"role": "model", "parts": []interface{}{map[string]interface{}{"text": "Qubits are..."}}},
			map[string]interface{}{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "More detail?"}}},
		},
	})
	resp, _ := json.Marshal(map[string]interface{}{
		"candidates": []interface{}{
			map[string]interface{}{
				"content": map[string]interface{}{
					"role":  "model",
					"parts": []interface{}{map[string]interface{}{"text": "Detailed explanation."}},
				},
			},
		},
	})
	state := buildState("gemini", "gemini_native", "gemini-2.5-flash", req, resp)

	if err := RecoverProviderHistory(context.Background(), client, state); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(client.history) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(client.history))
	}
	for i, role := range []string{"user", "model", "user", "model"} {
		if client.history[i].Role != role {
			t.Errorf("history[%d].Role = %q, want %q", i, client.history[i].Role, role)
		}
	}
}

// ---- OpenAI Responses (stateless) ----

func TestRecoverProviderHistory_OpenAIResponses_Stateless(t *testing.T) {
	req := json.RawMessage(`{"model":"gpt-4o-mini","prompt":"test"}`)
	resp := json.RawMessage(`{"id":"resp_123"}`)
	state := buildState("openai", "openai_responses", "gpt-4o-mini", req, resp)

	err := RecoverProviderHistory(context.Background(), &mockChatForRecovery{}, state)
	if err != nil {
		t.Errorf("stateless API should not error: %v", err)
	}
}

// ---- Unknown client type ----

func TestRecoverProviderHistory_UnsupportedClientType(t *testing.T) {
	req := json.RawMessage(`{}`)
	resp := json.RawMessage(`{}`)
	state := buildState("unknown", "unknown_client", "model", req, resp)

	err := RecoverProviderHistory(context.Background(), &mockChatForRecovery{}, state)
	if err == nil {
		t.Error("expected error for unsupported client type")
	}
}

// ---- mock ----

type mockChatForRecovery struct{}

func (m *mockChatForRecovery) Request(ctx context.Context, content []ContentPart, toolset *tools.ToolSet) (*Response, error) {
	return nil, nil
}
func (m *mockChatForRecovery) Respond(ctx context.Context, toolResponses map[string]string, additionalMessages []Message, toolset *tools.ToolSet) (*Response, error) {
	return nil, nil
}
func (m *mockChatForRecovery) SetSystemBlocks(blocks []SystemBlock)      {}
func (m *mockChatForRecovery) ClearHistory()                             {}
func (m *mockChatForRecovery) SetPendingToolResults(r map[string]string) {}
func (m *mockChatForRecovery) GetDanglingToolUseIDs() []string           { return nil }
