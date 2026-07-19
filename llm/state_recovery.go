package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/weka/go-weka-observability/instrumentation"
)

// RecoverProviderHistory recovers provider-native conversation state from saved
// provider recovery state. Returns nil when there is nothing to recover.
func RecoverProviderHistory(ctx context.Context, chat Chat, state *ProviderRecoveryState) error {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "RecoverProviderHistory")
	defer end()

	if state == nil || (len(state.LastRequest) == 0 && len(state.LastResponse) == 0) {
		logger.Info("No provider state to recover")
		return nil
	}

	logger.Info("Recovering provider history",
		"provider", state.Provider,
		"client_type", state.ClientType,
		"model_id", state.ModelID)

	switch ClientType(state.ClientType) {
	case ClientTypeAnthropic:
		return recoverAnthropicHistory(ctx, chat, state)
	case ClientTypeOpenAI:
		return recoverOpenAIHistory(ctx, chat, state)
	case ClientTypeOpenAIResponses:
		// Stateless — context maintained via response_id chaining.
		logger.Info("OpenAI Responses API is stateless, no history recovery needed")
		return nil
	case ClientTypeGeminiNative:
		return recoverGeminiHistory(ctx, chat, state)
	default:
		return fmt.Errorf("unsupported client type for recovery: %s", state.ClientType)
	}
}

// recoverAnthropicHistory recovers Anthropic conversation state.
// Anthropic is stateless — each request contains the full message history.
// We use LastRequest (full history) + LastResponse (final assistant reply).
func recoverAnthropicHistory(ctx context.Context, chat Chat, state *ProviderRecoveryState) error {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "recoverAnthropicHistory")
	defer end()

	anthropicClient, ok := chat.(*AnthropicLLMClient)
	if !ok {
		return fmt.Errorf("chat is not an AnthropicLLMClient")
	}

	var req map[string]interface{}
	if err := json.Unmarshal(state.LastRequest, &req); err != nil {
		return fmt.Errorf("failed to unmarshal Anthropic request: %w", err)
	}

	// Recover system blocks if present.
	if system, ok := req["system"].([]interface{}); ok && len(system) > 0 {
		anthropicClient.systemBlocks = make([]AnthropicSystemMessage, 0, len(system))
		for _, s := range system {
			if sysMsg, ok := s.(map[string]interface{}); ok {
				if text, ok := sysMsg["text"].(string); ok && text != "" {
					block := AnthropicSystemMessage{
						Text: text,
						Type: "text",
					}
					if cc, ok := sysMsg["cache_control"].(map[string]interface{}); ok {
						if ccType, ok := cc["type"].(string); ok {
							block.CacheControl = &AnthropicCacheControl{Type: ccType}
						}
					}
					anthropicClient.systemBlocks = append(anthropicClient.systemBlocks, block)
				}
			}
		}
	}

	anthropicClient.history = make([]Message, 0)

	messages, ok := req["messages"].([]interface{})
	if !ok {
		logger.Info("No messages in last request")
		return nil
	}
	for _, msgData := range messages {
		msgMap, ok := msgData.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msgMap["role"].(string)
		content := msgMap["content"]
		anthropicClient.history = append(anthropicClient.history, Message{
			Role:    role,
			Content: content,
		})
	}

	// Append the final assistant reply from the response.
	var resp map[string]interface{}
	if err := json.Unmarshal(state.LastResponse, &resp); err != nil {
		return fmt.Errorf("failed to unmarshal Anthropic response: %w", err)
	}
	if content, ok := resp["content"].([]interface{}); ok && len(content) > 0 {
		anthropicClient.history = append(anthropicClient.history, Message{
			Role:    "assistant",
			Content: content,
		})
	}

	logger.Info("Recovered Anthropic history", "history_length", len(anthropicClient.history))
	return nil
}

// recoverOpenAIHistory recovers OpenAI conversation state.
// OpenAI is stateless — each request contains the full message history.
func recoverOpenAIHistory(ctx context.Context, chat Chat, state *ProviderRecoveryState) error {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "recoverOpenAIHistory")
	defer end()

	openaiClient, ok := chat.(*OpenAiChat)
	if !ok {
		return fmt.Errorf("chat is not an OpenAiChat")
	}

	var req map[string]interface{}
	if err := json.Unmarshal(state.LastRequest, &req); err != nil {
		return fmt.Errorf("failed to unmarshal OpenAI request: %w", err)
	}

	openaiClient.history = make([]OpenaiMessage, 0)

	messages, ok := req["messages"].([]interface{})
	if !ok {
		logger.Info("No messages in last request")
		return nil
	}
	for _, msgData := range messages {
		msgBytes, _ := json.Marshal(msgData)
		var msg OpenaiMessage
		if err := json.Unmarshal(msgBytes, &msg); err != nil {
			continue
		}
		openaiClient.history = append(openaiClient.history, msg)
	}

	// Append the final assistant reply from the response.
	var resp map[string]interface{}
	if err := json.Unmarshal(state.LastResponse, &resp); err != nil {
		return fmt.Errorf("failed to unmarshal OpenAI response: %w", err)
	}
	if choices, ok := resp["choices"].([]interface{}); ok && len(choices) > 0 {
		choice := choices[0].(map[string]interface{})
		msgBytes, _ := json.Marshal(choice["message"])
		var msg OpenaiMessage
		if err := json.Unmarshal(msgBytes, &msg); err == nil {
			openaiClient.history = append(openaiClient.history, msg)
		}
	}

	logger.Info("Recovered OpenAI history", "history_length", len(openaiClient.history))
	return nil
}

// recoverGeminiHistory recovers Gemini conversation state.
// Gemini is stateless — each request contains the full contents history.
func recoverGeminiHistory(ctx context.Context, chat Chat, state *ProviderRecoveryState) error {
	ctx, logger, end := instrumentation.GetLogSpan(ctx, "recoverGeminiHistory")
	defer end()

	geminiClient, ok := chat.(*GeminiLLMClient)
	if !ok {
		return fmt.Errorf("chat is not a GeminiLLMClient")
	}

	var req map[string]interface{}
	if err := json.Unmarshal(state.LastRequest, &req); err != nil {
		return fmt.Errorf("failed to unmarshal Gemini request: %w", err)
	}

	geminiClient.history = make([]GeminiMessage, 0)

	contents, ok := req["contents"].([]interface{})
	if !ok {
		logger.Info("No contents in last request")
		return nil
	}
	for _, contentData := range contents {
		contentBytes, _ := json.Marshal(contentData)
		var msg GeminiMessage
		if err := json.Unmarshal(contentBytes, &msg); err != nil {
			continue
		}
		geminiClient.history = append(geminiClient.history, msg)
	}

	// Append the final model reply from the response.
	var resp map[string]interface{}
	if err := json.Unmarshal(state.LastResponse, &resp); err != nil {
		return fmt.Errorf("failed to unmarshal Gemini response: %w", err)
	}
	if candidates, ok := resp["candidates"].([]interface{}); ok && len(candidates) > 0 {
		candidate := candidates[0].(map[string]interface{})
		contentBytes, _ := json.Marshal(candidate["content"])
		var msg GeminiMessage
		if err := json.Unmarshal(contentBytes, &msg); err == nil {
			geminiClient.history = append(geminiClient.history, msg)
		}
	}

	logger.Info("Recovered Gemini history", "history_length", len(geminiClient.history))
	return nil
}
