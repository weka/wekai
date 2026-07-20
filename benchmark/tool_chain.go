package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// ToolChainResult represents the result of a tool chain evaluation
type ToolChainResult struct {
	Model           string
	TargetValue     int
	FinalValue      int
	ToolInvocations int
	ExpectedCalls   int
	Success         bool
	FinalResponse   string
	ErrorMessage    string
	RequestCount    int
	InputTokens     int
	OutputTokens    int
	ReasoningTokens int
	CachedTokens    int
	TotalCost       float64
}

// RunToolChainEvaluation runs the increment tool chain against the given model.
// It starts at 1, increments targetValue-1 times, and verifies the final result.
func RunToolChainEvaluation(ctx context.Context, modelName string, targetValue int) (*ToolChainResult, error) {
	// Create toolset with increment tool
	toolset, incrementTool := createIncrementToolset()

	chatGetter := config.GetChatGetter(modelName, &llm.ChatParams{
		ResponseCallback: func(s string) { fmt.Print(s) },
		ThinkingCallback: func(s string) { fmt.Print(s) },
		APIKeys:          config.GetAPIKeys(),
	})

	// System prompt that forces tool usage
	systemPrompt := fmt.Sprintf("You are a number incrementer. You start with the number 1. "+
		"Your ONLY job is to use the increment_by_one tool to reach the number %d. "+
		"You MUST use the tool for each increment - you are FORBIDDEN from generating numbers by any other method. "+
		"You cannot calculate, count, or generate numbers yourself. "+
		"The tool requires a 'current_value' parameter - always use the current number you have. "+
		"Call the increment_by_one tool repeatedly with the current value until it returns '%d'. "+
		"When the tool returns '%d', immediately respond with ONLY the text '%d' and STOP. "+
		"Do NOT call the tool again after it returns '%d'. "+
		"Do NOT provide any final answer or summary - only use the tool and respond with '%d' when done. "+
		"Start by calling increment_by_one with current_value: 1.", targetValue, targetValue, targetValue, targetValue, targetValue, targetValue)

	// Start the conversation
	invokeResult, err := InvokeChat(ctx, chatGetter, toolset, systemPrompt,
		fmt.Sprintf("Start incrementing from 1 to reach %d. Use the increment_by_one tool.", targetValue))

	result := &ToolChainResult{
		Model:           modelName,
		TargetValue:     targetValue,
		FinalValue:      incrementTool.GetCurrentValue(),
		ToolInvocations: incrementTool.GetInvocations(),
		ExpectedCalls:   targetValue - 1,
	}

	if err != nil {
		result.ErrorMessage = err.Error()
		return result, err
	}
	result.FinalResponse = invokeResult.Content

	// Get usage data
	usage := invokeResult.Usage
	result.RequestCount = usage.RequestCount
	result.InputTokens = usage.InputTokens.Count
	result.OutputTokens = usage.OutputTokens.Count
	result.ReasoningTokens = usage.ReasoningTokens.Count
	result.CachedTokens = usage.CachedTokens.Count
	result.TotalCost = usage.TotalCost

	// Check success
	targetStr := fmt.Sprintf("%d", targetValue)
	result.Success = result.ToolInvocations > 0 &&
		result.FinalValue == targetValue &&
		strings.Contains(result.FinalResponse, targetStr)

	// Check for critical infrastructure issues
	if result.ToolInvocations == 0 {
		result.ErrorMessage = "Tool was never called"
		result.Success = false
	}
	if usage.RequestCount == 0 {
		result.ErrorMessage = "No requests recorded"
		result.Success = false
	}
	if result.FinalResponse == "" {
		result.ErrorMessage = "Response was empty"
		result.Success = false
	}

	return result, nil
}

// createIncrementToolset creates a toolset with the increment_by_one tool
func createIncrementToolset() (*tools.ToolSet, *IncrementTool) {
	toolset := tools.NewToolSet()
	incrementTool := NewIncrementTool()
	toolset.AddTool(incrementTool.CreateTool())
	return toolset, incrementTool
}

// IncrementTool is a simple tool that increments a number by one
type IncrementTool struct {
	currentValue int
	invocations  int
}

// NewIncrementTool creates a new IncrementTool
func NewIncrementTool() *IncrementTool {
	return &IncrementTool{currentValue: 1}
}

// CreateTool returns the tool definition for the increment tool
func (t *IncrementTool) CreateTool() *tools.Tool {
	callback := func(ctx context.Context, rawParams string) (*tools.ToolResult, error) {
		// Parse parameters to get the current value
		type IncrementParams struct {
			CurrentValue int `json:"current_value"`
		}

		var params IncrementParams
		if err := json.Unmarshal([]byte(rawParams), &params); err != nil {
			return nil, fmt.Errorf("failed to unmarshal parameters: %w", err)
		}

		// Validate that the provided current_value matches our actual current value
		if params.CurrentValue != t.currentValue {
			return nil, fmt.Errorf("provided current_value (%d) does not match actual current value (%d)", params.CurrentValue, t.currentValue)
		}

		// Increment the value
		t.currentValue++
		t.invocations++

		return &tools.ToolResult{
			Content: fmt.Sprintf("%d", t.currentValue),
		}, nil
	}

	// Define parameter schema
	properties := map[string]jsonschema.Definition{
		"current_value": {
			Type:        jsonschema.Integer,
			Description: "The current value to increment",
		},
	}

	return tools.NewTool(
		"increment_by_one",
		"Increments the current value by exactly 1. Takes the current value as input and returns the new value.",
		properties,
		[]string{"current_value"},
		callback,
	)
}

// GetCurrentValue returns the current value
func (t *IncrementTool) GetCurrentValue() int {
	return t.currentValue
}

// GetInvocations returns the number of times the tool was called
func (t *IncrementTool) GetInvocations() int {
	return t.invocations
}
