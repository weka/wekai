// Package tools provides the base Tool/ToolSet types shared by the core LLM
// client layer (for definition serialization) and the benchmark tool-chain
// eval (for callback execution). Agentic tool implementations (filesystem,
// git, MCP, flow orchestration, etc.) are NOT part of this package — they
// live in the embedding application.
package tools

import (
	"context"
	"encoding/json"
	"sort"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

// TokenUsage represents usage and cost for a specific token type.
type TokenUsage struct {
	Count int     `json:"count"` // Number of tokens
	Cost  float64 `json:"cost"`  // Cost in USD
}

// ExecutionUsageData represents accumulated LLM usage data for an execution.
//
// This type lives here (rather than in the llm package) to avoid an import
// cycle: llm.Chat references tools.ToolSet, so llm already imports tools —
// tools cannot import llm back. ToolResult.UsageData needs this exact type,
// so it's defined where it's consumed. Embedding applications that persist
// this data (e.g. wekai's FSDB-backed datastore) should type-alias their
// storage struct to this type to keep identical field/JSON shape.
type ExecutionUsageData struct {
	RequestCount int `json:"request_count"`

	// Token usage with cost breakdown - these are the source of truth
	InputTokens     TokenUsage           `json:"input_tokens"`             // Prompt tokens with cost
	OutputTokens    TokenUsage           `json:"output_tokens"`            // Completion tokens with cost
	CachedTokens    TokenUsage           `json:"cached_tokens"`            // Cached tokens with cost
	ReasoningTokens TokenUsage           `json:"reasoning_tokens"`         // Reasoning tokens with cost
	TotalCost       float64              `json:"total_cost"`               // Total cost in USD
	ModelName       string               `json:"model_name"`               // Model used for cost calculation
	SubExecutions   []ExecutionUsageData `json:"sub_executions,omitempty"` // Nested execution usage data
	ChatId          string               `json:"chat_id"`                  // Chat ID for the execution
	LLMTimeMs       int64                `json:"llm_time_ms,omitempty"`    // Cumulative time in LLM API calls (ms)
	ExecTimeMs      int64                `json:"exec_time_ms,omitempty"`   // LLM time + tool execution time (ms)
}

// ToolResult represents the result of a tool execution.
type ToolResult struct {
	Content   string             `json:"content"`
	IsError   bool               `json:"is_error,omitempty"`
	UsageData ExecutionUsageData `json:"usage_data"`
	LogBlobID string             `json:"-"` // blob ID for persisted output log (transparent to LLM)
}

type Tool struct {
	Name                string                           `json:"name"`
	Description         string                           `json:"description"`
	Properties          map[string]jsonschema.Definition `json:"parameters"`
	run                 ToolCallback                     `json:"-"`
	paramsSchema        interface{}                      `json:"-"`
	Required            []string                         `json:"required"`
	ExcludeFromToolTime bool                             `json:"-"` // if true, duration is not counted toward toolTimeMs
	Callbacks           struct {
		ThinkingCallback func(string) `json:"thinking_callback"`
		ResponseCallback func(string) `json:"response_callback"`
	}
}

func (t *Tool) AsOpenAi() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters: jsonschema.Definition{
				Type:                 "object",
				Properties:           t.Properties,
				Required:             t.Required,
				AdditionalProperties: false,
			},
		},
	}
}

// AsOpenAiResponses converts the tool to OpenAI Responses API format
func (t *Tool) AsOpenAiResponses() map[string]interface{} {
	return map[string]interface{}{
		"type":        "function",
		"name":        t.Name,
		"description": t.Description,
		"parameters": map[string]interface{}{
			"type":                 "object",
			"properties":           t.Properties,
			"required":             t.Required,
			"additionalProperties": false,
		},
	}
}

func (t *Tool) Run(ctx context.Context, params string) (*ToolResult, error) {
	return t.run(ctx, params)
}

// GetRun returns the underlying run callback. Used for wrapping tools.
func (t *Tool) GetRun() ToolCallback {
	return t.run
}

// SetRun sets the run callback. Used when wrapping tools.
func (t *Tool) SetRun(cb ToolCallback) {
	t.run = cb
}

// AsAnthropic converts the tool to Anthropic API format.
func (t *Tool) AsAnthropic() anthropic.ToolParam {
	schema := jsonschema.Definition{
		Type:                 "object",
		Properties:           t.Properties,
		Required:             t.Required,
		AdditionalProperties: false,
	}
	b, _ := json.Marshal(schema)
	var schemaMap interface{}
	json.Unmarshal(b, &schemaMap)

	return anthropic.ToolParam{
		Name:        anthropic.F(t.Name),
		Description: anthropic.F(t.Description),
		InputSchema: anthropic.F(schemaMap),
	}
}

type ToolCallback func(ctx context.Context, params string) (*ToolResult, error)

func NewTool(name, description string, params map[string]jsonschema.Definition, required []string, callback ToolCallback) *Tool {
	return &Tool{
		Name:        name,
		Description: description,
		Properties:  params,
		run:         callback,
		Required:    required,
	}
}

type ToolSet struct {
	Tools map[string]*Tool
}

func (ts *ToolSet) AddTool(tool *Tool) {
	ts.Tools[tool.Name] = tool
}

func NewToolSet() *ToolSet {
	return &ToolSet{
		Tools: map[string]*Tool{},
	}
}

func (ts *ToolSet) AsOpenAi() []openai.Tool {
	var tools []openai.Tool
	if ts == nil {
		return tools
	}
	for _, name := range ts.sortedToolNames() {
		tools = append(tools, ts.Tools[name].AsOpenAi())
	}
	return tools
}

// AsOpenAiResponses converts the toolset to OpenAI Responses API format
func (ts *ToolSet) AsOpenAiResponses() []map[string]interface{} {
	var tools []map[string]interface{}
	if ts == nil {
		return tools
	}
	for _, name := range ts.sortedToolNames() {
		tools = append(tools, ts.Tools[name].AsOpenAiResponses())
	}
	return tools
}

// sortedToolNames returns tool names in deterministic order.
// This is critical for LLM prompt caching — non-deterministic tool ordering
// invalidates the cache prefix on every request, causing 0% cache hits.
func (ts *ToolSet) sortedToolNames() []string {
	names := make([]string, 0, len(ts.Tools))
	for name := range ts.Tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// SortedTools returns tools in deterministic alphabetical order.
// Use this instead of ranging over Tools map directly to ensure stable ordering.
func (ts *ToolSet) SortedTools() []*Tool {
	tools := make([]*Tool, 0, len(ts.Tools))
	for _, name := range ts.sortedToolNames() {
		tools = append(tools, ts.Tools[name])
	}
	return tools
}

func (ts *ToolSet) GetToolByName(name string) *Tool {
	return ts.Tools[name]
}

func (ts *ToolSet) AsAnthropic() []anthropic.ToolParam {
	var tools []anthropic.ToolParam
	if ts == nil {
		return tools
	}
	for _, name := range ts.sortedToolNames() {
		tools = append(tools, ts.Tools[name].AsAnthropic())
	}
	return tools
}

func (ts *ToolSet) AsJsonSchema() map[string]jsonschema.Definition {
	schema := make(map[string]jsonschema.Definition)
	for name, tool := range ts.Tools {
		schema[name] = jsonschema.Definition{
			Type:        "object",
			Description: tool.Description,
			Properties:  tool.Properties,
			Required:    tool.Required,
		}
	}
	return schema
}
