package benchmark

import (
	"context"
	"strings"
	"testing"

	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// An agentic model can emit tool_calls even in a call that configured no
// ToolSet at all (e.g. the coherency eval always passes nil — see
// InvokeChat's call site in cache_coherency.go). That must be recorded as a
// tool-not-found result the caller's normal scoring can see, never a panic:
// github.com/weka/wekai/tools.(*ToolSet).GetToolByName used to dereference a
// nil receiver's Tools field directly.
func TestExecuteToolCallsParallelNilToolSetDoesNotPanic(t *testing.T) {
	usage := &tools.ExecutionUsageData{}
	calls := llm.ToolsCalls{{Name: "search", Args: "{}", CallId: "call-1"}}

	responses := executeToolCallsParallel(context.Background(), nil, "test/model", calls, usage)

	got, ok := responses["call-1"]
	if !ok {
		t.Fatalf("no response recorded for call-1; responses=%v", responses)
	}
	if !strings.Contains(got, "search") || !strings.Contains(got, "not found") {
		t.Errorf("expected a %q not-found result naming the tool, got %q", "search", got)
	}
}

// Same contract when a real ToolSet exists but doesn't contain the name the
// model called — GetToolByName returns nil either way, and both paths must
// converge on the same "not found" result rather than one of them panicking.
func TestExecuteToolCallsParallelUnknownToolNameDoesNotPanic(t *testing.T) {
	toolset := tools.NewToolSet()
	toolset.AddTool(tools.NewTool("known_tool", "a tool that exists", nil, nil,
		func(ctx context.Context, params string) (*tools.ToolResult, error) {
			return &tools.ToolResult{Content: "should not be called"}, nil
		}))
	usage := &tools.ExecutionUsageData{}
	calls := llm.ToolsCalls{{Name: "unknown_tool", Args: "{}", CallId: "call-2"}}

	responses := executeToolCallsParallel(context.Background(), toolset, "test/model", calls, usage)

	got, ok := responses["call-2"]
	if !ok {
		t.Fatalf("no response recorded for call-2; responses=%v", responses)
	}
	if !strings.Contains(got, "unknown_tool") || !strings.Contains(got, "not found") {
		t.Errorf("expected a %q not-found result naming the tool, got %q", "unknown_tool", got)
	}
}

// A tool call the ToolSet DOES recognize still runs normally — the nil/
// unknown-name handling above must not have broken the found path.
func TestExecuteToolCallsParallelKnownToolRuns(t *testing.T) {
	toolset := tools.NewToolSet()
	toolset.AddTool(tools.NewTool("known_tool", "a tool that exists", nil, nil,
		func(ctx context.Context, params string) (*tools.ToolResult, error) {
			return &tools.ToolResult{Content: "ran ok"}, nil
		}))
	usage := &tools.ExecutionUsageData{}
	calls := llm.ToolsCalls{{Name: "known_tool", Args: "{}", CallId: "call-3"}}

	responses := executeToolCallsParallel(context.Background(), toolset, "test/model", calls, usage)

	if got := responses["call-3"]; got != "ran ok" {
		t.Errorf("responses[call-3] = %q, want %q", got, "ran ok")
	}
}
