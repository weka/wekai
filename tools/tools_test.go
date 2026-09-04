package tools

import "testing"

// GetToolByName must be safe to call on a nil *ToolSet, matching the
// nil-receiver pattern already used by AsOpenAi/AsOpenAiResponses/
// AsAnthropic on this type. A caller with no ToolSet configured (nil) still
// has to handle a model spontaneously emitting tool_calls without crashing.
func TestGetToolByNameNilReceiver(t *testing.T) {
	var ts *ToolSet
	if got := ts.GetToolByName("anything"); got != nil {
		t.Errorf("GetToolByName on nil ToolSet = %v, want nil", got)
	}
}

func TestGetToolByNameUnknownName(t *testing.T) {
	ts := NewToolSet()
	ts.AddTool(NewTool("known", "desc", nil, nil, nil))

	if got := ts.GetToolByName("unknown"); got != nil {
		t.Errorf("GetToolByName(%q) = %v, want nil", "unknown", got)
	}
	if got := ts.GetToolByName("known"); got == nil || got.Name != "known" {
		t.Errorf("GetToolByName(%q) = %v, want the registered tool", "known", got)
	}
}
