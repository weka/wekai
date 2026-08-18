package benchmark

import (
	"strings"
	"testing"
)

// TestMergedResponseReassemblesToolCalls: tool-call arguments arrive as
// fragments split at arbitrary points, and a merge that concatenates them
// wrongly produces plausible-looking JSON that is not what was sent — the
// worst outcome for a file whose job is to be believed.
func TestMergedResponseReassemblesToolCalls(t *testing.T) {
	t.Run("openai streamed", func(t *testing.T) {
		raw := strings.Join([]string{
			`data: {"choices":[{"delta":{"content":"let me look"}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"Read","arguments":"{\"path\""}}]}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"/etc/hosts\"}"}}]}}]}`,
			`data: [DONE]`,
			"",
		}, "\n\n")
		got := string(mergedResponse("let me look", []byte(raw)))
		if !strings.Contains(got, "let me look") {
			t.Error("merged file lost the scored text")
		}
		if !strings.Contains(got, `{"path":"/etc/hosts"}`) {
			t.Errorf("tool arguments not reassembled from their fragments:\n%s", got)
		}
		if !strings.Contains(got, "Read") {
			t.Errorf("tool name missing:\n%s", got)
		}
	})

	t.Run("anthropic streamed", func(t *testing.T) {
		raw := strings.Join([]string{
			`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","name":"Grep"}}`,
			`data: {"type":"content_block_delta","index":1,"delta":{"partial_json":"{\"q\":"}}`,
			`data: {"type":"content_block_delta","index":1,"delta":{"partial_json":"\"needle\"}"}}`,
			"",
		}, "\n\n")
		got := string(mergedResponse("thinking out loud", []byte(raw)))
		if !strings.Contains(got, "Grep") || !strings.Contains(got, `{"q":"needle"}`) {
			t.Errorf("anthropic tool call not reassembled:\n%s", got)
		}
	})

	t.Run("non-streaming body", func(t *testing.T) {
		raw := `{"content":[{"type":"text","text":"hi"},{"type":"tool_use","name":"Bash","input":{"cmd":"ls"}}]}`
		got := string(mergedResponse("hi", []byte(raw)))
		if !strings.Contains(got, "Bash") || !strings.Contains(got, `"cmd":"ls"`) {
			t.Errorf("whole-body tool call not extracted:\n%s", got)
		}
	})

	t.Run("plain text response carries no tool-call heading", func(t *testing.T) {
		raw := "data: {\"choices\":[{\"delta\":{\"content\":\"just prose\"}}]}\n\ndata: [DONE]\n\n"
		got := string(mergedResponse("just prose", []byte(raw)))
		if strings.Contains(got, "tool calls") {
			t.Errorf("a response with no tool calls grew a tool-call section:\n%s", got)
		}
		if strings.TrimSpace(got) != "just prose" {
			t.Errorf("merged = %q, want just the scored text", got)
		}
	})

	t.Run("garbage frames are skipped, not fatal", func(t *testing.T) {
		raw := "data: not json\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"
		if got := string(mergedResponse("ok", []byte(raw))); strings.TrimSpace(got) != "ok" {
			t.Errorf("merged = %q; an unparseable frame must be skipped", got)
		}
	})
}

// TestMergedResponseSeparatesScoredTextFromReconstruction: the two halves come
// from different places — one is what the scorer read, the other is this file's
// own reconstruction — and a reader deciding whether a PRESENCE_MISS is real
// has to be able to tell which is which.
func TestMergedResponseSeparatesScoredTextFromReconstruction(t *testing.T) {
	raw := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"X","arguments":"{}"}}]}}]}`
	got := string(mergedResponse("scored text here", []byte(raw)))
	scored := strings.Index(got, "scored text here")
	heading := strings.Index(got, "NOT part of the text above")
	if scored == -1 || heading == -1 || scored > heading {
		t.Errorf("scored text must come first and the reconstruction must be labelled:\n%s", got)
	}
}
