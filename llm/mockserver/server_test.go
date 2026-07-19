package mockserver

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

// sseEvent represents a parsed SSE event.
type sseEvent struct {
	Event string
	Data  map[string]any
}

func parseSSEEvents(t *testing.T, resp *http.Response) []sseEvent {
	t.Helper()
	var events []sseEvent
	scanner := bufio.NewScanner(resp.Body)
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			dataStr := strings.TrimPrefix(line, "data: ")
			var data map[string]any
			if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
				t.Fatalf("failed to parse SSE data JSON: %v\nraw: %s", err, dataStr)
			}
			events = append(events, sseEvent{Event: currentEvent, Data: data})
			currentEvent = ""
		}
		// blank lines are event separators, just skip
	}
	return events
}

func doMessagesRequest(t *testing.T, url string) *http.Response {
	t.Helper()
	body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req, err := http.NewRequest(http.MethodPost, url+"/v1/messages", strings.NewReader(body))
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	return resp
}

func TestMockServer_TwoTurnScript(t *testing.T) {
	script := []MockTurn{
		{
			// Turn 1: tool call
			Content: "Let me check that.",
			ToolCalls: []ToolCall{
				{Name: "read_file", Args: `{"path":"/tmp/test.txt"}`, ID: "toolu_abc123"},
			},
		},
		{
			// Turn 2: text only
			Content: "The file contains hello world.",
		},
	}

	srv := NewMockServer(script)
	defer srv.Close()

	// --- Turn 1 ---
	resp := doMessagesRequest(t, srv.URL)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	events := parseSSEEvents(t, resp)
	resp.Body.Close()

	if srv.CallCount() != 1 {
		t.Fatalf("expected call count 1, got %d", srv.CallCount())
	}

	// Verify event sequence for turn 1.
	eventTypes := make([]string, len(events))
	for i, e := range events {
		eventTypes[i] = e.Event
	}
	t.Logf("Turn 1 events: %v", eventTypes)

	// Should have: message_start, content_block_start, content_block_delta(s), content_block_stop,
	//              content_block_start (tool), content_block_delta (tool), content_block_stop (tool),
	//              message_delta, message_stop
	assertEventType(t, events, 0, "message_start")

	// Find the message_delta and check stop_reason is "tool_use"
	var messageDelta *sseEvent
	for _, e := range events {
		if e.Event == "message_delta" {
			messageDelta = &e
			break
		}
	}
	if messageDelta == nil {
		t.Fatal("no message_delta event found")
	}
	delta := messageDelta.Data["delta"].(map[string]any)
	if delta["stop_reason"] != "tool_use" {
		t.Fatalf("expected stop_reason tool_use, got %v", delta["stop_reason"])
	}

	// Verify text content was streamed.
	var textContent strings.Builder
	for _, e := range events {
		if e.Event == "content_block_delta" {
			d := e.Data["delta"].(map[string]any)
			if d["type"] == "text_delta" {
				textContent.WriteString(d["text"].(string))
			}
		}
	}
	if textContent.String() != "Let me check that." {
		t.Fatalf("expected text 'Let me check that.', got %q", textContent.String())
	}

	// Verify tool call was streamed.
	var toolName, toolArgs, toolID string
	for _, e := range events {
		if e.Event == "content_block_start" {
			cb, ok := e.Data["content_block"].(map[string]any)
			if ok && cb["type"] == "tool_use" {
				toolName = cb["name"].(string)
				toolID = cb["id"].(string)
			}
		}
		if e.Event == "content_block_delta" {
			d := e.Data["delta"].(map[string]any)
			if d["type"] == "input_json_delta" {
				toolArgs = d["partial_json"].(string)
			}
		}
	}
	if toolName != "read_file" {
		t.Fatalf("expected tool name read_file, got %q", toolName)
	}
	if toolID != "toolu_abc123" {
		t.Fatalf("expected tool ID toolu_abc123, got %q", toolID)
	}
	if toolArgs != `{"path":"/tmp/test.txt"}` {
		t.Fatalf("unexpected tool args: %q", toolArgs)
	}

	// Verify last event is message_stop.
	lastEvent := events[len(events)-1]
	if lastEvent.Event != "message_stop" {
		t.Fatalf("expected last event message_stop, got %s", lastEvent.Event)
	}

	// --- Turn 2 ---
	resp2 := doMessagesRequest(t, srv.URL)
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp2.StatusCode)
	}
	events2 := parseSSEEvents(t, resp2)
	resp2.Body.Close()

	if srv.CallCount() != 2 {
		t.Fatalf("expected call count 2, got %d", srv.CallCount())
	}

	// Verify stop_reason is "end_turn" for text-only turn.
	var messageDelta2 *sseEvent
	for _, e := range events2 {
		if e.Event == "message_delta" {
			messageDelta2 = &e
			break
		}
	}
	if messageDelta2 == nil {
		t.Fatal("no message_delta event in turn 2")
	}
	delta2 := messageDelta2.Data["delta"].(map[string]any)
	if delta2["stop_reason"] != "end_turn" {
		t.Fatalf("expected stop_reason end_turn, got %v", delta2["stop_reason"])
	}

	// Verify text content.
	var textContent2 strings.Builder
	for _, e := range events2 {
		if e.Event == "content_block_delta" {
			d := e.Data["delta"].(map[string]any)
			if d["type"] == "text_delta" {
				textContent2.WriteString(d["text"].(string))
			}
		}
	}
	if textContent2.String() != "The file contains hello world." {
		t.Fatalf("expected text 'The file contains hello world.', got %q", textContent2.String())
	}

	// No tool calls in turn 2.
	for _, e := range events2 {
		if e.Event == "content_block_start" {
			cb, ok := e.Data["content_block"].(map[string]any)
			if ok && cb["type"] == "tool_use" {
				t.Fatal("unexpected tool_use block in turn 2")
			}
		}
	}
}

func TestMockServer_ErrorTurn(t *testing.T) {
	script := []MockTurn{
		{
			Error: &MockError{
				StatusCode: 529,
				Type:       "overloaded_error",
				Message:    "API is overloaded",
			},
		},
	}

	srv := NewMockServer(script)
	defer srv.Close()

	resp := doMessagesRequest(t, srv.URL)
	defer resp.Body.Close()

	if resp.StatusCode != 529 {
		t.Fatalf("expected 529, got %d", resp.StatusCode)
	}

	var errResp map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		t.Fatalf("failed to decode error: %v", err)
	}
	if errResp["type"] != "error" {
		t.Fatalf("expected type error, got %v", errResp["type"])
	}
	errObj := errResp["error"].(map[string]any)
	if errObj["type"] != "overloaded_error" {
		t.Fatalf("expected overloaded_error, got %v", errObj["type"])
	}

	if srv.CallCount() != 1 {
		t.Fatalf("expected call count 1, got %d", srv.CallCount())
	}
}

func TestMockServer_Reset(t *testing.T) {
	srv := NewMockServer([]MockTurn{{Content: "first"}})
	defer srv.Close()

	resp := doMessagesRequest(t, srv.URL)
	resp.Body.Close()
	if srv.CallCount() != 1 {
		t.Fatalf("expected 1, got %d", srv.CallCount())
	}

	srv.Reset([]MockTurn{{Content: "second"}})
	if srv.CallCount() != 0 {
		t.Fatalf("expected 0 after reset, got %d", srv.CallCount())
	}

	resp2 := doMessagesRequest(t, srv.URL)
	events := parseSSEEvents(t, resp2)
	resp2.Body.Close()

	var text strings.Builder
	for _, e := range events {
		if e.Event == "content_block_delta" {
			d := e.Data["delta"].(map[string]any)
			if d["type"] == "text_delta" {
				text.WriteString(d["text"].(string))
			}
		}
	}
	if text.String() != "second" {
		t.Fatalf("expected 'second', got %q", text.String())
	}
}

func TestMockServer_ScriptExhausted(t *testing.T) {
	srv := NewMockServer([]MockTurn{{Content: "only one"}})
	defer srv.Close()

	resp := doMessagesRequest(t, srv.URL)
	resp.Body.Close()

	// Second call should fail.
	resp2 := doMessagesRequest(t, srv.URL)
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", resp2.StatusCode)
	}
}

func TestMockServer_AutoGeneratedToolID(t *testing.T) {
	script := []MockTurn{
		{
			ToolCalls: []ToolCall{
				{Name: "bash", Args: `{"command":"ls"}`},
				{Name: "read", Args: `{"path":"foo"}`},
			},
		},
	}

	srv := NewMockServer(script)
	defer srv.Close()

	resp := doMessagesRequest(t, srv.URL)
	events := parseSSEEvents(t, resp)
	resp.Body.Close()

	var toolIDs []string
	for _, e := range events {
		if e.Event == "content_block_start" {
			cb, ok := e.Data["content_block"].(map[string]any)
			if ok && cb["type"] == "tool_use" {
				toolIDs = append(toolIDs, cb["id"].(string))
			}
		}
	}

	if len(toolIDs) != 2 {
		t.Fatalf("expected 2 tool IDs, got %d", len(toolIDs))
	}
	if toolIDs[0] != "toolu_mock_001" {
		t.Fatalf("expected toolu_mock_001, got %s", toolIDs[0])
	}
	if toolIDs[1] != "toolu_mock_002" {
		t.Fatalf("expected toolu_mock_002, got %s", toolIDs[1])
	}
}

func TestMockServer_ValidationErrors(t *testing.T) {
	srv := NewMockServer([]MockTurn{{Content: "hi"}})
	defer srv.Close()

	// Missing messages field.
	body := `{"model":"test-model"}`
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing messages, got %d", resp.StatusCode)
	}

	// Missing model field.
	body2 := `{"messages":[{"role":"user","content":"hi"}]}`
	req2, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/messages", strings.NewReader(body2))
	req2.Header.Set("Content-Type", "application/json")
	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing model, got %d", resp2.StatusCode)
	}
}

func TestMockServer_Latency(t *testing.T) {
	script := []MockTurn{
		{
			Content: "delayed",
			Latency: 50 * time.Millisecond,
		},
	}
	srv := NewMockServer(script)
	defer srv.Close()

	start := time.Now()
	resp := doMessagesRequest(t, srv.URL)
	events := parseSSEEvents(t, resp)
	resp.Body.Close()
	elapsed := time.Since(start)

	if elapsed < 40*time.Millisecond {
		t.Fatalf("expected at least 40ms latency, got %v", elapsed)
	}

	// Verify content still arrives correctly.
	var text strings.Builder
	for _, e := range events {
		if e.Event == "content_block_delta" {
			d := e.Data["delta"].(map[string]any)
			if d["type"] == "text_delta" {
				text.WriteString(d["text"].(string))
			}
		}
	}
	if text.String() != "delayed" {
		t.Fatalf("expected 'delayed', got %q", text.String())
	}
}

func assertEventType(t *testing.T, events []sseEvent, idx int, expected string) {
	t.Helper()
	if idx >= len(events) {
		t.Fatalf("event index %d out of range (have %d events)", idx, len(events))
	}
	if events[idx].Event != expected {
		t.Fatalf("event[%d]: expected %q, got %q", idx, expected, events[idx].Event)
	}
}

// suppress unused import warning
var _ = fmt.Sprintf
