package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNewProviderRecoveryState(t *testing.T) {
	req := json.RawMessage(`{"model":"gpt-4"}`)
	resp := json.RawMessage(`{"id":"r1"}`)
	s := NewProviderRecoveryState("openai", "openai", "gpt-4", req, resp)

	if s.Version != StateVersion {
		t.Errorf("expected version %d, got %d", StateVersion, s.Version)
	}
	if s.Provider != "openai" {
		t.Errorf("expected provider openai, got %s", s.Provider)
	}
	if s.ClientType != "openai" {
		t.Errorf("expected client_type openai, got %s", s.ClientType)
	}
	if s.ModelID != "gpt-4" {
		t.Errorf("expected model_id gpt-4, got %s", s.ModelID)
	}
	if string(s.LastRequest) != string(req) {
		t.Errorf("LastRequest mismatch")
	}
	if string(s.LastResponse) != string(resp) {
		t.Errorf("LastResponse mismatch")
	}
}

func TestProviderRecoveryState_ToJSON_FromJSON(t *testing.T) {
	req := json.RawMessage(`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}`)
	resp := json.RawMessage(`{"id":"chatcmpl-123","choices":[{"message":{"role":"assistant","content":"hi"}}]}`)
	orig := NewProviderRecoveryState("openai", "openai", "gpt-4o-mini", req, resp)

	jsonStr, err := orig.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}
	if jsonStr == "" {
		t.Fatal("ToJSON returned empty string")
	}

	recovered, err := FromJSON(jsonStr)
	if err != nil {
		t.Fatalf("FromJSON failed: %v", err)
	}
	if recovered == nil {
		t.Fatal("FromJSON returned nil")
	}
	if recovered.Version != orig.Version {
		t.Errorf("version mismatch: got %d want %d", recovered.Version, orig.Version)
	}
	if recovered.Provider != orig.Provider {
		t.Errorf("provider mismatch")
	}
	if string(recovered.LastRequest) != string(orig.LastRequest) {
		t.Errorf("LastRequest mismatch after roundtrip")
	}
	if string(recovered.LastResponse) != string(orig.LastResponse) {
		t.Errorf("LastResponse mismatch after roundtrip")
	}
}

func TestFromJSON_EmptyString(t *testing.T) {
	state, err := FromJSON("")
	if err != nil {
		t.Errorf("FromJSON with empty string should not error: %v", err)
	}
	if state != nil {
		t.Error("FromJSON with empty string should return nil")
	}
}

func TestFromJSON_InvalidJSON(t *testing.T) {
	_, err := FromJSON(`{"version":2,"provider":"openai"`) // truncated
	if err == nil {
		t.Error("FromJSON should return error for invalid JSON")
	}
}

func TestProviderRecoveryState_JSONFieldNames(t *testing.T) {
	req := json.RawMessage(`{"req":1}`)
	resp := json.RawMessage(`{"resp":1}`)
	s := NewProviderRecoveryState("prov", "cli", "mod", req, resp)

	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	jsonStr := string(data)

	for _, field := range []string{`"version"`, `"provider"`, `"client_type"`, `"model_id"`, `"last_request"`, `"last_response"`} {
		if !strings.Contains(jsonStr, field) {
			t.Errorf("JSON missing field %s; got: %s", field, jsonStr)
		}
	}
}

func TestProviderStateLogEntry_ToJSONL(t *testing.T) {
	req := json.RawMessage(`{"model":"gpt-4"}`)
	resp := json.RawMessage(`{"id":"r1"}`)
	e := NewProviderStateLogEntry("openai", "openai", "gpt-4", req, resp)

	line := e.ToJSONL()
	if !strings.HasSuffix(line, "\n") {
		t.Error("ToJSONL should end with newline")
	}
	// Must be valid JSON (strip the trailing newline)
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSuffix(line, "\n")), &parsed); err != nil {
		t.Errorf("ToJSONL produced invalid JSON: %v", err)
	}
	if parsed["provider"] != "openai" {
		t.Errorf("unexpected provider in log entry: %v", parsed["provider"])
	}
	if parsed["ts"] == nil {
		t.Error("log entry missing ts field")
	}
}
