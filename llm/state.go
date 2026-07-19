package llm

import (
	"encoding/json"
	"fmt"
	"time"
)

// StateVersion is the current schema version for provider state serialization.
const StateVersion = 2

// ProviderRecoveryState stores the last API request/response needed to
// reconstruct provider-native conversation history after a restart.
type ProviderRecoveryState struct {
	Version      int             `json:"version"`
	Provider     string          `json:"provider"`
	ClientType   string          `json:"client_type"`
	ModelID      string          `json:"model_id"`
	LastRequest  json.RawMessage `json:"last_request"`
	LastResponse json.RawMessage `json:"last_response"`
}

// NewProviderRecoveryState creates a new ProviderRecoveryState.
func NewProviderRecoveryState(provider, clientType, modelID string, lastReq, lastResp json.RawMessage) *ProviderRecoveryState {
	return &ProviderRecoveryState{
		Version:      StateVersion,
		Provider:     provider,
		ClientType:   clientType,
		ModelID:      modelID,
		LastRequest:  lastReq,
		LastResponse: lastResp,
	}
}

// ToJSON serializes the recovery state to JSON.
func (s *ProviderRecoveryState) ToJSON() (string, error) {
	data, err := json.Marshal(s)
	if err != nil {
		return "", fmt.Errorf("failed to marshal provider recovery state: %w", err)
	}
	return string(data), nil
}

// FromJSON deserializes a ProviderRecoveryState from JSON.
// Returns nil, nil when data is empty (no prior state).
func FromJSON(data string) (*ProviderRecoveryState, error) {
	if data == "" {
		return nil, nil
	}
	var state ProviderRecoveryState
	if err := json.Unmarshal([]byte(data), &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal provider recovery state: %w", err)
	}
	return &state, nil
}

// ProviderStateLogEntry is one entry in the JSONL append log written on every
// API exchange. It captures the full request and raw SSE stream bytes for audit and debugging.
type ProviderStateLogEntry struct {
	Timestamp        int64           `json:"ts"`
	Provider         string          `json:"provider"`
	ClientType       string          `json:"client_type"`
	ModelID          string          `json:"model_id"`
	Request          json.RawMessage `json:"request"`
	RawResponseBytes []byte          `json:"raw_response,omitempty"` // raw HTTP response bytes; nil for non-streaming
}

// NewProviderStateLogEntry creates a log entry stamped with the current time.
func NewProviderStateLogEntry(provider, clientType, modelID string, req json.RawMessage, rawResp []byte) *ProviderStateLogEntry {
	return &ProviderStateLogEntry{
		Timestamp:        time.Now().UnixMilli(),
		Provider:         provider,
		ClientType:       clientType,
		ModelID:          modelID,
		Request:          req,
		RawResponseBytes: rawResp,
	}
}

// ToJSONL serializes the entry as a single JSON line followed by a newline.

func (e *ProviderStateLogEntry) ToJSONL() string {
	data, err := json.Marshal(e)
	if err != nil {
		return fmt.Sprintf(`{"ts":%d,"error":"marshal failed"}`+"\n", e.Timestamp)
	}
	return string(data) + "\n"
}
