package gateway_test

import (
	"testing"

	"github.com/weka/wekai/router/internal/gateway"
)

// The contract: under per-user prefixes EVERY request carries one, so the first
// segment of any multi-segment path is the user by definition. There is no way
// to tell a user named "v1" from a client that forgot its prefix, and guessing
// would make the rule depend on the very paths it is meant to expose.
func TestSplitUserPrefix(t *testing.T) {
	cases := []struct {
		input    string
		wantUser string
		wantPath string
	}{
		{"/mati.mizrahi/api/event_logging/batch", "mati.mizrahi", "/api/event_logging/batch"},
		{"/mati.mizrahi/v1/messages", "mati.mizrahi", "/v1/messages"},
		{"/mati.mizrahi/v1/messages/count_tokens", "mati.mizrahi", "/v1/messages/count_tokens"},
		{"/mati.mizrahi/v1", "mati.mizrahi", "/v1"},
		// Non-API two-segment paths surrender their first segment too.
		{"/mati.mizrahi/healthz", "mati.mizrahi", "/healthz"},
		// Input without a prefix while the mode is on is malformed by contract;
		// the first segment is taken as the user regardless.
		{"/v1/messages", "v1", "/messages"},
		{"/api/event_logging/batch", "api", "/event_logging/batch"},
		// Single-segment paths are infra probes: nobody prefixes those, and
		// taking their only segment would leave nothing to route.
		{"/healthz", "", "/healthz"},
		{"/", "", "/"},
		{"", "", ""},
	}

	for _, tc := range cases {
		gotUser, gotPath := gateway.SplitUserPrefix(tc.input)
		if gotUser != tc.wantUser || gotPath != tc.wantPath {
			t.Errorf("SplitUserPrefix(%q) = (%q, %q), want (%q, %q)",
				tc.input, gotUser, gotPath, tc.wantUser, tc.wantPath)
		}
	}
}
