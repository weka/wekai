package cli

import "testing"

func TestStripUserPrefix(t *testing.T) {
	cases := []struct {
		input    string
		wantUser string
		wantPath string
	}{
		{"/mati.mizrahi/api/event_logging/batch", "mati.mizrahi", "/api/event_logging/batch"},
		{"/mati.mizrahi/v1/messages", "mati.mizrahi", "/v1/messages"},
		{"/mati.mizrahi/v1/messages/count_tokens", "mati.mizrahi", "/v1/messages/count_tokens"},
		{"/mati.mizrahi/v1", "mati.mizrahi", "/v1"},
		// Non-API two-segment paths: user-prefix is stripped unconditionally.
		{"/mati.mizrahi/healthz", "mati.mizrahi", "/healthz"},
		// The cases below represent input without a user prefix while user-prefix
		// mode is on — malformed by contract; the first segment is taken as the
		// user by definition.
		{"/v1/messages", "v1", "/messages"},
		{"/api/event_logging/batch", "api", "/event_logging/batch"},
		// Single-segment paths (infra probes): pass through unchanged.
		{"/healthz", "", "/healthz"},
		{"/", "", "/"},
		{"", "", ""},
	}

	for _, tc := range cases {
		gotUser, gotPath := stripUserPrefix(tc.input)
		if gotUser != tc.wantUser || gotPath != tc.wantPath {
			t.Errorf("stripUserPrefix(%q) = (%q, %q), want (%q, %q)",
				tc.input, gotUser, gotPath, tc.wantUser, tc.wantPath)
		}
	}
}
