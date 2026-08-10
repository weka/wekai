package proxy

import "testing"

// The path a backend receives is composed from two places: the base in the
// router's configuration and the path the client actually sent. For a
// root-hosted API only the client's matters, and a base that is present has to
// stand in for the client's version segment rather than pile on top of it.
func TestJoinBase(t *testing.T) {
	for _, tc := range []struct {
		name, base, path, want string
	}{
		{"root-hosted backend passes the client path through unchanged",
			"", "/v1/chat/completions", "/v1/chat/completions"},
		{"anthropic at the root is likewise untouched",
			"", "/v1/messages", "/v1/messages"},
		{"google's compat surface gets its prefix and keeps one version segment",
			"/v1beta/openai", "/v1/chat/completions", "/v1beta/openai/chat/completions"},
		{"a trailing slash on the base does not double up",
			"/v1beta/openai/", "/v1/chat/completions", "/v1beta/openai/chat/completions"},
		{"the redundant /v1 people write composes back to the same URL",
			"/v1", "/v1/chat/completions", "/v1/chat/completions"},
		{"a path with no version segment is simply prefixed",
			"/v1beta/openai", "/metrics", "/v1beta/openai/metrics"},
		{"a base that is only a slash is no base at all",
			"/", "/v1/chat/completions", "/v1/chat/completions"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := joinBase(tc.base, tc.path); got != tc.want {
				t.Errorf("joinBase(%q, %q) = %q, want %q", tc.base, tc.path, got, tc.want)
			}
		})
	}
}
