package registry

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
			if got := JoinBasePath(tc.base, tc.path); got != tc.want {
				t.Errorf("JoinBasePath(%q, %q) = %q, want %q", tc.base, tc.path, got, tc.want)
			}
		})
	}
}

// ResolveURL is what every probe the router makes on its own behalf now goes
// through. Before, each site concatenated, which was right only for a backend
// hosted at the root — and silently wrong in both directions otherwise.
func TestResolveURL(t *testing.T) {
	for _, tc := range []struct {
		name, backend, path, want string
	}{
		{"root-hosted backend is plain concatenation",
			"http://vllm:8000", "/health", "http://vllm:8000/health"},
		{"a trailing slash does not double up",
			"http://vllm:8000/", "/health", "http://vllm:8000/health"},
		{"root-hosted model listing is unchanged",
			"http://vllm:8000", "/v1/models", "http://vllm:8000/v1/models"},
		{"the redundant /v1 no longer doubles the version segment",
			"http://vllm:8000/v1", "/v1/models", "http://vllm:8000/v1/models"},
		{"google's compat surface lists at <base>/models",
			"https://generativelanguage.googleapis.com/v1beta/openai", "/v1/models",
			"https://generativelanguage.googleapis.com/v1beta/openai/models"},
		{"a versionless path is prefixed by the base",
			"https://generativelanguage.googleapis.com/v1beta/openai", "/metrics",
			"https://generativelanguage.googleapis.com/v1beta/openai/metrics"},
		// Documented, not fixed: /v1 + /health has no correct join, because the
		// suffix is a mistake rather than a base. The router warns about it at
		// startup instead of guessing.
		{"a redundant /v1 still misplaces the health probe",
			"http://vllm:8000/v1", "/health", "http://vllm:8000/v1/health"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveURL(tc.backend, tc.path); got != tc.want {
				t.Errorf("ResolveURL(%q, %q) = %q, want %q", tc.backend, tc.path, got, tc.want)
			}
		})
	}
}
