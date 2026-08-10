package serve_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// A router that fronts a local fleet on the dialect's own routes AND a hosted
// API on paths the dialect never claims is the configuration this consolidation
// had to preserve: it is how the captured traffic behind every replay file was
// collected. The dialect claims /v1/chat/completions; Anthropic's /v1/messages
// it does not, and before the passthrough tier that request 404'd at the mux.
func TestPassthroughReachesUnclaimedPaths(t *testing.T) {
	// Guarded: the router makes its own requests to a backend on background
	// goroutines (health, model discovery), so this handler is not called only
	// from the test's goroutine.
	var mu sync.Mutex
	var gotPaths []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotPaths = append(gotPaths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	rt := newTestRouter(t, upstream.URL)
	defer rt.Close()

	for _, path := range []string{"/v1/messages", "/v1/chat/completions"} {
		resp, err := rt.Client().Post(rt.URL+path, "application/json",
			strings.NewReader(`{"model":"m","messages":[]}`))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("%s: status %d, want 200", path, resp.StatusCode)
		}
	}

	// Only the paths this test drove. The router also asks the backend things on
	// its own behalf — a model listing, a health probe — and those are not what
	// is being asserted here.
	mu.Lock()
	var clientPaths []string
	for _, p := range gotPaths {
		if p == "/v1/messages" || p == "/v1/chat/completions" {
			clientPaths = append(clientPaths, p)
		}
	}
	mu.Unlock()

	if len(clientPaths) != 2 {
		t.Fatalf("upstream saw %v, want both client paths forwarded", clientPaths)
	}
	for i, want := range []string{"/v1/messages", "/v1/chat/completions"} {
		if clientPaths[i] != want {
			t.Errorf("upstream saw %q, want %q — the path must be preserved, "+
				"not rewritten to a dialect route", clientPaths[i], want)
		}
	}
}
