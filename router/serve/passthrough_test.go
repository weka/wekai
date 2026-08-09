package serve_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A router that fronts a local fleet on the dialect's own routes AND a hosted
// API on paths the dialect never claims is the configuration this consolidation
// had to preserve: it is how the captured traffic behind every replay file was
// collected. The dialect claims /v1/chat/completions; Anthropic's /v1/messages
// it does not, and before the passthrough tier that request 404'd at the mux.
func TestPassthroughReachesUnclaimedPaths(t *testing.T) {
	var gotPaths []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPaths = append(gotPaths, r.URL.Path)
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

	if len(gotPaths) != 2 {
		t.Fatalf("upstream saw %v, want both paths forwarded", gotPaths)
	}
	for i, want := range []string{"/v1/messages", "/v1/chat/completions"} {
		if gotPaths[i] != want {
			t.Errorf("upstream saw %q, want %q — the path must be preserved, "+
				"not rewritten to a dialect route", gotPaths[i], want)
		}
	}
}
