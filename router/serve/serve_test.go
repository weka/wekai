package serve_test

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/weka/wekai/router/serve"
)

// newTestRouter starts a router with one catch-all route at endpoints and
// returns it as a live test server.
func newTestRouter(t *testing.T, endpoints ...string) *httptest.Server {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	opts := serve.Options{
		// Passive: a test upstream is a plain httptest server with no
		// /health, exactly like a hosted API.
		Routes:        []serve.Route{{Patterns: "*", Endpoints: endpoints, Passive: true}},
		MaxAttempts:   2,
		DrainDeadline: time.Second,
	}
	h, err := serve.Handler(ctx, opts)
	if err != nil {
		t.Fatalf("serve.Handler: %v", err)
	}
	return httptest.NewServer(h)
}
