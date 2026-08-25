package serve_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/weka/wekai/router/serve"
)

type capturedHook struct {
	mu   sync.Mutex
	recs []serve.Captured
}

func (h *capturedHook) WantsResponseBody() bool { return false }
func (h *capturedHook) Record(c serve.Captured) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.recs = append(h.recs, c)
}

// lastOK returns the most recent record for a request the router actually
// served. Not simply the last record: capture appends AFTER the response has
// been written, so a client that has read its 200 can reach here before the
// record for it exists, and the newest record is then the previous attempt.
func (h *capturedHook) lastOK() (serve.Captured, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := len(h.recs) - 1; i >= 0; i-- {
		if h.recs[i].Status == http.StatusOK {
			return h.recs[i], true
		}
	}
	return serve.Captured{}, false
}

// A captured record exists to say what happened to a request, and where it went
// is most of that. Capture wraps the gateway from outside, so it and the
// handlers within have to share one holder — with two, capture reads the one
// nothing ever wrote to and every record claims it does not know.
func TestCaptureRecordsTheRoutingOutcome(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			fmt.Fprint(w, "vllm:num_requests_running 0\n")
		case "/health":
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
		default:
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"ok":true}`)
		}
	}))
	defer up.Close()

	hook := &capturedHook{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		HealthInterval: 20 * time.Millisecond, HealthTimeout: 10 * time.Millisecond,
		AutoModel: "off", Capture: hook,
		Routes: []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: []string{up.URL}}},
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := rt.Client().Post(rt.URL+"/v1/messages", "application/json",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"x"}]}`))
		if err == nil {
			code := resp.StatusCode
			resp.Body.Close()
			if code == http.StatusOK {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
	}

	var rec serve.Captured
	eventually(t, func() bool {
		var ok bool
		rec, ok = hook.lastOK()
		return ok
	})
	if rec.Pool != "fleet" {
		t.Errorf("captured pool %q, want the pool that served it", rec.Pool)
	}
	if rec.Backend == "" {
		t.Error("captured record names no backend")
	}
}
