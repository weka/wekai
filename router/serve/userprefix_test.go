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

	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/serve"
)

// A client sets its base URL to http://router/<user>, so every path it sends
// carries that segment. The router has to remove it before anything else looks:
// the allowlist, the route table and the upstream all know only /v1/messages.
func TestUserPrefixIsStrippedBeforeRoutingAndForwarding(t *testing.T) {
	var mu sync.Mutex
	var upstreamPaths []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			fmt.Fprint(w, "vllm:num_requests_running 0\n")
			return
		case "/health":
			return
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
			return
		}
		mu.Lock()
		upstreamPaths = append(upstreamPaths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer up.Close()

	hook := &capturedHook{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		HealthInterval: 20 * time.Millisecond, HealthTimeout: 10 * time.Millisecond,
		AutoModel: "off", UserPrefix: true, Capture: hook,
		APIKey: "inbound-key", PathAllowlist: []string{"/v1/"},
		Routes: []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: []string{up.URL}}},
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	post := func(path string) int {
		req, _ := http.NewRequest(http.MethodPost, rt.URL+path, strings.NewReader(
			`{"model":"m","max_tokens":4,"system":"`+strings.Repeat("s", 400)+
				`","messages":[{"role":"user","content":"x"}]}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer inbound-key")
		resp, err := rt.Client().Do(req)
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) && post("/alice/v1/messages") != http.StatusOK {
		time.Sleep(20 * time.Millisecond)
	}
	mu.Lock()
	upstreamPaths = nil
	mu.Unlock()

	if code := post("/alice/v1/messages"); code != http.StatusOK {
		t.Fatalf("status %d: a prefixed path must be authorised and routed like the "+
			"bare one", code)
	}

	mu.Lock()
	got := append([]string(nil), upstreamPaths...)
	mu.Unlock()
	if len(got) != 1 || got[0] != "/v1/messages" {
		t.Errorf("upstream saw %v, want [/v1/messages]: forwarding the caller's "+
			"segment gives a hosted API a path it has never heard of", got)
	}

	// The user survives, as a field of its own — and the path the client sent
	// survives with it, because capture records what actually arrived.
	var rec serve.Captured
	eventually(t, func() bool {
		var ok bool
		rec, ok = hook.lastOK()
		return ok
	})
	if rec.User != "alice" {
		t.Errorf("captured user %q, want alice", rec.User)
	}
	if rec.Request.URL.Path != "/alice/v1/messages" {
		t.Errorf("captured path %q; capture records the path as the client sent it",
			rec.Request.URL.Path)
	}
	// The routing outcome reaches capture at all, which is what makes a record
	// worth keeping.
	if rec.Pool != "fleet" || rec.Backend == "" {
		t.Errorf("captured pool=%q backend=%q; a record that cannot say where a "+
			"request went is not much of a record", rec.Pool, rec.Backend)
	}
}

// Prefixed Anthropic traffic must reach the dialect route, not the passthrough
// tier — otherwise the router load balances every per-user request and the
// prefix affinity it exists to provide is off for the whole deployment.
func TestUserPrefixTrafficIsStillCacheRouted(t *testing.T) {
	urls, _ := mockFleet(t, 3, mockvllm.SurfaceAnthropic)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		HealthInterval: 20 * time.Millisecond, HealthTimeout: 10 * time.Millisecond,
		AutoModel: "off", UserPrefix: true,
		Routes: []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: urls}},
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	body := fmt.Sprintf(`{"model":"local-model","max_tokens":4,"system":%q,
	  "messages":[{"role":"user","content":"hello"}]}`,
		strings.Repeat("a long shared system prompt. ", 200))

	eventually(t, func() bool {
		code, _ := post(t, rt, "/alice/v1/messages", body)
		return code == http.StatusOK
	})

	before := counters(t, rt.URL, "router_route_decisions_total")
	for range 5 {
		if code, _ := post(t, rt, "/alice/v1/messages", body); code != http.StatusOK {
			t.Fatalf("status %d", code)
		}
	}
	after := counters(t, rt.URL, "router_route_decisions_total")

	const cache = `router_route_decisions_total{decision="cache",pool="fleet"}`
	const load = `router_route_decisions_total{decision="load",pool="fleet"}`
	if got := after[cache] - before[cache]; got != 5 {
		t.Errorf("%.0f of 5 per-user requests were cache routed (load: %.0f); a "+
			"prefixed path must reach the same route as a bare one",
			got, after[load]-before[load])
	}
}
