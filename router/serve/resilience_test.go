package serve_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/serve"
)

// killable wraps a mock vLLM so a test can take it down and bring it back,
// which is what a pod restart looks like to the router.
type killable struct {
	ts   *httptest.Server
	down bool
	// hits counts requests that actually reached the engine, so a test can tell
	// "routed elsewhere" from "routed here and failed".
	hits int
}

func newKillable(t *testing.T, surface mockvllm.Surface) *killable {
	t.Helper()
	cfg := mockvllm.DefaultConfig()
	cfg.ModelID = "m"
	cfg.MaxConcurrency = 64
	inner := mockvllm.NewServer(mockvllm.NewEngine(cfg))
	inner.Surface = surface
	k := &killable{}
	k.ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if k.down {
			// A dead vLLM: connection refused is the honest shape, but a 503
			// from an in-between proxy is just as common and is what a fatal
			// upstream failure looks like to the router.
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		k.hits++
		inner.Handler().ServeHTTP(w, r)
	}))
	t.Cleanup(k.ts.Close)
	return k
}

func chat(t *testing.T, ts *httptest.Server) int {
	t.Helper()
	resp, err := ts.Client().Post(ts.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

// A backend that dies must stop receiving traffic, and one that recovers must
// start receiving it again — quickly. Recovery speed is the whole reason the
// health checker probes unhealthy backends far more often than healthy ones:
// a recovered backend held out of rotation sends every request that could have
// used its warm cache somewhere colder.
func TestBackendDiesAndRecovers(t *testing.T) {
	a := newKillable(t, mockvllm.SurfaceVLLM)
	b := newKillable(t, mockvllm.SurfaceVLLM)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes:          []serve.Route{{Patterns: "*", Endpoints: []string{a.ts.URL, b.ts.URL}}},
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 20 * time.Millisecond,
		HealthTimeout:   50 * time.Millisecond,
		MaxAttempts:     3,
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	eventually(t, func() bool { return chat(t, rt) == http.StatusOK })

	// Kill one. Requests must keep succeeding on the survivor: a fatal upstream
	// failure has to take the backend out of rotation, not surface to the
	// client.
	a.down = true
	a.hits = 0
	deadline := time.Now().Add(3 * time.Second)
	ok := 0
	for time.Now().Before(deadline) && ok < 10 {
		if chat(t, rt) == http.StatusOK {
			ok++
		}
		time.Sleep(10 * time.Millisecond)
	}
	if ok < 10 {
		t.Fatalf("only %d/10 requests succeeded with one backend down; a dead backend "+
			"must be routed around, not returned to the client", ok)
	}

	// Bring it back. It has to re-enter rotation on its own.
	a.down = false
	before := a.hits
	recovered := false
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		chat(t, rt)
		if a.hits > before {
			recovered = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !recovered {
		t.Error("the recovered backend never received traffic again; it is stuck out of " +
			"rotation, and every request that could use its warm cache goes somewhere colder")
	}
}

// /v1/models must advertise what the ROUTER can serve — the union across pools —
// not whatever one pool happens to answer. A client discovering models through
// the router would otherwise never learn the other pools exist.
func TestModelsMergesEveryPool(t *testing.T) {
	mk := func(id string) *httptest.Server {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/models" {
				fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"object":"model"}]}`, id)
				return
			}
			if r.URL.Path == "/health" {
				return
			}
			fmt.Fprint(w, `{"ok":true}`)
		}))
		t.Cleanup(ts.Close)
		return ts
	}
	small, big := mk("small-7b"), mk("big-70b")
	// A pool with nothing alive must not fail the listing, only be absent.
	deadTS := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer deadTS.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes: []serve.Route{
			{Patterns: "small", Endpoints: []string{small.URL}},
			{Patterns: "big", Endpoints: []string{big.URL}},
			{Patterns: "gone", Endpoints: []string{deadTS.URL}, Passive: true},
		},
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 20 * time.Millisecond,
		HealthTimeout:   50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	var got struct {
		Data []struct {
			ID   string `json:"id"`
			Pool string `json:"pool"`
		} `json:"data"`
	}
	eventually(t, func() bool {
		resp, err := rt.Client().Get(rt.URL + "/v1/models")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		got.Data = nil
		_ = json.NewDecoder(resp.Body).Decode(&got)
		return len(got.Data) >= 2
	})

	ids := map[string]string{}
	for _, m := range got.Data {
		ids[m.ID] = m.Pool
	}
	for _, want := range []string{"small-7b", "big-70b"} {
		if _, ok := ids[want]; !ok {
			t.Errorf("/v1/models omits %q; it must advertise the union across pools, "+
				"or a client never learns the other pools exist. got %v", want, ids)
		}
	}
	if ids["small-7b"] != "small" || ids["big-70b"] != "big" {
		t.Errorf("each model must name the pool that serves it: %v", ids)
	}
}

// TestBackendURLEndingInV1IsFlagged: the router appends the dialect's own paths,
// so a backend written as http://host:8000/v1 still PROXIES correctly while
// every probe against it 404s — health, metrics and model discovery alike. The
// endpoint is then taken for a hosted API and never actively health-checked,
// which is a silent loss rather than a visible failure. It has to be loud.
func TestBackendURLEndingInV1IsFlagged(t *testing.T) {
	var logged strings.Builder
	h := slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelWarn}))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer up.Close()

	if _, err := serve.Handler(ctx, serve.Options{
		Routes: []serve.Route{{Patterns: "*", Endpoints: []string{up.URL + "/v1"}, Passive: true}},
		Log:    h,
	}); err != nil {
		t.Fatalf("Handler: %v", err)
	}
	if !strings.Contains(logged.String(), "ends in /v1") {
		t.Errorf("no warning for a /v1 backend URL; the degradation is silent otherwise:\n%s",
			logged.String())
	}
}
