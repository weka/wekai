package serve_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/serve"
)

// The three deployments the router is documented to support (docs/router.md),
// exercised against the mock vLLM engine rather than stubs — so cache
// accounting, admission and the vllm: metric surface are the real ones.

// mockFleet starts n mock vLLM instances on the given surface and returns their
// URLs plus the servers, for assertions on where traffic landed.
func mockFleet(t *testing.T, n int, surface mockvllm.Surface) ([]string, []*mockvllm.Server) {
	t.Helper()
	var urls []string
	var servers []*mockvllm.Server
	for range n {
		cfg := mockvllm.DefaultConfig()
		cfg.ModelID = "local-model"
		cfg.MaxConcurrency = 64
		eng := mockvllm.NewEngine(cfg)
		srv := mockvllm.NewServer(eng)
		srv.Surface = surface
		ts := httptest.NewServer(srv.Handler())
		t.Cleanup(ts.Close)
		urls = append(urls, ts.URL)
		servers = append(servers, srv)
	}
	return urls, servers
}

func startRouter(t *testing.T, opts serve.Options) *httptest.Server {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	if opts.HealthInterval == 0 {
		opts.HealthInterval = 20 * time.Millisecond
		opts.HealthTimeout = 10 * time.Millisecond
	}
	h, err := serve.Handler(ctx, opts)
	if err != nil {
		t.Fatalf("serve.Handler: %v", err)
	}
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)
	return ts
}

// post sends body to path and returns status plus the decoded JSON.
func post(t *testing.T, ts *httptest.Server, path, body string) (int, map[string]any) {
	t.Helper()
	resp, err := ts.Client().Post(ts.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

// eventually retries until cond holds or the deadline passes. Health has to
// converge before a backend is eligible, and that is a couple of probe
// intervals rather than instant.
func eventually(t *testing.T, cond func() bool) {
	t.Helper()
	// Generous: a poll that queries real hosted providers takes ~1s, so a short
	// deadline fits only a couple of attempts and fails on latency rather than
	// on the condition. Costs nothing when the condition holds early.
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("condition never held")
}

// Use case 1: one vLLM fleet, every model routed to it.
//
// The simplest deployment, and the one --backends exists for. Affinity should
// concentrate a shared prefix rather than spreading it, which is the whole
// reason to route rather than load-balance.
func TestUseCase1_SingleFleetAllTraffic(t *testing.T) {
	urls, servers := mockFleet(t, 3, mockvllm.SurfaceVLLM)
	rt := startRouter(t, serve.Options{
		Routes: []serve.Route{{Patterns: "*", Endpoints: urls}},
	})

	eventually(t, func() bool {
		code, _ := post(t, rt, "/v1/chat/completions",
			`{"model":"any-model","max_tokens":4,"messages":[{"role":"user","content":"warm up"}]}`)
		return code == http.StatusOK
	})

	const shared = "a long shared system prompt that several requests reuse verbatim"
	for range 8 {
		code, _ := post(t, rt, "/v1/chat/completions", fmt.Sprintf(
			`{"model":"any-model","max_tokens":4,"messages":[{"role":"user","content":%q}]}`, shared))
		if code != http.StatusOK {
			t.Fatalf("status %d", code)
		}
	}

	served := 0
	for _, s := range servers {
		if s.Engine().Stats().Admitted > 0 {
			served++
		}
	}
	if served == 0 {
		t.Fatal("no backend served anything")
	}
	if served == len(servers) {
		t.Errorf("all %d backends served the same shared prefix; affinity should "+
			"concentrate it, not spread it like a load balancer", len(servers))
	}
}

// Use case 2: per-model routes, over BOTH wire formats.
//
// A model name is a model name whichever schema carries it, so routing must not
// depend on the request shape. The Anthropic surface is a vLLM fronted with
// /v1/messages — same engine, same metrics — and must route by the same rules.
func TestUseCase2_PerModelRoutesAcrossBothAPIs(t *testing.T) {
	fastURLs, fast := mockFleet(t, 1, mockvllm.SurfaceVLLM)
	// The "big" pool speaks Anthropic, to prove the rule matches on the model
	// name rather than on which endpoint shape serves it.
	bigURLs, big := mockFleet(t, 1, mockvllm.SurfaceAnthropic)

	rt := startRouter(t, serve.Options{
		Routes: []serve.Route{
			{Patterns: "fast,small", Endpoints: fastURLs},
			{Patterns: "big,70b", Endpoints: bigURLs},
		},
	})

	eventually(t, func() bool {
		code, _ := post(t, rt, "/v1/chat/completions",
			`{"model":"fast-7b","max_tokens":4,"messages":[{"role":"user","content":"x"}]}`)
		return code == http.StatusOK
	})

	// OpenAI-format request for a "fast" model.
	if code, _ := post(t, rt, "/v1/chat/completions",
		`{"model":"fast-7b","max_tokens":4,"messages":[{"role":"user","content":"openai path"}]}`); code != http.StatusOK {
		t.Fatalf("openai-format request: status %d", code)
	}
	// Anthropic-format request for a "big" model, on the messages endpoint.
	if code, _ := post(t, rt, "/v1/messages",
		`{"model":"big-70b","max_tokens":4,"messages":[{"role":"user","content":"anthropic path"}]}`); code != http.StatusOK {
		t.Fatalf("anthropic-format request: status %d", code)
	}

	if fast[0].Engine().Stats().Admitted == 0 {
		t.Error("the fast pool served nothing; its OpenAI-format request did not arrive")
	}
	if big[0].Engine().Stats().Admitted == 0 {
		t.Error("the big pool served nothing; an Anthropic-format request must route " +
			"by model name exactly as an OpenAI one does")
	}
}

// Use case 3: self-hosted models by name, everything else to a hosted API.
//
// The mixed deployment. Rules are first-match-wins, so the specific pools come
// first and the catch-all last; the hosted fallback is passive because it has no
// /health to probe, which discovery works out on its own.
func TestUseCase3_SelfHostedModelsWithHostedFallback(t *testing.T) {
	localURLs, local := mockFleet(t, 2, mockvllm.SurfaceVLLM)

	var hostedHits int
	var hostedPaths []string
	hosted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A real hosted API: no /metrics, no /v1/models, no /health. Discovery
		// must work that out and fall back to passive health rather than
		// holding it unusable for failing probes it cannot answer.
		switch r.URL.Path {
		case "/metrics", "/v1/models", "/health":
			w.WriteHeader(http.StatusNotFound)
			return
		}
		hostedHits++
		hostedPaths = append(hostedPaths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg_1","type":"message","content":[{"type":"text","text":"ok"}]}`)
	}))
	defer hosted.Close()

	rt := startRouter(t, serve.Options{
		Routes: []serve.Route{
			{Patterns: "llama,mistral", Endpoints: localURLs},
			{Patterns: "*", Endpoints: []string{hosted.URL}},
		},
	})

	eventually(t, func() bool {
		code, _ := post(t, rt, "/v1/chat/completions",
			`{"model":"llama-3","max_tokens":4,"messages":[{"role":"user","content":"x"}]}`)
		return code == http.StatusOK
	})

	// A self-hosted model goes local.
	if code, _ := post(t, rt, "/v1/chat/completions",
		`{"model":"mistral-7b","max_tokens":4,"messages":[{"role":"user","content":"local"}]}`); code != http.StatusOK {
		t.Fatalf("local model: status %d", code)
	}
	// Anything else falls through to the hosted API, on its own path.
	if code, _ := post(t, rt, "/v1/messages",
		`{"model":"claude-sonnet-4","max_tokens":4,"messages":[{"role":"user","content":"hosted"}]}`); code != http.StatusOK {
		t.Fatalf("hosted fallback: status %d", code)
	}

	var localServed int64
	for _, s := range local {
		localServed += s.Engine().Stats().Admitted
	}
	if localServed == 0 {
		t.Error("the self-hosted pool served nothing")
	}
	if hostedHits == 0 {
		t.Error("the catch-all never reached the hosted API")
	}
	for _, p := range hostedPaths {
		if p != "/v1/messages" {
			t.Errorf("hosted API saw path %q; the path must be forwarded unchanged, "+
				"not rewritten to a dialect route", p)
		}
	}
	// A hosted API has no /health, and discovery must not hold it unhealthy for
	// failing a probe it cannot answer.
	if hostedHits < 1 {
		t.Error("hosted endpoint was never usable")
	}
}
