package serve_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/serve"
)

// Live tests against real hosted providers, mixed with a local mock fleet.
//
// They SKIP unless the relevant key is set, so an ordinary `go test ./...` is
// still hermetic. They exist because the merged model listing is the one place
// the router talks to an upstream on its own behalf rather than proxying, and
// hosted providers disagree about it in two ways a mock cannot reproduce:
// credentials are required, and the path is not the same everywhere.
//
// Only /models is called — a listing, not a completion, so running these costs
// nothing beyond a request.

type provider struct {
	name    string
	env     string
	base    string
	header  func(key string) (string, string)
	extra   map[string]string
	pattern string
}

var providers = []provider{
	{
		name: "openai", env: "OPENAI_API_KEY", pattern: "gpt",
		base:   "https://api.openai.com",
		header: func(k string) (string, string) { return "Authorization", "Bearer " + k },
	},
	{
		name: "anthropic", env: "ANTHROPIC_API_KEY", pattern: "claude",
		base:   "https://api.anthropic.com",
		header: func(k string) (string, string) { return "X-Api-Key", k },
		// Anthropic rejects a listing without a version.
		extra: map[string]string{"Anthropic-Version": "2023-06-01"},
	},
	{
		// Google's OpenAI-compatible surface. The /v1beta/openai prefix is part
		// of the BASE, so its listing is <base>/models — not <base>/v1/models,
		// which is why fetchModels tries both.
		name: "gemini", env: "GEMINI_API_KEY", pattern: "gemini",
		base:   "https://generativelanguage.googleapis.com/v1beta/openai",
		header: func(k string) (string, string) { return "Authorization", "Bearer " + k },
	},
}

// TestMergedModelsWithRealHostedProviders is the mixed deployment: a local mock
// fleet plus every hosted provider whose key is present, all behind one router.
func TestMergedModelsWithRealHostedProviders(t *testing.T) {
	// Opt-in, not merely key-gated. Keys live in a developer's shell, so
	// gating on the key alone quietly made `task verify` depend on three
	// third-party APIs answering within a timeout — which under -race they
	// intermittently do not, failing the suite for reasons that have nothing
	// to do with the change under test. Verification has to be deterministic;
	// run these on purpose with `task test:live`.
	if os.Getenv("WEKAI_LIVE") == "" {
		t.Skip("set WEKAI_LIVE=1 to run live hosted-provider tests (task test:live)")
	}

	local, _ := mockFleet(t, 1, mockvllm.SurfaceVLLM)

	routes := []serve.Route{{Patterns: "local", Endpoints: local}}
	var live []provider
	for _, p := range providers {
		if os.Getenv(p.env) == "" {
			t.Logf("%s: %s unset, skipping that pool", p.name, p.env)
			continue
		}
		live = append(live, p)
		routes = append(routes, serve.Route{
			Patterns: p.pattern, Endpoints: []string{p.base},
			// A hosted API has no /health; discovery would work this out, but
			// saying so skips a pointless probe over the internet.
			Passive: true, Name: p.name,
		})
	}
	if len(live) == 0 {
		t.Skip("no provider keys set")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes:          routes,
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 50 * time.Millisecond,
		HealthTimeout:   100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	// The local pool is actively probed, so wait for it before asserting that
	// merging keeps it — otherwise the first subtest races health convergence
	// and reads a missing pool as a merge bug.
	eventually(t, func() bool {
		resp, err := rt.Client().Get(rt.URL + "/v1/models")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		var got struct{ Data []struct{ Pool string } }
		_ = json.NewDecoder(resp.Body).Decode(&got)
		for _, m := range got.Data {
			if m.Pool == "local" {
				return true
			}
		}
		return false
	})

	// One credential per request, so each provider is exercised on its own —
	// which is also the honest limitation: a single caller cannot authenticate
	// to three providers at once, so a merged listing across hosted pools needs
	// per-route credentials the router does not have yet.
	for _, p := range live {
		t.Run(p.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodGet, rt.URL+"/v1/models", nil)
			k, v := p.header(os.Getenv(p.env))
			req.Header.Set(k, v)
			for hk, hv := range p.extra {
				req.Header.Set(hk, hv)
			}
			resp, err := rt.Client().Do(req)
			if err != nil {
				t.Fatalf("GET /v1/models: %v", err)
			}
			defer resp.Body.Close()

			var got struct {
				Data []struct {
					ID   string `json:"id"`
					Pool string `json:"pool"`
				} `json:"data"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
				t.Fatalf("decode: %v", err)
			}

			var fromProvider, fromLocal int
			for _, m := range got.Data {
				switch m.Pool {
				case p.name:
					fromProvider++
				case "local":
					fromLocal++
				}
			}
			if fromProvider == 0 {
				t.Errorf("%s contributed no models to the merged listing; with a valid "+
					"credential forwarded it should list its catalogue. got %d entries: %v",
					p.name, len(got.Data), summarise(got.Data))
			}
			if fromLocal == 0 {
				t.Errorf("the local mock pool vanished from the listing when %s was "+
					"queried; merging must not drop a pool", p.name)
			}
			t.Logf("%s: %d models, local: %d", p.name, fromProvider, fromLocal)
		})
	}
}

func summarise(data []struct {
	ID   string `json:"id"`
	Pool string `json:"pool"`
}) string {
	out := ""
	for i, m := range data {
		if i == 5 {
			return out + "..."
		}
		out += fmt.Sprintf("%s(%s) ", m.ID, m.Pool)
	}
	return out
}

// TestHostedPoolStaysUsableWithoutModelsOrHealth is the guarantee that matters
// for a mixed deployment: the two things the router cannot do to a hosted API —
// read vllm: metrics, and probe a liveness path — must not be read as "this
// endpoint is down".
//
// It uses a stub rather than a real provider so it runs everywhere, and asserts
// the whole chain: the pool is routable, it is counted for readiness, and a
// failed model listing removes neither it nor anyone else from the merged
// result.
func TestHostedPoolStaysUsableWithoutModelsOrHealth(t *testing.T) {
	local, _ := mockFleet(t, 1, mockvllm.SurfaceVLLM)

	var served int
	// A hosted API as the router meets it unauthenticated: 401 on the listing,
	// 404 on anything operational, but perfectly able to serve a request the
	// client authenticates itself.
	hosted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/models":
			w.WriteHeader(http.StatusUnauthorized)
		case "/health", "/metrics":
			w.WriteHeader(http.StatusNotFound)
		default:
			served++
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"msg_1","type":"message","content":[{"type":"text","text":"ok"}]}`)
		}
	}))
	defer hosted.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes: []serve.Route{
			{Patterns: "local", Endpoints: local, Name: "local"},
			{Patterns: "*", Endpoints: []string{hosted.URL}, Name: "hosted"},
		},
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 20 * time.Millisecond,
		HealthTimeout:   100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	// Routable despite failing every probe the router knows how to make.
	resp, err := rt.Client().Post(rt.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"claude-x","max_tokens":4,"messages":[]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("hosted pool returned %d; an endpoint with no /health and no "+
			"readable models is not therefore down", resp.StatusCode)
	}
	if served == 0 {
		t.Error("the hosted endpoint never received the request")
	}

	// Readiness counts it.
	rd, err := rt.Client().Get(rt.URL + "/readiness")
	if err != nil {
		t.Fatal(err)
	}
	rd.Body.Close()
	if rd.StatusCode != http.StatusOK {
		t.Errorf("/readiness = %d with a usable hosted pool", rd.StatusCode)
	}

	// And a pool whose listing 401s must not take the others down with it:
	// merge what can be merged.
	eventually(t, func() bool {
		r, err := rt.Client().Get(rt.URL + "/v1/models")
		if err != nil {
			return false
		}
		defer r.Body.Close()
		if r.StatusCode != http.StatusOK {
			return false
		}
		var got struct {
			Data []struct{ Pool string } `json:"data"`
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		for _, m := range got.Data {
			if m.Pool == "local" {
				return true
			}
		}
		return false
	})
}

// TestKnownHostedProvidersAreNotProbed: the well-known APIs are recognised by
// host, so the router does not spend two internet round trips per endpoint at
// startup discovering what it already knows.
func TestKnownHostedProvidersAreNotProbed(t *testing.T) {
	for _, ep := range []string{
		"https://api.anthropic.com",
		"https://api.openai.com/v1",
		"https://generativelanguage.googleapis.com/v1beta/openai",
		"https://my-tenant.openai.azure.com",
	} {
		if !serve.IsKnownHostedProvider(ep) {
			t.Errorf("%s not recognised as a hosted provider", ep)
		}
	}
	for _, ep := range []string{"http://vllm-a:8000", "http://127.0.0.1:9001"} {
		if serve.IsKnownHostedProvider(ep) {
			t.Errorf("%s wrongly treated as hosted; a local fleet must still be probed", ep)
		}
	}
}
