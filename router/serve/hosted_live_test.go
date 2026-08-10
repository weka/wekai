package serve_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
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
