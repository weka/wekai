package serve_test

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/serve"
)

// counters reads the router's own exposition from the serving listener, which a
// keyless router mounts there.
func counters(t *testing.T, base, prefix string) map[string]float64 {
	t.Helper()
	resp, err := http.Get(base + "/metrics")
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	out := map[string]float64{}
	for _, line := range strings.Split(string(b), "\n") {
		if line == "" || line[0] == '#' || !strings.HasPrefix(line, prefix) {
			continue
		}
		i := strings.LastIndexByte(line, ' ')
		if i < 0 {
			continue
		}
		var v float64
		if _, err := fmt.Sscanf(line[i+1:], "%g", &v); err == nil {
			out[line[:i]] = v
		}
	}
	return out
}

// Anthropic-format traffic must be CACHE routed, not merely load balanced.
//
// The passthrough tier extracts no units, so anything left there is served by a
// least-outstanding load balancer with no prefix affinity at all. /v1/messages
// must not be one of those paths: the two surfaces have to reach the same
// decision on the same conversation.
func TestAnthropicMessagesAreCacheRouted(t *testing.T) {
	urls, _ := mockFleet(t, 3, mockvllm.SurfaceAnthropic)
	rt := startRouter(t, serve.Options{
		AutoModel: "off",
		Routes:    []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: urls}},
	})

	system := strings.Repeat("a long shared system prompt. ", 200)
	body := fmt.Sprintf(`{"model":"local-model","max_tokens":4,"system":%q,
	  "messages":[{"role":"user","content":"hello"}]}`, system)

	eventually(t, func() bool {
		code, _ := post(t, rt, "/v1/messages", body)
		return code == http.StatusOK
	})

	before := counters(t, rt.URL, "router_route_decisions_total")
	for range 5 {
		if code, _ := post(t, rt, "/v1/messages", body); code != http.StatusOK {
			t.Fatalf("status %d", code)
		}
	}
	after := counters(t, rt.URL, "router_route_decisions_total")

	const cache = `router_route_decisions_total{decision="cache",pool="fleet"}`
	const load = `router_route_decisions_total{decision="load",pool="fleet"}`
	if got := after[cache] - before[cache]; got != 5 {
		t.Errorf("%.0f of 5 Anthropic-format requests were routed by prefix cache "+
			"(load: %.0f); /v1/messages must be cache routed, not load balanced",
			got, after[load]-before[load])
	}
}

// A token count carries the body of the generation it is counting, so nothing in
// the payload tells them apart. Committing units for it would record a backend as
// holding a prefix it never processed, and the next real request would be sent
// there to collect a hit that cannot exist.
func TestCountTokensDoesNotClaimAPrefix(t *testing.T) {
	urls, _ := mockFleet(t, 3, mockvllm.SurfaceAnthropic)
	rt := startRouter(t, serve.Options{
		AutoModel: "off",
		Routes:    []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: urls}},
	})

	body := fmt.Sprintf(`{"model":"local-model","max_tokens":4,"system":%q,
	  "messages":[{"role":"user","content":"hello"}]}`,
		strings.Repeat("a long shared system prompt. ", 200))

	eventually(t, func() bool {
		code, _ := post(t, rt, "/v1/messages", body)
		return code == http.StatusOK
	})

	before := counters(t, rt.URL, "router_route_decisions_total")
	for range 5 {
		post(t, rt, "/v1/messages/count_tokens", body)
	}
	after := counters(t, rt.URL, "router_route_decisions_total")

	const cache = `router_route_decisions_total{decision="cache",pool="fleet"}`
	const load = `router_route_decisions_total{decision="load",pool="fleet"}`
	if got := after[cache] - before[cache]; got != 0 {
		t.Errorf("%.0f token counts were routed as cache hits; a count populates no "+
			"KV, so it must not commit a holder", got)
	}
	// It is still ROUTED — by model, to a backend, by load. Asserting only the
	// absence of cache decisions would pass just as well if the request had never
	// reached routing at all.
	if got := after[load] - before[load]; got != 5 {
		t.Errorf("%.0f of 5 token counts reached a routing decision; they must still "+
			"be routed by model, just without a prefix", got)
	}
}
