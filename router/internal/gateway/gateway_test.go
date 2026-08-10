package gateway_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/wekai/router/internal/dialect/openai"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/routing"
	"github.com/weka/wekai/router/internal/testutil/mockvllm"
	"github.com/weka/wekai/router/internal/viz"
)

type harness struct {
	srv     *httptest.Server
	gw      *gateway.Server
	reg     *registry.Registry
	workers []*mockvllm.Worker
}

// harnessConfig spans what the tests configure across three now-separate
// structs: the gateway's own listener knobs, the pool's capacity signal, and
// the proxy's upstream behaviour. Splitting them in production was the point —
// each package's real dependencies are visible — but a test harness reads
// better with one knob bag.
type harnessConfig struct {
	APIKey                string
	RequireAuthForProbes  bool
	MaxBodyBytes          int64
	MaxConcurrentRequests int
	PathAllowlist         []string
	CORSOrigins           []string

	MaxNodeConcurrency int64
	RebalanceRatio     float64

	MaxAttempts        int
	UpstreamCredential string
	StreamBufferBytes  int
}

func defaultHarnessConfig() harnessConfig {
	return harnessConfig{
		MaxBodyBytes:      64 << 20,
		MaxAttempts:       2,
		StreamBufferBytes: 64 << 10,
	}
}

func (c harnessConfig) gateway() gateway.Config {
	return gateway.Config{
		APIKey:                c.APIKey,
		RequireAuthForProbes:  c.RequireAuthForProbes,
		MaxBodyBytes:          c.MaxBodyBytes,
		MaxConcurrentRequests: c.MaxConcurrentRequests,
		PathAllowlist:         c.PathAllowlist,
		CORSOrigins:           c.CORSOrigins,
		DefaultCapacity:       1,
	}
}

func newHarness(t *testing.T, nWorkers int, mutate func(*harnessConfig)) *harness {
	t.Helper()
	lease.ResetAccountingErrors()

	cfg := defaultHarnessConfig()
	if mutate != nil {
		mutate(&cfg)
	}

	reg := registry.New(registry.Options{})
	h := &harness{reg: reg}
	for i := 0; i < nWorkers; i++ {
		w := mockvllm.New()
		t.Cleanup(w.Close)
		h.workers = append(h.workers, w)
		b, err := reg.Add(registry.Spec{URL: w.URL(), Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		// Tests drive health directly; the checker is exercised separately.
		b.SetHealth(registry.Healthy)
	}

	d := openai.New()
	px := proxy.New(proxy.Config{
		MaxAttempts:        cfg.MaxAttempts,
		UpstreamCredential: cfg.UpstreamCredential,
		StreamBufferBytes:  cfg.StreamBufferBytes,
	})
	gw := gateway.New(cfg.gateway(), mustTable(t, reg, mustFlow(t, affinity.Config{
		NodeConcurrency: cfg.MaxNodeConcurrency,
		RebalanceRatio:  cfg.RebalanceRatio,
	})), px, d)
	h.gw = gw
	h.srv = httptest.NewServer(gw)
	t.Cleanup(h.srv.Close)
	return h
}

// mustFlow builds the router's one routing flow. Signals are enabled by being
// set in cfg; the refused signal is always on.
func mustFlow(t *testing.T, cfg affinity.Config) *affinity.Policy {
	t.Helper()
	p, err := affinity.New(cfg, policy.LeastOutstanding{})
	if err != nil {
		t.Fatalf("affinity.New: %v", err)
	}
	return p
}

// mustTable wraps one registry and flow as the single implicit pool — the
// shape a router has when it fronts one fleet and routes every model to it.
func mustTable(t *testing.T, reg *registry.Registry, flow *affinity.Policy) *routing.Table {
	t.Helper()
	tbl, err := routing.NewTable([]routing.Rule{{
		Pool: &pool.Pool{Name: affinity.DefaultPoolName, Registry: reg, Flow: flow},
	}})
	if err != nil {
		t.Fatalf("routing.NewTable: %v", err)
	}
	return tbl
}

func (h *harness) post(t *testing.T, path, body string, hdr map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, h.srv.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func (h *harness) assertNoLeaks(t *testing.T) {
	t.Helper()
	// Give any deferred release a moment to run after the client returns.
	deadline := time.Now().Add(2 * time.Second)
	for {
		ok := true
		for _, b := range h.reg.Snapshot().Backends {
			if b.Inflight() != 0 {
				ok = false
			}
		}
		if ok {
			break
		}
		if time.Now().After(deadline) {
			for _, b := range h.reg.Snapshot().Backends {
				if b.Inflight() != 0 {
					t.Errorf("backend %s leaked inflight = %d", b.URL, b.Inflight())
				}
			}
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if n := lease.AccountingErrors(); n != 0 {
		t.Errorf("load accounting errors = %d, want 0", n)
	}
}

func TestProxiesChatCompletion(t *testing.T) {
	h := newHarness(t, 1, nil)
	resp := h.post(t, "/v1/chat/completions", `{"model":"m","messages":[]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if h.workers[0].CallCount() != 1 {
		t.Fatalf("worker received %d calls, want 1", h.workers[0].CallCount())
	}
	h.assertNoLeaks(t)
}

// COMPAT-2: the body reaching the worker must be byte-identical to the client's.
func TestRequestBodyForwardedByteIdentical(t *testing.T) {
	h := newHarness(t, 1, nil)
	body := `{"model":"m","messages":[{"role":"user","content":"hello ünïcödé 🎉"}],"stream":false}`
	resp := h.post(t, "/v1/chat/completions", body, nil)
	defer resp.Body.Close()

	call, ok := h.workers[0].LastCall()
	if !ok {
		t.Fatal("worker received nothing")
	}
	if call.Body != body {
		t.Errorf("body was modified in transit:\n got %q\nwant %q", call.Body, body)
	}
	h.assertNoLeaks(t)
}

// TestPreflightSucceedsWithAuthEnabled guards GW-N3.
//
// In v1 auth was strictly outside CORS, so a browser preflight — which carries no
// Authorization header — got a 401 before the CORS layer could answer it. Any
// browser client was unusable whenever an API key was set.
func TestPreflightSucceedsWithAuthEnabled(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "secret-key"
		c.CORSOrigins = []string{"https://app.example.com"}
	})

	req, _ := http.NewRequest(http.MethodOptions, h.srv.URL+"/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://app.example.com")
	req.Header.Set("Access-Control-Request-Method", "POST")
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("preflight status = %d, want 204 — auth is ordered outside CORS", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "https://app.example.com" {
		t.Errorf("Allow-Origin = %q, want the request origin", got)
	}
}

func TestAuthRejectsMissingAndWrongKey(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) { c.APIKey = "secret-key" })

	for _, tc := range []struct {
		name string
		hdr  map[string]string
		want int
	}{
		{"no credential", nil, http.StatusUnauthorized},
		{"wrong bearer", map[string]string{"Authorization": "Bearer nope"}, http.StatusUnauthorized},
		{"malformed scheme", map[string]string{"Authorization": "Basic abc"}, http.StatusUnauthorized},
		{"empty bearer", map[string]string{"Authorization": "Bearer "}, http.StatusUnauthorized},
		{"correct bearer", map[string]string{"Authorization": "Bearer secret-key"}, http.StatusOK},
		{"correct x-api-key", map[string]string{"X-Api-Key": "secret-key"}, http.StatusOK},
	} {
		resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`, tc.hdr)
		if resp.StatusCode != tc.want {
			t.Errorf("%s: status = %d, want %d", tc.name, resp.StatusCode, tc.want)
		}
		if tc.want == http.StatusUnauthorized {
			// Errors must use the dialect's envelope, not bare text (API-9).
			var env struct {
				Error struct{ Message, Type, Code string } `json:"error"`
			}
			b, _ := io.ReadAll(resp.Body)
			if err := json.Unmarshal(b, &env); err != nil || env.Error.Code == "" {
				t.Errorf("%s: body %q is not an OpenAI error envelope", tc.name, b)
			}
		}
		resp.Body.Close()
	}
}

// AUTH-N4: the client's credential must never reach a backend. v1 forwarded the
// inbound Authorization header verbatim to every worker and its logs.
func TestWorkerReceivesNoClientCredential(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) { c.APIKey = "secret-key" })
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`,
		map[string]string{"Authorization": "Bearer secret-key"})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	call, ok := h.workers[0].LastCall()
	if !ok {
		t.Fatal("worker received nothing")
	}
	if got := call.Header.Get("Authorization"); got != "" {
		t.Errorf("worker received Authorization %q — the client credential leaked to the backend", got)
	}
	if got := call.Header.Get("X-Api-Key"); got != "" {
		t.Errorf("worker received X-Api-Key %q", got)
	}
}

func TestUpstreamCredentialIsInjected(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "inbound"
		c.UpstreamCredential = "outbound"
	})
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`,
		map[string]string{"Authorization": "Bearer inbound"})
	defer resp.Body.Close()

	call, _ := h.workers[0].LastCall()
	if got := call.Header.Get("Authorization"); got != "Bearer outbound" {
		t.Errorf("worker Authorization = %q, want the configured upstream credential", got)
	}
}

// GW-N1: v1's catch-all read its body with an unbounded limit, an
// unauthenticated memory-exhaustion DoS. Here the limit is armed on every path.
func TestOversizeBodyIsRejected(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) { c.MaxBodyBytes = 1024 })
	big := `{"model":"m","pad":"` + strings.Repeat("x", 8192) + `"}`
	resp := h.post(t, "/v1/chat/completions", big, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", resp.StatusCode)
	}
	if h.workers[0].CallCount() != 0 {
		t.Error("oversize body was forwarded to a backend")
	}
	h.assertNoLeaks(t)
}

// AUTH-N3: allowlist matching must respect segment boundaries.
func TestAllowlistSegmentBoundary(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.PathAllowlist = []string{"/v1/chat/completions", "/v1/models"}
	})
	cases := []struct {
		path string
		want int
	}{
		{"/v1/chat/completions", http.StatusOK},
		{"/v1/models", http.StatusOK},           // exact
		{"/v1/modelsXXX", http.StatusNotFound},  // must NOT match a prefix
		{"/v1/embeddings", http.StatusNotFound}, // not listed
	}
	for _, c := range cases {
		var resp *http.Response
		if c.path == "/v1/models" || c.path == "/v1/modelsXXX" {
			r, err := h.srv.Client().Get(h.srv.URL + c.path)
			if err != nil {
				t.Fatal(err)
			}
			resp = r
		} else {
			resp = h.post(t, c.path, `{"model":"m"}`, nil)
		}
		if resp.StatusCode != c.want {
			t.Errorf("%s: status = %d, want %d", c.path, resp.StatusCode, c.want)
		}
		resp.Body.Close()
	}
}

// AUTH-8 (withdrawn) pins what an EMPTY allowlist does, because the requirements
// once demanded the opposite and the comments claimed the opposite was shipped.
//
// Empty means "serve every path", and auth still guards every path. That is v1's
// behaviour. The withdrawn requirement's premise was that empty-means-allow-all
// "turns a config typo into an open router"; the second half of this test is what
// makes that false — with no allowlist at all, an uncredentialed request to a
// normal inference path is still 401, not 200.
func TestEmptyAllowlistServesAllPathsUnderAuth(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "secret"
		c.PathAllowlist = nil // the case under test
	})

	// Reachable: no 404 from the allowlist, because there is no allowlist.
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`,
		map[string]string{"Authorization": "Bearer secret"})
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		t.Error("empty allowlist 404'd a path; empty must serve everything")
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("authorized request = %d, want 200", resp.StatusCode)
	}

	// Still guarded: an empty allowlist exempts nothing from auth.
	un := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	defer un.Body.Close()
	if un.StatusCode != http.StatusUnauthorized {
		t.Errorf("uncredentialed request with empty allowlist = %d, want 401 — "+
			"an absent allowlist must not disable auth", un.StatusCode)
	}
}

// AUTH-11: the allowlist must not be able to exempt an admin endpoint from auth.
//
// Listing an admin path is the adversarial case: it makes the path reachable, and
// the requirement is that reachable still means authenticated. Unlisted admin
// paths 404 instead, which is the other half.
func TestAdminNotExemptibleByAllowlist(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "secret"
		// Deliberately list an admin path and a mutating one.
		c.PathAllowlist = []string{"/get_loads", "/add_worker"}
	})

	get := func(path string, hdr map[string]string) int {
		req, err := http.NewRequest(http.MethodGet, h.srv.URL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range hdr {
			req.Header.Set(k, v)
		}
		resp, err := h.srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		return resp.StatusCode
	}

	if code := get("/get_loads", nil); code != http.StatusUnauthorized {
		t.Errorf("allowlisted GET /get_loads without credential = %d, want 401 — "+
			"the allowlist must not exempt admin from auth", code)
	}
	if code := get("/get_loads", map[string]string{"Authorization": "Bearer secret"}); code != http.StatusOK {
		t.Errorf("allowlisted GET /get_loads with credential = %d, want 200", code)
	}

	// POST /add_worker is listed, so it is reachable; without a credential it must
	// still be refused before it can mutate the registry.
	before := len(h.reg.Snapshot().Backends)
	aw := h.post(t, "/add_worker?url=http://evil.invalid:8000", "", nil)
	defer aw.Body.Close()
	if aw.StatusCode != http.StatusUnauthorized {
		t.Errorf("allowlisted POST /add_worker without credential = %d, want 401", aw.StatusCode)
	}
	if after := len(h.reg.Snapshot().Backends); after != before {
		t.Errorf("registry changed from %d to %d backends on an unauthenticated "+
			"add_worker", before, after)
	}

	// An admin path left off the allowlist is not reachable at all.
	if code := get("/workers", map[string]string{"Authorization": "Bearer secret"}); code != http.StatusNotFound {
		t.Errorf("unlisted GET /workers with valid credential = %d, want 404", code)
	}
}

// AUTH-5: GET /liveness is the one public endpoint, even with auth and an
// allowlist configured, so a kubelet httpGet probe still works.
func TestLivenessIsPublic(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "secret"
		c.PathAllowlist = []string{"/v1/chat/completions"}
	})
	resp, err := h.srv.Client().Get(h.srv.URL + "/liveness")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /liveness = %d, want 200 with auth and an allowlist set", resp.StatusCode)
	}
}

func TestRequestIDIsEchoedAndForwarded(t *testing.T) {
	h := newHarness(t, 1, nil)
	const id = "client-supplied-id-123"
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`,
		map[string]string{"X-Request-Id": id})
	defer resp.Body.Close()

	if got := resp.Header.Get("X-Request-Id"); got != id {
		t.Errorf("response X-Request-Id = %q, want the inbound id %q", got, id)
	}
	call, _ := h.workers[0].LastCall()
	if got := call.Header.Get("X-Request-Id"); got != id {
		t.Errorf("worker X-Request-Id = %q, want %q — the id must survive the hop unchanged", got, id)
	}
}

// STR-N2: v1 forcibly rewrote Content-Type to text/event-stream whenever the
// request asked to stream, so a JSON error body reached the client as a
// malformed SSE stream instead of a readable error.
func TestNon2xxOnStreamPassesContentTypeThrough(t *testing.T) {
	h := newHarness(t, 1, nil)
	h.workers[0].SetScript(mockvllm.Script{
		Status: http.StatusBadRequest, ContentType: "application/json",
		Body: `{"error":{"message":"bad model"}}`,
	})
	resp := h.post(t, "/v1/chat/completions", `{"model":"m","stream":true}`, nil)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 relayed as-is", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json passed through unchanged", ct)
	}
	h.assertNoLeaks(t)
}

func TestStreamingRelaysAllChunks(t *testing.T) {
	h := newHarness(t, 1, nil)
	h.workers[0].SetScript(mockvllm.Script{Stream: true, Tokens: 5})
	resp := h.post(t, "/v1/chat/completions", `{"model":"m","stream":true}`, nil)
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if n := bytes.Count(body, []byte("data: ")); n != 6 { // 5 tokens + [DONE]
		t.Errorf("relayed %d SSE events, want 6:\n%s", n, body)
	}
	if !bytes.Contains(body, []byte("data: [DONE]")) {
		t.Error("terminal marker missing from the relayed stream")
	}
	h.assertNoLeaks(t)
}

// REL-1/REL-2: a 503 is retried on a DIFFERENT backend.
func TestRetryGoesToADifferentBackend(t *testing.T) {
	h := newHarness(t, 2, nil)
	for _, w := range h.workers {
		w.SetScript(mockvllm.Script{Status: http.StatusServiceUnavailable, Body: `{}`})
	}
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	defer resp.Body.Close()

	total := h.workers[0].CallCount() + h.workers[1].CallCount()
	if total != 2 {
		t.Errorf("total upstream calls = %d, want exactly 2 (max_attempts)", total)
	}
	if h.workers[0].CallCount() == 2 || h.workers[1].CallCount() == 2 {
		t.Error("both attempts hit the same backend; retry must pick a different one")
	}
	h.assertNoLeaks(t)
}

// REL-2: a 500 must NOT be retried — the backend may already have done
// non-idempotent work.
func TestServerErrorIsNotRetried(t *testing.T) {
	h := newHarness(t, 2, nil)
	for _, w := range h.workers {
		w.SetScript(mockvllm.Script{Status: http.StatusInternalServerError, Body: `{}`})
	}
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	defer resp.Body.Close()

	if total := h.workers[0].CallCount() + h.workers[1].CallCount(); total != 1 {
		t.Errorf("total upstream calls = %d, want 1: 500 is not retryable", total)
	}
	h.assertNoLeaks(t)
}

// HLT-11: with no healthy backend the router returns a distinguishable 503 and
// never routes to a known-bad one.
func TestNoHealthyBackendsReturns503(t *testing.T) {
	h := newHarness(t, 1, nil)
	for _, b := range h.reg.Snapshot().Backends {
		b.SetHealth(registry.Unhealthy)
	}
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	var env struct {
		Error struct{ Code string } `json:"error"`
	}
	b, _ := io.ReadAll(resp.Body)
	if json.Unmarshal(b, &env); env.Error.Code != "no_healthy_backends" {
		t.Errorf("error code = %q, want no_healthy_backends (a distinguishable code)", env.Error.Code)
	}
	if h.workers[0].CallCount() != 0 {
		t.Error("routed to an unhealthy backend")
	}
}

func TestReadinessReflectsHealthyBackends(t *testing.T) {
	h := newHarness(t, 1, nil)
	resp, err := h.srv.Client().Get(h.srv.URL + "/readiness")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("readiness = %d with a healthy backend, want 200", resp.StatusCode)
	}

	for _, b := range h.reg.Snapshot().Backends {
		b.SetHealth(registry.Unhealthy)
	}
	resp2, err := h.srv.Client().Get(h.srv.URL + "/readiness")
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("readiness = %d with no healthy backend, want 503", resp2.StatusCode)
	}
}

// SEC-6/CFG-7: /get_server_info must not leak secrets, and must state plainly
// that residency is predicted rather than observed (RES-4).
func TestServerInfoRedactsSecretsAndStatesResidencySource(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "super-secret-value"
		c.UpstreamCredential = "upstream-secret-value"
	})
	req, _ := http.NewRequest(http.MethodGet, h.srv.URL+"/get_server_info", nil)
	req.Header.Set("Authorization", "Bearer super-secret-value")
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	for _, secret := range []string{"super-secret-value", "upstream-secret-value"} {
		if bytes.Contains(body, []byte(secret)) {
			t.Errorf("/get_server_info leaked %q", secret)
		}
	}
	var out struct {
		ResidencySource string `json:"residency_source"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.ResidencySource != "predicted" {
		t.Errorf("residency_source = %q, want \"predicted\" — FR-RTR-01 is approximated, "+
			"and saying so is a requirement", out.ResidencySource)
	}
}

// SEC-5: the admin API must reject non-http(s) worker URLs (SSRF guard).
func TestAddWorkerRejectsBadSchemes(t *testing.T) {
	h := newHarness(t, 1, nil)
	for _, u := range []string{"file:///etc/passwd", "gopher://x", ""} {
		resp := h.post(t, "/add_worker?url="+u, "", nil)
		if resp.StatusCode == http.StatusCreated {
			t.Errorf("add_worker accepted %q", u)
		}
		resp.Body.Close()
	}
}

func TestListWorkersIsDeterministicallyOrdered(t *testing.T) {
	h := newHarness(t, 5, nil)
	var first []string
	for i := 0; i < 10; i++ {
		resp, err := h.srv.Client().Get(h.srv.URL + "/workers")
		if err != nil {
			t.Fatal(err)
		}
		var out struct {
			Workers []struct{ URL string } `json:"workers"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()

		got := make([]string, len(out.Workers))
		for j, w := range out.Workers {
			got[j] = w.URL
		}
		if first == nil {
			first = got
			continue
		}
		if fmt.Sprint(got) != fmt.Sprint(first) {
			t.Fatalf("worker order changed between calls:\n%v\n%v", first, got)
		}
	}
}

// Concurrency smoke test: many requests through the whole stack, asserting the
// lease invariant holds end-to-end (LB-7 at the integration level).
func TestConcurrentRequestsLeaveNoInflight(t *testing.T) {
	h := newHarness(t, 4, nil)
	h.workers[1].SetScript(mockvllm.Script{Status: http.StatusServiceUnavailable, Body: `{}`})
	h.workers[2].SetScript(mockvllm.Script{Stream: true, Tokens: 3})

	const n = 200
	done := make(chan struct{}, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer func() { done <- struct{}{} }()
			body := `{"model":"m","stream":false}`
			if i%3 == 0 {
				body = `{"model":"m","stream":true}`
			}
			resp := h.post(t, "/v1/chat/completions", body, nil)
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}(i)
	}
	for i := 0; i < n; i++ {
		<-done
	}
	h.assertNoLeaks(t)
}

// TestMetricsCarryTheMatchedRouteClass is a regression test for a bug found while
// smoke-testing the binary: every metric and log line was labelled
// route="unmatched".
//
// The cause is structural and easy to reintroduce. The access-log middleware sits
// OUTSIDE the mux, so it cannot know the route class when it builds its context,
// and a context the handler derives with context.WithValue is invisible to the
// middleware that wrapped it. The handler must therefore write into a holder the
// middleware installed. Without this, the `route` dimension — the one operators
// slice dashboards by — is worthless.
func TestMetricsCarryTheMatchedRouteClass(t *testing.T) {
	h := newHarness(t, 1, nil)

	before := testutil.ToFloat64(metrics.RequestsTotal.WithLabelValues("chat", "openai", "2xx"))
	unmatchedBefore := testutil.ToFloat64(metrics.RequestsTotal.WithLabelValues("unmatched", "", "2xx"))

	resp := h.post(t, "/v1/chat/completions", `{"model":"m","messages":[]}`, nil)
	resp.Body.Close()

	after := testutil.ToFloat64(metrics.RequestsTotal.WithLabelValues("chat", "openai", "2xx"))
	unmatchedAfter := testutil.ToFloat64(metrics.RequestsTotal.WithLabelValues("unmatched", "", "2xx"))

	if after != before+1 {
		t.Errorf("router_requests_total{route=chat,dialect=openai} went %v -> %v, want +1: "+
			"the matched route class did not reach the access-log middleware", before, after)
	}
	if unmatchedAfter != unmatchedBefore {
		t.Errorf("a matched request was counted as route=unmatched (%v -> %v)",
			unmatchedBefore, unmatchedAfter)
	}
}

// TestKubeletProbesWorkWithAuthEnabled is a regression test for a bug found by
// running the container the way Kubernetes runs it.
//
// The kubelet's httpGet probe cannot present a credential — doing so would mean
// embedding the secret in the pod spec in plaintext. With /readiness behind auth
// the probe got a 401, so the pod never became Ready and the Deployment never
// received traffic. That is exactly why v1's README instructed operators to
// replace httpGet with an exec probe, an instruction that cannot be followed on a
// distroless image with no shell.
func TestKubeletProbesWorkWithAuthEnabled(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) { c.APIKey = "secret-key" })

	for _, path := range []string{"/liveness", "/readiness", "/health"} {
		resp, err := h.srv.Client().Get(h.srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("unauthenticated GET %s = %d, want 200: a kubelet probe cannot "+
				"authenticate, so this makes the pod permanently NotReady\nbody: %s",
				path, resp.StatusCode, body)
		}
	}
}

// The fleet size is operational detail: an unauthenticated probe gets only the
// boolean it needs.
func TestUnauthenticatedReadinessDisclosesOnlyReadiness(t *testing.T) {
	h := newHarness(t, 3, func(c *harnessConfig) { c.APIKey = "secret-key" })

	resp, err := h.srv.Client().Get(h.srv.URL + "/readiness")
	if err != nil {
		t.Fatal(err)
	}
	var anon map[string]any
	json.NewDecoder(resp.Body).Decode(&anon)
	resp.Body.Close()
	if _, leaked := anon["healthy_backends"]; leaked {
		t.Errorf("unauthenticated /readiness disclosed the backend count: %v", anon)
	}
	if anon["ready"] != true {
		t.Errorf("unauthenticated /readiness should still report readiness: %v", anon)
	}

	req, _ := http.NewRequest(http.MethodGet, h.srv.URL+"/readiness", nil)
	req.Header.Set("Authorization", "Bearer secret-key")
	resp2, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var authed map[string]any
	json.NewDecoder(resp2.Body).Decode(&authed)
	resp2.Body.Close()
	if got, ok := authed["healthy_backends"]; !ok || got.(float64) != 3 {
		t.Errorf("authenticated /readiness should report the count, got %v", authed)
	}
}

// An operator who wants probes locked down can have that, at the cost of needing
// probes that can authenticate.
func TestRequireAuthForProbesIsHonoured(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.APIKey = "secret-key"
		c.RequireAuthForProbes = true
	})
	resp, err := h.srv.Client().Get(h.srv.URL + "/readiness")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("with require_auth_for_probes, /readiness = %d, want 401", resp.StatusCode)
	}

	// /liveness stays public regardless: it discloses only that the process is up.
	resp2, err := h.srv.Client().Get(h.srv.URL + "/liveness")
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("/liveness = %d, want 200 even with require_auth_for_probes", resp2.StatusCode)
	}
}

// A readiness probe reporting "not ready" is the mechanism working, so it must
// not be logged at ERROR. Observed in the cluster: a 15-minute vLLM model load
// produced ~90 ERROR lines that were all normal startup, which is how operators
// learn to ignore errors. A 503 on an inference path must still be an error.
func TestNotReadyProbeIsNotLoggedAsError(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(prev)

	h := newHarness(t, 1, nil)
	for _, b := range h.reg.Snapshot().Backends {
		b.SetHealth(registry.Unhealthy) // nothing healthy: readiness must 503
	}

	resp, err := h.srv.Client().Get(h.srv.URL + "/readiness")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("readiness = %d, want 503", resp.StatusCode)
	}
	if strings.Contains(buf.String(), `"level":"ERROR"`) {
		t.Errorf("a not-ready readiness probe was logged at ERROR:\n%s", buf.String())
	}

	// But a 503 from the inference path is a genuine error and must stay one.
	buf.Reset()
	r2 := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	r2.Body.Close()
	if !strings.Contains(buf.String(), `"level":"ERROR"`) {
		t.Errorf("an inference 503 should still log at ERROR:\n%s", buf.String())
	}
}

// TestClientDisconnectIsNotCountedAsPanic is a regression test for a bug found by
// running a real benchmark against the cluster.
//
// httputil.ReverseProxy panics with http.ErrAbortHandler when it cannot finish
// copying a response — i.e. the client hung up mid-stream. That is net/http's
// sentinel for "abort this response", not a fault. Treating it as a panic logged
// ERROR for a routine disconnect and drove router_panics_total to 16 when a load
// driver hit its timeout and cancelled its in-flight streams, which turns an alarm
// metric into background noise.
func TestClientDisconnectIsNotCountedAsPanic(t *testing.T) {
	h := newHarness(t, 1, nil)

	panicsBefore := testutil.ToFloat64(metrics.PanicsTotal)
	discBefore := testutil.ToFloat64(metrics.ClientDisconnects)

	// A worker that closes the connection abruptly mid-response makes
	// ReverseProxy raise http.ErrAbortHandler.
	h.workers[0].SetScript(mockvllm.Script{Stream: true, Tokens: 5, Hijack: true})
	resp, err := h.srv.Client().Post(h.srv.URL+"/v1/chat/completions",
		"application/json", strings.NewReader(`{"model":"m","stream":true}`))
	if err == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}

	if got := testutil.ToFloat64(metrics.PanicsTotal); got != panicsBefore {
		t.Errorf("router_panics_total went %v -> %v: a client disconnect must not "+
			"count as a panic, or the alarm metric fires on routine hang-ups",
			panicsBefore, got)
	}
	// The disconnect path may or may not trigger depending on timing; if it did,
	// it must have been counted as a disconnect rather than a panic.
	if d := testutil.ToFloat64(metrics.ClientDisconnects); d > discBefore {
		t.Logf("counted %v client disconnect(s), as intended", d-discBefore)
	}
	h.assertNoLeaks(t)
}

// A genuine panic must still be caught, counted, and turned into a 500.
func TestRealPanicStillBecomes500(t *testing.T) {
	h := newHarness(t, 1, nil)
	before := testutil.ToFloat64(metrics.PanicsTotal)

	// /v1/models is a GET route; drive a panic through the recover boundary by
	// asking the dialect to render into a hijacked writer is awkward, so instead
	// assert the boundary directly on a handler that panics.
	mux := http.NewServeMux()
	mux.HandleFunc("GET /boom", func(http.ResponseWriter, *http.Request) {
		panic("deliberate test panic")
	})
	srv := httptest.NewServer(h.gw.RecoverForTest(mux))
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL + "/boom")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("panic gave status %d, want 500", resp.StatusCode)
	}
	if got := testutil.ToFloat64(metrics.PanicsTotal); got != before+1 {
		t.Errorf("router_panics_total went %v -> %v, want +1 for a real panic", before, got)
	}
}

// End-to-end cache affinity: the same conversation must keep landing on the same
// backend, and a request must only be committed to a backend that actually
// accepted it.
func TestPrefixCacheAffinityEndToEnd(t *testing.T) {
	lease.ResetAccountingErrors()
	cfg := defaultHarnessConfig()
	cfg.MaxAttempts = 2

	reg := registry.New(registry.Options{})
	cp := mustFlow(t, affinity.Config{})
	var workers []*mockvllm.Worker
	for i := 0; i < 4; i++ {
		w := mockvllm.New()
		t.Cleanup(w.Close)
		workers = append(workers, w)
		b, err := reg.Add(registry.Spec{URL: w.URL(), Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		cp.AddBackend(b)
	}
	d := openai.New()
	px := proxy.New(proxy.Config{MaxAttempts: cfg.MaxAttempts, StreamBufferBytes: cfg.StreamBufferBytes})
	srv := httptest.NewServer(gateway.New(cfg.gateway(), mustTable(t, reg, cp), px, d))
	defer srv.Close()

	// One long shared system prompt plus a per-turn suffix — the shape of a real
	// multi-turn conversation.
	system := strings.Repeat("you are a helpful assistant. ", 200)
	post := func(turn int) {
		body := fmt.Sprintf(
			`{"model":"m","messages":[{"role":"system","content":%q},{"role":"user","content":"turn %d"}]}`,
			system, turn)
		resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("turn %d: status %d", turn, resp.StatusCode)
		}
	}

	post(0) // first turn lands somewhere and warms that backend
	before := make([]int, len(workers))
	for i, w := range workers {
		before[i] = w.CallCount()
	}
	for turn := 1; turn <= 30; turn++ {
		post(turn)
	}

	// Every subsequent turn shares the system prompt, so affinity should pin them
	// to whichever backend served the first.
	var moved int
	for i, w := range workers {
		got := w.CallCount() - before[i]
		if got > 0 && got < 30 {
			moved++
		}
	}
	var maxShare int
	for i, w := range workers {
		if got := w.CallCount() - before[i]; got > maxShare {
			maxShare = got
		}
	}
	if maxShare < 25 {
		t.Errorf("the busiest backend served only %d of 30 shared-prefix turns; "+
			"affinity is not holding", maxShare)
	}
	if n := lease.AccountingErrors(); n != 0 {
		t.Errorf("load accounting errors = %d", n)
	}
}

// R3: a backend that never accepted the request must not be credited with its
// prefix. Committing at selection time would poison it permanently.
func TestFailedAttemptDoesNotCommit(t *testing.T) {
	cfg := defaultHarnessConfig()
	cfg.MaxAttempts = 2

	reg := registry.New(registry.Options{})
	cp := mustFlow(t, affinity.Config{})
	var workers []*mockvllm.Worker
	for i := 0; i < 2; i++ {
		w := mockvllm.New()
		t.Cleanup(w.Close)
		workers = append(workers, w)
		b, _ := reg.Add(registry.Spec{URL: w.URL(), Capacity: 1})
		b.SetHealth(registry.Healthy)
		cp.AddBackend(b)
	}
	// Both refuse with a retryable status, so nothing is ever accepted.
	for _, w := range workers {
		w.SetScript(mockvllm.Script{Status: http.StatusServiceUnavailable, Body: `{}`})
	}

	d := openai.New()
	px := proxy.New(proxy.Config{MaxAttempts: cfg.MaxAttempts, StreamBufferBytes: cfg.StreamBufferBytes})
	srv := httptest.NewServer(gateway.New(cfg.gateway(), mustTable(t, reg, cp), px, d))
	defer srv.Close()

	resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	for _, bm := range cp.Snapshot(viz.SnapshotOptions{}).Backends {
		if bm.Nodes != 0 {
			t.Errorf("%s was credited with %d blocks despite never accepting the "+
				"request — it will look warm for a prefix it never received", bm.URL, bm.Nodes)
		}
	}
}

// REL-10: the concurrency cap is what actually bounds memory. Each in-flight
// request can hold up to max_body_bytes buffered for retry, so at the shipped
// 64 MiB limit roughly eight concurrent uploads reach a 1 GiB container limit.
// The chain documented this at position 7 and it did not exist.
func TestConcurrencyCapShedsWith503(t *testing.T) {
	release := make(chan struct{})
	h := newHarness(t, 1, func(c *harnessConfig) { c.MaxConcurrentRequests = 2 })
	h.workers[0].Behave(func(*http.Request) mockvllm.Script {
		<-release // hold the slot until the test lets go
		return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
	})

	codes := make(chan int, 6)
	for i := 0; i < 6; i++ {
		go func() {
			resp, err := h.srv.Client().Post(h.srv.URL+"/v1/chat/completions",
				"application/json", strings.NewReader(`{"model":"m"}`))
			if err != nil {
				codes <- 0
				return
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			codes <- resp.StatusCode
		}()
	}

	// Wait for the cap to bite: some requests must be shed while 2 are parked.
	shed := 0
	deadline := time.After(10 * time.Second)
	for shed == 0 {
		select {
		case c := <-codes:
			if c == http.StatusServiceUnavailable {
				shed++
			} else if c == 0 {
				t.Fatal("request error")
			}
		case <-deadline:
			close(release)
			t.Fatal("no request was shed; the concurrency cap is not enforced")
		}
	}
	close(release)
	if shed == 0 {
		t.Error("expected at least one 503 from the concurrency cap")
	}
	t.Logf("shed %d of 6 with a cap of 2", shed)
}

// RES-3: the closed loop on prediction. Without this the observed fraction is
// never emitted and there is no way to tell whether the router's cache guesses
// correspond to anything the worker actually did.
func TestObservedCacheFractionIsRecorded(t *testing.T) {
	h := newHarness(t, 1, nil)
	before := testutil.CollectAndCount(metrics.CacheObservedFraction)
	_ = before

	h.workers[0].SetScript(mockvllm.Script{PromptTokens: 1000, CachedTokens: 750})
	resp := h.post(t, "/v1/chat/completions", `{"model":"m","messages":[]}`, nil)
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	got := testutil.ToFloat64(metrics.CacheObservedFractionSumForTest())
	if got <= 0 {
		t.Errorf("router_cache_observed_fraction was never observed; " +
			"prediction accuracy remains unmeasurable")
	}
}

// TestMaxNodeConcurrencyRejects429WhenAllBackendsAtCap is anton's exact
// scenario for --max-node-concurrency: with cap=1 and 2 backends, two
// concurrent requests each land on a DIFFERENT backend (both are under cap
// the instant each is admitted), a third request while both are held gets
// 429 with Retry-After — distinguishable from the existing 503s: NOT
// no_healthy_backends (every backend is healthy) and NOT the router-wide
// MaxConcurrentRequests shed — and once one backend's request completes, the
// next request is admitted again.
func TestMaxNodeConcurrencyRejects429WhenAllBackendsAtCap(t *testing.T) {
	h := newHarness(t, 2, func(c *harnessConfig) { c.MaxNodeConcurrency = 1 })

	// Which backend a fresh, idle-tied request lands on is an intentionally
	// unspecified 50/50 tie-break (policy.pickBest's reservoir sample over
	// least-outstanding), so this test tracks "first"/"other" dynamically
	// rather than assuming index 0 goes first.
	release := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
	entered := make(chan int, 2)
	for i, w := range h.workers {
		i := i
		w.Behave(func(*http.Request) mockvllm.Script {
			entered <- i
			<-release[i]
			return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
		})
	}

	shedBefore := testutil.ToFloat64(metrics.SaturationRejects)

	// Admit the two "held" requests SEQUENTIALLY. Once the first is admitted,
	// its backend is immediately AT cap, so the SECOND request is
	// deterministically forced onto the OTHER backend by the cap filter
	// itself — which is the exact mechanism under test, not an assumption
	// about load-balance luck.
	done := [2]chan *http.Response{make(chan *http.Response, 1), make(chan *http.Response, 1)}
	go func() { done[0] <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	first := <-entered

	go func() { done[1] <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	second := <-entered
	if second == first {
		t.Fatalf("expected the second held request to land on the OTHER backend (backend %d is "+
			"now at cap=1), but it landed on the same backend again", first)
	}
	other := second

	bFirst, ok := h.reg.Snapshot().Get(h.workers[first].URL())
	if !ok {
		t.Fatal("backend not found in registry")
	}
	bOther, ok := h.reg.Snapshot().Get(h.workers[other].URL())
	if !ok {
		t.Fatal("backend not found in registry")
	}
	capBeforeFirst := testutil.ToFloat64(metrics.BackendCapExceeded.WithLabelValues(bFirst.URL))
	capBeforeOther := testutil.ToFloat64(metrics.BackendCapExceeded.WithLabelValues(bOther.URL))

	// A third request now finds BOTH backends at cap: 429, not 503.
	resp3 := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	body3, _ := io.ReadAll(resp3.Body)
	resp3.Body.Close()
	if resp3.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429 (body: %s)", resp3.StatusCode, body3)
	}
	if got := resp3.Header.Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want \"1\"", got)
	}
	var env struct {
		Error struct{ Code string } `json:"error"`
	}
	_ = json.Unmarshal(body3, &env)
	// Every capacity rejection looks the same to a caller. Which mechanism
	// declined — the split guard, fleet-wide saturation, the router's own cap —
	// is internal bookkeeping, and a caller can act on exactly one thing
	// regardless: back off and retry. The distinction lives in the log and in
	// router_saturation_rejects_total, asserted just below.
	if env.Error.Code != "rate_limit_error" {
		t.Errorf("error code = %q, want rate_limit_error; capacity rejections must not "+
			"publish which internal mechanism declined", env.Error.Code)
	}
	if strings.Contains(string(body3), "prefix") || strings.Contains(string(body3), "backend") {
		t.Errorf("429 body describes the router's internals to the caller: %s", body3)
	}
	if got := testutil.ToFloat64(metrics.SaturationRejects); got != shedBefore+1 {
		t.Errorf("router_saturation_rejects_total went %v -> %v, want +1", shedBefore, got)
	}
	if got := testutil.ToFloat64(metrics.BackendCapExceeded.WithLabelValues(bFirst.URL)); got != capBeforeFirst+1 {
		t.Errorf("router_backend_cap_exceeded_total{backend=%s} went %v -> %v, want +1", bFirst.URL, capBeforeFirst, got)
	}
	if got := testutil.ToFloat64(metrics.BackendCapExceeded.WithLabelValues(bOther.URL)); got != capBeforeOther+1 {
		t.Errorf("router_backend_cap_exceeded_total{backend=%s} went %v -> %v, want +1", bOther.URL, capBeforeOther, got)
	}

	// Release the first held request; its lease frees up.
	close(release[first])
	r0 := <-done[0]
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	if r0.StatusCode != http.StatusOK {
		t.Fatalf("held request 0 status = %d, want 200", r0.StatusCode)
	}

	// The lease releases in a defer after ServeHTTP returns, which races the
	// client goroutine reading the response — poll rather than assume.
	deadline := time.Now().Add(2 * time.Second)
	for bFirst.Inflight() != 0 {
		if time.Now().After(deadline) {
			t.Fatal("the released backend's lease never cleared after its request completed")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// A new request must be admitted again, landing on the now-freed backend
	// (the only one under cap). It gets a fresh Behave call, which hangs on
	// the already-closed release channel and returns immediately.
	resp4 := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	io.Copy(io.Discard, resp4.Body)
	resp4.Body.Close()
	if resp4.StatusCode != http.StatusOK {
		t.Fatalf("status after a slot freed = %d, want 200 (admission should resume)", resp4.StatusCode)
	}

	close(release[other])
	r1 := <-done[1]
	io.Copy(io.Discard, r1.Body)
	r1.Body.Close()

	h.assertNoLeaks(t)
}

// TestMaxNodeConcurrencyExcludesAtCapBackendFromCacheAffinity is the
// affinity-path half of "an at-cap backend is excluded from candidates":
// the cache-aware policy would confidently pick the warm backend by
// predicted-hit fraction, but once that backend is at its
// --max-node-concurrency cap it must never appear in the candidate set
// Select is even given, so the request falls to the cold backend instead.
func TestMaxNodeConcurrencyExcludesAtCapBackendFromCacheAffinity(t *testing.T) {
	lease.ResetAccountingErrors()
	cfg := defaultHarnessConfig()
	cfg.MaxAttempts = 1
	cfg.MaxNodeConcurrency = 1

	reg := registry.New(registry.Options{})
	cp := mustFlow(t, affinity.Config{NodeConcurrency: cfg.MaxNodeConcurrency})
	var workers []*mockvllm.Worker
	for i := 0; i < 2; i++ {
		w := mockvllm.New()
		t.Cleanup(w.Close)
		workers = append(workers, w)
		b, err := reg.Add(registry.Spec{URL: w.URL(), Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		cp.AddBackend(b)
	}
	d := openai.New()
	px := proxy.New(proxy.Config{MaxAttempts: cfg.MaxAttempts, StreamBufferBytes: cfg.StreamBufferBytes})
	srv := httptest.NewServer(gateway.New(cfg.gateway(), mustTable(t, reg, cp), px, d))
	defer srv.Close()

	system := strings.Repeat("you are a helpful assistant. ", 200)
	post := func(user string) *http.Response {
		body := fmt.Sprintf(`{"model":"m","messages":[{"role":"system","content":%q},{"role":"user","content":%q}]}`,
			system, user)
		resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// Warm one backend via a real, completed request sharing the system
	// prompt.
	r0 := post("turn 0")
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	warm := 0
	if workers[0].CallCount() == 0 {
		warm = 1
	}
	cold := 1 - warm

	// Hold the WARM backend at its cap=1 with a second request that shares
	// the same system prompt — affinity should (and, per the base
	// TestPrefixCacheAffinityEndToEnd, does) route it there too.
	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	workers[warm].Behave(func(*http.Request) mockvllm.Script {
		entered <- struct{}{}
		<-release
		return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
	})
	go func() {
		r := post("occupy the warm backend's only slot")
		io.Copy(io.Discard, r.Body)
		r.Body.Close()
	}()
	<-entered // warm backend now at Inflight()==1==cap

	beforeCold := workers[cold].CallCount()
	r2 := post("turn 1, sharing the same system prompt")
	body2, _ := io.ReadAll(r2.Body)
	r2.Body.Close()
	if r2.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", r2.StatusCode, body2)
	}
	if got := workers[cold].CallCount(); got != beforeCold+1 {
		t.Errorf("expected the warm-but-at-cap backend to be excluded from affinity selection so "+
			"the request fell to the cold backend: cold backend call count = %d, want %d", got, beforeCold+1)
	}

	close(release)
	if n := lease.AccountingErrors(); n != 0 {
		t.Errorf("load accounting errors = %d", n)
	}
}

// TestMaxNodeConcurrencyExcludesAtCapBackendFromFallback is the
// fallback-path half: a request with no routable prefix at all makes the
// cache-aware policy decline immediately and delegate to its fallback
// (least-outstanding) — exercising the OTHER branch through which a
// candidate could reach a backend — and the at-cap backend must be excluded
// there too, not just in the affinity scorer.
func TestMaxNodeConcurrencyExcludesAtCapBackendFromFallback(t *testing.T) {
	lease.ResetAccountingErrors()
	cfg := defaultHarnessConfig()
	cfg.MaxAttempts = 1
	cfg.MaxNodeConcurrency = 1

	reg := registry.New(registry.Options{})
	cp := mustFlow(t, affinity.Config{NodeConcurrency: cfg.MaxNodeConcurrency})
	var workers []*mockvllm.Worker
	for i := 0; i < 2; i++ {
		w := mockvllm.New()
		t.Cleanup(w.Close)
		workers = append(workers, w)
		b, err := reg.Add(registry.Spec{URL: w.URL(), Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		cp.AddBackend(b)
	}

	// Both workers start with a blocking Behave so whichever one the FIRST
	// request lands on can be held — the tie-break among two idle backends is
	// intentionally random (see the note in the ...RejectsWith429... test),
	// so which index that is isn't known yet.
	release := make(chan struct{})
	entered := make(chan int, 2)
	for i, w := range workers {
		i := i
		w.Behave(func(*http.Request) mockvllm.Script {
			entered <- i
			<-release
			return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
		})
	}

	d := openai.New()
	px := proxy.New(proxy.Config{MaxAttempts: cfg.MaxAttempts, StreamBufferBytes: cfg.StreamBufferBytes})
	srv := httptest.NewServer(gateway.New(cfg.gateway(), mustTable(t, reg, cp), px, d))
	defer srv.Close()

	post := func() *http.Response {
		resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions",
			"application/json", strings.NewReader(`{"model":"m"}`))
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// No messages/prompt/input at all -> ExtractUnits declines -> the cache
	// policy's len(rr.Units)==0 branch fires immediately, so this request is
	// decided by the FALLBACK (least-outstanding), never the affinity
	// scorer.
	go func() {
		resp := post()
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	held := <-entered // which backend the fallback picked, now at cap
	other := 1 - held

	heldBackend, ok := reg.Snapshot().Get(workers[held].URL())
	if !ok {
		t.Fatal("held backend not found in registry")
	}

	// The OTHER backend now answers normally: it must NOT share the held
	// backend's release gate, or the assertion below would block forever
	// waiting on a channel this test never closes for it.
	workers[other].Behave(func(*http.Request) mockvllm.Script {
		return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
	})

	beforeOther := workers[other].CallCount()
	resp2 := post()
	body2, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", resp2.StatusCode, body2)
	}
	if got := workers[other].CallCount(); got != beforeOther+1 {
		t.Errorf("expected the at-cap backend to be excluded from the FALLBACK path so the "+
			"request fell to the other backend: other backend call count = %d, want %d", got, beforeOther+1)
	}
	if got := heldBackend.Inflight(); got != 1 {
		t.Errorf("held backend Inflight = %d, want 1 (unchanged) — a second request must "+
			"not have been routed to an already-at-cap backend via the fallback path", got)
	}

	close(release)
	if n := lease.AccountingErrors(); n != 0 {
		t.Errorf("load accounting errors = %d", n)
	}
}

// TestProbePathsNeverReachABackend covers the failure that CrashLoopBackOffed a
// deployed router: probes on the kubernetes-convention paths were proxied
// upstream, because any path the dialect does not claim is passed through.
//
// A probe answered by a backend is not a probe of the router, and the backend's
// 404 reads as the router being unhealthy.
func TestProbePathsNeverReachABackend(t *testing.T) {
	h := newHarness(t, 1, nil)
	for _, path := range []string{"/readiness", "/liveness", "/healthz", "/livez", "/health"} {
		resp, err := http.Get(h.srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == http.StatusNotFound {
			t.Errorf("GET %s = 404; a probe path must be answered by the router, "+
				"not proxied to a backend that knows nothing about it", path)
		}
		if strings.Contains(string(body), "chat.completion") {
			t.Errorf("GET %s was answered by a backend", path)
		}
	}
}
