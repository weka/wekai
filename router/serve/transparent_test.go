package serve_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/weka/wekai/router/serve"
)

// hostedUpstream stands in for a provider: it answers whatever it is told to and
// counts the requests it actually received, which is how "no retry" is proved.
type hostedUpstream struct {
	srv    *httptest.Server
	hits   atomic.Int64
	status atomic.Int64
}

func newHostedUpstream(t *testing.T) *hostedUpstream {
	t.Helper()
	u := &hostedUpstream{}
	u.status.Store(http.StatusOK)
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A real hosted API answers none of the router's probes.
		switch r.URL.Path {
		case "/metrics", "/health", "/v1/models":
			w.WriteHeader(http.StatusNotFound)
			return
		}
		u.hits.Add(1)
		code := int(u.status.Load())
		// Headers a provider sends with a refusal and a caller acts on. These
		// are the substance of "relayed verbatim" — a synthesized 429 has none.
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "17")
		w.Header().Set("Anthropic-Ratelimit-Requests-Remaining", "0")
		w.WriteHeader(code)
		fmt.Fprintf(w, `{"type":"error","error":{"type":"overloaded_error","message":"upstream said %d"}}`, code)
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func startTransparentRouter(t *testing.T, endpoint string) *httptest.Server {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	h, err := serve.Handler(ctx, serve.Options{
		HealthInterval: 20 * time.Millisecond,
		HealthTimeout:  10 * time.Millisecond,
		Routes:         []serve.Route{{Name: "hosted", Patterns: "*", Endpoints: []string{endpoint}}},
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)
	return ts
}

func call(t *testing.T, ts *httptest.Server, path string) (int, string, http.Header) {
	t.Helper()
	resp, err := ts.Client().Post(ts.URL+path, "application/json",
		strings.NewReader(`{"model":"claude-sonnet-4-5","max_tokens":4,"messages":[{"role":"user","content":"x"}]}`))
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), resp.Header
}

// The provider's answer reaches the caller intact — status, body and the headers
// a client acts on. Anything the router substitutes here is information the
// caller needed and no longer has.
func TestTransparentRelaysTheUpstreamAnswerVerbatim(t *testing.T) {
	up := newHostedUpstream(t)
	rt := startTransparentRouter(t, up.srv.URL)

	for _, status := range []int{http.StatusOK, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusServiceUnavailable,
		http.StatusBadRequest, http.StatusUnauthorized} {
		up.status.Store(int64(status))
		code, body, hdr := call(t, rt, "/v1/messages")
		if code != status {
			t.Errorf("upstream answered %d, caller saw %d", status, code)
		}
		if !strings.Contains(body, fmt.Sprintf("upstream said %d", status)) {
			t.Errorf("status %d: caller saw body %q, not the upstream's", status, body)
		}
		if status == http.StatusTooManyRequests {
			if hdr.Get("Retry-After") != "17" {
				t.Errorf("Retry-After %q; the provider's own backoff must survive",
					hdr.Get("Retry-After"))
			}
			if hdr.Get("Anthropic-Ratelimit-Requests-Remaining") != "0" {
				t.Error("the provider's rate-limit headers were dropped")
			}
		}
	}
}

// The headline property. A provider having a bad hour must never cause the router
// to start answering on its behalf: no breaker opens, so no request is ever
// refused with a status the upstream did not send.
func TestTransparentNeverOpensACircuitBreaker(t *testing.T) {
	up := newHostedUpstream(t)
	rt := startTransparentRouter(t, up.srv.URL)

	// Comfortably past the breaker's floor: it opens at 20 requests in a 30s
	// window with half failing, and this is 40 consecutive failures.
	const n = 40
	up.status.Store(http.StatusInternalServerError)
	for i := range n {
		code, body, _ := call(t, rt, "/v1/messages")
		if code != http.StatusInternalServerError {
			t.Fatalf("request %d: got %d, want the upstream's 500 — the router "+
				"started answering for the provider", i+1, code)
		}
		if strings.Contains(body, "no_healthy_backends") {
			t.Fatalf("request %d: the breaker opened and the router refused on the "+
				"provider's behalf", i+1)
		}
	}
	// Exactly one upstream request per client request: no retry, so a metered
	// API is never billed twice for one call.
	if got := up.hits.Load(); got != n {
		t.Errorf("upstream received %d requests for %d client requests; a "+
			"transparent route must not retry", got, n)
	}

	// And it recovers the instant the provider does — there is no timer to wait
	// out, because nothing was ever latched.
	up.status.Store(http.StatusOK)
	if code, _, _ := call(t, rt, "/v1/messages"); code != http.StatusOK {
		t.Errorf("got %d after the upstream recovered, want 200 immediately", code)
	}
}

// When there is no response to relay, the router has to speak for itself. 502 is
// the one honest thing it can say: this gateway could not reach the upstream.
func TestTransparentAnswers502WhenTheUpstreamIsUnreachable(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	addr := dead.URL
	dead.Close() // nothing is listening there now

	rt := startTransparentRouter(t, addr)
	code, body, _ := call(t, rt, "/v1/messages")
	if code != http.StatusBadGateway {
		t.Errorf("got %d, want 502 when the upstream cannot be reached", code)
	}
	if !strings.Contains(body, "upstream_unreachable") {
		t.Errorf("body %q does not say what went wrong", body)
	}
}

// The single-endpoint condition is load-bearing. Two endpoints are a fleet the
// router picks between, and there it must keep the breaker: passive health never
// changes on its own, so without it a dead endpoint stays in rotation forever.
func TestTwoUnmanagedEndpointsStayManaged(t *testing.T) {
	a, b := newHostedUpstream(t), newHostedUpstream(t)
	a.status.Store(http.StatusInternalServerError)
	b.status.Store(http.StatusInternalServerError)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		HealthInterval: 20 * time.Millisecond,
		HealthTimeout:  10 * time.Millisecond,
		Routes: []serve.Route{{Name: "pair", Patterns: "*",
			Endpoints: []string{a.srv.URL, b.srv.URL}}},
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	refused := false
	for range 60 {
		if code, body, _ := call(t, rt, "/v1/messages"); code == http.StatusServiceUnavailable &&
			strings.Contains(body, "no_healthy_backends") {
			refused = true
			break
		}
	}
	if !refused {
		t.Error("a two-endpoint pool never opened a breaker; a dead endpoint would " +
			"stay in rotation permanently, since passive health never changes")
	}
}
