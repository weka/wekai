package serve_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/serve"
)

// Two routers, which is the deployment this feature exists for.
//
//	inner  — protected by an API key, fronting the internal vLLM fleet.
//	outer  — user facing, no inbound key. It holds a MOUNTED secret that is
//	         inner's key, and a default route to a hosted API where it forwards
//	         the USER's key instead.
//
// The property that matters is the one a leak would violate: a user's
// credential must reach the hosted API and must never reach the internal
// fleet, and inner's key must never be visible to a user.
func TestTwoRoutersUserKeyOutRouterKeyIn(t *testing.T) {
	const innerKey = "inner-secret-key"
	const userKey = "user-personal-key"

	fleet, servers := mockFleet(t, 1, mockvllm.SurfaceVLLM)

	// Inner: authenticated, internal only.
	innerCtx, cancelInner := context.WithCancel(context.Background())
	defer cancelInner()
	innerH, err := serve.Handler(innerCtx, serve.Options{
		Routes:          []serve.Route{{Patterns: "*", Endpoints: fleet}},
		APIKey:          innerKey,
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 20 * time.Millisecond,
		HealthTimeout:   100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("inner Handler: %v", err)
	}
	inner := httptest.NewServer(innerH)
	defer inner.Close()

	// A stand-in for the hosted API, recording what credential arrived.
	var hostedSawAuth, hostedSawKey string
	var hostedHits int
	hosted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/health", "/metrics":
			w.WriteHeader(http.StatusNotFound)
			return
		}
		hostedHits++
		hostedSawAuth = r.Header.Get("Authorization")
		hostedSawKey = r.Header.Get("X-Api-Key")
		fmt.Fprint(w, `{"id":"msg_1","type":"message","content":[{"type":"text","text":"ok"}]}`)
	}))
	defer hosted.Close()

	// The secret as it is actually delivered: a mounted file, never a flag.
	dir := t.TempDir()
	secret := filepath.Join(dir, "inner-key")
	if err := os.WriteFile(secret, []byte(innerKey+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	outerCtx, cancelOuter := context.WithCancel(context.Background())
	defer cancelOuter()
	outerH, err := serve.Handler(outerCtx, serve.Options{
		Routes: []serve.Route{
			{Patterns: "local", Endpoints: []string{inner.URL},
				CredentialFile: secret, Name: "internal"},
			// Forwarding is opt-in: the user pays for this call, so their key is
			// the only one that can work.
			{Patterns: "*", Endpoints: []string{hosted.URL}, Passive: true, Name: "hosted",
				ForwardClientCredential: true},
		},
		HealthInterval:  200 * time.Millisecond,
		HealthUnhealthy: 20 * time.Millisecond,
		HealthTimeout:   100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("outer Handler: %v", err)
	}
	outer := httptest.NewServer(outerH)
	defer outer.Close()

	post := func(model, path string) int {
		req, _ := http.NewRequest(http.MethodPost, outer.URL+path,
			strings.NewReader(fmt.Sprintf(
				`{"model":%q,"max_tokens":4,"messages":[{"role":"user","content":"hi"}]}`, model)))
		req.Header.Set("Content-Type", "application/json")
		// The user's own credential, as a client would send it.
		req.Header.Set("Authorization", "Bearer "+userKey)
		resp, err := outer.Client().Do(req)
		if err != nil {
			t.Fatalf("post %s: %v", model, err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	// Internal: the outer router substitutes its mounted key, so inner's auth
	// accepts a request whose user never knew that key.
	eventually(t, func() bool { return post("local-model", "/v1/chat/completions") == http.StatusOK })
	if servers[0].Engine().Stats().Admitted == 0 {
		t.Fatal("the internal fleet served nothing; the router's own key did not satisfy inner's auth")
	}

	// Hosted: the USER's credential is what arrives, not the router's.
	if code := post("claude-sonnet", "/v1/messages"); code != http.StatusOK {
		t.Fatalf("hosted route status %d", code)
	}
	if hostedHits == 0 {
		t.Fatal("the hosted API was never reached")
	}
	if !strings.Contains(hostedSawAuth, userKey) {
		t.Errorf("hosted API saw Authorization %q; the caller's own credential must be "+
			"forwarded on a route that does not carry the router's", redact(hostedSawAuth, innerKey))
	}
	if strings.Contains(hostedSawAuth, innerKey) || strings.Contains(hostedSawKey, innerKey) {
		t.Error("the router's internal key LEAKED to the hosted API; a mounted secret must " +
			"only be used on the pool it was configured for")
	}
}

// TestRouterCredentialReplacesTheCallersRatherThanAddingToIt: an internal
// service must not receive a user's personal credential alongside the router's.
// Forwarding both is the leak this feature exists to prevent, in the direction
// that is easy to miss.
func TestRouterCredentialReplacesTheCallersRatherThanAddingToIt(t *testing.T) {
	const routerKey = "router-owned"
	const userKey = "user-owned"

	var sawAuth, sawKey string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models", "/health", "/metrics":
			w.WriteHeader(http.StatusNotFound)
			return
		}
		sawAuth, sawKey = r.Header.Get("Authorization"), r.Header.Get("X-Api-Key")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer upstream.Close()

	dir := t.TempDir()
	secret := filepath.Join(dir, "key")
	if err := os.WriteFile(secret, []byte(routerKey), 0o600); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h, err := serve.Handler(ctx, serve.Options{
		Routes: []serve.Route{{Patterns: "*", Endpoints: []string{upstream.URL},
			CredentialFile: secret, Passive: true}},
	})
	if err != nil {
		t.Fatalf("Handler: %v", err)
	}
	rt := httptest.NewServer(h)
	defer rt.Close()

	req, _ := http.NewRequest(http.MethodPost, rt.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"m","max_tokens":4,"messages":[]}`))
	req.Header.Set("Authorization", "Bearer "+userKey)
	req.Header.Set("X-Api-Key", userKey)
	resp, err := rt.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	if strings.Contains(sawAuth, userKey) || strings.Contains(sawKey, userKey) {
		t.Error("the caller's credential reached an upstream the router authenticates to " +
			"itself; it must be replaced, not merged")
	}
	if !strings.Contains(sawAuth, routerKey) && !strings.Contains(sawKey, routerKey) {
		t.Errorf("the router's own credential did not arrive: auth=%q key=%q", sawAuth, sawKey)
	}
	// A trailing newline in a mounted secret is normal and must not be sent.
	if strings.Contains(sawAuth, "\n") || strings.Contains(sawKey, "\n") {
		t.Error("credential sent with a trailing newline; a secret file usually has one")
	}
}

func redact(s, secret string) string { return strings.ReplaceAll(s, secret, "<REDACTED>") }
