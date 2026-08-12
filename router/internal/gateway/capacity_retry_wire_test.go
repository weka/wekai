package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/dialect/openai"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

// The retry budget through the real handler: a capacity refusal is waited out
// and RE-DECIDED, a transport failure is not, and the budget is honoured.
//
// In-package because the seam that matters is Target.Selector, and a Target
// built from a routing.Rule always carries the pool's own affinity flow. Going
// through a stub Router lets a selector refuse on demand, which is what makes
// "the fleet freed a slot while we waited" expressible at all.

type stubRouter struct{ target Target }

func (s stubRouter) Route(string) (Target, bool) { return s.target, true }
func (s stubRouter) Targets() []Target           { return []Target{s.target} }

// refusingSelector fails the first n selections with a capacity error, then
// behaves — a fleet that frees a slot while we wait, which is the premise.
type refusingSelector struct {
	err       error
	remaining atomic.Int32
	calls     atomic.Int32
}

func (s *refusingSelector) Name() string { return "refusing" }

func (s *refusingSelector) Select(_ context.Context, cands []*registry.Backend, _ *policy.RoutingRequest) (*registry.Backend, error) {
	s.calls.Add(1)
	if s.remaining.Add(-1) >= 0 {
		return nil, s.err
	}
	return cands[0], nil
}

const chatBody = `{"model":"m","messages":[{"role":"user","content":"hi"}]}`

func retryHarness(t *testing.T, sel proxy.Selector, limit time.Duration, upstream http.HandlerFunc) (*httptest.Server, *clock.Fake) {
	t.Helper()
	if upstream == nil {
		upstream = func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","choices":[]}`))
		}
	}
	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)

	reg := registry.New(registry.Options{})
	b, err := reg.Add(registry.Spec{URL: up.URL, Prov: registry.ProvStatic, Capacity: 4})
	if err != nil {
		t.Fatal(err)
	}
	b.SetHealth(registry.Healthy)

	clk := clock.NewFake(time.Time{})
	gw := New(Config{
		MaxBodyBytes: 64 << 20, DefaultCapacity: 1,
		RetryTimeLimit: limit, Clock: clk,
	}, stubRouter{Target{Name: "default", Registry: reg, Selector: sel}},
		proxy.New(proxy.Config{MaxAttempts: 2, StreamBufferBytes: 64 << 10}), openai.New())

	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	return srv, clk
}

// drive advances the fake clock while the handler is parked in clock.After.
// The handler runs on another goroutine, so this has to keep ticking until the
// request is answered.
func drive(clk *clock.Fake, done <-chan struct{}) {
	for {
		select {
		case <-done:
			return
		default:
			clk.Advance(time.Second)
			//clockexempt: yields to the handler goroutine; not a timing decision
			time.Sleep(time.Millisecond)
		}
	}
}

func TestCapacityRefusalIsWaitedOutAndReDecided(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"every backend saturated", policy.ErrAllBackendsSaturated},
		{"split guard blocked", policy.ErrSplitGuardBlocked},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sel := &refusingSelector{err: tc.err}
			sel.remaining.Store(3) // frees up on the fourth decision
			srv, clk := retryHarness(t, sel, 10*time.Second, nil)

			done := make(chan struct{})
			go drive(clk, done)
			resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
			close(done)
			if err != nil {
				t.Fatalf("post: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				t.Errorf("status %d, want 200: the refusal describes one instant, and the fleet "+
					"freed a slot inside the budget", resp.StatusCode)
			}
			if n := sel.calls.Load(); n < 4 {
				t.Errorf("the selector was asked %d times, want at least 4 — each wait must be "+
					"followed by a fresh decision, not a replay of the old one", n)
			}
		})
	}
}

func TestCapacityRefusalGivesUpAtTheLimit(t *testing.T) {
	sel := &refusingSelector{err: policy.ErrAllBackendsSaturated}
	sel.remaining.Store(1 << 30) // never recovers
	srv, clk := retryHarness(t, sel, 2*time.Second, nil)

	done := make(chan struct{})
	go drive(clk, done)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
	close(done)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status %d, want 429 once the budget is spent", resp.StatusCode)
	}
}

func TestRetryLimitOffAnswersImmediately(t *testing.T) {
	sel := &refusingSelector{err: policy.ErrAllBackendsSaturated}
	sel.remaining.Store(1 << 30)
	srv, _ := retryHarness(t, sel, 0, nil) // the shipped default

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status %d, want 429", resp.StatusCode)
	}
	if n := sel.calls.Load(); n != 1 {
		t.Errorf("the selector was asked %d times with the budget at 0, want exactly 1", n)
	}
}

// TestFailuresAreNotWaitedOut: the budget covers capacity, not breakage. A
// backend returning 502 is the proxy's problem under its own tighter rules, and
// waiting on it would only delay an error the caller has to handle anyway.
func TestFailuresAreNotWaitedOut(t *testing.T) {
	sel := &refusingSelector{err: policy.ErrAllBackendsSaturated}
	sel.remaining.Store(0) // never refuses; the UPSTREAM is what fails
	srv, _ := retryHarness(t, sel, time.Hour, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	})

	// No clock driving at all: if this path waited, the fake clock would never
	// advance and the request would hang until the test binary's own deadline.
	answered := make(chan int, 1)
	go func() {
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
		if err != nil {
			answered <- 0
			return
		}
		answered <- resp.StatusCode
		resp.Body.Close()
	}()

	select {
	case got := <-answered:
		if got == http.StatusTooManyRequests {
			t.Errorf("an upstream failure was answered %d; it must not be treated as a capacity "+
				"refusal", got)
		}
	//clockexempt: bounds the test itself, not the code under test
	case <-time.After(10 * time.Second):
		t.Fatal("an upstream failure hung against an hour-long CAPACITY budget: the budget is " +
			"covering transport errors, which it must not")
	}
}
