package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

// retryFixture is a gateway under test, the fake clock its retry loop waits on,
// and a barrier that a metrics assertion has to cross first.
type retryFixture struct {
	srv      *httptest.Server
	clk      *clock.Fake
	handlers sync.WaitGroup
}

// settled blocks until every request served so far has RETURNED from the
// gateway, not merely been answered.
//
// The two are not the same instant. The proxy writes the response from inside
// the retry loop, and only once that loop returns does it settle its histogram
// — so a test reading a metric the moment http.Post returns is racing the tail
// of the handler it just triggered. It is a race the test usually wins on one
// core and loses on several, which makes it a CI failure rather than a local
// one: about 1 run in 80 at GOMAXPROCS=2, on whichever commit happened to be
// building.
//
// Every assertion about a metric this package records therefore has to cross
// this barrier, and assertions about the RESPONSE do not.
func (f *retryFixture) settled() { f.handlers.Wait() }

func retryHarness(t *testing.T, sel proxy.Selector, limit time.Duration, upstream http.HandlerFunc) *retryFixture {
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

	fx := &retryFixture{clk: clk}
	// Add runs before the gateway writes a single byte, so a caller that reaches
	// settled() by way of a returned response is guaranteed to see the counter
	// already raised.
	fx.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fx.handlers.Add(1)
		defer fx.handlers.Done()
		gw.ServeHTTP(w, r)
	}))
	t.Cleanup(fx.srv.Close)
	return fx
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
			fx := retryHarness(t, sel, 10*time.Second, nil)

			done := make(chan struct{})
			go drive(fx.clk, done)
			resp, err := http.Post(fx.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
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
	fx := retryHarness(t, sel, 2*time.Second, nil)

	done := make(chan struct{})
	go drive(fx.clk, done)
	resp, err := http.Post(fx.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
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
	fx := retryHarness(t, sel, 0, nil) // the shipped default

	resp, err := http.Post(fx.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
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
	fx := retryHarness(t, sel, time.Hour, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	})

	// No clock driving at all: if this path waited, the fake clock would never
	// advance and the request would hang until the test binary's own deadline.
	answered := make(chan int, 1)
	go func() {
		resp, err := http.Post(fx.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
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

// TestRetryReasonNamesWhichRefusal. "Retries happened and no transient did" is
// the observation an operator will actually make, and it has two opposite
// explanations: every backend was saturated, so the fallback could not apply at
// all, or the guard refused and the fallback found nobody inside its margin —
// a threshold set too tight. Sharing one label makes those indistinguishable
// and invites the wrong conclusion, that the router waited when it should have
// fallen back.
func TestRetryReasonNamesWhichRefusal(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{"saturated", policy.ErrAllBackendsSaturated, "capacity_saturated"},
		{"guard", policy.ErrSplitGuardBlocked, "capacity_guard_blocked"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := capacityReason(tc.err); got != tc.want {
				t.Errorf("capacityReason = %q, want %q", got, tc.want)
			}
		})
	}
	if got := capacityReason(errors.New("connection refused")); got != "" {
		t.Errorf("a transport error was named %q; only capacity refusals are waited out", got)
	}
	if isCapacityRefusal(nil) {
		t.Error("a nil error must not read as a capacity refusal")
	}
}
