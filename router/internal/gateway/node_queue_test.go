package gateway_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/testutil/mockvllm"
)

// queueWaitCount reads router_backend_queue_wait_seconds' sample count for one
// outcome, the same way gateway_test.go's observedFractionStats reads a
// histogram directly rather than through a last-value gauge.
func queueWaitCount(t *testing.T, outcome string) uint64 {
	t.Helper()
	var pb dto.Metric
	if err := metrics.BackendQueueWaitSeconds.WithLabelValues(outcome).(prometheus.Metric).Write(&pb); err != nil {
		t.Fatalf("write histogram: %v", err)
	}
	return pb.GetHistogram().GetSampleCount()
}

// waitForQueued polls until b's wait queue reaches exactly want, or fails the
// test. Needed because joining the queue happens inside the handler goroutine
// a test only observes from the outside, the same reason assertNoLeaks polls
// Inflight rather than assuming it has already settled.
func waitForQueued(t *testing.T, b interface{ Queued() int64 }, want int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if b.Queued() == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("Queued() never reached %d (stuck at %d)", want, b.Queued())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// holdWorker arranges w to block every request until release is closed,
// reporting each entry on entered — the pattern
// TestMaxNodeConcurrencyRejects429WhenAllBackendsAtCap uses.
func holdWorker(w *mockvllm.Worker, idx int, entered chan<- int, release <-chan struct{}) {
	w.Behave(func(*http.Request) mockvllm.Script {
		entered <- idx
		<-release
		return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
	})
}

// errorCode extracts the OpenAI-shaped error code from a response body, or ""
// if it doesn't parse as one.
func errorCode(body []byte) string {
	var env struct {
		Error struct{ Code string } `json:"error"`
	}
	_ = json.Unmarshal(body, &env)
	return env.Error.Code
}

// TestNodeQueueSpilloverUnaffectedByRetryBudget guards that merely setting
// --retry-time-limit does not change the EXISTING spillover behaviour: while
// any backend is still usable, a new request lands on it immediately, never
// touching the wait queue at all. The queue only ever engages from the
// len(usable)==0 branch (see affinity.Policy.Select), so this is really a
// regression check that wiring RetryTimeLimit through didn't change the
// ladder above it.
func TestNodeQueueSpilloverUnaffectedByRetryBudget(t *testing.T) {
	h := newHarness(t, 2, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1
		c.RetryTimeLimit = 2 * time.Second
	})

	release := make(chan struct{})
	entered := make(chan int, 4)
	// Both workers start out ready to hold a request, since which one the
	// first request lands on is an unspecified 50/50 tie-break (see
	// TestMaxNodeConcurrencyRejects429WhenAllBackendsAtCap). Once that is
	// known, the OTHER worker is switched back to answering normally — it
	// must stay free to prove spillover, not become a second held request.
	for i, w := range h.workers {
		holdWorker(w, i, entered, release)
	}

	done := make(chan *http.Response, 1)
	go func() { done <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	first := <-entered
	other := 1 - first
	h.workers[other].Behave(func(*http.Request) mockvllm.Script {
		return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
	})

	bFirst, _ := h.reg.Snapshot().Get(h.workers[first].URL())
	bOther, _ := h.reg.Snapshot().Get(h.workers[other].URL())
	waitForQueued(t, bFirst, 0) // sanity: nobody should ever queue here

	// A second, differently-prefixed request finds the other backend still
	// usable, so it must be admitted immediately rather than waiting.
	resp2 := h.post(t, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"unrelated"}]}`, nil)
	io.Copy(io.Discard, resp2.Body)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (spillover should not have waited)", resp2.StatusCode)
	}
	if got := bOther.Queued(); got != 0 {
		t.Errorf("Queued() = %d, want 0: spillover must never touch the wait queue", got)
	}
	if got := bFirst.Queued(); got != 0 {
		t.Errorf("Queued() = %d, want 0", got)
	}

	close(release)
	r0 := <-done
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	h.assertNoLeaks(t)
}

// TestNodeQueueAdmitsOnRelease is the basic positive case: with a single
// backend (so there is nowhere to spill over to), a request arriving while it
// is at --max-node-concurrency joins the wait queue instead of being shed, and
// is admitted the instant the held request's lease releases.
func TestNodeQueueAdmitsOnRelease(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1
		// The wake here is an explicit close(release), not a real timeout, so
		// an hour-long budget costs nothing in practice but keeps a loaded
		// machine's scheduling delays from racing it into "expired" instead —
		// see retry_wait_test.go's own hour-long budgets for the same reason.
		c.RetryTimeLimit = time.Hour
	})
	b, ok := h.reg.Snapshot().Get(h.workers[0].URL())
	if !ok {
		t.Fatal("backend not found")
	}

	release := make(chan struct{})
	entered := make(chan int, 4)
	holdWorker(h.workers[0], 0, entered, release)

	heldDone := make(chan *http.Response, 1)
	go func() { heldDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	<-entered // the held request is now occupying the only backend

	wokenBefore := queueWaitCount(t, "woken")

	queuedDone := make(chan *http.Response, 1)
	go func() { queuedDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	waitForQueued(t, b, 1)

	// The queued request must not have been shed while waiting.
	select {
	case r := <-queuedDone:
		t.Fatalf("queued request returned early with status %d; want it to keep waiting", r.StatusCode)
	case <-time.After(100 * time.Millisecond):
	}

	close(release)
	r0 := <-heldDone
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	if r0.StatusCode != http.StatusOK {
		t.Fatalf("held request status = %d, want 200", r0.StatusCode)
	}

	r1 := <-queuedDone
	io.Copy(io.Discard, r1.Body)
	r1.Body.Close()
	if r1.StatusCode != http.StatusOK {
		t.Fatalf("queued request status = %d, want 200 once the slot freed", r1.StatusCode)
	}

	waitForQueued(t, b, 0)
	if got := queueWaitCount(t, "woken"); got != wokenBefore+1 {
		t.Errorf("router_backend_queue_wait_seconds{outcome=woken} count = %d, want %d", got, wokenBefore+1)
	}
	h.assertNoLeaks(t)
}

// TestNodeQueueWaitsForPreferredBackendSpecifically is the locality guarantee
// that is the whole point of this feature: a queued request is bound to the
// ONE backend the policy picked for it (the holder of its prefix), not the
// first backend anywhere in the fleet to free a slot. A plain fleet-wide poll
// would have re-decided the instant the OTHER (non-holder) backend freed and
// split onto it immediately; the per-backend queue must not.
func TestNodeQueueWaitsForPreferredBackendSpecifically(t *testing.T) {
	h := newHarness(t, 2, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1
		// Every wake in this test is an explicit close(release), never a real
		// timeout, so an hour-long budget costs nothing but removes any risk
		// of a loaded machine's scheduling delays racing the queued request
		// into "expired" before the test gets around to releasing a holder.
		c.RetryTimeLimit = time.Hour
	})
	const prefixBody = `{"model":"m","messages":[{"role":"user","content":"a shared prefix long enough to hash"}]}`

	// Seed the tree: a fast, non-held request with the target prefix marks
	// whichever backend serves it as "home" — deterministically, since after
	// this every future request with the same prefix is a tier-1 cache hit on
	// home rather than a 50/50 tie-break.
	seedEntered := make(chan int, 1)
	for i, w := range h.workers {
		i := i
		w.Behave(func(*http.Request) mockvllm.Script {
			seedEntered <- i
			return mockvllm.Script{Status: 200, Body: `{"ok":true}`}
		})
	}
	seedResp := h.post(t, "/v1/chat/completions", prefixBody, nil)
	io.Copy(io.Discard, seedResp.Body)
	seedResp.Body.Close()
	if seedResp.StatusCode != http.StatusOK {
		t.Fatalf("seed request status = %d, want 200", seedResp.StatusCode)
	}
	home := <-seedEntered
	cold := 1 - home
	bHome, _ := h.reg.Snapshot().Get(h.workers[home].URL())
	bCold, _ := h.reg.Snapshot().Get(h.workers[cold].URL())

	// Now hold every request so both backends can be driven to cap.
	release := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
	entered := make(chan int, 8)
	for i, w := range h.workers {
		i := i
		holdWorker(w, i, entered, release[i])
	}

	// Held request A: same prefix, must land on home (tier-1 cache hit).
	doneA := make(chan *http.Response, 1)
	go func() { doneA <- h.post(t, "/v1/chat/completions", prefixBody, nil) }()
	if got := <-entered; got != home {
		t.Fatalf("held request A landed on worker %d, want home (%d): the seed didn't pin affinity as expected", got, home)
	}

	// Held request B: home is now saturated, so this splits onto cold — the
	// only usable backend left.
	doneB := make(chan *http.Response, 1)
	go func() { doneB <- h.post(t, "/v1/chat/completions", prefixBody, nil) }()
	if got := <-entered; got != cold {
		t.Fatalf("held request B landed on worker %d, want cold (%d) via the split tier", got, cold)
	}

	// Both backends are now at cap=1. The request under test joins home's
	// queue specifically (home is the only MARKED holder so far; cold's mark
	// hasn't committed yet since request B is still held).
	queuedDone := make(chan *http.Response, 1)
	go func() { queuedDone <- h.post(t, "/v1/chat/completions", prefixBody, nil) }()
	waitForQueued(t, bHome, 1)
	if got := bCold.Queued(); got != 0 {
		t.Fatalf("Queued() on cold = %d, want 0: the queued request must be bound to home, not cold", got)
	}

	// Free cold FIRST. A fleet-wide poll would re-decide right now and land on
	// cold (usable, 0 in-flight); the per-backend queue must not even notice.
	close(release[cold])
	rB := <-doneB
	io.Copy(io.Discard, rB.Body)
	rB.Body.Close()

	select {
	case r := <-queuedDone:
		t.Fatalf("queued request was served (status %d) after COLD freed; it must wait for HOME specifically", r.StatusCode)
	case <-time.After(150 * time.Millisecond):
	}
	if got := bHome.Queued(); got != 1 {
		t.Fatalf("Queued() on home = %d, want 1: still waiting after cold freed", got)
	}

	// Now free home. The queued request must be woken and complete.
	close(release[home])
	rA := <-doneA
	io.Copy(io.Discard, rA.Body)
	rA.Body.Close()

	r := <-queuedDone
	io.Copy(io.Discard, r.Body)
	r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("queued request status = %d, want 200 once home freed", r.StatusCode)
	}
	h.assertNoLeaks(t)
}

// TestNodeQueueTimesOutTo429 covers the budget running out while queued: the
// caller gets the same 429 it would have gotten without queueing at all, not
// a different status or body shape.
func TestNodeQueueTimesOutTo429(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1
		c.RetryTimeLimit = 60 * time.Millisecond
	})

	release := make(chan struct{})
	entered := make(chan int, 2)
	holdWorker(h.workers[0], 0, entered, release)

	heldDone := make(chan *http.Response, 1)
	go func() { heldDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	<-entered

	expiredBefore := queueWaitCount(t, "expired")
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429 (body: %s)", resp.StatusCode, body)
	}
	if got := resp.Header.Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want \"1\"", got)
	}
	if got := errorCode(body); got != "rate_limit_error" {
		t.Errorf("error code = %q, want rate_limit_error", got)
	}
	if got := queueWaitCount(t, "expired"); got != expiredBefore+1 {
		t.Errorf("router_backend_queue_wait_seconds{outcome=expired} count = %d, want %d", got, expiredBefore+1)
	}

	close(release)
	r0 := <-heldDone
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	h.assertNoLeaks(t)
}

// TestNodeQueueDepthOverflowReturns503 covers the queue's own depth bound
// (one waiter per concurrency slot): a request that cannot even get a place
// in line is refused immediately with 503, never waiting at all — distinct
// from the 429 a queued-then-timed-out request gets.
func TestNodeQueueDepthOverflowReturns503(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1 // queue depth bound = max(1, limit) = 1
		// The first-in-line request's wake is an explicit close(release), not
		// a real timeout; an hour-long budget removes any risk of a loaded
		// machine's scheduling delays racing it into "expired" instead.
		c.RetryTimeLimit = time.Hour
	})
	b, _ := h.reg.Snapshot().Get(h.workers[0].URL())

	release := make(chan struct{})
	entered := make(chan int, 2)
	holdWorker(h.workers[0], 0, entered, release)

	heldDone := make(chan *http.Response, 1)
	go func() { heldDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	<-entered

	queuedDone := make(chan *http.Response, 1)
	go func() { queuedDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	waitForQueued(t, b, 1) // the queue is now at its depth bound (1)

	fullBefore := testutil.ToFloat64(metrics.BackendQueueFullTotal.WithLabelValues(b.URL))
	resp := h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 (body: %s)", resp.StatusCode, body)
	}
	if strings.Contains(string(body), "prefix") || strings.Contains(string(body), "backend") {
		t.Errorf("503 body describes router internals to the caller: %s", body)
	}
	if got := testutil.ToFloat64(metrics.BackendQueueFullTotal.WithLabelValues(b.URL)); got != fullBefore+1 {
		t.Errorf("router_backend_queue_full_total{backend=%s} = %v, want %v", b.URL, got, fullBefore+1)
	}
	// The overflow must not itself have joined the queue.
	if got := b.Queued(); got != 1 {
		t.Errorf("Queued() = %d, want 1 (unchanged by the overflowing request)", got)
	}

	close(release)
	r0 := <-heldDone
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	r1 := <-queuedDone
	io.Copy(io.Discard, r1.Body)
	r1.Body.Close()
	if r1.StatusCode != http.StatusOK {
		t.Fatalf("first-in-line request status = %d, want 200 once the slot freed", r1.StatusCode)
	}
	h.assertNoLeaks(t)
}

// TestNodeQueueDepthOverrideAllowsDeeperQueue covers MaxNodeQueueDepth: with
// --max-node-concurrency at 1 (so the default queue depth would be 1, exactly
// what TestNodeQueueDepthOverflowReturns503 pins to 503 on the SECOND
// waiter), a large override lets MORE than one waiter queue behind the held
// request without hitting that overflow path at all.
func TestNodeQueueDepthOverrideAllowsDeeperQueue(t *testing.T) {
	const depth = 3
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1
		c.MaxNodeQueueDepth = depth
		// Every wake here is an explicit close(release), never a real timeout —
		// see the identical reasoning on the other tests in this file.
		c.RetryTimeLimit = time.Hour
	})
	b, _ := h.reg.Snapshot().Get(h.workers[0].URL())

	release := make(chan struct{})
	entered := make(chan int, 1+depth)
	holdWorker(h.workers[0], 0, entered, release)

	heldDone := make(chan *http.Response, 1)
	go func() { heldDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	<-entered // the held request now occupies the only backend

	fullBefore := testutil.ToFloat64(metrics.BackendQueueFullTotal.WithLabelValues(b.URL))

	// Queue `depth` more requests — more than the concurrency limit (1), which
	// is exactly what the DEFAULT depth bound (max(1, limit) = 1) would refuse
	// on the second of these with a 503. The override must admit all of them.
	queuedDone := make([]chan *http.Response, depth)
	for i := range queuedDone {
		queuedDone[i] = make(chan *http.Response, 1)
		i := i
		go func() { queuedDone[i] <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
		waitForQueued(t, b, int64(i+1))
	}

	if got := testutil.ToFloat64(metrics.BackendQueueFullTotal.WithLabelValues(b.URL)); got != fullBefore {
		t.Errorf("router_backend_queue_full_total{backend=%s} = %v, want unchanged %v: the override must prevent any overflow", b.URL, got, fullBefore)
	}

	// None of the queued requests may have been shed while waiting.
	for i, ch := range queuedDone {
		select {
		case r := <-ch:
			t.Fatalf("queued request %d returned early with status %d; want it to keep waiting", i, r.StatusCode)
		case <-time.After(50 * time.Millisecond):
		}
	}

	// Release the held request and then each queued one in turn, draining the
	// line down to empty.
	close(release)
	r0 := <-heldDone
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	if r0.StatusCode != http.StatusOK {
		t.Fatalf("held request status = %d, want 200", r0.StatusCode)
	}

	for i, ch := range queuedDone {
		r := <-ch
		body, _ := io.ReadAll(r.Body)
		r.Body.Close()
		if r.StatusCode != http.StatusOK {
			t.Fatalf("queued request %d status = %d, want 200 once its turn came (body: %s)", i, r.StatusCode, body)
		}
	}

	waitForQueued(t, b, 0)
	h.assertNoLeaks(t)
}

// TestNodeQueueAbandonedOnClientDisconnect covers the caller giving up while
// queued: the ticket must be released (not leaked) and counted as abandoned,
// freeing its place for anyone behind it.
func TestNodeQueueAbandonedOnClientDisconnect(t *testing.T) {
	h := newHarness(t, 1, func(c *harnessConfig) {
		c.MaxNodeConcurrency = 1
		// An hour, not a few seconds: on a loaded machine (the whole module's
		// suite running at once) scheduling delays alone can eat a short
		// budget, and the request would then expire rather than being
		// abandoned — failing for a reason this test is not about. See the
		// identical reasoning on retry_wait_test.go's own hour-long budgets.
		c.RetryTimeLimit = time.Hour
	})
	b, _ := h.reg.Snapshot().Get(h.workers[0].URL())

	release := make(chan struct{})
	entered := make(chan int, 2)
	holdWorker(h.workers[0], 0, entered, release)

	heldDone := make(chan *http.Response, 1)
	go func() { heldDone <- h.post(t, "/v1/chat/completions", `{"model":"m"}`, nil) }()
	<-entered

	abandonedBefore := queueWaitCount(t, "abandoned")

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.srv.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"m"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	clientErr := make(chan error, 1)
	go func() {
		resp, err := h.srv.Client().Do(req)
		if resp != nil {
			resp.Body.Close()
		}
		clientErr <- err
	}()

	waitForQueued(t, b, 1)
	cancel()

	select {
	case err := <-clientErr:
		if err == nil {
			t.Fatal("client request succeeded, want a cancellation error")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("client request never returned after cancel")
	}

	waitForQueued(t, b, 0)
	if got := queueWaitCount(t, "abandoned"); got != abandonedBefore+1 {
		t.Errorf("router_backend_queue_wait_seconds{outcome=abandoned} count = %d, want %d", got, abandonedBefore+1)
	}

	close(release)
	r0 := <-heldDone
	io.Copy(io.Discard, r0.Body)
	r0.Body.Close()
	h.assertNoLeaks(t)
}
