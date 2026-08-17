package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
)

// What the retry budget costs and what it buys, per REQUEST.
//
// RetriesTotal counts attempts, so a request that went round four times adds
// four and "retried minus exhausted" counts nothing in particular. The
// histogram's _count is the missing figure: how many requests entered the retry
// path, and how many the waiting actually rescued.

func post(srv *httptest.Server) (*http.Response, error) {
	return http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
}

func waitStats(t *testing.T, reason, outcome string) (count uint64, sum float64) {
	t.Helper()
	o := metrics.RetryWaitSeconds.WithLabelValues(reason, outcome)
	m, ok := o.(prometheus.Metric)
	if !ok {
		t.Fatalf("RetryWaitSeconds child is not a Metric: %T", o)
	}
	var pb dto.Metric
	if err := m.Write(&pb); err != nil {
		t.Fatalf("read histogram: %v", err)
	}
	return pb.GetHistogram().GetSampleCount(), pb.GetHistogram().GetSampleSum()
}

func retryAttempts(t *testing.T, reason, outcome string) float64 {
	t.Helper()
	var pb dto.Metric
	if err := metrics.RetriesTotal.WithLabelValues(reason, outcome).(prometheus.Metric).Write(&pb); err != nil {
		t.Fatalf("read counter: %v", err)
	}
	return pb.GetCounter().GetValue()
}

func TestRetryWaitCountsRequestsNotAttempts(t *testing.T) {
	const reason = metrics.ReasonGuardBlocked
	countBefore, sumBefore := waitStats(t, reason, "satisfied")
	attemptsBefore := retryAttempts(t, reason, "retried")

	sel := &refusingSelector{err: policy.ErrSplitGuardBlocked}
	sel.remaining.Store(3) // three refusals, then the fleet frees a slot
	// An hour, because drive() advances a fake second per real millisecond: a
	// budget denominated in fake seconds is really one denominated in
	// milliseconds of goroutine scheduling, and on a loaded machine the refusals
	// alone can spend it. The request would then expire rather than be
	// satisfied, and this test would fail for a reason it is not about. Where
	// the budget's edge is under test the fixture says so — see
	// TestRetryWaitSeparatesRescuedFromExpired.
	srv, clk := retryHarness(t, sel, time.Hour, nil)

	done := make(chan struct{})
	go drive(clk, done)
	resp, err := post(srv)
	close(done)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	countAfter, sumAfter := waitStats(t, reason, "satisfied")
	if got := countAfter - countBefore; got != 1 {
		t.Errorf("router_retry_wait_seconds_count moved by %d, want 1: one request waited, however "+
			"many times it went round", got)
	}
	if got := retryAttempts(t, reason, "retried") - attemptsBefore; got != 3 {
		t.Errorf("router_retries_total moved by %v, want 3: the counter is per attempt, and the "+
			"pair is only useful if the two disagree in exactly this way", got)
	}
	if sumAfter <= sumBefore {
		t.Errorf("the observed wait was %v seconds; a request that waited three times must record "+
			"the latency it added, or the budget's cost is unmeasurable", sumAfter-sumBefore)
	}
}

// TestRetryWaitSeparatesRescuedFromExpired: the two outcomes are the numerator
// and the denominator of "was this flag worth it", so they must not merge.
func TestRetryWaitSeparatesRescuedFromExpired(t *testing.T) {
	const reason = metrics.ReasonSaturated
	satisfiedBefore, _ := waitStats(t, reason, "satisfied")
	expiredBefore, _ := waitStats(t, reason, "expired")

	sel := &refusingSelector{err: policy.ErrAllBackendsSaturated}
	sel.remaining.Store(1 << 30) // never recovers
	srv, clk := retryHarness(t, sel, 2*time.Second, nil)

	done := make(chan struct{})
	go drive(clk, done)
	resp, err := post(srv)
	close(done)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	satisfiedAfter, _ := waitStats(t, reason, "satisfied")
	expiredAfter, _ := waitStats(t, reason, "expired")
	if got := expiredAfter - expiredBefore; got != 1 {
		t.Errorf("outcome=\"expired\" moved by %d, want 1", got)
	}
	if got := satisfiedAfter - satisfiedBefore; got != 0 {
		t.Errorf("outcome=\"satisfied\" moved by %d on a request that spent the whole budget and "+
			"still got a 429", got)
	}
}

// TestRetryWaitIgnoresRequestsThatNeverWaited: a router serving normally must
// not accumulate observations, or the count stops meaning "entered the retry
// path" and the p50 describes the fleet's happy path instead of the budget.
func TestRetryWaitIgnoresRequestsThatNeverWaited(t *testing.T) {
	var before uint64
	for _, r := range metrics.CapacityRetryReasons {
		for _, o := range []string{"satisfied", "expired", "abandoned"} {
			c, _ := waitStats(t, r, o)
			before += c
		}
	}

	sel := &refusingSelector{err: policy.ErrAllBackendsSaturated}
	sel.remaining.Store(0) // refuses nothing
	srv, _ := retryHarness(t, sel, 10*time.Second, nil)

	resp, err := post(srv)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	var after uint64
	for _, r := range metrics.CapacityRetryReasons {
		for _, o := range []string{"satisfied", "expired", "abandoned"} {
			c, _ := waitStats(t, r, o)
			after += c
		}
	}
	if after != before {
		t.Errorf("a request that was served on its first attempt added %d observations", after-before)
	}
}

// TestRetryWaitExcludesTheServiceTimeOfTheAttemptThatSucceeded.
//
// Measured to the end of the loop, the attempt that resolves it is inside the
// span — and that attempt is a microsecond refusal when the outcome is
// `expired` but a whole completion when it is `satisfied`. One series then
// reports a bounded ~10s for one outcome and an unbounded ~30s for the other
// against the same 10s budget, and any average across them describes nothing.
//
// The upstream here burns four minutes of clock, twenty-four times the budget.
// If that leg is inside the span it cannot be missed.
func TestRetryWaitExcludesTheServiceTimeOfTheAttemptThatSucceeded(t *testing.T) {
	const (
		reason  = metrics.ReasonGuardBlocked
		budget  = time.Hour // see TestRetryWaitCountsRequestsNotAttempts
		service = 4 * time.Minute
	)
	countBefore, sumBefore := waitStats(t, reason, "satisfied")

	sel := &refusingSelector{err: policy.ErrSplitGuardBlocked}
	sel.remaining.Store(2)

	var (
		clk  *clock.Fake
		stop sync.Once
		// Unix nanos rather than a time.Time: written on the server goroutine,
		// read on this one.
		serviceBegan atomic.Int64
	)
	done := make(chan struct{})
	srv, c := retryHarness(t, sel, budget, func(w http.ResponseWriter, r *http.Request) {
		// The driver stops before the service leg, so this handler is the only
		// writer of the clock while it runs. That gives the span under test an
		// exact ceiling rather than one that moves with how often the driver
		// goroutine happened to be scheduled.
		stop.Do(func() { close(done) })
		serviceBegan.Store(clk.Now().UnixNano())
		clk.Advance(service) // a slow completion, on the attempt that succeeds
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","choices":[]}`))
	})
	clk = c
	start := clk.Now()

	go drive(clk, done)
	resp, err := post(srv)
	stop.Do(func() { close(done) })
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()

	countAfter, sumAfter := waitStats(t, reason, "satisfied")
	if countAfter-countBefore != 1 {
		t.Fatalf("observations moved by %d, want 1", countAfter-countBefore)
	}
	// The wait can only cover clock advanced before the successful attempt
	// began, so that is its exact ceiling — and the ceiling sits four minutes
	// below where including the service leg would put it.
	ceiling := time.Unix(0, serviceBegan.Load()).Sub(start).Seconds()
	if got := sumAfter - sumBefore; got > ceiling {
		t.Errorf("observed %.1fs against a ceiling of %.1fs, on a request whose successful attempt "+
			"alone took %v: the attempt that ends the wait is inside the span, which makes this "+
			"series mean one thing for satisfied and another for expired", got, ceiling, service)
	}
}

// TestCapacityReasonsAreTheWarmedSet. The set that is emitted and the set that
// is pre-registered live in different packages; if they drift, a new reason
// ships absent-until-first-fire and reintroduces exactly the ambiguity the
// warm-up exists to remove.
func TestCapacityReasonsAreTheWarmedSet(t *testing.T) {
	warmed := map[string]bool{}
	for _, r := range metrics.CapacityRetryReasons {
		warmed[r] = true
	}
	for _, err := range []error{policy.ErrAllBackendsSaturated, policy.ErrSplitGuardBlocked} {
		r := capacityReason(err)
		if r == "" {
			t.Errorf("%v is waited out but names no reason", err)
			continue
		}
		if !warmed[r] {
			t.Errorf("capacityReason returns %q, which metrics.CapacityRetryReasons does not warm; "+
				"the series would be absent until the first retry", r)
		}
	}
}
