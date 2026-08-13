package gateway

import (
	"errors"
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/proxy"
)

// Waiting out a capacity refusal instead of passing it to the client.
//
// Both refusals this covers are statements about one instant. "Every backend is
// saturated" and "nothing was far enough below the holders to be worth a copy"
// are true until any in-flight request anywhere completes, which on a fleet
// serving thousands of requests is a matter of milliseconds. The client's own
// answer to a 429 is to back off and try again, so the only question is whether
// that waiting happens here — where the fleet's state is already known and the
// request body is already buffered — or across a round trip that re-parses,
// re-extracts prefix units and re-decides from scratch.
//
// It is deliberately NOT a general retry. A 502 means a backend broke and the
// proxy already has its own tight rules for that; waiting on it would delay an
// error the caller has to handle either way.

// backoffSteps is the delay ladder, in the shape Anton specified: 10ms, 20ms,
// 50ms and so on by 1-2-5, flattening at 3s.
//
// A 1-2-5 progression rather than pure doubling because the useful range spans
// three orders of magnitude. The first rungs are short enough that a slot
// freeing up immediately costs almost nothing; the last are long enough that a
// genuinely saturated fleet is not being asked the same question hundreds of
// times while it works through its queue.
var backoffSteps = []time.Duration{
	10 * time.Millisecond,
	20 * time.Millisecond,
	50 * time.Millisecond,
	100 * time.Millisecond,
	200 * time.Millisecond,
	500 * time.Millisecond,
	time.Second,
	2 * time.Second,
	3 * time.Second,
}

// backoffAt returns the nth delay, with jitter, clamped so the wait never
// overruns what is left of the budget.
//
// Equal jitter — half the step plus a random half — rather than none, because
// the requests being delayed here were all refused by the same fleet at the
// same moment. Released in lockstep they would arrive together, saturate it
// again, and re-synchronise on the next rung; spread across the interval they
// probe it continuously instead.
func backoffAt(n int, remaining time.Duration) time.Duration {
	step := backoffSteps[len(backoffSteps)-1]
	if n < len(backoffSteps) {
		step = backoffSteps[n]
	}
	d := step/2 + time.Duration(rand.Int64N(int64(step/2)+1))
	if d > remaining {
		d = remaining
	}
	return d
}

// isCapacityRefusal reports whether an error means "the fleet cannot take this
// right now" as opposed to "something went wrong".
func isCapacityRefusal(err error) bool {
	return capacityReason(err) != ""
}

// capacityReason names WHICH refusal is being waited out, and the distinction
// is the whole diagnostic value of these counters — the two mean opposite
// things about the configuration, spelled out where the names are declared.
//
// A fleet retrying constantly on guard_blocked with
// router_cache_overflows_total at zero is not evidence that the ordering is
// wrong; it is evidence that the threshold is too tight, or off. Without the
// label the two are indistinguishable, and the natural reading of "retries
// happened and no transient did" is that the router waited when it should have
// fallen back — which is the one thing it cannot do.
func capacityReason(err error) string {
	switch {
	case errors.Is(err, policy.ErrAllBackendsSaturated):
		return metrics.ReasonSaturated
	case errors.Is(err, policy.ErrSplitGuardBlocked):
		return metrics.ReasonGuardBlocked
	}
	return ""
}

// serveWithCapacityRetry runs the request, and on a capacity refusal waits and
// re-decides until the budget is spent.
//
// The candidate set is rebuilt on every attempt rather than reused. Health,
// in-flight counts and the prefix tree all move while we wait — that movement
// is the entire reason for waiting — so re-deciding against the snapshot that
// produced the refusal would be asking the same question and expecting a
// different answer.
func (s *Server) serveWithCapacityRetry(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	deadline := s.cfg.Clock.Now().Add(s.cfg.RetryTimeLimit)
	var res proxy.Result

	// Set on the first refusal that is actually waited on, so a request that
	// never waits observes nothing and the histogram counts REQUESTS THAT
	// WAITED rather than requests.
	var firstRefusal time.Time
	// waited is the time spent in the retry path BEFORE the attempt currently
	// running — snapshotted at the top of each iteration rather than read at the
	// end.
	//
	// The distinction is the difference between a number that means one thing
	// and one that means two. Measured at settle time, the attempt that ENDS the
	// loop is inside the span: for `expired` that is a refusal costing
	// microseconds, but for `satisfied` it is a whole completion, so the same
	// series would report a bounded ~10s for one outcome and an unbounded 30s
	// for the other against the same 10s budget. Averaging across outcomes then
	// produces a figure describing nothing, which is exactly what a dashboard
	// does by default.
	//
	// Bounded by the budget in every outcome, this answers the question the flag
	// is tuned against: how much latency did the waiting add. End-to-end cost is
	// already router_request_duration_seconds, measured around the whole handler
	// and therefore inclusive of both.
	var waited time.Duration
	lastReason := ""
	settle := func(how string) {
		if lastReason == "" {
			return
		}
		metrics.RetryWaitSeconds.WithLabelValues(lastReason, how).Observe(waited.Seconds())
	}

	for attempt := 0; ; attempt++ {
		if lastReason != "" {
			waited = s.cfg.Clock.Now().Sub(firstRefusal)
		}
		cands := s.candidates(target)
		if len(cands) == 0 {
			// Every backend went unhealthy while we waited. The caller is owed
			// the refusal we already have rather than a fresh error about an
			// empty set: the request was refused for capacity, and the fleet
			// disappearing afterwards does not change what happened to it. The
			// handler's own up-front check covers an empty set on entry.
			settle("abandoned")
			return res
		}

		res = s.px.Serve(w, r, cands, target.Selector, s.dialect, rr, body, accepted, outcome, auth)
		if s.cfg.RetryTimeLimit <= 0 || !isCapacityRefusal(res.Err) || res.Committed {
			// satisfied means the waiting worked: a later attempt was served.
			// Anything else — a backend broke, the fleet emptied — ended the
			// wait without answering the question it was asked, and counting
			// that as a rescue would inflate the one number the budget is
			// judged on.
			if res.Err == nil {
				settle("satisfied")
			} else {
				settle("abandoned")
			}
			return res
		}

		// Stamped before the budget is checked, so a request refused with the
		// budget already spent still lands in the histogram — at ~0s, which is
		// the true added latency — rather than incrementing `exhausted` in
		// RetriesTotal while appearing in no per-request series at all.
		if lastReason == "" {
			firstRefusal = s.cfg.Clock.Now()
		}
		lastReason = capacityReason(res.Err)

		remaining := deadline.Sub(s.cfg.Clock.Now())
		if remaining <= 0 {
			metrics.RetriesTotal.WithLabelValues(lastReason, "exhausted").Inc()
			settle("expired")
			return res
		}

		metrics.RetriesTotal.WithLabelValues(lastReason, "retried").Inc()
		select {
		case <-s.cfg.Clock.After(backoffAt(attempt, remaining)):
		case <-r.Context().Done():
			// The caller gave up while we were waiting. Returning the refusal
			// unchanged is right: the response writer is already dead, and the
			// handler above renders nothing on a cancelled request.
			//
			// This leaves without looping, so the sleep just taken has to be
			// added here or it goes unrecorded.
			waited = s.cfg.Clock.Now().Sub(firstRefusal)
			settle("abandoned")
			return res
		}
	}
}
