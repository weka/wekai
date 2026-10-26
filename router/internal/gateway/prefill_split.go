package gateway

import (
	"net/http"

	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/prefill"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

// splitAndServe answers a request its backend refused as a large prefill,
// without the caller ever seeing the refusal.
//
// The prefill runs on a node the manager has set aside, as a copy of the
// request limited to one token. The request itself then goes through ordinary
// decode routing, over decode nodes only, and its response is the caller's.
// Both carry the bypass, so neither can be refused again: if the shared KV tier
// lagged the job, the decode node recomputes the rest instead of looping.
//
// When no prefill node can be used the request goes back to the backend that
// refused it, with the bypass, and is served there.
func (s *Server) splitAndServe(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte, rejected proxy.Result,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	m := target.Split
	m.CountRejection()
	// Everything sent from here carries the bypass, so a refusal of it is not
	// absorbed: it goes to the caller instead of starting another round.
	r = r.WithContext(proxy.WithoutRejectCapture(r.Context()))
	bypassed := prefill.WithBypass(body)
	// A fresh decision: the rejected attempt's selection state is not this
	// request's final one.
	rr.PolicyState = nil

	job, err := m.Reserve(s.healthy(target), rejected.PrefillTokens, s.prefillPicker(target, rr, r))
	if err != nil {
		m.FallbackNoNode()
		return s.serveOnOriginal(w, r, target, rr, bypassed, rejected.Backend, accepted, outcome, auth)
	}
	defer job.Done()

	if err := m.WaitReady(r.Context(), job); err != nil {
		return proxy.Result{Err: err}
	}
	ok := s.runPrefillJob(r, job.Backend, rr, prefill.JobBody(body), auth)
	m.CountJob(ok)
	if !ok {
		m.FallbackJobFailed()
	}
	job.Done()

	rr.PolicyState = nil
	return s.serveWithCapacityRetry(w, r, target, rr, bypassed, accepted, outcome, auth)
}

// prefillPicker chooses among eligible prefill nodes with the pool's own flow,
// so the choice sees the same prefix knowledge and load rules as ordinary
// routing. It selects on a copy of the request: the flow's per-request state
// belongs to the decode decision that comes after.
func (s *Server) prefillPicker(t Target, rr *policy.RoutingRequest, r *http.Request) prefill.Picker {
	return func(eligible []*registry.Backend) (*registry.Backend, error) {
		sel := *rr
		sel.PolicyState = nil
		return t.Selector.Select(r.Context(), eligible, &sel)
	}
}

// runPrefillJob sends the job to its node through the ordinary proxy path, so
// the lease, circuit breaker, credentials and timeouts all apply, and discards
// the response. It does not commit the prefix to the flow: the node is outside
// decode routing, and a mark there would make the request's real holders look
// unavailable to the decode decision that follows.
func (s *Server) runPrefillJob(r *http.Request, b *registry.Backend,
	rr *policy.RoutingRequest, body []byte, auth proxy.Auth) bool {
	job := *rr
	job.PolicyState = nil
	res := s.px.Serve(&discardWriter{h: http.Header{}}, r, []*registry.Backend{b},
		policy.LeastOutstanding{}, s.dialect, &job, body, nil, nil, auth)
	return res.Err == nil && !res.PrefillRejected && res.Status >= 200 && res.Status < 300
}

// serveOnOriginal resends a refused request to the backend that refused it. A
// backend that has since left decode mode or health is not sent decode work;
// the request is routed normally instead.
func (s *Server) serveOnOriginal(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte, orig *registry.Backend,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	if orig != nil && orig.Available() && target.Split.ModeOf(orig) == prefill.Decode {
		return s.px.Serve(w, r, []*registry.Backend{orig}, policy.LeastOutstanding{},
			s.dialect, rr, body, accepted, outcome, auth)
	}
	return s.serveWithCapacityRetry(w, r, target, rr, body, accepted, outcome, auth)
}

// discardWriter swallows a prefill job's response.
type discardWriter struct{ h http.Header }

func (d *discardWriter) Header() http.Header         { return d.h }
func (d *discardWriter) Write(p []byte) (int, error) { return len(p), nil }
func (d *discardWriter) WriteHeader(int)             {}
