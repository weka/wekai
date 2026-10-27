package gateway

import (
	"bytes"
	"net/http"
	"strconv"
	"time"

	"github.com/weka/wekai/router/internal/pd"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

// pdRoute reports whether a route class carries NIXL kv_transfer_params. Only
// the OpenAI generation endpoints do: the Anthropic messages surface has no
// field for them, so it is served over the D nodes alone.
func pdRoute(class string) bool { return class == "chat" || class == "completions" }

// pdServe serves a request as a prefill leg on a P node followed by a decode
// leg on a D node, as vLLM's own disaggregation proxy does.
//
// Both legs are routed by the pool's own flow, restricted to their node set, so
// each sees the prefix knowledge and load rules of ordinary routing, and each
// records the prefix against the node that served it. A prefill leg that cannot
// be completed sends the request to a D node without kv_transfer_params, which
// recomputes the prompt.
func (s *Server) pdServe(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	set := target.PD
	decodeBody := body
	if params, ok := s.runPDPrefill(r, target, rr, body, outcome, auth); ok {
		decodeBody = pd.DecodeBody(body, params)
	}
	rr.PolicyState = nil
	start := s.cfg.Clock.Now()
	res := s.serveWithCapacityRetry(w, r, target, rr, decodeBody, accepted, outcome, auth)
	set.M.DecodeSeconds.Observe(s.cfg.Clock.Now().Sub(start).Seconds())
	return res
}

// runPDPrefill runs the prefill leg and returns the kv_transfer_params the P
// node answered with. ok is false when the request must fall back to a D node
// alone.
//
// A router-side saturation of every P node, or a 429 from the chosen one, is
// waited out and retried, like a prefill-split job: the leg holds its place
// until it is served, --retry-time-limit runs out when that is set, or the
// caller leaves.
func (s *Server) runPDPrefill(
	r *http.Request, target Target, rr *policy.RoutingRequest, body []byte,
	outcome proxy.OnOutcome, auth proxy.Auth,
) (pd.Params, bool) {
	set := target.PD
	m := set.M
	start := s.cfg.Clock.Now()
	defer func() { m.PrefillSeconds.Observe(s.cfg.Clock.Now().Sub(start).Seconds()) }()

	leg := *rr
	sel := target.PrefillSelector
	var accepted proxy.OnAccepted
	if c, ok := sel.(policy.Committer); ok {
		accepted = func(b *registry.Backend) { c.Commit(b, &leg) }
	}
	if o, ok := sel.(policy.Observer); ok {
		outcome = func(b *registry.Backend, status int) {
			if status == http.StatusTooManyRequests {
				o.OnRefused(b)
			} else if status >= 200 && status < 500 {
				o.OnAccepted(b)
			}
		}
	}
	pbody := pd.PrefillBody(body)
	deadline := start.Add(s.cfg.RetryTimeLimit)

	for attempt := 0; ; attempt++ {
		cands := set.PrefillNodes(s.healthy(target))
		if len(cands) == 0 {
			m.FallbackNoNode.Inc()
			return nil, false
		}
		leg.PolicyState = nil
		cw := &captureWriter{h: http.Header{}}
		res := s.px.Serve(cw, r, cands, sel, s.dialect, &leg, pbody, accepted, outcome, auth)

		refused := isCapacityRefusal(res.Err) || (res.Err == nil && res.Status == http.StatusTooManyRequests)
		if refused && r.Context().Err() == nil &&
			(s.cfg.RetryTimeLimit <= 0 || s.cfg.Clock.Now().Before(deadline)) {
			m.Retries.Inc()
			wait := backoffAt(attempt, time.Hour)
			if ra, err := strconv.Atoi(cw.h.Get("Retry-After")); err == nil && ra >= 0 {
				wait = min(time.Duration(ra)*time.Second, maxJobRetryWait)
			}
			select {
			case <-s.cfg.Clock.After(wait):
				continue
			case <-r.Context().Done():
			}
		}

		if res.Err != nil || res.Status < 200 || res.Status >= 300 {
			m.LegFailed.Inc()
			m.FallbackFailed.Inc()
			return nil, false
		}
		params, ok := pd.ParseParams(cw.buf.Bytes())
		if !ok {
			m.LegNoParams.Inc()
			m.FallbackNoParams.Inc()
			return nil, false
		}
		m.LegOK.Inc()
		return params, true
	}
}

// captureWriter keeps a prefill leg's response, which is small: one token and
// the transfer params.
type captureWriter struct {
	h   http.Header
	buf bytes.Buffer
}

func (c *captureWriter) Header() http.Header         { return c.h }
func (c *captureWriter) Write(p []byte) (int, error) { return c.buf.Write(p) }
func (c *captureWriter) WriteHeader(int)             {}
