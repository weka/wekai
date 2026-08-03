// Package proxy forwards a request to a selected backend.
//
// It owns the three lifecycles that v1 got wrong:
//
//   - The load lease. Acquired once per attempt, released when the response body
//     is finished — not when headers arrive. v1 charged the decode worker's load
//     at header time, which is exactly the long phase, so measured load for
//     streaming traffic was ~0 (LB-4, A8).
//   - Circuit admission. Allow() is called once, for the selected backend only,
//     after selection. Filtering reads State() and must never consume a
//     half-open probe token (R2).
//   - Retry. Bounded, never after a byte has reached the client, always on a
//     different backend, each attempt with its own lease (REL-1..REL-5, LB-3).
package proxy

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"

	"github.com/weka/wekai/router/internal/circuit"
	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/obs"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// Config tunes the outbound side.
type Config struct {
	// MaxAttempts bounds total tries including the first (REL-1).
	MaxAttempts int
	// UpstreamCredential is sent to backends. The CLIENT's credential is never
	// forwarded (AUTH-9, SEC-4) — v1 relayed the router's own inbound secret to
	// every worker and its logs.
	UpstreamCredential string
	// StreamBufferBytes bounds in-flight buffering per stream, giving the client
	// real backpressure. v1 used an unbounded channel, so a slow client made the
	// router absorb an entire generation (STR-1, STR-N1).
	StreamBufferBytes int
	RequestTimeout    time.Duration
	IdleTimeout       time.Duration
	DialTimeout       time.Duration
}

func DefaultConfig() Config {
	return Config{
		MaxAttempts:       2,
		StreamBufferBytes: 64 << 10,
		RequestTimeout:    600 * time.Second,
		IdleTimeout:       300 * time.Second,
		DialTimeout:       10 * time.Second,
	}
}

// hopByHop headers are stripped in both directions (SEC-3). ReverseProxy handles
// the Connection-listed ones; these are the standard set.
var hopByHop = []string{
	"Connection", "Proxy-Connection", "Keep-Alive", "Proxy-Authenticate",
	"Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade",
}

type Proxy struct {
	cfg  Config
	rp   *httputil.ReverseProxy
	pool *sync.Pool
}

func New(cfg Config) *Proxy {
	if cfg.StreamBufferBytes <= 0 {
		cfg.StreamBufferBytes = 64 << 10
	}
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 1
	}
	p := &Proxy{
		cfg: cfg,
		pool: &sync.Pool{New: func() any {
			b := make([]byte, cfg.StreamBufferBytes)
			return &b
		}},
	}
	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: cfg.DialTimeout}).DialContext,
		MaxIdleConnsPerHost:   256,
		IdleConnTimeout:       50 * time.Second,
		ResponseHeaderTimeout: 0, // bounded by the request context instead
		ForceAttemptHTTP2:     false,
	}
	p.rp = &httputil.ReverseProxy{
		Transport: transport,
		// FlushInterval -1 flushes after every write, so inter-token latency is
		// not inflated by proxy-side coalescing (STR-5).
		FlushInterval: -1,
		BufferPool:    p,
		Rewrite:       func(*httputil.ProxyRequest) {}, // set per attempt
		ErrorHandler:  func(http.ResponseWriter, *http.Request, error) {},
	}
	return p
}

// Get and Put implement httputil.BufferPool over a sync.Pool.
func (p *Proxy) Get() []byte  { return *(p.pool.Get().(*[]byte)) }
func (p *Proxy) Put(b []byte) { p.pool.Put(&b) }

// Result reports what happened, for the caller's logging and metrics.
type Result struct {
	Backend  *registry.Backend
	Status   int
	Attempts int
	Err      error
	// Committed is true once any byte reached the client, after which retry is
	// forbidden regardless of what went wrong (REL-3, STR-9).
	Committed bool
}

var errNoBackend = errors.New("proxy: no backend accepted the request")

// errRetryable is returned from ModifyResponse to make ReverseProxy discard a
// response instead of copying it to the client, so another backend can be tried.
// It never reaches the client.
var errRetryable = errors.New("proxy: retryable upstream response")

// Selector picks a backend from a candidate set.
type Selector interface {
	Select(ctx context.Context, candidates []*registry.Backend, rr *policy.RoutingRequest) (*registry.Backend, error)
	Name() string
}

// OnAccepted is invoked once an upstream has returned response headers for an
// attempt that will actually be relayed — i.e. the backend demonstrably accepted
// the request. Cache-affinity policies commit here rather than at selection time
// (R3): an attempt that fails at connect must leave no trace, or the backend
// looks warm for a prefix it never received.
type OnAccepted func(*registry.Backend)

// Serve routes one request, retrying on a different backend where allowed.
func (p *Proxy) Serve(
	w http.ResponseWriter, r *http.Request,
	candidates []*registry.Backend,
	sel Selector, d dialect.Dialect,
	rr *policy.RoutingRequest, body []byte,
	accepted OnAccepted,
) Result {
	res := Result{}
	remaining := append([]*registry.Backend(nil), candidates...)
	committed := false

	for attempt := 0; attempt < p.cfg.MaxAttempts; attempt++ {
		if len(remaining) == 0 {
			break
		}
		//clockexempt: measures the policy Select budget (OBS-8, NFR-2)
		start := time.Now()
		b, err := sel.Select(r.Context(), remaining, rr)
		//clockexempt: measures the policy Select budget (OBS-8, NFR-2)
		metrics.RoutingDecisionDuration.WithLabelValues(sel.Name()).Observe(time.Since(start).Seconds())
		if err != nil || b == nil {
			res.Err = err
			break
		}

		// Circuit admission: exactly once, for the selected backend only.
		permitted, token := b.CB.Allow()
		if !permitted {
			remaining = without(remaining, b)
			attempt-- // a refused admission is not a delivery attempt
			continue
		}

		metrics.PolicySelections.WithLabelValues(sel.Name(), b.URL).Inc()
		res.Backend = b
		res.Attempts++

		canRetry := attempt+1 < p.cfg.MaxAttempts && len(remaining) > 1
		out := p.attempt(w, r, b, d, body, &committed, canRetry, accepted)
		b.CB.Record(circuit.Classify(out.status, out.err), token)
		if out.err != nil {
			metrics.UpstreamErrors.WithLabelValues(b.URL, kindOf(out.err)).Inc()
			b.Failed.Add(1)
		} else {
			b.Served.Add(1)
		}

		res.Status, res.Err, res.Committed = out.status, out.err, committed

		if !out.retryable || committed || attempt+1 >= p.cfg.MaxAttempts {
			return res
		}
		metrics.RetriesTotal.WithLabelValues(kindOf(out.err), "retried").Inc()
		remaining = without(remaining, b) // REL-1: a *different* backend
	}

	if res.Backend == nil && res.Err == nil {
		res.Err = errNoBackend
	}
	return res
}

type attemptOut struct {
	status    int
	err       error
	retryable bool
}

// attempt performs one upstream round trip.
//
// canRetry tells ModifyResponse whether a retryable status should be *rejected*
// (so nothing is written to the client and the caller can try another backend) or
// relayed as the final answer. This is the one genuinely awkward part of building
// on ReverseProxy: by the time the response reaches the client the decision is
// already made, so it has to be made inside ModifyResponse.
func (p *Proxy) attempt(
	w http.ResponseWriter, r *http.Request,
	b *registry.Backend, d dialect.Dialect,
	body []byte, committed *bool, canRetry bool, accepted OnAccepted,
) attemptOut {
	// The lease is the only in-flight increment in the program (LB-1), and the
	// deferred release fires after ServeHTTP returns — i.e. after the response
	// body has been fully copied or the stream has aborted (LB-2, LB-4).
	lse := lease.Acquire(b)
	defer lse.Release()

	target, err := url.Parse(b.URL)
	if err != nil {
		return attemptOut{status: 0, err: err, retryable: false}
	}

	var cancelAttempt context.CancelFunc
	// Bound the upstream call. Without this nothing limits a backend that
	// completes the handshake and then goes silent: the goroutine, the connection
	// and — worse — the load lease are held indefinitely, so that backend's
	// in-flight count never returns and every load-based decision is skewed for
	// the life of the process. RequestTimeout was configured and plumbed but never
	// actually read, so DialTimeout was the only outbound bound in the program.
	{
		var ctx context.Context
		if p.cfg.RequestTimeout > 0 {
			ctx, cancelAttempt = context.WithTimeout(r.Context(), p.cfg.RequestTimeout)
		} else {
			ctx, cancelAttempt = context.WithCancel(r.Context())
		}
		defer cancelAttempt()
		r = r.WithContext(ctx)
	}

	aw := &attemptWriter{ResponseWriter: w, committed: committed}
	scanner := d.NewStreamScanner()

	var upstreamStatus int
	var upstreamErr error
	var rejectedForRetry bool

	rp := *p.rp // shallow copy: per-attempt hooks, shared transport and pool
	rp.Rewrite = func(pr *httputil.ProxyRequest) {
		pr.Out.URL.Scheme = target.Scheme
		pr.Out.URL.Host = target.Host
		pr.Out.Host = target.Host
		if body != nil {
			pr.Out.Body = io.NopCloser(bytes.NewReader(body))
			pr.Out.ContentLength = int64(len(body))
		}
		// Never forward the client's credential to a backend (AUTH-9, SEC-4).
		pr.Out.Header.Del("Authorization")
		pr.Out.Header.Del("X-Api-Key")
		if p.cfg.UpstreamCredential != "" {
			pr.Out.Header.Set("Authorization", "Bearer "+p.cfg.UpstreamCredential)
		}
		for _, h := range hopByHop {
			pr.Out.Header.Del(h)
		}
		// Rewrite (not Director) clears inbound X-Forwarded-*, which is the
		// not-trusting-forwarded-headers default we want (SEC-9).
		pr.Out.Header.Set("X-Request-Id", obs.RequestID(r.Context()))
	}
	rp.ModifyResponse = func(resp *http.Response) error {
		upstreamStatus = resp.StatusCode
		if canRetry && !*committed && isRetryable(resp.StatusCode, nil) {
			// Returning an error makes ReverseProxy close the upstream body and
			// call ErrorHandler INSTEAD of copying the response. That is what
			// keeps the response uncommitted so another backend can be tried;
			// letting it through first would fix the answer at this attempt.
			rejectedForRetry = true
			return errRetryable
		}
		// Record that this backend served the prefix — but ONLY on success.
		//
		// `< 500` is not sufficient. 429 and 503 are refusals: the backend did no
		// prefill and cached nothing. Committing on a 429 is actively harmful and
		// self-reinforcing — it teaches the trie that the most overloaded backend
		// holds the prefix, so the next matching request is steered at the very
		// node that is shedding load. 4xx generally means the request was rejected
		// before any KV work happened.
		if accepted != nil && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			accepted(b)
		}
		// Content-Type is passed through untouched. v1 forcibly rewrote it to
		// text/event-stream whenever the request asked to stream, so a JSON error
		// body reached the client as a malformed SSE stream (STR-2, STR-N2).
		sb := &scanBody{rc: resp.Body, sc: scanner, d: d, backend: b.URL}
		if p.cfg.IdleTimeout > 0 && cancelAttempt != nil {
			sb.idleTimeout = p.cfg.IdleTimeout
			sb.idle = time.AfterFunc(p.cfg.IdleTimeout, func() {
				metrics.StreamAborted.WithLabelValues("idle_timeout").Inc()
				cancelAttempt()
			})
		}
		resp.Body = sb
		return nil
	}
	rp.ErrorHandler = func(_ http.ResponseWriter, _ *http.Request, e error) {
		// Deliberately writes nothing: the caller renders the dialect's error
		// envelope, and writing here would both bypass that and commit the
		// response, foreclosing a retry (API-9, REL-3).
		if !errors.Is(e, errRetryable) {
			upstreamErr = e
		}
	}

	rp.ServeHTTP(aw, r)

	out := attemptOut{status: upstreamStatus, err: upstreamErr}
	out.retryable = (rejectedForRetry || isRetryable(upstreamStatus, upstreamErr)) && !*committed
	return out
}

// isRetryable is deliberately narrower than v1's set.
//
// 500 is excluded: it may mean the backend already performed non-idempotent
// work, and replaying generation is expensive. 429 and 503 are retryable here
// but are still recorded as circuit *failures* — retrying elsewhere and telling
// the breaker the backend is fine is precisely v1's bug (REL-2, HLT-N4).
func isRetryable(status int, err error) bool {
	if err != nil {
		return true // connection refused / reset / TLS / timeout before response
	}
	switch status {
	case http.StatusBadGateway, http.StatusServiceUnavailable,
		http.StatusGatewayTimeout, http.StatusRequestTimeout,
		http.StatusTooManyRequests:
		return true
	}
	return false
}

func kindOf(err error) error0 {
	switch {
	case err == nil:
		return "status"
	case errors.Is(err, context.Canceled):
		return "client_cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	default:
		return "transport"
	}
}

// error0 keeps the metric label a closed enum.
type error0 = string

func without(bs []*registry.Backend, drop *registry.Backend) []*registry.Backend {
	out := bs[:0:0]
	for _, b := range bs {
		if b != drop {
			out = append(out, b)
		}
	}
	return out
}

// attemptWriter records whether any byte has reached the client. Once it has,
// retry is forbidden and the response status is fixed (REL-3).
type attemptWriter struct {
	http.ResponseWriter
	committed *bool
}

func (a *attemptWriter) WriteHeader(code int) {
	*a.committed = true
	a.ResponseWriter.WriteHeader(code)
}

func (a *attemptWriter) Write(p []byte) (int, error) {
	*a.committed = true
	return a.ResponseWriter.Write(p)
}

func (a *attemptWriter) Flush() {
	if f, ok := a.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (a *attemptWriter) Unwrap() http.ResponseWriter { return a.ResponseWriter }

// scanBody feeds the response body through the dialect's terminal scanner so an
// abort can be distinguished from a clean end of stream.
//
// The scan is used ONLY for metrics and abort classification. Binding anything
// load-bearing to it would reintroduce API-N3: a hard-coded OpenAI marker never
// fires on an Anthropic stream, and if the lease depended on it the leak would
// look like gradual capacity loss rather than a bug.
type scanBody struct {
	rc      io.ReadCloser
	sc      dialect.StreamScanner
	d       dialect.Dialect
	backend string
	sawEnd  bool

	// tail retains the last maxSniff bytes so usage can be read on Close without
	// buffering the whole response. For a non-streaming reply the body is usually
	// smaller than this, so the tail IS the body; for SSE the final usage frame is
	// by construction at the end.
	tail  []byte
	total int

	// idle cancels the attempt when no body byte arrives for idleTimeout. The
	// overall RequestTimeout bounds the whole call, but a generation that legitimately
	// runs for minutes needs a shorter "stopped producing" signal — otherwise a
	// silent backend holds a goroutine and a load lease for the full request budget.
	idle        *time.Timer
	idleTimeout time.Duration
}

const maxSniff = 64 << 10

func (s *scanBody) Read(p []byte) (int, error) {
	n, err := s.rc.Read(p)
	if s.idle != nil {
		s.idle.Reset(s.idleTimeout)
	}
	if n > 0 {
		s.sawEnd = s.sc.Feed(p[:n])
		s.appendTail(p[:n])
	}
	if err != nil && err != io.EOF && !s.sawEnd {
		metrics.StreamAborted.WithLabelValues("upstream_error").Inc()
	}
	return n, err
}

func (s *scanBody) appendTail(p []byte) {
	s.total += len(p)
	s.tail = append(s.tail, p...)
	if len(s.tail) > maxSniff {
		s.tail = s.tail[len(s.tail)-maxSniff:]
	}
}

// Close extracts the worker's reported cached-token count.
//
// This is the closed loop on prefix-cache prediction: without it
// router_cache_observed_fraction is never emitted and there is no way to tell
// whether the router's guesses correspond to anything real. Note the worker must
// run with --enable-prompt-tokens-details or the field is simply absent, in which
// case nothing is recorded rather than a misleading zero.
func (s *scanBody) Close() error {
	if s.idle != nil {
		s.idle.Stop()
	}
	s.observeUsage()
	return s.rc.Close()
}

func (s *scanBody) observeUsage() {
	if s.d == nil || len(s.tail) == 0 {
		return
	}
	span := s.tail
	// For SSE the usage lives in the last complete `data: {...}` frame.
	if i := bytes.LastIndex(span, []byte("data: ")); i >= 0 {
		span = span[i+len("data: "):]
		if j := bytes.IndexByte(span, '\n'); j >= 0 {
			span = span[:j]
		}
	} else if s.total > len(s.tail) {
		// A non-SSE body larger than the tail buffer: the retained bytes are a
		// fragment, not valid JSON. Skip rather than parse garbage.
		return
	}
	u, ok := s.d.ExtractUsage(span)
	if !ok || u.PromptTokens <= 0 {
		return
	}
	frac := float64(u.CachedTokens) / float64(u.PromptTokens)
	metrics.CacheObservedFraction.Observe(frac)
	metrics.SetObservedShadow(frac)
}
