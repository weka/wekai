package gateway

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/obs"
)

// The middleware chain, outermost first. The ORDER is the design:
//
//	1 recover      REL-9  — must wrap everything, including later middleware
//	2 requestID    GW-7   — must precede logging; adopts the inbound id unchanged
//	3 accessLog    OBS-7  — wraps below everything, so 401/413/508 are all logged
//	4 cors         GW-10  — MUST be OUTSIDE auth (see corsMiddleware)
//	5 bodyLimit    GW-8   — arms MaxBytesReader on EVERY path, reads nothing
//	6 auth         AUTH-4 — the single enforcement site in the binary
//	7 concurrency  REL-10 — inside auth, so a flood cannot consume slots
//	8 mux                 — the matched pattern determines route class and dialect
//
// v1 got 4 and 5 wrong, and enforced 6 twice.
func chain(h http.Handler, ms ...func(http.Handler) http.Handler) http.Handler {
	for i := len(ms) - 1; i >= 0; i-- {
		h = ms[i](h)
	}
	return h
}

// recoverMiddleware converts a handler panic into a 500 rather than taking the
// process down with it (REL-9).
func (s *Server) recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			v := recover()
			if v == nil {
				return
			}
			// http.ErrAbortHandler is not a fault. It is the sentinel net/http
			// uses to abort a response, and httputil.ReverseProxy raises it when
			// it cannot finish copying the body to the client — i.e. the client
			// went away mid-stream. Re-panicking hands it back to net/http, which
			// recovers it and closes the connection without logging a stack trace.
			//
			// Found in a real benchmark: a load driver hitting its timeout
			// cancelled 16 in-flight streams at once, and every one of them landed
			// here as a "panic". That both logged ERROR for a routine disconnect
			// and drove router_panics_total to 16 — turning an alarm metric into
			// something that fires whenever a client hangs up.
			if err, ok := v.(error); ok && errors.Is(err, http.ErrAbortHandler) {
				metrics.ClientDisconnects.Inc()
				obs.Logger(r.Context()).Debug("client disconnected mid-response",
					"path", r.URL.Path)
				panic(v)
			}
			metrics.PanicsTotal.Inc()
			obs.Logger(r.Context()).Error("handler panic recovered",
				"panic", v, "path", r.URL.Path)
			s.dialect.WriteError(w, http.StatusInternalServerError,
				"internal_error", "internal error")
		}()
		next.ServeHTTP(w, r)
	})
}

// requestIDMiddleware adopts a well-formed inbound id or generates one.
//
// The id is never re-minted, so a request that traverses more than one router
// keeps a single trace (GW-7, HIER-10, HIER-N5).
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if !validRequestID(id) {
			id = newRequestID()
		}
		ctx := obs.WithRequestID(r.Context(), id)
		w.Header().Set("X-Request-Id", id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func validRequestID(s string) bool {
	if s == "" || len(s) > 128 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x20 || c > 0x7e {
			return false
		}
	}
	return true
}

func newRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "req-fallback"
	}
	return hex.EncodeToString(b[:])
}

// statusRecorder captures the status for logging and metrics without buffering
// the body.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written bool
	firstAt time.Time
}

func (s *statusRecorder) WriteHeader(code int) {
	if !s.written {
		//clockexempt: latency measurement for a histogram, not a decision
		s.status, s.written, s.firstAt = code, true, time.Now()
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Write(p []byte) (int, error) {
	if !s.written {
		//clockexempt: latency measurement for a histogram, not a decision
		s.status, s.written, s.firstAt = http.StatusOK, true, time.Now()
	}
	return s.ResponseWriter.Write(p)
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *statusRecorder) Unwrap() http.ResponseWriter { return s.ResponseWriter }

// accessLogMiddleware logs and counts EVERY request.
//
// It sits above auth and the body limit so a 401 or a 413 is still recorded. v1's
// catch-all proxy escaped the logging, request-id and tracing layers entirely,
// because those were applied to the mux before the fallback was attached
// (GW-N2, OBS-N3).
func accessLogMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		//clockexempt: access-log and request-duration measurement
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		// Install the holder the handler will populate once the mux has matched.
		rctx, _ := obs.WithRouteHolder(r.Context())
		r = r.WithContext(rctx)
		next.ServeHTTP(rec, r)

		//clockexempt: access-log and request-duration measurement
		dur := time.Since(start)
		ctx := r.Context()
		class := obs.RouteClass(ctx)
		if class == "" {
			class = "unmatched"
		}
		metrics.RequestsTotal.WithLabelValues(class, obs.Dialect(ctx), metrics.StatusClass(rec.status)).Inc()
		metrics.RequestDuration.WithLabelValues(class).Observe(dur.Seconds())
		if !rec.firstAt.IsZero() {
			metrics.TimeToFirstByte.WithLabelValues(class).Observe(rec.firstAt.Sub(start).Seconds())
		}

		lg := obs.Logger(ctx)
		msg := "request"
		switch {
		case isExpectedProbe503(r, rec.status):
			// A readiness probe reporting 503 while the fleet is still loading is
			// the mechanism working, not a fault. Logging it at ERROR produced ~90
			// error lines during one observed 15-minute vLLM model load, which is
			// exactly how operators learn to ignore errors.
			lg.Debug(msg, "method", r.Method, "path", r.URL.Path,
				"status", rec.status, "duration_ms", dur.Milliseconds(),
				"note", "no healthy backend yet")
		case rec.status >= 500:
			lg.Error(msg, "method", r.Method, "path", r.URL.Path,
				"status", rec.status, "duration_ms", dur.Milliseconds())
		case rec.status >= 400:
			lg.Warn(msg, "method", r.Method, "path", r.URL.Path,
				"status", rec.status, "duration_ms", dur.Milliseconds())
		default:
			lg.Info(msg, "method", r.Method, "path", r.URL.Path,
				"status", rec.status, "duration_ms", dur.Milliseconds())
		}
	})
}

// isExpectedProbe503 reports a readiness/health probe that is correctly saying
// "not ready". Only that exact combination is demoted — a 503 on an inference
// path stays an error.
func isExpectedProbe503(r *http.Request, status int) bool {
	if status != http.StatusServiceUnavailable || r.Method != http.MethodGet {
		return false
	}
	return r.URL.Path == "/readiness" || r.URL.Path == "/health"
}

// concurrencyMiddleware caps requests in flight through the gateway.
//
// It sits INSIDE auth so an unauthenticated flood cannot consume slots, and
// inside the body limit so the cap is on requests actually being processed. Each
// in-flight request can hold up to max_body_bytes buffered for retry, so without
// this the memory ceiling is unbounded in practice: at the shipped 64 MiB limit
// roughly eight concurrent uploads reach a 1 GiB container limit.
//
// Shedding with 503 + Retry-After is deliberate: queueing here would just move
// the memory problem behind a queue.
func (s *Server) concurrencyMiddleware(next http.Handler) http.Handler {
	if s.sem == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case s.sem <- struct{}{}:
			defer func() { <-s.sem }()
			next.ServeHTTP(w, r)
		default:
			metrics.RequestsShed.Inc()
			w.Header().Set("Retry-After", "1")
			s.dialect.WriteError(w, http.StatusServiceUnavailable,
				"router_at_capacity", "router is at capacity; retry shortly")
		}
	})
}

// corsMiddleware answers preflight and must sit OUTSIDE auth.
//
// This is GW-10 and it is not a style preference. A browser preflight is an
// OPTIONS request carrying no Authorization header. In v1 the auth layer was
// strictly outer, so every preflight got a 401 before CorsLayer could answer it,
// making any browser client unusable whenever an API key was configured — and
// v1's restricted-origins branch even declared OPTIONS in allow_methods, which
// could never run (GW-N3, AUTH-N5).
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && s.originAllowed(origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key, X-Request-Id")
			w.Header().Set("Access-Control-Max-Age", "600")
		}
		if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
			// Answered here, before auth ever runs.
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) originAllowed(origin string) bool {
	for _, o := range s.cfg.CORSOrigins {
		if o == "*" || strings.EqualFold(o, origin) {
			return true
		}
	}
	return false
}

// bodyLimitMiddleware arms a size limit on every path with no exceptions.
//
// It only wraps the reader; it reads nothing, so placing it before auth cannot be
// used to make the router buffer memory pre-auth. v1's catch-all proxy read its
// body with `to_bytes(body, usize::MAX)` and, because layers were applied before
// the fallback was registered, escaped the limit entirely — an unbounded-memory
// DoS on any unmatched path (GW-8, GW-9, GW-N1).
func (s *Server) bodyLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxBodyBytes)
		}
		next.ServeHTTP(w, r)
	})
}

// authMiddleware is the ONE enforcement site.
//
// v1 layered a global auth middleware and *also* called authorize_request at the
// top of all ~28 handlers. Two mechanisms for one invariant is two things to
// drift; hack/ asserts this stays single (AUTH-4, AUTH-N1).
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.pathAllowed(r) {
			// 404, not 403: an unlisted path must not reveal whether a
			// credential would have been accepted.
			http.NotFound(w, r)
			return
		}
		if s.apiKey == nil {
			// Auth disabled: everything is "authenticated" for disclosure purposes.
			next.ServeHTTP(w, r.WithContext(withAuthed(r.Context(), true)))
			return
		}
		authed := s.credentialValid(r)
		if s.isPublicProbe(r) {
			// Admitted without a credential, but the handler is told whether one was
			// actually presented so it can disclose the minimum.
			next.ServeHTTP(w, r.WithContext(withAuthed(r.Context(), authed)))
			return
		}
		if !authed {
			// Never log the presented credential, not even a prefix (AUTH-10).
			obs.Logger(r.Context()).Warn("authentication failed", "path", r.URL.Path)
			s.dialect.WriteError(w, http.StatusUnauthorized,
				"invalid_api_key", "invalid or missing API key")
			return
		}
		next.ServeHTTP(w, r.WithContext(withAuthed(r.Context(), true)))
	})
}

type authedKey struct{}

func withAuthed(ctx context.Context, ok bool) context.Context {
	return context.WithValue(ctx, authedKey{}, ok)
}

// Authed reports whether the caller presented a valid credential. Probes may be
// admitted without one, so handlers use this to decide how much to disclose.
func Authed(ctx context.Context) bool {
	v, _ := ctx.Value(authedKey{}).(bool)
	return v
}

// credentialValid is the ONE place a presented credential is checked.
//
// Keeping it to a single call site is AUTH-4, and it is enforced mechanically by
// hack/fence_test.go rather than by this comment — which matters, because the
// probe-disclosure change briefly introduced a second comparison here and the
// fence is what caught it.
func (s *Server) credentialValid(r *http.Request) bool {
	tok, ok := s.dialect.Credential(r.Header)
	if !ok {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(tok), s.apiKey) == 1
}

// isPublicProbe decides which endpoints a kubelet may reach without a credential.
//
// GET /liveness is always public: it reports only that the process is up (AUTH-5).
// GET /readiness and GET /health are public unless require_auth_for_probes is set,
// because a kubelet probe cannot authenticate and a 401 there means the pod never
// becomes Ready (AUTH-6).
func (s *Server) isPublicProbe(r *http.Request) bool {
	if r.Method != http.MethodGet {
		return false
	}
	switch r.URL.Path {
	case "/liveness":
		return true
	case "/readiness", "/health":
		return !s.cfg.RequireAuthForProbes
	}
	return false
}

// pathAllowed applies the allowlist with segment-boundary matching.
//
// An entry ending in "/" denotes a subtree; otherwise the match is exact or a
// path-segment child. So /v1/responses admits /v1/responses/abc but rejects
// /v1/responses_evil, and /v1/mod never matches /v1/models (AUTH-7, AUTH-N3).
//
// An empty allowlist means NO exemptions, i.e. everything is subject to auth —
// the inverse of v1, where empty meant allow-all and a typo opened the router.
func (s *Server) pathAllowed(r *http.Request) bool {
	if len(s.cfg.PathAllowlist) == 0 {
		return true // no restriction configured; auth still applies
	}
	if s.isPublicProbe(r) {
		return true
	}
	p := r.URL.Path
	for _, entry := range s.cfg.PathAllowlist {
		if strings.HasSuffix(entry, "/") {
			if strings.HasPrefix(p, entry) {
				return true
			}
			continue
		}
		if p == entry {
			return true
		}
		if strings.HasPrefix(p, entry) && len(p) > len(entry) && p[len(entry)] == '/' {
			return true
		}
	}
	// Admin paths are never exemptible by the allowlist, but they are also not
	// admitted by it: they require auth like everything else (AUTH-11).
	return false
}
