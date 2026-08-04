package cli

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// RouterServeCommand implements the `wekai router serve` subcommand: a model-aware HTTP
// reverse proxy. Inspects each request's JSON body for the `model` field and
// dispatches to upstreams declared on the command line.
//
// Routing rule syntax (repeatable --route, plus an optional --default):
//
//		"<pattern> => <upstream>[ as <rewrite-model>]"
//
//	  - <pattern>: comma-separated case-insensitive substrings of the requested
//	    model name, or "*" for catch-all.
//	  - <upstream>: scheme://host[:port][/base-path] — the request's path and
//	    query are appended.
//	  - <rewrite-model>: optional. If set, the JSON body's "model" field is
//	    overwritten before forwarding.
//
// First matching --route wins; --default applies if nothing matched.
type RouterServeCommand struct {
	Listen        string        `short:"l" long:"listen" default:":25201" description:"Address to listen on (e.g. :25201 or 127.0.0.1:25201)"`
	Routes        []string      `short:"r" long:"route" description:"Routing rule: '<pattern> => <upstream>[ as <model>]'. Pattern is comma-separated substrings or '*'. Repeat for multiple rules; first match wins."`
	Default       string        `short:"d" long:"default" description:"Catch-all rule: '<upstream>[ as <model>]'. Used when no --route matches."`
	StripAuth     []string      `long:"strip-auth-when" description:"Strip Authorization/x-api-key headers when route pattern matches (comma-separated patterns, repeatable). Useful for unauth'd local upstreams."`
	AutoModel     string        `long:"auto-model" choice:"auto" choice:"off" choice:"force" default:"auto" description:"For routes with no explicit 'as <model>', ask the upstream (GET /v1/models) what it serves and rewrite the request's model to it. 'auto' (default) only rewrites when the upstream serves exactly one model; 'force' always takes the first listed; 'off' disables probing."`
	LogHeaders    bool          `long:"log-headers" description:"Log full request/response headers (verbose, may leak tokens)"`
	Capture       string        `long:"capture" choice:"raw" choice:"redacted" description:"Capture each request/response to JSONL for later analysis or replay. 'raw' saves bodies as-is (auth headers always redacted); 'redacted' replaces bodies with 'REDACTED_TOKENS=<approx>'."`
	CaptureDir    string        `long:"capture-dir" description:"Override capture output directory. Default: ~/.wekai/router/capture/<mode>/"`
	CaptureBuffer int           `long:"capture-buffer" default:"512" description:"Async capture channel buffer size; overflow records are dropped and counted."`
	DrainTimeout  time.Duration `long:"drain-timeout" default:"5m" description:"Grace period on SIGTERM/SIGINT: readiness flips to 503 (pod is removed from Service endpoints) and in-flight requests are allowed to finish. Forced shutdown after this deadline."`
	UserPrefix    bool          `long:"user-prefix" description:"Enable per-user path prefix routing. When set, requests at /<user>/v1/<rest> route as /v1/<rest> upstream and the leading path segment is captured as the request's user. Lets clients distinguish themselves by setting ANTHROPIC_BASE_URL=http://router:port/<user>. Default off — behavior is unchanged unless this is enabled."`
}

type routeRule struct {
	patterns     []string // lowercased substrings; empty means catch-all
	upstream     *url.URL
	rewriteModel string
	stripAuth    bool

	// autoModel holds the model discovered from the upstream's /v1/models when
	// the rule carries no explicit `as <model>` (see command_router_automodel.go).
	// Nil until a probe succeeds, and written by a background retry goroutine
	// while requests are being served — hence the atomic. Read via
	// effectiveRewrite, never directly.
	autoModel atomic.Pointer[string]
}

// effectiveRewrite is the model name to write into the forwarded body, or ""
// to forward the client's own. An explicit `as <model>` always wins over
// discovery: the operator said it, we don't second-guess it.
func (r *routeRule) effectiveRewrite() string {
	if r.rewriteModel != "" {
		return r.rewriteModel
	}
	if m := r.autoModel.Load(); m != nil {
		return *m
	}
	return ""
}

func (r *routeRule) matches(model string) bool {
	if len(r.patterns) == 0 {
		return true
	}
	lower := strings.ToLower(model)
	for _, p := range r.patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func (r *routeRule) describe() string {
	pat := "*"
	if len(r.patterns) > 0 {
		pat = strings.Join(r.patterns, ",")
	}
	out := fmt.Sprintf("%s => %s", pat, r.upstream.String())
	if r.rewriteModel != "" {
		out += " as " + r.rewriteModel
	} else if m := r.autoModel.Load(); m != nil {
		out += " as " + *m + " (auto-discovered)"
	}
	if r.stripAuth {
		out += " (strip-auth)"
	}
	return out
}

func (c *RouterServeCommand) Execute(args []string) error {
	rules, err := c.buildRules()
	if err != nil {
		return err
	}
	if len(rules) == 0 {
		return fmt.Errorf("no routes configured: provide at least one --route or --default")
	}

	stripPatterns := normalizePatternList(c.StripAuth)
	for _, r := range rules {
		if matchesAny(r.patterns, stripPatterns) || (len(r.patterns) == 0 && hasCatchAll(stripPatterns)) {
			r.stripAuth = true
		}
	}

	// Probe before the listener opens so the route lines logged below already
	// name whatever was discovered.
	resolveAutoModels(rules, c.AutoModel)

	handler := &routerHandler{rules: rules, logHeaders: c.LogHeaders, userPrefix: c.UserPrefix}

	if c.Capture != "" {
		dir := resolveCaptureDir(c.CaptureDir, c.Capture)
		sink, err := newCaptureSink(captureMode(c.Capture), dir, c.CaptureBuffer)
		if err != nil {
			return fmt.Errorf("init capture: %w", err)
		}
		handler.capture = sink
		log.Printf("capture: mode=%s dir=%s buffer=%d", c.Capture, dir, c.CaptureBuffer)
	}

	// Drain state: on shutdown signal we flip `draining` so readiness probes
	// fail (k8s removes the pod from Service endpoints) while in-flight
	// requests are tracked by `inFlight` and given DrainTimeout to finish.
	var draining atomic.Bool
	var inFlight sync.WaitGroup

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if draining.Load() {
			http.Error(w, "draining", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok")
	})
	mux.HandleFunc("/livez", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok")
	})
	// /metrics on the same listener — Prometheus PodMonitor scrapes the
	// router's regular port. promhttp uses the global default registerer,
	// which is what the auto-registered counters in
	// command_router_metrics.go write to.
	mux.Handle("/metrics", promhttp.Handler())
	mux.Handle("/", trackInFlight(&inFlight, handler))

	srv := &http.Server{Addr: c.Listen, Handler: mux}

	log.Printf("wekai router listening on %s (drain-timeout=%s)", c.Listen, c.DrainTimeout)
	for _, r := range rules {
		log.Printf("  route: %s", r.describe())
	}

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ListenAndServe() }()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case sig := <-sigCh:
		log.Printf("received %s: draining (timeout=%s) — readiness now reports 503", sig, c.DrainTimeout)
		draining.Store(true)

		done := make(chan struct{})
		go func() { inFlight.Wait(); close(done) }()

		select {
		case <-done:
			log.Printf("in-flight requests drained; shutting down")
		case <-time.After(c.DrainTimeout):
			log.Printf("drain timeout exceeded; forcing shutdown (in-flight requests will be cut)")
		}

		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutCtx); err != nil {
			log.Printf("graceful shutdown error: %v (closing)", err)
			_ = srv.Close()
		}
		<-serveErr
		return nil
	}
}

// trackInFlight counts active proxy requests so the drain path can wait for
// them to finish. Health endpoints are handled elsewhere and are not counted.
func trackInFlight(wg *sync.WaitGroup, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wg.Add(1)
		defer wg.Done()
		next.ServeHTTP(w, r)
	})
}

func (c *RouterServeCommand) buildRules() ([]*routeRule, error) {
	var rules []*routeRule
	for _, spec := range c.Routes {
		r, err := parseRoute(spec)
		if err != nil {
			return nil, fmt.Errorf("invalid --route %q: %w", spec, err)
		}
		rules = append(rules, r)
	}
	if c.Default != "" {
		// Treat --default as "* => <upstream>[ as <model>]"
		r, err := parseRoute("* => " + c.Default)
		if err != nil {
			return nil, fmt.Errorf("invalid --default %q: %w", c.Default, err)
		}
		rules = append(rules, r)
	}
	return rules, nil
}

func parseRoute(spec string) (*routeRule, error) {
	sep := "=>"
	idx := strings.Index(spec, sep)
	if idx < 0 {
		// Allow single '=' as a fallback separator.
		if eq := strings.Index(spec, "="); eq > 0 {
			idx = eq
			sep = "="
		} else {
			return nil, fmt.Errorf("expected '<pattern> => <upstream>'")
		}
	}
	lhs := strings.TrimSpace(spec[:idx])
	rhs := strings.TrimSpace(spec[idx+len(sep):])
	if lhs == "" || rhs == "" {
		return nil, fmt.Errorf("empty pattern or upstream")
	}

	upstream, rewrite := splitAsModel(rhs)
	u, err := url.Parse(upstream)
	if err != nil {
		return nil, fmt.Errorf("bad upstream URL %q: %w", upstream, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("upstream must include scheme and host: %q", upstream)
	}

	var patterns []string
	if lhs != "*" {
		patterns = normalizePatternList([]string{lhs})
	}
	return &routeRule{patterns: patterns, upstream: u, rewriteModel: rewrite}, nil
}

// splitAsModel separates "<upstream> as <model>" (case-insensitive " as ").
func splitAsModel(s string) (upstream, model string) {
	lower := strings.ToLower(s)
	idx := strings.Index(lower, " as ")
	if idx < 0 {
		return strings.TrimSpace(s), ""
	}
	return strings.TrimSpace(s[:idx]), strings.TrimSpace(s[idx+4:])
}

// normalizePatternList flattens comma-separated entries, trims, and lowercases.
// Returns nil if any entry is "*" (signaling catch-all).
func normalizePatternList(specs []string) []string {
	var out []string
	for _, spec := range specs {
		for _, p := range strings.Split(spec, ",") {
			p = strings.TrimSpace(strings.ToLower(p))
			if p == "" {
				continue
			}
			out = append(out, p)
		}
	}
	return out
}

func hasCatchAll(patterns []string) bool {
	for _, p := range patterns {
		if p == "*" {
			return true
		}
	}
	return false
}

func matchesAny(rulePatterns, stripPatterns []string) bool {
	for _, sp := range stripPatterns {
		for _, rp := range rulePatterns {
			if rp == sp {
				return true
			}
		}
	}
	return false
}

// inferencePaths is the whitelist of paths that carry Anthropic inference
// payloads and therefore qualify for per-user routing and redacted capture.
// Non-listed paths use the catch-all (default) rule unchanged and are skipped
// in redacted capture mode. Conservative — prefer false negatives over
// misrouting non-inference traffic through user-prefix logic.
var inferencePaths = map[string]bool{
	"/v1/messages":              true, // Anthropic Messages API
	"/v1/messages/count_tokens": true, // token-count auxiliary endpoint
}

// isInferencePath reports whether path (after user-prefix stripping) is an
// Anthropic inference endpoint qualifying for full per-user routing.
func isInferencePath(path string) bool {
	return inferencePaths[path]
}

type routerHandler struct {
	rules      []*routeRule
	logHeaders bool
	userPrefix bool // when true, extract leading /<user>/ from request paths
	reqCounter atomic.Uint64
	capture    *captureSink
}

// stripUserPrefix strips the leading /<user>/ segment from p and returns
// (user, remainingPath). The caller only invokes this when user-prefix mode
// is configured, in which case every request is prefixed by construction, so
// the first path segment is definitionally the user.
//
// Special cases:
//   - Empty path or path not starting with '/': returned unchanged, user="".
//   - Single-segment path (e.g. /healthz, /metrics, /): returned unchanged,
//     user="". A lone segment cannot be "user + API path" (nothing left to
//     forward), and this keeps infra health probes working.
//
// For all other paths the first segment is taken as the user and the
// remainder becomes the new path.
func stripUserPrefix(p string) (user, newPath string) {
	if len(p) == 0 || p[0] != '/' {
		return "", p
	}
	rest := p[1:]
	slash := strings.IndexByte(rest, '/')
	if slash <= 0 {
		return "", p // single segment — infra probe or bare path, pass through
	}
	first := rest[:slash]
	tail := rest[slash:] // starts with '/'
	return first, tail
}

// Hop-by-hop headers per RFC 7230, plus Anthropic-specific stripping handled
// elsewhere. Stripped from both directions.
var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"TE",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

func (h *routerHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	reqID := h.reqCounter.Add(1)
	started := time.Now()

	// Optional user-prefix extraction: strip the leading /<user>/ segment
	// unconditionally when user-prefix mode is on. By contract every client
	// request carries this prefix in that mode, so the first segment is always
	// the user. Default off (no-op). When on, the user shows up in capture
	// records and log lines, and the upstream sees the canonical path.
	user := ""
	if h.userPrefix {
		user, r.URL.Path = stripUserPrefix(r.URL.Path)
	}

	// Classify the request: inference paths get full per-user model-based routing;
	// everything else is forwarded via the catch-all rule with no user-prefix
	// semantics. Single-segment paths (e.g. HEAD /<user>) are not stripped by
	// stripUserPrefix, so user="" and they land in the catch-all unchanged.
	inference := isInferencePath(r.URL.Path)
	if !inference {
		log.Printf("[#%d] >> non-inference path %s %s (user=%q) — using catch-all rule", reqID, r.Method, r.URL.Path, user)
	}

	// Snapshot inbound headers up front — later we mutate upReq.Header.
	inboundHeaders := r.Header.Clone()

	body, err := io.ReadAll(r.Body)
	_ = r.Body.Close()
	if err != nil {
		log.Printf("[#%d] !! read body: %v", reqID, err)
		http.Error(w, "read body: "+err.Error(), http.StatusBadGateway)
		h.captureError(reqID, started, r, inboundHeaders, body, "", nil, "", 0, err, user, inference)
		return
	}

	var rule *routeRule
	var model string
	if inference {
		model = extractModel(body)
		rule = h.match(model)
		if rule == nil {
			log.Printf("[#%d] !! no route matches model=%q method=%s path=%s", reqID, model, r.Method, r.URL.Path)
			http.Error(w, fmt.Sprintf("no route matches model %q", model), http.StatusBadGateway)
			h.captureError(reqID, started, r, inboundHeaders, body, "", nil, "", 0, fmt.Errorf("no route matches model %q", model), user, inference)
			return
		}
	} else {
		// Non-inference: force catch-all routing (match "" model, first catch-all wins).
		// Any user prefix has already been stripped above; the path here is canonical.
		rule = h.match("")
		if rule == nil {
			log.Printf("[#%d] !! no catch-all route for non-inference path method=%s path=%s", reqID, r.Method, r.URL.Path)
			http.Error(w, "no catch-all route configured", http.StatusBadGateway)
			return
		}
	}

	outBody := body
	rewroteTo := ""
	if rw := rule.effectiveRewrite(); rw != "" && len(body) > 0 {
		if rewritten, ok := rewriteModelField(body, rw); ok {
			outBody = rewritten
			rewroteTo = rw
		}
	}

	target := *rule.upstream
	target.Path = joinPaths(target.Path, r.URL.Path)
	if r.URL.RawQuery != "" {
		if target.RawQuery == "" {
			target.RawQuery = r.URL.RawQuery
		} else {
			target.RawQuery += "&" + r.URL.RawQuery
		}
	}

	upReq, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), bytes.NewReader(outBody))
	if err != nil {
		log.Printf("[#%d] !! build upstream request: %v", reqID, err)
		http.Error(w, "build upstream request: "+err.Error(), http.StatusBadGateway)
		h.captureError(reqID, started, r, inboundHeaders, body, target.String(), rule, rewroteTo, 0, err, user, inference)
		return
	}

	copyHeadersExcept(upReq.Header, r.Header, hopByHopHeaders)
	upReq.Host = target.Host
	upReq.ContentLength = int64(len(outBody))
	upReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(outBody)))

	authNote := "auth=passthrough"
	if rule.stripAuth {
		upReq.Header.Del("Authorization")
		upReq.Header.Del("X-Api-Key")
		upReq.Header.Del("Anthropic-Beta")
		authNote = "auth=STRIPPED"
	}

	rewriteNote := ""
	if rewroteTo != "" {
		rewriteNote = fmt.Sprintf(", model rewritten to %q", rewroteTo)
	}
	userNote := ""
	if user != "" {
		userNote = fmt.Sprintf("  user=%q", user)
	}
	log.Printf("[#%d] >> %s %s  model=%q%s  ROUTE→ %s://%s%s  %s%s",
		reqID, r.Method, r.URL.Path, model, userNote,
		rule.upstream.Scheme, rule.upstream.Host, rule.upstream.Path,
		authNote, rewriteNote)
	if h.logHeaders {
		for k, v := range redactHeaders(upReq.Header) {
			log.Printf("[#%d]    req hdr %s: %v", reqID, k, v)
		}
	}

	resp, err := http.DefaultClient.Do(upReq)
	if err != nil {
		log.Printf("[#%d] !! upstream error after %s: %v", reqID, time.Since(started).Truncate(time.Millisecond), err)
		http.Error(w, "upstream: "+err.Error(), http.StatusBadGateway)
		// 502 is what we send to the client; record it that way in metrics
		// so failures aren't silently invisible on the dashboard.
		metricModelErr := model
		if rewroteTo != "" {
			metricModelErr = rewroteTo
		}
		recordRequestMetric(user, r.Header.Get("X-Claude-Code-Session-Id"), metricModelErr, http.StatusBadGateway, inference)
		h.captureError(reqID, started, r, inboundHeaders, body, target.String(), rule, rewroteTo, 0, err, user, inference)
		return
	}
	defer resp.Body.Close()

	upstreamLatency := time.Since(started)
	copyHeadersExcept(w.Header(), resp.Header, hopByHopHeaders)
	w.WriteHeader(resp.StatusCode)
	log.Printf("[#%d] << %d from %s://%s in %s", reqID, resp.StatusCode, rule.upstream.Scheme, rule.upstream.Host, upstreamLatency.Truncate(time.Millisecond))
	if h.logHeaders {
		for k, v := range redactHeaders(resp.Header) {
			log.Printf("[#%d]    resp hdr %s: %v", reqID, k, v)
		}
	}

	sessionID := r.Header.Get("X-Claude-Code-Session-Id")
	metricModel := model
	if rewroteTo != "" {
		metricModel = rewroteTo
	}

	// respTee needs to be on whenever capture wants the body OR when metrics
	// want to extract token usage from an inference 2xx response. Otherwise
	// allocating a buffer on every request would waste memory.
	captureWantsBody := h.capture != nil && (inference || h.capture.mode != captureRedacted)
	metricsWantsUsage := inference && resp.StatusCode >= 200 && resp.StatusCode < 300

	var respTee *bytes.Buffer
	if captureWantsBody || metricsWantsUsage {
		respTee = &bytes.Buffer{}
	}
	streamCopy(w, resp.Body, respTee)
	total := time.Since(started)
	log.Printf("[#%d] -- done in %s", reqID, total.Truncate(time.Millisecond))

	recordRequestMetric(user, sessionID, metricModel, resp.StatusCode, inference)
	if metricsWantsUsage && respTee != nil {
		plain := respTee.Bytes()
		if ce := resp.Header.Get("Content-Encoding"); ce != "" {
			if decomp, err := decompressBody(plain, ce); err == nil {
				plain = decomp
			}
		}
		var rresp redactedResponse
		parseSSEResponse(plain, &rresp)
		if rresp.Usage != nil {
			recordTokenMetrics(user, sessionID, metricModel,
				rresp.Usage.InputTokens,
				rresp.Usage.CacheReadInputTokens,
				rresp.Usage.CacheCreationInputTokens,
				rresp.Usage.OutputTokens)
		}
	}

	if h.capture != nil {
		// In redacted mode, skip non-inference paths entirely: they have no
		// usage data and produce only noise (404s, 0-token records). Raw mode
		// is the forensic/debug mode and captures everything unchanged.
		if !inference && h.capture.mode == captureRedacted {
			log.Printf("[#%d] capture: skipping non-inference path in redacted mode", reqID)
		} else {
			capHeader := resp.Header.Clone()
			capBody := respTee.Bytes()
			// Client got the compressed response unchanged; for capture we decompress
			// so the stored body is plaintext (Go's json.Marshal would otherwise
			// destroy non-UTF-8 gzip bytes via U+FFFD replacement). Content-Encoding
			// is cleared on the captured headers so the record doesn't lie about
			// the stored body's encoding.
			if ce := capHeader.Get("Content-Encoding"); ce != "" {
				if decomp, err := decompressBody(capBody, ce); err != nil {
					log.Printf("[#%d] capture decompress (%s) failed: %v (storing raw)", reqID, ce, err)
				} else {
					capBody = decomp
					capHeader.Del("Content-Encoding")
				}
			}
			rec := buildCaptureRecord(h.capture.mode, reqID, started, r, inboundHeaders, body, target.String(), rule, rewroteTo, resp.StatusCode, capHeader, capBody, upstreamLatency, total, nil, user)
			h.capture.offer(rec)
		}
	}
}

func (h *routerHandler) captureError(reqID uint64, started time.Time, r *http.Request, inboundHeaders http.Header, body []byte, upstreamURL string, rule *routeRule, rewroteTo string, status int, err error, user string, inference bool) {
	if h.capture == nil {
		return
	}
	// Mirror the same gate as the success path: skip non-inference errors in
	// redacted mode — they're just noise with no usage data.
	if !inference && h.capture.mode == captureRedacted {
		return
	}
	rec := buildCaptureRecord(h.capture.mode, reqID, started, r, inboundHeaders, body, upstreamURL, rule, rewroteTo, status, nil, nil, 0, time.Since(started), err, user)
	h.capture.offer(rec)
}

func (h *routerHandler) match(model string) *routeRule {
	for _, r := range h.rules {
		if r.matches(model) {
			return r
		}
	}
	return nil
}

func extractModel(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var probe struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(body, &probe)
	return probe.Model
}

func rewriteModelField(body []byte, newModel string) ([]byte, bool) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, false
	}
	encoded, err := json.Marshal(newModel)
	if err != nil {
		return nil, false
	}
	raw["model"] = encoded
	out, err := json.Marshal(raw)
	if err != nil {
		return nil, false
	}
	return out, true
}

func copyHeadersExcept(dst, src http.Header, drop []string) {
	dropSet := make(map[string]struct{}, len(drop))
	for _, k := range drop {
		dropSet[strings.ToLower(k)] = struct{}{}
	}
	// Drop anything listed in the inbound Connection header too.
	for _, c := range src.Values("Connection") {
		for _, h := range strings.Split(c, ",") {
			dropSet[strings.ToLower(strings.TrimSpace(h))] = struct{}{}
		}
	}
	for k, vs := range src {
		if _, skip := dropSet[strings.ToLower(k)]; skip {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

func joinPaths(base, extra string) string {
	if extra == "" || extra == "/" {
		return base
	}
	if base == "" {
		return extra
	}
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(extra, "/")
}

func streamCopy(w http.ResponseWriter, src io.Reader, tee *bytes.Buffer) {
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if tee != nil {
				tee.Write(buf[:n])
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

type captureMode string

const (
	captureRaw      captureMode = "raw"
	captureRedacted captureMode = "redacted"
)

type captureRecord struct {
	ID                uint64 `json:"id"`
	Ts                string `json:"ts"`
	Method            string `json:"method"`
	Path              string `json:"path"`
	Query             string `json:"query"`
	UpstreamURL       string `json:"upstream_url"`
	RoutePattern      string `json:"route_pattern"`
	ModelIn           string `json:"model_in"`
	ModelOut          string `json:"model_out,omitempty"`
	Status            int    `json:"status"`
	UpstreamLatencyMs int64  `json:"upstream_latency_ms"`
	TotalMs           int64  `json:"total_ms"`
	Error             string `json:"error,omitempty"`
	// User is set when --user-prefix is enabled and the request path was
	// /<user>/v1/...; the leading segment is stripped before forwarding
	// upstream. Empty otherwise.
	User     string      `json:"user,omitempty"`
	Request  captureBody `json:"request"`
	Response captureBody `json:"response"`
}

type captureBody struct {
	Headers http.Header     `json:"headers"`
	Body    json.RawMessage `json:"body"`
}

type captureSink struct {
	mode    captureMode
	dir     string
	ch      chan *captureRecord
	dropped atomic.Uint64
	mu      sync.Mutex
	file    *os.File
	fileDay string
}

func resolveCaptureDir(override, mode string) string {
	if override != "" {
		if strings.HasPrefix(override, "~/") {
			home, _ := os.UserHomeDir()
			return filepath.Join(home, override[2:])
		}
		return override
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".wekai", "router", "capture", mode)
}

func newCaptureSink(mode captureMode, dir string, buf int) (*captureSink, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	if buf <= 0 {
		buf = 512
	}
	s := &captureSink{mode: mode, dir: dir, ch: make(chan *captureRecord, buf)}
	go s.run()
	return s, nil
}

func (s *captureSink) offer(rec *captureRecord) {
	select {
	case s.ch <- rec:
	default:
		s.dropped.Add(1)
	}
}

func (s *captureSink) run() {
	for rec := range s.ch {
		if err := s.write(rec); err != nil {
			log.Printf("capture write: %v", err)
		}
	}
}

func (s *captureSink) write(rec *captureRecord) error {
	day := time.Now().UTC().Format("2006-01-02")
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.fileDay != day {
		if s.file != nil {
			s.file.Close()
			s.file = nil
		}
		s.fileDay = ""
	}
	if s.file == nil {
		path := filepath.Join(s.dir, "router-"+day+".jsonl")
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		s.file = f
		s.fileDay = day
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = s.file.Write(data)
	return err
}

// sensitiveHeaders are redacted before any capture or log write. Headers are
// still forwarded to the upstream unchanged (unless --strip-auth-when matches)
// — this list only gates what lands in capture JSONL and kubectl logs.
var sensitiveHeaders = []string{
	"authorization",       // Anthropic, OpenAI, Groq, Cohere — "Bearer …"
	"proxy-authorization", // any proxy in front
	"x-api-key",           // Anthropic
	"anthropic-beta",      // Anthropic feature-gating (not a secret, but routinely bundled with keys)
	"x-goog-api-key",      // Gemini
	"openai-organization", // OpenAI
	"openai-project",      // OpenAI
	"cookie",              // session tokens
	"set-cookie",          // server-issued session tokens
}

// sensitiveQueryParams are query-string keys whose values are redacted in
// captures and logs. Gemini's REST API accepts `?key=<api_key>`; generic
// `access_token`/`token` are covered for completeness.
var sensitiveQueryParams = []string{"key", "api_key", "apikey", "access_token", "token"}

func redactHeaders(h http.Header) http.Header {
	if h == nil {
		return nil
	}
	out := make(http.Header, len(h))
	for k, v := range h {
		out[k] = v
	}
	for _, k := range sensitiveHeaders {
		if out.Get(k) != "" {
			out.Set(k, "REDACTED")
		}
	}
	return out
}

// redactQuery returns the raw query with sensitive parameter values replaced
// by "REDACTED". Order and unknown params are preserved.
func redactQuery(raw string) string {
	if raw == "" {
		return raw
	}
	values, err := url.ParseQuery(raw)
	if err != nil {
		// On parse failure, err on the side of safety.
		return "REDACTED"
	}
	redacted := false
	for _, k := range sensitiveQueryParams {
		if _, ok := values[k]; ok {
			values.Set(k, "REDACTED")
			redacted = true
		}
	}
	if !redacted {
		return raw
	}
	return values.Encode()
}

func buildCaptureRecord(mode captureMode, reqID uint64, started time.Time, r *http.Request, inboundHeaders http.Header, reqBody []byte, upstreamURL string, rule *routeRule, rewroteTo string, status int, respHeader http.Header, respBody []byte, upstreamLatency, total time.Duration, callErr error, user string) *captureRecord {
	rec := &captureRecord{
		ID:                reqID,
		Ts:                started.UTC().Format(time.RFC3339Nano),
		Method:            r.Method,
		Path:              r.URL.Path,
		Query:             redactQuery(r.URL.RawQuery),
		UpstreamURL:       upstreamURL,
		RoutePattern:      "*",
		ModelIn:           extractModel(reqBody),
		ModelOut:          rewroteTo,
		Status:            status,
		UpstreamLatencyMs: upstreamLatency.Milliseconds(),
		TotalMs:           total.Milliseconds(),
		User:              user,
	}
	if rule != nil && len(rule.patterns) > 0 {
		rec.RoutePattern = strings.Join(rule.patterns, ",")
	} else if rule == nil {
		rec.RoutePattern = ""
	}
	if callErr != nil {
		rec.Error = callErr.Error()
	}
	// Auth headers are always redacted, even in raw mode — live OAuth tokens
	// must not land on disk. Body redaction is gated on mode.
	// In raw mode, store body as a JSON-quoted string inside json.RawMessage.
	var reqBodyJSON, respBodyJSON json.RawMessage
	if len(reqBody) > 0 {
		reqBodyJSON, _ = json.Marshal(string(reqBody))
	} else {
		reqBodyJSON = json.RawMessage("\"\"")
	}
	if len(respBody) > 0 {
		respBodyJSON, _ = json.Marshal(string(respBody))
	} else {
		respBodyJSON = json.RawMessage("\"\"")
	}
	rec.Request = captureBody{Headers: redactHeaders(inboundHeaders), Body: reqBodyJSON}
	rec.Response = captureBody{Headers: redactHeaders(respHeader), Body: respBodyJSON}
	if mode == captureRedacted {
		applyBodyRedaction(rec)
	}
	return rec
}

// decompressBody undoes the common HTTP Content-Encodings so the capture
// file holds plaintext bodies (Go's json.Marshal would otherwise convert
// invalid UTF-8 bytes to U+FFFD, making compressed SSE streams unrecoverable
// from the capture). Only runs on the capture path — the proxied response
// bytes forwarded to the client are untouched.
func decompressBody(body []byte, encoding string) ([]byte, error) {
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case "", "identity":
		return body, nil
	case "gzip":
		zr, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer zr.Close()
		return io.ReadAll(zr)
	case "deflate":
		zr := flate.NewReader(bytes.NewReader(body))
		defer zr.Close()
		return io.ReadAll(zr)
	default:
		return nil, fmt.Errorf("unsupported Content-Encoding %q", encoding)
	}
}

// applyBodyRedaction replaces body content with structured redacted schemas.
// Shared between the live `--capture redacted` path and the offline
// `router-redact-raw` conversion so the two produce byte-identical output
// for the same inputs.
func applyBodyRedaction(rec *captureRecord) {
	rec.Request.Headers = redactHeaders(rec.Request.Headers)
	rec.Response.Headers = redactHeaders(rec.Response.Headers)

	// Idempotency check: if already redacted with schema, skip
	if len(rec.Request.Body) > 0 {
		var probe map[string]interface{}
		if err := json.Unmarshal(rec.Request.Body, &probe); err == nil {
			if schema, ok := probe["_schema"].(string); ok && (schema == "req-v1" || schema == "resp-v1") {
				// Already redacted, skip
				return
			}
		}
	}

	reqBodyBytes := rawBodyBytes(rec.Request.Body)
	respBodyBytes := rawBodyBytes(rec.Response.Body)
	// Build together so per-block token counts can be allocated from the
	// response's server-reported usage — the offline and live paths share
	// this helper, producing byte-identical output for the same inputs.
	rec.Request.Body, rec.Response.Body = BuildRedactedPair(reqBodyBytes, respBodyBytes)
}

// rawBodyBytes unwraps a raw-mode Body (stored as a JSON-quoted string
// inside json.RawMessage) back to its original []byte. Falls back to
// returning the RawMessage unchanged for inputs that aren't a JSON string.
func rawBodyBytes(body json.RawMessage) []byte {
	if len(body) == 0 {
		return nil
	}
	var s string
	if err := json.Unmarshal(body, &s); err == nil {
		return []byte(s)
	}
	return body
}
