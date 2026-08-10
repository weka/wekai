package cli

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"encoding/json"
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

	"github.com/weka/wekai/router/serve"
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

	// --- Fleet routing. A --route with several pipe-separated endpoints, or
	// --backends, turns on prefix-cache affinity across them.
	Backends       []string      `long:"backends" description:"Endpoints for the implicit catch-all pool, comma- or pipe-separated. Shorthand for --route '* => a|b|c' — the simplest router: route every model to this set."`
	Passive        bool          `long:"passive-health" description:"Skip active health probing. Required for upstreams with no /health endpoint (a hosted API); their health is inferred from proxied request outcomes."`
	MetricsListen  string        `long:"metrics-listen" default:"127.0.0.1:29000" description:"Address for /metrics and the live KV map at /router-viz. Separate from the inference listener: diagnostic surface is never reachable on the serving path. Empty disables it."`
	MaxNodeConc    int64         `long:"max-node-concurrency" description:"Enables the concurrency split signal: treat a backend at or above this many in-flight requests as saturated without waiting for it to say so. Set it to the backends' vLLM --max-num-seqs. 0 = off; the backend's own 429 remains the ultimate signal either way."`
	RebalanceRatio float64       `long:"rebalance-ratio" description:"Enables the imbalance split signal: a backend is saturated while (inflight - fleetMin)/inflight exceeds this. 0.5 rebalances once the gap is more than half the busier side. 0 = off — a fleet where affinity works is supposed to look imbalanced."`
	SplitGuard     float64       `long:"cache-split-guard" default:"0.20" description:"A prefix is split onto a backend only while its in-flight is below limit*(1-this). Higher keeps the holder set tighter at the cost of splitting less readily; too low and every backend ends up holding every prefix."`
	TailTTL        time.Duration `long:"cache-tail-ttl" default:"5m" description:"How long a leaf of the shared prefix tree may go untouched before eviction. Memory pressure only: eviction never removes a run that still has children."`
	RefusalTTL     time.Duration `long:"cache-refusal-ttl" default:"2s" description:"How long a backend's own 429 keeps it out of its prefixes. Cleared early by any success from it, and by its in-flight dropping below the level it refused at."`

	// --- Listener behaviour.
	APIKeyFile         string        `long:"api-key-file" description:"Read the inbound API key from this file. PREFER this to --api-key: a key given as a flag is visible in a process listing and in the pod spec that launched it."`
	APIKey             string        `long:"api-key" description:"Require this key on inference and admin requests. Empty leaves the listener unauthenticated, which is logged loudly at startup."`
	CORSOrigins        []string      `long:"cors-origins" description:"Origins permitted to call the inference listener. '*' cannot be combined with --api-key."`
	PathAllowlist      []string      `long:"path-allowlist" description:"Restrict which upstream paths may be proxied. Empty allows every path."`
	MaxBodyBytes       int64         `long:"max-body-bytes" default:"67108864" description:"Maximum request body. Bodies are buffered whole so a retry can replay them, so this is the real per-request memory bound."`
	MaxConcurrent      int           `long:"max-concurrent-requests" default:"256" description:"Router-wide in-flight cap protecting the router's own memory; sheds 503 router_at_capacity. Distinct from per-backend capacity, which sheds 429. 0 disables."`
	MaxAttempts        int           `long:"max-attempts" default:"2" description:"Upstream attempts including the first, after a FAILURE. A 429 does not spend this budget: refusals draw on their own, bounded by the number of endpoints."`
	RequestTimeout     time.Duration `long:"request-timeout" default:"600s" description:"Overall upstream request deadline."`
	IdleTimeout        time.Duration `long:"idle-timeout" default:"300s" description:"Abort a stream that has produced nothing for this long."`
	UpstreamCred       string        `long:"upstream-credential" description:"Credential presented to upstreams, replacing the client's."`
	HealthInterval     time.Duration `long:"health-interval" default:"15s" description:"How often a HEALTHY backend is re-probed. Can be slow: a healthy backend that breaks is caught by real traffic failing, since an upstream 5xx trips its circuit and removes it from selection."`
	DiscoverNamespace  string        `long:"discover-namespace" description:"Namespace searched by pods:<selector> routes. Empty means the router's own namespace in-cluster."`
	DiscoverPort       int           `long:"discover-port" default:"8000" description:"Port used for a discovered pod that declares none of its own. Each pod's declared containerPort wins over this, which is what lets several DaemonSets on different ports be one pool."`
	DiscoverPortName   string        `long:"discover-port-name" description:"Name of the containerPort to use when a pod declares several, e.g. 'http'."`
	DiscoverKubeconfig string        `long:"discover-kubeconfig" description:"Kubeconfig path for out-of-cluster discovery. In-cluster credentials are used when empty."`
	HealthUnhealthy    time.Duration `long:"health-unhealthy-interval" default:"1s" description:"How often a backend that is NOT healthy is re-probed. Much shorter on purpose: a recovered backend stays out of rotation for this long, and every request that could have used its warm cache goes somewhere colder."`
	HealthTimeout      time.Duration `long:"health-timeout" default:"5s" description:"Health probe timeout. Must be below the interval, or probes fall behind forever."`
	HealthPath         string        `long:"health-path" default:"/health" description:"Path probed on each backend."`
	VLLMMetrics        bool          `long:"vllm-metrics" description:"Aggregate upstream vLLM counters into router-level totals served on --metrics-listen. Only endpoints discovered to be vLLM are scraped. Totals accumulate DELTAS, so they never go backwards when a pod restarts or the fleet scales down — a decreasing counter silently breaks rate() and increase()."`
	VLLMMetricsEvery   time.Duration `long:"vllm-metrics-interval" default:"30s" description:"How often each discovered vLLM endpoint is scraped."`
	VLLMMetricsNames   []string      `long:"vllm-metrics-name" description:"Upstream counter to aggregate; repeatable. Default: vllm:prompt_tokens_by_source_total. Re-exporting every series would multiply cardinality by the fleet size."`
	LogLevel           string        `long:"log-level" default:"info" description:"debug, info, warn or error."`
	LogFormat          string        `long:"log-format" choice:"json" choice:"text" default:"json" description:"Log output format."`
}

type routeRule struct {
	patterns []string // lowercased substrings; empty means catch-all
	// endpoints are the interchangeable upstreams serving these models. One is
	// a plain proxy; several and the affinity flow routes between them. The
	// pipe-separated form is the same syntax the client already accepts for a
	// multi-endpoint model, so `a|b|c` means the same thing on both sides.
	endpoints []string
	// discoverSelector populates the pool from Kubernetes pods by label,
	// written as an endpoint of the form pods:<selector>.
	discoverSelector string
	rewriteModel     string
	// credentialFile holds the router's own key for this pool, from
	// "<upstream> using <file>".
	credentialFile string
	// forwardClient passes the caller's own credential upstream ("using client").
	forwardClient bool
	stripAuth     bool
	// passive skips active health probing, for an upstream with no /health —
	// a hosted API. Its health is inferred from proxied request outcomes.
	passive bool

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
	out := fmt.Sprintf("%s => %s", pat, strings.Join(r.endpoints, "|"))
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
		return fmt.Errorf("no routes configured: provide --backends, --route or --default")
	}

	stripPatterns := normalizePatternList(c.StripAuth)
	for _, r := range rules {
		if matchesAny(r.patterns, stripPatterns) || (len(r.patterns) == 0 && hasCatchAll(stripPatterns)) {
			r.stripAuth = true
		}
	}

	var hook serve.CaptureHook
	if c.Capture != "" {
		dir := resolveCaptureDir(c.CaptureDir, c.Capture)
		sink, err := newCaptureSink(captureMode(c.Capture), dir, c.CaptureBuffer)
		if err != nil {
			return fmt.Errorf("init capture: %w", err)
		}
		hook = &captureAdapter{sink: sink, userPrefix: c.UserPrefix}
		log.Printf("capture: mode=%s dir=%s buffer=%d", c.Capture, dir, c.CaptureBuffer)
	}

	routes := make([]serve.Route, 0, len(rules))
	for _, r := range rules {
		pat := "*"
		if len(r.patterns) > 0 {
			pat = strings.Join(r.patterns, ",")
		}
		routes = append(routes, serve.Route{
			Patterns:                pat,
			Endpoints:               r.endpoints,
			DiscoverySelector:       r.discoverSelector,
			CredentialFile:          r.credentialFile,
			ForwardClientCredential: r.forwardClient,
			RewriteModel:            r.rewriteModel,
			StripAuth:               r.stripAuth,
			Passive:                 c.Passive,
		})
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	return serve.Run(ctx, serve.Options{
		Listen:                c.Listen,
		MetricsListen:         c.MetricsListen,
		Routes:                routes,
		APIKey:                c.APIKey,
		APIKeyFile:            c.APIKeyFile,
		CORSOrigins:           c.CORSOrigins,
		PathAllowlist:         c.PathAllowlist,
		MaxBodyBytes:          c.MaxBodyBytes,
		MaxConcurrentRequests: c.MaxConcurrent,
		NodeConcurrency:       c.MaxNodeConc,
		RebalanceRatio:        c.RebalanceRatio,
		AutoModel:             c.AutoModel,
		SplitGuard:            c.SplitGuard,
		TailTTL:               c.TailTTL,
		RefusalTTL:            c.RefusalTTL,
		DiscoveryNamespace:    c.DiscoverNamespace,
		DiscoveryPort:         c.DiscoverPort,
		DiscoveryPortName:     c.DiscoverPortName,
		DiscoveryKubeconfig:   c.DiscoverKubeconfig,
		HealthInterval:        c.HealthInterval,
		HealthUnhealthy:       c.HealthUnhealthy,
		HealthTimeout:         c.HealthTimeout,
		HealthPath:            c.HealthPath,
		VLLMMetrics:           c.VLLMMetrics,
		VLLMMetricsInterval:   c.VLLMMetricsEvery,
		VLLMMetricsNames:      c.VLLMMetricsNames,
		MaxAttempts:           c.MaxAttempts,
		RequestTimeout:        c.RequestTimeout,
		IdleTimeout:           c.IdleTimeout,
		UpstreamCredential:    c.UpstreamCred,
		DrainDeadline:         c.DrainTimeout,
		Capture:               hook,
		LogLevel:              c.LogLevel,
		LogFormat:             c.LogFormat,
	})
}

// captureAdapter turns serve's routing-outcome view into the capture record
// schema the replay tooling reads. serve knows nothing about that schema, and
// this file knows nothing about routing — which is the point of the seam.
type captureAdapter struct {
	sink       *captureSink
	userPrefix bool
}

func (a *captureAdapter) WantsResponseBody() bool { return a.sink.mode != captureRedacted }

func (a *captureAdapter) Record(ev serve.Captured) {
	user := ""
	if a.userPrefix {
		user, _ = stripUserPrefix(ev.Request.URL.Path)
	}
	rec := buildCaptureRecord(a.sink.mode, ev.ID, ev.Started, ev.Request,
		ev.InboundHeaders, ev.ReqBody, ev.Backend, ev.Pool, ev.ModelOut,
		ev.Status, ev.RespHeaders, ev.RespBody, ev.Total, ev.Total, nil, user)
	if rec != nil {
		a.sink.offer(rec)
	}
}

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
	// --backends is the simplest form: route every model to this set. It is
	// shorthand for "* => a|b|c", which is why it produces a rule rather than a
	// special case downstream.
	if len(c.Backends) > 0 {
		joined := strings.Join(c.Backends, "|")
		joined = strings.ReplaceAll(joined, ",", "|")
		r, err := parseRoute("* => " + joined)
		if err != nil {
			return nil, fmt.Errorf("invalid --backends %q: %w", joined, err)
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

	upstreams, credRef := splitUsing(rhs)
	forwardClient := strings.EqualFold(credRef, "client")
	credFile := credRef
	if forwardClient {
		credFile = ""
	}
	upstreams, rewrite := splitAsModel(upstreams)
	var endpoints []string
	var selector string
	for _, raw := range strings.Split(upstreams, "|") {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		// pods:<selector> populates the pool from Kubernetes instead of naming
		// endpoints. It is an endpoint position rather than a separate flag so
		// one route still describes one pool, and so a pool can mix discovered
		// pods with statically named ones during a migration.
		if rest, ok := strings.CutPrefix(raw, "pods:"); ok {
			selector = strings.TrimSpace(rest)
			continue
		}
		u, err := url.Parse(raw)
		if err != nil {
			return nil, fmt.Errorf("bad upstream URL %q: %w", raw, err)
		}
		if u.Scheme == "" || u.Host == "" {
			return nil, fmt.Errorf("upstream must include scheme and host: %q", raw)
		}
		endpoints = append(endpoints, strings.TrimRight(u.String(), "/"))
	}
	if len(endpoints) == 0 && selector == "" {
		return nil, fmt.Errorf("no upstream endpoints")
	}

	var patterns []string
	if lhs != "*" {
		patterns = normalizePatternList([]string{lhs})
	}
	return &routeRule{patterns: patterns, endpoints: endpoints,
		discoverSelector: selector, rewriteModel: rewrite,
		credentialFile: credFile, forwardClient: forwardClient}, nil
}

// splitUsing separates "<upstream> using <ref>", how this pool authenticates.
//
//	using /path/to/secret   the ROUTER's own key, read from a mounted file — a
//	                        path rather than a value so the secret never appears
//	                        in a process listing or a pod spec
//	using client            forward the CALLER's credential, which a hosted API
//	                        the user pays for requires
//
// Neither means send no credential at all, which is the safe default: a user's
// key reaching an internal backend has to take saying so out loud.
func splitUsing(s string) (rest, credFile string) {
	lower := strings.ToLower(s)
	idx := strings.LastIndex(lower, " using ")
	if idx < 0 {
		return strings.TrimSpace(s), ""
	}
	return strings.TrimSpace(s[:idx]), strings.TrimSpace(s[idx+7:])
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

func buildCaptureRecord(mode captureMode, reqID uint64, started time.Time, r *http.Request, inboundHeaders http.Header, reqBody []byte, upstreamURL string, routePattern string, rewroteTo string, status int, respHeader http.Header, respBody []byte, upstreamLatency, total time.Duration, callErr error, user string) *captureRecord {
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
	rec.RoutePattern = routePattern
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
