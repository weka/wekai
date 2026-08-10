// Package serve is the router's entrypoint, and the only part of the router
// visible outside it.
//
// It exists because of Go's internal rule: everything under router/internal is
// importable only from router/, so `wekai router serve` — which lives in cli/ —
// cannot wire the gateway, pools and proxy itself. Rather than promote those
// packages to public module API, which is a one-way door and would freeze
// internals that are still moving, this package is the seam. cli/ describes
// what it wants; serve builds it.
//
// The alternative shape, a standalone router binary, is what this replaces.
// There is no reason for two programs: the router IS `wekai router serve`.
package serve

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/dialect/openai"
	k8sdisc "github.com/weka/wekai/router/internal/discovery/k8s"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/health"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/routing"
	"github.com/weka/wekai/router/internal/viz"
	"github.com/weka/wekai/router/internal/vllmmetrics"
)

// registerDialectOnce guards the process-global dialect registry; see Handler.
var registerDialectOnce sync.Once

// Route is one routing rule as an operator writes it.
type Route struct {
	// Patterns are comma-separated substrings of the model name, or "*" /
	// empty for the catch-all a single-fleet router uses.
	Patterns string
	// Endpoints are the interchangeable upstreams serving these models. More
	// than one and the affinity flow routes between them; the pipe-separated
	// form an operator writes (`a|b|c`) is the same syntax the client already
	// accepts for a multi-endpoint model.
	Endpoints []string
	// RewriteModel replaces the request's model before forwarding.
	RewriteModel string
	// StripAuth drops inbound credentials, for an unauthenticated upstream.
	StripAuth bool
	// Name labels the pool in logs and on metrics. Defaults to the patterns,
	// or "default" for the catch-all.
	Name string

	// DiscoverySelector, when set, populates this pool from Kubernetes pods
	// matching a label selector instead of (or alongside) a static endpoint
	// list. Each pod contributes its OWN declared port, which is what a Service
	// cannot express: a fleet run as several DaemonSets, one per GPU topology,
	// commonly listens on a different port per set.
	DiscoverySelector string

	// Passive forces this route's endpoints to skip active health probing.
	//
	// Normally this is DISCOVERED rather than declared: a vLLM-style backend
	// answers GET /v1/models, and one that does is probed actively and gets
	// prefix affinity. Anything that does not — a hosted API behind a
	// different surface — falls back to passive automatically.
	//
	// Set this when discovery would guess wrong, or to skip the probe entirely
	// for an upstream you already know is not vLLM. Anthropic is the case that
	// motivated keeping the override: it answers nothing useful at /v1/models
	// and there is no reason to ask it twice.
	Passive bool
}

// Options is everything `wekai router serve` needs.
type Options struct {
	Listen        string
	MetricsListen string

	Routes []Route

	// --- Listener behaviour.
	APIKey                string
	RequireAuthForProbes  bool
	CORSOrigins           []string
	PathAllowlist         []string
	MaxBodyBytes          int64
	MaxConcurrentRequests int

	// --- The routing flow, shared by every pool.
	NodeConcurrency int64
	RebalanceRatio  float64
	SplitGuard      float64
	TailTTL         time.Duration
	RefusalTTL      time.Duration

	// --- Kubernetes discovery, applied to any route carrying a selector.
	DiscoveryNamespace  string
	DiscoveryPort       int
	DiscoveryPortName   string
	DiscoveryKubeconfig string

	// --- Health probing.
	// HealthInterval is how often a HEALTHY backend is re-probed;
	// HealthUnhealthy how often one that is not. They are asymmetric on
	// purpose — see health.Config.
	HealthInterval  time.Duration
	HealthUnhealthy time.Duration
	HealthTimeout   time.Duration
	HealthPath      string

	// VLLMMetrics turns on upstream vLLM counter aggregation. Only endpoints
	// DISCOVERED to be vLLM are scraped, so a hosted API in the same router is
	// never asked for metrics it does not have — the probe already answered
	// that, once, and the answer is reused rather than rediscovered per cycle.
	VLLMMetrics         bool
	VLLMMetricsInterval time.Duration
	VLLMMetricsNames    []string

	MaxAttempts    int
	RequestTimeout time.Duration
	IdleTimeout    time.Duration
	// UpstreamCredential is presented to upstreams, replacing the client's.
	UpstreamCredential string
	StreamBufferBytes  int
	DrainDeadline      time.Duration

	// Capture, when set, records every proxied exchange. The router's original
	// job — capture traffic for later replay — survives the merge by wrapping
	// the routing gateway rather than being wired through it.
	Capture CaptureHook

	// LogLevel and LogFormat configure the router's own logger when Log is nil.
	LogLevel  string
	LogFormat string
	Log       *slog.Logger
}

// gatewayConfig, flowConfig and healthConfig translate the public Options into
// the internal structs. Options deliberately names no internal type: cli/
// cannot import them, and a facade whose signature leaks what it is hiding
// would not be one.
func (o Options) gatewayConfig() gateway.Config {
	return gateway.Config{
		APIKey:                o.APIKey,
		RequireAuthForProbes:  o.RequireAuthForProbes,
		MaxBodyBytes:          o.MaxBodyBytes,
		MaxConcurrentRequests: o.MaxConcurrentRequests,
		PathAllowlist:         o.PathAllowlist,
		CORSOrigins:           o.CORSOrigins,
		DefaultCapacity:       1,
	}
}

func (o Options) flowConfig() affinity.Config {
	return affinity.Config{
		NodeConcurrency: o.NodeConcurrency,
		RebalanceRatio:  o.RebalanceRatio,
		SplitGuard:      o.SplitGuard,
		TailTTL:         o.TailTTL,
		RefusalTTL:      o.RefusalTTL,
	}
}

func (o Options) healthConfig() health.Config {
	return health.Config{
		Interval: o.HealthInterval, UnhealthyInterval: o.HealthUnhealthy,
		Timeout: o.HealthTimeout, Path: o.HealthPath,
		FailureThreshold: 3, SuccessThreshold: 2,
	}
}

// withDefaults fills what an operator did not set. These used to live in the
// deleted config loader; they belong here, next to the thing that needs them,
// rather than in a layer everything had to construct before it could start.
//
// The health defaults in particular are not optional: a zero probe interval
// panics the ticker, which is a poor way to learn a field was missed.
func (o Options) withDefaults() Options {
	if o.HealthInterval <= 0 {
		o.HealthInterval = 15 * time.Second
	}
	if o.HealthUnhealthy <= 0 || o.HealthUnhealthy > o.HealthInterval {
		o.HealthUnhealthy = min(time.Second, o.HealthInterval)
	}
	if o.HealthTimeout <= 0 || o.HealthTimeout >= o.HealthInterval {
		// A timeout at or above the interval lets probes fall behind forever,
		// which is how v1's health state went stale indefinitely (HLT-2).
		o.HealthTimeout = o.HealthInterval / 2
	}
	if o.HealthPath == "" {
		o.HealthPath = "/health"
	}
	if o.MaxAttempts < 1 {
		o.MaxAttempts = 2
	}
	if o.StreamBufferBytes <= 0 {
		o.StreamBufferBytes = 64 << 10
	}
	if o.DrainDeadline <= 0 {
		o.DrainDeadline = 60 * time.Second
	}
	if o.MaxBodyBytes <= 0 {
		o.MaxBodyBytes = 64 << 20
	}
	return o
}

// Run builds the router and serves until ctx is done.
func Run(ctx context.Context, opts Options) error {
	h, err := Handler(ctx, opts)
	if err != nil {
		return err
	}
	return runServers(ctx, opts, h)
}

// Handler builds the routing handler and starts each pool's background work,
// without owning a listener. Exposed so a test — or an embedder — can drive the
// real router through httptest rather than a port.
func Handler(ctx context.Context, opts Options) (http.Handler, error) {
	log := opts.Log
	if log == nil {
		log = slog.Default()
	}
	if len(opts.Routes) == 0 {
		return nil, errors.New("router: no routes configured; give --route or --backends")
	}
	opts = opts.withDefaults()

	// One dialect, registered at the wiring layer rather than as a package side
	// effect, so what is compiled in is explicit (API-3).
	//
	// Once per process: registration panics on a duplicate, and Handler is
	// callable more than once — a test builds several routers, and an embedder
	// may too. The registry is process-global, so this is the one piece of
	// wiring that cannot be per-router.
	d := openai.New()
	registerDialectOnce.Do(func() { dialect.Register(d) })

	clk := clock.Real{}
	var rules []routing.Rule
	var pools []*pool.Pool
	// Endpoints discovery identified as vLLM instances, and so worth scraping.
	var vllmEndpoints []string

	for i, rt := range opts.Routes {
		name := rt.Name
		if name == "" {
			name = poolName(rt.Patterns, i)
		}
		var backends []pool.Backend
		for _, ep := range rt.Endpoints {
			passive := rt.Passive
			if !passive {
				isVLLM := probeIsVLLM(ctx, ep, log)
				passive = !isVLLM
				if isVLLM {
					vllmEndpoints = append(vllmEndpoints, ep)
				}
			}
			backends = append(backends, pool.Backend{URL: ep, Passive: passive})
		}
		flow := opts.flowConfig()
		flow.PoolName = name
		flow.Clock = clk

		p, err := pool.New(pool.Config{
			Name:            name,
			Backends:        backends,
			Flow:            flow,
			Health:          opts.healthConfig(),
			DefaultCapacity: 1,
			DefaultDialect:  d.ID(),
			DrainDeadline:   opts.DrainDeadline,
			NewGauge: func(url string) registry.Gauge {
				return metrics.BackendInflight.WithLabelValues(url)
			},
		}, clk, log)
		if err != nil {
			return nil, err
		}
		if rt.DiscoverySelector != "" {
			if err := startPodDiscovery(ctx, opts, rt, p, log); err != nil {
				return nil, fmt.Errorf("pool %q discovery: %w", name, err)
			}
		}
		pools = append(pools, p)
		rules = append(rules, routing.Rule{
			Patterns:     routing.NormalizePatterns(rt.Patterns),
			Pool:         p,
			RewriteModel: rt.RewriteModel,
			StripAuth:    rt.StripAuth,
		})
	}

	tbl, err := routing.NewTable(rules)
	if err != nil {
		return nil, err
	}

	px := proxy.New(proxy.Config{
		MaxAttempts:        opts.MaxAttempts,
		UpstreamCredential: opts.UpstreamCredential,
		StreamBufferBytes:  opts.StreamBufferBytes,
		RequestTimeout:     opts.RequestTimeout,
		IdleTimeout:        opts.IdleTimeout,
	})

	gw := gateway.New(opts.gatewayConfig(), tbl, px, d)

	handler := captureMiddleware(opts.Capture, gw)

	for _, p := range pools {
		p.Run(ctx)
	}
	// Fleet load, summarised across every pool so a dashboard does not need a
	// PromQL aggregation just to see whether the fleet is balanced. Computed
	// over AVAILABLE backends only — the same set routing chooses among — so a
	// dead or draining one holding stale load cannot skew it.
	go func() {
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				publishFleetLoad(pools)
			}
		}
	}()

	var agg *vllmmetrics.Aggregator
	if opts.VLLMMetrics && len(vllmEndpoints) > 0 {
		agg = vllmmetrics.New(vllmmetrics.Config{
			Interval: opts.VLLMMetricsInterval,
			Names:    opts.VLLMMetricsNames,
		})
		eps := vllmEndpoints
		log.Info("aggregating upstream vLLM metrics",
			"endpoints", len(eps), "interval", agg.Interval())
		go agg.Run(ctx, func() []string { return eps })
	}

	// Metrics and the live KV map on their own listener: diagnostic surface,
	// never reachable on the inference path (GW-13).
	if opts.MetricsListen != "" {
		mux := http.NewServeMux()
		mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
			metrics.Handler(metrics.Registry()).ServeHTTP(w, r)
			// Upstream totals are appended to the router's own exposition, so
			// one scrape target covers both. They are rendered rather than
			// registered as collectors because their names and labels come from
			// the upstreams, not from anything declared here.
			if agg != nil {
				_ = agg.Render(w)
			}
		})
		mux.HandleFunc("/router-viz", viz.PageHandler())
		// One KV map per pool; the page takes ?pool=, defaulting to the first.
		mux.HandleFunc("/router-viz/data", vizData(pools))
		srv := &http.Server{
			Addr: opts.MetricsListen, Handler: mux,
			ReadHeaderTimeout: 5 * time.Second,
		}
		go func() {
			if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error("metrics listener failed", "err", err)
			}
		}()
		// Shut down on ctx, NOT on a defer here: Handler returns as soon as the
		// handler is built, so a deferred Shutdown would close this listener
		// immediately and /metrics would answer nothing for the process's whole
		// life.
		go func() {
			<-ctx.Done()
			shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = srv.Shutdown(shutCtx)
		}()
	}

	for _, r := range tbl.Rules() {
		log.Info("route", "rule", r.String())
	}

	return handler, nil
}

func runServers(ctx context.Context, opts Options, handler http.Handler) error {
	log := opts.Log
	if log == nil {
		log = slog.Default()
	}
	srv := &http.Server{
		Addr: opts.Listen, Handler: handler,
		ReadHeaderTimeout: 15 * time.Second,
	}
	errCh := make(chan error, 1)
	go func() {
		log.Info("router listening", "listen", opts.Listen,
			"metrics", opts.MetricsListen, "signals", signalSummary(opts.flowConfig()))
		errCh <- srv.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
	}

	// Drain: stop accepting, let in-flight requests finish.
	shutCtx, cancel := context.WithTimeout(context.Background(), opts.DrainDeadline)
	defer cancel()
	return srv.Shutdown(shutCtx)
}

// vizData serves the KV map for one pool, selected by ?pool= and defaulting to
// the first — the only one on a single-pool router.
func vizData(pools []*pool.Pool) http.HandlerFunc {
	byName := map[string]*pool.Pool{}
	for _, p := range pools {
		byName[p.Name] = p
	}
	return func(w http.ResponseWriter, r *http.Request) {
		p := pools[0]
		if name := r.URL.Query().Get("pool"); name != "" {
			got, ok := byName[name]
			if !ok {
				http.Error(w, "unknown pool "+name, http.StatusNotFound)
				return
			}
			p = got
		}
		viz.DataHandler(p.Flow)(w, r)
	}
}

// probeIsVLLM asks an endpoint, ONCE, whether it looks like a vLLM-style
// OpenAI backend: it answers GET /v1/models.
//
// Once, and never again. A failed probe is latched by simply not being
// repeated — there is no retry loop and no ticker. An endpoint that is not
// vLLM will never become vLLM, and a router that kept asking would spend a
// request per interval per endpoint forever to learn the same thing. That
// retry-forever shape is exactly what the benchmark's vLLM metrics sampler
// does today, logging "unavailable — still polling every 1m0s" against an
// endpoint that will never answer.
//
// The consequence of guessing wrong is small and self-correcting in the safe
// direction: a false negative means passive health, so the endpoint is still
// served, its health inferred from real traffic instead of probes. A false
// positive would mean probing /health on something that lacks it, which is why
// the probe asks for the thing vLLM definitely serves rather than sniffing.
// knownHostedProviders are APIs that are never a vLLM instance and never
// expose a liveness path. They are recognised by host so the router does not
// conclude "unreachable" from the two things it cannot do to them: read
// vllm: metrics, and probe /health.
//
// Without this they still end up passive — the probe fails and the fallback is
// correct — but at the cost of two requests over the internet per endpoint at
// startup, and a log line that reads like a problem. Naming them makes the
// intent explicit and the startup silent.
//
// A miss is harmless: an unlisted provider takes the probe path and lands in
// the same place. This is an optimisation and a clarity measure, not a
// correctness dependency, which is why it is a short list rather than an
// exhaustive one.
var knownHostedProviders = []string{
	"api.anthropic.com",
	"api.openai.com",
	"generativelanguage.googleapis.com",
	"api.mistral.ai",
	"api.cohere.ai",
	"api.groq.com",
	"openrouter.ai",
	"api.deepseek.com",
	"api.x.ai",
	"bedrock-runtime",  // AWS Bedrock, region-prefixed
	"openai.azure.com", // Azure OpenAI, tenant-prefixed
}

// IsKnownHostedProvider reports whether the endpoint is a hosted API we already
// know cannot be probed.
func IsKnownHostedProvider(endpoint string) bool {
	u, err := url.Parse(endpoint)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	for _, known := range knownHostedProviders {
		if host == known || strings.Contains(host, known) {
			return true
		}
	}
	return false
}

func probeIsVLLM(ctx context.Context, endpoint string, log *slog.Logger) bool {
	if IsKnownHostedProvider(endpoint) {
		log.Info("known hosted provider: passive health, no probe",
			"endpoint", endpoint)
		return false
	}
	// /metrics FIRST, because it is the question that actually matters and the
	// only one whose answer is unambiguous.
	//
	// Wire format and engine are independent: a vLLM instance may be fronted so
	// it accepts Anthropic-format messages, and it still serves vLLM's own
	// Prometheus metrics at /metrics either way. Deciding "is this vLLM" from
	// which request schema it accepts would therefore get exactly that
	// deployment wrong — it would be classed a hosted API and its metrics left
	// unread. What identifies the engine is the engine's own metric names.
	if body, ok := probeGet(ctx, endpoint+"/metrics", log); ok && strings.Contains(body, "vllm:") {
		return true
	}
	// Otherwise fall back to the OpenAI model listing. This is a weaker signal —
	// plenty of things serve it — but it is enough to decide the only thing
	// riding on this: whether active health probing can work.
	_, ok := probeGet(ctx, endpoint+"/v1/models", log)
	if !ok {
		log.Info("endpoint looks like neither a vLLM instance nor an OpenAI model server; "+
			"treating as passive", "endpoint", endpoint)
	}
	return ok
}

// probeGet performs one bounded GET and returns a prefix of the body. It never
// retries: see probeIsVLLM.
func probeGet(ctx context.Context, url string, log *slog.Logger) (string, bool) {
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, url, nil)
	if err != nil {
		return "", false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", false
	}
	defer resp.Body.Close()
	// Bounded: /metrics on a busy vLLM is large, and this only needs to spot a
	// metric-name prefix near the top.
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<10))
	if resp.StatusCode != http.StatusOK {
		return "", false
	}
	return string(b), true
}

// startPodDiscovery populates a pool from pods matching a label selector.
//
// It only ever PROPOSES backends; the registry decides admission and health
// decides eligibility (SD-4), so a discovered pod is not routed to until it
// passes the same checks a statically configured one does.
func startPodDiscovery(ctx context.Context, opts Options, rt Route, p *pool.Pool, log *slog.Logger) error {
	client, err := k8sdisc.NewInClusterOrKubeconfig(opts.DiscoveryKubeconfig)
	if err != nil {
		return err
	}
	ns := opts.DiscoveryNamespace
	if ns == "" {
		// "Empty means my own namespace" is what an operator expects, and it is
		// the only value that is right by default: a router discovers the fleet
		// it is deployed beside. Kubernetes publishes it to every pod through
		// the service account, which is the same place the token comes from.
		b, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
		if err != nil {
			return fmt.Errorf("no --discover-namespace given and this is not a pod: %w", err)
		}
		ns = strings.TrimSpace(string(b))
	}
	d, err := k8sdisc.New(k8sdisc.Config{
		Mode:            k8sdisc.ModePod,
		Namespace:       ns,
		Selector:        rt.DiscoverySelector,
		Port:            opts.DiscoveryPort,
		PortName:        opts.DiscoveryPortName,
		Scheme:          "http",
		DefaultCapacity: 1,
		DefaultDialect:  "openai",
	}, client, p.Registry, log.With("pool", p.Name))
	if err != nil {
		return err
	}
	go func() {
		if err := d.Run(ctx); err != nil && ctx.Err() == nil {
			log.Error("discovery stopped", "pool", p.Name, "err", err)
		}
	}()
	log.Info("discovering pods by label", "pool", p.Name, "selector", rt.DiscoverySelector,
		"namespace", ns)
	return nil
}

func publishFleetLoad(pools []*pool.Pool) {
	var sum, max, min float64
	n := 0
	for _, p := range pools {
		for _, b := range p.Registry.Snapshot().Backends {
			if !b.Available() {
				continue
			}
			l := b.NormalizedLoad()
			if n == 0 || l > max {
				max = l
			}
			if n == 0 || l < min {
				min = l
			}
			sum += l
			n++
		}
	}
	if n == 0 {
		return
	}
	metrics.WorkerLoadAvg.Set(sum / float64(n))
	metrics.WorkerLoadMax.Set(max)
	metrics.WorkerLoadMin.Set(min)
}

func poolName(patterns string, i int) string {
	p := strings.TrimSpace(patterns)
	if p == "" || p == "*" {
		return affinity.DefaultPoolName
	}
	p = strings.ReplaceAll(p, ",", "-")
	if p == "" {
		return fmt.Sprintf("pool-%d", i)
	}
	return p
}

// signalSummary renders the enabled capacity signals for the startup log. An
// operator could not otherwise tell from the router's own output which are on.
func signalSummary(c affinity.Config) string {
	out := "refused"
	if c.NodeConcurrency > 0 {
		out += fmt.Sprintf(",concurrency=%d", c.NodeConcurrency)
	}
	if c.RebalanceRatio > 0 {
		out += fmt.Sprintf(",imbalance=%.3g", c.RebalanceRatio)
	}
	return out
}
