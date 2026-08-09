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
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/dialect/openai"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/health"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/routing"
	"github.com/weka/wekai/router/internal/viz"
)

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

	// Passive skips active health probing. A hosted API has no /health to
	// probe, so probing one means a 404 every interval and a backend that is
	// never marked healthy — health is inferred from real traffic instead.
	// This is the first piece of per-endpoint typing; richer typing (which
	// upstreams expose vLLM metrics, say) belongs here too.
	Passive bool
}

// Options is everything `wekai router serve` needs.
type Options struct {
	Listen        string
	MetricsListen string

	Routes []Route

	// Gateway carries the listener's own behaviour: auth, CORS, body limits.
	Gateway gateway.Config

	// Flow tunes the routing flow shared by every pool — the split guard and
	// which capacity signals are enabled.
	Flow affinity.Config

	Health         health.Config
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

	Log *slog.Logger
}

// withDefaults fills what an operator did not set. These used to live in the
// deleted config loader; they belong here, next to the thing that needs them,
// rather than in a layer everything had to construct before it could start.
//
// The health defaults in particular are not optional: a zero probe interval
// panics the ticker, which is a poor way to learn a field was missed.
func (o Options) withDefaults() Options {
	if o.Health.Interval <= 0 {
		o.Health.Interval = 10 * time.Second
	}
	if o.Health.Timeout <= 0 || o.Health.Timeout >= o.Health.Interval {
		// A timeout at or above the interval lets probes fall behind forever,
		// which is how v1's health state went stale indefinitely (HLT-2).
		o.Health.Timeout = o.Health.Interval / 2
	}
	if o.Health.Path == "" {
		o.Health.Path = "/health"
	}
	if o.Health.FailureThreshold <= 0 {
		o.Health.FailureThreshold = 3
	}
	if o.Health.SuccessThreshold <= 0 {
		o.Health.SuccessThreshold = 2
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
	if o.Gateway.MaxBodyBytes <= 0 {
		o.Gateway.MaxBodyBytes = 64 << 20
	}
	if o.Gateway.DefaultCapacity <= 0 {
		o.Gateway.DefaultCapacity = 1
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

	// One dialect. Registered here, at the wiring layer, rather than as a
	// package side effect, so what is compiled in is explicit (API-3).
	d := openai.New()
	dialect.Register(d)

	clk := clock.Real{}
	var rules []routing.Rule
	var pools []*pool.Pool

	for i, rt := range opts.Routes {
		name := rt.Name
		if name == "" {
			name = poolName(rt.Patterns, i)
		}
		var backends []pool.Backend
		for _, ep := range rt.Endpoints {
			backends = append(backends, pool.Backend{URL: ep, Passive: rt.Passive})
		}
		flow := opts.Flow
		flow.PoolName = name
		flow.Clock = clk

		p, err := pool.New(pool.Config{
			Name:            name,
			Backends:        backends,
			Flow:            flow,
			Health:          opts.Health,
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

	gw := gateway.New(opts.Gateway, tbl, px, d)

	handler := captureMiddleware(opts.Capture, gw)

	for _, p := range pools {
		p.Run(ctx)
	}

	// Metrics and the live KV map on their own listener: diagnostic surface,
	// never reachable on the inference path (GW-13).
	if opts.MetricsListen != "" {
		mux := http.NewServeMux()
		mux.Handle("/metrics", metrics.Handler(metrics.Registry()))
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
		defer func() {
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
			"metrics", opts.MetricsListen, "signals", signalSummary(opts.Flow))
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
