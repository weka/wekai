// Command wllm-router is an OpenAI-compatible routing gateway for vLLM fleets.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/weka/wekai/router/internal/circuit"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/config"
	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/dialect/openai"
	k8sdisc "github.com/weka/wekai/router/internal/discovery/k8s"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/health"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/obs"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/routing"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/viz"
)

// Build metadata, injected with -ldflags at build time so the running image can
// be identified from its logs and /get_server_info.
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		// Config errors are reported before the logger is configured, so they go
		// to stderr plainly. Note what is NOT printed anywhere: the argv itself.
		// v1 dumped the whole command line — including secrets passed as flags —
		// to stdout before logging existed, where no log level could suppress it
		// (CFG-9, CFG-N2).
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	// -version is handled before config loading so it works with no other flags
	// and in a container with no configuration mounted.
	for _, a := range args {
		if a == "-version" || a == "--version" {
			fmt.Printf("wllm-router %s (commit %s, built %s, %s)\n",
				version, commit, date, runtime.Version())
			return nil
		}
	}

	cfg, err := config.Load(args, os.Getenv)
	if err != nil {
		return err
	}

	// Logging is configured before any other subsystem can emit output (CFG-10).
	log := obs.Init(obs.Config{Level: cfg.LogLevel, Format: cfg.LogFormat})
	if cfg.APIKey == "" {
		log.Warn("no API key configured: the inference listener is unauthenticated",
			"listen", cfg.Listen)
	}

	// One dialect in v2.0. Registration lives here, at the wiring layer, rather
	// than as a package side effect, so what is compiled in is explicit (API-3).
	d := openai.New()
	dialect.Register(d)

	reg := metrics.Registry()
	clk := clock.Real{}

	pol, cachePol, err := buildFlow(cfg, clk)
	if err != nil {
		return err
	}

	opts := registry.Options{
		Clock:         clk,
		DrainDeadline: cfg.DrainDeadline.D(),
		NewGauge: func(url string) registry.Gauge {
			// Resolved once here, not per request (R5).
			return metrics.BackendInflight.WithLabelValues(url)
		},
	}
	if cachePol != nil {
		// Per-backend models follow backend lifecycle, so a backend discovered and
		// later removed does not leak its model, and its prefixes are never
		// reassigned to anyone else (CACHE-10, CU-4, CU-12).
		opts.OnAdd = cachePol.AddBackend
		opts.OnDrop = cachePol.DropBackend
	}
	rtr := registry.New(opts)

	for _, b := range cfg.Backends {
		spec := registry.Spec{
			URL: b.URL, DialectID: orDefault(b.Dialect, d.ID()),
			Prov: registry.ProvStatic, Model: b.Model, Locality: b.Locality,
			Capacity: orDefaultInt(b.Capacity, cfg.MaxInflightPerBackend),
		}
		if b.Kind == "router" {
			spec.Kind = registry.KindRouter
		}
		if b.Health == "passive" {
			spec.Health = registry.HealthPassive
		}
		be, err := rtr.Add(spec)
		if err != nil {
			return fmt.Errorf("backend %q: %w", b.URL, err)
		}
		be.CB.OnTransition(transitionLogger(log, be.URL))
	}

	px := proxy.New(proxy.Config{
		MaxAttempts:        cfg.MaxAttempts,
		UpstreamCredential: cfg.UpstreamCredential,
		StreamBufferBytes:  cfg.StreamBufferBytes,
		RequestTimeout:     cfg.RequestTimeout.D(),
		IdleTimeout:        cfg.IdleTimeout.D(),
	})

	tbl, err := routing.NewTable([]routing.Rule{{Pool: &pool.Pool{
		Name: affinity.DefaultPoolName, Registry: rtr, Flow: cachePol.(*affinity.Policy),
	}}})
	if err != nil {
		return err
	}
	gw := gateway.New(gateway.Config{
		APIKey:                cfg.APIKey,
		RequireAuthForProbes:  cfg.RequireAuthForProbes,
		MaxBodyBytes:          cfg.MaxBodyBytes,
		MaxConcurrentRequests: cfg.MaxConcurrentRequests,
		PathAllowlist:         cfg.PathAllowlist,
		CORSOrigins:           cfg.CORSOrigins,
		DefaultCapacity:       cfg.MaxInflightPerBackend,
	}, tbl, px, d)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Kubernetes discovery, when enabled. It only ever proposes backends; the
	// registry decides admission and health decides eligibility (SD-4).
	if cfg.Discovery.Enabled {
		client, err := k8sdisc.NewInClusterOrKubeconfig(cfg.Discovery.Kubeconfig)
		if err != nil {
			return fmt.Errorf("kubernetes client: %w", err)
		}
		disc, err := k8sdisc.New(k8sdisc.Config{
			Mode:            k8sdisc.Mode(cfg.Discovery.Mode),
			Namespace:       cfg.Discovery.Namespace,
			Service:         cfg.Discovery.Service,
			Selector:        cfg.Discovery.Selector,
			Port:            cfg.Discovery.Port,
			PortName:        cfg.Discovery.PortName,
			Scheme:          cfg.Discovery.Scheme,
			ResyncInterval:  cfg.Discovery.ResyncInterval.D(),
			DefaultCapacity: cfg.MaxInflightPerBackend,
			DefaultDialect:  d.ID(),
		}, client, rtr, log)
		if err != nil {
			return err
		}
		go func() {
			if err := disc.Run(ctx); err != nil && ctx.Err() == nil {
				log.Error("discovery stopped", "err", err)
			}
		}()
	}

	checker := health.New(health.Config{
		Interval: cfg.HealthInterval.D(), Timeout: cfg.HealthTimeout.D(), Path: cfg.HealthPath,
		FailureThreshold: 3, SuccessThreshold: 2,
	}, rtr, clk)
	go checker.Run(ctx)

	if cachePol != nil {
		// Publish per-backend model size on the health interval so cache memory is
		// observable in production, not just in a unit test.
		go func() {
			t := clk.NewTicker(cfg.HealthInterval.D())
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C():
					cachePol.PublishGauges()
				}
			}
		}()
	}

	// TTL eviction runs on its own ticker rather than from the request path, so
	// a routing decision never pays for a sweep. Only prefix-cache-split has a
	// tail set to sweep; the older policies bound their tries by size on insert.
	if sw, ok := cachePol.(sweeper); ok {
		go func() {
			t := clk.NewTicker(sweepInterval(sw.TailTTL()))
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C():
					sw.Sweep()
				}
			}
		}()
	}

	// Publish fleet-wide load stats on the health interval, regardless of
	// policy: router_worker_load_{avg,max,min} answer "how balanced is the
	// fleet right now" without a PromQL aggregation over per-backend series.
	go func() {
		t := clk.NewTicker(cfg.HealthInterval.D())
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C():
				avg, max, min, n := rtr.Snapshot().LoadStats()
				if n == 0 {
					continue
				}
				metrics.WorkerLoadAvg.Set(avg)
				metrics.WorkerLoadMax.Set(max)
				metrics.WorkerLoadMin.Set(min)
			}
		}
	}()

	// Metrics on a separate listener; never reachable on the inference mux
	// (GW-13). Also carries the live KV block map at /router-viz — same
	// listener, same "internal-only, never exposed on the inference path"
	// posture, since it's diagnostic surface, not part of serving traffic.
	// cachePol is nil for a non-cache policy (round-robin/least-outstanding);
	// viz.DataHandler handles a nil DataSource by reporting
	// policy_active:false rather than erroring.
	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", metrics.Handler(reg))
	metricsMux.HandleFunc("/router-viz", viz.PageHandler())
	metricsMux.HandleFunc("/router-viz/data", viz.DataHandler(cachePol))
	metricsSrv := &http.Server{
		Addr:              cfg.MetricsListen,
		Handler:           metricsMux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		if err := metricsSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("metrics listener failed", "err", err)
		}
	}()

	srv := gw.HTTPServer(cfg.Listen)
	errCh := make(chan error, 1)
	go func() {
		// Log the SIGNALS, not a policy name. There is one routing flow, so
		// "policy" said nothing; what actually differs between two deployments
		// is which capacity signals are on and what they are tuned to, and
		// that was previously invisible — an operator could not tell from the
		// router's own output whether --rebalance-ratio was set.
		log.Info("router listening",
			"listen", cfg.Listen, "metrics", cfg.MetricsListen,
			"flow", pol.Name(), "signals", signalSummary(cfg),
			"split_guard", cfg.Cache.SplitGuard,
			"static_backends", len(cfg.Backends),
			"discovery", cfg.Discovery.Enabled, "auth", cfg.APIKey != "",
			"version", version, "commit", commit)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	// Graceful shutdown: in-flight requests, including streams, are allowed to
	// finish up to the drain deadline (REL-8).
	log.Info("shutting down", "drain_deadline", cfg.DrainDeadline.D())
	sdCtx, cancel := context.WithTimeout(context.Background(), cfg.DrainDeadline.D())
	defer cancel()
	_ = metricsSrv.Shutdown(sdCtx)
	if err := srv.Shutdown(sdCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	log.Info("stopped")
	return nil
}

// cacheLifecycle is implemented by every cache-aware policy so the registry
// can drive per-backend model lifecycle without main knowing which cache
// policy (if any) is active. It embeds viz.DataSource for the same reason:
// the /router-viz wiring above needs a narrow interface, not the concrete
// Policy/ThresholdPolicy type, and every cache-aware policy already has to
// implement Snapshot to be usable here at all.
type cacheLifecycle interface {
	AddBackend(*registry.Backend)
	DropBackend(*registry.Backend)
	PublishGauges()
	viz.DataSource
}

// sweeper is implemented by a cache policy that evicts on a timer rather than
// on insert. Detected structurally, like policy.Committer, so wiring does not
// have to name a concrete policy type.
type sweeper interface {
	Sweep() int64
	TailTTL() time.Duration
}

// sweepInterval picks how often to look for expired tails. A fraction of the
// TTL so a run is released reasonably promptly after going idle, clamped so a
// very short TTL cannot spin and a very long one cannot leave the tail set
// unvisited for an hour.
func sweepInterval(ttl time.Duration) time.Duration {
	d := ttl / 10
	return min(max(d, time.Second), time.Minute)
}

// buildFlow builds the router's one routing flow.
//
// There is no policy switch any more. Five other policies used to live here —
// least-outstanding, round-robin, random, prefix-cache-aware,
// prefix-cache-candidates — and the parts of them worth keeping are now signals
// or the selector inside this flow: prefix-cache-aware's load-imbalance spill
// guard became the imbalance signal, prefix-cache-candidates' MaxPending became
// the concurrency signal, and least-outstanding became the tie-break the flow
// uses throughout. The threshold both cache policies gated on is gone
// deliberately: it is the defect that made a long session's fixed shared prefix
// a shrinking fraction of its own growing request.
//
// What varies between deployments is which signals are on, and each is enabled
// by setting its own value rather than by naming it in a list.
func buildFlow(cfg config.Config, clk clock.Clock) (proxy.Selector, cacheLifecycle, error) {
	p, err := affinity.New(affinity.Config{
		NodeConcurrency: cfg.MaxNodeConcurrency,
		RebalanceRatio:  cfg.RebalanceRatio,
		SplitGuard:      cfg.Cache.SplitGuard,
		TailTTL:         cfg.Cache.TailTTL.D(),
		RefusalTTL:      cfg.Cache.RefusalTTL.D(),
		Clock:           clk,
	}, policy.LeastOutstanding{})
	if err != nil {
		return nil, nil, err
	}
	return p, p, nil
}

// signalSummary renders the enabled split signals for the startup log.
// "refused" is always present: a backend's own 429 needs no configuration and
// cannot be turned off.
func signalSummary(cfg config.Config) string {
	out := "refused"
	if cfg.MaxNodeConcurrency > 0 {
		out += fmt.Sprintf(",concurrency=%d", cfg.MaxNodeConcurrency)
	}
	if cfg.RebalanceRatio > 0 {
		out += fmt.Sprintf(",imbalance=%.3g", cfg.RebalanceRatio)
	}
	return out
}

// transitionLogger logs every circuit transition with the counters that caused
// it, so an operator can tell an overload from an outage (HLT-10).
func transitionLogger(log *slog.Logger, url string) circuit.TransitionFunc {
	return func(from, to circuit.State, ok, fail int) {
		log.Warn("circuit breaker transition",
			"backend", url, "from", from.String(), "to", to.String(),
			"window_ok", ok, "window_fail", fail)
		metrics.CircuitTransitions.WithLabelValues(url, from.String(), to.String()).Inc()
	}
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func orDefaultInt(v, def int64) int64 {
	if v == 0 {
		return def
	}
	return v
}
