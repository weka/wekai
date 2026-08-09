// Package pool bundles everything needed to route across ONE set of
// interchangeable endpoints: the registry that holds them, the health checker
// that decides which are eligible, the affinity flow that chooses among them,
// and the optional Kubernetes discovery that populates them.
//
// This is the unit the router was missing. `wllm-router` had exactly one of
// these, implicit in its main() wiring, which is why it could front a fleet but
// not a fleet PER MODEL. `wekai router serve` had the opposite shape: many
// routes, but each pointing at a single URL with no health, no affinity and no
// failover. A pool is what lets one router do both — a route names a pool, and
// inside it the endpoints are interchangeable in exactly the way the client's
// pipe-separated `a|b|c` syntax already implies.
//
// A pool owns its own tree, so two pools never share prefix state. That is
// required rather than convenient: two models' KV caches are unrelated, and
// crediting one for the other's prompt would be the same defect the tree's
// per-model root sentinels exist to prevent, one level up.
package pool

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/health"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/registry"
)

// Backend is one statically configured endpoint.
type Backend struct {
	URL      string
	Model    string
	Locality string
	Dialect  string
	Capacity int64
	// Passive skips active health probing for endpoints that have no health
	// path — a hosted API, say. Health is then inferred from traffic alone.
	Passive bool
	// Router marks an endpoint that is itself a router, so this one does not
	// try to reason about its cache.
	Router bool
}

// Config describes one pool.
type Config struct {
	// Name identifies the pool in logs and in the `pool` label on metrics. It
	// must be unique across a router; the implicit whole-router pool is
	// "default".
	Name string

	Backends []Backend

	// Flow configures the routing flow — the split guard and which signals are
	// enabled. Zero value is valid: the refused signal alone.
	Flow affinity.Config

	Health health.Config

	// DefaultCapacity applies to backends that do not carry their own, and is
	// what Backend.Capacity falls back to.
	DefaultCapacity int64

	// DefaultDialect is the dialect ID for backends that do not name one.
	DefaultDialect string

	DrainDeadline time.Duration

	// NewGauge resolves a backend's in-flight gauge once, at add time, rather
	// than per request (R5).
	NewGauge func(url string) registry.Gauge

	// OnCircuitTransition, when set, is attached to every backend as it joins,
	// so an operator can tell an overload from an outage.
	OnCircuitTransition func(url string) func(from, to any, ok, fail int)
}

// Pool is one routable set of endpoints.
type Pool struct {
	Name     string
	Registry *registry.Registry
	Flow     *affinity.Policy

	checker *health.Checker
	log     *slog.Logger
}

// New builds a pool and admits its static backends. It does not start any
// background work; call Run for that.
func New(cfg Config, clk clock.Clock, log *slog.Logger) (*Pool, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("pool: name is required")
	}
	flow, err := affinity.New(cfg.Flow, policy.LeastOutstanding{})
	if err != nil {
		return nil, fmt.Errorf("pool %q: %w", cfg.Name, err)
	}

	opts := registry.Options{
		Clock:         clk,
		DrainDeadline: cfg.DrainDeadline,
		NewGauge:      cfg.NewGauge,
		// Per-backend prefix state follows backend lifecycle, so a backend that
		// is discovered and later removed does not leak its marks, and its
		// prefixes are never reassigned to whoever takes its place
		// (CACHE-10, CU-4, CU-12).
		OnAdd:  flow.AddBackend,
		OnDrop: flow.DropBackend,
	}
	reg := registry.New(opts)

	p := &Pool{
		Name:     cfg.Name,
		Registry: reg,
		Flow:     flow,
		log:      log.With("pool", cfg.Name),
	}

	for _, b := range cfg.Backends {
		spec := registry.Spec{
			URL:       b.URL,
			DialectID: orDefault(b.Dialect, cfg.DefaultDialect),
			Prov:      registry.ProvStatic,
			Model:     b.Model,
			Locality:  b.Locality,
			Capacity:  orDefaultInt(b.Capacity, cfg.DefaultCapacity),
		}
		if b.Router {
			spec.Kind = registry.KindRouter
		}
		if b.Passive {
			spec.Health = registry.HealthPassive
		}
		be, err := reg.Add(spec)
		if err != nil {
			return nil, fmt.Errorf("pool %q backend %q: %w", cfg.Name, b.URL, err)
		}
		if b.Passive {
			// A passive backend is never probed — its health comes solely from
			// proxied request outcomes (HLT-12). It must therefore start
			// ELIGIBLE, or it waits forever for a check that never runs. This
			// is what a hosted API needs: there is no /health to probe, and
			// probing one means a 404 every interval and a backend that is
			// never usable.
			be.SetHealth(registry.Healthy)
		}
	}

	p.checker = health.New(cfg.Health, reg, clk)
	return p, nil
}

// Run starts the pool's background work: health probing, plus the flow's own
// tail-eviction sweep and gauge publication. It returns when ctx is done.
//
// Eviction and gauge publication run on their own tickers rather than from the
// request path, so a routing decision never pays for either.
func (p *Pool) Run(ctx context.Context) {
	go p.checker.Run(ctx)

	go p.tick(ctx, sweepInterval(p.Flow.TailTTL()), func() { p.Flow.Sweep() })
	go p.tick(ctx, gaugeInterval, p.Flow.PublishGauges)
}

func (p *Pool) tick(ctx context.Context, every time.Duration, fn func()) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			fn()
		}
	}
}

// gaugeInterval is how often tree size and duplication are republished. It is
// deliberately not the health interval: those gauges walk the forest per
// backend, and that cost should not be tied to how aggressively health is
// probed.
const gaugeInterval = 10 * time.Second

// sweepInterval keeps eviction responsive relative to the TTL without spinning:
// a tenth of the TTL, clamped so a very short TTL cannot busy-loop and a very
// long one still gets swept within the minute.
func sweepInterval(ttl time.Duration) time.Duration {
	d := ttl / 10
	return min(max(d, time.Second), time.Minute)
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
