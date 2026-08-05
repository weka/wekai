// Package health runs active health checks against backends.
//
// Two v1 defects are fixed structurally:
//
//   - Checks run CONCURRENTLY. v1 awaited each check in a for loop, so with a
//     5s timeout and N unreachable workers a single round took 5N seconds and
//     could exceed the 60s interval entirely — checks silently fell behind and
//     stale health persisted indefinitely (HLT-1, HLT-N1).
//   - The checker NEVER touches in-flight load. v1's registry checker called
//     reset_load() on every worker every 10 cycles, with no guard, destroying
//     the signal every load-based decision depended on (HLT-4, HLT-N5). The
//     hack/ fence makes that impossible here.
package health

import (
	"context"
	"net/http"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

type Config struct {
	Interval time.Duration
	Timeout  time.Duration
	Path     string
	// FailureThreshold consecutive failures mark a backend unhealthy;
	// SuccessThreshold consecutive successes mark it healthy (HLT-3).
	FailureThreshold int
	SuccessThreshold int
	// Concurrency bounds simultaneous checks. It defaults to
	// min(256, max(32, N)) per round: a small fixed pool cannot satisfy the
	// round-time requirement at large N, and the real bound here is file
	// descriptors, not memory.
	Concurrency int
}

func DefaultConfig() Config {
	return Config{
		Interval: 10 * time.Second, Timeout: 5 * time.Second, Path: "/health",
		FailureThreshold: 3, SuccessThreshold: 2,
	}
}

type Checker struct {
	cfg    Config
	reg    *registry.Registry
	clk    clock.Clock
	client *http.Client

	streaks map[string]int // >0 consecutive successes, <0 consecutive failures
}

func New(cfg Config, reg *registry.Registry, clk clock.Clock) *Checker {
	if clk == nil {
		clk = clock.Real{}
	}
	if cfg.FailureThreshold < 1 {
		cfg.FailureThreshold = 1
	}
	if cfg.SuccessThreshold < 1 {
		cfg.SuccessThreshold = 1
	}
	return &Checker{
		cfg: cfg, reg: reg, clk: clk,
		client:  &http.Client{Timeout: cfg.Timeout},
		streaks: map[string]int{},
	}
}

// Run checks on the configured interval until ctx is cancelled. One goroutine,
// owned by the context, so shutdown leaks nothing (NFR-9).
func (c *Checker) Run(ctx context.Context) {
	c.Round(ctx) // check immediately so startup does not wait a full interval
	t := c.clk.NewTicker(c.cfg.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C():
			c.Round(ctx)
		}
	}
}

// Round performs one concurrent pass over all active-health backends.
func (c *Checker) Round(ctx context.Context) {
	snap := c.reg.Snapshot()

	var targets []*registry.Backend
	for _, b := range snap.Backends {
		// Passive backends are NEVER probed; their health comes solely from
		// proxied request outcomes (API-16, HLT-12).
		if b.HealthMod == registry.HealthActive {
			targets = append(targets, b)
		}
	}

	limit := c.cfg.Concurrency
	if limit <= 0 {
		limit = min(256, max(32, len(targets)))
	}
	sem := make(chan struct{}, limit)
	results := make(chan result, len(targets))

	for _, b := range targets {
		sem <- struct{}{}
		go func(b *registry.Backend) {
			defer func() { <-sem }()
			results <- result{b: b, ok: c.probe(ctx, b)}
		}(b)
	}
	for i := 0; i < len(targets); i++ {
		r := <-results
		c.observe(r.b, r.ok)
	}
	c.publishGauges(snap)
}

type result struct {
	b  *registry.Backend
	ok bool
}

func (c *Checker) probe(ctx context.Context, b *registry.Backend) bool {
	ctx, cancel := context.WithTimeout(ctx, c.cfg.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.URL+c.cfg.Path, nil)
	if err != nil {
		return false
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// observe applies hysteresis. Called from the single Round goroutine, so the
// streak map needs no lock.
func (c *Checker) observe(b *registry.Backend, ok bool) {
	s := c.streaks[b.URL]
	if ok {
		if s < 0 {
			s = 0
		}
		s++
	} else {
		if s > 0 {
			s = 0
		}
		s--
	}
	c.streaks[b.URL] = s

	switch {
	case ok && s >= c.cfg.SuccessThreshold && b.Health() != registry.Healthy:
		b.SetHealth(registry.Healthy)
	case !ok && -s >= c.cfg.FailureThreshold && b.Health() != registry.Unhealthy:
		b.SetHealth(registry.Unhealthy)
	}
	// Note what is absent: nothing here writes in-flight load.
}

func (c *Checker) publishGauges(snap *registry.Snapshot) {
	var healthy, unhealthy, unknown, draining int
	for _, b := range snap.Backends {
		metrics.BackendHealth.WithLabelValues(b.URL).Set(float64(b.Health()))
		metrics.CircuitState.WithLabelValues(b.URL).Set(float64(b.CB.State()))
		if b.Draining() {
			draining++
		}
		switch b.Health() {
		case registry.Healthy:
			healthy++
		case registry.Unhealthy:
			unhealthy++
		default:
			unknown++
		}
	}
	metrics.BackendsTotal.WithLabelValues("healthy").Set(float64(healthy))
	metrics.BackendsTotal.WithLabelValues("unhealthy").Set(float64(unhealthy))
	metrics.BackendsTotal.WithLabelValues("unknown").Set(float64(unknown))
	metrics.BackendsTotal.WithLabelValues("draining").Set(float64(draining))
}
