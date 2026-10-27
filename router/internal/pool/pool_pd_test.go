package pool_test

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/registry"
)

// With a band of its own the prefill flow saturates a node at the prefill hard
// limit, while the decode flow keeps taking requests on a node of the same load.
func TestPDRolesRunTheirOwnConcurrencyBands(t *testing.T) {
	newPool := func(prefill *affinity.Config) *pool.Pool {
		p, err := pool.New(pool.Config{
			Name:          "default",
			Backends:      []pool.Backend{{URL: "http://n:8000", Capacity: 64}},
			Flow:          affinity.Config{NodeConcurrency: 16, SoftNodeConcurrency: 8},
			PDPrefill:     []string{"http://other:8000"},
			PDPrefillFlow: prefill,
		}, clock.NewFake(time.Time{}), slog.Default())
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	selectOn := func(p *affinity.Policy, b *registry.Backend) error {
		_, err := p.Select(context.Background(), []*registry.Backend{b},
			&policy.RoutingRequest{Units: []kvcache.Unit{{Hash: 1, Tokens: 256}}, Model: "m"})
		return err
	}
	saturated := func(err error) bool { return errors.Is(err, policy.ErrAllBackendsSaturated) }

	p := newPool(&affinity.Config{NodeConcurrency: 4})
	b := p.Registry.Snapshot().Backends[0]
	b.SetHealth(registry.Healthy)
	for range 4 {
		lease.Acquire(b)
	}
	if err := selectOn(p.PDFlow, b); !saturated(err) {
		t.Errorf("prefill flow at its max 4: err = %v, want saturated", err)
	}
	if err := selectOn(p.Flow, b); err != nil {
		t.Errorf("decode flow at 4 of 16: %v", err)
	}
	for range 12 {
		lease.Acquire(b)
	}
	if err := selectOn(p.Flow, b); !saturated(err) {
		t.Errorf("decode flow at its max 16: err = %v, want saturated", err)
	}

	// No override: the prefill flow inherits the pool's flow, band included.
	p = newPool(nil)
	b = p.Registry.Snapshot().Backends[0]
	b.SetHealth(registry.Healthy)
	for range 15 {
		lease.Acquire(b)
	}
	if err := selectOn(p.PDFlow, b); err != nil {
		t.Errorf("inherited prefill band at 15 of 16: %v", err)
	}
	lease.Acquire(b)
	if err := selectOn(p.PDFlow, b); !saturated(err) {
		t.Errorf("inherited prefill band at 16: err = %v, want saturated", err)
	}
}
