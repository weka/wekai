package registry

import (
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/router/internal/circuit"
	"github.com/weka/wekai/router/internal/clock"
)

// Snapshot is an immutable view of the backend set.
//
// Backends is sorted by canonical URL and MUST NOT be mutated by readers. The
// *Backend pointers are shared across snapshots — only the slice and index are
// copied — so per-backend counters stay live while membership stays stable
// (WRK-3, WRK-4).
type Snapshot struct {
	Version  uint64
	Backends []*Backend
	byURL    map[string]*Backend
}

func (s *Snapshot) Get(url string) (*Backend, bool) {
	b, ok := s.byURL[url]
	return b, ok
}

// Available returns the backends eligible for new traffic, in snapshot order.
// This is the candidate set handed to policies (LB-9) — policies never see
// unhealthy, draining, or open-circuit backends, and never mutate circuit state
// while filtering (R2).
func (s *Snapshot) Available() []*Backend {
	out := make([]*Backend, 0, len(s.Backends))
	for _, b := range s.Backends {
		if b.Available() {
			out = append(out, b)
		}
	}
	return out
}

// Registry is the single source of truth for backend membership.
type Registry struct {
	clk clock.Clock
	cbc circuit.Config

	mu      sync.Mutex // serializes writers only; readers are wait-free
	cur     atomic.Pointer[Snapshot]
	version uint64

	drainDeadline time.Duration
	newGauge      func(url string) Gauge
	onAdd, onDrop func(*Backend)
}

type Options struct {
	Clock         clock.Clock
	Circuit       circuit.Config
	DrainDeadline time.Duration
	// NewGauge resolves a backend's in-flight gauge once, at registration (R5).
	NewGauge func(url string) Gauge
	// OnAdd / OnDrop drive per-backend resource lifecycle — notably the cache
	// model, which must be created on add and dropped on removal with its
	// prefixes never reassigned to another backend (CACHE-10, CU-4, CU-12).
	OnAdd  func(*Backend)
	OnDrop func(*Backend)
}

func New(o Options) *Registry {
	if o.Clock == nil {
		o.Clock = clock.Real{}
	}
	if o.Circuit.Window == 0 {
		o.Circuit = circuit.DefaultConfig()
	}
	if o.DrainDeadline == 0 {
		o.DrainDeadline = 60 * time.Second
	}
	if o.NewGauge == nil {
		o.NewGauge = func(string) Gauge { return nopGauge{} }
	}
	r := &Registry{
		clk: o.Clock, cbc: o.Circuit, drainDeadline: o.DrainDeadline,
		newGauge: o.NewGauge, onAdd: o.OnAdd, onDrop: o.OnDrop,
	}
	r.cur.Store(&Snapshot{byURL: map[string]*Backend{}})
	return r
}

// Snapshot is wait-free: a single atomic load.
func (r *Registry) Snapshot() *Snapshot { return r.cur.Load() }

// Add registers or updates a backend. It is idempotent (WRK-2).
//
// v1 reused the WorkerId for an existing URL but still appended the worker to
// every secondary index, so re-registering weighted one backend N times in
// selection and let power-of-two "sample two" and get the same backend twice
// (WRK-N2, E1). Here there is exactly one map, keyed by canonical URL.
func (r *Registry) Add(spec Spec) (*Backend, error) {
	canon, err := Canonical(spec.URL)
	if err != nil {
		return nil, err
	}
	spec.URL = canon

	r.mu.Lock()
	defer r.mu.Unlock()

	old := r.cur.Load()
	if existing, ok := old.byURL[canon]; ok {
		// Update in place. Discovery must never override a static entry.
		if existing.Prov == ProvStatic && spec.Prov == ProvDiscovered {
			return existing, ErrStaticConflict
		}
		existing.SetKind(spec.Kind)
		existing.SetModel(spec.Model)
		existing.SetLocality(spec.Locality)
		existing.SetCapacity(spec.Capacity)
		existing.draining.Store(false) // re-adding cancels a pending drain
		return existing, nil
	}

	b := &Backend{
		URL: canon, DialectID: spec.DialectID,
		HealthMod: spec.Health, Prov: spec.Prov,
		CB:            circuit.New(r.cbc, r.clk),
		InflightGauge: r.newGauge(canon),
	}
	b.SetKind(spec.Kind)
	b.SetModel(spec.Model)
	b.SetLocality(spec.Locality)
	b.SetCapacity(spec.Capacity)
	// Unknown, not Healthy: a backend is not routable until its first
	// successful health check, so discovery cannot send traffic to a pod that
	// has not loaded its model yet (HLT-5, SD-N3).
	b.SetHealth(Unknown)
	if b.DialectID == "" {
		b.DialectID = "openai"
	}

	r.publishLocked(append(slices.Clone(old.Backends), b))
	if r.onAdd != nil {
		r.onAdd(b)
	}
	return b, nil
}

// ErrStaticConflict reports that discovery tried to modify a static backend.
var ErrStaticConflict = fmt.Errorf("backend is statically configured")

// Remove drains a backend: it stops receiving new traffic immediately, but its
// entry is retained until in-flight reaches zero or the drain deadline elapses
// (WRK-6, WRK-N3). v1 hard-removed workers with requests still attached.
func (r *Registry) Remove(rawURL string) error {
	canon, err := Canonical(rawURL)
	if err != nil {
		return err
	}
	r.mu.Lock()
	b, ok := r.cur.Load().byURL[canon]
	r.mu.Unlock()
	if !ok {
		return fmt.Errorf("unknown backend %q", canon)
	}
	r.drain(b)
	return nil
}

func (r *Registry) drain(b *Backend) {
	if b.draining.Swap(true) {
		return // already draining
	}
	go r.awaitDrain(b)
}

func (r *Registry) awaitDrain(b *Backend) {
	deadline := r.clk.After(r.drainDeadline)
	tick := r.clk.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if b.Inflight() <= 0 {
			r.unlink(b)
			return
		}
		select {
		case <-deadline:
			r.unlink(b)
			return
		case <-tick.C():
		}
	}
}

func (r *Registry) unlink(b *Backend) {
	r.mu.Lock()
	old := r.cur.Load()
	if _, ok := old.byURL[b.URL]; !ok {
		r.mu.Unlock()
		return
	}
	next := make([]*Backend, 0, len(old.Backends))
	for _, x := range old.Backends {
		if x.URL != b.URL {
			next = append(next, x)
		}
	}
	r.publishLocked(next)
	r.mu.Unlock()

	// Drop AFTER unlinking, so no reader can still select this backend while
	// its cache model is being torn down. v1 had this inverted: it removed the
	// worker from the registry and *then* looked it up by URL to clean up the
	// cache tree, so the lookup always returned None and dead tenants stayed
	// in the tree forever (E2, E3).
	if r.onDrop != nil {
		r.onDrop(b)
	}
}

// ReconcileDiscovered applies a complete desired set of discovered backends.
//
// Full reconciliation rather than incremental patching is what makes double
// application a no-op (SD-7, SD-N2): there is no additive index to corrupt.
// Static entries are never added, updated, or removed by this pass (HIER-19).
// Returns the number of desired entries that collided with a static backend.
func (r *Registry) ReconcileDiscovered(desired []Spec) (conflicts []string, err error) {
	want := make(map[string]Spec, len(desired))
	for _, s := range desired {
		canon, cerr := Canonical(s.URL)
		if cerr != nil {
			return nil, cerr
		}
		s.URL = canon
		s.Prov = ProvDiscovered
		want[canon] = s
	}

	for _, s := range want {
		if _, aerr := r.Add(s); aerr == ErrStaticConflict {
			conflicts = append(conflicts, s.URL)
		} else if aerr != nil {
			return nil, aerr
		}
	}

	// Drain discovered backends that are no longer desired. Removal always
	// goes through drain, never a hard delete (SD-6).
	for _, b := range r.Snapshot().Backends {
		if b.Prov != ProvDiscovered || b.Draining() {
			continue
		}
		if _, keep := want[b.URL]; !keep {
			r.drain(b)
		}
	}
	slices.Sort(conflicts)
	return conflicts, nil
}

// publishLocked builds and atomically installs a new snapshot. Caller holds mu.
func (r *Registry) publishLocked(bs []*Backend) {
	slices.SortFunc(bs, func(a, b *Backend) int {
		switch {
		case a.URL < b.URL:
			return -1
		case a.URL > b.URL:
			return 1
		}
		return 0
	})
	idx := make(map[string]*Backend, len(bs))
	for _, b := range bs {
		idx[b.URL] = b
	}
	r.version++
	r.cur.Store(&Snapshot{Version: r.version, Backends: bs, byURL: idx})
}
