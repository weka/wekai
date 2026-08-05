package registry_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/registry"
)

func TestCanonicalizationTable(t *testing.T) {
	ok := []struct{ in, want string }{
		{"http://a:8000", "http://a:8000"},
		{"a:8000", "http://a:8000"},
		{"HTTP://A:8000", "http://a:8000"},
		{"http://a:8000/", "http://a:8000"},
		{"http://a:8000/v1/", "http://a:8000"}, // path is not part of identity
		{"http://a", "http://a:80"},            // port made explicit
		{"https://a", "https://a:443"},
		{"  http://a:8000  ", "http://a:8000"},
		{"http://[::1]:8000", "http://[::1]:8000"}, // IPv6 stays bracketed (SD-9)
		{"http://[2001:db8::1]", "http://[2001:db8::1]:80"},
	}
	for _, c := range ok {
		got, err := registry.Canonical(c.in)
		if err != nil {
			t.Errorf("Canonical(%q) error: %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("Canonical(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// Scheme restriction is the SSRF guard on the admin API (SEC-5).
	bad := []string{"", "   ", "file:///etc/passwd", "gopher://a", "http://"}
	for _, in := range bad {
		if got, err := registry.Canonical(in); err == nil {
			t.Errorf("Canonical(%q) = %q, want error", in, got)
		}
	}
}

// TestRegisterTwiceYieldsOneEntry guards WRK-N2. v1 appended a re-registered
// URL to every secondary index, so one backend was weighted N times in
// selection and power-of-two could "sample two" and draw the same backend.
func TestRegisterTwiceYieldsOneEntry(t *testing.T) {
	r := registry.New(registry.Options{})
	for _, u := range []string{"http://a:8000", "a:8000", "HTTP://A:8000/", "http://a:8000"} {
		if _, err := r.Add(registry.Spec{URL: u}); err != nil {
			t.Fatalf("add %q: %v", u, err)
		}
	}
	snap := r.Snapshot()
	if got := len(snap.Backends); got != 1 {
		t.Fatalf("len(Backends) = %d, want 1 — all four inputs canonicalize equal", got)
	}
	if _, ok := snap.Get("http://a:8000"); !ok {
		t.Fatal("canonical URL not indexed")
	}
}

// TestSnapshotOrderStableOverManyMutations guards WRK-N1/LB-N3. v1 returned
// worker lists from a DashMap in unspecified order that varied between calls,
// which by itself turns round-robin into random.
func TestSnapshotOrderStableOverManyMutations(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for iter := 0; iter < 200; iter++ {
		r := registry.New(registry.Options{})
		var urls []string
		n := 2 + rng.IntN(12)
		for i := 0; i < n; i++ {
			urls = append(urls, fmt.Sprintf("http://w%02d:8000", rng.IntN(50)))
		}
		rng.Shuffle(len(urls), func(i, j int) { urls[i], urls[j] = urls[j], urls[i] })
		for _, u := range urls {
			if _, err := r.Add(registry.Spec{URL: u}); err != nil {
				t.Fatalf("add: %v", err)
			}
		}
		got := backendURLs(r.Snapshot().Backends)
		if !slices.IsSorted(got) {
			t.Fatalf("iter %d: snapshot not sorted: %v", iter, got)
		}
		// Repeated reads must be identical, not merely sorted.
		if again := backendURLs(r.Snapshot().Backends); !slices.Equal(got, again) {
			t.Fatalf("iter %d: two reads differ:\n%v\n%v", iter, got, again)
		}
	}
}

func backendURLs(bs []*registry.Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.URL
	}
	return out
}

// A newly added backend must not be routable until its first successful health
// check (HLT-5, SD-N3) — otherwise discovery sends traffic to a pod that has
// not loaded its model.
func TestNewBackendStartsUnknownAndUnavailable(t *testing.T) {
	r := registry.New(registry.Options{})
	b, err := r.Add(registry.Spec{URL: "http://a:8000"})
	if err != nil {
		t.Fatal(err)
	}
	if got := b.Health(); got != registry.Unknown {
		t.Fatalf("health = %v, want unknown", got)
	}
	if b.Available() {
		t.Fatal("backend available before first health check")
	}
	if got := len(r.Snapshot().Available()); got != 0 {
		t.Fatalf("available = %d, want 0", got)
	}
	b.SetHealth(registry.Healthy)
	if got := len(r.Snapshot().Available()); got != 1 {
		t.Fatalf("available after health pass = %d, want 1", got)
	}
}

func TestNormalizedLoadUsesCapacity(t *testing.T) {
	r := registry.New(registry.Options{})
	leaf, _ := r.Add(registry.Spec{URL: "http://leaf:8000", Capacity: 1})
	child, _ := r.Add(registry.Spec{URL: "http://child:8000", Kind: registry.KindRouter, Capacity: 40})

	// The HIER-N1 scenario: a child fronting 40 GPUs with 40 requests in
	// flight must not look "more loaded" than a leaf with 1.
	for i := 0; i < 40; i++ {
		child.AddInflight(1)
	}
	leaf.AddInflight(1)

	if got := child.NormalizedLoad(); got != 1.0 {
		t.Fatalf("child normalized load = %v, want 1.0", got)
	}
	if got := leaf.NormalizedLoad(); got != 1.0 {
		t.Fatalf("leaf normalized load = %v, want 1.0", got)
	}
	// And raw counts would have said the opposite.
	if child.Inflight() <= leaf.Inflight() {
		t.Fatal("test setup wrong: child should have a higher raw count")
	}
}

// Drain must exclude a backend from new traffic immediately while keeping its
// entry until in-flight reaches zero (WRK-6, WRK-N3).
func TestDrainExcludesImmediatelyAndUnlinksWhenIdle(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	dropped := make(chan string, 1)
	r := registry.New(registry.Options{
		Clock:         clk,
		DrainDeadline: 60 * time.Second,
		OnDrop:        func(b *registry.Backend) { dropped <- b.URL },
	})
	b, _ := r.Add(registry.Spec{URL: "http://a:8000"})
	b.SetHealth(registry.Healthy)
	l := b.AddInflight(1) // one request in flight
	if l != 1 {
		t.Fatalf("inflight = %d", l)
	}

	if err := r.Remove("http://a:8000"); err != nil {
		t.Fatal(err)
	}
	if b.Available() {
		t.Fatal("draining backend still available for new traffic")
	}
	if got := len(r.Snapshot().Backends); got != 1 {
		t.Fatalf("entry removed while a request was in flight: len = %d, want 1", got)
	}

	b.AddInflight(-1) // request completes
	waitFor(t, dropped)
	if got := len(r.Snapshot().Backends); got != 0 {
		t.Fatalf("len(Backends) = %d after drain, want 0", got)
	}
}

func TestDrainDeadlineForcesUnlink(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	dropped := make(chan string, 1)
	r := registry.New(registry.Options{
		Clock: clk, DrainDeadline: 30 * time.Second,
		OnDrop: func(b *registry.Backend) { dropped <- b.URL },
	})
	b, _ := r.Add(registry.Spec{URL: "http://a:8000"})
	b.AddInflight(1) // never completes

	if err := r.Remove("http://a:8000"); err != nil {
		t.Fatal(err)
	}
	// Let the drain goroutine register its waiters before advancing.
	for i := 0; i < 100 && len(r.Snapshot().Backends) == 1; i++ {
		clk.Advance(time.Second)
		time.Sleep(time.Millisecond)
	}
	waitFor(t, dropped)
}

func waitFor(t *testing.T, ch chan string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for drain to complete")
	}
}

// TestReconcileIsIdempotent guards SD-7/SD-N2: applying the same desired set
// twice must produce identical state. Full reconciliation rather than
// incremental patching is what makes this true by construction.
func TestReconcileIsIdempotent(t *testing.T) {
	r := registry.New(registry.Options{Clock: clock.NewFake(time.Time{})})
	desired := []registry.Spec{
		{URL: "http://b:8000"}, {URL: "http://a:8000"}, {URL: "http://c:8000"},
	}
	for pass := 0; pass < 5; pass++ {
		conflicts, err := r.ReconcileDiscovered(desired)
		if err != nil {
			t.Fatalf("pass %d: %v", pass, err)
		}
		if len(conflicts) != 0 {
			t.Fatalf("pass %d: unexpected conflicts %v", pass, conflicts)
		}
		got := backendURLs(r.Snapshot().Backends)
		want := []string{"http://a:8000", "http://b:8000", "http://c:8000"}
		if !slices.Equal(got, want) {
			t.Fatalf("pass %d: backends = %v, want %v", pass, got, want)
		}
	}
}

// TestStaticWinsOverDiscovered guards HIER-19. With topology coming from both
// config and discovery, a reconcile pass must never be able to delete or
// override a statically-declared backend.
func TestStaticWinsOverDiscovered(t *testing.T) {
	r := registry.New(registry.Options{Clock: clock.NewFake(time.Time{})})
	st, err := r.Add(registry.Spec{URL: "http://pinned:8000", Prov: registry.ProvStatic, Model: "static-model"})
	if err != nil {
		t.Fatal(err)
	}

	conflicts, err := r.ReconcileDiscovered([]registry.Spec{
		{URL: "http://pinned:8000", Model: "discovered-model"},
		{URL: "http://fresh:8000"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(conflicts, []string{"http://pinned:8000"}) {
		t.Fatalf("conflicts = %v, want [http://pinned:8000]", conflicts)
	}
	if st.Model != "static-model" {
		t.Errorf("static backend overwritten: model = %q", st.Model)
	}
	if st.Prov != registry.ProvStatic {
		t.Errorf("provenance changed to %v", st.Prov)
	}

	// A later reconcile that omits the static entry must not drain it.
	if _, err := r.ReconcileDiscovered([]registry.Spec{{URL: "http://fresh:8000"}}); err != nil {
		t.Fatal(err)
	}
	if st.Draining() {
		t.Fatal("discovery drained a statically-configured backend")
	}
	if _, ok := r.Snapshot().Get("http://pinned:8000"); !ok {
		t.Fatal("static backend removed by discovery")
	}
}

func TestReconcileDrainsUndesiredDiscovered(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	r := registry.New(registry.Options{Clock: clk})
	if _, err := r.ReconcileDiscovered([]registry.Spec{{URL: "http://a:8000"}, {URL: "http://b:8000"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.ReconcileDiscovered([]registry.Spec{{URL: "http://a:8000"}}); err != nil {
		t.Fatal(err)
	}
	b, ok := r.Snapshot().Get("http://b:8000")
	if !ok {
		t.Skip("already unlinked")
	}
	if !b.Draining() {
		t.Fatal("undesired discovered backend not draining")
	}
	if b.Available() {
		t.Fatal("draining backend still available")
	}
}

// A backend whose circuit is open must be filtered out, and filtering must not
// mutate circuit state (R2) — verified here end-to-end through Available().
func TestAvailableExcludesOpenCircuitWithoutConsumingProbes(t *testing.T) {
	clk := clock.NewFake(time.Time{})
	r := registry.New(registry.Options{Clock: clk})
	b, _ := r.Add(registry.Spec{URL: "http://a:8000"})
	b.SetHealth(registry.Healthy)
	if !b.Available() {
		t.Fatal("healthy closed-circuit backend not available")
	}
	for i := 0; i < 40; i++ {
		b.CB.Record(1 /* Failure */, false)
	}
	for i := 0; i < 500; i++ {
		if b.Available() {
			t.Fatalf("pass %d: open-circuit backend reported available", i)
		}
	}
}
