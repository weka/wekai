package lease_test

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/registry"
)

func newBackends(t *testing.T, n int) (*registry.Registry, []*registry.Backend) {
	t.Helper()
	r := registry.New(registry.Options{})
	var bs []*registry.Backend
	for i := 0; i < n; i++ {
		b, err := r.Add(registry.Spec{URL: fmt.Sprintf("http://w%d:8000", i)})
		if err != nil {
			t.Fatalf("add: %v", err)
		}
		bs = append(bs, b)
	}
	return r, bs
}

func TestAcquireRelease(t *testing.T) {
	_, bs := newBackends(t, 1)
	b := bs[0]

	if got := b.Inflight(); got != 0 {
		t.Fatalf("initial inflight = %d, want 0", got)
	}
	l := lease.Acquire(b)
	if got := b.Inflight(); got != 1 {
		t.Fatalf("after acquire = %d, want 1", got)
	}
	l.Release()
	if got := b.Inflight(); got != 0 {
		t.Fatalf("after release = %d, want 0", got)
	}
}

// The v1 defect in miniature: one increment, many decrements. Idempotency is
// what makes the number of release call sites irrelevant (LB-2, LB-N1).
func TestReleaseIsIdempotent(t *testing.T) {
	lease.ResetAccountingErrors()
	_, bs := newBackends(t, 1)
	b := bs[0]

	l := lease.Acquire(b)
	for i := 0; i < 10; i++ {
		l.Release()
	}
	if got := b.Inflight(); got != 0 {
		t.Fatalf("inflight = %d, want 0", got)
	}
	if got := lease.AccountingErrors(); got != 0 {
		t.Fatalf("accounting errors = %d, want 0 (idempotent release must not underflow)", got)
	}
}

func TestConcurrentReleaseReleasesOnce(t *testing.T) {
	lease.ResetAccountingErrors()
	_, bs := newBackends(t, 1)
	b := bs[0]

	const goroutines = 64
	for iter := 0; iter < 200; iter++ {
		l := lease.Acquire(b)
		var wg sync.WaitGroup
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func() { defer wg.Done(); l.Release() }()
		}
		wg.Wait()
		if got := b.Inflight(); got != 0 {
			t.Fatalf("iter %d: inflight = %d, want 0", iter, got)
		}
	}
	if got := lease.AccountingErrors(); got != 0 {
		t.Fatalf("accounting errors = %d, want 0", got)
	}
}

func TestNilLeaseReleaseIsSafe(t *testing.T) {
	var l *lease.Lease
	l.Release() // must not panic: `defer lse.Release()` before Acquire succeeds
}

// TestLeasePropertyAllCountersReturnToZero is the LB-7 property test.
//
// It drives many concurrent request lifecycles through every termination path
// that v1 got wrong — including double release, concurrent release, retry to a
// different backend, and a panic mid-request — and asserts that every backend's
// in-flight count returns to exactly zero with no detected underflow.
func TestLeasePropertyAllCountersReturnToZero(t *testing.T) {
	lease.ResetAccountingErrors()
	const (
		backends   = 8
		lifecycles = 10000
		workers    = 64
	)
	_, bs := newBackends(t, backends)

	seed := rand.Uint64()
	t.Logf("seed = %d (reported so a failure is reproducible)", seed)

	var wg sync.WaitGroup
	wg.Add(workers)
	per := lifecycles / workers
	for w := 0; w < workers; w++ {
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(seed, uint64(w)))
			for i := 0; i < per; i++ {
				runLifecycle(rng, bs)
			}
		}(w)
	}
	wg.Wait()

	for _, b := range bs {
		if got := b.Inflight(); got != 0 {
			t.Errorf("backend %s: inflight = %d, want 0", b.URL, got)
		}
	}
	if got := lease.AccountingErrors(); got != 0 {
		t.Errorf("accounting errors = %d, want 0", got)
	}
}

// runLifecycle models one request, choosing uniformly among the termination
// paths that a real proxy has to survive.
func runLifecycle(rng *rand.Rand, bs []*registry.Backend) {
	defer func() { _ = recover() }() // the panic case is deliberate

	b := bs[rng.IntN(len(bs))]
	l := lease.Acquire(b)
	// Every path below must end with the lease released exactly once in effect.
	defer l.Release()

	switch rng.IntN(8) {
	case 0: // normal completion
		return
	case 1: // explicit release then the deferred one fires too
		l.Release()
	case 2: // retry to a different backend: release first, acquire second
		l.Release()
		other := bs[rng.IntN(len(bs))]
		l2 := lease.Acquire(other)
		defer l2.Release()
	case 3: // concurrent release from two goroutines
		var wg sync.WaitGroup
		wg.Add(2)
		for i := 0; i < 2; i++ {
			go func() { defer wg.Done(); l.Release() }()
		}
		wg.Wait()
	case 4: // client cancellation: unwinds through the defer
		return
	case 5: // upstream error, released by the handler then again by defer
		l.Release()
		l.Release()
	case 6: // stream abort mid-body
		return
	case 7: // panic in the handler; release happens during unwinding
		panic("simulated handler panic")
	}
}

// TestReleaseWakesAQueuedTicket covers the lease<->waitq hookup: Release is
// the single wake source for a backend's wait queue (waitq.go's Wake is never
// called anywhere else on the request path), so a ticket joined on a backend
// must be signalled the moment that backend's lease is released.
func TestReleaseWakesAQueuedTicket(t *testing.T) {
	_, bs := newBackends(t, 1)
	b := bs[0]

	l := lease.Acquire(b)
	tk, ok := b.Waiters().Join(0, false)
	if !ok {
		t.Fatal("Join: want true")
	}

	select {
	case <-tk.C():
		t.Fatal("ticket woken before any release")
	default:
	}

	l.Release()

	select {
	case <-tk.C():
	default:
		t.Fatal("ticket was not woken by Release")
	}
}

// TestDoubleReleaseWakesOnlyOnce guards the idempotency half of the hookup:
// Release's once.Do must mean the wake fires exactly once too, not once per
// call site that happens to invoke Release on an already-released lease.
func TestDoubleReleaseWakesOnlyOnce(t *testing.T) {
	_, bs := newBackends(t, 1)
	b := bs[0]

	l := lease.Acquire(b)
	l.Release()
	l.Release() // idempotent: must not wake a second ticket

	first, ok := b.Waiters().Join(0, false)
	if !ok {
		t.Fatal("Join first: want true")
	}
	second, ok := b.Waiters().Join(0, false)
	if !ok {
		t.Fatal("Join second: want true")
	}

	// Both tickets were joined AFTER the lease already released, so neither
	// should ever be woken by it — this just establishes a clean baseline
	// before the real assertion: one more release wakes exactly one of them.
	l2 := lease.Acquire(b)
	l2.Release()
	l2.Release()

	woken := 0
	select {
	case <-first.C():
		woken++
	default:
	}
	select {
	case <-second.C():
		woken++
	default:
	}
	if woken != 1 {
		t.Fatalf("tickets woken = %d, want exactly 1 (one release, one wake)", woken)
	}
}
