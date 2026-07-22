package benchmark

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestConcurrencyGateFIFODefault verifies that with randomOrder off, Release()
// serves waiters in exact FIFO order (index 0 first) -- the pre-existing,
// unchanged behavior. Byte-for-byte order must be preserved when the flag is
// off.
func TestConcurrencyGateFIFODefault(t *testing.T) {
	g := newConcurrencyGate(1, false)
	ctx := context.Background()

	if err := g.Acquire(ctx); err != nil {
		t.Fatalf("initial Acquire: %v", err)
	}

	const n = 20
	var mu sync.Mutex
	order := make([]int, 0, n)
	var wg sync.WaitGroup

	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := g.Acquire(ctx); err != nil {
				t.Errorf("Acquire %d: %v", i, err)
				return
			}
			mu.Lock()
			order = append(order, i)
			mu.Unlock()
		}()
		// Wait for goroutine i to actually enqueue as waiter #i+1 before
		// spawning i+1, so g.waiters insertion order is deterministically
		// 0..n-1 (goroutine spawn order alone does not guarantee this: the
		// Go scheduler can run spawned goroutines in any order).
		waitForLen(t, g, i+1)
	}

	// Release one at a time, waiting for each waiter to actually record its
	// turn before releasing the next. Since limit=1, exactly one waiter is
	// ever unblocked at a time, so this makes `order` a faithful trace of
	// release sequence rather than a race between goroutine wakeups.
	for i := 0; i < n; i++ {
		g.Release()
		waitForOrderLen(t, &mu, &order, i+1)
	}
	wg.Wait()

	if len(order) != n {
		t.Fatalf("got %d completions, want %d", len(order), n)
	}
	for i, v := range order {
		if v != i {
			t.Fatalf("FIFO order violated: order[%d] = %d, want %d (full order: %v)", i, v, i, order)
		}
	}
}

// TestConcurrencyGateRandomOrderServesAllExactlyOnce verifies the core
// correctness property of random mode: every waiter is served exactly once
// (no double-serve, no lost wakeups), and active accounting settles back to
// the limit once every waiter has been handed a slot.
func TestConcurrencyGateRandomOrderServesAllExactlyOnce(t *testing.T) {
	const limit = 4
	const waiters = 200

	g := newConcurrencyGate(limit, true)
	ctx := context.Background()

	// Fill all `limit` slots synchronously first so every subsequently
	// spawned goroutine is guaranteed to block and enqueue as a waiter.
	for i := 0; i < limit; i++ {
		if err := g.Acquire(ctx); err != nil {
			t.Fatalf("seed Acquire %d: %v", i, err)
		}
	}

	var served sync.Map // id -> count
	var wg sync.WaitGroup
	for i := 0; i < waiters; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := g.Acquire(ctx); err != nil {
				t.Errorf("Acquire %d: %v", i, err)
				return
			}
			v, _ := served.LoadOrStore(i, new(int))
			*(v.(*int))++
		}()
	}

	waitForWaiters(t, g, waiters)

	for i := 0; i < waiters; i++ {
		g.Release()
	}
	wg.Wait()

	seen := 0
	for i := 0; i < waiters; i++ {
		v, ok := served.Load(i)
		if !ok {
			t.Fatalf("waiter %d was never served (lost wakeup)", i)
		}
		count := *(v.(*int))
		if count != 1 {
			t.Fatalf("waiter %d served %d times, want exactly 1 (double-serve)", i, count)
		}
		seen++
	}
	if seen != waiters {
		t.Fatalf("served %d distinct waiters, want %d", seen, waiters)
	}

	active, cold, normal := g.GateStats()
	if active != limit {
		t.Fatalf("active = %d, want %d (limit) after all releases settle back to steady state", active, limit)
	}
	if cold != 0 || normal != 0 {
		t.Fatalf("expected no residual waiters, got cold=%d normal=%d", cold, normal)
	}
}

// TestConcurrencyGateRandomOrderNotAlwaysFIFO checks distribution sanity:
// across repeated release rounds, the waiter served isn't deterministically
// index 0 the way strict FIFO would be. This is a statistical smoke test,
// not a rigorous uniformity check -- it just guards against an accidental
// no-op random-mode implementation that silently falls back to FIFO.
func TestConcurrencyGateRandomOrderNotAlwaysFIFO(t *testing.T) {
	g := newConcurrencyGate(1, true)
	ctx := context.Background()
	if err := g.Acquire(ctx); err != nil {
		t.Fatalf("seed Acquire: %v", err)
	}

	const n = 50
	firstServed := -1
	sawNonFirst := false

	for round := 0; round < 30; round++ {
		var wg sync.WaitGroup
		resultCh := make(chan int, n)
		for i := 0; i < n; i++ {
			i := i
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := g.Acquire(ctx); err != nil {
					t.Errorf("Acquire %d: %v", i, err)
					return
				}
				resultCh <- i
			}()
		}
		waitForWaiters(t, g, n)
		g.Release() // wakes exactly one waiter
		winner := <-resultCh
		if round == 0 {
			firstServed = winner
		} else if winner != firstServed {
			sawNonFirst = true
		}
		// Drain the rest so the next round starts clean (all n waiters from
		// this round must be released before the gate returns to its
		// steady active=1 state that the next round's preconditions rely on).
		for i := 0; i < n-1; i++ {
			g.Release()
		}
		wg.Wait()
	}

	if !sawNonFirst {
		t.Fatalf("random mode always served the same waiter id (%d) across 30 rounds of %d waiters -- looks like FIFO, not random", firstServed, n)
	}
}

// TestConcurrencyGateColdPriorityPreservedUnderRandomOrder verifies that
// random mode only reorders normal waiters; coldWaiters keep FIFO priority
// over normal waiters regardless of the flag.
func TestConcurrencyGateColdPriorityPreservedUnderRandomOrder(t *testing.T) {
	g := newConcurrencyGate(1, true)
	ctx := context.Background()
	if err := g.Acquire(ctx); err != nil {
		t.Fatalf("seed Acquire: %v", err)
	}

	normalDone := make(chan struct{})
	go func() {
		if err := g.Acquire(ctx); err != nil {
			t.Errorf("normal Acquire: %v", err)
		}
		close(normalDone)
	}()
	waitForWaiters(t, g, 1)

	coldDone := make(chan struct{})
	go func() {
		if err := g.AcquireCold(ctx); err != nil {
			t.Errorf("cold Acquire: %v", err)
		}
		close(coldDone)
	}()
	waitForColdWaiters(t, g, 1)

	g.Release()

	// The cold waiter must be the one released, even though a normal waiter
	// was queued first. Block (with a deadline) rather than checking
	// immediately -- Release() only hands off the token synchronously, the
	// woken goroutine still needs to be scheduled before it closes the
	// channel.
	select {
	case <-coldDone:
	case <-time.After(5 * time.Second):
		t.Fatalf("cold waiter was not served first (coldDone never closed)")
	}
	// normalDone closing is gated purely on Release() having sent it a
	// token, which provably has not happened yet (only one Release() call
	// so far, and it went to coldWaiters) -- no scheduling race here.
	select {
	case <-normalDone:
		t.Fatalf("normal waiter was served before cold waiter released its slot")
	default:
	}

	g.Release()
	select {
	case <-normalDone:
	case <-time.After(5 * time.Second):
		t.Fatalf("normal waiter was never served after cold waiter's slot was released")
	}
}

func waitForWaiters(t *testing.T, g *concurrencyGate, want int) {
	t.Helper()
	spinUntil(t, func() bool {
		_, _, n := g.GateStats()
		return n >= want
	})
}

func waitForColdWaiters(t *testing.T, g *concurrencyGate, want int) {
	t.Helper()
	spinUntil(t, func() bool {
		_, c, _ := g.GateStats()
		return c >= want
	})
}

func waitForLen(t *testing.T, g *concurrencyGate, want int) {
	t.Helper()
	spinUntil(t, func() bool {
		_, _, n := g.GateStats()
		return n == want
	})
}

func waitForOrderLen(t *testing.T, mu *sync.Mutex, order *[]int, want int) {
	t.Helper()
	spinUntil(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(*order) >= want
	})
}

func spinUntil(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		runtime.Gosched()
	}
	t.Fatalf("condition not met within deadline")
}

// TestGateOrderDefaultIsRandom locks the 2026-07-22 default flip at the
// config level: a default-constructed AutoBenchmarkConfig produces a
// random-order gate; explicit FIFOGateOrder restores the legacy FIFO.
func TestGateOrderDefaultIsRandom(t *testing.T) {
	var cfg AutoBenchmarkConfig
	if g := newConcurrencyGate(1, !cfg.FIFOGateOrder); !g.randomOrder {
		t.Fatal("default-constructed config must yield a random-order gate")
	}
	cfg.FIFOGateOrder = true
	if g := newConcurrencyGate(1, !cfg.FIFOGateOrder); g.randomOrder {
		t.Fatal("explicit FIFOGateOrder must yield a FIFO gate")
	}
}
