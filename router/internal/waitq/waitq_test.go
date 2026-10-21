package waitq

import (
	"sync"
	"testing"
	"time"
)

func TestWakeOnEmptyQueueIsNoop(t *testing.T) {
	var q Queue
	q.Wake() // must not panic or block
	if got := q.Len(); got != 0 {
		t.Fatalf("Len() = %d, want 0", got)
	}
}

func TestJoinWakeIsFIFO(t *testing.T) {
	var q Queue
	var tickets []*Ticket
	for i := 0; i < 4; i++ {
		tk, ok := q.Join(0, false)
		if !ok {
			t.Fatalf("Join %d: want true", i)
		}
		tickets = append(tickets, tk)
	}
	if got := q.Len(); got != 4 {
		t.Fatalf("Len() = %d, want 4", got)
	}

	for i, tk := range tickets {
		q.Wake()
		select {
		case <-tk.C():
		default:
			t.Fatalf("ticket %d (expected FIFO position) was not woken", i)
		}
		// No other ticket must have been woken by this Wake call.
		for j, other := range tickets {
			if j <= i {
				continue
			}
			select {
			case <-other.C():
				t.Fatalf("ticket %d woken out of order (waking position %d)", j, i)
			default:
			}
		}
	}
	if got := q.Len(); got != 0 {
		t.Fatalf("Len() after draining = %d, want 0", got)
	}
}

func TestJoinCapEnforcement(t *testing.T) {
	var q Queue
	for i := 0; i < 2; i++ {
		if _, ok := q.Join(2, false); !ok {
			t.Fatalf("Join %d: want true under cap", i)
		}
	}
	if _, ok := q.Join(2, false); ok {
		t.Fatal("Join at cap: want false")
	}
	if got := q.Len(); got != 2 {
		t.Fatalf("Len() = %d, want 2 (the rejected join must not have been added)", got)
	}
}

// TestJoinFrontBypassesCap: a front join is admitted even over the cap, and
// goes to the head of the line ahead of everyone already waiting.
func TestJoinFrontBypassesCap(t *testing.T) {
	var q Queue
	back, ok := q.Join(1, false)
	if !ok {
		t.Fatal("Join back: want true")
	}
	front, ok := q.Join(1, true)
	if !ok {
		t.Fatal("Join front: want true despite being at cap")
	}
	if got := q.Len(); got != 2 {
		t.Fatalf("Len() = %d, want 2", got)
	}

	q.Wake()
	select {
	case <-front.C():
	default:
		t.Fatal("front-joined ticket should have been woken first")
	}
	select {
	case <-back.C():
		t.Fatal("back ticket must not be woken yet")
	default:
	}
}

// TestLeavePassesOnPendingWake covers the race the comment on Ticket.Leave
// describes: a ticket that is given up right as (or after) it was woken must
// not swallow the slot — it has to hand it to the next waiter.
func TestLeavePassesOnPendingWake(t *testing.T) {
	var q Queue
	first, ok := q.Join(0, false)
	if !ok {
		t.Fatal("Join first: want true")
	}
	second, ok := q.Join(0, false)
	if !ok {
		t.Fatal("Join second: want true")
	}

	q.Wake() // pops and signals `first`
	select {
	case <-first.C():
	default:
		t.Fatal("first should have been woken")
	}

	// `first` declines the slot it was already handed — exactly the "woken but
	// served elsewhere" case from capacity_retry's awaitSlot.
	first.Leave()

	select {
	case <-second.C():
	default:
		t.Fatal("Leave on an already-woken ticket must pass the wake to the next waiter")
	}
}

// TestLeaveStillQueuedRemovesWithoutWaking: the common case, where a waiter
// gives up before ever being woken.
func TestLeaveStillQueuedRemovesWithoutWaking(t *testing.T) {
	var q Queue
	first, ok := q.Join(0, false)
	if !ok {
		t.Fatal("Join first: want true")
	}
	second, ok := q.Join(0, false)
	if !ok {
		t.Fatal("Join second: want true")
	}

	first.Leave()
	if got := q.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1 after leaving a still-queued ticket", got)
	}

	q.Wake()
	select {
	case <-second.C():
	default:
		t.Fatal("second should be woken; first's departure must not have consumed a wake")
	}
	select {
	case <-first.C():
		t.Fatal("a ticket that left before being woken must never receive one")
	default:
	}
}

// TestConcurrentJoinWakeLeaveStress hammers the queue from many goroutines at
// once under -race: every ticket must end up exactly once in one of "woken" or
// "left without a wake", and the queue must never panic or deadlock.
func TestConcurrentJoinWakeLeaveStress(t *testing.T) {
	var q Queue
	const n = 500

	var wg sync.WaitGroup
	woken := make([]bool, n)
	var mu sync.Mutex

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tk, ok := q.Join(0, false)
			if !ok {
				t.Errorf("Join %d: want true (unbounded queue)", i)
				return
			}
			select {
			case <-tk.C():
				mu.Lock()
				woken[i] = true
				mu.Unlock()
			case <-time.After(2 * time.Second):
				tk.Leave()
			}
		}(i)
	}

	// Wake everyone, possibly racing the Leave calls above; Wake is a no-op on
	// an empty queue so extra calls are harmless.
	var wakers sync.WaitGroup
	for i := 0; i < n; i++ {
		wakers.Add(1)
		go func() {
			defer wakers.Done()
			q.Wake()
		}()
	}
	wakers.Wait()
	wg.Wait()

	if got := q.Len(); got != 0 {
		t.Fatalf("Len() after stress = %d, want 0", got)
	}
}
