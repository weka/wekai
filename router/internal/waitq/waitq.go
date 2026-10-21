// Package waitq is a small per-backend FIFO wait list.
//
// It exists so a request refused only because one specific backend is at its
// concurrency cap can wait for THAT backend's next free slot instead of either
// polling the whole fleet or spilling onto a colder one and losing prefix-cache
// locality. A Queue is usable from its zero value so it can be embedded by
// value in registry.Backend, the same way that type already embeds plain
// atomics with no constructor.
package waitq

import (
	"sync"
	"sync/atomic"
)

// Queue is a FIFO list of goroutines waiting for a slot on one backend.
//
// Wake's fast path — an atomic load before taking the lock — keeps the cost of
// "nobody is waiting" at one atomic read per lease release, which matters
// because Wake runs on every release, not only the ones with a queue behind
// them. The slice itself is guarded by mu; membership in it and a ticket's
// state are kept in lock-step so Wake and Leave can never both believe they
// own the same ticket (see Ticket.Leave).
type Queue struct {
	mu      sync.Mutex
	waiters []*Ticket
	n       atomic.Int64 // mirrors len(waiters), for the lock-free fast path
}

// Ticket is one waiter's place in line.
type Ticket struct {
	q     *Queue
	ch    chan struct{}
	state atomic.Int64 // ticketQueued, ticketWoken or ticketLeft
}

const (
	ticketQueued int64 = iota
	ticketWoken
	ticketLeft
)

// Join adds a waiter to the back of the queue, or to the front when front is
// true. max bounds the queue depth; front bypasses that bound as well as the
// ordering, for a waiter that must always get a place. Join returns false,
// with a nil ticket, when the queue is at max and front is false.
//
// A non-positive max is treated as unbounded, matching the zero-value-usable
// contract: an embedded Queue nobody has configured must not silently refuse
// every Join.
func (q *Queue) Join(max int, front bool) (*Ticket, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if !front && max > 0 && len(q.waiters) >= max {
		return nil, false
	}
	t := &Ticket{q: q, ch: make(chan struct{}, 1)}
	if front {
		q.waiters = append([]*Ticket{t}, q.waiters...)
	} else {
		q.waiters = append(q.waiters, t)
	}
	q.n.Store(int64(len(q.waiters)))
	return t, true
}

// Wake pops the head of the queue and signals it. No-op if the queue is empty.
//
// It is the single wake source for a backend's queue, called from
// lease.Release after every completed attempt on that backend, and again by
// Ticket.Leave when a woken ticket declines its slot — so a release always
// finds, eventually, a waiter that actually uses it.
func (q *Queue) Wake() {
	if q.n.Load() == 0 {
		return // fast path: no lock for the overwhelmingly common case
	}
	q.mu.Lock()
	if len(q.waiters) == 0 {
		q.mu.Unlock()
		return
	}
	t := q.waiters[0]
	q.waiters = q.waiters[1:]
	q.n.Store(int64(len(q.waiters)))
	q.mu.Unlock()

	// t was in the slice, so by the membership invariant (see Ticket.Leave)
	// nothing else can have transitioned it yet: this store cannot race
	// Leave's.
	t.state.Store(ticketWoken)
	t.ch <- struct{}{} // buffered(1); exactly one send in a ticket's life
}

// Len reports the current queue depth.
func (q *Queue) Len() int { return int(q.n.Load()) }

// C returns the channel that receives once when this ticket is woken.
func (t *Ticket) C() <-chan struct{} { return t.ch }

// Leave gives up the ticket: the waiter timed out, its caller disconnected, or
// it was woken but ended up served elsewhere and does not need the slot.
//
// Membership in the queue's slice and a ticket's state are changed together,
// under the queue's lock, so exactly one of Wake and Leave ever removes a
// given ticket from the slice. Whichever loses that race finds the ticket
// already gone and knows a wake was (or is about to be) delivered for it —
// and since this ticket is declining it, Leave passes it on to the next
// waiter rather than letting it strand that slot.
func (t *Ticket) Leave() {
	q := t.q
	q.mu.Lock()
	for i, w := range q.waiters {
		if w == t {
			q.waiters = append(q.waiters[:i], q.waiters[i+1:]...)
			q.n.Store(int64(len(q.waiters)))
			q.mu.Unlock()
			t.state.Store(ticketLeft)
			return
		}
	}
	q.mu.Unlock()
	// Not in the slice: Wake already popped it (or is in the middle of doing
	// so). Either way a wake is spoken for here and would otherwise be lost,
	// so hand it to whoever is next.
	t.state.Store(ticketLeft)
	q.Wake()
}
