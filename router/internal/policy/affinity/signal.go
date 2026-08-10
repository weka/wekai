package affinity

import (
	"sync"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

// A signal answers one question about one backend: should it be treated as
// unable to take more work right now, and if so, what in-flight level counts as
// "as loaded as it" for the split guard.
//
// The second half is not decoration. Anton's rule is relative — "we already
// have 32 in the air by inflight atomic, so if candidate has 30 in the air we
// are not splitting onto it" — so the guard needs the level that made this
// backend unusable, not a number from the config file. A signal that fires
// without supplying one would leave the guard measuring against nothing.
//
// Signals never route. They only shrink the usable set and set the guard's
// reference; the flow is the same whichever ones are enabled.
type signal interface {
	name() string
	// saturated reports whether b cannot take more work, and the reference
	// in-flight level for the guard when it cannot.
	saturated(b *registry.Backend, view loadView) (bool, int64)
}

// loadView is the fleet context a signal may need, computed once per decision
// so a signal never walks the candidate list itself.
type loadView struct {
	minInflight int64
}

// ---------------------------------------------------------------- refused

// refusedSignal is the ultimate signal and the only one always on: the backend
// itself answered 429.
//
// Nothing else is ground truth about a vLLM's capacity. A router-side limit is
// a guess at --max-num-seqs, an imbalance is a heuristic; a 429 is the engine
// saying it will not take this request. The flow is built on this one and every
// other signal is an opt-in early warning that saves the round trip.
//
// A refusal is LATCHED AGAINST THE IN-FLIGHT COUNT IT HAPPENED AT, not merely
// against the clock. The backend told us something precise — "I cannot take
// more while I am carrying N" — so it is treated as saturated while its
// in-flight is still at or above N, and becomes usable again the moment that
// falls. Nothing needs to expire for recovery to happen.
//
// This is what bounds retry multiplication. When a holder refuses, the flow
// fails over to the next holder, and the next, before it will consider a split;
// under fleet-wide saturation a purely time-based latch would let each of those
// requests walk back into backends that have already said no, and the retries
// would multiply exactly when the fleet can least afford them. Keyed to
// in-flight, a backend that has refused is skipped by every subsequent failover
// until it genuinely has room.
//
// The TTL remains as a backstop only, for the entry itself: a backend that
// refuses and then goes completely quiet at the same in-flight level would
// otherwise hold its latch forever. A success clears it early.
type refusedSignal struct {
	clk clock.Clock
	ttl time.Duration

	mu  sync.RWMutex
	hot map[string]refusal
}

type refusal struct {
	at time.Time
	// inflight is what the backend was carrying when it refused — the guard's
	// reference, observed rather than configured. This is the number Anton's
	// rule is written against.
	inflight int64
}

func newRefusedSignal(clk clock.Clock, ttl time.Duration) *refusedSignal {
	return &refusedSignal{clk: clk, ttl: ttl, hot: map[string]refusal{}}
}

func (r *refusedSignal) name() string { return "refused" }

// record latches a 429 from b. Called from the proxy once the upstream status
// is known, never from the routing path.
func (r *refusedSignal) record(b *registry.Backend) {
	if b == nil {
		return
	}
	r.mu.Lock()
	r.hot[b.URL] = refusal{at: r.clk.Now(), inflight: b.Inflight()}
	r.mu.Unlock()
}

// clear drops b's latch after a successful response. Recovery is immediate:
// waiting out the TTL after the backend has demonstrably taken work would keep
// a healthy node out of its own prefixes for no reason.
func (r *refusedSignal) clear(b *registry.Backend) {
	if b == nil {
		return
	}
	r.mu.RLock()
	_, latched := r.hot[b.URL]
	r.mu.RUnlock()
	if !latched {
		return // the common case: no write lock on every successful request
	}
	r.mu.Lock()
	delete(r.hot, b.URL)
	r.mu.Unlock()
}

func (r *refusedSignal) saturated(b *registry.Backend, _ loadView) (bool, int64) {
	r.mu.RLock()
	f, ok := r.hot[b.URL]
	r.mu.RUnlock()
	if !ok {
		return false, 0
	}
	if r.clk.Now().Sub(f.at) > r.ttl {
		return false, 0
	}
	// It refused while carrying f.inflight. Below that it has demonstrably
	// freed a slot since, so the refusal no longer describes it.
	if b.Inflight() < f.inflight {
		return false, 0
	}
	return true, f.inflight
}

// ---------------------------------------------------------------- concurrency

// concurrencySignal is the router's own guess at the backend's concurrency
// ceiling, enabled by setting --max-node-concurrency.
//
// It exists to avoid paying a wasted round trip per saturation event: a real
// 429 only arrives after the request has been dispatched and refused. Set it to
// the backends' vLLM --max-num-seqs and saturation is predicted rather than
// discovered. Set it wrong and it is merely early or late; the refused signal
// still backstops it, which is why this one is opt-in and that one is not.
type concurrencySignal struct{ limit int64 }

func (concurrencySignal) name() string { return "concurrency" }

func (c concurrencySignal) saturated(b *registry.Backend, _ loadView) (bool, int64) {
	if b.Inflight() >= c.limit {
		metrics.BackendCapExceeded.WithLabelValues(b.URL).Inc()
		return true, c.limit
	}
	return false, 0
}

// ---------------------------------------------------------------- imbalance

// imbalanceSignal treats a backend as unusable while it is carrying
// disproportionately more than the least-loaded backend in the fleet, enabled
// by setting --rebalance-ratio.
//
// The test is relative to the HIGHER side: b is unusable while
//
//	(inflight(b) - min) / inflight(b) > ratio
//
// so at 0.5 a holder on 10 is unusable against a fleet minimum of 4, and usable
// against a minimum of 6. Expressing it against the higher side rather than as
// an absolute gap is what makes one number work at any fleet size or request
// rate; the pair of absolute-and-relative thresholds this replaces
// (balance_abs_threshold / balance_rel_threshold, from the retired
// prefix-cache-aware policy) needed retuning per deployment.
//
// A ratio alone is not enough, because it degenerates exactly where it matters
// least. The fleet minimum is 0 whenever any backend is momentarily idle, and
// then (load - 0)/load is 1.0 for every backend carrying anything at all — so
// no ratio below 1.0 can fail, and a fleet of 1,1,1,0 is called as imbalanced
// as one of 20,20,20,0. Those two are ratio-IDENTICAL: same minimum, same
// proportions. Only their magnitude differs, so only an absolute term can tell
// them apart.
//
// Hence minRebalanceLoad: a backend carrying less than that is never treated as
// overloaded, whatever the proportions say. Nothing is under pressure there, and
// copying a prefix away from it costs more than the imbalance does. Above it the
// ratio decides, as before.
//
// This is the signal that trades locality for evenness: a fleet where affinity
// is working is SUPPOSED to look imbalanced. Setting the ratio to 0 turns it
// off for a deployment that values locality more.
type imbalanceSignal struct{ ratio float64 }

// minRebalanceLoad is the in-flight floor, in requests, below which a backend is
// never rebalanced away from. Deliberately a constant and not a knob: it does
// not separate deployments, it separates "this backend is under pressure" from
// "it is not", and that boundary is the same everywhere.
const minRebalanceLoad = 8

func (imbalanceSignal) name() string { return "imbalance" }

func (i imbalanceSignal) saturated(b *registry.Backend, view loadView) (bool, int64) {
	load := b.Inflight()
	if load <= 0 || load <= view.minInflight {
		return false, 0
	}
	// Both tests, and the absolute one first because it is the cheap one and
	// the one that rejects the degenerate cases.
	if load < minRebalanceLoad {
		return false, 0
	}
	if float64(load-view.minInflight)/float64(load) > i.ratio {
		return true, load
	}
	return false, 0
}
