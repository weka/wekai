// Package lease is the load-accounting primitive, and the reason this rewrite
// exists.
//
// v1's in-flight counter was incremented on exactly one code path (the
// cache-aware selection branch) and decremented on three — normal completion,
// retryable-status handling, and body-read error — so a backend returning 503
// was decremented twice for one increment. The health checker then zeroed every
// counter every ten cycles. Downstream, power-of-two degenerated to random, the
// cache-aware imbalance guard latched on stale load from dead workers, and the
// worker_load / max_load / min_load gauges reported noise to operators.
//
// The fix is not a better formula. It is making the lifecycle symmetric by
// construction: ONE increment site, and a release that is idempotent, so the
// number of call sites stops mattering.
//
//	lse := lease.Acquire(b)
//	defer lse.Release()      // safe to call again anywhere; first wins
//
// Satisfies LB-1..LB-7.
package lease

import (
	"log/slog"
	"sync"
	"sync/atomic"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

// accountingErrors counts detected underflows. Any non-zero value is a bug in
// the router, not a condition to tolerate — the release gate for a canary is
// that this stays at zero (LB-5, AC-R1).
var accountingErrors atomic.Int64

// AccountingErrors reports the number of detected in-flight underflows.
func AccountingErrors() int64 { return accountingErrors.Load() }

// ResetAccountingErrors is for tests only.
func ResetAccountingErrors() { accountingErrors.Store(0) }

// Lease represents one backend's share of in-flight work for one attempt.
//
// Leases are never transferred or reused. A retry to a different backend
// releases the first lease and acquires a new one (LB-3).
type Lease struct {
	b    *registry.Backend
	once sync.Once
}

// Acquire is the ONLY place in the program that increments in-flight load
// (LB-1). It is called immediately after a backend is selected and before the
// upstream request is issued, for every policy without exception.
func Acquire(b *registry.Backend) *Lease {
	b.AddInflight(+1)
	b.InflightGauge.Inc()
	return &Lease{b: b}
}

// Release returns the lease. It is idempotent and safe from any number of
// goroutines (LB-2), and nil-receiver-safe so that
//
//	var lse *lease.Lease
//	defer lse.Release()
//
// is valid before the acquire has happened.
//
// Idempotency is what makes correctness independent of call-site count: the
// retry loop releases explicitly at the end of each attempt, the proxy releases
// in a defer, and a panic releases during unwinding. All three can fire; only
// the first has an effect.
//
// For streaming responses the release must not happen until the body is fully
// read or the stream aborts — not when headers arrive (LB-4). Callers get this
// by deferring inside the scope that wraps the body copy; v1 charged the decode
// worker's load at header time, which is exactly the long phase, so measured
// load for streaming traffic was ~0.
func (l *Lease) Release() {
	if l == nil {
		return
	}
	l.once.Do(func() {
		if n := l.b.AddInflight(-1); n < 0 {
			// Never wrap. A negative count means the accounting invariant is
			// broken somewhere; say so loudly and clamp (LB-5).
			accountingErrors.Add(1)
			// Both: the local counter is what tests assert on, the collector is
			// what an operator alerts on. Incrementing only the local one made the
			// exported metric permanently zero — indistinguishable from healthy.
			metrics.LoadAccountingErrors.Inc()
			slog.Error("in-flight underflow: this is a bug in load accounting",
				"backend", l.b.URL, "value", n)
			l.b.StoreInflight(0)
		}
		l.b.InflightGauge.Dec()
	})
}

// Backend returns the leased backend.
func (l *Lease) Backend() *registry.Backend {
	if l == nil {
		return nil
	}
	return l.b
}
