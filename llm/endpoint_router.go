package llm

import (
	"hash/fnv"
	"sort"
	"sync"
	"sync/atomic"
)

// DefaultOverloadThreshold is the multiple of the well-balanced in-flight share
// at which an endpoint is considered overloaded and requests fail over.
const DefaultOverloadThreshold = 1.5

// EndpointRouter assigns series to endpoints stickily — each series always
// prefers the same endpoint — but fails over when that endpoint is carrying
// disproportionately more in-flight requests than its fair share.
//
// Why failover exists: sticky assignment balances the number of SERIES per
// endpoint, which is not the same as balancing concurrent REQUESTS. Router
// replay fans out (one turn can spawn many concurrent sub-agent instances), so
// two endpoints owning identical series counts can differ several-fold in
// instantaneous load. Measured on a 6-node fleet: one endpoint sat pinned at 64
// concurrent requests, returning HTTP 429 "server at capacity", while another
// idled at 19 — with series counts exactly equal (256 each) throughout.
//
// Failover targets are chosen by hashing the set of endpoints already rejected,
// so the choice is deterministic and spreads across the remaining endpoints
// instead of stampeding onto whichever one happens to be next in index order.
type EndpointRouter struct {
	endpoints []string

	mu          sync.Mutex
	seriesMap   map[int]int // seriesNum → home endpoint index
	assignments []int       // count of series assigned per endpoint

	inFlight []atomic.Int64 // live in-flight requests per endpoint

	// wellBalanced is the per-endpoint in-flight count we would see if load
	// were spread perfectly: total benchmark concurrency / endpoint count.
	// Zero disables failover entirely (router degrades to pure stickiness).
	wellBalanced int64
	threshold    float64
}

// NewEndpointRouter creates a router with sticky assignment and no failover.
func NewEndpointRouter(endpoints []string) *EndpointRouter {
	return NewEndpointRouterWithFailover(endpoints, 0, DefaultOverloadThreshold)
}

// NewEndpointRouterWithFailover creates a router that fails over once an
// endpoint exceeds threshold × its fair share of in-flight requests.
//
// totalConcurrency is the benchmark's aggregate concurrency across all
// endpoints — for router replay that is Concurrency + HotSeriesConcurrency.
// With 168 + 24 over 6 endpoints the fair share is 32, and at the default
// threshold of 1.5 an endpoint fails over above 48 in flight.
func NewEndpointRouterWithFailover(endpoints []string, totalConcurrency int, threshold float64) *EndpointRouter {
	if threshold <= 0 {
		threshold = DefaultOverloadThreshold
	}
	var wb int64
	if n := len(endpoints); n > 0 && totalConcurrency > 0 {
		wb = int64(totalConcurrency / n)
		if wb < 1 {
			wb = 1
		}
	}
	return &EndpointRouter{
		endpoints:    endpoints,
		seriesMap:    make(map[int]int),
		assignments:  make([]int, len(endpoints)),
		inFlight:     make([]atomic.Int64, len(endpoints)),
		wellBalanced: wb,
		threshold:    threshold,
	}
}

// HomeIndexForSeries returns the sticky endpoint index for a series, assigning
// one on first call (fewest series so far; lower index breaks ties).
func (r *EndpointRouter) HomeIndexForSeries(seriesNum int) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	if idx, ok := r.seriesMap[seriesNum]; ok {
		return idx
	}
	best := 0
	for i := 1; i < len(r.assignments); i++ {
		if r.assignments[i] < r.assignments[best] {
			best = i
		}
	}
	r.seriesMap[seriesNum] = best
	r.assignments[best]++
	return best
}

// EndpointForSeries returns the sticky endpoint URL for the given series,
// ignoring load. Returns "" if there is nothing to route between.
func (r *EndpointRouter) EndpointForSeries(seriesNum int) string {
	if len(r.endpoints) <= 1 {
		return ""
	}
	return r.endpoints[r.HomeIndexForSeries(seriesNum)]
}

// overloaded reports whether endpoint idx is carrying more than its allowed
// share of in-flight requests.
func (r *EndpointRouter) overloaded(idx int) bool {
	if r.wellBalanced <= 0 {
		return false
	}
	limit := int64(float64(r.wellBalanced) * r.threshold)
	if limit < 1 {
		limit = 1
	}
	return r.inFlight[idx].Load() > limit
}

// hashPick deterministically chooses one of the endpoints not in skipped.
// The hash is taken over the whole skipped set, so the second choice depends on
// the first rejection, the third on the first two, and so on. That keeps the
// decision reproducible for a given rejection history while sending different
// histories to different endpoints, rather than every overloaded endpoint
// spilling onto the same neighbour.
func (r *EndpointRouter) hashPick(skipped []int) int {
	n := len(r.endpoints)
	inSkipped := make([]bool, n)
	for _, s := range skipped {
		if s >= 0 && s < n {
			inSkipped[s] = true
		}
	}
	remaining := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if !inSkipped[i] {
			remaining = append(remaining, i)
		}
	}
	if len(remaining) == 0 {
		return -1
	}

	// Sort so the hash input depends on the SET of rejected endpoints, not the
	// order they were rejected in — two paths that rejected the same endpoints
	// then agree on the next pick.
	ordered := append([]int(nil), skipped...)
	sort.Ints(ordered)

	h := fnv.New64a()
	var buf [8]byte
	for _, s := range ordered {
		v := uint64(s) + 1 // +1 so index 0 still perturbs the hash
		for i := 0; i < 8; i++ {
			buf[i] = byte(v >> (8 * i))
		}
		_, _ = h.Write(buf[:])
	}
	return remaining[h.Sum64()%uint64(len(remaining))]
}

// PickIndex returns the endpoint index a request for this series should use:
// the series' home endpoint, or — if that endpoint is overloaded — a
// deterministically hash-selected alternative that is not. If every endpoint is
// overloaded there is nowhere better to go, so it returns home and lets the
// server queue or reject.
//
// This is the read-only form; concurrent callers can all observe the same
// under-limit endpoint and pile onto it. Prefer AcquireForRequest, which makes
// the decision and the accounting atomic.
func (r *EndpointRouter) PickIndex(seriesNum int) int {
	home := r.HomeIndexForSeries(seriesNum)
	return r.pickLocked(home, func(i int) bool { return r.overloaded(i) })
}

// pickLocked walks home -> hash-selected alternatives until isOverloaded says
// no. Returns home when everything is saturated.
func (r *EndpointRouter) pickLocked(home int, isOverloaded func(int) bool) int {
	n := len(r.endpoints)
	if n == 0 {
		return -1
	}
	if n == 1 || !isOverloaded(home) {
		return home
	}
	skipped := []int{home}
	for len(skipped) < n {
		cand := r.hashPick(skipped)
		if cand < 0 {
			break
		}
		if !isOverloaded(cand) {
			return cand
		}
		skipped = append(skipped, cand)
	}
	return home
}

// AcquireForRequest picks an endpoint for ONE request and takes its in-flight
// slot in the same critical section, so concurrent callers see each other's
// increments. Without that atomicity a fan-out group — whose siblings share a
// seriesNum and therefore a home endpoint — can have every sibling read
// "under the limit" before any of them increments, and all land on the same
// endpoint. That is exactly the burst this router exists to spread.
//
// The caller must call ReleaseIndex with the returned index once the request
// completes. Returns -1 when there is nothing to route between.
func (r *EndpointRouter) AcquireForRequest(seriesNum int) int {
	if len(r.endpoints) <= 1 {
		return -1
	}
	home := r.HomeIndexForSeries(seriesNum)

	r.mu.Lock()
	defer r.mu.Unlock()
	idx := r.pickLocked(home, r.overloadedLocked)
	if idx >= 0 {
		r.inFlight[idx].Add(1)
	}
	return idx
}

// overloadedLocked is overloaded() for use inside the router mutex. The
// counters are atomics, so the read itself needs no lock — holding the mutex is
// what makes check-then-increment indivisible against other pickers.
func (r *EndpointRouter) overloadedLocked(idx int) bool {
	return r.overloaded(idx)
}

// PickEndpoint is PickIndex as a URL, with the index it resolved to. Returns
// ("", -1) when there is nothing to route between, matching EndpointForSeries
// so callers can treat "" as "no override".
func (r *EndpointRouter) PickEndpoint(seriesNum int) (string, int) {
	if len(r.endpoints) <= 1 {
		return "", -1
	}
	idx := r.PickIndex(seriesNum)
	if idx < 0 {
		return "", -1
	}
	return r.endpoints[idx], idx
}

// Endpoints returns the routed endpoint URLs in index order.
func (r *EndpointRouter) Endpoints() []string {
	return append([]string(nil), r.endpoints...)
}

// AcquireIndex increments the in-flight counter for an endpoint index.
func (r *EndpointRouter) AcquireIndex(idx int) {
	if idx >= 0 && idx < len(r.inFlight) {
		r.inFlight[idx].Add(1)
	}
}

// ReleaseIndex decrements the in-flight counter for an endpoint index.
func (r *EndpointRouter) ReleaseIndex(idx int) {
	if idx >= 0 && idx < len(r.inFlight) {
		r.inFlight[idx].Add(-1)
	}
}

// InFlight returns the current in-flight count per endpoint, for reporting.
func (r *EndpointRouter) InFlight() []int64 {
	out := make([]int64, len(r.inFlight))
	for i := range r.inFlight {
		out[i] = r.inFlight[i].Load()
	}
	return out
}

// WellBalanced returns the per-endpoint in-flight share considered balanced,
// and the threshold multiple above which failover kicks in.
func (r *EndpointRouter) WellBalanced() (int64, float64) {
	return r.wellBalanced, r.threshold
}

// Acquire increments the in-flight counter for the series' home endpoint.
//
// Deprecated: prefer AcquireIndex with the index actually chosen by PickIndex —
// with failover the request may not have gone to the home endpoint.
func (r *EndpointRouter) Acquire(seriesNum int) {
	if len(r.endpoints) <= 1 {
		return
	}
	r.mu.Lock()
	idx, ok := r.seriesMap[seriesNum]
	r.mu.Unlock()
	if ok {
		r.AcquireIndex(idx)
	}
}

// Release decrements the in-flight counter for the series' home endpoint.
//
// Deprecated: prefer ReleaseIndex, for the same reason as Acquire.
func (r *EndpointRouter) Release(seriesNum int) {
	if len(r.endpoints) <= 1 {
		return
	}
	r.mu.Lock()
	idx, ok := r.seriesMap[seriesNum]
	r.mu.Unlock()
	if ok {
		r.ReleaseIndex(idx)
	}
}
