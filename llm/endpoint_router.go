package llm

import (
	"sync"
	"sync/atomic"
)

// EndpointRouter assigns series to endpoints stickily — each series always uses
// the same endpoint, and initial assignment picks the endpoint with the fewest
// series (ties broken by lower index).
type EndpointRouter struct {
	endpoints   []string
	mu          sync.Mutex
	seriesMap   map[int]int // seriesNum → endpoint index
	assignments []int       // count of series assigned per endpoint
	inFlight    []atomic.Int64
}

// NewEndpointRouter creates a new EndpointRouter for the given endpoints.
func NewEndpointRouter(endpoints []string) *EndpointRouter {
	r := &EndpointRouter{
		endpoints:   endpoints,
		seriesMap:   make(map[int]int),
		assignments: make([]int, len(endpoints)),
		inFlight:    make([]atomic.Int64, len(endpoints)),
	}
	return r
}

// EndpointForSeries returns the sticky endpoint URL for the given series.
// On first call for a series, assigns to the endpoint with fewest assignments
// (ties broken by lower index). Returns "" if endpoints is empty or has 1 entry.
func (r *EndpointRouter) EndpointForSeries(seriesNum int) string {
	if len(r.endpoints) <= 1 {
		return ""
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if idx, ok := r.seriesMap[seriesNum]; ok {
		return r.endpoints[idx]
	}

	// Pick endpoint with fewest assignments (lower index breaks ties)
	best := 0
	for i := 1; i < len(r.assignments); i++ {
		if r.assignments[i] < r.assignments[best] {
			best = i
		}
	}

	r.seriesMap[seriesNum] = best
	r.assignments[best]++
	return r.endpoints[best]
}

// Acquire increments the in-flight counter for the series' endpoint.
func (r *EndpointRouter) Acquire(seriesNum int) {
	if len(r.endpoints) <= 1 {
		return
	}
	r.mu.Lock()
	idx, ok := r.seriesMap[seriesNum]
	r.mu.Unlock()
	if ok {
		r.inFlight[idx].Add(1)
	}
}

// Release decrements the in-flight counter for the series' endpoint.
func (r *EndpointRouter) Release(seriesNum int) {
	if len(r.endpoints) <= 1 {
		return
	}
	r.mu.Lock()
	idx, ok := r.seriesMap[seriesNum]
	r.mu.Unlock()
	if ok {
		r.inFlight[idx].Add(-1)
	}
}
