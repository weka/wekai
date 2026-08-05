package policy

import (
	"context"
	"math"
	"sync"

	"github.com/weka/wekai/router/internal/registry"
)

// RoundRobin is a virtual-time scheduler: it always serves the
// least-recently-served candidate.
//
// The obvious implementation — an atomic cursor modulo len(candidates) — is
// rejected, and it is worth being explicit about why, because v1 shipped it and
// it fails two independent ways:
//
//  1. The modulus changes when the candidate set resizes, so index k suddenly
//     denotes a different backend and the rotation restarts at an arbitrary
//     phase. Combined with v1 returning worker lists from a DashMap in
//     nondeterministic order, "round-robin" was indistinguishable from random
//     (LB-N3, B1, B2).
//  2. A backend that is briefly unhealthy loses its slot permanently once the
//     set grows back, because the cursor keeps marching while index k now means
//     someone else. That backend can be starved indefinitely.
//
// Keying last-served on the backend rather than on a position fixes both.
// last[url] is a fact about that backend, so removing and re-adding it does not
// change its position in the rotation: on return it holds the oldest sequence in
// the set and is served next. It is *compensated*, not skipped.
//
// Over 10N requests to N stable backends each is served exactly 10 times,
// because "serve the least-recently-served" is a strict rotation on a stable
// set.
type RoundRobin struct {
	mu   sync.Mutex
	seq  int64
	last map[string]int64
}

func NewRoundRobin() *RoundRobin {
	return &RoundRobin{last: map[string]int64{}}
}

func (p *RoundRobin) Name() string { return "round-robin" }

func (p *RoundRobin) Select(_ context.Context, cands []*registry.Backend, _ *RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, ErrNoCandidates
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	// A newcomer enters the rotation at the FRONT, inheriting the minimum
	// last-served sequence among current candidates, so it competes on equal
	// terms immediately. Entering at the back would let a steady trickle of new
	// backends starve behind incumbents.
	min := int64(math.MaxInt64)
	seen := 0
	for _, c := range cands {
		if v, ok := p.last[c.URL]; ok {
			seen++
			if v < min {
				min = v
			}
		}
	}
	if seen == 0 {
		min = p.seq
	}
	for _, c := range cands {
		if _, ok := p.last[c.URL]; !ok {
			p.last[c.URL] = min
		}
	}

	// Deliberately NOT the shared reservoir tie-break. Here the ordering is
	// total and meaningful, so a random tie-break would break the
	// exactly-10-each guarantee. Canonical URL is the deterministic tiebreak
	// (LB-10).
	best := cands[0]
	bestV := p.last[best.URL]
	for _, c := range cands[1:] {
		v := p.last[c.URL]
		if v < bestV || (v == bestV && c.URL < best.URL) {
			best, bestV = c, v
		}
	}

	p.seq++
	p.last[best.URL] = p.seq
	p.pruneLocked(cands)
	return best, nil
}

// pruneLocked bounds the map against backends that have gone away for good.
//
// Pruning is lazy and generous on purpose: an entry must survive a transient
// health flap or a brief drain, or the backend would return as a "newcomer" and
// jump the queue. Only once the map has grown well past the live set do we drop
// entries not currently present.
func (p *RoundRobin) pruneLocked(cands []*registry.Backend) {
	if len(p.last) <= 4*len(cands)+16 {
		return
	}
	live := make(map[string]struct{}, len(cands))
	for _, c := range cands {
		live[c.URL] = struct{}{}
	}
	for url := range p.last {
		if _, ok := live[url]; !ok {
			delete(p.last, url)
		}
	}
}
