// Package policy selects a backend for a request.
//
// Two properties are enforced across every policy here:
//
//   - Candidates arrive already filtered to healthy, non-draining,
//     non-open-circuit backends (LB-9). A policy never sees or reasons about an
//     unhealthy backend. v1's cache-aware policy computed its load-imbalance
//     check over the *full* worker list, so a single dead worker holding stale
//     load latched the guard permanently on and silently disabled cache routing
//     forever (CACHE-N4, LB-N7).
//   - Ties are broken by an explicit rule that is never "lowest index"
//     (LB-11). v1 used min_by_key, which returns the first minimum, so on a
//     cold fleet every request piled onto candidate 0 until it exceeded the
//     imbalance threshold — a 32-deep thundering herd on one backend while its
//     peers idled (LB-N4, CACHE-N5).
//
// This package imports no dialect package; request shape never reaches it
// (API-1).
package policy

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/registry"
)

// ErrNoCandidates reports an empty candidate set. Callers turn this into a 503
// with a distinguishable code; a policy never invents a backend (HLT-11).
var ErrNoCandidates = errors.New("policy: no candidates")

// RoutingRequest is everything a policy may know about a request.
//
// Note what is absent: headers, body, and any dialect type. Units are
// pre-extracted opaque hashes, computed once per request and reused by every
// policy and by observability (CACHE-4).
type RoutingRequest struct {
	RequestID  string
	RouteClass string
	DialectID  string
	Model      string
	Stream     bool
	// Units is nil when the route has no routable prefix, in which case cache
	// policies decline and fall back rather than failing (CU-11).
	Units    []kvcache.Unit
	Locality string
	Deadline time.Time
	// PolicyState carries opaque per-request state from Select to Commit.
	//
	// A policy whose commit depends on WHICH branch its selection took has
	// nowhere else to put that: Committer.Commit receives only the backend and
	// this struct. The affinity policy uses it to distinguish a request it
	// routed onto a cold backend deliberately, to use idle capacity, from one
	// it means to record as a genuine holder of the prefix.
	//
	// Owned by whichever policy is active and meaningless to any other. The
	// proxy may call Select more than once per request when it retries onto a
	// different backend, so the last selection wins; Commit runs at most once.
	PolicyState any
}

type Policy interface {
	Name() string
	Select(ctx context.Context, candidates []*registry.Backend, rr *RoutingRequest) (*registry.Backend, error)
}

// Committer is implemented only by cache-affinity policies. The commit happens
// after the upstream has accepted the request, never before dispatch, so a
// failed attempt leaves no trace in any cache model (CACHE-9, R3).
type Committer interface {
	Commit(b *registry.Backend, rr *RoutingRequest)
}

// direction selects whether the best score is the lowest or the highest.
//
// Parameterizing this is not cosmetic: the load policies minimize while a
// cache-usefulness scorer maximizes an expected-time-saved value. With the
// comparison hard-coded, that policy would have grown its own selection loop
// and quietly stopped enforcing the shared tie-break above (R4).
type direction bool

const (
	minimize direction = false
	maximize direction = true
)

// pickBest is the single tie-break implementation: one pass, no allocation, and
// uniform over the tied set via reservoir sampling.
func pickBest(cands []*registry.Backend, dir direction, score func(*registry.Backend) float64) *registry.Backend {
	var best *registry.Backend
	bestScore := math.Inf(1)
	if dir == maximize {
		bestScore = math.Inf(-1)
	}
	ties := 0
	for _, c := range cands {
		s := score(c)
		better := s < bestScore
		if dir == maximize {
			better = s > bestScore
		}
		switch {
		case better:
			best, bestScore, ties = c, s, 1
		case s == bestScore:
			ties++
			// Reservoir: the k-th tied element replaces the incumbent with
			// probability 1/k, giving a uniform draw over all ties in one pass.
			if rand.IntN(ties) == 0 {
				best = c
			}
		}
	}
	return best
}

// LeastOutstanding is the default policy.
//
// It compares NormalizedLoad — in-flight divided by capacity — rather than raw
// in-flight counts, so heterogeneous backends compare meaningfully (HIER-5). On
// a uniform fleet of leaves with capacity 1 this degenerates to the raw count at
// no cost.
//
// The load signal comes from the lease primitive, which is the only writer.
// That is what separates this from v1's power-of-two: the algorithm there was
// fine, but the counter it read was incremented on one path, decremented on
// three, and zeroed every ten health cycles, so it was indistinguishable from
// random (LB-N1, A2).
//
// O(N) atomic loads; ~200ns at N=64, against an NFR-2 p99 budget of 250us.
type LeastOutstanding struct{}

func (LeastOutstanding) Name() string { return "least-outstanding" }

func (LeastOutstanding) Select(_ context.Context, cands []*registry.Backend, _ *RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, ErrNoCandidates
	}
	return pickBest(cands, minimize, (*registry.Backend).NormalizedLoad), nil
}

// Random selects uniformly. Kept as a predictable baseline for tests and small
// deployments.
type Random struct{}

func (Random) Name() string { return "random" }

func (Random) Select(_ context.Context, cands []*registry.Backend, _ *RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, ErrNoCandidates
	}
	return cands[rand.IntN(len(cands))], nil
}
