package benchmark

import (
	"sort"
	"time"
)

// truncateSessionRequests keeps a session's first maxRequests requests in
// CAPTURE ORDER and drops the rest, along with any instance left holding none.
// It reports how many requests were dropped; 0 means the session was already
// within the cap and was not touched.
//
// A capture's sessions are wildly uneven — on the July corpus the median holds
// 17 requests and one holds 26,253, 42% of a 256-session slice by itself — so a
// run over such a corpus spends most of its time on a handful of sessions, and
// the fleet is measured on their shape rather than the workload's. Capping per
// session normalizes that: it is a tax on the outliers, since the median
// session is far below any cap worth setting.
//
// Chronological and not per-instance-prefix, because that is what keeps the
// spawn tree consistent. A parent's spawn-bearing request always precedes the
// child's first request in time, so a cut can orphan a child only by cutting
// everything the child would have run — leaving it with no requests, and
// dropped here. The reverse, a surviving child whose parent's spawn was cut,
// cannot happen. Truncating each instance independently gives up that property:
// it keeps late children whose parents were cut short, and they then fire as
// roots because their parent request is gone from the wait map.
//
// A request with an unparseable timestamp sorts to the end rather than the
// beginning, so a corpus missing them degrades to instance order instead of
// promoting untimed requests over real ones.
func truncateSessionRequests(sess *RouterReplaySession, maxRequests int) int {
	if maxRequests <= 0 {
		return 0
	}
	total := 0
	for i := range sess.Instances {
		total += len(sess.Instances[i].Requests)
	}
	if total <= maxRequests {
		return 0
	}

	type ref struct {
		inst, req int
		at        time.Time
		timed     bool
	}
	refs := make([]ref, 0, total)
	for i := range sess.Instances {
		for j, r := range sess.Instances[i].Requests {
			at, err := time.Parse(time.RFC3339Nano, r.Ts)
			refs = append(refs, ref{inst: i, req: j, at: at, timed: err == nil})
		}
	}
	// Ties and untimed requests fall back to the file's own order, so the same
	// corpus and cap always truncate to the same set — a run whose content
	// varied between passes would not be comparable with itself.
	sort.SliceStable(refs, func(a, b int) bool {
		x, y := refs[a], refs[b]
		if x.timed != y.timed {
			return x.timed
		}
		if x.timed && !x.at.Equal(y.at) {
			return x.at.Before(y.at)
		}
		if x.inst != y.inst {
			return x.inst < y.inst
		}
		return x.req < y.req
	})

	keep := make([][]bool, len(sess.Instances))
	for i := range sess.Instances {
		keep[i] = make([]bool, len(sess.Instances[i].Requests))
	}
	for _, r := range refs[:maxRequests] {
		keep[r.inst][r.req] = true
	}

	kept := make([]RouterReplayInstance, 0, len(sess.Instances))
	for i := range sess.Instances {
		inst := sess.Instances[i]
		reqs := make([]RouterReplayRequest, 0, len(inst.Requests))
		for j, r := range inst.Requests {
			if keep[i][j] {
				reqs = append(reqs, r)
			}
		}
		if len(reqs) == 0 {
			// Nothing left to run. Kept, it would still take a done-channel and
			// a goroutine, and its children would wait on a request that is
			// never sent.
			continue
		}
		inst.Requests = reqs
		kept = append(kept, inst)
	}
	sess.Instances = kept
	return total - maxRequests
}
