package benchmark

import (
	"crypto/sha256"
	"sync"
	"sync/atomic"
)

// Online UUID registry for the router-replay cache-coherency check.
//
// A marker has to satisfy two things at once, and separating them is what
// makes this cheap. It has to leave cross-session prefix sharing intact —
// two sessions that shared a block in the capture must still send
// byte-identical content for it — and it has to let a response be scored for
// content the request never sent.
//
// Deriving the UUID from the block hash satisfies the first by construction.
// Two sessions carrying the same hash carry the same marker, so the shared
// prefix stays shared and there is nothing to discover before dispatch. What
// used to establish that property — two full streaming passes over the
// corpus, counting how many distinct sessions referenced each hash — was
// answering a question that only had to be asked because the marker was
// per-session. On the 5441-session corpus those passes cost 140s of pure
// parsing before the first request, scaled with the FILE rather than with
// the run, and gated a 0.64% tail of the turns eligible for stamping.
//
// The second is answered per request instead of per session: a leak is a
// marker of ours in the response that THIS request's own prompt did not
// carry. That needs no lookahead at all, only the set of markers currently
// live, which is what this registry holds.
//
// Refcounting falls out of the same structure. A session acquires its
// markers when it starts and releases them when it finishes; an entry with
// count > 1 is a hash genuinely shared by concurrent sessions — the number
// the precompute used to derive, now observed over the sessions that
// actually run rather than assumed from the whole file.
//
// The registry is therefore also the detection window, and a narrow one: a
// marker is unknown once every session holding it has finished, so a leak
// from a retired session reads as an unrecognised UUID-shaped string and is
// ignored rather than counted. That is a deliberately narrower claim than
// "no session ever saw another's content" and callers must report it as
// such — see Stats.
type uuidRegistry struct {
	mu sync.RWMutex
	m  map[string]*uuidEntry

	// peak is the largest the live set ever reached, and peakSessions the
	// most sessions ever holding at once, so a run can report the width of
	// its own detection window rather than leaving a reader to assume it
	// covered the corpus.
	peak         int
	liveSessions int
	peakSessions int
}

type uuidEntry struct {
	// n is the number of live sessions holding this marker. Atomic so the
	// common case — a session acquiring a marker another session already
	// registered — needs only a read lock.
	n atomic.Int64
	// series is the first series number to register it, used only to label a
	// leak. Shared markers name their first holder plus a shared flag rather
	// than pretending to a single owner.
	series int
}

func newUUIDRegistry() *uuidRegistry {
	return &uuidRegistry{m: map[string]*uuidEntry{}}
}

// Acquire registers every marker in uuids as live, held by series. Safe to
// call with duplicates; each distinct marker is counted once per call.
func (r *uuidRegistry) Acquire(uuids []string, series int) {
	if r == nil || len(uuids) == 0 {
		return
	}
	r.mu.Lock()
	r.liveSessions++
	if r.liveSessions > r.peakSessions {
		r.peakSessions = r.liveSessions
	}
	r.mu.Unlock()

	seen := make(map[string]bool, len(uuids))
	var missing []string
	r.mu.RLock()
	for _, u := range uuids {
		if seen[u] {
			continue
		}
		seen[u] = true
		if e, ok := r.m[u]; ok {
			e.n.Add(1)
		} else {
			missing = append(missing, u)
		}
	}
	r.mu.RUnlock()
	if len(missing) == 0 {
		return
	}
	r.mu.Lock()
	for _, u := range missing {
		// Re-check: another session may have inserted it between the two
		// locks, and double-inserting would drop that session's hold.
		if e, ok := r.m[u]; ok {
			e.n.Add(1)
			continue
		}
		e := &uuidEntry{series: series}
		e.n.Store(1)
		r.m[u] = e
	}
	if len(r.m) > r.peak {
		r.peak = len(r.m)
	}
	r.mu.Unlock()
}

// Release drops one hold per distinct marker, deleting entries no live
// session holds any more.
func (r *uuidRegistry) Release(uuids []string) {
	if r == nil || len(uuids) == 0 {
		return
	}
	seen := make(map[string]bool, len(uuids))
	r.mu.Lock()
	r.liveSessions--
	for _, u := range uuids {
		if seen[u] {
			continue
		}
		seen[u] = true
		if e, ok := r.m[u]; ok && e.n.Add(-1) <= 0 {
			delete(r.m, u)
		}
	}
	r.mu.Unlock()
}

// lookup returns the entry for a marker, or nil if no live session holds it.
func (r *uuidRegistry) lookup(u string) *uuidEntry {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	e := r.m[u]
	r.mu.RUnlock()
	return e
}

// Stats reports the live set and its high-water marks — the width of the
// window cross-contamination was actually checked against. peakSessions is
// the figure a reader needs: "no session saw another's content" is only ever
// true of sessions that were live at the same time, and that is how many.
func (r *uuidRegistry) Stats() (live, peak, peakSessions int) {
	if r == nil {
		return 0, 0, 0
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.m), r.peak, r.peakSessions
}

// uuidForHash derives a stable marker from a block hash and the stamp of the
// pass that is replaying it.
//
// Deterministic in (stamp, hash) alone, so within one pass the same block
// yields the same marker in every session that carries it — which is what
// keeps cross-session prefix sharing intact, the property the corpus-wide
// session counting used to buy.
//
// The stamp is the PASS stamp, not the run stamp, and the difference matters
// under --replay-reuse-sessions. A second pass over the corpus deliberately
// carries a different stamp so its content lands in a disjoint keyspace; keyed
// on the run stamp the markers would not follow, and the same session replayed
// twice would mint identical markers in both passes. Two live passes of one
// session would then look like a shared block rather than two distinct
// sessions, and a genuine leak between them would be scored as each holding
// its own.
//
// It also bounds the live set by distinct blocks per pass rather than letting
// it grow with elapsed time.
func uuidForHash(hash, stamp string) string {
	sum := sha256.Sum256([]byte(stamp + "\x00" + hash))
	return formatUUID(sum[:16])
}

// sessionUUIDs is one session's view of the stamping scheme, built when the
// session is dequeued from the material already parsed to dispatch it.
//
// Turn order is first-appearance order across the session's instances,
// requests and messages, walked in file order — the same order every request
// of that session sees, because a request carries the whole growing history
// and only a turn's first appearance defines its index.
type sessionUUIDs struct {
	// hashToTurn maps a qualifying turn's block hash to its 0-based index
	// within the session.
	hashToTurn map[string]int
	// uuids is indexed by turn, parallel to hashToTurn's values.
	uuids []string
}

// buildSessionUUIDs walks one session and returns its turn view, or nil if
// the session has no qualifying turn.
//
// A qualifying turn is a role=="user" message carrying at least one text
// block. There is deliberately no cross-session test here: a marker derived
// from the hash is identical wherever that hash appears, so stamping a block
// two sessions share perturbs neither session's prefix relative to the other
// and cannot be mistaken for a leak — both hold it legitimately, and the
// registry records the shared hold.
func buildSessionUUIDs(sess RouterReplaySession, stamp string) *sessionUUIDs {
	su := &sessionUUIDs{hashToTurn: map[string]int{}}
	for _, inst := range sess.Instances {
		for _, req := range inst.Requests {
			for _, m := range req.Messages {
				if !isQualifyingUserTurn(m) {
					continue
				}
				if _, dup := su.hashToTurn[m.Hash]; dup {
					continue
				}
				su.hashToTurn[m.Hash] = len(su.uuids)
				su.uuids = append(su.uuids, uuidForHash(m.Hash, stamp))
			}
		}
	}
	if len(su.uuids) == 0 {
		return nil
	}
	return su
}

// isQualifyingUserTurn reports whether m is a "user-input turn" for stamping:
// role=="user" with at least one text content block. tool_result-only
// messages and assistant/system messages never qualify — a marker has to land
// where the model reads it as prose it can be asked to repeat.
func isQualifyingUserTurn(m RouterReplayMessage) bool {
	if m.Role != "user" || m.Hash == "" {
		return false
	}
	for _, t := range m.BlockTypes {
		if t == "text" {
			return true
		}
	}
	return false
}
