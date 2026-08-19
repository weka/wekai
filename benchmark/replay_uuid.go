package benchmark

// Path-agnostic UUID-based response validation primitives, shared by the
// router-replay UUID cache-coherency check (replay_router_uuid.go,
// --verify + --router-replay-file) and the cache-coherency eval
// CLI (cli/eval_commands.go). Distinct UUIDs are stamped somewhere in a
// request's content; the model is later asked (directly or implicitly) to
// recall them, and the response is scored for presence (did it recall its
// own UUID?) and cross-contamination (did it leak a UUID belonging to a
// DIFFERENT session/series, i.e. a KV/scheduling leak?).
//
// This is a presence-based check (Contains), not the cache-coherency eval's
// exact-conformity check (matchesExpectedUUIDList) — replay responses are
// real multi-turn chat turns with real prose, not a bare comma-joined list.

import (
	"fmt"
	"regexp"
	"strings"
)

// uuidRe matches a canonical hyphenated UUID string (8-4-4-4-12 hex digits).
// Used by findLeakedUUIDsByOwner to pull UUID-shaped substrings out of a
// response/thinking blob without needing to know the candidate set up
// front.
var uuidRe = regexp.MustCompile(`[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}`)

// findLeakedUUIDs scans resp and thinking for UUID-shaped substrings and
// flags any that is a LIVE marker (some session currently holds it — see
// uuidRegistry) which THIS request's own prompt did not carry.
//
// The comparison is against the prompt, not against a session's assigned
// set. What the request sent is known exactly, so no ownership table and no
// corpus-wide pass is needed to decide whether a marker belongs here; and
// because a marker is derived from its block hash, a block two sessions
// genuinely share yields the same marker in both, so each holds it
// legitimately and neither is flagged for reciting it. The narrower question
// this cannot answer — did the engine serve one session's copy of a block
// that is byte-identical to another's — has no observable answer and no
// defect behind it: serving identical content from a shared prefix entry is
// what prefix caching is.
//
// Scoped per request rather than per session, it also catches a marker
// surfacing in a response whose own prompt had not reached that turn yet,
// which a per-session test cannot see at all.
//
// A UUID-shaped string no live session holds is ignored rather than flagged:
// it is either a hallucination or a marker whose sessions have all finished,
// and the registry is explicit that its window is the live set.
//
// Returns "uuid(series=N)" entries — with ",shared" when more than one live
// session holds the marker — one per distinct leak, in first-appearance
// order, so a given response always yields the same list.
func findLeakedUUIDs(resp, thinking string, own map[string]bool, reg *uuidRegistry) []string {
	if reg == nil {
		return nil
	}
	combined := resp
	if thinking != "" {
		combined = resp + "\n" + thinking
	}
	var leaked []string
	seen := map[string]bool{}
	for _, u := range uuidRe.FindAllString(combined, -1) {
		if seen[u] || own[u] {
			continue
		}
		seen[u] = true
		e := reg.lookup(u)
		if e == nil {
			continue
		}
		if e.n.Load() > 1 {
			leaked = append(leaked, fmt.Sprintf("%s(series=%d,shared)", u, e.holder.Series))
		} else {
			leaked = append(leaked, fmt.Sprintf("%s(series=%d)", u, e.holder.Series))
		}
	}
	return leaked
}

// injectUUIDMarker appends one visible marker per uuid to turnValue. Unlike
// the cache-coherency eval's <ignore>...</ignore> filler, the model MUST see
// and be able to repeat this text — it's asked to recite every ref-id later
// — so the marker is plain, readable text. Detection downstream is a raw
// substring match on the UUID itself, so the wrapper text ("[ref-id: ...]")
// is purely cosmetic and not load-bearing for scoring.
func injectUUIDMarker(turnValue string, uuids []string) string {
	if len(uuids) == 0 {
		return turnValue
	}
	var b strings.Builder
	b.WriteString(turnValue)
	for _, u := range uuids {
		b.WriteString("\n\n[ref-id: ")
		b.WriteString(u)
		b.WriteString("]")
	}
	return b.String()
}

// validateReplayResponse scores one replay response/thinking pair:
//   - found[i] reports whether expected[i] (this conversation's in-scope
//     ref-ids at this point) appears in resp or thinking (Contains, mirroring
//     the cache-coherency eval's presence check).
//   - leaked reports any OTHER conversation's UUID found in resp/thinking —
//     cross-contamination — via the shared FindLeakedUUIDs helper (moved
//     here from cli/eval_commands.go so both the coherency CLI and replay
//     validation share one implementation).
func validateReplayResponse(resp, thinking string, expected []string, convIdx int, allSets [][]string) (found []bool, leaked []string) {
	found = make([]bool, len(expected))
	for i, u := range expected {
		found[i] = strings.Contains(resp, u) || strings.Contains(thinking, u)
	}
	leaked = FindLeakedUUIDs(resp, thinking, convIdx, allSets)
	return found, leaked
}

// FindLeakedUUIDs scans resp and thinking for UUIDs belonging to a series/
// conversation OTHER than ownIdx, per the ordered allSets list (allSets[i] =
// full UUID list "owned" by index i — this doubles as the uuid -> owner
// mapping without needing an actual map, keeping iteration order — and
// therefore leak-report order — deterministic for a given seed). Returns
// "uuid(series=N)" entries, one per leaked UUID found.
//
// Exported (moved from cli/eval_commands.go) so both the cache-coherency
// eval CLI (cli/eval_commands.go, where ownIdx is a coherency series index
// and allSets is CacheCoherencyResult.SeriesUUIDs) and replay UUID
// validation above (where ownIdx is a conversation/session index and
// allSets is AutoBenchmarkConfig.replayUUIDSets) share one implementation,
// not two.
func FindLeakedUUIDs(resp, thinking string, ownIdx int, allSets [][]string) []string {
	var leaked []string
	for si, uuids := range allSets {
		if si == ownIdx {
			continue
		}
		for _, u := range uuids {
			if strings.Contains(resp, u) || strings.Contains(thinking, u) {
				leaked = append(leaked, fmt.Sprintf("%s(series=%d)", u, si))
			}
		}
	}
	return leaked
}
