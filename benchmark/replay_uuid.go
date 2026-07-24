package benchmark

// Path-agnostic UUID-based response validation primitives, shared by the
// router-replay UUID cache-coherency check (replay_router_uuid.go,
// --replay-inject-uuids + --router-replay-file) and the cache-coherency eval
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
	"strings"
)

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
