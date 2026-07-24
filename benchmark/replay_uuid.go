package benchmark

// UUID-based response validation for the dataset-replay benchmark
// (--from-dataset, --replay-inject-uuids). Distinct per-conversation UUIDs
// are stamped into injectable turns as the conversation is walked; the model
// is periodically asked to recite every ref-id it has seen so far, and the
// response is scored for presence (did it recall its own conversation's
// UUIDs?) and cross-contamination (did it leak a UUID belonging to a
// DIFFERENT conversation, i.e. a KV/scheduling leak?).
//
// This is a presence-based check (Contains), not the cache-coherency eval's
// exact-conformity check (matchesExpectedUUIDList) — replay responses are
// real multi-turn chat turns with real prose, not a bare comma-joined list.
//
// DATASET PATH ONLY: this file must never be reached from the router-replay
// path (replay_router*.go), which reconstructs prefixes from block
// hashes+token counts rather than raw dataset text — injecting visible text
// there would break cache-hit reproduction. The CLI enforces this (see
// cli/benchmark_commands.go: --replay-inject-uuids requires --from-dataset
// and is rejected together with --router-replay-file).

import (
	"fmt"
	"strings"
)

// replayTurnInjectable reports whether turn t should carry an injected UUID
// marker under the given --replay-uuid-mode. This is the SAME predicate used
// both to precompute each conversation's UUID count (buildReplayUUIDSets) and
// to walk turns for real (runReplayConversation in replay.go) — the two MUST
// agree exactly, or the precomputed UUID list and the turn walk's cursor
// diverge mid-conversation.
//
// Modes:
//   - "human" (default): only turns from the human/user.
//   - "all-non-gpt": human, plus tool-result turns and any stray system turn
//     (a conversation's LEADING system turn is never walked at all — it
//     becomes the cached system prompt instead — so "stray" here means any
//     system turn that is NOT at index 0).
func replayTurnInjectable(t HermesTurn, mode string) bool {
	if mode == "all-non-gpt" {
		return t.From == "human" || t.From == "tool" || t.From == "system"
	}
	return t.From == "human"
}

// buildReplayUUIDSets precomputes, for every conversation in convs, the full
// ordered list of UUIDs its injectable turns will carry over the course of
// the conversation: injectableTurns * perTurn UUIDs. Every UUID across every
// conversation is drawn from a SINGLE seeded generator in conversation-major,
// turn-minor order, which is what makes the result both deterministic (same
// seed -> same output) and disjoint by construction (each draw is unique —
// the seeded PCG generator never repeats a UUID within a run).
//
// The returned slice is indexed by conversation index (parallel to convs),
// matching cfg.replayUUIDSets / cfg.replayConversations elsewhere.
func buildReplayUUIDSets(convs []Conversation, seed int64, perTurn int, mode string) [][]string {
	newUUID := newUUIDGenerator(seed)
	sets := make([][]string, len(convs))
	for i, conv := range convs {
		// Skip a LEADING system turn exactly like the real turn walk does
		// (runReplayConversation / computeInScopeAtEachGptTurn, both in
		// replay.go) — it becomes the cached system prompt, not a walked turn.
		// Without this skip, "all-non-gpt" mode (which counts system turns)
		// would count that turn here but the real walk would never consume a
		// UUID for it, drifting the precomputed count out of lockstep with
		// what's actually assigned.
		turns := conv.Turns
		if len(turns) > 0 && turns[0].From == "system" {
			turns = turns[1:]
		}
		injectable := 0
		for _, t := range turns {
			if replayTurnInjectable(t, mode) {
				injectable++
			}
		}
		n := injectable * perTurn
		if n <= 0 {
			continue
		}
		uuids := make([]string, n)
		for j := range uuids {
			uuids[j] = newUUID()
		}
		sets[i] = uuids
	}
	return sets
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

// replayReciteInstruction returns the boilerplate appended to an outgoing
// user turn, asking the model to FIRST recite every ref-id it has seen so
// far (inScope, in order) on a delimited line, THEN answer normally.
// Presence scoring is Contains-based (see validateReplayResponse), so the
// exact wording/format here is not load-bearing — the "SEEN_REFS:" delimiter
// just keeps the recited list easy to spot in a transcript/log.
func replayReciteInstruction(inScope []string) string {
	return fmt.Sprintf("\n\n(Before your normal answer, first output one line in the exact form `SEEN_REFS: %s` listing every ref-id you have seen anywhere in this conversation so far, comma-separated. Then answer normally.)",
		strings.Join(inScope, ","))
}

// replayReciteBudgetFraction caps how much of a request's max_tokens output
// budget the recited ref-id list is allowed to consume (see
// capRecitedUUIDs) — recite-every-turn means the list only grows, so without
// a cap a long conversation eventually asks the model to reproduce more
// ref-ids than max_tokens can hold, truncating the SEEN_REFS line itself.
const replayReciteBudgetFraction = 0.5

// capRecitedUUIDs trims inScope down to (at most) however many entries fit
// within replayReciteBudgetFraction of maxOutputTokens, keeping the MOST
// RECENT entries (dropping the oldest first — the model is more likely to
// have retained recent context). Returns the (possibly untouched) list and
// whether trimming occurred.
//
// This is deliberately a simple heuristic (reuses the standard
// len/4 estimateTokens idiom already used elsewhere in this package, see
// cache_sim.go) — not exact tokenizer accounting. maxOutputTokens <= 0
// (budget unknown/unbounded) disables capping entirely.
func capRecitedUUIDs(inScope []string, maxOutputTokens int) ([]string, bool) {
	if maxOutputTokens <= 0 || len(inScope) == 0 {
		return inScope, false
	}
	budget := int(float64(maxOutputTokens) * replayReciteBudgetFraction)
	if estimateTokens(strings.Join(inScope, ",")) <= budget {
		return inScope, false
	}
	perUUID := estimateTokens(inScope[0] + ",")
	if perUUID < 1 {
		perUUID = 1
	}
	n := budget / perUUID
	if n < 1 {
		n = 1
	}
	if n >= len(inScope) {
		return inScope, false
	}
	return inScope[len(inScope)-n:], true
}

// computeInScopeAtEachGptTurn walks turns exactly like runReplayConversation's
// real turn loop (replay.go) — skipping turns[0] when it's the leading system
// turn (that becomes the cached system prompt, not a walked turn) — and
// returns, for each 'gpt' turn encountered in order, a snapshot of every UUID
// assigned to an injectable turn seen so far. len(result) == the number of
// 'gpt' turns in turns (after the leading-system skip).
//
// This exists purely so the cumulative in-scope tracking is unit-testable in
// isolation (see replay_uuid_test.go); replay.go's real loop additionally
// needs the PER-TURN uuid slice (to wrap the turn text via injectUUIDMarker),
// so it maintains the same cursor/slicing logic inline rather than calling
// this function directly — the two must be kept in lockstep, which the test
// suite verifies.
func computeInScopeAtEachGptTurn(turns []HermesTurn, sets []string, perTurn int, mode string) [][]string {
	firstIdx := 0
	if len(turns) > 0 && turns[0].From == "system" {
		firstIdx = 1
	}

	var result [][]string
	var inScope []string
	cursor := 0
	for i := firstIdx; i < len(turns); i++ {
		t := turns[i]
		if t.From == "gpt" {
			result = append(result, append([]string(nil), inScope...))
			continue
		}
		if replayTurnInjectable(t, mode) {
			end := cursor + perTurn
			if end > len(sets) {
				end = len(sets)
			}
			if cursor < end {
				inScope = append(inScope, sets[cursor:end]...)
			}
			cursor = end
		}
	}
	return result
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
// and allSets is CacheCoherencyResult.SeriesUUIDs) and dataset-replay UUID
// validation above (where ownIdx is a conversation index and allSets is
// AutoBenchmarkConfig.replayUUIDSets) share one implementation, not two.
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
