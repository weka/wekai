package benchmark

// UUID-based cache-coherency validation for the ROUTER-REPLAY path
// (--router-replay-file, --verify).
//
// Strategy: a deterministic UUID is injected inline into every qualifying
// user turn — a role=="user" message with >=1 text block — spread through
// the conversation rather than clustered at a session boundary. Turn i's
// marker ("\n\n[turn-N id: <uuid>]") is appended to that turn's own
// synthesized text content (see buildMessageContent's stampByHash param in
// replay_router_wire.go), keyed by the turn's first-appearance index within
// the session (see buildSessionUUIDs in replay_uuid_registry.go).
//
// The marker is derived from the block hash, not from the session (see
// uuidForHash). Content synthesis is already deterministic in that hash
// (synthText, seeded "<hash>:block:<i>"), so every request repeating a turn
// in its growing history re-emits byte-identical content for it, and two
// sessions that shared a block in the capture still send byte-identical
// content for it and still collide on the server's prefix cache exactly as
// the original traffic did. That is the fidelity invariant this feature must
// not break, and deriving from the hash satisfies it without needing to know
// anything about the rest of the corpus.
//
// Scoring is per request rather than per session: presence is whether the
// response recited the markers this request asked for, and a leak is a live
// marker in the response that this request's own prompt did not carry (see
// findLeakedUUIDs). Neither needs a marker to belong to exactly one session,
// which is what lets the whole scheme run with no precompute — see the
// registry's doc for what that replaced and what its detection window is.
//
// Each request asks the model to recite a WINDOW of turns rather than the
// whole history: the first (visible) turn plus up to 3 most-recent turns,
// EXCLUDING the current turn, deduplicated and capped at 4 (see
// replayPoster.buildInjection in replay_router_post.go and
// replayReciteWindowInstruction below). This keeps the recite cost/response
// budget constant regardless of session length while still spreading
// coverage across the whole conversation over time — turn N's stamp gets
// asked about at turns N, N+1, N+2, N+3, then ages out of the window (but
// stays warm in KV via the always-embedded StampByHash markers above, which
// cover every visible turn, not just the recited window).

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

// turnStamp is the per-user-turn UUID marker injected inline into that
// turn's own message content (see buildMessageContent's stampByHash param
// in replay_router_wire.go).
type turnStamp struct {
	Idx   int    // this session's global turn index (0-based, first-appearance order)
	UUID  string // this turn's deterministic UUID
	Label string // "turn-N" (Idx+1 — 1-based, for human-readable instructions)
}

// uuidInjection describes the per-request UUID injection to apply when
// building a router-replay wire body. A nil *uuidInjection means "no
// injection" — buildAnthropicMessagesBody / buildOpenAIChatCompletionsBody
// must behave identically to before this feature existed.
type uuidInjection struct {
	// StampByHash carries one turnStamp per user-turn message VISIBLE in
	// this request (keyed by RouterReplayMessage.Hash) — every qualifying
	// turn gets its UUID marker embedded inline in its own synthesized
	// content (see buildMessageContent), keeping every turn's stamp warm in
	// KV as later requests repeat that history in full, not just the turns
	// named in the recite window below.
	StampByHash map[string]turnStamp
	// ReciteLabels is the ordered window of turn labels ("turn-N") the
	// instruction names — first (visible) turn, then up to 3 most-recent
	// turns EXCLUDING the current turn, deduplicated, capped at 4.
	ReciteLabels []string
	// ReciteUUIDs is ReciteLabels' matching ordered UUID values — what
	// firstLineConformity/matchesExpectedUUIDList checks the response's
	// first line against.
	ReciteUUIDs []string
}

// firstLineConformity implements this feature's output-conformity check:
// whether the FIRST LINE of resp (up to the first '\n', trimmed) is EXACTLY
// the comma-separated expected UUID list, in order (see
// matchesExpectedUUIDList) — the router-replay analogue of the coherency
// eval's ExactMatch, adapted for a recite-FIRST instruction that lets the
// model keep generating after line 1 (so forced-output/ignore_eos still
// fills the remainder of the output budget with a normal continuation).
// expected is inj.ReciteUUIDs — the current request's recite WINDOW, not
// every UUID ever stamped in the session.
func firstLineConformity(resp string, expected []string) bool {
	line := resp
	if idx := strings.Index(resp, "\n"); idx >= 0 {
		line = resp[:idx]
	}
	return matchesExpectedUUIDList(strings.TrimSpace(line), expected)
}

// recall from cached KV context, not an echo of the ask.
func replayReciteWindowInstruction(labels []string) string {
	if len(labels) == 0 {
		return ""
	}
	tagged := make([]string, len(labels))
	for i, l := range labels {
		tagged[i] = "[" + l + "]"
	}
	return "\n\nSomewhere above, several ids are tagged like [turn-N id: ...]. On the FIRST line output ONLY the id values for these tags, " +
		"in this exact order, comma-separated and nothing else: " + strings.Join(tagged, ", ") + ". Then continue normally."
}

// ---- max_tokens recite floor ----

// replayReciteFloorMultiplier mirrors the cache-coherency eval's default
// --max-output-multiplier (see computeMaxOutputTokens): the recite floor is
// sized at multiplier x the expected N-UUID first-line list size, giving
// the model headroom to emit the full list without truncation.
const replayReciteFloorMultiplier = 3.0

// replayReciteFloorTokens returns the minimum max_tokens budget to enforce
// on a request that carries an injection — every request carrying a
// qualifying turn asks for a recite — sized
// to fit the FIRST-LINE numUUIDs-UUID comma-joined list this feature asks
// for (reuses the cache-coherency eval's computeMaxOutputTokens sizing:
// numUUIDs*36 chars + separating commas, /4 for an approximate token count,
// x replayReciteFloorMultiplier). numUUIDs is now len(inj.ReciteUUIDs) — the
// current request's recite WINDOW, capped at 4 (see uuidInjection) — so the
// floor itself is now bounded and constant regardless of session length,
// unlike the retired per-session-N scheme where a long session's floor grew
// without bound. The tradeoff: for a request whose recorded output budget
// is tiny (a handful of tokens, e.g. a pure tool-call turn), this constant
// floor is a much LARGER ratio of the original budget than an N-scaled
// floor would have been at N=2 — i.e. the recite ask now perturbs a small
// turn's output-size profile proportionally more. Accepted: correctness
// (not truncating the recite line into a false PRESENCE_MISS) takes
// priority over preserving the exact captured output-size ratio.
//
// A router-replay request's max_tokens is normally sized off the ORIGINAL
// capture's output_tokens (see pickMaxTokens) — which for a tool-call-only
// turn can be a handful of tokens, nowhere near enough to also emit the
// first-line UUID list. Without this floor, a tiny budget truncates that
// first line, which would misread as PRESENCE_MISS/NOT_EXACT (coherency
// failure) when it's actually just an output-size artifact.
func replayReciteFloorTokens(numUUIDs int) int {
	return computeMaxOutputTokens(numUUIDs, replayReciteFloorMultiplier)
}

var reciteFloorWarnOnce sync.Once

// applyReciteFloor raises maxTokens to replayReciteFloorTokens(numUUIDs)
// when recite is requested and the original budget falls short, warning
// once per process (mirrors the dataset path's reciteTruncWarned one-shot
// pattern, but this is a single global warning rather than per-conversation
// since the router path computes one floor value per call, not a per-
// conversation truncation computation).
func applyReciteFloor(maxTokens int, recite bool, numUUIDs int) int {
	if !recite {
		return maxTokens
	}
	floor := replayReciteFloorTokens(numUUIDs)
	if maxTokens >= floor {
		return maxTokens
	}
	reciteFloorWarnOnce.Do(func() {
		fmt.Fprintf(os.Stderr,
			"[router-replay] WARNING: max_tokens raised to the UUID-recite floor (%d) for one or more requests — "+
				"a tiny captured output budget would otherwise truncate the first-line recite list into a false PRESENCE_MISS, not real corruption\n",
			floor)
	})
	return floor
}
