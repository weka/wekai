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
	"strings"
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
	// BudgetShort marks a request whose captured output budget cannot carry
	// even one id (see reciteCapacity). It is asked nothing and scored on
	// nothing: a question the response had no room to answer must not be
	// recorded as a wrong answer, and the count of these is reported so the
	// validated population is never mistaken for the whole run.
	BudgetShort bool
	// ReciteLabels is the ordered window of turn labels ("turn-N") the
	// instruction names — first (visible) turn, then up to 3 most-recent
	// turns EXCLUDING the current turn, deduplicated, capped at 4.
	ReciteLabels []string
	// ReciteUUIDs is ReciteLabels' matching ordered UUID values — what
	// firstLineConformity/matchesExpectedUUIDList checks the response's
	// first line against.
	ReciteUUIDs []string
}

// firstLineConformity implements this feature's output-conformity check: the
// response LEADS with the expected ordered, comma-joined GUID list.
//
// Leads with — not "the first line equals". The strict form was measured
// against a real fleet and turned out to test whether the model emits a
// newline after complying: 31 of 33 responses began with the complete,
// correctly-ordered list and then continued on the same line, and rewording
// the instruction moved nothing. A newline is a decoding habit with no
// bearing on ordering, position or completeness, which are the properties
// this check exists to add on top of presence. A response that buries the
// list mid-prose, reorders it, or leads with something else still fails.
func firstLineConformity(resp string, expected []string) bool {
	if len(expected) == 0 {
		return false
	}
	return strings.HasPrefix(strings.TrimSpace(resp), strings.Join(expected, ", "))
}

// recall from cached KV context, not an echo of the ask.
func replayReciteWindowInstruction(labels []string) string {
	if len(labels) == 0 {
		return ""
	}
	// The turn names are listed WITHOUT brackets, and the ask is phrased around
	// the guid rather than the tag.
	//
	// The previous wording ended with the bracketed tags themselves —
	// "...comma-separated and nothing else: [turn-1], [turn-50], ..." — and a
	// trailing list immediately after "output only" reads as the thing to
	// output. Measured against a real session, four responses in eight copied
	// that list back verbatim and scored zero, which the summary then reported
	// as PRESENCE_MISS: a prompt defect arriving as a cache-coherency figure.
	//
	// Naming what a guid looks like, showing one worked example, and saying
	// outright that the turn names are not the answer removes every reading in
	// which echoing the list is correct.
	// The first line's terms are spelled out to the point of pedantry because
	// each clause closes a failure that was actually observed. Naming the
	// turns without brackets stops the tag list being echoed back; "end the
	// line immediately after the last GUID" stops prose being appended to a
	// correct list with no newline; "do not repeat any GUID" stops the
	// degenerate tail-babble a smaller model produces after complying. None
	// of these change what is measured — presence is a substring test — but
	// each rescues the conformity signal from a failure mode that is about
	// formatting, not about the fleet.
	return "\n\nSome messages above end with a tag of the form [turn-N id: GUID], where GUID is a " +
		"36-character identifier like 550e8400-e29b-41d4-a716-446655440000. On the FIRST line of your " +
		"reply output ONLY those GUID values, comma-separated, for these turns in this order: " +
		strings.Join(labels, ", ") + ". End that line immediately after the last GUID and start a new " +
		"line — no words, symbols or repeated GUIDs may follow on the first line, and no GUID appears " +
		"twice. Output the GUID values themselves — never the turn names, and never the square " +
		"brackets. For example, if a message ends with " +
		"[turn-7 id: 550e8400-e29b-41d4-a716-446655440000] then for turn-7 you output " +
		"550e8400-e29b-41d4-a716-446655440000. From the second line on, continue normally."
}

// ---- how many ids this request's own output budget can carry ----

// reciteReserveTokens is what the response needs for everything that is not
// the id list: any reasoning the model emits before answering, and the prose
// it continues with afterwards.
const reciteReserveTokens = 100

// reciteTokensPerID is a deliberately generous per-id estimate. A hyphenated
// UUID splits into roughly sixteen BPE pieces plus its separator, and being
// wrong in the cheap direction costs one fewer id in the window while being
// wrong in the expensive direction truncates the list mid-way and reports a
// PRESENCE_MISS that describes the budget rather than the fleet.
const reciteTokensPerID = 20

// reciteMaxRecent caps how many recent turns join the first one, so the ask
// stays bounded on a session hundreds of turns deep.
//
// 39 (a 40-id window with the pinned first turn) rather than the earlier 10:
// every recited id is a probe into a distinct region of the KV prefix, so a
// wider window covers four times the places corruption could surface per
// response. The budget arithmetic keeps it honest — at ~20 tokens per id a
// full window only happens on requests whose captured output budget clears
// ~900 tokens, and anything smaller degrades to what fits rather than
// forcing the ask. The cost is model effort: recalling 40 ids is harder than
// 10, so presence dips on weaker models raise the ask-quality counters, and
// those say prompt, not fleet, by design.
const reciteMaxRecent = 39

// reciteCapacity answers how many ids this request can be asked for, given the
// output budget the CAPTURE recorded for it.
//
// The budget is never raised to fit the ask. A replay benchmark that edits
// max_tokens is no longer replaying: the output-size profile is part of the
// workload under test, and a request whose captured budget was a handful of
// tokens — a tool-call turn, say — is exactly the kind of traffic whose shape
// matters. The previous design raised such budgets to a floor and warned about
// it, which traded a false PRESENCE_MISS for a real distortion of the thing
// being measured.
//
// Capacity 0 means this request cannot answer a recite at all. It is then not
// asked, not scored, and counted as excluded rather than as a miss — an
// unanswerable question must not be recorded as a wrong answer.
func reciteCapacity(maxTokens int) int {
	n := (maxTokens - reciteReserveTokens) / reciteTokensPerID
	if n < 0 {
		return 0
	}
	if n > reciteMaxRecent+1 {
		return reciteMaxRecent + 1
	}
	return n
}
