package benchmark

// UUID-based cache-coherency validation for the ROUTER-REPLAY path
// (--router-replay-file, --replay-inject-uuids).
//
// Strategy: one deterministic UUID is injected inline into EVERY qualifying
// user turn — a role=="user" message with >=1 text block whose hash is
// referenced by exactly one session (see isQualifyingUserTurn, which reuses
// computeBlockSessionCounts' distinct-session-count map to exclude blocks
// shared across sessions) — spread through the conversation rather than
// clustered at a session boundary. Turn i's marker
// ("\n\n[turn-N id: <uuid>]") is appended to that turn's own synthesized
// text content (see buildMessageContent's stampByHash param in
// replay_router_wire.go), keyed by the turn's session-global
// first-appearance index (see computeSessionTurnHashes/
// buildSessionTurnUUIDs below). Content synthesis is already deterministic
// in the block hash (synthText, seeded "<hash>:block:<i>"), so every request
// that repeats a given turn in its growing history re-emits byte-identical
// content for it — the fidelity invariant that makes cache-hit reproduction
// against the original capture possible: two sessions sharing a leading
// block still collide on the server's prefix cache exactly as in the
// original traffic, because only count==1 (genuinely per-session) turns
// ever carry a stamp.
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
	"bufio"
	"encoding/json"
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
	// Recite asks the model to output, on the FIRST line of its response,
	// the ReciteUUIDs values (identified to the model by ReciteLabels'
	// inline tags — see replayReciteWindowInstruction), then continue
	// normally.
	Recite bool
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

// computeBlockSessionCounts streams a replay-v3 file once and returns, for
// every distinct block hash seen, the number of DISTINCT SESSIONS that
// reference it at least once (not the number of requests — a hash reused
// many times within one session still counts once for that session).
//
// A hash with count > 1 is shared across sessions — the router's own
// leading system prompt(s), repeated verbatim capture after capture, are
// the common case — and is never eligible for UUID stamping: stamping it
// would perturb content this session shares with others, poisoning their
// cross-session prefix-cache key too. computeSessionTurnHashes uses this
// map (via isQualifyingUserTurn) to restrict turn-stamping to hashes with
// count == 1: genuinely per-session, per-turn content.
//
// allowed and sessionLimit mirror openRouterReplayStream's filtering
// exactly (nil allowed = every session; sessionLimit <= 0 = no cap) so the
// set of sessions counted here matches the set the real run will dispatch.
// Only hashes are read here — synthText/synthesized content is never
// generated, so this pass is cheap even over large files.
func computeBlockSessionCounts(path string, allowed map[int]bool, sessionLimit int) (map[string]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)
	if _, err := br.ReadBytes('\n'); err != nil {
		return nil, fmt.Errorf("read header line: %w", err)
	}

	counts := map[string]int{}
	lineIdx := 0
	produced := 0
	for {
		if sessionLimit > 0 && produced >= sessionLimit {
			break
		}
		if allowed != nil && produced >= len(allowed) {
			break
		}
		line, rerr := br.ReadBytes('\n')
		if len(line) > 0 {
			line = trimNL(line)
			if len(line) > 0 {
				currentIdx := lineIdx
				lineIdx++
				if allowed != nil && !allowed[currentIdx] {
					if rerr != nil {
						break
					}
					continue
				}
				var sess RouterReplaySession
				if jerr := json.Unmarshal(line, &sess); jerr == nil {
					seen := map[string]bool{}
					for _, inst := range sess.Instances {
						for _, req := range inst.Requests {
							hashes, _ := BuildReplayRequestPrefix(req)
							for _, h := range hashes {
								if h != "" {
									seen[h] = true
								}
							}
						}
					}
					for h := range seen {
						counts[h]++
					}
					produced++
				}
			}
		}
		if rerr != nil {
			break
		}
	}
	return counts, nil
}

// isQualifyingUserTurn reports whether m is a "user-input turn" for the
// purposes of UUID stamping: role=="user", at least one "text" content
// block, and a hash referenced by exactly one session (counts[m.Hash]==1 —
// see computeBlockSessionCounts). tool_result-only messages, assistant/
// system messages, and any message whose hash is shared across sessions
// never qualify.
func isQualifyingUserTurn(m RouterReplayMessage, counts map[string]int) bool {
	if m.Role != "user" {
		return false
	}
	if m.Hash == "" || counts[m.Hash] != 1 {
		return false
	}
	for _, t := range m.BlockTypes {
		if t == "text" {
			return true
		}
	}
	return false
}

// computeSessionTurnHashes makes a second streaming pass (same filtering as
// computeBlockSessionCounts — nil allowed = every session; sessionLimit <= 0
// = no cap) and returns, in dispatch order (index i = the i-th session
// encountered, the SAME order buildSessionTurnUUIDs' returned slice is
// indexed by and cfg.replayUUIDSets ends up in), that session's ordered list
// of DISTINCT qualifying user-turn hashes (see isQualifyingUserTurn), in
// first-appearance order across the session's instances/requests/messages
// (walked in file order — Instances, then each instance's Requests, then
// each request's Messages). A request's Messages carries the FULL growing
// conversation history, so the same turn hash reappears in every later
// request of the same instance; only its FIRST appearance contributes an
// entry here — turnHashes[i][t] is session i's turn-t hash, and len(...) is
// that session's total turn count.
func computeSessionTurnHashes(path string, allowed map[int]bool, sessionLimit int, counts map[string]int) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)
	if _, err := br.ReadBytes('\n'); err != nil {
		return nil, fmt.Errorf("read header line: %w", err)
	}

	var turnHashes [][]string
	lineIdx := 0
	produced := 0
	for {
		if sessionLimit > 0 && produced >= sessionLimit {
			break
		}
		if allowed != nil && produced >= len(allowed) {
			break
		}
		line, rerr := br.ReadBytes('\n')
		if len(line) > 0 {
			line = trimNL(line)
			if len(line) > 0 {
				currentIdx := lineIdx
				lineIdx++
				if allowed != nil && !allowed[currentIdx] {
					if rerr != nil {
						break
					}
					continue
				}
				var sess RouterReplaySession
				if jerr := json.Unmarshal(line, &sess); jerr == nil {
					seen := map[string]bool{}
					var hashes []string
					for _, inst := range sess.Instances {
						for _, req := range inst.Requests {
							for _, m := range req.Messages {
								if !isQualifyingUserTurn(m, counts) {
									continue
								}
								if seen[m.Hash] {
									continue
								}
								seen[m.Hash] = true
								hashes = append(hashes, m.Hash)
							}
						}
					}
					turnHashes = append(turnHashes, hashes)
					produced++
				}
			}
		}
		if rerr != nil {
			break
		}
	}
	return turnHashes, nil
}

// buildSessionTurnUUIDs returns, for each session, one UUID per turn
// (sets[i][t] = session i's turn-t UUID), drawn session-major/turn-minor
// from a single seeded generator (see newUUIDGenerator) — same determinism/
// disjointness rationale as the dataset path's buildReplayUUIDSets: same
// seed -> same per-session-per-turn UUID assignment across runs and across
// every model in a multi-model run (see the precompute call site in
// RunAutoBenchmark, which populates cfg.replayUUIDSets once, before any
// per-model goroutine spawns, so every model sees the identical
// assignment).
//
// owner is the reverse mapping (uuid -> the owning session's index i) used
// by findLeakedUUIDsByOwner (replay_uuid.go) to flag cross-session
// contamination in O(response) time — a substring scan of the response
// plus map lookups, rather than iterating every session's UUID set.
func buildSessionTurnUUIDs(turnCounts []int, seed int64) (sets [][]string, owner map[string]int) {
	owner = map[string]int{}
	if len(turnCounts) == 0 {
		return nil, owner
	}
	newUUID := newUUIDGenerator(seed)
	sets = make([][]string, len(turnCounts))
	for i, n := range turnCounts {
		if n <= 0 {
			continue
		}
		uuids := make([]string, n)
		for j := range uuids {
			u := newUUID()
			uuids[j] = u
			owner[u] = i
		}
		sets[i] = uuids
	}
	return sets, owner
}

// replayReciteWindowInstruction returns the tail instruction asking the
// model to output, as the FIRST line of its response, the id VALUES for the
// given ordered window of inline "[turn-N id: ...]" tags — in the same
// order, comma-separated — then continue normally. Unlike the retired
// per-session replayReciteFirstLineInstruction (which asked for "every UUID
// shown above" verbatim), this instruction names the exact turns to recite
// by LABEL, not position or value, so the model must locate each tagged
// turn in its own (possibly long) context rather than simply copying
// whatever happens to be nearby. Mirrors the cache-coherency eval's
// userMessage ("List every UUID shown in the request...") but (a)
// front-loads the ask to line 1 specifically, so a forced-output/
// ignore_eos budget can still fill the remainder with the model's normal
// continuation, and (b) references labels instead of embedding the UUIDs
// themselves, so first-line conformity in the response reflects genuine
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
// on a request that carries the recite ask (see uuidInjection.Recite), sized
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
