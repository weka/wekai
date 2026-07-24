package benchmark

// UUID-based cache-coherency validation for the ROUTER-REPLAY path
// (--router-replay-file, --replay-inject-uuids). This mirrors the mechanics
// of the cache-coherency eval's --shared-prefix-per-series mode
// (cache_coherency.go's buildCoherencySharedSeriesPrompt/userMessage/
// matchesExpectedUUIDList) as closely as the router-replay wire shape
// allows: N bare, space-separated UUIDs stamped once per session, a recite-
// the-list instruction, and exact first-line conformity scoring — rather
// than a single wrapped "[ref-id: ...]" marker recited anywhere in the
// response. (Path-agnostic primitives — injectUUIDMarker/
// validateReplayResponse/FindLeakedUUIDs — are still shared with, not
// duplicated from, replay_uuid.go; injectUUIDMarker itself remains the
// dataset-replay path's own wrapper and is untouched here.)
//
// Strategy (Option C — boundary injection, with tail fallback): every
// session in a replay-v3 capture opens with one or more blocks (system
// blocks, tools, or leading messages) whose content hash is shared across
// MANY OTHER sessions too — the router's own leading system prompt(s),
// repeated verbatim capture after capture. Everything AFTER that shared
// run is genuinely per-session (the user's actual turn). We inject exactly
// N deterministic, bare, space-separated UUIDs per session at that
// boundary — mirroring buildCoherencySharedSeriesPrompt's tail
// ("UUID0 UUID1 … UUIDlast") rather than the dataset path's wrapped
// "[ref-id: <uuid>]" marker:
//
//	[ RUN_GUID ][ shared system blocks ][ UUID0 UUID1 … UUIDlast ][ forceOutput instr ] [ messages... ] [ recite-first-line ask ]
//	 \_______________________ byte-identical across sessions ________________________/ \_ per-session, grows each turn _/
//
// Putting the UUID block there means:
//   - the cross-session shared prefix stays byte-identical (cache-hit
//     reproduction against the original capture is preserved: two sessions
//     that shared a system prompt still collide on the server's prefix
//     cache exactly as they did in the original traffic)
//   - the UUID block itself lands in a region that IS cached WITHIN a
//     session (every subsequent request in the same session repeats it,
//     byte-identical), so asking the model to recall it later is a genuine
//     KV-coherency signal, not an artifact of it being freshly re-sent
//     every turn.
//
// A session with no shared leading block at all (empirically none, across
// 5441 real sessions, lack one — but the file format doesn't guarantee it)
// falls back to tail injection: the UUID block is folded into the end of
// the request instead, forfeiting the "cached within a session" property
// but still producing a valid, scorable request.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
)

// uuidInjection describes the per-request UUID injection to apply when
// building a router-replay wire body. A nil *uuidInjection means "no
// injection" — buildAnthropicMessagesBody / buildOpenAIChatCompletionsBody
// must behave identically to before this feature existed.
type uuidInjection struct {
	// UUIDs is the session's full ordered N-UUID list (see buildSessionUUIDs),
	// spliced bare and space-separated (see bareUUIDBlock) — mirrors the
	// cache-coherency eval's buildCoherencySharedSeriesPrompt tail. Nil/empty
	// means no UUID block this call (still allows Recite alone, though
	// callers currently always set both together).
	UUIDs []string
	// Recite asks the model to output, as the FIRST line of its response,
	// the exact ordered UUID list (see replayReciteFirstLineInstruction),
	// then continue normally.
	Recite bool
	// SharedPrefixLen is this request's leading run of cross-session-shared
	// prefix blocks (see sharedPrefixBlockCount). It tells the wire builder
	// whether the UUID block can be spliced in at the natural system/
	// message boundary (SharedPrefixLen covers every emitted system block)
	// or must fall back to tail injection (SharedPrefixLen == 0).
	SharedPrefixLen int
}

// bareUUIDBlock returns uuids joined bare and space-separated — mirrors the
// cache-coherency eval's buildCoherencySharedSeriesPrompt tail
// ("UUID0 UUID1 … UUIDlast", no wrapper text) — for splicing as one system
// message/block at the session boundary (or the tail, on fallback). Bare
// UUIDs (vs. the dataset path's "[ref-id: <uuid>]" wrapper) both match the
// coherency test's mechanics and avoid the model treating a marker-shaped
// wrapper as instruction text to echo verbatim.
func bareUUIDBlock(uuids []string) string {
	return strings.Join(uuids, " ")
}

// firstLineConformity implements this feature's output-conformity check:
// whether the FIRST LINE of resp (up to the first '\n', trimmed) is EXACTLY
// the comma-separated expected UUID list, in order (see
// matchesExpectedUUIDList) — the router-replay analogue of the coherency
// eval's ExactMatch, adapted for a recite-FIRST instruction that lets the
// model keep generating after line 1 (so forced-output/ignore_eos still
// fills the remainder of the output budget with a normal continuation).
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
// many times within one session still counts once for that session). A
// hash with count > 1 is shared across sessions, which is exactly the
// "safe to leave byte-identical" prefix content sharedPrefixBlockCount looks
// for.
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

// countSessionsWithUsableBoundary makes a second lightweight streaming pass
// (same filtering as computeBlockSessionCounts) purely to report, for the
// startup diagnostic, how many sessions have at least one request whose
// leading run of shared blocks is non-empty (a "usable boundary" — the
// marker can be spliced in at the natural system/message boundary) versus
// how many would fall back to tail injection for every one of their
// requests. Returns (usable, total, error); usable <= total.
func countSessionsWithUsableBoundary(path string, allowed map[int]bool, sessionLimit int, counts map[string]int) (usable int, total int, err error) {
	f, ferr := os.Open(path)
	if ferr != nil {
		return 0, 0, ferr
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)
	if _, rerr := br.ReadBytes('\n'); rerr != nil {
		return 0, 0, fmt.Errorf("read header line: %w", rerr)
	}

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
					total++
					hasBoundary := false
					for _, inst := range sess.Instances {
						for _, req := range inst.Requests {
							if sharedPrefixBlockCount(req, counts) > 0 {
								hasBoundary = true
								break
							}
						}
						if hasBoundary {
							break
						}
					}
					if hasBoundary {
						usable++
					}
					produced++
				}
			}
		}
		if rerr != nil {
			break
		}
	}
	return usable, total, nil
}

// computePerSessionCachedChars makes a second streaming pass (same filtering
// as computeBlockSessionCounts) and returns, in dispatch order (index i =
// the i-th session encountered — the SAME order buildSessionUUIDs' returned
// slice is indexed by, and the same order cfg.replayUUIDSets ends up in),
// each session's per-session cached-region byte size: the ROOT request's
// (first instance, first request) prefix bytes (see requestPrefixBytes) AT
// OR AFTER that request's cross-session-shared boundary
// (sharedPrefixBlockCount, using counts from computeBlockSessionCounts).
//
// This is a proxy for "how much content is genuinely per-session and reused,
// byte-identical, across the session's own later requests" — the region the
// injected UUID block actually sits within once spliced at the boundary
// (see the package doc comment above). computeStampsPerSeries then turns
// this byte count into a stamp count exactly as the cache-coherency eval
// turns --garbage-chars into --stamps-per-series. A session with no
// requests at all (degenerate/empty) contributes 0, which
// computeStampsPerSeries floors to its minimum of 2.
func computePerSessionCachedChars(path string, allowed map[int]bool, sessionLimit int, counts map[string]int) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)
	if _, err := br.ReadBytes('\n'); err != nil {
		return nil, fmt.Errorf("read header line: %w", err)
	}

	var perSession []int
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
					chars := 0
					if len(sess.Instances) > 0 && len(sess.Instances[0].Requests) > 0 {
						root := sess.Instances[0].Requests[0]
						boundary := sharedPrefixBlockCount(root, counts)
						prefixBytes := requestPrefixBytes(root)
						for i := boundary; i < len(prefixBytes); i++ {
							chars += prefixBytes[i]
						}
					}
					perSession = append(perSession, chars)
					produced++
				}
			}
		}
		if rerr != nil {
			break
		}
	}
	return perSession, nil
}

// sharedPrefixBlockCount returns the length of req's LEADING run of blocks
// (per BuildReplayRequestPrefix's cache-order hash sequence: system blocks,
// then tools, then messages) whose hash is shared across more than one
// session, per counts (see computeBlockSessionCounts). The run stops at the
// first hash that is either unshared (counts[hash] <= 1) or empty.
//
// This is the offline analogue of "how much of this request's prefix is
// safe to leave byte-identical" — the wire builder uses it to decide
// whether the UUID marker can be spliced in at the natural boundary
// (SharedPrefixLen covers every system block) or must fall back to tail
// injection.
func sharedPrefixBlockCount(req RouterReplayRequest, counts map[string]int) int {
	hashes, _ := BuildReplayRequestPrefix(req)
	n := 0
	for _, h := range hashes {
		if h == "" || counts[h] <= 1 {
			break
		}
		n++
	}
	return n
}

// replayReciteFirstLineInstruction returns the tail instruction asking the
// model to output, as the FIRST line of its response, every UUID injected
// at the session boundary — in order, comma-separated — then continue
// normally. Mirrors the cache-coherency eval's userMessage ("List every
// UUID shown in the request, in order, separated by commas. Output only the
// UUIDs and commas, nothing else.") but front-loads the ask to line 1
// specifically (rather than the entire response), so a forced-output /
// ignore_eos budget can still fill the remainder with the model's normal
// continuation. Like the dataset-path instruction it replaces, this text
// never embeds the UUIDs themselves: first-line conformity in the response
// therefore reflects genuine recall from cached KV context, not an echo of
// the ask.
func replayReciteFirstLineInstruction() string {
	return "\n\nBefore anything else, output on the FIRST line every UUID shown above in the request, in order, " +
		"separated by commas, and nothing else on that first line. Then continue with your normal response."
}

// buildSessionUUIDs returns len(perSessionChars) UUID sets, one per session,
// drawn in order (session-major, stamp-minor) from a single seeded
// generator — same determinism/disjointness rationale as the dataset path's
// buildReplayUUIDSets: same seed -> same per-session UUID assignment across
// runs and across every model in a multi-model run (see the precompute call
// site in RunAutoBenchmark, which populates cfg.replayUUIDSets once, before
// any per-model goroutine spawns, so every model sees the identical
// assignment).
//
// Session i's set size is N = computeStampsPerSeries(perSessionChars[i]) —
// the SAME sizing rule the cache-coherency eval uses to turn a garbage-char
// budget into a stamp count (min 2) — applied here to perSessionChars[i],
// session i's per-session cached-region byte size (see
// computePerSessionCachedChars): the blocks AFTER the cross-session-shared
// boundary that get reused, byte-identical, across every one of the
// session's own requests. A larger reused region gets more UUID stamps
// spread across it, mirroring the coherency test's
// garbageChars -> numStamps relationship.
func buildSessionUUIDs(perSessionChars []int, seed int64) [][]string {
	if len(perSessionChars) == 0 {
		return nil
	}
	newUUID := newUUIDGenerator(seed)
	sets := make([][]string, len(perSessionChars))
	for i, chars := range perSessionChars {
		n := computeStampsPerSeries(chars)
		uuids := make([]string, n)
		for j := range uuids {
			uuids[j] = newUUID()
		}
		sets[i] = uuids
	}
	return sets
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
// x replayReciteFloorMultiplier). A router-replay request's max_tokens is
// normally sized off the ORIGINAL capture's output_tokens (see
// pickMaxTokens) — which for a tool-call-only turn can be a handful of
// tokens, nowhere near enough to also emit the first-line UUID list. Without
// this floor, a tiny budget truncates that first line, which would misread
// as PRESENCE_MISS/NOT_EXACT (coherency failure) when it's actually just an
// output-size artifact.
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
