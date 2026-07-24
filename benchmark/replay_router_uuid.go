package benchmark

// UUID-based cache-coherency validation for the ROUTER-REPLAY path
// (--router-replay-file, --replay-inject-uuids). This is the router-path
// counterpart of the dataset-replay UUID validation (replay_uuid.go's
// path-agnostic primitives — injectUUIDMarker / validateReplayResponse /
// FindLeakedUUIDs — are shared with, not duplicated from, that file).
//
// Strategy (Option C — boundary injection, with tail fallback): every
// session in a replay-v3 capture opens with one or more blocks (system
// blocks, tools, or leading messages) whose content hash is shared across
// MANY OTHER sessions too — the router's own leading system prompt(s),
// repeated verbatim capture after capture. Everything AFTER that shared
// run is genuinely per-session (the user's actual turn). We inject exactly
// ONE deterministic UUID per session at that boundary:
//
//	[ RUN_GUID ][ shared system blocks ][ MARKER ][ forceOutput instr ] [ messages... ] [ recite ask ]
//	 \_________________ byte-identical across sessions _________________/ \_ per-session, grows each turn _/
//
// Putting the marker there means:
//   - the cross-session shared prefix stays byte-identical (cache-hit
//     reproduction against the original capture is preserved: two sessions
//     that shared a system prompt still collide on the server's prefix
//     cache exactly as they did in the original traffic)
//   - the marker itself lands in a region that IS cached WITHIN a session
//     (every subsequent request in the same session repeats it), so asking
//     the model to recall it later is a genuine KV-coherency signal, not an
//     artifact of it being freshly re-sent every turn.
//
// A session with no shared leading block at all (empirically none, across
// 5441 real sessions, lack one — but the file format doesn't guarantee it)
// falls back to tail injection: the marker is folded into the end of the
// request instead, forfeiting the "cached within a session" property but
// still producing a valid, scorable request.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// uuidInjection describes the per-request UUID injection to apply when
// building a router-replay wire body. A nil *uuidInjection means "no
// injection" — buildAnthropicMessagesBody / buildOpenAIChatCompletionsBody
// must behave identically to before this feature existed.
type uuidInjection struct {
	// Marker is the exact text to splice in (see injectUUIDMarker) — e.g.
	// "\n\n[ref-id: <uuid>]". Empty means no marker this call (still allows
	// Recite alone, though callers currently always set both together).
	Marker string
	// Recite asks the model to find and echo the marker verbatim from
	// earlier in its own context (see replayReciteFromContextInstruction).
	Recite bool
	// SharedPrefixLen is this request's leading run of cross-session-shared
	// prefix blocks (see sharedPrefixBlockCount). It tells the wire builder
	// whether the marker can be spliced in at the natural system/message
	// boundary (SharedPrefixLen covers every emitted system block) or must
	// fall back to tail injection (SharedPrefixLen == 0).
	SharedPrefixLen int
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

// replayReciteFromContextInstruction returns the tail instruction asking
// the model to find and echo, verbatim, the ref-id marker planted earlier
// in its own context — WITHOUT restating it from the instruction itself.
// Unlike the dataset-path replayReciteInstruction (replay_uuid.go), this
// text never embeds the UUID: presence in the response therefore reflects
// genuine recall from cached KV, not an echo of the ask.
func replayReciteFromContextInstruction() string {
	return "\n\n(Somewhere earlier in this conversation there is a line of the exact form `[ref-id: <uuid>]`. " +
		"Find it and, before your normal answer, output one line in the exact form `SEEN_REF: <uuid>` reproducing " +
		"that id verbatim from your context — do not invent one, and do not simply repeat this instruction. Then answer normally.)"
}

// buildSessionUUIDs returns n singleton UUID sets (one UUID per session),
// drawn in order from a single seeded generator — same determinism/
// disjointness rationale as the dataset path's buildReplayUUIDSets: same
// seed -> same per-session UUID assignment across runs and across every
// model in a multi-model run (see the precompute call site in
// RunAutoBenchmark, which populates cfg.replayUUIDSets once, before any
// per-model goroutine spawns, so every model sees the identical
// assignment).
func buildSessionUUIDs(n int, seed int64) [][]string {
	if n <= 0 {
		return nil
	}
	newUUID := newUUIDGenerator(seed)
	sets := make([][]string, n)
	for i := range sets {
		sets[i] = []string{newUUID()}
	}
	return sets
}

// ---- max_tokens recite floor ----

// replayReciteFloorTokens is the minimum max_tokens budget enforced on a
// request that carries the recite ask (see uuidInjection.Recite). A
// router-replay request's max_tokens is normally sized off the ORIGINAL
// capture's output_tokens (see pickMaxTokens) — which for a tool-call-only
// turn can be a handful of tokens, nowhere near enough to also emit the
// "SEEN_REF: <uuid>" line the recite ask asks for. Without a floor, a tiny
// budget truncates the recite line itself, which would misread as a
// PRESENCE_MISS (coherency failure) when it's actually just an output-size
// artifact.
const replayReciteFloorTokens = 64

var reciteFloorWarnOnce sync.Once

// applyReciteFloor raises maxTokens to replayReciteFloorTokens when recite
// is requested and the original budget falls short, warning once per
// process (mirrors the dataset path's reciteTruncWarned one-shot pattern,
// but this is a single global warning rather than per-conversation since
// the router path's floor is a fixed constant, not a per-conversation
// truncation computation).
func applyReciteFloor(maxTokens int, recite bool) int {
	if !recite || maxTokens >= replayReciteFloorTokens {
		return maxTokens
	}
	reciteFloorWarnOnce.Do(func() {
		fmt.Fprintf(os.Stderr,
			"[router-replay] WARNING: max_tokens raised to the UUID-recite floor (%d) for one or more requests — "+
				"a tiny captured output budget would otherwise truncate the recite line into a false PRESENCE_MISS, not real corruption\n",
			replayReciteFloorTokens)
	})
	return replayReciteFloorTokens
}
