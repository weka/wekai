package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/weka/wekai/benchmark"
)

// RouterConvertAIPerfWekaTraceCommand converts a replay-v3 capture (wekai's
// own tree-aware replay format — see benchmark/replay_router.go) into one
// JSON file per session in AIPerf's native `weka_trace` custom-dataset-type
// schema (see agentx-harness's src/aiperf/dataset/loader/weka_trace_models.py).
//
// This is a pure offline data transformation: no HTTP calls, no capture, no
// replay. It exists so the same captured agentic-coding traffic can be
// benchmarked through AIPerf, not just wekai's own replay tool — the two
// tools disagree on almost everything (session tree shape, cache-block
// bookkeeping, timing fields) so the conversion has to rebuild those from
// what replay-v3 actually preserves rather than pass values through.
type RouterConvertAIPerfWekaTraceCommand struct {
	Out                    string `short:"o" long:"out" required:"true" description:"Output directory; one <session_id>.json file is written per session"`
	IncludeEphemeralProbes bool   `long:"include-ephemeral-probes" description:"Keep 'ephemeral (no system)' instances instead of dropping them. These are single-token liveness probes (max_tokens=1), not real agent traffic, so they are dropped by default"`
	BlockSize              int    `long:"block-size" default:"64" description:"KV-cache block size in tokens. Embedded in every output trace's block_size field and used to size newly-minted hash_ids"`

	Args struct {
		Src string `positional-arg-name:"src" description:"replay-v3 JSONL file (header line + one session per line)"`
	} `positional-args:"yes" required:"yes"`
}

func (c *RouterConvertAIPerfWekaTraceCommand) Execute(args []string) error {
	if c.BlockSize <= 0 {
		return fmt.Errorf("--block-size must be positive, got %d", c.BlockSize)
	}
	if err := os.MkdirAll(c.Out, 0o755); err != nil {
		return fmt.Errorf("create output dir %s: %w", c.Out, err)
	}

	f, err := os.Open(c.Args.Src)
	if err != nil {
		return err
	}
	defer f.Close()

	// bufio.Scanner's 64KB default line cap is far too small here: a single
	// session line embeds every request's full structured spec (system
	// blocks, messages, growing conversation history) and the real corpus
	// this runs against has lines up to ~54MB. ReadBytes('\n') has no such
	// cap. See openRouterReplayStream in benchmark/replay_router.go for the
	// same reasoning against the same file format.
	br := bufio.NewReaderSize(f, 1<<20)

	headerLine, err := br.ReadBytes('\n')
	if err != nil && len(headerLine) == 0 {
		return fmt.Errorf("read header line: %w", err)
	}
	headerLine = trimTrailingNewline(headerLine)
	var hdr benchmark.RouterReplayHeader
	if err := json.Unmarshal(headerLine, &hdr); err != nil {
		return fmt.Errorf("parse header: %w", err)
	}
	if hdr.Schema != "replay-v3" {
		return fmt.Errorf("unsupported replay schema %q (expected replay-v3)", hdr.Schema)
	}

	opts := convertOptions{
		IncludeEphemeralProbes: c.IncludeEphemeralProbes,
		BlockSize:              c.BlockSize,
	}

	var totalDropped, totalFlattened, totalDangling, totalRootFallback, written int
	for {
		line, readErr := br.ReadBytes('\n')
		line = trimTrailingNewline(line)
		if len(line) > 0 {
			var sess benchmark.RouterReplaySession
			if err := json.Unmarshal(line, &sess); err != nil {
				fmt.Fprintf(os.Stderr, "convert-aiperf-weka-trace: skipping unparsable session line: %v\n", err)
			} else if trace, stats, err := convertSessionToWekaTrace(sess, opts); err != nil {
				fmt.Fprintf(os.Stderr, "convert-aiperf-weka-trace: session %s: %v\n", sess.SessionID, err)
			} else {
				if stats.EphemeralDropped > 0 || stats.RootFallbackUsed || stats.MultiLevelFlattened > 0 || stats.DanglingAnchorFallback > 0 {
					fmt.Fprintf(os.Stderr,
						"convert-aiperf-weka-trace: session %s: dropped %d ephemeral probe(s), root-fallback=%v, %d multi-level-flattened, %d dangling-anchor-fallback\n",
						sess.SessionID, stats.EphemeralDropped, stats.RootFallbackUsed, stats.MultiLevelFlattened, stats.DanglingAnchorFallback)
				}
				totalDropped += stats.EphemeralDropped
				totalFlattened += stats.MultiLevelFlattened
				totalDangling += stats.DanglingAnchorFallback
				if stats.RootFallbackUsed {
					totalRootFallback++
				}

				b, err := json.MarshalIndent(trace, "", "  ")
				if err != nil {
					return fmt.Errorf("marshal session %s: %w", sess.SessionID, err)
				}
				outPath := filepath.Join(c.Out, trace.ID+".json")
				if err := os.WriteFile(outPath, b, 0o644); err != nil {
					return fmt.Errorf("write %s: %w", outPath, err)
				}
				written++
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}

	fmt.Fprintf(os.Stderr,
		"convert-aiperf-weka-trace: wrote %d session(s) to %s (dropped %d ephemeral probe(s) total, %d root-fallback session(s), %d multi-level-flattened, %d dangling-anchor-fallback)\n",
		written, c.Out, totalDropped, totalRootFallback, totalFlattened, totalDangling)
	return nil
}

func trimTrailingNewline(line []byte) []byte {
	if len(line) > 0 && line[len(line)-1] == '\n' {
		line = line[:len(line)-1]
	}
	return line
}

// ---- weka_trace output schema ----
//
// Mirrors agentx-harness's src/aiperf/dataset/loader/weka_trace_models.py
// pydantic models field-for-field, including the `in`/`out` aliases and the
// literal "n"/"s"/"subagent" discriminator strings the loader switches on.
// Kept private to this file: nothing outside the converter needs them, and
// the source of truth for the schema lives in the other repo, not here.

// wekaRequest serves both WekaNormalRequest ("n") and WekaStreamingRequest
// ("s") — the two are field-for-field identical except for the discriminator
// and streaming's optional ttft, and this converter never has a ttft value
// to offer (see the field comment below), so one Go struct covers both.
type wekaRequest struct {
	T            float64  `json:"t"`
	Type         string   `json:"type"` // "n" or "s"
	Model        string   `json:"model"`
	InputLength  int      `json:"in"`
	OutputLength int      `json:"out"`
	HashIDs      []int    `json:"hash_ids"`
	InputTypes   []string `json:"input_types"`
	OutputTypes  []string `json:"output_types"`
	Stop         string   `json:"stop"`
	// APITime and ThinkTime are pointers because "no measurement" (nil) and
	// "measured zero" (0.0) are different claims — see convertRequest.
	APITime   *float64 `json:"api_time,omitempty"`
	ThinkTime *float64 `json:"think_time,omitempty"`
	// TTFT has no source signal in replay-v3 (no separate streaming-response
	// capture) and is always omitted; present only so a future signal has
	// somewhere to go without a schema change.
	TTFT *float64 `json:"ttft,omitempty"`
}

// wekaSubagentEntry mirrors WekaSubagentEntry. DurationMs/TotalTokens/
// ToolUseCount are plain ints rather than pointers: the pydantic model
// allows null for the async_launched status this converter never produces
// (every entry here comes from a completed, fully-captured instance), so a
// value is always available.
type wekaSubagentEntry struct {
	T            float64       `json:"t"`
	Type         string        `json:"type"` // always "subagent"
	AgentID      string        `json:"agent_id"`
	SubagentType string        `json:"subagent_type"`
	DurationMs   int           `json:"duration_ms"`
	TotalTokens  int           `json:"total_tokens"`
	ToolUseCount int           `json:"tool_use_count"`
	Status       string        `json:"status"` // always "completed"
	Requests     []interface{} `json:"requests"`
	Models       []string      `json:"models"`
	ToolTokens   int           `json:"tool_tokens"`
	SystemTokens int           `json:"system_tokens"`
}

// wekaTrace mirrors WekaTrace, the one-file-per-session output.
type wekaTrace struct {
	ID           string        `json:"id"`
	Models       []string      `json:"models"`
	BlockSize    int           `json:"block_size"`
	HashIDScope  string        `json:"hash_id_scope"` // always "local"
	ToolTokens   int           `json:"tool_tokens"`
	SystemTokens int           `json:"system_tokens"`
	Requests     []interface{} `json:"requests"`
}

// ---- conversion ----

type convertOptions struct {
	IncludeEphemeralProbes bool
	BlockSize              int
}

// convertStats reports the fallback paths a session's conversion took, so
// the caller can log them instead of silently reshaping the tree. None of
// these are errors — the spec explicitly expects the source data to
// occasionally need them — but a corpus where they fire constantly instead
// of rarely would mean an assumption in this converter is wrong.
type convertStats struct {
	EphemeralDropped       int
	RootFallbackUsed       bool // true iff no "main"-role instance survived filtering
	MultiLevelFlattened    int  // instances anchored via >1 hop of ParentSpawnRequestID
	DanglingAnchorFallback int  // instances with no resolvable parent, anchored by nearest-preceding-Ts
}

// owner locates a request within the retained (post-filter) instance list.
type owner struct{ instIdx, reqIdx int }

// convertSessionToWekaTrace is the pure, file-I/O-free transform: one
// replay-v3 session in, one weka_trace document out. Kept separate from
// Execute so it is unit-testable without ever touching disk.
func convertSessionToWekaTrace(sess benchmark.RouterReplaySession, opts convertOptions) (*wekaTrace, convertStats, error) {
	var stats convertStats

	var retained []benchmark.RouterReplayInstance
	for _, inst := range sess.Instances {
		if inst.Role == "ephemeral (no system)" && !opts.IncludeEphemeralProbes {
			stats.EphemeralDropped++
			continue
		}
		retained = append(retained, inst)
	}
	if len(retained) == 0 {
		return nil, stats, fmt.Errorf("no instances remain after filtering")
	}

	sessionStart, err := time.Parse(time.RFC3339Nano, sess.StartTs)
	if err != nil {
		return nil, stats, fmt.Errorf("parse start_ts %q: %w", sess.StartTs, err)
	}

	// Sort every retained instance's own requests by Ts once, up front.
	// Everything downstream (root ordering, per-instance duration/think-time,
	// the global hash-id timeline) reads from these rather than re-sorting.
	instOrders := make([][]int, len(retained))
	instTs := make([][]time.Time, len(retained))
	for i, inst := range retained {
		order, tsList, err := sortedRequestIndices(inst.Requests)
		if err != nil {
			return nil, stats, fmt.Errorf("instance %s: %w", inst.InstanceID, err)
		}
		instOrders[i] = order
		instTs[i] = tsList
	}

	// Root selection keys off Role=="main" alone. ParentSpawnRequestID is
	// NOT a reliable signal for this: the producer's implicit-parent pass
	// can anchor "main" to an earlier ephemeral probe's request, so an
	// empty/zero parent field is neither necessary nor sufficient here.
	rootIdx := -1
	for i, inst := range retained {
		if inst.Role == "main" {
			rootIdx = i
			break
		}
	}
	if rootIdx == -1 {
		stats.RootFallbackUsed = true
		var earliest time.Time
		for i, order := range instOrders {
			if len(order) == 0 {
				continue
			}
			ts := instTs[i][order[0]]
			if rootIdx == -1 || ts.Before(earliest) {
				rootIdx = i
				earliest = ts
			}
		}
	}
	if rootIdx == -1 {
		return nil, stats, fmt.Errorf("no instance with any requests to use as root")
	}
	root := retained[rootIdx]
	rootOrder := instOrders[rootIdx]
	rootTs := instTs[rootIdx]

	// reqOwner and requestIDToRootPos are built only from retained
	// instances: an instance whose parent points at a dropped (ephemeral)
	// instance's request is deliberately indistinguishable from one with no
	// parent at all — both fall back to nearest-preceding-root-request.
	reqOwner := map[uint64]owner{}
	for i, inst := range retained {
		for j, r := range inst.Requests {
			if r.RequestID != 0 {
				reqOwner[r.RequestID] = owner{instIdx: i, reqIdx: j}
			}
		}
	}
	requestIDToRootPos := map[uint64]int{}
	for pos, idx := range rootOrder {
		requestIDToRootPos[root.Requests[idx].RequestID] = pos
	}

	// Global hash-id namespace: one map for the whole session, walked in
	// session-wide chronological order (not per instance) per the spec's
	// hash_id_scope: "local" contract. Also gives us the trace-level
	// `models` list in genuine first-appearance order for free.
	type globalRef struct {
		instIdx, reqIdx int
		ts              time.Time
	}
	var refs []globalRef
	for i, order := range instOrders {
		for _, idx := range order {
			refs = append(refs, globalRef{instIdx: i, reqIdx: idx, ts: instTs[i][idx]})
		}
	}
	sort.SliceStable(refs, func(a, b int) bool { return refs[a].ts.Before(refs[b].ts) })

	// Reorder each request's slots so ones that behave as a growing, reused
	// prefix land before ones that don't, regardless of real wire order —
	// see idsForBlock and slotVerdicts for why AIPerf's downstream chain
	// detection needs this. Two earlier designs were tried and rejected
	// before this one; both are worth knowing about, since the failure
	// mode of each is exactly what the other gets right:
	//
	//  1. Classify a slot as stable the moment it is EVER reused anywhere
	//     across the instance's whole history, and keep it there for every
	//     later request too. Wrong: confirmed on the real corpus, a
	//     1,122-request single conversation (no subagents at all) had a
	//     slot that coincidentally repeated once, early on, then changed on
	//     nearly every one of the following ~1,121 turns. "Ever" latched
	//     onto the one coincidence and kept the slot at position 0
	//     throughout, breaking the prefix check on almost every transition
	//     and splitting one real conversation into 1,106 disconnected ones.
	//  2. Classify PER OCCURRENCE instead of per slot: does this specific
	//     request's content match the immediately preceding request's, at
	//     this exact slot. Wrong differently: a slot with no predecessor
	//     yet (a request's own first appearance) has nothing to compare
	//     against, so its own internal ordering falls back to real wire
	//     order — which can then put that slot in a different relative
	//     position than a LATER request (whose own copy of that content DID
	//     have a predecessor to connect to, and got promoted) uses for the
	//     very same slot. Two requests that share content, ordered
	//     differently, still fail the prefix check against EACH OTHER even
	//     though the content genuinely matches.
	//
	// What's actually needed is a per-slot verdict that's consistent across
	// every request of an instance (fixing #2) but based on how the slot
	// behaves on AVERAGE rather than whether it EVER once matched (fixing
	// #1): a slot connects on a MAJORITY of its own transitions, or it
	// doesn't. See slotVerdicts.
	assigner := newHashIDAssigner(opts.BlockSize)

	type occurrence struct {
		key blockKey
		ids []int
	}
	occurrencesByOwner := map[owner][]occurrence{}
	transitions := map[blockKey]int{}
	connectingTransitions := map[blockKey]int{}
	globalModels := []string{}
	seenModel := map[string]bool{}
	for _, ref := range refs {
		req := retained[ref.instIdx].Requests[ref.reqIdx]
		var occs []occurrence
		record := func(key blockKey, ids []int, connects bool, hasPredecessor bool) {
			if hasPredecessor {
				transitions[key]++
				if connects {
					connectingTransitions[key]++
				}
			}
			occs = append(occs, occurrence{key: key, ids: ids})
		}
		for j, sb := range req.SystemBlocks {
			key := blockKey{ref.instIdx, "sys", j}
			ids, connects, hadPredecessor := assigner.idsForBlock(key, blockInfo{
				role: "", hash: sb.Hash, bytes: sb.Bytes, tokens: sb.Tokens,
				cacheControl: sb.CacheControl != "",
			})
			record(key, ids, connects, hadPredecessor)
		}
		for j, m := range req.Messages {
			key := blockKey{ref.instIdx, "msg", j}
			ids, connects, hadPredecessor := assigner.idsForBlock(key, blockInfo{
				role: m.Role, hash: m.Hash, bytes: m.Bytes, tokens: m.Tokens,
				cacheControl: m.CacheControl != "",
			})
			record(key, ids, connects, hadPredecessor)
		}
		occurrencesByOwner[owner{ref.instIdx, ref.reqIdx}] = occs

		if req.Model != "" && !seenModel[req.Model] {
			seenModel[req.Model] = true
			globalModels = append(globalModels, req.Model)
		}
	}

	// slotVerdicts: a slot is stable iff it connects to its predecessor on
	// at least half of the transitions it actually had. A slot with zero
	// transitions (appears exactly once in the whole instance) has no
	// evidence either way and defaults to unstable — its placement can't
	// need to align with any other request's copy of it, since there isn't
	// one.
	slotIsStable := map[blockKey]bool{}
	for key, total := range transitions {
		slotIsStable[key] = total > 0 && connectingTransitions[key]*2 >= total
	}

	hashIDs := map[owner][]int{}
	for ownerKey, occs := range occurrencesByOwner {
		var stable, unstable []int
		for _, o := range occs {
			if slotIsStable[o.key] {
				stable = append(stable, o.ids...)
			} else {
				unstable = append(unstable, o.ids...)
			}
		}
		hashIDs[ownerKey] = append(stable, unstable...)
	}

	// Anchor every non-root instance to a position in the root's request
	// list: -1 means "before the root's first request", otherwise it's the
	// 0-based index (in Ts order) of the root request this anchor follows.
	type anchor struct {
		instIdx    int
		posInRoot  int
		firstReqTs time.Time
	}
	var anchors []anchor
	for i := range retained {
		if i == rootIdx || len(instOrders[i]) == 0 {
			continue
		}
		firstTs := instTs[i][instOrders[i][0]]
		pos := resolveAnchor(i, retained, reqOwner, rootIdx, requestIDToRootPos, rootOrder, rootTs, firstTs, &stats)
		anchors = append(anchors, anchor{instIdx: i, posInRoot: pos, firstReqTs: firstTs})
	}

	groups := map[int][]anchor{}
	for _, a := range anchors {
		groups[a.posInRoot] = append(groups[a.posInRoot], a)
	}
	for pos := range groups {
		g := groups[pos]
		sort.SliceStable(g, func(a, b int) bool { return g[a].firstReqTs.Before(g[b].firstReqTs) })
		groups[pos] = g
	}

	rootConverted := buildRequests(root.Requests, rootOrder, rootTs, sessionStart, func(reqIdx int) []int {
		return hashIDs[owner{rootIdx, reqIdx}]
	})

	agentCounter := 1
	emitGroup := func(out []interface{}, pos int) []interface{} {
		for _, a := range groups[pos] {
			inst := retained[a.instIdx]
			order := instOrders[a.instIdx]
			tsList := instTs[a.instIdx]
			firstTs := tsList[order[0]]
			lastTs := tsList[order[len(order)-1]]

			totalTokens := 0
			toolUseCount := 0
			models := []string{}
			seen := map[string]bool{}
			for _, idx := range order {
				r := inst.Requests[idx]
				totalTokens += r.InputTokens + r.OutputTokens
				if r.Model != "" && !seen[r.Model] {
					seen[r.Model] = true
					models = append(models, r.Model)
				}
				for _, m := range r.Messages {
					// ToolUseIDs enumerates the tool_use blocks in the message
					// one-for-one, so prefer it; fall back to counting
					// "tool_use" occurrences in BlockTypes for messages
					// captured without ToolUseIDs populated.
					if len(m.ToolUseIDs) > 0 {
						toolUseCount += len(m.ToolUseIDs)
						continue
					}
					for _, bt := range m.BlockTypes {
						if bt == "tool_use" {
							toolUseCount++
						}
					}
				}
			}

			toolTokens, systemTokens := 0, 0
			if len(inst.Requests[order[0]].SystemBlocks) > 0 {
				toolTokens, systemTokens = splitSystemTokens(inst.Requests[order[0]].SystemBlocks)
			}

			entryIdx := a.instIdx
			converted := buildRequests(inst.Requests, order, tsList, sessionStart, func(reqIdx int) []int {
				return hashIDs[owner{entryIdx, reqIdx}]
			})

			out = append(out, wekaSubagentEntry{
				T:            firstTs.Sub(sessionStart).Seconds(),
				Type:         "subagent",
				AgentID:      fmt.Sprintf("agent_%03d", agentCounter),
				SubagentType: inst.Role,
				DurationMs:   int(lastTs.Sub(firstTs).Milliseconds()),
				TotalTokens:  totalTokens,
				ToolUseCount: toolUseCount,
				Status:       "completed",
				Requests:     converted,
				Models:       models,
				ToolTokens:   toolTokens,
				SystemTokens: systemTokens,
			})
			agentCounter++
		}
		return out
	}

	finalRequests := []interface{}{}
	finalRequests = emitGroup(finalRequests, -1)
	for i := range rootOrder {
		finalRequests = append(finalRequests, rootConverted[i])
		finalRequests = emitGroup(finalRequests, i)
	}

	rootToolTokens, rootSystemTokens := 0, 0
	if len(rootOrder) > 0 {
		rootToolTokens, rootSystemTokens = splitSystemTokens(root.Requests[rootOrder[0]].SystemBlocks)
	}

	trace := &wekaTrace{
		ID:           sess.SessionID,
		Models:       globalModels,
		BlockSize:    opts.BlockSize,
		HashIDScope:  "local",
		ToolTokens:   rootToolTokens,
		SystemTokens: rootSystemTokens,
		Requests:     finalRequests,
	}
	return trace, stats, nil
}

// resolveAnchor walks ParentSpawnRequestID from inst upward until it lands
// on a request the root instance owns, capped at 50 hops as a defensive
// guard against any residual cycle repairParentCycles/breakParentCycles
// (wekai's own producer) did not fully clean up. Every hop past the first
// means a level of nesting AIPerf's flat WekaSubagentEntry.requests cannot
// represent, so it gets flattened to the root anchor instead — a
// deliberate loss of fidelity, logged via stats so it's visible how often
// it happens on real data.
//
// On the real corpus this runs against, the dangling-fallback path below
// fires for roughly 45% of non-root instances — confirmed NOT a bug in this
// walk (a faithful independent re-implementation reproduces the same count
// exactly): "main" is often not the only causal root in a session's parent
// graph. "helper-or-isolated" is wekai's own label for an instance its
// spawn-detection could not confidently link to anything (see
// personaSignature/spawnRegistry in cli/command_router_tree.go), and its
// implicit-parent fallback just chains it to whatever request happened
// immediately before it in wall-clock time — which is frequently another
// disconnected helper, not "main". On this corpus "main" starts first in
// only ~70% of sessions; the rest have a helper-or-isolated instance
// starting earlier, forming its own chain that never reaches "main" at any
// hop count. Anchoring these by nearest-preceding-root-timestamp instead of
// dropping them is a deliberate choice to keep their token/request volume in
// the benchmark, at the cost of a fabricated (but plausible) attachment
// point rather than a reconstructed real spawn relationship.
func resolveAnchor(
	startIdx int,
	retained []benchmark.RouterReplayInstance,
	reqOwner map[uint64]owner,
	rootIdx int,
	requestIDToRootPos map[uint64]int,
	rootOrder []int,
	rootTs []time.Time,
	firstReqTs time.Time,
	stats *convertStats,
) int {
	cur := startIdx
	for hop := 0; hop < 50; hop++ {
		pid := retained[cur].ParentSpawnRequestID
		if pid == 0 {
			break
		}
		own, ok := reqOwner[pid]
		if !ok {
			break // parent request not owned by any retained instance: dangling
		}
		if own.instIdx == rootIdx {
			if hop > 0 {
				stats.MultiLevelFlattened++
			}
			return requestIDToRootPos[pid]
		}
		if own.instIdx == cur {
			break // self-referential parent guard; treat as dangling
		}
		cur = own.instIdx
	}
	stats.DanglingAnchorFallback++
	return nearestPrecedingRootPos(firstReqTs, rootOrder, rootTs)
}

// nearestPrecedingRootPos returns the 0-based (Ts-order) position of the
// last root request at or before t, or -1 if t precedes every root request.
func nearestPrecedingRootPos(t time.Time, rootOrder []int, rootTs []time.Time) int {
	pos := -1
	for i, idx := range rootOrder {
		if !rootTs[idx].After(t) {
			pos = i
		} else {
			break // rootOrder is Ts-ascending, so nothing later can qualify
		}
	}
	return pos
}

// splitSystemTokens divides a request's system-block token budget between
// tool_tokens and system_tokens. In the real corpus every RouterReplaySystemBlock.Type
// is "text" (see parseSystemBlocks in command_router_capture.go) — there is
// no field that distinguishes a tools-schema block from a plain system
// block — so per the spec's documented fallback, the whole budget goes to
// system_tokens.
func splitSystemTokens(blocks []benchmark.RouterReplaySystemBlock) (toolTokens, systemTokens int) {
	for _, b := range blocks {
		systemTokens += b.Tokens
	}
	return 0, systemTokens
}

// buildRequests converts one instance's own requests (already Ts-sorted via
// order/tsList) into the weka_trace request shape, threading think_time and
// the input_types tool_result/text signal across the sequence.
func buildRequests(
	reqs []benchmark.RouterReplayRequest,
	order []int,
	tsList []time.Time,
	sessionStart time.Time,
	hashIDsFor func(reqIdx int) []int,
) []interface{} {
	out := make([]interface{}, 0, len(order))
	var prevTs time.Time
	var prevAPISeconds float64
	havePrev := false
	prevMsgCount := 0

	for _, idx := range order {
		req := reqs[idx]
		ts := tsList[idx]

		typ := "n"
		if req.Stream {
			typ = "s"
		}

		// UpstreamLatencyMs/TotalMs are confirmed always 0 in the real
		// corpus. A nil api_time means "not measured"; emitting 0.0 would
		// falsely claim a measured zero-duration call, so the two must stay
		// distinguishable rather than collapsing "always nil" into a
		// hardcoded omission.
		var apiTime *float64
		apiSeconds := 0.0
		if req.TotalMs != 0 {
			apiSeconds = float64(req.TotalMs) / 1000.0
			v := apiSeconds
			apiTime = &v
		}

		var thinkTime *float64
		if havePrev {
			tt := ts.Sub(prevTs).Seconds() - prevAPISeconds
			if tt < 0 {
				tt = 0
			}
			thinkTime = &tt
		}

		// Messages carries the full growing conversation on every request,
		// so "this turn's new input" is the slice past what the previous
		// request in this same instance already carried. A shrinking
		// Messages length (context reset) is treated as "everything is new"
		// rather than indexing negative.
		newStart := prevMsgCount
		if newStart > len(req.Messages) {
			newStart = 0
		}
		inputTypes := []string{"text"}
		for _, m := range req.Messages[newStart:] {
			for _, bt := range m.BlockTypes {
				if bt == "tool_result" {
					inputTypes = []string{"tool_result"}
				}
			}
		}
		prevMsgCount = len(req.Messages)

		ids := hashIDsFor(idx)
		if ids == nil {
			ids = []int{}
		}

		out = append(out, wekaRequest{
			T:            ts.Sub(sessionStart).Seconds(),
			Type:         typ,
			Model:        req.Model,
			InputLength:  req.InputTokens,
			OutputLength: req.OutputTokens,
			HashIDs:      ids,
			InputTypes:   inputTypes,
			OutputTypes:  []string{"text"}, // no signal for the assistant's own output block types; see spec §5
			Stop:         req.StopReason,
			APITime:      apiTime,
			ThinkTime:    thinkTime,
		})

		prevTs = ts
		prevAPISeconds = apiSeconds
		havePrev = true
	}
	return out
}

// sortedRequestIndices returns reqs' indices in Ts-ascending order plus the
// parsed timestamps, so callers can look up either by original index.
func sortedRequestIndices(reqs []benchmark.RouterReplayRequest) ([]int, []time.Time, error) {
	order := make([]int, len(reqs))
	tsList := make([]time.Time, len(reqs))
	for i, r := range reqs {
		ts, err := time.Parse(time.RFC3339Nano, r.Ts)
		if err != nil {
			return nil, nil, fmt.Errorf("parse request %d ts %q: %w", r.RequestID, r.Ts, err)
		}
		order[i] = i
		tsList[i] = ts
	}
	sort.SliceStable(order, func(a, b int) bool { return tsList[order[a]].Before(tsList[order[b]]) })
	return order, tsList, nil
}

// hashIDAssigner reconstructs stable per-session KV-cache block ids from
// content hashes. One instance is used per session (hash_id_scope: "local")
// and reset for the next.
type hashIDAssigner struct {
	blockSize int
	assigned  map[string][]int
	next      int
	// bySlot remembers the most recently seen block at each (instance, kind,
	// position) slot, keyed by blockKey — see idsForBlock's cache-control-churn
	// reuse path.
	bySlot map[blockKey]blockInfo
}

// blockKey identifies one system-block or message SLOT within an instance's
// growing request history: the same instIdx+kind+position recurs across that
// instance's successive requests as long as the slot isn't compacted away,
// since Messages/SystemBlocks only ever grow (see ReplayMessage's doc comment
// on replaying the full history verbatim).
type blockKey struct {
	instIdx  int
	kind     string // "sys" or "msg"
	position int
}

// blockInfo is the metadata idsForBlock needs to decide whether two blocks
// at the same slot, seen on different requests, are the same underlying
// content despite carrying different hashes.
type blockInfo struct {
	role         string // messages only; empty for system_blocks
	hash         string
	bytes        int
	tokens       int
	cacheControl bool
	ids          []int
}

func newHashIDAssigner(blockSize int) *hashIDAssigner {
	return &hashIDAssigner{
		blockSize: blockSize,
		assigned:  map[string][]int{},
		bySlot:    map[blockKey]blockInfo{},
	}
}

// cacheControlChurnByteTolerance bounds how much a block's byte count may
// differ across a suspected cache-control-marker move and still count as
// "the same content." Anthropic's client resends prior turns verbatim, so a
// real content edit at the same slot should either match closely (the
// marker-move case, observed within a fraction of a percent) or differ
// substantially (a genuine compaction/edit, which must NOT be merged as a
// cache hit it never was). 10% comfortably separates the two on real data
// without needing an exact-byte match that a redaction step could perturb.
const cacheControlChurnByteTolerance = 0.10

// idsForBlock returns the hash-id run for one system-block or message,
// whether this occurrence CONNECTS TO ITS IMMEDIATE PREDECESSOR — the same
// (instance, kind, position) slot's most recent prior occurrence — and
// whether a predecessor existed to compare against at all (false only on a
// slot's first-ever appearance in this instance). Callers use hadPredecessor
// to count TRANSITIONS (see slotVerdicts in convertSessionToWekaTrace): a
// slot's first appearance isn't a transition either way, since there was
// nothing yet to connect to or diverge from.
//
// ID assignment (which numeric ids to hand back) is deliberately a separate
// concern from the connects-to-predecessor answer: a hash that exactly
// matches something from much earlier in the session (not the immediate
// predecessor) still reuses the SAME ids for correct cache-hit accounting,
// even though it's reported as NOT connecting for placement purposes, since
// it doesn't help the immediate-predecessor prefix match either way.
//
// Two ways to connect:
//   - exact hash match against the predecessor's own hash at this slot.
//   - the cache-control-churn heuristic: same role, `cache_control`
//     presence DIFFERS, byte count within cacheControlChurnByteTolerance.
//     Exists because wekai's capture hash is computed over the block's
//     serialized form INCLUDING its `cache_control` field, and Anthropic's
//     prompt-caching convention moves the cache breakpoint forward to the
//     newest block on nearly every multi-turn call — so the block that
//     carried the marker last time loses it this time (or vice versa), and
//     the underlying text is unchanged but the hash flips. Confirmed on the
//     real corpus: 62% of multi-request instances show exactly this
//     pattern. Gated on the presence FLIP specifically (not just "hash
//     differs") so a genuinely dynamic per-call block, which typically
//     carries no cache_control on either side, is never merged into a
//     false cache hit.
//
// A THIRD case, independent of connection: a matched hash's id-count is
// always reconciled against THIS occurrence's own declared tokens before
// being returned (see reconcileIDCount). Confirmed on the real corpus: some
// blocks carry a stable hash and a constant, tiny `bytes` value across many
// requests while `tokens` grows steadily between them (e.g. 1,577 -> 1,829
// -> 2,440 -> ... -> 5,050 tokens, same hash, same 82 bytes, every time) --
// almost certainly a dynamically-injected reference (a growing scratchpad
// or reminder block) that the capture represents by a small fixed pointer
// rather than its real, current size. Blindly reusing the id-run minted for
// an EARLIER, smaller occurrence of that hash under-counts the block's
// current length, which AIPerf's own loader detects and hard-aborts on:
// "turn-0 system prefix requires N hash blocks but only M were recorded".
// Reconciling on every hash hit -- not just the first -- extends a grown
// block with freshly minted trailing ids (preserving the shared prefix as a
// genuine, valid cache hit) and truncates a shrunk one, so the returned
// id-count always matches what this occurrence actually declares.
func (a *hashIDAssigner) idsForBlock(key blockKey, info blockInfo) (ids []int, connectsToPredecessor bool, hadPredecessor bool) {
	prev, hasPrev := a.bySlot[key]
	cacheControlChurnMatch := hasPrev &&
		prev.role == info.role &&
		prev.cacheControl != info.cacheControl &&
		byteCountsClose(prev.bytes, info.bytes)
	connects := hasPrev && (prev.hash == info.hash || cacheControlChurnMatch)
	want := blockCount(info.tokens, a.blockSize)

	if info.hash != "" {
		if hit, ok := a.assigned[info.hash]; ok {
			reconciled := a.reconcileIDCount(hit, want)
			if info.hash != "" {
				a.assigned[info.hash] = reconciled
			}
			info.ids = reconciled
			a.bySlot[key] = info
			return reconciled, connects, hasPrev
		}
	}

	if cacheControlChurnMatch {
		reconciled := a.reconcileIDCount(prev.ids, want)
		info.ids = reconciled
		a.bySlot[key] = info
		if info.hash != "" {
			a.assigned[info.hash] = reconciled
		}
		return reconciled, true, hasPrev
	}

	minted := make([]int, want)
	for i := range minted {
		minted[i] = a.next
		a.next++
	}
	if info.hash != "" {
		a.assigned[info.hash] = minted
	}
	info.ids = minted
	a.bySlot[key] = info
	return minted, connects, hasPrev
}

// blockCount is ceil(tokens/blockSize), minimum 1 -- the number of hash-id
// blocks a request's declared token count requires.
func blockCount(tokens, blockSize int) int {
	if tokens <= 0 {
		return 1
	}
	count := int(math.Ceil(float64(tokens) / float64(blockSize)))
	if count < 1 {
		count = 1
	}
	return count
}

// reconcileIDCount returns ids adjusted to exactly `want` entries: unchanged
// if already the right length, the existing prefix plus freshly minted
// trailing ids if want is longer (a grown block — the shared prefix is
// still a genuine cache hit), or truncated to the prefix if want is shorter
// (a shrunk block — arbitrarily discarding the excess tail rather than
// guessing which portion of it might still be valid).
func (a *hashIDAssigner) reconcileIDCount(ids []int, want int) []int {
	if len(ids) == want {
		return ids
	}
	if len(ids) > want {
		return ids[:want]
	}
	extended := make([]int, want)
	copy(extended, ids)
	for i := len(ids); i < want; i++ {
		extended[i] = a.next
		a.next++
	}
	return extended
}

// byteCountsClose reports whether b differs from a by no more than
// cacheControlChurnByteTolerance, guarding against a zero base value.
func byteCountsClose(a, b int) bool {
	if a <= 0 {
		return b == a
	}
	diff := float64(b - a)
	if diff < 0 {
		diff = -diff
	}
	return diff/float64(a) <= cacheControlChurnByteTolerance
}
