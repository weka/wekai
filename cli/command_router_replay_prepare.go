package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// RouterReplayPrepareCommand transforms a directory of redacted JSONL captures
// into one replay-friendly JSON file under ../replays/. The output bakes in
// every linkage the replayer needs (parent_spawn_request_id, fan-out grouping,
// per-request input/output token budgets) so the replayer doesn't re-resolve
// the tree at run time.
type RouterReplayPrepareCommand struct {
	Out  string `short:"o" long:"out" description:"Output file path. Default: <src>/../replays/replay-<earliest>-to-<latest>.json"`
	Name string `long:"name" description:"Embed this name in the file (default: derived from output path)"`
	Args struct {
		Src string `positional-arg-name:"src" description:"Directory of redacted *.jsonl files (or one .jsonl). Default: ~/.wekai/router/capture/redacted/"`
	} `positional-args:"yes"`
}

// ReplayHeader is the first line of a replay-v3 JSONL file. The remaining
// lines are one ReplaySession per line — this lets producers stream output
// and consumers stream input with a bounded queue, instead of holding the
// full file (and its decoded objects) in memory.
//
// v3 (vs v2) embeds per-request structured spec — system_blocks, tools,
// messages — so the replayer can reconstruct a byte/token-faithful HTTP
// body at run time, with synthetic content generated deterministically
// from each block's hash (same hash → same content → server prefix-cache
// hits the same way the original capture did).
type ReplayHeader struct {
	Schema      string        `json:"_schema"` // "replay-v3"
	Name        string        `json:"name"`
	GeneratedAt string        `json:"generated_at"`
	Source      string        `json:"source"`
	Summary     ReplaySummary `json:"summary"`
}

type ReplaySummary struct {
	Sessions           int    `json:"sessions"`
	Instances          int    `json:"instances"`
	Requests           int    `json:"requests"`
	Spawns             int    `json:"spawns"`
	MatchedSpawns      int    `json:"matched_spawns"`
	FanOutTurns        int    `json:"fan_out_turns"`
	MaxFanOutInOneTurn int    `json:"max_fan_out_in_one_turn"`
	RepairedParents    int    `json:"repaired_parents"`
	CyclesBroken       int    `json:"cycles_broken"`
	StartTs            string `json:"start_ts"`
	EndTs              string `json:"end_ts"`
}

type ReplaySession struct {
	SessionID string           `json:"session_id"`
	StartTs   string           `json:"start_ts"`
	Instances []ReplayInstance `json:"instances"`
}

type ReplayInstance struct {
	InstanceID           string          `json:"instance_id"`
	Role                 string          `json:"role"`
	ParentInstanceID     string          `json:"parent_instance_id,omitempty"`
	ParentSpawnRequestID uint64          `json:"parent_spawn_request_id,omitempty"`
	FanOutGroup          string          `json:"fan_out_group,omitempty"`
	FanOutSize           int             `json:"fan_out_size,omitempty"`
	FanOutPosition       int             `json:"fan_out_position,omitempty"`
	Requests             []ReplayRequest `json:"requests"`
}

type ReplayRequest struct {
	RequestID uint64 `json:"request_id"`
	Ts        string `json:"ts"`
	Model     string `json:"model"`
	Stream    bool   `json:"stream"`
	MaxTokens int    `json:"max_tokens"` // original max_tokens setting

	// Per-request usage from server (used for budgets, telemetry).
	InputTokens         int `json:"input_tokens"`   // server-reported total input (incl. cached)
	PrefillTokens       int `json:"prefill_tokens"` // input not served from cache
	CacheReadTokens     int `json:"cache_read_tokens"`
	CacheCreationTokens int `json:"cache_creation_tokens"`
	OutputTokens        int `json:"output_tokens"` // original output length — replayer sets max_tokens=this

	// Sampling knobs preserved verbatim so the replay HTTP body looks
	// like the original wire shape.
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	TopK        *int            `json:"top_k,omitempty"`
	Thinking    json.RawMessage `json:"thinking,omitempty"`

	// Structured spec (mirrors the redacted-capture schema). The replayer
	// uses these to materialize byte-equivalent synthetic content.
	// SystemBlocks: each original system block, one entry per block.
	// Tools: original tools array fingerprint (count + total bytes).
	// Messages: full conversation as the original request sent it —
	// every prior turn is preserved here, since the original capture
	// carried the full history per request.
	SystemBlocks []ReplaySystemBlock `json:"system_blocks,omitempty"`
	Tools        *ReplayToolsSpec    `json:"tools,omitempty"`
	Messages     []ReplayMessage     `json:"messages,omitempty"`

	StopReason        string `json:"stop_reason,omitempty"`
	UpstreamLatencyMs int64  `json:"upstream_latency_ms"`
	TotalMs           int64  `json:"total_ms"`
}

// ReplaySystemBlock mirrors redactedSystemBlock so the replayer can rebuild
// a system-array entry of the right size from each block's hash.
type ReplaySystemBlock struct {
	Type         string `json:"type"`
	Hash         string `json:"hash"`
	Bytes        int    `json:"bytes"`
	Tokens       int    `json:"tokens,omitempty"`
	CacheControl string `json:"cache_control,omitempty"`
}

// ReplayToolsSpec mirrors redactedToolsInfo — used by the replayer to
// generate N synthetic tools whose canonical JSON totals the original
// bytes count (deterministic from Hash, so the same tools array
// fingerprints identically across replays of the same session).
type ReplayToolsSpec struct {
	Count  int    `json:"count"`
	Bytes  int    `json:"bytes"`
	Tokens int    `json:"tokens,omitempty"`
	Hash   string `json:"hash"`
}

// ReplayMessage mirrors redactedMessage — the replayer rebuilds the
// message's content blocks from block_types + bytes (sized from Bytes,
// hashed from Hash for cross-request determinism), preserving tool_use
// and tool_result ids so the conversation maintains a valid tool-call
// reference graph.
type ReplayMessage struct {
	Role          string   `json:"role"`
	Hash          string   `json:"hash"`
	BlockTypes    []string `json:"block_types"`
	Bytes         int      `json:"bytes"`
	Tokens        int      `json:"tokens,omitempty"`
	CacheControl  string   `json:"cache_control,omitempty"`
	ToolUseIDs    []string `json:"tool_use_ids,omitempty"`
	ToolResultIDs []string `json:"tool_result_ids,omitempty"`
	SeedHash      string   `json:"seed_hash,omitempty"`
}

func (c *RouterReplayPrepareCommand) Execute(args []string) error {
	src := c.Args.Src
	if src == "" {
		home, _ := os.UserHomeDir()
		src = filepath.Join(home, ".wekai", "router", "capture", "redacted")
	}
	files, err := collectJSONLFiles(src)
	if err != nil {
		return err
	}

	// Reuse the tree command's parser + tree builder so the linkage logic
	// is single-sourced.
	var records []treeRecord
	for _, f := range files {
		recs, err := parseTreeFile(f)
		if err != nil {
			return fmt.Errorf("parse %s: %w", f, err)
		}
		records = append(records, recs...)
	}
	if len(records) == 0 {
		return fmt.Errorf("no records found in %s", src)
	}
	rep := buildTreeReport(records)

	// Index records and build per-instance request streams.
	recByID := make(map[uint64]*treeRecord, len(records))
	for i := range records {
		recByID[records[i].ID] = &records[i]
	}
	// Group records by their owning instance id (recompute the same key the
	// tree builder uses).
	instReqs := map[string][]*treeRecord{}
	for i := range records {
		r := &records[i]
		if r.IsPlain {
			continue
		}
		sig := personaSignature(r.Req.SystemBlocks)
		fp := instanceFingerprint(r.Req)
		// Use SessionID() (not r.Sid) so sid-less records resolve to the
		// same synthetic id buildTreeReport used; otherwise their requests
		// never get routed into instReqs and the replay session emits empty.
		sid := r.SessionID()
		key := sid + ":" + sig + ":" + fp
		// May have been collapsed onto a canonical main instance — find the
		// canonical key by checking if rep.Instances has this exact key; if
		// not, we look it up against the report's instance set by best-match
		// (same session, same persona, earliest-seen-by-time main).
		if _, ok := rep.Instances[key]; !ok {
			key = canonicalInstanceKey(rep, sid, sig)
			if key == "" {
				continue
			}
		}
		instReqs[key] = append(instReqs[key], r)
	}
	for k := range instReqs {
		sort.Slice(instReqs[k], func(i, j int) bool {
			return instReqs[k][i].Ts.Before(instReqs[k][j].Ts)
		})
	}

	// Stable session order: by first-seen timestamp.
	sessIDs := make([]string, 0, len(rep.Sessions))
	for sid := range rep.Sessions {
		sessIDs = append(sessIDs, sid)
	}
	sort.Slice(sessIDs, func(i, j int) bool {
		return rep.Sessions[sessIDs[i]].FirstSeenTs.Before(rep.Sessions[sessIDs[j]].FirstSeenTs)
	})

	// Resolve output path.
	dst := c.Out
	if dst == "" {
		base := filepath.Dir(strings.TrimRight(src, string(filepath.Separator)))
		dst = filepath.Join(base, "replays")
		if err := os.MkdirAll(dst, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dst, err)
		}
		fname := fmt.Sprintf("replay-%s-to-%s.jsonl",
			ymd(rep.StartTs), ymd(rep.EndTs))
		dst = filepath.Join(dst, fname)
	}
	name := c.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(dst), filepath.Ext(dst))
	}

	// Stream-write: header on line 1, then one session per line. Build each
	// session in turn rather than accumulating the whole structure in
	// memory — that way replay-prepare itself doesn't blow up on huge
	// captures.
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	bw := bufio.NewWriterSize(f, 1<<20)
	defer bw.Flush()

	header := ReplayHeader{
		Schema:      "replay-v3",
		Name:        name,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Source:      src,
		Summary: ReplaySummary{
			Sessions:           rep.TotalSessions,
			Instances:          rep.TotalInstances,
			Spawns:             rep.TotalSpawns,
			MatchedSpawns:      rep.MatchedSpawns,
			FanOutTurns:        rep.FanOutTurns,
			MaxFanOutInOneTurn: rep.MaxFanoutSeen,
			StartTs:            rep.StartTs,
			EndTs:              rep.EndTs,
		},
	}

	// Implicit-parent pass: for any instance not linked via Agent-tool
	// spawn, find the latest request in the SAME SESSION (any instance,
	// any role) whose ts < this instance's first ts; set that as
	// implicit parent_spawn_request_id. Only the absolute earliest
	// request in each session stays unparented (the true session root).
	//
	// Bakes the entire causal ordering of the session into the tree, so
	// the replayer just walks parent_done channels with no timestamp
	// inference at run time.
	type sessReq struct {
		ts time.Time
		id uint64
	}
	implicitParents := map[string]uint64{}
	for _, sid := range sessIDs {
		sess := rep.Sessions[sid]
		// Collect every request in the session, sorted by ts.
		var all []sessReq
		for _, iid := range sess.InstanceIDs {
			for _, r := range instReqs[iid] {
				if r.ID == 0 {
					continue
				}
				all = append(all, sessReq{ts: r.Ts, id: r.ID})
			}
		}
		if len(all) == 0 {
			continue
		}
		sort.Slice(all, func(i, j int) bool { return all[i].ts.Before(all[j].ts) })

		for _, iid := range sess.InstanceIDs {
			inst := rep.Instances[iid]
			if inst.HasParent {
				continue // already linked via Agent-tool spawn
			}
			myReqs := instReqs[iid]
			if len(myReqs) == 0 {
				continue
			}
			firstTs := myReqs[0].Ts
			// Find latest request strictly before firstTs. Binary
			// search would be faster but the per-session list is short
			// enough that linear from the back is fine.
			var bestID uint64
			for i := len(all) - 1; i >= 0; i-- {
				if all[i].ts.Before(firstTs) {
					bestID = all[i].id
					break
				}
			}
			if bestID != 0 {
				implicitParents[iid] = bestID
			}
		}
	}

	totalReqs := 0
	emittedSessions := 0
	emittedInstances := 0
	totalRepaired := 0
	totalCyclesBroken := 0
	// Two-pass write: build all sessions first to fill in Summary.Requests,
	// then write header + sessions. Sessions go through a slice but each
	// session is small relative to the file (median 2.7 KB) — we only ever
	// hold the marshaled bytes briefly. If we wanted true zero-buffering
	// we'd have to write the header last; sticking it on line 1 wins on
	// readability.
	var sessionLines [][]byte
	for _, sid := range sessIDs {
		sess := rep.Sessions[sid]
		rs := ReplaySession{
			SessionID: sid,
			StartTs:   sess.FirstSeenTs.UTC().Format(time.RFC3339Nano),
		}
		iids := append([]string(nil), sess.InstanceIDs...)
		sort.Slice(iids, func(i, j int) bool {
			return rep.Instances[iids[i]].FirstSeenTs.Before(rep.Instances[iids[j]].FirstSeenTs)
		})
		for _, iid := range iids {
			inst := rep.Instances[iid]
			ri := ReplayInstance{
				InstanceID: iid,
				Role:       inst.PersonaLabel,
			}
			if inst.HasParent {
				ri.ParentInstanceID = inst.ParentInstID
			}
			if inst.GroupKey != "" {
				ri.FanOutGroup = inst.GroupKey
				ri.FanOutSize = inst.GroupSize
				ri.FanOutPosition = inst.Position
				if id, err := parseUint(inst.GroupKey); err == nil {
					ri.ParentSpawnRequestID = id
				}
			}
			// Fall back to the implicit timestamp-derived parent if this
			// instance wasn't matched to an Agent-tool spawn.
			if ri.ParentSpawnRequestID == 0 {
				if pid, ok := implicitParents[iid]; ok {
					ri.ParentSpawnRequestID = pid
				}
			}
			for _, r := range instReqs[iid] {
				rr := buildReplayRequest(r)
				if rr != nil {
					ri.Requests = append(ri.Requests, *rr)
					totalReqs++
				}
			}
			if len(ri.Requests) == 0 {
				continue
			}
			rs.Instances = append(rs.Instances, ri)
			emittedInstances++
		}
		if len(rs.Instances) == 0 {
			continue
		}
		repaired := repairDanglingParents(&rs)
		totalRepaired += repaired
		cyclesBroken := breakParentCycles(&rs)
		totalCyclesBroken += cyclesBroken
		if cyclesBroken > 0 {
			fmt.Fprintf(os.Stderr, "replay-prepare: broke %d parent-dependency cycle(s) (instances promoted to roots)\n", cyclesBroken)
		}
		emittedSessions++
		// Compact JSON encoding (no indent) so each session sits on one
		// line. The header line uses the same encoding for consistency.
		line, err := json.Marshal(&rs)
		if err != nil {
			return fmt.Errorf("marshal session %s: %w", sid, err)
		}
		sessionLines = append(sessionLines, line)
	}
	header.Summary.Requests = totalReqs
	header.Summary.RepairedParents = totalRepaired
	header.Summary.CyclesBroken = totalCyclesBroken
	// Write header on line 1.
	hb, err := json.Marshal(&header)
	if err != nil {
		return err
	}
	if _, err := bw.Write(hb); err != nil {
		return err
	}
	if err := bw.WriteByte('\n'); err != nil {
		return err
	}
	for _, ln := range sessionLines {
		if _, err := bw.Write(ln); err != nil {
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			return err
		}
	}
	fmt.Printf("wrote %d sessions / %d instances / %d requests / %d repaired_parents / %d cycles_broken -> %s\n",
		emittedSessions, emittedInstances, totalReqs, totalRepaired, totalCyclesBroken, dst)
	return nil
}

// parseTreeFile is a thin wrapper around RouterTreeCommand.parseFile so we
// don't duplicate the raw/redacted detection logic here.
func parseTreeFile(path string) ([]treeRecord, error) {
	c := &RouterTreeCommand{}
	recs, _, _, _, err := c.parseFile(path)
	return recs, err
}

// canonicalInstanceKey resolves a non-canonical (sid, sig, fp) triple to the
// canonical instance id under the same session+persona, used after the tree
// builder collapses compaction-induced main-agent duplicates.
func canonicalInstanceKey(rep *treeReport, sid, sig string) string {
	sess := rep.Sessions[sid]
	if sess == nil {
		return ""
	}
	for _, iid := range sess.InstanceIDs {
		if rep.Instances[iid].PersonaSig == sig {
			return iid
		}
	}
	return ""
}

func buildReplayRequest(r *treeRecord) *ReplayRequest {
	if r.Resp.PlainJSON != nil {
		// count_tokens responses: not part of conversational replay.
		return nil
	}
	rr := &ReplayRequest{
		RequestID:   r.ID,
		Ts:          r.Ts.UTC().Format(time.RFC3339Nano),
		Model:       r.Req.Model,
		Stream:      r.Req.Stream,
		MaxTokens:   r.Req.MaxTokens,
		Temperature: r.Req.Temperature,
		TopP:        r.Req.TopP,
		TopK:        r.Req.TopK,
		Thinking:    r.Req.Thinking,
		StopReason:  r.Resp.StopReason,
	}
	if r.Resp.Usage != nil {
		rr.InputTokens = r.Resp.Usage.InputTokens +
			r.Resp.Usage.CacheReadInputTokens +
			r.Resp.Usage.CacheCreationInputTokens
		rr.PrefillTokens = r.Resp.Usage.InputTokens
		rr.CacheReadTokens = r.Resp.Usage.CacheReadInputTokens
		rr.CacheCreationTokens = r.Resp.Usage.CacheCreationInputTokens
		rr.OutputTokens = r.Resp.Usage.OutputTokens
	}
	// Structured spec — pulled verbatim from the redacted request so the
	// replayer can rebuild every system block, the tools array, and the
	// full messages array to byte-equivalent shape.
	for _, b := range r.Req.SystemBlocks {
		rr.SystemBlocks = append(rr.SystemBlocks, ReplaySystemBlock{
			Type:         b.Type,
			Hash:         b.Hash,
			Bytes:        b.Bytes,
			Tokens:       b.Tokens,
			CacheControl: b.CacheControl,
		})
	}
	if r.Req.Tools != nil {
		rr.Tools = &ReplayToolsSpec{
			Count:  r.Req.Tools.Count,
			Bytes:  r.Req.Tools.Bytes,
			Tokens: r.Req.Tools.Tokens,
			Hash:   r.Req.Tools.Hash,
		}
	}
	for _, m := range r.Req.Messages {
		rr.Messages = append(rr.Messages, ReplayMessage{
			Role:          m.Role,
			Hash:          m.Hash,
			BlockTypes:    m.BlockTypes,
			Bytes:         m.Bytes,
			Tokens:        m.Tokens,
			CacheControl:  m.CacheControl,
			ToolUseIDs:    m.ToolUseIDs,
			ToolResultIDs: m.ToolResultIDs,
			SeedHash:      m.SeedHash,
		})
	}
	return rr
}

// repairDanglingParents validates every instance's ParentSpawnRequestID against
// the set of request IDs actually emitted in this session. Any instance whose
// parent ref does not resolve within the session is re-parented by timestamp to
// the latest emitted request strictly before the instance's own first request,
// or to 0 (true root) if none exists. Invalid fan-out metadata is cleared on
// repaired instances. Returns the number of instances repaired.
func repairDanglingParents(rs *ReplaySession) int {
	emitted := map[uint64]bool{}
	type timedReq struct {
		ts time.Time
		id uint64
	}
	var all []timedReq
	for _, inst := range rs.Instances {
		for _, req := range inst.Requests {
			emitted[req.RequestID] = true
			ts, err := time.Parse(time.RFC3339Nano, req.Ts)
			if err == nil {
				all = append(all, timedReq{ts: ts, id: req.RequestID})
			}
		}
	}
	if len(all) == 0 {
		return 0
	}
	sort.Slice(all, func(i, j int) bool { return all[i].ts.Before(all[j].ts) })

	repaired := 0
	for i := range rs.Instances {
		inst := &rs.Instances[i]
		if inst.ParentSpawnRequestID == 0 {
			continue // already a root
		}
		if emitted[inst.ParentSpawnRequestID] {
			continue // valid: parent request exists in this session
		}
		// Dangling. Determine the instance's first request timestamp.
		firstTs := time.Time{}
		if len(inst.Requests) > 0 {
			ts, err := time.Parse(time.RFC3339Nano, inst.Requests[0].Ts)
			if err == nil {
				firstTs = ts
			}
		}
		// Find latest emitted request with ts strictly before firstTs.
		var bestID uint64
		for j := len(all) - 1; j >= 0; j-- {
			if all[j].ts.Before(firstTs) {
				bestID = all[j].id
				break
			}
		}
		inst.ParentSpawnRequestID = bestID
		inst.FanOutGroup = ""
		inst.FanOutSize = 0
		inst.FanOutPosition = 0
		repaired++
	}
	return repaired
}

// breakParentCycles makes the session's parent graph acyclic. Each instance's
// ParentSpawnRequestID links it to the request that spawned it; the builder's
// heuristics can produce cycles (A's parent owned by B, B's owned by A, or
// longer). Any instance whose parent chain does not reach a true root
// (ParentSpawnRequestID == 0, or a parent request owned by no instance) sits
// in or behind a cycle — its ParentSpawnRequestID is reset to 0 (promoted to
// a root). Returns the number of instances promoted.
func breakParentCycles(sess *ReplaySession) int {
	instances := sess.Instances

	// Build reqOwner: every request's RequestID (!= 0) → its instance index,
	// over ALL instances in the session.
	reqOwner := map[uint64]int{}
	for i := range instances {
		for _, r := range instances[i].Requests {
			if r.RequestID != 0 {
				reqOwner[r.RequestID] = i
			}
		}
	}

	// Anchor: an instance is anchored if ParentSpawnRequestID == 0 (true root)
	// OR its parent request id is not in reqOwner (dangling → treat as root).
	anchored := make([]bool, len(instances))
	for i := range instances {
		pid := instances[i].ParentSpawnRequestID
		if pid == 0 {
			anchored[i] = true
			continue
		}
		_, ok := reqOwner[pid]
		if !ok {
			// Dangling parent — already handled by repairDanglingParents, but
			// treat as anchored here so it doesn't get promoted again.
			anchored[i] = true
		}
	}

	// Fixpoint: propagate anchored status through the dependency graph.
	// An unanchored instance becomes anchored if the owner of its
	// ParentSpawnRequestID is anchored.
	for {
		changed := false
		for i := range instances {
			if anchored[i] {
				continue
			}
			pid := instances[i].ParentSpawnRequestID
			owner, ok := reqOwner[pid]
			if !ok {
				// Shouldn't happen after the first pass, but be defensive.
				anchored[i] = true
				changed = true
				continue
			}
			if anchored[owner] {
				anchored[i] = true
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	// Any instance still not anchored sits in or behind a cycle.
	// Promote it to a root.
	promoted := 0
	for i := range instances {
		if !anchored[i] {
			instances[i].ParentSpawnRequestID = 0
			promoted++
		}
	}
	return promoted
}

func parseUint(s string) (uint64, error) {
	var n uint64
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("not a uint: %q", s)
		}
		n = n*10 + uint64(c-'0')
	}
	if s == "" {
		return 0, fmt.Errorf("empty")
	}
	return n, nil
}

func ymd(ts string) string {
	if len(ts) >= 10 {
		return ts[:10]
	}
	return ts
}
