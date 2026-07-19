package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/weka/wekai-core/llm"
)

// RouterTreeCommand reads redacted (or raw — auto-redacts in memory) capture
// JSONL and emits the agent tree: distinct sessions, agent instances per
// session, parent→child edges resolved by PromptHash↔SeedHash equality, and
// parallel sibling groups (siblings emitted by one response).
//
// This validates that the redacted schema (with PromptHash + SeedHash)
// carries enough information to rebuild the tree without reading any
// prompt content.
type RouterTreeCommand struct {
	Format     string `short:"f" long:"format" choice:"text" choice:"json" default:"text" description:"Output format"`
	PerSession bool   `long:"per-session" description:"Also print a per-session tree (verbose)"`
	TopN       int    `long:"top-sessions" default:"10" description:"How many top sessions to detail"`
	TopUsers   int    `long:"top-users" default:"10" description:"How many top users to detail"`
	ShowCosts  bool   `long:"show-costs" description:"Compute per-user dollar cost from response.usage and model pricing"`
	Args       struct {
		Path string `positional-arg-name:"path" description:"Directory of *.jsonl or a single .jsonl file" required:"yes"`
	} `positional-args:"yes"`
}

// treeRecord is the minimal projection of a capture record needed for
// tree-building. We pull session id from headers and re-use the redacted
// schema for body fields.
type treeRecord struct {
	ID      uint64
	Ts      time.Time
	Sid     string
	User    string
	ModelIn string
	Req     redactedRequest
	Resp    redactedResponse
	IsPlain bool
}

// OrphanSessionID is the synthetic session id assigned to every record that
// lacks an X-Claude-Code-Session-Id header. Bundling all sid-less records
// under one session means: (a) the real-session count and the per-session
// averages aren't diluted by N single-record placeholders, (b) at replay
// time the whole orphan stream consumes one concurrency slot rather than N.
const OrphanSessionID = "orphan:no-session-id"

// SessionID returns Sid if the request carried an X-Claude-Code-Session-Id
// header, otherwise the shared OrphanSessionID so the record still lands
// somewhere in rep.Sessions. Every consumer that keys by session must use
// this method, not r.Sid directly — otherwise orphan records get lost from
// instance maps even though they're in rep.Sessions.
func (r treeRecord) SessionID() string {
	if r.Sid != "" {
		return r.Sid
	}
	return OrphanSessionID
}

// captureRecordTreeRaw mirrors the on-disk record shape but typed loosely so
// we can decode either raw or redacted without two parsers.
type captureRecordTreeRaw struct {
	ID       uint64      `json:"id"`
	Ts       string      `json:"ts"`
	Method   string      `json:"method"`
	Path     string      `json:"path"`
	Status   int         `json:"status,omitempty"`
	ModelIn  string      `json:"model_in,omitempty"`
	ModelOut string      `json:"model_out,omitempty"`
	User     string      `json:"user,omitempty"`
	Request  captureBody `json:"request"`
	Response captureBody `json:"response"`
}

func (c *RouterTreeCommand) Execute(args []string) error {
	files, err := collectJSONLFiles(c.Args.Path)
	if err != nil {
		return err
	}

	var records []treeRecord
	rawCount, redactedCount, errorCount := 0, 0, 0
	for _, f := range files {
		recs, raw, red, errs, err := c.parseFile(f)
		if err != nil {
			return fmt.Errorf("parse %s: %w", f, err)
		}
		records = append(records, recs...)
		rawCount += raw
		redactedCount += red
		errorCount += errs
	}

	if len(records) == 0 {
		return fmt.Errorf("no records found")
	}

	report := buildTreeReport(records)
	report.Files = files
	report.RawCount = rawCount
	report.RedactedCount = redactedCount
	report.ErrorRecordsDropped = errorCount

	if c.Format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}
	report.PrintText(os.Stdout, c.PerSession, c.TopN, c.TopUsers, c.ShowCosts)
	return nil
}

// collectJSONLFiles returns one or more .jsonl paths under p (file or dir).
func collectJSONLFiles(p string) ([]string, error) {
	info, err := os.Stat(p)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		if !strings.HasSuffix(p, ".jsonl") {
			return nil, fmt.Errorf("expected .jsonl file or directory: %s", p)
		}
		return []string{p}, nil
	}
	entries, err := os.ReadDir(p)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			out = append(out, filepath.Join(p, e.Name()))
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no .jsonl files in %s", p)
	}
	sort.Strings(out)
	return out, nil
}

func (c *RouterTreeCommand) parseFile(path string) ([]treeRecord, int, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer f.Close()

	var out []treeRecord
	rawCount, redactedCount, errorCount := 0, 0, 0
	// Use bufio.NewReader + ReadBytes('\n') because raw capture lines can
	// be hundreds of MB (a long conversation's full body on one line) and
	// bufio.Scanner has a hard line-size cap. Same approach as
	// command_router_redact.go.
	br := bufio.NewReaderSize(f, 1<<20)

	for {
		line, readErr := br.ReadBytes('\n')
		if len(line) > 0 {
			// Strip trailing newline before unmarshal.
			if line[len(line)-1] == '\n' {
				line = line[:len(line)-1]
			}
			if err := c.parseLine(line, &out, &rawCount, &redactedCount, &errorCount); err != nil {
				return out, rawCount, redactedCount, errorCount, err
			}
		}
		if readErr == io.EOF {
			return out, rawCount, redactedCount, errorCount, nil
		}
		if readErr != nil {
			return out, rawCount, redactedCount, errorCount, readErr
		}
	}
}

func (c *RouterTreeCommand) parseLine(line []byte, out *[]treeRecord, rawCount, redactedCount, errorCount *int) error {
	if len(line) == 0 {
		return nil
	}
	var rec captureRecordTreeRaw
	if err := json.Unmarshal(line, &rec); err != nil {
		// Skip malformed lines; not fatal.
		return nil
	}
	if rec.Method != "POST" {
		return nil
	}
	// Drop non-2xx responses. Errored requests have no usable usage/output
	// data and would otherwise inflate request/instance counts without
	// contributing useful structure. Status 0 (no response captured)
	// counts as an error too.
	if rec.Status < 200 || rec.Status >= 300 {
		*errorCount++
		return nil
	}
	ts, _ := time.Parse(time.RFC3339Nano, rec.Ts)
	var req redactedRequest
	var resp redactedResponse

	// Detect format by probing the request body.
	var probe map[string]interface{}
	isRedacted := false
	if err := json.Unmarshal(rec.Request.Body, &probe); err == nil {
		if schema, ok := probe["_schema"].(string); ok && schema == "req-v1" {
			isRedacted = true
		}
	}
	if isRedacted {
		*redactedCount++
		_ = json.Unmarshal(rec.Request.Body, &req)
		_ = json.Unmarshal(rec.Response.Body, &resp)
	} else {
		*rawCount++
		var reqBody, respBody string
		_ = json.Unmarshal(rec.Request.Body, &reqBody)
		_ = json.Unmarshal(rec.Response.Body, &respBody)
		reqRaw, respRaw := BuildRedactedPair([]byte(reqBody), []byte(respBody))
		_ = json.Unmarshal(reqRaw, &req)
		_ = json.Unmarshal(respRaw, &resp)
	}
	sid := ""
	if rec.Request.Headers != nil {
		if v := rec.Request.Headers.Get("X-Claude-Code-Session-Id"); v != "" {
			sid = v
		}
	}
	// model_out wins when the router rewrote the model (e.g., when a wekai
	// alias was the target); else model_in is the actual billed model.
	model := rec.ModelOut
	if model == "" {
		model = rec.ModelIn
	}
	*out = append(*out, treeRecord{
		ID: rec.ID, Ts: ts, Sid: sid, User: rec.User, ModelIn: model,
		Req: req, Resp: resp,
		IsPlain: resp.PlainJSON != nil,
	})
	return nil
}

// ---- Tree analysis -----------------------------------------------------

type spawnEdge struct {
	ParentRecID  uint64
	ParentInstID string // logical id of the parent agent instance
	ParentRole   string
	ResponseID   uint64 // response that emitted this spawn (= ParentRecID)
	GroupKey     string // unique per parent response; used to group siblings
	GroupSize    int    // total spawns in that parent response
	ToolUseID    string
	Position     int
	// Synthetic is true when the parent record had no X-Claude-Code-Session-Id
	// (i.e., the spawn was emitted from the orphan bucket). Tracked so we can
	// split the matched-spawn percentage between real sessions and orphans
	// — otherwise a noisy orphan stream can distort the headline metric.
	Synthetic bool
}

type instanceInfo struct {
	ID            string // synthesized stable id
	SessionID     string
	PersonaSig    string // role fingerprint
	PersonaLabel  string // human-friendly label, see classifyPersona
	FirstSeenTs   time.Time
	LastSeenTs    time.Time
	RequestCount  int
	SeedHash      string
	HasParent     bool
	ParentInstID  string
	GroupKey      string
	GroupSize     int
	Position      int
	ChildrenInsts []string
	IsRoot        bool
	OrphanRoot    bool // root that's neither a "main" nor a known helper persona
}

type sessionInfo struct {
	ID           string
	FirstSeenTs  time.Time
	LastSeenTs   time.Time
	RequestCount int
	InstanceIDs  []string
	FanOutTurns  int
	MaxFanout    int
	User         string
	// Synthetic is true for sessions we fabricated to host a record that
	// had no X-Claude-Code-Session-Id header. One synth session per such
	// record so every captured request still lands somewhere on the tree.
	Synthetic bool
}

// userInfo aggregates per-user activity. Only populated for capture records
// that carry a non-empty top-level "user" field (router was run with
// --user-prefix). Records without user metadata contribute to a sentinel
// bucket that's excluded from user-level stats.
type userInfo struct {
	ID           string          `json:"id"`
	RequestCount int             `json:"request_count"`
	Sessions     map[string]bool `json:"-"`
	SessionCount int             `json:"session_count"`
	FirstSeenTs  time.Time       `json:"first_seen_ts"`
	LastSeenTs   time.Time       `json:"last_seen_ts"`

	// Token totals from response.usage. InputTokens is the full input billed
	// to this user — uncached + cache_read + cache_creation. CacheReadTokens
	// is the portion served from cache, which we use to compute cache %.
	InputTokens     int64 `json:"input_tokens"`
	OutputTokens    int64 `json:"output_tokens"`
	CacheReadTokens int64 `json:"cache_read_tokens"`

	// Cost is the running USD total across requests, derived per-record from
	// llm.LookupModelByIdentifier(record.ModelIn) + llm.CalculateCost.
	// Records whose model isn't in the registry contribute 0 and are counted
	// in UnknownModelRecords so the user sees how much spend is unpriced.
	Cost                float64 `json:"cost_usd"`
	UnknownModelRecords int     `json:"unknown_model_records"`
}

type treeReport struct {
	Files               []string                 `json:"files"`
	TotalRecords        int                      `json:"total_records"`
	RawCount            int                      `json:"raw_records"`
	RedactedCount       int                      `json:"redacted_records"`
	Sessions            map[string]*sessionInfo  `json:"sessions"`
	Instances           map[string]*instanceInfo `json:"instances"`
	Roles               map[string]int           `json:"role_request_counts"` // persona label → request count
	RoleInstanceCounts  map[string]int           `json:"role_instance_counts"`
	TotalSessions       int                      `json:"total_sessions"`
	TotalInstances      int                      `json:"total_instances"`
	TotalSpawns         int                      `json:"total_spawns"`
	MatchedSpawns       int                      `json:"matched_spawns"`
	TotalSpawnsReal     int                      `json:"total_spawns_real"`
	TotalSpawnsOrphan   int                      `json:"total_spawns_orphan"`
	MatchedSpawnsReal   int                      `json:"matched_spawns_real"`
	MatchedSpawnsOrphan int                      `json:"matched_spawns_orphan"`
	FanOutTurns         int                      `json:"fan_out_turns"`
	MaxFanoutSeen       int                      `json:"max_fanout_seen"`
	StartTs             string                   `json:"start_ts"`
	EndTs               string                   `json:"end_ts"`

	// User metadata. UsersDetected is false when no record carried a
	// non-empty "user" field (router lacked --user-prefix or upstream
	// captures pre-date that flag). The remaining fields are zero in
	// that case.
	UsersDetected       bool                 `json:"users_detected"`
	Users               map[string]*userInfo `json:"users,omitempty"`
	TotalUsers          int                  `json:"total_users"`
	MaxRequestsPerUser  int                  `json:"max_requests_per_user"`
	MaxSessionsPerUser  int                  `json:"max_sessions_per_user"`
	AvgRequestsPerUser  float64              `json:"avg_requests_per_user"`
	AvgSessionsPerUser  float64              `json:"avg_sessions_per_user"`
	TopUserByRequests   string               `json:"top_user_by_requests,omitempty"`
	TopUserBySessions   string               `json:"top_user_by_sessions,omitempty"`
	RequestsWithoutUser int                  `json:"requests_without_user"`
	SessionsWithoutUser int                  `json:"sessions_without_user"`

	// Session-id coverage. Records missing X-Claude-Code-Session-Id get a
	// synthesized one-record session so the report covers every captured
	// request. OrphanSessions counts those synth sessions; it equals
	// RequestsWithoutSessionID under the current one-record-per-orphan rule.
	RequestsWithSessionID    int     `json:"requests_with_session_id"`
	RequestsWithoutSessionID int     `json:"requests_without_session_id"`
	SessionIDCoveragePct     float64 `json:"session_id_coverage_pct"`
	OrphanSessions           int     `json:"orphan_sessions"`
	PlainResponseRecords     int     `json:"plain_response_records"`

	// ErrorRecordsDropped is the count of non-2xx records filtered out
	// at parse time. They're never counted in TotalRecords or any
	// downstream metric — this is the only field that surfaces them.
	ErrorRecordsDropped int `json:"error_records_dropped"`

	// Cost tracking. TotalCost is the sum across all user-tagged records.
	// UnknownModels maps each unregistered model id to how many records used
	// it — surfaced as a warning so the registry can be filled in. Records
	// without a user tag still contribute to UnknownModels but not to
	// per-user costs.
	TotalCost     float64        `json:"total_cost_usd"`
	UnknownModels map[string]int `json:"unknown_models,omitempty"`
}

// personaSignature builds a stable role fingerprint from system blocks.
//
// The system field for /v1/messages is structured as up to three blocks:
//
//	[0] tiny billing header containing a per-request "cch=" cache hash —
//	    changes on every request, so we MUST exclude it.
//	[1] short persona banner ("You are Claude Code...") — stable per
//	    agent role; this is the role-discriminating block we want.
//	[2] large env block (cwd, CLAUDE.md, available tools etc.) — varies
//	    across sessions but is stable within a session for one role.
//
// We pick block[1] when present (the canonical role banner), fall back
// to block[0] if there's only one block, and return "" for no-system
// requests (ephemeral haiku side-calls).
func personaSignature(blocks []redactedSystemBlock) string {
	if len(blocks) == 0 {
		return ""
	}
	if len(blocks) >= 2 {
		return blocks[1].Hash
	}
	return blocks[0].Hash
}

// instanceFingerprint identifies a logical agent instance within a session.
// For records with a SeedHash on messages[0] (subagent seed shape), we use
// the SeedHash. For records without (e.g., a continuation that has tool
// results in messages[0]), we use messages[0].Hash. Same instance across all
// turns iff its first user message stays the same — same caveat as raw-mode
// analysis: compaction rewrites the seed and creates a synthetic new
// instance under this scheme.
func instanceFingerprint(req redactedRequest) string {
	if len(req.Messages) == 0 {
		return ""
	}
	m0 := req.Messages[0]
	if m0.SeedHash != "" {
		return m0.SeedHash
	}
	return m0.Hash
}

func buildTreeReport(records []treeRecord) *treeReport {
	rep := &treeReport{
		Sessions:           map[string]*sessionInfo{},
		Instances:          map[string]*instanceInfo{},
		Roles:              map[string]int{},
		RoleInstanceCounts: map[string]int{},
		Users:              map[string]*userInfo{},
		UnknownModels:      map[string]int{},
	}
	rep.TotalRecords = len(records)

	sort.Slice(records, func(i, j int) bool { return records[i].Ts.Before(records[j].Ts) })
	if len(records) > 0 {
		rep.StartTs = records[0].Ts.Format(time.RFC3339)
		rep.EndTs = records[len(records)-1].Ts.Format(time.RFC3339)
	}

	// Session-id coverage. Done in a dedicated walk so the counters reflect
	// the original capture, not the synthesized state used by passes below.
	for _, r := range records {
		if r.IsPlain {
			rep.PlainResponseRecords++
			continue
		}
		if r.Sid == "" {
			rep.RequestsWithoutSessionID++
		} else {
			rep.RequestsWithSessionID++
		}
	}
	if rep.TotalRecords > 0 {
		rep.SessionIDCoveragePct = 100.0 * float64(rep.RequestsWithSessionID) /
			float64(rep.TotalRecords)
	}

	// Pass 1: index spawns by (session, prompt_hash).
	spawnRegistry := map[string]*spawnEdge{}
	matchedKeys := map[string]bool{} // populated in pass 3
	personaSpawnsOthers := map[string]bool{}
	personaIsSpawnTarget := map[string]bool{}
	fanOutGroups := map[string]bool{} // dedupe parent turns across many requests
	for _, r := range records {
		if r.IsPlain {
			continue
		}
		spawnsInResp := []int{}
		for i, ob := range r.Resp.OutputBlocks {
			if ob.Type == "tool_use" && ob.PromptHash != "" {
				spawnsInResp = append(spawnsInResp, i)
			}
		}
		groupKey := fmt.Sprintf("%d", r.ID)
		groupSize := len(spawnsInResp)
		for pos, i := range spawnsInResp {
			ob := r.Resp.OutputBlocks[i]
			key := r.SessionID() + "|" + ob.PromptHash
			if _, exists := spawnRegistry[key]; !exists {
				spawnRegistry[key] = &spawnEdge{
					ParentRecID: r.ID,
					ResponseID:  r.ID,
					ToolUseID:   ob.ToolUseID,
					GroupKey:    groupKey,
					GroupSize:   groupSize,
					Position:    pos,
					Synthetic:   r.Sid == "",
				}
			}
		}
		if groupSize > 0 {
			personaSpawnsOthers[personaSignature(r.Req.SystemBlocks)] = true
			if groupSize >= 2 {
				fanOutGroups[groupKey] = true
			}
			if groupSize > rep.MaxFanoutSeen {
				rep.MaxFanoutSeen = groupSize
			}
		}
	}
	// TotalSpawns = distinct (session, prompt_hash) spawn events; one entry
	// per Agent tool call regardless of how many request records echo it.
	rep.TotalSpawns = len(spawnRegistry)
	for _, e := range spawnRegistry {
		if e.Synthetic {
			rep.TotalSpawnsOrphan++
		} else {
			rep.TotalSpawnsReal++
		}
	}
	rep.FanOutTurns = len(fanOutGroups)

	// Pass 2: build instances and link parents.
	for _, r := range records {
		if r.IsPlain {
			continue
		}
		sid := r.SessionID()
		synthetic := r.Sid == ""
		s, ok := rep.Sessions[sid]
		if !ok {
			// Orphan session is multi-user by construction; leave User blank
			// so it doesn't get attributed to whichever user happened to be
			// the first sid-less record we saw.
			user := r.User
			if synthetic {
				user = ""
			}
			s = &sessionInfo{ID: sid, FirstSeenTs: r.Ts, LastSeenTs: r.Ts, User: user, Synthetic: synthetic}
			rep.Sessions[sid] = s
			if synthetic {
				rep.OrphanSessions++
			}
		} else if !s.Synthetic && s.User == "" && r.User != "" {
			s.User = r.User
		}
		s.RequestCount++
		if r.Ts.Before(s.FirstSeenTs) {
			s.FirstSeenTs = r.Ts
		}
		if r.Ts.After(s.LastSeenTs) {
			s.LastSeenTs = r.Ts
		}

		if r.User != "" {
			rep.UsersDetected = true
			u, ok := rep.Users[r.User]
			if !ok {
				u = &userInfo{ID: r.User, Sessions: map[string]bool{}, FirstSeenTs: r.Ts, LastSeenTs: r.Ts}
				rep.Users[r.User] = u
			}
			u.RequestCount++
			// Don't count the shared orphan session toward this user's
			// session set — the orphan bucket holds many users' sid-less
			// records and would falsely add +1 session to each of them.
			if !synthetic {
				u.Sessions[sid] = true
			}
			if r.Ts.Before(u.FirstSeenTs) {
				u.FirstSeenTs = r.Ts
			}
			if r.Ts.After(u.LastSeenTs) {
				u.LastSeenTs = r.Ts
			}
			if r.Resp.Usage != nil {
				usage := r.Resp.Usage
				u.InputTokens += int64(usage.InputTokens) +
					int64(usage.CacheCreationInputTokens) +
					int64(usage.CacheReadInputTokens)
				u.OutputTokens += int64(usage.OutputTokens)
				u.CacheReadTokens += int64(usage.CacheReadInputTokens)

				// Cost lookup. ModelIn may be empty for old captures or for
				// non-/v1/messages paths; treat as unknown and contribute 0
				// rather than skipping the record (the user still made the
				// request — we just can't price it).
				if r.ModelIn == "" {
					u.UnknownModelRecords++
					rep.UnknownModels[""]++
				} else if llm.LookupModelByIdentifier == nil {
					u.UnknownModelRecords++
					rep.UnknownModels[r.ModelIn]++
				} else if info, ok := llm.LookupModelByIdentifier(r.ModelIn); ok {
					c := llm.CalculateCost(info, llm.Usage{
						InputTokens:              usage.InputTokens,
						CacheReadInputTokens:     usage.CacheReadInputTokens,
						CacheCreationInputTokens: usage.CacheCreationInputTokens,
						OutputTokens:             usage.OutputTokens,
					})
					u.Cost += c
					rep.TotalCost += c
				} else {
					u.UnknownModelRecords++
					rep.UnknownModels[r.ModelIn]++
				}
			}
		}

		sig := personaSignature(r.Req.SystemBlocks)
		fp := instanceFingerprint(r.Req)
		instID := sid + ":" + sig + ":" + fp
		// Special-collapse "main" candidates: instances whose persona spawns
		// others get collapsed by (sid, sig) so compaction-induced first-msg
		// rewrites don't multiply their count. We don't know which persona is
		// "main" until pass-3 classification, so we'll fix this up after
		// classification — for now, keep the raw key.

		inst, exists := rep.Instances[instID]
		if !exists {
			inst = &instanceInfo{
				ID: instID, SessionID: sid, PersonaSig: sig,
				FirstSeenTs: r.Ts, LastSeenTs: r.Ts,
			}
			rep.Instances[instID] = inst
			s.InstanceIDs = append(s.InstanceIDs, instID)
		}
		inst.RequestCount++
		if r.Ts.Before(inst.FirstSeenTs) {
			inst.FirstSeenTs = r.Ts
		}
		if r.Ts.After(inst.LastSeenTs) {
			inst.LastSeenTs = r.Ts
		}
		// Capture seed hash on the first record we see for this instance —
		// this is what we'll join against the spawn registry.
		if inst.SeedHash == "" && len(r.Req.Messages) > 0 {
			m0 := r.Req.Messages[0]
			if m0.SeedHash != "" {
				inst.SeedHash = m0.SeedHash
			}
		}
	}

	// Pass 3: link children to parents via SeedHash → spawnRegistry.
	for _, inst := range rep.Instances {
		if inst.SeedHash == "" {
			continue
		}
		key := inst.SessionID + "|" + inst.SeedHash
		edge, ok := spawnRegistry[key]
		if !ok {
			continue
		}
		inst.HasParent = true
		inst.GroupKey = edge.GroupKey
		inst.GroupSize = edge.GroupSize
		inst.Position = edge.Position
		matchedKeys[key] = true
	}
	// MatchedSpawns = distinct (session, prompt_hash) keys with at least
	// one child instance — i.e., resolved spawn events.
	rep.MatchedSpawns = len(matchedKeys)
	for k := range matchedKeys {
		if e, ok := spawnRegistry[k]; ok && e.Synthetic {
			rep.MatchedSpawnsOrphan++
		} else {
			rep.MatchedSpawnsReal++
		}
	}

	// Pass 3.5: fill ParentInstID by mapping records to instances.
	recIDToInstID := map[uint64]string{}
	for _, r := range records {
		if r.IsPlain {
			continue
		}
		sig := personaSignature(r.Req.SystemBlocks)
		fp := instanceFingerprint(r.Req)
		recIDToInstID[r.ID] = r.SessionID() + ":" + sig + ":" + fp
	}
	for _, inst := range rep.Instances {
		if !inst.HasParent {
			continue
		}
		key := inst.SessionID + "|" + inst.SeedHash
		edge := spawnRegistry[key]
		if edge == nil {
			continue
		}
		if pid, ok := recIDToInstID[edge.ParentRecID]; ok {
			inst.ParentInstID = pid
		}
	}

	// Pass 4: classify each instance by tree position.
	// Per-instance — not per-persona — because the same persona can serve
	// in different roles in different sessions. A child whose persona
	// happens to spawn other agents elsewhere is still a sub-agent here.
	for _, inst := range rep.Instances {
		if inst.HasParent {
			personaIsSpawnTarget[inst.PersonaSig] = true
		}
	}
	for _, inst := range rep.Instances {
		var label string
		switch {
		case inst.HasParent:
			label = "sub-agent"
			inst.IsRoot = false
		case inst.PersonaSig == "":
			label = "ephemeral (no system)"
			inst.IsRoot = true
		case personaIsSpawnTarget[inst.PersonaSig] && !personaSpawnsOthers[inst.PersonaSig]:
			// Persona shape matches a child we successfully linked elsewhere,
			// but this instance couldn't be linked (hash mismatch, compaction,
			// missing parent capture). Render as root so it still appears on
			// the session tree, but label it so it's not confused with a
			// real main agent.
			label = "orphan-sub-agent"
			inst.IsRoot = true
			inst.OrphanRoot = true
		case personaSpawnsOthers[inst.PersonaSig]:
			label = "main"
			inst.IsRoot = true
		default:
			label = "helper-or-isolated"
			inst.IsRoot = true
		}
		inst.PersonaLabel = label
		rep.RoleInstanceCounts[label]++
		rep.Roles[label] += inst.RequestCount
	}

	// Pass 5: collapse main-persona instances per session to one
	// (compaction handling).
	collapse := map[string]string{} // old instID -> new instID
	for _, sess := range rep.Sessions {
		// find main persona's instances in this session
		mainIDs := []string{}
		for _, iid := range sess.InstanceIDs {
			if rep.Instances[iid].PersonaLabel == "main" {
				mainIDs = append(mainIDs, iid)
			}
		}
		if len(mainIDs) <= 1 {
			continue
		}
		sort.Slice(mainIDs, func(i, j int) bool {
			return rep.Instances[mainIDs[i]].FirstSeenTs.Before(rep.Instances[mainIDs[j]].FirstSeenTs)
		})
		canon := mainIDs[0]
		canonInst := rep.Instances[canon]
		for _, iid := range mainIDs[1:] {
			merged := rep.Instances[iid]
			canonInst.RequestCount += merged.RequestCount
			if merged.LastSeenTs.After(canonInst.LastSeenTs) {
				canonInst.LastSeenTs = merged.LastSeenTs
			}
			collapse[iid] = canon
			delete(rep.Instances, iid)
		}
	}
	if len(collapse) > 0 {
		// rewire children's ParentInstID and prune session.InstanceIDs
		for _, inst := range rep.Instances {
			if newID, ok := collapse[inst.ParentInstID]; ok {
				inst.ParentInstID = newID
			}
		}
		for _, sess := range rep.Sessions {
			out := sess.InstanceIDs[:0]
			for _, iid := range sess.InstanceIDs {
				if _, gone := collapse[iid]; gone {
					continue
				}
				out = append(out, iid)
			}
			sess.InstanceIDs = out
		}
		// rebuild role counts
		rep.RoleInstanceCounts = map[string]int{}
		rep.Roles = map[string]int{}
		for _, inst := range rep.Instances {
			rep.RoleInstanceCounts[inst.PersonaLabel]++
			rep.Roles[inst.PersonaLabel] += inst.RequestCount
		}
	}

	// Pass 6: link children list and per-session parallel stats.
	for _, inst := range rep.Instances {
		if inst.ParentInstID != "" {
			if p, ok := rep.Instances[inst.ParentInstID]; ok {
				p.ChildrenInsts = append(p.ChildrenInsts, inst.ID)
			}
		}
	}
	// per-session parallel stats: group children by GroupKey
	for _, sess := range rep.Sessions {
		groups := map[string]int{}
		for _, iid := range sess.InstanceIDs {
			inst := rep.Instances[iid]
			if inst.GroupKey != "" {
				groups[inst.GroupKey]++
			}
		}
		for _, sz := range groups {
			if sz >= 2 {
				sess.FanOutTurns++
			}
			if sz > sess.MaxFanout {
				sess.MaxFanout = sz
			}
		}
	}

	rep.TotalSessions = len(rep.Sessions)
	rep.TotalInstances = len(rep.Instances)

	// User aggregate stats. Only meaningful when --user-prefix was on at
	// capture time; otherwise rep.Users is empty and we report nothing.
	if rep.UsersDetected {
		totalReqs, totalSess := 0, 0
		for _, u := range rep.Users {
			u.SessionCount = len(u.Sessions)
			// Tie-break by user ID (lexicographic) so the chosen top is
			// deterministic across runs even if two users tie on count.
			if u.RequestCount > rep.MaxRequestsPerUser ||
				(u.RequestCount == rep.MaxRequestsPerUser && (rep.TopUserByRequests == "" || u.ID < rep.TopUserByRequests)) {
				rep.MaxRequestsPerUser = u.RequestCount
				rep.TopUserByRequests = u.ID
			}
			if u.SessionCount > rep.MaxSessionsPerUser ||
				(u.SessionCount == rep.MaxSessionsPerUser && (rep.TopUserBySessions == "" || u.ID < rep.TopUserBySessions)) {
				rep.MaxSessionsPerUser = u.SessionCount
				rep.TopUserBySessions = u.ID
			}
			totalReqs += u.RequestCount
			totalSess += u.SessionCount
		}
		rep.TotalUsers = len(rep.Users)
		if rep.TotalUsers > 0 {
			rep.AvgRequestsPerUser = float64(totalReqs) / float64(rep.TotalUsers)
			rep.AvgSessionsPerUser = float64(totalSess) / float64(rep.TotalUsers)
		}
		// Count sessions/requests that arrived without a user tag — useful
		// when only some captures had --user-prefix enabled. Skip the
		// synthetic orphan bucket: it has no user by construction (multi-
		// user) so it would always inflate this counter.
		for _, s := range rep.Sessions {
			if s.Synthetic {
				continue
			}
			if s.User == "" {
				rep.SessionsWithoutUser++
				rep.RequestsWithoutUser += s.RequestCount
			}
		}
	}
	return rep
}

// ---- Text rendering ----------------------------------------------------

func (rep *treeReport) PrintText(w *os.File, perSession bool, topN, topUsers int, showCosts bool) {
	fmt.Fprintln(w, "Router capture tree analysis")
	fmt.Fprintf(w, "  files:     %d\n", len(rep.Files))
	fmt.Fprintf(w, "  records:   %d (raw=%d, redacted=%d)\n", rep.TotalRecords, rep.RawCount, rep.RedactedCount)
	if rep.ErrorRecordsDropped > 0 {
		fmt.Fprintf(w, "  errors dropped (non-2xx):     %d\n", rep.ErrorRecordsDropped)
	}
	fmt.Fprintf(w, "  range:     %s -> %s\n", rep.StartTs, rep.EndTs)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Aggregate")
	// Find the orphan session (if any) so we can report its request count
	// alongside the real-session count.
	var orphanReqs int
	if rep.OrphanSessions > 0 {
		if s, ok := rep.Sessions[OrphanSessionID]; ok {
			orphanReqs = s.RequestCount
		}
		fmt.Fprintf(w, "  sessions:                    %d  (%d real + 1 orphan bucket with %d reqs)\n",
			rep.TotalSessions, rep.TotalSessions-rep.OrphanSessions, orphanReqs)
	} else {
		fmt.Fprintf(w, "  sessions:                    %d\n", rep.TotalSessions)
	}
	fmt.Fprintf(w, "  agent instances total:       %d\n", rep.TotalInstances)
	fmt.Fprintf(w, "  requests w/ session-id:      %d / %d  (%.2f%%)\n",
		rep.RequestsWithSessionID, rep.TotalRecords, rep.SessionIDCoveragePct)
	if rep.PlainResponseRecords > 0 {
		fmt.Fprintf(w, "  plain (non-SSE) records:     %d  (excluded from tree analysis)\n",
			rep.PlainResponseRecords)
	}
	fmt.Fprintf(w, "  spawns observed:             %d\n", rep.TotalSpawns)
	fmt.Fprintf(w, "  spawns matched to a child:   %d  (%.1f%%)\n",
		rep.MatchedSpawns, percent(rep.MatchedSpawns, rep.TotalSpawns))
	if rep.TotalSpawnsOrphan > 0 {
		fmt.Fprintf(w, "    real sessions:             %d / %d  (%.1f%%)\n",
			rep.MatchedSpawnsReal, rep.TotalSpawnsReal,
			percent(rep.MatchedSpawnsReal, rep.TotalSpawnsReal))
		fmt.Fprintf(w, "    orphan bucket:             %d / %d  (%.1f%%)\n",
			rep.MatchedSpawnsOrphan, rep.TotalSpawnsOrphan,
			percent(rep.MatchedSpawnsOrphan, rep.TotalSpawnsOrphan))
	}
	fmt.Fprintf(w, "  fan-out turns (>=2 spawns):  %d\n", rep.FanOutTurns)
	fmt.Fprintf(w, "  max fan-out in one turn:     %d\n", rep.MaxFanoutSeen)
	if rep.UsersDetected {
		fmt.Fprintf(w, "  users:                       %d\n", rep.TotalUsers)
	} else {
		fmt.Fprintf(w, "  users:                       (no user metadata captured)\n")
	}
	fmt.Fprintln(w)
	if rep.UsersDetected {
		fmt.Fprintln(w, "Per-user stats")
		fmt.Fprintf(w, "  distinct users:              %d\n", rep.TotalUsers)
		fmt.Fprintf(w, "  max requests per user:       %d (%s)\n", rep.MaxRequestsPerUser, rep.TopUserByRequests)
		fmt.Fprintf(w, "  avg requests per user:       %.1f\n", rep.AvgRequestsPerUser)
		fmt.Fprintf(w, "  max sessions per user:       %d (%s)\n", rep.MaxSessionsPerUser, rep.TopUserBySessions)
		fmt.Fprintf(w, "  avg sessions per user:       %.1f\n", rep.AvgSessionsPerUser)
		if rep.SessionsWithoutUser > 0 {
			fmt.Fprintf(w, "  sessions w/o user tag:       %d  (%d reqs)\n",
				rep.SessionsWithoutUser, rep.RequestsWithoutUser)
		}
		fmt.Fprintln(w)

		// Top users by request volume. Governed by --top-users so a wide
		// user table doesn't force a wide session table (or vice versa).
		urows := make([]*userInfo, 0, len(rep.Users))
		for _, u := range rep.Users {
			urows = append(urows, u)
		}
		sort.Slice(urows, func(i, j int) bool {
			if urows[i].RequestCount != urows[j].RequestCount {
				return urows[i].RequestCount > urows[j].RequestCount
			}
			return urows[i].ID < urows[j].ID
		})
		uTop := topUsers
		if uTop <= 0 || uTop > len(urows) {
			uTop = len(urows)
		}
		fmt.Fprintf(w, "Top %d users by request volume\n", uTop)
		if showCosts {
			fmt.Fprintf(w, "  %-40s %10s %10s %14s %12s %8s %10s\n",
				"user", "reqs", "sessions", "in_tok", "out_tok", "cache%", "cost_usd")
		} else {
			fmt.Fprintf(w, "  %-40s %10s %10s %14s %12s %8s\n",
				"user", "reqs", "sessions", "in_tok", "out_tok", "cache%")
		}
		for i := 0; i < uTop; i++ {
			u := urows[i]
			cachePct := 0.0
			if u.InputTokens > 0 {
				cachePct = 100.0 * float64(u.CacheReadTokens) / float64(u.InputTokens)
			}
			if showCosts {
				fmt.Fprintf(w, "  %-40s %10d %10d %14d %12d %7.1f%% %10.2f\n",
					u.ID, u.RequestCount, u.SessionCount,
					u.InputTokens, u.OutputTokens, cachePct, u.Cost)
			} else {
				fmt.Fprintf(w, "  %-40s %10d %10d %14d %12d %7.1f%%\n",
					u.ID, u.RequestCount, u.SessionCount,
					u.InputTokens, u.OutputTokens, cachePct)
			}
		}
		if showCosts {
			fmt.Fprintf(w, "  total cost across all users: $%.2f\n", rep.TotalCost)
		}
		fmt.Fprintln(w)
	}

	// Unknown-model warning. Surface even without --show-costs so the user
	// knows the registry needs an entry. Sort by count desc for readability.
	if len(rep.UnknownModels) > 0 {
		type umRow struct {
			Name  string
			Count int
		}
		rows := make([]umRow, 0, len(rep.UnknownModels))
		for n, c := range rep.UnknownModels {
			rows = append(rows, umRow{n, c})
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].Count > rows[j].Count })
		fmt.Fprintln(w, "Unknown models (no pricing in registry — costs underreported)")
		for _, r := range rows {
			name := r.Name
			if name == "" {
				name = "(empty model_in field)"
			}
			fmt.Fprintf(w, "  %-40s %d records\n", name, r.Count)
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintln(w, "Per-role breakdown")
	roles := make([]string, 0, len(rep.RoleInstanceCounts))
	for r := range rep.RoleInstanceCounts {
		roles = append(roles, r)
	}
	sort.Slice(roles, func(i, j int) bool {
		return rep.RoleInstanceCounts[roles[i]] > rep.RoleInstanceCounts[roles[j]]
	})
	fmt.Fprintf(w, "  %-32s %10s %10s\n", "label", "instances", "requests")
	for _, r := range roles {
		fmt.Fprintf(w, "  %-32s %10d %10d\n", r, rep.RoleInstanceCounts[r], rep.Roles[r])
	}

	// Top sessions
	type sessRow struct {
		ID     string
		Reqs   int
		Insts  int
		FanOut int
		MaxFan int
	}
	rows := make([]sessRow, 0, len(rep.Sessions))
	for _, s := range rep.Sessions {
		if s.Synthetic {
			// Orphan bucket isn't a real session — skip it so it doesn't
			// crowd out real sessions in the leaderboard. It's still
			// reported in the Aggregate line with its request count.
			continue
		}
		rows = append(rows, sessRow{s.ID, s.RequestCount, len(s.InstanceIDs), s.FanOutTurns, s.MaxFanout})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Reqs > rows[j].Reqs })
	if topN <= 0 || topN > len(rows) {
		topN = len(rows)
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Top %d sessions by request volume\n", topN)
	fmt.Fprintf(w, "  %-40s %8s %8s %8s %8s\n", "session", "reqs", "insts", "fan-out", "max-fan")
	for i := 0; i < topN; i++ {
		r := rows[i]
		fmt.Fprintf(w, "  %-40s %8d %8d %8d %8d\n", r.ID, r.Reqs, r.Insts, r.FanOut, r.MaxFan)
	}

	if perSession {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Per-session trees")
		ids := make([]string, 0, len(rep.Sessions))
		for id := range rep.Sessions {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool {
			return rep.Sessions[ids[i]].FirstSeenTs.Before(rep.Sessions[ids[j]].FirstSeenTs)
		})
		for _, id := range ids {
			rep.printSessionTree(w, id)
		}
	}
}

func (rep *treeReport) printSessionTree(w *os.File, sid string) {
	sess := rep.Sessions[sid]
	fmt.Fprintf(w, "\nSession %s  (%d reqs, %d insts, %d fan-out turns, max fan-out %d)\n",
		sid, sess.RequestCount, len(sess.InstanceIDs), sess.FanOutTurns, sess.MaxFanout)
	// Roots first, recurse children.
	children := map[string][]string{}
	roots := []string{}
	for _, iid := range sess.InstanceIDs {
		inst := rep.Instances[iid]
		if inst.HasParent && inst.ParentInstID != "" {
			children[inst.ParentInstID] = append(children[inst.ParentInstID], iid)
		} else {
			roots = append(roots, iid)
		}
	}
	for _, kids := range children {
		sort.Slice(kids, func(i, j int) bool {
			return rep.Instances[kids[i]].FirstSeenTs.Before(rep.Instances[kids[j]].FirstSeenTs)
		})
	}
	sort.Slice(roots, func(i, j int) bool {
		return rep.Instances[roots[i]].FirstSeenTs.Before(rep.Instances[roots[j]].FirstSeenTs)
	})
	for _, id := range roots {
		rep.printInstanceLine(w, id, "", children, true)
	}
}

func (rep *treeReport) printInstanceLine(w *os.File, id, prefix string, children map[string][]string, isLast bool) {
	inst := rep.Instances[id]
	branch := "├─"
	contPrefix := "│  "
	if isLast {
		branch = "└─"
		contPrefix = "   "
	}
	groupTag := ""
	if inst.GroupSize >= 2 {
		groupTag = fmt.Sprintf(" [fan-out %d/%d]", inst.Position+1, inst.GroupSize)
	}
	if prefix == "" {
		fmt.Fprintf(w, "  [%s] reqs=%d  fp=%s%s\n", inst.PersonaLabel, inst.RequestCount, shortFP(inst.SeedHash), groupTag)
	} else {
		fmt.Fprintf(w, "  %s%s [%s] reqs=%d  fp=%s%s\n", prefix, branch, inst.PersonaLabel, inst.RequestCount, shortFP(inst.SeedHash), groupTag)
	}
	kids := children[id]
	for i, kid := range kids {
		newPrefix := prefix
		if prefix != "" {
			newPrefix = prefix + contPrefix
		} else {
			newPrefix = contPrefix
		}
		rep.printInstanceLine(w, kid, newPrefix, children, i == len(kids)-1)
	}
}

func shortFP(h string) string {
	if h == "" {
		return "(no seed hash)"
	}
	// strip "sha256:" prefix if present, keep first 8
	x := h
	if strings.HasPrefix(x, "sha256:") {
		x = x[7:]
	}
	if len(x) > 8 {
		x = x[:8]
	}
	return x
}

func percent(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return 100.0 * float64(n) / float64(d)
}
