package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/weka/wekai/benchmark"
)

// RouterReplayPrepareCommand transforms a directory of redacted JSONL captures
// into one replay-friendly JSON file under ../replays/. The output bakes in
// every linkage the replayer needs (parent_spawn_request_id, fan-out grouping,
// per-request input/output token budgets) so the replayer doesn't re-resolve
// the tree at run time.
type RouterReplayPrepareCommand struct {
	Out  string `short:"o" long:"out" description:"Output file path. Default: <src>/../replays/replay-<earliest>-to-<latest>.json"`
	Name string `long:"name" description:"Embed this name in the file (default: derived from output path)"`
	// Since/Until bound the capture by request timestamp (RFC3339, inclusive),
	// so a corpus can be rebuilt from a directory that has grown since.
	Since string `long:"since" description:"Only records with ts >= this RFC3339 instant"`
	Until string `long:"until" description:"Only records with ts <= this RFC3339 instant"`
	// CompleteBy drops records still in flight at an instant: a capture synced
	// while requests were running holds only those already finished, so
	// rebuilding the same corpus from the finished file needs the same cut.
	CompleteBy string `long:"complete-by" description:"Only records whose response had finished (ts + total_ms) by this RFC3339 instant"`
	// Shards bounds memory: records are split by session into this many
	// temp files and built one shard at a time.
	Shards  int `long:"shards" default:"256" description:"Split the capture by session into this many temp shards; peak memory is about one shard per worker"`
	Workers int `long:"workers" description:"Shards parsed in parallel (default: number of CPUs)"`
	Args    struct {
		Src string `positional-arg-name:"src" description:"Directory of redacted *.jsonl files (or one .jsonl). Default: ~/.wekai/router/capture/redacted/"`
	} `positional-args:"yes"`
}

// ReplayHeader is the first line of a replay JSONL file (schema replay-v5;
// readers also accept replay-v4, which has no index trailer, and replay-v3,
// which also has no think_ms / think_ref). The following lines are one
// ReplaySession per line — this lets producers stream output and consumers
// stream input with a bounded queue, instead of holding the full file (and its
// decoded objects) in memory.
//
// v5 (vs v4) ends with one more line, the session index (benchmark.ReplayIndex):
// each session's byte offset and length, start/end time and request count, so a
// reader can seek to any session and play the corpus newest-first.
//
// v4 (vs v3) adds per-request think_ms / think_ref and a header timing
// summary: see replay_think.go. upstream_latency_ms and total_ms are now
// filled from the capture (v3 files carry them as 0).
//
// v3 (vs v2) embeds per-request structured spec — system_blocks, tools,
// messages — so the replayer can reconstruct a byte/token-faithful HTTP
// body at run time, with synthetic content generated deterministically
// from each block's hash (same hash → same content → server prefix-cache
// hits the same way the original capture did).
type ReplayHeader struct {
	Schema      string        `json:"_schema"` // "replay-v5"
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

	Timing *ReplayTimingSummary `json:"timing,omitempty"`
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

	// ThinkMs is the captured gap between the end of the reference named by
	// ThinkRef and the start of this request, clamped at 0. Absent when it
	// could not be computed. See replay_think.go.
	ThinkMs  *int64 `json:"think_ms,omitempty"`
	ThinkRef string `json:"think_ref,omitempty"`
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
	// A shard's records are held parsed in memory; keep GC headroom from
	// pushing a large one past physical RAM.
	setSoftMemoryLimit(0.75)

	// The output directory is known up front (the default file name needs
	// the corpus time range, known only at the end); shards and the session
	// spool live there so they share the output's disk.
	dstDir := filepath.Dir(c.Out)
	if c.Out == "" {
		base := filepath.Dir(strings.TrimRight(src, string(filepath.Separator)))
		dstDir = filepath.Join(base, "replays")
		if err := os.MkdirAll(dstDir, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dstDir, err)
		}
	}
	work, err := os.MkdirTemp(dstDir, ".replay-prepare-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)

	// Pass 1: split the capture by session. Everything replay-prepare links
	// (spawns, instances, implicit parents, think times) is scoped to one
	// session, so a shard holding whole sessions can be built on its own, and
	// only one shard is in memory at a time. A session resumed days later
	// hashes to the same shard as its earlier records.
	shards, orphans, err := shardBySession(files, work, c.Shards)
	if err != nil {
		return err
	}

	// Pass 2: persona labels depend on whether a persona spawns, or is
	// spawned, anywhere in the corpus — the one cross-session fact. Collect
	// it from every shard (a few entries per distinct system prompt).
	workers := c.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	persona := newPersonaFlags()
	err = eachShardOrdered(shards, workers,
		func(path string) (*personaFlags, error) {
			records, err := c.loadShard(path)
			if err != nil || len(records) == 0 {
				return nil, err
			}
			return buildTreeReport(records).persona, nil
		},
		func(f *personaFlags) error {
			if f != nil {
				persona.merge(f)
			}
			return nil
		})
	if err != nil {
		return err
	}

	// Pass 3: build each shard's sessions against the corpus-wide flags and
	// spool their lines; the header carries totals known only once every
	// shard is done.
	spool, err := os.CreateTemp(work, "sessions-")
	if err != nil {
		return err
	}
	defer spool.Close()
	b := &replayBuilder{sw: bufio.NewWriterSize(spool, 1<<20), think: newThinkAccumulator(), orphans: orphans, persona: persona}
	// Shards parse in parallel; sessions are built in shard order so the
	// output does not depend on scheduling.
	err = eachShardOrdered(shards, workers,
		func(path string) ([]treeRecord, error) {
			records, err := c.loadShard(path)
			if err != nil {
				return nil, err
			}
			return records, os.Remove(path)
		},
		b.addRecords)
	if err != nil {
		return err
	}
	if err := b.sw.Flush(); err != nil {
		return err
	}
	if b.records == 0 {
		return fmt.Errorf("no records found in %s", src)
	}

	dst := c.Out
	if dst == "" {
		dst = filepath.Join(dstDir, fmt.Sprintf("replay-%s-to-%s.jsonl",
			ymd(b.startTs.Format(time.RFC3339)), ymd(b.endTs.Format(time.RFC3339))))
	}
	name := c.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(dst), filepath.Ext(dst))
	}
	b.summary.StartTs = b.startTs.Format(time.RFC3339)
	b.summary.EndTs = b.endTs.Format(time.RFC3339)
	b.summary.Timing = b.think.summary()
	header := ReplayHeader{
		Schema:      "replay-v5",
		Name:        name,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		// Base name only: replay files get published as artifacts, and a full
		// path would leak the producer's local filesystem layout.
		Source:  filepath.Base(strings.TrimRight(src, "/")),
		Summary: b.summary,
	}
	if err := b.write(dst, &header, spool); err != nil {
		return err
	}
	fmt.Printf("wrote %d sessions / %d instances / %d requests / %d repaired_parents / %d cycles_broken / %d orphan records dropped / %d unplaced records -> %s\n",
		len(b.entries), b.instances, b.summary.Requests, b.summary.RepairedParents, b.summary.CyclesBroken, b.orphans, b.unplaced, dst)
	return nil
}

// loadShard parses one shard and applies the time filters.
func (c *RouterReplayPrepareCommand) loadShard(path string) ([]treeRecord, error) {
	// Fresh interner per shard: sessions never span shards, so its tables
	// would only grow across them.
	records, err := parseTreeFile(path, newTreeInterner())
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return filterRecordsByTime(records, c.Since, c.Until, c.CompleteBy)
}

// eachShardOrdered runs work on every shard with up to workers in flight and
// hands the results to consume in shard order. A slot frees only once its
// result is consumed, so at most workers results are held at a time.
func eachShardOrdered[T any](shards []string, workers int, work func(string) (T, error), consume func(T) error) error {
	type result struct {
		v   T
		err error
	}
	results := make([]chan result, len(shards))
	for i := range results {
		results[i] = make(chan result, 1)
	}
	slots := make(chan struct{}, max(workers, 1))
	done := make(chan struct{})
	defer close(done)
	go func() {
		for i, path := range shards {
			select {
			case slots <- struct{}{}:
			case <-done:
				return
			}
			go func() {
				v, err := work(path)
				results[i] <- result{v, err}
			}()
		}
	}()
	for i := range shards {
		r := <-results[i]
		if r.err != nil {
			return r.err
		}
		err := consume(r.v)
		var zero T
		r.v = zero
		<-slots
		if err != nil {
			return err
		}
	}
	return nil
}

// shardBySession copies every POST 2xx record line of files into one of n
// shard files under dir, picked by a hash of the record's session id, and
// returns the shard paths and how many records were dropped for having no
// session id. Records with no session id are dropped: they would
// form one synthetic session spanning the whole capture, which replays as an
// unrealistic single stream and is the one session that does not shard.
func shardBySession(files []string, dir string, n int) ([]string, int, error) {
	if n <= 0 {
		n = 1
	}
	paths := make([]string, n)
	outs := make([]*os.File, n)
	ws := make([]*bufio.Writer, n)
	defer func() {
		for _, f := range outs {
			if f != nil {
				f.Close()
			}
		}
	}()
	for i := range paths {
		paths[i] = filepath.Join(dir, fmt.Sprintf("shard-%04d.jsonl", i))
		f, err := os.Create(paths[i])
		if err != nil {
			return nil, 0, err
		}
		outs[i] = f
		ws[i] = bufio.NewWriterSize(f, 256<<10)
	}
	// Only the fields that route a line; the line itself is copied verbatim
	// and fully parsed in pass 2.
	var probe struct {
		Method  string `json:"method"`
		Status  int    `json:"status"`
		Request struct {
			Headers http.Header `json:"headers"`
		} `json:"request"`
	}
	orphans := 0
	for _, path := range files {
		f, err := os.Open(path)
		if err != nil {
			return nil, 0, err
		}
		br := bufio.NewReaderSize(f, 1<<20)
		for {
			line, readErr := br.ReadBytes('\n')
			if len(line) > 1 {
				probe.Method, probe.Status, probe.Request.Headers = "", 0, nil
				// Same skips as parseLine, so dropped lines never reach disk.
				if json.Unmarshal(line, &probe) == nil && probe.Method == "POST" &&
					probe.Status >= 200 && probe.Status < 300 {
					if sid := probe.Request.Headers.Get("X-Claude-Code-Session-Id"); sid == "" {
						orphans++
					} else {
						h := fnv.New32a()
						h.Write([]byte(sid))
						w := ws[h.Sum32()%uint32(n)]
						if _, err := w.Write(line); err != nil {
							f.Close()
							return nil, 0, err
						}
						if line[len(line)-1] != '\n' {
							w.WriteByte('\n')
						}
					}
				}
			}
			if readErr == io.EOF {
				break
			}
			if readErr != nil {
				f.Close()
				return nil, 0, fmt.Errorf("read %s: %w", path, readErr)
			}
		}
		f.Close()
	}
	for i, w := range ws {
		if err := w.Flush(); err != nil {
			return nil, 0, err
		}
		if err := outs[i].Close(); err != nil {
			return nil, 0, err
		}
		outs[i] = nil
	}
	return paths, orphans, nil
}

// replayBuilder accumulates replay sessions shard by shard: session lines go
// to the spool, and only the index entries, totals and think-time samples stay
// in memory.
type replayBuilder struct {
	sw       *bufio.Writer
	spoolLen int64
	entries  []benchmark.ReplayIndexSession // Offset is into the spool until write
	starts   []time.Time                    // session first-seen, for output order
	think    *thinkAccumulator
	summary  ReplaySummary
	persona  *personaFlags // corpus-wide, see personaFlags

	records        int
	instances      int
	orphans        int
	unplaced       int // records whose instance the tree no longer has
	startTs, endTs time.Time
}

// addRecords builds and spools the sessions of records, which must hold every
// record of each session they touch.
func (b *replayBuilder) addRecords(records []treeRecord) error {
	if len(records) == 0 {
		return nil
	}
	// Reuse the tree command's tree builder so the linkage logic is
	// single-sourced.
	rep := buildTreeReportWith(records, b.persona) // sorts records by ts
	b.records += len(records)
	if b.startTs.IsZero() || records[0].Ts.Before(b.startTs) {
		b.startTs = records[0].Ts
	}
	if last := records[len(records)-1].Ts; last.After(b.endTs) {
		b.endTs = last
	}
	b.summary.Sessions += rep.TotalSessions
	b.summary.Instances += rep.TotalInstances
	b.summary.Spawns += rep.TotalSpawns
	b.summary.MatchedSpawns += rep.MatchedSpawns
	b.summary.FanOutTurns += rep.FanOutTurns
	b.summary.MaxFanOutInOneTurn = max(b.summary.MaxFanOutInOneTurn, rep.MaxFanoutSeen)

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
		// May have been collapsed onto a canonical main instance, which can
		// carry a different persona (a session's main agent changes system
		// prompt mid-way): follow the tree's own collapse map. Persona
		// best-match is only a fallback.
		if _, ok := rep.Instances[key]; !ok {
			if canon, ok := rep.collapsed[key]; ok {
				key = canon
			} else if key = canonicalInstanceKey(rep, sid, sig); key == "" {
				b.unplaced++
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

	sessIDs := make([]string, 0, len(rep.Sessions))
	for sid := range rep.Sessions {
		sessIDs = append(sessIDs, sid)
	}
	sort.Slice(sessIDs, func(i, j int) bool {
		return rep.Sessions[sessIDs[i]].FirstSeenTs.Before(rep.Sessions[sessIDs[j]].FirstSeenTs)
	})

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
					b.summary.Requests++
				}
				// Each record feeds exactly one instance, and only its
				// ID/Ts are read after this point; drop the bodies so
				// memory shrinks as sessions are written.
				r.Req = redactedRequest{}
				r.Resp = redactedResponse{}
			}
			if len(ri.Requests) == 0 {
				continue
			}
			rs.Instances = append(rs.Instances, ri)
			b.instances++
		}
		if len(rs.Instances) == 0 {
			continue
		}
		b.summary.RepairedParents += repairDanglingParents(&rs)
		cyclesBroken := breakParentCycles(&rs)
		b.summary.CyclesBroken += cyclesBroken
		if cyclesBroken > 0 {
			fmt.Fprintf(os.Stderr, "replay-prepare: broke %d parent-dependency cycle(s) (instances promoted to roots)\n", cyclesBroken)
		}
		annotateThink(&rs, b.think)
		// Compact JSON encoding (no indent) so each session sits on one
		// line. The header line uses the same encoding for consistency.
		line, err := json.Marshal(&rs)
		if err != nil {
			return fmt.Errorf("marshal session %s: %w", sid, err)
		}
		e := sessionIndexEntry(&rs, int64(len(line)))
		e.Offset = b.spoolLen
		b.entries = append(b.entries, e)
		b.starts = append(b.starts, sess.FirstSeenTs)
		if _, err := b.sw.Write(line); err != nil {
			return err
		}
		if err := b.sw.WriteByte('\n'); err != nil {
			return err
		}
		b.spoolLen += int64(len(line)) + 1
	}
	return nil
}

// write emits the replay file: header on line 1, the spooled session lines in
// first-seen order (shards emit them interleaved in time), and the session
// index (replay-v5) last, once the offsets are known; readers find it by
// reading the file's tail.
func (b *replayBuilder) write(dst string, header *ReplayHeader, spool *os.File) error {
	order := make([]int, len(b.entries))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, c := order[i], order[j]
		if !b.starts[a].Equal(b.starts[c]) {
			return b.starts[a].Before(b.starts[c])
		}
		return b.entries[a].SessionID < b.entries[c].SessionID
	})

	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	bw := bufio.NewWriterSize(f, 1<<20)
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	if _, err := bw.Write(hb); err != nil {
		return err
	}
	if err := bw.WriteByte('\n'); err != nil {
		return err
	}
	offset := int64(len(hb)) + 1
	index := make([]benchmark.ReplayIndexSession, 0, len(order))
	for _, i := range order {
		e := b.entries[i]
		n := e.Length + 1 // with its newline
		if _, err := io.Copy(bw, io.NewSectionReader(spool, e.Offset, n)); err != nil {
			return fmt.Errorf("copy session %s: %w", e.SessionID, err)
		}
		e.Offset = offset
		offset += n
		index = append(index, e)
	}
	ib, err := json.Marshal(&benchmark.ReplayIndex{Schema: benchmark.ReplayIndexSchema, Sessions: index})
	if err != nil {
		return err
	}
	if _, err := bw.Write(ib); err != nil {
		return err
	}
	if err := bw.WriteByte('\n'); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return f.Close()
}

// sessionIndexEntry describes one session line for the index trailer; the
// writer fills in Offset. EndTs is when the session's last request finished in
// the capture (ts + total_ms), falling back to its last start when a request
// has no recorded duration.
func sessionIndexEntry(rs *ReplaySession, length int64) benchmark.ReplayIndexSession {
	e := benchmark.ReplayIndexSession{SessionID: rs.SessionID, Length: length, StartTs: rs.StartTs}
	var end time.Time
	for _, inst := range rs.Instances {
		for _, r := range inst.Requests {
			e.Requests++
			t, err := time.Parse(time.RFC3339Nano, r.Ts)
			if err != nil {
				continue
			}
			if fin := t.Add(time.Duration(r.TotalMs) * time.Millisecond); fin.After(end) {
				end = fin
			}
		}
	}
	if !end.IsZero() {
		e.EndTs = end.UTC().Format(time.RFC3339Nano)
	}
	return e
}

// parseTreeFile is a thin wrapper around RouterTreeCommand.parseFile so we
// don't duplicate the raw/redacted detection logic here.
func parseTreeFile(path string, intern *treeInterner) ([]treeRecord, error) {
	c := &RouterTreeCommand{intern: intern}
	recs, _, _, _, err := c.parseFile(path)
	return recs, err
}

// filterRecordsByTime keeps records with since <= ts <= until whose response
// finished (ts + total_ms) by completeBy; an empty bound is open.
func filterRecordsByTime(recs []treeRecord, since, until, completeBy string) ([]treeRecord, error) {
	if since == "" && until == "" && completeBy == "" {
		return recs, nil
	}
	var lo, hi, done time.Time
	var err error
	if since != "" {
		if lo, err = time.Parse(time.RFC3339Nano, since); err != nil {
			return nil, fmt.Errorf("--since: %w", err)
		}
	}
	if until != "" {
		if hi, err = time.Parse(time.RFC3339Nano, until); err != nil {
			return nil, fmt.Errorf("--until: %w", err)
		}
	}
	if completeBy != "" {
		if done, err = time.Parse(time.RFC3339Nano, completeBy); err != nil {
			return nil, fmt.Errorf("--complete-by: %w", err)
		}
	}
	out := recs[:0]
	for _, r := range recs {
		if completeBy != "" && r.Ts.Add(time.Duration(r.TotalMs)*time.Millisecond).After(done) {
			continue
		}
		if since != "" && r.Ts.Before(lo) {
			continue
		}
		if until != "" && r.Ts.After(hi) {
			continue
		}
		out = append(out, r)
	}
	return out, nil
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

		UpstreamLatencyMs: r.UpstreamLatencyMs,
		TotalMs:           r.TotalMs,
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
