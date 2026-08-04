package benchmark

// Tree-aware router replay. Drives sessions captured by the wekai router (see
// `wekai router replay-prepare`) preserving:
//
//   - per-session series boundary (each session = one replay-queue item)
//   - per-instance sequential turn ordering
//   - parent->child sequencing (child waits for the parent's spawn-bearing
//     request to finish before its own first turn)
//   - sibling fan-out concurrency (when a parent's response carried K Agent
//     tool_uses, the K children may run in parallel)
//   - per-request input length (bytes drawn from the embedded docs sized to
//     the original input_tokens)
//   - per-request output budget (max_tokens = original output_tokens)
//
// Concurrency model: the existing series-pool is reused as-is. Each series
// goroutine pulls one ReplaySession from the queue and runs the full session
// tree to completion via runRouterReplaySession; that function spins up one
// goroutine per agent instance, all gated by st.gate. So with 32 series and
// some sessions in a fan-out moment, the in-flight count can exceed 32 — the
// gate queues the surplus until slots free up. This matches the user-stated
// expectation (queue in the middle absorbs bursts).

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
)

// ---- File schema (mirrors wekai router replay-prepare output) ----
//
// replay-v3 is a JSONL file:
//   line 1: header (RouterReplayHeader)
//   line 2..N+1: one RouterReplaySession per line
//
// Each ReplayRequest embeds the full per-request structured spec
// (system_blocks, tools, messages) — so the replayer can materialize
// byte/token-faithful synthetic content keyed by each block's hash,
// preserving the original wire shape and the original tool-use / tool-
// result reference graph across turns.

// RouterReplayHeader is the first-line metadata of a replay-v3 file.
type RouterReplayHeader struct {
	Schema      string              `json:"_schema"` // "replay-v3"
	Name        string              `json:"name"`
	GeneratedAt string              `json:"generated_at"`
	Source      string              `json:"source"`
	Summary     RouterReplaySummary `json:"summary"`
}

type RouterReplaySummary struct {
	Sessions           int    `json:"sessions"`
	Instances          int    `json:"instances"`
	Requests           int    `json:"requests"`
	Spawns             int    `json:"spawns"`
	MatchedSpawns      int    `json:"matched_spawns"`
	FanOutTurns        int    `json:"fan_out_turns"`
	MaxFanOutInOneTurn int    `json:"max_fan_out_in_one_turn"`
	StartTs            string `json:"start_ts"`
	EndTs              string `json:"end_ts"`
}

type RouterReplaySession struct {
	SessionID string                 `json:"session_id"`
	StartTs   string                 `json:"start_ts"`
	Instances []RouterReplayInstance `json:"instances"`
}

type RouterReplayInstance struct {
	InstanceID           string                `json:"instance_id"`
	Role                 string                `json:"role"`
	ParentInstanceID     string                `json:"parent_instance_id,omitempty"`
	ParentSpawnRequestID uint64                `json:"parent_spawn_request_id,omitempty"`
	FanOutGroup          string                `json:"fan_out_group,omitempty"`
	FanOutSize           int                   `json:"fan_out_size,omitempty"`
	FanOutPosition       int                   `json:"fan_out_position,omitempty"`
	Requests             []RouterReplayRequest `json:"requests"`
}

type RouterReplayRequest struct {
	RequestID uint64 `json:"request_id"`
	Ts        string `json:"ts"`
	Model     string `json:"model"`
	Stream    bool   `json:"stream"`
	MaxTokens int    `json:"max_tokens"`

	InputTokens         int `json:"input_tokens"`
	PrefillTokens       int `json:"prefill_tokens"`
	CacheReadTokens     int `json:"cache_read_tokens"`
	CacheCreationTokens int `json:"cache_creation_tokens"`
	OutputTokens        int `json:"output_tokens"`

	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	TopK        *int            `json:"top_k,omitempty"`
	Thinking    json.RawMessage `json:"thinking,omitempty"`

	SystemBlocks []RouterReplaySystemBlock `json:"system_blocks,omitempty"`
	Tools        *RouterReplayToolsSpec    `json:"tools,omitempty"`
	Messages     []RouterReplayMessage     `json:"messages,omitempty"`

	StopReason        string `json:"stop_reason,omitempty"`
	UpstreamLatencyMs int64  `json:"upstream_latency_ms"`
	TotalMs           int64  `json:"total_ms"`
}

// RouterReplaySystemBlock mirrors the prep-side ReplaySystemBlock.
type RouterReplaySystemBlock struct {
	Type         string `json:"type"`
	Hash         string `json:"hash"`
	Bytes        int    `json:"bytes"`
	Tokens       int    `json:"tokens,omitempty"`
	CacheControl string `json:"cache_control,omitempty"`
}

// RouterReplayToolsSpec mirrors the prep-side ReplayToolsSpec.
type RouterReplayToolsSpec struct {
	Count  int    `json:"count"`
	Bytes  int    `json:"bytes"`
	Tokens int    `json:"tokens,omitempty"`
	Hash   string `json:"hash"`
}

// RouterReplayMessage mirrors the prep-side ReplayMessage.
type RouterReplayMessage struct {
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

// ---- Streaming reader + queue ----

// routerReplayStream reads a replay-v3 JSONL file line-by-line and serves
// sessions through a bounded channel. The producer goroutine reads exactly
// one session ahead of what consumers have pulled (channel cap = small),
// so memory stays bounded by chan_capacity * max_session_size — no matter
// how big the file gets.
//
// Only Pull, Total, and Remaining are externally visible; the producer
// goroutine is internal.
type routerReplayStream struct {
	header         RouterReplayHeader
	f              *os.File
	br             *bufio.Reader
	ch             chan RouterReplaySession
	idx            atomic.Int64 // monotonic pull counter
	total          int          // total sessions the producer will emit (capped by sessionLimit if set)
	produced       atomic.Int64 // how many sessions the producer has emitted so far
	limit          int          // 0 = no cap; >0 stop after this many sessions
	allowedIndices map[int]bool // nil = allow all; non-nil = only emit sessions at these 0-based line indices
	done           chan struct{}
	ctx            context.Context
	cancel         context.CancelFunc
}

// readRouterReplayHeader reads only line 1 (the header) and closes the
// file. Used at startup to surface a one-line summary before per-model
// streams are opened.
func readRouterReplayHeader(path string) (RouterReplayHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return RouterReplayHeader{}, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)
	line, err := br.ReadBytes('\n')
	if err != nil && len(line) == 0 {
		return RouterReplayHeader{}, fmt.Errorf("read header: %w", err)
	}
	line = trimNL(line)
	var hdr RouterReplayHeader
	if err := json.Unmarshal(line, &hdr); err != nil {
		return RouterReplayHeader{}, fmt.Errorf("parse header: %w", err)
	}
	if hdr.Schema != "replay-v3" {
		return RouterReplayHeader{}, fmt.Errorf("unsupported replay schema %q (expected replay-v3)", hdr.Schema)
	}
	return hdr, nil
}

// openRouterReplayStream parses the header line, then starts a producer
// goroutine that streams subsequent lines through ch (cap defines how
// many sessions can be buffered ahead of consumers). sessionLimit > 0
// caps the producer to that many sessions (useful for bounded runs that
// shouldn't continue past the first N sessions in the file).
// allowedIndices, when non-nil, restricts the producer to only emit
// sessions at the given 0-based line indices (after the header line);
// sessions at other positions are read and discarded. sessionLimit is
// applied independently and still stops the producer after N sessions
// from the beginning of the file regardless of allowedIndices.
func openRouterReplayStream(path string, chanCap, sessionLimit int, allowedIndices map[int]bool) (*routerReplayStream, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	// Use bufio.NewReader.ReadBytes('\n') because session lines can exceed
	// bufio.Scanner's default 64KB cap (largest in our captures: ~3.3 MB).
	br := bufio.NewReaderSize(f, 1<<20)
	headerLine, err := br.ReadBytes('\n')
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("read header line: %w", err)
	}
	headerLine = trimNL(headerLine)
	var hdr RouterReplayHeader
	if err := json.Unmarshal(headerLine, &hdr); err != nil {
		f.Close()
		return nil, fmt.Errorf("parse header: %w", err)
	}
	if hdr.Schema != "replay-v3" {
		f.Close()
		return nil, fmt.Errorf("unsupported replay schema %q (expected replay-v3)", hdr.Schema)
	}
	if chanCap < 1 {
		chanCap = 4
	}
	ctx, cancel := context.WithCancel(context.Background())
	total := hdr.Summary.Sessions
	if sessionLimit > 0 && sessionLimit < total {
		total = sessionLimit
	}
	// When an index filter is active, total is the number of matching indices
	// (capped by sessionLimit if set) — this is what the progress display uses.
	if len(allowedIndices) > 0 {
		total = len(allowedIndices)
		if sessionLimit > 0 && sessionLimit < total {
			total = sessionLimit
		}
	}
	s := &routerReplayStream{
		header:         hdr,
		f:              f,
		br:             br,
		ch:             make(chan RouterReplaySession, chanCap),
		total:          total,
		limit:          sessionLimit,
		allowedIndices: allowedIndices,
		done:           make(chan struct{}),
		ctx:            ctx,
		cancel:         cancel,
	}
	go s.produce()
	return s, nil
}

// produce reads session lines and pushes them into ch. Closes ch on EOF or
// when Stop() is called. Errors mid-stream are logged via stderr but don't
// surface to the consumer (the workers can still drain whatever was emitted
// before the error).
// When s.allowedIndices is non-nil, only lines whose 0-based position
// (after the header) appear in the set are emitted; others are read and
// discarded so the file position advances correctly.
func (s *routerReplayStream) produce() {
	defer close(s.done)
	defer close(s.ch)
	lineIdx := 0 // 0-based index of the current session line (header already consumed)
	for {
		if s.ctx.Err() != nil {
			return
		}
		if s.limit > 0 && s.produced.Load() >= int64(s.limit) {
			// session cap reached — stop reading further lines.
			return
		}
		// When filtering by index: stop once we've read past the highest
		// allowed index (no point scanning the rest of the file).
		if s.allowedIndices != nil && s.produced.Load() >= int64(len(s.allowedIndices)) {
			return
		}
		line, err := s.br.ReadBytes('\n')
		if len(line) > 0 {
			line = trimNL(line)
			if len(line) > 0 {
				currentIdx := lineIdx
				lineIdx++
				// Skip lines not in the allowed set (when a filter is active).
				if s.allowedIndices != nil && !s.allowedIndices[currentIdx] {
					if err != nil {
						return
					}
					continue
				}
				var sess RouterReplaySession
				if jerr := json.Unmarshal(line, &sess); jerr == nil {
					// Backpressure: blocks when ch is full, so the next
					// file read happens only after a worker pulls.
					// Respect ctx so Close() can unblock us when the
					// run is wrapping up and consumers stopped pulling.
					select {
					case <-s.ctx.Done():
						return
					case s.ch <- sess:
						s.produced.Add(1)
					}
				}
			}
		}
		if err != nil {
			return
		}
	}
}

// Pull blocks until a session is available, the stream is done, or ctx is
// cancelled. Returns (session, idx, true) on success, (zero, 0, false) on
// drain or cancel — caller can't tell the difference (and shouldn't need to:
// either way, the worker should exit its loop).
func (s *routerReplayStream) Pull(ctx context.Context) (RouterReplaySession, int, bool) {
	select {
	case <-ctx.Done():
		return RouterReplaySession{}, 0, false
	case sess, ok := <-s.ch:
		if !ok {
			return RouterReplaySession{}, 0, false
		}
		idx := int(s.idx.Add(1)) - 1
		return sess, idx, true
	}
}

// Total returns the session count from the header summary. May be 0 if the
// producer didn't populate it.
func (s *routerReplayStream) Total() int { return s.total }

// Remaining is an approximation: total minus the number we've handed out.
// It does NOT count sessions still buffered in the channel (those count as
// "remaining" for the consumer's purposes).
func (s *routerReplayStream) Remaining() int {
	r := s.total - int(s.idx.Load())
	if r < 0 {
		r = 0
	}
	return r
}

// Stop tells the producer to stop. Cancels the producer context, which
// also unblocks any send-to-channel that's waiting for a slow consumer.
// Existing buffered sessions remain pullable until the channel drains.
func (s *routerReplayStream) Stop() {
	s.cancel()
}

// Close releases the underlying file. Stops the producer first.
func (s *routerReplayStream) Close() error {
	s.Stop()
	<-s.done
	return s.f.Close()
}

func trimNL(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}

// ---- Pre-flight dependency-graph resolution ----

// Instance launch actions, decided up front from the parent-dependency graph.
const (
	instActWait = iota // block on the parent's done-channel before running
	instActFire        // run immediately: a true root, OR promoted out of a cycle
	instActDrop        // do not run: parent excluded by --router-replay-roles
)

// resolveInstanceActions classifies how each instance of a session must be
// launched. The replay file's parent links (ParentSpawnRequestID) can contain
// cycles or dangle; an instance only ever unblocks if its parent chain reaches
// a true root (ParentSpawnRequestID == 0) through instances that actually run.
// Any instance not provably anchored to such a root is "promoted" to fire
// immediately, breaking the cycle. Returns the per-instance action slice
// (indexed like `instances`) and the count of cycle-promoted instances.
func resolveInstanceActions(instances []RouterReplayInstance, includeRole func(string) bool) (actions []int, promoted int) {
	// reqOwnerAll maps every RequestID (!= 0) to its instance index, over ALL
	// instances (filtered or not). This distinguishes a role-excluded parent
	// from a genuinely missing one.
	reqOwnerAll := map[uint64]int{}
	for i := range instances {
		for _, r := range instances[i].Requests {
			if r.RequestID != 0 {
				reqOwnerAll[r.RequestID] = i
			}
		}
	}

	actions = make([]int, len(instances))
	anchored := make([]bool, len(instances))

	// First pass: classify each instance.
	for i := range instances {
		if !includeRole(instances[i].Role) {
			actions[i] = instActDrop
			continue
		}
		p := instances[i].ParentSpawnRequestID
		if p == 0 {
			actions[i] = instActFire
			anchored[i] = true
			continue
		}
		owner, ok := reqOwnerAll[p]
		if !ok {
			// Dangling parent (not owned by anyone).
			actions[i] = instActFire
			anchored[i] = true
			continue
		}
		if !includeRole(instances[owner].Role) {
			// Parent role-excluded.
			actions[i] = instActDrop
			continue
		}
		actions[i] = instActWait // provisional
	}

	// Fixpoint: propagate anchored status through the dependency graph.
	for {
		changed := false
		for i := range instances {
			if actions[i] != instActWait || anchored[i] {
				continue
			}
			owner := reqOwnerAll[instances[i].ParentSpawnRequestID]
			if anchored[owner] {
				anchored[i] = true
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	// Final pass: any remaining instActWait that isn't anchored is cycle-bound.
	for i := range instances {
		if actions[i] == instActWait && !anchored[i] {
			actions[i] = instActFire
			promoted++
		}
	}

	return actions, promoted
}

// ---- Series-loop body for tree replay ----

// endpointPicker chooses an endpoint PER REQUEST and accounts the in-flight
// slot for exactly that request's duration.
//
// Selection is per-request, not per-series or per-instance, because in-flight
// load is a per-request property: an instance runs its turns sequentially, so
// holding a slot for a whole instance measures "live agents", not "concurrent
// requests" — and the 429 cap we are avoiding counts the latter.
//
// Stickiness is preserved as a PREFERENCE: every request first tries its
// series' home endpoint, and only fails over when that endpoint is above the
// overload threshold. So the common case keeps prefix locality and only the
// overflow moves.
type endpointPicker struct {
	router *llm.EndpointRouter
	// posters holds one poster per endpoint index, mirroring router.Endpoints().
	// A poster latches its resolved URL form on first success, so a request
	// cannot simply retarget an existing one — it selects the right one.
	posters []*replayPoster
	// single is used when there is nothing to route between.
	single *replayPoster
}

// acquire selects the endpoint for one request and takes its in-flight slot.
// Returns the poster to use and the index to release afterwards (-1 = nothing
// to release).
func (p *endpointPicker) acquire(seriesNum int) (*replayPoster, int) {
	if p.router == nil || len(p.posters) == 0 {
		return p.single, -1
	}
	idx := p.router.AcquireForRequest(seriesNum)
	if idx < 0 || idx >= len(p.posters) || p.posters[idx] == nil {
		return p.single, -1
	}
	return p.posters[idx], idx
}

func (p *endpointPicker) release(idx int) {
	if p.router != nil && idx >= 0 {
		p.router.ReleaseIndex(idx)
	}
}

func runRouterReplaySeriesLoop(
	benchCtx context.Context,
	cfg AutoBenchmarkConfig,
	st *autoState,
	rdw *requestDataWriter,
	queue *routerReplayStream,
	picker endpointPicker,
	docs string,
	updateSnap func(*autoState),
	gate *concurrencyGate,
) {
	reqTimeout := cfg.RequestTimeout
	if reqTimeout == 0 {
		reqTimeout = 5 * time.Minute
	}

	for {
		select {
		case <-benchCtx.Done():
			return
		default:
		}
		if cfg.Total > 0 && st.totalEmitted.Load() >= int64(cfg.Total) {
			return
		}

		sess, idx, ok := queue.Pull(benchCtx)
		if !ok {
			return
		}
		seriesNum := idx + 1
		runRouterReplaySession(benchCtx, cfg, st, rdw,
			sess, seriesNum, picker, reqTimeout, docs, gate)
		// Session slot retired — drop its active-dataset snapshot.
		st.datasetTracker.Reset(seriesNum)
		st.seriesReplayCompleted.Add(1)
		updateSnap(st)
	}
}

// runRouterReplaySession runs all instances of one session, honoring
// parent->child sequencing and fan-out concurrency. Returns when every
// instance's request stream has finished or benchCtx is cancelled.
//
// Cancellation policy:
//   - benchCtx is the only signal that aborts in-flight work; it's
//     cancelled by the orchestrator at end of run.
//   - When an instance exits early (--total reached, error, ctx done),
//     its defer closes any of its own request done-channels that didn't
//     fire — that wakes up children waiting on those channels so they
//     can exit cleanly via their own --total checks. Each channel has a
//     single owner (the instance whose request id it is), so the close
//     is unambiguous.
//   - We do NOT cancel a session-level context on --total: that would
//     kill the in-flight HTTP request which already counted toward total.
func runRouterReplaySession(
	benchCtx context.Context,
	cfg AutoBenchmarkConfig,
	st *autoState,
	rdw *requestDataWriter,
	sess RouterReplaySession,
	seriesNum int,
	picker endpointPicker,
	reqTimeout time.Duration,
	docs string,
	gate *concurrencyGate,
) {
	includeRole := buildRoleFilter(cfg.RouterReplayRoles)

	// Build done-channels for every included request id. Helpers and
	// ephemerals carry an implicit parent_spawn_request_id baked in at
	// prepare time (see router replay-prepare), so the tree is complete:
	// no virtual-time inference needed at replay time.
	requestDone := map[uint64]chan struct{}{}
	for i := range sess.Instances {
		if !includeRole(sess.Instances[i].Role) {
			continue
		}
		for _, r := range sess.Instances[i].Requests {
			if r.RequestID != 0 {
				requestDone[r.RequestID] = make(chan struct{})
			}
		}
	}

	actions, promoted := resolveInstanceActions(sess.Instances, includeRole)
	if promoted > 0 {
		fmt.Fprintf(os.Stderr, "[router-replay] session %s: promoted %d instance(s) in a parent-dependency cycle to roots\n",
			sess.SessionID, promoted)
	}

	var wg sync.WaitGroup
	for i := range sess.Instances {
		inst := sess.Instances[i]
		act := actions[i]
		if act == instActDrop {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			switch act {
			case instActWait:
				ch := requestDone[inst.ParentSpawnRequestID]
				select {
				case <-benchCtx.Done():
					return
				case <-ch:
				}
			case instActDrop:
				return
			}
			runRouterReplayInstance(benchCtx, cfg, st, rdw,
				sess.SessionID, seriesNum, inst, picker, reqTimeout, docs, requestDone, gate)
		}()
	}
	wg.Wait()
}

// buildRoleFilter returns a predicate: empty list -> include everything;
// otherwise -> include only roles listed (case-sensitive, comma-separated).
// Common opt-in for a clean per-session benchmark: "main,sub-agent".
func buildRoleFilter(spec string) func(string) bool {
	if spec == "" {
		return func(string) bool { return true }
	}
	allowed := map[string]bool{}
	for _, r := range strings.Split(spec, ",") {
		r = strings.TrimSpace(r)
		if r != "" {
			allowed[r] = true
		}
	}
	return func(role string) bool { return allowed[role] }
}

// runRouterReplayInstance issues the requests of a single agent instance in
// order, gated by st.gate. After each request completes, signals its
// done-channel so any waiting children may start. On any kind of early exit
// (--total reached, ctx cancelled, etc.), a deferred sweep closes the
// done-channels of unfired requests so descendant instances waiting on
// those channels unblock — they'll then notice --total/ctx themselves and
// exit cleanly without deadlocking the wg.Wait at the session level.
//
// Wire shape: each request is rebuilt from its replay-v3 structured spec
// (system_blocks, tools, messages) into a byte/token-faithful Anthropic
// /v1/messages body, then POSTed directly. This bypasses llm.Chat — which
// builds the body itself from accumulated chat history and can't take a
// pre-built multi-turn conversation with tool_use / tool_result blocks.
func runRouterReplayInstance(
	ctx context.Context,
	cfg AutoBenchmarkConfig,
	st *autoState,
	rdw *requestDataWriter,
	sessionID string,
	seriesNum int,
	inst RouterReplayInstance,
	picker endpointPicker,
	reqTimeout time.Duration,
	docs string,
	requestDone map[uint64]chan struct{},
	gate *concurrencyGate,
) {
	defer func() {
		for _, r := range inst.Requests {
			if ch, ok := requestDone[r.RequestID]; ok {
				closeOnce(ch)
			}
		}
	}()

	replayRunID := ""
	if !cfg.ReplayNoStamp {
		replayRunID = cfg.RunID
	}
	// With a multi-endpoint spec the posters were built once per series loop and
	// live on the picker; this per-instance poster is only the single-endpoint
	// fallback. err is still checked so a bad model spec fails the same way.
	poster, err := newReplayPoster(cfg.Model, config.GetAPIKeys(), "", replayRunID, cfg.DryRun, cfg.DryRunColdTPS, cfg.DryRunWarmTPS, cfg.DryRunOutputTPS, st.estimator)
	if err == nil {
		poster.outputRatio = cfg.ReplayOutputRatio
		poster.forceOutput = cfg.ReplayForceOutput
	}
	if err != nil {
		// Configuration error — record one error per request in this
		// instance and exit. Future improvement: surface this once at
		// startup rather than per-instance.
		for ti, req := range inst.Requests {
			st.totalEmitted.Add(1)
			recordReplayRequest(cfg, st, rdw, RequestMetrics{
				RequestNum: int(st.totalCompleted.Load()) + 1,
				SeriesNum:  seriesNum,
				CycleNum:   ti + 1,
				SeriesGUID: sessionID + ":" + inst.InstanceID,
				Error:      err,
			}, ti == 0, new(time.Duration))
			if ch, ok := requestDone[req.RequestID]; ok {
				closeOnce(ch)
			}
		}
		return
	}

	isFirstRequest := true
	var coldStartTTFT time.Duration

	for ti, req := range inst.Requests {
		if ctx.Err() != nil {
			return
		}
		if cfg.Total > 0 && st.totalEmitted.Load() >= int64(cfg.Total) {
			return
		}
		st.totalEmitted.Add(1)

		if isFirstRequest {
			if gerr := gate.AcquireCold(ctx); gerr != nil {
				return
			}
		} else {
			if gerr := gate.Acquire(ctx); gerr != nil {
				return
			}
		}

		// Cap max_tokens if cfg.MaxOutputTokens is set; otherwise the
		// builder uses the original output_tokens.
		if cfg.MaxOutputTokens > 0 {
			eff := req.OutputTokens
			if eff <= 0 {
				eff = req.MaxTokens
			}
			if eff == 0 || eff > cfg.MaxOutputTokens {
				req.MaxTokens = cfg.MaxOutputTokens
				req.OutputTokens = 0 // force max_tokens path
			}
		}

		reqCtx, reqCancel := context.WithTimeout(ctx, reqTimeout)
		var metrics RequestMetrics
		// Per-request endpoint selection: prefer this series' home endpoint,
		// fail over only if it is overloaded. The slot is held for exactly this
		// request, so the counter tracks concurrent requests — the same thing
		// the server's capacity limit counts.
		reqPoster, epIdx := picker.acquire(seriesNum)
		if reqPoster == nil {
			reqPoster = poster
		}
		if reqPoster.dryRun {
			metrics = reqPoster.dryDo(reqCtx, req, docs, ti+1, sessionID, inst.InstanceID, seriesNum, st)
		} else {
			metrics = reqPoster.do(reqCtx, req, docs, ti+1, sessionID, inst.InstanceID, seriesNum, st)
		}
		picker.release(epIdx)
		reqCancel()
		gate.Release()

		if ch, ok := requestDone[req.RequestID]; ok {
			closeOnce(ch)
		}

		recordReplayRequest(cfg, st, rdw, metrics, isFirstRequest, &coldStartTTFT)
		isFirstRequest = false
	}
}

// closeOnce closes ch unless already closed. Each replay request's done
// channel has a single owner (the instance whose request id it is), so
// the select-default check is unambiguous in our use.
func closeOnce(ch chan struct{}) {
	select {
	case <-ch:
	default:
		close(ch)
	}
}

// BuildReplayRequestPrefix reconstructs the cacheable-prefix hash/token
// sequence for a single replay request: system blocks (skipping the tiny
// per-request billing header at index 0 when Bytes < 200), then tools, then
// messages — in cache order. Live replay and offline analysis share this
// single definition of "prefix".
func BuildReplayRequestPrefix(req RouterReplayRequest) (hashes []string, tokens []int) {
	for i, sb := range req.SystemBlocks {
		if i == 0 && sb.Bytes < 200 {
			continue
		}
		hashes = append(hashes, sb.Hash)
		tokens = append(tokens, sb.Tokens)
	}
	if req.Tools != nil && req.Tools.Hash != "" {
		hashes = append(hashes, req.Tools.Hash)
		tokens = append(tokens, req.Tools.Tokens)
	}
	for _, m := range req.Messages {
		hashes = append(hashes, m.Hash)
		tokens = append(tokens, m.Tokens)
	}
	return
}
