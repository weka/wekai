package cli

// Pure, offline unit tests for the replay-v3 -> AIPerf weka_trace converter
// (command_router_convert_aiperf_weka_trace.go). No mocked LLM/Chat flows —
// these exercise convertSessionToWekaTrace, the data-transformation function,
// directly, per repo testing policy.

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/weka/wekai/benchmark"
)

// ts returns an RFC3339Nano timestamp offsetSeconds after a fixed epoch, so
// tests can express request ordering as plain numbers.
func ts(offsetSeconds float64) string {
	base := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	return base.Add(time.Duration(offsetSeconds * float64(time.Second))).Format(time.RFC3339Nano)
}

func TestConvertSessionToWekaTrace_Simple(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess1",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 100, OutputTokens: 10, StopReason: "end_turn"},
					{RequestID: 2, Ts: ts(1), Model: "m1", InputTokens: 150, OutputTokens: 20, StopReason: "end_turn"},
				},
			},
		},
	}

	trace, stats, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if stats.EphemeralDropped != 0 || stats.RootFallbackUsed || stats.MultiLevelFlattened != 0 || stats.DanglingAnchorFallback != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if trace.ID != "sess1" {
		t.Fatalf("id = %q, want sess1", trace.ID)
	}
	if trace.HashIDScope != "local" {
		t.Fatalf("hash_id_scope = %q, want local", trace.HashIDScope)
	}
	if trace.BlockSize != 64 {
		t.Fatalf("block_size = %d, want 64", trace.BlockSize)
	}
	if len(trace.Requests) != 2 {
		t.Fatalf("expected 2 top-level requests, got %d", len(trace.Requests))
	}

	r0, ok := trace.Requests[0].(wekaRequest)
	if !ok {
		t.Fatalf("requests[0] is %T, want wekaRequest", trace.Requests[0])
	}
	if r0.T != 0 || r0.Type != "n" || r0.Model != "m1" || r0.InputLength != 100 || r0.OutputLength != 10 || r0.Stop != "end_turn" {
		t.Fatalf("unexpected r0: %+v", r0)
	}
	if len(r0.HashIDs) != 0 {
		t.Fatalf("r0 hash_ids = %v, want empty (no system blocks or messages)", r0.HashIDs)
	}

	r1, ok := trace.Requests[1].(wekaRequest)
	if !ok {
		t.Fatalf("requests[1] is %T, want wekaRequest", trace.Requests[1])
	}
	if r1.T != 1 {
		t.Fatalf("r1.T = %v, want 1", r1.T)
	}
}

// TestConvertSessionToWekaTrace_APITimeOmittedWhenUnmeasured verifies the
// nil-vs-zero distinction for api_time/think_time survives JSON marshaling:
// a request with TotalMs==0 (confirmed always-zero in the real corpus) must
// omit api_time entirely rather than claim a measured 0.0, and the first
// request in a chain has no predecessor to compute think_time from. ttft is
// never populated (no source signal).
func TestConvertSessionToWekaTrace_APITimeOmittedWhenUnmeasured(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess_apitime",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	b, err := json.Marshal(trace.Requests[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := m["api_time"]; ok {
		t.Fatalf("api_time should be omitted when TotalMs==0, got %v", m)
	}
	if _, ok := m["think_time"]; ok {
		t.Fatalf("think_time should be omitted for the first request in a chain, got %v", m)
	}
	if _, ok := m["ttft"]; ok {
		t.Fatalf("ttft should always be omitted (no source signal), got %v", m)
	}
}

func TestConvertSessionToWekaTrace_SubagentAnchoredAtRootRequest(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess2",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "tool_use"},
					{RequestID: 3, Ts: ts(2), Model: "m1", InputTokens: 20, OutputTokens: 2, StopReason: "end_turn"},
				},
			},
			{
				InstanceID:           "sub",
				Role:                 "sub-agent",
				ParentSpawnRequestID: 1, // root's first request
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 2, Ts: ts(1), Model: "m2", InputTokens: 5, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
		},
	}

	trace, stats, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if stats.MultiLevelFlattened != 0 || stats.DanglingAnchorFallback != 0 || stats.RootFallbackUsed {
		t.Fatalf("unexpected fallback stats: %+v", stats)
	}
	if len(trace.Requests) != 3 {
		t.Fatalf("expected 3 top-level entries (root req, subagent, root req), got %d", len(trace.Requests))
	}
	if _, ok := trace.Requests[0].(wekaRequest); !ok {
		t.Fatalf("requests[0] should be a normal request, got %T", trace.Requests[0])
	}
	sub, ok := trace.Requests[1].(wekaSubagentEntry)
	if !ok {
		t.Fatalf("requests[1] should be a subagent entry, got %T", trace.Requests[1])
	}
	if sub.AgentID != "agent_001" || sub.SubagentType != "sub-agent" || sub.Status != "completed" {
		t.Fatalf("unexpected subagent entry: %+v", sub)
	}
	if sub.T != 1 {
		t.Fatalf("sub.T = %v, want 1", sub.T)
	}
	if len(sub.Requests) != 1 {
		t.Fatalf("expected 1 inner request, got %d", len(sub.Requests))
	}
	if _, ok := trace.Requests[2].(wekaRequest); !ok {
		t.Fatalf("requests[2] should be a normal request, got %T", trace.Requests[2])
	}
}

// TestConvertSessionToWekaTrace_MultiLevelFlattening covers an instance whose
// ParentSpawnRequestID points at a request owned by ANOTHER non-root
// instance (B), not the root. It must be walked up to B's own anchor and
// flattened to a root-anchored sibling, since WekaSubagentEntry.requests has
// no nested-subagent variant.
func TestConvertSessionToWekaTrace_MultiLevelFlattening(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess3",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "tool_use"},
					{RequestID: 4, Ts: ts(3), Model: "m1", InputTokens: 30, OutputTokens: 3, StopReason: "end_turn"},
				},
			},
			{
				InstanceID:           "B",
				Role:                 "sub-agent",
				ParentSpawnRequestID: 1, // root's first request
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 2, Ts: ts(1), Model: "m2", InputTokens: 5, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
			{
				InstanceID:           "C",
				Role:                 "sub-agent",
				ParentSpawnRequestID: 2, // owned by B, NOT the root
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 3, Ts: ts(1.5), Model: "m3", InputTokens: 7, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
		},
	}

	trace, stats, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if stats.MultiLevelFlattened != 1 {
		t.Fatalf("expected 1 multi-level flattening, got %d (stats=%+v)", stats.MultiLevelFlattened, stats)
	}
	if stats.DanglingAnchorFallback != 0 {
		t.Fatalf("C should resolve via B, not fall back to dangling: stats=%+v", stats)
	}
	if len(trace.Requests) != 4 {
		t.Fatalf("expected 4 top-level entries, got %d", len(trace.Requests))
	}
	b, ok := trace.Requests[1].(wekaSubagentEntry)
	if !ok {
		t.Fatalf("requests[1] should be subagent B, got %T", trace.Requests[1])
	}
	c, ok := trace.Requests[2].(wekaSubagentEntry)
	if !ok {
		t.Fatalf("requests[2] should be subagent C, got %T", trace.Requests[2])
	}
	if b.AgentID != "agent_001" || c.AgentID != "agent_002" {
		t.Fatalf("unexpected agent ids: b=%s c=%s (both should anchor at root's first request, ordered by own start time)", b.AgentID, c.AgentID)
	}
	if _, ok := trace.Requests[3].(wekaRequest); !ok {
		t.Fatalf("requests[3] should be root's second request, got %T", trace.Requests[3])
	}
}

func TestConvertSessionToWekaTrace_EphemeralProbeDroppedByDefault(t *testing.T) {
	mkSess := func() benchmark.RouterReplaySession {
		return benchmark.RouterReplaySession{
			SessionID: "sess4",
			StartTs:   ts(0),
			Instances: []benchmark.RouterReplayInstance{
				{
					InstanceID: "root",
					Role:       "main",
					Requests: []benchmark.RouterReplayRequest{
						{RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "end_turn"},
					},
				},
				{
					InstanceID: "probe",
					Role:       "ephemeral (no system)",
					Requests: []benchmark.RouterReplayRequest{
						{RequestID: 2, Ts: ts(0.5), Model: "m1", InputTokens: 8, OutputTokens: 1, MaxTokens: 1, StopReason: "max_tokens"},
					},
				},
			},
		}
	}

	trace, stats, err := convertSessionToWekaTrace(mkSess(), convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if stats.EphemeralDropped != 1 {
		t.Fatalf("expected 1 ephemeral probe dropped, got %d", stats.EphemeralDropped)
	}
	if len(trace.Requests) != 1 {
		t.Fatalf("expected the ephemeral instance excluded from output, got %d top-level entries", len(trace.Requests))
	}

	trace2, stats2, err := convertSessionToWekaTrace(mkSess(), convertOptions{BlockSize: 64, IncludeEphemeralProbes: true})
	if err != nil {
		t.Fatalf("convert (include-ephemeral): %v", err)
	}
	if stats2.EphemeralDropped != 0 {
		t.Fatalf("expected 0 dropped when --include-ephemeral-probes is set, got %d", stats2.EphemeralDropped)
	}
	if len(trace2.Requests) != 2 {
		t.Fatalf("expected the ephemeral instance retained as a subagent entry, got %d entries", len(trace2.Requests))
	}
	found := false
	for _, r := range trace2.Requests {
		if e, ok := r.(wekaSubagentEntry); ok && e.SubagentType == "ephemeral (no system)" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected an ephemeral-no-system subagent entry when included, got %+v", trace2.Requests)
	}
}

// TestConvertSessionToWekaTrace_HashIDReuse exercises the growing-history
// property of RouterReplayRequest.Messages: the same Hash recurring across
// requests (a resent system prompt / earlier turn) must reuse its
// previously-minted block ids exactly, while a genuinely new hash mints
// fresh, non-overlapping ids.
func TestConvertSessionToWekaTrace_HashIDReuse(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess5",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{
						RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "end_turn",
						Messages: []benchmark.RouterReplayMessage{
							{Role: "user", Hash: "H1", Tokens: 10, BlockTypes: []string{"text"}},
						},
					},
					{
						// Full growing history: H1 resent verbatim, plus a new H2.
						RequestID: 2, Ts: ts(1), Model: "m1", InputTokens: 220, OutputTokens: 2, StopReason: "end_turn",
						Messages: []benchmark.RouterReplayMessage{
							{Role: "user", Hash: "H1", Tokens: 10, BlockTypes: []string{"text"}},
							{Role: "assistant", Hash: "H2", Tokens: 200, BlockTypes: []string{"text"}},
						},
					},
				},
			},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	r0 := trace.Requests[0].(wekaRequest)
	r1 := trace.Requests[1].(wekaRequest)

	if len(r0.HashIDs) != 1 { // ceil(10/64) = 1
		t.Fatalf("r0 hash_ids = %v, want 1 id", r0.HashIDs)
	}
	if len(r1.HashIDs) != 5 { // H1 reused (1) + H2 fresh (ceil(200/64)=4)
		t.Fatalf("r1 hash_ids = %v, want 5 ids", r1.HashIDs)
	}
	if r1.HashIDs[0] != r0.HashIDs[0] {
		t.Fatalf("expected H1's id reused identically: r0=%v r1[0]=%v", r0.HashIDs, r1.HashIDs[0])
	}
	for _, id := range r1.HashIDs[1:] {
		if id == r0.HashIDs[0] {
			t.Fatalf("H2's freshly-minted ids must not collide with H1's reused id: r1.HashIDs=%v", r1.HashIDs)
		}
	}
}

// TestConvertSessionToWekaTrace_DanglingAnchorFallback covers a non-root
// instance whose ParentSpawnRequestID resolves to nothing retained (either
// a genuinely dangling id, or the zero sentinel) — it must fall back to the
// nearest-preceding root request, or position -1 (very start) when its
// first request precedes the root's first request entirely.
func TestConvertSessionToWekaTrace_DanglingAnchorFallback(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess6",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "end_turn"},
					{RequestID: 2, Ts: ts(2), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
			{
				InstanceID:           "dangling",
				Role:                 "orphan-sub-agent",
				ParentSpawnRequestID: 999, // does not resolve to any retained request
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 50, Ts: ts(1), Model: "m2", InputTokens: 5, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
			{
				InstanceID:           "beforeall",
				Role:                 "orphan-sub-agent",
				ParentSpawnRequestID: 0, // zero sentinel, and starts before the root's first request
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 60, Ts: ts(-1), Model: "m3", InputTokens: 5, OutputTokens: 1, StopReason: "end_turn"},
				},
			},
		},
	}

	trace, stats, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if stats.DanglingAnchorFallback != 2 {
		t.Fatalf("expected 2 dangling-anchor fallbacks, got %d", stats.DanglingAnchorFallback)
	}
	if len(trace.Requests) != 4 {
		t.Fatalf("expected 4 top-level entries, got %d", len(trace.Requests))
	}

	first, ok := trace.Requests[0].(wekaSubagentEntry)
	if !ok || first.AgentID != "agent_001" {
		t.Fatalf("expected 'beforeall' anchored at the very start, got %+v", trace.Requests[0])
	}
	if _, ok := trace.Requests[1].(wekaRequest); !ok {
		t.Fatalf("requests[1] should be root's first request, got %T", trace.Requests[1])
	}
	dangling, ok := trace.Requests[2].(wekaSubagentEntry)
	if !ok || dangling.AgentID != "agent_002" {
		t.Fatalf("expected 'dangling' anchored right after root's first request, got %+v", trace.Requests[2])
	}
	if _, ok := trace.Requests[3].(wekaRequest); !ok {
		t.Fatalf("requests[3] should be root's second request, got %T", trace.Requests[3])
	}
}

// TestConvertSessionToWekaTrace_JSONShapeMatchesFixture round-trips a
// converted trace through JSON and checks its keys/discriminators against
// agentx-harness's tests/fixtures/weka_traces/one_subagent.json — the best
// defense against a schema mismatch with the Python pydantic model
// available without being able to run Python from here.
func TestConvertSessionToWekaTrace_JSONShapeMatchesFixture(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "trace_sa",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 1, Ts: ts(0), Model: "claude-opus-4-5-20251101", InputTokens: 200, OutputTokens: 30, StopReason: "tool_use", TotalMs: 1000},
					{RequestID: 3, Ts: ts(6), Model: "claude-opus-4-5-20251101", InputTokens: 400, OutputTokens: 40, StopReason: "end_turn", TotalMs: 1500},
				},
			},
			{
				InstanceID:           "sub",
				Role:                 "sub-agent",
				ParentSpawnRequestID: 1,
				Requests: []benchmark.RouterReplayRequest{
					{RequestID: 2, Ts: ts(2), Model: "claude-haiku-4-5-20251001", InputTokens: 100, OutputTokens: 50, StopReason: "end_turn", TotalMs: 500},
				},
			},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}

	b, err := json.Marshal(trace)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var doc map[string]interface{}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	for _, key := range []string{"id", "models", "block_size", "hash_id_scope", "tool_tokens", "system_tokens", "requests"} {
		if _, ok := doc[key]; !ok {
			t.Fatalf("trace missing top-level key %q: %v", key, doc)
		}
	}
	if _, bad := doc["input_length"]; bad {
		t.Fatalf("trace must use the pydantic alias 'in', not the field name 'input_length'")
	}
	if _, bad := doc["output_length"]; bad {
		t.Fatalf("trace must use the pydantic alias 'out', not the field name 'output_length'")
	}

	reqs, ok := doc["requests"].([]interface{})
	if !ok || len(reqs) != 3 {
		t.Fatalf("requests = %v, want a 3-element list", doc["requests"])
	}

	normal, ok := reqs[0].(map[string]interface{})
	if !ok {
		t.Fatalf("requests[0] is not an object: %v", reqs[0])
	}
	for _, key := range []string{"t", "type", "model", "in", "out", "hash_ids", "input_types", "output_types", "stop", "api_time"} {
		if _, ok := normal[key]; !ok {
			t.Fatalf("normal request missing key %q: %v", key, normal)
		}
	}
	if normal["type"] != "n" {
		t.Fatalf("expected discriminator 'n' for a non-streaming request, got %v", normal["type"])
	}

	sub, ok := reqs[1].(map[string]interface{})
	if !ok {
		t.Fatalf("requests[1] is not an object: %v", reqs[1])
	}
	for _, key := range []string{"t", "type", "agent_id", "subagent_type", "duration_ms", "total_tokens", "tool_use_count", "status", "requests", "models", "tool_tokens", "system_tokens"} {
		if _, ok := sub[key]; !ok {
			t.Fatalf("subagent entry missing key %q: %v", key, sub)
		}
	}
	if sub["type"] != "subagent" {
		t.Fatalf("expected discriminator 'subagent', got %v", sub["type"])
	}
	innerReqs, ok := sub["requests"].([]interface{})
	if !ok || len(innerReqs) != 1 {
		t.Fatalf("subagent requests = %v, want a 1-element list", sub["requests"])
	}
	inner, ok := innerReqs[0].(map[string]interface{})
	if !ok {
		t.Fatalf("subagent requests[0] is not an object: %v", innerReqs[0])
	}
	if _, ok := inner["in"]; !ok {
		t.Fatalf("subagent inner request missing 'in' alias: %v", inner)
	}
}

// TestConvertSessionToWekaTrace_CacheControlChurnReusesIDs covers the fix
// for the 62%-of-multi-turn-instances pattern found on the real corpus:
// Anthropic's client resends the full message history verbatim every turn,
// but wekai's capture hash includes the `cache_control` field, and the
// cache-breakpoint marker conventionally moves to the newest block each
// turn — so the SAME message gets a different hash the moment the marker
// moves off it. Without the positional fallback, this mints a fresh,
// unearned id run for old content and breaks the growing-prefix shape
// AIPerf's downstream flattened-agent detection needs to recognize a
// continued conversation.
func TestConvertSessionToWekaTrace_CacheControlChurnReusesIDs(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess-cc",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{
						RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 6577, OutputTokens: 1, StopReason: "tool_use",
						Messages: []benchmark.RouterReplayMessage{
							// Cache breakpoint marked HERE (matches the real
							// corpus example: 6577 tokens, ephemeral marker).
							{Role: "user", Hash: "H1-marked", Tokens: 6577, Bytes: 17551, BlockTypes: []string{"text"}, CacheControl: "ephemeral"},
						},
					},
					{
						// Same slot 0, same role, hash CHANGED because the
						// marker moved off it (now absent here), byte count
						// within tolerance (17503 vs 17551, ~0.3%) -- must
						// reuse slot 0's ids despite the hash mismatch. Slot
						// 1 is new content (the tool_use response) and must
						// mint fresh, non-colliding ids.
						RequestID: 2, Ts: ts(1), Model: "m1", InputTokens: 6559 + 100, OutputTokens: 1, StopReason: "end_turn",
						Messages: []benchmark.RouterReplayMessage{
							{Role: "user", Hash: "H1-unmarked", Tokens: 6559, Bytes: 17503, BlockTypes: []string{"text"}},
							{Role: "assistant", Hash: "H2", Tokens: 100, Bytes: 300, BlockTypes: []string{"tool_use"}},
						},
					},
				},
			},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	r0 := trace.Requests[0].(wekaRequest)
	r1 := trace.Requests[1].(wekaRequest)

	wantSlot0IDs := mathCeilForTest(6577, 64)
	if len(r0.HashIDs) != wantSlot0IDs {
		t.Fatalf("r0 hash_ids = %v, want %d ids", r0.HashIDs, wantSlot0IDs)
	}
	if len(r1.HashIDs) != wantSlot0IDs+mathCeilForTest(100, 64) {
		t.Fatalf("r1 hash_ids = %v, want %d ids", r1.HashIDs, wantSlot0IDs+mathCeilForTest(100, 64))
	}
	for i := range r0.HashIDs {
		if r1.HashIDs[i] != r0.HashIDs[i] {
			t.Fatalf("expected slot 0's ids reused despite hash change (cache_control moved): r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
		}
	}
	for _, id := range r1.HashIDs[wantSlot0IDs:] {
		for _, prior := range r0.HashIDs {
			if id == prior {
				t.Fatalf("H2's freshly-minted ids must not collide with reused slot-0 ids: r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
			}
		}
	}
}

// TestConvertSessionToWekaTrace_CacheControlChurnGatedByPresenceFlip ensures
// the positional-reuse fallback does NOT fire for a block whose
// cache_control presence is identical on both sides (e.g. a genuinely
// dynamic per-call preamble that never carries a marker) even when its byte
// count happens to match exactly -- that block is a real cache miss on
// every call, and merging it would fabricate a cache hit that never
// happened.
func TestConvertSessionToWekaTrace_CacheControlChurnGatedByPresenceFlip(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess-cc2",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{
						RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 41, OutputTokens: 1, StopReason: "end_turn",
						SystemBlocks: []benchmark.RouterReplaySystemBlock{
							{Type: "text", Hash: "DYN-A", Tokens: 41, Bytes: 110},
						},
					},
					{
						// Same slot, same byte/token count, hash changed --
						// but cache_control is absent on BOTH sides, so this
						// must NOT be treated as the same content.
						RequestID: 2, Ts: ts(1), Model: "m1", InputTokens: 41, OutputTokens: 1, StopReason: "end_turn",
						SystemBlocks: []benchmark.RouterReplaySystemBlock{
							{Type: "text", Hash: "DYN-B", Tokens: 41, Bytes: 110},
						},
					},
				},
			},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	r0 := trace.Requests[0].(wekaRequest)
	r1 := trace.Requests[1].(wekaRequest)
	if len(r0.HashIDs) != 1 || len(r1.HashIDs) != 1 {
		t.Fatalf("expected 1 id each: r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
	}
	if r0.HashIDs[0] == r1.HashIDs[0] {
		t.Fatalf("dynamic block with no cache_control on either side must NOT be merged as a cache hit: r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
	}
}

// mathCeilForTest mirrors the assigner's own block-count math for test
// expectations, without importing math just for integer division here.
func mathCeilForTest(tokens, blockSize int) int {
	n := tokens / blockSize
	if tokens%blockSize != 0 {
		n++
	}
	if n < 1 {
		n = 1
	}
	return n
}

// TestConvertSessionToWekaTrace_VolatileSlotMovedToEnd covers the reordering
// fix on top of the cache-control-churn reuse: AIPerf's chain detection
// compares hash_ids by LONGEST COMMON PREFIX from index 0
// (_hash_list_lcp), so a slot that never reuses across an instance's
// requests (real system-block position 0 on the corpus: a small preamble
// with no cache_control on either side, changing every call) must not sit
// at position 0 of the emitted hash_ids -- it would defeat prefix matching
// for the entire request regardless of how well the rest of the content
// matches. A never-reused slot must be pushed after every ever-reused slot,
// even on requests where it happens to appear first in real wire order.
func TestConvertSessionToWekaTrace_VolatileSlotMovedToEnd(t *testing.T) {
	sess := benchmark.RouterReplaySession{
		SessionID: "sess-vol",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{
				InstanceID: "root",
				Role:       "main",
				Requests: []benchmark.RouterReplayRequest{
					{
						RequestID: 1, Ts: ts(0), Model: "m1", InputTokens: 41 + 50, OutputTokens: 1, StopReason: "end_turn",
						SystemBlocks: []benchmark.RouterReplaySystemBlock{
							// slot 0: dynamic preamble, no cache_control on
							// either side -- never reuses, must end up last.
							{Type: "text", Hash: "DYN-A", Tokens: 41, Bytes: 110},
							// slot 1: stable system content, reused verbatim.
							{Type: "text", Hash: "STABLE", Tokens: 50, Bytes: 135},
						},
					},
					{
						RequestID: 2, Ts: ts(1), Model: "m1", InputTokens: 41 + 50, OutputTokens: 1, StopReason: "end_turn",
						SystemBlocks: []benchmark.RouterReplaySystemBlock{
							{Type: "text", Hash: "DYN-B", Tokens: 41, Bytes: 110},
							{Type: "text", Hash: "STABLE", Tokens: 50, Bytes: 135},
						},
					},
				},
			},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	r0 := trace.Requests[0].(wekaRequest)
	r1 := trace.Requests[1].(wekaRequest)

	// Both requests have 2 ids (ceil(41/64)=1 + ceil(50/64)=1). The stable
	// slot's id must be first (position 0) in both, with the volatile
	// slot's id last, despite the volatile slot coming FIRST in real wire
	// order (SystemBlocks[0]).
	if len(r0.HashIDs) != 2 || len(r1.HashIDs) != 2 {
		t.Fatalf("expected 2 ids each: r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
	}
	if r0.HashIDs[0] != r1.HashIDs[0] {
		t.Fatalf("expected the stable slot's id at position 0 in both requests: r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
	}
	if r0.HashIDs[1] == r1.HashIDs[1] {
		t.Fatalf("expected the volatile slot's id to differ (never reused) and sit last: r0=%v r1=%v", r0.HashIDs, r1.HashIDs)
	}
}

// TestConvertSessionToWekaTrace_MajorityVoteNotEverReused is a miniature of
// the real corpus's worst case: a 1,122-request single conversation (no
// subagents) where position 0 coincidentally repeated ONCE, early on, and
// then minted fresh content on nearly every one of the following ~1,121
// transitions. Classifying "reused at least once, anywhere" as stable (an
// earlier, rejected design) kept that slot at position 0 throughout,
// breaking AIPerf's prefix-based chain detection on almost every turn and
// splitting one real conversation into 1,106 disconnected ones.
//
// Five requests here: position 0 connects on only 1 of 4 transitions (a
// minority) and must be classified unstable — pushed to the end — despite
// that one coincidental match. Position 1 connects on 3 of 4 transitions
// (a majority) and must stay stable — kept at the front — despite one
// genuine miss partway through.
func TestConvertSessionToWekaTrace_MajorityVoteNotEverReused(t *testing.T) {
	// hashes chosen so position 0 matches only between req 2 and req 3
	// (both "P0-B"), and position 1 matches everywhere except between req
	// 3 and req 4 ("P1-A" -> "P1-B").
	pos0 := []string{"P0-A", "P0-B", "P0-B", "P0-C", "P0-D"}
	pos1 := []string{"P1-A", "P1-A", "P1-A", "P1-B", "P1-B"}

	var reqs []benchmark.RouterReplayRequest
	for i := range pos0 {
		reqs = append(reqs, benchmark.RouterReplayRequest{
			RequestID: uint64(i + 1), Ts: ts(float64(i)), Model: "m1", InputTokens: 10, OutputTokens: 1, StopReason: "end_turn",
			Messages: []benchmark.RouterReplayMessage{
				{Role: "user", Hash: pos0[i], Tokens: 10, BlockTypes: []string{"text"}},
				{Role: "user", Hash: pos1[i], Tokens: 20, BlockTypes: []string{"text"}},
			},
		})
	}
	sess := benchmark.RouterReplaySession{
		SessionID: "sess-majority",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{InstanceID: "root", Role: "main", Requests: reqs},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	got := make([][]int, len(reqs))
	for i := range reqs {
		got[i] = trace.Requests[i].(wekaRequest).HashIDs
	}

	// Position 1's id (majority-stable) must be identical and FIRST across
	// every request, even the one (index 3) where it doesn't match its own
	// immediate predecessor -- global consistency wins over a lone local
	// mismatch, otherwise index 3 and 4 wouldn't align with each other or
	// with 0-2 either.
	pos1ID := got[0][0]
	for i, ids := range got {
		if len(ids) != 2 {
			t.Fatalf("req[%d] hash_ids = %v, want 2 ids", i, ids)
		}
		if ids[0] != pos1ID && i < 3 {
			// req 3, 4 legitimately mint a new id for pos1 (P1-B is new
			// content), but it must still be pos1ID's SLOT (position 0) --
			// checked structurally below instead of by exact id equality.
			t.Fatalf("req[%d]: expected stable slot's id at position 0 to match req[0]: got %v", i, ids)
		}
	}
	// req 3 and req 4 share pos1 = "P1-B": their position-0 id must match
	// each other even though it differs from req[0..2]'s.
	if got[3][0] != got[4][0] {
		t.Fatalf("req[3] and req[4] share pos1 content (P1-B) and must share the same position-0 id: got[3]=%v got[4]=%v", got[3], got[4])
	}
	// pos0's id (minority-connecting, must be classified unstable) must sit
	// LAST in every request, never first, despite connecting once between
	// req[1] and req[2].
	for i, ids := range got {
		if ids[1] == pos1ID {
			t.Fatalf("req[%d]: position 1 (pos0 content) must not hold the stable slot's id: got %v", i, ids)
		}
	}
}

// TestConvertSessionToWekaTrace_GrowingBlockUnderSameHashExtendsIDCount
// covers a real corpus bug: a dynamically-injected reference block (e.g. a
// growing scratchpad or reminder) keeps a stable hash and a constant, tiny
// `bytes` value across many requests while `tokens` grows steadily between
// them (confirmed on real data: 1,577 -> 1,829 -> 2,440 -> ... -> 5,050
// tokens, same hash, same 82 bytes, every time). Blindly reusing the
// id-run minted for an earlier, smaller occurrence under-counts the
// block's CURRENT length -- AIPerf's own loader hard-aborts when the
// declared system_tokens needs more hash blocks than were actually
// recorded ("turn-0 system prefix requires N hash blocks but only M were
// recorded"). The id-count must always be reconciled to the CURRENT
// occurrence's own declared tokens, growing (new trailing ids, prefix
// preserved as a genuine cache hit) or shrinking as needed, every time the
// hash matches -- not just the first.
func TestConvertSessionToWekaTrace_GrowingBlockUnderSameHashExtendsIDCount(t *testing.T) {
	sameHash := "GROWING-REF"
	tokensPerRequest := []int{1577, 1829, 2440, 2791, 2099, 2516, 3415, 5050}

	var reqs []benchmark.RouterReplayRequest
	for i, tok := range tokensPerRequest {
		reqs = append(reqs, benchmark.RouterReplayRequest{
			RequestID: uint64(i + 1), Ts: ts(float64(i)), Model: "m1", InputTokens: tok, OutputTokens: 1, StopReason: "end_turn",
			SystemBlocks: []benchmark.RouterReplaySystemBlock{
				// bytes constant (82, matching the real corpus example) while
				// tokens grows/shrinks -- the tell that this hash isn't a
				// valid proxy for "same content, same length".
				{Type: "text", Hash: sameHash, Tokens: tok, Bytes: 82},
			},
		})
	}
	sess := benchmark.RouterReplaySession{
		SessionID: "sess-growing-ref",
		StartTs:   ts(0),
		Instances: []benchmark.RouterReplayInstance{
			{InstanceID: "root", Role: "main", Requests: reqs},
		},
	}

	trace, _, err := convertSessionToWekaTrace(sess, convertOptions{BlockSize: 64})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	for i, tok := range tokensPerRequest {
		got := trace.Requests[i].(wekaRequest).HashIDs
		want := mathCeilForTest(tok, 64)
		if len(got) != want {
			t.Fatalf("req[%d]: tokens=%d wants %d hash_ids, got %d: %v", i, tok, want, len(got), got)
		}
	}
	// The shared prefix must actually be shared: req[1] (grown from req[0])
	// must carry req[0]'s ids as its own leading prefix, not a fresh mint.
	r0 := trace.Requests[0].(wekaRequest).HashIDs
	r1 := trace.Requests[1].(wekaRequest).HashIDs
	for i := range r0 {
		if r1[i] != r0[i] {
			t.Fatalf("req[1] must carry req[0]'s ids as its own prefix (a real cache hit): r0=%v r1=%v", r0, r1)
		}
	}
	// req[4] shrinks relative to req[3]; its ids must be req[3]'s own
	// prefix, truncated -- not an unrelated fresh mint.
	r3 := trace.Requests[3].(wekaRequest).HashIDs
	r4 := trace.Requests[4].(wekaRequest).HashIDs
	for i := range r4 {
		if r4[i] != r3[i] {
			t.Fatalf("req[4] (shrunk) must be req[3]'s prefix, truncated: r3=%v r4=%v", r3, r4)
		}
	}
}
