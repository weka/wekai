package benchmark

// Pure, offline unit tests for router-replay UUID cache-coherency injection
// (replay_router_uuid.go). No mocked LLM/Chat — these exercise data
// transforms and the wire builders directly, per repo testing policy.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeReplayV3File marshals a header + sessions into a replay-v3 JSONL
// file under t.TempDir() and returns its path.
func writeReplayV3File(t *testing.T, sessions []RouterReplaySession) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "replay.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create temp replay file: %v", err)
	}
	defer f.Close()

	hdr := RouterReplayHeader{
		Schema: "replay-v3",
		Summary: RouterReplaySummary{
			Sessions: len(sessions),
		},
	}
	enc := json.NewEncoder(f)
	if err := enc.Encode(hdr); err != nil {
		t.Fatalf("encode header: %v", err)
	}
	for _, s := range sessions {
		if err := enc.Encode(s); err != nil {
			t.Fatalf("encode session: %v", err)
		}
	}
	return path
}

// syntheticSessionsForBoundaryTests builds 5 sessions exercising every
// sharedPrefixBlockCount regime:
//   - sessions 0,1: share one system block (sys1) but each carries its own
//     unique message -> leading run length 1, which equals the (single)
//     emitted system block count -> "covers all system blocks" boundary case.
//   - session 2: no block it carries is shared with any other session at
//     all -> leading run length 0 -> tail-fallback case.
//   - sessions 3,4: share BOTH their system block AND their message block
//     (identical request shape) -> leading run length 2 == full prefix
//     length -> "fully shared" edge case.
func syntheticSessionsForBoundaryTests() []RouterReplaySession {
	mkSession := func(id string, sysHash string, msgHash string) RouterReplaySession {
		return RouterReplaySession{
			SessionID: id,
			Instances: []RouterReplayInstance{
				{
					InstanceID: id + "-inst",
					Requests: []RouterReplayRequest{
						{
							RequestID:   1,
							InputTokens: 500,
							SystemBlocks: []RouterReplaySystemBlock{
								{Hash: sysHash, Bytes: 250, Tokens: 60},
							},
							Messages: []RouterReplayMessage{
								{Hash: msgHash, Role: "user", BlockTypes: []string{"text"}, Bytes: 100, Tokens: 25},
							},
						},
					},
				},
			},
		}
	}
	return []RouterReplaySession{
		mkSession("s0", "sys1", "msgUniq0"),
		mkSession("s1", "sys1", "msgUniq1"),
		mkSession("s2", "sys2only", "msgUniq2"),
		mkSession("s3", "sysShared34", "msgShared34"),
		mkSession("s4", "sysShared34", "msgShared34"),
	}
}

func TestComputeBlockSessionCounts(t *testing.T) {
	path := writeReplayV3File(t, syntheticSessionsForBoundaryTests())

	counts, err := computeBlockSessionCounts(path, nil, 0)
	if err != nil {
		t.Fatalf("computeBlockSessionCounts: %v", err)
	}

	cases := []struct {
		hash string
		want int
	}{
		{"sys1", 2},          // sessions 0 and 1
		{"msgUniq0", 1},      // only session 0
		{"msgUniq1", 1},      // only session 1
		{"sys2only", 1},      // only session 2
		{"msgUniq2", 1},      // only session 2
		{"sysShared34", 2},   // sessions 3 and 4
		{"msgShared34", 2},   // sessions 3 and 4
		{"never-appears", 0}, // absent hash
	}
	for _, c := range cases {
		if got := counts[c.hash]; got != c.want {
			t.Errorf("counts[%q] = %d, want %d", c.hash, got, c.want)
		}
	}
}

func TestComputeBlockSessionCountsRespectsFilters(t *testing.T) {
	path := writeReplayV3File(t, syntheticSessionsForBoundaryTests())

	t.Run("sessionLimit caps to the first N sessions", func(t *testing.T) {
		// sessionLimit=2 -> only s0, s1 counted; sys1 still shared (2), but
		// session 2/3/4 hashes never appear.
		counts, err := computeBlockSessionCounts(path, nil, 2)
		if err != nil {
			t.Fatalf("computeBlockSessionCounts: %v", err)
		}
		if got := counts["sys1"]; got != 2 {
			t.Errorf("sys1 = %d, want 2", got)
		}
		if got := counts["sysShared34"]; got != 0 {
			t.Errorf("sysShared34 = %d, want 0 (beyond sessionLimit)", got)
		}
	})

	t.Run("allowed index set restricts to those sessions only", func(t *testing.T) {
		// Only session index 3 and 4 (0-based) allowed -> sysShared34 still
		// shows count 2, but sys1 (sessions 0,1) never counted.
		allowed := map[int]bool{3: true, 4: true}
		counts, err := computeBlockSessionCounts(path, allowed, 0)
		if err != nil {
			t.Fatalf("computeBlockSessionCounts: %v", err)
		}
		if got := counts["sysShared34"]; got != 2 {
			t.Errorf("sysShared34 = %d, want 2", got)
		}
		if got := counts["sys1"]; got != 0 {
			t.Errorf("sys1 = %d, want 0 (session 0/1 excluded)", got)
		}
	})
}

func TestSharedPrefixBlockCount(t *testing.T) {
	path := writeReplayV3File(t, syntheticSessionsForBoundaryTests())
	counts, err := computeBlockSessionCounts(path, nil, 0)
	if err != nil {
		t.Fatalf("computeBlockSessionCounts: %v", err)
	}

	reqBoundary := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgUniq0", Role: "user", Bytes: 100}},
	}
	if got := sharedPrefixBlockCount(reqBoundary, counts); got != 1 {
		t.Errorf("boundary case: sharedPrefixBlockCount = %d, want 1 (covers the single system block)", got)
	}

	reqNoShare := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys2only", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgUniq2", Role: "user", Bytes: 100}},
	}
	if got := sharedPrefixBlockCount(reqNoShare, counts); got != 0 {
		t.Errorf("no-shared-block case: sharedPrefixBlockCount = %d, want 0 (tail fallback)", got)
	}

	reqFullyShared := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sysShared34", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgShared34", Role: "user", Bytes: 100}},
	}
	if got, want := sharedPrefixBlockCount(reqFullyShared, counts), 2; got != want {
		t.Errorf("fully-shared case: sharedPrefixBlockCount = %d, want %d (full prefix length)", got, want)
	}
}

// TestComputePerSessionCachedChars verifies the per-session cached-region
// byte size computation: the ROOT request's (first instance, first
// request) prefix bytes AT OR AFTER its sharedPrefixBlockCount boundary,
// using the same synthetic 5-session fixture TestSharedPrefixBlockCount
// exercises (sys=250 bytes, msg=100 bytes each request).
func TestComputePerSessionCachedChars(t *testing.T) {
	path := writeReplayV3File(t, syntheticSessionsForBoundaryTests())
	counts, err := computeBlockSessionCounts(path, nil, 0)
	if err != nil {
		t.Fatalf("computeBlockSessionCounts: %v", err)
	}

	got, err := computePerSessionCachedChars(path, nil, 0, counts)
	if err != nil {
		t.Fatalf("computePerSessionCachedChars: %v", err)
	}
	// s0,s1: boundary=1 (only the shared sys1 block) -> chars = msg bytes (100).
	// s2: boundary=0 (nothing shared) -> chars = sys(250) + msg(100) = 350.
	// s3,s4: boundary=2 (both blocks shared, full prefix) -> chars = 0.
	want := []int{100, 100, 350, 0, 0}
	if len(got) != len(want) {
		t.Fatalf("computePerSessionCachedChars len = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("session %d: chars = %d, want %d", i, got[i], w)
		}
	}
}

// TestBuildSessionUUIDsDeterminism verifies buildSessionUUIDs matches the
// dataset path's determinism contract: same seed -> same per-session UUID
// assignment; different seed -> different assignment; every UUID unique
// across the whole run (not just within a session).
func TestBuildSessionUUIDsDeterminism(t *testing.T) {
	perSessionChars := []int{0, 0, 0, 0, 0} // all -> computeStampsPerSeries floors to 2
	a := buildSessionUUIDs(perSessionChars, 42)
	b := buildSessionUUIDs(perSessionChars, 42)
	if len(a) != 5 || len(b) != 5 {
		t.Fatalf("expected 5 sets each, got %d and %d", len(a), len(b))
	}
	for i := range a {
		if len(a[i]) != 2 || len(b[i]) != 2 {
			t.Fatalf("session %d: expected 2-UUID sets (min floor), got %v / %v", i, a[i], b[i])
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Errorf("session %d stamp %d: same seed produced different UUIDs: %q vs %q", i, j, a[i][j], b[i][j])
			}
		}
	}

	c := buildSessionUUIDs(perSessionChars, 43)
	same := true
	for i := range a {
		if a[i][0] != c[i][0] {
			same = false
		}
	}
	if same {
		t.Error("different seeds produced an identical UUID assignment")
	}

	seen := map[string]bool{}
	for _, set := range a {
		for _, u := range set {
			if seen[u] {
				t.Errorf("uuid %q assigned to more than one stamp", u)
			}
			seen[u] = true
		}
	}

	if got := buildSessionUUIDs(nil, 42); got != nil {
		t.Errorf("buildSessionUUIDs(nil, ...) = %v, want nil", got)
	}
}

// TestBuildSessionUUIDsScalesWithBytes verifies each session's N is exactly
// computeStampsPerSeries(perSessionChars[i]) -- min 2, scaling with bytes --
// mirroring the cache-coherency eval's garbageChars -> numStamps rule.
func TestBuildSessionUUIDsScalesWithBytes(t *testing.T) {
	perSessionChars := []int{0, 8192 * 5, 8192*10 + 100}
	sets := buildSessionUUIDs(perSessionChars, 7)
	want := []int{2, 5, 10}
	for i, w := range want {
		if got := len(sets[i]); got != w {
			t.Errorf("session %d: len = %d, want %d (computeStampsPerSeries(%d))", i, got, w, perSessionChars[i])
		}
	}
	for _, chars := range []int{0, 1, 8192, 8192 * 3, 100000} {
		got := len(buildSessionUUIDs([]int{chars}, 1)[0])
		want := computeStampsPerSeries(chars)
		if got != want {
			t.Errorf("chars=%d: N = %d, want %d (computeStampsPerSeries)", chars, got, want)
		}
	}
}

// TestWireInjectionDeterminism verifies that, for a fixed session's N-UUID
// block, buildOpenAIChatCompletionsBody / buildAnthropicMessagesBody produce
// byte-identical bodies across repeated calls (same request + same
// injection in -> same bytes out), and that two DIFFERENT sessions' UUID
// blocks diverge the body.
func TestWireInjectionDeterminism(t *testing.T) {
	docs := strings.Repeat("wire-injection-docs ", 100)
	req := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgUniq0", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	sets := buildSessionUUIDs([]int{0, 0}, 7)
	injA := &uuidInjection{UUIDs: sets[0], Recite: true, SharedPrefixLen: 1}
	injA2 := &uuidInjection{UUIDs: sets[0], Recite: true, SharedPrefixLen: 1}
	injB := &uuidInjection{UUIDs: sets[1], Recite: true, SharedPrefixLen: 1}

	for _, kind := range []string{"openai", "anthropic"} {
		build := func(r RouterReplayRequest, inj *uuidInjection) []byte {
			var body []byte
			var err error
			if kind == "openai" {
				body, _, err = buildOpenAIChatCompletionsBody(r, docs, "model", "", 0, false, inj)
			} else {
				body, _, err = buildAnthropicMessagesBody(r, docs, "model", "", 0, false, inj)
			}
			if err != nil {
				t.Fatalf("%s build: %v", kind, err)
			}
			return body
		}
		bodyA1 := build(req, injA)
		bodyA2 := build(req, injA2)
		bodyB := build(req, injB)

		if string(bodyA1) != string(bodyA2) {
			t.Errorf("%s: identical injection produced different bytes", kind)
		}
		if string(bodyA1) == string(bodyB) {
			t.Errorf("%s: different sessions' UUID blocks produced identical bytes", kind)
		}
		for _, u := range sets[0] {
			if !strings.Contains(string(bodyA1), u) {
				t.Errorf("%s: body missing session A's own UUID %q", kind, u)
			}
		}
		for _, u := range sets[1] {
			if strings.Contains(string(bodyA1), u) {
				t.Errorf("%s: body A leaked session B's UUID %q into the wire body", kind, u)
			}
		}
		if strings.Contains(string(bodyA1), "ref-id") {
			t.Errorf("%s: injected block still carries the old [ref-id: ...] wrapper", kind)
		}
	}
}

// TestBoundaryInjectionEmitsBareSpaceSeparatedUUIDs verifies the injected
// block is exactly N bare, space-separated UUIDs (no wrapper text) —
// mirroring the cache-coherency eval's buildCoherencySharedSeriesPrompt tail
// — and that it is byte-identical across two DIFFERENT requests belonging
// to the SAME session (the within-session cache-reuse property), while the
// cross-session-shared leading system block stays untouched.
func TestBoundaryInjectionEmitsBareSpaceSeparatedUUIDs(t *testing.T) {
	docs := strings.Repeat("boundary-multi-docs ", 100)
	uuids := []string{"uuid-0", "uuid-1", "uuid-2"}
	inj := &uuidInjection{UUIDs: uuids, Recite: false, SharedPrefixLen: 1}

	req1 := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgTurn1", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	req2 := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}}, // same shared system block
		Messages:     []RouterReplayMessage{{Hash: "msgTurn2", Role: "user", BlockTypes: []string{"text"}, Bytes: 140}},
	}

	body1, _, err := buildAnthropicMessagesBody(req1, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("build 1: %v", err)
	}
	body2, _, err := buildAnthropicMessagesBody(req2, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("build 2: %v", err)
	}

	var parsed1, parsed2 map[string]interface{}
	if err := json.Unmarshal(body1, &parsed1); err != nil {
		t.Fatalf("unmarshal 1: %v", err)
	}
	if err := json.Unmarshal(body2, &parsed2); err != nil {
		t.Fatalf("unmarshal 2: %v", err)
	}

	sys1, _ := parsed1["system"].([]interface{})
	sys2, _ := parsed2["system"].([]interface{})
	if len(sys1) != 2 || len(sys2) != 2 {
		t.Fatalf("expected system = [shared block, uuid block], got lens %d and %d", len(sys1), len(sys2))
	}

	wantText := "uuid-0 uuid-1 uuid-2"
	block1 := sys1[1].(map[string]interface{})
	block2 := sys2[1].(map[string]interface{})
	if block1["text"] != wantText {
		t.Errorf("uuid block 1 text = %q, want %q (bare, space-separated)", block1["text"], wantText)
	}
	if block2["text"] != wantText {
		t.Errorf("uuid block diverged across two requests in the SAME session: %v vs %v", block2["text"], wantText)
	}

	// The shared leading system block (index 0) must stay byte-identical —
	// injection must never perturb the cross-session-shared prefix.
	shared1, _ := json.Marshal(sys1[0])
	shared2, _ := json.Marshal(sys2[0])
	if string(shared1) != string(shared2) {
		t.Errorf("shared leading system block diverged across requests: %s vs %s", shared1, shared2)
	}
}

// TestCacheFidelityBoundaryInvariant verifies the core Option-C guarantee:
// two DIFFERENT sessions that share a leading system block emit
// byte-identical content for that shared block, diverging only at (or
// after) the injected per-session UUID block — i.e. injection never
// perturbs the cross-session-shared prefix a real server would
// prefix-cache on.
func TestCacheFidelityBoundaryInvariant(t *testing.T) {
	docs := strings.Repeat("fidelity-docs ", 100)
	reqA := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgUniq-A", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	reqB := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}}, // SAME shared system block
		Messages:     []RouterReplayMessage{{Hash: "msgUniq-B", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	injA := &uuidInjection{UUIDs: []string{"uuid-session-A"}, Recite: false, SharedPrefixLen: 1}
	injB := &uuidInjection{UUIDs: []string{"uuid-session-B"}, Recite: false, SharedPrefixLen: 1}

	bodyA, _, err := buildAnthropicMessagesBody(reqA, docs, "model", "", 0, false, injA)
	if err != nil {
		t.Fatalf("build A: %v", err)
	}
	bodyB, _, err := buildAnthropicMessagesBody(reqB, docs, "model", "", 0, false, injB)
	if err != nil {
		t.Fatalf("build B: %v", err)
	}

	var parsedA, parsedB map[string]interface{}
	if err := json.Unmarshal(bodyA, &parsedA); err != nil {
		t.Fatalf("unmarshal A: %v", err)
	}
	if err := json.Unmarshal(bodyB, &parsedB); err != nil {
		t.Fatalf("unmarshal B: %v", err)
	}

	sysA, _ := parsedA["system"].([]interface{})
	sysB, _ := parsedB["system"].([]interface{})
	if len(sysA) != 2 || len(sysB) != 2 {
		t.Fatalf("expected system = [shared block, uuid block], got lens %d and %d", len(sysA), len(sysB))
	}
	// Index 0 (the shared system block, "sys1") must be byte-identical.
	sharedA, _ := json.Marshal(sysA[0])
	sharedB, _ := json.Marshal(sysB[0])
	if string(sharedA) != string(sharedB) {
		t.Errorf("shared leading system block diverged between sessions:\nA: %s\nB: %s", sharedA, sharedB)
	}
	// Index 1 (the injected uuid block) MUST diverge — that's the whole point.
	blockA, _ := json.Marshal(sysA[1])
	blockB, _ := json.Marshal(sysB[1])
	if string(blockA) == string(blockB) {
		t.Error("injected uuid blocks were identical across two different sessions")
	}
	if !strings.Contains(string(blockA), "uuid-session-A") {
		t.Errorf("session A's block missing its own uuid: %s", blockA)
	}
	if !strings.Contains(string(blockB), "uuid-session-B") {
		t.Errorf("session B's block missing its own uuid: %s", blockB)
	}
}

// TestTailFallbackInjection verifies that when SharedPrefixLen == 0 (no
// usable boundary), the UUID block is folded into the tail (messages array)
// rather than the system array, and the request remains well-formed.
func TestTailFallbackInjection(t *testing.T) {
	docs := strings.Repeat("tail-fallback-docs ", 100)
	req := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys-unique", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msg-unique", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	inj := &uuidInjection{UUIDs: []string{"uuid-tail"}, Recite: true, SharedPrefixLen: 0}

	body, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	sys, _ := parsed["system"].([]interface{})
	if len(sys) != 1 {
		t.Fatalf("expected system to carry ONLY the original block (no boundary splice), got %d entries", len(sys))
	}
	if strings.Contains(string(body), "uuid-tail") == false {
		t.Fatal("uuid block missing from body entirely")
	}
	msgs, _ := parsed["messages"].([]interface{})
	if len(msgs) == 0 {
		t.Fatal("expected messages to carry the tail-injected uuid block/recite content")
	}
	last := msgs[len(msgs)-1].(map[string]interface{})
	if last["role"] != "user" {
		t.Errorf("tail-injected message role = %v, want user", last["role"])
	}
}

// TestUUIDValidationEndToEnd exercises validateReplayResponse (presence +
// cross-session leak) directly against N-stamp sessions (N=2), mirroring
// how replayPoster.do() wires it: a response containing ALL of the OWN
// session's uuids scores found/no-leak; a response containing ANOTHER
// session's uuid scores CROSS_CONTAMINATION against the correct series
// index; a response missing one of its own uuids scores a partial
// PRESENCE_MISS. Semantics are unchanged from the single-uuid path — only
// the stamp count (N) is now typically > 1.
func TestUUIDValidationEndToEnd(t *testing.T) {
	sets := [][]string{
		{"uuid-s0-a", "uuid-s0-b"},
		{"uuid-s1-a", "uuid-s1-b"},
		{"uuid-s2-a", "uuid-s2-b"},
	}

	t.Run("own uuids present, no leak", func(t *testing.T) {
		resp := "Sure, the ids I recall are " + sets[0][0] + " and " + sets[0][1] + ". Anyway, here's your answer."
		found, leaked := validateReplayResponse(resp, "", sets[0], 0, sets)
		if len(found) != 2 || !found[0] || !found[1] {
			t.Errorf("found = %v, want [true true]", found)
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})

	t.Run("cross contamination from another session", func(t *testing.T) {
		resp := sets[0][0] + ", " + sets[0][1] + ", and also " + sets[2][0]
		found, leaked := validateReplayResponse(resp, "", sets[0], 0, sets)
		if len(found) != 2 || !found[0] || !found[1] {
			t.Errorf("found = %v, want [true true]", found)
		}
		if len(leaked) != 1 || !strings.Contains(leaked[0], sets[2][0]) || !strings.Contains(leaked[0], "series=2") {
			t.Errorf("leaked = %v, want one entry naming session 2's uuid", leaked)
		}
	})

	t.Run("partial presence miss", func(t *testing.T) {
		resp := "only " + sets[1][0] + " here"
		found, leaked := validateReplayResponse(resp, "", sets[1], 1, sets)
		if len(found) != 2 || !found[0] || found[1] {
			t.Errorf("found = %v, want [true false]", found)
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})

	t.Run("total presence miss", func(t *testing.T) {
		found, leaked := validateReplayResponse("no ref ids here", "", sets[1], 1, sets)
		if found[0] || found[1] {
			t.Error("expected PRESENCE_MISS on both stamps (found=[false false])")
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})
}

// TestFirstLineConformity exercises firstLineConformity (the output-
// conformity check --replay-inject-uuids scores, mirroring the coherency
// eval's matchesExpectedUUIDList/ExactMatch): pass on an exact ordered
// comma-joined first line; fail on missing, reordered, or chatty first
// lines; pass when line 1 is exact even though LATER lines contain filler
// (the whole point of front-loading the ask to line 1 while forced-output
// keeps generating).
func TestFirstLineConformity(t *testing.T) {
	expected := []string{"uuid-a", "uuid-b", "uuid-c"}

	cases := []struct {
		name string
		resp string
		want bool
	}{
		{"exact single line", "uuid-a, uuid-b, uuid-c", true},
		{"exact with surrounding whitespace tolerated", "  uuid-a, uuid-b, uuid-c  ", true},
		{"exact first line, filler after", "uuid-a, uuid-b, uuid-c\nHere is more detail about your request...", true},
		{"exact first line, multiple filler lines after", "uuid-a, uuid-b, uuid-c\nline2\nline3 with more text", true},
		{"missing a uuid", "uuid-a, uuid-c", false},
		{"reordered", "uuid-b, uuid-a, uuid-c", false},
		{"chatty first line", "Sure! The UUIDs are uuid-a, uuid-b, uuid-c", false},
		{"uuids only on line 2, not line 1", "Sure, here you go:\nuuid-a, uuid-b, uuid-c", false},
		{"empty response", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := firstLineConformity(c.resp, expected); got != c.want {
				t.Errorf("firstLineConformity(%q) = %v, want %v", c.resp, got, c.want)
			}
		})
	}
}

// TestApplyReciteFloor verifies the max_tokens recite-floor helper: raises
// a too-small budget to replayReciteFloorTokens(numUUIDs) only when recite
// is requested; leaves larger budgets and non-recite calls untouched.
func TestApplyReciteFloor(t *testing.T) {
	cases := []struct {
		name     string
		tokens   int
		recite   bool
		numUUIDs int
	}{
		{"below floor, recite -> raised", 5, true, 2},
		{"at floor, recite -> unchanged", replayReciteFloorTokens(2), true, 2},
		{"above floor, recite -> unchanged", 100000, true, 2},
		{"below floor, no recite -> unchanged", 5, false, 2},
		{"below floor, more uuids, recite -> raised higher", 5, true, 20},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			floor := replayReciteFloorTokens(c.numUUIDs)
			want := c.tokens
			if c.recite && c.tokens < floor {
				want = floor
			}
			if got := applyReciteFloor(c.tokens, c.recite, c.numUUIDs); got != want {
				t.Errorf("applyReciteFloor(%d, %v, %d) = %d, want %d", c.tokens, c.recite, c.numUUIDs, got, want)
			}
		})
	}
}

// TestReciteFloorScalesWithN verifies the recite floor grows with numUUIDs —
// more UUIDs to recite on the first line needs a bigger budget — the
// N-per-session analogue of the old fixed-64-token constant.
func TestReciteFloorScalesWithN(t *testing.T) {
	small := replayReciteFloorTokens(2)
	large := replayReciteFloorTokens(20)
	if large <= small {
		t.Errorf("replayReciteFloorTokens(20) = %d, want > replayReciteFloorTokens(2) = %d", large, small)
	}
}

// TestMaxTokensFloorAppliedInWireBuilders verifies the floor is actually
// wired into both body builders' emitted max_tokens when a recite
// injection is present and the original/recorded budget is tiny — the
// scenario a real tool-call-only turn would hit.
func TestMaxTokensFloorAppliedInWireBuilders(t *testing.T) {
	docs := strings.Repeat("floor-docs ", 100)
	req := RouterReplayRequest{
		InputTokens:  500,
		OutputTokens: 5, // tiny recorded budget -- would truncate the recite line
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msg1", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	uuids := []string{"uuid-floor-0", "uuid-floor-1"}
	inj := &uuidInjection{UUIDs: uuids, Recite: true, SharedPrefixLen: 1}
	wantFloor := float64(replayReciteFloorTokens(len(uuids)))

	anthBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("anthropic build: %v", err)
	}
	var anthParsed map[string]interface{}
	if err := json.Unmarshal(anthBody, &anthParsed); err != nil {
		t.Fatalf("anthropic unmarshal: %v", err)
	}
	if got := anthParsed["max_tokens"].(float64); got != wantFloor {
		t.Errorf("anthropic max_tokens = %v, want %v (floor)", got, wantFloor)
	}

	openaiBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("openai build: %v", err)
	}
	var openaiParsed map[string]interface{}
	if err := json.Unmarshal(openaiBody, &openaiParsed); err != nil {
		t.Fatalf("openai unmarshal: %v", err)
	}
	if got := openaiParsed["max_tokens"].(float64); got != wantFloor {
		t.Errorf("openai max_tokens = %v, want %v (floor)", got, wantFloor)
	}

	// Without injection (nil), the tiny recorded output_tokens is honored
	// as before -- the floor must never apply when there's no recite ask.
	plainBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, nil)
	if err != nil {
		t.Fatalf("anthropic build (no injection): %v", err)
	}
	var plainParsed map[string]interface{}
	if err := json.Unmarshal(plainBody, &plainParsed); err != nil {
		t.Fatalf("anthropic unmarshal (no injection): %v", err)
	}
	if got, want := plainParsed["max_tokens"].(float64), float64(5); got != want {
		t.Errorf("anthropic max_tokens (no injection) = %v, want %v (unfloored)", got, want)
	}
}
