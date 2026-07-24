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

// TestBuildSessionUUIDsDeterminism verifies buildSessionUUIDs matches the
// dataset path's determinism contract: same seed -> same per-session UUID
// assignment; different seed -> different assignment; every UUID unique.
func TestBuildSessionUUIDsDeterminism(t *testing.T) {
	a := buildSessionUUIDs(5, 42)
	b := buildSessionUUIDs(5, 42)
	if len(a) != 5 || len(b) != 5 {
		t.Fatalf("expected 5 sets each, got %d and %d", len(a), len(b))
	}
	for i := range a {
		if len(a[i]) != 1 || len(b[i]) != 1 {
			t.Fatalf("session %d: expected singleton sets, got %v / %v", i, a[i], b[i])
		}
		if a[i][0] != b[i][0] {
			t.Errorf("session %d: same seed produced different UUIDs: %q vs %q", i, a[i][0], b[i][0])
		}
	}

	c := buildSessionUUIDs(5, 43)
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
		if seen[set[0]] {
			t.Errorf("uuid %q assigned to more than one session", set[0])
		}
		seen[set[0]] = true
	}

	if got := buildSessionUUIDs(0, 42); got != nil {
		t.Errorf("buildSessionUUIDs(0, ...) = %v, want nil", got)
	}
}

// TestWireInjectionDeterminism verifies that, for a fixed session's marker,
// buildOpenAIChatCompletionsBody / buildAnthropicMessagesBody produce
// byte-identical bodies across repeated calls (same request + same
// injection in -> same bytes out), and that two DIFFERENT sessions' markers
// diverge the body.
func TestWireInjectionDeterminism(t *testing.T) {
	docs := strings.Repeat("wire-injection-docs ", 100)
	req := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys1", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msgUniq0", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	sets := buildSessionUUIDs(2, 7)
	injA := &uuidInjection{Marker: injectUUIDMarker("", sets[0]), Recite: true, SharedPrefixLen: 1}
	injA2 := &uuidInjection{Marker: injectUUIDMarker("", sets[0]), Recite: true, SharedPrefixLen: 1}
	injB := &uuidInjection{Marker: injectUUIDMarker("", sets[1]), Recite: true, SharedPrefixLen: 1}

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
			t.Errorf("%s: different sessions' markers produced identical bytes", kind)
		}
		if !strings.Contains(string(bodyA1), sets[0][0]) {
			t.Errorf("%s: body missing session A's own UUID", kind)
		}
		if strings.Contains(string(bodyA1), sets[1][0]) {
			t.Errorf("%s: body A leaked session B's UUID into the wire body", kind)
		}
	}
}

// TestCacheFidelityBoundaryInvariant verifies the core Option-C guarantee:
// two DIFFERENT sessions that share a leading system block emit
// byte-identical content for that shared block, diverging only at (or
// after) the injected per-session marker — i.e. injection never perturbs
// the cross-session-shared prefix a real server would prefix-cache on.
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
	injA := &uuidInjection{Marker: injectUUIDMarker("", []string{"uuid-session-A"}), Recite: false, SharedPrefixLen: 1}
	injB := &uuidInjection{Marker: injectUUIDMarker("", []string{"uuid-session-B"}), Recite: false, SharedPrefixLen: 1}

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
		t.Fatalf("expected system = [shared block, marker], got lens %d and %d", len(sysA), len(sysB))
	}
	// Index 0 (the shared system block, "sys1") must be byte-identical.
	sharedA, _ := json.Marshal(sysA[0])
	sharedB, _ := json.Marshal(sysB[0])
	if string(sharedA) != string(sharedB) {
		t.Errorf("shared leading system block diverged between sessions:\nA: %s\nB: %s", sharedA, sharedB)
	}
	// Index 1 (the injected marker) MUST diverge — that's the whole point.
	markerA, _ := json.Marshal(sysA[1])
	markerB, _ := json.Marshal(sysB[1])
	if string(markerA) == string(markerB) {
		t.Error("injected markers were identical across two different sessions")
	}
	if !strings.Contains(string(markerA), "uuid-session-A") {
		t.Errorf("session A's marker missing its own uuid: %s", markerA)
	}
	if !strings.Contains(string(markerB), "uuid-session-B") {
		t.Errorf("session B's marker missing its own uuid: %s", markerB)
	}
}

// TestTailFallbackInjection verifies that when SharedPrefixLen == 0 (no
// usable boundary), the marker is folded into the tail (messages array)
// rather than the system array, and the request remains well-formed.
func TestTailFallbackInjection(t *testing.T) {
	docs := strings.Repeat("tail-fallback-docs ", 100)
	req := RouterReplayRequest{
		InputTokens:  500,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys-unique", Bytes: 250}},
		Messages:     []RouterReplayMessage{{Hash: "msg-unique", Role: "user", BlockTypes: []string{"text"}, Bytes: 100}},
	}
	inj := &uuidInjection{Marker: injectUUIDMarker("", []string{"uuid-tail"}), Recite: true, SharedPrefixLen: 0}

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
		t.Fatal("marker missing from body entirely")
	}
	msgs, _ := parsed["messages"].([]interface{})
	if len(msgs) == 0 {
		t.Fatal("expected messages to carry the tail-injected marker/recite content")
	}
	last := msgs[len(msgs)-1].(map[string]interface{})
	if last["role"] != "user" {
		t.Errorf("tail-injected message role = %v, want user", last["role"])
	}
}

// TestUUIDValidationEndToEnd exercises buildSessionUUIDs + injectUUIDMarker +
// validateReplayResponse together, mirroring how replayPoster.do() wires
// them: a response containing the OWN session's uuid scores found/no-leak;
// a response containing ANOTHER session's uuid scores CROSS_CONTAMINATION
// against the correct series index.
func TestUUIDValidationEndToEnd(t *testing.T) {
	sets := buildSessionUUIDs(3, 123)

	t.Run("own uuid present, no leak", func(t *testing.T) {
		resp := "Sure, the ref-id I recall is " + sets[0][0] + ". Anyway, here's your answer."
		found, leaked := validateReplayResponse(resp, "", sets[0], 0, sets)
		if len(found) != 1 || !found[0] {
			t.Errorf("found = %v, want [true]", found)
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})

	t.Run("cross contamination from another session", func(t *testing.T) {
		resp := "SEEN_REF: " + sets[0][0] + " and also " + sets[2][0]
		found, leaked := validateReplayResponse(resp, "", sets[0], 0, sets)
		if len(found) != 1 || !found[0] {
			t.Errorf("found = %v, want [true]", found)
		}
		if len(leaked) != 1 || !strings.Contains(leaked[0], sets[2][0]) || !strings.Contains(leaked[0], "series=2") {
			t.Errorf("leaked = %v, want one entry naming session 2's uuid", leaked)
		}
	})

	t.Run("presence miss", func(t *testing.T) {
		found, leaked := validateReplayResponse("no ref ids here", "", sets[1], 1, sets)
		if found[0] {
			t.Error("expected PRESENCE_MISS (found=false)")
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})
}

// TestApplyReciteFloor verifies the max_tokens recite-floor helper: raises
// a too-small budget to replayReciteFloorTokens only when recite is
// requested; leaves larger budgets and non-recite calls untouched.
func TestApplyReciteFloor(t *testing.T) {
	cases := []struct {
		name   string
		tokens int
		recite bool
		want   int
	}{
		{"below floor, recite -> raised", 10, true, replayReciteFloorTokens},
		{"at floor, recite -> unchanged", replayReciteFloorTokens, true, replayReciteFloorTokens},
		{"above floor, recite -> unchanged", 1000, true, 1000},
		{"below floor, no recite -> unchanged", 10, false, 10},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := applyReciteFloor(c.tokens, c.recite); got != c.want {
				t.Errorf("applyReciteFloor(%d, %v) = %d, want %d", c.tokens, c.recite, got, c.want)
			}
		})
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
	inj := &uuidInjection{Marker: injectUUIDMarker("", []string{"uuid-floor"}), Recite: true, SharedPrefixLen: 1}

	anthBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("anthropic build: %v", err)
	}
	var anthParsed map[string]interface{}
	if err := json.Unmarshal(anthBody, &anthParsed); err != nil {
		t.Fatalf("anthropic unmarshal: %v", err)
	}
	if got, want := anthParsed["max_tokens"].(float64), float64(replayReciteFloorTokens); got != want {
		t.Errorf("anthropic max_tokens = %v, want %v (floor)", got, want)
	}

	openaiBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("openai build: %v", err)
	}
	var openaiParsed map[string]interface{}
	if err := json.Unmarshal(openaiBody, &openaiParsed); err != nil {
		t.Fatalf("openai unmarshal: %v", err)
	}
	if got, want := openaiParsed["max_tokens"].(float64), float64(replayReciteFloorTokens); got != want {
		t.Errorf("openai max_tokens = %v, want %v (floor)", got, want)
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
