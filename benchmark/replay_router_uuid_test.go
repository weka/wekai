package benchmark

// Pure, offline unit tests for router-replay UUID cache-coherency injection
// (replay_router_uuid.go, replay_uuid.go, and the buildInjection/wire-body
// plumbing in replay_router_post.go/replay_router_wire.go). No mocked LLM/
// Chat — these exercise data transforms and the wire builders directly, per
// repo testing policy.

import (
	"encoding/json"
	"fmt"
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

// userText is a shorthand for a qualifying user-turn message (role=="user",
// one "text" block).
func userText(hash string, bytes int) RouterReplayMessage {
	return RouterReplayMessage{Role: "user", Hash: hash, BlockTypes: []string{"text"}, Bytes: bytes}
}

// assistantText is a shorthand for an assistant message — never a
// qualifying turn (wrong role), included in fixtures purely to exercise
// that non-user messages are skipped.
func assistantText(hash string, bytes int) RouterReplayMessage {
	return RouterReplayMessage{Role: "assistant", Hash: hash, BlockTypes: []string{"text"}, Bytes: bytes}
}

// toolResultOnly is a shorthand for a role=="user" message carrying ONLY a
// tool_result block (no text) — the "exclude tool_result-only" case from
// isQualifyingUserTurn.
func toolResultOnly(hash string, bytes int) RouterReplayMessage {
	return RouterReplayMessage{Role: "user", Hash: hash, BlockTypes: []string{"tool_result"}, Bytes: bytes, ToolResultIDs: []string{"tr1"}}
}

// turnFixtureSessions builds two sessions exercising every
// isQualifyingUserTurn exclusion plus multi-instance turn ordering:
//
//   - both sessions' first request opens with the SAME message hash
//     ("shared-msg", a role=="user" text message) — shared across sessions
//     (count==2) so it must NEVER qualify as a turn despite being
//     role==user with a text block.
//   - each session's main instance accumulates 3 genuine user turns
//     (u1,u2,u3) interleaved with assistant replies (never turns) and, for
//     session 0 only, a tool_result-only message (never a turn, even
//     though role=="user").
//   - session 0 has a SECOND instance ("s0-sub") appearing AFTER the main
//     instance in the Instances slice, contributing one more turn
//     (s0-sub-u1) — verifies turn indices are session-global and ordered
//     by instance/request/message file order, not per-instance.
func turnFixtureSessions() []RouterReplaySession {
	mkMainInstance := func(id, prefix string, includeToolResultOnly bool) RouterReplayInstance {
		req1 := RouterReplayRequest{
			RequestID: 1,
			Messages:  []RouterReplayMessage{userText("shared-msg", 60)},
		}
		req2Msgs := []RouterReplayMessage{
			userText("shared-msg", 60),
			assistantText(prefix+"-a1", 80),
			userText(prefix+"-u1", 100),
		}
		req2 := RouterReplayRequest{RequestID: 2, Messages: req2Msgs}

		req3Msgs := append(append([]RouterReplayMessage{}, req2Msgs...),
			assistantText(prefix+"-a2", 80),
			userText(prefix+"-u2", 100),
		)
		req3 := RouterReplayRequest{RequestID: 3, Messages: req3Msgs}

		req4Msgs := append([]RouterReplayMessage{}, req3Msgs...)
		if includeToolResultOnly {
			req4Msgs = append(req4Msgs, toolResultOnly(prefix+"-tool1", 40))
		}
		req4 := RouterReplayRequest{RequestID: 4, Messages: req4Msgs}

		req5Msgs := append(append([]RouterReplayMessage{}, req4Msgs...),
			assistantText(prefix+"-a3", 80),
			userText(prefix+"-u3", 100),
		)
		req5 := RouterReplayRequest{RequestID: 5, Messages: req5Msgs}

		return RouterReplayInstance{
			InstanceID: id,
			Role:       "main",
			Requests:   []RouterReplayRequest{req1, req2, req3, req4, req5},
		}
	}

	subInstance := RouterReplayInstance{
		InstanceID: "s0-sub",
		Role:       "sub-agent",
		Requests: []RouterReplayRequest{
			{RequestID: 6, Messages: []RouterReplayMessage{userText("s0-sub-u1", 90)}},
		},
	}

	s0 := RouterReplaySession{
		SessionID: "s0",
		Instances: []RouterReplayInstance{
			mkMainInstance("s0-main", "s0", true),
			subInstance,
		},
	}
	s1 := RouterReplaySession{
		SessionID: "s1",
		Instances: []RouterReplayInstance{
			mkMainInstance("s1-main", "s1", false),
		},
	}
	return []RouterReplaySession{s0, s1}
}

func TestComputeSessionTurnHashesIdentifiesQualifyingTurns(t *testing.T) {
	path := writeReplayV3File(t, turnFixtureSessions())
	counts, err := computeBlockSessionCounts(path, nil, 0)
	if err != nil {
		t.Fatalf("computeBlockSessionCounts: %v", err)
	}
	if got := counts["shared-msg"]; got != 2 {
		t.Fatalf("counts[shared-msg] = %d, want 2 (referenced by both sessions)", got)
	}

	turnHashes, err := computeSessionTurnHashes(path, nil, 0, counts)
	if err != nil {
		t.Fatalf("computeSessionTurnHashes: %v", err)
	}
	if len(turnHashes) != 2 {
		t.Fatalf("len(turnHashes) = %d, want 2 sessions", len(turnHashes))
	}

	wantS0 := []string{"s0-u1", "s0-u2", "s0-u3", "s0-sub-u1"}
	if !equalStrSlices(turnHashes[0], wantS0) {
		t.Errorf("session 0 turnHashes = %v, want %v", turnHashes[0], wantS0)
	}
	wantS1 := []string{"s1-u1", "s1-u2", "s1-u3"}
	if !equalStrSlices(turnHashes[1], wantS1) {
		t.Errorf("session 1 turnHashes = %v, want %v", turnHashes[1], wantS1)
	}

	for _, session := range turnHashes {
		for _, h := range session {
			if h == "shared-msg" {
				t.Error("shared-msg (count==2) qualified as a turn — cross-session-shared content must never be stamped")
			}
			if strings.Contains(h, "tool1") {
				t.Errorf("tool_result-only hash %q qualified as a turn", h)
			}
			if strings.Contains(h, "-a") {
				t.Errorf("assistant hash %q qualified as a turn", h)
			}
		}
	}
}

func TestComputeSessionTurnHashesRespectsFilters(t *testing.T) {
	path := writeReplayV3File(t, turnFixtureSessions())
	counts, err := computeBlockSessionCounts(path, nil, 0)
	if err != nil {
		t.Fatalf("computeBlockSessionCounts: %v", err)
	}

	t.Run("sessionLimit caps to the first N sessions", func(t *testing.T) {
		turnHashes, err := computeSessionTurnHashes(path, nil, 1, counts)
		if err != nil {
			t.Fatalf("computeSessionTurnHashes: %v", err)
		}
		if len(turnHashes) != 1 {
			t.Fatalf("len(turnHashes) = %d, want 1 (sessionLimit=1)", len(turnHashes))
		}
	})

	t.Run("allowed index set restricts to those sessions only", func(t *testing.T) {
		// counts computed globally (both sessions), but the turn-hash pass
		// only walks session index 1 (s1).
		allowed := map[int]bool{1: true}
		turnHashes, err := computeSessionTurnHashes(path, allowed, 0, counts)
		if err != nil {
			t.Fatalf("computeSessionTurnHashes: %v", err)
		}
		if len(turnHashes) != 1 {
			t.Fatalf("len(turnHashes) = %d, want 1 (only index 1 allowed)", len(turnHashes))
		}
		want := []string{"s1-u1", "s1-u2", "s1-u3"}
		if !equalStrSlices(turnHashes[0], want) {
			t.Errorf("turnHashes[0] = %v, want %v (session s1)", turnHashes[0], want)
		}
	})
}

func equalStrSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestBuildSessionTurnUUIDsDeterminism verifies buildSessionTurnUUIDs
// matches the dataset path's determinism contract: same seed -> same
// per-session-per-turn UUID assignment; different seed -> different
// assignment; every UUID unique across the whole run; owner correctly
// reverse-maps each UUID to its issuing session.
func TestBuildSessionTurnUUIDsDeterminism(t *testing.T) {
	turnCounts := []int{3, 2, 4}
	setsA, ownerA := buildSessionTurnUUIDs(turnCounts, 42)
	setsB, ownerB := buildSessionTurnUUIDs(turnCounts, 42)
	if len(setsA) != 3 || len(setsB) != 3 {
		t.Fatalf("expected 3 sets each, got %d and %d", len(setsA), len(setsB))
	}
	for i := range setsA {
		if len(setsA[i]) != turnCounts[i] || len(setsB[i]) != turnCounts[i] {
			t.Fatalf("session %d: expected %d UUIDs, got %v / %v", i, turnCounts[i], setsA[i], setsB[i])
		}
		for j := range setsA[i] {
			if setsA[i][j] != setsB[i][j] {
				t.Errorf("session %d turn %d: same seed produced different UUIDs: %q vs %q", i, j, setsA[i][j], setsB[i][j])
			}
		}
	}
	if len(ownerA) != len(ownerB) {
		t.Errorf("owner map sizes differ: %d vs %d", len(ownerA), len(ownerB))
	}

	setsC, _ := buildSessionTurnUUIDs(turnCounts, 43)
	same := true
	for i := range setsA {
		if len(setsA[i]) > 0 && len(setsC[i]) > 0 && setsA[i][0] != setsC[i][0] {
			same = false
		}
	}
	if same {
		t.Error("different seeds produced an identical UUID assignment")
	}

	seen := map[string]bool{}
	for i, set := range setsA {
		for _, u := range set {
			if seen[u] {
				t.Errorf("uuid %q assigned to more than one turn", u)
			}
			seen[u] = true
			if got := ownerA[u]; got != i {
				t.Errorf("owner[%q] = %d, want %d", u, got, i)
			}
		}
	}

	if sets, owner := buildSessionTurnUUIDs(nil, 42); sets != nil || len(owner) != 0 {
		t.Errorf("buildSessionTurnUUIDs(nil, ...) = (%v, %v), want (nil, empty)", sets, owner)
	}
}

// TestBuildSessionTurnUUIDsScalesWithTurnCount verifies each session gets
// EXACTLY turnCounts[i] UUIDs (one per turn, no floor/multiplier — unlike
// the retired per-session-N scheme) and a zero-turn session gets none.
func TestBuildSessionTurnUUIDsScalesWithTurnCount(t *testing.T) {
	turnCounts := []int{0, 1, 5}
	sets, owner := buildSessionTurnUUIDs(turnCounts, 7)
	if len(sets[0]) != 0 {
		t.Errorf("session 0 (0 turns): len = %d, want 0", len(sets[0]))
	}
	if len(sets[1]) != 1 {
		t.Errorf("session 1 (1 turn): len = %d, want 1", len(sets[1]))
	}
	if len(sets[2]) != 5 {
		t.Errorf("session 2 (5 turns): len = %d, want 5", len(sets[2]))
	}
	if len(owner) != 6 {
		t.Errorf("len(owner) = %d, want 6 (1+5 issued UUIDs)", len(owner))
	}
}

// newFixturePoster builds a replayPoster wired up exactly as
// runRouterReplayInstance does (see replay_router.go), for a single session
// with nTurns turns, without touching HTTP/newReplayPoster.
func newFixturePoster(nTurns int, seed int64, reciteEveryRequest bool) *replayPoster {
	turnHashes := make([]string, nTurns)
	for i := range turnHashes {
		turnHashes[i] = fmt.Sprintf("h%d", i)
	}
	sets, owner := buildSessionTurnUUIDs([]int{nTurns}, seed)
	hashToTurn := make(map[string]int, nTurns)
	for i, h := range turnHashes {
		hashToTurn[h] = i
	}
	return &replayPoster{
		uuidEnabled:        true,
		sessionIdx:         0,
		allUUIDSets:        sets,
		turnHashes:         turnHashes,
		hashToTurn:         hashToTurn,
		owner:              owner,
		reciteEveryRequest: reciteEveryRequest,
	}
}

// visibleTurnsRequest builds a RouterReplayRequest whose Messages carry the
// qualifying-turn hashes h0..h(n-1) (in order), simulating a growing
// conversation history at turn n.
func visibleTurnsRequest(n int) RouterReplayRequest {
	msgs := make([]RouterReplayMessage, 0, n)
	for i := 0; i < n; i++ {
		msgs = append(msgs, userText(fmt.Sprintf("h%d", i), 50))
	}
	return RouterReplayRequest{Messages: msgs}
}

// TestBuildInjectionWindowSelection verifies buildInjection's recite window:
// first (visible) turn + up to 3 most-recent turns EXCLUDING the current
// (highest-index visible) turn, deduplicated, capped at 4 — and that
// StampByHash always covers EVERY visible turn, not just the window.
// Exercises the edge cases at turns 1, 2, 3 explicitly (D1/D2/D3 in the
// plan) plus the steady-state 4-cap and window-sliding behavior beyond it.
func TestBuildInjectionWindowSelection(t *testing.T) {
	p := newFixturePoster(6, 1, true)

	cases := []struct {
		turn       int      // 1-based "current turn" being requested
		wantLabels []string // expected inj.ReciteLabels
	}{
		{1, []string{"turn-1"}},
		{2, []string{"turn-1"}},
		{3, []string{"turn-1", "turn-2"}},
		{4, []string{"turn-1", "turn-2", "turn-3"}},
		{5, []string{"turn-1", "turn-2", "turn-3", "turn-4"}},
		{6, []string{"turn-1", "turn-3", "turn-4", "turn-5"}},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("turn-%d", c.turn), func(t *testing.T) {
			req := visibleTurnsRequest(c.turn)
			inj := p.buildInjection(req, false)
			if inj == nil {
				t.Fatal("buildInjection returned nil, want a non-nil injection")
			}
			if !equalStrSlices(inj.ReciteLabels, c.wantLabels) {
				t.Errorf("ReciteLabels = %v, want %v", inj.ReciteLabels, c.wantLabels)
			}
			if len(inj.ReciteLabels) > 4 {
				t.Errorf("ReciteLabels len = %d, want <= 4", len(inj.ReciteLabels))
			}
			if len(inj.ReciteUUIDs) != len(inj.ReciteLabels) {
				t.Fatalf("ReciteUUIDs len = %d, want %d (matching ReciteLabels)", len(inj.ReciteUUIDs), len(inj.ReciteLabels))
			}
			// Every UUID in the window must be this session's own (index 0).
			for i, u := range inj.ReciteUUIDs {
				if p.owner[u] != 0 {
					t.Errorf("ReciteUUIDs[%d] = %q owned by session %d, want session 0", i, u, p.owner[u])
				}
			}
			// StampByHash must cover EVERY visible turn (h0..h(turn-1)), not
			// just the recite window — this is what keeps every turn's
			// marker warm in KV regardless of whether it's being recited
			// this request.
			if len(inj.StampByHash) != c.turn {
				t.Errorf("StampByHash covers %d turns, want %d (every visible turn)", len(inj.StampByHash), c.turn)
			}
			for i := 0; i < c.turn; i++ {
				h := fmt.Sprintf("h%d", i)
				stamp, ok := inj.StampByHash[h]
				if !ok {
					t.Errorf("StampByHash missing visible turn hash %q", h)
					continue
				}
				if stamp.Idx != i {
					t.Errorf("StampByHash[%q].Idx = %d, want %d", h, stamp.Idx, i)
				}
				if stamp.Label != fmt.Sprintf("turn-%d", i+1) {
					t.Errorf("StampByHash[%q].Label = %q, want turn-%d", h, stamp.Label, i+1)
				}
			}
		})
	}
}

// TestBuildInjectionNilCases verifies buildInjection degrades to "no
// injection" (nil) rather than panicking: disabled, session index out of
// range, session with zero turns, and a request with no visible qualifying
// turn at all.
func TestBuildInjectionNilCases(t *testing.T) {
	t.Run("uuidEnabled false", func(t *testing.T) {
		p := newFixturePoster(3, 1, true)
		p.uuidEnabled = false
		if got := p.buildInjection(visibleTurnsRequest(2), false); got != nil {
			t.Errorf("buildInjection = %v, want nil (disabled)", got)
		}
	})
	t.Run("sessionIdx out of range", func(t *testing.T) {
		p := newFixturePoster(3, 1, true)
		p.sessionIdx = 5
		if got := p.buildInjection(visibleTurnsRequest(2), false); got != nil {
			t.Errorf("buildInjection = %v, want nil (sessionIdx out of range)", got)
		}
	})
	t.Run("zero-turn session", func(t *testing.T) {
		p := newFixturePoster(0, 1, true)
		if got := p.buildInjection(visibleTurnsRequest(0), false); got != nil {
			t.Errorf("buildInjection = %v, want nil (no turns)", got)
		}
	})
	t.Run("no qualifying turn visible in this request", func(t *testing.T) {
		p := newFixturePoster(3, 1, true)
		req := RouterReplayRequest{Messages: []RouterReplayMessage{assistantText("unrelated", 30)}}
		if got := p.buildInjection(req, false); got != nil {
			t.Errorf("buildInjection = %v, want nil (nothing visible)", got)
		}
	})
}

// TestBuildInjectionRecite verifies Recite = reciteEveryRequest ||
// isLastRequest, independent of window selection.
func TestBuildInjectionRecite(t *testing.T) {
	req := visibleTurnsRequest(2)

	pAlways := newFixturePoster(3, 1, true)
	if inj := pAlways.buildInjection(req, false); inj == nil || !inj.Recite {
		t.Error("reciteEveryRequest=true, isLastRequest=false: expected Recite=true")
	}

	pFinalOnly := newFixturePoster(3, 1, false)
	if inj := pFinalOnly.buildInjection(req, false); inj == nil || inj.Recite {
		t.Error("reciteEveryRequest=false, isLastRequest=false: expected Recite=false")
	}
	if inj := pFinalOnly.buildInjection(req, true); inj == nil || !inj.Recite {
		t.Error("reciteEveryRequest=false, isLastRequest=true: expected Recite=true")
	}
}

// TestWireInjectionDeterminism verifies that, for a fixed set of turn
// stamps, buildOpenAIChatCompletionsBody / buildAnthropicMessagesBody
// produce byte-identical bodies across repeated calls (same request + same
// injection in -> same bytes out) and across two DIFFERENT requests that
// both carry the SAME turn message (the within-session cache-reuse
// property), while two DIFFERENT sessions' turn stamps diverge the body.
func TestWireInjectionDeterminism(t *testing.T) {
	docs := strings.Repeat("wire-injection-docs ", 100)
	stampA := map[string]turnStamp{"msg1": {Idx: 0, UUID: "uuid-session-A", Label: "turn-1"}}
	stampB := map[string]turnStamp{"msg1": {Idx: 0, UUID: "uuid-session-B", Label: "turn-1"}}
	req := RouterReplayRequest{
		InputTokens: 500,
		Messages:    []RouterReplayMessage{userText("msg1", 100)},
	}

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
		injA := &uuidInjection{StampByHash: stampA}
		injA2 := &uuidInjection{StampByHash: stampA}
		injB := &uuidInjection{StampByHash: stampB}

		bodyA1 := build(req, injA)
		bodyA2 := build(req, injA2)
		bodyB := build(req, injB)

		if string(bodyA1) != string(bodyA2) {
			t.Errorf("%s: identical injection produced different bytes", kind)
		}
		if string(bodyA1) == string(bodyB) {
			t.Errorf("%s: different sessions' turn stamps produced identical bytes", kind)
		}
		if !strings.Contains(string(bodyA1), "uuid-session-A") {
			t.Errorf("%s: body missing session A's own turn UUID", kind)
		}
		if strings.Contains(string(bodyA1), "uuid-session-B") {
			t.Errorf("%s: body A leaked session B's turn UUID into the wire body", kind)
		}

		// A SECOND, different request that repeats the SAME turn message
		// (msg1) must emit the byte-identical stamped content for it.
		req2 := RouterReplayRequest{
			InputTokens: 500,
			Messages: []RouterReplayMessage{
				userText("msg1", 100),
				assistantText("msg2", 60),
			},
		}
		bodyA3 := build(req2, injA)
		var p1, p3 map[string]interface{}
		if err := json.Unmarshal(bodyA1, &p1); err != nil {
			t.Fatalf("unmarshal bodyA1: %v", err)
		}
		if err := json.Unmarshal(bodyA3, &p3); err != nil {
			t.Fatalf("unmarshal bodyA3: %v", err)
		}
		msgs1, _ := p1["messages"].([]interface{})
		msgs3, _ := p3["messages"].([]interface{})
		if len(msgs1) == 0 || len(msgs3) == 0 {
			t.Fatalf("%s: expected non-empty messages in both bodies", kind)
		}
		first1, _ := json.Marshal(msgs1[0])
		first3, _ := json.Marshal(msgs3[0])
		if string(first1) != string(first3) {
			t.Errorf("%s: turn msg1's stamped content diverged across two different requests:\n%s\nvs\n%s", kind, first1, first3)
		}
	}
}

// TestInlineMarkerAppendedToTurnMessage verifies the injected marker's exact
// shape ("\n\n[turn-N id: <uuid>]") lands inside the stamped message's OWN
// text content (both wire body and canonical text), and that an unrelated
// message in the same request is left untouched.
func TestInlineMarkerAppendedToTurnMessage(t *testing.T) {
	docs := strings.Repeat("inline-marker-docs ", 100)
	stampByHash := map[string]turnStamp{"stamped-msg": {Idx: 3, UUID: "abc-uuid", Label: "turn-4"}}
	inj := &uuidInjection{StampByHash: stampByHash}
	req := RouterReplayRequest{
		InputTokens: 500,
		Messages: []RouterReplayMessage{
			userText("stamped-msg", 100),
			assistantText("other-msg", 80),
		},
	}
	wantMarker := "\n\n[turn-4 id: abc-uuid]"

	body, canonical, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !strings.Contains(canonical, wantMarker) {
		t.Errorf("canonical text missing marker %q", wantMarker)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	msgs, _ := parsed["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	// Compare against the UNMARSHALED content (fmt.Sprintf, not
	// json.Marshal) so a "\n" in the marker is a real newline byte, not the
	// two-character JSON escape sequence "\\n".
	first, _ := msgs[0].(map[string]interface{})
	firstContent := fmt.Sprintf("%v", first["content"])
	if !strings.Contains(firstContent, wantMarker) {
		t.Errorf("stamped message content missing marker: %s", firstContent)
	}
	second, _ := msgs[1].(map[string]interface{})
	secondContent := fmt.Sprintf("%v", second["content"])
	if strings.Contains(secondContent, "abc-uuid") {
		t.Errorf("marker leaked into the unrelated (unstamped) message: %s", secondContent)
	}
}

// TestCacheFidelitySharedBlockUnaffectedByInjection verifies the core
// fidelity invariant: a message hash that is NOT a key in StampByHash
// (standing in for a cross-session-shared, count>1 block — see
// isQualifyingUserTurn, which excludes those from ever being stamped) emits
// byte-identical content whether or not UUID injection is active elsewhere
// in the SAME request.
func TestCacheFidelitySharedBlockUnaffectedByInjection(t *testing.T) {
	docs := strings.Repeat("fidelity-docs ", 100)
	req := RouterReplayRequest{
		InputTokens: 500,
		Messages: []RouterReplayMessage{
			userText("shared-unstamped-msg", 100), // stands in for a count>1 block
			userText("own-msg", 80),
		},
	}

	bodyNoInj, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, nil)
	if err != nil {
		t.Fatalf("build (no injection): %v", err)
	}
	inj := &uuidInjection{StampByHash: map[string]turnStamp{"own-msg": {Idx: 0, UUID: "own-uuid", Label: "turn-1"}}}
	bodyWithInj, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, inj)
	if err != nil {
		t.Fatalf("build (with injection): %v", err)
	}

	var parsedNoInj, parsedWithInj map[string]interface{}
	if err := json.Unmarshal(bodyNoInj, &parsedNoInj); err != nil {
		t.Fatalf("unmarshal (no injection): %v", err)
	}
	if err := json.Unmarshal(bodyWithInj, &parsedWithInj); err != nil {
		t.Fatalf("unmarshal (with injection): %v", err)
	}
	msgsNoInj, _ := parsedNoInj["messages"].([]interface{})
	msgsWithInj, _ := parsedWithInj["messages"].([]interface{})
	if len(msgsNoInj) != 2 || len(msgsWithInj) != 2 {
		t.Fatalf("expected 2 messages in both bodies, got %d and %d", len(msgsNoInj), len(msgsWithInj))
	}

	sharedNoInj, _ := json.Marshal(msgsNoInj[0])
	sharedWithInj, _ := json.Marshal(msgsWithInj[0])
	if string(sharedNoInj) != string(sharedWithInj) {
		t.Errorf("unstamped (shared) message diverged with injection active elsewhere:\nwithout: %s\nwith:    %s", sharedNoInj, sharedWithInj)
	}

	ownWithInj, _ := json.Marshal(msgsWithInj[1])
	if !strings.Contains(string(ownWithInj), "own-uuid") {
		t.Errorf("stamped message missing its own marker: %s", ownWithInj)
	}
}

// TestFindLeakedUUIDsByOwner exercises the router path's O(response)
// contamination scanner: a known OTHER-session UUID is flagged with the
// correct series index; the caller's OWN uuid is never flagged; a
// UUID-shaped string with no entry in owner is silently ignored; and
// multiple leaks are returned in deterministic (scan-order) sequence.
// Mirrors validateReplayResponse/FindLeakedUUIDs' semantics for the dataset
// path, but via the reverse uuid->owner map instead of iterating every
// session's UUID set.
func TestFindLeakedUUIDsByOwner(t *testing.T) {
	owner := map[string]int{
		"11111111-1111-1111-1111-111111111111": 0,
		"22222222-2222-2222-2222-222222222222": 0,
		"33333333-3333-3333-3333-333333333333": 1,
		"44444444-4444-4444-4444-444444444444": 2,
	}

	t.Run("other-session uuid flagged with correct series", func(t *testing.T) {
		resp := "here it is: 33333333-3333-3333-3333-333333333333"
		got := findLeakedUUIDsByOwner(resp, "", 0, owner)
		if len(got) != 1 || !strings.Contains(got[0], "33333333-3333-3333-3333-333333333333") || !strings.Contains(got[0], "series=1") {
			t.Errorf("got %v, want one entry naming series=1", got)
		}
	})

	t.Run("own uuid never flagged", func(t *testing.T) {
		resp := "own ids: 11111111-1111-1111-1111-111111111111, 22222222-2222-2222-2222-222222222222"
		got := findLeakedUUIDsByOwner(resp, "", 0, owner)
		if len(got) != 0 {
			t.Errorf("got %v, want none (both are the caller's own uuids)", got)
		}
	})

	t.Run("unowned uuid-shaped string ignored", func(t *testing.T) {
		resp := "random: 99999999-9999-9999-9999-999999999999"
		got := findLeakedUUIDsByOwner(resp, "", 0, owner)
		if len(got) != 0 {
			t.Errorf("got %v, want none (not a real stamp)", got)
		}
	})

	t.Run("thinking channel scanned too", func(t *testing.T) {
		got := findLeakedUUIDsByOwner("no leak here", "but here: 44444444-4444-4444-4444-444444444444", 0, owner)
		if len(got) != 1 || !strings.Contains(got[0], "series=2") {
			t.Errorf("got %v, want one entry naming series=2 (found in thinking)", got)
		}
	})

	t.Run("deterministic scan order for multiple leaks", func(t *testing.T) {
		resp := "first 44444444-4444-4444-4444-444444444444 then 33333333-3333-3333-3333-333333333333"
		got1 := findLeakedUUIDsByOwner(resp, "", 0, owner)
		got2 := findLeakedUUIDsByOwner(resp, "", 0, owner)
		if len(got1) != 2 {
			t.Fatalf("got %v, want 2 leaked entries", got1)
		}
		for i := range got1 {
			if got1[i] != got2[i] {
				t.Errorf("non-deterministic order: %v vs %v", got1, got2)
			}
		}
		if !strings.Contains(got1[0], "series=2") || !strings.Contains(got1[1], "series=1") {
			t.Errorf("got %v, want series=2 before series=1 (scan order)", got1)
		}
	})

	t.Run("empty owner map returns nil", func(t *testing.T) {
		if got := findLeakedUUIDsByOwner("11111111-1111-1111-1111-111111111111", "", 0, map[string]int{}); got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})
}

// TestFirstLineConformity exercises firstLineConformity (the output-
// conformity check --replay-inject-uuids scores, mirroring the coherency
// eval's matchesExpectedUUIDList/ExactMatch): pass on an exact ordered
// comma-joined first line; fail on missing, reordered, or chatty first
// lines; pass when line 1 is exact even though LATER lines contain filler
// (the whole point of front-loading the ask to line 1 while forced-output
// keeps generating). expected here plays the role of inj.ReciteUUIDs — the
// request's recite WINDOW, not the session's full turn history.
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
		{"below floor, more uuids (capped at 4), recite -> raised higher", 5, true, 4},
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

// TestReciteFloorScalesWithN verifies the recite floor grows with numUUIDs
// up to the window cap (4) — more UUIDs to recite on the first line needs a
// bigger budget — but is now BOUNDED, unlike the retired per-session-N
// scheme where a long session's floor grew without bound.
func TestReciteFloorScalesWithN(t *testing.T) {
	small := replayReciteFloorTokens(1)
	large := replayReciteFloorTokens(4)
	if large <= small {
		t.Errorf("replayReciteFloorTokens(4) = %d, want > replayReciteFloorTokens(1) = %d", large, small)
	}
}

// TestMaxTokensFloorAppliedInWireBuilders verifies the floor is actually
// wired into both body builders' emitted max_tokens when a recite
// injection is present and the original/recorded budget is tiny — the
// scenario a real tool-call-only turn would hit. numUUIDs is now
// len(inj.ReciteUUIDs) (the window, capped at 4), not the old per-session N.
func TestMaxTokensFloorAppliedInWireBuilders(t *testing.T) {
	docs := strings.Repeat("floor-docs ", 100)
	req := RouterReplayRequest{
		InputTokens:  500,
		OutputTokens: 5, // tiny recorded budget -- would truncate the recite line
		Messages:     []RouterReplayMessage{userText("msg1", 100)},
	}
	reciteUUIDs := []string{"uuid-floor-0", "uuid-floor-1"}
	inj := &uuidInjection{
		StampByHash:  map[string]turnStamp{"msg1": {Idx: 0, UUID: reciteUUIDs[0], Label: "turn-1"}},
		Recite:       true,
		ReciteLabels: []string{"turn-1"},
		ReciteUUIDs:  reciteUUIDs,
	}
	wantFloor := float64(replayReciteFloorTokens(len(reciteUUIDs)))

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
