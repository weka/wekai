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

// TestBuildSessionUUIDsIdentifiesQualifyingTurns: a turn is a role=="user"
// message with a text block. Assistant and system messages, and messages
// with no text block, never qualify. Order is first appearance across the
// session's instances/requests/messages, and a hash repeated by the growing
// history contributes one turn, not one per request.
func TestBuildSessionUUIDsIdentifiesQualifyingTurns(t *testing.T) {
	sess := RouterReplaySession{
		SessionID: "s1",
		Instances: []RouterReplayInstance{{
			InstanceID: "i1",
			Requests: []RouterReplayRequest{
				{Messages: []RouterReplayMessage{userText("t1", 10)}},
				{Messages: []RouterReplayMessage{
					userText("t1", 10), // repeated history: still turn 0
					assistantText("a1", 10),
					{Role: "user", Hash: "tool-only", BlockTypes: []string{"tool_result"}},
					userText("t2", 10),
				}},
			},
		}},
	}
	su := buildSessionUUIDs(sess, 1)
	if su == nil {
		t.Fatal("buildSessionUUIDs returned nil for a session with qualifying turns")
	}
	if len(su.uuids) != 2 {
		t.Fatalf("got %d turns, want 2 (t1, t2)", len(su.uuids))
	}
	if su.hashToTurn["t1"] != 0 || su.hashToTurn["t2"] != 1 {
		t.Errorf("turn indices = %v, want t1->0 t2->1", su.hashToTurn)
	}
	for _, h := range []string{"a1", "tool-only"} {
		if _, ok := su.hashToTurn[h]; ok {
			t.Errorf("%q qualified as a turn; only user messages with a text block may", h)
		}
	}
}

// TestBuildSessionUUIDsNilWithoutQualifyingTurns: a session with nothing to
// stamp must produce no view, so buildInjection short-circuits rather than
// carrying an empty one through every request.
func TestBuildSessionUUIDsNilWithoutQualifyingTurns(t *testing.T) {
	sess := RouterReplaySession{Instances: []RouterReplayInstance{{
		Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{assistantText("a1", 10)}}},
	}}}
	if su := buildSessionUUIDs(sess, 1); su != nil {
		t.Errorf("got %d turns for a session with no qualifying turn, want nil", len(su.uuids))
	}
}

// TestMarkerIsHashDerivedNotSessionDerived is the invariant that removes the
// need for any corpus pass.
//
// Two sessions carrying the same block must stamp it identically. That is
// what keeps a block they shared in the capture shared on the server's
// prefix cache — the fidelity property the old design bought by scanning the
// whole file to find such blocks and refusing to stamp them. Derived from
// the hash, it holds by construction, for blocks nobody has looked for.
func TestMarkerIsHashDerivedNotSessionDerived(t *testing.T) {
	mk := func(id string, msgs ...RouterReplayMessage) RouterReplaySession {
		return RouterReplaySession{SessionID: id, Instances: []RouterReplayInstance{{
			Requests: []RouterReplayRequest{{Messages: msgs}},
		}}}
	}
	// The shared block sits at a DIFFERENT turn index in each session, so a
	// marker keyed by position rather than by content would diverge here.
	a := buildSessionUUIDs(mk("a", userText("shared", 10), userText("a-only", 10)), 9)
	b := buildSessionUUIDs(mk("b", userText("b-only", 10), userText("shared", 10)), 9)

	if a.uuids[a.hashToTurn["shared"]] != b.uuids[b.hashToTurn["shared"]] {
		t.Errorf("shared block stamped differently per session: %q vs %q — the prefix the two "+
			"sessions shared in the capture no longer collides on the server",
			a.uuids[a.hashToTurn["shared"]], b.uuids[b.hashToTurn["shared"]])
	}
	if a.uuids[a.hashToTurn["a-only"]] == b.uuids[b.hashToTurn["b-only"]] {
		t.Error("distinct blocks produced the same marker, which would make a genuine leak unattributable")
	}
}

// TestUUIDForHashShapeAndSeed: markers must be recognisable on the way back
// out of a response (uuidRe) and must vary with the seed, or two runs
// sharing a corpus could not be told apart.
func TestUUIDForHashShapeAndSeed(t *testing.T) {
	u := uuidForHash("block-a", 42)
	if !uuidRe.MatchString(u) {
		t.Fatalf("uuidForHash produced %q, which the response scanner would never match", u)
	}
	if u != uuidForHash("block-a", 42) {
		t.Error("uuidForHash is not deterministic")
	}
	if u == uuidForHash("block-a", 43) {
		t.Error("uuidForHash ignores the seed; two runs over one corpus would mint identical markers")
	}
	if u == uuidForHash("block-b", 42) {
		t.Error("uuidForHash ignores the hash")
	}
}

// TestUUIDRegistryHoldsWhileAnySessionDoes: the refcount is what makes a
// block shared by concurrent sessions visible without a corpus pass, and it
// is also the detection window — a marker must stay live until the LAST
// holder finishes, or one session retiring would blind the check for the
// others still running.
func TestUUIDRegistryHoldsWhileAnySessionDoes(t *testing.T) {
	r := newUUIDRegistry()
	shared := uuidForHash("shared", 1)
	aOnly := uuidForHash("a-only", 1)

	r.Acquire([]string{shared, aOnly}, 1)
	r.Acquire([]string{shared}, 2)

	if e := r.lookup(shared); e == nil || e.n.Load() != 2 {
		t.Fatalf("shared marker refcount = %v, want 2", e)
	}
	if e := r.lookup(shared); e.series != 1 {
		t.Errorf("shared marker labelled series %d, want its first holder (1)", e.series)
	}

	r.Release([]string{shared, aOnly}) // session 1 finishes
	if r.lookup(aOnly) != nil {
		t.Error("marker held by nobody is still live; the window would grow without bound")
	}
	if e := r.lookup(shared); e == nil || e.n.Load() != 1 {
		t.Fatalf("shared marker dropped while session 2 still holds it: %v", e)
	}

	r.Release([]string{shared}) // session 2 finishes
	if r.lookup(shared) != nil {
		t.Error("shared marker outlived its last holder")
	}
	if live, peak := r.Stats(); live != 0 || peak != 2 {
		t.Errorf("Stats() = (%d live, %d peak), want (0, 2) — peak is what a run reports as the "+
			"width of the window it actually checked against", live, peak)
	}
}

// TestUUIDRegistryAcquireIsIdempotentPerCall: a session whose turn list
// repeats a marker must take one hold, not several, or its own release
// leaves the entry stranded and the live set never shrinks.
func TestUUIDRegistryAcquireIsIdempotentPerCall(t *testing.T) {
	r := newUUIDRegistry()
	u := uuidForHash("h", 1)
	r.Acquire([]string{u, u, u}, 1)
	if e := r.lookup(u); e == nil || e.n.Load() != 1 {
		t.Fatalf("refcount = %v after acquiring the same marker three times in one call, want 1", e)
	}
	r.Release([]string{u, u})
	if r.lookup(u) != nil {
		t.Error("marker survived its holder's release")
	}
}

// newFixturePoster builds a replayPoster wired up exactly as
// runRouterReplayInstance does (see replay_router.go), for a single session
// (sessionIdx 0) with nTurns turns, without touching HTTP/newReplayPoster.
// All UUID state is global/read-only on the poster (see replay_router_post.go);
// callers pass the returned view into buildInjection, exactly as
// runRouterReplaySession does.
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

func newFixturePoster(nTurns int, seed int64, reciteEveryRequest bool) (*replayPoster, *sessionUUIDs) {
	su := &sessionUUIDs{hashToTurn: map[string]int{}}
	for i := 0; i < nTurns; i++ {
		h := fmt.Sprintf("h%d", i)
		su.hashToTurn[h] = i
		su.uuids = append(su.uuids, uuidForHash(h, seed))
	}
	reg := newUUIDRegistry()
	reg.Acquire(su.uuids, 1)
	return &replayPoster{
		uuidEnabled:        true,
		uuidSeed:           seed,
		registry:           reg,
		reciteEveryRequest: reciteEveryRequest,
	}, su
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
	p, su := newFixturePoster(6, 1, true)

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
			inj := p.buildInjection(req, su, false)
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
			// Every UUID in the window must be one this session carries.
			own := map[string]bool{}
			for _, u := range su.uuids {
				own[u] = true
			}
			for i, u := range inj.ReciteUUIDs {
				if !own[u] {
					t.Errorf("ReciteUUIDs[%d] = %q is not a marker this session carries", i, u)
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
		p, su := newFixturePoster(3, 1, true)
		p.uuidEnabled = false
		if got := p.buildInjection(visibleTurnsRequest(2), su, false); got != nil {
			t.Errorf("buildInjection = %v, want nil (disabled)", got)
		}
	})
	t.Run("no session view", func(t *testing.T) {
		p, _ := newFixturePoster(3, 1, true)
		if got := p.buildInjection(visibleTurnsRequest(2), nil, false); got != nil {
			t.Errorf("buildInjection = %v, want nil (session has no markers)", got)
		}
	})
	t.Run("zero-turn session", func(t *testing.T) {
		p, su := newFixturePoster(0, 1, true)
		if got := p.buildInjection(visibleTurnsRequest(0), su, false); got != nil {
			t.Errorf("buildInjection = %v, want nil (no turns)", got)
		}
	})
	t.Run("no qualifying turn visible in this request", func(t *testing.T) {
		p, su := newFixturePoster(3, 1, true)
		req := RouterReplayRequest{Messages: []RouterReplayMessage{assistantText("unrelated", 30)}}
		if got := p.buildInjection(req, su, false); got != nil {
			t.Errorf("buildInjection = %v, want nil (nothing visible)", got)
		}
	})
}

// TestBuildInjectionRecite verifies Recite = reciteEveryRequest ||
// isLastRequest, independent of window selection.
func TestBuildInjectionRecite(t *testing.T) {
	req := visibleTurnsRequest(2)

	pAlways, suA := newFixturePoster(3, 1, true)
	if inj := pAlways.buildInjection(req, suA, false); inj == nil || !inj.Recite {
		t.Error("reciteEveryRequest=true, isLastRequest=false: expected Recite=true")
	}

	pFinalOnly, suF := newFixturePoster(3, 1, false)
	if inj := pFinalOnly.buildInjection(req, suF, false); inj == nil || inj.Recite {
		t.Error("reciteEveryRequest=false, isLastRequest=false: expected Recite=false")
	}
	if inj := pFinalOnly.buildInjection(req, suF, true); inj == nil || !inj.Recite {
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
				body, _, err = buildOpenAIChatCompletionsBody(r, docs, "model", "", 0, false, 0, inj)
			} else {
				body, _, err = buildAnthropicMessagesBody(r, docs, "model", "", 0, false, 0, inj)
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

	body, canonical, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, 0, inj)
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

	bodyNoInj, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, 0, nil)
	if err != nil {
		t.Fatalf("build (no injection): %v", err)
	}
	inj := &uuidInjection{StampByHash: map[string]turnStamp{"own-msg": {Idx: 0, UUID: "own-uuid", Label: "turn-1"}}}
	bodyWithInj, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, 0, inj)
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

// TestFindLeakedUUIDsRouterPath exercises the router path's contamination scanner.
//
// The comparison is against what THIS request sent, not against a
// session-wide assignment, which is what lets markers be derived from block
// hashes: a block two sessions genuinely share yields one marker that both
// hold legitimately, and neither is flagged for reciting it.
func TestFindLeakedUUIDsRouterPath(t *testing.T) {
	reg := newUUIDRegistry()
	mine := uuidForHash("mine", 1)
	theirs := uuidForHash("theirs", 1)
	shared := uuidForHash("shared", 1)
	reg.Acquire([]string{mine, shared}, 1)
	reg.Acquire([]string{theirs, shared}, 2)

	own := map[string]bool{mine: true, shared: true}

	t.Run("own marker is never flagged", func(t *testing.T) {
		if got := findLeakedUUIDs("here it is: "+mine, "", own, reg); len(got) != 0 {
			t.Errorf("findLeakedUUIDs = %v, want empty", got)
		}
	})
	t.Run("shared marker this request carried is not flagged", func(t *testing.T) {
		// The point of hash-derived markers: session 2 also holds this one,
		// but this request sent it, so reciting it is correct behaviour.
		if got := findLeakedUUIDs("recall: "+shared, "", own, reg); len(got) != 0 {
			t.Errorf("findLeakedUUIDs = %v, want empty — a block both sessions carry is not a leak", got)
		}
	})
	t.Run("live marker this request did not carry is flagged", func(t *testing.T) {
		got := findLeakedUUIDs("stray "+theirs, "", own, reg)
		if len(got) != 1 || got[0] != theirs+"(series=2)" {
			t.Errorf("findLeakedUUIDs = %v, want [%s(series=2)]", got, theirs)
		}
	})
	t.Run("shared marker this request did NOT carry names the shared hold", func(t *testing.T) {
		got := findLeakedUUIDs("stray "+shared, "", map[string]bool{mine: true}, reg)
		if len(got) != 1 || got[0] != shared+"(series=1,shared)" {
			t.Errorf("findLeakedUUIDs = %v, want [%s(series=1,shared)] — the label must not "+
				"claim a single owner for a marker two sessions hold", got, shared)
		}
	})
	t.Run("unknown UUID-shaped string is ignored", func(t *testing.T) {
		if got := findLeakedUUIDs("deadbeef-0000-4000-8000-000000000000", "", own, reg); len(got) != 0 {
			t.Errorf("findLeakedUUIDs = %v, want empty (no live session holds it)", got)
		}
	})
	t.Run("retired marker is ignored", func(t *testing.T) {
		// The registry IS the detection window: once every holder finishes,
		// its markers stop being recognisable. Narrower than "no session
		// ever leaked", and the run reports the window it checked.
		reg.Release([]string{theirs, shared})
		if got := findLeakedUUIDs("stray "+theirs, "", own, reg); len(got) != 0 {
			t.Errorf("findLeakedUUIDs = %v, want empty once the holding session retired", got)
		}
	})
	t.Run("thinking is scanned too", func(t *testing.T) {
		reg2 := newUUIDRegistry()
		reg2.Acquire([]string{theirs}, 3)
		if got := findLeakedUUIDs("clean", "leaked "+theirs, own, reg2); len(got) != 1 {
			t.Errorf("findLeakedUUIDs = %v, want the leak found in the thinking blob", got)
		}
	})
	t.Run("multiple leaks in scan order", func(t *testing.T) {
		reg3 := newUUIDRegistry()
		a := uuidForHash("a", 1)
		b := uuidForHash("b", 1)
		reg3.Acquire([]string{a}, 4)
		reg3.Acquire([]string{b}, 5)
		got := findLeakedUUIDs(b+" then "+a, "", nil, reg3)
		if len(got) != 2 || got[0] != b+"(series=5)" || got[1] != a+"(series=4)" {
			t.Errorf("findLeakedUUIDs = %v, want scan order [%s(series=5) %s(series=4)]", got, b, a)
		}
	})
}

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

	anthBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, 0, inj)
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

	openaiBody, _, err := buildOpenAIChatCompletionsBody(req, docs, "model", "", 0, false, 0, inj)
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
	plainBody, _, err := buildAnthropicMessagesBody(req, docs, "model", "", 0, false, 0, nil)
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
