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
	su := buildSessionUUIDs(sess, "stamp-1")
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
	if su := buildSessionUUIDs(sess, "stamp-1"); su != nil {
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
	a := buildSessionUUIDs(mk("a", userText("shared", 10), userText("a-only", 10)), "stamp-9")
	b := buildSessionUUIDs(mk("b", userText("b-only", 10), userText("shared", 10)), "stamp-9")

	if a.uuids[a.hashToTurn["shared"]] != b.uuids[b.hashToTurn["shared"]] {
		t.Errorf("shared block stamped differently per session: %q vs %q — the prefix the two "+
			"sessions shared in the capture no longer collides on the server",
			a.uuids[a.hashToTurn["shared"]], b.uuids[b.hashToTurn["shared"]])
	}
	if a.uuids[a.hashToTurn["a-only"]] == b.uuids[b.hashToTurn["b-only"]] {
		t.Error("distinct blocks produced the same marker, which would make a genuine leak unattributable")
	}
}

// TestUUIDForHashShapeAndStamp: markers must be recognisable on the way back
// out of a response (uuidRe) and must vary with the pass stamp, or two runs
// over one corpus — or two passes within one run — could not be told apart.
func TestUUIDForHashShapeAndStamp(t *testing.T) {
	u := uuidForHash("block-a", "stamp-42")
	if !uuidRe.MatchString(u) {
		t.Fatalf("uuidForHash produced %q, which the response scanner would never match", u)
	}
	if u != uuidForHash("block-a", "stamp-42") {
		t.Error("uuidForHash is not deterministic")
	}
	if u == uuidForHash("block-a", "stamp-43") {
		t.Error("uuidForHash ignores the stamp; two runs over one corpus would mint identical markers")
	}
	if u == uuidForHash("block-b", "stamp-42") {
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
	shared := uuidForHash("shared", "stamp-1")
	aOnly := uuidForHash("a-only", "stamp-1")

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
	if live, peak, sessions := r.Stats(); live != 0 || peak != 2 || sessions != 2 {
		t.Errorf("Stats() = (%d live, %d peak, %d peak sessions), want (0, 2, 2) — the peaks are what "+
			"a run reports as the width of the window it actually checked against", live, peak, sessions)
	}
}

// TestUUIDRegistryAcquireIsIdempotentPerCall: a session whose turn list
// repeats a marker must take one hold, not several, or its own release
// leaves the entry stranded and the live set never shrinks.
func TestUUIDRegistryAcquireIsIdempotentPerCall(t *testing.T) {
	r := newUUIDRegistry()
	u := uuidForHash("h", "stamp-1")
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

func newFixturePoster(nTurns int, stamp string) (*replayPoster, *sessionUUIDs) {
	su := &sessionUUIDs{hashToTurn: map[string]int{}}
	for i := 0; i < nTurns; i++ {
		h := fmt.Sprintf("h%d", i)
		su.hashToTurn[h] = i
		su.uuids = append(su.uuids, uuidForHash(h, stamp))
	}
	reg := newUUIDRegistry()
	reg.Acquire(su.uuids, 1)
	return &replayPoster{
		uuidEnabled: true,
		registry:    reg,
	}, su
}

// visibleTurnsRequest builds a RouterReplayRequest whose Messages carry the
// qualifying-turn hashes h0..h(n-1) (in order), simulating a growing
// conversation history at turn n.
func visibleTurnsRequest(n int) RouterReplayRequest {
	return visibleTurnsRequestBudget(n, 1000) // room for the full window
}

// visibleTurnsRequestBudget carries an explicit captured output budget, which
// is what now decides how many ids the request can be asked for.
func visibleTurnsRequestBudget(n, outputTokens int) RouterReplayRequest {
	msgs := make([]RouterReplayMessage, 0, n)
	for i := 0; i < n; i++ {
		msgs = append(msgs, userText(fmt.Sprintf("h%d", i), 50))
	}
	return RouterReplayRequest{Messages: msgs, OutputTokens: outputTokens}
}

// TestBuildInjectionWindowSelection verifies the recite window: the first
// visible turn, plus as many of the most-recent turns EXCLUDING the current
// one as the request's own captured output budget allows, up to
// reciteMaxRecent — and that StampByHash always covers EVERY visible turn,
// not just the window, so an unrecited turn still keeps its identity in KV.
func TestBuildInjectionWindowSelection(t *testing.T) {
	p, su := newFixturePoster(14, "fixture")

	for _, c := range []struct {
		turn       int      // 1-based "current turn" being requested
		wantLabels []string // expected inj.ReciteLabels
	}{
		{1, []string{"turn-1"}},
		{2, []string{"turn-1"}},
		{3, []string{"turn-1", "turn-2"}},
		{4, []string{"turn-1", "turn-2", "turn-3"}},
		{6, []string{"turn-1", "turn-2", "turn-3", "turn-4", "turn-5"}},
		// Past the cap the first turn stays pinned and the oldest middle
		// turns fall out, so coverage keeps sliding forward.
		{13, []string{"turn-1", "turn-3", "turn-4", "turn-5", "turn-6", "turn-7",
			"turn-8", "turn-9", "turn-10", "turn-11", "turn-12"}},
	} {
		t.Run(fmt.Sprintf("turn-%d", c.turn), func(t *testing.T) {
			req := visibleTurnsRequest(c.turn)
			inj := p.buildInjection(req, su)
			if inj == nil {
				t.Fatal("buildInjection returned nil, want a non-nil injection")
			}
			if !equalStrSlices(inj.ReciteLabels, c.wantLabels) {
				t.Errorf("ReciteLabels = %v, want %v", inj.ReciteLabels, c.wantLabels)
			}
			if len(inj.ReciteLabels) > reciteMaxRecent+1 {
				t.Errorf("ReciteLabels len = %d, want <= %d", len(inj.ReciteLabels), reciteMaxRecent+1)
			}
			if len(inj.ReciteUUIDs) != len(inj.ReciteLabels) {
				t.Fatalf("ReciteUUIDs len = %d, want %d", len(inj.ReciteUUIDs), len(inj.ReciteLabels))
			}
			own := map[string]bool{}
			for _, u := range su.uuids {
				own[u] = true
			}
			for i, u := range inj.ReciteUUIDs {
				if !own[u] {
					t.Errorf("ReciteUUIDs[%d] = %q is not a marker this session carries", i, u)
				}
			}
			if len(inj.StampByHash) != c.turn {
				t.Errorf("StampByHash covers %d turns, want all %d visible — an unrecited turn must "+
					"still carry its marker so it stays identifiable in KV", len(inj.StampByHash), c.turn)
			}
		})
	}
}

// TestBuildInjectionWindowFollowsTheCapturedBudget is the point of the
// rewrite: the ask shrinks to fit the budget the capture recorded, instead of
// the budget being raised to fit the ask.
func TestBuildInjectionWindowFollowsTheCapturedBudget(t *testing.T) {
	p, su := newFixturePoster(14, "fixture")

	for _, c := range []struct {
		name       string
		out        int
		wantLabels []string
		wantShort  bool
	}{
		{"no room for even one id", reciteReserveTokens + reciteTokensPerID - 1, nil, true},
		{"room for exactly one", reciteReserveTokens + reciteTokensPerID, []string{"turn-1"}, false},
		{"room for two", reciteReserveTokens + 2*reciteTokensPerID, []string{"turn-1", "turn-11"}, false},
		{"room for three", reciteReserveTokens + 3*reciteTokensPerID, []string{"turn-1", "turn-10", "turn-11"}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			inj := p.buildInjection(visibleTurnsRequestBudget(12, c.out), su)
			if inj == nil {
				t.Fatal("buildInjection returned nil; the inline markers must be sent regardless")
			}
			if inj.BudgetShort != c.wantShort {
				t.Errorf("BudgetShort = %v, want %v", inj.BudgetShort, c.wantShort)
			}
			if !equalStrSlices(inj.ReciteLabels, c.wantLabels) {
				t.Errorf("ReciteLabels = %v, want %v", inj.ReciteLabels, c.wantLabels)
			}
			if c.wantShort && len(inj.StampByHash) == 0 {
				t.Error("a budget-short request dropped its inline markers; the turn would lose its " +
					"identity for every later request that could afford to ask about it")
			}
		})
	}
}

// TestReciteCapacityNeverRaisesTheBudget: the estimate must be a function of
// the captured budget alone. A replay that edits max_tokens to fit its own
// instrumentation is measuring a workload nobody captured.
func TestReciteCapacityNeverRaisesTheBudget(t *testing.T) {
	if got := reciteCapacity(0); got != 0 {
		t.Errorf("reciteCapacity(0) = %d, want 0", got)
	}
	if got := reciteCapacity(reciteReserveTokens); got != 0 {
		t.Errorf("reciteCapacity(reserve) = %d, want 0: the reserve is what the prose needs before "+
			"any id fits", got)
	}
	if got := reciteCapacity(1 << 20); got != reciteMaxRecent+1 {
		t.Errorf("reciteCapacity(huge) = %d, want %d — the ask stays bounded on a deep session",
			got, reciteMaxRecent+1)
	}
	prev := 0
	for b := 0; b < 5000; b += 7 {
		if got := reciteCapacity(b); got < prev {
			t.Fatalf("reciteCapacity is not monotonic: %d tokens gave %d after %d", b, got, prev)
		} else {
			prev = got
		}
	}
}

// turn at all.
func TestBuildInjectionNilCases(t *testing.T) {
	t.Run("uuidEnabled false", func(t *testing.T) {
		p, su := newFixturePoster(3, "fixture")
		p.uuidEnabled = false
		if got := p.buildInjection(visibleTurnsRequest(2), su); got != nil {
			t.Errorf("buildInjection = %v, want nil (disabled)", got)
		}
	})
	t.Run("no session view", func(t *testing.T) {
		p, _ := newFixturePoster(3, "fixture")
		if got := p.buildInjection(visibleTurnsRequest(2), nil); got != nil {
			t.Errorf("buildInjection = %v, want nil (session has no markers)", got)
		}
	})
	t.Run("zero-turn session", func(t *testing.T) {
		p, su := newFixturePoster(0, "fixture")
		if got := p.buildInjection(visibleTurnsRequest(0), su); got != nil {
			t.Errorf("buildInjection = %v, want nil (no turns)", got)
		}
	})
	t.Run("no qualifying turn visible in this request", func(t *testing.T) {
		p, su := newFixturePoster(3, "fixture")
		req := RouterReplayRequest{Messages: []RouterReplayMessage{assistantText("unrelated", 30)}}
		if got := p.buildInjection(req, su); got != nil {
			t.Errorf("buildInjection = %v, want nil (nothing visible)", got)
		}
	})
}

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
	mine := uuidForHash("mine", "stamp-1")
	theirs := uuidForHash("theirs", "stamp-1")
	shared := uuidForHash("shared", "stamp-1")
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
		a := uuidForHash("a", "stamp-1")
		b := uuidForHash("b", "stamp-1")
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

// TestMarkersDifferPerPass covers --replay-reuse-sessions: the same session
// replayed a second time must mint different markers.
//
// A pass carries its own stamp so its content lands in a disjoint keyspace and
// cannot trivially hit the previous pass's cache entries. Keying markers on the
// RUN stamp instead of the PASS stamp would leave them identical across passes
// while the content around them changed — and two live passes of one session
// would then present as a single block shared by two sessions, which is exactly
// the case the scoring treats as legitimate. A real leak between passes would
// be counted as each pass holding its own marker.
func TestMarkersDifferPerPass(t *testing.T) {
	sess := RouterReplaySession{Instances: []RouterReplayInstance{{
		Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("turn-a", 10)}}},
	}}}

	// The stamps runRouterReplaySession derives: the run's own for pass 0, and
	// a suffixed one for each pass after it.
	const runID = "3fa1c2d4-0000-4000-8000-000000000000"
	p0 := buildSessionUUIDs(sess, runID)
	p1 := buildSessionUUIDs(sess, runID+"-p2")

	if p0.uuids[0] == p1.uuids[0] {
		t.Errorf("pass 1 and pass 2 of the same session minted the same marker %q — two live passes "+
			"would read as one block shared by two sessions, and a leak between them would be "+
			"scored as each holding its own", p0.uuids[0])
	}

	// Same pass, same marker: this is the property that keeps a block two
	// sessions shared in the capture colliding on the server's prefix cache.
	if again := buildSessionUUIDs(sess, runID); again.uuids[0] != p0.uuids[0] {
		t.Errorf("same pass produced different markers on two builds (%q vs %q) — sharing within a "+
			"pass would break and every session would look unique to the cache",
			p0.uuids[0], again.uuids[0])
	}
}

// TestRunIDIsAFunctionOfTheSeed: one printed number has to reproduce a run, and
// a run with no seed given has to be distinct from every other.
func TestRunIDIsAFunctionOfTheSeed(t *testing.T) {
	if runIDFromSeed(7) != runIDFromSeed(7) {
		t.Error("runIDFromSeed is not deterministic; --seed could not reproduce a run")
	}
	if runIDFromSeed(7) == runIDFromSeed(8) {
		t.Error("runIDFromSeed ignores the seed; every run would share content and start warm")
	}
	// Deliberately NOT UUID-shaped. The stamp sits above every marker in the
	// prompt, so a UUID-shaped one is a plausible wrong answer to "output the
	// id for this tag" — and it could be picked up by the contamination scan.
	if uuidRe.MatchString(runIDFromSeed(7)) {
		t.Errorf("run id %q is UUID-shaped: it is the first UUID in every prompt, and a model asked "+
			"for an id has every reason to reach for it", runIDFromSeed(7))
	}
	a, err := resolveRunSeed(0)
	if err != nil {
		t.Fatalf("resolveRunSeed: %v", err)
	}
	b, _ := resolveRunSeed(0)
	if a == b {
		t.Error("an unseeded run drew the same seed twice; two runs would share prefixes and the " +
			"second's cache hit rate would measure the first's leftovers")
	}
	if a < 0 || b < 0 {
		t.Errorf("drew a negative seed (%d, %d); it is printed for a human to paste back into --seed", a, b)
	}
	if got, err := resolveRunSeed(42); err != nil || got != 42 {
		t.Errorf("resolveRunSeed(42) = %d, %v; an explicit seed must be used verbatim", got, err)
	}
}
