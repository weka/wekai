package benchmark

import (
	"strings"
	"testing"
)

// syntheticMixedRoleConvs builds a small synthetic conversation set with mixed
// From roles (human/gpt/tool/system, including one LEADING system turn and one
// STRAY mid-conversation system turn) — enough to exercise both
// --replay-uuid-mode values.
func syntheticMixedRoleConvs() []Conversation {
	return []Conversation{
		{ID: "c0", Turns: []HermesTurn{
			{From: "system", Value: "c0 leading system prompt"},
			{From: "human", Value: "c0 h1"},
			{From: "gpt", Value: "c0 g1"},
			{From: "tool", Value: "c0 t1"},
			{From: "human", Value: "c0 h2"},
			{From: "gpt", Value: "c0 g2"},
		}},
		{ID: "c1", Turns: []HermesTurn{
			{From: "human", Value: "c1 h1"}, // no leading system turn at all
			{From: "gpt", Value: "c1 g1"},
			{From: "system", Value: "c1 stray system"}, // NOT at index 0
			{From: "human", Value: "c1 h2"},
			{From: "gpt", Value: "c1 g2"},
		}},
	}
}

func TestBuildReplayUUIDSets(t *testing.T) {
	convs := syntheticMixedRoleConvs()

	t.Run("determinism under fixed seed", func(t *testing.T) {
		a := buildReplayUUIDSets(convs, 42, 2, "human")
		b := buildReplayUUIDSets(convs, 42, 2, "human")
		if len(a) != len(b) {
			t.Fatalf("length mismatch: %d vs %d", len(a), len(b))
		}
		for i := range a {
			if len(a[i]) != len(b[i]) {
				t.Fatalf("conv %d length mismatch: %d vs %d", i, len(a[i]), len(b[i]))
			}
			for j := range a[i] {
				if a[i][j] != b[i][j] {
					t.Errorf("conv %d uuid %d mismatch across identical-seed calls: %q vs %q", i, j, a[i][j], b[i][j])
				}
			}
		}
	})

	t.Run("different seeds diverge", func(t *testing.T) {
		a := buildReplayUUIDSets(convs, 1, 2, "human")
		b := buildReplayUUIDSets(convs, 2, 2, "human")
		same := true
		for i := range a {
			for j := range a[i] {
				if a[i][j] != b[i][j] {
					same = false
				}
			}
		}
		if same {
			t.Errorf("different seeds produced identical uuid sets")
		}
	})

	t.Run("disjoint across conversations", func(t *testing.T) {
		sets := buildReplayUUIDSets(convs, 7, 2, "all-non-gpt")
		owner := make(map[string]int)
		for ci, uuids := range sets {
			for _, u := range uuids {
				if prevCi, ok := owner[u]; ok {
					t.Errorf("uuid %q appears in both conversation %d and conversation %d", u, prevCi, ci)
				}
				owner[u] = ci
			}
		}
	})

	t.Run("count == injectableTurns*perTurn, mode human", func(t *testing.T) {
		const perTurn = 3
		sets := buildReplayUUIDSets(convs, 1, perTurn, "human")
		// c0: human turns = h1, h2 -> 2. c1: human turns = h1, h2 -> 2.
		if got, want := len(sets[0]), 2*perTurn; got != want {
			t.Errorf("conv0 human mode: len = %d, want %d", got, want)
		}
		if got, want := len(sets[1]), 2*perTurn; got != want {
			t.Errorf("conv1 human mode: len = %d, want %d", got, want)
		}
	})

	t.Run("count == injectableTurns*perTurn, mode all-non-gpt", func(t *testing.T) {
		const perTurn = 2
		sets := buildReplayUUIDSets(convs, 1, perTurn, "all-non-gpt")
		// c0: leading system turn (index 0) is stripped before counting, so
		// injectable turns are h1, tool t1, h2 -> 3.
		if got, want := len(sets[0]), 3*perTurn; got != want {
			t.Errorf("conv0 all-non-gpt mode: len = %d, want %d", got, want)
		}
		// c1: no leading system turn to strip; injectable turns are h1, the
		// STRAY mid-conversation system turn, h2 -> 3.
		if got, want := len(sets[1]), 3*perTurn; got != want {
			t.Errorf("conv1 all-non-gpt mode: len = %d, want %d", got, want)
		}
	})
}

func TestReplayTurnInjectable(t *testing.T) {
	tests := []struct {
		from        string
		wantHuman   bool
		wantAllNGpt bool
	}{
		{"human", true, true},
		{"gpt", false, false},
		{"tool", false, true},
		{"system", false, true},
	}
	for _, tt := range tests {
		t.Run(tt.from, func(t *testing.T) {
			turn := HermesTurn{From: tt.from, Value: "x"}
			if got := replayTurnInjectable(turn, "human"); got != tt.wantHuman {
				t.Errorf("replayTurnInjectable(%q, \"human\") = %v, want %v", tt.from, got, tt.wantHuman)
			}
			if got := replayTurnInjectable(turn, "all-non-gpt"); got != tt.wantAllNGpt {
				t.Errorf("replayTurnInjectable(%q, \"all-non-gpt\") = %v, want %v", tt.from, got, tt.wantAllNGpt)
			}
		})
	}
}

func TestInjectUUIDMarker(t *testing.T) {
	original := "what is the weather today?"
	uuids := []string{"uuid-aaa", "uuid-bbb"}

	got := injectUUIDMarker(original, uuids)

	if !strings.Contains(got, original) {
		t.Errorf("injectUUIDMarker() dropped the original turn text: %q", got)
	}
	for _, u := range uuids {
		if !strings.Contains(got, u) {
			t.Errorf("injectUUIDMarker() result missing uuid %q: %q", u, got)
		}
	}

	// No uuids -> turnValue passed through unchanged.
	if got := injectUUIDMarker(original, nil); got != original {
		t.Errorf("injectUUIDMarker(_, nil) = %q, want unchanged %q", got, original)
	}
}

// TestComputeInScopeAtEachGptTurn walks a synthetic turn sequence and asserts
// the snapshot at each gpt-turn boundary equals the expected prefix union of
// uuids assigned to injectable turns seen so far — the invariant
// replay.go's real turn loop maintains inline (see the comment there pointing
// back at this test).
func TestComputeInScopeAtEachGptTurn(t *testing.T) {
	turns := []HermesTurn{
		{From: "system", Value: "leading system prompt"}, // skipped entirely
		{From: "human", Value: "h1"},                     // injectable (human mode): uuids[0:2]
		{From: "gpt", Value: "g1"},                       // snapshot #0 -> uuids[0:2]
		{From: "tool", Value: "t1"},                      // NOT injectable in human mode
		{From: "human", Value: "h2"},                     // injectable: uuids[2:4]
		{From: "gpt", Value: "g2"},                       // snapshot #1 -> uuids[0:4]
		{From: "human", Value: "h3"},                     // injectable: uuids[4:6]
		{From: "gpt", Value: "g3"},                       // snapshot #2 -> uuids[0:6]
	}
	sets := []string{"u0", "u1", "u2", "u3", "u4", "u5"}
	const perTurn = 2

	got := computeInScopeAtEachGptTurn(turns, sets, perTurn, "human")
	want := [][]string{
		{"u0", "u1"},
		{"u0", "u1", "u2", "u3"},
		{"u0", "u1", "u2", "u3", "u4", "u5"},
	}
	if len(got) != len(want) {
		t.Fatalf("computeInScopeAtEachGptTurn() = %d snapshots, want %d", len(got), len(want))
	}
	for i := range want {
		if strings.Join(got[i], ",") != strings.Join(want[i], ",") {
			t.Errorf("snapshot[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Mutating a returned snapshot must not corrupt a previous snapshot
	// (defensive copy per call) or later ones.
	got[0][0] = "MUTATED"
	got2 := computeInScopeAtEachGptTurn(turns, sets, perTurn, "human")
	if got2[0][0] != "u0" {
		t.Errorf("computeInScopeAtEachGptTurn() snapshots are not independently allocated: got2[0][0] = %q", got2[0][0])
	}

	// all-non-gpt mode additionally picks up the mid-conversation 'tool' turn,
	// so the second snapshot's union grows by one more perTurn slice than in
	// human mode (the leading system turn is still skipped).
	gotAll := computeInScopeAtEachGptTurn(turns, sets, perTurn, "all-non-gpt")
	if len(gotAll) != 3 {
		t.Fatalf("all-non-gpt mode: got %d snapshots, want 3", len(gotAll))
	}
	if len(gotAll[0]) != 2 {
		t.Errorf("all-non-gpt mode snapshot[0] = %v, want len 2", gotAll[0])
	}
	if len(gotAll[1]) != 6 {
		// h1 (2) + tool t1 (2) + h2 (2) = 6, vs 4 in human-only mode.
		t.Errorf("all-non-gpt mode snapshot[1] = %v, want len 6 (tool turn now counted)", gotAll[1])
	}
}

func TestValidateReplayResponse(t *testing.T) {
	allSets := [][]string{
		{"own-0", "own-1"},     // conversation 0 (this response's own conversation)
		{"other-0", "other-1"}, // conversation 1
	}

	t.Run("presence in content only", func(t *testing.T) {
		found, leaked := validateReplayResponse("SEEN_REFS: own-0,own-1", "", []string{"own-0", "own-1"}, 0, allSets)
		if len(found) != 2 || !found[0] || !found[1] {
			t.Errorf("found = %v, want both true", found)
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})

	t.Run("presence in thinking only", func(t *testing.T) {
		found, _ := validateReplayResponse("some unrelated prose", "I recall own-0 and own-1 from earlier", []string{"own-0", "own-1"}, 0, allSets)
		if !found[0] || !found[1] {
			t.Errorf("found = %v, want both true (present in thinking)", found)
		}
	})

	t.Run("presence in general prose (not the SEEN_REFS line)", func(t *testing.T) {
		found, _ := validateReplayResponse("Sure! Earlier you mentioned own-0, and also own-1 came up.", "", []string{"own-0", "own-1"}, 0, allSets)
		if !found[0] || !found[1] {
			t.Errorf("found = %v, want both true (Contains-based, no format requirement)", found)
		}
	})

	t.Run("PRESENCE_MISS: expected uuid absent from both", func(t *testing.T) {
		found, leaked := validateReplayResponse("I don't remember any ref ids.", "", []string{"own-0", "own-1"}, 0, allSets)
		if found[0] || found[1] {
			t.Errorf("found = %v, want both false", found)
		}
		if len(leaked) != 0 {
			t.Errorf("leaked = %v, want none", leaked)
		}
	})

	t.Run("CROSS_CONTAMINATION: another conversation's uuid appears", func(t *testing.T) {
		found, leaked := validateReplayResponse("SEEN_REFS: own-0,own-1,other-0", "", []string{"own-0", "own-1"}, 0, allSets)
		if !found[0] || !found[1] {
			t.Errorf("found = %v, want both true", found)
		}
		if len(leaked) != 1 || !strings.Contains(leaked[0], "other-0") || !strings.Contains(leaked[0], "series=1") {
			t.Errorf("leaked = %v, want one entry naming other-0 from series=1", leaked)
		}
	})
}

func TestFindLeakedUUIDs(t *testing.T) {
	allSets := [][]string{
		{"uuid-0-a", "uuid-0-b"}, // 0
		{"uuid-1-a", "uuid-1-b"}, // 1
		{"uuid-2-a", "uuid-2-b"}, // 2
	}

	t.Run("no leak: response only contains own uuids", func(t *testing.T) {
		got := FindLeakedUUIDs("uuid-0-a,uuid-0-b", "", 0, allSets)
		if len(got) != 0 {
			t.Errorf("expected no leaks, got %v", got)
		}
	})

	t.Run("single leak from another set", func(t *testing.T) {
		got := FindLeakedUUIDs("uuid-0-a,uuid-1-a", "", 0, allSets)
		if len(got) != 1 {
			t.Fatalf("expected 1 leak, got %v", got)
		}
		if !strings.Contains(got[0], "uuid-1-a") || !strings.Contains(got[0], "series=1") {
			t.Errorf("leak entry %q missing uuid or owning index", got[0])
		}
	})

	t.Run("multiple leaks, deterministic order", func(t *testing.T) {
		resp := "uuid-0-a,uuid-1-a,uuid-2-b"
		got1 := FindLeakedUUIDs(resp, "", 0, allSets)
		got2 := FindLeakedUUIDs(resp, "", 0, allSets)
		if len(got1) != 2 {
			t.Fatalf("expected 2 leaks, got %v", got1)
		}
		for i := range got1 {
			if got1[i] != got2[i] {
				t.Errorf("non-deterministic leak order: %v vs %v", got1, got2)
			}
		}
	})

	t.Run("leak detected in thinking, not just response", func(t *testing.T) {
		got := FindLeakedUUIDs("uuid-0-a", "I recall uuid-2-a from earlier", 0, allSets)
		if len(got) != 1 || !strings.Contains(got[0], "uuid-2-a") {
			t.Errorf("expected leak of uuid-2-a via thinking, got %v", got)
		}
	})

	t.Run("own index never reported as a leak", func(t *testing.T) {
		got := FindLeakedUUIDs("uuid-1-a,uuid-1-b", "", 1, allSets)
		if len(got) != 0 {
			t.Errorf("expected no leaks for own index, got %v", got)
		}
	})
}

func TestCapRecitedUUIDs(t *testing.T) {
	t.Run("unbounded budget (<=0) never trims", func(t *testing.T) {
		in := []string{"a", "b", "c"}
		got, trimmed := capRecitedUUIDs(in, 0)
		if trimmed {
			t.Errorf("expected no trim with maxOutputTokens<=0")
		}
		if len(got) != len(in) {
			t.Errorf("got %v, want unchanged %v", got, in)
		}
	})

	t.Run("small list well within budget is untouched", func(t *testing.T) {
		in := []string{"11111111-1111-1111-1111-111111111111"}
		got, trimmed := capRecitedUUIDs(in, 10000)
		if trimmed {
			t.Errorf("expected no trim for a single uuid against a large budget")
		}
		if len(got) != 1 {
			t.Errorf("got %v, want unchanged", got)
		}
	})

	t.Run("large list against a tiny budget is trimmed to the most recent entries", func(t *testing.T) {
		in := make([]string, 200)
		for i := range in {
			in[i] = "11111111-1111-1111-1111-11111111111" + string(rune('0'+i%10))
		}
		got, trimmed := capRecitedUUIDs(in, 20) // tiny budget forces a cap
		if !trimmed {
			t.Fatalf("expected trimming for a 200-uuid list against a 20-token budget")
		}
		if len(got) == 0 || len(got) >= len(in) {
			t.Fatalf("got %d entries, want a proper subset of %d", len(got), len(in))
		}
		// Kept entries must be the MOST RECENT (tail) of the input, in order.
		wantTail := in[len(in)-len(got):]
		for i := range got {
			if got[i] != wantTail[i] {
				t.Errorf("capRecitedUUIDs did not keep the most-recent tail: got[%d]=%q, want %q", i, got[i], wantTail[i])
			}
		}
	})
}
