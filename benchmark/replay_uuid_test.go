package benchmark

import (
	"strings"
	"testing"
)

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
