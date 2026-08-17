package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Replaying the corpus a second time is only correct if the stamp is per PASS
// and never per session. Two properties, and they pull against each other:
//
//   - WITHIN a pass, sessions that shared a prefix in the capture must still
//     share it, or the sharing topology the benchmark exists to measure is
//     destroyed — and a run with a flattened topology still produces a cache
//     hit rate, just a meaningless one.
//   - ACROSS passes, a session must not hit its own earlier pass, or the second
//     pass measures nothing and drags the aggregate toward 100%.
//
// A per-session stamp satisfies the second and destroys the first. That is the
// failure this file exists to catch.

func writeReplayFile(t *testing.T, sessions int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	hdr := map[string]any{
		"_schema": "replay-v3", "name": "t", "source": "t",
		"summary": map[string]any{"sessions": sessions},
	}
	b, _ := json.Marshal(hdr)
	fmt.Fprintf(f, "%s\n", b)
	for i := range sessions {
		sess := map[string]any{
			"session_id": fmt.Sprintf("s%d", i),
			"start_ts":   "2026-05-12T08:50:06Z",
			"instances":  []any{},
		}
		b, _ := json.Marshal(sess)
		fmt.Fprintf(f, "%s\n", b)
	}
	return path
}

func TestReuseReplaysTheCorpusAgain(t *testing.T) {
	path := writeReplayFile(t, 3)
	st, err := openRouterReplayStream(path, 1, 0, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	seen := map[string][]int{} // session id -> passes it was handed out on
	for range 9 {              // three passes' worth
		sess, _, ok := st.Pull(context.Background())
		if !ok {
			t.Fatalf("stream drained after %d pulls; reuse must not run out", len(seen))
		}
		seen[sess.SessionID] = append(seen[sess.SessionID], sess.pass)
	}
	for id, passes := range seen {
		if len(passes) != 3 {
			t.Errorf("session %s handed out %d times, want 3", id, len(passes))
		}
		for i, p := range passes {
			if p != i {
				t.Errorf("session %s pass %d reported as %d; the generation must advance once per "+
					"lap, not per session", id, i, p)
			}
		}
	}
}

func TestWithoutReuseTheStreamStillDrains(t *testing.T) {
	path := writeReplayFile(t, 2)
	st, err := openRouterReplayStream(path, 1, 0, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	for range 2 {
		if _, _, ok := st.Pull(context.Background()); !ok {
			t.Fatal("drained early")
		}
	}
	if _, _, ok := st.Pull(context.Background()); ok {
		t.Error("stream kept going with reuse off; the underfill abort depends on it draining")
	}
}

// stampOf is the production rule, isolated: one value per pass, applied to
// every session on that pass.
func stampOf(runID string, pass int, noStamp bool) string {
	if noStamp {
		return ""
	}
	if pass > 0 {
		return fmt.Sprintf("%s-p%d", runID, pass+1)
	}
	return runID
}

// TestPassTopologyMatchesAcrossPasses is the guard that matters. It compares the
// WHOLE pass's collision structure rather than sampling a pair, because the two
// simpler properties can both hold over a topology that has been flattened.
func TestPassTopologyMatchesAcrossPasses(t *testing.T) {
	const runID = "RUN"
	sessions := []string{"a", "b", "c", "d"}

	// Collision structure of a pass: which sessions carry the same stamp, and
	// therefore share any prefix they shared in the capture.
	topology := func(pass int) map[string][]string {
		byStamp := map[string][]string{}
		for _, s := range sessions {
			byStamp[stampOf(runID, pass, false)] = append(byStamp[stampOf(runID, pass, false)], s)
		}
		out := map[string][]string{}
		for _, group := range byStamp {
			for _, s := range group {
				out[s] = group
			}
		}
		return out
	}

	p1, p2 := topology(0), topology(1)
	for _, s := range sessions {
		if len(p1[s]) != len(p2[s]) {
			t.Fatalf("session %s shares with %d others on pass 1 and %d on pass 2; the topology "+
				"was flattened, and a flattened topology still yields a hit rate", s, len(p1[s])-1, len(p2[s])-1)
		}
		for i := range p1[s] {
			if p1[s][i] != p2[s][i] {
				t.Errorf("session %s's sharing group differs between passes: %v vs %v",
					s, p1[s], p2[s])
			}
		}
	}
	// Every session on a pass must share ONE stamp — that is what makes the
	// topology survive. Four sessions in one group here.
	if len(p1["a"]) != len(sessions) {
		t.Errorf("pass 1 split %d sessions into groups; the stamp must be uniform across the pass, "+
			"never per session", len(sessions))
	}
}

func TestPassesDoNotShareAKeyspace(t *testing.T) {
	const runID = "RUN"
	s1, s2, s3 := stampOf(runID, 0, false), stampOf(runID, 1, false), stampOf(runID, 2, false)
	if s1 == s2 || s2 == s3 || s1 == s3 {
		t.Errorf("passes share a stamp (%q, %q, %q); pass two would replay into pass one's cache "+
			"entries and drag the hit rate toward 100%%", s1, s2, s3)
	}
	if s1 != runID {
		t.Errorf("pass 1 stamp = %q, want the run's own %q — a single-pass run must be identical "+
			"to what it was before reuse existed", s1, runID)
	}
	if stampOf(runID, 3, true) != "" {
		t.Error("--replay-no-stamp must still suppress the stamp on every pass")
	}
}
