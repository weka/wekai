package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// writeIndexedReplayFile writes a replay-v5 file the way replay-prepare does:
// header, one session per line (session i starts i minutes after base), then
// the index trailer. pad lengthens every session id so the trailer can be made
// larger than the first tail window.
func writeIndexedReplayFile(t *testing.T, sessions, pad int) (string, *ReplayIndex) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	hb, _ := json.Marshal(map[string]any{
		"_schema": "replay-v5", "name": "t", "source": "t",
		"summary": map[string]any{"sessions": sessions},
	})
	fmt.Fprintf(f, "%s\n", hb)
	off := int64(len(hb)) + 1
	base := time.Date(2026, 5, 12, 8, 0, 0, 0, time.UTC)
	idx := &ReplayIndex{Schema: ReplayIndexSchema}
	for i := range sessions {
		id := fmt.Sprintf("s%d%s", i, strings.Repeat("x", pad))
		start := base.Add(time.Duration(i) * time.Minute).Format(time.RFC3339Nano)
		b, _ := json.Marshal(map[string]any{"session_id": id, "start_ts": start, "instances": []any{}})
		fmt.Fprintf(f, "%s\n", b)
		idx.Sessions = append(idx.Sessions, ReplayIndexSession{
			SessionID: id, Offset: off, Length: int64(len(b)), StartTs: start, EndTs: start, Requests: 1,
		})
		off += int64(len(b)) + 1
	}
	ib, _ := json.Marshal(idx)
	fmt.Fprintf(f, "%s\n", ib)
	return path, idx
}

func pullIDs(t *testing.T, st *routerReplayStream, n int) (ids []string, passes []int) {
	t.Helper()
	for range n {
		sess, _, ok := st.Pull(context.Background())
		if !ok {
			t.Fatalf("stream drained after %d pulls", len(ids))
		}
		ids = append(ids, strings.TrimRight(sess.SessionID, "x"))
		passes = append(passes, sess.pass)
	}
	return ids, passes
}

// The trailer is found whether it fits the first tail window or needs several
// doublings, and a file without one reads as unindexed rather than failing.
func TestReadReplayIndex(t *testing.T) {
	for _, tc := range []struct {
		name          string
		sessions, pad int
	}{{"fits first window", 3, 0}, {"needs doubling", 400, 300}} {
		t.Run(tc.name, func(t *testing.T) {
			old := replayIndexInitialRead
			replayIndexInitialRead = 32 << 10
			defer func() { replayIndexInitialRead = old }()
			path, want := writeIndexedReplayFile(t, tc.sessions, tc.pad)
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			got, err := readReplayIndex(f)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("index mismatch: got %d sessions, want %d", len(got.Sessions), len(want.Sessions))
			}
		})
	}
	f, err := os.Open(writeReplayFile(t, 2))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if got, err := readReplayIndex(f); got != nil || err != nil {
		t.Fatalf("unindexed file: got %v, %v; want nil, nil", got, err)
	}
}

// writeWeeklyReplayFile writes an indexed corpus of n sessions, one a day, so
// the last week is the final 8 sessions (day n-8 is exactly seven days before
// the last).
func writeWeeklyReplayFile(t *testing.T, n int) string {
	t.Helper()
	path, idx := writeIndexedReplayFile(t, n, 0)
	base := time.Date(2026, 5, 12, 8, 0, 0, 0, time.UTC)
	for i := range idx.Sessions {
		idx.Sessions[i].StartTs = base.AddDate(0, 0, i).Format(time.RFC3339Nano)
	}
	// Rewrite only the trailer: session lines keep their offsets.
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body := b[:len(b)-1]
	body = body[:strings.LastIndexByte(string(body), '\n')+1]
	ib, _ := json.Marshal(idx)
	if err := os.WriteFile(path, append(append(body, ib...), '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func ids(n ...int) []string {
	out := make([]string, len(n))
	for i, v := range n {
		out[i] = fmt.Sprintf("s%d", v)
	}
	return out
}

func TestSelectionStrategies(t *testing.T) {
	path := writeWeeklyReplayFile(t, 10) // last week = s2..s9
	for _, tc := range []struct {
		strategy, from string
		pulls          int
		want           []string
		passes         []int
	}{
		// The default: whole laps newest first, wrapping to the end again.
		{"", "", 12, ids(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8),
			[]int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
		// last-week: the last week, then whole laps from the top.
		{ReplayStrategyLastWeek, "", 20, append(ids(2, 3, 4, 5, 6, 7, 8, 9), ids(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1)...),
			[]int{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2}},
		{ReplayStrategyBeginning, "", 12, ids(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1), nil},
		{ReplayStrategyReverse, "", 12, ids(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8), nil},
		{ReplayStrategyBeginning, "7", 5, ids(7, 8, 9, 0, 1), nil},
		{ReplayStrategyReverse, "2", 5, ids(2, 1, 0, 9, 8), nil},
		// s5 started 05-17 08:00; noon that day is after it and before s6.
		{ReplayStrategyBeginning, "2026-05-17T12:00:00Z", 2, ids(6, 7), nil},
		{ReplayStrategyReverse, "2026-05-17T12:00:00Z", 2, ids(5, 4), nil},
		{ReplayStrategyLastWeek, "8", 4, ids(8, 9, 0, 1), nil},
		{"", "8", 4, ids(8, 7, 6, 5), nil},
	} {
		t.Run(tc.strategy+"/"+tc.from, func(t *testing.T) {
			st, err := openRouterReplayStream(path, routerReplayStreamOpts{
				ChanCap: 1, Reuse: true, Strategy: tc.strategy, From: tc.from,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer st.Close()
			got, passes := pullIDs(t, st, tc.pulls)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("sessions = %v, want %v", got, tc.want)
			}
			if tc.passes != nil && !reflect.DeepEqual(passes, tc.passes) {
				t.Errorf("passes = %v, want %v", passes, tc.passes)
			}
		})
	}
}

// Random plays every session once per lap before any repeats, in a different
// order each lap.
func TestRandomStrategyCoversEachLap(t *testing.T) {
	path, _ := writeIndexedReplayFile(t, 50, 0)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1, Reuse: true, Strategy: ReplayStrategyRandom})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	got, passes := pullIDs(t, st, 100)
	for lap := range 2 {
		seen := map[string]bool{}
		for i := lap * 50; i < (lap+1)*50; i++ {
			seen[got[i]] = true
			if passes[i] != lap {
				t.Fatalf("pull %d on pass %d, want %d", i, passes[i], lap)
			}
		}
		if len(seen) != 50 {
			t.Errorf("lap %d played %d distinct sessions, want 50", lap, len(seen))
		}
	}
	if reflect.DeepEqual(got[:50], got[50:]) {
		t.Error("both laps played the same permutation")
	}
	if reflect.DeepEqual(got[:50], ids(seq(50)...)) {
		t.Error("random lap is in file order")
	}
}

func seq(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// Without reuse the run is the first lap, so its session count is what the
// progress display divides by.
func TestLastWeekWithoutReuseIsTheWeek(t *testing.T) {
	path := writeWeeklyReplayFile(t, 10)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1, Strategy: ReplayStrategyLastWeek})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if st.Total() != 8 {
		t.Errorf("Total = %d, want 8", st.Total())
	}
	got, _ := pullIDs(t, st, 8)
	if !reflect.DeepEqual(got, ids(2, 3, 4, 5, 6, 7, 8, 9)) {
		t.Errorf("sessions = %v", got)
	}
	if _, _, ok := st.Pull(context.Background()); ok {
		t.Error("stream without reuse kept going after the week")
	}
}

// An index filter and a session cap still apply on the indexed path, and
// fileIdx stays the session's position in the file whatever the order.
func TestIndexedStreamFilterAndLimit(t *testing.T) {
	path, _ := writeIndexedReplayFile(t, 5, 0)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{
		ChanCap: 1, Strategy: ReplayStrategyReverse,
		AllowedIndices: map[int]bool{0: true, 2: true, 4: true}, SessionLimit: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	var got []int
	for {
		sess, _, ok := st.Pull(context.Background())
		if !ok {
			break
		}
		got = append(got, sess.fileIdx)
	}
	if want := []int{4, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("file indices = %v, want %v", got, want)
	}
}

// An unindexed file replays from the beginning under the default, and refuses
// a strategy or start it cannot honour instead of silently ignoring it.
func TestUnindexedFileStrategies(t *testing.T) {
	path := writeReplayFile(t, 3)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := pullIDs(t, st, 3)
	st.Close()
	if want := ids(0, 1, 2); !reflect.DeepEqual(got, want) {
		t.Errorf("default on unindexed file = %v, want %v", got, want)
	}
	for _, o := range []routerReplayStreamOpts{
		{Strategy: ReplayStrategyLastWeek}, {Strategy: ReplayStrategyReverse}, {Strategy: ReplayStrategyRandom}, {From: "1"},
	} {
		if st, err := openRouterReplayStream(path, o); err == nil {
			st.Close()
			t.Errorf("opts %+v on an unindexed file opened; want an error", o)
		}
	}
}

func TestReplayFirstStartRejects(t *testing.T) {
	_, idx := writeIndexedReplayFile(t, 3, 0)
	for _, tc := range []struct{ strategy, from string }{
		{ReplayStrategyBeginning, "3"},
		{ReplayStrategyBeginning, "-1"},
		{ReplayStrategyReverse, "2026-05-12T07:00:00Z"},
		{ReplayStrategyBeginning, "2026-05-12T09:00:00Z"},
		{ReplayStrategyBeginning, "yesterday"},
		{ReplayStrategyRandom, "1"},
	} {
		if i, err := replayFirstStart(idx, tc.strategy, tc.from); err == nil {
			t.Errorf("%s from %q resolved to %d; want an error", tc.strategy, tc.from, i)
		}
	}
}

func TestIsReplayIndexLine(t *testing.T) {
	ib, _ := json.Marshal(&ReplayIndex{Schema: ReplayIndexSchema})
	if !IsReplayIndexLine(ib) {
		t.Errorf("%s not recognized as the index", ib)
	}
	if IsReplayIndexLine([]byte(`{"session_id":"s0","instances":[]}`)) {
		t.Error("a session line was taken for the index")
	}
}
