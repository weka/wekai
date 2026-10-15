package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"time"
)

// ReplayIndexSchema marks the last line of a replay-v5 file: a table of where
// every session line sits in the file. It is the last line, not the first, so
// replay-prepare can write sessions as it builds them and only then emit the
// table.
const ReplayIndexSchema = "replay-index-v1"

// replayIndexMarker is how a line is recognized as the index without decoding
// it. The writer always emits _schema as the first key.
var replayIndexMarker = []byte(`{"_schema":"` + ReplayIndexSchema + `"`)

// ReplayIndex locates each session line so a reader can seek straight to one,
// which is what lets a replay start from the newest sessions (the end of the
// file) or from any session, instead of reading the file front to back.
type ReplayIndex struct {
	Schema   string               `json:"_schema"`
	Sessions []ReplayIndexSession `json:"sessions"`
}

// ReplayIndexSession is one session line: its byte span (Length excludes the
// newline), the capture window it covers, and how many requests it holds.
// StartTs is the session's first-seen time; EndTs is when its last request
// finished in the capture.
type ReplayIndexSession struct {
	SessionID string `json:"session_id"`
	Offset    int64  `json:"offset"`
	Length    int64  `json:"length"`
	StartTs   string `json:"start_ts"`
	EndTs     string `json:"end_ts"`
	Requests  int    `json:"requests"`
}

// IsReplayIndexLine reports whether a (newline-trimmed) line is the index
// trailer rather than a session.
func IsReplayIndexLine(line []byte) bool { return bytes.HasPrefix(line, replayIndexMarker) }

// replayIndexInitialRead is the first tail window readReplayIndex tries. The
// 5,441-session corpus has a 1.0 MB index, so 2 MiB reads it in one go with
// room to grow; a larger index costs one more read per doubling.
var replayIndexInitialRead int64 = 2 << 20

// readReplayIndex loads the index trailer of a replay file. It reads the last
// window of the file and looks for the start of the final line; when the window
// holds no newline before it, the window doubles. A file whose last line is not
// an index (replay-v3/v4) returns nil, nil.
func readReplayIndex(f *os.File) (*ReplayIndex, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	for window := replayIndexInitialRead; ; window *= 2 {
		if window > size {
			window = size
		}
		buf := make([]byte, window)
		if _, err := f.ReadAt(buf, size-window); err != nil && err != io.EOF {
			return nil, fmt.Errorf("read index tail: %w", err)
		}
		body := bytes.TrimRight(buf, "\n")
		nl := bytes.LastIndexByte(body, '\n')
		if nl < 0 && window < size {
			continue
		}
		last := body[nl+1:]
		if !IsReplayIndexLine(last) {
			return nil, nil
		}
		var idx ReplayIndex
		if err := json.Unmarshal(last, &idx); err != nil {
			return nil, fmt.Errorf("parse replay index: %w", err)
		}
		return &idx, nil
	}
}

// Replay selection strategies: which sessions each lap over the corpus
// dispatches, and in what order. Sessions sit in the file by start time.
//
//   - last-week: the first lap plays the final seven days of the capture (from
//     the first session that started within seven days of the last one, to the
//     end of the file); every later lap is the whole corpus in file order. A
//     run that needs more sessions than the corpus holds therefore opens on
//     the most recent traffic and then cycles everything.
//   - beginning: every lap is the whole corpus in file order.
//   - reverse-from-end: every lap is the whole corpus, newest first.
//   - random: every lap is a fresh permutation of the whole corpus, so no
//     session repeats before every other one has played.
//
// --replay-from moves where the first lap of beginning or reverse-from-end
// starts; that lap then runs to the end of its direction, and later laps are
// whole again. last-week is beginning with that start chosen for you.
const (
	ReplayStrategyLastWeek  = "last-week"
	ReplayStrategyBeginning = "beginning"
	ReplayStrategyReverse   = "reverse-from-end"
	ReplayStrategyRandom    = "random"
)

// replayLastWeekSpan is how far back from the last session last-week starts.
const replayLastWeekSpan = 7 * 24 * time.Hour

// ValidReplayStrategy reports whether s names a selection strategy.
func ValidReplayStrategy(s string) bool {
	switch s {
	case ReplayStrategyLastWeek, ReplayStrategyBeginning, ReplayStrategyReverse, ReplayStrategyRandom:
		return true
	}
	return false
}

// replayLaps produces the file indices of each lap for a strategy over n
// sessions. Lap 0 starts at first (forward for last-week and beginning,
// backward for reverse-from-end); every later lap covers all n. allowed, when
// non-nil, drops indices outside the set.
type replayLaps struct {
	n        int
	strategy string
	first    int
	allowed  map[int]bool
}

func (l replayLaps) lap(k int) []int {
	var out []int
	add := func(i int) {
		if l.allowed == nil || l.allowed[i] {
			out = append(out, i)
		}
	}
	switch l.strategy {
	case ReplayStrategyRandom:
		// Seeded by the lap number: reproducible run to run, different lap to lap.
		for _, i := range rand.New(rand.NewPCG(uint64(k), 0x7265706c6179)).Perm(l.n) {
			add(i)
		}
	case ReplayStrategyReverse:
		start := l.n - 1
		if k == 0 {
			start = l.first
		}
		for i := start; i >= 0; i-- {
			add(i)
		}
	default:
		start := 0
		if k == 0 {
			start = l.first
		}
		for i := start; i < l.n; i++ {
			add(i)
		}
	}
	return out
}

// replayFirstStart resolves where lap 0 starts. from is a --replay-from value:
// an integer session index (0 = first session line), or an RFC3339 time, which
// selects the first session started at or after it (forward) or the last
// started at or before it (reverse-from-end). Empty means the strategy's own
// start: seven days before the last session for last-week, the first session
// for beginning, the last for reverse-from-end.
func replayFirstStart(idx *ReplayIndex, strategy, from string) (int, error) {
	n := len(idx.Sessions)
	if n == 0 {
		return 0, fmt.Errorf("replay index lists no sessions")
	}
	if strategy == ReplayStrategyRandom {
		if from != "" {
			return 0, fmt.Errorf("--replay-from has no meaning with the random strategy")
		}
		return 0, nil
	}
	if from == "" {
		switch strategy {
		case ReplayStrategyReverse:
			return n - 1, nil
		case ReplayStrategyBeginning:
			return 0, nil
		}
	} else if i, err := strconv.Atoi(from); err == nil {
		if i < 0 || i >= n {
			return 0, fmt.Errorf("--replay-from %d outside 0..%d", i, n-1)
		}
		return i, nil
	}
	starts := make([]time.Time, n)
	for i, s := range idx.Sessions {
		t, err := time.Parse(time.RFC3339Nano, s.StartTs)
		if err != nil {
			return 0, fmt.Errorf("index session %d start_ts %q: %w", i, s.StartTs, err)
		}
		starts[i] = t
	}
	var at time.Time
	if from == "" { // last-week
		at = starts[n-1].Add(-replayLastWeekSpan)
	} else {
		t, err := time.Parse(time.RFC3339Nano, from)
		if err != nil {
			return 0, fmt.Errorf("--replay-from %q: want a session index or an RFC3339 time", from)
		}
		at = t
	}
	if strategy == ReplayStrategyReverse {
		after := sort.Search(n, func(i int) bool { return starts[i].After(at) })
		if after == 0 {
			return 0, fmt.Errorf("--replay-from %s precedes the first session (%s)", from, idx.Sessions[0].StartTs)
		}
		return after - 1, nil
	}
	first := sort.Search(n, func(i int) bool { return !starts[i].Before(at) })
	if first == n {
		return 0, fmt.Errorf("--replay-from %s follows the last session (%s)", from, idx.Sessions[n-1].StartTs)
	}
	return first, nil
}
