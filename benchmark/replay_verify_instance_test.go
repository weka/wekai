package benchmark

import (
	"strings"
	"testing"
)

// observe's arguments in the order the recorder passes them, so a test reads
// like the sequence of requests it describes.
type seriesTurn struct {
	guid       string
	classified bool
	garbage    bool
	scored     bool
	missed     bool
}

// foldSeries numbers the series in first-seen order and the turns within each
// series from 1, which is what the recorder does with SeriesNum and CycleNum.
func foldSeries(t *testing.T, turns ...seriesTurn) instanceVerifyTotals {
	t.Helper()
	var h instanceVerifyHistory
	seriesOf := map[string]int{}
	turnOf := map[string]int{}
	for _, tn := range turns {
		if _, ok := seriesOf[tn.guid]; !ok {
			seriesOf[tn.guid] = len(seriesOf) + 1
		}
		turnOf[tn.guid]++
		h.observe(tn.guid, seriesOf[tn.guid], turnOf[tn.guid],
			tn.classified, tn.garbage, tn.scored, tn.missed)
	}
	return h.totals()
}

func seriesClean(guid string) seriesTurn {
	return seriesTurn{guid: guid, classified: true, scored: true}
}
func seriesCorrupt(guid string) seriesTurn {
	return seriesTurn{guid: guid, classified: true, garbage: true, scored: true}
}
func seriesMissed(guid string) seriesTurn {
	return seriesTurn{guid: guid, classified: true, scored: true, missed: true}
}

// seriesFailed is a request that never produced a response to look at: it was neither
// scanned nor scored, and is no evidence in either direction.
func seriesFailed(guid string) seriesTurn { return seriesTurn{guid: guid} }

// TestSeriesGarbageRun: adjacency is the whole signal. Independent per-request
// faults land next to each other at the square of their rate; a prefix that has
// gone bad stays bad, so back-to-back corruption in one series is the shape
// that separates a corrupted cache entry from a flaky decode.
func TestSeriesGarbageRun(t *testing.T) {
	t.Run("two in a row is a run", func(t *testing.T) {
		got := foldSeries(t, seriesClean("a"), seriesCorrupt("a"), seriesCorrupt("a"), seriesClean("a"))
		if got.garbageRunInstances != 1 {
			t.Errorf("garbageRunInstances = %d, want 1", got.garbageRunInstances)
		}
		if got.garbageInstances != 1 || got.classifiedInstances != 1 {
			t.Errorf("denominators = %d of %d, want 1 of 1", got.garbageInstances, got.classifiedInstances)
		}
	})

	t.Run("a clean response between them is not", func(t *testing.T) {
		got := foldSeries(t, seriesCorrupt("a"), seriesClean("a"), seriesCorrupt("a"))
		if got.garbageRunInstances != 0 {
			t.Error("two corrupt responses with a good one between them are not a run: the series " +
				"recovered, which is exactly what a persistent fault does not do")
		}
		if got.garbageInstances != 1 {
			t.Errorf("garbageInstances = %d, want the series still counted as having produced garbage", got.garbageInstances)
		}
	})

	t.Run("a failed request between them breaks nothing", func(t *testing.T) {
		got := foldSeries(t, seriesCorrupt("a"), seriesFailed("a"), seriesCorrupt("a"))
		if got.garbageRunInstances != 1 {
			t.Error("a request that never produced a response is no evidence of recovery; treating " +
				"it as clean lets one 429 hide the continuity this signal exists to see")
		}
	})

	t.Run("different series do not chain into each other", func(t *testing.T) {
		got := foldSeries(t, seriesCorrupt("a"), seriesCorrupt("b"), seriesCorrupt("c"))
		if got.garbageRunInstances != 0 {
			t.Errorf("garbageRunInstances = %d; one corrupt response each in three series is not a run "+
				"in any of them", got.garbageRunInstances)
		}
		if got.garbageInstances != 3 {
			t.Errorf("garbageInstances = %d, want 3", got.garbageInstances)
		}
	})
}

// TestSeriesMissToEnd: a model that declines one request answers the next, so
// what distinguishes lost context from an unhelpful response is whether
// anything after it ever came back right.
func TestSeriesMissToEnd(t *testing.T) {
	t.Run("never reciting again after the first miss", func(t *testing.T) {
		got := foldSeries(t, seriesClean("a"), seriesMissed("a"), seriesMissed("a"), seriesMissed("a"))
		if got.missToEndInstances != 1 {
			t.Errorf("missToEndInstances = %d, want 1", got.missToEndInstances)
		}
		if got.missInstances != 1 || got.askedInstances != 1 {
			t.Errorf("denominators = %d of %d, want 1 of 1", got.missInstances, got.askedInstances)
		}
	})

	t.Run("a later correct recite is a recovery", func(t *testing.T) {
		got := foldSeries(t, seriesMissed("a"), seriesMissed("a"), seriesClean("a"), seriesMissed("a"), seriesClean("a"))
		if got.missToEndInstances != 0 {
			t.Error("the series recited correctly after missing, so it did not lose its context; " +
				"counting it would put ordinary model non-compliance in a cache-corruption row")
		}
		if got.missInstances != 1 {
			t.Errorf("missInstances = %d, want the series still counted as having missed", got.missInstances)
		}
	})

	t.Run("one miss on the last request is where it ended, not a failure to recover", func(t *testing.T) {
		got := foldSeries(t, seriesClean("a"), seriesClean("a"), seriesMissed("a"))
		if got.missToEndInstances != 0 {
			t.Error("a single trailing miss says only that the session stopped there; with no later " +
				"request there was never an opportunity to recover, and counting it would make this " +
				"row a statistic about where sessions end")
		}
	})

	t.Run("a series that never missed is not in the numerator or its denominator", func(t *testing.T) {
		got := foldSeries(t, seriesClean("a"), seriesClean("a"))
		if got.missToEndInstances != 0 || got.missInstances != 0 {
			t.Errorf("missToEnd=%d miss=%d, want 0 0", got.missToEndInstances, got.missInstances)
		}
		if got.askedInstances != 1 {
			t.Errorf("askedInstances = %d, want the clean series counted as watched", got.askedInstances)
		}
	})

	t.Run("each series is judged on its own tail", func(t *testing.T) {
		got := foldSeries(t,
			seriesMissed("a"), seriesMissed("a"), // a never recovers
			seriesMissed("b"), seriesClean("b"), // b does
			seriesClean("c"), seriesMissed("c"), seriesMissed("c"), // c goes bad and stays bad
		)
		if got.missToEndInstances != 2 {
			t.Errorf("missToEndInstances = %d, want 2 (a and c)", got.missToEndInstances)
		}
		if got.missInstances != 3 || got.askedInstances != 3 {
			t.Errorf("denominators = %d of %d, want 3 of 3", got.missInstances, got.askedInstances)
		}
	})
}

// TestSeriesVerifyIgnoresUnidentifiedRequests: a request with no series guid
// cannot be placed in a sequence, and folding them all into one bucket would
// manufacture runs out of unrelated requests.
func TestSeriesVerifyIgnoresUnidentifiedRequests(t *testing.T) {
	got := foldSeries(t, seriesCorrupt(""), seriesCorrupt(""), seriesMissed(""), seriesMissed(""))
	if got.classifiedInstances != 0 || got.askedInstances != 0 {
		t.Errorf("unidentified requests entered the series population: %+v", got)
	}
}

// A count says the fleet has the fault. Locating it needs three things: which
// series, the turn it stopped reciting on, and how long it stayed that way —
// and the first two are the same s/t coordinates the per-request error lines
// carry, so a reported run can be grepped straight out of a log.
func TestSeriesMissRunsAreLocated(t *testing.T) {
	got := foldSeries(t,
		// series 1: clean throughout, must not be named.
		seriesClean("a"), seriesClean("a"),
		// series 2: recites through turn 2, then misses turns 3, 4 and 5.
		seriesClean("b"), seriesClean("b"),
		seriesMissed("b"), seriesMissed("b"), seriesMissed("b"),
	)
	if len(got.missToEnd) != 1 {
		t.Fatalf("named %v, want exactly the one series that never recovered", got.missToEnd)
	}
	if s := got.missToEnd[0].String(); s != "2:3:3" {
		t.Errorf("run = %s, want 2:3:3 — series 2 stopped reciting on turn 3 and missed 3 in a row", s)
	}
}

// A miss that IS recovered from resets the tail, so the reported turn must be
// where the final unbroken run began — not where the series first stumbled.
// The earlier number would send an investigation to a turn the session went on
// to answer correctly.
func TestSeriesMissRunReportsTheUnrecoveredTail(t *testing.T) {
	got := foldSeries(t,
		seriesMissed("a"), // turn 1: missed, then recovered — not the run
		seriesClean("a"),  // turn 2
		seriesClean("a"),  // turn 3
		seriesMissed("a"), // turn 4: the tail starts here
		seriesMissed("a"), // turn 5
	)
	if len(got.missToEnd) != 0 {
		t.Fatalf("named %v; a series that recited correctly after its first miss recovered from it, "+
			"so it is not in this population at all", got.missToEnd)
	}
}

// Map iteration is random. Two arms of the same experiment have to produce
// lists that can be diffed, so the order is the series order.
func TestSeriesMissRunsAreSorted(t *testing.T) {
	var turns []seriesTurn
	for _, g := range []string{"e", "c", "a", "d", "b"} {
		turns = append(turns, seriesMissed(g), seriesMissed(g))
	}
	first := foldSeries(t, turns...)
	for range 20 {
		got := foldSeries(t, turns...)
		if len(got.missToEnd) != 5 {
			t.Fatalf("named %d runs, want 5", len(got.missToEnd))
		}
		for i := range got.missToEnd {
			if got.missToEnd[i] != first.missToEnd[i] {
				t.Fatalf("order moved between folds: %v vs %v", got.missToEnd, first.missToEnd)
			}
		}
		for i := 1; i < len(got.missToEnd); i++ {
			if got.missToEnd[i-1].Series > got.missToEnd[i].Series {
				t.Fatalf("not sorted by series: %v", got.missToEnd)
			}
		}
	}
}

// A truncated list that does not say it is truncated reads as the whole set.
func TestMissRunListSaysWhatItLeftOut(t *testing.T) {
	var runs []seriesMissRun
	for i := 1; i <= 14; i++ {
		runs = append(runs, seriesMissRun{Series: i, Turn: i * 2, Length: 3})
	}
	out := formatMissRuns(runs, 10)
	if !strings.Contains(out, "1:2:3") || !strings.Contains(out, "10:20:3") {
		t.Errorf("first ten runs are not all present: %q", out)
	}
	if strings.Contains(out, "11:22:3") {
		t.Errorf("the cap did not apply: %q", out)
	}
	if !strings.Contains(out, "and 4 more") {
		t.Errorf("the remainder went unmentioned, so the list reads as complete: %q", out)
	}
	if got := formatMissRuns(nil, 10); got != "" {
		t.Errorf("empty input rendered %q, want nothing to print", got)
	}
	// Under the cap, no remainder clause at all.
	if out := formatMissRuns(runs[:3], 10); strings.Contains(out, "more") {
		t.Errorf("a complete list claimed a remainder: %q", out)
	}
}

// Under --replay-reuse-sessions the corpus is replayed in laps, and every lap
// hands out the same session under the same SeriesGUID. Folding those laps into
// one history lets a later lap's correct recite erase an earlier lap that never
// recovered — the signal reads 0 on a fleet that failed every time.
func TestSeriesLapsAreCountedSeparately(t *testing.T) {
	var h instanceVerifyHistory
	// Two laps of one session, same GUID, different series numbers. Lap 1 stops
	// reciting for good; lap 2 is clean.
	for _, turn := range []struct {
		series, turn int
		missed       bool
	}{
		{1, 1, false}, {1, 2, true}, {1, 3, true},
		{2, 1, false}, {2, 2, false}, {2, 3, false},
	} {
		h.observe("sess:inst", turn.series, turn.turn, true, false, true, turn.missed)
	}
	got := h.totals()
	if got.missToEndInstances != 1 {
		t.Fatalf("missToEndInstances = %d, want 1: lap 1 never recovered, and lap 2 reciting correctly "+
			"is a different conversation rather than a recovery", got.missToEndInstances)
	}
	if got.askedInstances != 2 {
		t.Errorf("askedInstances = %d, want 2: two laps are two conversations", got.askedInstances)
	}
	if s := got.missToEnd[0].String(); s != "1:2:2" {
		t.Errorf("run = %s, want 1:2:2", s)
	}
}

// The unit is the AGENT, not the session, and that is why these counts exceed
// the session count a run was given: one session that fans out to three
// sub-agents contributes four.
//
// The confusion this pins is a real one — a 256-session run reported 686
// instances scanned, which reads as impossible until you know a session holds
// 4.3 agents on average in this corpus.
func TestInstancesOfOneSessionAreCountedSeparately(t *testing.T) {
	var h instanceVerifyHistory
	// One session (series 7), a main agent and two sub-agents. Only the second
	// sub-agent loses its context.
	for _, inst := range []struct {
		guid   string
		missed bool
	}{
		{"sess-a:main", false},
		{"sess-a:sub-1", false},
		{"sess-a:sub-2", true},
	} {
		h.observe(inst.guid, 7, 1, true, false, true, false)
		h.observe(inst.guid, 7, 2, true, false, true, inst.missed)
		h.observe(inst.guid, 7, 3, true, false, true, inst.missed)
	}
	got := h.totals()
	if got.askedInstances != 3 {
		t.Errorf("askedInstances = %d, want 3: one session's agents each own their own prefix, so "+
			"each is its own conversation here", got.askedInstances)
	}
	if got.missToEndInstances != 1 {
		t.Fatalf("missToEndInstances = %d, want 1: only one sub-agent lost its context, and its "+
			"siblings' health says nothing about it", got.missToEndInstances)
	}
	// The reported series number is the SESSION's, shared by every agent in it,
	// so it stays greppable against the s%d of a per-request error line.
	if s := got.missToEnd[0].String(); s != "7:2:2" {
		t.Errorf("run = %s, want 7:2:2 — session 7's failing agent stopped reciting on turn 2", s)
	}
}
