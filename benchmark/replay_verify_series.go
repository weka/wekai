package benchmark

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Verify signals that only exist ACROSS the requests of one series.
//
// Every other number in the verify section is a property of a single response:
// this one was corrupt, that one did not recite its marker. Read that way, a
// run with 40 garbage responses out of 20,000 looks like background noise
// whatever produced it — one flaky decode per 500 requests is the same figure
// as one session whose KV went bad and stayed bad for its next fifteen turns.
//
// The difference between those two runs is entirely in the ORDER, and the
// order is exactly what a per-request count discards. So two shapes are
// tracked per series, both of them things a transient fault cannot easily
// produce and a corrupted prefix produces naturally:
//
//	back-to-back garbage  two consecutive classified responses in one series
//	                      both corrupt. Independent per-request faults land
//	                      adjacent at the square of their rate; a prefix that
//	                      is bad stays bad until the session ends or the
//	                      entry is evicted.
//	miss to the end       the series' first PRESENCE_MISS is followed by no
//	                      correct recite for the rest of the session. A model
//	                      that declines one request answers the next; a
//	                      session that lost its context never gets it back.
//
// Both are counted over the requests that were actually SCANNED. An errored,
// empty or skipped request is no evidence either way, so it neither joins a
// run of corrupt responses nor breaks one — treating it as clean would let
// one 429 in the middle hide the very continuity these signals exist to see.
type seriesVerifyHistory struct {
	mu sync.Mutex
	// Keyed by DISPATCH, not by conversation: the series number joined to the
	// GUID. Under --replay-reuse-sessions the corpus is replayed in laps, and
	// every lap hands out the same session with the same SeriesGUID — so a
	// GUID-keyed history folds all of a session's laps together, and one lap
	// reciting correctly erases the lap before it that never recovered. The
	// series number is the pull counter, unique per dispatch, so joining it to
	// the GUID keeps each lap of each instance its own conversation.
	byDispatch map[string]*seriesVerifyState
}

// seriesVerifyState is one series' running history. Keyed by SeriesGUID
// (session:instance), which is the unit that owns a conversation prefix: a
// session's sub-agents fan out with their own context, so a marker lost by one
// says nothing about its siblings.
type seriesVerifyState struct {
	// seriesNum is the run's own index for this series — the same s%d the
	// per-request error lines carry, so a reported run can be grepped for.
	seriesNum   int
	classified  int  // responses scanned for corruption
	garbageReqs int  // of those, ones that carried it
	prevGarbage bool // was the previous classified response corrupt
	garbageRun  bool // two classified responses in a row were

	scored   int // responses that were asked to recite
	misses   int // of those, ones missing at least one expected marker
	missTail int // consecutive misses ending at the latest scored response
	// missTailTurn is the turn the current unbroken tail of misses began on,
	// reset with the tail. Held because "2 of 97 series never recovered" says
	// a fleet has the fault but not where to look for it, and the turn a
	// session stopped reciting is the first thing an investigation needs.
	missTailTurn int
}

// observe folds one completed request into its series' history.
//
// Called from recordReplayRequest, which runs on the series' own goroutine in
// request order — an instance issues its turns strictly in sequence, and that
// is the ordering both signals are built on. The lock guards the map against
// other series, not against this series' own ordering.
func (h *seriesVerifyHistory) observe(guid string, seriesNum, turn int, classified, garbage, scored, missed bool) {
	if guid == "" || (!classified && !scored) {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.byDispatch == nil {
		h.byDispatch = make(map[string]*seriesVerifyState)
	}
	key := strconv.Itoa(seriesNum) + "|" + guid
	s := h.byDispatch[key]
	if s == nil {
		s = &seriesVerifyState{seriesNum: seriesNum}
		h.byDispatch[key] = s
	}
	if classified {
		s.classified++
		if garbage {
			s.garbageReqs++
			if s.prevGarbage {
				s.garbageRun = true
			}
		}
		s.prevGarbage = garbage
	}
	if scored {
		s.scored++
		if missed {
			if s.missTail == 0 {
				s.missTailTurn = turn
			}
			s.misses++
			s.missTail++
		} else {
			// A correct recite is the recovery that makes everything before it
			// survivable, so the tail restarts from here.
			s.missTail = 0
		}
	}
}

// seriesVerifyTotals is the run-level view, with the denominators each count
// belongs to: "3 series went bad" means nothing without how many were watched,
// and the two counts are watched over different populations.
type seriesVerifyTotals struct {
	classifiedSeries int64 // series with >=1 response scanned for corruption
	garbageSeries    int64 // of those, ones that produced any garbage at all
	garbageRunSeries int64 // of those, ones that produced it twice in a row
	scoredSeries     int64 // series with >=1 response asked to recite
	missSeries       int64 // of those, ones that missed at least once
	missToEndSeries  int64 // of those, ones that never recited correctly again
	// missToEnd names them, sorted by series. A count says the fault exists;
	// these say which conversation to open.
	missToEnd []seriesMissRun
}

// seriesMissRun locates one series' unrecovered tail of misses.
type seriesMissRun struct {
	Series int // the run's series index, as in the s%d of an error line
	Turn   int // the turn its first unrecovered miss landed on, as in t%d
	Length int // how many consecutive misses ran from there to the end
}

func (r seriesMissRun) String() string {
	return fmt.Sprintf("%d:%d:%d", r.Series, r.Turn, r.Length)
}

// maxReportedMissRuns caps the named list in the summary. Ten is what fits
// beside the label on one line; past that the count above is the number that
// matters and the rest are in the request data.
const maxReportedMissRuns = 10

// formatMissRuns renders runs as series:turn:length, capped so one pathological
// arm cannot push the rest of the summary off the screen. The remainder is
// counted rather than dropped silently: a truncated list that does not say it
// is truncated reads as the whole set.
func formatMissRuns(runs []seriesMissRun, limit int) string {
	if len(runs) == 0 {
		return ""
	}
	shown := runs
	if limit > 0 && len(shown) > limit {
		shown = shown[:limit]
	}
	parts := make([]string, len(shown))
	for i, r := range shown {
		parts[i] = r.String()
	}
	out := strings.Join(parts, " ")
	if n := len(runs) - len(shown); n > 0 {
		out += fmt.Sprintf(" … and %d more", n)
	}
	return out
}

func (h *seriesVerifyHistory) totals() seriesVerifyTotals {
	var t seriesVerifyTotals
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, s := range h.byDispatch {
		if s.classified > 0 {
			t.classifiedSeries++
		}
		if s.garbageReqs > 0 {
			t.garbageSeries++
		}
		if s.garbageRun {
			t.garbageRunSeries++
		}
		if s.scored > 0 {
			t.scoredSeries++
		}
		if s.misses > 0 {
			t.missSeries++
		}
		// misses == missTail says no correct recite followed the first miss;
		// missTail >= 2 says there were later requests to get it right in.
		// Without that second clause every series whose LAST request happened
		// to miss would qualify, which is a statement about where the session
		// ended rather than about anything failing to recover.
		if s.misses > 0 && s.misses == s.missTail && s.missTail >= 2 {
			t.missToEndSeries++
			t.missToEnd = append(t.missToEnd, seriesMissRun{
				Series: s.seriesNum, Turn: s.missTailTurn, Length: s.missTail,
			})
		}
	}
	// Map order is random, so a run would otherwise report the same failures in
	// a different order every time and two arms could not be diffed.
	sort.Slice(t.missToEnd, func(i, j int) bool {
		if t.missToEnd[i].Series != t.missToEnd[j].Series {
			return t.missToEnd[i].Series < t.missToEnd[j].Series
		}
		return t.missToEnd[i].Turn < t.missToEnd[j].Turn
	})
	return t
}
