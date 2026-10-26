package benchmark

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"time"
)

// Offline model of the real-time replay's "active dataset": the sum, over live
// session slots, of each slot's latest full prompt (activeDatasetTracker).
//
// Under real-time pacing a slot holds one session for that session's whole
// wall-clock life, idle think gaps included, so the dataset is set by how long
// sessions live and how big they grow, and by how many slots the governor
// keeps. Both drift for as long as the slots' session-age mix is still filling
// in with long sessions: the dataset settles only once that mix is stationary.

// DatasetRequest is one request of a session timeline, in ms from the start of
// the session on the replay's clock.
type DatasetRequest struct {
	StartMs, EndMs int64
	Prompt         int64
}

// DatasetSession is a session as real-time think pacing would play it with a
// fixed response time per request.
type DatasetSession struct {
	Requests   []DatasetRequest // sorted by StartMs
	DurationMs int64            // last response end
	// CapturedMs is the session's span in the capture (first start to last
	// end), for comparing with DurationMs.
	CapturedMs int64
	// PromptMsArea is the integral of the slot's tracked prompt over the
	// session's life, in token-ms: the session's contribution to dataset-time.
	PromptMsArea float64
	// BusyMs is the total time requests of this session are in flight.
	BusyMs int64
	// MaxWaitMs is the session's longest think gap: a turn's captured wait
	// after the end of the request it follows (its instance's previous one,
	// or the spawning one). Offsets from the session start are not waits.
	MaxWaitMs int64
	// Waits counts the session's gaps at or above each DatasetWaitBounds edge.
	Waits []int
}

// BuildDatasetSession schedules a session's requests the way the real-time
// think pacer does: each turn is due at its reference's end plus its captured
// think_ms (the previous request of its instance, the parent's spawning
// request, or the session start), and runs for its captured total_ms times
// respScale. Requests without think_ms fall back to their captured offset.
func BuildDatasetSession(s *RouterReplaySession, respScale float64) DatasetSession {
	origin, err := time.Parse(time.RFC3339Nano, s.StartTs)
	type loc struct{ inst, req int }
	byID := map[uint64]loc{}
	for i := range s.Instances {
		for j := range s.Instances[i].Requests {
			if id := s.Instances[i].Requests[j].RequestID; id != 0 {
				byID[id] = loc{i, j}
			}
		}
	}
	start := make([][]int64, len(s.Instances))
	end := make([][]int64, len(s.Instances))
	state := make([][]uint8, len(s.Instances)) // 0 unvisited, 1 in progress, 2 done
	for i := range s.Instances {
		n := len(s.Instances[i].Requests)
		start[i], end[i], state[i] = make([]int64, n), make([]int64, n), make([]uint8, n)
	}
	offset := func(r *RouterReplayRequest) int64 {
		if err != nil {
			return 0
		}
		ts, perr := time.Parse(time.RFC3339Nano, r.Ts)
		if perr != nil {
			return 0
		}
		return max(0, ts.Sub(origin).Milliseconds())
	}
	var visit func(i, j int) int64
	visit = func(i, j int) int64 {
		switch state[i][j] {
		case 2:
			return end[i][j]
		case 1: // a reference cycle the converter failed to break: use the capture
			return offset(&s.Instances[i].Requests[j])
		}
		state[i][j] = 1
		r := &s.Instances[i].Requests[j]
		var st int64
		if r.ThinkMs == nil {
			st = offset(r)
		} else {
			think := *r.ThinkMs
			switch r.ThinkRef {
			case ThinkRefPrev:
				if j > 0 {
					st = visit(i, j-1) + think
				} else {
					st = offset(r)
				}
			case ThinkRefSpawn:
				if l, ok := byID[s.Instances[i].ParentSpawnRequestID]; ok {
					st = visit(l.inst, l.req) + think
				} else {
					st = offset(r)
				}
			default:
				st = think
			}
		}
		if j > 0 && state[i][j-1] == 2 {
			st = max(st, end[i][j-1]) // an instance never overlaps itself
		}
		svc := int64(math.Round(float64(max(r.TotalMs, 1)) * respScale))
		start[i][j], end[i][j], state[i][j] = st, st+svc, 2
		return end[i][j]
	}

	out := DatasetSession{Waits: make([]int, len(DatasetWaitBounds))}
	capFirst, capLast := int64(math.MaxInt64), int64(0)
	for i := range s.Instances {
		for j := range s.Instances[i].Requests {
			visit(i, j)
			r := &s.Instances[i].Requests[j]
			out.Requests = append(out.Requests, DatasetRequest{StartMs: start[i][j], EndMs: end[i][j], Prompt: int64(r.InputTokens)})
			out.DurationMs = max(out.DurationMs, end[i][j])
			out.BusyMs += end[i][j] - start[i][j]
			if r.ThinkMs != nil && (r.ThinkRef == ThinkRefPrev || r.ThinkRef == ThinkRefSpawn) {
				out.MaxWaitMs = max(out.MaxWaitMs, *r.ThinkMs)
				for k, b := range DatasetWaitBounds {
					if time.Duration(*r.ThinkMs)*time.Millisecond >= b {
						out.Waits[k]++
					}
				}
			}
			o := offset(r)
			capFirst = min(capFirst, o)
			capLast = max(capLast, o+r.TotalMs)
		}
	}
	if capFirst <= capLast {
		out.CapturedMs = capLast - capFirst
	}
	sort.Slice(out.Requests, func(a, b int) bool { return out.Requests[a].StartMs < out.Requests[b].StartMs })

	// The tracker holds the prompt of the slot's most recent RESPONSE.
	ends := make([]DatasetRequest, len(out.Requests))
	copy(ends, out.Requests)
	sort.Slice(ends, func(a, b int) bool { return ends[a].EndMs < ends[b].EndMs })
	for k, r := range ends {
		next := out.DurationMs
		if k+1 < len(ends) {
			next = ends[k+1].EndMs
		}
		out.PromptMsArea += float64(r.Prompt) * float64(next-r.EndMs)
	}
	return out
}

// DatasetSimConfig drives SimulateDataset. Exactly one of Series and
// TargetInflight is set.
type DatasetSimConfig struct {
	Strategy string // reverse-from-end (default), beginning or random
	// Series fixes the slot count.
	Series int
	// TargetInflight models the TTFT governor as a load target: StartSeries
	// slots, then AdmitCount more every AdmitEvery while the interval's mean
	// in-flight requests stayed below it; with Retire, a slot whose session
	// ends while the last interval ran above TargetInflight*(1+RetireMargin)
	// is retired instead of refilled.
	TargetInflight float64
	StartSeries    int
	AdmitEvery     time.Duration
	AdmitCount     int
	MaxSeries      int
	Retire         bool
	RetireMargin   float64
	Horizon        time.Duration
	SampleEvery    time.Duration
}

// DatasetSample is the simulated state at one sample time.
type DatasetSample struct {
	At       time.Duration
	Slots    int     // slots holding a session
	Inflight float64 // mean requests in flight over the sample interval
	Dataset  int64   // active dataset tokens
	Series   int     // slots contributing (completed at least one request)
	Started  int     // sessions dispatched so far
	Lap      int     // 0-based lap of the session last dispatched
	// MeanAge is the mean age of the live slots' sessions.
	MeanAge time.Duration
	// ByLife splits the live slots by the total life of the session they
	// hold, at DatasetLifeBounds.
	ByLife []DatasetLifeShare
}

// DatasetWaitBounds are the think-gap bucket edges of the wait breakdown.
var DatasetWaitBounds = []time.Duration{time.Minute, 10 * time.Minute, time.Hour, 6 * time.Hour, 24 * time.Hour, 3 * 24 * time.Hour, 7 * 24 * time.Hour}

// DatasetLifeBounds are the session-life bucket edges for ByLife: a slot
// holding a session that lives past a day is load a one-day run never sheds.
var DatasetLifeBounds = []time.Duration{time.Hour, 6 * time.Hour, 24 * time.Hour, 3 * 24 * time.Hour, 7 * 24 * time.Hour}

// DatasetLifeShare is one session-life bucket of a sample.
type DatasetLifeShare struct {
	Slots   int   // live slots holding a session of this life
	Busy    int   // of those, with a request in flight
	Dataset int64 // their tracked prompts
}

func lifeBucket(ms int64) int {
	d := time.Duration(ms) * time.Millisecond
	for i, b := range DatasetLifeBounds {
		if d < b {
			return i
		}
	}
	return len(DatasetLifeBounds)
}

type datasetEvent struct {
	at   int64
	slot int
	seq  int64
}

type datasetEventHeap []datasetEvent

func (h datasetEventHeap) Len() int { return len(h) }
func (h datasetEventHeap) Less(a, b int) bool {
	if h[a].at != h[b].at {
		return h[a].at < h[b].at
	}
	return h[a].seq < h[b].seq
}
func (h datasetEventHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h *datasetEventHeap) Push(x any)   { *h = append(*h, x.(datasetEvent)) }
func (h *datasetEventHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// datasetSlot walks one session's requests: starts at their StartMs, ends at
// their EndMs, both offset by the slot's base time.
type datasetSlot struct {
	sess     *DatasetSession
	base     int64
	starts   int // next request to start (Requests order)
	ends     []DatasetRequest
	endPos   int
	tracked  int64
	inflight int
	live     bool
}

func (sl *datasetSlot) nextAt() int64 {
	next := int64(math.MaxInt64)
	if sl.starts < len(sl.sess.Requests) {
		next = sl.sess.Requests[sl.starts].StartMs
	}
	if sl.endPos < len(sl.ends) {
		next = min(next, sl.ends[sl.endPos].EndMs)
	}
	if next == math.MaxInt64 {
		next = sl.sess.DurationMs
	}
	return sl.base + next
}

// SimulateDataset plays sessions (in file order) through real-time slots and
// samples the active dataset every SampleEvery up to Horizon.
func SimulateDataset(sessions []DatasetSession, cfg DatasetSimConfig) ([]DatasetSample, error) {
	if len(sessions) == 0 {
		return nil, fmt.Errorf("no sessions")
	}
	if (cfg.Series > 0) == (cfg.TargetInflight > 0) {
		return nil, fmt.Errorf("set exactly one of Series and TargetInflight")
	}
	if cfg.Strategy == "" {
		cfg.Strategy = ReplayStrategyReverse
	}
	switch cfg.Strategy {
	case ReplayStrategyReverse, ReplayStrategyBeginning, ReplayStrategyRandom:
	default:
		return nil, fmt.Errorf("strategy %q: want reverse-from-end, beginning or random", cfg.Strategy)
	}
	if cfg.SampleEvery <= 0 {
		cfg.SampleEvery = 10 * time.Minute
	}
	if cfg.AdmitEvery <= 0 {
		cfg.AdmitEvery = time.Minute
	}
	cfg.AdmitCount = max(cfg.AdmitCount, 1)
	if cfg.StartSeries <= 0 {
		cfg.StartSeries = 32
	}
	n := len(sessions)
	first := n - 1
	if cfg.Strategy != ReplayStrategyReverse {
		first = 0
	}
	laps := replayLaps{n: n, strategy: cfg.Strategy, first: first}
	lap, lapPos, curLap := 0, 0, laps.lap(0)
	nextSession := func() *DatasetSession {
		for lapPos >= len(curLap) {
			lap++
			lapPos, curLap = 0, laps.lap(lap)
		}
		s := &sessions[curLap[lapPos]]
		lapPos++
		return s
	}

	var (
		h        datasetEventHeap
		seq      int64
		slots    []*datasetSlot
		now      int64
		inflight int
		area     float64 // in-flight integral since the last admit tick
		sArea    float64 // ... since the last sample
		gArea    float64 // ... since the last gate sample
		lastT    int64
		dataset  int64
		started  int
		lastMean float64
		samples  []DatasetSample
	)
	advance := func(t int64) {
		d := float64(t - lastT)
		area += float64(inflight) * d
		sArea += float64(inflight) * d
		gArea += float64(inflight) * d
		lastT, now = t, t
	}
	push := func(i int) {
		seq++
		heap.Push(&h, datasetEvent{at: slots[i].nextAt(), slot: i, seq: seq})
	}
	assign := func(i int) {
		sl := slots[i]
		sl.sess = nextSession()
		started++
		sl.base, sl.starts, sl.endPos, sl.live = now, 0, 0, true
		sl.ends = append(sl.ends[:0], sl.sess.Requests...)
		sort.Slice(sl.ends, func(a, b int) bool { return sl.ends[a].EndMs < sl.ends[b].EndMs })
		push(i)
	}
	addSlot := func() {
		slots = append(slots, &datasetSlot{})
		assign(len(slots) - 1)
	}
	live := func() int {
		c := 0
		for _, sl := range slots {
			if sl.live {
				c++
			}
		}
		return c
	}
	initial := cfg.Series
	if cfg.TargetInflight > 0 {
		initial = cfg.StartSeries
	}
	for range initial {
		addSlot()
	}

	horizon := cfg.Horizon.Milliseconds()
	admitMs, sampleMs := cfg.AdmitEvery.Milliseconds(), cfg.SampleEvery.Milliseconds()
	// The gate mirrors admitSeries: an interval is open only if every
	// admitSampleEvery sample stayed under target, and after a block it takes
	// admitUnblockIntervals open intervals in a row to admit again.
	gateMs := admitSampleEvery.Milliseconds()
	nextAdmit, nextSample, nextGate := admitMs, sampleMs, gateMs
	sustained, blocked, good := true, false, 0
	for now < horizon {
		t := min(nextAdmit, nextSample, nextGate, horizon)
		if h.Len() > 0 && h[0].at < t {
			ev := heap.Pop(&h).(datasetEvent)
			advance(ev.at)
			sl := slots[ev.slot]
			rel := now - sl.base
			for sl.endPos < len(sl.ends) && sl.ends[sl.endPos].EndMs <= rel {
				r := sl.ends[sl.endPos]
				sl.endPos++
				inflight--
				sl.inflight--
				dataset += r.Prompt - sl.tracked
				sl.tracked = r.Prompt
			}
			for sl.starts < len(sl.sess.Requests) && sl.sess.Requests[sl.starts].StartMs <= rel {
				sl.starts++
				inflight++
				sl.inflight++
			}
			if sl.starts >= len(sl.sess.Requests) && sl.endPos >= len(sl.ends) && rel >= sl.sess.DurationMs {
				dataset -= sl.tracked
				sl.tracked, sl.live = 0, false
				retire := cfg.TargetInflight > 0 && cfg.Retire && lastMean > cfg.TargetInflight*(1+cfg.RetireMargin)
				if !retire {
					assign(ev.slot)
				}
			} else {
				push(ev.slot)
			}
			continue
		}
		advance(t)
		if t == nextGate {
			if gArea/float64(gateMs) >= cfg.TargetInflight {
				sustained = false
			}
			gArea = 0
			nextGate += gateMs
		}
		if t == nextAdmit {
			lastMean = area / float64(admitMs)
			area = 0
			open := sustained && lastMean < cfg.TargetInflight
			sustained = true
			if open {
				good++
			} else {
				good, blocked = 0, true
			}
			if open && blocked && good >= admitUnblockIntervals {
				blocked = false
			}
			if cfg.TargetInflight > 0 && open && !blocked && (cfg.MaxSeries <= 0 || live() < cfg.MaxSeries) {
				count := cfg.AdmitCount
				if cfg.MaxSeries > 0 {
					count = min(count, cfg.MaxSeries-live())
				}
				for range count {
					// Reuse a retired slot before growing the table.
					reused := false
					for i, sl := range slots {
						if !sl.live {
							assign(i)
							reused = true
							break
						}
					}
					if !reused {
						addSlot()
					}
				}
			}
			nextAdmit += admitMs
		}
		if t == nextSample {
			s := DatasetSample{At: time.Duration(t) * time.Millisecond, Inflight: sArea / float64(sampleMs), Dataset: dataset, Started: started, Lap: lap}
			var age int64
			s.ByLife = make([]DatasetLifeShare, len(DatasetLifeBounds)+1)
			for _, sl := range slots {
				if sl.live {
					b := &s.ByLife[lifeBucket(sl.sess.DurationMs)]
					b.Slots++
					b.Dataset += sl.tracked
					if sl.inflight > 0 {
						b.Busy++
					}
					s.Slots++
					age += now - sl.base
					if sl.tracked > 0 {
						s.Series++
					}
				}
			}
			if s.Slots > 0 {
				s.MeanAge = time.Duration(age/int64(s.Slots)) * time.Millisecond
			}
			samples = append(samples, s)
			sArea = 0
			nextSample += sampleMs
		}
	}
	return samples, nil
}

// DatasetStationary is the renewal-reward long-run average over whole laps: a
// slot cycling through the corpus spends each session's duration on it, so its
// time-averaged dataset and load are the corpus totals divided by total
// duration, whatever the order.
type DatasetStationary struct {
	PerSlotDataset  float64 // tokens
	PerSlotInflight float64 // requests in flight
	// LapSlotTime is the total session duration of one lap: divided by the
	// slot count it is how long one lap takes.
	LapSlotTime time.Duration
}

// StationaryDataset computes the renewal-reward averages over sessions.
func StationaryDataset(sessions []DatasetSession) DatasetStationary {
	var area, busy, dur float64
	for i := range sessions {
		area += sessions[i].PromptMsArea
		busy += float64(sessions[i].BusyMs)
		dur += float64(sessions[i].DurationMs)
	}
	if dur == 0 {
		return DatasetStationary{}
	}
	return DatasetStationary{
		PerSlotDataset:  area / dur,
		PerSlotInflight: busy / dur,
		LapSlotTime:     time.Duration(dur) * time.Millisecond,
	}
}

// SettleTime is the first sample time after which the trailing-window mean of
// value stays within tol (a fraction) of target until the end; -1 if never.
func SettleTime(samples []DatasetSample, value func(DatasetSample) float64, target, tol float64, window int) time.Duration {
	window = max(window, 1)
	means := make([]float64, len(samples))
	var sum float64
	for i, s := range samples {
		sum += value(s)
		if i >= window {
			sum -= value(samples[i-window])
		}
		means[i] = sum / float64(min(i+1, window))
	}
	settled := -1
	for i := len(means) - 1; i >= 0; i-- {
		if math.Abs(means[i]-target) > tol*target {
			break
		}
		settled = i
	}
	if settled < 0 {
		return -1
	}
	return samples[settled].At
}
