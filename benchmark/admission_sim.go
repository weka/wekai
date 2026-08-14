package benchmark

import (
	"container/heap"
	"math"
	"sort"
)

// A discrete-event model of the TTFT-governed session-admission loop, used to
// size a run before it is spent on hardware.
//
// The question it answers is not "how fast is the fleet" — a benchmark measures
// that. It is "where does the fleet stop taking more work, and what client-side
// concurrency does that correspond to", which has to be known BEFORE the run:
// a series pool set below the concurrency the governor will reach silently
// becomes the bottleneck, and the run then measures the client.
//
// The load shape is a closed loop with no concurrency limit of its own. What
// bounds it is the number of active sessions, and sessions are added one per
// second while time-to-first-token stays under a threshold. Past it, admission
// pauses; when it recovers, admission resumes. The fleet's own latency is the
// only throttle, so the loop settles at the largest session count the fleet can
// carry — which is the number a capacity plan actually needs.

// ServerProfile is one backend's throughput model.
//
// Prefill is a shared, serialised resource and decode is per-sequence, which is
// the distinction that decides which of the two binds first. Modelling decode
// as a flat per-sequence rate ignores that real batching slows each sequence as
// the batch grows; that makes this optimistic about decode and therefore
// conservative about the prefill ceiling being the one that matters.
type ServerProfile struct {
	Name string

	// PrefillTokensPerSec is aggregate prefill bandwidth, shared across every
	// request on this backend.
	PrefillTokensPerSec float64

	// DecodeTokensPerSec is the per-sequence output rate.
	DecodeTokensPerSec float64

	// MaxSeqs is vLLM's --max-num-seqs: how many sequences may be resident at
	// once. A request holds a slot from the start of its prefill until its last
	// output token.
	MaxSeqs int

	// CacheHitRate is the share of a request's input tokens already resident,
	// and so costing no prefill. This is the single number that separates a
	// small HBM-only KV tier from a large shared one, and the ceiling moves
	// with 1/(1-hit): the difference between 0.90 and 0.96 is not 6%, it is
	// 2.5x the sustainable request rate.
	CacheHitRate float64
}

// SimConfig is one simulated run.
type SimConfig struct {
	Servers []ServerProfile

	// AdmitEvery is how often a session is added while the gate is open.
	AdmitEvery float64

	// TTFTLimitSec closes the gate. Sessions stop being added while observed
	// TTFT is at or above it, and resume below.
	TTFTLimitSec float64

	// MaxSeries caps client-side concurrency. 0 means unbounded, which is what
	// the run wants; a positive value answers "would this pool have been the
	// bottleneck".
	MaxSeries int

	// HorizonSec ends the run.
	HorizonSec float64

	// TTFTWindow is how many recent completions the gate averages over. A gate
	// reading a single sample chases noise and oscillates; one reading too long
	// a window admits well past the limit before it notices.
	TTFTWindow int
}

// SimWorkload is the replayed traffic: sessions of requests at their original
// offsets, which is what makes the arrival process real rather than Poisson.
type SimWorkload struct {
	Sessions []SimSession `json:"sessions"`
}

// SimSession is one session's requests as [offsetSeconds, inputTokens,
// outputTokens] triples, ordered by offset.
type SimSession struct {
	Reqs [][3]float64 `json:"reqs"`
}

// SimResult is what the run learned.
type SimResult struct {
	Profile string

	// MaxSessions is the largest number of sessions concurrently active — the
	// answer the loop exists to find.
	MaxSessions   int
	FinalSessions int

	// MaxConcurrency is peak requests in flight, and MeanConcurrency its
	// time-weighted average. The peak sizes the series pool; the mean is what
	// the fleet actually ran at.
	MaxConcurrency  int
	MeanConcurrency float64

	Completed   int
	SeriesStall float64 // seconds spent blocked on MaxSeries, 0 if it never bound

	// MeanInputCompleted is the mean input size of the requests that finished,
	// which is NOT the workload's mean. A session's prompt grows as it runs, so
	// a run short enough that most sessions are young measures a cheaper
	// workload than the same sessions would impose at maturity. When this sits
	// well below the corpus mean, the ceiling found is optimistic.
	MeanInputCompleted float64
	SessionsRetired    int

	ThroughputReqPerSec float64
	MeanTTFT            float64
	P50TTFT             float64
	P99TTFT             float64

	// Bound names what actually stopped the run, which is the whole point of
	// simulating rather than guessing.
	Bound string

	PrefillUtil float64 // share of wall time the prefill engines were busy
	SeqSlotUtil float64 // share of sequence slots held, time-weighted
}

// ---------------------------------------------------------------- event queue

type evKind int

const (
	evArrive evKind = iota // a session's next request is due
	evDone                 // a request finished
	evAdmit                // the admission tick
)

type event struct {
	at   float64
	kind evKind
	sess int
	srv  int
	seq  int
}

type evQueue []event

func (q evQueue) Len() int { return len(q) }
func (q evQueue) Less(i, j int) bool {
	if q[i].at != q[j].at {
		return q[i].at < q[j].at
	}
	// Completions before arrivals at the same instant: a slot freed at t is
	// available to a request arriving at t, and the reverse ordering would
	// invent a stall that the real scheduler does not have.
	return q[i].kind < q[j].kind
}
func (q evQueue) Swap(i, j int)       { q[i], q[j] = q[j], q[i] }
func (q *evQueue) Push(x interface{}) { *q = append(*q, x.(event)) }
func (q *evQueue) Pop() interface{} {
	old := *q
	n := len(old)
	e := old[n-1]
	*q = old[:n-1]
	return e
}

// ---------------------------------------------------------------- simulation

// f64heap is a min-heap of sequence completion times, so "when does a slot free"
// is O(1) to read and O(log n) to maintain. A sorted slice here made the 4-hour
// horizon quadratic.
type f64heap []float64

func (h f64heap) Len() int            { return len(h) }
func (h f64heap) Less(i, j int) bool  { return h[i] < h[j] }
func (h f64heap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *f64heap) Push(x interface{}) { *h = append(*h, x.(float64)) }
func (h *f64heap) Pop() interface{} {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}

type srvState struct {
	prof          ServerProfile
	prefillFreeAt float64
	resident      f64heap // completion times of sequences holding a slot
	outstanding   int     // requests the client has open against this backend
	prefillBusy   float64
	slotSecs      float64
}

type sessState struct {
	reqs       [][3]float64
	cursor     int
	admittedAt float64
	active     bool
}

// RunAdmissionSim plays the workload through the model and reports where it
// settled.
func RunAdmissionSim(cfg SimConfig, wl SimWorkload) SimResult {
	if cfg.AdmitEvery <= 0 {
		cfg.AdmitEvery = 1
	}
	if cfg.TTFTLimitSec <= 0 {
		cfg.TTFTLimitSec = 5
	}
	if cfg.TTFTWindow <= 0 {
		cfg.TTFTWindow = 32
	}
	if cfg.HorizonSec <= 0 {
		cfg.HorizonSec = 3600
	}

	srv := make([]srvState, len(cfg.Servers))
	for i, p := range cfg.Servers {
		srv[i] = srvState{prof: p}
	}
	// Live sessions are separate from the templates they are drawn from. A run
	// long enough to find a ceiling admits far more sessions than any sample
	// holds, so templates are reused — legitimately, because sessions are
	// independent and the cache hit rate here is a parameter rather than
	// something derived from the content. A real run generates unique content
	// and must not lean on this.
	var sess []sessState

	var q evQueue
	heap.Init(&q)
	heap.Push(&q, event{at: 0, kind: evAdmit})

	var (
		now          float64
		nextSession  int
		active       int
		maxSessions  int
		inflight     int
		maxInflight  int
		concSecs     float64
		completed    int
		ttfts        []float64
		recent       []float64
		recentSum    float64
		seriesStall  float64
		blockedSince float64
		pending      []int // sessions waiting on the client series pool, FIFO
		inputSum     float64
		retired      int
	)

	admit := func() {
		if len(wl.Sessions) == 0 {
			return
		}
		tpl := wl.Sessions[nextSession%len(wl.Sessions)]
		sess = append(sess, sessState{reqs: tpl.Reqs, admittedAt: now, active: true})
		heap.Push(&q, event{at: now, kind: evArrive, sess: len(sess) - 1})
		nextSession++
		active++
		if active > maxSessions {
			maxSessions = active
		}
	}

	gateOpen := func() bool {
		if len(recent) < cfg.TTFTWindow {
			return true // not enough evidence to close it yet
		}
		return recentSum/float64(len(recent)) < cfg.TTFTLimitSec
	}

	observe := func(ttft float64) {
		ttfts = append(ttfts, ttft)
		recent = append(recent, ttft)
		recentSum += ttft
		if len(recent) > cfg.TTFTWindow {
			recentSum -= recent[0]
			recent = recent[1:]
		}
	}

	// pick chooses a backend by least outstanding, the fleet's own rule.
	pick := func() int {
		best, bestN := 0, math.MaxInt32
		for i := range srv {
			if srv[i].outstanding < bestN {
				best, bestN = i, srv[i].outstanding
			}
		}
		return best
	}

	// dispatch places one request and schedules its completion.
	dispatch := func(si int) {
		s := &sess[si]
		r := s.reqs[s.cursor]
		in, out := r[1], r[2]

		i := pick()
		st := &srv[i]
		// Retire sequences that have finished, then read when the next slot
		// frees. A sequence holds its slot from the start of prefill, so a full
		// batch delays prefill and not only decode.
		for st.resident.Len() > 0 && st.resident[0] <= now {
			heap.Pop(&st.resident)
		}
		slotAt := now
		if st.resident.Len() >= st.prof.MaxSeqs {
			slotAt = st.resident[0]
		}
		start := math.Max(math.Max(now, st.prefillFreeAt), slotAt)
		pdur := in * (1 - st.prof.CacheHitRate) / st.prof.PrefillTokensPerSec
		ddur := out / st.prof.DecodeTokensPerSec
		end := start + pdur + ddur

		st.prefillFreeAt = start + pdur
		st.prefillBusy += pdur
		st.slotSecs += end - start
		heap.Push(&st.resident, end)
		st.outstanding++

		inflight++
		if inflight > maxInflight {
			maxInflight = inflight
		}
		inputSum += in
		observe(start + pdur - now)
		heap.Push(&q, event{at: end, kind: evDone, sess: si, srv: i})
	}

	for q.Len() > 0 {
		e := heap.Pop(&q).(event)
		if e.at > cfg.HorizonSec {
			break
		}
		// Idle skip is implicit: `now` jumps straight to the next event, so a
		// stretch where nothing is in flight and no request is due costs no
		// simulated work. Time-weighted totals are accumulated across the jump
		// rather than per tick.
		if e.at > now {
			concSecs += float64(inflight) * (e.at - now)
			now = e.at
		}

		switch e.kind {
		case evAdmit:
			if gateOpen() {
				admit()
			}
			if now+cfg.AdmitEvery <= cfg.HorizonSec {
				heap.Push(&q, event{at: now + cfg.AdmitEvery, kind: evAdmit})
			}

		case evArrive:
			s := &sess[e.sess]
			if s.cursor >= len(s.reqs) {
				if s.active {
					s.active = false
					active--
					retired++
				}
				continue
			}
			if cfg.MaxSeries > 0 && inflight >= cfg.MaxSeries {
				// The client pool is full. Queue rather than drop or spin: this
				// is the stall a real series pool imposes, and measuring it is
				// how the run learns the pool — not the fleet — was the limit.
				if len(pending) == 0 {
					blockedSince = now
				}
				pending = append(pending, e.sess)
				continue
			}
			dispatch(e.sess)

		case evDone:
			st := &srv[e.srv]
			st.outstanding--
			inflight--
			completed++

			s := &sess[e.sess]
			s.cursor++
			if s.cursor >= len(s.reqs) {
				if s.active {
					s.active = false
					active--
					retired++
				}
			} else {
				// Original timings, with the response as a floor: a session is a
				// conversation, so request i+1 cannot precede response i however
				// the captured timings ran.
				due := math.Max(s.admittedAt+s.reqs[s.cursor][0], now)
				heap.Push(&q, event{at: due, kind: evArrive, sess: e.sess})
			}

			// A slot just freed; let one waiter through.
			if len(pending) > 0 && (cfg.MaxSeries == 0 || inflight < cfg.MaxSeries) {
				next := pending[0]
				pending = pending[1:]
				if len(pending) == 0 {
					seriesStall += now - blockedSince
				}
				if sess[next].cursor < len(sess[next].reqs) {
					dispatch(next)
				}
			}
		}
	}
	if len(pending) > 0 {
		seriesStall += now - blockedSince
	}

	res := SimResult{
		Profile:         cfg.Servers[0].Name,
		MaxSessions:     maxSessions,
		FinalSessions:   active,
		MaxConcurrency:  maxInflight,
		Completed:       completed,
		SeriesStall:     seriesStall,
		SessionsRetired: retired,
	}
	if len(ttfts) > 0 {
		res.MeanInputCompleted = inputSum / float64(len(ttfts))
		sorted := append([]float64(nil), ttfts...)
		sort.Float64s(sorted)
		var sum float64
		for _, v := range sorted {
			sum += v
		}
		res.MeanTTFT = sum / float64(len(sorted))
		res.P50TTFT = sorted[len(sorted)/2]
		res.P99TTFT = sorted[min(len(sorted)-1, len(sorted)*99/100)]
	}
	if now > 0 {
		res.MeanConcurrency = concSecs / now
		res.ThroughputReqPerSec = float64(completed) / now
		var pb, ss float64
		for i := range srv {
			pb += srv[i].prefillBusy
			ss += srv[i].slotSecs
		}
		res.PrefillUtil = pb / (now * float64(len(srv)))
		res.SeqSlotUtil = ss / (now * float64(len(srv)) * float64(cfg.Servers[0].MaxSeqs))
	}

	switch {
	case cfg.MaxSeries > 0 && res.SeriesStall > 0.01*now:
		res.Bound = "CLIENT series pool"
	case res.PrefillUtil > 0.85:
		res.Bound = "prefill bandwidth"
	case res.SeqSlotUtil > 0.85:
		res.Bound = "sequence slots (max-num-seqs)"
	default:
		res.Bound = "TTFT gate, below any hard limit"
	}
	return res
}
