package benchmark

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"time"
)

// Live view: an opt-in HTTP server (--live-port) that renders the request
// timeline while a benchmark runs. It reuses the timeline model and renderer
// of the report; the page polls an incremental endpoint, so the server only
// ever serializes what changed since the client's last sequence number.
//
// Cost on the benchmark is one mutex-guarded append per request completion
// plus one map insert/delete per request. Memory is bounded: each arm keeps at
// most maxRequests completed requests (oldest dropped) and at most
// maxInflight in-flight entries are shipped per poll.

const (
	// DefaultLiveMaxRequests bounds the completed requests kept per arm. An 8h
	// run at c28 completes ~70k, so the default holds a full run.
	DefaultLiveMaxRequests = 100000
	// DefaultLiveBind is loopback: a pod's `kubectl port-forward` reaches it
	// there, and nothing else on the network can. Bind a wider address only
	// deliberately.
	DefaultLiveBind = "127.0.0.1"

	liveMaxInflight = 5000
	// liveRateBuckets bounds the run-long token-rate history (5 float64 series).
	liveRateBuckets = 20000
	// liveMaxTTFTSamples bounds the TTFT-over-time history; past it every other
	// sample is dropped so the whole run stays covered at half the resolution.
	liveMaxTTFTSamples = 20000
	livePollMs         = 2000
)

// liveToken identifies an in-flight request. The zero value (no live view
// attached) is inert.
type liveToken struct {
	arm  *liveArm
	id   uint64
	prog *liveProgress
}

// progress is the tracker the response consumers report the first token and
// stream deltas to (nil for the zero token).
func (t liveToken) progressTracker() *liveProgress { return t.prog }

type liveDone struct {
	seq uint64
	req timelineReq
}

type liveInflight struct {
	series, turn, lane, sub int
	start                   time.Time // when the request became eligible (ready)
	sentAt                  time.Time // zero while it still waits for a concurrency slot
	guid                    string
	slot                    slotInfo
	newSession              bool
	prog                    *liveProgress
}

// liveArm is the live state of one model/arm. All methods are safe on a nil
// receiver, so call sites need no "is the live view on" checks.
type liveArm struct {
	name string
	t0   time.Time
	max  int

	mu       sync.Mutex
	seq      uint64
	done     []liveDone
	dropped  int
	lanes    map[rowKey]int
	lastSess map[rowKey]int // series_num last seen on each row, to flag session boundaries
	packers  map[rowKey]*subRowPacker
	inflight map[uint64]liveInflight
	nextID   uint64
	span     float64

	rates       *rateAccum // tokens over time for the whole run, bounded (coarsens)
	ttftSamples []ttftSample
	lastSample  time.Time
	thinking    map[rowKey]int // slots currently pausing before a scheduled turn (count: fan-out)
	slotsFn     func() (admitted, max int)
	gateFn      func() *liveGate
}

// ttftSample is the governor's TTFT window statistics at one instant (ms).
type ttftSample struct{ T, Mean, P50, P95, P99 float64 }

// liveGate is the admission governor's state as of now.
type liveGate struct {
	Enabled    bool    `json:"enabled"`
	Stat       string  `json:"stat"`
	ValueMs    float64 `json:"value_ms"`
	LimitMs    float64 `json:"limit_ms"`
	WindowS    float64 `json:"window_s"`
	N          int     `json:"n"`
	MeanMs     float64 `json:"mean_ms"`
	P50Ms      float64 `json:"p50_ms"`
	P95Ms      float64 `json:"p95_ms"`
	P99Ms      float64 `json:"p99_ms"`
	Open       bool    `json:"open"`
	Capped     bool    `json:"capped"`
	AdmitCount int     `json:"admit_count"`
	EveryS     float64 `json:"every_s"`
	NextInS    float64 `json:"next_in_s"`
	// --ttft-retire: the threshold above which finished sessions' slots retire
	// (0 = off) and how many have.
	RetireMs float64 `json:"retire_ms,omitempty"`
	Retired  int64   `json:"retired,omitempty"`
	// --decode-rate-min: the per-request decode rate now and the minimum.
	DecodeRate float64 `json:"decode_rate,omitempty"`
	DecodeMin  float64 `json:"decode_min,omitempty"`
}

// liveRates are token throughputs (tokens/s) over trailing windows.
type liveRates struct {
	In10  float64 `json:"in10"`
	Out10 float64 `json:"out10"`
	In60  float64 `json:"in60"`
	Out60 float64 `json:"out60"`
}

// liveState counts the admitted session slots by what each is doing now.
type liveState struct {
	Slots    int        `json:"slots"`     // admitted session slots
	Max      int        `json:"max"`       // admission ceiling (--max-series); 0 = none
	InFlight int        `json:"in_flight"` // slots with a request outstanding at the engine
	Waiting  int        `json:"waiting"`   // slots whose next turn is due but waits for a free concurrency slot
	Think    int        `json:"think"`     // slots pausing until their next turn is due
	Idle     int        `json:"idle"`      // slots with no turn pending (between sessions)
	Requests int        `json:"requests"`  // outstanding HTTP requests (a fan-out slot can hold several)
	Rates    *liveRates `json:"rates,omitempty"`
	Gate     *liveGate  `json:"gate,omitempty"`
}

func newLiveArm(name string, max int) *liveArm {
	if max <= 0 {
		max = DefaultLiveMaxRequests
	}
	return &liveArm{name: name, t0: time.Now(), max: max, lanes: map[rowKey]int{}, lastSess: map[rowKey]int{}, rates: newRateAccum(1000, liveRateBuckets), thinking: map[rowKey]int{}, packers: map[rowKey]*subRowPacker{}, inflight: map[uint64]liveInflight{}}
}

// lane returns the stable row group of a slot (or, without one, a series),
// assigned in order of first sight. Callers hold a.mu.
func (a *liveArm) lane(k rowKey) int {
	l, ok := a.lanes[k]
	if !ok {
		l = len(a.lanes)
		a.lanes[k] = l
	}
	return l
}

// newSession reports whether series is a different session from the one the
// row last ran, and records it. Callers hold a.mu.
func (a *liveArm) newSession(k rowKey, series int) bool {
	prev, seen := a.lastSess[k]
	a.lastSess[k] = series
	return !seen || prev != series
}

// packer returns the series' sub-row packer. Callers hold a.mu.
func (a *liveArm) packer(k rowKey) *subRowPacker {
	p := a.packers[k]
	if p == nil {
		p = &subRowPacker{}
		a.packers[k] = p
	}
	return p
}

// begin registers a request that is about to be sent.
//
// ready is when the request became eligible to send; until sent is called it
// shows as client wait (no free concurrency slot).
func (a *liveArm) begin(series, turn int, guid string, sl slotInfo, ready time.Time) liveToken {
	if a == nil {
		return liveToken{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.nextID++
	k := rowKeyOf(sl.N, series)
	sub := a.packer(k).place(float64(ready.Sub(a.t0).Milliseconds()), math.Inf(1))
	prog := &liveProgress{}
	a.inflight[a.nextID] = liveInflight{series: series, turn: turn, lane: a.lane(k), sub: sub, start: ready, guid: guid,
		slot: sl, newSession: a.newSession(k, series), prog: prog}
	return liveToken{arm: a, id: a.nextID, prog: prog}
}

// setSources wires the run's admission state into the live view.
func (a *liveArm) setSources(slots func() (int, int), gate func() *liveGate) {
	if a == nil {
		return
	}
	a.mu.Lock()
	a.slotsFn, a.gateFn = slots, gate
	a.mu.Unlock()
}

// thinkUntil marks the slot as pausing until due (on the pacer's clock `now`).
// It reports whether a pause was registered; pair it with thinkDone.
func (a *liveArm) thinkUntil(sl slotInfo, now, due time.Time) bool {
	if a == nil || sl.N <= 0 || !due.After(now) {
		return false
	}
	a.mu.Lock()
	a.thinking[rowKeyOf(sl.N, 0)]++
	a.mu.Unlock()
	return true
}

// thinkDone ends a pause registered by thinkUntil.
func (a *liveArm) thinkDone(sl slotInfo, registered bool) {
	if a == nil || !registered {
		return
	}
	a.mu.Lock()
	k := rowKeyOf(sl.N, 0)
	if a.thinking[k] <= 1 {
		delete(a.thinking, k)
	} else {
		a.thinking[k]--
	}
	a.mu.Unlock()
}

// state counts slots by activity. In flight beats waiting beats think beats idle.
func (a *liveArm) state() liveState {
	a.mu.Lock()
	var st liveState
	sent, unsent := map[rowKey]bool{}, map[rowKey]bool{}
	for _, f := range a.inflight {
		k := rowKeyOf(f.slot.N, f.series)
		if f.sentAt.IsZero() {
			unsent[k] = true
		} else {
			sent[k] = true
			st.Requests++
		}
	}
	for k := range a.lanes {
		switch {
		case sent[k]:
			st.InFlight++
		case unsent[k]:
			st.Waiting++
		case a.thinking[k] > 0:
			st.Think++
		default:
			st.Idle++
		}
	}
	seen := len(a.lanes)
	slotsFn, gateFn := a.slotsFn, a.gateFn
	nowMs := float64(time.Since(a.t0).Milliseconds())
	st.Rates = a.windowRatesLocked(nowMs)
	a.mu.Unlock()
	st.Slots = seen
	if slotsFn != nil {
		st.Slots, st.Max = slotsFn()
		if st.Slots > seen { // admitted but not yet running a request
			st.Idle += st.Slots - seen
		}
	}
	if gateFn != nil {
		st.Gate = gateFn()
		a.recordTTFT(st.Gate, nowMs)
	}
	return st
}

// windowRatesLocked is the trailing 10s / 60s token rate: completed requests'
// attributed tokens plus what requests still running have shown so far.
func (a *liveArm) windowRatesLocked(nowMs float64) *liveRates {
	r := &liveRates{}
	for _, w := range [2]float64{10000, 60000} {
		from := math.Max(0, nowMs-w)
		in, out := a.rates.windowSum(from, nowMs)
		for _, f := range a.inflight {
			firstNs := f.prog.firstNs.Load()
			if firstNs == 0 {
				continue
			}
			first := float64(firstNs/1e6) - float64(a.t0.UnixMilli())
			if first >= from {
				in += float64(f.prog.promptEst.Load())
			}
			if dur := nowMs - first; dur > 0 {
				overlap := nowMs - math.Max(from, first)
				out += float64(f.prog.out.Load()) * overlap / dur
			}
		}
		// Always divided by the full window: early in a run that under-reports
		// rather than inflating a handful of tokens over a fraction of a second.
		span := w / 1000
		if w == 10000 {
			r.In10, r.Out10 = in/span, out/span
		} else {
			r.In60, r.Out60 = in/span, out/span
		}
	}
	return r
}

// recordTTFT keeps the governor's window statistics over time, one sample per
// couple of seconds, for the TTFT strip.
func (a *liveArm) recordTTFT(g *liveGate, nowMs float64) {
	if g == nil || g.N == 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if time.Since(a.lastSample) < 2*time.Second {
		return
	}
	a.lastSample = time.Now()
	a.ttftSamples = append(a.ttftSamples, ttftSample{nowMs, g.MeanMs, g.P50Ms, g.P95Ms, g.P99Ms})
	if len(a.ttftSamples) > liveMaxTTFTSamples {
		for i := 0; 2*i < len(a.ttftSamples); i++ {
			a.ttftSamples[i] = a.ttftSamples[2*i]
		}
		a.ttftSamples = a.ttftSamples[:(len(a.ttftSamples)+1)/2]
	}
}

// liveSeries is the /api/series response: token rates and TTFT over the run.
type liveSeries struct {
	Rates *rateSeries `json:"rates"`
	TTFT  *ttftSeries `json:"ttft"`
}

// series returns the run-long rate and TTFT history reduced to about n points,
// with in-flight requests' provisional tokens folded into the newest buckets.
func (a *liveArm) series(n int) liveSeries {
	a.mu.Lock()
	defer a.mu.Unlock()
	nowMs := float64(time.Since(a.t0).Milliseconds())
	rs := a.rates.export(n)
	// Fold in what running requests have shown, at the exported resolution.
	prov := newRateAccum(rs.Bw, int(nowMs/rs.Bw)+2)
	for _, f := range a.inflight {
		firstNs := f.prog.firstNs.Load()
		if firstNs == 0 {
			continue
		}
		first := float64(firstNs/1e6) - float64(a.t0.UnixMilli())
		prov.impulse(first, float64(f.prog.promptEst.Load()), 0, 0, 0)
		prov.spread(first, nowMs, float64(f.prog.out.Load()))
	}
	for i := range prov.out {
		for len(rs.In) <= i {
			rs.In, rs.IC, rs.IE, rs.IG, rs.Out = append(rs.In, 0), append(rs.IC, 0), append(rs.IE, 0), append(rs.IG, 0), append(rs.Out, 0)
		}
		rs.In[i] += int(math.Round(prov.in[i]))
		rs.Out[i] += int(math.Round(prov.out[i]))
	}
	ts := &ttftSeries{}
	step := 1
	if n > 0 && len(a.ttftSamples) > n {
		step = (len(a.ttftSamples) + n - 1) / n
	}
	for i := 0; i < len(a.ttftSamples); i += step {
		s := a.ttftSamples[i]
		ts.T = append(ts.T, s.T)
		ts.Mean = append(ts.Mean, s.Mean)
		ts.P50 = append(ts.P50, s.P50)
		ts.P95 = append(ts.P95, s.P95)
		ts.P99 = append(ts.P99, s.P99)
	}
	return liveSeries{Rates: rs, TTFT: ts}
}

// sent marks that the request has been issued (the concurrency gate opened).
func (a *liveArm) sent(tok liveToken, at time.Time) {
	if a == nil || tok.arm != a {
		return
	}
	a.mu.Lock()
	if f, ok := a.inflight[tok.id]; ok {
		f.sentAt = at
		a.inflight[tok.id] = f
	}
	a.mu.Unlock()
}

// complete retires the request tok began (if any) and records its result.
func (a *liveArm) complete(tok liveToken, rec requestDataRecord) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	req := timelineFromRecord(rec, a.t0.UnixMilli())
	req.GUID = rec.SeriesGUID
	k := rowKeyOf(rec.Slot, rec.SeriesNum)
	req.Row = a.lane(k)
	end := req.Start + req.E2E
	if f, ok := a.inflight[tok.id]; ok && tok.arm == a {
		// Keep the sub-row (and boundary flag) the request had while running.
		req.Sub = f.sub
		if f.newSession {
			req.Flags |= flagSession
		}
		a.packer(k).release(f.sub, float64(time.Since(a.t0).Milliseconds()))
		delete(a.inflight, tok.id)
	} else {
		req.Sub = a.packer(k).place(req.Start, end)
		if a.newSession(k, rec.SeriesNum) {
			req.Flags |= flagSession
		}
	}
	a.rates.addRequest(&req)
	a.seq++
	a.done = append(a.done, liveDone{seq: a.seq, req: req})
	if end := req.Start + req.E2E; end > a.span {
		a.span = end
	}
	// Drop in blocks of max/10 so the copy is amortized.
	if over := len(a.done) - a.max; over > a.max/10 {
		n := copy(a.done, a.done[over:])
		clear(a.done[n:])
		a.done = a.done[:n]
		a.dropped += over
	}
}

// abandon retires a begun request that produced no record (skipped,
// cancelled at shutdown).
func (a *liveArm) abandon(tok liveToken) {
	if a == nil || tok.arm != a {
		return
	}
	a.mu.Lock()
	if f, ok := a.inflight[tok.id]; ok {
		a.packer(rowKeyOf(f.slot.N, f.series)).release(f.sub, float64(time.Since(a.t0).Milliseconds()))
		delete(a.inflight, tok.id)
	}
	a.mu.Unlock()
}

// liveData is the /api/data response.
type liveData struct {
	Seq      uint64        `json:"seq"`
	Now      float64       `json:"now"`  // ms since the arm's t0, at response time
	Span     float64       `json:"span"` // ms extent of completed requests
	T0       int64         `json:"t0"`
	Reset    bool          `json:"reset,omitempty"` // client must discard what it holds
	Dropped  int           `json:"dropped,omitempty"`
	Reqs     []timelineReq `json:"reqs"`
	Inflight []timelineReq `json:"inflight"`
	State    *liveState    `json:"state,omitempty"`
}

// snapshot returns the requests completed after since, plus every in-flight
// request.
func (a *liveArm) snapshot(since uint64) liveData {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := time.Now()
	d := liveData{Seq: a.seq, Now: float64(now.Sub(a.t0).Milliseconds()), Span: a.span, T0: a.t0.UnixMilli(), Dropped: a.dropped}
	// A client whose cursor predates the oldest retained request has missed
	// some; it must start over from what is left.
	if len(a.done) > 0 && since > 0 && a.done[0].seq > since+1 {
		d.Reset, since = true, 0
	}
	if since > a.seq { // server restarted under the client
		d.Reset, since = true, 0
	}
	i := sort.Search(len(a.done), func(i int) bool { return a.done[i].seq > since })
	d.Reqs = make([]timelineReq, 0, len(a.done)-i)
	for _, x := range a.done[i:] {
		d.Reqs = append(d.Reqs, x.req)
	}
	d.Inflight = make([]timelineReq, 0, len(a.inflight))
	for _, f := range a.inflight {
		if len(d.Inflight) >= liveMaxInflight {
			break
		}
		q := timelineReq{
			Row: f.lane, Sub: f.sub, Series: f.series, Slot: f.slot.N, Turn: f.turn, GUID: f.guid,
			Start: float64(f.start.Sub(a.t0).Milliseconds()), Flags: flagInflight | f.flags(),
		}
		if f.sentAt.IsZero() {
			q.Flags |= flagWaiting
		} else {
			q.CW = float64(f.sentAt.Sub(f.start).Milliseconds())
		}
		d.Inflight = append(d.Inflight, q)
	}
	return d
}

// liveServer serves the live page and its data API.
type liveServer struct {
	arms []*liveArm
	srv  *http.Server
	ln   net.Listener
	max  int
}

// startLiveServer listens on bind:port and serves until Close. port 0 picks a
// free port (tests).
func startLiveServer(bind string, port int, armNames []string, maxRequests int) (*liveServer, error) {
	if bind == "" {
		bind = DefaultLiveBind
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(bind, strconv.Itoa(port)))
	if err != nil {
		return nil, fmt.Errorf("--live-port: %w", err)
	}
	if maxRequests <= 0 {
		maxRequests = DefaultLiveMaxRequests
	}
	s := &liveServer{ln: ln, max: maxRequests}
	for _, n := range armNames {
		s.arms = append(s.arms, newLiveArm(n, maxRequests))
	}
	s.srv = &http.Server{Handler: s.handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := s.srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Printf("live view: server stopped: %v\n", err)
		}
	}()
	return s, nil
}

func (s *liveServer) Addr() string { return s.ln.Addr().String() }

// Close stops the server.
func (s *liveServer) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = s.srv.Shutdown(ctx)
}

func (s *liveServer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		page, err := renderTimelinePage("wekai live timeline", "connecting…", nil, timelinePageConfig{
			Live: true, PollMs: livePollMs, MaxRequests: s.max, Kinds: timelineKinds,
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(page)
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	mux.HandleFunc("/api/arms", func(w http.ResponseWriter, r *http.Request) {
		type armInfo struct {
			Name string `json:"name"`
			T0   int64  `json:"t0"`
		}
		out := struct {
			Kinds []timelineKind `json:"kinds"`
			Arms  []armInfo      `json:"arms"`
		}{Kinds: timelineKinds}
		for _, a := range s.arms {
			out.Arms = append(out.Arms, armInfo{Name: a.name, T0: a.t0.UnixMilli()})
		}
		writeJSON(w, out)
	})
	mux.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
		i := s.armIndex(r.URL.Query().Get("arm"))
		if i < 0 {
			http.Error(w, "unknown arm (use an index or an arm name from /api/arms)", http.StatusBadRequest)
			return
		}
		var since uint64
		if v := r.URL.Query().Get("since"); v != "" {
			var err error
			since, err = strconv.ParseUint(v, 10, 64)
			if err != nil {
				http.Error(w, "bad since", http.StatusBadRequest)
				return
			}
		}
		d := s.arms[i].snapshot(since)
		st := s.arms[i].state()
		d.State = &st
		writeJSON(w, d)
	})
	mux.HandleFunc("/api/series", func(w http.ResponseWriter, r *http.Request) {
		i := s.armIndex(r.URL.Query().Get("arm"))
		if i < 0 {
			http.Error(w, "unknown arm (use an index or an arm name from /api/arms)", http.StatusBadRequest)
			return
		}
		n, _ := strconv.Atoi(r.URL.Query().Get("points"))
		if n <= 0 || n > 4000 {
			n = 1500
		}
		out := s.arms[i].series(n)
		if g := s.arms[i].state().Gate; g != nil {
			out.TTFT.LimitMs, out.TTFT.WindowS, out.TTFT.Stat = g.LimitMs, g.WindowS, g.Stat
		}
		writeJSON(w, out)
	})
	mux.HandleFunc("/api/state", func(w http.ResponseWriter, r *http.Request) {
		i := s.armIndex(r.URL.Query().Get("arm"))
		if i < 0 {
			i = 0
		}
		if len(s.arms) == 0 {
			http.Error(w, "no arms", http.StatusNotFound)
			return
		}
		writeJSON(w, s.arms[i].state())
	})
	return mux
}

// armIndex resolves an arm given as an index or a name; -1 if unknown.
func (s *liveServer) armIndex(v string) int {
	for i, a := range s.arms {
		if a.name == v {
			return i
		}
	}
	if i, err := strconv.Atoi(v); err == nil && i >= 0 && i < len(s.arms) {
		return i
	}
	return -1
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(v)
}

// flags returns the per-request flag bits an in-flight request carries.
func (f liveInflight) flags() int {
	x := 0
	if f.newSession {
		x |= flagSession
	}
	if f.slot.Hot {
		x |= flagHot
	}
	return x
}
