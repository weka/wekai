package benchmark

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	livePollMs      = 2000
)

// liveToken identifies an in-flight request. The zero value (no live view
// attached) is inert.
type liveToken struct {
	arm *liveArm
	id  uint64
}

type liveDone struct {
	seq uint64
	req timelineReq
}

type liveInflight struct {
	series, turn, lane int
	start              time.Time
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
	lanes    map[int]int
	inflight map[uint64]liveInflight
	nextID   uint64
	span     float64
}

func newLiveArm(name string, max int) *liveArm {
	if max <= 0 {
		max = DefaultLiveMaxRequests
	}
	return &liveArm{name: name, t0: time.Now(), max: max, lanes: map[int]int{}, inflight: map[uint64]liveInflight{}}
}

// lane returns the series' stable row, assigned in order of first sight.
// Callers hold a.mu.
func (a *liveArm) lane(series int) int {
	l, ok := a.lanes[series]
	if !ok {
		l = len(a.lanes)
		a.lanes[series] = l
	}
	return l
}

// begin registers a request that is about to be sent.
func (a *liveArm) begin(series, turn int) liveToken {
	if a == nil {
		return liveToken{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.nextID++
	a.inflight[a.nextID] = liveInflight{series: series, turn: turn, lane: a.lane(series), start: time.Now()}
	return liveToken{arm: a, id: a.nextID}
}

// complete retires the request tok began (if any) and records its result.
func (a *liveArm) complete(tok liveToken, rec requestDataRecord) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if tok.arm == a {
		delete(a.inflight, tok.id)
	}
	req := timelineFromRecord(rec, a.t0.UnixMilli())
	req.Row = a.lane(rec.SeriesNum)
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
	delete(a.inflight, tok.id)
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
		d.Inflight = append(d.Inflight, timelineReq{
			Row: f.lane, Series: f.series, Turn: f.turn,
			Start: float64(f.start.Sub(a.t0).Milliseconds()), Flags: flagInflight,
		})
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
			Name   string `json:"name"`
			T0     int64  `json:"t0"`
			Layout string `json:"layout"`
		}
		out := struct {
			Kinds []timelineKind `json:"kinds"`
			Arms  []armInfo      `json:"arms"`
		}{Kinds: timelineKinds}
		for _, a := range s.arms {
			out.Arms = append(out.Arms, armInfo{Name: a.name, T0: a.t0.UnixMilli(), Layout: "series"})
		}
		writeJSON(w, out)
	})
	mux.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
		i, err := strconv.Atoi(r.URL.Query().Get("arm"))
		if err != nil || i < 0 || i >= len(s.arms) {
			http.Error(w, "unknown arm", http.StatusBadRequest)
			return
		}
		var since uint64
		if v := r.URL.Query().Get("since"); v != "" {
			since, err = strconv.ParseUint(v, 10, 64)
			if err != nil {
				http.Error(w, "bad since", http.StatusBadRequest)
				return
			}
		}
		writeJSON(w, s.arms[i].snapshot(since))
	})
	return mux
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(v)
}
