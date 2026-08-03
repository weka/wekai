// Package mockvllm is a programmable stand-in for a vLLM worker (AC-0.1).
//
// It exists so every failure mode the router has to survive can be produced on
// demand: latency, error statuses, SSE streaming with a configurable inter-token
// delay, mid-stream abort, connection reset, and a slow body. It also records the
// headers it received, which is how tests assert that the client's credential
// never reaches a backend and that request ids are preserved.
package mockvllm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"
)

// Script describes how the worker should respond.
type Script struct {
	Status      int
	ContentType string
	Body        string

	// TTFT delays the response headers; InterToken delays each SSE chunk.
	TTFT       time.Duration
	InterToken time.Duration

	// Stream emits Tokens SSE chunks followed by the OpenAI terminal marker.
	Stream bool
	Tokens int

	// AbortAfter truncates a stream after N chunks without a terminal marker.
	AbortAfter int
	// Hijack closes the connection abruptly, simulating a reset.
	Hijack bool

	// Usage, when set, is appended as an OpenAI usage block on non-streaming
	// responses so cache-prediction accuracy can be measured (RES-3).
	PromptTokens int
	CachedTokens int

	// Unhealthy makes /health fail while inference still works — the WEKA
	// degraded-mode shape, where a node is slower but not broken (WEKA-1).
	Unhealthy bool
}

// Call records one received request.
type Call struct {
	Method string
	Path   string
	Header http.Header
	Body   string
}

type Worker struct {
	srv *httptest.Server

	mu       sync.Mutex
	script   Script
	behave   func(*http.Request) Script
	calls    []Call
	healthOK bool
}

// New starts a worker with a default 200/JSON script.
func New() *Worker {
	w := &Worker{
		script:   Script{Status: 200, ContentType: "application/json", Body: `{"ok":true}`},
		healthOK: true,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", w.health)
	mux.HandleFunc("/", w.handle)
	w.srv = httptest.NewServer(mux)
	return w
}

func (w *Worker) URL() string { return w.srv.URL }
func (w *Worker) Close()      { w.srv.Close() }

// SetScript replaces the response script.
func (w *Worker) SetScript(s Script) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if s.Status == 0 {
		s.Status = 200
	}
	w.script = s
}

// Behave programs a per-request script, for tests that need the first attempt to
// fail and the retry to succeed.
func (w *Worker) Behave(fn func(*http.Request) Script) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.behave = fn
}

// SetHealthy controls /health independently of inference, so a test can model a
// backend that is degraded but still serving.
func (w *Worker) SetHealthy(ok bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.healthOK = ok
}

// Calls returns the requests received so far.
func (w *Worker) Calls() []Call {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]Call(nil), w.calls...)
}

func (w *Worker) CallCount() int { return len(w.Calls()) }

// LastCall returns the most recent call, or false if there were none.
func (w *Worker) LastCall() (Call, bool) {
	c := w.Calls()
	if len(c) == 0 {
		return Call{}, false
	}
	return c[len(c)-1], true
}

func (w *Worker) health(rw http.ResponseWriter, r *http.Request) {
	w.mu.Lock()
	ok := w.healthOK
	w.mu.Unlock()
	if !ok {
		rw.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	rw.WriteHeader(http.StatusOK)
}

func (w *Worker) handle(rw http.ResponseWriter, r *http.Request) {
	buf := make([]byte, 0, 1024)
	if r.Body != nil {
		tmp := make([]byte, 4096)
		for {
			n, err := r.Body.Read(tmp)
			buf = append(buf, tmp[:n]...)
			if err != nil {
				break
			}
		}
	}

	w.mu.Lock()
	s := w.script
	if w.behave != nil {
		s = w.behave(r)
		if s.Status == 0 {
			s.Status = 200
		}
	}
	w.calls = append(w.calls, Call{
		Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: string(buf),
	})
	w.mu.Unlock()

	if s.TTFT > 0 {
		time.Sleep(s.TTFT)
	}
	if s.Hijack {
		if hj, ok := rw.(http.Hijacker); ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				_ = conn.Close()
				return
			}
		}
	}

	if !s.Stream {
		ct := s.ContentType
		if ct == "" {
			ct = "application/json"
		}
		rw.Header().Set("Content-Type", ct)
		rw.WriteHeader(s.Status)
		body := s.Body
		if body == "" && s.PromptTokens > 0 {
			body = fmt.Sprintf(
				`{"usage":{"prompt_tokens":%d,"total_tokens":%d,"prompt_tokens_details":{"cached_tokens":%d}}}`,
				s.PromptTokens, s.PromptTokens, s.CachedTokens)
		}
		_, _ = rw.Write([]byte(body))
		return
	}

	rw.Header().Set("Content-Type", "text/event-stream")
	rw.WriteHeader(s.Status)
	flusher, _ := rw.(http.Flusher)
	n := s.Tokens
	if n == 0 {
		n = 3
	}
	for i := 0; i < n; i++ {
		if s.AbortAfter > 0 && i >= s.AbortAfter {
			return // truncated: no terminal marker
		}
		if s.InterToken > 0 {
			time.Sleep(s.InterToken)
		}
		if _, err := fmt.Fprintf(rw, "data: {\"choices\":[{\"delta\":{\"content\":\"t%d\"}}]}\n\n", i); err != nil {
			return
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	_, _ = rw.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
}
