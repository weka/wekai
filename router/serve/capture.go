package serve

import (
	"bytes"
	"io"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/router/internal/obs"
)

// Captured is one proxied request/response pair, as the capture sink sees it.
type Captured struct {
	ID             uint64
	Started        time.Time
	Request        *http.Request
	InboundHeaders http.Header
	ReqBody        []byte
	RespHeaders    http.Header
	RespBody       []byte
	Status         int
	// Pool, Backend and ModelOut are the routing outcome. A record that cannot
	// say where a request went is not much of a record.
	Pool     string
	Backend  string
	ModelOut string
	Total    time.Duration
}

// CaptureHook receives each proxied exchange. It is an interface so this
// package needs to know nothing about capture formats, redaction or sinks —
// those live in cli/, which owns the schema the replay tooling reads.
type CaptureHook interface {
	// WantsResponseBody reports whether response bodies are needed. A redacted
	// capture keeps none, and teeing one it will discard is pure cost on every
	// streamed token.
	WantsResponseBody() bool
	Record(Captured)
}

// captureMiddleware records each exchange, wrapping the routing gateway rather
// than living inside it.
//
// Capture used to be interleaved through the old proxy handler's own request
// lifecycle, which is why the two routers could not merge without it: the
// gateway streams responses and knows nothing about capture, and the capture
// code knew everything about a proxy that no longer exists. As middleware it
// needs exactly two things from inside — which pool served the request, and
// which upstream — and those arrive through obs.RouteInfo, the same
// fill-in-from-the-handler indirection the access log already uses.
//
// Response bodies are teed, not buffered whole and replayed: a streaming
// generation runs for minutes and can reach tens of megabytes, and holding one
// to rewrite it afterwards would add that to live memory for every concurrent
// stream.
func captureMiddleware(hook CaptureHook, next http.Handler) http.Handler {
	if hook == nil {
		return next
	}
	var seq atomic.Uint64
	keepBody := hook.WantsResponseBody()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		//clockexempt: measures the exchange for the capture record, not a decision
		started := time.Now()
		id := seq.Add(1)
		// Snapshotted before anything downstream can strip credentials: a
		// record should say what the client actually sent.
		inbound := r.Header.Clone()

		var reqBody []byte
		if r.Body != nil && r.Method != http.MethodGet {
			// Already bounded by the gateway's MaxBytesReader.
			if b, err := io.ReadAll(r.Body); err == nil {
				reqBody = b
				r.Body = io.NopCloser(bytes.NewReader(b))
			}
		}

		// Installed HERE, outside the gateway, so the holder capture reads is the
		// same one the handlers within write to. The gateway's own access log
		// reuses it rather than replacing it.
		rctx, ri := obs.WithRouteHolder(r.Context())
		r = r.WithContext(rctx)

		cw := &captureWriter{ResponseWriter: w, keepBody: keepBody}
		next.ServeHTTP(cw, r)

		hook.Record(Captured{
			ID: id, Started: started, Request: r, InboundHeaders: inbound,
			ReqBody: reqBody, RespHeaders: cw.Header(), RespBody: cw.body.Bytes(),
			Status: cw.statusOr(http.StatusOK), Pool: ri.Pool, Backend: ri.Backend,
			//clockexempt: measures the exchange for the capture record, not a decision
			ModelOut: ri.ModelOut, Total: time.Since(started),
		})
	})
}

// captureWriter records the status and, when wanted, tees a bounded prefix of
// the body.
//
// It forwards Flush because the responses it wraps are usually SSE streams:
// without that the chunks pile up and a client sees a whole generation arrive
// at once, which is the difference between a working router and one that
// appears to hang.
type captureWriter struct {
	http.ResponseWriter
	status   int
	keepBody bool
	body     bytes.Buffer
}

func (c *captureWriter) WriteHeader(status int) {
	c.status = status
	c.ResponseWriter.WriteHeader(status)
}

func (c *captureWriter) Write(b []byte) (int, error) {
	if c.status == 0 {
		c.status = http.StatusOK
	}
	if c.keepBody && c.body.Len() < captureBodyLimit {
		c.body.Write(b[:min(len(b), captureBodyLimit-c.body.Len())])
	}
	return c.ResponseWriter.Write(b)
}

func (c *captureWriter) Flush() {
	if f, ok := c.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (c *captureWriter) statusOr(def int) int {
	if c.status == 0 {
		return def
	}
	return c.status
}

// captureBodyLimit bounds the retained response body per record: a capture is a
// record of an exchange, not a copy of the whole generation.
const captureBodyLimit = 1 << 20
