package benchmark

import (
	"io"
	"net/http"
	"sync/atomic"
)

// Counting what is genuinely outstanding at the server, as distinct from what
// the client has started working on.
//
// The progress line's in_flight has always been the concurrency gate's active
// counter: a slot is taken before the request body is built and released after
// the response is consumed. That is a true statement about the CLIENT — how many
// requests it has in hand — and it was being read as a statement about the
// FLEET. Measured against the backends' own num_requests_running plus
// num_requests_waiting, it ran about 35% high, with the fleet's queue in the
// teens so the difference was not vLLM-side queueing.
//
// The gap is real work in a real place: these prompts average ~159k tokens, so
// each request synthesises and marshals several hundred kilobytes before a
// socket is written, and the upload itself takes time at six hundred concurrent.
// A request in that state is outstanding from the client and does not exist yet
// at the backend, and both facts matter — one is the client's own cost, the
// other is the concurrency the fleet actually carried.
//
// So the two are counted separately rather than one being corrected into the
// other. A transport is the right place: RoundTrip returns when the response
// headers arrive, and the body's Close is when the exchange is finished, which
// brackets exactly the interval a request is the server's problem.
type inflightTransport struct {
	base http.RoundTripper
	n    *atomic.Int64
}

func newInflightTransport(base http.RoundTripper, n *atomic.Int64) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	// No counter, no wrapper. A caller with nothing to report should get plain
	// transport behaviour rather than a panic on the first request — counting is
	// an observation, and an observation must never be able to break the thing
	// it observes.
	if n == nil {
		return base
	}
	return &inflightTransport{base: base, n: n}
}

func (t *inflightTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	t.n.Add(1)
	resp, err := t.base.RoundTrip(r)
	if err != nil {
		t.n.Add(-1)
		return nil, err
	}
	// Held until the body is closed, not until the headers land. A streaming
	// completion returns its headers immediately and then occupies the backend
	// for the whole generation; releasing at RoundTrip would report a fleet
	// doing nothing while it decoded.
	resp.Body = &countedBody{ReadCloser: resp.Body, n: t.n}
	return resp, nil
}

// countedBody decrements once, however many times Close is called — a
// double-close would otherwise drive the count negative and, worse, make it
// drift steadily wrong rather than obviously wrong.
type countedBody struct {
	io.ReadCloser
	n      *atomic.Int64
	closed atomic.Bool
}

func (b *countedBody) Close() error {
	if b.closed.CompareAndSwap(false, true) {
		b.n.Add(-1)
	}
	return b.ReadCloser.Close()
}
