package benchmark

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// in_flight must mean "the server has this request", not "the client is working
// on it". Measured against the backends' own counters the gate figure ran about
// a third high, and it was being quoted as the concurrency the fleet carried.

func TestInflightHeldUntilTheBodyIsClosed(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush() // headers out, generation continues
		<-release
		_, _ = io.WriteString(w, "done")
	}))
	defer srv.Close()

	var n atomic.Int64
	c := &http.Client{Transport: newInflightTransport(nil, &n)}

	done := make(chan *http.Response, 1)
	go func() {
		resp, err := c.Get(srv.URL)
		if err != nil {
			t.Errorf("get: %v", err)
			done <- nil
			return
		}
		done <- resp
	}()

	resp := <-done
	if resp == nil {
		t.Fatal("no response")
	}
	// Headers have arrived and the exchange is still open: a streaming
	// completion looks exactly like this for the whole generation, and releasing
	// here would report a fleet doing nothing while it decoded.
	if got := n.Load(); got != 1 {
		t.Errorf("in-flight = %d after headers, want 1", got)
	}
	close(release)
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if got := n.Load(); got != 0 {
		t.Errorf("in-flight = %d after the body was closed, want 0", got)
	}
}

// TestInflightDoesNotLeakOnTransportError: a failed round trip never produces a
// body to close, so the decrement has to happen on that path or the count
// ratchets up forever and reports a fleet that is busier every hour.
func TestInflightDoesNotLeakOnTransportError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	addr := srv.URL
	srv.Close() // refused

	var n atomic.Int64
	c := &http.Client{Transport: newInflightTransport(nil, &n), Timeout: 2 * time.Second}
	if _, err := c.Get(addr); err == nil {
		t.Fatal("expected a transport error")
	}
	if got := n.Load(); got != 0 {
		t.Errorf("in-flight = %d after a failed round trip, want 0", got)
	}
}

// TestInflightSurvivesADoubleClose. A double close would drive the count
// negative — and a counter that drifts steadily wrong is worse than one that is
// obviously broken, because it keeps looking plausible.
func TestInflightSurvivesADoubleClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	defer srv.Close()

	var n atomic.Int64
	c := &http.Client{Transport: newInflightTransport(nil, &n)}
	resp, err := c.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	resp.Body.Close()
	if got := n.Load(); got != 0 {
		t.Errorf("in-flight = %d after a double close, want 0", got)
	}
}

// TestNilCounterIsAPlainTransport: an observation must never be able to break
// the thing it observes.
func TestNilCounterIsAPlainTransport(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	defer srv.Close()

	c := &http.Client{Transport: newInflightTransport(nil, nil)}
	resp, err := c.Get(srv.URL)
	if err != nil {
		t.Fatalf("get with no counter: %v", err)
	}
	resp.Body.Close()
}
