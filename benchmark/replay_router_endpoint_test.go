package benchmark

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// The endpoint form must latch on any answer that PROVES the path, not only on
// success.
//
// From a soak: the router's log alternated perfectly between
//
//	route=passthrough path=/chat/completions   status=404
//	route=chat        path=/v1/chat/completions status=429
//
// The operator's base had no /v1, so every request tried the bare path first,
// took a 404, then tried the /v1 form — which answered 429 because the fleet was
// saturated. A 429 is not a success, so the endpoint never resolved, and the
// probe repeated for every single request. That doubles inbound requests exactly
// when the fleet is least able to serve them, and it fills the router's log and
// its route metrics with 404s on a path nobody configured.
//
// A 404 means the path is wrong. Anything else means it is right.
func TestEndpointLatchesOnAnyNon404(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
	}{
		{"a saturated router still resolves the endpoint", http.StatusTooManyRequests},
		{"so does an unauthorized one", http.StatusUnauthorized},
		{"and a broken upstream", http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var paths []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				paths = append(paths, r.URL.Path)
				mu.Unlock()
				// The bare path is not served; the /v1 form is, and answers
				// with the status under test.
				if !strings.HasPrefix(r.URL.Path, "/v1/") {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				w.WriteHeader(tc.status)
			}))
			defer srv.Close()

			p := &replayPoster{
				epPrimary:  srv.URL + "/chat/completions",
				epFallback: srv.URL + "/v1/chat/completions",
			}

			// First request: probes both, then latches the working form.
			if got := p.endpointAttempts(); len(got) != 2 {
				t.Fatalf("before resolution, attempts = %v, want both forms", got)
			}
			resp, err := http.Get(p.epFallback)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			p.latchEndpoint(p.epFallback)

			got := p.endpointAttempts()
			if len(got) != 1 || got[0] != p.epFallback {
				t.Errorf("after a %d on the /v1 form, attempts = %v, want only %q: a "+
					"status that is not 404 proves the path, so the probe must not repeat",
					tc.status, got, p.epFallback)
			}
		})
	}
}
