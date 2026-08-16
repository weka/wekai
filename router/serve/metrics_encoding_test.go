package serve

import (
	"bytes"
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/weka/wekai/router/internal/metrics"
)

// The metrics endpoint carries TWO expositions — the router's own collectors and
// the upstream vLLM totals appended after them — and both have to arrive inside
// one encoding.
//
// Letting the Prometheus handler negotiate compression and then appending plain
// text to the same ResponseWriter produced [gzip member][plain text]. Go's gzip
// reader is multistream: having finished the first member it reads on for the
// next one, hits the appended text, and fails with "gzip: invalid header". The
// scrape then SUCCEEDS without Accept-Encoding and FAILS with it — and every Go
// client sends it, because the transport adds the header itself and decompresses
// transparently. A whole benchmark campaign recorded zero upstream samples that
// way while a curl check of the first bytes looked fine.

// appendingHandler is the shape the router's /metrics mux has: a Prometheus
// handler, then a second writer.
func appendingHandler(trailer string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var body bytes.Buffer
		rec := &metricsBuffer{hdr: http.Header{}, buf: &body, status: http.StatusOK}
		metrics.Handler(metrics.Registry()).ServeHTTP(rec, r)
		_, _ = body.WriteString(trailer)

		h := w.Header()
		for k, v := range rec.hdr {
			h[k] = v
		}
		h.Del("Content-Length")
		if acceptsGzip(r) {
			h.Set("Content-Encoding", "gzip")
			w.WriteHeader(rec.status)
			zw := gzip.NewWriter(w)
			_, _ = zw.Write(body.Bytes())
			_ = zw.Close()
			return
		}
		w.WriteHeader(rec.status)
		_, _ = w.Write(body.Bytes())
	}
}

const upstreamTrailer = "vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\"} 39669683456\n"

// TestMetricsScrapeSurvivesTransparentGzip drives the endpoint through a real
// http.Client, which is what makes this a regression test rather than a unit
// test: the transport adds Accept-Encoding and decompresses on the way back, so
// a body that is only valid uncompressed fails here and nowhere else.
func TestMetricsScrapeSurvivesTransparentGzip(t *testing.T) {
	srv := httptest.NewServer(appendingHandler(upstreamTrailer))
	defer srv.Close()

	resp, err := http.Get(srv.URL) // no explicit Accept-Encoding: the transport adds it
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v — the appended exposition is outside the encoding the handler "+
			"declared, so every Go client's scrape fails while curl without Accept-Encoding "+
			"succeeds", err)
	}
	if !strings.Contains(string(body), "vllm:prompt_tokens_by_source_total") {
		t.Error("the upstream totals are missing; appending them is the whole reason one scrape " +
			"target covers both")
	}
	if !strings.Contains(string(body), "router_") {
		t.Error("the router's own collectors are missing")
	}
}

// TestMetricsScrapeWithExplicitGzip covers the client that asks for gzip itself,
// which disables the transport's transparent decode and hands back the raw body.
func TestMetricsScrapeWithExplicitGzip(t *testing.T) {
	srv := httptest.NewServer(appendingHandler(upstreamTrailer))
	defer srv.Close()

	req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept-Encoding", "gzip")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	defer resp.Body.Close()
	if resp.Header.Get("Content-Encoding") != "gzip" {
		t.Fatalf("Content-Encoding = %q, want gzip", resp.Header.Get("Content-Encoding"))
	}

	zr, err := gzip.NewReader(resp.Body)
	if err != nil {
		t.Fatalf("gzip.NewReader: %v", err)
	}
	defer zr.Close()
	body, err := io.ReadAll(zr)
	if err != nil {
		// This is the exact failure the fleet saw. Reading the FIRST member
		// succeeds, which is why a curl check of the leading bytes passes; the
		// reader then looks for a second member and finds plain text.
		t.Fatalf("read gzip: %v — the body is a valid member followed by unencoded bytes", err)
	}
	if !strings.Contains(string(body), "vllm:prompt_tokens_by_source_total") {
		t.Error("upstream totals missing from the compressed body")
	}
}

// TestMetricsScrapeUncompressed: the path that always worked must keep working,
// and must not claim an encoding it did not apply.
func TestMetricsScrapeUncompressed(t *testing.T) {
	srv := httptest.NewServer(appendingHandler(upstreamTrailer))
	defer srv.Close()

	req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept-Encoding", "identity")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("Content-Encoding"); got != "" {
		t.Errorf("Content-Encoding = %q on a client that did not offer gzip", got)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "vllm:prompt_tokens_by_source_total") {
		t.Error("upstream totals missing")
	}
}

func TestAcceptsGzipMatchesTheToken(t *testing.T) {
	for _, tc := range []struct {
		hdr  string
		want bool
	}{
		{"gzip", true},
		{"gzip, deflate", true},
		{"deflate, gzip;q=1.0", true},
		{" GZIP ", true},
		{"", false},
		{"identity", false},
		{"deflate", false},
		// Substring matching would false-positive on both of these, and
		// declaring gzip we did not apply is the same corruption in reverse.
		{"x-gzip", false},
		{"gzipfoo", false},
	} {
		r, err := http.NewRequest(http.MethodGet, "http://x/metrics", nil)
		if err != nil {
			t.Fatal(err)
		}
		if tc.hdr != "" {
			r.Header.Set("Accept-Encoding", tc.hdr)
		}
		if got := acceptsGzip(r); got != tc.want {
			t.Errorf("Accept-Encoding %q -> %v, want %v", tc.hdr, got, tc.want)
		}
	}
}
