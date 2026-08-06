package viz_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/viz"
)

// fakeSource is a minimal, in-package-independent viz.DataSource for
// testing the HTTP layer without any dependency on the real
// router/internal/policy/cache implementation (kept isolated per the
// package's own doc: viz knows nothing about kvcache or policies).
type fakeSource struct {
	snap     viz.Snapshot
	lastOpts viz.SnapshotOptions
	calls    int
}

func (f *fakeSource) Snapshot(opts viz.SnapshotOptions) viz.Snapshot {
	f.calls++
	f.lastOpts = opts
	return f.snap
}

func TestPageHandler_ServesHTML(t *testing.T) {
	ts := httptest.NewServer(viz.PageHandler())
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("Content-Type = %q, want text/html prefix", ct)
	}

	buf := make([]byte, 4096)
	n, _ := resp.Body.Read(buf)
	body := string(buf[:n])
	// The page must actually reference its own data endpoint and contain
	// the DOCTYPE — a cheap structural check that this is real HTML, not an
	// empty or truncated embed.
	if !strings.Contains(body, "<!DOCTYPE html>") {
		t.Fatalf("page body missing <!DOCTYPE html>, got: %.200s", body)
	}
}

// TestPageHandler_ReferencesRelativeDataEndpoint reads the FULL embedded
// page (not just the first buffer) and checks the JS derives its fetch URL
// from location.pathname rather than a hardcoded absolute path — this is
// what makes the page work unmodified if ever mounted under a different
// prefix, and it's the one thing a pure httptest 200-check can't otherwise
// verify since httptest doesn't execute JS.
func TestPageHandler_ReferencesRelativeDataEndpoint(t *testing.T) {
	ts := httptest.NewServer(viz.PageHandler())
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	body := new(strings.Builder)
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			body.Write(buf[:n])
		}
		if err != nil {
			break
		}
	}
	html := body.String()
	if !strings.Contains(html, "location.pathname") {
		t.Fatalf("page JS does not derive its data URL from location.pathname (hardcoded path risk)")
	}
	if !strings.Contains(html, "'/data'") {
		t.Fatalf("page JS does not append '/data' to its own path")
	}
	if strings.Contains(html, "cdn.") || strings.Contains(html, "unpkg.com") || strings.Contains(html, "googleapis.com") {
		t.Fatalf("page references an external CDN, violates the self-contained requirement")
	}
}

// TestPageHandler_HasUIControlsNotJustQueryParams is anton's explicit
// requirement: any limiting must be a visible UI control the user sets, not
// a URL query parameter they have to know about. Checks the page defines
// input elements for the three options and reads THEIR values (not
// location.search) when building its fetch — a structural proxy for "the
// configuration surface is the UI," since httptest can't execute the JS.
func TestPageHandler_HasUIControlsNotJustQueryParams(t *testing.T) {
	ts := httptest.NewServer(viz.PageHandler())
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	body := new(strings.Builder)
	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			body.Write(buf[:n])
		}
		if err != nil {
			break
		}
	}
	html := body.String()
	for _, id := range []string{"ctl-rows", "ctl-children", "ctl-depth"} {
		if !strings.Contains(html, `id="`+id+`"`) {
			t.Fatalf("page missing UI control #%s", id)
		}
	}
	// The page must not seed its request params from its own URL — the UI
	// inputs are the only configuration surface.
	if strings.Contains(html, "location.search") {
		t.Fatalf("page reads location.search — configuration must come from the UI controls only")
	}
}

func TestDataHandler_NilSourceReportsInactive(t *testing.T) {
	ts := httptest.NewServer(viz.DataHandler(nil))
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var snap viz.Snapshot
	if err := json.NewDecoder(resp.Body).Decode(&snap); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if snap.PolicyActive {
		t.Fatalf("PolicyActive = true with a nil DataSource, want false")
	}
	if len(snap.Backends) != 0 || len(snap.Tree) != 0 {
		t.Fatalf("expected an empty snapshot, got %+v", snap)
	}
}

func TestDataHandler_ServesJSONShape(t *testing.T) {
	healthy := true
	src := &fakeSource{snap: viz.Snapshot{
		GeneratedAt:  time.Now(),
		PolicyActive: true,
		Tree: []viz.TreeNode{
			{Hash: "abcd1234", RunLen: 1, Tokens: 16, Depth: 0, Parent: -1, Children: []int{1}, Present: []bool{true}},
			{Hash: "ef567890", RunLen: 1, Tokens: 16, Depth: 1, Parent: 0, Present: []bool{true}},
		},
		Backends: []viz.BackendMeta{
			{URL: "http://w0:8000", Healthy: &healthy, Inflight: 2, Nodes: 2, Tokens: 32},
		},
		AvgCopies:  1.0,
		NodesShown: 2,
		NodesTotal: 2,
	}}

	ts := httptest.NewServer(viz.DataHandler(src))
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}

	var snap viz.Snapshot
	if err := json.NewDecoder(resp.Body).Decode(&snap); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !snap.PolicyActive {
		t.Fatalf("PolicyActive = false, want true")
	}
	if len(snap.Tree) != 2 {
		t.Fatalf("len(Tree) = %d, want 2", len(snap.Tree))
	}
	if snap.Tree[0].Parent != -1 || len(snap.Tree[0].Children) != 1 || snap.Tree[0].Children[0] != 1 {
		t.Fatalf("root/child linkage did not round-trip: %+v", snap.Tree[0])
	}
	if snap.Tree[1].Parent != 0 {
		t.Fatalf("child's Parent = %d, want 0", snap.Tree[1].Parent)
	}
	if len(snap.Backends) != 1 || snap.Backends[0].URL != "http://w0:8000" {
		t.Fatalf("unexpected Backends: %+v", snap.Backends)
	}
	if snap.Backends[0].Healthy == nil || !*snap.Backends[0].Healthy {
		t.Fatalf("Healthy = %v, want true", snap.Backends[0].Healthy)
	}
}

// TestDataHandler_NoParamsMeansUnlimited is anton's explicit requirement at
// the wire level: a request with no query params at all must reach the
// DataSource as a completely zero SnapshotOptions — unlimited in every
// dimension, no hidden default cap.
func TestDataHandler_NoParamsMeansUnlimited(t *testing.T) {
	src := &fakeSource{}
	ts := httptest.NewServer(viz.DataHandler(src))
	defer ts.Close()

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	resp.Body.Close()
	if src.lastOpts != (viz.SnapshotOptions{}) {
		t.Fatalf("lastOpts = %+v, want the zero value (unlimited) with no query params", src.lastOpts)
	}
}

// TestDataHandler_QueryParamsSetOptions checks each of the three
// UI-settable knobs is parsed into SnapshotOptions correctly, and that an
// explicitly-huge value is clamped rather than passed through unbounded —
// these params are the mechanism the page's OWN UI controls use; a client
// is still free to hit them directly (e.g. for scripting), but nothing
// defaults to using them.
func TestDataHandler_QueryParamsSetOptions(t *testing.T) {
	src := &fakeSource{}
	ts := httptest.NewServer(viz.DataHandler(src))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "?limit=15&max_children=3&max_depth=7")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	resp.Body.Close()
	want := viz.SnapshotOptions{MaxRows: 15, MaxChildren: 3, MaxDepth: 7}
	if src.lastOpts != want {
		t.Fatalf("lastOpts = %+v, want %+v", src.lastOpts, want)
	}

	// An explicitly huge value must be clamped, not passed through
	// unbounded — this is what stops a client forcing an unbounded walk
	// while still keeping "omitted -> unlimited" true.
	resp2, err := http.Get(ts.URL + "?limit=999999999")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	resp2.Body.Close()
	if src.lastOpts.MaxRows != viz.MaxParamValue {
		t.Fatalf("MaxRows = %d, want clamped to MaxParamValue=%d", src.lastOpts.MaxRows, viz.MaxParamValue)
	}

	// A malformed or non-positive value must fall back to unlimited (0)
	// rather than erroring the whole page.
	resp3, err := http.Get(ts.URL + "?limit=not-a-number&max_children=-5&max_depth=0")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp3.Body.Close()
	if resp3.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 even with malformed params", resp3.StatusCode)
	}
	if src.lastOpts != (viz.SnapshotOptions{}) {
		t.Fatalf("lastOpts = %+v, want the zero value for malformed/non-positive params", src.lastOpts)
	}
}
