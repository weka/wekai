package benchmark

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

func getJSON(t *testing.T, url string, v any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("GET %s: %d %s", url, resp.StatusCode, b)
	}
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatal(err)
	}
}

func liveRec(arm *liveArm, series int, kv bool) requestDataRecord {
	now := time.Now()
	r := requestDataRecord{StartTime: now.Add(-400 * time.Millisecond), EndTime: now, ResponseMs: 400, TTFT: 100, SeriesNum: series, Turn: 1}
	if kv {
		r.KV = goldenKV()
	}
	return r
}

func TestLiveServerEndpoints(t *testing.T) {
	s, err := startLiveServer("127.0.0.1", 0, []string{"armA", "armB"}, 100)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	base := "http://" + s.Addr()

	var meta struct {
		Kinds []timelineKind `json:"kinds"`
		Arms  []struct {
			Name   string `json:"name"`
			Layout string `json:"layout"`
		} `json:"arms"`
	}
	getJSON(t, base+"/api/arms", &meta)
	if len(meta.Arms) != 2 || meta.Arms[1].Name != "armB" || meta.Arms[0].Layout != "series" || len(meta.Kinds) != len(timelineKinds) {
		t.Fatalf("meta = %+v", meta)
	}

	a := s.arms[0]
	tok1 := a.begin(7, 1)
	tok2 := a.begin(3, 4)
	var d liveData
	getJSON(t, base+"/api/data?arm=0&since=0", &d)
	if len(d.Reqs) != 0 || len(d.Inflight) != 2 {
		t.Fatalf("before completion: reqs=%d inflight=%d", len(d.Reqs), len(d.Inflight))
	}
	for _, f := range d.Inflight {
		if f.Flags&flagInflight == 0 {
			t.Errorf("in-flight request not flagged: %+v", f)
		}
	}
	// Series get stable lanes in order of first sight: 7 -> 0, 3 -> 1.
	lanes := map[int]int{}
	for _, f := range d.Inflight {
		lanes[f.Series] = f.Row
	}
	if lanes[7] != 0 || lanes[3] != 1 {
		t.Errorf("lanes = %v", lanes)
	}

	a.complete(tok1, liveRec(a, 7, true))
	getJSON(t, base+"/api/data?arm=0&since=0", &d)
	if len(d.Reqs) != 1 || len(d.Inflight) != 1 || d.Seq != 1 || d.Reqs[0].Row != 0 || d.Reqs[0].Series != 7 {
		t.Fatalf("after one completion: %+v", d)
	}
	if d.Reqs[0].Flags&flagTimings == 0 {
		t.Error("engine timings not carried to the live request")
	}
	cursor := d.Seq

	a.complete(tok2, liveRec(a, 3, false))
	a.complete(liveToken{}, liveRec(a, 9, false)) // a request with no begin is still recorded
	getJSON(t, base+"/api/data?arm=0&since="+itoa(cursor), &d)
	if len(d.Reqs) != 2 || len(d.Inflight) != 0 || d.Reset {
		t.Fatalf("incremental poll: %+v", d)
	}
	getJSON(t, base+"/api/data?arm=0&since="+itoa(d.Seq), &d)
	if len(d.Reqs) != 0 {
		t.Errorf("poll at the head returned %d requests", len(d.Reqs))
	}

	// The other arm is independent.
	getJSON(t, base+"/api/data?arm=1&since=0", &d)
	if len(d.Reqs) != 0 || d.Seq != 0 {
		t.Errorf("arm 1 = %+v", d)
	}

	for url, code := range map[string]int{
		"/api/data?arm=5": 400, "/api/data?arm=x": 400, "/api/data?arm=0&since=zz": 400, "/nope": 404, "/healthz": 200,
	} {
		resp, err := http.Get(base + url)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != code {
			t.Errorf("GET %s = %d, want %d", url, resp.StatusCode, code)
		}
	}
}

func itoa(u uint64) string { b, _ := json.Marshal(u); return string(b) }

func TestLiveServerPage(t *testing.T) {
	s, err := startLiveServer("127.0.0.1", 0, []string{"a"}, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	resp, err := http.Get("http://" + s.Addr() + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	html := string(b)
	// The live page embeds the same renderer as the report.
	if resp.StatusCode != 200 || !strings.Contains(html, "var WekaiTimeline = (function") || !strings.Contains(html, `"live":true`) {
		t.Errorf("status %d, live page incomplete", resp.StatusCode)
	}
}

// Memory is bounded: past the cap the oldest requests drop, and a client whose
// cursor predates what is left is told to start over.
func TestLiveArmBoundedMemory(t *testing.T) {
	a := newLiveArm("a", 50)
	for i := 0; i < 500; i++ {
		a.complete(liveToken{}, liveRec(a, i%4+1, false))
	}
	d := a.snapshot(0)
	if len(d.Reqs) > 55 || len(d.Reqs) < 50 {
		t.Errorf("retained %d requests, want ~50", len(d.Reqs))
	}
	if d.Seq != 500 || d.Dropped < 400 {
		t.Errorf("seq=%d dropped=%d", d.Seq, d.Dropped)
	}
	if last := d.Reqs[len(d.Reqs)-1]; last.Row < 0 {
		t.Errorf("bad row %+v", last)
	}
	if got := a.snapshot(5); !got.Reset {
		t.Error("a stale cursor must reset the client")
	}
	if got := a.snapshot(d.Seq); got.Reset || len(got.Reqs) != 0 {
		t.Errorf("a current cursor got %+v", got)
	}
	if got := a.snapshot(d.Seq + 99); !got.Reset {
		t.Error("a cursor from the future (server restarted) must reset the client")
	}
}

// Every hook is a no-op without a live view.
func TestLiveArmNilSafe(t *testing.T) {
	var a *liveArm
	tok := a.begin(1, 1)
	a.complete(tok, requestDataRecord{})
	a.abandon(tok)
}

func TestLiveArmAbandon(t *testing.T) {
	a := newLiveArm("a", 10)
	tok := a.begin(1, 1)
	a.abandon(tok)
	if d := a.snapshot(0); len(d.Inflight) != 0 {
		t.Errorf("inflight = %d after abandon", len(d.Inflight))
	}
}

func TestLiveServerBindFailure(t *testing.T) {
	s, err := startLiveServer("127.0.0.1", 0, []string{"a"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	port := s.ln.Addr().(*net.TCPAddr).Port
	if _, err := startLiveServer("127.0.0.1", port, []string{"a"}, 1); err == nil {
		t.Error("binding a taken port must fail the run up front")
	}
}

// Workers complete requests while pollers snapshot; run under -race.
func TestLiveArmConcurrent(t *testing.T) {
	a := newLiveArm("a", 200)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 300; i++ {
				tok := a.begin(w+1, i)
				a.complete(tok, liveRec(a, w+1, i%2 == 0))
			}
		}(w)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		var since uint64
		for i := 0; i < 200; i++ {
			d := a.snapshot(since)
			since = d.Seq
			if _, err := json.Marshal(d); err != nil {
				t.Error(err)
			}
		}
	}()
	wg.Wait()
	if d := a.snapshot(0); d.Seq != 2400 || len(d.Inflight) != 0 {
		t.Errorf("seq=%d inflight=%d", d.Seq, len(d.Inflight))
	}
}
