package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

// A run asks for more concurrent sessions than the corpus holds whenever the
// fleet under test is bigger than the capture: 8,000 slots against a 5,441-
// session file is the ordinary case for a many-node router, not a mistake.
//
// The two outcomes are opposite and look alike from the outside. With the
// corpus cycling, every slot holds a live session and the run offers the load
// it reports. Without it, the surplus slots pull once, find the stream drained
// and exit — the run still prints its slot count while a fraction of it is
// actually running.
//
// These tests drive the real session loop against a mock fleet, because the
// property is about what reaches the wire, not about what the producer intends.

// oversubscribeCorpus is n sessions, each a single instance with one user turn,
// so a dispatched session is exactly one request on the wire and in-flight
// requests count live sessions.
func oversubscribeCorpus(t *testing.T, n int) string {
	t.Helper()
	sessions := make([]RouterReplaySession, n)
	for i := range sessions {
		sessions[i] = RouterReplaySession{
			SessionID: fmt.Sprintf("s%d", i),
			StartTs:   "2026-05-12T08:50:06Z",
			Instances: []RouterReplayInstance{{
				InstanceID: "i0",
				Role:       "main",
				Requests: []RouterReplayRequest{{
					RequestID:    uint64(i*10 + 1),
					OutputTokens: 8,
					Messages:     []RouterReplayMessage{userText(fmt.Sprintf("h%d", i), 200)},
				}},
			}},
		}
	}
	return writeReplayV3File(t, sessions)
}

// blockingFleet is a mock fleet that holds every request open until released,
// so the number of requests it is holding IS the number of sessions the run has
// managed to put in flight at once.
type blockingFleet struct {
	srv     *httptest.Server
	release chan struct{}

	mu       sync.Mutex
	inflight int
	stamps   []string // content stamp of each arriving request, arrival order
}

// The stamp rides in a system block as "<ignore>RUN_GUID: ...</ignore>", and
// encoding/json escapes the angle brackets, so match up to the escape.
var stampRe = regexp.MustCompile(`RUN_GUID: ([^\\<"]+)`)

// oversubscribeRunID is the run's own stamp: lap one carries it unchanged, and
// later laps carry it with a "-pN" suffix.
const oversubscribeRunID = "3fa1c2d4-0000-4000-8000-000000000001"

func newBlockingFleet() *blockingFleet {
	bf := &blockingFleet{release: make(chan struct{})}
	bf.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		stamp := ""
		if m := stampRe.FindStringSubmatch(string(body)); m != nil {
			stamp = m[1]
		}
		bf.mu.Lock()
		bf.stamps = append(bf.stamps, stamp)
		bf.inflight++
		bf.mu.Unlock()

		<-bf.release

		bf.mu.Lock()
		bf.inflight--
		bf.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"c","object":"chat.completion","model":"test-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
	}))
	return bf
}

func (bf *blockingFleet) close() {
	select {
	case <-bf.release:
	default:
		close(bf.release)
	}
	bf.srv.Close()
}

// settle waits until `want` requests have arrived, or until arrivals have
// stopped growing for a beat — so the "surplus slots never fill" case costs a
// beat rather than the whole deadline.
func (bf *blockingFleet) settle(want int, deadline time.Duration) {
	end := time.Now().Add(deadline)
	last, quiet := -1, 0
	for time.Now().Before(end) {
		bf.mu.Lock()
		got := len(bf.stamps)
		bf.mu.Unlock()
		if got >= want {
			return
		}
		if got == last {
			if quiet++; quiet > 25 { // ~250ms with nothing new
				return
			}
		} else {
			last, quiet = got, 0
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (bf *blockingFleet) snapshot() (inflight int, stamps []string) {
	bf.mu.Lock()
	defer bf.mu.Unlock()
	return bf.inflight, append([]string(nil), bf.stamps...)
}

// runOversubscribed spawns `slots` session workers over a `sessions`-session
// corpus and reports what the fleet saw once every slot has had its chance.
func runOversubscribed(t *testing.T, sessions, slots int, reuse bool) (*blockingFleet, int) {
	t.Helper()
	path := oversubscribeCorpus(t, sessions)
	bf := newBlockingFleet()
	defer bf.close()

	stream, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 8, Reuse: reuse})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	cfg := AutoBenchmarkConfig{
		Model:          fmt.Sprintf("dynamic/%s,type=openai,model=test-model", bf.srv.URL),
		RunID:          oversubscribeRunID,
		RequestTimeout: 30 * time.Second,
	}
	st := newTestAutoState(256)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	for range slots {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runRouterReplaySeriesLoop(ctx, cfg, st, nil, stream, endpointPicker{},
				strings.Repeat("reuse-docs ", 100), func(*autoState) {}, st.gate)
		}()
	}

	bf.settle(slots, 15*time.Second)
	// Every request the fleet is holding is one slot with a live session in it.
	live, _ := bf.snapshot()

	cancel()
	bf.close()
	wg.Wait()
	return bf, live
}

// TestReuseFillsSlotsThatOutnumberTheCorpus is the case a big fleet forces: more
// session slots than the file has sessions. With the corpus cycling, every slot
// must end up holding a live session — the surplus is served by going round
// again, not by leaving the slots empty.
func TestReuseFillsSlotsThatOutnumberTheCorpus(t *testing.T) {
	const sessions, slots = 3, 8
	bf, live := runOversubscribed(t, sessions, slots, true)
	if live != slots {
		_, stamps := bf.snapshot()
		t.Fatalf("%d of %d slots held a live session over a %d-session corpus (%d requests arrived); "+
			"cycling the corpus is what fills the surplus, and a run that leaves them empty offers "+
			"less load than the slot count it reports", live, slots, sessions, len(stamps))
	}
}

// TestWithoutReuseSlotsBeyondTheCorpusNeverFill is the same run with cycling
// off. It is not a bug — draining is what --replay-series and the underfill
// abort are built on — but it is the failure mode of asking for 8,000 slots
// against a 5,441-session file and forgetting the flag, so it is pinned here.
func TestWithoutReuseSlotsBeyondTheCorpusNeverFill(t *testing.T) {
	const sessions, slots = 3, 8
	_, live := runOversubscribed(t, sessions, slots, false)
	if live != sessions {
		t.Errorf("with reuse off, %d slots were live over a %d-session corpus; the stream drains, so "+
			"exactly the corpus can be in flight and the surplus workers exit", live, sessions)
	}
}

// TestCyclingKeepsEachPassInItsOwnKeyspace: the surplus slots are the SAME
// sessions again. If they carried pass one's stamp they would replay into pass
// one's cache entries — every one of them a guaranteed hit on content the fleet
// already holds — and the run's hit rate would be a measurement of its own
// duplication.
func TestCyclingKeepsEachPassInItsOwnKeyspace(t *testing.T) {
	const sessions, slots = 3, 8
	bf, _ := runOversubscribed(t, sessions, slots, true)
	_, stamps := bf.snapshot()
	if len(stamps) < slots {
		t.Fatalf("only %d requests reached the fleet, want %d", len(stamps), slots)
	}
	counts := map[string]int{}
	for _, s := range stamps {
		counts[s]++
	}
	// 8 slots over 3 sessions: passes 1..3 (the third partly), so three stamps.
	if len(counts) != 3 {
		t.Errorf("%d distinct stamps across %d requests (%v); each lap through the corpus needs its "+
			"own, or the second lap hits the first lap's cache entries", len(counts), len(stamps), counts)
	}
	if counts[oversubscribeRunID] != sessions {
		t.Errorf("pass one stamped %d requests, want the whole corpus (%d): the stamp is per PASS, "+
			"and every session on a pass must share it or the prefix sharing the capture recorded is "+
			"broken up", counts[oversubscribeRunID], sessions)
	}
	for _, want := range []string{oversubscribeRunID + "-p2", oversubscribeRunID + "-p3"} {
		if counts[want] == 0 {
			t.Errorf("no request carried %q; laps beyond the first are not being stamped apart (%v)",
				want, counts)
		}
	}
}

// TestEveryLapGetsFreshSeriesNumbers: series number comes from the pull counter,
// so the same session on lap two is a different series. Without that, two live
// sessions share an identity and the per-series bookkeeping (dataset tracker,
// marker holders) collides between laps.
func TestEveryLapGetsFreshSeriesNumbers(t *testing.T) {
	path := oversubscribeCorpus(t, 3)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1, Reuse: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	seen := map[int]bool{}
	for i := range 8 {
		sess, idx, ok := st.Pull(context.Background())
		if !ok {
			t.Fatalf("stream drained after %d pulls", i)
		}
		if seen[idx] {
			t.Fatalf("pull %d reused series index %d (session %s, pass %d)", i, idx, sess.SessionID, sess.pass)
		}
		seen[idx] = true
	}
}

// TestCappedReuseStopsAndReportsDrained: --replay-series is a cap on how many
// sessions the run dispatches, and under reuse it is the only thing that ever
// stops the producer. Once it has, the stream has to report itself drained —
// the drain watcher and the evaluator both decide the run is finished by asking
// Remaining(), and a stream that always answers "more" leaves every worker idle
// while the run ticks on to its timeout.
func TestCappedReuseStopsAndReportsDrained(t *testing.T) {
	path := oversubscribeCorpus(t, 3)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1, SessionLimit: 5, Reuse: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	if got := st.Total(); got != 5 {
		t.Errorf("Total() = %d, want the cap 5: under reuse a cap above the corpus size is what the "+
			"run will actually dispatch, and dividing progress by the corpus reports past 100%%", got)
	}
	for i := range 5 {
		if _, _, ok := st.Pull(context.Background()); !ok {
			t.Fatalf("drained after %d pulls, want the full cap of 5", i)
		}
	}
	if _, _, ok := st.Pull(context.Background()); ok {
		t.Error("stream kept going past --replay-series; the cap must bound the run across laps")
	}
	if got := st.Remaining(); got != 0 {
		t.Errorf("Remaining() = %d after the capped producer stopped, want 0: the run finishes when "+
			"the stream says it is empty, and under reuse nothing else ever says so", got)
	}
}

// TestUncappedReuseNeverReportsDrained is the other half of the same contract:
// while the producer is still cycling there is always another session, so the
// underfill abort has nothing to fire on.
func TestUncappedReuseNeverReportsDrained(t *testing.T) {
	path := oversubscribeCorpus(t, 2)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1, Reuse: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	for range 6 {
		if _, _, ok := st.Pull(context.Background()); !ok {
			t.Fatal("uncapped reuse drained; it must not")
		}
	}
	if got := st.Remaining(); got == 0 {
		t.Error("Remaining() = 0 while the corpus is still cycling; the underfill abort would fire on " +
			"a stream that is not empty and kill a healthy run")
	}
}

// TestIndexFilteredReuseKeepsCycling: --replay-series-range picks a slice of the
// corpus, and a slice is exactly what a run cycles when it wants a smaller
// working set than the file. The filter must bound WHICH sessions repeat, not
// how many laps there are.
func TestIndexFilteredReuseKeepsCycling(t *testing.T) {
	path := oversubscribeCorpus(t, 4)
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1,
		AllowedIndices: map[int]bool{1: true, 3: true}, Reuse: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	var gotIdx []int
	var gotPass []int
	for i := range 6 {
		sess, _, ok := st.Pull(context.Background())
		if !ok {
			t.Fatalf("index-filtered reuse drained after %d pulls; the filter selects the working set, "+
				"it does not end the run", i)
		}
		gotIdx = append(gotIdx, sess.fileIdx)
		gotPass = append(gotPass, sess.pass)
	}
	wantIdx := []int{1, 3, 1, 3, 1, 3}
	wantPass := []int{0, 0, 1, 1, 2, 2}
	for i := range wantIdx {
		if gotIdx[i] != wantIdx[i] || gotPass[i] != wantPass[i] {
			t.Fatalf("pulls = idx %v pass %v, want idx %v pass %v", gotIdx, gotPass, wantIdx, wantPass)
		}
	}
}

// TestReuseOnAnEmptyCorpusStops: a pass that produced nothing will produce
// nothing next time either. Rewinding on it is an unbounded loop over an empty
// file — a pinned core and a run that never starts and never ends.
func TestReuseOnAnEmptyCorpusStops(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.jsonl")
	hdr, _ := json.Marshal(RouterReplayHeader{
		Schema: "replay-v3", Name: "empty", Source: "test",
		Summary: RouterReplaySummary{Sessions: 0},
	})
	if err := os.WriteFile(path, append(hdr, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	st, err := openRouterReplayStream(path, routerReplayStreamOpts{ChanCap: 1, Reuse: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	select {
	case <-st.done:
	case <-time.After(2 * time.Second):
		t.Fatal("the producer is still cycling an empty corpus after 2s; a lap that emitted nothing " +
			"must stop the producer, not spin the file open and shut forever")
	}
	if _, _, ok := st.Pull(context.Background()); ok {
		t.Error("an empty corpus handed out a session")
	}
}
