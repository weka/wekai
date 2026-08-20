package benchmark

import (
	"context"
	"fmt"
	"testing"
)

// tsReq is a request at a given capture time.
func tsReq(id uint64, ts string) RouterReplayRequest {
	return RouterReplayRequest{
		RequestID: id, Ts: ts, OutputTokens: 8,
		Messages: []RouterReplayMessage{userText(fmt.Sprintf("h%d", id), 100)},
	}
}

func requestIDs(sess RouterReplaySession) []uint64 {
	var out []uint64
	for _, inst := range sess.Instances {
		for _, r := range inst.Requests {
			out = append(out, r.RequestID)
		}
	}
	return out
}

func TestTruncateKeepsTheEarliestRequestsAcrossTheWholeSession(t *testing.T) {
	// Two instances interleaved in time. A per-instance prefix would keep 1,2
	// and 10,11; capture order keeps 1,10,2 — what the session actually did
	// first.
	sess := RouterReplaySession{
		SessionID: "s",
		Instances: []RouterReplayInstance{
			{InstanceID: "a", Requests: []RouterReplayRequest{
				tsReq(1, "2026-05-12T08:00:00Z"),
				tsReq(2, "2026-05-12T08:00:02Z"),
				tsReq(3, "2026-05-12T08:00:04Z"),
			}},
			{InstanceID: "b", Requests: []RouterReplayRequest{
				tsReq(10, "2026-05-12T08:00:01Z"),
				tsReq(11, "2026-05-12T08:00:03Z"),
			}},
		},
	}
	dropped := truncateSessionRequests(&sess, 3)
	if dropped != 2 {
		t.Errorf("dropped %d, want 2", dropped)
	}
	got := requestIDs(sess)
	want := map[uint64]bool{1: true, 2: true, 10: true}
	if len(got) != 3 {
		t.Fatalf("kept %v, want 3 requests", got)
	}
	for _, id := range got {
		if !want[id] {
			t.Errorf("kept request %d; the cap keeps the session's earliest requests across every "+
				"instance, not the first N of each", id)
		}
	}
}

// The property the whole design rests on: a cut can remove a child entirely,
// but can never leave one running whose parent's spawn was cut away. A parent's
// spawn-bearing request precedes its child's first request in time, so anything
// that removes the spawn removes everything the child would have run.
func TestTruncateNeverOrphansALiveChild(t *testing.T) {
	sess := RouterReplaySession{
		SessionID: "s",
		Instances: []RouterReplayInstance{
			{InstanceID: "parent", Requests: []RouterReplayRequest{
				tsReq(1, "2026-05-12T08:00:00Z"),
				tsReq(2, "2026-05-12T08:00:01Z"), // the spawn
			}},
			{InstanceID: "child", ParentSpawnRequestID: 2, Requests: []RouterReplayRequest{
				tsReq(3, "2026-05-12T08:00:02Z"),
				tsReq(4, "2026-05-12T08:00:03Z"),
			}},
		},
	}
	// Cut before the spawn: the child must be gone entirely, not left to fire
	// as a root because its parent request vanished from the wait map.
	cut := sess
	cut.Instances = append([]RouterReplayInstance(nil), sess.Instances...)
	truncateSessionRequests(&cut, 1)
	for _, inst := range cut.Instances {
		if inst.InstanceID == "child" {
			t.Errorf("the child survived a cut that removed its parent's spawn; it would fire "+
				"immediately as a root and replay a turn the capture never reached. instances=%v",
				cut.Instances)
		}
	}
	if len(cut.Instances) != 1 || len(cut.Instances[0].Requests) != 1 {
		t.Errorf("expected one instance holding one request, got %d instances", len(cut.Instances))
	}
}

func TestTruncateDropsInstancesLeftWithNothing(t *testing.T) {
	sess := RouterReplaySession{
		SessionID: "s",
		Instances: []RouterReplayInstance{
			{InstanceID: "a", Requests: []RouterReplayRequest{tsReq(1, "2026-05-12T08:00:00Z")}},
			{InstanceID: "b", Requests: []RouterReplayRequest{tsReq(2, "2026-05-12T08:00:01Z")}},
		},
	}
	truncateSessionRequests(&sess, 1)
	if len(sess.Instances) != 1 || sess.Instances[0].InstanceID != "a" {
		t.Errorf("instances = %d; one holding no requests still takes a goroutine and a done-channel "+
			"its children would wait on", len(sess.Instances))
	}
}

func TestTruncateLeavesASessionUnderTheCapAlone(t *testing.T) {
	sess := RouterReplaySession{
		SessionID: "s",
		Instances: []RouterReplayInstance{{InstanceID: "a", Requests: []RouterReplayRequest{
			tsReq(1, "2026-05-12T08:00:00Z"), tsReq(2, "2026-05-12T08:00:01Z"),
		}}},
	}
	before := requestIDs(sess)
	if dropped := truncateSessionRequests(&sess, 5); dropped != 0 {
		t.Errorf("dropped %d from a session under the cap", dropped)
	}
	if got := requestIDs(sess); len(got) != len(before) {
		t.Errorf("requests = %v, want %v untouched", got, before)
	}
	// The median session is far below any cap worth setting, so the untouched
	// path is the common one; a cap of 0 must be off entirely.
	if dropped := truncateSessionRequests(&sess, 0); dropped != 0 || len(requestIDs(sess)) != 2 {
		t.Error("a cap of 0 truncated something; 0 means off")
	}
}

// Same corpus and same cap must truncate to the same set every time, or a run
// is not comparable with itself — let alone with the arm it is being compared
// against.
func TestTruncateIsDeterministic(t *testing.T) {
	build := func() RouterReplaySession {
		return RouterReplaySession{
			SessionID: "s",
			Instances: []RouterReplayInstance{
				// Identical timestamps across instances: the tie-break has to be
				// the file's own order, not map or scheduling order.
				{InstanceID: "a", Requests: []RouterReplayRequest{
					tsReq(1, "2026-05-12T08:00:00Z"), tsReq(2, "2026-05-12T08:00:00Z")}},
				{InstanceID: "b", Requests: []RouterReplayRequest{
					tsReq(10, "2026-05-12T08:00:00Z"), tsReq(11, "2026-05-12T08:00:00Z")}},
				// No timestamp at all: must sort last rather than being promoted
				// ahead of real ones.
				{InstanceID: "c", Requests: []RouterReplayRequest{tsReq(20, "")}},
			},
		}
	}
	first := build()
	truncateSessionRequests(&first, 3)
	want := requestIDs(first)
	if len(want) != 3 || want[len(want)-1] == 20 {
		t.Fatalf("kept %v; an untimed request must not displace a timed one", want)
	}
	for i := range 20 {
		s := build()
		truncateSessionRequests(&s, 3)
		got := requestIDs(s)
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Fatalf("run %d kept %v, first run kept %v", i, got, want)
		}
	}
}

// The cap is applied in the producer, so a worker never holds the requests it
// drops — which is the whole reason it is there rather than at dispatch.
func TestStreamAppliesTheCapBeforeHandingSessionsOut(t *testing.T) {
	sessions := []RouterReplaySession{{
		SessionID: "big", StartTs: "2026-05-12T08:00:00Z",
		Instances: []RouterReplayInstance{{InstanceID: "i0", Role: "main", Requests: []RouterReplayRequest{
			tsReq(1, "2026-05-12T08:00:00Z"), tsReq(2, "2026-05-12T08:00:01Z"),
			tsReq(3, "2026-05-12T08:00:02Z"), tsReq(4, "2026-05-12T08:00:03Z"),
		}}},
	}}
	st, err := openRouterReplayStream(writeReplayV3File(t, sessions),
		routerReplayStreamOpts{ChanCap: 1, MaxRequestsPerSession: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	sess, _, ok := st.Pull(context.Background())
	if !ok {
		t.Fatal("stream drained")
	}
	if got := requestIDs(sess); len(got) != 2 {
		t.Errorf("worker received %v, want 2 requests: the cap has to be applied before the session "+
			"is handed out, or the dropped requests stay resident for as long as it runs", got)
	}
	if s, r := st.Truncated(); s != 1 || r != 2 {
		t.Errorf("Truncated() = (%d, %d), want (1, 2): a capped run is not comparable with an "+
			"uncapped one, so what it dropped has to be reported rather than inferred", s, r)
	}
}
