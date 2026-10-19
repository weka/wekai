package benchmark

import (
	"reflect"
	"testing"
	"time"
)

func retentionReq(at time.Duration, msgs ...string) RouterReplayRequest {
	t0 := time.Date(2026, 5, 12, 8, 0, 0, 0, time.UTC)
	r := RouterReplayRequest{Ts: t0.Add(at).Format(time.RFC3339Nano)}
	for _, h := range msgs {
		r.Messages = append(r.Messages, RouterReplayMessage{Role: "user", Hash: h, Tokens: 100})
	}
	return r
}

func TestSimulateRetention(t *testing.T) {
	reqs := []RouterReplayRequest{
		retentionReq(0, "a", "b"),
		retentionReq(10*time.Minute, "a", "b", "c"),      // a,b idle 10m
		retentionReq(20*time.Minute, "a", "b", "c", "d"), // a,b,c idle 10m: refreshed at 10m
		retentionReq(3*time.Hour, "a", "x"),              // a idle 2h40m
	}
	windows := []time.Duration{5 * time.Minute, 15 * time.Minute, time.Hour, 24 * time.Hour}
	rep := SimulateRetention(reqs, windows)

	if rep.Total != 1100 {
		t.Fatalf("total = %d, want 1100", rep.Total)
	}
	// 5m: nothing survives. 15m and 1h: a,b then a,b,c (refresh keeps a,b
	// alive although they were written 20m before the third request). 1d and
	// infinite: also "a" of the last request.
	want := []int64{0, 500, 500, 600}
	if !reflect.DeepEqual(rep.Cached, want) {
		t.Fatalf("cached = %v, want %v", rep.Cached, want)
	}
	if rep.CachedInfinite != 600 {
		t.Fatalf("infinite = %d, want 600", rep.CachedInfinite)
	}
	if sim := SimulateReplayCache(reqs); int64(sim.SimCachedTokens) != rep.CachedInfinite {
		t.Fatalf("infinite %d disagrees with SimulateReplayCache %d", rep.CachedInfinite, sim.SimCachedTokens)
	}
}

// A node behind an expired one is not reusable even if itself recently used.
func TestSimulateRetentionExpiredPrefixCutsTail(t *testing.T) {
	reqs := []RouterReplayRequest{
		retentionReq(0, "a", "b"),
		retentionReq(50*time.Minute, "z", "b"), // touches no node of a's path
		retentionReq(time.Hour, "a", "b"),      // a idle 1h, b idle 1h
	}
	rep := SimulateRetention(reqs, []time.Duration{30 * time.Minute, 2 * time.Hour})
	if want := []int64{0, 200}; !reflect.DeepEqual(rep.Cached, want) {
		t.Fatalf("cached = %v, want %v", rep.Cached, want)
	}
}

func TestParseRetentionWindows(t *testing.T) {
	got, err := ParseRetentionWindows("2d, 5m,1h,5m,1.5d")
	if err != nil {
		t.Fatal(err)
	}
	want := []time.Duration{5 * time.Minute, time.Hour, 36 * time.Hour, 48 * time.Hour}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if got, _ := ParseRetentionWindows(""); len(got) != 0 {
		t.Fatalf("empty: got %v", got)
	}
	if _, err := ParseRetentionWindows("0m"); err == nil {
		t.Fatal("0m accepted")
	}
	for d, s := range map[time.Duration]string{
		5 * time.Minute: "5m", time.Hour: "1h", 90 * time.Minute: "1h30m",
		48 * time.Hour: "2d", 36 * time.Hour: "36h", 30 * time.Second: "30s",
	} {
		if f := FormatRetentionWindow(d); f != s {
			t.Errorf("format %v = %q, want %q", d, f, s)
		}
	}
}
