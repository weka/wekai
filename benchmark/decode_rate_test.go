package benchmark

import (
	"context"
	"math"
	"testing"
	"time"
)

func prog(firstAgo time.Duration, now time.Time, out int64) *liveProgress {
	p := &liveProgress{}
	p.firstNs.Store(now.Add(-firstAgo).UnixNano())
	p.out.Store(out)
	return p
}

func TestDecodeRateTimeWeightsCompletedAndRunning(t *testing.T) {
	now := time.Now()
	d := newDecodeRate(30 * time.Second)
	// completed: 10s of decode at 40 tok/s, entirely inside the window
	p1 := prog(15*time.Second, now, 0)
	d.begin(p1)
	d.end(p1, now.Add(-5*time.Second), 400)
	// running: 10s of decode so far at 10 tok/s
	p2 := prog(10*time.Second, now, 100)
	d.begin(p2)
	r, secs := d.rate(now)
	if math.Abs(secs-20) > 0.01 || math.Abs(r-25) > 0.01 {
		t.Fatalf("rate=%.2f secs=%.2f, want 25 tok/s over 20 request-seconds", r, secs)
	}
}

func TestDecodeRateOnlyCountsTheWindow(t *testing.T) {
	now := time.Now()
	d := newDecodeRate(30 * time.Second)
	// 60s decode ending now at 20 tok/s: only the last 30s (600 tok) count
	p := prog(60*time.Second, now, 0)
	d.begin(p)
	d.end(p, now, 1200)
	r, secs := d.rate(now)
	if math.Abs(secs-30) > 0.01 || math.Abs(r-20) > 0.01 {
		t.Fatalf("rate=%.2f secs=%.2f, want 20 over 30", r, secs)
	}
	// a request finished long before the window is pruned
	old := prog(200*time.Second, now, 0)
	d.begin(old)
	d.end(old, now.Add(-100*time.Second), 999999)
	if r2, _ := d.rate(now); math.Abs(r2-20) > 0.01 {
		t.Fatalf("stale request leaked into the window: %.2f", r2)
	}
}

func TestDecodeGateOpenRules(t *testing.T) {
	now := time.Now()
	d := newDecodeRate(30 * time.Second)
	if !d.open(now, 20) {
		t.Fatal("empty window must not hold requests")
	}
	slow := prog(20*time.Second, now, 100) // 5 tok/s for 20 request-seconds
	d.begin(slow)
	if d.open(now, 20) {
		t.Fatal("5 tok/s < 20 must hold")
	}
	if !d.open(now, 0) || !(*decodeRate)(nil).open(now, 20) {
		t.Fatal("gate off must be open")
	}
	tiny := newDecodeRate(30 * time.Second)
	tiny.begin(prog(2*time.Second, now, 1)) // only 2 request-seconds: too little to judge
	if !tiny.open(now, 20) {
		t.Fatal("under decodeGateMinSeconds of decoding must stay open")
	}
	// never-streamed requests are not decode samples
	nf := &liveProgress{}
	d2 := newDecodeRate(30 * time.Second)
	d2.begin(nf)
	d2.end(nf, now, 50)
	if _, s := d2.rate(now); s != 0 {
		t.Fatal("request without a first token counted as decoding")
	}
}

func TestDecodeHoldReleasesWhenRateRecovers(t *testing.T) {
	d := newDecodeRate(30 * time.Second)
	now := time.Now()
	slow := prog(20*time.Second, now, 100)
	d.begin(slow)
	go func() {
		time.Sleep(600 * time.Millisecond)
		d.end(slow, time.Now(), 100)                   // leaves 20s @5 tok/s in the window...
		fast := prog(20*time.Second, time.Now(), 2000) // ...plus 20s @100 tok/s -> 52.5
		d.begin(fast)
	}()
	held := d.hold(context.Background(), 20)
	if held < 500*time.Millisecond || held > 3*time.Second {
		t.Fatalf("held %v, want released shortly after the rate recovered", held)
	}
}

func TestLiveProgressSetOutNeverMovesBack(t *testing.T) {
	p := &liveProgress{}
	p.setOut(10)
	p.setOut(7)
	if p.out.Load() != 10 {
		t.Fatalf("out=%d", p.out.Load())
	}
}

func TestSplitDecodeHoldLeadsTheClientWait(t *testing.T) {
	segs := []tlSeg{{segWaitClient, 3000}, {segDecode, 500}}
	got := splitDecodeHold(segs, 2000)
	if len(got) != 3 || got[0] != (tlSeg{segDecodeHold, 2000}) || got[1] != (tlSeg{segWaitClient, 1000}) || got[2].kind != segDecode {
		t.Fatalf("got %+v", got)
	}
	if got := splitDecodeHold(segs, 9000); len(got) != 2 || got[0] != (tlSeg{segDecodeHold, 3000}) {
		t.Fatalf("hold longer than the wait must be clamped: %+v", got)
	}
	if got := splitDecodeHold(segs, 0); len(got) != 2 {
		t.Fatalf("no hold must be a no-op: %+v", got)
	}
}
