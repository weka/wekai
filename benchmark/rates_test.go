package benchmark

import (
	"math"
	"strings"
	"testing"
	"time"
)

func sum(v []float64) (s float64) {
	for _, x := range v {
		s += x
	}
	return
}

// Prompt tokens land when prefill ends (the first token); output tokens spread
// across the decode interval, so a long decode is a steady rate rather than a
// spike at its end.
func TestRateAttribution(t *testing.T) {
	r := newRateAccum(1000, 1000)
	q := &timelineReq{Start: 0, E2E: 10500, CW: 500, TTFT: 2000, In: 800, Cached: 200, Out: 800,
		Src: []int{200, 0, 0, 800}} // 200 GPU, 800 computed
	r.addRequest(q)
	// First token at 500 (client wait) + 2000 = 2500 ms: bucket 2 holds the prompt.
	if r.in[2] != 1000 || r.ic[2] != 800 || r.ig[2] != 200 || r.in[0] != 0 || r.in[5] != 0 {
		t.Errorf("prompt impulse: in=%v ic=%v ig=%v", r.in[:6], r.ic[:4], r.ig[:4])
	}
	// 800 output tokens over [2500, 10500] = 100/s; first bucket is half covered.
	if math.Abs(r.out[2]-50) > 0.01 || math.Abs(r.out[3]-100) > 0.01 || math.Abs(r.out[9]-100) > 0.01 || r.out[1] != 0 {
		t.Errorf("output spread: %v", r.out[:12])
	}
	if math.Abs(sum(r.out)-800) > 0.01 {
		t.Errorf("output total = %v, want 800", sum(r.out))
	}
	// Errors carry no tokens.
	r2 := newRateAccum(1000, 1000)
	r2.addRequest(&timelineReq{E2E: 1000, In: 10, Out: 10, Flags: flagError})
	if len(r2.in) != 0 && sum(r2.in) != 0 {
		t.Error("an errored request was attributed tokens")
	}
	// No TTFT reported: everything lands at the end.
	r3 := newRateAccum(1000, 1000)
	r3.addRequest(&timelineReq{Start: 0, E2E: 3000, In: 5, Out: 6})
	if r3.in[3] != 5 || r3.out[3] != 6 {
		t.Errorf("no-ttft request: in=%v out=%v", r3.in, r3.out)
	}
}

// Past the bucket cap the width doubles and pairs merge: bounded, nothing lost.
func TestRateAccumCoarsensWithoutLosingTokens(t *testing.T) {
	r := newRateAccum(1000, 100)
	var want float64
	for i := 0; i < 5000; i++ { // 5000 s of requests against a 100-bucket cap
		q := &timelineReq{Start: float64(i) * 1000, E2E: 900, TTFT: 100, In: 10, Out: 20}
		r.addRequest(q)
		want += 20
	}
	if len(r.in) > 100 || r.bw < 50000 {
		t.Errorf("buckets=%d bw=%v: not bounded", len(r.in), r.bw)
	}
	if math.Abs(sum(r.out)-want) > 1 {
		t.Errorf("output total %v, want %v after coarsening", sum(r.out), want)
	}
	s := r.export(10)
	if len(s.In) > 10 {
		t.Errorf("export gave %d buckets", len(s.In))
	}
	var tot int
	for _, v := range s.Out {
		tot += v
	}
	if tot < int(want)-10 || tot > int(want)+10 {
		t.Errorf("exported output = %d, want ~%v", tot, want)
	}
}

func TestRateWindowSum(t *testing.T) {
	r := newRateAccum(1000, 100)
	r.spread(0, 10000, 1000) // 100 tok/s for 10 s
	_, out := r.windowSum(2000, 7000)
	if math.Abs(out-500) > 0.01 {
		t.Errorf("window out = %v, want 500", out)
	}
	_, out = r.windowSum(8500, 20000)
	if math.Abs(out-150) > 0.01 {
		t.Errorf("window past the data = %v, want 150", out)
	}
}

func TestTTFTOverTime(t *testing.T) {
	var reqs []timelineReq
	for i := 0; i < 100; i++ { // 1 req/s, TTFT 1s for the first 50s then 9s
		ttft := 1000.0
		if i >= 50 {
			ttft = 9000
		}
		reqs = append(reqs, timelineReq{Start: float64(i) * 1000, E2E: 20000, TTFT: ttft})
	}
	s := ttftOverTime(reqs, 120000, 30000, 100)
	if len(s.T) == 0 || len(s.T) != len(s.Mean) || len(s.Mean) != len(s.P95) {
		t.Fatalf("series = %+v", s)
	}
	first, last := s.Mean[0], s.Mean[len(s.Mean)-1]
	if first < 900 || first > 1100 || last < 8000 {
		t.Errorf("mean from %v to %v, want ~1000 rising to ~9000", first, last)
	}
	if s.P95[len(s.P95)-1] != 9000 {
		t.Errorf("p95 = %v", s.P95[len(s.P95)-1])
	}
}

// In-flight requests contribute what the client has seen: tokens streamed so far
// spread from the first token, and the prompt at the first token.
func TestLiveRatesIncludeInflightProgress(t *testing.T) {
	a := newLiveArm("a", 100)
	tok := a.begin(1, 1, "", slotInfo{N: 1}, time.Now())
	a.sent(tok, time.Now())
	tok.prog.setPromptEst(5000)
	if r := a.state().Rates; r.In10 != 0 || r.Out10 != 0 {
		t.Errorf("before the first token: %+v", r)
	}
	tok.prog.markFirst()
	tok.prog.addOut(400)
	time.Sleep(50 * time.Millisecond)
	r := a.state().Rates
	if r.In10 < 400 || r.In10 > 600 || r.Out10 <= 0 { // 5000 tokens over a 10 s window
		t.Errorf("rates with a request running = %+v, want in10~500 and out10>0", r)
	}
	// Completing it replaces the provisional figures with the attributed ones.
	rec := liveRec(a, 1, false)
	rec.InputTokens, rec.OutputTokens = 5000, 400
	a.complete(tok, rec)
	if a.state().Rates.In10 <= 0 {
		t.Error("prompt tokens vanished on completion")
	}
	s := a.series(100)
	if s.Rates == nil || len(s.Rates.In) == 0 {
		t.Error("series has no rates")
	}
}

// The response consumer reports the first token and each streamed delta to the
// request's live progress, which is what in-flight rates are built from.
func TestConsumeOpenAISSEReportsProgress(t *testing.T) {
	sse := "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"b\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"c\"}}]}\n\ndata: [DONE]\n\n"
	p := &liveProgress{}
	m := RequestMetrics{progress: p}
	consumeOpenAISSE(strings.NewReader(sse), time.Now(), &m)
	if p.out.Load() != 3 || p.firstNs.Load() == 0 {
		t.Errorf("progress: out=%d first=%d, want 3 deltas and a first-token time", p.out.Load(), p.firstNs.Load())
	}
	// No tracker attached: nothing to report, nothing breaks.
	var plain RequestMetrics
	consumeOpenAISSE(strings.NewReader(sse), time.Now(), &plain)
}
