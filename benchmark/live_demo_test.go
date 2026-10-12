package benchmark

import (
	"fmt"
	"github.com/weka/wekai/llm"
	"math/rand"
	"os"
	"testing"
	"time"
)

// TestLiveDemoServer serves a synthetic live view for manual and headless-browser
// checks: LIVE_DEMO_PORT=18099 go test ./benchmark -run TestLiveDemoServer. Skipped otherwise.
func TestLiveDemoServer(t *testing.T) {
	port := os.Getenv("LIVE_DEMO_PORT")
	if port == "" {
		t.Skip()
	}
	var p int
	fmt.Sscan(port, &p)
	s, err := startLiveServer("127.0.0.1", p, []string{"demo"}, 100000)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	a := s.arms[0]
	a.setSources(func() (int, int) { return 24, 64 }, func() *liveGate {
		v := 2000 + 7500*float64(time.Now().UnixMilli()%60000)/60000 // sweeps through amber to near the limit
		return &liveGate{Enabled: true, Stat: "mean", ValueMs: v, MeanMs: v, P50Ms: v * 0.8, P95Ms: v * 2.4, P99Ms: v * 3.5, LimitMs: 10000, WindowS: 30, N: 412, Open: v < 10000, AdmitCount: 16, EveryS: 60, NextInS: 23}
	})
	for slot := 1; slot <= 24; slot++ {
		go func(slot int) {
			rng := rand.New(rand.NewSource(int64(slot)))
			sess := slot * 100
			for turn := 1; ; turn++ {
				if turn%6 == 0 {
					sess++
					turn = 1
				}
				sl := slotInfo{N: slot, Hot: slot <= 2}
				pause := time.Duration(rng.Intn(1500)) * time.Millisecond
				thinking := a.thinkUntil(sl, time.Now(), time.Now().Add(pause))
				time.Sleep(pause)
				a.thinkDone(sl, thinking)
				ready := time.Now()
				tok := a.begin(sess, turn, fmt.Sprintf("s%d:main", sess), sl, ready)
				time.Sleep(time.Duration(rng.Intn(1500)) * time.Millisecond) // waits for a free slot
				sent := time.Now()
				a.sent(tok, sent)
				var rw time.Duration
				var retries int
				if rng.Intn(5) == 0 {
					retries, rw = 1+rng.Intn(3), time.Duration(300+rng.Intn(900))*time.Millisecond
					time.Sleep(rw) // backing off after 429s
				}
				dur := time.Duration(300+rng.Intn(2500)) * time.Millisecond
				tok.progressTracker().setPromptEst(8000)
				time.Sleep(dur / 3) // prefill
				tok.progressTracker().markFirst()
				for i := 0; i < 10; i++ {
					time.Sleep(dur / 15)
					tok.progressTracker().addOut(20)
				}
				now := time.Now()
				dur += rw
				ext := rng.Intn(20000)
				rec := requestDataRecord{StartTime: sent, EndTime: now, ResponseMs: float64(dur.Milliseconds()), TTFT: float64(dur.Milliseconds()) / 3,
					Retries429: retries, RetryWaitMs: rw.Milliseconds(), ThinkPauseMs: pause.Milliseconds(), ReadyMs: ready.UnixMilli(), SentMs: sent.UnixMilli(), SeriesNum: sess, Slot: slot, SlotHot: sl.Hot, Turn: turn, InputTokens: 3000, CachedTokens: ext + 5000, OutputTokens: 200}
				if rng.Intn(3) > 0 {
					d := dur.Seconds()
					rec.KV = goldenKV()
					rec.KV.Sources = &llmSources
					rec.KV.Timings.QueueWaitKV, rec.KV.Timings.QueueWaitOther, rec.KV.Timings.KVFetch = 0.05, 0.1, 0.1
					rec.KV.Timings.Prefill, rec.KV.Timings.Decode, rec.KV.Timings.PreemptedTime = d/4, d/2, 0
				}
				a.complete(tok, rec)
				time.Sleep(time.Duration(rng.Intn(400)) * time.Millisecond)
			}
		}(slot)
		time.Sleep(200 * time.Millisecond)
	}
	fmt.Println("LIVE ready", s.Addr())
	time.Sleep(10 * time.Minute)
}

var llmSources = llm.TokenSources{GPU: 8000, External: 6000, Computed: 3000}
