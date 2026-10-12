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
				ready := time.Now()
				tok := a.begin(sess, turn, fmt.Sprintf("s%d:main", sess), sl, ready)
				time.Sleep(time.Duration(rng.Intn(1500)) * time.Millisecond) // waits for a free slot
				sent := time.Now()
				a.sent(tok, sent)
				dur := time.Duration(300+rng.Intn(2500)) * time.Millisecond
				time.Sleep(dur)
				now := time.Now()
				ext := rng.Intn(20000)
				rec := requestDataRecord{StartTime: now.Add(-dur), EndTime: now, ResponseMs: float64(dur.Milliseconds()), TTFT: float64(dur.Milliseconds()) / 3,
					ReadyMs: ready.UnixMilli(), SentMs: sent.UnixMilli(), SeriesNum: sess, Slot: slot, SlotHot: sl.Hot, Turn: turn, InputTokens: 3000, CachedTokens: ext + 5000, OutputTokens: 200}
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
