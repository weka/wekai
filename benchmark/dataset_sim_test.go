package benchmark

import (
	"math"
	"testing"
	"time"
)

func i64(v int64) *int64 { return &v }

// A turn is due at its reference's end plus think, and runs for total_ms
// scaled: response scaling stretches the session, think gaps stay fixed.
func TestBuildDatasetSessionThinkPacing(t *testing.T) {
	s := &RouterReplaySession{
		StartTs: "2026-01-01T00:00:00Z",
		Instances: []RouterReplayInstance{
			{Requests: []RouterReplayRequest{
				{RequestID: 1, Ts: "2026-01-01T00:00:00Z", InputTokens: 100, TotalMs: 1000, ThinkMs: i64(0), ThinkRef: ThinkRefSessionStart},
				{RequestID: 2, Ts: "2026-01-01T00:00:11Z", InputTokens: 300, TotalMs: 1000, ThinkMs: i64(10000), ThinkRef: ThinkRefPrev},
			}},
			{ParentSpawnRequestID: 1, Requests: []RouterReplayRequest{
				{RequestID: 3, Ts: "2026-01-01T00:00:01.5Z", InputTokens: 50, TotalMs: 2000, ThinkMs: i64(500), ThinkRef: ThinkRefSpawn},
			}},
		},
	}
	d := BuildDatasetSession(s, 2)
	want := []DatasetRequest{{0, 2000, 100}, {2500, 6500, 50}, {12000, 14000, 300}}
	if len(d.Requests) != len(want) {
		t.Fatalf("requests = %+v", d.Requests)
	}
	for i, w := range want {
		if d.Requests[i] != w {
			t.Errorf("request %d = %+v, want %+v", i, d.Requests[i], w)
		}
	}
	if d.DurationMs != 14000 || d.BusyMs != 8000 {
		t.Errorf("duration %d busy %d, want 14000 8000", d.DurationMs, d.BusyMs)
	}
	// Tracked prompt: 100 over [2000,6500), 50 over [6500,14000), 300 at the end.
	if want := 100.0*4500 + 50*7500; d.PromptMsArea != want {
		t.Errorf("prompt area %v, want %v", d.PromptMsArea, want)
	}
}

func flatSession(durMs, prompt int64) DatasetSession {
	return DatasetSession{
		Requests:     []DatasetRequest{{StartMs: 0, EndMs: 1000, Prompt: prompt}},
		DurationMs:   durMs,
		PromptMsArea: float64(prompt) * float64(durMs-1000),
		BusyMs:       1000,
	}
}

// With fixed slots the long-run dataset is the renewal average, and a corpus
// of long sessions takes about one session life to get there.
func TestSimulateDatasetFixedSlotsConvergesToRenewal(t *testing.T) {
	sessions := []DatasetSession{flatSession(int64(time.Hour/time.Millisecond), 1000), flatSession(int64(9*time.Hour/time.Millisecond), 5000)}
	samples, err := SimulateDataset(sessions, DatasetSimConfig{Series: 200, Strategy: ReplayStrategyRandom, Horizon: 20 * 24 * time.Hour, SampleEvery: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	st := StationaryDataset(sessions)
	var tail float64
	last := samples[len(samples)/2:]
	for _, s := range last {
		tail += float64(s.Dataset)
	}
	tail /= float64(len(last))
	want := 200 * st.PerSlotDataset
	if math.Abs(tail-want)/want > 0.05 {
		t.Errorf("long-run dataset %.0f, renewal %.0f", tail, want)
	}
	if samples[0].Slots != 200 {
		t.Errorf("slots %d, want 200", samples[0].Slots)
	}
}

// The governor grows slots until the in-flight target is met: with every
// session busy 1s of each 100s, 10 in flight takes about 1000 slots.
func TestSimulateDatasetGovernorMeetsTarget(t *testing.T) {
	s := DatasetSession{DurationMs: 100000}
	for k := int64(0); k < 100; k++ {
		s.Requests = append(s.Requests, DatasetRequest{StartMs: k * 1000, EndMs: k*1000 + 10, Prompt: 10})
	}
	s.BusyMs = 1000
	samples, err := SimulateDataset([]DatasetSession{s}, DatasetSimConfig{TargetInflight: 0.5, StartSeries: 1, AdmitEvery: time.Minute, AdmitCount: 4, Horizon: 6 * time.Hour, SampleEvery: 10 * time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	end := samples[len(samples)-1]
	if end.Slots < 40 || end.Slots > 60 {
		t.Errorf("slots %d, want ~50 for 0.5 in flight at 0.01 per slot", end.Slots)
	}
}

func TestSimulateDatasetRejectsAmbiguousMode(t *testing.T) {
	if _, err := SimulateDataset([]DatasetSession{flatSession(2000, 1)}, DatasetSimConfig{Series: 1, TargetInflight: 1, Horizon: time.Hour}); err == nil {
		t.Error("want an error with both Series and TargetInflight")
	}
}

func TestSettleTime(t *testing.T) {
	var samples []DatasetSample
	for i := range 10 {
		v := int64(100)
		if i < 4 {
			v = int64(10 * (i + 1))
		}
		samples = append(samples, DatasetSample{At: time.Duration(i) * time.Hour, Dataset: v})
	}
	got := SettleTime(samples, func(s DatasetSample) float64 { return float64(s.Dataset) }, 100, 0.05, 1)
	if got != 4*time.Hour {
		t.Errorf("settle %v, want 4h", got)
	}
}
