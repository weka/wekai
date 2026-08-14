package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// Sizing the TTFT-governed admission loop before it is run on hardware.
//
// The load driver has no concurrency limit of its own: it adds one session per
// second while TTFT stays under the gate and pauses above it, so the fleet's
// own latency is the only throttle and the loop settles at the largest session
// count the fleet can carry. What has to be known in advance is the CLIENT-side
// concurrency that corresponds to — a series pool set below it silently becomes
// the bottleneck, and the run then measures the client instead of the fleet.

func loadSimWorkload(t *testing.T) SimWorkload {
	t.Helper()
	b, err := os.ReadFile("testdata/admission_workload.json")
	if err != nil {
		t.Skipf("workload sample absent: %v", err)
	}
	var wl SimWorkload
	if err := json.Unmarshal(b, &wl); err != nil {
		t.Fatalf("workload: %v", err)
	}
	return wl
}

// fleet builds an 8-backend fleet at one profile.
func fleet(name string, prefill, decode float64, seqs int, hit float64) []ServerProfile {
	out := make([]ServerProfile, 8)
	for i := range out {
		out[i] = ServerProfile{
			Name:                name,
			PrefillTokensPerSec: prefill,
			DecodeTokensPerSec:  decode,
			MaxSeqs:             seqs,
			CacheHitRate:        hit,
		}
	}
	return out
}

func report(t *testing.T, r SimResult) {
	t.Helper()
	t.Logf("%-24s sess max=%-6d final=%-6d | conc max=%-5d mean=%-7.0f | %6.1f req/s | "+
		"TTFT p50=%5.2fs p99=%6.2fs | in/req=%7.0f | prefill=%3.0f%% slots=%3.0f%% | %s",
		r.Profile, r.MaxSessions, r.FinalSessions, r.MaxConcurrency, r.MeanConcurrency,
		r.ThroughputReqPerSec, r.P50TTFT, r.P99TTFT, r.MeanInputCompleted,
		r.PrefillUtil*100, r.SeqSlotUtil*100, r.Bound)
}

// TestAdmissionGateFindsDifferentCeilingsForDifferentFleets is the check that
// the loop works at all: a faster fleet must settle at more sessions than a
// slower one. A governor that returns the same answer whatever it is governing
// is measuring the workload, not the fleet.
func TestAdmissionGateFindsDifferentCeilingsForDifferentFleets(t *testing.T) {
	wl := loadSimWorkload(t)
	base := SimConfig{AdmitEvery: 1, TTFTLimitSec: 5, HorizonSec: 14400, TTFTWindow: 32}

	slow := base
	slow.Servers = fleet("hbm-only (hit 0.50)", 200_000, 50, 512, 0.50)
	fast := base
	fast.Servers = fleet("shared tier (hit 0.90)", 200_000, 50, 512, 0.90)

	rs := RunAdmissionSim(slow, wl)
	rf := RunAdmissionSim(fast, wl)
	report(t, rs)
	report(t, rf)

	if rf.MaxSessions <= rs.MaxSessions {
		t.Errorf("the faster fleet settled at %d sessions and the slower at %d; the gate is not "+
			"discovering the fleet's ceiling", rf.MaxSessions, rs.MaxSessions)
	}
	if rf.ThroughputReqPerSec <= rs.ThroughputReqPerSec {
		t.Errorf("throughput did not rise with the cache tier: %.2f vs %.2f req/s",
			rf.ThroughputReqPerSec, rs.ThroughputReqPerSec)
	}
}

// TestSeriesPoolOf512 is the sizing question: with the client capped at 512
// concurrent requests, is the cap ever what stops the run?
func TestSeriesPoolOf512(t *testing.T) {
	wl := loadSimWorkload(t)
	for _, hit := range []float64{0.50, 0.90, 0.96, 0.99} {
		cfg := SimConfig{
			Servers:      fleet(fmt.Sprintf("hit %.2f", hit), 200_000, 50, 512, hit),
			AdmitEvery:   1,
			TTFTLimitSec: 5,
			HorizonSec:   14400,
			TTFTWindow:   32,
			MaxSeries:    512,
		}
		r := RunAdmissionSim(cfg, wl)
		report(t, r)
	}
}

// TestUncappedConcurrency removes the client cap so the number the pool would
// have to accommodate is visible rather than clipped.
func TestUncappedConcurrency(t *testing.T) {
	wl := loadSimWorkload(t)
	for _, hit := range []float64{0.50, 0.90, 0.96, 0.99} {
		cfg := SimConfig{
			Servers:      fleet(fmt.Sprintf("hit %.2f uncapped", hit), 200_000, 50, 512, hit),
			AdmitEvery:   1,
			TTFTLimitSec: 5,
			HorizonSec:   14400,
			TTFTWindow:   32,
		}
		report(t, RunAdmissionSim(cfg, wl))
	}
}

// TestPrefillBindsBeforeSequenceSlots. Both ceilings exist and only one is
// reached; which one decides what the run is actually measuring. At this
// workload's mean of ~159k input tokens against ~558 output, prefill bandwidth
// runs out long before the 512 sequence slots do, at every hit rate short of
// near-perfect — so a run that shows HBM stalling and a shared tier continuing
// is showing a prefill result, not a batching one.
func TestPrefillBindsBeforeSequenceSlots(t *testing.T) {
	wl := loadSimWorkload(t)
	cfg := SimConfig{
		Servers:      fleet("bound check", 200_000, 50, 512, 0.90),
		AdmitEvery:   1,
		TTFTLimitSec: 5,
		HorizonSec:   14400,
		TTFTWindow:   32,
	}
	r := RunAdmissionSim(cfg, wl)
	report(t, r)
	if r.PrefillUtil <= r.SeqSlotUtil {
		t.Errorf("prefill %.0f%% vs slots %.0f%%: if slots bind first the cache tier is not what "+
			"the experiment is varying", r.PrefillUtil*100, r.SeqSlotUtil*100)
	}
}

// TestGateTracksRawServerSpeed varies the mock servers' prefill bandwidth
// rather than their cache tier, because the gate must respond to the fleet
// being faster whatever made it faster. A governor that only reacts to one
// parameter is fitting that parameter, not finding a ceiling.
func TestGateTracksRawServerSpeed(t *testing.T) {
	wl := loadSimWorkload(t)
	var prev SimResult
	for _, prefill := range []float64{100_000, 200_000, 400_000} {
		cfg := SimConfig{
			Servers:      fleet(fmt.Sprintf("prefill %.0fk/s", prefill/1000), prefill, 50, 512, 0.50),
			AdmitEvery:   1,
			TTFTLimitSec: 5,
			HorizonSec:   14400,
			TTFTWindow:   32,
		}
		r := RunAdmissionSim(cfg, wl)
		report(t, r)
		if prev.MaxSessions > 0 && r.MaxSessions <= prev.MaxSessions {
			t.Errorf("doubling prefill bandwidth did not raise the sustained session count: "+
				"%d then %d", prev.MaxSessions, r.MaxSessions)
		}
		prev = r
	}
}

// TestSeriesPoolOf512DistortsTheComparison is the sizing result that matters
// for the planned hardware run.
//
// The client pool is not a neutral safety limit. It binds only on the fast arm
// — the slow arm never reaches 512 because the fleet throttles it first — so
// capping at 512 penalises exactly the configuration the experiment is trying
// to show winning, and compresses the gap it exists to measure. A cap has to
// sit above the concurrency the FASTEST arm will reach, or it is an instrument
// that reports its own setting.
func TestSeriesPoolOf512DistortsTheComparison(t *testing.T) {
	wl := loadSimWorkload(t)
	mk := func(hit float64, cap int) SimResult {
		return RunAdmissionSim(SimConfig{
			Servers:      fleet(fmt.Sprintf("hit %.2f cap %d", hit, cap), 200_000, 50, 512, hit),
			AdmitEvery:   1,
			TTFTLimitSec: 5,
			HorizonSec:   14400,
			TTFTWindow:   32,
			MaxSeries:    cap,
		}, wl)
	}
	slowFree, slowCap := mk(0.50, 0), mk(0.50, 512)
	fastFree, fastCap := mk(0.90, 0), mk(0.90, 512)
	for _, r := range []SimResult{slowFree, slowCap, fastFree, fastCap} {
		report(t, r)
	}

	slowLoss := 1 - slowCap.ThroughputReqPerSec/slowFree.ThroughputReqPerSec
	fastLoss := 1 - fastCap.ThroughputReqPerSec/fastFree.ThroughputReqPerSec
	t.Logf("throughput lost to the 512 cap: slow arm %.1f%%, fast arm %.1f%%", slowLoss*100, fastLoss*100)

	if fastLoss <= slowLoss+0.02 {
		t.Errorf("the cap cost the fast arm %.1f%% and the slow arm %.1f%%; if it no longer bites "+
			"asymmetrically this test has stopped describing the risk it was written for",
			fastLoss*100, slowLoss*100)
	}
	if fastCap.Bound != "CLIENT series pool" {
		t.Errorf("fast arm bound = %q, want the client pool: the point is that 512 becomes the "+
			"limit before the fleet does", fastCap.Bound)
	}
}

// TestRampRateDecidesTheAnswerOnAFastFleet.
//
// One session per second is not a neutral ramp. It is a bound on how fast load
// can grow, and on a fleet that answers quickly it is the SMALLER bound — the
// run then reports how long it ran rather than what the fleet could carry.
//
// The size of the error is what makes this worth a knob. At a 0.90 hit rate,
// admitting once a second finds 52.8 req/s at 36% prefill utilisation, with the
// gate never closing; admitting five times a second finds 140.7 req/s at 97%.
// The second is the fleet's actual ceiling and the first is the clock's. Run
// both arms of a cache comparison at one-per-second and the fast arm is
// understated by 2.7x, turning a 5x result into a 1.9x one.
//
// The slow arm barely moves across the same range, which is the trap: a ramp
// rate validated on a saturating fleet looks harmless and then silently caps
// the arm it was never tested against.
func TestRampRateDecidesTheAnswerOnAFastFleet(t *testing.T) {
	wl := loadSimWorkload(t)
	run := func(hit, every float64) SimResult {
		return RunAdmissionSim(SimConfig{
			Servers:      fleet(fmt.Sprintf("hit %.2f admit 1/%.2fs", hit, every), 200_000, 50, 512, hit),
			AdmitEvery:   every,
			TTFTLimitSec: 5,
			HorizonSec:   14400,
			TTFTWindow:   32,
		}, wl)
	}

	slowSlow, slowFast := run(0.50, 1), run(0.50, 0.2)
	fastSlow, fastFast := run(0.90, 1), run(0.90, 0.2)
	for _, r := range []SimResult{slowSlow, slowFast, fastSlow, fastFast} {
		report(t, r)
	}

	slowSens := fastRatio(slowFast.ThroughputReqPerSec, slowSlow.ThroughputReqPerSec)
	fastSens := fastRatio(fastFast.ThroughputReqPerSec, fastSlow.ThroughputReqPerSec)
	t.Logf("throughput found by a 5x faster ramp: slow arm %.2fx, fast arm %.2fx", slowSens, fastSens)

	if fastSens < 1.5 {
		t.Errorf("the fast arm gained only %.2fx from a faster ramp; if one-per-second now reaches "+
			"its ceiling, the reason for making the ramp configurable has gone", fastSens)
	}
	if fastSlow.Bound == "prefill bandwidth" {
		t.Errorf("the fast arm saturated at one session per second (%s); this test's premise is "+
			"that it does not", fastSlow.Bound)
	}
	// Converged: a further 4x on the ramp must not move the answer much, or the
	// ceiling reported is still the ramp's rather than the fleet's.
	if conv := fastRatio(run(0.90, 0.05).ThroughputReqPerSec, fastFast.ThroughputReqPerSec); conv > 1.15 {
		t.Errorf("ramping 4x faster again moved throughput %.2fx: 1/0.2s has not converged, so the "+
			"run must ramp faster still or report a floor rather than a ceiling", conv)
	}
}

func fastRatio(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
