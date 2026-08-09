package affinity

import (
	"fmt"
	"testing"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
)

// buildAffinityMode is buildAffinity with the shallow-anchor policy under test.
func buildAffinityMode(mode ladderMode) func(clk clock.Clock, cfg simConfig) simPolicy {
	return func(clk clock.Clock, cfg simConfig) simPolicy {
		p, err := New(Config{
			NodeConcurrency: cfg.conc, SplitGuard: cfg.guard, TailTTL: cfg.ttl,
			Ladder: mode, Clock: clk,
		}, nil)
		if err != nil {
			panic(err)
		}
		return simPolicy{pol: p, committer: p, sweep: func() { p.Sweep() }, tree: p.tree.stats}
	}
}

var anchorModeNames = map[ladderMode]string{
	LadderServeAnyway: "serve-anyway (was)",
	LadderStrict:      "strict       (now)",
}

// This file is the measurement behind "why is avg copies not ~1.0". It is a
// sweep, not an assertion: TestFleetHolderSetsDoNotExplode already pins the
// moderate-load bar. What is missing there is ATTRIBUTION — which tier actually
// creates the duplication, and how that scales with offered load.
//
// The hypothesis under test: the copies are not made by the split tier (which
// is guarded) but by tier 1 anchoring on a shared ancestor when the request's
// own holders are all at the concurrency cap and have been filtered out of the
// candidate set by the gateway. Commit then marks the whole path, so a backend
// that held only the 44-block system prompt is recorded as holding the
// session's private tail as well. metrics.CacheShallowAnchors counts exactly
// those decisions.

// copyProfile is one sweep point.
type copyProfile struct {
	sessions int
	peakUtil float64
	copies   float64
	// trueCopies is measured from what the backends really served, not from
	// the tree's marks — the check that a drop in copies is a real reduction
	// in duplicated KV and not just a quieter model.
	trueCopies float64
	hit        float64
	accepted int
	rej429   int
	// shallow is tier-1 decisions that anchored above the deepest marked run,
	// and shallowBlocks the prefix depth they gave up (and so duplicated).
	shallow      int
	shallowBlock int
	byDecision   map[string]int
	// overflows counts every request served without recording a holder,
	// whether it reached tier 3 or was suppressed by AnchorGuardMark.
	overflows int
	runs      int64
}

func (p copyProfile) String() string {
	cache := p.byDecision["cache"]
	var shallowPct float64
	if cache > 0 {
		shallowPct = 100 * float64(p.shallow) / float64(cache)
	}
	return fmt.Sprintf(
		"sessions=%-4d util=%3.0f%% copies=%.2f REAL=%.2f hit=%5.1f%% accepted=%-6d 429=%-5d | cache=%-6d split=%-4d "+
			"unmarked=%-5d load=%-4d | shallow=%-6d (%4.1f%% of cache) blocksForgone=%-8d runs=%d",
		p.sessions, 100*p.peakUtil, p.copies, p.trueCopies, 100*p.hit, p.accepted, p.rej429,
		cache, p.byDecision["split"], p.overflows, p.byDecision["load"],
		p.shallow, shallowPct, p.shallowBlock, p.runs)
}

// sweepCopies runs the real policy at one offered-load level and attributes the
// duplication it produced.
func sweepCopies(t *testing.T, nodes, sessions int, mode ladderMode) copyProfile {
	t.Helper()
	return sweepSeeded(t, nodes, sessions, mode, defaultSimConfig().seed)
}

func sweepSeeded(t *testing.T, nodes, sessions int, mode ladderMode, seed uint64) copyProfile {
	t.Helper()
	cfg := defaultSimConfig()
	cfg.nodes = nodes
	cfg.sessions = sessions
	cfg.seed = seed

	beforeShallow := readCounter(metrics.CacheShallowAnchors)
	beforeBlocks := readCounter(metrics.CacheShallowAnchorBlocks)

	f := newSimFleet(t, cfg, "prefix-cache-split", buildAffinityMode(mode))
	st := f.run()

	// Ground truth, independent of what the tree believes: how many backends
	// have ACTUALLY served each distinct block. Suppressing a mark lowers
	// tree.AvgCopies by construction, so this is the number that says whether
	// real KV duplication moved or only the router's model of it did.
	holders := map[uint64]int{}
	for _, n := range f.nodes {
		for blk := range n.blocks {
			holders[blk]++
		}
	}
	var sum int
	for _, h := range holders {
		sum += h
	}
	trueCopies := 0.0
	if len(holders) > 0 {
		trueCopies = float64(sum) / float64(len(holders))
	}

	return copyProfile{
		trueCopies: trueCopies,
		sessions:     sessions,
		peakUtil:     st.peakUtil,
		copies:       st.avgCopies,
		hit:          st.hitRate(),
		accepted:     st.accepted,
		rej429:       st.rej429,
		shallow:      int(readCounter(metrics.CacheShallowAnchors) - beforeShallow),
		shallowBlock: int(readCounter(metrics.CacheShallowAnchorBlocks) - beforeBlocks),
		byDecision:   st.byDecision,
		overflows:    st.overflows,
		runs:         st.treeRuns,
	}
}

// TestCopiesVsUtilisation is the diagnostic sweep. Run with -v; it asserts
// nothing about the numbers, it reports them.
//
// Four nodes mirrors the mock-fleet recipe in .ainav/router-testing, where
// router_cache_avg_copies read 1.68 against a fleet held at 100% utilisation by
// construction (client concurrency 128 = 4 nodes x 32 cap).
func TestCopiesVsUtilisation(t *testing.T) {
	for _, nodes := range []int{4, 8} {
		t.Logf("--- %d nodes x 32 concurrency (%d slots) ---", nodes, nodes*32)
		for _, sessions := range []int{40, 80, 120, 160, 220, 300, 420} {
			t.Log(sweepCopies(t, nodes, sessions, LadderServeAnyway))
		}
	}
}

// TestShallowAnchorModesNearSaturation is the only regime where suppressing a
// shallow anchor's mark could be a real win rather than a quieter gauge.
//
// At 100% utilisation every session is forced onto whatever node has a free
// slot, so duplication is a property of the offered load and no marking policy
// can change it — TestShallowAnchorModesAB shows all three modes landing on the
// same REAL figure. Below ~80% there are no shallow anchors to suppress. The
// band between is where the guard could plausibly steer a bounce onto a
// genuinely idle node instead of a nearly-full one.
//
// Averaged over seeds because the tie-break sampler is deliberately unseeded
// (LB-11) and a single run moves by more than the effect being measured.
func TestShallowAnchorModesNearSaturation(t *testing.T) {
	const runs = 5
	for _, nodes := range []int{4, 8} {
		for _, sessions := range []int{110, 130, 150} {
			t.Logf("--- %d nodes x 32, %d sessions, mean of %d seeds ---", nodes, sessions, runs)
			for _, mode := range []ladderMode{LadderServeAnyway, LadderStrict} {
				var util, copies, real, hit, acc float64
				for seed := range runs {
					p := sweepSeeded(t, nodes, sessions, mode, uint64(seed)*0x9e3779b9+1)
					util += p.peakUtil
					copies += p.copies
					real += p.trueCopies
					hit += p.hit
					acc += float64(p.accepted)
				}
				n := float64(runs)
				t.Logf("%-21s util=%3.0f%% copies=%.3f REAL=%.3f hit=%5.2f%% accepted=%.0f",
					anchorModeNames[mode], 100*util/n, copies/n, real/n, 100*hit/n, acc/n)
			}
		}
	}
}

// TestShallowAnchorModesAB is the A/B: same seeded workload, same fleet, three
// answers to the shallow anchor. Reports the whole trade — copies, ground-truth
// hit rate, and work completed — because suppressing a mark can only be an
// improvement if it does not cost the hit rate that motivates the policy.
func TestShallowAnchorModesAB(t *testing.T) {
	for _, nodes := range []int{4, 8} {
		for _, sessions := range []int{160, 300} {
			t.Logf("--- %d nodes x 32, %d sessions ---", nodes, sessions)
			for _, mode := range []ladderMode{LadderServeAnyway, LadderStrict} {
				p := sweepCopies(t, nodes, sessions, mode)
				t.Logf("%-21s %s", anchorModeNames[mode], p)
			}
		}
	}
}
