package affinity

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// This file is the fleet-level check: an offline replay of agentic traffic
// against the REAL policy code, reproducing the acceptance criterion encoded in
// kv-router-sim.html's renderVerdict().
//
// On repeatability: the WORKLOAD is seeded and replays identically, but a run's
// totals still move by around a percent, because every policy here breaks ties
// among equally-loaded backends with the package-level reservoir sampler that
// LB-11 requires and that deliberately is not seeded. That is why every
// assertion below is a bound or a comparison between two arms of the same run,
// never a transcribed number.
//
// It differs from the reference simulator in three ways, each deliberate:
//
//   - Admission is the gateway's. The harness reproduces gateway.candidates()
//     verbatim, so a rejection means every backend was at its limit, which is
//     what makes "premature rejection" measurable rather than definitional.
//
//   - Commit is delayed. The router commits from the proxy's ModifyResponse,
//     i.e. once response headers arrive, so there is a full prefill of lag
//     between choosing a backend and recording it. Every request that starts
//     inside that window still sees the pre-commit tree. The simulator commits
//     at issue and so cannot show this at all; scheduling the commit at
//     T+prefill reproduces it exactly, and deterministically — goroutines would
//     have shown the same effect but made the numbers unassertable.
//
//   - Backends carry ground truth. Each node tracks the blocks it has actually
//     served, separately from what the tree PREDICTS it holds, so prediction
//     accuracy is measured rather than assumed (kvcache's package doc is
//     explicit that residency is predicted and never observed).

// simConfig is one run's parameters. Defaults mirror the reference simulator's
// sliders: 8 nodes x 32 concurrency, a 20% split guard, a 5-minute tail TTL.
type simConfig struct {
	seed     uint64
	nodes    int
	conc     int64
	guard    float64
	ttl      time.Duration
	sessions int // target live sessions; above nodes*conc this saturates
	duration time.Duration
	dt       time.Duration
}

func defaultSimConfig() simConfig {
	return simConfig{
		seed: 0x9e3779b9, nodes: 8, conc: 32, guard: DefaultSplitGuard,
		ttl: 5 * time.Minute, duration: 15 * time.Minute, dt: 200 * time.Millisecond,
	}
}

// saturatedSimConfig offers more work than the fleet's 8x32 = 256 slots can
// take, which is the only regime in which the admission verdict means anything:
// Anton's bar is about what happens at full load, and "ignore any optimization
// ideas that are done not for the worst case".
func saturatedSimConfig() simConfig {
	c := defaultSimConfig()
	c.sessions = 420
	return c
}

// moderateSimConfig stays under the fleet's capacity, so affinity is actually
// achievable and duplication is a policy choice rather than a consequence of
// every session being forced to bounce.
func moderateSimConfig() simConfig {
	c := defaultSimConfig()
	c.sessions = 150
	return c
}

// Rates from the reference simulator's CFG.
const (
	simBlockTokens = 256
	simPrefillTps  = 25000.0
	simDecodeTps   = 46.0
	simContention  = 0.014
	simSweepEvery  = 10 * time.Second
)

type simNode struct {
	b       *registry.Backend
	blocks  map[uint64]bool // ground truth: what this backend has actually served
	maxedAt time.Duration   // when it first reached its limit; -1 until then
	peak    int64
	served  int64
	hit     int64
	total   int64
}

type simSession struct {
	id       int
	blocks   []uint64
	parent   int
	tokens   float64
	cap      float64
	dieAt    time.Duration
	nextAt   time.Duration
	pending  bool
	reqEnd   time.Duration
	reqLen   int
	attempts int
	kids     int
}

type simStats struct {
	name string

	turns     int
	attempts  int
	accepted  int
	completed int

	rej429       int
	rejHard      int
	rejPremature int
	idleSum      int64
	idleN        int
	worstIdle    int64

	firstRejectAt time.Duration // -1 if none
	allMaxedAt    time.Duration // -1 if never
	neverMaxed    []string

	splits     int
	overflows  int
	byDecision map[string]int

	hitBlocks   int64
	totalBlocks int64

	// predictedHit is what the tree implied the chosen backend held; actualHit
	// is what it really held. The gap is the prediction error.
	predictedHit int64
	actualHit    int64

	peakUtil  float64
	avgCopies float64
	treeRuns  int64
	tailSet   int64
	expired   int64
}

func (s simStats) hitRate() float64 {
	if s.totalBlocks == 0 {
		return 0
	}
	return float64(s.hitBlocks) / float64(s.totalBlocks)
}

func (s simStats) meanIdlePct(cap int64) float64 {
	if s.idleN == 0 {
		return 0
	}
	return 100 * (float64(s.idleSum) / float64(s.idleN)) / float64(cap)
}

// saturatedFirst is renderVerdict()'s bar: every node reached its concurrency
// limit before the router rejected anybody.
func (s simStats) saturatedFirst() bool {
	if s.firstRejectAt < 0 {
		return true // nothing was ever rejected
	}
	return s.allMaxedAt >= 0 && s.allMaxedAt <= s.firstRejectAt
}

func (s simStats) String() string {
	return fmt.Sprintf(
		"%-26s turns=%-6d accepted=%-6d 429=%-5d (hard=%-5d premature=%-4d) "+
			"hit=%5.1f%% splits=%-5d overflow=%-5d copies=%.2f runs=%-6d peakUtil=%3.0f%%",
		s.name, s.turns, s.accepted, s.rej429, s.rejHard, s.rejPremature,
		100*s.hitRate(), s.splits, s.overflows, s.avgCopies, s.treeRuns, 100*s.peakUtil)
}

// simFleet is the harness. It owns the workload, the clock and the ground
// truth; the policy under test is a black box behind policy.Policy.
type simFleet struct {
	cfg   simConfig
	clk   *clock.Fake
	rng   *rand.Rand
	start time.Time

	sp simPolicy

	nodes    []*simNode
	sessions []*simSession
	stats    simStats

	// pending work, kept as sorted-by-time slices rather than a heap: the
	// counts here are small and a heap would only obscure the model.
	inflight []*simInflight
	commits  []*simCommit

	blockSeq    uint64
	sessSeq     int
	spawnBudget float64
	lastSweep   time.Duration

	shared     []uint64
	frameworks [][]uint64
}

type simInflight struct {
	node  *simNode
	lse   *lease.Lease
	endAt time.Duration
	sess  *simSession
}

type simCommit struct {
	at time.Duration
	b  *registry.Backend
	rr *policy.RoutingRequest
}

func (f *simFleet) now() time.Duration { return f.clk.Now().Sub(f.start) }

// simPolicy is the policy under test plus the optional hooks the harness uses
// to drive and inspect it.
type simPolicy struct {
	pol       policy.Policy
	committer policy.Committer
	sweep     func()
	tree      func() treeStats
	// unfiltered gives the policy the whole backend list instead of the
	// gateway's under-the-limit filter, for a policy that makes its own
	// admission decision.
	unfiltered bool
}

func newSimFleet(t *testing.T, cfg simConfig, name string,
	build func(clk clock.Clock, cfg simConfig) simPolicy,
) *simFleet {
	t.Helper()
	clk := clock.NewFake(time.Time{})
	sp := build(clk, cfg)

	reg := registry.New(registry.Options{})
	f := &simFleet{
		cfg: cfg, clk: clk, start: clk.Now(),
		rng: rand.New(rand.NewPCG(cfg.seed, cfg.seed+1)),
		sp:  sp,
	}
	f.stats = simStats{name: name, firstRejectAt: -1, allMaxedAt: -1, byDecision: map[string]int{}}

	for i := range cfg.nodes {
		b, err := reg.Add(registry.Spec{URL: fmt.Sprintf("http://sim%02d:8000", i), Capacity: cfg.conc})
		if err != nil {
			t.Fatalf("add sim backend: %v", err)
		}
		b.SetHealth(registry.Healthy)
		if lc, ok := sp.pol.(interface{ AddBackend(*registry.Backend) }); ok {
			lc.AddBackend(b)
		}
		f.nodes = append(f.nodes, &simNode{b: b, blocks: map[uint64]bool{}, maxedAt: -1})
	}

	// The workload's fixed prefixes, exactly as the reference builds them: a
	// 44-block Claude Code system prompt used by most traffic, and three
	// framework preambles.
	for range 44 {
		f.blockSeq++
		f.shared = append(f.shared, f.blockSeq)
	}
	for _, n := range []int{12, 30, 68} {
		var fw []uint64
		for range n {
			f.blockSeq++
			fw = append(fw, f.blockSeq)
		}
		f.frameworks = append(f.frameworks, fw)
	}
	return f
}

func (f *simFleet) newSession(parent *simSession) *simSession {
	f.sessSeq++
	s := &simSession{id: f.sessSeq, parent: -1}
	switch {
	case parent != nil:
		// A subagent forks the parent's LIVE prefix, multiplying pressure on
		// one hash path — and, because siblings fire together, it is the main
		// generator of concurrent first-touch.
		s.blocks = append([]uint64(nil), parent.blocks...)
		s.parent = parent.id
		s.cap = 40000 + f.rng.Float64()*110000
		s.dieAt = f.now() + time.Duration(60+f.rng.Float64()*180)*time.Second
	default:
		r := f.rng.Float64()
		switch {
		case r < 0.68:
			s.blocks = append([]uint64(nil), f.shared...)
		case r < 0.90:
			s.blocks = append([]uint64(nil), f.frameworks[f.rng.IntN(len(f.frameworks))]...)
		default:
			f.blockSeq++
			s.blocks = []uint64{f.blockSeq}
		}
		s.cap = math.Exp(math.Log(50000) + math.Pow(f.rng.Float64(), 1.8)*math.Log(1000000/50000))
		s.dieAt = f.now() + time.Duration(120+f.rng.Float64()*1500)*time.Second
	}
	s.tokens = float64(len(s.blocks)*simBlockTokens + 2000)
	s.nextAt = f.now() + time.Duration(f.rng.Float64()*2*float64(time.Second))
	f.sessions = append(f.sessions, s)
	return s
}

func (f *simFleet) growSession(s *simSession) {
	add := 200 + f.rng.Float64()*1500
	if f.rng.Float64() < 0.06 {
		add += 8000 + f.rng.Float64()*32000 // a fat tool result or file read
	}
	s.tokens += add
	want := int(math.Ceil(s.tokens / simBlockTokens))
	for len(s.blocks) < want {
		f.blockSeq++
		s.blocks = append(s.blocks, f.blockSeq)
	}
}

// candidates reproduces gateway.candidates(): healthy backends strictly below
// the per-node concurrency limit. This is the admission boundary, and the
// reason a rejection by THIS harness means zero idle slots fleet-wide.
// candidates reproduces gateway.candidates(): HEALTH ONLY.
//
// It used to also apply the --max-node-concurrency cap, because the gateway
// did. Capacity is now judged entirely by the flow's signals, so a harness that
// kept filtering here would be testing a gateway that no longer exists — and
// would hide the case the refused signal is for, where the router hands over a
// backend that turns out to be full.
func (f *simFleet) candidates() []*registry.Backend {
	out := make([]*registry.Backend, 0, len(f.nodes))
	for _, n := range f.nodes {
		if n.b.Available() {
			out = append(out, n.b)
		}
	}
	return out
}

func (f *simFleet) allBackends() []*registry.Backend {
	out := make([]*registry.Backend, 0, len(f.nodes))
	for _, n := range f.nodes {
		out = append(out, n.b)
	}
	return out
}

func (f *simFleet) freeSlots() (free int64) {
	for _, n := range f.nodes {
		if d := f.cfg.conc - n.b.Inflight(); d > 0 {
			free += d
		}
	}
	return free
}

func (f *simFleet) nodeFor(b *registry.Backend) *simNode {
	for _, n := range f.nodes {
		if n.b == b {
			return n
		}
	}
	return nil
}

// actualPrefixHit is the ground truth: how many leading blocks this backend has
// really served before. Prefix semantics, so a gap ends the run.
func (n *simNode) actualPrefixHit(blocks []uint64) int {
	i := 0
	for i < len(blocks) && n.blocks[blocks[i]] {
		i++
	}
	return i
}

func (f *simFleet) issue(s *simSession) {
	now := f.now()
	if s.reqLen == 0 {
		f.growSession(s)
		s.reqLen = len(s.blocks)
		s.attempts = 0
		f.stats.turns++
	}
	s.attempts++
	f.stats.attempts++

	units := make([]kvcache.Unit, s.reqLen)
	for i := range units {
		units[i] = kvcache.Unit{Hash: s.blocks[i], Tokens: simBlockTokens}
	}
	rr := &policy.RoutingRequest{Units: units, Model: "sim", RequestID: fmt.Sprint(s.id)}

	cands := f.candidates()
	if f.sp.unfiltered {
		cands = f.allBackends()
	}
	var chosen *registry.Backend
	var err error
	if len(cands) > 0 {
		chosen, err = f.sp.pol.Select(context.Background(), cands, rr)
	} else {
		err = policy.ErrNoCandidates
	}

	if chosen == nil || err != nil {
		free := f.freeSlots()
		f.stats.rej429++
		if free > 0 {
			f.stats.rejPremature++
			f.stats.idleSum += free
			f.stats.idleN++
			f.stats.worstIdle = max(f.stats.worstIdle, free)
		} else {
			f.stats.rejHard++
		}
		if f.stats.firstRejectAt < 0 {
			f.stats.firstRejectAt = now
		}
		// Client backoff, as the reference models it.
		back := math.Min(25, math.Pow(1.9, float64(s.attempts-1))) * (0.7 + f.rng.Float64()*0.6)
		s.nextAt = now + time.Duration(back*float64(time.Second))
		return
	}

	s.reqLen = 0
	n := f.nodeFor(chosen)

	// Ground truth decides the work; the router only decided where.
	hit := n.actualPrefixHit(s.blocks[:len(units)])
	uncached := len(units) - hit
	n.hit += int64(hit)
	n.total += int64(len(units))
	f.stats.hitBlocks += int64(hit)
	f.stats.totalBlocks += int64(len(units))

	prefill := float64(uncached*simBlockTokens) / simPrefillTps
	outTok := 150 + f.rng.Float64()*1300
	dur := (prefill + outTok/simDecodeTps) * (1 + simContention*float64(chosen.Inflight()))

	lse := lease.Acquire(chosen)
	n.served++
	f.stats.accepted++
	end := now + time.Duration(dur*float64(time.Second))
	f.inflight = append(f.inflight, &simInflight{node: n, lse: lse, endAt: end, sess: s})
	s.pending = true
	s.reqEnd = end

	// The backend now genuinely holds every block of this request.
	for _, h := range s.blocks[:len(units)] {
		n.blocks[h] = true
	}

	// Commit lands when response headers would arrive — after prefill, not
	// after decode. Everything issued inside that window still sees the
	// pre-commit tree.
	if f.sp.committer != nil {
		f.commits = append(f.commits, &simCommit{
			at: now + time.Duration(prefill*float64(time.Second)), b: chosen, rr: rr,
		})
	}

	// Subagent fan-out: forks the live prefix, so siblings hit one hash path
	// simultaneously.
	if f.rng.Float64() < 0.035 && s.parent < 0 && s.kids < 6 &&
		len(f.sessions) < int(float64(f.cfg.sessions)*1.6) {
		k := 2 + f.rng.IntN(3)
		s.kids += k
		for range k {
			f.newSession(s)
		}
	}
}

func (f *simFleet) step() {
	f.clk.Advance(f.cfg.dt)
	now := f.now()

	// Completions.
	kept := f.inflight[:0]
	for _, in := range f.inflight {
		if in.endAt <= now {
			in.lse.Release()
			f.stats.completed++
			continue
		}
		kept = append(kept, in)
	}
	f.inflight = kept

	// Deferred commits.
	keptC := f.commits[:0]
	for _, c := range f.commits {
		if c.at <= now {
			f.sp.committer.Commit(c.b, c.rr)
			continue
		}
		keptC = append(keptC, c)
	}
	f.commits = keptC

	// Session lifecycle.
	live := f.sessions[:0]
	for _, s := range f.sessions {
		if s.pending && s.reqEnd <= now {
			s.pending = false
			s.nextAt = now + time.Duration((0.4+f.rng.Float64()*1.6)*float64(time.Second))
		}
		if !s.pending && s.reqLen == 0 && (s.tokens > s.cap || now > s.dieAt) {
			continue
		}
		live = append(live, s)
	}
	f.sessions = live

	// Ramp new sessions in at a bounded rate: no thundering herd at t=0.
	f.spawnBudget = math.Min(f.spawnBudget+8*f.cfg.dt.Seconds(), 40)
	for len(f.sessions) < f.cfg.sessions && f.spawnBudget >= 1 {
		f.newSession(nil)
		f.spawnBudget--
	}
	if len(f.sessions) >= f.cfg.sessions {
		f.spawnBudget = math.Min(f.spawnBudget, 8)
	}

	for _, s := range f.sessions {
		if !s.pending && now >= s.nextAt {
			f.issue(s)
		}
	}

	if f.sp.sweep != nil && now-f.lastSweep >= simSweepEvery {
		f.lastSweep = now
		f.sp.sweep()
	}

	// Saturation bookkeeping, the input to the verdict.
	var total int64
	allMaxed := true
	for _, n := range f.nodes {
		fl := n.b.Inflight()
		total += fl
		n.peak = max(n.peak, fl)
		if fl >= f.cfg.conc && n.maxedAt < 0 {
			n.maxedAt = now
		}
		if n.maxedAt < 0 {
			allMaxed = false
		}
	}
	if util := float64(total) / float64(int64(len(f.nodes))*f.cfg.conc); util > f.stats.peakUtil {
		f.stats.peakUtil = util
	}
	if allMaxed && f.stats.allMaxedAt < 0 {
		f.stats.allMaxedAt = now
	}
}

// decisionLabels is the closed enum RouteDecisions uses; captured as deltas
// because the collectors are package-level and shared across the test binary.
var decisionLabels = []string{"cache", "split", "overflow", "load", "other"}

func (f *simFleet) run() simStats {
	before := map[string]float64{}
	for _, l := range decisionLabels {
		before[l] = readCounter(metrics.RouteDecisions.WithLabelValues(l))
	}
	beforeSplits := readCounter(metrics.CacheSplits)
	beforeOverflow := readCounter(metrics.CacheOverflows)

	for f.now() < f.cfg.duration {
		f.step()
	}

	for _, l := range decisionLabels {
		if d := int(readCounter(metrics.RouteDecisions.WithLabelValues(l)) - before[l]); d > 0 {
			f.stats.byDecision[l] = d
		}
	}
	f.stats.splits = int(readCounter(metrics.CacheSplits) - beforeSplits)
	f.stats.overflows = int(readCounter(metrics.CacheOverflows) - beforeOverflow)

	if f.sp.tree != nil {
		ts := f.sp.tree()
		f.stats.avgCopies, f.stats.treeRuns = ts.AvgCopies, ts.Runs
		f.stats.tailSet, f.stats.expired = ts.Tails, ts.Expired
	}
	for _, n := range f.nodes {
		if n.maxedAt < 0 {
			f.stats.neverMaxed = append(f.stats.neverMaxed, fmt.Sprintf("%s(peak %d)", n.b.URL, n.peak))
		}
	}
	sort.Strings(f.stats.neverMaxed)
	// Release whatever is still in flight so the leases balance.
	for _, in := range f.inflight {
		in.lse.Release()
	}
	return f.stats
}

func readCounter(c interface{ Write(*dto.Metric) error }) float64 {
	var m dto.Metric
	if err := c.Write(&m); err != nil {
		return 0
	}
	if m.Counter != nil {
		return m.Counter.GetValue()
	}
	return 0
}

// ---------------------------------------------------------------- builders

func buildAffinity(clk clock.Clock, cfg simConfig) simPolicy {
	p, err := New(Config{
		NodeConcurrency: cfg.conc, SplitGuard: cfg.guard, TailTTL: cfg.ttl, Clock: clk,
	}, nil)
	if err != nil {
		panic(err)
	}
	return simPolicy{pol: p, committer: p, sweep: func() { p.Sweep() }, tree: p.tree.stats}
}

func buildLeastOutstanding(clock.Clock, simConfig) simPolicy {
	return simPolicy{pol: policy.LeastOutstanding{}}
}

// errSimReject is how the reference ladder says "429". Our own policy has no
// equivalent: it never rejects, because admission is the gateway's.
var errSimReject = errors.New("sim: reference ladder rejected")

// simLadder is kv-router-sim.html's route() transcribed onto the same shared
// tree, for the cross-check in TestSimulatorLadderRejectsWhileCapacityIsIdle.
//
// Three tiers, not four: least-loaded holder if one is under the limit, else a
// split onto a backend outside the holder set that clears the guard, else
// reject. It is handed the UNFILTERED backend list, because the reference makes
// its own admission decision rather than inheriting the gateway's.
type simLadder struct {
	tree  *tree
	conc  int64
	guard float64
}

func (s *simLadder) Name() string { return "sim-reference-ladder" }

func (s *simLadder) Select(_ context.Context, cands []*registry.Backend, rr *policy.RoutingRequest) (*registry.Backend, error) {
	if len(cands) == 0 {
		return nil, policy.ErrNoCandidates
	}
	if len(rr.Units) == 0 {
		return leastLoadedOf(cands), nil
	}
	pth := path{units: rr.Units, modelKey: modelKey(rr.Model)}

	var everything markSet
	slots := make([]int, len(cands))
	for i, c := range cands {
		slots[i] = s.tree.slotOrCreate(c.URL)
		everything.Add(slots[i])
	}
	a := s.tree.walk(pth, everything)

	// The pool is the holder set if there is one, otherwise the whole fleet.
	pool := cands
	if !a.held.Empty() {
		pool = pool[:0:0]
		for i, c := range cands {
			if a.held.Has(slots[i]) {
				pool = append(pool, c)
			}
		}
	}
	best := leastLoadedOf(pool)
	if best != nil && best.Inflight() < s.conc {
		return best, nil
	}

	// Every holder is saturated: split onto a node outside the set that is far
	// enough below the reference signal.
	own := s.conc
	if best != nil {
		own = best.Inflight()
	}
	thr := float64(own) * (1 - s.guard)
	if !a.held.Empty() {
		var out []*registry.Backend
		for i, c := range cands {
			if a.held.Has(slots[i]) {
				continue
			}
			if float64(c.Inflight()) >= thr || c.Inflight() >= s.conc {
				continue
			}
			out = append(out, c)
		}
		if b := leastLoadedOf(out); b != nil {
			return b, nil
		}
	}
	return nil, errSimReject
}

func (s *simLadder) Commit(b *registry.Backend, rr *policy.RoutingRequest) {
	if b == nil || len(rr.Units) == 0 {
		return
	}
	s.tree.commit(path{units: rr.Units, modelKey: modelKey(rr.Model)}, s.tree.slotOrCreate(b.URL))
}

func leastLoadedOf(cands []*registry.Backend) *registry.Backend {
	var best *registry.Backend
	for _, c := range cands {
		if best == nil || c.Inflight() < best.Inflight() {
			best = c
		}
	}
	return best
}

func buildSimLadder(clk clock.Clock, cfg simConfig) simPolicy {
	l := &simLadder{tree: newTree(clk, cfg.ttl), conc: cfg.conc, guard: cfg.guard}
	return simPolicy{
		pol: l, committer: l, unfiltered: true,
		sweep: func() { l.tree.sweep() }, tree: l.tree.stats,
	}
}

// ---------------------------------------------------------------- the tests

// TestFleetVerdictDuplicationHoldsAtSaturation is the acceptance criterion.
//
// It is NOT the reference simulator's renderVerdict(). That function fails a
// run in which any 429 is issued while idle capacity exists anywhere, and this
// policy issues a great many of them: with a 20% guard, a backend between 80%
// and 100% of its limit is idle and refused on purpose, because serving there
// means a second copy of the prefix. Anton's call, made with these numbers in
// front of him: mean holders per block near 1.05 is worth the 429s and worth
// nodes not always reaching full concurrency.
//
// So what is asserted here is the thing that call was made FOR — duplication
// stays near one even at full saturation — and the cost is measured and logged
// rather than asserted away. The previous ladder scored 3.4 on this workload.
//
// The cost turned out to be much smaller than the trade implied: a request that
// lands warm finishes sooner and frees its slot sooner, so accepted work is
// unchanged to within a percent while the hit rate rises ~20 points. What is
// genuinely paid is that the first 429 now arrives before every node has maxed.
func TestFleetVerdictDuplicationHoldsAtSaturation(t *testing.T) {
	cfg := saturatedSimConfig()
	st := newSimFleet(t, cfg, "prefix-cache-split", buildAffinity).run()
	t.Log(st)

	if st.rej429 == 0 {
		t.Fatalf("the workload never saturated the fleet, so the verdict is untested "+
			"(peak utilisation %.0f%%); raise simConfig.sessions", 100*st.peakUtil)
	}
	if st.peakUtil < 0.99 {
		t.Errorf("peak utilisation %.1f%%, want ~100%%: the guard is only under test at full load",
			100*st.peakUtil)
	}

	// The bar. Under 164% oversubscription the previous ladder reached 3.4
	// holders per block; the guard exists to keep this near 1.
	if st.avgCopies > 1.20 {
		t.Errorf("mean holders per block = %.2f at %.0f%% utilisation, want <= 1.20: the split "+
			"guard is the only thing that adds a holder, so anything higher means something "+
			"is marking outside it", st.avgCopies, 100*st.peakUtil)
	}

	// A rejection must never happen while a backend that HOLDS the prefix was
	// available — that would be affinity failing, not the guard biting.
	if st.rejHard == 0 {
		t.Error("no rejection happened with the fleet genuinely full, so the workload is not " +
			"exercising the admission path the guard sits behind")
	}

	// Measured, not asserted: this is the price of the bar above.
	t.Logf("cost of the guard: %d of %d rejections found idle capacity (mean %.1f%% of the fleet "+
		"idle, worst %d slots); first 429 at %v, all nodes maxed at %v",
		st.rejPremature, st.rej429, st.meanIdlePct(int64(cfg.nodes)*cfg.conc), st.worstIdle,
		st.firstRejectAt, st.allMaxedAt)
	if !st.saturatedFirst() {
		t.Logf("note: the reference simulator's renderVerdict() would call this run FAIL "+
			"(first 429 at %v precedes all-nodes-maxed at %v); accepted deliberately",
			st.firstRejectAt, st.allMaxedAt)
	}
}

// TestMatchesReferenceLadderAtSaturation is the cross-check against the
// reference design, which this policy no longer diverges from on the axis that
// used to differ.
//
// Same seed, same workload, same fleet — only the ladder differs. The one
// structural difference left is where admission lives: the reference makes its
// own, so it is handed the unfiltered backend list, while ours inherits the
// gateway's under-the-limit filter. On behaviour they should now agree, because
// both refuse rather than serve a prefix onto a backend that fails the guard.
//
// The shape of the disagreement is the interesting part if this ever breaks:
// ours rejecting MORE than the reference would mean the gateway filter and the
// guard are double-counting; ours duplicating more would mean something still
// marks outside a split.
func TestMatchesReferenceLadderAtSaturation(t *testing.T) {
	cfg := saturatedSimConfig()
	faithful := newSimFleet(t, cfg, "reference 3-tier ladder", buildSimLadder).run()
	ours := newSimFleet(t, cfg, "prefix-cache-split", buildAffinity).run()

	t.Log(faithful)
	t.Log(ours)
	t.Logf("our decisions by tier: %v", ours.byDecision)

	if faithful.rejPremature == 0 {
		t.Fatal("the reference ladder rejected nothing prematurely, so either the port " +
			"or the workload is not reproducing the conditions the guard bites under")
	}

	// Duplication must match the reference's, since both now grow a holder set
	// only through a guarded split.
	if d := ours.avgCopies - faithful.avgCopies; d > 0.15 {
		t.Errorf("avg copies %.2f vs reference %.2f (+%.2f): ours is marking through some path "+
			"the reference does not have", ours.avgCopies, faithful.avgCopies, d)
	}

	// Hit rate must not trail the reference by more than noise.
	if d := faithful.hitRate() - ours.hitRate(); d > 0.03 {
		t.Errorf("hit rate %.1f%% vs reference %.1f%% (-%.1f points): affinity is landing worse "+
			"than the design it implements", 100*ours.hitRate(), 100*faithful.hitRate(), 100*d)
	}

	// Work completed must not trail either: refusing to serve cold frees slots
	// sooner, so this should come out level rather than paying for the guard.
	if float64(ours.accepted) < 0.95*float64(faithful.accepted) {
		t.Errorf("accepted %d vs reference %d: the guard is costing throughput the reference "+
			"does not pay", ours.accepted, faithful.accepted)
	}

	t.Logf("at saturation: hit rate %.1f%% (reference %.1f%%), avg copies %.2f (reference %.2f), "+
		"accepted %d (reference %d), premature 429s %d (reference %d)",
		100*ours.hitRate(), 100*faithful.hitRate(),
		ours.avgCopies, faithful.avgCopies, ours.accepted, faithful.accepted,
		ours.rejPremature, faithful.rejPremature)
}

// TestFleetHolderSetsDoNotExplode is the tripwire for the one hazard this
// design knowingly accepts: a run under continuous traffic never reaches its
// idle TTL, and TTL is the only thing that removes a holder, so holder sets on
// hot runs can only grow.
//
// Measured at moderate load, deliberately. Above the fleet's capacity every
// session is forced to bounce between backends and the resulting duplication is
// a property of the offered load, not of the policy — a bar set there would
// measure the workload. Where affinity is achievable, it should be achieved,
// and the mean holders per block should stay near one.
func TestFleetHolderSetsDoNotExplode(t *testing.T) {
	cfg := moderateSimConfig()
	st := newSimFleet(t, cfg, "prefix-cache-split", buildAffinity).run()
	t.Log(st)
	t.Logf("avg copies %.3f over %d nodes, %d runs, %d tails, %d blocks expired",
		st.avgCopies, cfg.nodes, st.treeRuns, st.tailSet, st.expired)

	if st.peakUtil > 0.99 {
		t.Fatalf("the 'moderate' config saturated (peak %.0f%%); it no longer measures "+
			"avoidable duplication", 100*st.peakUtil)
	}
	if st.avgCopies > 1.5 {
		t.Errorf("mean holders per block = %.2f across %d nodes at %.0f%% peak utilisation: "+
			"holder sets are growing where affinity was achievable, which is the failure mode "+
			"per-backend mark budgets would fix",
			st.avgCopies, cfg.nodes, 100*st.peakUtil)
	}
	if st.expired == 0 {
		t.Error("no blocks were ever released by TTL eviction, so the tree grows forever")
	}
}

// TestFleetPredictionMatchesReality measures what kvcache's package doc warns
// about: the tree PREDICTS residency and never observes it. The harness tracks
// what each backend has really served, so "affinity sent this to a holder" is
// checked against "that backend really held the prefix" rather than assumed.
func TestFleetPredictionMatchesReality(t *testing.T) {
	for _, cfg := range []simConfig{moderateSimConfig(), saturatedSimConfig()} {
		st := newSimFleet(t, cfg, "prefix-cache-split", buildAffinity).run()
		t.Logf("%d sessions: ground-truth prefix hit rate %.1f%% of %d blocks (peak util %.0f%%)",
			cfg.sessions, 100*st.hitRate(), st.totalBlocks, 100*st.peakUtil)
		if st.hitRate() < 0.5 {
			t.Errorf("%d sessions: ground-truth hit rate %.1f%%: affinity is not landing on "+
				"backends that actually hold the prefix", cfg.sessions, 100*st.hitRate())
		}
	}
}

// TestFleetBeatsLeastOutstandingOnCacheHits is the A/B that says the policy is
// worth its complexity: the same seeded workload and fleet, routed by load
// alone versus by affinity, at both load levels.
func TestFleetBeatsLeastOutstandingOnCacheHits(t *testing.T) {
	for _, cfg := range []simConfig{moderateSimConfig(), saturatedSimConfig()} {
		aff := newSimFleet(t, cfg, "prefix-cache-split", buildAffinity).run()
		lo := newSimFleet(t, cfg, "least-outstanding", buildLeastOutstanding).run()
		t.Log(aff)
		t.Log(lo)
		t.Logf("%d sessions: prefix hit rate affinity %.1f%% vs least-outstanding %.1f%% (%+.1f points)",
			cfg.sessions, 100*aff.hitRate(), 100*lo.hitRate(), 100*(aff.hitRate()-lo.hitRate()))

		if aff.hitRate() <= lo.hitRate() {
			t.Errorf("%d sessions: affinity hit rate %.1f%% does not beat least-outstanding's %.1f%%",
				cfg.sessions, 100*aff.hitRate(), 100*lo.hitRate())
		}
	}
}
