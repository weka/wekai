// Package prefill moves heavy prefills off decode nodes.
//
// In the common case every node serves prefill and decode. When the engine
// refuses a request because recomputing its prompt would stall the node's
// running decodes (HTTP 503, header X-Weka-Prefill-Reject), the router runs that
// prefill on a node it has taken out of decode routing, then sends the request
// for decode as usual. The prefill leaves its KV in the fleet's shared tier, so
// the decode node loads it rather than recomputing.
//
// A node is in one of three modes. It starts in decode, becomes draining when a
// rejected request needs a prefill node and none has room, becomes prefill once
// its running decode requests have finished, and returns to decode after going
// idle. Which nodes are prefill nodes is therefore the router's decision, made
// from traffic, with nothing configured per node.
package prefill

import (
	"context"
	"errors"
	"math/rand/v2"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

// Mode is a node's role under the split.
type Mode int

const (
	// Decode is the default: the node takes ordinary traffic.
	Decode Mode = iota
	// Draining takes no new decode routing and waits for its running decode
	// requests to finish. Prefill jobs may already be queued on it.
	Draining
	// Prefill runs prefill jobs only.
	Prefill
)

func (m Mode) String() string { return metrics.PrefillModes[m] }

const (
	DefaultNodeBudgetTokens = 4_000_000
	DefaultMaxFleetFraction = 0.5
	DefaultReleaseIdle      = 60 * time.Second

	// drainPoll is how often a queued job re-checks whether its node's decode
	// work has finished. Drain is bounded by the longest running decode, so
	// this is for responsiveness, not precision.
	drainPoll = 25 * time.Millisecond
)

// ErrNoPrefillNode reports that no node can take the job: none is a prefill
// node, and none can become one without leaving the fleet no decode node or
// exceeding the prefill share.
var ErrNoPrefillNode = errors.New("prefill: no prefill node available")

// Config tunes the split.
type Config struct {
	// NodeBudgetTokens bounds the prefill tokens queued plus in flight on one
	// node. A job bigger than the budget still runs on an otherwise empty node.
	NodeBudgetTokens int64
	// MaxFleetFraction caps the share of the fleet that is draining or
	// prefill. At least one node always stays in decode.
	MaxFleetFraction float64
	// NodeConcurrency is the router's hard per-node in-flight limit
	// (--max-node-concurrency). A prefill node never runs more jobs than this
	// at once, and counts the node's other in-flight requests against it; jobs
	// beyond it wait their turn. 0 is no limit.
	NodeConcurrency int64
	// ReleaseIdle is how long a node goes without prefill work before it
	// returns to decode.
	ReleaseIdle time.Duration

	Clock    clock.Clock
	PoolName string
}

func (c Config) withDefaults() Config {
	if c.NodeBudgetTokens <= 0 {
		c.NodeBudgetTokens = DefaultNodeBudgetTokens
	}
	if c.MaxFleetFraction <= 0 || c.MaxFleetFraction > 1 {
		c.MaxFleetFraction = DefaultMaxFleetFraction
	}
	if c.ReleaseIdle <= 0 {
		c.ReleaseIdle = DefaultReleaseIdle
	}
	if c.Clock == nil {
		c.Clock = clock.Real{}
	}
	return c
}

// Picker chooses among the eligible prefill nodes. The gateway passes the
// pool's routing flow, so the choice uses the same prefix knowledge and load
// rules as ordinary routing, over the eligible set.
type Picker func(eligible []*registry.Backend) (*registry.Backend, error)

type node struct {
	b *registry.Backend
	// running is the jobs holding a slot; queue the jobs waiting for one, in
	// arrival order.
	running  int
	queue    []*Job
	mode     Mode
	pending  int64
	lastWork time.Time
}

// Manager owns the node modes for one pool.
type Manager struct {
	cfg Config
	m   *metrics.PrefillMetrics

	mu    sync.Mutex
	nodes map[string]*node

	// Mirrors of the Prometheus counters, which cannot be read back, for the
	// /router-viz state.
	rejections, jobRetries, jobsOK, jobsFailed, conversions, releases, fallbackNoNode, fallbackJobFailed atomic.Uint64
}

// New builds a manager for one pool.
func New(cfg Config) *Manager {
	cfg = cfg.withDefaults()
	return &Manager{cfg: cfg, m: metrics.ForPrefill(cfg.PoolName), nodes: map[string]*node{}}
}

// Job is a charge against a prefill node, released with Done.
type Job struct {
	mgr    *Manager
	n      *node
	tokens int64
	once   sync.Once
	// granted closes when the job holds a slot; held is set with it.
	granted chan struct{}
	held    bool
	queued  bool
	// Backend is the node the job runs on.
	Backend *registry.Backend
}

// Done releases the job's tokens and stamps the node as recently busy.
func (j *Job) Done() {
	j.once.Do(func() {
		j.mgr.mu.Lock()
		defer j.mgr.mu.Unlock()
		j.mgr.dequeueLocked(j)
		if j.held {
			j.held = false
			j.n.running--
			j.mgr.grantLocked(j.n)
		}
		j.n.pending -= j.tokens
		j.n.lastWork = j.mgr.cfg.Clock.Now()
		j.mgr.publishLocked(j.n)
	})
}

func (m *Manager) nodeLocked(b *registry.Backend) *node {
	n := m.nodes[b.URL]
	if n == nil {
		n = &node{b: b}
		m.nodes[b.URL] = n
	}
	n.b = b
	return n
}

// ModeOf reports a backend's current mode.
func (m *Manager) ModeOf(b *registry.Backend) Mode {
	m.mu.Lock()
	defer m.mu.Unlock()
	if n := m.nodes[b.URL]; n != nil {
		return n.mode
	}
	return Decode
}

// DecodeNodes is the candidate set for ordinary routing: everything not
// draining or prefill. When that would leave nothing — the decode nodes have
// all gone unhealthy — it returns cands unchanged, since a request served on a
// prefill node beats one refused.
func (m *Manager) DecodeNodes(cands []*registry.Backend) []*registry.Backend {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*registry.Backend, 0, len(cands))
	for _, b := range cands {
		if n := m.nodes[b.URL]; n == nil || n.mode == Decode {
			out = append(out, b)
		}
	}
	if len(out) == 0 {
		return cands
	}
	return out
}

// Reserve picks the node a prefill job of tokens runs on and charges it.
//
//  1. Among draining and prefill nodes with budget room, pick by the routing
//     policy.
//  2. Failing that, and while the prefill share is under the cap, take the
//     least-loaded decode node and start draining it.
//  3. Failing that, queue on the prefill node with the least pending tokens.
//  4. With no prefill node at all, ErrNoPrefillNode.
//
// cands is the pool's healthy backends: the fleet the cap is a share of.
func (m *Manager) Reserve(cands []*registry.Backend, tokens int64, pick Picker) (*Job, error) {
	tokens = max(tokens, 1)
	m.mu.Lock()
	defer m.mu.Unlock()

	var decode, nonDecode, eligible []*registry.Backend
	for _, b := range cands {
		n := m.nodeLocked(b)
		if n.mode == Decode {
			decode = append(decode, b)
			continue
		}
		nonDecode = append(nonDecode, b)
		if n.pending == 0 || n.pending+tokens <= m.cfg.NodeBudgetTokens {
			eligible = append(eligible, b)
		}
	}

	var chosen *registry.Backend
	switch {
	case len(eligible) > 0:
		chosen = m.pickLocked(eligible, pick)
	case m.canConvertLocked(len(cands), len(nonDecode), len(decode)):
		chosen = leastLoaded(decode)
		n := m.nodes[chosen.URL]
		n.mode = Draining
		m.conversions.Add(1)
		m.m.Conversions.Inc()
	case len(nonDecode) > 0:
		chosen = m.leastPendingLocked(nonDecode)
	default:
		return nil, ErrNoPrefillNode
	}

	n := m.nodes[chosen.URL]
	n.pending += tokens
	n.lastWork = m.cfg.Clock.Now()
	m.publishLocked(n)
	j := &Job{mgr: m, n: n, tokens: tokens, Backend: chosen, granted: make(chan struct{}), queued: true}
	n.queue = append(n.queue, j)
	m.publishQueuedLocked()
	return j, nil
}

func (m *Manager) pickLocked(eligible []*registry.Backend, pick Picker) *registry.Backend {
	if pick != nil {
		if b, err := pick(eligible); err == nil && b != nil {
			return b
		}
	}
	// The policy declined — a saturation signal on a node already running
	// prefill jobs is expected — so fall back to the emptiest queue rather
	// than refuse work the budget says fits.
	return m.leastPendingLocked(eligible)
}

func (m *Manager) leastPendingLocked(bs []*registry.Backend) *registry.Backend {
	best, ties := bs[0], 1
	for _, b := range bs[1:] {
		switch p, bp := m.nodes[b.URL].pending, m.nodes[best.URL].pending; {
		case p < bp:
			best, ties = b, 1
		case p == bp:
			ties++
			if rand.IntN(ties) == 0 {
				best = b
			}
		}
	}
	return best
}

func leastLoaded(bs []*registry.Backend) *registry.Backend {
	best, ties := bs[0], 1
	for _, b := range bs[1:] {
		switch l, bl := b.NormalizedLoad(), best.NormalizedLoad(); {
		case l < bl:
			best, ties = b, 1
		case l == bl:
			ties++
			if rand.IntN(ties) == 0 {
				best = b
			}
		}
	}
	return best
}

// canConvertLocked applies the cap: floor(fleet * fraction) nodes may be
// draining or prefill, and the last decode node is never converted.
func (m *Manager) canConvertLocked(fleet, nonDecode, decode int) bool {
	limit := min(int(float64(fleet)*m.cfg.MaxFleetFraction), fleet-1)
	return decode > 1 && nonDecode < limit
}

// WaitReady blocks until the job may run on its node; a job is queued from
// Reserve, so call it straight away. A draining node must
// first finish its in-flight requests — all decode, since no prefill job has
// been dispatched to it yet — and then becomes a prefill node. The job then
// takes a slot under the node's concurrency limit, queueing in arrival order
// behind the jobs ahead of it. The caller releases the slot with Done.
func (m *Manager) WaitReady(ctx context.Context, j *Job) error {
	for {
		m.mu.Lock()
		if j.n.mode == Draining && j.Backend.Inflight() == 0 {
			j.n.mode = Prefill
			m.publishLocked(j.n)
		}
		if j.n.mode != Draining {
			break
		}
		m.mu.Unlock()
		select {
		case <-m.cfg.Clock.After(drainPoll):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	// m.mu is held. The job joined its node's queue when it was reserved, so
	// the queue is in arrival order whatever the drain wait did to timing.
	for {
		m.grantLocked(j.n)
		if j.held {
			m.mu.Unlock()
			return nil
		}
		m.mu.Unlock()
		select {
		case <-j.granted:
		case <-m.cfg.Clock.After(drainPoll):
		case <-ctx.Done():
			m.mu.Lock()
			m.dequeueLocked(j)
			if j.held {
				// Granted in the same instant: hand the slot on.
				j.held = false
				j.n.running--
				m.grantLocked(j.n)
			}
			m.mu.Unlock()
			return ctx.Err()
		}
		m.mu.Lock()
	}
}

// grantLocked hands free slots to the head of the node's queue. Slots are the
// node's limit less its running jobs, and the node's own in-flight count — the
// one routing reads — must be under the limit too, so decode work still on a
// node counts against the same ceiling. The poll in WaitReady re-checks the
// latter, which changes without anyone telling the manager.
func (m *Manager) grantLocked(n *node) {
	limit := m.cfg.NodeConcurrency
	for len(n.queue) > 0 && n.mode != Draining {
		if limit > 0 && (int64(n.running) >= limit || n.b.Inflight() >= limit) {
			return
		}
		j := n.queue[0]
		n.queue = n.queue[1:]
		j.queued, j.held = false, true
		n.running++
		close(j.granted)
		m.publishQueuedLocked()
	}
}

func (m *Manager) dequeueLocked(j *Job) {
	if !j.queued {
		return
	}
	for i, q := range j.n.queue {
		if q == j {
			j.n.queue = append(j.n.queue[:i:i], j.n.queue[i+1:]...)
			break
		}
	}
	j.queued = false
	m.publishQueuedLocked()
}

func (m *Manager) publishQueuedLocked() {
	q := 0
	for _, n := range m.nodes {
		q += len(n.queue)
	}
	m.m.Queued.Set(float64(q))
}

// CountJobRetry records a prefill job retried after its node answered 429.
func (m *Manager) CountJobRetry() {
	m.jobRetries.Add(1)
	m.m.JobRetries.Inc()
}

// Sweep returns idle nodes to decode and forgets nodes that have left the
// pool. A node is idle when it has no pending prefill and none was finished
// within ReleaseIdle.
func (m *Manager) Sweep(known []*registry.Backend) {
	m.mu.Lock()
	defer m.mu.Unlock()
	live := make(map[string]bool, len(known))
	for _, b := range known {
		live[b.URL] = true
	}
	now := m.cfg.Clock.Now()
	for url, n := range m.nodes {
		if !live[url] {
			delete(m.nodes, url)
			m.m.DropNode(url)
			continue
		}
		if n.mode != Decode && n.pending == 0 && now.Sub(n.lastWork) >= m.cfg.ReleaseIdle {
			n.mode = Decode
			m.releases.Add(1)
			m.m.Releases.Inc()
			m.publishLocked(n)
		}
	}
	m.publishCountsLocked()
}

// Run sweeps on a ticker until ctx is done. known supplies the pool's current
// backends.
func (m *Manager) Run(ctx context.Context, known func() []*registry.Backend) {
	every := min(max(m.cfg.ReleaseIdle/4, 100*time.Millisecond), 5*time.Second)
	t := m.cfg.Clock.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C():
			m.Sweep(known())
		}
	}
}

func (m *Manager) publishLocked(n *node) {
	m.m.SetNode(n.b.URL, int(n.mode), n.pending)
	m.publishCountsLocked()
}

func (m *Manager) publishCountsLocked() {
	var counts [3]int
	for _, n := range m.nodes {
		counts[n.mode]++
	}
	m.m.SetModeCounts(counts)
}

// CountRejection records a large_prefill_rejected answer the router absorbed.
func (m *Manager) CountRejection() {
	m.rejections.Add(1)
	m.m.Rejections.Inc()
}

// CountJob records the outcome of a prefill job.
func (m *Manager) CountJob(ok bool) {
	if ok {
		m.jobsOK.Add(1)
		m.m.JobsOK.Inc()
		return
	}
	m.jobsFailed.Add(1)
	m.m.JobsFailed.Inc()
}

// FallbackNoNode records a rejected request served with the bypass because no
// prefill node could be used.
func (m *Manager) FallbackNoNode() {
	m.fallbackNoNode.Add(1)
	m.m.FallbackNoNode.Inc()
}

// FallbackJobFailed records a request sent for decode with the bypass because
// its prefill job did not succeed.
func (m *Manager) FallbackJobFailed() {
	m.fallbackJobFailed.Add(1)
	m.m.FallbackJobFail.Inc()
}

// NodeState is one node's view for /router-viz.
type NodeState struct {
	URL           string `json:"url"`
	Mode          string `json:"mode"`
	PendingTokens int64  `json:"pending_tokens"`
	RunningJobs   int    `json:"running_jobs"`
	QueuedJobs    int    `json:"queued_jobs"`
}

// State is the split's view for /router-viz.
type State struct {
	Enabled          bool        `json:"enabled"`
	NodeBudgetTokens int64       `json:"node_budget_tokens"`
	MaxFleetFraction float64     `json:"max_fleet_fraction"`
	Nodes            []NodeState `json:"nodes"`
	Rejections       uint64      `json:"rejections"`
	Jobs             uint64      `json:"jobs"`
	JobsFailed       uint64      `json:"jobs_failed"`
	JobRetries       uint64      `json:"job_retries"`
	Conversions      uint64      `json:"conversions"`
	Releases         uint64      `json:"releases"`
	BypassFallbacks  uint64      `json:"bypass_fallbacks"`
}

// State snapshots the modes, queues and counters, nodes sorted by URL.
func (m *Manager) State() State {
	m.mu.Lock()
	st := State{Enabled: true, NodeBudgetTokens: m.cfg.NodeBudgetTokens, MaxFleetFraction: m.cfg.MaxFleetFraction}
	for url, n := range m.nodes {
		st.Nodes = append(st.Nodes, NodeState{URL: url, Mode: n.mode.String(), PendingTokens: n.pending, RunningJobs: n.running, QueuedJobs: len(n.queue)})
	}
	m.mu.Unlock()
	sort.Slice(st.Nodes, func(i, j int) bool { return st.Nodes[i].URL < st.Nodes[j].URL })
	st.Rejections = m.rejections.Load()
	st.Jobs = m.jobsOK.Load() + m.jobsFailed.Load()
	st.JobsFailed = m.jobsFailed.Load()
	st.JobRetries = m.jobRetries.Load()
	st.Conversions = m.conversions.Load()
	st.Releases = m.releases.Load()
	st.BypassFallbacks = m.fallbackNoNode.Load() + m.fallbackJobFailed.Load()
	return st
}
