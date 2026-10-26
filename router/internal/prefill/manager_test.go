package prefill

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/registry"
)

type fleet struct {
	clk *clock.Fake
	bs  []*registry.Backend
	m   *Manager
}

func newFleet(t *testing.T, n int, cfg Config) *fleet {
	t.Helper()
	reg := registry.New(registry.Options{})
	f := &fleet{clk: clock.NewFake(time.Time{})}
	for i := range n {
		b, err := reg.Add(registry.Spec{URL: fmt.Sprintf("http://n%d:8000", i), Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		f.bs = append(f.bs, b)
	}
	cfg.Clock = f.clk
	cfg.PoolName = t.Name()
	f.m = New(cfg)
	return f
}

func (f *fleet) modes() []Mode {
	out := make([]Mode, len(f.bs))
	for i, b := range f.bs {
		out[i] = f.m.ModeOf(b)
	}
	return out
}

func count(ms []Mode, want Mode) (n int) {
	for _, m := range ms {
		if m == want {
			n++
		}
	}
	return n
}

func TestFirstRejectionDrainsTheLeastLoadedDecodeNode(t *testing.T) {
	f := newFleet(t, 4, Config{})
	for i, b := range f.bs {
		if i != 2 {
			defer lease.Acquire(b).Release()
		}
	}
	j, err := f.m.Reserve(f.bs, 50_000, nil)
	if err != nil {
		t.Fatal(err)
	}
	if j.Backend != f.bs[2] {
		t.Errorf("job on %s, want the idle node %s", j.Backend.URL, f.bs[2].URL)
	}
	if got := f.m.ModeOf(j.Backend); got != Draining {
		t.Errorf("mode = %v, want draining until its decode work is gone", got)
	}
	if got := f.m.State().Conversions; got != 1 {
		t.Errorf("conversions = %d, want 1", got)
	}
}

func TestDrainWaitsForDecodeToFinishThenPrefill(t *testing.T) {
	f := newFleet(t, 2, Config{})
	busy := lease.Acquire(f.bs[0])
	// Make node 0 the only candidate to convert: node 1 is busier.
	l1a, l1b := lease.Acquire(f.bs[1]), lease.Acquire(f.bs[1])
	defer l1a.Release()
	defer l1b.Release()

	j, err := f.m.Reserve(f.bs, 1000, nil)
	if err != nil || j.Backend != f.bs[0] {
		t.Fatalf("Reserve = %v, %v; want node 0", j, err)
	}

	ready := make(chan error, 1)
	go func() { ready <- f.m.WaitReady(context.Background(), j) }()
	select {
	case err := <-ready:
		t.Fatalf("job released while the node still ran decode work (err=%v)", err)
	case <-time.After(60 * time.Millisecond):
	}
	if got := f.m.ModeOf(j.Backend); got != Draining {
		t.Errorf("mode = %v while decode is in flight, want draining", got)
	}

	busy.Release()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case err := <-ready:
			if err != nil {
				t.Fatal(err)
			}
			if got := f.m.ModeOf(j.Backend); got != Prefill {
				t.Errorf("mode = %v after draining, want prefill", got)
			}
			return
		case <-deadline:
			t.Fatal("WaitReady never returned after the node drained")
		default:
			f.clk.Advance(drainPoll)
			time.Sleep(time.Millisecond)
		}
	}
}

func TestWaitReadyStopsWhenTheCallerGoesAway(t *testing.T) {
	f := newFleet(t, 2, Config{})
	defer lease.Acquire(f.bs[0]).Release()
	defer lease.Acquire(f.bs[1]).Release()
	defer lease.Acquire(f.bs[1]).Release()
	j, err := f.m.Reserve(f.bs, 1000, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := f.m.WaitReady(ctx, j); !errors.Is(err, context.Canceled) {
		t.Errorf("WaitReady = %v, want context.Canceled", err)
	}
}

func TestFleetFractionCapAndLastDecodeNode(t *testing.T) {
	cases := []struct {
		nodes    int
		fraction float64
		want     int
	}{
		{4, 0.5, 2},
		{5, 0.5, 2},
		{3, 0.5, 1},
		{2, 0.5, 1},
		{2, 1.0, 1}, // a decode node always remains
		{8, 0.25, 2},
		{1, 0.5, 0},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("%dx%.2f", c.nodes, c.fraction), func(t *testing.T) {
			// A budget of 1 token makes every second job want a new node.
			f := newFleet(t, c.nodes, Config{MaxFleetFraction: c.fraction, NodeBudgetTokens: 1})
			var jobs []*Job
			for range c.nodes + 2 {
				j, err := f.m.Reserve(f.bs, 10, nil)
				if errors.Is(err, ErrNoPrefillNode) {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				jobs = append(jobs, j)
			}
			ms := f.modes()
			if got := count(ms, Draining) + count(ms, Prefill); got != c.want {
				t.Errorf("%d nodes left decode, want %d (modes %v)", got, c.want, ms)
			}
			if count(ms, Decode) == 0 {
				t.Error("no decode node remains")
			}
			if c.want == 0 && len(jobs) != 0 {
				t.Errorf("reserved %d jobs on a fleet that cannot spare a node", len(jobs))
			}
		})
	}
}

func TestNoPrefillNodeWhenNoneCanBeAdded(t *testing.T) {
	f := newFleet(t, 1, Config{})
	if _, err := f.m.Reserve(f.bs, 1000, nil); !errors.Is(err, ErrNoPrefillNode) {
		t.Errorf("Reserve on a one-node fleet = %v, want ErrNoPrefillNode", err)
	}
	if got := f.m.ModeOf(f.bs[0]); got != Decode {
		t.Errorf("the only node left decode: %v", got)
	}
}

func TestBudgetSpillsToAnotherNodeThenQueuesOnTheLeastPending(t *testing.T) {
	f := newFleet(t, 4, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.5})
	j1, _ := f.m.Reserve(f.bs, 80, nil)
	// 80+50 exceeds the 100 budget, so a second node is converted.
	j2, err := f.m.Reserve(f.bs, 50, nil)
	if err != nil {
		t.Fatal(err)
	}
	if j2.Backend == j1.Backend {
		t.Fatalf("job %d tokens landed on the node already holding 80 of a 100 budget", 50)
	}
	// The cap (2) is reached: further jobs queue on the least-pending node.
	j3, err := f.m.Reserve(f.bs, 60, nil)
	if err != nil {
		t.Fatal(err)
	}
	if j3.Backend != j2.Backend {
		t.Errorf("over-budget job went to %s, want the lighter prefill node %s",
			j3.Backend.URL, j2.Backend.URL)
	}
	pending := map[string]int64{}
	for _, n := range f.m.State().Nodes {
		pending[n.URL] = n.PendingTokens
	}
	if pending[j1.Backend.URL] != 80 || pending[j2.Backend.URL] != 110 {
		t.Errorf("pending = %v, want 80 and 110", pending)
	}
	j1.Done()
	j2.Done()
	j3.Done()
	for _, n := range f.m.State().Nodes {
		if n.PendingTokens != 0 {
			t.Errorf("%s still holds %d pending tokens after every job finished", n.URL, n.PendingTokens)
		}
	}
}

func TestJobLargerThanTheBudgetStillRunsOnAnEmptyNode(t *testing.T) {
	f := newFleet(t, 3, Config{NodeBudgetTokens: 100})
	j, err := f.m.Reserve(f.bs, 5_000, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := f.m.ModeOf(j.Backend); got != Draining {
		t.Errorf("mode = %v", got)
	}
}

func TestPickerChoosesAmongEligibleNodesOnly(t *testing.T) {
	f := newFleet(t, 6, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.5})
	a, _ := f.m.Reserve(f.bs, 90, nil) // node a is nearly full
	b, _ := f.m.Reserve(f.bs, 90, nil) // forces a second prefill node
	c, _ := f.m.Reserve(f.bs, 90, nil) // and a third
	if a.Backend == b.Backend || b.Backend == c.Backend || a.Backend == c.Backend {
		t.Fatal("setup: expected three distinct prefill nodes")
	}
	a.Done() // a has room again; b and c stay full

	var offered []*registry.Backend
	pick := func(eligible []*registry.Backend) (*registry.Backend, error) {
		offered = eligible
		return eligible[0], nil
	}
	j, err := f.m.Reserve(f.bs, 50, pick)
	if err != nil {
		t.Fatal(err)
	}
	if len(offered) != 1 || offered[0] != a.Backend || j.Backend != a.Backend {
		t.Errorf("picker offered %d nodes, job on %s; want only the node with room (%s)",
			len(offered), j.Backend.URL, a.Backend.URL)
	}

	// With several eligible, the policy's choice wins over load order.
	b.Done()
	c.Done()
	want := c.Backend
	j2, _ := f.m.Reserve(f.bs, 10, func(el []*registry.Backend) (*registry.Backend, error) {
		for _, e := range el {
			if e == want {
				return e, nil
			}
		}
		return nil, errors.New("holder not offered")
	})
	if j2.Backend != want {
		t.Errorf("job on %s, want the policy's pick %s", j2.Backend.URL, want.URL)
	}
}

func TestPickerErrorFallsBackToLeastPending(t *testing.T) {
	f := newFleet(t, 4, Config{NodeBudgetTokens: 1000})
	a, _ := f.m.Reserve(f.bs, 400, nil)
	b, _ := f.m.Reserve(f.bs, 900, nil) // over a's room, so a second node
	if a.Backend == b.Backend {
		t.Fatal("setup")
	}
	j, err := f.m.Reserve(f.bs, 50, func([]*registry.Backend) (*registry.Backend, error) {
		return nil, errors.New("saturated")
	})
	if err != nil || j.Backend != a.Backend {
		t.Errorf("Reserve = %v, %v; want the emptier node %s", j, err, a.Backend.URL)
	}
}

func TestDecodeRoutingExcludesDrainingAndPrefillNodes(t *testing.T) {
	f := newFleet(t, 4, Config{NodeBudgetTokens: 10})
	a, _ := f.m.Reserve(f.bs, 50, nil)
	b, _ := f.m.Reserve(f.bs, 50, nil)
	f.m.mu.Lock()
	f.m.nodes[b.Backend.URL].mode = Prefill
	f.m.mu.Unlock()

	got := f.m.DecodeNodes(f.bs)
	if len(got) != 2 {
		t.Fatalf("decode candidates = %d, want 2", len(got))
	}
	for _, c := range got {
		if c == a.Backend || c == b.Backend {
			t.Errorf("%s (mode %v) offered for decode", c.URL, f.m.ModeOf(c))
		}
	}
}

func TestDecodeNodesNeverEmptiesTheCandidateSet(t *testing.T) {
	f := newFleet(t, 3, Config{})
	j, _ := f.m.Reserve(f.bs, 10, nil)
	only := []*registry.Backend{j.Backend}
	if got := f.m.DecodeNodes(only); len(got) != 1 {
		t.Errorf("a pool whose decode nodes are all down served nothing: got %d candidates", len(got))
	}
}

func TestIdleNodeReturnsToDecode(t *testing.T) {
	f := newFleet(t, 4, Config{ReleaseIdle: time.Minute})
	j, _ := f.m.Reserve(f.bs, 10, nil)
	n := j.Backend

	f.clk.Advance(2 * time.Minute)
	f.m.Sweep(f.bs)
	if got := f.m.ModeOf(n); got == Decode {
		t.Fatal("a node with a job outstanding was released")
	}

	j.Done()
	f.clk.Advance(59 * time.Second)
	f.m.Sweep(f.bs)
	if got := f.m.ModeOf(n); got == Decode {
		t.Fatal("released before the idle period elapsed")
	}

	f.clk.Advance(2 * time.Second)
	f.m.Sweep(f.bs)
	if got := f.m.ModeOf(n); got != Decode {
		t.Errorf("mode = %v after %s idle, want decode", got, time.Minute)
	}
	if got := f.m.State().Releases; got != 1 {
		t.Errorf("releases = %d, want 1", got)
	}
}

func TestSweepForgetsDepartedNodes(t *testing.T) {
	f := newFleet(t, 3, Config{})
	j, _ := f.m.Reserve(f.bs, 10, nil)
	j.Done()
	var rest []*registry.Backend
	for _, b := range f.bs {
		if b != j.Backend {
			rest = append(rest, b)
		}
	}
	f.m.Sweep(rest)
	for _, n := range f.m.State().Nodes {
		if n.URL == j.Backend.URL {
			t.Errorf("%s left the pool but is still tracked", n.URL)
		}
	}
}

func TestDoneIsIdempotent(t *testing.T) {
	f := newFleet(t, 2, Config{})
	j, _ := f.m.Reserve(f.bs, 10, nil)
	j.Done()
	j.Done()
	for _, n := range f.m.State().Nodes {
		if n.PendingTokens < 0 {
			t.Errorf("pending went negative: %d", n.PendingTokens)
		}
	}
}

func TestJobsQueueBehindTheNodeConcurrencyLimitInOrder(t *testing.T) {
	const limit, total = 3, 9
	f := newFleet(t, 2, Config{NodeConcurrency: limit})
	var jobs []*Job
	for range total {
		j, err := f.m.Reserve(f.bs, 10, nil)
		if err != nil {
			t.Fatal(err)
		}
		if j.Backend != jobs0(jobs, j) {
			t.Fatalf("jobs spread over two nodes; the fleet should have one prefill node")
		}
		jobs = append(jobs, j)
	}

	var mu sync.Mutex
	running, peak := 0, 0
	started := make([]chan struct{}, total)
	release := make([]chan struct{}, total)
	var wg sync.WaitGroup
	for i, j := range jobs {
		started[i], release[i] = make(chan struct{}), make(chan struct{})
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := f.m.WaitReady(context.Background(), j); err != nil {
				t.Error(err)
				return
			}
			mu.Lock()
			running++
			peak = max(peak, running)
			mu.Unlock()
			close(started[i])
			<-release[i]
			mu.Lock()
			running--
			mu.Unlock()
			j.Done()
		}()
	}
	isStarted := func(i int) bool {
		select {
		case <-started[i]:
			return true
		default:
			return false
		}
	}
	for i := range limit {
		<-started[i]
	}
	if n := prefillNode(f); n.RunningJobs != limit || n.QueuedJobs != total-limit {
		t.Errorf("state = %+v, want %d running and %d queued", n, limit, total-limit)
	}
	// Each slot freed starts the longest-waiting job, and nothing else.
	for i := range total - limit {
		time.Sleep(20 * time.Millisecond)
		if isStarted(i + limit) {
			t.Fatalf("job %d started before a slot freed", i+limit)
		}
		close(release[i])
		<-started[i+limit]
		for k := i + limit + 1; k < total; k++ {
			if isStarted(k) {
				t.Fatalf("job %d jumped the queue", k)
			}
		}
	}
	for i := total - limit; i < total; i++ {
		close(release[i])
	}
	wg.Wait()
	if peak > limit {
		t.Errorf("%d jobs ran at once, limit %d", peak, limit)
	}
}

func jobs0(jobs []*Job, j *Job) *registry.Backend {
	if len(jobs) == 0 {
		return j.Backend
	}
	return jobs[0].Backend
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	for end := time.Now().Add(3 * time.Second); time.Now().Before(end); time.Sleep(2 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatal("condition not reached")
}

func TestQueuedJobLeavesTheQueueWhenItsCallerGoes(t *testing.T) {
	f := newFleet(t, 2, Config{NodeConcurrency: 1})
	a, _ := f.m.Reserve(f.bs, 10, nil)
	b, _ := f.m.Reserve(f.bs, 10, nil)
	if err := f.m.WaitReady(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- f.m.WaitReady(ctx, b) }()
	waitFor(t, func() bool { return prefillNode(f).QueuedJobs == 1 })
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("WaitReady = %v", err)
	}
	b.Done()
	a.Done()
	if n := prefillNode(f); n.QueuedJobs != 0 || n.RunningJobs != 0 {
		t.Errorf("node still holds work: %+v", n)
	}
}

func TestOtherInflightCountsAgainstTheSameLimit(t *testing.T) {
	f := newFleet(t, 2, Config{NodeConcurrency: 2})
	j, _ := f.m.Reserve(f.bs, 10, nil)
	if err := f.m.WaitReady(context.Background(), j); err != nil {
		t.Fatal(err)
	}
	// Two other requests already fill the node's in-flight.
	l1, l2 := lease.Acquire(j.Backend), lease.Acquire(j.Backend)
	defer l2.Release()
	j2, _ := f.m.Reserve(f.bs, 10, nil)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(2 * time.Millisecond):
				f.clk.Advance(drainPoll)
			}
		}
	}()
	done := make(chan error, 1)
	go func() { done <- f.m.WaitReady(context.Background(), j2) }()
	select {
	case err := <-done:
		t.Fatalf("job ran with the node's in-flight at its limit: %v", err)
	case <-time.After(80 * time.Millisecond):
	}
	l1.Release()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("job never ran after the node's in-flight dropped")
	}
}

func prefillNode(f *fleet) NodeState {
	for _, n := range f.m.State().Nodes {
		if n.Mode != Decode.String() {
			return n
		}
	}
	return NodeState{}
}
