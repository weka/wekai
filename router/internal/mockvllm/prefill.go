package mockvllm

import (
	"context"
	"math"
	"time"

	"github.com/weka/wekai/router/internal/clock"
)

// prefillEpsilon absorbs float64 accumulation error in the settle loop —
// remaining work this close to zero is done, not "still owed 40ns worth."
const prefillEpsilon = 1e-9

// prefillJob is one request's outstanding prefill work, tracked in SECONDS
// of solo-rate GPU time still owed. Touched ONLY by prefillScheduler.run —
// no other goroutine ever reads or writes its fields — so it needs no
// synchronization of its own.
type prefillJob struct {
	remaining float64
	done      chan struct{}
}

// prefillScheduler models ONE mock vLLM instance's prefill throughput as a
// shared, egalitarian processor-sharing (PS) resource: when N requests are
// concurrently prefilling, the instance's aggregate token rate is split N
// ways, so each request's own prefill drains at 1/N of the solo rate.
//
// This is what makes --cold-input-tps/--cached-input-tps INSTANCE-aggregate
// rates rather than independent per-request rates. Real vLLM's prefill
// throughput is a genuinely shared GPU resource; modeling requests as
// independent was the mock's biggest remaining fidelity gap — with
// per-request rates, mock warm/cached share converged (46/48) where a real
// fleet under load spreads (51/35), because CONTENTION, not just per-token
// cost, is what makes deep cold turns expensive fleet-wide under
// concurrency. Queueing for the shared resource is the missing mechanism,
// not a rate-tuning problem.
//
// Decode (output) throughput is deliberately NOT modeled this way — see
// Engine.DecodeDuration / Engine.OutputTokenInterval: real vLLM's
// continuous batching keeps each in-flight request's own decode rate
// roughly constant until the batch itself saturates, so treating output as
// per-request remains a reasonable approximation.
//
// Implementation: a single long-lived goroutine per instance (run, below)
// owns all mutable scheduling state and is the only thing that ever reads
// or writes it — "share memory by communicating," so no mutex guards it. A
// submitting request's goroutine sends a join, then blocks on its own done
// channel (or asks to leave, on cancellation). The scheduler is
// event-driven, not polled: state is settled (every active job's remaining
// work brought up to date for elapsed real time under the job count that
// was active throughout that interval) only at a join, a leave, or a
// completion, and exactly one timer is armed at a time, for the next job to
// complete under the CURRENT job count — team-lead's spec calls this
// "recomputed timer," as opposed to a condition-variable design.
type prefillScheduler struct {
	clk clock.Clock

	joinCh  chan *prefillJob
	leaveCh chan *prefillJob
	stopCh  chan struct{}
}

// newPrefillScheduler starts the scheduler's background goroutine. clk nil
// defaults to clock.Real{}.
func newPrefillScheduler(clk clock.Clock) *prefillScheduler {
	if clk == nil {
		clk = clock.Real{}
	}
	s := &prefillScheduler{
		clk:     clk,
		joinCh:  make(chan *prefillJob),
		leaveCh: make(chan *prefillJob),
		stopCh:  make(chan struct{}),
	}
	go s.run()
	return s
}

// Close stops the background goroutine. Only needed by callers that create
// many short-lived Engines (tests); router/cmd/mock-vllm's process-lifetime
// instances never need to call it — the process exit takes the goroutine
// with it, same as any other server loop.
func (s *prefillScheduler) Close() { close(s.stopCh) }

// submit blocks until workSeconds of solo-rate prefill work has drained
// under processor sharing with every other request concurrently prefilling
// on this instance, or ctx is done first. workSeconds <= 0 returns
// immediately (nothing to schedule), same as an unset/zero-rate config
// making a term instant elsewhere in this package.
func (s *prefillScheduler) submit(ctx context.Context, workSeconds float64) bool {
	if workSeconds <= 0 {
		select {
		case <-ctx.Done():
			return false
		default:
			return true
		}
	}

	job := &prefillJob{remaining: workSeconds, done: make(chan struct{})}
	select {
	case s.joinCh <- job:
	case <-ctx.Done():
		return false
	}

	select {
	case <-job.done:
		return true
	case <-ctx.Done():
		// Ask the scheduler to remove us. This round-trips through run's
		// single event loop, so by the time the send below completes, any
		// completion that was already in flight for this job has already
		// happened (run processes events strictly in order) — checking
		// job.done immediately afterward is race-free, not a second racy
		// select against the same two channels.
		select {
		case s.leaveCh <- job:
		case <-job.done:
			return true
		}
		select {
		case <-job.done:
			return true // completed before the leave was processed
		default:
			return false // genuinely removed while still active
		}
	}
}

// run is the scheduler's single event loop. active/lastUpdate/wake are
// touched ONLY here.
func (s *prefillScheduler) run() {
	active := map[*prefillJob]struct{}{}
	lastUpdate := s.clk.Now()
	var wake <-chan time.Time

	// settle brings every active job's remaining work up to date for
	// elapsed real time since lastUpdate, at the job count that was active
	// THROUGHOUT that interval (i.e. before whatever join/leave/completion
	// triggered this call) — the defining property of processor sharing:
	// each of N active jobs drains at 1/N of the instance's full rate, so N
	// concurrent jobs together drain at exactly the instance's aggregate
	// rate regardless of N.
	settle := func(now time.Time) {
		n := len(active)
		elapsed := now.Sub(lastUpdate)
		lastUpdate = now
		if n == 0 || elapsed <= 0 {
			return
		}
		share := elapsed.Seconds() / float64(n)
		for j := range active {
			j.remaining -= share
		}
	}

	// reschedule arms a single timer for the NEXT job to complete under the
	// current (just-settled) state: the job with the least remaining work,
	// which — again by the PS property — fully drains after
	// (its remaining) * (current job count) more seconds of real time (the
	// other N-1 jobs are each also draining at 1/N, so nothing changes the
	// schedule until either this timer fires or the active set changes).
	reschedule := func() {
		if len(active) == 0 {
			wake = nil
			return
		}
		minRemaining := math.Inf(1)
		for j := range active {
			if j.remaining < minRemaining {
				minRemaining = j.remaining
			}
		}
		if minRemaining < 0 {
			minRemaining = 0
		}
		n := float64(len(active))
		wake = s.clk.After(time.Duration(minRemaining * n * float64(time.Second)))
	}

	for {
		select {
		case <-s.stopCh:
			return
		case j := <-s.joinCh:
			settle(s.clk.Now())
			active[j] = struct{}{}
			reschedule()
		case j := <-s.leaveCh:
			if _, ok := active[j]; ok {
				settle(s.clk.Now())
				delete(active, j)
				reschedule()
			}
			// else: already completed and removed by the branch below,
			// racing this leave request — nothing to do, and job.done is
			// already closed for submit's goroutine to observe.
		case now := <-wake:
			settle(now)
			for j := range active {
				if j.remaining <= prefillEpsilon {
					delete(active, j)
					close(j.done)
				}
			}
			reschedule()
		}
	}
}
