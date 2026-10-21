package benchmark

import "context"

// slotInfo names the line a request is drawn on and counted against.
//
// Router replay draws one line per SESSION: N is the session's 1-based run
// number (the order sessions started in) and Session is set. The goroutine that
// happens to run it is an implementation detail and appears nowhere.
//
// The other modes run a fixed pool of workers (--series N) that play sessions
// one after another, and there the line is the worker: N is its 1-based
// number, and workers 1..--hot-series-concurrency are the hot pool.
type slotInfo struct {
	N       int  // 1-based line number; 0 = unknown
	Hot     bool // member of the hot pool
	Session bool // N is a session run number, not a worker
}

type slotCtxKey struct{}

// withSlot tags ctx with the slot that drives everything beneath it.
func withSlot(ctx context.Context, n int, hot bool) context.Context {
	return context.WithValue(ctx, slotCtxKey{}, slotInfo{N: n, Hot: hot})
}

// withSession tags ctx with the run number of the session everything beneath
// it belongs to, keeping the hot-pool membership of the worker running it.
func withSession(ctx context.Context, run int) context.Context {
	return context.WithValue(ctx, slotCtxKey{}, slotInfo{N: run, Hot: slotOf(ctx).Hot, Session: true})
}

// slotOf returns the slot ctx was tagged with (zero value if none).
func slotOf(ctx context.Context) slotInfo {
	s, _ := ctx.Value(slotCtxKey{}).(slotInfo)
	return s
}

type progressCtxKey struct{}

// withProgress attaches a live request's progress tracker to ctx so the response
// consumers can report the first token and streamed deltas as they arrive.
func withProgress(ctx context.Context, p *liveProgress) context.Context {
	if p == nil {
		return ctx
	}
	return context.WithValue(ctx, progressCtxKey{}, p)
}

func progressFrom(ctx context.Context) *liveProgress {
	p, _ := ctx.Value(progressCtxKey{}).(*liveProgress)
	return p
}

// slotNum / sessionRun split a slotInfo into the record's two fields.
func slotNum(s slotInfo) int {
	if s.Session {
		return 0
	}
	return s.N
}

func sessionRun(s slotInfo) int {
	if s.Session {
		return s.N
	}
	return 0
}

// lineOf is the line a record is drawn on: its session run, else its worker,
// else (neither recorded) its series_num.
func lineOf(r requestDataRecord) rowKey {
	if r.SessionRun > 0 {
		return rowKey{session: r.SessionRun}
	}
	return rowKeyOf(r.Slot, r.SeriesNum)
}

// lineOfSlot is lineOf for a request still in flight.
func lineOfSlot(s slotInfo, series int) rowKey {
	if s.Session && s.N > 0 {
		return rowKey{session: s.N}
	}
	return rowKeyOf(s.N, series)
}
