package benchmark

import "context"

// A slot is one of the benchmark's concurrency lanes: the worker goroutine that
// replays sessions one after another (--series N runs N of them). It is the unit
// the timeline draws as one line, as opposed to series_num, which for replay is
// the index of the SESSION a slot happens to be running and keeps growing as
// slots move on to the next session. Slots 1..--hot-series-concurrency are the
// hot pool.
type slotInfo struct {
	N   int  // 1-based slot number; 0 = unknown
	Hot bool // member of the hot pool
}

type slotCtxKey struct{}

// withSlot tags ctx with the slot that drives everything beneath it.
func withSlot(ctx context.Context, n int, hot bool) context.Context {
	return context.WithValue(ctx, slotCtxKey{}, slotInfo{N: n, Hot: hot})
}

// slotOf returns the slot ctx was tagged with (zero value if none).
func slotOf(ctx context.Context) slotInfo {
	s, _ := ctx.Value(slotCtxKey{}).(slotInfo)
	return s
}
