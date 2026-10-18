package cli

import (
	"os"
	"runtime/debug"
)

// setSoftMemoryLimit caps the Go heap at frac of physical RAM unless
// GOMEMLIMIT is set. With the default GOGC=100 the heap may grow to twice the
// live data before a collection, so a command whose live data is a large share
// of RAM is killed by the OS while half its heap is garbage; under a limit the
// collector runs harder instead. Returns the limit set, or 0.
func setSoftMemoryLimit(frac float64) int64 {
	if os.Getenv("GOMEMLIMIT") != "" {
		return 0
	}
	total := physicalMemory()
	if total == 0 {
		return 0
	}
	limit := int64(float64(total) * frac)
	debug.SetMemoryLimit(limit)
	return limit
}
