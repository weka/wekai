package benchmark

import (
	"time"

	"github.com/weka/wekai/tools"
)

// RequestMetrics captures metrics for a single LLM request
type RequestMetrics struct {
	// Skipped: request filtered out pre-send (--limit-context); counted as
	// neither completed nor error by the auto aggregator.
	Skipped    bool
	RequestNum int    // Request number (1-based) within a series
	SeriesNum  int    // Series number (1-based) within a cycle
	CycleNum   int    // Cycle number (1-based)
	SeriesGUID string // GUID for the series this request belongs to
	// TimeToFirstToken is what the CALLER waited: every 429 backed off plus the
	// attempt that finally answered. There is no per-attempt variant, on
	// purpose.
	//
	// Retries are a client transport detail and must not become a dimension a
	// reader has to reason about. A caller who waited twelve seconds waited
	// twelve seconds, and "the last attempt was fast" is not a fact any consumer
	// of a benchmark needs — including the cache-hit heuristic, which will
	// therefore call more requests misses on a saturated fleet. That is Anton's
	// call and it is the right one: a fleet that made the client queue did not
	// serve that request from cache in any sense the caller experienced.
	TimeToFirstToken  time.Duration
	TotalResponseTime time.Duration // Total time for request
	UsageData         tools.ExecutionUsageData
	Error             error
	Response          string  // The actual response content
	IsImplicitCached  bool    // True if TTFT is 50%+ faster than baseline for this series
	IsEmpty           bool    // True if the response body was empty (potential server error)
	LocalCacheRatio   float64 // estimated fraction of this request's prompt that was repeated (in [0,1])
	CachedPrompt      string  // Full system prompt text (only populated on error/empty for diagnostics)
	Question          string  // The user question sent with the request (only populated on error/empty)
	RawResponseTail   string  // raw SSE tail (last bytes); only populated on error/empty for diagnostics
	// Retries429 / RetryWait record the client-side backoff a request spent
	// waiting out 429s before it was served (or gave up). TimeToFirstToken
	// measures the attempt that actually ran, so it stays a server-latency
	// number; TotalResponseTime INCLUDES RetryWait, because that is how long
	// the caller really waited.
	Retries429 int
	RetryWait  time.Duration

	// UUID validation (router-replay --verify only). All nil/zero
	// when the feature is off (default) or on the synthetic path, which never
	// populates these.
	ConvIdx       int      // session index within cfg.replayUUIDSets (== seriesNum-1)
	ExpectedUUIDs []string // this session's own N-UUID list, in order
	UUIDFound     []bool   // parallel to ExpectedUUIDs: whether each was found in Response or thinking
	// LeakChecked records that this request's response WAS scanned for
	// contamination, which is a different population from the recite-scored
	// requests below: a zero contamination count is only meaningful against
	// the number of responses actually looked at.
	LeakChecked bool
	// ReciteBudgetShort marks a request whose captured output budget could not
	// carry even one id, so presence was never asked and must not be scored.
	// Counted and reported separately: an unanswerable question recorded as a
	// miss would read as a coherency failure.
	ReciteBudgetShort bool
	LeakedUUIDs       []string // "uuid(series=N)" entries for any OTHER session's UUID found here
	ExactMatch        bool     // first line of Response is exactly the ordered, comma-joined ExpectedUUIDs list (output conformity)
}
