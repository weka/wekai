package benchmark

import (
	"time"

	"github.com/weka/wekai/tools"
)

// RequestMetrics captures metrics for a single LLM request
type RequestMetrics struct {
	RequestNum        int           // Request number (1-based) within a series
	SeriesNum         int           // Series number (1-based) within a cycle
	CycleNum          int           // Cycle number (1-based)
	SeriesGUID        string        // GUID for the series this request belongs to
	TimeToFirstToken  time.Duration // Time until first content callback
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

	// UUID validation (dataset-replay --replay-inject-uuids only). All nil/zero
	// when the feature is off (default) or on the router-replay/synthetic paths,
	// which never populate these.
	ConvIdx       int      // conversation index within cfg.replayConversations / cfg.replayUUIDSets
	ExpectedUUIDs []string // this conversation's in-scope ref-id UUIDs at this turn (defensive copy)
	UUIDFound     []bool   // parallel to ExpectedUUIDs: whether each was found in Response or thinking
	LeakedUUIDs   []string // "uuid(series=N)" entries for any OTHER conversation's UUID found here
}
