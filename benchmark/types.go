package benchmark

import (
	"time"

	"github.com/weka/wekai/tools"
)

// EmbedBenchmarkRequest represents a benchmark request for a single model
type EmbedBenchmarkRequest struct {
	Model        string
	DocsDir      string
	DocsContent  string // Pre-loaded documentation content (used instead of DocsDir when set)
	Question     string
	NumRequests  int
	SleepBetween time.Duration
	NumSeries    int    // Number of series to run (each series gets unique GUID)
	NumCycles    int    // Number of times to repeat all series (tests long-term cache)
	Concurrency  int    // Number of series to run concurrently within each cycle (default: 1)
	SeriesGUID   string // GUID for this specific series (for cache isolation)
}

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
}

// ModelBenchmarkResult contains aggregated metrics for a single model
type ModelBenchmarkResult struct {
	ModelName        string // Full model string (e.g., "dynamic/...")
	ModelDisplayName string // Display name (alias if available, otherwise ModelName)
	Requests         []RequestMetrics

	// Aggregate TTFT metrics
	AvgTTFT time.Duration
	MinTTFT time.Duration
	MaxTTFT time.Duration

	// Aggregate response time metrics
	AvgResponseTime time.Duration
	MinResponseTime time.Duration
	MaxResponseTime time.Duration

	// Request statistics
	TotalRequests          int
	CachedRequests         int // Requests where CachedTokens > 0
	ImplicitCachedRequests int // Requests where TTFT is 50%+ faster than baseline
	FailedRequests         int

	// Wall-clock duration for this model's benchmark run
	WallDuration time.Duration
	Concurrency  int

	// Aggregated usage data
	TotalCost         float64
	TotalInputTokens  int
	TotalOutputTokens int
	TotalCachedTokens int
}

// BenchmarkResult contains results for all models benchmarked
type BenchmarkResult struct {
	Models        []ModelBenchmarkResult
	TotalDuration time.Duration
	DocsDir       string
	Question      string
	NumSeries     int  // Number of series per cycle
	NumCycles     int  // Number of cycles executed
	Concurrency   int  // Concurrency level used
	TimedOut      bool // True if benchmark was stopped due to timeout or signal
}
