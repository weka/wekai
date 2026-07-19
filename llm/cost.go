package llm

import (
	"regexp"
	"strings"
)

// Usage captures the four token buckets that determine the cost of one
// Anthropic-style request. Field names follow the wire shape so callers can
// translate from response.usage with no renaming.
//
// Per Anthropic billing: InputTokens already excludes cache reads and writes;
// CacheCreationInputTokens is billed at the cache-write rate; the four buckets
// sum to the total prompt + completion size that drove cost.
type Usage struct {
	InputTokens              int
	CacheReadInputTokens     int
	CacheCreationInputTokens int
	OutputTokens             int
}

// CalculateCost returns the USD cost for one request using the prices on info.
// Cost components:
//   - uncached input: InputTokens × InputCostPerMillion
//   - cache reads:    CacheReadInputTokens × CachedCostPerMillion
//   - cache writes:   CacheCreationInputTokens × CacheTokens5MinCostPerMillion
//   - output:         OutputTokens × OutputCostPerMillion
//
// Models that don't expose a price for a bucket (e.g., providers without a
// dedicated cache-write rate) simply yield 0 for that component.
func CalculateCost(info ModelInfo, usage Usage) float64 {
	const perMillion = 1_000_000.0
	cost := float64(usage.InputTokens) * info.InputCostPerMillion / perMillion
	cost += float64(usage.OutputTokens) * info.OutputCostPerMillion / perMillion
	cost += float64(usage.CacheReadInputTokens) * info.CachedCostPerMillion / perMillion
	cost += float64(usage.CacheCreationInputTokens) * info.CacheTokens5MinCostPerMillion / perMillion
	return cost
}

// modelIDDateSuffix matches a trailing "-YYYYMMDD" version stamp added by
// Anthropic to some model IDs (e.g., claude-haiku-4-5-20251001).
var modelIDDateSuffix = regexp.MustCompile(`-\d{8}$`)

// NormalizeModelIdentifier collapses provider-side variations of the same
// model into the canonical id used in the registry. Currently:
//   - strips a "[1m]" / "[256k]" / "[...]" trailing context-window flag
//   - strips a trailing "-YYYYMMDD" date stamp
//
// Pricing is shared across these variants (the [1m] context flag changes
// per-request pricing only when the prompt exceeds 200k tokens; we report
// the base price and let the caller flag oversize prompts separately).
func NormalizeModelIdentifier(id string) string {
	if i := strings.Index(id, "["); i >= 0 {
		id = id[:i]
	}
	id = strings.TrimSpace(id)
	id = modelIDDateSuffix.ReplaceAllString(id, "")
	return id
}
