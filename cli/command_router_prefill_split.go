package cli

import (
	"fmt"
	"sort"

	"github.com/weka/wekai/benchmark"
	"github.com/weka/wekai/kvcache"
)

// prefillSensitivityThresholds are the fixed reference points for the
// sensitivity table (item 5 of the --prefill-split ask): it says whether the
// chosen --prefill-min-missing-blocks sits on a cliff or a plateau.
var prefillSensitivityThresholds = []int{1, 2, 4, 8, 16, 32}

// prefillSplitOutput bundles everything --prefill-split adds to the report.
// nil means the flag is off; outputText/outputJSON must then be byte-
// identical to the plain cache-ratio report.
type prefillSplitOutput struct {
	MinMissingBlocks int
	CacheMaxTokens   int64 // 0 = unbounded (the default lower-bound model)

	// Headline/PerModel are both AggregatePrefillSplit at MinMissingBlocks —
	// PerModel keyed by model, Headline over every model combined.
	Headline benchmark.PrefillSplitReport
	PerModel map[string]benchmark.PrefillSplitReport

	// Sensitivity is one AggregatePrefillSplit per prefillSensitivityThresholds
	// entry (same order), over every model combined.
	Sensitivity []benchmark.PrefillSplitReport

	// MissingBlocks is every request's missing-block count, all models
	// combined, in simulation order — source data for percentiles/histogram.
	MissingBlocks []int
}

// computePrefillSplit runs the router-block simulation once per model (each
// model's KV cache is a separate partition — see affinity.Policy.modelKey —
// so a shared trie across models would be unfaithful) and derives the
// headline, per-model, and sensitivity numbers from that single pass.
func (c *RouterAnalyzeReplayCommand) computePrefillSplit(models []string, byModel map[string][]benchmark.RouterReplayRequest) (*prefillSplitOutput, error) {
	docs, err := benchmark.GetEmbeddedDocs(0)
	if err != nil {
		return nil, fmt.Errorf("load embedded docs for --prefill-split content synthesis: %w", err)
	}

	cacheCfg := kvcache.Config{}
	if c.PrefillCacheMaxTokens > 0 {
		cacheCfg.MaxTokens = c.PrefillCacheMaxTokens
	}
	// Shared across every model: content synthesis depends only on a block's
	// own hash, not its model, so reuse is still valid and a system prompt
	// shared by two models is only synthesized once. Bounded (LRU) so memory
	// doesn't scale with input size — see PrefillUnitCache.
	unitCache := benchmark.NewPrefillUnitCache()

	out := &prefillSplitOutput{
		MinMissingBlocks: c.PrefillMinMissingBlocks,
		CacheMaxTokens:   c.PrefillCacheMaxTokens,
		PerModel:         make(map[string]benchmark.PrefillSplitReport, len(models)),
	}

	var allStats []benchmark.PrefillBlockStat
	for _, m := range models {
		stats := benchmark.SimulatePrefillBlocks(byModel[m], docs, cacheCfg, unitCache)
		allStats = append(allStats, stats...)
		out.PerModel[m] = benchmark.AggregatePrefillSplit(stats, c.PrefillMinMissingBlocks)
	}
	out.Headline = benchmark.AggregatePrefillSplit(allStats, c.PrefillMinMissingBlocks)
	for _, th := range prefillSensitivityThresholds {
		out.Sensitivity = append(out.Sensitivity, benchmark.AggregatePrefillSplit(allStats, th))
	}
	out.MissingBlocks = make([]int, len(allStats))
	for i, s := range allStats {
		out.MissingBlocks[i] = s.MissingBlocks()
	}
	return out, nil
}

// ---- percentiles + histogram ----

type percentileStat struct {
	Label string
	Value int
}

// prefillPercentiles returns p50/p75/p90/p95/p99/max over vals. Nil if vals
// is empty.
func prefillPercentiles(vals []int) []percentileStat {
	if len(vals) == 0 {
		return nil
	}
	sorted := append([]int(nil), vals...)
	sort.Ints(sorted)
	pick := func(p float64) int {
		idx := int(p * float64(len(sorted)-1))
		if idx < 0 {
			idx = 0
		} else if idx >= len(sorted) {
			idx = len(sorted) - 1
		}
		return sorted[idx]
	}
	return []percentileStat{
		{"p50", pick(0.50)},
		{"p75", pick(0.75)},
		{"p90", pick(0.90)},
		{"p95", pick(0.95)},
		{"p99", pick(0.99)},
		{"max", sorted[len(sorted)-1]},
	}
}

type histBucket struct {
	Label string
	Count int
}

// prefillHistogram buckets missing-block counts against the same reference
// points as the sensitivity table (1, 2, 4, 8, 16, 32) so the two sections
// read together.
func prefillHistogram(vals []int) []histBucket {
	bounds := []int{0, 1, 2, 3, 4, 8, 16, 32, 64}
	labels := []string{"0", "1", "2", "3", "4-7", "8-15", "16-31", "32-63", "64+"}
	counts := make([]int, len(labels))
	for _, v := range vals {
		idx := len(labels) - 1
		for i := 0; i < len(bounds)-1; i++ {
			if v < bounds[i+1] {
				idx = i
				break
			}
		}
		counts[idx]++
	}
	out := make([]histBucket, len(labels))
	for i, l := range labels {
		out[i] = histBucket{Label: l, Count: counts[i]}
	}
	return out
}

// ---- text output ----

func (c *RouterAnalyzeReplayCommand) printPrefillSplitText(p *prefillSplitOutput) {
	fmt.Println("  Prefill/decode split estimate (--prefill-split)")
	fmt.Println()

	cacheDesc := "unbounded (infinite retention)"
	if p.CacheMaxTokens > 0 {
		cacheDesc = fmt.Sprintf("bounded at %s estimated tokens", formatNumber(int(p.CacheMaxTokens)))
	}
	fmt.Printf("    Model:      single fleet-wide cache, %s — \"perfect affinity\".\n", cacheDesc)
	fmt.Println("                LOWER BOUND: the router predicts PER-BACKEND residency; a replay")
	fmt.Println("                file has no fleet. A real fleet splits prefixes across nodes, so")
	fmt.Println("                real missing-block counts run higher than this.")
	fmt.Printf("    Block size: ~256 estimated tokens (%d bytes, kvcache.DefaultChunkBytes)\n", kvcache.DefaultChunkBytes)
	fmt.Printf("    Threshold:  missing blocks > %d (--prefill-min-missing-blocks)\n", p.MinMissingBlocks)
	fmt.Println()

	h := p.Headline
	fmt.Println("    Headline (all models):")
	fmt.Printf("      Requests to prefill:             %s / %s  (%s%%)\n",
		formatNumber(h.PrefillRequests), formatNumber(h.Requests), formatPct(h.RequestShare()))
	fmt.Printf("      Input tokens to prefill:         %s / %s  (%s%%)  [whole prompt, sent once each]\n",
		formatNumber(int(h.PrefillInputTokens)), formatNumber(int(h.TotalInputTokens)), formatPct(h.InputTokenShare()))
	fmt.Printf("      New/uncached tokens to prefill:  %s / %s  (%s%%)  [estimate; GPU work actually moved off decode]\n",
		formatNumber(int(h.PrefillMissingTokens)), formatNumber(int(h.TotalInputTokens)), formatPct(h.MissingTokenShare()))
	fmt.Println()

	if len(p.PerModel) > 1 {
		fmt.Println("    Per-model:")
		var names []string
		for m := range p.PerModel {
			names = append(names, m)
		}
		sort.Strings(names)
		for _, m := range names {
			r := p.PerModel[m]
			label := m
			if label == "" {
				label = "(unknown)"
			}
			fmt.Printf("      %-24s requests %s/%s (%s%%)   input tokens %s/%s (%s%%)\n",
				label, formatNumber(r.PrefillRequests), formatNumber(r.Requests), formatPct(r.RequestShare()),
				formatNumber(int(r.PrefillInputTokens)), formatNumber(int(r.TotalInputTokens)), formatPct(r.InputTokenShare()))
		}
		fmt.Println()
	}

	if len(p.MissingBlocks) > 0 {
		fmt.Println("    Missing-blocks distribution (all models):")
		fmt.Print("     ")
		for _, ps := range prefillPercentiles(p.MissingBlocks) {
			fmt.Printf("  %s=%s", ps.Label, formatNumber(ps.Value))
		}
		fmt.Println()
		fmt.Println()

		fmt.Println("    Missing-blocks histogram:")
		total := len(p.MissingBlocks)
		for _, b := range prefillHistogram(p.MissingBlocks) {
			pct := 0.0
			if total > 0 {
				pct = 100.0 * float64(b.Count) / float64(total)
			}
			fmt.Printf("      %-8s %14s  (%s%%)\n", b.Label, formatNumber(b.Count), formatPct(pct/100))
		}
		fmt.Println()
	}

	fmt.Println("    Sensitivity to threshold (all models):")
	fmt.Printf("      %10s %16s %10s %20s %10s %20s %10s\n",
		"missing>", "requests", "req%", "input tokens", "tok%", "new tokens", "new%")
	for _, r := range p.Sensitivity {
		fmt.Printf("      %10d %16s %9s%% %20s %9s%% %20s %9s%%\n",
			r.MinMissingBlocks,
			formatNumber(r.PrefillRequests), formatPct(r.RequestShare()),
			formatNumber(int(r.PrefillInputTokens)), formatPct(r.InputTokenShare()),
			formatNumber(int(r.PrefillMissingTokens)), formatPct(r.MissingTokenShare()))
	}
	fmt.Println()
}

// formatPct renders a [0,1] fraction as a percentage with 4 decimal places —
// coarse-precision "%.2f" rounds every one of these headline numbers to
// 0.00% on workloads where the whole point is telling 0.001% from 0.01%.
func formatPct(frac float64) string {
	return fmt.Sprintf("%.4f", frac*100)
}

// ---- JSON output ----

type jsonPrefillReport struct {
	MinMissingBlocks int `json:"min_missing_blocks"`

	Requests        int     `json:"requests"`
	PrefillRequests int     `json:"prefill_requests"`
	RequestShare    float64 `json:"request_share"`

	TotalInputTokens   int64   `json:"total_input_tokens"`
	PrefillInputTokens int64   `json:"prefill_input_tokens"`
	InputTokenShare    float64 `json:"input_token_share"`

	TotalMissingTokensEstimate   int64   `json:"total_missing_tokens_estimate"`
	PrefillMissingTokensEstimate int64   `json:"prefill_missing_tokens_estimate"`
	MissingTokenShareEstimate    float64 `json:"missing_token_share_estimate"`
}

func toJSONPrefillReport(r benchmark.PrefillSplitReport) jsonPrefillReport {
	return jsonPrefillReport{
		MinMissingBlocks:             r.MinMissingBlocks,
		Requests:                     r.Requests,
		PrefillRequests:              r.PrefillRequests,
		RequestShare:                 r.RequestShare(),
		TotalInputTokens:             r.TotalInputTokens,
		PrefillInputTokens:           r.PrefillInputTokens,
		InputTokenShare:              r.InputTokenShare(),
		TotalMissingTokensEstimate:   r.TotalMissingTokens,
		PrefillMissingTokensEstimate: r.PrefillMissingTokens,
		MissingTokenShareEstimate:    r.MissingTokenShare(),
	}
}

type jsonHistBucket struct {
	Label string `json:"label"`
	Count int    `json:"count"`
}

type jsonPrefillSplit struct {
	// Model describes the assumptions behind every number below: see
	// SimulatePrefillBlocks's doc. Always a lower bound.
	Model            string `json:"model"`
	MinMissingBlocks int    `json:"min_missing_blocks"`
	CacheMaxTokens   int64  `json:"cache_max_tokens"` // 0 = unbounded
	BlockChunkBytes  int    `json:"block_chunk_bytes"`

	Headline jsonPrefillReport            `json:"headline"`
	PerModel map[string]jsonPrefillReport `json:"per_model,omitempty"`

	MissingBlocksPercentiles map[string]int   `json:"missing_blocks_percentiles,omitempty"`
	MissingBlocksHistogram   []jsonHistBucket `json:"missing_blocks_histogram,omitempty"`

	Sensitivity []jsonPrefillReport `json:"sensitivity"`
}

func toJSONPrefillSplit(p *prefillSplitOutput) *jsonPrefillSplit {
	out := &jsonPrefillSplit{
		Model: "single fleet-wide cache, perfect affinity — LOWER BOUND " +
			"(the router predicts per-backend residency; a real fleet splits prefixes " +
			"across nodes, so real missing-block counts run higher than this)",
		MinMissingBlocks: p.MinMissingBlocks,
		CacheMaxTokens:   p.CacheMaxTokens,
		BlockChunkBytes:  kvcache.DefaultChunkBytes,
		Headline:         toJSONPrefillReport(p.Headline),
	}
	if len(p.PerModel) > 0 {
		out.PerModel = make(map[string]jsonPrefillReport, len(p.PerModel))
		for m, r := range p.PerModel {
			out.PerModel[m] = toJSONPrefillReport(r)
		}
	}
	if len(p.MissingBlocks) > 0 {
		out.MissingBlocksPercentiles = map[string]int{}
		for _, ps := range prefillPercentiles(p.MissingBlocks) {
			out.MissingBlocksPercentiles[ps.Label] = ps.Value
		}
		for _, b := range prefillHistogram(p.MissingBlocks) {
			out.MissingBlocksHistogram = append(out.MissingBlocksHistogram, jsonHistBucket{Label: b.Label, Count: b.Count})
		}
	}
	for _, r := range p.Sensitivity {
		out.Sensitivity = append(out.Sensitivity, toJSONPrefillReport(r))
	}
	return out
}
