package cli

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/weka/wekai/benchmark"
	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
)

// EvalSimpleToolCommand implements the eval simple-tool subcommand
type EvalSimpleToolCommand struct {
	*EvalSimpleToolOptions
}

func (c *EvalSimpleToolCommand) Execute(args []string) error {
	if err := runPreExecute(context.Background()); err != nil {
		return err
	}

	// Validate model
	if err := llm.ValidateModel(c.Model); err != nil {
		return fmt.Errorf("invalid model %s: %w", c.Model, err)
	}

	// Validate global model override if provided
	if config.Config.GlobalModelOverride != "" {
		return fmt.Errorf("--global-model-override cannot be used with eval command; specify model via --model flag")
	}

	// Initialize config
	ctx := context.Background()
	ctx, shutdown, err := config.Init(ctx, "eval_simple_tool")
	if err != nil {
		return fmt.Errorf("failed to initialize config: %w", err)
	}
	defer shutdown(ctx)

	// Run the tool chain evaluation
	result, err := benchmark.RunToolChainEvaluation(ctx, c.Model, c.TargetCount)
	if err != nil && result == nil {
		return fmt.Errorf("evaluation failed: %w", err)
	}

	// Print results
	fmt.Printf("\n=== TOOL CHAIN EVALUATION RESULTS ===\n")
	fmt.Printf("Model: %s\n", result.Model)
	fmt.Printf("Target value: %d\n", result.TargetValue)
	fmt.Printf("Expected tool calls: %d\n", result.ExpectedCalls)
	fmt.Printf("Actual tool calls: %d\n", result.ToolInvocations)
	fmt.Printf("Final value: %d\n", result.FinalValue)
	fmt.Printf("Success: %v\n", result.Success)
	if result.ErrorMessage != "" {
		fmt.Printf("Error: %s\n", result.ErrorMessage)
	}
	fmt.Printf("\nUsage Statistics:\n")
	fmt.Printf("  Requests: %d\n", result.RequestCount)
	fmt.Printf("  Input tokens: %d\n", result.InputTokens)
	fmt.Printf("  Output tokens: %d\n", result.OutputTokens)
	fmt.Printf("  Reasoning tokens: %d\n", result.ReasoningTokens)
	fmt.Printf("  Cached tokens: %d\n", result.CachedTokens)
	fmt.Printf("  Total cost: $%.6f\n", result.TotalCost)
	fmt.Printf("\nFinal response: %s\n", result.FinalResponse)

	// Return error if not successful
	if !result.Success {
		return fmt.Errorf("evaluation did not succeed")
	}

	return nil
}

// EvalCacheCoherencyCommand implements the eval cache-coherency-garbage-clean subcommand
type EvalCacheCoherencyCommand struct {
	*EvalCacheCoherencyOptions
}

func (c *EvalCacheCoherencyCommand) Execute(args []string) error {
	if err := runPreExecute(context.Background()); err != nil {
		return err
	}

	// Validate model
	if err := llm.ValidateModel(c.Model); err != nil {
		return fmt.Errorf("invalid model %s: %w", c.Model, err)
	}

	if config.Config.GlobalModelOverride != "" {
		return fmt.Errorf("--global-model-override cannot be used with eval command; specify model via --model flag")
	}

	// Resolve --garbage-characters / deprecated --garbage-tokens precedence.
	garbageChars := resolveGarbageChars(c.GarbageCharacters, c.GarbageTokens, os.Stderr)

	ctx := context.Background()
	ctx, shutdown, err := config.Init(ctx, "eval_cache_coherency")
	if err != nil {
		return fmt.Errorf("failed to initialize config: %w", err)
	}
	defer shutdown(ctx)

	adv := benchmark.AdversarialOptions{
		AbortFraction: c.AbortFraction,
		AbortDelayMs:  c.AbortDelayMs,
		ResetEveryN:   c.ResetEveryN,
	}
	result, err := benchmark.RunCacheCoherencyEval(ctx, c.Model, c.Series, c.Concurrency, garbageChars, c.Seed, c.Total, c.SharedPrefixPerSeries, c.MaxOutputMultiplier, adv)
	if err != nil {
		return fmt.Errorf("evaluation failed: %w", err)
	}

	// Test 1 (UUID correctness) is now scored PER UUID STAMP, not per request: each
	// request expects StampsPerSeries UUIDs back, so the denominator is
	// TotalRequests * StampsPerSeries. Test 2 (output conformity) stays per-request
	// (the whole response must be exactly the comma-joined UUID list).
	//
	// Requests intentionally aborted via --abort-fraction (r.Aborted) have no usable
	// response and are excluded entirely from both — they create the corruption
	// *opportunity*, not something to score. nonAbortedRequests is Test 2's
	// denominator; it equals result.TotalRequests whenever AbortedCount is 0 (the
	// --abort-fraction=0 default), so the happy path is unaffected.
	var uuidCorrect, totalUUIDChecks, outputConformant, nonAbortedRequests int
	for _, r := range result.Results {
		if r.Aborted {
			continue
		}
		nonAbortedRequests++
		totalUUIDChecks += len(r.ExpectedUUIDs)
		for _, found := range r.UUIDFound {
			if found {
				uuidCorrect++
			}
		}
		if r.ExactMatch {
			outputConformant++
		}
	}

	uuidPass := uuidCorrect == totalUUIDChecks
	conformPass := outputConformant == nonAbortedRequests

	// Print results
	fmt.Printf("\n=== CACHE COHERENCY EVALUATION RESULTS ===\n")
	fmt.Printf("Model: %s\n", result.Model)
	fmt.Printf("Series: %d | Concurrency: %d | Garbage characters: %d | UUID stamps/series: %d | Auto max output: %d tokens\n",
		result.SeriesCount, result.Concurrency, result.GarbageChars, result.StampsPerSeries, result.MaxOutputTokens)
	fmt.Printf("Total requests: %d | Elapsed: %.1fs\n", result.TotalRequests, result.ElapsedSeconds)
	if result.AbortedCount > 0 {
		fmt.Printf("Aborted mid-flight (--abort-fraction, excluded from scoring above): %d\n", result.AbortedCount)
	}
	if result.ResetTriggerCount > 0 || c.ResetEveryN > 0 {
		fmt.Printf("reset_prefix_cache triggers (--reset-every-n): %d attempted, %d failed\n", result.ResetTriggerCount, result.ResetErrorCount)
	}
	fmt.Printf("\n")

	// Test 1: UUID correctness (per UUID stamp)
	status := "PASS"
	if !uuidPass {
		status = "FAIL"
	}
	fmt.Printf("[%s] Test 1 — UUID correctness: %d/%d (expected UUID present in response, counted per UUID stamp)\n", status, uuidCorrect, totalUUIDChecks)

	// Test 2: Output conformity (per request)
	status = "PASS"
	if !conformPass {
		status = "FAIL"
	}
	fmt.Printf("[%s] Test 2 — Output conformity: %d/%d (response is exactly the comma-joined UUID list)\n", status, outputConformant, nonAbortedRequests)

	// Per-request details for failures. ERROR and NOT_EXACT are counted per request;
	// CROSS_CONTAMINATION and UUID_MISSING_FLAKY are counted PER UUID (a single
	// request can contribute more than one of either, since it now carries
	// StampsPerSeries UUIDs instead of one).
	//
	// Aborted requests (r.Aborted) are skipped here too: an intentional
	// --abort-fraction cancellation is not a coherency failure to report, just the
	// corruption *opportunity* — result.AbortedCount above is their only mention.
	hasFailures := false
	var errorCount, crossContamCount, uuidMissingFlakyCount, notExactCount int
	// Cold/warm cycle breakdown of misses and NOT_EXACT responses (cycle 1 = cold,
	// cycle >= 2 = warm), plus the set of distinct cycle numbers actually seen — needed
	// to normalize by cycle cardinality (there is always exactly one cold cycle per
	// series, but --total can produce many warm cycles) in the summary below.
	var coldMissingCount, warmMissingCount, coldNotExactCount, warmNotExactCount int
	cyclesSeen := make(map[int]bool)
	for _, r := range result.Results {
		if r.Aborted {
			continue
		}
		cyclesSeen[r.Cycle] = true
		isError := r.Error != ""

		var missing []string
		var leaked []string
		if !isError {
			for idx, found := range r.UUIDFound {
				if !found {
					missing = append(missing, r.ExpectedUUIDs[idx])
				}
			}
			// Any uuid belonging to a DIFFERENT series found in this response/thinking
			// is cross-contamination (KV/scheduling leak across series).
			leaked = benchmark.FindLeakedUUIDs(r.Response, r.Thinking, r.SeriesIdx, result.SeriesUUIDs)
		}
		uuidMissingFlakyCount += len(missing)
		crossContamCount += len(leaked)
		if r.Cycle == 1 {
			coldMissingCount += len(missing)
		} else {
			warmMissingCount += len(missing)
		}

		if !isError && len(missing) == 0 && len(leaked) == 0 && r.ExactMatch {
			continue // fully correct request — nothing to report
		}

		if !hasFailures {
			fmt.Printf("\nDetails:\n")
			hasFailures = true
		}

		var marker string
		switch {
		case isError:
			marker = "ERROR"
			errorCount++
		case len(leaked) > 0:
			marker = "CROSS_CONTAMINATION"
		case len(missing) > 0:
			marker = "UUID_MISSING_FLAKY"
		default:
			// All expected UUIDs present, nothing leaked, but response isn't the
			// exact comma-joined format (extra prose/whitespace/etc).
			marker = "NOT_EXACT"
			notExactCount++
			if r.Cycle == 1 {
				coldNotExactCount++
			} else {
				warmNotExactCount++
			}
		}

		// --full-missing-responses: by default (false) content/thinking are truncated
		// to a short preview; opt in for the FULL untruncated dump (output is
		// typically redirected to a log file, so verbose dumps are fine there).
		resp := r.Response
		if !c.FullMissingResponses && len(resp) > 200 {
			resp = resp[:200] + "..."
		}
		suffix := ""
		if len(missing) > 0 {
			suffix += fmt.Sprintf(" missing=%v", missing)
		}
		if len(leaked) > 0 {
			suffix += fmt.Sprintf(" leaked_from=%v", leaked)
		}
		if r.Thinking != "" {
			thinking := r.Thinking
			if !c.FullMissingResponses && len(thinking) > 100 {
				thinking = thinking[:100] + "..."
			}
			fmt.Printf("  [%s] series=%d cycle=%d: content=%q thinking=%q%s\n", marker, r.SeriesIdx, r.Cycle, resp, thinking, suffix)
		} else {
			fmt.Printf("  [%s] series=%d cycle=%d: %q%s\n", marker, r.SeriesIdx, r.Cycle, resp, suffix)
		}
	}

	// Failure breakdown summary. Note ERROR/NOT_EXACT are request counts while
	// CROSS_CONTAMINATION/UUID_MISSING_FLAKY are UUID counts (see loop above) —
	// totalFailures is only used as a >0 gate for whether to print this block.
	totalFailures := errorCount + crossContamCount + uuidMissingFlakyCount + notExactCount
	if totalFailures > 0 {
		fmt.Printf("\nFailure breakdown:\n")
		fmt.Printf("  ERROR (request failed):                              %d requests\n", errorCount)
		fmt.Printf("  CROSS_CONTAMINATION (other series UUID in response): %d UUIDs\n", crossContamCount)
		fmt.Printf("  UUID_MISSING_FLAKY (expected UUID absent):           %d UUIDs\n", uuidMissingFlakyCount)
		fmt.Printf("  NOT_EXACT (correct UUIDs but extra text):            %d requests\n", notExactCount)
		if crossContamCount > 0 {
			fmt.Printf("  ⚠ %d cross-contamination — KV/scheduling leak between series\n", crossContamCount)
		}
	}

	fmt.Printf("\nUsage Statistics:\n")
	fmt.Printf("  Total cost: $%.6f\n", result.TotalCost)
	fmt.Printf("  Total input tokens: %d\n", result.TotalInputTokens)
	fmt.Printf("  Total output tokens: %d\n", result.TotalOutputTokens)
	fmt.Printf("  Total cached tokens: %d\n", result.TotalCachedTokens)

	fmt.Printf("\nCache Performance (TTFT-based):\n")
	fmt.Printf("  Mean cold TTFT (cycle 1): %.0f ms\n", result.MeanColdTTFT)
	fmt.Printf("  Mean warm TTFT (cycle 2): %.0f ms\n", result.MeanWarmTTFT)
	if result.CacheTotal > 0 {
		hitRate := float64(result.CacheHits) / float64(result.CacheTotal) * 100
		fmt.Printf("  Implicit cache hit rate: %.1f%% (%d/%d)\n", hitRate, result.CacheHits, result.CacheTotal)
	} else {
		fmt.Printf("  Implicit cache hit rate: N/A (no cycle-2 requests)\n")
	}
	nCold, nWarm := coldWarmCycleCounts(cyclesSeen)
	fmt.Printf("  Miss distribution:      %s\n", formatCycleDistribution(coldMissingCount, warmMissingCount, nCold, nWarm))
	fmt.Printf("  NOT_EXACT distribution: %s\n", formatCycleDistribution(coldNotExactCount, warmNotExactCount, nCold, nWarm))

	if !uuidPass {
		return fmt.Errorf("cache coherency FAILED: UUID correctness %d/%d", uuidCorrect, totalUUIDChecks)
	}
	if !conformPass {
		return fmt.Errorf("output conformity FAILED: %d/%d exact matches (UUID correctness passed)", outputConformant, result.TotalRequests)
	}
	return nil
}

// defaultGarbageChars is the effective default garbage-character budget when neither
// --garbage-characters nor the deprecated --garbage-tokens is given. 213000 preserves
// the historical prompt token budget (≈50k input tokens/request on the Nex tokenizer)
// after the filler content changed from repeated 'A' to repeated
// "<ignore>ignore this text</ignore>". The tagged filler tokenizes ~1.88× denser than
// BPE-compressible 'A' runs (400000 'A' chars = 51680 tokens vs 400000 tagged chars =
// 97025 tokens on Nex), so ~213000 tagged chars lands back at ~50.5k tokens. Budget
// drifts per model tokenizer — that is accepted. 213000 ÷ stampIntervalChars (8192) ≈
// 26 UUID stamps per series at default settings.
const defaultGarbageChars = 213000

// resolveGarbageChars applies --garbage-characters / --garbage-tokens precedence and
// prints the required deprecation/precedence warnings to w:
//   - --garbage-characters (literal character count), when set, always wins.
//   - the deprecated --garbage-tokens (N tokens ≈ N*4 characters — preserves its
//     historical meaning) is used only when --garbage-characters is not set; using it
//     prints a deprecation warning.
//   - if both are set, --garbage-characters wins and a warning notes --garbage-tokens
//     was ignored.
//   - if neither is set, defaultGarbageChars (213000) is used.
//
// Flags are treated as "set" when > 0 — the CLI options intentionally carry no `default`
// tag for these two fields so an unset flag surfaces here as the Go zero value (0).
func resolveGarbageChars(garbageCharacters, garbageTokens int, w io.Writer) int {
	charsSet := garbageCharacters > 0
	tokensSet := garbageTokens > 0

	switch {
	case charsSet && tokensSet:
		fmt.Fprintln(w, "WARNING: both --garbage-characters and --garbage-tokens given; --garbage-characters takes precedence and --garbage-tokens is ignored.")
		return garbageCharacters
	case charsSet:
		return garbageCharacters
	case tokensSet:
		fmt.Fprintln(w, "WARNING: --garbage-tokens is deprecated and will be removed; use --garbage-characters (N tokens ≈ N*4 characters).")
		return garbageTokens * 4
	default:
		return defaultGarbageChars
	}
}

// findLeakedUUIDs moved to benchmark.FindLeakedUUIDs (benchmark/replay_uuid.go)
// so both this CLI and the dataset-replay UUID validation path share one
// implementation.

// coldWarmCycleCounts returns the number of distinct COLD (cycle 1) and WARM (cycle >= 2)
// cycle numbers present in cyclesSeen — cycle-number CARDINALITY, not a request count.
// There is always at most one cold cycle (cycle 1), but --total can produce many warm
// cycles (e.g. SERIES=64 TOTAL=1024 -> cycles 2..16, nWarm=15). formatCycleDistribution
// uses this cardinality to normalize raw counts by per-cycle rate instead of raw volume.
func coldWarmCycleCounts(cyclesSeen map[int]bool) (nCold, nWarm int) {
	for c := range cyclesSeen {
		switch {
		case c == 1:
			nCold++
		case c >= 2:
			nWarm++
		}
	}
	return nCold, nWarm
}

// formatCycleDistribution renders a "cold N (A% abs / B% norm)  warm M (C% abs / D% norm)"
// summary of how a raw (coldCount, warmCount) pair — e.g. missing UUIDs or NOT_EXACT
// responses — splits across cold (cycle 1) vs warm (cycle >= 2) cycles:
//
//   - abs  = coldCount / (coldCount+warmCount): the raw count share. "n/a" when both
//     counts are zero.
//   - norm = (coldCount/nCold) / (coldCount/nCold + warmCount/nWarm): the PER-CYCLE-RATE
//     share, correcting for cold always contributing exactly one cycle while warm can
//     contribute many (--total mode) — without this, raw counts alone look warm-heavy
//     purely from cycle volume, not miss rate. "n/a" when either bucket's cycle count is
//     zero (e.g. a --total run too short to reach cycle 2, or no requests survived).
//
// A norm share far below 50% for cold flags warm-biased loss (a warm-read data problem);
// far above 50% flags a cold-path problem.
func formatCycleDistribution(coldCount, warmCount, nCold, nWarm int) string {
	total := coldCount + warmCount
	absCold, absWarm := "n/a", "n/a"
	if total > 0 {
		absCold = fmt.Sprintf("%.1f%%", float64(coldCount)/float64(total)*100)
		absWarm = fmt.Sprintf("%.1f%%", float64(warmCount)/float64(total)*100)
	}

	normCold, normWarm := "n/a", "n/a"
	if nCold > 0 && nWarm > 0 {
		coldRate := float64(coldCount) / float64(nCold)
		warmRate := float64(warmCount) / float64(nWarm)
		if rateTotal := coldRate + warmRate; rateTotal > 0 {
			normCold = fmt.Sprintf("%.1f%%", coldRate/rateTotal*100)
			normWarm = fmt.Sprintf("%.1f%%", warmRate/rateTotal*100)
		}
	}

	return fmt.Sprintf("cold %d (%s abs / %s norm)  warm %d (%s abs / %s norm)", coldCount, absCold, normCold, warmCount, absWarm, normWarm)
}
