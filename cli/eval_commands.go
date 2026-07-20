package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

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
	for _, r := range result.Results {
		if r.Aborted {
			continue
		}
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
			leaked = findLeakedUUIDs(r.Response, r.Thinking, r.SeriesIdx, result.SeriesUUIDs)
		}
		uuidMissingFlakyCount += len(missing)
		crossContamCount += len(leaked)

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

// findLeakedUUIDs scans resp and thinking for UUIDs belonging to a series OTHER than
// ownSeries, per the ordered seriesUUIDs list (seriesUUIDs[i] = full UUID stamp list of
// series i — this doubles as the uuid -> owning-series mapping without needing an actual
// map, keeping iteration order — and therefore leak-report order — deterministic for a
// given seed). Returns "uuid(series=N)" entries, one per leaked UUID found.
func findLeakedUUIDs(resp, thinking string, ownSeries int, seriesUUIDs [][]string) []string {
	var leaked []string
	for si, uuids := range seriesUUIDs {
		if si == ownSeries {
			continue
		}
		for _, u := range uuids {
			if strings.Contains(resp, u) || strings.Contains(thinking, u) {
				leaked = append(leaked, fmt.Sprintf("%s(series=%d)", u, si))
			}
		}
	}
	return leaked
}
