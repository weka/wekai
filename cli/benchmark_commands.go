package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/weka/wekai/benchmark"
	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
)

const defaultBenchmarkQuestion = "summarize in three words first 3%, last 3%, and middle 20%, clearly separate this three parts in response"

func parseTokenSize(s string) (int, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" || s == "0" {
		return 0, nil
	}
	multiplier := 1
	if strings.HasSuffix(s, "k") {
		multiplier = 1000
		s = s[:len(s)-1]
	} else if strings.HasSuffix(s, "m") {
		multiplier = 1000000
		s = s[:len(s)-1]
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("invalid token size %q: %w", s, err)
	}
	return n * multiplier, nil
}

// parseSeriesIndices parses a comma-separated list of non-negative integers
// (e.g. "3,7,42") into a set. Returns an error if any value is invalid or
// negative.
func parseSeriesIndices(s string) (map[int]bool, error) {
	result := map[int]bool{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("non-integer value %q in comma list", part)
		}
		if n < 0 {
			return nil, fmt.Errorf("negative index %d is not allowed", n)
		}
		result[n] = true
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("empty index list")
	}
	return result, nil
}

// parseSeriesRange parses an inclusive range expression "A-B" (both 0-based)
// into a set of indices. A must be <= B; both must be non-negative.
func parseSeriesRange(s string) (map[int]bool, error) {
	s = strings.TrimSpace(s)
	parts := strings.SplitN(s, "-", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("expected format A-B (e.g. 0-50), got %q", s)
	}
	lo, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return nil, fmt.Errorf("invalid lower bound %q: %w", parts[0], err)
	}
	hi, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil {
		return nil, fmt.Errorf("invalid upper bound %q: %w", parts[1], err)
	}
	if lo < 0 || hi < 0 {
		return nil, fmt.Errorf("range bounds must be non-negative, got %d-%d", lo, hi)
	}
	if lo > hi {
		return nil, fmt.Errorf("lower bound %d must be <= upper bound %d", lo, hi)
	}
	result := make(map[int]bool, hi-lo+1)
	for i := lo; i <= hi; i++ {
		result[i] = true
	}
	return result, nil
}

// BenchmarkAutoCommand implements the benchmark auto subcommand
type BenchmarkAutoCommand struct {
	*BenchmarkAutoOptions
}

func (c *BenchmarkAutoCommand) Execute(args []string) error {
	if err := runPreExecute(context.Background()); err != nil {
		return err
	}

	// Resolve models: --model (repeatable) and --models (repeatable) are
	// aliases, merged. Earlier versions had --model as a singular string,
	// which silently dropped all but the last value when callers passed
	// --model multiple times.
	models := append([]string{}, c.Models...)
	models = append(models, c.Model...)
	if len(models) == 0 {
		return fmt.Errorf("at least one model must be specified via --model or --models")
	}

	// Shorthand: a bare http(s) URL is promoted to a dynamic/ spec with
	// type=openai_vllm so users can write --model http://host:port/v1 without
	// repeating the dynamic/type boilerplate every time.
	for i, m := range models {
		models[i] = llm.NormalizeModelSpec(m)
	}

	// Validate each model
	for _, m := range models {
		if err := llm.ValidateModel(m); err != nil {
			return fmt.Errorf("invalid model %s: %w", m, err)
		}
	}

	if config.Config.GlobalModelOverride != "" {
		return fmt.Errorf("--global-model-override cannot be used with benchmark auto; specify model via --model or --models flag")
	}

	// --series is a shortcut for --start-series=N --max-series=N; enforce mutual exclusivity
	startSeries := c.StartSeries
	maxSeries := c.MaxSeries
	if c.Series > 0 {
		if c.StartSeries > 0 || c.MaxSeries > 0 {
			return fmt.Errorf("--series is mutually exclusive with --start-series and --max-series")
		}
		startSeries = c.Series
		maxSeries = c.Series
	}

	// Validate --hot-series-concurrency.
	if c.HotSeriesConcurrency < 0 {
		return fmt.Errorf("--hot-series-concurrency must be >= 0, got %d", c.HotSeriesConcurrency)
	}
	if c.HotSeriesConcurrency > 0 && startSeries > 0 && c.HotSeriesConcurrency > startSeries {
		return fmt.Errorf("--hot-series-concurrency (%d) must be <= --series/--start-series (%d)", c.HotSeriesConcurrency, startSeries)
	}

	// Determine question
	question := c.Args.Question
	if question == "" {
		question = defaultBenchmarkQuestion
	}

	// Apply profile defaults (only for fields still at their zero/default value)
	const defaultAutoTokens = 100000
	if c.Profile == "agentic" {
		if c.Step == "" {
			c.Step = "3k"
		}
		if c.SharedPrefixPerSeries == 0 {
			c.SharedPrefixPerSeries = 4
		}
		if c.GlobalCacheHitRateTarget == 0 {
			c.GlobalCacheHitRateTarget = 0.95
		}
		if c.Tokens == defaultAutoTokens {
			c.Tokens = 200000
		}
		if c.MaxOutputTokens == 0 {
			c.MaxOutputTokens = 32000
		}
		if !c.ExhaustSessions {
			c.ExhaustSessions = true
		}
	} else if c.Profile != "" {
		return fmt.Errorf("unknown profile %q (supported: agentic)", c.Profile)
	}

	if c.ExhaustSessions && (c.Step == "" || c.Step == "0") {
		return fmt.Errorf("--exhaust-sessions requires --step (directly or via --profile=agentic)")
	}

	// Replay defaults: conversations carry real tool-using assistant turns, so
	// give the model room to respond fully. Manual --max-output-tokens still wins.
	if c.FromDataset != "" && c.MaxOutputTokens == 0 {
		c.MaxOutputTokens = 32000
	}

	// Replay series-count defaults: in replay mode, --replay-series N defines
	// the total work. Unless the user explicitly set --start-series / --max-series
	// (or the --series shortcut), pin both to N so all N conversations can be
	// in flight at once — concurrency is bounded by --concurrency anyway, so
	// extra series slots just wait idle on the gate. Saves the user from
	// repeating the number three times.
	if c.FromDataset != "" && c.ReplaySeries > 0 && c.Series == 0 {
		if startSeries <= 0 {
			startSeries = c.ReplaySeries
		}
		if maxSeries <= 0 {
			maxSeries = c.ReplaySeries
		}
	}

	ctx := context.Background()
	ctx, shutdown, err := config.Init(ctx, "benchmark_auto")
	if err != nil {
		return fmt.Errorf("failed to initialize config: %w", err)
	}
	defer func() {
		_ = shutdown(ctx)
	}()

	// Parse optional timeout
	var timeout time.Duration
	if c.Timeout != "" {
		timeout, err = time.ParseDuration(c.Timeout)
		if err != nil {
			return fmt.Errorf("invalid timeout duration: %w", err)
		}
		fmt.Printf("Timeout: %v\n", timeout)
	}

	// Resolve documentation source
	var autoDocsContent string
	autoDocsDir := c.DocsDir
	if autoDocsDir == "" {
		content, err := benchmark.GetEmbeddedDocs(c.Tokens)
		if err != nil {
			return fmt.Errorf("failed to get embedded docs: %w", err)
		}
		autoDocsContent = content
		autoDocsDir = "(embedded)"
	}

	cfg := benchmark.AutoBenchmarkConfig{
		DocsDir:                    autoDocsDir,
		DocsContent:                autoDocsContent,
		Models:                     models,
		Question:                   question,
		Timeout:                    timeout,
		MaxSeries:                  maxSeries,
		StartSeries:                startSeries,
		MaxConcurrency:             c.MaxConcurrency,
		CacheTarget:                c.CacheTarget,
		CacheWindowSize:            c.CacheWindowSize,
		TTFTDegradationFactor:      c.TTFTDegradationFactor,
		TTFTHitThreshold:           c.TTFTHitThreshold,
		Concurrency:                c.Concurrency,
		VerboseCache:               c.VerboseCache,
		PrintResponses:             c.PrintResponses,
		SaveRequestDataDir:         c.SaveRequestData,
		ErrorRateLimit:             c.ErrorRateLimit,
		MaxConsecutiveFailures:     c.MaxConsecutiveFailures,
		MaxTotalErrors:             c.MaxTotalErrors,
		Total:                      c.Total,
		Tokens:                     c.Tokens,
		HotSeriesConcurrency:       c.HotSeriesConcurrency,
		SharedPrefixPerSeries:      c.SharedPrefixPerSeries,
		GlobalCacheHitRateTarget:   c.GlobalCacheHitRateTarget,
		MaxOutputTokens:            c.MaxOutputTokens,
		ExhaustSessions:            c.ExhaustSessions,
		FromDataset:                c.FromDataset,
		ReplaySeries:               c.ReplaySeries,
		ReplayNoStamp:              c.ReplayNoStamp,
		AbortOnCollapse:            c.AbortOnCollapse,
		ReplayStopAtLowConcurrency: c.ReplayStopAtLowConcurrency,
		RouterReplayFile:           c.RouterReplayFile,
		RouterReplayRoles:          c.RouterReplayRoles,
		DryRun:                     c.DryRun,
		DryRunColdTPS:              c.DryRunColdTPS,
		DryRunWarmTPS:              c.DryRunWarmTPS,
		DryRunOutputTPS:            c.DryRunOutputTPS,
		CacheSimChunkBytes:         c.CacheSimChunkBytes,
		RandomGateOrder:            c.RandomGateOrder,
	}

	if c.FromDataset != "" && c.RouterReplayFile != "" {
		return fmt.Errorf("--from-dataset and --router-replay-file are mutually exclusive")
	}
	if c.DryRun && c.RouterReplayFile == "" {
		return fmt.Errorf("--dry-run requires --router-replay-file")
	}
	if c.RouterReplayFile != "" && c.MaxOutputTokens != 0 {
		fmt.Fprintln(os.Stderr,
			"warning: --max-output-tokens overrides per-request budgets baked into the router replay file")
	}

	// Parse --replay-series-indices / --replay-series-range into a set of
	// 0-based session indices. Mutually exclusive with each other; both
	// require --router-replay-file.
	if c.RouterReplaySeriesIndices != "" && c.RouterReplaySeriesRange != "" {
		return fmt.Errorf("--replay-series-indices and --replay-series-range are mutually exclusive")
	}
	if c.RouterReplaySeriesIndices != "" || c.RouterReplaySeriesRange != "" {
		if c.RouterReplayFile == "" {
			return fmt.Errorf("--replay-series-indices/--replay-series-range require --router-replay-file")
		}
	}
	if c.RouterReplaySeriesIndices != "" {
		indices, err := parseSeriesIndices(c.RouterReplaySeriesIndices)
		if err != nil {
			return fmt.Errorf("invalid --replay-series-indices: %w", err)
		}
		cfg.RouterReplaySeriesIndices = indices
	}
	if c.RouterReplaySeriesRange != "" {
		indices, err := parseSeriesRange(c.RouterReplaySeriesRange)
		if err != nil {
			return fmt.Errorf("invalid --replay-series-range: %w", err)
		}
		cfg.RouterReplaySeriesIndices = indices
	}

	if c.Step != "" && c.Step != "0" {
		stepTokens, err := parseTokenSize(c.Step)
		if err != nil {
			return fmt.Errorf("invalid --step: %w", err)
		}
		cfg.Step = stepTokens
	}

	if c.StepStartingTokens != "" && c.StepStartingTokens != "0" {
		startTokens, err := parseTokenSize(c.StepStartingTokens)
		if err != nil {
			return fmt.Errorf("invalid --step-starting-tokens: %w", err)
		}
		cfg.StepStartingTokens = startTokens
	}

	if c.RequestTimeout != "" {
		d, err := time.ParseDuration(c.RequestTimeout)
		if err != nil {
			return fmt.Errorf("invalid --request-timeout: %w", err)
		}
		cfg.RequestTimeout = d
	}

	if c.PrintErrorsThreshold != "" && c.PrintErrorsThreshold != "0" {
		d, err := time.ParseDuration(c.PrintErrorsThreshold)
		if err != nil {
			return fmt.Errorf("invalid --print-errors-threshold: %w", err)
		}
		cfg.PrintErrorsThreshold = d
	}

	if c.DryRun {
		fmt.Fprintf(os.Stderr, "\n*** DRY RUN — no remote requests. Synthetic rates: cold=%d warm=%d output=%d tok/s ***\n\n",
			c.DryRunColdTPS, c.DryRunWarmTPS, c.DryRunOutputTPS)
	}

	return benchmark.RunAutoBenchmark(ctx, cfg)
}

// BenchmarkVisualizeCommand implements the benchmark visualize subcommand
type BenchmarkVisualizeCommand struct {
	*BenchmarkVisualizeOptions
}

func (c *BenchmarkVisualizeCommand) Execute(args []string) error {
	if err := runPreExecute(context.Background()); err != nil {
		return err
	}

	dir := c.Args.Dir
	htmlPath, err := benchmark.GenerateVisualization(dir, c.Concurrency)
	if err != nil {
		return fmt.Errorf("generate visualization: %w", err)
	}
	fmt.Printf("Visualization saved to: %s\n", htmlPath)
	return nil
}

// BenchmarkVisualizeMergeCommand implements the benchmark visualize-merge subcommand
type BenchmarkVisualizeMergeCommand struct {
	*BenchmarkVisualizeMergeOptions
}

func (c *BenchmarkVisualizeMergeCommand) Execute(args []string) error {
	if err := runPreExecute(context.Background()); err != nil {
		return err
	}

	dirs := c.Args.Dirs
	if len(dirs) == 0 {
		return fmt.Errorf("at least one directory argument is required")
	}

	if c.All {
		if len(dirs) != 1 {
			return fmt.Errorf("--all expects exactly one parent directory argument")
		}
		parent := dirs[0]
		entries, err := os.ReadDir(parent)
		if err != nil {
			return fmt.Errorf("read parent directory %s: %w", parent, err)
		}
		dirs = nil
		for _, e := range entries {
			if e.IsDir() && !strings.HasPrefix(e.Name(), "merged") {
				dirs = append(dirs, filepath.Join(parent, e.Name()))
			}
		}
		if len(dirs) == 0 {
			return fmt.Errorf("no subdirectories found in %s", parent)
		}
	}

	var labels []string
	if c.Labels != "" {
		for _, l := range strings.Split(c.Labels, ",") {
			labels = append(labels, strings.TrimSpace(l))
		}
		if len(labels) != len(dirs) {
			return fmt.Errorf("--labels count (%d) does not match directory count (%d)", len(labels), len(dirs))
		}
	}

	htmlPath, err := benchmark.GenerateVisualizationMerged(dirs, labels, c.Output, c.Concurrency)
	if err != nil {
		return fmt.Errorf("generate merged visualization: %w", err)
	}
	fmt.Printf("Merged visualization saved to: %s\n", htmlPath)
	return nil
}

// BenchmarkThroughputCommand implements the benchmark throughput subcommand
type BenchmarkThroughputCommand struct {
	*BenchmarkThroughputOptions
}

func (c *BenchmarkThroughputCommand) Execute(args []string) error {
	if err := runPreExecute(context.Background()); err != nil {
		return err
	}

	for _, model := range c.Models {
		if err := llm.ValidateModel(model); err != nil {
			return fmt.Errorf("invalid model %s: %w", model, err)
		}
	}

	if config.Config.GlobalModelOverride != "" {
		return fmt.Errorf("--global-model-override cannot be used with benchmark throughput; specify models via --models flag")
	}

	if c.ColdPrefillConcurrency < 0 {
		return fmt.Errorf("--cold-prefill-concurrency must be >= 0, got %d", c.ColdPrefillConcurrency)
	}

	question := c.Question
	if c.Args.Question != "" {
		question = c.Args.Question
	}

	ctx := context.Background()
	ctx, shutdown, err := config.Init(ctx, "benchmark_throughput")
	if err != nil {
		return fmt.Errorf("failed to initialize config: %w", err)
	}
	defer func() {
		_ = shutdown(ctx)
	}()

	// Signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		sig := <-sigChan
		if !config.Config.JSONOutput {
			fmt.Printf("\n\nReceived signal %v. Stopping benchmark...\n\n", sig)
		}
		cancel()
	}()

	// Apply timeout
	if c.Timeout != "" {
		duration, err := time.ParseDuration(c.Timeout)
		if err != nil {
			return fmt.Errorf("invalid timeout duration: %w", err)
		}
		ctx, cancel = context.WithTimeout(ctx, duration)
		defer cancel()
		if !config.Config.JSONOutput {
			fmt.Printf("  Timeout: %v\n", duration)
		}
	}

	// Resolve documentation
	var docsContent string
	docsDir := c.DocsDir
	if docsDir == "" {
		content, err := benchmark.GetEmbeddedDocs(c.Tokens)
		if err != nil {
			return fmt.Errorf("failed to get embedded docs: %w", err)
		}
		docsContent = content
		docsDir = "(embedded)"
	} else {
		return fmt.Errorf("--docs-dir is not yet supported for throughput benchmark; use embedded docs")
	}

	if !config.Config.JSONOutput {
		numSeries := c.StartingConcurrency * c.SeriesPerConcurrency
		fmt.Printf("Throughput Benchmark Configuration:\n")
		fmt.Printf("  Documentation: %s\n", docsDir)
		fmt.Printf("  Models: %v\n", c.Models)
		fmt.Printf("  Concurrency: %d\n", c.StartingConcurrency)
		if c.ColdPrefillConcurrency > 0 {
			fmt.Printf("  Cold prefill concurrency: %d\n", c.ColdPrefillConcurrency)
		}
		fmt.Printf("  Series per concurrency: %d (total series: %d)\n", c.SeriesPerConcurrency, numSeries)
		fmt.Printf("  Decode tokens: %d\n", c.DecodeTokens)
		if question != "" {
			fmt.Printf("  Question: %s\n", question)
		}
		fmt.Println()
	}

	var progressCallback func(string)
	if config.Config.JSONOutput {
		progressCallback = func(msg string) { fmt.Fprint(os.Stderr, msg) }
	} else {
		progressCallback = func(msg string) { fmt.Print(msg) }
	}

	if c.AutoSweep {
		// Validate sweep-specific constraints.
		if c.MaxConcurrency < c.StartingConcurrency {
			return fmt.Errorf("--max-concurrency (%d) must be >= --starting-concurrency (%d)", c.MaxConcurrency, c.StartingConcurrency)
		}
		if c.SweepImprovementThreshold < 0 {
			return fmt.Errorf("--sweep-improvement-threshold must be >= 0, got %f", c.SweepImprovementThreshold)
		}

		if !config.Config.JSONOutput {
			fmt.Printf("  Auto-sweep: starting-concurrency=%d, max-concurrency=%d, improvement-threshold=%.0f%%\n\n",
				c.StartingConcurrency, c.MaxConcurrency, c.SweepImprovementThreshold*100)
		}

		// Run sweep per model sequentially (sweeps are slow; parallel would overwhelm the API).
		// Collect into a ThroughputResult so we can reuse FormatText/FormatJSON.
		modelResults := make([]benchmark.ThroughputModelResult, 0, len(c.Models))
		startTime := time.Now()

		for _, model := range c.Models {
			sweepResult, hsResult, err := benchmark.RunThroughputSweep(ctx, model, docsContent, question,
				c.StartingConcurrency, c.ColdPrefillConcurrency, c.SeriesPerConcurrency, c.DecodeTokens, c.Tokens,
				c.MaxConcurrency, c.SweepImprovementThreshold, c.HighSeriesCheck, progressCallback)
			if err != nil {
				if config.Config.JSONOutput {
					errorResult := map[string]interface{}{"success": false, "error": err.Error()}
					return json.NewEncoder(os.Stdout).Encode(errorResult)
				}
				return fmt.Errorf("sweep for model %s failed: %w", model, err)
			}
			// Build a ThroughputModelResult from the best level so the existing
			// summary table still renders useful data.
			var bestLevel benchmark.ThroughputLevelResult
			if len(sweepResult.Levels) > 0 {
				for _, l := range sweepResult.Levels {
					if l.DecodeRate > bestLevel.DecodeRate {
						bestLevel = l
					}
				}
			}

			mr := benchmark.ThroughputModelResult{
				ModelName:        model,
				ModelDisplayName: benchmark.GetModelDisplayName(model),
				ColdPrefillRate:  bestLevel.ColdPrefillRate,
				WarmPrefillRate:  bestLevel.WarmPrefillRate,
				DecodeRate:       bestLevel.DecodeRate,
				DecodeTokens:     c.DecodeTokens,
				Concurrency:      sweepResult.BestConcurrency,
				NumSeries:        bestLevel.PoolSize,
				Sweep:            sweepResult,
				HighSeriesCheck:  hsResult,
			}
			modelResults = append(modelResults, mr)
		}

		result := &benchmark.ThroughputResult{
			Models:               modelResults,
			TotalDuration:        time.Since(startTime),
			Concurrency:          c.StartingConcurrency,
			SeriesPerConcurrency: c.SeriesPerConcurrency,
			DecodeTokens:         c.DecodeTokens,
			Question:             question,
		}

		if config.Config.JSONOutput {
			out, err := result.FormatJSON()
			if err != nil {
				return fmt.Errorf("failed to format JSON output: %w", err)
			}
			fmt.Println(out)
		} else {
			fmt.Println(result.FormatText())
		}
		return nil
	}

	result, err := benchmark.RunThroughput(ctx, c.Models, docsContent, question,
		c.StartingConcurrency, c.ColdPrefillConcurrency, c.SeriesPerConcurrency, c.DecodeTokens, c.Tokens, c.HighSeriesCheck, progressCallback)
	if err != nil {
		if config.Config.JSONOutput {
			errorResult := map[string]interface{}{"success": false, "error": err.Error()}
			return json.NewEncoder(os.Stdout).Encode(errorResult)
		}
		return fmt.Errorf("benchmark failed: %w", err)
	}

	if config.Config.JSONOutput {
		out, err := result.FormatJSON()
		if err != nil {
			return fmt.Errorf("failed to format JSON output: %w", err)
		}
		fmt.Println(out)
	} else {
		fmt.Println(result.FormatText())
	}

	return nil
}
