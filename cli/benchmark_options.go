package cli

// BenchmarkThroughputOptions contains options for the benchmark throughput subcommand
type BenchmarkThroughputOptions struct {
	DocsDir                string   `long:"docs-dir" description:"Documentation directory (default: use built-in benchmark doc)" env:"BENCHMARK_DOCS_DIR"`
	Tokens                 int      `long:"tokens" description:"Truncate embedded doc to this many tokens (max 305465)" default:"100000" env:"BENCHMARK_TOKENS"`
	Models                 []string `long:"models" description:"Models to benchmark (can be specified multiple times)" required:"yes"`
	StartingConcurrency    int      `long:"starting-concurrency" description:"Concurrency level C for all phases (also the first sweep level)" default:"16" env:"BENCHMARK_STARTING_CONCURRENCY"`
	ColdPrefillConcurrency int      `long:"cold-prefill-concurrency" description:"Concurrency for the cold prefill phase only (0 = use --starting-concurrency). Lets cold prefill run at a different rate than the warm/decode phases." default:"0" env:"BENCHMARK_COLD_PREFILL_CONCURRENCY"`
	SeriesPerConcurrency   int      `long:"series-per-concurrency" description:"Series multiplier: total series = C * this" default:"1" env:"BENCHMARK_SERIES_PER_CONCURRENCY"`
	DecodeTokens           int      `long:"decode-tokens" description:"max_tokens for the decode phase (phase 3)" default:"1000" env:"BENCHMARK_DECODE_TOKENS"`
	Timeout                string   `long:"timeout" description:"Maximum duration (e.g. 10m)" env:"BENCHMARK_TIMEOUT"`
	Question               string   `long:"question" description:"Question to ask" env:"BENCHMARK_QUESTION"`

	// Sweep mode flags
	AutoSweep                 bool    `long:"auto-sweep" description:"Sweep concurrency from --starting-concurrency upward (doubling) until decode rate plateaus"`
	MaxConcurrency            int     `long:"max-concurrency" description:"Upper bound for sweep" default:"64" env:"BENCHMARK_MAX_CONCURRENCY"`
	SweepImprovementThreshold float64 `long:"sweep-improvement-threshold" description:"Stop sweep when decode-rate improvement falls below this fraction (e.g. 0.05 = 5%)" default:"0.05" env:"BENCHMARK_SWEEP_IMPROVEMENT_THRESHOLD"`

	// Auxiliary high-series warm prefill check
	HighSeriesCheck int `long:"high-series-check" description:"After main run, measure warm prefill at this series count (0 = disabled). Upfills the existing series pool, cold-primes the new GUIDs, then measures warm prefill only." default:"0" env:"BENCHMARK_HIGH_SERIES_CHECK"`

	Args struct {
		Question string `positional-arg-name:"question" description:"Question to ask (overrides --question)"`
	} `positional-args:"yes"`
}

// BenchmarkAutoOptions contains options for the benchmark auto subcommand
type BenchmarkAutoOptions struct {
	DocsDir                    string   `long:"docs-dir" description:"Documentation directory to embed (default: use built-in benchmark doc)" env:"BENCHMARK_DOCS_DIR"`
	Tokens                     int      `long:"tokens" description:"Upper bound on per-request prompt tokens. With --docs-dir, caps --step prefix growth at this value (otherwise growth is bounded only by source-doc length). Without --docs-dir, also truncates the embedded doc (max 305465)." default:"100000" env:"BENCHMARK_TOKENS"`
	Model                      []string `long:"model" description:"Model to benchmark. Can be repeated to benchmark multiple endpoints concurrently (e.g. --model A --model B). Alias for --models."`
	Models                     []string `long:"models" description:"Model endpoints to benchmark concurrently (can be repeated; merged with --model values)."`
	Timeout                    string   `long:"timeout" description:"Maximum duration for benchmark (e.g. 10m, 1h). Prints summary on timeout." env:"BENCHMARK_TIMEOUT"`
	MaxSeries                  int      `long:"max-series" description:"Maximum number of series (safety cap, default 64). Mutually exclusive with --series." env:"BENCHMARK_MAX_SERIES"`
	StartSeries                int      `long:"start-series" description:"Initial number of series to spawn (default 1). Mutually exclusive with --series." env:"BENCHMARK_START_SERIES"`
	Series                     int      `long:"series" description:"Shortcut to pin both --start-series and --max-series to the same value (fixed series count). Mutually exclusive with --start-series and --max-series." env:"BENCHMARK_SERIES"`
	MaxConcurrency             int      `long:"max-concurrency" default:"0" env:"BENCHMARK_MAX_CONCURRENCY" description:"Maximum concurrency limit (0 = unlimited)"`
	CacheTarget                float64  `long:"cache-target" description:"Cache hit rate threshold (0.0-1.0) at which series scaling begins" default:"0.90" env:"BENCHMARK_CACHE_TARGET"`
	CacheWindowSize            int      `long:"cache-window-size" description:"Number of recent requests used to measure cache hit rate" default:"20" env:"BENCHMARK_CACHE_WINDOW_SIZE"`
	TTFTDegradationFactor      int      `long:"ttft-degradation-factor" default:"4" env:"BENCHMARK_TTFT_DEGRADATION_FACTOR" description:"TTFT multiplier above early cold-start baseline at which the heuristic is disqualified entirely (default 4)"`
	TTFTHitThreshold           float64  `long:"ttft-hit-threshold" default:"0.5" env:"BENCHMARK_TTFT_HIT_THRESHOLD" description:"Fraction of cold-start baseline below which a request counts as a cache hit (default 0.5 = 50%)"`
	Concurrency                int      `long:"concurrency" env:"BENCHMARK_CONCURRENCY" description:"Fixed concurrency; disables hill-climber (default 0 = auto)"`
	VerboseCache               bool     `long:"verbose-cache" description:"Print TTFT baseline/threshold/miss distribution when hit rate shifts; forces periodic (non-TTY) display mode"`
	PrintResponses             bool     `long:"print-responses" description:"Print each request and LLM response to stdout for sanity-checking; forces non-TTY display mode"`
	PrintErrorsThreshold       string   `long:"print-errors-threshold" description:"Print errors to stderr as they occur, rate-limited to at most one per this interval (e.g. 1s, 500ms). Empty/0 = disabled." env:"BENCHMARK_PRINT_ERRORS_THRESHOLD"`
	SaveRequestData            string   `long:"save-request-data" description:"Directory to save per-request JSONL data files (one file per model)" env:"BENCHMARK_SAVE_REQUEST_DATA"`
	Total                      int      `long:"total" description:"Stop after emitting this many total requests (0 = unlimited). Useful for fixed-count time measurements." env:"BENCHMARK_TOTAL"`
	ErrorRateLimit             float64  `long:"error-rate-limit" description:"DEPRECATED: no-op. Retained for backwards compatibility. See --max-consecutive-failures." default:"0.10" env:"BENCHMARK_ERROR_RATE_LIMIT"`
	MaxConsecutiveFailures     int      `long:"max-consecutive-failures" description:"Abort the benchmark after this many request failures in a row (any successful request resets the counter). Default 512 — high enough that brief server overload recovers before triggering." default:"512" env:"BENCHMARK_MAX_CONSECUTIVE_FAILURES"`
	MaxTotalErrors             int      `long:"max-total-errors" description:"Abort the benchmark after this many TOTAL request errors (monotonic; NOT reset by successes). 0 = disabled. Set to 1 to stop on the first error for investigation." default:"0" env:"BENCHMARK_MAX_TOTAL_ERRORS"`
	EndpointOverloadThreshold  float64  `long:"endpoint-overload-threshold" default:"1.2" env:"BENCHMARK_ENDPOINT_OVERLOAD_THRESHOLD" description:"Multi-endpoint specs only (a|b|c): a series sticks to its assigned endpoint until that endpoint's in-flight requests exceed this multiple of its fair share (total concurrency / endpoints), then the request fails over to a deterministically hash-selected endpoint that is not overloaded. 1.2 = default."`
	HotSeriesConcurrency       int      `long:"hot-series-concurrency" default:"0" env:"BENCHMARK_HOT_SERIES_CONCURRENCY" description:"H of the --series workers run as a 'hot' pool with a dedicated gate so they issue back-to-back requests; the rest share --concurrency. 0 = no hot pool."`
	RequestTimeout             string   `long:"request-timeout" description:"Per-request timeout (e.g. 5m, 2m). Default 5m." default:"5m" env:"BENCHMARK_REQUEST_TIMEOUT"`
	Step                       string   `long:"step" description:"Token step size for incremental prefix growth per request (e.g., 3k, 3000). Simulates agentic context growth. 0 or empty = disabled." env:"BENCHMARK_STEP"`
	StepStartingTokens         string   `long:"step-starting-tokens" description:"Initial prompt-prefix token size for each series when --step is enabled (e.g. 50000, 50k). Default 0 = start at --step value." env:"BENCHMARK_STEP_STARTING_TOKENS"`
	SharedPrefixPerSeries      int      `long:"shared-prefix-per-series" description:"Every N series share the same prefix content (unique session ID appended). Tests cross-series cache reuse. 0 = disabled." default:"0" env:"BENCHMARK_SHARED_PREFIX_PER_SERIES"`
	GlobalCacheHitRateTarget   float64  `long:"global-cache-hit-rate-target" description:"Do not spawn new series until global (all-time) cache hit rate reaches this threshold (0 = disabled)." default:"0" env:"BENCHMARK_GLOBAL_CACHE_HIT_RATE_TARGET"`
	MaxOutputTokens            int      `long:"max-output-tokens" description:"Override max output tokens for all requests (0 = use model spec default)." default:"0" env:"BENCHMARK_MAX_OUTPUT_TOKENS"`
	ExhaustSessions            bool     `long:"exhaust-sessions" description:"When a series' --step prefix reaches the token cap, recycle the slot with a fresh session GUID instead of pinning at 100% cache on a stable prefix. New cold session continues in the same slot, naturally maintaining --global-cache-hit-rate-target as old sessions retire. Requires --step (directly or via --profile=agentic, which enables this by default)." env:"BENCHMARK_EXHAUST_SESSIONS"`
	Profile                    string   `long:"profile" description:"Preset configuration profile. 'agentic': step=3k, shared-prefix-per-series=4, global-cache-hit-rate-target=0.95, tokens=200000, max-output-tokens=32000, exhaust-sessions=true." env:"BENCHMARK_PROFILE"`
	FromDataset                string   `long:"from-dataset" description:"Replay conversations from a dataset instead of synthetic prompts (short name, e.g. 'hermes-lambda'). Each series = one conversation; when done the slot pulls the next from the queue until the queue drains." env:"BENCHMARK_FROM_DATASET"`
	ReplaySeries               int      `long:"replay-series" description:"Cap on number of conversations / sessions replayed. With --from-dataset: limits how many parquet rows are loaded. With --router-replay-file: caps the streaming producer to the first N sessions in the file (workers run them to completion, then exit — useful for sweeping just an initial slice for comparison against the original capture)." env:"BENCHMARK_REPLAY_SERIES"`
	LimitContext               int      `long:"limit-context" description:"Replay only: skip any request whose capture-recorded prompt tokens (usage input + cache read/creation) exceed this limit. Use to avoid 400 storms when replaying long-session captures against a small-context model. 0 = no limit." env:"BENCHMARK_LIMIT_CONTEXT"`
	ReplayCharsPerToken        float64  `long:"replay-chars-per-token" description:"Router-replay only: sizes synthesized replay content as block Tokens x this many chars instead of the capture's byte sizes, so the serving tokenizer's counts land near the ORIGINAL capture's token counts. 0 = off (default: byte-faithful sizing from the capture's recorded Bytes)." default:"0" env:"BENCHMARK_REPLAY_CHARS_PER_TOKEN"`
	ReplayNoStamp              bool     `long:"replay-no-stamp" description:"Disable per-run <ignore>RUN_GUID</ignore> stamping in replay mode. By default each replay run prepends a fresh UUID to every request's system prompt (both --from-dataset and --router-replay-file paths) so server prefix caches from prior runs can't be reused — pristine per-run cache state." env:"BENCHMARK_REPLAY_NO_STAMP"`
	AbortOnCollapse            bool     `long:"abort-on-collapse" description:"Abort the benchmark if the windowed cache hit rate stays below 50% for 2 minutes. Off by default — this heuristic fires on legitimate workloads with low cache reuse (e.g. replay across many distinct conversations)." env:"BENCHMARK_ABORT_ON_COLLAPSE"`
	ReplayStopAtLowConcurrency bool     `long:"replay-stop-at-low-concurrency" description:"Terminate the replay run once the queue is drained AND the number of active worker goroutines has dropped below --concurrency. Avoids long-tail measurements where only a handful of long conversations remain and the gate is underutilized." env:"BENCHMARK_REPLAY_STOP_AT_LOW_CONCURRENCY"`
	RouterReplayFile           string   `long:"router-replay-file" description:"Path to a tree-aware replay file produced by 'wekai router replay-prepare'. Each series = one CLI session; within the session sub-agents fan out concurrently, honoring parent->child sequencing baked into the file. Mutually exclusive with --from-dataset." env:"BENCHMARK_ROUTER_REPLAY_FILE"`
	ReplayOutputRatio          float64  `long:"replay-output-ratio" description:"Router-replay only: retarget each request's max_tokens to InputTokens * this ratio, overriding the recorded original output_tokens (which otherwise pins max_tokens to what the model produced in the original capture, so the model stops almost immediately on replay). 0 = off (default: use original output_tokens/max_tokens)." default:"0" env:"BENCHMARK_REPLAY_OUTPUT_RATIO"`
	ReplayNaturalOutput        bool     `long:"replay-natural-output" description:"Let the model stop generation naturally instead of forcing it to fill max_tokens (disables the continue-generating instruction and vLLM ignore_eos). Router-replay only." env:"BENCHMARK_REPLAY_NATURAL_OUTPUT"`
	RouterReplayRoles          string   `long:"router-replay-roles" description:"Comma-separated list of instance roles to replay (default: all). E.g. 'main,sub-agent' excludes the CLI's background helper:title / helper:summarize / ephemeral (haiku side-calls) instances and keeps only the agentic workload — gives a cleaner 'in_flight ~= series' steady state. Other values: 'helper-or-isolated', 'ephemeral (no system)', 'other'." env:"BENCHMARK_ROUTER_REPLAY_ROLES"`
	RouterReplaySeriesIndices  string   `long:"replay-series-indices" description:"Comma-separated list of 0-based session indices to replay from --router-replay-file (e.g. '3,7,42'). Only sessions at those line positions (0 = first session after the header) are dispatched; others are skipped. Mutually exclusive with --replay-series-range. Overrides --replay-series." env:"BENCHMARK_REPLAY_SERIES_INDICES"`
	RouterReplaySeriesRange    string   `long:"replay-series-range" description:"Inclusive range of 0-based session indices to replay from --router-replay-file (e.g. '0-50' or '100-199'). Mutually exclusive with --replay-series-indices. Overrides --replay-series." env:"BENCHMARK_REPLAY_SERIES_RANGE"`
	DryRun                     bool     `long:"dry-run" description:"Dry run: skip remote HTTP requests; drive the router-replay pipeline with synthetic timing so gcache evolution can be observed offline. Requires --router-replay-file." env:"BENCHMARK_DRY_RUN"`
	DryRunColdTPS              int      `long:"dry-run-cold-tps" default:"1000000" description:"Dry-run: cold (uncached) input tokens processed per second." env:"BENCHMARK_DRY_RUN_COLD_TPS"`
	DryRunWarmTPS              int      `long:"dry-run-warm-tps" default:"10000000" description:"Dry-run: warm (cached) input tokens processed per second." env:"BENCHMARK_DRY_RUN_WARM_TPS"`
	DryRunOutputTPS            int      `long:"dry-run-output-tps" default:"100000" description:"Dry-run: output tokens generated per second." env:"BENCHMARK_DRY_RUN_OUTPUT_TPS"`
	CacheSimChunkBytes         int      `long:"cache-sim-chunk-bytes" description:"Chunk size in bytes for the content-level cache estimator (0 = default 1024)." default:"0" env:"BENCHMARK_CACHE_SIM_CHUNK_BYTES"`
	RandomGateOrder            string   `long:"random-gate-order" choice:"true" choice:"false" default:"true" optional:"yes" optional-value:"true" description:"Wake the concurrency gate's waiting series in uniformly random order when oversubscribed (the DEFAULT). Strict FIFO forces every series to wait behind all other waiting series before its next turn -- the adversarial worst case for GPU prefix-cache LRU. Pass --random-gate-order=false for the legacy exact-FIFO order. Cold-start waiters are unaffected (always served first, FIFO)." env:"BENCHMARK_RANDOM_GATE_ORDER"`

	// Positional arguments
	Args struct {
		Question string `positional-arg-name:"question" description:"Question to ask (overrides default)"`
	} `positional-args:"yes"`
}

// BenchmarkVisualizeOptions contains options for the benchmark visualize subcommand
type BenchmarkVisualizeOptions struct {
	Concurrency int    `long:"concurrency" description:"Override the concurrency used for the moving-average window (window = concurrency*3). Normally unnecessary: runs record their own concurrency in the reqdata JSONL and each arm is sized from its own. Use only for data recorded before run params were saved, or to force a different smoothing window." default:"0"`
	MaxElapsed  string `long:"max-elapsed" description:"Drop records past this elapsed time from each run's own start (per input directory/arm, not global wall-clock) -- e.g. 7h45m, 465m, 27900s. Also truncates vllm_metrics_sample rows so the cache-mix overlay, ingest volume, and dataset rows stop at the cutoff. Use to strip a crashed run's terminal error-storm from the report."`
	Args        struct {
		Dir string `positional-arg-name:"directory" description:"Directory containing .jsonl request data files" required:"yes"`
	} `positional-args:"yes"`
}

// BenchmarkVisualizeMergeOptions contains options for the benchmark visualize-merge subcommand
type BenchmarkVisualizeMergeOptions struct {
	All         bool   `long:"all" description:"Treat the first argument as a parent directory and include all subdirectories"`
	Output      string `long:"output" short:"o" description:"Output directory for merged results (default: auto-generated next to input)"`
	Concurrency int    `long:"concurrency" description:"Override the concurrency used for the moving-average window (window = concurrency*3). Normally unnecessary: runs record their own concurrency in the reqdata JSONL and each arm is sized from its own. Use only for data recorded before run params were saved, or to force a different smoothing window." default:"0"`
	Labels      string `long:"labels" description:"Comma-separated labels for each input directory, in positional order (overrides auto-detected model aliases / directory names). Count must exactly match the number of directories (post --all expansion)."`
	MaxElapsed  string `long:"max-elapsed" description:"Drop records past this elapsed time from each run's own start (per input directory/arm, not global wall-clock) -- e.g. 7h45m, 465m, 27900s. Also truncates vllm_metrics_sample rows so the cache-mix overlay, ingest volume, and dataset rows stop at the cutoff. Use to strip a crashed run's terminal error-storm from the report."`
	Args        struct {
		Dirs []string `positional-arg-name:"directories" description:"Directories containing .jsonl request data files"`
	} `positional-args:"yes"`
}

// EvalSimpleToolOptions contains options for the eval simple-tool subcommand
type EvalSimpleToolOptions struct {
	Model       string `long:"model" description:"Model to evaluate (e.g., gemini/gemini-2.5-flash-native)" required:"yes" env:"BENCHMARK_MODEL"`
	TargetCount int    `long:"target" description:"Target count for increment chain (number of tool calls)" default:"10" env:"BENCHMARK_TARGET"`
}

// EvalCacheCoherencyOptions contains options for the eval cache-coherency-garbage-clean subcommand
type EvalCacheCoherencyOptions struct {
	Model       string `long:"model" description:"Model to evaluate" required:"yes" env:"BENCHMARK_MODEL"`
	Series      int    `long:"series" description:"Number of unique series (each with its own UUID list)" default:"1"`
	Concurrency int    `long:"concurrency" description:"Max concurrent requests" default:"1"`
	// GarbageCharacters and GarbageTokens intentionally have no `default` tag: a zero
	// value means "not explicitly set" and is resolved by resolveGarbageChars(), which
	// applies precedence + the real default (213000 characters). See resolveGarbageChars.
	GarbageCharacters int   `long:"garbage-characters" description:"Total garbage character count in the system prompt, literal characters (default: 213000). Takes precedence over the deprecated --garbage-tokens."`
	GarbageTokens     int   `long:"garbage-tokens" description:"DEPRECATED — use --garbage-characters instead. Approximate number of garbage tokens (N tokens ≈ N*4 characters); kept for backward compatibility and mapped internally to characters."`
	Seed              int64 `long:"seed" description:"PRNG seed for reproducible UUID and prompt generation (0 = random, default)" default:"0"`
	Total             int   `long:"total" description:"Total number of requests across all cycles (0 = use 2*series default). Each request picks a series in round-robin order; cycles repeat as needed for cache-warmth tests." default:"0"`

	// Adversarial injection flags: exercise conditions the happy-path eval never
	// creates (aborts / cache-resets mid-flight) so the KV-offload load-pin's
	// guarantees get tested, not just assumed. All default to 0 (disabled), which
	// reproduces the eval's original behavior identically.
	AbortFraction float64 `long:"abort-fraction" description:"Fraction (0.0-1.0) of requests to cancel mid-flight (HTTP context canceled / connection closed), simulating a client disconnect while vLLM is mid-prefill or mid-load (WAITING_FOR_REMOTE_KVS) — exercises the abort-during-load pin path (has_pending_load). Aborted requests have no response to check and are excluded from coherency scoring; only their count is reported. 0 = disabled (default)." default:"0"`
	AbortDelayMs  int     `long:"abort-delay-ms" description:"Milliseconds after sending an aborted request (see --abort-fraction) to wait before canceling it. 0 (default) = pick a random delay in [0, live cold/cycle-1 TTFT estimate) instead of a fixed value, so the cancel lands sometime during that request's expected prefill/load window." default:"0"`
	ResetEveryN   int     `long:"reset-every-n" description:"Every N completed requests, POST vLLM's dev-mode reset_prefix_cache endpoint (?reset_external=true) — requires the server was started with VLLM_SERVER_DEV_MODE=1 — injecting a cache reset while other requests may still be mid-load. Requires --model to be a dynamic/ vLLM endpoint spec; a no-op with a warning otherwise. 0 = disabled (default)." default:"0"`

	SharedPrefixPerSeries int `long:"shared-prefix-per-series" description:"Group series into cohorts of N that share one byte-identical leading garbage prefix (grouped by seriesIdx/N), so peers concurrently co-hit the same prefix-cache blocks — creating the cross-series cache reuse (peer-prefix-hit of a mid-scatter block) the fully-unique default never produces. Each series' unique UUID stamps still trail the shared prefix, so cross-contamination detection stays valid and per-series. 0 = disabled, every series fully unique (default)." default:"0"`

	MaxOutputMultiplier float64 `long:"max-output-multiplier" description:"Multiplier applied to the auto-computed expected UUID-list size when sizing the response max_tokens budget (max_tokens = multiplier x expected tokens). Reasoning models (e.g. Kimi) can exhaust max_tokens on reasoning before emitting the answer; raise this to give them room to think + answer. Default 3.0 preserves the eval's original fixed 3x sizing." default:"3.0"`

	FullMissingResponses bool `long:"full-missing-responses" description:"Print the FULL untruncated response content and thinking/reasoning text in the per-request failure dump, instead of the default truncated preview (200 chars content / 100 chars thinking). Opt-in — output is typically redirected to a log file, so verbose full dumps are fine. Default false preserves the current truncated behavior."`
}
