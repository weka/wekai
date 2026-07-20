package cli

// BenchmarkCommands groups the `benchmark` subcommand tree. Shared between
// the full `wekai` binary and the slim `wekai-benchmark` binary so the
// subcommand surface stays identical without duplicating the struct.
type BenchmarkCommands struct {
	Auto           BenchmarkAutoCommand           `command:"auto" description:"Auto-scaling benchmark to find max throughput"`
	Throughput     BenchmarkThroughputCommand     `command:"throughput" description:"Three-phase throughput benchmark (cold prefill, warm prefill, decode)"`
	Visualize      BenchmarkVisualizeCommand      `command:"visualize" description:"Generate HTML visualization from request data directory"`
	VisualizeMerge BenchmarkVisualizeMergeCommand `command:"visualize-merge" description:"Merge multiple result directories into a single visualization"`
}

func (b *BenchmarkCommands) Init() {
	b.Auto.BenchmarkAutoOptions = &BenchmarkAutoOptions{}
	b.Throughput.BenchmarkThroughputOptions = &BenchmarkThroughputOptions{}
	b.Visualize.BenchmarkVisualizeOptions = &BenchmarkVisualizeOptions{}
	b.VisualizeMerge.BenchmarkVisualizeMergeOptions = &BenchmarkVisualizeMergeOptions{}
}

// RouterCommands groups the `router` subcommand tree.
type RouterCommands struct {
	Serve         RouterServeCommand         `command:"serve" description:"Run the model-aware HTTP reverse proxy"`
	Redact        RouterRedactCommand        `command:"redact" description:"Convert raw router capture JSONL into redacted JSONL"`
	Analyze       RouterAnalyzeCommand       `command:"analyze" description:"Analyze router capture files and emit per-model analytics"`
	Tree          RouterTreeCommand          `command:"tree" description:"Reconstruct the agent tree (sessions, instances, parent->child edges, parallelism) from redacted captures"`
	ReplayPrepare RouterReplayPrepareCommand `command:"replay-prepare" description:"Convert a directory of redacted captures into one replay-friendly JSON file"`
	AnalyzeReplay RouterAnalyzeReplayCommand `command:"analyze-replay" description:"Simulate a replay or source capture offline and report expected cache hit ratio"`
}

func (r *RouterCommands) Init() {}

// EvalCommands groups the `eval` subcommand tree.
type EvalCommands struct {
	SimpleTool     EvalSimpleToolCommand     `command:"simple-tool" description:"Evaluate model tool-calling capability"`
	CacheCoherency EvalCacheCoherencyCommand `command:"coherency" alias:"cache-coherency-garbage-clean" description:"Evaluate cache coherency with garbage-padded prompts"`
}

func (e *EvalCommands) Init() {
	e.SimpleTool.EvalSimpleToolOptions = &EvalSimpleToolOptions{}
	e.CacheCoherency.EvalCacheCoherencyOptions = &EvalCacheCoherencyOptions{}
}
