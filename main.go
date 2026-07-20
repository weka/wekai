// Command wekai-core is a standalone binary exposing the benchmark, router,
// and eval subcommand trees.
package main

import (
	"fmt"
	"os"

	"github.com/jessevdk/go-flags"
	"github.com/weka/wekai/cli"
)

// Options defines the wekai-core command structure: benchmark, router, and
// eval subcommand trees plus global flags.
type Options struct {
	cli.GlobalOptions

	Benchmark cli.BenchmarkCommands `command:"benchmark" description:"Benchmark LLM models"`
	Router    cli.RouterCommands    `command:"router" description:"Router commands for LLM proxy and capture analysis"`
	Eval      cli.EvalCommands      `command:"eval" description:"Evaluate LLM model capabilities"`
}

func main() {
	var opts Options
	opts.Benchmark.Init()
	opts.Router.Init()
	opts.Eval.Init()

	parser := flags.NewParser(&opts, flags.Default)
	parser.ShortDescription = "wekai-core - LLM benchmarking, router/proxy, and replay toolkit"
	parser.LongDescription = `wekai-core exposes the benchmark, router, and eval subcommand trees against
any dynamic model spec (dynamic/<url>,type=anthropic|openai|...) or
openrouter/... spec. No static named-model registry is included — embed this
module in an application that registers llm.ResolveModel /
llm.LookupModelByIdentifier hooks for named-model support.`

	cli.SetGlobalOptions(&opts.GlobalOptions)

	_, err := parser.Parse()
	if err != nil {
		if flagsErr, ok := err.(*flags.Error); ok {
			if flagsErr.Type == flags.ErrHelp {
				os.Exit(0)
			}
		}
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
