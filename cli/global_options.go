package cli

import (
	"context"

	"github.com/weka/wekai-core/config"
)

// GlobalOptions contains global flags available to all wekai-core commands.
// Only the subset relevant to benchmark/router/eval is present — no
// bot/agent/sandbox/datastore flags, since those concepts don't exist here.
type GlobalOptions struct {
	GlobalModelOverride string `long:"global-model-override" description:"Global model override for all model types" env:"GLOBAL_MODEL_OVERRIDE"`
	TmpDir              string `long:"tmp-dir" description:"Temporary directory" default:"/tmp/wekai-core" env:"TMP_DIR"`
	LogFormat           string `long:"log-format" choice:"text" choice:"json" description:"Log format" default:"text" env:"LOG_FORMAT"`
	StderrLogs          bool   `long:"stderr-logs" description:"Log to stderr instead of file"`
	LogHTTPRequests     bool   `long:"log-http-requests" description:"Log HTTP requests and responses for debugging"`
	JSONOutput          bool   `long:"json-output" description:"Output structured JSON result"`
}

// Global options storage for access by command handlers.
var globalOptions *GlobalOptions

// SetGlobalOptions stores the global options for access by command handlers.
// For standalone binaries that parse a GlobalOptions of their own (core's
// own main.go, cmd/wekai-benchmark in wekai): call this before parsing —
// go-flags mutates the same struct SetGlobalOptions was pointed at, so by
// the time any command's Execute() runs, syncConfig() below sees the real
// parsed values.
//
// Applications that embed BenchmarkCommands/EvalCommands/RouterCommands
// directly, without ever constructing a GlobalOptions of their own (e.g.
// wekai, which has its own richer global-options type), should set
// PreExecute instead — see below.
func SetGlobalOptions(opts *GlobalOptions) {
	globalOptions = opts
}

// PreExecute, when set, runs at the very start of every command's Execute(),
// before any command reads config.Config or does command-specific work.
// nil by default (no-op).
//
// This is the extension point for applications that embed this package's
// command groups but manage their own configuration/global-options flow
// (e.g. wekai's internal/cli): set PreExecute once, to whatever setup needs
// to happen before a wekai-core command runs (registering model-registry
// hooks, copying the embedder's own parsed config into config.Config, ...),
// and every embedded command picks it up automatically — no per-command
// wrapping needed, so BenchmarkCommands/EvalCommands/RouterCommands can be
// plain type aliases in the embedder.
//
// Standalone binaries using SetGlobalOptions don't need this; syncConfig()
// already keeps config.Config current from the parsed GlobalOptions.
var PreExecute func(ctx context.Context) error

// runPreExecute is called at the top of every command's Execute(). It runs
// the PreExecute hook (if set) and then syncConfig(), so config.Config is
// current by the time command logic reads it — whichever mechanism
// (PreExecute or SetGlobalOptions) the embedder/binary uses.
func runPreExecute(ctx context.Context) error {
	if PreExecute != nil {
		if err := PreExecute(ctx); err != nil {
			return err
		}
	}
	syncConfig()
	return nil
}

// syncConfig copies globalOptions into config.Config, mirroring wekai's own
// per-command "set config before Init" boilerplate. Safe to call when
// globalOptions is nil (no-op) — e.g. when config.Config is instead being
// managed directly by an embedder's PreExecute hook.
func syncConfig() {
	if globalOptions == nil {
		return
	}
	config.Config.JSONOutput = globalOptions.JSONOutput
	config.Config.TmpDir = globalOptions.TmpDir
	config.Config.LogFormat = globalOptions.LogFormat
	config.Config.StderrLogs = globalOptions.StderrLogs
	config.Config.LogHTTPRequests = globalOptions.LogHTTPRequests
	if globalOptions.GlobalModelOverride != "" {
		config.Config.GlobalModelOverride = globalOptions.GlobalModelOverride
	}
}
