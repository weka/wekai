# wekai-core

An LLM benchmarking, HTTP-proxy/router, and capture-replay toolkit for
Anthropic, OpenAI (Chat Completions and Responses), and Gemini-native APIs.

wekai-core ships three command groups:

- **benchmark** — throughput/latency benchmarking against embedded documentation,
  auto-scaling load tests, and result visualization.
- **router** — a model-aware HTTP reverse proxy with request/response capture,
  redaction, analysis, and replay simulation.
- **eval** — model capability evaluations (tool-calling, cache coherency).

It works against any endpoint reachable via a dynamic model spec
(`dynamic/<url>,type=anthropic|openai|openai_responses|gemini_native,...`) —
no static model registry is required. Applications that embed wekai-core can
inject their own named-model registry via the `llm.ResolveModel` and
`llm.LookupModelByIdentifier` hooks.

## Build

```
go build -o wekai-core .
```

## Usage

```
wekai-core benchmark embed --help
wekai-core router serve --help
wekai-core eval simple-tool --help
```

## Embedding

Applications that embed wekai-core's command groups directly (rather than
running the standalone binary) have two extension points, both in `cli/`:

- `cli.SetGlobalOptions(*GlobalOptions)` — for binaries that parse their own
  `cli.GlobalOptions` before calling `flags.Parse` (this is what `main.go`
  above does). go-flags mutates that struct in place, so by the time any
  command runs, wekai-core's internal `config.Config` is already in sync.
- `cli.PreExecute func(ctx context.Context) error` — for applications with
  their own richer global-options type that don't construct a
  `cli.GlobalOptions` at all. Set this once; it runs at the top of every
  embedded command's `Execute()`, before the command touches
  `config.Config`. Use it to copy the embedder's own parsed flags into
  `config.Config` and to register the model-registry hooks below. This lets
  `BenchmarkCommands`/`RouterCommands`/`EvalCommands` be embedded as plain
  type aliases with no per-command wrapping.

Either mechanism should also set `llm.ResolveModel` and
`llm.LookupModelByIdentifier` (see above) if the embedder has its own named
models — otherwise only dynamic/openrouter specs resolve.

## Deployment

`Dockerfile` builds a self-contained replay image: a golang builder stage
compiles the `wekai-core` binary, and a second `COPY --from=` stage embeds
one router-replay JSONL artifact at `/wekai/replay.jsonl`, pulled from a
separately-published scratch image (default
`quay.io/weka.io/wekai-benchmark:replay-<sha12>` — published by wekai's
`.dagger` `push_replay` / `task benchmark:push-replay`; the `wekai-benchmark`
repo name there is historical, where replay artifacts have always lived —
nothing built from this Dockerfile uses "benchmark" in its own name). Build
locally with:

```
task docker:build   # override REPLAY_IMAGE=... to embed a different capture
```

`chart/wekai-core/` is a run-once Helm chart (`; sleep infinity` after the
command, same pattern as wekai's retired `chart/benchmark`) that runs the
embedded replay directly — no other run mode is supported. The container
command is `wekai-core benchmark auto --router-replay-file
/wekai/replay.jsonl ...`, with values for `replay.models`,
`replay.replaySeries`, `replay.concurrency`, `replay.maxConcurrency`,
`replay.dryRun` (+ dry-run TPS knobs), timeouts, and an optional
`llmApiKeySecretName` envFrom secret. Results can optionally persist to a
PVC via `storeResults`. See `chart/wekai-core/values.yaml` for the full
list, or `helm show values chart/wekai-core`.

```
helm lint chart/wekai-core
helm template test chart/wekai-core --values my-values.yaml
```

## Layout

- `llm/` — raw LLM client layer (Anthropic, OpenAI, OpenAI Responses, Gemini
  native), dynamic model-spec parsing, cost calculation, mock servers for tests.
- `tools/` — base tool/toolset types shared by the LLM client layer and the
  benchmark tool-chain eval.
- `benchmark/` — benchmark and replay engine.
- `cli/` — go-flags command trees for `benchmark`, `router`, `eval`.
- `config/` — minimal global config and environment-based API key loading.

## License

Apache-2.0, see [LICENSE](LICENSE).
