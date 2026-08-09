# wekai Navigation Index

> `.ainav/` is a navigation map and operational memory for this repo, mirroring the
> convention in the parent `wllm` repo's `.ainav/`. It captures dev workflows and
> commands that aren't obvious from the code alone — not prose documentation.
> Keep it updated as the project evolves.
>
> **No environment-specific values** (IPs, hostnames, pod names, credentials) belong
> here. Use placeholders (`<pod>`, `<model>`, `$VAR`) in snippets.

## What is this project?

**wekai** is Weka's LLM benchmarking/replay CLI and its routing gateway. One binary:

- `wekai` (root `main.go`, `cli/`) — capture/replay traffic, run coherency evals,
  benchmark real or replayed workloads against any OpenAI-compatible endpoint.
- `router/` — the routing gateway behind `wekai router serve`, fronting a vLLM fleet with
  prefix-cache-affinity routing, plus a GPU-less mock vLLM fleet (`cmd/mock-vllm`) for
  developing and load-testing router policies without hardware.

## Primary Reference Documents

| Doc | What it covers |
|-----|----------------|
| **[docs/router.md](../docs/router.md)** | The router: routes, pools, signals, the three supported deployments, CLI and Helm for each |
| **[README.md](../README.md)** | Build, CLI usage, replay benchmark, coherency eval, deployment — **read this first** |
| [CLAUDE.md](../CLAUDE.md) | Commit convention (Conventional Commits, gates releases), versioning, build/test policy |

## Navigation

| Topic | File | Description |
|-------|------|--------------|
| Router mock-testing | [router-testing/](router-testing/index.md) | End-to-end: build the router + mock fleet, run a replay benchmark against it |
| Live KV map (`/router-viz`) | [viz/](viz/index.md) | The router's live prefix-tree visualization on the metrics listener |
| Architecture | [architecture/](architecture/index.md) | `kvcache`, `router/internal/policy/affinity`, `router/internal/mockvllm`, `router/hack` fences |

## Key Directories (this repo)

| Directory | Purpose |
|-----------|---------|
| `kvcache/` | Shared prefix-cache trie engine (stdlib-only leaf). Used by the benchmark and the mock vLLM; NOT by the router, which owns its own shared tree. |
| `router/serve/` | The router entrypoint (`wekai router serve`); the only part of `router/` visible outside it |
| `router/cmd/mock-vllm/` | GPU-less mock vLLM server binary entrypoint |
| `router/internal/` | Router packages: `gateway` (HTTP surface), `policy`/`policy/affinity` (the one routing flow + its signals), `registry`/`lease` (backend state, in-flight leases), `proxy` (upstream forwarding), `dialect/openai` (wire format), `mockvllm` (mock engine), `viz` (`/router-viz`), `hack` (architectural fence tests) |
| `benchmark/` | Benchmark and replay engine (`wekai benchmark auto`, router-replay mode) |
| `cli/` | go-flags command trees for `benchmark`, `router`, `eval` |
| `llm/` | Raw LLM client layer, dynamic model-spec parsing, mock servers for tests |

## Build Config (this repo)

- Go module: `github.com/weka/wekai`. `task build`, `task test`, `task docker:build` (root `Taskfile.yaml`).
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) is **mandatory** — CI derives the next semver release from commit types on every push to `main`. See [CLAUDE.md](../CLAUDE.md).
- Testing policy: no mocked LLM/chat flows — pure unit tests on data-transformation functions, `httptest` for HTTP utilities, real-endpoint e2e tests only.
