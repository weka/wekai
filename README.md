# wekai

An LLM benchmarking, HTTP-proxy/router, and capture-replay toolkit for
Anthropic, OpenAI (Chat Completions and Responses), and Gemini-native APIs.

wekai ships three command groups:

- **benchmark** — throughput/latency benchmarking against embedded documentation,
  auto-scaling load tests, and result visualization.
- **router** — a model-aware HTTP reverse proxy with request/response capture,
  redaction, analysis, and replay simulation.
- **eval** — model capability evaluations (tool-calling, cache coherency).

It works against any endpoint reachable via a dynamic model spec
(`dynamic/<url>,type=anthropic|openai|openai_responses|gemini_native,...`) —
no static model registry is required. Applications that embed wekai can
inject their own named-model registry via the `llm.ResolveModel` and
`llm.LookupModelByIdentifier` hooks.

## Build

```
go build -o wekai .
# or: go install github.com/weka/wekai@latest
```

## Usage

```
wekai benchmark auto --help
wekai router serve --help
wekai eval simple-tool --help
```

## Embedding

Applications that embed wekai's command groups directly (rather than
running the standalone binary) have two extension points, both in `cli/`:

- `cli.SetGlobalOptions(*GlobalOptions)` — for binaries that parse their own
  `cli.GlobalOptions` before calling `flags.Parse` (this is what `main.go`
  above does). go-flags mutates that struct in place, so by the time any
  command runs, wekai's internal `config.Config` is already in sync.
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
compiles the `wekai` binary (module github.com/weka/wekai), and a second `COPY --link --from=` stage
embeds one router-replay JSONL artifact at `/wekai/replay.jsonl`, pulled
from a separately-published scratch image (default
`quay.io/weka.io/wekai:replay-<sha12>` — published by this repo's own
`task replay:push`, see "CI / Publishing" below; replay artifacts and the
app image share the wekai quay repo, distinguished by the replay- tag prefix). Both
runtime-stage `COPY`s use BuildKit's `--link` — each becomes an
independent, content-addressed layer, so a rebuild triggered by a Go source
change (which busts the builder stage) does not recopy or reupload the
replay layer, which can be several GB; its digest stays identical across
rebuilds and registries can cross-mount the existing blob instead of
re-uploading it. Build locally with:

```
task docker:build   # override REPLAY_IMAGE=... to embed a different capture
```

`chart/wekai/` is a run-once Helm chart that runs the embedded replay
directly — no other run mode is supported. The container command is
`wekai benchmark auto --router-replay-file /wekai/replay.jsonl ...`. After
the benchmark completes (or fails) the pod sleeps forever so results stay
explorable via `kubectl exec`/`kubectl cp`; delete the pod and the
Deployment restarts the benchmark. Per-request JSONL data plus an
auto-generated `report.html` visualization are always written under
`resultsMountPath/<run-timestamp>/` (`--save-request-data`) — set
`storeResults=true` to back that path with a PVC instead of a pod-local
emptyDir. The chart is deliberately minimal: the only value most installs
need to set is `endpoint`, the target model server; everything else has a
working default (all sessions replayed by 256 parallel series workers at
fixed concurrency 28 with a 4-worker hot pool).

| Value | Default | Purpose |
|---|---|---|
| `endpoint` | `""` | Target model server — the one value most installs set. Bare URL (`http://host:8000`, autodiscovers `/v1` + model id) or a full `dynamic/...,type=...` spec |
| `model` | `""` | Optional explicit model id (skips autodiscovery); appended as `,model=<v>` |
| `duration` | `8h` | Benchmark run length (maps to `--timeout`); e.g. `3m` for smoke tests |
| `imageRepository` / `imageTag` | `quay.io/weka.io/wekai` / `""` | Image; tag defaults to the chart's `appVersion` (see "Releases") |
| `imagePullSecrets` | `[]` | Pull secrets for the private quay repo |
| `replay.replaySeries` | `0` | Subset cap on which sessions get replayed (0 = all 5k+ in the embedded file) |
| `replay.series` | `256` | Parallel series workers replaying sessions (`--series`) |
| `replay.replayRoles` | `""` | Comma-separated instance roles to replay (empty = all) |
| `replay.concurrency` | `28` | Fixed concurrency (0 = auto hill-climber) |
| `replay.hotConcurrency` | `4` | Hot-pool workers with a dedicated gate (`--hot-series-concurrency`; 0 = off) |
| `replay.maxConcurrency` | `0` | Hill-climber upper bound, only relevant when `concurrency=0` |
| `replay.maxOutputTokens` | `0` | Override per-request output budget (0 = replay-file budgets) |
| `replay.requestTimeout` | `""` | Per-request timeout, e.g. `5m` |
| `replay.replayNoStamp` | `false` | Disable per-run cache-busting stamp (bitwise-faithful replay) |
| `replay.abortOnCollapse` / `replay.replayStopAtLowConcurrency` | `false` | Early-stop behaviors |
| `replay.dryRun` + `replay.dryRun{Cold,Warm,Output}TPS` | `false` / `0` | Synthetic timing mode, no real LLM calls |
| `replay.stderrLogs` | `false` | Log to stderr |
| `llmApiKeySecretName` | `""` | K8s secret with LLM API-key env vars (for endpoints that need auth) |
| `storeResults` / `storageSize` / `storageClassName` / `resultsMountPath` | `false` / `10Gi` / `""` / `/results` | Results (JSONL + `report.html`) are always saved under `resultsMountPath`; `storeResults=true` backs it with a PVC, otherwise emptyDir |
| `resources` | 256Mi/250m → 4Gi/4 | Pod resources |

Authoritative list with inline docs: `chart/wekai/values.yaml`
(or `helm show values oci://quay.io/weka.io/helm/wekai --version <vX>`).

Charts are published to `oci://quay.io/weka.io/helm/wekai`. The chart
`--version` is mandatory (versions are `v999.0.0-<sha12>` prerelease stamps,
so Helm never auto-picks a "latest") and it pins the image purely by
propagation: `push-helm` stamps `Chart.yaml`'s `version`/`appVersion` in
lockstep with the image it just pushed, and the deployment template resolves
the image tag via `imageTag | default .Chart.AppVersion` — no version is
hardcoded in the packaged values. `helm show chart` alone tells you exactly
which image a chart version runs.

Get the concrete `<vX>` for the snippets below from the
[releases page](https://github.com/weka/wekai/releases) — every release
description includes these commands pre-filled with its own version.

Default install — runs for the default duration (8h):

```
helm install my-replay oci://quay.io/weka.io/helm/wekai \
  --version <vX> --set endpoint=http://10.71.0.4:8000
```

Smoke test — shorten `duration` (maps to `--timeout`), e.g. 3 minutes:

```
helm install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set duration=3m
```

Explicit model override — by default the model id is autodiscovered (see
"Bare-URL model selector" below); set `model` to skip discovery:

```
helm install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set model=nvidia/Kimi-K2.6-NVFP4
```

`endpoint` accepts any dynamic model spec, not just a bare URL — e.g. to
target an Anthropic-shaped server, append `,type=anthropic`. Because Helm's
`--set` splits on unescaped commas, either escape it or use `--set-string`
with a values file instead:

```
helm install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set-string endpoint='http://10.71.0.4:8000\,type=anthropic' \
  --set duration=3m
```

Private registry — `quay.io/weka.io/wekai` requires auth to pull; create
a `kubernetes.io/dockerconfigjson` secret once and reference it via
`imagePullSecrets`:

```
kubectl create secret docker-registry quay-pull \
  --docker-server=quay.io \
  --docker-username=<user> --docker-password=<token>

helm install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set 'imagePullSecrets[0].name=quay-pull'
```

For local development installs from the chart directory, pass the image tag
explicitly (the in-tree `Chart.yaml` carries a placeholder `appVersion`):
`helm install my-replay chart/wekai --set imageTag=<vX> --set endpoint=...`

```
helm lint chart/wekai
helm template test chart/wekai --set endpoint=http://10.71.0.4:8000
```

### Bare-URL model selector

A bare `http://` or `https://` URL passed as `--model`/`--models` (or, in the
chart, `endpoint`) is promoted by `llm.NormalizeModelSpec` to a
`dynamic/<url>,type=openai_vllm` spec — no need to spell out the dynamic
model boilerplate for the common case. From there `ParseDynamicModel` /
`GetChatGetter` autodiscover, against the endpoint itself, whatever the spec
didn't already say:

- **`/v1` path** — if the URL has no path (e.g. `http://host:8000`), a
  `GET <url>/v1/models` probe checks whether the server answers there; on
  success, `<url>/v1/` becomes the effective base for all requests. A URL
  that already ends in `/v1` (or carries any other explicit path) is left
  exactly as given — no probe.
- **Model id** — if `model=` is absent from the spec, the first entry in
  that same `/v1/models` response (`data[0].id`) is used as the model id.
  For `type=anthropic` specifically, a model id is mandatory for the actual
  request to succeed against a real Anthropic-compatible server, so failed
  autodiscovery there is a hard error (specify `model=` explicitly) rather
  than the softer `"default"` placeholder fallback used for other types.
- **type=anthropic works the same way** — `http://host:8000,type=anthropic`
  autodiscovers identically; only the request-shaping client differs.

Discovery is memoized per distinct raw endpoint for the life of the process,
not per request: `GetChatGetter` (and therefore this resolution) runs fresh
on every request in several benchmark code paths, so without memoization a
concurrent benchmark run would re-probe the endpoint on every single
request. The underlying network probe(s) fire exactly once per endpoint no
matter how many times or how concurrently the spec is resolved.

## Releases

**Install commands for every version live on the
[releases page](https://github.com/weka/wekai/releases)** — each release
description carries copy-paste `helm install`, `docker pull`, and
`go install` snippets pinned to that exact version. Pick a release there and
substitute its version wherever this README says `<vX>`.

Pushes to `main` cut releases automatically (`.github/workflows/release.yml`):
the next semver is derived from Conventional Commits (mandatory — see
CLAUDE.md), the image and chart are published under that `vX.Y.Z` (overriding
the local content-hash scheme below), and the GitHub Release is created with
those per-version install instructions.

## CI / Publishing

Publishing is a self-contained Dagger module (Python SDK, engine pinned to
`v0.18.6` to match wekai's own `.dagger` module) rooted at `dagger.json` /
`.dagger/src/wekai_core_flows/`. It was ported from wekai's
`.dagger/src/wekai_flows/main.py` (`push_replay`, `_calc_version`) — same
tag scheme, same default registry, no reimplementation via crane/shell.
Only its Go dependencies are needed to build the image, and they're all
public, so unlike wekai's module this one needs no SSH socket forwarding
for private-repo access.

Three functions:

- `push-replay` — publishes a replay JSONL as a minimal scratch image,
  tagged `replay-<sha12>` (sha256 of the file), to
  `quay.io/weka.io/wekai` by default, alongside the app image — replay
  artifacts are distinguished by the `replay-` tag prefix (registry
  overridable).
- `publish` — builds this repo's own `Dockerfile` via `Directory.docker_build()`
  (the Dockerfile is the single source of truth — the module does not
  reimplement its steps), tags with `v999.0.0-<sha12>` (sha12 of the
  source directory's content digest, same scheme wekai uses), and
  publishes to `quay.io/weka.io/wekai` by default. The Dockerfile's
  `REPLAY_IMAGE` build-arg is exposed as a `--replay-image` function param.
- `push-helm` — first runs the exact same image build+publish as `publish`
  (shared internal helper, not a separate code path), then packages
  `chart/wekai` and pushes it to an OCI Helm registry
  (`quay.io/weka.io/helm` by default). Version pinning is pure propagation
  (same pattern as wekai's `push_restricted` charts): only `Chart.yaml`'s
  `version`/`appVersion` are stamped — in lockstep with the image the
  publish step just pushed — and the deployment template resolves the image
  tag via `imageTag | default .Chart.AppVersion`; the packaged `values.yaml`
  carries no hardcoded version (`imageTag` stays `""`, and only
  `imageRepository` is synced to the actual push registry). A `helm install`
  of a chart version with zero further `--set` flags always deploys the
  exact image published under that same version. Because the image publish
  is `await`-ed before any chart packaging happens, "image pushed before
  chart push" holds by construction, not by convention. Takes
  `helm-username`/`helm-password` as Dagger secrets.

```
task replay:push REPLAY=/path/to/replay.jsonl   # dagger call push-replay
task app:push                                    # dagger call publish (image only)
task helm:push                                   # dagger call push-helm (image + chart, image first)
```

`helm:push` reads Helm registry credentials from `QUAY_USERNAME`/`QUAY_PASSWORD`
env vars (same convention wekai's retired chart-push flow used).

`.dagger/sdk/` (the generated Dagger client bindings) is gitignored, same
as wekai — run `dagger develop` once after a fresh clone (or the first
`dagger call`/`task app:push` will do it implicitly) to regenerate it
locally; nothing under `sdk/` is meant to be hand-edited or committed.

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
