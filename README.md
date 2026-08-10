# wekai

> **Active development — no API stability guarantees.** CLI flags, Helm
> values, and Go APIs are all subject to change between releases. Getting
> the defaults and the functionality right is a higher priority here than
> keeping a stable interface, and defaults do change. **Compare runs only
> head to head on the same version** — numbers from different versions are
> not comparable, since a defaults change can move them on its own.

An LLM benchmarking, HTTP-proxy/router, and capture-replay toolkit for
Anthropic, OpenAI (Chat Completions and Responses), and Gemini-native APIs.

wekai ships three command groups:

- **benchmark** — throughput/latency benchmarking against embedded documentation,
  auto-scaling load tests, and result visualization.
- **router** — a model-aware LLM router with prefix-cache affinity across a vLLM
  fleet, plus request/response capture, redaction, analysis and replay
  simulation. See **[docs/router.md](docs/router.md)**.
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

## Run the router

Route every model across a vLLM fleet, choosing among endpoints by prefix-cache
affinity so a conversation keeps landing on the backend that already holds its
KV cache:

```
wekai router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --backends 'http://vllm-a:8000|http://vllm-b:8000|http://vllm-c:8000'
```

Endpoints are pipe-separated, the same syntax the client uses for a
multi-endpoint model. One endpoint is a plain proxy; several get affinity.

Two further shapes are supported, both covered in
**[docs/router.md](docs/router.md)** with CLI and Helm for each:

- **Per-model routes** — `--route 'big,70b => http://a:8000|http://b:8000'`.
  Routing is by model name, so OpenAI-format and Anthropic-format requests
  follow the same rules.
- **Self-hosted models with a hosted fallback** — named models to your fleet,
  everything else through `--default https://api.anthropic.com`, so one base URL
  serves both.

### Fronting a single local endpoint

Front a local model endpoint on `http://127.0.0.1:25201` with the router on
`http://127.0.0.1:25202`, sending every model to that one upstream:

```
wekai router serve \
  --listen 127.0.0.1:25202 \
  --default 'http://127.0.0.1:25201' \
  --strip-auth-when '*'
```

- `--default` is the catch-all rule (`* => <upstream>`): every requested model
  name matches. The request path and query are appended to the upstream, so a
  `POST /v1/messages` is forwarded to
  `http://127.0.0.1:25201/v1/messages`.
- `--strip-auth-when '*'` drops the inbound `Authorization`/`x-api-key`
  headers, which unauth'd local servers (vLLM, llama.cpp, …) otherwise reject.
- The model name is handled for you. Claude Code sends `claude-*`, which a
  single-model server doesn't recognize — vLLM answers `POST /v1/messages`
  with a bare `404`. On startup the router asks each upstream that has no
  explicit `as <model>` what it serves (`GET <upstream>/v1/models`); if the
  answer is exactly one model, every matching request's `model` is rewritten
  to it:

  ```
  auto-model: rewriting every matching request's model to what the pool serves pool=default model=zai-org/GLM-5.2-FP8
    route: * => http://127.0.0.1:25201 as zai-org/GLM-5.2-FP8 (auto-discovered) (strip-auth)
  ```

  A multi-model upstream is left alone, so this can't silently collapse a
  real fan-out onto one model. Control it with `--auto-model`: `auto`
  (default), `force` (always take the first listed model), `off` (never
  probe).

  The question is asked of the *pool*, not of one endpoint, and only once a
  backend has passed a health check — so a pool whose first endpoint is dead
  still resolves from any sibling that answers, and a backend still loading
  weights is simply waited for. The answer takes effect the moment it lands,
  including for pools whose members arrive later from pod discovery.
- Append `as <model>` to pin the rewrite explicitly and skip discovery:
  `--default 'http://127.0.0.1:25201 as my-local-model'`.
- Use repeated `--route '<substr>,<substr> => <upstream>[ as <model>]'` rules
  (first match wins) when you want per-model fan-out instead of one catch-all.
- Add `--capture redacted` to record traffic under
  `~/.wekai/router/capture/redacted/` for later `wekai router replay-prepare`.

`/healthz`, `/livez`, and `/metrics` are served on the same listener.

### Pointing Claude Code at it

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:25202
export ANTHROPIC_AUTH_TOKEN=dummy   # any non-empty value — stripped by --strip-auth-when '*'
claude
```

With `--user-prefix` on the router, each client tags itself via the first path
segment, and requests/captures/log lines carry that user:

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:25202/alice
```

## Replay benchmark

`wekai benchmark auto --router-replay-file` replays previously captured
agentic traffic against a target endpoint — the same run mode the Helm
chart below wraps. Canonical run (the chart's exact defaults, expressed as
flags):

```
wekai benchmark auto \
  --router-replay-file replay.jsonl \
  --models http://YOUR-LLM-HOST:8000 \
  --series 256 --concurrency 28 --hot-series-concurrency 4 \
  --timeout 8h \
  --save-request-data ./results
```

**Where the replay file comes from.** `wekai router serve` proxies live
LLM traffic and captures it (redacted); `wekai router replay-prepare`
compiles those captures into a single replay-v3 JSONL file — one header
line plus one session per line, so the replayer streams it with a bounded
queue regardless of file size. Each request embeds the original structured
spec (system blocks, tools, messages) with content regenerated
deterministically from each block's hash: same hash → same bytes → the
server's prefix cache hits exactly the way the original traffic did,
without shipping any real captured text.

**Downloading it.** The capture embedded in the image and run by the Helm
chart is published as a scratch image holding just `/replay.jsonl`, pulled
anonymously with `curl` — no Docker engine, no registry client (~1.3 GB
download, ~9 GB on disk). `Dockerfile`'s `REPLAY_IMAGE` ARG is the source
of truth for the tag:

```
REPO=weka.io/wekai; TAG=replay-24e7f15ba0ea
TOKEN=$(curl -s "https://quay.io/v2/auth?service=quay.io&scope=repository:$REPO:pull" | jq -r .token)
LAYER=$(curl -s -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
  "https://quay.io/v2/$REPO/manifests/$TAG" | jq -r '.layers[0].digest')
curl -L -H "Authorization: Bearer $TOKEN" "https://quay.io/v2/$REPO/blobs/$LAYER" | tar -xz replay.jsonl
```

**What "replay" preserves.** This is not a request firehose — it re-enacts
each captured session as a tree: per-session series boundaries, sequential
turn order per agent instance, parent→child sequencing (a sub-agent starts
only after its parent's spawning request completes), sibling fan-out
concurrency (K spawned agents run in parallel), and per-request
input/output token budgets. Prefix growth and sub-agent bursts therefore
hit the endpoint's KV/prefix cache with the same shape as real agentic
load.

**The knobs, and how they differ:**

- `--series 256` — how many parallel workers replay sessions; each worker
  pulls the next session from the stream and runs its full tree to
  completion. Not to be confused with…
- `--replay-series N` — a *subset cap* on which sessions get replayed at
  all (0/omitted = every session in the file).
- `--concurrency 28` — the gate on simultaneously in-flight HTTP requests.
  During fan-out moments a session tree can want more slots than its one
  worker; the gate queues the surplus, absorbing bursts.
- `--hot-series-concurrency 4` — carves 4 of the workers into a "hot" pool
  with its own dedicated gate, so a few sessions issue back-to-back
  requests at full speed while the rest share the main gate — a
  foreground-agents-plus-background-fleet traffic mix.
- By default each run injects a per-run `<ignore>RUN_GUID</ignore>` stamp
  ahead of every prompt, so a rerun starts against a pristine prefix cache
  while within-run cross-session cache hits still occur;
  `--replay-no-stamp` disables it for bitwise-faithful replay.

**Results.** `--save-request-data` writes per-request JSONL under
`./results/<run-timestamp>/` and auto-generates an interactive
`report.html` scatter-plot (TTFT / response time / cache hits over time)
there at the end of the run. Regenerate or combine runs later with
`wekai benchmark visualize <dir>` and `wekai benchmark visualize-merge`.

**Server-side cache-source sampling.** When `--save-request-data` is on and
the model spec points at an OpenAI-compatible (chat/completions) endpoint,
a sampler polls the server's Prometheus endpoint — `<base>/metrics`, with
the `/v1` API suffix stripped — once a minute and appends the result to the
same JSONL. This is what splits the report's prompt tokens into compute vs
local cache vs external cache.

Exactly one series is read — `vllm:prompt_tokens_by_source_total` — keeping
the three `source` label values `local_compute`, `local_cache_hit`, and
`external_kv_transfer`. Values are summed over every other label
(`model_name`, engine index); nothing else from `/metrics` is retained. The
`_total` is `prometheus_client` appending the counter suffix to the family
declared as `vllm:prompt_tokens_by_source`, which is why the bare name still
appears on the `# HELP`/`# TYPE` lines. Each sample is one JSONL row with
`record_type: "vllm_metrics_sample"` and fields `ts`, `model`,
`sources.compute` / `sources.local_cache` / `sources.external_cache`, plus
the locally-computed `active_dataset_tokens` and `active_series`.

Collection is best-effort and never affects the benchmark: an unreachable
or non-Prometheus endpoint just skips samples. Since any chat/completions
endpoint is probed — it may well be vLLM — a spec that didn't say
`type=openai_vllm` gives up after 3 consecutive failures rather than
polling a public API for the rest of the run. Spelling out
`type=openai_vllm` asserts the server *is* vLLM and keeps it polling
through restarts and slow weight loads.

## Cache coherency eval

`wekai eval coherency` verifies that an inference server's prefix/KV cache
returns *correct* bytes, not just *fast* ones. Each series gets a large
garbage-padded system prompt (~213k characters by default) with a list of
unique UUID stamps scattered through it; the model's only job is to echo
back exactly its own comma-joined UUID list. Any deviation is evidence of
cache corruption, not model weakness.

Canonical run:

```
wekai eval coherency \
  --model dynamic/http://YOUR-LLM-HOST:8000,type=openai_vllm \
  --series 256 --shared-prefix-per-series 4 --abort-fraction 0.1
```

**Cycles: every series is sent twice.** With `--total` unset the default is
`2 × series` requests — e.g. `--series 128` issues 256 requests total:
cycle 1 sends each of the 128 unique prompts cold (full prefill), cycle 2
re-sends the identical prompts so they should be served from cache. The
report prints **mean cold TTFT (cycle 1)** vs **mean warm TTFT (cycle 2)**
and an implicit cache hit rate (a cycle-2 request counts as a hit when its
TTFT is ≤ 50% of the cycle-1 mean) — so you see in one run both that the
cache *works* (warm ≪ cold) and that it's *coherent* (the checks below).

**What the coherency report means.** Two pass/fail tests plus a failure
breakdown:

- `UUID_MISSING_FLAKY` — an expected UUID is absent from the response
  ("missing"): the model never saw or lost part of its own prompt —
  typically truncated/corrupted prefill or a cache block served from the
  wrong content.
- `CROSS_CONTAMINATION` — a UUID belonging to a *different* series appears
  in the response: a KV/scheduling leak where one request was served
  another request's cached blocks. This is the worst failure class.
- `NOT_EXACT` — all UUIDs correct but extra prose/whitespace around them
  (output conformity, per request).
- `ERROR` — request failed outright.

**Shared cache (`--shared-prefix-per-series 4`).** By default every series
is fully unique, so series never touch each other's cache entries — which
means cross-series reuse is never exercised. With cohorts of 4, each group
of 4 series shares one byte-identical leading garbage prefix, so peers
concurrently co-hit the same prefix-cache blocks — exactly the cross-series
sharing a production cache does. Each series' unique UUID stamps still
trail the shared prefix, so contamination detection remains valid and
per-series.

**Adversarial aborts (`--abort-fraction 0.1`).** 10% of requests are
canceled mid-flight (connection closed while the server is mid-prefill or
mid-KV-load), simulating client disconnects — the abort-during-load path
where cache pinning bugs hide. Aborted requests create the corruption
*opportunity* and are excluded from scoring; only their count is reported.

Related knobs: `--concurrency` (max in-flight requests, default 1 — raise
it to create real cache contention), `--garbage-characters`, `--seed` for
reproducible prompts, `--reset-every-n` to inject vLLM dev-mode
`reset_prefix_cache` calls mid-run, and `--total` to override the 2-cycle
default. See `wekai eval coherency --help`.

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
`quay.io/weka.io/wekai:replay-<sha12>` — published by `task replay:push`,
see "CI / Publishing" below; replay artifacts and the app image share the
wekai quay repo, distinguished by the `replay-` tag prefix; to fetch that
artifact standalone see "Downloading it" above). Both
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
`resultsMountPath/resultsSubPath/<run-timestamp>/` (`--save-request-data`);
the only choices are where under the volume they land and what storage
backs it — see below. The chart is
deliberately minimal: the only value most installs
need to set is `endpoint`, the target model server; everything else has a
working default (all sessions replayed by 256 parallel series workers at
fixed concurrency 28 with a 4-worker hot pool).

| Value | Default | Purpose |
|---|---|---|
| `endpoint` | `""` | Target model server — the one value most installs set. Bare URL (`http://host:8000`, autodiscovers `/v1` + model id) or a full `dynamic/...,type=...` spec |
| `model` | `""` | Optional explicit model id (skips autodiscovery); appended as `,model=<v>` |
| `duration` | `8h` | Benchmark run length (maps to `--timeout`); e.g. `3m` for smoke tests |
| `imageRepository` / `imageTag` | `quay.io/weka.io/wekai` / `""` | Image; tag defaults to the chart's `appVersion` (see "Releases") |
| `imagePullSecrets` | `[]` | Optional pull secrets — the quay repo is public, only needed behind an authenticated mirror |
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
| `replay.printErrorsThreshold` | `"1s"` | Print request errors to the pod log as they happen, at most one line per interval (`""` or `0` = silent) |
| `replay.stderrLogs` | `false` | Log to stderr |
| `llmApiKeySecretName` | `""` | K8s secret with LLM API-key env vars (for endpoints that need auth) |
| `resultsClaim` | `""` | Mount an existing PVC at `resultsMountPath` — the one value needed to persist results (see below) |
| `createResultsClaim` / `storageSize` / `storageClassName` | `false` / `10Gi` / `""` | Have the chart provision the PVC instead |
| `resultsMountPath` | `/results` | Where the results volume is mounted inside the pod |
| `resultsSubPath` | `wekai-requests-data` | Subdirectory of `resultsMountPath` that wekai writes into, so run directories don't land loose at the root of a shared PVC. Set to `""` to write at the root |
| `nodeSelector` / `tolerations` | `{}` / `[]` | Standard pod scheduling — keep the load generator off the inference nodes it measures, or tolerate a tainted pool to reach them |
| `resources` | requests 8Gi / 4 CPU, no limits | Pod resources. Limits are omitted deliberately — CPU throttling on the load generator would show up as inflated TTFT in the results |

Authoritative list with inline docs: `chart/wekai/values.yaml`
(or `helm show values oci://quay.io/weka.io/helm/wekai --version <vX>`).

### Where results are stored

Results are *always* written, to
`resultsMountPath/resultsSubPath/<run-timestamp>/`. Two independent
decisions: what backs the path, and where under it the run lands.

What backs it:

| | |
|---|---|
| nothing set | ephemeral `emptyDir` — results go away with the pod |
| `--set resultsClaim=<pvc-name>` | mounts a PVC you already created; no other value required |
| `--set createResultsClaim=true` | chart provisions `<release>-results` (`storageSize`, `storageClassName` apply) |

Where under it:

| | |
|---|---|
| default | `/results/wekai-requests-data/<run-timestamp>/` |
| `--set resultsSubPath=""` | `/results/<run-timestamp>/` — flat, at the volume root |
| `--set resultsSubPath=some/dir` | `/results/some/dir/<run-timestamp>/` |

The subdirectory exists because a results PVC is usually shared across
purposes, and run directories at the volume root mix in with whatever else
is already there. The *whole* volume is still mounted at
`resultsMountPath` (this is not a volumeMount `subPath`), so everything
else on the PVC stays reachable via `kubectl exec`/`kubectl cp` — only
what wekai writes is namespaced. Leaving `resultsSubPath` unset keeps the
default; setting it to an explicit empty string is what selects the flat
layout.

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --namespace weka \
  --set endpoint=http://YOUR-LLM-HOST:8000 \
  --set resultsClaim=my-existing-pvc
```

`resultsClaim` wins over `createResultsClaim`: the volume is yours to size
and delete, so the chart won't provision a second one beside it. A
chart-provisioned claim carries `helm.sh/resource-policy: keep`, so
`helm uninstall` leaves the results behind — delete that PVC by hand once
you've pulled them off. Either way the volume outlives the pod, so deleting
the pod to rerun the benchmark is safe.

`storeResults` is the old name for `createResultsClaim` and still works, so
existing values files keep their PVC.

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
helm upgrade --install my-replay oci://quay.io/weka.io/helm/wekai \
  --version <vX> --set endpoint=http://10.71.0.4:8000
```

Smoke test — shorten `duration` (maps to `--timeout`), e.g. 3 minutes:

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set duration=3m
```

Explicit model override — by default the model id is autodiscovered (see
"Bare-URL model selector" below); set `model` to skip discovery:

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set model=nvidia/Kimi-K2.6-NVFP4
```

`endpoint` accepts any dynamic model spec, not just a bare URL — e.g. to
target an Anthropic-shaped server, append `,type=anthropic`. Because Helm's
`--set` splits on unescaped commas, either escape it or use `--set-string`
with a values file instead:

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set-string endpoint='http://10.71.0.4:8000\,type=anthropic' \
  --set duration=3m
```

Pull secrets — `quay.io/weka.io/wekai` is public, so no credentials are
needed. If your cluster still pulls through an authenticated mirror or
proxy, reference an existing `kubernetes.io/dockerconfigjson` secret via
`imagePullSecrets`:

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/wekai --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set 'imagePullSecrets[0].name=my-pull-secret'
```

For local development installs from the chart directory, pass the image tag
explicitly (the in-tree `Chart.yaml` carries a placeholder `appVersion`):
`helm upgrade --install my-replay chart/wekai --set imageTag=<vX> --set endpoint=...`

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
`v0.18.6`) rooted at `dagger.json` / `.dagger/src/wekai_core_flows/`. All
Go dependencies needed to build the image are public, so no SSH socket
forwarding or registry credentials are required for the image build
itself.

Three functions:

- `push-replay` — publishes a replay JSONL as a minimal scratch image,
  tagged `replay-<sha12>` (sha256 of the file), to
  `quay.io/weka.io/wekai` by default, alongside the app image — replay
  artifacts are distinguished by the `replay-` tag prefix (registry
  overridable).
- `publish` — builds the repo's `Dockerfile` via `Directory.docker_build()`
  (the Dockerfile is the single source of truth — the module does not
  reimplement its steps), tags with `v999.0.0-<sha12>` (sha12 of the
  source directory's content digest), and
  publishes to `quay.io/weka.io/wekai` by default. The Dockerfile's
  `REPLAY_IMAGE` build-arg is exposed as a `--replay-image` function param.
- `push-helm` — first runs the exact same image build+publish as `publish`
  (shared internal helper, not a separate code path), then packages
  `chart/wekai` and pushes it to an OCI Helm registry
  (`quay.io/weka.io/helm` by default). Version pinning is pure propagation:
  only `Chart.yaml`'s `version`/`appVersion` are stamped — in lockstep with the image the
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
env vars.

`.dagger/sdk/` (the generated Dagger client bindings) is gitignored — run
`dagger develop` once after a fresh clone (or let the first
`dagger call`/`task app:push` do it implicitly) to regenerate it locally;
nothing under `sdk/` is meant to be hand-edited or committed.

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
