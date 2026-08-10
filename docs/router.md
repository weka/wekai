# The router

`wekai router serve` is a model-aware LLM router. It matches each request's
model to a **pool** of interchangeable endpoints, then chooses among them by
prefix-cache affinity — so a conversation keeps landing on the backend that
already holds its KV cache instead of being spread by a load balancer.

It is one subcommand of the `wekai` binary. There is no separate router program.

## Concepts, briefly

**A route maps model names to a pool.** Rules are first-match-wins, so specific
rules come first and a catch-all last:

```
--route '<patterns> => <endpoint>[|<endpoint>...] [as <model>]'
```

Patterns are comma-separated case-insensitive substrings of the model name, or
`*` for the catch-all. Endpoints are pipe-separated — the same syntax the wekai
client already accepts for a multi-endpoint model, meaning the same thing on
both sides.

**A pool of one endpoint is a plain proxy. Several get affinity.** Same code
path either way; there is no mode to switch.

**Routing is by model, not by wire format.** Both OpenAI (`/v1/chat/completions`)
and Anthropic (`/v1/messages`) bodies carry a `model` field, and the same rules
apply to both. Any path the OpenAI dialect does not claim is still forwarded to
the matched pool, unchanged — which is what lets one router front a local fleet
and a hosted API at once.

**Endpoint kind is discovered, once.** An endpoint serving `vllm:` metrics at
`/metrics` is treated as a vLLM instance: health-probed actively and eligible for
upstream metric aggregation. Anything else falls back to passive health — still
served, health inferred from real traffic rather than from probes it would always
fail. The probe never repeats. Because it keys on metric names rather than wire
format, **a vLLM fronted with an Anthropic API is still recognised as vLLM**.

**Capacity is signalled, not assumed.** The backend's own `429` is the ultimate
signal and is always on. `--max-node-concurrency` and `--rebalance-ratio` are
opt-in early warnings that save a round trip.

---

## Use case 1 — one vLLM fleet, all traffic

The simplest deployment. Every model goes to one pool, and affinity concentrates
each conversation on the backend holding its prefix.

### CLI

```bash
wekai router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --backends 'http://vllm-a:8000|http://vllm-b:8000|http://vllm-c:8000'
```

`--backends` is shorthand for `--route '* => a|b|c'`.

Set `--max-node-concurrency` to the backends' vLLM `--max-num-seqs` to predict
saturation rather than discovering it one refusal at a time:

```bash
wekai router serve \
  --backends 'http://vllm-a:8000|http://vllm-b:8000' \
  --max-node-concurrency 32
```

### Helm

```yaml
# values.yaml
router:
  backends:
    - http://vllm-a:8000
    - http://vllm-b:8000
    - http://vllm-c:8000
  signals:
    maxNodeConcurrency: 32
```

```bash
helm upgrade --install my-router \
  oci://quay.io/weka.io/helm/wekai-router --version v1.2.3 \
  -f values.yaml
```

Backend URLs contain no commas, so `--set` is safe here and no values file is
needed:

```bash
helm upgrade --install my-router \
  oci://quay.io/weka.io/helm/wekai-router --version v1.2.3 \
  --set router.backends[0]=http://vllm-a:8000 \
  --set router.backends[1]=http://vllm-b:8000 \
  --set router.signals.maxNodeConcurrency=32
```


Covered by `TestUseCase1_SingleFleetAllTraffic`.

---

## Use case 2 — per-model routes, both APIs

Different models on different fleets. Routing is by model name, so clients using
the OpenAI API and clients using the Anthropic API hit the same rules.

### CLI

```bash
wekai router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route 'fast,small => http://vllm-small-a:8000|http://vllm-small-b:8000' \
  --route 'big,70b    => http://vllm-large-a:8000|http://vllm-large-b:8000'
```

Both of these route to the `big` pool, because both name the model the same way:

```bash
curl :8080/v1/chat/completions -d '{"model":"big-70b","messages":[...]}'   # OpenAI
curl :8080/v1/messages         -d '{"model":"big-70b","messages":[...]}'   # Anthropic
```

Use `as <model>` when the client's name for a model differs from the backend's:

```bash
--route 'sonnet => http://vllm-a:8000 as Qwen/Qwen3-32B'
```

### Helm

```yaml
router:
  routes:
    - "fast,small => http://vllm-small-a:8000|http://vllm-small-b:8000"
    - "big,70b    => http://vllm-large-a:8000|http://vllm-large-b:8000"
    # `as <model>` rewrites the model before forwarding, same as on the CLI —
    # a route is the same string either way.
    - "sonnet     => http://vllm-a:8000 as Qwen/Qwen3-32B"
```

**With `--set`, use the structured form instead.** Helm's `--set` splits values
on commas, so a multi-pattern route written as one string is truncated at its
first pattern — it renders without error and fails at runtime. Given as lists,
no single value contains a comma:

```yaml
router:
  routes:
    - patterns: [fast, small]
      endpoints: [http://vllm-small-a:8000, http://vllm-small-b:8000]
    - patterns: [sonnet]
      endpoints: [http://vllm-a:8000]
      as: Qwen/Qwen3-32B
```

```bash
--set router.routes[0].patterns[0]=fast \
--set router.routes[0].endpoints[0]=http://vllm-small-a:8000
```

Both forms are accepted; a values file can use whichever reads better.

Covered by `TestUseCase2_PerModelRoutesAcrossBothAPIs`.

---

## Use case 3 — self-hosted models, hosted fallback

Named models go to your fleet; everything else falls through to a hosted API, so
a client can point at one base URL for both.

### CLI

```bash
wekai router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route 'llama,mistral => http://vllm-a:8000|http://vllm-b:8000' \
  --default 'https://api.anthropic.com' \
  --strip-auth-when 'llama,mistral'
```

Order matters: the specific rule is matched first, and `--default` is the
catch-all. `--strip-auth-when` drops the caller's credentials before forwarding
to an unauthenticated local upstream, so a client's key is not leaked into your
own logs.

The hosted endpoint has no `/health` or `/metrics`; discovery works that out and
treats it as passive, so it is usable immediately. Pass `--passive-health` to
skip the probe entirely when you already know.

### Helm

```yaml
router:
  routes:
    - "llama,mistral => http://vllm-a:8000|http://vllm-b:8000"
  default: "https://api.anthropic.com"
  stripAuthWhen:
    - "llama,mistral"
```

Covered by `TestUseCase3_SelfHostedModelsWithHostedFallback`.

---

## What is published

Every release publishes two images and two charts from the same commit under the
same semver, so a benchmark and a router from one release are known to agree.

| | image | chart |
|---|---|---|
| router | `quay.io/weka.io/wekai-router:<version>` | `oci://quay.io/weka.io/helm/wekai-router:<version>` |
| benchmark | `quay.io/weka.io/wekai:<version>` | `oci://quay.io/weka.io/helm/wekai:<version>` |

The two images are the same `wekai` binary. The benchmark one additionally
embeds a multi-GB replay artifact so a benchmark pod can replay captured traffic
with no volume; the router never reads it, so the router image does not carry it.

The chart pins its image purely by propagation — `Chart.yaml` `appVersion` feeds
the deployment's `imageTag | default .Chart.AppVersion` — so a `helm install` of
a published chart with no further flags deploys exactly the image it was
packaged with.

## Building and publishing locally

| task | what it needs | what it does |
|---|---|---|
| `go test ./chart/` | helm only | renders the chart and asserts on the manifests — the fastest check, and enough for template changes |
| `task router:image` | docker | plain `docker build` of the router image |
| `task router:build` | a working Dagger engine | image **and** packaged chart, no push, no credentials — runs the release's own packaging code, so a chart checked here cannot differ from the published one |
| `task router:push` | Dagger + `QUAY_USERNAME`/`QUAY_PASSWORD` | publishes the image and the chart pinned to it; what the release workflow runs |

If Dagger cannot start — a locked-down Docker Desktop refusing to pull its engine
image is the usual cause — `go test ./chart/` and `task router:image` still cover
template and image correctness between them.

## Backend URLs and base paths

A backend is configured as a **base**; the client's request supplies the
**path**. For anything hosted at the root — vLLM, OpenAI, Anthropic — write the
bare host and nothing is added or stripped:

```
http://vllm:8000            +  POST /v1/chat/completions
  -> http://vllm:8000/v1/chat/completions
```

A base path exists for providers that are *not* at the root. Google's
OpenAI-compatible surface is one, and the prefix can only come from
configuration — without it the request lands on Google's native API, which
speaks a different protocol:

```
https://generativelanguage.googleapis.com/v1beta/openai  +  POST /v1/chat/completions
  -> https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
```

The base *replaces* the client's version segment rather than stacking on top of
it, which is what `base_url` means everywhere else it appears: an OpenAI
`base_url` ends in `/v1` and the caller appends `chat/completions`.

### The `/v1` suffix

Out of that same `base_url` habit, people often write:

```
http://vllm:8000/v1
```

Inference is unaffected — it composes back to exactly the same URL. But the
requests the router makes *on its own behalf* concatenate onto the base, so
`/v1/health`, `/v1/metrics` and `/v1/v1/models` all 404. The endpoint is then
taken for a hosted API, marked passive, and never actively health-checked — so
it serves fine until it dies and keeps looking healthy afterwards, while
contributing no models to `/v1/models` and no upstream metrics.

The router logs a warning at startup for any backend URL ending in `/v1`. Drop
the suffix; a genuine base path like `/v1beta/openai` is kept and used.

## Credentials

Two questions, kept separate: who may call the router, and how the router
authenticates to each pool.

### Protecting the router

```bash
--api-key-file /etc/router/inbound-key    # preferred
--api-key <value>                         # visible in `ps` and in the pod spec
```

Every inference and admin request then needs the key. `/liveness` and
`/readiness` stay open so a kubelet needs no credential; `--require-auth-for-probes`
changes that.

### How a pool authenticates

A route says it in the same position as `as <model>`:

```
=> http://vllm:8000 using /etc/secrets/inner-key    the ROUTER's own key
=> https://api.anthropic.com using client           forward the CALLER's key
=> http://vllm:8000                                 send no credential
```

**Forwarding is opt-in and the default sends nothing.** A hosted API the user
pays for is the case that needs `using client` — the router has no key that
could work there. An internal backend must never receive a user's personal key,
so defaulting to forward would leak one on any route somebody forgot to
annotate. `--strip-auth-when` remains for dropping credentials by pattern.

`using <file>` wins over `using client`: a pool the router authenticates to
itself never also carries the caller's credential.

### Two routers: internal fleet behind a user-facing edge

The deployment this exists for. An inner router owns the fleet and requires a
key; an outer router is public, holds that key as a mounted secret, and sends
everything else to Anthropic with the user's own credential.

```bash
# inner — internal only, authenticated
wekai router serve --listen :8080 \
  --api-key-file /etc/router/inbound-key \
  --backends 'http://vllm-a:8000|http://vllm-b:8000'

# outer — public, no inbound key
wekai router serve --listen :8080 \
  --route 'llama,mistral => http://inner-router:8080 using /etc/router/inner-key' \
  --default 'https://api.anthropic.com using client'
```

A user calling the outer router for `llama-3` never learns the inner key; a
user calling it for `claude-*` pays with their own. Capture, if enabled, records
both.

```yaml
# outer values.yaml
router:
  routes:
    - patterns: [llama, mistral]
      endpoints: [http://inner-router:8080]
      using: /etc/router/inner-key
    - patterns: ["*"]
      endpoints: [https://api.anthropic.com]
      using: client
  secretMounts:
    - name: inner-key
      secretName: router-inner-key
      mountPath: /etc/router
```

## Discovering endpoints from Kubernetes

A route's endpoints can be discovered from pod labels instead of listed:

```
--route '* => pods:app=vllm'
```

Each discovered pod contributes **its own declared `containerPort`**. That is
the reason to prefer pod discovery over a Service, and it is not a detail: a
fleet run as several DaemonSets — one per GPU topology, or per model — commonly
listens on a different port per set. A Service maps ONE port, so covering three
DaemonSets needs three Services, and the router loses the single label selector
that made them one pool. Pod discovery keeps them one pool.

Port precedence: the pod's named port when `--discover-port-name` is set, then
its sole declared port, then `--discover-port` as a floor. The pod is the
authority on what it listens on — a flag silently overriding it would send
traffic to a port nothing is bound to. A pod declaring several unnamed ports
with no `--discover-port-name` falls back to the flag rather than guessing.

Discovery only ever PROPOSES backends. The registry decides admission and health
decides eligibility, so a discovered pod is not routed to until it passes the
same checks a statically listed one does — and a pod that goes NotReady leaves
the pool on its own.

Static and discovered endpoints can be mixed in one route, which is what a
migration looks like:

```
--route '* => http://legacy-vllm:8000|pods:app=vllm'
```

### CLI

```bash
wekai router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route '* => pods:app=vllm' \
  --discover-namespace inference \
  --discover-port-name http
```

### Helm

Discovery needs RBAC, which the chart creates only when asked — a
namespace-scoped Role, never a ClusterRole:

Note the two blocks. Top-level `discovery` carries one field and it is a
permission: it creates the ServiceAccount, Role and RoleBinding and mounts the
API token. Everything about HOW to look pods up lives under `router.discovery`,
because those become CLI flags.

```yaml
discovery:
  enabled: true      # the ONLY field here; grants API access

router:
  routes:
    - "* => pods:app=vllm"
  discovery:
    namespace: ""      # empty: the router's own namespace
    portName: http     # which containerPort serves inference
    port: 8000         # only for a pod declaring none of its own
```

Several DaemonSets on different ports, as one pool:

```yaml
router:
  routes:
    - "* => pods:app=vllm"      # matches every set
  discovery:
    portName: http              # each set names its own port `http`
```

## Readiness

A router pod is **not ready until at least one endpoint is alive**. `/readiness`
returns 503 while no backend is healthy, so an orchestrator removes the pod from
its Service rather than sending it traffic it cannot serve. Backends are probed
immediately at startup and then on `--health-interval`, so readiness reflects the
fleet rather than the router's own process being up — that is `/liveness`.

Probe the router on `/readiness` and `/liveness` (`/healthz` and `/livez` are
accepted aliases). **Do not probe an arbitrary path**: any path the dialect does
not claim is proxied to a backend, so a probe on the wrong path is answered by
something that knows nothing about it.

A saturated router stays ready. It sheds with 429 rather than going NotReady,
because it is working, not broken.

## Observability

`/metrics` and the live KV map at `/router-viz` are on `--metrics-listen`, never
on the serving path.

| metric | what it tells you |
|---|---|
| `router_route_decisions_total` | which tier decided: `cache`, `split`, `load`. Aggregate by label. |
| `router_cache_avg_copies` | mean backends holding each block; ~1.0 means no duplication |
| `router_cache_guard_rejects_total` | 429s the split guard caused. **Read with `avg_copies`** — a misconfiguration lands in one or the other depending on where the guard sits, and either alone misses half the failure space |
| `router_signal_fired_total` | which capacity signal is actually driving decisions |
| `router_cache_tree_runs` / `_tail_set` | tree size, for memory |

Every cache and signal metric carries a **`pool`** label. A router may front
several pools whose caches are unrelated, so summing across them describes
nothing — aggregate by pool.

`--vllm-metrics` additionally aggregates upstream vLLM counters into
router-level totals served alongside the router's own. Only endpoints discovered
to be vLLM are scraped, and totals accumulate **deltas**, so they never rewind
when a pod restarts or the fleet scales down. A counter that decreases silently
breaks `rate()` and `increase()`.

## Capture

`--capture raw|redacted` (off by default) records every proxied exchange to JSONL for later
analysis or replay. `redacted` keeps structured metadata — block hashes, token
counts, tool-use ids — without bodies. This is how the replay corpora used for
benchmarking are collected.

## Two 429s, and what they mean

- **`all_backends_saturated`** — nothing could take work.
- **`split_guard_blocked`** — capacity existed, and the guard refused to spend it
  on a duplicate copy of the prefix. Not an error: the policy working as
  designed.

Both carry `Retry-After: 1`. **Clients must retry.** A client treating 429 as
fatal will see throughput collapse under load; the wekai replay client backs off
10ms doubling to a 3s cap with a 30s budget.

## Testing without hardware

See [`.ainav/router-testing/`](../.ainav/router-testing/index.md). `mock-vllm`
is a GPU-less vLLM with three surfaces (`--surface vllm|anthropic|hosted`) so
every deployment shape above can be exercised offline, including a vLLM fronted
with an Anthropic API.
