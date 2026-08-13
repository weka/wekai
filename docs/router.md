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

**An endpoint is a URL or a pod selector.** One grammar, used everywhere an
endpoint can appear, and the two forms mix in one pool:

```
http://vllm-a:8000              a URL
pods:app=vllm                   every matching pod IN THE ROUTER'S NAMESPACE
pods:app=vllm:http              ... using each pod's containerPort named "http"
pods:app=vllm:8000              ... using 8000 for a pod declaring no port
http://legacy:8000|pods:app=vllm    both, which is what a migration looks like
```

Each discovered pod contributes **its own declared port**. That is the reason to
prefer a selector over a Service, and it is not a detail: a fleet run as several
DaemonSets — one per GPU topology, or per model — commonly listens on a different
port per set. A Service maps one port, so covering three sets needs three
Services, and the single selector that made them one pool is gone.

The port belongs to the route because it describes the *pool*. A router fronting
two fleets on different ports could not say so while this was a router-wide flag.
A colon is safe as the separator: Kubernetes label keys and values never contain
one.

Discovery searches the router's **own namespace** only. There is no syntax for
another, deliberately — reading pods across namespaces needs a ClusterRole, and a
standing cluster-wide pod read on every router is a poor trade for a case a
second router covers. It is what keeps the chart's RBAC a plain Role, which it
creates by default.

Discovery only ever *proposes* backends: the registry decides admission and
health decides eligibility, so a discovered pod is not routed to until it passes
the same checks a statically named one does, and a pod going NotReady leaves the
pool on its own.

**Capacity is signalled, not assumed.** The backend's own `429` is the ultimate
signal and is always on. `--max-node-concurrency` is an opt-in early warning
that saves a round trip: set it to the backends' vLLM `--max-num-seqs` and
saturation is predicted rather than discovered one refusal at a time.

`--rebalance-ratio` defaults to `0.5`, so a backend carrying more than twice the
fleet minimum stops taking new work and the idle capacity beside it gets used.
It trades locality for evenness — a fleet where affinity is working is *supposed*
to look imbalanced — so set it to `0` when locality matters more.

A backend under **8 in-flight** is never rebalanced away from, whatever the
proportions say. The floor is fixed, and it is what keeps the ratio meaningful:
the fleet minimum is 0 whenever any backend is momentarily idle, and then
`(inflight − 0)/inflight` is 1.0 for every backend carrying anything at all. A
fleet of `1,1,1,0` would read as imbalanced as `20,20,20,0` — they are
ratio-identical, and only magnitude separates them. Above the floor the ratio
decides; below it nothing is under pressure and moving a prefix costs more than
the imbalance does.

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

`--backends` is shorthand for `--route '* => a|b|c'` — no model patterns, no
rules, every request to one pool. It takes pod selectors too, which is the
tidiest form there is:

```bash
wekai router serve --listen :8080 --backends pods:app=vllm:http
```

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

The same pool from pod labels — still no patterns, still just `backends`. An
endpoint is a URL string or a selector map, and that is the same grammar a
route's `endpoints` takes:

```yaml
router:
  backends:
    - pods: {app: vllm}
      port: http
```

```bash
--set-string 'router.backends[0].pods.app=vllm' \
--set-string 'router.backends[0].port=http'
```

Use the map with `--set`. A label selector is comma-separated and `--set` splits
on commas, so the string `pods:app=vllm,tier=prod` arrives truncated at its first
label — silently, rendering fine and failing at runtime. In a values *file* the
string form is fine; the problem is `--set` specifically.

All forms render to the one CLI grammar, so the pod ends up running
`--backends pods:app=vllm:http` either way. A URL and a selector can share the
list, which is what a migration looks like:

```yaml
router:
  backends:
    - http://legacy-vllm:8000
    - pods: {app: vllm}
      port: http
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

The hosted endpoint has no `/health` or `/metrics`; the router works that out and
treats it as passive, so it is usable immediately. Pass `--passive-health` to
skip the probe entirely when you already know.

### The full shape

Everything at once: two local fleets discovered from pod labels on different
ports, a model-name rewrite, and Anthropic as the fallback paid for with the
caller's own key.

```bash
wekai router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route 'fast,small => pods:app=vllm,size=7b:http' \
  --route 'sonnet     => pods:app=vllm,size=70b:http as Qwen/Qwen3-32B' \
  --default 'https://api.anthropic.com' \
  --strip-auth-when 'fast,small,sonnet' \
  --max-node-concurrency 48
```

What each line does:

- **`fast,small`** — either substring in the model name picks the 7b pool,
  discovered from pod labels, each pod on its own port named `http`.
- **`sonnet`** — a client asking for `sonnet` reaches the 70b fleet, with the
  model rewritten to the checkpoint the backend actually serves. The client
  never learns the local name.
- **`--default`** — anything unmatched goes to Anthropic. The caller's key is
  forwarded, so they pay for their own hosted calls.
- **`--strip-auth-when`** — the caller's key is removed before it reaches a
  local fleet, which is unauthenticated and would otherwise log someone's
  credential.
- **`--max-node-concurrency`** — the backends' vLLM `--max-num-seqs`, so
  saturation is predicted rather than discovered one refusal at a time.

Both APIs route through the same rules, so a client on `/v1/messages` and one on
`/v1/chat/completions` are matched identically.

### Helm

```yaml
router:
  routes:
    - "llama,mistral => http://vllm-a:8000|http://vllm-b:8000"
  default: "https://api.anthropic.com"
  stripAuthWhen:
    - "llama,mistral"
```

The full shape above, structured — which is what `--set` requires, since both
model patterns and label selectors are comma-separated:

```yaml
router:
  routes:
    - patterns: [fast, small]
      endpoints:
        - pods: {app: vllm, size: 7b}
          port: http
    - patterns: [sonnet]
      endpoints:
        - pods: {app: vllm, size: 70b}
          port: http
      as: Qwen/Qwen3-32B
  default: "https://api.anthropic.com"
  stripAuthWhen: ["fast,small,sonnet"]
  signals:
    maxNodeConcurrency: 48
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

### Serving only what you intend

**Setting an API key already does this.** A protected router serves the
dialect's own routes — `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/models`, and the rest of the table — all of them
requiring the key, and nothing else. The passthrough tier is closed, and so are
the admin endpoints; ask for those explicitly if you want them.

Setting a key says this listener faces users, and proxying arbitrary paths
through to a backend is not something a user-facing listener should do. The two
belong together, so one implies the other rather than being a second thing to
remember.

`--path-allowlist` overrides that default when you want a different set. Without
a key it is the only way to restrict anything: **empty means every path**, since
the passthrough tier is what lets one router front both a local fleet and a
hosted API on paths this dialect never claims.

```bash
wekai router serve \
  --api-key-file /etc/wekai/key \
  --path-allowlist /v1/chat/completions \
  --path-allowlist /v1/messages \
  --path-allowlist /v1/models \
  --backends pods:app=vllm
```

```yaml
router:
  pathAllowlist:
    - /v1/chat/completions
    - /v1/messages
    - /v1/models
```

Nothing outside the list is served — not the admin endpoints, and not a probe
path. There is no exemption. Matching is on segment boundaries, so
`/v1/mod` never admits `/v1/models`, and a trailing `/` denotes a subtree
(`/v1/responses/` admits `/v1/responses/abc`). A path off the list answers 404
rather than 401, so it does not confirm what the router has.

### Probes are on the metrics listener

`/liveness` and `/readiness` are served on `--metrics-listen`, not on the
serving port. Their answer is operational detail — how many backends exist, how
many are healthy, and why the router is not ready — and a user-facing router is
routinely unauthenticated, so on the serving port that detail would be public.

The chart probes the `metrics` port accordingly. A router started without
`--metrics-listen` serves no probes at all and says so at startup.

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
| `router_cache_overflows_total` | requests `--transient-fallback-threshold` served without marking a holder. **Read with `guard_rejects`** — same situation, opposite outcomes; the ratio is what the threshold buys |
| `router_retries_total{reason="capacity_saturated"}` | waits caused by every backend being full. The transient fallback cannot apply here — there was no candidate — so waiting is the only move |
| `router_retries_total{reason="capacity_guard_blocked"}` | waits caused by the split guard. The fallback is tried BEFORE this error is returned, so a count here with `overflows_total` at zero means the threshold is too tight or off — not that the router waited instead of falling back |
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
