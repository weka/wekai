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
helm upgrade --install my-router oci://quay.io/weka.io/helm/wekai-router \
  --version <release> -f values.yaml
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

`--capture raw|redacted` records every proxied exchange to JSONL for later
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
