# Migrating from vllm-router (v1) to wllm-router

For operators. This is the delta between the Rust `vllm-router` and the Go
`wllm-router` for someone running the thing — not a feature list.

Everything here was verified against both codebases and against a running
binary. Where something was not verified, it says so.

Read [Breaks silently](#1-breaks-silently) first. Those are the ones that do not
announce themselves.

---

## 0. The five-minute version

| | v1 | wllm-router |
|---|---|---|
| Metric prefix | `vllm_router_*` | `router_*` — **every name changed** |
| Unmatched paths | transparently proxied | `404` — only known routes are served |
| Inbound key env | `INBOUND_API_KEY` | `WLLM_API_KEY` — **wrong name means auth is OFF** |
| Allowlist env | `FORWARD_PATH_ALLOWLIST` | `FORWARD_PATH_ALLOWLIST` (kept deliberately) |
| Probes | required auth → `exec` probe | public by default → use `httpGet` |
| Base image | `python:3.12-slim` | distroless static, nonroot, **no shell** |
| Default policy | `cache_aware` family | `least-outstanding` |
| Policy names | `round_robin`, `cache_aware`, … | `round-robin`, `prefix-cache-aware`, … |
| Config | flags + 3 env vars | JSON file (optional) + flags + `WLLM_*` env |
| Metrics port | `:29000` | `:29000` (unchanged) |
| Inference port | image ran `--port 8080` | `:8080` (unchanged in practice) |

---

## 1. Breaks silently

Three things change without producing an error. Everything in section 2 fails
loudly at startup, which is the safe case; these do not.

### 1.1 Every metric was renamed

`vllm_router_*` → `router_*`, with **no overlap**. A Grafana panel does not
error, it goes blank. A Prometheus alert that can never fire looks exactly like
an alert that is healthy. Update dashboards and rules *before* cutting over.

Direct renames:

| v1 | wllm-router |
|---|---|
| `vllm_router_requests_total` | `router_requests_total` |
| `vllm_router_request_duration_seconds` | `router_request_duration_seconds` |
| `vllm_router_worker_health` | `router_backend_health` |
| `vllm_router_active_workers` | `router_backends_total` |
| `vllm_router_cb_state` | `router_circuit_state` |
| `vllm_router_cb_state_transitions_total` | `router_circuit_transitions_total` |
| `vllm_router_policy_decisions_total` | `router_policy_selections_total` |
| `vllm_router_request_errors_total` | `router_upstream_errors_total` |
| `vllm_router_retries_total` | `router_retries_total` |

Renamed *and* reshaped — check the query, not just the name:

| v1 | wllm-router | What changed |
|---|---|---|
| `vllm_router_worker_load` | `router_backend_inflight`, `router_worker_load_{avg,max,min}` | per-backend gauge split from fleet aggregates |
| `vllm_router_max_load` / `min_load` | `router_worker_load_max` / `_min` | — |
| `vllm_router_tree_size` | `router_cache_entries`, `router_cache_tokens` | one gauge became two, nodes and tokens |
| `vllm_router_cache_hits_total` / `misses_total` | `router_cache_predicted_fraction`, `router_cache_observed_fraction` | **not a rename.** v1 counted its own trie hits. These are a predicted fraction at routing time and the fraction vLLM actually reported. Different question, different units — a ratio, not a counter. |

Gone, with no replacement:

- `vllm_router_pd_*` — nine metrics. No prefill/decode disaggregation.
- `vllm_tokenizer_*` — the router does not tokenize.
- `vllm_router_embeddings_*` — folded into the generic request metrics, labelled
  by route class instead of having their own family.
- `vllm_router_discovery_workers_added` / `_removed` / `_updates_total` — only
  `router_discovery_conflicts_total` survives.
- `vllm_router_load_balancing_events_total`, `vllm_router_processed_requests_total`,
  `vllm_router_running_requests`, `vllm_router_retries_exhausted_total`,
  `vllm_router_retry_backoff_duration_seconds`.

New, worth putting on a dashboard on day one:

| Metric | Why |
|---|---|
| `router_load_accounting_errors_total` | **Must stay at 0.** Non-zero means in-flight accounting under- or over-flowed, which is the exact class of corruption v1 shipped with. |
| `router_panics_total` | Recovered handler panics. |
| `router_client_disconnects_total` | Client hangups, counted separately so they never masquerade as panics. |
| `router_requests_shed_total` | Requests rejected by the concurrency cap. |
| `router_policy_fallback_total` | The chosen policy could not pick and something else did. |
| `router_ttft_seconds` | Time to first token. |
| `router_routing_decision_duration_seconds` | Time spent choosing a backend. |
| `router_stream_aborted_total` | Streams that ended abnormally. |

Scrape config itself needs no change: both default to `:29000`, and `/metrics`
is on that separate listener, never routable on the inference port.

### 1.2 Unmatched paths now 404 instead of being proxied

v1 registered a transparent catch-all, so **any** path reached a backend. This
router serves only routes it knows. Verified live, with no allowlist and no auth
configured:

```
/tokenize                 -> 404      /flush_cache           -> 404
/v1/audio/transcriptions  -> 404      /health_generate       -> 404
/v1/score                 -> 404      /v1/responses/resp_123 -> 404
```

Removed endpoints that v1 served explicitly:

- `POST /flush_cache`
- `GET /health_generate`
- `GET` and `DELETE /v1/responses/{id}` (`POST /v1/responses` is still served)
- `GET` and `DELETE /workers/{url}` — use `GET /workers` and `POST /remove_worker`

Inventory what your clients actually call before cutting over. Anything hitting
a vLLM endpoint not in the served list below stops working.

**Served:** `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/responses`, `/v1/rerank`, `/rerank`, `/generate`, `/inference/v1/generate`;
`GET /v1/models`, `/get_model_info`; `GET /liveness`, `/readiness`, `/health`,
`/get_server_info`, `/get_loads`, `/workers`, `/list_workers`;
`POST /workers`, `/add_worker`, `/remove_worker`.

### 1.3 `INBOUND_API_KEY` is ignored — and that disables auth

v1 reads `INBOUND_API_KEY`. This router reads `WLLM_API_KEY` (or
`-api-key-file`, which is what the manifest uses). Carry a v1 deployment over
unchanged and the router starts with **no key**, which means **auth is off** —
including on the admin endpoints. `GET /get_loads` returns 200 to anyone who can
reach the Service.

**No key meaning no authentication is intended behaviour, not an oversight.** It
is what makes the router runnable on a laptop or a trusted network without
ceremony, and it will not change: the router will not refuse to start, and it
will not require an explicit "yes, really unauthenticated" flag. The consequence
is that the failure mode of a mistyped or missing key variable is an open
router rather than a crash, so the check belongs in your rollout, not in the
binary.

There is a startup warning:

```
no API key configured: the inference listener is unauthenticated
```

but that is one line in a pod log. Grep for it after cutover, and assert that an
uncredentialed `GET /get_loads` returns 401 — step 2 of the post-cutover
verification below. That assertion is the real guard here.

`FORWARD_PATH_ALLOWLIST` deliberately keeps v1's name for exactly this reason —
it is the one variable that does *not* take the `WLLM_` prefix.

`API_KEY_VALIDATION_URLS` (remote key validation) does not exist and will not be
added. Static key only.

---

## 2. Breaks loudly

These stop the router at startup with a message. Annoying, not dangerous.

### 2.1 Policy names changed

| v1 | wllm-router | Note |
|---|---|---|
| `round_robin` | `round-robin` | underscore → hyphen |
| `random` | `random` | unchanged |
| `cache_aware` | `prefix-cache-aware` | renamed *and* reimplemented |
| `power_of_two` | `least-outstanding` | closest equivalent, **different algorithm** |
| `consistent_hash` | — | **no equivalent** |

Default is now `least-outstanding`. An unknown policy is a startup error listing
the valid names, not a silent fallback.

### 2.2 Config is stricter

- An unknown config key is a hard error (v1 ignored them).
- Validation reports **every** problem at once instead of dying on the first.
- `health_timeout` must be `<` `health_interval`. v1 let a checker fall behind
  indefinitely.
- `max_body_bytes` must be `> 0`; unbounded is a memory-exhaustion DoS.
- Allowlist entries must start with `/`.

---

## 3. Pod spec changes

### 3.1 Probes invert — switch `exec` back to `httpGet`

v1 required auth on `/readiness` and `/health`, which is why its README told you
to use an `exec` probe: a kubelet cannot present a credential, so an `httpGet`
probe got 401, the pod never went Ready, and the Deployment never received
traffic.

Here `GET /liveness`, `/readiness` and `/health` are public by default, so
`httpGet` works:

```yaml
livenessProbe:
  httpGet: { path: /liveness,  port: 8080 }
readinessProbe:
  httpGet: { path: /readiness, port: 8080 }
```

An `exec` probe is now **impossible** anyway — see 3.2.

Set `require_auth_for_probes: true` to opt out, but then the probes need
credentials the kubelet cannot supply. Unauthenticated probes disclose only a
boolean; `healthy_backends` is included in the body only for an authenticated
caller. `/readiness` returns 503 when zero backends are healthy.

### 3.2 The image is distroless — no shell

`python:3.12-slim-bullseye` → `gcr.io/distroless/static-debian12:nonroot`.

- No shell. `kubectl exec -- /bin/sh` fails. Use
  `kubectl debug -it <pod> --image=busybox --target=wllm-router`.
- No `curl`, no `python`, no package manager.
- Runs as `nonroot:nonroot`. **Remove any `runAsUser: 0`** from the spec, and
  make sure mounted Secrets are readable by that UID.
- `exec` probes cannot work, which is fine — see 3.1.

### 3.3 Ports

Unchanged in practice. Inference `:8080`, metrics `:29000`. v1's *CLI* default
port was 30000, but its image overrode it with `--port 8080`, so a containerized
v1 deployment already used 8080.

Metrics default to `127.0.0.1:29000` — loopback, therefore unscrapable from
outside the pod. The shipped manifest overrides this to `0.0.0.0:29000`.

---

## 4. Configuration

Precedence: **flag > environment > config file > default**.

**There is no config file in the image.** It holds the binary and CA certs; the
`ENTRYPOINT` is the bare binary with no default arguments. A config file is
entirely optional — flags alone are a valid way to run it, and so is env alone.
The shipped manifest chooses to mount one from a ConfigMap at
`/etc/wllm-router/config.json`, but nothing requires that.

If you *do* pass `-config` and the file is not there, the router exits 1:

```
error: read config "/etc/wllm-router/config.json": no such file or directory
```

So a ConfigMap that failed to mount gives you CrashLoopBackOff, not a router
quietly running on defaults.

The API key is **not** settable by flag — a flag value is visible in `ps` to
anything on the host. Use `WLLM_API_KEY` or `-api-key-file` with a mounted
Secret, which lets the key rotate without a pod-spec change.

### Environment variables

| v1 | wllm-router |
|---|---|
| `INBOUND_API_KEY` | `WLLM_API_KEY` — see 1.3 |
| `FORWARD_PATH_ALLOWLIST` | `FORWARD_PATH_ALLOWLIST` (unchanged on purpose) |
| `API_KEY_VALIDATION_URLS` | removed, not coming back |

Everything else is new and `WLLM_`-prefixed: `WLLM_CONFIG`, `WLLM_LISTEN`,
`WLLM_METRICS_LISTEN`, `WLLM_POLICY`, `WLLM_API_KEY_FILE`,
`WLLM_UPSTREAM_CREDENTIAL`, `WLLM_BACKENDS`, `WLLM_CORS_ORIGINS`,
`WLLM_LOG_LEVEL`, `WLLM_LOG_FORMAT`, `WLLM_HEALTH_PATH`, `WLLM_DISCOVERY_*`.

### Path allowlist

Semantics are v1's, including segment-boundary matching. It gates which paths
are **served**, independently of auth:

- Unset or empty serves every path; auth still applies to every path.
- Non-empty serves only listed paths. Everything else is `404` — including for a
  caller holding a valid key, and including admin endpoints. `404` rather than
  `403` so an unlisted path does not reveal whether a credential would have been
  accepted.
- Listing a path does **not** exempt it from auth. `/get_loads` on the allowlist
  is reachable, not public.
- `/v1/responses` admits `/v1/responses/abc` but not `/v1/responses_evil`; a
  trailing `/` denotes a subtree.
- One difference from v1: `GET /liveness`, `/readiness` and `/health` are served
  in strict mode without being listed. v1 404'd `/readiness` and `/health`
  unless you listed them explicitly.

---

## 5. Cutover checklist

1. Update Grafana dashboards and Prometheus rules to `router_*` (§1.1).
2. Add a `router_load_accounting_errors_total > 0` alert. It should never fire.
3. Inventory client paths against the served list; confirm nothing relied on the
   catch-all (§1.2).
4. Rename `INBOUND_API_KEY` → `WLLM_API_KEY`, or switch to `-api-key-file`
   (§1.3). Do not skip the 401 assertion below — this step failing is silent.
5. Translate the policy name (§2.1). `consistent_hash` has no equivalent —
   decide what replaces it.
6. Switch `exec` probes to `httpGet` (§3.1).
7. Remove `runAsUser: 0`; check Secret file permissions for `nonroot` (§3.2).
8. Deploy to one replica behind a canary Service first.

Post-cutover, verify in this order:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://ROUTER:8080/liveness    # 200
curl -s -o /dev/null -w '%{http_code}\n' http://ROUTER:8080/get_loads   # 401, NOT 200
curl -s http://ROUTER:29000/metrics | grep router_load_accounting_errors_total  # 0
```

The second one is the important one. A 200 there means auth is off.

---

## 6. Not verified

Stated plainly so nobody assumes otherwise:

- **Error response body shapes** were not diffed against v1. If a client parses
  the error envelope rather than the status code, check it.
- **Log field names** were not diffed. Both emit JSON by default, but log-based
  alerts may reference fields that changed. v1 could also log to a file; this
  router writes to stdout only.
