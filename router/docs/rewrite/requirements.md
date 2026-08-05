# Requirements Specification: `wllm-router` Go Rewrite (v2)

> ## Amendments applied after review — read before implementing
>
> Where this banner conflicts with body text, the banner wins. **`plan.md` is authoritative** for product-requirement traceability (FR-RTR-01..05), scope, and milestones.
>
> **Product-requirement status:**
>
> | FR | Status |
> |---|---|
> | **FR-RTR-01** Opportunistic HBM routing | ⚠️ **Approximated by prediction, not satisfied.** No residency feed is available in our deployment — see `RES-*` below. Recorded as a known gap. |
> | **FR-RTR-02** Configurable routing policies | ✅ `--policy` flag |
> | **FR-RTR-03** Load-balanced routing (GPU util + queue depth) | ⚠️ Partial — needs a per-node vLLM metrics scraper (`LOAD-*`). *Nice to Have*, deferred. |
> | **FR-RTR-04** DRAM-aware multi-tier routing | ⚠️ **Blocked** on the same feed as FR-RTR-01. Interface-shaped only. |
> | **FR-RTR-05** Resilient storage fallback | ⚠️ **Mostly not the router** — LMCache's `HealthMonitor` already implements it. Router obligations are `WEKA-*`. |
>
> **Amendments to requirements in this document:**
>
> | ID | Change |
> |---|---|
> | **CU-6 / CU-7** | The continuous `queueing_penalty` is **no longer the default**. `prefix-cache-aware` uses the `balance_abs_threshold` (32) / `balance_rel_threshold` (1.5) spill guard per FR-RTR-01 — residency wins outright below the threshold. The continuous form ships only as the experimental `cache-usefulness` policy. |
> | **CU-8** | `max_nodes_per_worker` default **500k → 100k**. 500k × ~96 B × 64 backends = 3.0 GiB, a direct `NFR-5` violation. The token budget binds first anyway (~7,800 nodes at 1024-byte granularity). |
> | **CACHE-6 / OQ-2** | Units are **text-prefix** windows (1024 bytes), *not* token ids — so the router needs **no tokenizer**, preserving NG-3. |
> | **HLT-1 / HLT-N1** | Health-check pool defaults to `min(256, max(32, N))`. A small fixed pool cannot meet the "200 workers, 5 s timeout, round < 6 s" criterion; the real bound is fds. |
> | **HIER-\*** | Hierarchical routing is **documented but deferred** to post-v2.0. `Backend.kind` and `Backend.capacity` ship as no-op defaults so it stays additive. |
> | **API-3** | Anthropic dialect deferred; the seven-method `Dialect` interface and the CI import fence ship in v2.0. |
>
> **New requirement areas (full text in `plan.md`):**
>
> - **`RES-1..RES-5`** — residency behind one `Lookup()` interface with a predictive-trie implementation, so the vLLM ZMQ KV event stream or LMCache's `POST /lookup` slot in later without touching any policy. Every decision records its source; predicted-vs-observed accuracy is emitted continuously.
> - **`WEKA-1..WEKA-5`** — the router MUST NOT mark a node unhealthy because WEKA is degraded (a node in `RECOMPUTE` fallback is *slower, not broken*; failing it converts a storage slowdown into a capacity outage). Invalidate that node's cache model, recover with no restart, surface the state to operators.
> - **`LOAD-*`** — per-node vLLM metrics scraper (`vllm:num_requests_running`, `vllm:num_requests_waiting{,_by_reason}`, `vllm:kv_cache_usage_perc`) feeding the FR-RTR-03 composite. Note vLLM exposes **no** direct SM-utilization metric.
>
> **Deployment dependency:** `usage.prompt_tokens_details.cached_tokens` requires `--enable-prompt-tokens-details` on the vLLM nodes (default **off**). Without it there is no feedback signal and `prefix-cache-aware` is unfalsifiable — this gates whether that policy ships at all.

**Status:** Draft for review, amended post-review
**Supersedes:** `vllm_router_rs` v0.1.14 (Rust), `/Users/ofer.kiselovnahman/workspace/wllm-router`
**Target:** `github.com/weka/wllm-router` — Go binary + container image, owned end-to-end by the team

---

## 1. Context & Motivation

### 1.1 Current state

The existing router is a fork/derivative of the SGLang Rust router, at ~36,250 lines of Rust across `src/` (verified: `wc -l` over `src/**/*.rs`). It carries a PyO3 extension module, a Python CLI, a Python `mini_lb.py`, a full HuggingFace/tiktoken tokenizer subsystem, and PD (prefill/decode) disaggregation routers — none of which the team uses. The largest files are `src/protocols/spec.rs` (3,823 LOC), `src/routers/http/vllm_pd_router.rs` (2,617), `src/tree.rs` (2,320), `src/routers/http/router.rs` (2,218), `src/core/worker.rs` (1,961).

Three problems drive the rewrite:

1. **Over-complication.** The Rust codebase is large, generic, and abstracted for use cases the team does not have (PD disaggregation, Python embedding, multi-tokenizer support). Nobody on the team can confidently change routing behavior.
2. **The load-balancing policies do not work.** This is not a tuning problem; the load signal itself is corrupt (see §1.3). Every load-sensitive policy — `power_of_two`, `cache_aware`'s balance check, `get_loads` — is making decisions on noise.
3. **Ownership.** Rust + PyO3 + a vendored upstream lineage means changes are risky and slow. Go is the team's primary language (`wekai` is Go 1.25), and the cache-usefulness engine we want to reuse is already Go.

### 1.2 Why Go

- Team fluency; `wekai` (module `github.com/weka/wekai`, Go 1.25.7) is the source of the cache-prediction engine we intend to reuse.
- `net/http` + `httputil.ReverseProxy` gives a correct, streaming, backpressured HTTP proxy out of the box; the Rust implementation hand-rolled this and got backpressure, Content-Type, and SSE framing wrong.
- Performance is adequate: the router is I/O-bound (proxying tokens), not CPU-bound. GC pressure is the only real risk and is addressed by NFR-3/NFR-4.

### 1.3 The load-accounting defect (root cause of "policies are broken")

Verified in the Rust source and treated as the central motivating bug:

- In-flight load is **incremented on exactly one code path** (the `cache_aware` selection path) but **decremented on up to three** — normal completion, retryable-status handling, and body-read error — producing double-decrements and wrap-around/underflow.
- The health checker **wipes all load counters wholesale every 10 cycles**, destroying whatever signal survived.
- Therefore `power_of_two`, the `cache_aware` imbalance check (`max_load - min_load > balance_abs_threshold && max_load > min_load * balance_rel_threshold`, `src/policies/cache_aware.rs:253`), `get_loads`, and the `vllm_router_worker_load` / `max_load` / `min_load` gauges are all reporting noise.

The v2 design treats load accounting as a **single, symmetric, RAII-style lifecycle primitive** (LB-1..LB-4) — the one invariant the whole load-balancing area rests on.

---

## 2. Goals

- G1: A single Go binary and container image that is wire-compatible with the current router for both clients and vLLM workers.
- G2: Correct, testable load balancing with a default policy (`least-outstanding`) that is right by construction.
- G3: A rebuilt prefix-cache-aware policy with token-based (not character-based) accounting and bounded memory.
- G4: A new cache-usefulness policy derived from `wekai`'s prediction engine.
- G5: Kubernetes service discovery that supports Services/EndpointSlices, not just pod IPs.
- G6: Code small enough that any team member can read the routing path end-to-end in an afternoon. Target: **under 8,000 LOC excluding tests and vendored deps.**

## 3. Non-Goals (explicit)

- **NG-1:** PD (prefill/decode) disaggregation. No prefill worker class, no bootstrap ports, no `--prefill` flag, no PD service-discovery mode.
- **NG-2:** Python packaging. No PyO3, no `py_src/`, no `mini_lb.py`, no `pip install`. Go binary + container only.
- **NG-3:** Tokenization. The router does not tokenize. All token counts are estimates (see CACHE-6). No HuggingFace hub download, no tiktoken, no chat-template rendering.
- **NG-4:** Request body transformation. The router does not rewrite, validate, or re-serialize OpenAI request/response bodies beyond what routing requires (see GW-6).
- **NG-5:** Response storage / the stateful side of the Responses API (`data_connector`). `/v1/responses` and its sub-routes are proxied, not stored (see GW-3).
- **NG-6:** Multi-model routing / model-aware worker pools beyond a single label. Deferred (see §16 OQ-4).
- **NG-7:** CLI/env compatibility with v1. Flag names, env var names, and policy names are redesigned.

---

## 4. Compatibility Contract

**COMPAT-1 (MUST).** The set of HTTP paths, methods, request bodies, and response bodies exposed to clients MUST be a superset-compatible match for v1 for all endpoints listed in GW-1..GW-5. An existing OpenAI-SDK client pointed at v2 MUST work with no change.

**COMPAT-2 (MUST).** The requests the router issues to vLLM workers MUST be byte-identical to the client's request body, with only hop-by-hop and auth headers adjusted (SEC-4). An unmodified vLLM worker MUST work with no change.

**COMPAT-3 (MUST NOT).** v2 MUST NOT preserve v1's CLI flags, environment variable names, config-file schema, or policy identifiers. These are redesigned (§13).

**COMPAT-4 (SHOULD).** Prometheus metric names SHOULD be redesigned (§14) rather than preserved; a migration table for existing dashboards MUST be published with the release.

---

## 5. Gateway / HTTP

### Requirements

| ID | Level | Requirement |
|---|---|---|
| **GW-1** | MUST | Expose inference endpoints, proxied to a selected worker: `POST /generate`, `POST /inference/v1/generate`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, `POST /rerank`, `POST /v1/rerank`, `POST /v1/responses`. |
| **GW-2** | MUST | Expose Responses sub-resources: `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, `DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input`. |
| **GW-3** | MUST | GW-2 routes MUST be routed by `{id}` affinity when a mapping is known, and otherwise MUST return `404` with an OpenAI-shaped error body rather than being broadcast or sent to an arbitrary worker. The id→worker map is in-memory, bounded (GW-4), and lost on restart; this is an accepted limitation (see §16 OQ-5). |
| **GW-4** | MUST | The response-id→worker map MUST be bounded by both entry count and TTL, with LRU eviction, defaults 100k entries / 1h. |
| **GW-5** | MUST | Expose operational endpoints: `GET /liveness`, `GET /readiness`, `GET /health`, `GET /health_generate`, `GET /v1/models`, `GET /get_model_info`, `GET /get_server_info`. |
| **GW-6** | MUST | The router MUST NOT deserialize the full request body into a typed schema. It MUST extract only the fields routing needs (`stream`, `model`, and the routing text — see CACHE-4) via a streaming/partial JSON scan, and MUST forward the original bytes unmodified. |
| **GW-7** | MUST | Every request MUST have a request id: taken from inbound `X-Request-Id` if present and well-formed (≤128 chars, printable ASCII), else generated. It MUST be echoed in the response, attached to the access log line, and forwarded to the worker. |
| **GW-8** | MUST | A maximum request body size MUST be enforced on **every** path with no exceptions, including the catch-all proxy. Default 64 MiB, configurable. Exceeding it MUST return `413`. |
| **GW-9** | MUST | The catch-all/transparent proxy fallback (if retained — see §16 OQ-1) MUST pass through the identical middleware chain as named routes: body limit, auth, request id, CORS, access log, metrics, tracing. |
| **GW-10** | MUST | CORS preflight (`OPTIONS`) MUST succeed regardless of auth configuration. The CORS handler MUST be ordered **outside** (before) the auth handler. |
| **GW-11** | MUST | Admin endpoints: `POST /add_worker`, `POST /remove_worker`, `GET /list_workers`, `POST /flush_cache`, `GET /get_loads`, plus RESTful `POST /workers`, `GET /workers`, `GET /workers/{url}`, `DELETE /workers/{url}`. |
| **GW-12** | SHOULD | Admin endpoints SHOULD be servable on a separate listener from inference traffic, controlled by config; default is same-listener for compatibility. |
| **GW-13** | MUST | Prometheus `/metrics` MUST be served on a separate listener, default `127.0.0.1:29000`, and MUST NOT be reachable on the inference listener. |
| **GW-14** | MUST | All error responses on `/v1/*` paths MUST use the OpenAI error envelope (`{"error":{"message","type","param","code"}}`) with the correct `Content-Type: application/json`. |
| **GW-15** | MUST | Client request cancellation (context cancel / connection close) MUST propagate to the upstream worker request within 100 ms, and MUST decrement in-flight load exactly once (LB-2). |
| **GW-16** | SHOULD | HTTP/2 cleartext (h2c) SHOULD be supported on the inference listener for client→router; router→worker MAY remain HTTP/1.1. |

### Must not regress into

- **GW-N1:** Reading a body with an unbounded limit (`to_bytes(body, usize::MAX)` in v1's transparent-proxy fallback) — an unauthenticated unbounded-memory DoS.
- **GW-N2:** A fallback path that bypasses request-id assignment, CORS, access logging, and tracing spans.
- **GW-N3:** CORS preflight failing whenever auth is enabled, because the auth layer is outer and `OPTIONS` carries no `Authorization` header.
- **GW-N4:** Full typed deserialization of OpenAI bodies (v1's 3,823-LOC `protocols/spec.rs` + 1,224-LOC `protocols/validation.rs`), which couples the router to vLLM's evolving schema.

---

## 6. Authentication & Authorization

| ID | Level | Requirement |
|---|---|---|
| **AUTH-1** | MUST | Inbound authentication MUST be a single static API key, supplied by config/env, compared with `crypto/subtle.ConstantTimeCompare`. |
| **AUTH-2** | MUST | If no key is configured, the router MUST start with auth disabled and MUST log a `WARN` at startup naming the exposure. |
| **AUTH-3** | MUST | Accepted credential forms: `Authorization: Bearer <key>` and `X-Api-Key: <key>`. Any other form MUST be rejected with `401`. |
| **AUTH-4** | MUST | Auth MUST be enforced in exactly **one** place — a single middleware. Handlers MUST NOT re-check auth. |
| **AUTH-5** | MUST | `GET /liveness` MUST be public. No other endpoint is public by default. |
| **AUTH-6** | SHOULD | `GET /readiness` and `GET /metrics` SHOULD be public when bound to a non-public interface; configurable. |
| **AUTH-7** | MUST | A path allowlist MUST be supported and MUST match on **segment boundaries only**. `/v1/models` MUST NOT be matched by an allowlist entry of `/v1/mod`. A trailing `/` denotes a subtree (`/v1/` allows `/v1/anything`). |
| **AUTH-8** | ~~MUST~~ **WITHDRAWN** | Originally: an empty allowlist MUST mean deny-by-default, "a deliberate, breaking inversion of v1 semantics". **Not implemented, and the requirement is withdrawn rather than outstanding.** Its premise — that empty-means-allow-all "turns a config typo into an open router" (AUTH-N2) — does not hold: the allowlist gates *reachability*, auth gates *access*, and they are independent. An empty or mistyped allowlist widens the served surface but leaves every path behind auth. Deny-by-default would also mean an unset variable serves nothing at all, kubelet probes included, so the chart default would have to re-list the probes to boot. Shipped behaviour is v1's: empty serves all paths, auth applies to all paths. **Do not put an inversion in the release notes.** |
| **AUTH-9** | MUST | The router MUST NOT forward the client's inbound `Authorization` or `X-Api-Key` header to workers. Upstream credentials, if any, are separately configured (SEC-4). |
| **AUTH-10** | MUST | Auth failures MUST NOT log the presented credential, in whole or in prefix/suffix form. |
| **AUTH-11** | MUST | Admin endpoints (GW-11) MUST require auth even when a path allowlist is configured; the allowlist MUST NOT be able to exempt them. |
| **AUTH-12** | MUST NOT | Remote key validation (`API_KEY_VALIDATION_URLS` fan-out) MUST NOT exist in v2. |

### Must not regress into

- **AUTH-N1:** Double enforcement — a global middleware *plus* an explicit call in ~28 handlers. One of them will drift.
- **AUTH-N2:** ~~Empty-allowlist-means-allow-all, which turns a config typo into an open router.~~ **Retracted with AUTH-8** — it does not turn a typo into an open router, because auth is enforced independently of the allowlist. The real regression to guard is an allowlist that is *silently inactive*, e.g. because the environment variable was renamed away from what the deployment sets.
- **AUTH-N3:** Prefix matching that permits `/v1/modelsXXX` or similar boundary escapes.
- **AUTH-N4:** Verbatim forwarding of the client's `Authorization` header to backend workers (credential leakage into the worker fleet and its logs).
- **AUTH-N5:** Auth ordered outside CORS, breaking preflight.

---

## 7. Worker Model & Registry

| ID | Level | Requirement |
|---|---|---|
| **WRK-1** | MUST | A worker is identified by a **canonical URL** (normalized scheme, lowercase host, explicit port, no trailing slash, no path). Two inputs that canonicalize equal are the same worker. |
| **WRK-2** | MUST | Worker registration MUST be idempotent. Registering an already-present URL MUST update metadata in place and MUST NOT create a duplicate entry in any index. |
| **WRK-3** | MUST | The registry MUST expose a **deterministically ordered snapshot** (`[]*Worker`, sorted by canonical URL) as the single source of worker lists for every policy, admin endpoint, and metric. |
| **WRK-4** | MUST | Snapshots MUST be immutable and copy-on-write: a policy holding a snapshot MUST see a stable list for the whole duration of a routing decision, with no lock held. |
| **WRK-5** | MUST | Per-worker state MUST include: canonical URL, labels (model, zone, discovery source), health status, circuit-breaker state, in-flight count, cumulative counters, last-transition timestamps. |
| **WRK-6** | MUST | Worker removal MUST be graceful: the worker is marked draining, excluded from new selections immediately, and its entry is retained until in-flight reaches zero or a drain deadline (default 60 s) elapses. |
| **WRK-7** | SHOULD | Static workers (config/admin API) and discovered workers (K8s) SHOULD be tracked with distinct provenance; discovery MUST NOT remove a statically-configured worker. |
| **WRK-8** | MUST | `GET /list_workers` and `GET /workers` MUST return workers in the deterministic snapshot order, with health, in-flight, and circuit state included. |

### Must not regress into

- **WRK-N1:** Returning worker lists from a `DashMap` in nondeterministic iteration order. This alone degenerates round-robin into random and makes any `index 0` fallback pin traffic to an arbitrary node that changes between calls.
- **WRK-N2:** Re-registering a URL and having it appended to every secondary index, so a single worker is weighted N times in selection.
- **WRK-N3:** Hard-removing a worker with in-flight requests attached.

---

## 8. Health & Circuit Breaking

| ID | Level | Requirement |
|---|---|---|
| **HLT-1** | MUST | Active health checks MUST run **concurrently across workers**, with a bounded worker pool. Total check-round wall time MUST NOT scale with worker count. |
| **HLT-2** | MUST | Per-check timeout MUST be strictly less than the check interval; config validation MUST reject `timeout >= interval` at startup. |
| **HLT-3** | MUST | Health state MUST be hysteretic: N consecutive failures to mark unhealthy (default 3), M consecutive successes to mark healthy (default 2). |
| **HLT-4** | MUST | The health checker MUST NOT mutate in-flight load counters. Ever. |
| **HLT-5** | MUST | A newly discovered worker MUST start in `Unknown` and become eligible only after its first successful check, unless `--assume-healthy-on-add` is explicitly set. |
| **HLT-6** | MUST | Circuit breaker state per worker: `Closed` → `Open` → `HalfOpen` → (`Closed`\|`Open`). |
| **HLT-7** | MUST | The open trigger MUST be evaluated over a **sliding time window** (default 30 s): open when `failures >= min_requests (default 20)` AND `failure_rate >= threshold (default 0.5)`. If a `window_duration` is configurable it MUST be used. |
| **HLT-8** | MUST | `HalfOpen` MUST admit at most `half_open_max_concurrent` in-flight probes (default 1), enforced by a semaphore-style counter, not by state check alone. |
| **HLT-9** | MUST | Outcome classification MUST be explicit: `5xx`, connection error, and timeout are **failures**; `429` and `503` are **failures** (worker overloaded); `4xx` other than 408/425/429 are **successes** (client's fault, worker healthy); `2xx`/`3xx` are successes. |
| **HLT-10** | MUST | Circuit state transitions MUST be logged at `INFO` with worker URL, old state, new state, and the counters that triggered it. |
| **HLT-11** | MUST | If **all** workers are unhealthy or open-circuited, the router MUST return `503` with a distinguishable error code, and MUST NOT silently route to a known-bad worker. |
| **HLT-12** | SHOULD | The router SHOULD support passive health signals from proxy outcomes in addition to active checks, feeding the same circuit breaker. |

### Must not regress into

- **HLT-N1:** Sequential health checks where `5 s timeout × N workers` can exceed the 60 s interval, so checks silently fall behind and stale health persists indefinitely.
- **HLT-N2:** A `window_duration` field that is configured, documented, and never read — with only consecutive-failure counting actually implemented.
- **HLT-N3:** `HalfOpen` admitting unbounded concurrency, so a recovering worker is instantly re-flooded and re-opens.
- **HLT-N4:** Recording `4xx` including `429` as **success**, which makes the breaker blind to the single most common overload signal.
- **HLT-N5:** The health checker zeroing load counters as a side effect of its cycle.

---

## 9. Load-Balancing Policies

### 9.1 The load-accounting invariant

| ID | Level | Requirement |
|---|---|---|
| **LB-1** | MUST | In-flight load MUST be incremented at exactly **one** point in the code: immediately after a worker is selected, before the upstream request is issued — for **every** policy, without exception. |
| **LB-2** | MUST | The decrement MUST be performed by a single `defer`-guarded release owned by the same scope as the increment. The release MUST be idempotent (guarded by a `sync.Once` or equivalent) so no code path can double-release. |
| **LB-3** | MUST | On retry to a different worker, the first worker's lease MUST be released and a new lease acquired on the second. Leases MUST NOT be transferred or reused. |
| **LB-4** | MUST | For streaming responses, the lease MUST be held until the response body is fully read or the stream is aborted — **not** until headers are received. |
| **LB-5** | MUST | In-flight counters MUST be `uint`-safe: a release that would underflow MUST be treated as a bug, MUST NOT wrap, MUST increment a `router_load_accounting_errors_total` counter, and MUST log at `ERROR`. |
| **LB-6** | MUST | Nothing outside the lease lifecycle may write in-flight counters. Not health checks, not admin endpoints, not discovery, not `flush_cache`. |
| **LB-7** | MUST | A property test MUST exist proving that after an arbitrary interleaving of N request lifecycles (including cancellations, retries, timeouts, upstream errors, and stream aborts), every worker's in-flight count returns to exactly 0. |

### 9.2 Policy contract

| ID | Level | Requirement |
|---|---|---|
| **LB-8** | MUST | Every policy MUST implement one narrow interface: `Select(ctx, candidates []*Worker, req *RoutingRequest) (*Worker, error)` plus an optional `Commit(worker, req)` hook (used by cache policies, CACHE-9). |
| **LB-9** | MUST | `candidates` MUST already be filtered to **healthy, non-draining, closed-or-half-open-circuit** workers by the caller. A policy MUST NOT see or reason about unhealthy workers. |
| **LB-10** | MUST | A policy MUST be deterministic given (candidates, request, internal state) except where randomness is the policy. |
| **LB-11** | MUST | All tie-breaks MUST be resolved by an explicit rule that is **not** "lowest index": either a per-policy rotating cursor or uniform random among the tied set. Tie-break behavior MUST be unit-tested with N identical candidates for uniformity. |
| **LB-12** | MUST | Every policy MUST behave correctly for `len(candidates)` of 0 (return a typed `ErrNoCandidates`), 1, and 2. |

### 9.3 Shipped policies

**LB-13 (MUST) — `least-outstanding` (DEFAULT).**
Select the candidate with the smallest in-flight count. Ties broken per LB-11. Reads the lease counter from §9.1, which is the only trusted load signal. This is the default policy for all configurations.

**LB-14 (MUST) — `round-robin`.**
Strict rotation over the deterministically-ordered snapshot (WRK-3) using a monotonically increasing atomic cursor modulo `len(candidates)`. When the candidate set changes size, the cursor MUST be remapped such that no worker is starved — a rotation over a set that shrinks then grows MUST still visit every member. Verified by a test asserting that over `10 × N` requests to `N` stable workers, every worker receives exactly 10.

**LB-15 (MUST) — `random`.**
Uniform selection using `math/rand/v2`. Chi-square uniformity test over 100k draws across 8 workers.

**LB-16 (MUST) — `prefix-cache-aware`.** See §10.

**LB-17 (MUST) — `cache-usefulness`.** See §11.

**LB-18 (SHOULD) — `power-of-two`** MAY be offered as a variant of `least-outstanding` (sample 2, pick lower). If shipped it MUST use the same lease counter and MUST NOT introduce a separate "load check interval" concept.

**LB-19 (MUST NOT).** `consistent_hash`, `rendezvous_hash`, and `hash_key` from v1 MUST NOT be ported. Session/prefix affinity is served by LB-16 and LB-17.

**LB-20 (MUST).** Every configurable knob accepted by a policy MUST be read by that policy. Config validation MUST fail startup on any knob that the selected policy does not consume.

### Must not regress into

- **LB-N1:** Load incremented on one path, decremented on three (v1: normal completion + retryable-status + body-read-error), producing systematic double-decrement.
- **LB-N2:** The health checker wiping all load counters every 10 cycles.
- **LB-N3:** Round-robin over a nondeterministically ordered map, which is just random with extra steps.
- **LB-N4:** `min_by_key`-style tie-breaks that always return the first element, producing a synchronized thundering herd onto one worker on cold start (v1: 32-deep queue on a single node while peers idle).
- **LB-N5:** Accepting `PowerOfTwo.load_check_interval_secs` and `ConsistentHash.virtual_nodes` in config and silently discarding them.
- **LB-N6:** A policy default that differs between the Rust CLI, the PyO3 binding, and the Python CLI (v1: `cache_aware` in one, `RoundRobin` in another).
- **LB-N7:** Computing a load-imbalance decision over the full worker list including unhealthy workers, so one dead worker holding a stale nonzero load permanently disables the policy.

---

## 10. Cache-Aware Routing (`prefix-cache-aware`)

This is the **rebuilt** prefix policy. It is not a port of `src/tree.rs` (2,320 LOC) or `src/policies/cache_aware.rs`.

| ID | Level | Requirement |
|---|---|---|
| **CACHE-1** | MUST | Maintain one radix/prefix structure **per worker**, keyed on the request's routing text (CACHE-4), recording which worker most recently served each prefix. |
| **CACHE-2** | MUST | Selection: compute the longest matching prefix across per-worker structures; if `matched_fraction >= cache_threshold` (default 0.5) select the owning worker; otherwise fall back to the configured fallback policy (default `least-outstanding`). |
| **CACHE-3** | MUST | An imbalance guard MUST override cache affinity: if the selected worker's in-flight exceeds `min(in-flight over candidates) + abs_threshold` **and** `> min × rel_threshold`, fall back to `least-outstanding`. The guard MUST be computed **over `candidates` only** (LB-9), never over unhealthy workers. |
| **CACHE-4** | MUST | "Routing text" MUST be derived from a documented, versioned extraction rule (see CACHE-11), computed once per request and reused by all policies and by observability. |
| **CACHE-5** | MUST | Eviction MUST be bounded by **estimated tokens**, not characters, with a default budget of `max_tokens_per_worker` (default 2,000,000 est. tokens ≈ 8 MB of prompt text). Configurable. |
| **CACHE-6** | MUST | Token estimation MUST be `len(bytes)/4`, clamped to ≥1 for non-empty content — the same heuristic as `wekai`'s `estimateTokens`. The router does not tokenize (NG-3). The estimate's crudeness MUST be documented at every point where it influences a decision. |
| **CACHE-7** | MUST | Eviction MUST be LRU, MUST be incremental (amortized per-insert or a bounded work budget per tick), and MUST hold the same lock discipline as insertion so that no reader can observe a partially-unlinked subtree. |
| **CACHE-8** | MUST | All background work MUST run on a goroutine owned by a `context.Context` that is cancelled on policy shutdown. No unkillable threads. Leak-checked with `go.uber.org/goleak` in tests. |
| **CACHE-9** | MUST | The structure MUST be split into a **read-only `Query`** used during selection and a **`Commit`** applied only after a worker is actually selected and the request dispatched. A request that is rejected, rate-limited, or fails candidate filtering MUST NOT mutate cache state. |
| **CACHE-10** | MUST | On worker removal, that worker's structure MUST be dropped entirely. Its prefixes MUST NOT be reassigned to any other worker. |
| **CACHE-11** | MUST | Routing-text extraction MUST be pluggable per endpoint: `/v1/chat/completions` → concatenation of message roles+contents in order; `/v1/completions` → `prompt`; `/v1/embeddings` → `input`; `/generate` → `text`/`inputs`. Unknown shape → policy declines and falls back. |
| **CACHE-12** | SHOULD | The prefix structure SHOULD segment on a fixed unit size aligned to vLLM's block hashing granularity (see §11 and OQ-2), not on whole messages. |
| **CACHE-13** | MUST | `POST /flush_cache` MUST clear all per-worker prefix structures and MUST NOT touch load counters, health state, or circuit state. |

### Must not regress into

- **CACHE-N1:** Evicting by **character count** with a default of ~10,000 chars/worker — roughly 2–3 realistic prompts, meaning the tree is thrashed continuously and cache affinity is effectively random.
- **CACHE-N2:** A stop-the-world DFS eviction every 30 s that races structural mutation and can unlink live subtrees.
- **CACHE-N3:** Leaking an unkillable OS thread per policy instance (v1: `eviction_handle` in an infinite loop with an explicit code comment admitting it cannot be stopped).
- **CACHE-N4:** Computing the imbalance check over unhealthy workers.
- **CACHE-N5:** Tie-breaking to index 0 on a cold tree, producing a 32-deep thundering herd.
- **CACHE-N6:** Mutating cache state during a speculative/abandoned selection.

---

## 11. The `cache-usefulness` Policy (wekai-derived)

### 11.1 Naming correction

The user referred to this as `wekai benchmark analyze`. The actual command is **`wekai router analyze`**, implemented in `/Users/ofer.kiselovnahman/workspace/wekai/cli/command_router_analyze.go` (736 LOC). The reusable engine is `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/cache_sim.go` (281 LOC).

### 11.2 What wekai actually gives us (verified)

`benchmark/cache_sim.go` depends only on `crypto/sha256`, `encoding/hex`, `sync`, `sync/atomic` — no wekai-internal packages. It contains:

- `prefixTrie` / `trieNode` — a mutex-guarded trie of prefix-unit hashes; each node stores the estimated token delta of the unit that created it.
- `prefixTrie.RecordAndCount(hashes []string, tokens []int) (cached, total int)` — walks the longest matching prefix crediting `child.tokens` per matched node, then inserts the novel tail. O(prefix length). Online-capable.
- `cacheEstimator.Observe(content string) float64` — chunks raw prompt bytes into `promptChunkBytes = 1024`-byte windows (~256 est. tokens), hashes each with `hashMessage("chunk", …)` (SHA-256 truncated to 64 bits / 16 hex chars), walks/extends the trie, and returns the predicted cached **fraction** in `[0,1]`.
- `estimateTokens` = `len(content)/4`, min 1.

`Observe` is already on a hot path (`benchmark/replay_router_post.go:307`, `benchmark/auto.go:1675`), so its cost profile is known-acceptable.

**Prefix-unit definition** — `BuildReplayRequestPrefix` (`benchmark/replay_router.go:731`, exported) orders units as: **system blocks → tools → messages**, and **skips system block index 0 when `Bytes < 200`**, because that per-request billing header carries a near-unique hash that poisons vLLM's sequential prefix-block hashing. This heuristic is hard-won and must survive the port.

**Cost model** — `benchmark/replay_router_post.go:692-716` and `dryRunDurations` model `TTFT ≈ cold/coldTPS + warm/warmTPS` against three measured rates (`coldTPS`, `warmTPS`, `outputTPS`). The value of steering a request to a warm node is therefore:

```
saved_seconds ≈ predicted_warm_tokens × (1/coldTPS − 1/warmTPS)
```

### 11.3 Gaps — what must be built new

Be explicit with reviewers: wekai gives us a **validated single-cache simulator**, not a router. The following are new work:

1. **Singleton → per-node.** The trie is one global instance. The router needs one per worker, plus lifecycle tied to worker add/remove.
2. **Walk-and-mutate → query/commit split.** `Observe()` walks *and* inserts under one lock. The router must query all N workers read-only, then commit to exactly one.
3. **No eviction.** The trie models an **infinite** cache. Real vLLM nodes evict under LRU / KV-memory pressure. Unbounded, the prediction drifts optimistic and memory grows without limit.
4. **Unit granularity mismatch.** wekai's units are whole messages (thousands of tokens) or 1024-byte windows; vLLM hashes **fixed 16-token blocks**. Nothing in wekai models vLLM's block size. Prediction resolution is therefore coarse by a factor of ~16–100.
5. **No per-node ground truth.** wekai has no per-node cache-content tracking. `llm/endpoint_router.go` gets locality only via sticky series assignment. `benchmark/vllm_metrics.go` polls `vllm:prompt_tokens_by_source` every 60 s but **sums across all endpoints**, discarding per-node attribution. Per-node validation must be built.
6. **Unexported and in package `benchmark`.** `prefixTrie`, `trieNode`, `cacheEstimator` are all lowercase inside a package that pulls in the whole benchmark harness.

### 11.4 Requirements

| ID | Level | Requirement |
|---|---|---|
| **CU-1** | MUST | The engine MUST be extracted into a standalone, dependency-free package (proposal: `github.com/weka/wllm-router/internal/cachepredict`) that imports only the standard library. |
| **CU-2** | MUST | The extraction MUST be a **shared-source or vendored-copy with a documented provenance header** naming `wekai@<commit> benchmark/cache_sim.go`, and a CI check that flags upstream drift. Cross-repo import of `benchmark` MUST NOT be used (it would drag the benchmark harness into the router binary). |
| **CU-3** | MUST | `BuildReplayRequestPrefix`'s ordering (system → tools → messages) and its **`i == 0 && Bytes < 200` skip rule** MUST be preserved in the router's prefix-unit builder, with the rationale documented in a comment and asserted by a unit test named after the behavior. |
| **CU-4** | MUST | Maintain one predictor instance **per worker**, created on worker registration and destroyed on removal. Memory is per-worker-bounded (CU-8). |
| **CU-5** | MUST | The API MUST be `Query(units []Unit) (predictedCachedTokens, totalTokens int)` — pure, read-only, safe for concurrent use across all workers — and `Commit(units []Unit)` — mutating, called on exactly one worker after selection succeeds. |
| **CU-6** | MUST | Selection MUST maximize **expected saved time**, not raw predicted hit ratio: `score(w) = predictedCachedTokens(w) × (1/coldTPS − 1/warmTPS) − queueingPenalty(inflight(w))`. All three rates MUST be configurable, with defaults documented as measured-on-our-hardware and clearly labeled as such. |
| **CU-7** | MUST | `queueingPenalty` MUST be a documented function of in-flight count, so a very warm but saturated node loses to a cold idle node. At minimum: `inflight(w) × est_service_time`. |
| **CU-8** | MUST | Each per-worker predictor MUST be bounded by node count **and** estimated tokens, with LRU eviction, defaults `500k nodes` / `2M est. tokens` per worker. This is the new-work item that models vLLM's real KV eviction. |
| **CU-9** | SHOULD | Eviction SHOULD be parameterized by the worker's advertised KV cache capacity when known (from `/get_server_info` or vLLM metrics), so the model tracks the real cache size rather than a fixed constant. |
| **CU-10** | MUST | Prefix units MUST be built at a configurable granularity, default 1024 bytes (`promptChunkBytes`), with the vLLM 16-token-block mismatch documented (see OQ-2). |
| **CU-11** | MUST | Degradation: if routing text cannot be extracted, if all predictors return `total == 0`, or if the best score is within `epsilon` of the worst, the policy MUST fall back to `least-outstanding` and increment `router_cache_usefulness_fallback_total{reason=…}`. The policy MUST NEVER fail a request. |
| **CU-12** | MUST | On worker removal, that worker's predictor MUST be dropped. Its prefixes MUST NOT be reassigned. |
| **CU-13** | SHOULD | The router SHOULD record, per request, `predicted_cached_fraction` alongside the worker's **reported** `prompt_tokens_details.cached_tokens` (when vLLM returns it in `usage`) and emit both, so prediction accuracy is measurable in production. This is the per-node ground truth wekai lacks. |
| **CU-14** | MAY | The router MAY expose a `shadow` mode where `cache-usefulness` computes its choice and emits metrics but the actual routing decision is taken by the configured production policy — for safe accuracy validation before cutover. |
| **CU-15** | MUST | Prediction MUST NOT be on the critical path for more than the NFR-2 budget. `Query` across N workers is O(N × prefix_units); with N ≤ 64 and units ≤ 64 this is bounded, but MUST be benchmarked and MUST short-circuit (fall back) if it exceeds a hard deadline (default 2 ms). |

### Must not regress into

- **CU-N1:** A global singleton trie shared across workers, which predicts the cluster's aggregate cache, not any node's.
- **CU-N2:** A combined walk-and-insert call, which pollutes every worker's model with every request regardless of where it was actually sent — making the prediction converge to uniform and useless.
- **CU-N3:** An unevicting trie that models an infinite cache and grows without bound.
- **CU-N4:** Summing cache metrics across endpoints and discarding per-node attribution (wekai's current `vllm_metrics.go` behavior), which is exactly the signal this policy needs.
- **CU-N5:** Importing `github.com/weka/wekai/benchmark` into the router, pulling a benchmark harness and its transitive dependencies into a production proxy.

---

## 12. Service Discovery (Kubernetes)

| ID | Level | Requirement |
|---|---|---|
| **SD-1** | MUST | Support discovery via **EndpointSlice** for a named Service (namespace + service name), in addition to label-selected Pods. EndpointSlice MUST be the recommended default. |
| **SD-2** | MUST | Support discovery via **Pod label selector** with configurable port (v1 parity). |
| **SD-3** | MUST | Use a shared informer / watch with resync, not a poll loop, and MUST handle watch expiry and re-list with exponential backoff capped at 5 min. |
| **SD-4** | MUST | Discovery MUST only propose workers; the registry (WRK-*) decides admission, and health (HLT-5) decides eligibility. A discovered pod MUST NOT be routable before its first successful health check. |
| **SD-5** | MUST | Endpoint readiness (`conditions.ready`) MUST be honored; not-ready endpoints MUST be removed from candidates. |
| **SD-6** | MUST | Removal of a discovered endpoint MUST go through graceful drain (WRK-6). |
| **SD-7** | MUST | Reconciliation MUST be **idempotent and convergent**: applying the same observed endpoint set twice MUST produce identical registry state (directly addresses WRK-N2). |
| **SD-8** | MUST | Required RBAC MUST be documented and minimal: `get`/`list`/`watch` on `endpointslices` (and `pods` if SD-2 is used) in the configured namespace(s). |
| **SD-9** | SHOULD | IPv6 endpoints SHOULD be supported, with correct `[addr]:port` URL bracketing. |
| **SD-10** | MUST | The router MUST run correctly with discovery disabled and a static worker list. |
| **SD-11** | MUST NOT | PD-mode discovery (prefill/decode pod classification, bootstrap ports) MUST NOT exist (NG-1). |

### Must not regress into

- **SD-N1:** Pod-IP-only discovery with no Service/EndpointSlice support, which breaks the moment workers sit behind a Service or a headless Service.
- **SD-N2:** Non-idempotent reconciliation that duplicates a URL across secondary indices, weighting one worker N times.
- **SD-N3:** Discovery-driven registration that bypasses health gating and immediately routes to a pod that has not yet loaded its model.

---

## 13. Streaming & Backpressure

| ID | Level | Requirement |
|---|---|---|
| **STR-1** | MUST | Streaming responses MUST be relayed with **direct backpressure**: the router MUST NOT buffer more than a bounded amount (default 64 KiB) between upstream read and downstream write. A slow client MUST slow the upstream read. |
| **STR-2** | MUST | The upstream `Content-Type` MUST be passed through **unmodified**. The router MUST NOT force `text/event-stream`. |
| **STR-3** | MUST | If the upstream returns a non-2xx status on a streaming request, the router MUST relay the upstream status and body as-is (typically `application/json`), and MUST NOT wrap it in SSE framing. |
| **STR-4** | MUST | SSE terminal-event detection (`data: [DONE]`) MUST be performed by a **line-oriented scanner over the byte stream**, correct across arbitrary chunk boundaries. It MUST NOT use a fixed-size trailing-byte window. |
| **STR-5** | MUST | The response MUST be flushed to the client per upstream chunk (`http.Flusher`), with no proxy-side coalescing that would inflate inter-token latency. |
| **STR-6** | MUST | Upstream disconnect mid-stream MUST terminate the client stream promptly, emit a `router_stream_aborted_total{reason}` metric, log at `WARN` with request id and worker, and release the load lease exactly once. |
| **STR-7** | MUST | Client disconnect mid-stream MUST cancel the upstream request within 100 ms (GW-15). |
| **STR-8** | MUST | A per-stream idle timeout (no bytes from upstream) MUST be enforced, default 300 s, configurable, distinct from the overall request timeout. |
| **STR-9** | MUST | Streaming requests MUST NOT be retried once the first response byte has reached the client (see REL-3). |
| **STR-10** | SHOULD | Where possible the implementation SHOULD use `httputil.ReverseProxy` with `FlushInterval: -1` rather than a hand-rolled copy loop. |

### Must not regress into

- **STR-N1:** An **unbounded** `mpsc` channel between backend and client, where a slow client causes unbounded router memory growth (the router absorbs the entire generation).
- **STR-N2:** Force-rewriting `Content-Type` to `text/event-stream` even for JSON error bodies, so clients see a malformed SSE stream instead of a parseable error.
- **STR-N3:** Detecting `data: [DONE]` with a 12-byte sliding window that misses the sentinel whenever it is split across chunk boundaries.

---

## 14. Configuration

| ID | Level | Requirement |
|---|---|---|
| **CFG-1** | MUST | Exactly one configuration model. Precedence: CLI flag > environment variable > config file > default. Every setting MUST be reachable by all three mechanisms. |
| **CFG-2** | MUST | Every flag MUST be defined in the flag parser. No hand-rolled `os.Args` scanning outside it. |
| **CFG-3** | MUST | Defaults MUST be defined in exactly one place. There MUST be exactly one binary and one default set. |
| **CFG-4** | MUST | Default policy MUST be `least-outstanding` (LB-13). |
| **CFG-5** | MUST | Configuration MUST be validated at startup with a **fail-fast, aggregated** error report listing all problems, not just the first. |
| **CFG-6** | MUST | Validation MUST reject: unknown keys, knobs not consumed by the selected policy (LB-20), `health_check_timeout >= health_check_interval` (HLT-2), a body limit of 0 or unbounded, and a circuit-breaker window of 0. |
| **CFG-7** | MUST | `GET /get_server_info` MUST report the effective, post-validation configuration with all secrets redacted. |
| **CFG-8** | MUST | The API key MUST be accepted only via environment variable or a file path, never as a CLI flag value (it would appear in `ps` output). |
| **CFG-9** | MUST | The router MUST NOT print raw command-line arguments to stdout at any log level. |
| **CFG-10** | MUST | Logging MUST be configured (level, format, destination) before any other subsystem emits output. |
| **CFG-11** | SHOULD | Policy names SHOULD be the kebab-case identifiers used in this document: `least-outstanding`, `prefix-cache-aware`, `cache-usefulness`, `round-robin`, `random`. Unknown names MUST fail startup with the list of valid names. |

### Must not regress into

- **CFG-N1:** `--prefill` parsed by hand outside the flag library.
- **CFG-N2:** `println!("DEBUG: Raw args: {:?}", args)` dumping the full command line — including any secret passed as a flag — to stdout **before logging is configured** (so it cannot be suppressed by log level).
- **CFG-N3:** Three divergent default sets (Rust CLI, PyO3 binding, Python CLI) where the default policy is `cache_aware` in one and `RoundRobin` in another.
- **CFG-N4:** Silently accepting and discarding configuration (`PowerOfTwo.load_check_interval_secs`, `ConsistentHash.virtual_nodes`).

---

## 15. Observability

| ID | Level | Requirement |
|---|---|---|
| **OBS-1** | MUST | Structured JSON logging with a configurable level (`log/slog`). Every request-scoped log line MUST carry `request_id`, `route`, `policy`, `worker`, `status`, `duration_ms`. |
| **OBS-2** | MUST | Prometheus metrics on a separate listener (GW-13). |
| **OBS-3** | MUST | The metric set MUST be **small and every metric MUST be emitted**. Required minimum: `router_requests_total{route,status}`, `router_request_duration_seconds{route}`, `router_ttft_seconds{route}`, `router_worker_inflight{worker}`, `router_worker_health{worker}`, `router_circuit_state{worker,state}`, `router_routing_decision_duration_seconds{policy}`, `router_policy_selections_total{policy,worker}`, `router_policy_fallback_total{policy,reason}`, `router_upstream_errors_total{worker,kind}`, `router_stream_aborted_total{reason}`, `router_load_accounting_errors_total`, `router_workers_total{state}`. |
| **OBS-4** | MUST | Cache-policy metrics: `router_cache_predicted_fraction` (histogram), `router_cache_observed_fraction` (histogram, from vLLM `usage.prompt_tokens_details.cached_tokens` when present), `router_cache_entries{worker}`, `router_cache_evictions_total{worker}`. |
| **OBS-5** | MUST | A CI check MUST assert that every registered metric name is referenced by at least one emission site. Dead metric families are a build failure. |
| **OBS-6** | SHOULD | OpenTelemetry tracing SHOULD be supported and MUST be zero-overhead-by-default when disabled (no span allocation on the hot path). |
| **OBS-7** | MUST | Access logs MUST cover **every** request, including catch-all-proxied ones (GW-9). |
| **OBS-8** | MUST | `router_routing_decision_duration_seconds` MUST measure only the policy `Select` call, so NFR-2 is directly observable. |

### Must not regress into

- **OBS-N1:** An entire metric family registered for a subsystem that does not exist (v1 registers tokenizer metrics; routing never tokenizes).
- **OBS-N2:** Load gauges (`vllm_router_worker_load`, `max_load`, `min_load`) fed from a corrupt counter, silently misleading operators.
- **OBS-N3:** A proxy path with no access log and no trace span.

---

## 16. Reliability & Retries

| ID | Level | Requirement |
|---|---|---|
| **REL-1** | MUST | Retries MUST be **bounded** (default `max_attempts = 2`, i.e. one retry) and MUST always select a **different** worker than the failed attempt where one is available. |
| **REL-2** | MUST | Retryable conditions: connection refused/reset, TLS handshake failure, `502`/`503`/`504`, and request timeout **before any response byte**. `500` MUST NOT be retried by default (may be non-idempotent work already performed). |
| **REL-3** | MUST | A request MUST NOT be retried once any response byte has been written to the client. |
| **REL-4** | MUST | Retrying requires the request body; the router MUST buffer it up to the GW-8 limit for replay, and MUST NOT retry if the body was streamed beyond the buffer. |
| **REL-5** | MUST | Each attempt MUST acquire and release its own load lease (LB-3). |
| **REL-6** | MUST | A global request deadline MUST bound total time across all attempts, default 600 s, configurable per route class. |
| **REL-7** | MUST | Retries MUST be recorded: `router_retries_total{reason,outcome}`. |
| **REL-8** | MUST | Graceful shutdown: on `SIGTERM` the router MUST (a) immediately fail `GET /readiness`, (b) stop accepting new connections after a configurable pre-drain delay (default 5 s, to let load balancers observe the readiness flip), (c) allow in-flight requests including streams to complete up to a drain deadline (default 120 s), (d) then force-close and exit non-zero if any remain. |
| **REL-9** | MUST | The router MUST NOT panic-crash a request into a process exit; a top-level recover MUST convert a handler panic into `500`, a logged stack, and `router_panics_total`. |
| **REL-10** | SHOULD | Outbound concurrency to a single worker SHOULD be limitable (`max_inflight_per_worker`), returning `503` or queueing per config when exceeded. |

---

## 17. Security

| ID | Level | Requirement |
|---|---|---|
| **SEC-1** | MUST | Constant-time inbound key comparison (AUTH-1). |
| **SEC-2** | MUST | Body-size limit enforced universally (GW-8). |
| **SEC-3** | MUST | Hop-by-hop headers (`Connection`, `Keep-Alive`, `Proxy-*`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`) MUST be stripped in both directions. |
| **SEC-4** | MUST | The client's credentials MUST NOT reach workers (AUTH-9). If workers require auth, a separately configured upstream credential is injected. |
| **SEC-5** | MUST | Worker URLs supplied via the admin API MUST be validated against an allowed scheme set (`http`, `https`) and, when configured, a host allowlist/CIDR allowlist — mitigating SSRF via `POST /add_worker`. |
| **SEC-6** | MUST | Secrets MUST NOT appear in logs, `/get_server_info`, metrics labels, or error messages. |
| **SEC-7** | MUST | The container image MUST run as a non-root user, with a read-only root filesystem and no shell in the final layer (distroless or scratch + static binary). |
| **SEC-8** | MUST | Dependencies MUST be scanned in CI (`govulncheck`); a known-exploitable CVE in a direct dependency MUST fail the build. |
| **SEC-9** | SHOULD | `X-Forwarded-For` / `X-Forwarded-Proto` handling SHOULD be explicit and configurable (trust-proxy setting), defaulting to not trusting inbound forwarded headers. |
| **SEC-10** | MUST | CORS allowed origins MUST default to none (same-origin only); `*` MUST require explicit opt-in and MUST be rejected in combination with credentials. |

---

## 18. Non-Functional Requirements

| ID | Level | Requirement |
|---|---|---|
| **NFR-1** | MUST | Throughput: ≥ 20,000 non-streaming proxied req/s on 8 vCPU against a null-latency mock worker with 1 KiB request / 1 KiB response bodies, at ≤ 70% CPU. |
| **NFR-2** | MUST | Routing-decision overhead (policy `Select` only, OBS-8): **p50 ≤ 50 µs, p99 ≤ 250 µs** for `least-outstanding` / `round-robin` / `random` with 64 workers; **p99 ≤ 1 ms** for `prefix-cache-aware` and `cache-usefulness` with 64 workers and a 32 KiB prompt. Hard deadline with fallback at 2 ms (CU-15). |
| **NFR-3** | MUST | End-to-end added latency (router in path vs. direct-to-worker), non-streaming, p99 ≤ 3 ms at NFR-1 load. |
| **NFR-4** | MUST | Streaming: added inter-token latency p99 ≤ 1 ms; 10,000 concurrent open streams sustained on 8 vCPU / 4 GiB. |
| **NFR-5** | MUST | Memory: total RSS MUST be bounded and predictable. `RSS ≤ base(256 MiB) + workers × cache_budget_bytes + concurrent_streams × 64 KiB`. A 24 h soak at NFR-1 load MUST show no unbounded growth (< 5% RSS drift after the first hour). |
| **NFR-6** | MUST | Cold start to serving: ≤ 2 s excluding first health-check round. |
| **NFR-7** | MUST | Graceful shutdown per REL-8. |
| **NFR-8** | SHOULD | Container image ≤ 40 MiB. |
| **NFR-9** | MUST | No goroutine leaks: `goleak` assertions in every package's `TestMain`. |
| **NFR-10** | MUST | The routing path MUST be race-free under `-race` for the full integration suite. |
| **NFR-11** | SHOULD | Total non-test, non-generated Go LOC ≤ 8,000 (G6). Reviewed at each milestone; exceeding it requires an explicit justification in the PR.

---

## 19. Open Questions / Deferred to Design

- **OQ-1: Keep the transparent catch-all proxy at all?** It exists in v1 to forward unknown vLLM endpoints. Keeping it means every future vLLM endpoint works for free; dropping it means a strictly enumerated surface, which is easier to secure and reason about. **Recommendation:** keep it, but behind an explicit `--enable-passthrough` flag defaulting to **off**, and fully inside the middleware chain (GW-9).
- **OQ-2: Prefix-unit granularity.** vLLM hashes fixed **16-token** blocks; wekai uses 1024-byte (~256-token) windows or whole messages. Finer units = better prediction resolution, ~16× more trie nodes and hashing cost. Needs an offline accuracy/cost sweep against `wekai router analyze` traces before fixing the default (CU-10, CACHE-12).
- **OQ-3: Do `prefix-cache-aware` and `cache-usefulness` merge?** They share a trie, a routing-text extractor, and an eviction model; they differ in scoring (longest-match owner vs. expected-saved-time). Shipping both may be redundant. **Recommendation:** implement one engine with two scoring functions; decide before GA whether to expose both names.
- **OQ-4: Multi-model routing.** If one router fronts workers serving different models, candidate filtering must be model-aware. Currently out of scope (NG-6); confirm no deployment needs it.
- **OQ-5: Responses-API affinity durability.** The `{id}` → worker map is in-memory (GW-3/GW-4). Multi-replica routers or restarts break `GET /v1/responses/{id}`. Options: sticky-hash on the id (works only if worker set is stable), shared store (adds a dependency), or accept the limitation. **Recommendation:** accept and document for v2.0; revisit if the Responses API is actually used.
- **OQ-6: Where do `coldTPS` / `warmTPS` / `outputTPS` come from (CU-6)?** Static config from a benchmark run, or continuously estimated from observed TTFT? Continuous estimation is better but adds a feedback loop that can oscillate. **Recommendation:** static config for v2.0, with the measured values recorded in the release notes; continuous estimation as a follow-up behind a flag.
- **OQ-7: Is `power-of-two` (LB-18) worth shipping** given a correct `least-outstanding` with an O(N) scan is fast enough at N ≤ 64 (NFR-2)? Likely no.
- **OQ-8: Per-worker KV capacity discovery (CU-9)** — does the deployed vLLM version expose usable cache-size info via `/get_server_info` or `/metrics`? Needs verification against the actual worker build.
- **OQ-9: Metric-name migration (COMPAT-4).** Renaming breaks existing dashboards/alerts. Option: dual-emit old `vllm_router_*` names for one release behind `--emit-legacy-metrics`.

---

## 20. Acceptance Criteria

Each area is signed off only when all listed checks pass in CI.

### 20.1 Test infrastructure (prerequisite)

- **AC-0.1:** A **mock vLLM worker** exists as a Go test helper with programmable behavior: latency, status codes, SSE streaming with configurable inter-token delay, mid-stream abort, connection reset, slow-body, and `usage.prompt_tokens_details.cached_tokens` in responses.
- **AC-0.2:** A **deterministic clock** abstraction is used by health checks, circuit breakers, TTL/LRU eviction, and drain — so all time-dependent logic is unit-testable without sleeps.
- **AC-0.3:** All packages run under `-race` and `goleak`.

### 20.2 Per-area verification

| Area | Verification |
|---|---|
| **Gateway/HTTP** | Unit: routing table, request-id handling, OpenAI error envelope. Integration: every GW-1..GW-5 endpoint against the mock worker, asserting byte-identical body forwarding (COMPAT-2). Negative: a 1 GiB body on the catch-all path returns `413` with bounded RSS growth (**GW-N1**). Negative: `OPTIONS` preflight succeeds with auth enabled (**GW-N3**). Negative: catch-all request appears in access log with a request id (**GW-N2**). |
| **Auth** | Unit: constant-time compare (timing variance test), allowlist segment-boundary table test including `/v1/mod` vs `/v1/models` (**AUTH-N3**). Unit: empty allowlist denies (**AUTH-N2**). Integration: worker receives no `Authorization`/`X-Api-Key` from client (**AUTH-N4**). Grep-based CI check: exactly one auth call site (**AUTH-N1**). |
| **Registry** | Unit: canonicalization table; register-same-URL-twice yields one entry across all indices (**WRK-N2**). Property: snapshot order is stable and sorted across 1,000 randomized mutation sequences (**WRK-N3**/**LB-N3**). Integration: drain holds an in-flight stream to completion. |
| **Health & CB** | Unit: hysteresis, sliding-window open trigger with a fake clock (**HLT-N2**), half-open concurrency cap under 100 concurrent probes (**HLT-N3**), `429` classified as failure (**HLT-N4**). Integration: 200 workers with 5 s-timeout checks complete a round in < 6 s (**HLT-N1**). Grep/CI: no write to in-flight counters outside the lease package (**HLT-N5**, LB-6). |
| **Load balancing** | Property (LB-7): random interleaving of 10k lifecycles — cancel/retry/timeout/upstream-error/stream-abort — leaves every in-flight counter at exactly 0 (**LB-N1**). Unit: `round-robin` over N stable workers gives each exactly `10` of `10N` requests (**LB-N3**). Unit: tie-break uniformity, chi-square over 100k draws with N identical candidates (**LB-N4**). Unit: `least-outstanding` picks correctly for N=1,2,64. Config test: an unconsumed knob fails startup (**LB-N5**). Config test: exactly one default set, default is `least-outstanding` (**LB-N6**). Unit: imbalance guard sees only healthy candidates (**LB-N7**). |
| **Cache-aware** | Unit: eviction is token-budgeted and the tree survives 1M inserts within the memory bound (**CACHE-N1**). Race: 8 concurrent inserters + 1 evictor under `-race` for 60 s, with an invariant checker asserting no reader observes an unlinked live subtree (**CACHE-N2**). `goleak`: policy shutdown leaves zero goroutines (**CACHE-N3**). Unit: rejected request performs no commit (**CACHE-N6**). Integration: 1,000-request session trace shows ≥ 80% affinity to the same worker under a stable worker set. |
| **Cache-usefulness** | Unit: golden test that `Query` reproduces `wekai` `cacheEstimator.Observe` outputs on a fixed corpus (proves faithful extraction). Unit: the `Bytes < 200` system-block-0 skip is asserted by name (**CU-3**). Unit: `Query` is pure — 1,000 `Query` calls leave state byte-identical (**CU-N2**). Unit: per-worker isolation — committing to worker A does not change worker B's prediction (**CU-N1**). Unit: eviction bounds node count and tokens (**CU-N3**). Bench: `Query` across 64 workers with a 32 KiB prompt meets NFR-2. Integration: `predicted_cached_fraction` vs. mock-reported `cached_tokens` correlate ≥ 0.7 on a replayed `wekai router analyze` trace. Degradation: with routing text unavailable, all requests still succeed via fallback with the fallback counter incremented (**CU-11**). |
| **Service discovery** | Integration against `envtest` or a fake clientset: EndpointSlice add/update/delete converges (SD-7 — apply the same set twice, assert identical registry state, **SD-N2**). Assert a newly discovered endpoint is not routable until its first health check passes (**SD-N3**). Watch-expiry re-list with backoff. IPv6 URL bracketing. |
| **Streaming** | Integration: slow client (1 byte/s reader) against a fast worker — assert router RSS stays within `64 KiB × streams` (**STR-N1**). Integration: worker returns `400 application/json` on a `stream:true` request — assert `Content-Type` is passed through unchanged (**STR-N2**). Unit: `data: [DONE]` split across every possible byte offset (parameterized over all split points) is detected (**STR-N3**). Integration: client disconnect cancels upstream within 100 ms. |
| **Reliability** | Integration: worker returns `503` → retried on a different worker; `500` → not retried; retry after first byte → not retried (REL-3). Chaos: **kill a worker mid-stream** — assert client sees a clean termination, `router_stream_aborted_total` increments, the lease releases exactly once, the circuit opens, and subsequent requests route away within one health interval. Chaos: kill a worker between selection and dispatch. Chaos: `SIGTERM` with 100 open streams — assert readiness flips immediately, all streams complete, exit code 0 within the drain deadline. |
| **Config** | Table test: every setting reachable via flag, env, and file with correct precedence. Test: unknown key fails. Test: `timeout >= interval` fails. Test: stdout contains no argv dump at any log level (**CFG-N2**). Test: the API key cannot be set via a CLI flag (CFG-8). |
| **Observability** | CI check: every registered metric has ≥ 1 emission site (**OBS-N1**). Integration: `/metrics` scrape after a mixed workload contains every OBS-3 metric with non-default values. Test: `/metrics` is not reachable on the inference listener. |
| **Security** | `govulncheck` clean. Image scan: non-root, no shell. Test: `POST /add_worker` with `file://` and with a disallowed host is rejected (SEC-5). Test: `/get_server_info` output contains no secret material. |
| **Non-functional** | Load test (k6 or `vegeta`) against the mock worker asserting NFR-1/NFR-3. Streaming load test asserting NFR-4. 24 h soak asserting NFR-5. Benchmark suite in CI with regression gates on NFR-2, failing the build on > 20% regression. |

### 20.3 Release gate

**AC-R1:** A shadow-mode (CU-14) or canary deployment MUST run against production-shaped traffic for ≥ 72 h, showing: no increase in error rate, p99 TTFT no worse than v1, and `router_load_accounting_errors_total == 0`.

---

## Critical Files for Implementation

Reference material the implementation will be read against (all read-only inputs; v2 is a new tree):

- `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/cache_sim.go` — the engine to extract (`prefixTrie`, `cacheEstimator`, `estimateTokens`, `promptChunkBytes`); CU-1, CU-2, CU-5.
- `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/replay_router.go` — `BuildReplayRequestPrefix` at line 731, including the `i == 0 && sb.Bytes < 200` skip rule; CU-3.
- `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/replay_router_post.go` — the cold/warm TTFT cost model at lines 692–716 and the `Observe` hot-path call site at line 307; CU-6.
- `/Users/ofer.kiselovnahman/workspace/wllm-router/src/server.rs` — the authoritative HTTP surface (routes at lines 825–901); GW-1..GW-13, COMPAT-1.
- `/Users/ofer.kiselovnahman/workspace/wllm-router/src/policies/cache_aware.rs` — the imbalance formula at line 253 and thresholds at 314/342, plus the unkillable eviction thread at 96–122 and the comment at 467; CACHE-2, CACHE-3, CACHE-N3.
- `/Users/ofer.kiselovnahman/workspace/wllm-router/src/core/circuit_breaker.rs` — the unused `window_duration` at line 16/25 and the half-open handling; HLT-7, HLT-8.
- `/Users/ofer.kiselovnahman/workspace/wllm-router/src/service_discovery.rs` — current pod-IP-only discovery (`PodInfo::from_pod`, `worker_url`); SD-1, SD-9.