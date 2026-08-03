# Plan: Go rewrite of `wllm-router`

## Context

The current router (`/Users/ofer.kiselovnahman/workspace/wllm-router`) is ~36k LOC of Rust derived from the SGLang router. Three drivers:

1. **Over-complication** — PyO3 bindings, a Python CLI, `mini_lb.py`, a ~2,100 LOC tokenizer subsystem routing never calls, PD disaggregation (~4,600 LOC), and ~5,000 LOC of dead files not even declared in `src/lib.rs`.
2. **Structurally broken policies** — in-flight load is incremented on exactly one code path, decremented on three, and wiped wholesale by the health checker every 10 cycles. Every load-sensitive decision is made on noise, including v1's cache-aware imbalance guard.
3. **Ownership** — Go is the team's language, and the cache-prediction engine we reuse is already Go.

Artifacts drafted: a requirements spec (`scratchpad/requirements.md`) and technical design (`scratchpad/design.md`), plus the `HIER-*` and `API-*` areas below. **Both scratchpad files are ephemeral — M0 persists them to `docs/rewrite/`.**

> Naming correction: the prediction referred to as `wekai benchmark analyze` is actually `wekai router analyze`; the reusable engine is `benchmark/cache_sim.go` (281 LOC, stdlib-only, all unexported).

---

# Product requirement traceability (FR-RTR-01 … 05)

| FR | Status | Mechanism |
|---|---|---|
| **FR-RTR-01** Opportunistic HBM routing | ⚠️ **Partially satisfied — approximated by prediction.** Stated as a known gap, not papered over. | `CACHE-*` predictive trie + threshold spill (§C1, §C2) |
| **FR-RTR-02** Configurable routing policies (RR ⟷ KV-aware) | ✅ Covered | `--policy` flag, `LB-13..LB-17` |
| **FR-RTR-03** Load-balanced routing (GPU util + queue depth) | ⚠️ Partial — needs a per-node vLLM metrics scraper (§C3). *Nice to Have* → M7. | new `LOAD-*` |
| **FR-RTR-04** DRAM-aware multi-tier routing | ⚠️ **Blocked on the same feed as FR-RTR-01** (§C4). *Nice to Have* → deferred. | new `RES-*` (interface only) |
| **FR-RTR-05** Resilient storage fallback | ⚠️ **Mostly not the router.** Already implemented by LMCache. Router obligations specified (§C5). | new `WEKA-*` |

I verified the enabling vLLM/LMCache capabilities against `~/workspace/vllm` @ `702f4814f` and `~/workspace/wllm-lmcache` rather than assuming. Five consequences follow.

## C1 — FR-RTR-01: threshold-based spill, reusing v1's operator-facing model

**Decision:** adopt v1's imbalance-guard shape — cache residency wins outright until the fleet is measurably imbalanced, then traffic spills to less-loaded nodes. Two knobs, same semantics operators already know:

```
spill_to_least_loaded  ⟺  (max_load − min_load) > balance_abs_threshold
                      AND   max_load > min_load × balance_rel_threshold
```

This replaces the drafted design's continuous `queueing_penalty` term, which is **rejected**: its worked example abandons an 8,000-token cache hit for a single queued request (crossover at `inflight < 1.4`), which would systematically hide the cache value this product exists to demonstrate. A threshold guard keeps residency primary — FR-RTR-01's "even if heavily loaded" — while still bounding pathological pile-up. It is also *less* machinery than the drafted scoring.

**But v1's guard must not be ported — it was broken in four ways, and all four are fixed by construction here:**

| v1 defect | Fix |
|---|---|
| Fed by the corrupt load counter (incremented on one path, decremented on three, zeroed every 10 health cycles) — so the guard evaluated **noise** | M1's `Lease` primitive is the only writer; the guard is the *first consumer* of a load signal that is actually correct. This is the whole point of sequencing M1 first. |
| `max_load` computed over **all** workers including unhealthy ones, so one dead worker holding stale load latched the guard permanently ON and silently disabled cache routing forever (`CACHE-N4`) | Computed over **candidates only** — healthy, non-draining, circuit-closed (`LB-9`) |
| `min_by_key` tie-break always returned index 0, so on a cold fleet every request piled onto one node until it exceeded the threshold — a 32-deep thundering herd (`CACHE-N5`) | One shared reservoir tie-break, uniform over the tied set (`LB-11`), chi-square tested |
| Divergent defaults: `abs=32, rel=1.1` in the policy vs `abs=64, rel=1.5` on the CLI | One default set (`CFG-3`). **Proposed: `abs=32, rel=1.5`** — recorded in `/get_server_info` and revisited from replay data |

**Bound on FR-RTR-01:** `max_inflight_per_worker` (`REL-10`) still applies, so a resident node at its hard cap sheds rather than queueing without limit. That is the one limit on "even if heavily loaded", and it should be documented as such.

## C2 — FR-RTR-01 says "identify"; we can only predict, and neither residency feed is available

FR-RTR-01 requires identifying nodes where the KV slice **is** resident. I checked what vLLM actually exposes:

- **No query API.** I enumerated every route under `vllm/entrypoints/`. Nothing answers "do you hold hash X". `/reset_prefix_cache` mutates only; `/server_info` and `/collective_rpc` are dev-mode.
- **A push-based stream exists** — vLLM's ZMQ KV event publisher (`vllm/distributed/kv_events.py`) emits `BlockStored` / `BlockRemoved` / `AllBlocksCleared` with block hashes, `token_ids`, and a `medium` tier label. Requires `--kv-events-config` (default **off**).
- **LMCache offers a real query API** — cache-controller `POST /lookup` returns `{instance_id: (location, matched_token_count)}` from token ids.

**Neither is available in our deployment today.** So:

- **`kv-aware` ships purely predictive** — a rebuilt trie mirroring what vLLM *probably* holds. **FR-RTR-01 is therefore approximated, not satisfied.** That is recorded here as a known gap so nobody later mistakes prediction for identification.
- **Indexing is by text prefix**, not token ids — 1024-byte windows, per `wekai`'s `promptChunkBytes`. This keeps the drafted design's granularity decision and, importantly, means **no tokenizer in the router**. (Token-id indexing would have required either an in-router tokenizer or a `/tokenize` round trip per candidate; both were worse.)
- The accuracy instrument stops being optional and becomes **the only way to know whether the prediction is worth anything**:

| ID | Level | Requirement |
|---|---|---|
| **RES-1** | MUST | Residency is exposed to policies behind one interface — `Lookup(prefixUnits) → []{node, matchedTokens, tier, source}` — with a predictive-trie implementation. Event-stream and LMCache implementations slot in later **without touching the policy**. |
| **RES-2** | MUST | Every decision records its source: `router_residency_source_total{source=predicted\|events\|lookup}`. Today always `predicted`. |
| **RES-3** | MUST | Emit predicted-vs-observed cached fraction from `usage.prompt_tokens_details.cached_tokens`, labelled by source. Prediction quality is measured continuously, not assumed. |
| **RES-4** | MUST | Documentation and `/get_server_info` MUST state that FR-RTR-01 is served by prediction, naming the two feeds that would make it exact and what enabling them requires. |
| **RES-5** | — | Tier-aware residency (FR-RTR-04) is **interface-shaped now, unimplemented** — the `tier` field exists and is always `unknown`. |

**Deployment ask:** `usage.prompt_tokens_details.cached_tokens` needs `--enable-prompt-tokens-details` on the vLLM nodes (default **off**). Without it we have no feedback signal at all and `kv-aware` is unfalsifiable. This is a small config change with high value — worth raising with whoever owns the vLLM deployment.

## C3 — FR-RTR-03 needs GPU utilization; deferred to M7 as *Nice to Have*

FR-RTR-03 defines load as *"GPU compute utilization + request count queued."* The router's own lease count is neither, though it is more accurate per-request than any scrape.

Verified per-node metrics (note `gpu_cache_usage_perc` and `gpu_prefix_cache_hit_rate` **no longer exist** in this vLLM version):

- `vllm:num_requests_running`, `vllm:num_requests_waiting` — the queue depth FR-RTR-03 asks for
- `vllm:num_requests_waiting_by_reason{reason="capacity"|"deferred"}` — a useful discriminator: high `deferred` means stalled on KV transfer, not compute-saturated
- `vllm:kv_cache_usage_perc` — the closest available proxy for GPU occupancy
- cheaper: `GET /load` → `{"server_load": N}` (needs `--enable-server-load-tracking`)

**vLLM exposes no direct SM-utilization metric.** `kv_cache_usage_perc` + `num_requests_running` is the honest approximation and the limitation should be stated rather than implied away. Ships in M7 as an optional composite feeding the C1 threshold's `load` definition.

## C4 — FR-RTR-04 blocked on the same feed as FR-RTR-01

Tier attribution requires the KV event `medium` field (`"GPU"`, `"CPU"` for host-DRAM offload, plus `fs`/`p2p`/`obj` secondary tiers). Aggregate metrics collapse everything non-local into `external_kv_transfer`, so `vllm:prompt_tokens_by_source` cannot distinguish HBM from DRAM per node.

With no event feed, **FR-RTR-04 cannot be implemented** — a predictive trie has no way to know which tier holds a prefix. `RES-5` keeps the `tier` field in the interface so this is additive, not a redesign, the day events are enabled. Deferred; *Nice to Have*, so no schedule impact.

## C5 — FR-RTR-05 is mostly not the router's job, and already exists

FR-RTR-05 requires the **KV cache manager** to fall back to local prefill on WEKA failure and recover without restart. That is LMCache-side and **already implemented** — I should not claim the router does it:

- `wllm-lmcache/lmcache/v1/health_monitor/` has `FallbackPolicy.RECOMPUTE` (default) and `LOCAL_CPU`; `ping_timeout` 5 s, `ping_interval` 30 s, `waiting_time_for_recovery` 300 s.
- Recovery is automatic and restart-free: `HealthMonitor` tracks `_bypassed_backends` and calls `set_backend_bypass(name, false)` on recovery.
- Externally readable: `GET /bypass/list` → `{"bypassed_backends":[...],"all_backends":[...]}`, plus `lmcache:remote_ping_latency`.
- vLLM has **zero** WEKA code (`grep -ril weka` → no hits); the WEKA path is LMCache's GDS backend (`fstype == "wekafs"`, RDMA → cuFile/GDS → POSIX fallback).

| ID | Level | Requirement |
|---|---|---|
| **WEKA-1** | MUST | The router MUST NOT mark a node unhealthy because WEKA is degraded. A node in `RECOMPUTE` fallback is **slower, not broken**; failing it converts a storage slowdown into a capacity outage. |
| **WEKA-2** | MUST | On observing degraded mode (`GET /bypass/list` non-empty, or `AllBlocksCleared`), the router MUST invalidate that node's cache-model entries — predictions for a bypassed tier are actively wrong. |
| **WEKA-3** | MUST | Recovery is automatic with no router restart; the transition is visible in `router_residency_source_total`. |
| **WEKA-4** | SHOULD | Surface degraded state on `GET /workers` and as `router_backend_kv_degraded{worker}`, so an operator can see *why* hit rates dropped. |
| **WEKA-5** | — | Local-prefill fallback itself is **out of scope** — LMCache owns it. |

---

# What this adds beyond the original router

### Net removals (the bulk of the simplification)

| Removed | ~LOC |
|---|---|
| PD disaggregation | 4,600 |
| Tokenizer subsystem + its dead metric family | 2,100 |
| Dead files (`handler.rs`, `types.rs`, `logger.rs`, `routes/`, `utils/`) | 5,000 |
| `protocols/spec.rs` + `validation.rs` → a ~300 LOC partial JSON scanner | 4,700 |
| Python/PyO3 layer, `mini_lb.py` | — |
| `consistent_hash`, `rendezvous_hash`, `hash_key` | 1,260 |
| Remote API-key validation fan-out | — |

### Genuine additions

| Added | Why |
|---|---|
| **Correct load accounting** (`Lease`) | the defect motivating the rewrite; also what finally makes C1's threshold guard meaningful |
| **Rebuilt cache engine** | FR-RTR-01/02; token-budgeted eviction, bounded memory, no background thread |
| **Residency interface** (`RES-*`) | so event/LMCache feeds are additive later, not a redesign |
| EndpointSlice discovery | v1 was pod-IP only, which breaks behind a Service |
| Graceful drain, deterministic snapshots | fixes round-robin degenerating to random |
| WEKA degraded-mode observer | FR-RTR-05, router side |
| Dialect interface (one implementation) | future Anthropic without v1's 5,000-LOC schema coupling |
| Per-node metrics scraper | FR-RTR-03 — M7 |

### Confirmed cut

**Hierarchical routing is deferred** to post-v2.0 — ~500 LOC plus complexity threaded through registry, policy, health and the forwarding path, and nothing in FR-RTR-01..05 requires it. Requirements stay documented below; v2.0 keeps `Backend.kind` and `Backend.capacity` as no-op defaults so it is additive later rather than a refactor.

Everything else stays as designed: the five-policy set, Responses-API affinity, and the dialect interface with a single OpenAI implementation.

### Consequence for OQ-3 (do the two cache policies merge?)

The drafted design merged them into one engine with two scorer presets. **That still holds at the engine level** — one trie, one extractor, one eviction model. But both names now ship, because C1 makes the distinction product-meaningful rather than cosmetic:

| Policy | Spill behaviour |
|---|---|
| `prefix-cache-aware` | **Threshold** spill (C1's `abs`/`rel` guard). The FR-RTR-01/02 policy. |
| `cache-usefulness` | **Continuous** tier-weighted scoring. Experimental; the path FR-RTR-04 would light up. |

---

# Policy set (FR-RTR-02)

`--policy` — a single deployment flag, per FR-RTR-02:

| Name | Behaviour |
|---|---|
| `round-robin` | Virtual-time (least-recently-served) rotation, starvation-free across a changing candidate set — FR-RTR-02 (1) |
| `prefix-cache-aware` | Predictive prefix residency, threshold spill per C1 — FR-RTR-02 (2), FR-RTR-01 |
| `least-outstanding` | Correct in-flight count; default and universal fallback |
| `cache-usefulness` | Continuous tier-weighted scoring; experimental |
| `random` | Baseline for testing and small deployments |

---

# Review findings on the drafted design

Four issues found by checking the design's claims rather than accepting them:

**R1 — BLOCKING: the cache LRU invariant is inverted.** §D.4 argues the LRU tail is always a leaf because `Commit` touches ancestors before descendants — but that ordering makes the *deepest* node most-recently-used, so the tail trends toward nodes *with children*. `evictLocked` treats a non-leaf tail as an invariant violation and returns early, so **eviction silently stops and the trie grows unbounded**, signalled only by an unwatched metric. Fix: leaves-only LRU (unlink the parent on insert, relink when its last child is evicted) so the tail is a leaf *by construction*. Now on the critical path, since prediction is the only residency source.

**R2 — BLOCKING: half-open circuit tokens leak during candidate filtering.** Filtering calls `circuit.Allow()` on every candidate but `Record()` only on the selected one. With `HalfOpenMax = 1`, one filtering pass permanently exhausts a backend's probe budget — it can never close, so it never returns to rotation. This is v1's bug F1 (selection mutating circuit state) in new clothing. Fix: filter on `State()` only; call `Allow()` once after selection and re-select if denied.

**R3 — Cache commit happens before dispatch.** A request failing at connect has already written its prefix into that backend's trie, so it looks warm for a prefix it never received — permanently and self-reinforcingly. Fix: commit after response headers arrive.

**R4/R5 — minor.** The shared `pickBest` tie-break helper minimizes while `cache-usefulness` maximizes, so that policy can't use it — quietly dropping `LB-11` enforcement in one place. And `lease.Acquire/Release` call `WithLabelValues` per request (40k label resolutions/s at `NFR-1`); resolve the child gauge once at registration.

**Design self-flagged amendments:** `CU-8`'s 500k-node default breaks `NFR-5` at 64 backends (reduce to 100k — the token budget binds first at ~7,800 nodes anyway), and `HLT-N1`'s acceptance criterion forces the health pool to `min(256, max(32, N))`.

**What held up:** the `Lease` primitive (`sync.Once` + nil-safe + underflow-clamps-and-alarms; three call sites, one effect); the round-robin virtual-time scheduler; `httputil.ReverseProxy` validated with a narrow dissent and a pre-committed fallback for the risky retry glue; the allocation-free partial JSON scanner replacing 4,700 LOC of typed schema; and declining to dual-emit v1's three load gauges because they were fed by the corrupt counter.

---

# Milestones

| M | Milestone | FR |
|---|---|---|
| **M0** | Persist specs to `docs/rewrite/`. Skeleton, deterministic clock, metrics, structured logging, mock vLLM worker. | — |
| **M1** | **Load accounting first.** `Lease`, registry with COW snapshots + drain, circuit breaker with the R2 fix. Everything downstream depends on this signal being correct. | FR-RTR-03 basis |
| **M2** | **Runnable proxy.** Static backends, `round-robin` + `least-outstanding`, auth, streaming, retries, dialect interface + OpenAI. Deployable. Resolve the ReverseProxy retry-glue risk here, not later. | FR-RTR-02 (1) |
| **M3** | **`prefix-cache-aware`.** Partial JSON scanner, 1024-byte text-prefix units, trie with leaves-only LRU (R1) and commit-after-headers (R3), C1 threshold guard, `Lookup` interface (`RES-1`), accuracy instrumentation (`RES-3`). | **FR-RTR-01** (approximated), **FR-RTR-02 (2)** |
| **M4** | K8s EndpointSlice discovery with idempotent reconciliation. | — |
| **M5** | WEKA degraded-mode observer. | **FR-RTR-05** (router side) |
| **M6** | Hardening: soak, benchmarks, dead-metric + import-fence CI, container. | — |
| **M7** | *Nice to Have:* per-node vLLM metrics scraper feeding the C1 threshold's load definition. | FR-RTR-03 |
| **later** | Residency feed (KV events / LMCache `/lookup`) → exact FR-RTR-01 + FR-RTR-04 tiers; hierarchical routing; Anthropic dialect. | FR-RTR-04 |

## Verification

- **FR-RTR-01 / C1** — with a mock node holding prefix P and several requests in flight against an idle node with no match: assert the **resident** node wins below the imbalance threshold, and that traffic spills only once `(max−min) > abs && max > min×rel`. Assert the guard is computed over healthy candidates only (a dead worker holding stale load must not latch it on — v1's `CACHE-N4`), and that cold-fleet ties spread uniformly rather than piling on index 0 (`CACHE-N5`).
- **FR-RTR-02** — flipping `--policy` changes selection with no other config change; `round-robin` gives each of N stable workers exactly 10 of 10N requests, including across a shrink-then-grow of the candidate set.
- **FR-RTR-05** — kill WEKA reachability behind a mock LMCache: the node stays **healthy** (WEKA-1), its cache entries are invalidated (WEKA-2), and on restore the model repopulates with no router restart (WEKA-3).
- **Core correctness** — `TestLeasePropertyAllCountersReturnToZero` (10k randomized lifecycles including double- and concurrent-release, under `-race`); the trie invariant walk under concurrent inserters + evictor; SSE terminal marker split at every byte offset; chaos test killing a worker mid-stream.
- **Prediction quality** — offline replay against `wekai router analyze` traces, gated on predicted-vs-observed correlation ≥ 0.7. **This is the gate that decides whether `prefix-cache-aware` is worth shipping at all**, and it needs `--enable-prompt-tokens-details` on the nodes.
- **Rollout** — shadow mode (compute the choice, emit metrics, route by `least-outstanding`), then canary 1→10→50→100% with ≥72 h at the last two steps. Rollback is an ingress weight change; v2 shares no state with v1.

---

# Deferred: Hierarchical routing (`HIER-*`)

Documented, not implemented in v2.0. Model: a directed tree, same binary at every node, backends either leaf workers or child routers. A child router is *addressable* like a worker but must never be *scored* like one.

| ID | Level | Requirement |
|---|---|---|
| **HIER-1** | MUST | Explicit `kind: worker \| router`; never inferred by probing. |
| **HIER-2** | MUST | Loop prevention: `X-Wllm-Hops` + appended `X-Wllm-Via`; `508` on self-in-`Via` or hops > `max_hops` (default 4). |
| **HIER-3** | MUST | Deadline propagation via `X-Wllm-Deadline-Ms`, reduced per hop; a child may only shrink it. |
| **HIER-4** | MUST | Retry-amplification bound: `X-Wllm-Attempts-Remaining: k` → at most `k` attempts, forward `k−1`. Tree-wide calls bounded by `max_attempts`, not `max_attempts^depth`. |
| **HIER-5** | MUST | Capacity-weighted load: compare `inflight / capacity`. Leaf capacity is `max_inflight_per_worker`; a child's is its reported subtree capacity. |
| **HIER-6** | MUST | `GET /v1/internal/node_state` reports node id, role, depth, healthy leaves, subtree capacity/inflight, dialects, ready. Absence degrades to "leaf, capacity 1". |
| **HIER-7/8** | MUST | Readiness fails at zero healthy backends so parents drain children through the ordinary health path; `SIGTERM` flips readiness first. |
| **HIER-9** | MUST | Per-hop auth with a separate upstream key; the client's credential never reaches a child. |
| **HIER-10** | MUST | Request id byte-identical across hops; `hop_depth` on logs/metrics; `traceparent` propagated. |
| **HIER-11** | MUST | Streaming bounds hold transitively: buffered bytes ≤ `bounded_buffer × depth`. |
| **HIER-12/13/14** | MUST | Opaque cross-tier affinity only (a parent models a child as one aggregate cache); bid mode deferred; if built, only hashes and token counts cross the wire; **no cross-tier commit**. |
| **HIER-15/16** | SHOULD/MUST | `locality` label with `prefer_local` spill threshold; discovery works for child routers via `wllm.weka.io/backend-kind`. |
| **HIER-17** | MUST | Startup rejects a self-loop URL; warns if `max_hops × per_hop_reserve > global_deadline`. |
| **HIER-19** | MUST | Provenance `static \| discovered`; discovery only touches `discovered` entries; collisions ignored with `router_discovery_conflicts_total{url}` + `WARN`. |

**Must not regress into:** comparing a child's raw in-flight against a leaf's (**N1**, the likeliest bug — a child fronting 40 GPUs at 40 in-flight looks "more loaded" than an idle leaf at 1, starving a healthy subtree); retry amplification (**N2**); forwarding client credentials to children (**N3**); modeling a child as one cache with one eviction budget when it has k independently-evicting leaves, optimism factor in `[1, k]` (**N4**); re-minting request ids per hop (**N5**); O(nodes²) state polling (**N6**).

---

# API dialects (`API-*`) — interface in v2.0, Anthropic later

**Structural insight:** `wekai`'s prefix builder is already **Anthropic-native** — `BuildReplayRequestPrefix` orders units *system blocks → tools → messages* (the Messages shape), and the `i == 0 && Bytes < 200` skip exists because Anthropic's per-request billing header carries a near-unique hash that poisons prefix-block hashing. The OpenAI extractor is the adaptation, not the reverse — which inverts the natural reading of `CU-3`.

| ID | Level | Requirement |
|---|---|---|
| **API-1** | MUST | Core (`registry`, `lease`, `policy`, `residency`, `cachetrie`, `proxy`, `circuit`, `health`) contains no request-shape knowledge. Enforced by a `go list -deps` import fence in CI. |
| **API-2** | MUST | `Dialect` covers exactly seven concerns: routes claimed; prefix-unit extraction; stream terminal framing; error-envelope rendering; usage/cached-token extraction; credential header form; model-identifier location. |
| **API-3** | MUST | v2.0 ships **OpenAI only**; adding Anthropic requires no core change, proven by the fence plus a stub dialect driven through the gateway in tests. |
| **API-5/6/7** | MUST | Dialect comes from the matched route, never body sniffing; every backend declares its dialect; **passthrough only** — a dialect-D request only reaches dialect-D backends, else `503`. |
| **API-8** | MUST | Stream terminal detection is dialect-provided: OpenAI `data: [DONE]`, Anthropic `event: message_stop`; line-oriented and correct across arbitrary chunk boundaries. |
| **API-9/10** | MUST | Error envelopes render in the **inbound** dialect; cached-token extraction reads both OpenAI `prompt_tokens_details.cached_tokens` and Anthropic `cache_read_input_tokens`. |
| **API-11/12** | MUST | Anthropic shape is the reference model; the `Bytes < 200` skip is documented as an Anthropic billing-header artifact and **not** applied to OpenAI. Golden test against a `wekai` `req-v1` corpus. |
| **API-15** | MUST | Partial scan only — no typed model, no re-serialization, for every dialect. |
| **API-16** | MUST | Backend health model declared `active \| passive`; `passive` derives health from proxied outcomes and is never probed. |
| **API-17** | MUST | Policy scoring is a sum of named, individually-toggleable terms — what makes C1, C3 and C4 config changes rather than redesigns. |

**NG-8:** cross-dialect translation is out of scope — the one thing that would force abandoning `GW-6` (never deserialize or re-serialize a body).

**Must not regress into:** body-sniffing the dialect (**N1**, ambiguous and an injection vector); dialect knowledge leaking into core (**N2**, the coupling that grew v1's `protocols/` to 5,000 LOC); a hard-coded `data: [DONE]` scanner (**N3**); OpenAI envelopes to Anthropic clients (**N4**); Anthropic requests to OpenAI-only workers, producing a 4xx storm the breaker reads as client error and never trips (**N5**).
