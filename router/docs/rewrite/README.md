# `wllm-router` v2 — Go rewrite

Design documentation for replacing the current Rust router with a Go implementation.

## Why

1. **Over-complication.** ~36k LOC of Rust carrying PyO3 bindings, a Python CLI, `mini_lb.py`, a ~2,100 LOC tokenizer subsystem that routing never calls, PD disaggregation (~4,600 LOC), and ~5,000 LOC of dead files not even declared in `src/lib.rs`.
2. **Structurally broken load balancing.** In-flight load is incremented on exactly one code path, decremented on three, and wiped wholesale by the health checker every 10 cycles. Every load-sensitive decision — `power_of_two`, the cache-aware imbalance guard, `get_loads`, and the `worker_load`/`max_load`/`min_load` gauges — is made on noise. This is the central motivating defect.
3. **Ownership.** Go is the team's language, and the cache-prediction engine being reused (`wekai`) is already Go.

## Read in this order

| Document | Contents |
|---|---|
| **[plan.md](plan.md)** | **Start here, and treat it as authoritative.** Product-requirement traceability (FR-RTR-01..05), scope decisions, the `RES-*` / `WEKA-*` / `LOAD-*` areas, review findings, and milestones. |
| [requirements.md](requirements.md) | Numbered, testable requirements by area with stable IDs (`GW-*`, `AUTH-*`, `LB-*`, `CACHE-*`, `CU-*`, `SD-*`, `STR-*`, `CFG-*`, `OBS-*`, `REL-*`, `SEC-*`, `NFR-*`), each with a "must not regress into" list citing the specific v1 defect it guards against. |
| [design.md](design.md) | Technical design: package layout, the `Lease` primitive, policy implementations, the cache engine, middleware ordering, testing strategy, rollout. |

Both `requirements.md` and `design.md` open with an **amendments banner** listing corrections applied after review. Where a banner conflicts with body text, the banner wins.

## Scope at a glance

**In:** OpenAI-compatible gateway (wire-compatible with v1), correct load accounting, five policies, rebuilt prefix-cache-aware routing, K8s EndpointSlice discovery, static API-key auth, WEKA degraded-mode awareness.

**Out:** PD disaggregation, the Python/PyO3 layer, tokenization, cross-dialect translation. **Deferred:** hierarchical router trees, the Anthropic dialect, per-tier (DRAM-aware) routing, GPU-utilization-based load.

## The one thing to get right first

Milestone M1 builds the `Lease` primitive — a single, symmetric, `sync.Once`-guarded lifecycle that is the *only* writer of in-flight counters anywhere in the program. Everything downstream depends on that signal being correct, including the FR-RTR-01 spill threshold, which in v1 was evaluating a counter that was structurally garbage.

## Honest gaps

- **FR-RTR-01 is approximated, not satisfied.** vLLM exposes no residency query API. A push-based feed exists (ZMQ `BlockStored`/`BlockRemoved` with a `medium` tier label, via `--kv-events-config`) and LMCache offers `POST /lookup`, but neither is enabled in our deployment. v2.0 predicts residency; `RES-1` shapes the interface so a real feed is additive rather than a redesign.
- **FR-RTR-04 cannot be implemented** without that feed — a predictive model has no way to know which memory tier holds a prefix.
- **FR-RTR-05's fallback belongs to LMCache**, not the router. See `WEKA-5`.
