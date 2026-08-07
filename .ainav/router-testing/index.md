# Router Mock-Testing Flow

End-to-end loop for developing/evaluating `wllm-router` routing policies (prefix-cache
affinity, load balancing, `--max-node-concurrency`) against a GPU-less mock vLLM fleet,
driven by a real captured-traffic replay — no hardware, no real model.

```
mock-vllm fleet (N instances)  <--  wllm-router (prefix-cache-aware)  <--  wekai benchmark auto (replay-v3)
```

See [calibration.md](calibration.md) for tuning the mock's latency/tokenizer rates to
match a real fleet, and [replay-notes.md](replay-notes.md) for replay-v3 file mechanics
and recent correctness fixes.

## Validated Standard Recipe (routing-improvement work)

The recommended, validated flow for evaluating a routing-policy change — everything
below is one worked, known-good example of steps 1-5; read those for the explanation
of each flag. 4 mock instances, `--max-node-concurrency 32` at the ROUTER (not on the
mock instances — they keep their own `--max-concurrency 256`, matching a real backend's
own admission limit; the per-node cap under test is a ROUTER-side concurrency lease,
see step 3), client concurrency sized to the fleet's total node capacity
(4 nodes × 32 = 128), ×10 calibrated rates (see
[calibration.md](calibration.md#speedup-for-fast-iteration)), and a LONG run so the
cache numbers are meaningful (see
[calibration.md](calibration.md#depth-matters-run-length-changes-what-a-run-is-valid-evidence-for)).
Total runtime ~6 minutes.

```bash
# 1. Build (once)
go build -o /tmp/mock-vllm ./router/cmd/mock-vllm
go build -o /tmp/wllm-router ./router/cmd/wllm-router
go build -o /tmp/wekai-local .

# 2. Fleet: 4 instances, ×10 aggregate rates, admission cap stays at the backend default
/tmp/mock-vllm --instances 4 --base-port 9001 --model-id <model-id> \
  --block-size-tokens 256 --block-capacity 7952 --max-concurrency 256 \
  --chars-per-token 3.729 --output-kv-multiplier 1.0 \
  --cold-input-tps 640000 --cached-input-tps 1330000 --output-tps 740 --base-latency 10ms

# 3. Router: the per-node cap under test lives HERE, not on the mock instances
/tmp/wllm-router \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003,http://127.0.0.1:9004 \
  --policy prefix-cache-aware --max-node-concurrency 32

# 4. Readiness-gate (always — see below)
until curl -sf http://127.0.0.1:8080/v1/models > /dev/null; do sleep 0.5; done

# 5. Benchmark: concurrency = nodes x node-cap = 4x32 = 128; hot pool OFF for
#    precise measurements; a LONG total for cache-realistic numbers
/tmp/wekai-local benchmark auto \
  --router-replay-file <replay.jsonl> \
  --models "dynamic/http://127.0.0.1:8080/v1,type=openai_vllm,alias=<run-name>" \
  --concurrency 128 --series 256 --hot-series-concurrency 0 \
  --limit-context 140000 --print-errors-threshold=1s --total 30000 \
  [--save-request-data <dir>]
```

- **Viz is PER ROUTER INSTANCE.** `/router-viz` lives on the metrics listener above
  (`127.0.0.1:29000` here). Running a second router for A/B on the same machine: give
  it its own `--metrics-listen` (conventionally `127.0.0.1:29001`) as well as its own
  `--listen`, so both KV maps stay independently reachable.
- **The drain tail is normal, not a hang.** Real captures include requests with
  `max_tokens` up to ~32k; at ×10 (output 740 tok/s) one of those alone takes ~43s to
  decode. Near the end of a run, once the replay queue has emptied, the last few
  giant-decode requests dominate: `active` in the ledger sense drops and `in_flight`
  can sit at a small number (heading to 0) for tens of seconds while those finish. Let
  it drain — that is the run completing correctly, not stalling.

## 1. Build

```bash
cd $WEKAI_DIR   # or submodules/wekai
go build -o /tmp/mock-vllm ./router/cmd/mock-vllm
go build -o /tmp/wllm-router ./router/cmd/wllm-router
go build -o /tmp/wekai-local .
```

## 2. Launch the mock fleet

One process, N independent instances (own cache/counters each — never share state),
on consecutive ports:

```bash
/tmp/mock-vllm --instances 4 --base-port 9001 --model-id <model-id> \
  --block-size-tokens 256 --block-capacity 7952 --max-concurrency 256 \
  --chars-per-token 3.729 --output-kv-multiplier 1.0 \
  --cold-input-tps <rate> --cached-input-tps <rate> --output-tps <rate> \
  --base-latency <dur>
```

- `--block-capacity` should mirror the real fleet's `GPU KV cache size` (tokens) ÷
  `--block-size-tokens`, so the mock's eviction pressure matches reality.
- `--chars-per-token` is the mock's OWN byte→token estimator (independent of
  `kvcache`'s fixed 4.0 used elsewhere) — calibrate it against the real tokenizer; see
  [calibration.md](calibration.md).
- `--output-kv-multiplier 1.0` models real vLLM writing decode KV into the same pool as
  prompt KV (appended to the chain on completion — see
  `router/internal/mockvllm/engine.go`'s `AppendOutputBlocks`).
- Rate flags (`--cold-input-tps`/`--cached-input-tps`/`--output-tps`/`--base-latency`)
  are what let a mock run be calibrated against a real fleet's measured throughput
  instead of an arbitrary latency shape — see calibration.md for fitted values and how
  to speed a run up uniformly for fast iteration.

## 3. Launch the router

```bash
/tmp/wllm-router \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003,http://127.0.0.1:9004 \
  --policy prefix-cache-aware \
  --max-node-concurrency <N>          # optional: per-backend router-enforced cap; 0 = off
```

- Metrics (`/metrics`), the live KV map (`/router-viz`, `/router-viz/data`), backend
  listing (`/workers`), and readiness (`/readiness`) all live on `--metrics-listen` /
  `--listen` respectively — see [../viz/index.md](../viz/index.md) for the KV map.
- `--policy prefix-cache-aware` is the routing policy under test; `least-outstanding`
  (default) is the load-only baseline for A/B. `prefix-cache-split` is the newer
  shared-marked-tree policy — see [A/B'ing prefix-cache-split](#abing-prefix-cache-split)
  below, and note it REQUIRES `--max-node-concurrency`.
- `--max-node-concurrency N`: a backend at or above N router-leased in-flight requests
  is excluded from candidate selection for EVERY policy (affinity and fallback alike —
  filtered once at the candidate-set level, not per-policy). If every healthy backend
  is at cap, the router returns `429 all_backends_at_capacity` with `Retry-After: 1`
  (distinct from `503 no_healthy_backends`, an outage, and from the router-wide
  `--max-concurrent-requests` `503 router_at_capacity` shed). Use this to test a lower
  ceiling than the real fleet's own `WEKA_MAX_CONCURRENT_REQUESTS` without restarting
  vLLM. `/readiness` deliberately ignores the cap (a saturated router still answers
  ready=true; it just sheds 429s) — only backend health affects readiness.

## 4. Readiness-gate BEFORE benchmarking

**Always** poll the router's own `/v1/models` (or `/readiness`) for a 200 before
starting the benchmark:

```bash
until curl -sf http://127.0.0.1:8080/v1/models > /dev/null; do sleep 0.5; done
```

Starting the benchmark while backends are still `Unknown` health (first health check
hasn't landed yet) causes the benchmark's one-time-per-endpoint model-discovery GET to
503, and — before `960b6a0`/`cdeac09` fixed this — cascaded into thousands of phantom
errors and instant run termination. The router side is fine now, but there is no
reason to race it: always gate.

## 5. Run the benchmark

```bash
/tmp/wekai-local benchmark auto \
  --router-replay-file <replay.jsonl> \
  --models "dynamic/http://127.0.0.1:8080/v1,type=openai_vllm,alias=<run-name>" \
  --concurrency 90 --series 256 --hot-series-concurrency 6 \
  --limit-context 140000 \
  --print-errors-threshold=1s \
  --total 4800 \
  [--save-request-data <dir>]
```

This particular sizing (`--total 4800`, `--concurrency 90`, hot pool on) is illustrative,
not the validated one — it is a SHORT run: fine for a fast correctness/logic check, but
its cache-hit numbers read optimistic (see
[calibration.md](calibration.md#depth-matters-run-length-changes-what-a-run-is-valid-evidence-for)).
For anything whose cache-hit numbers matter, use the
[Validated Standard Recipe](#validated-standard-recipe-routing-improvement-work) above.

- Point `--models` at the ROUTER's listen address (`:8080` above), not at any one
  backend — that's the whole point of the exercise.
- `type=openai_vllm` keeps the run's Prometheus cache-source sampler polling through
  restarts/slow loads (see README's Replay benchmark section for what that sampler
  does) — always set it when the target is really vLLM (or this mock, which speaks the
  same wire format).
- `--limit-context` skips any request whose CAPTURE-recorded prompt tokens exceed the
  limit, to avoid 400 storms replaying long-session captures against a small-context
  model/mock config — see [replay-notes.md](replay-notes.md) for exactly what counts as
  a skip vs a retirement.
- `--save-request-data <dir>` writes per-request JSONL + an interactive `report.html`;
  see README's Replay benchmark section for the full shape.

## Operational gotchas

- **Readiness-gate, every time** (step 4) — the single most common cause of an
  instantly-dead run.
- **One benchmark arm at a time per machine.** Concurrent arms compete for the same
  CPU/network and starve each other's router health probes, producing misleading
  503s/skips that look like a routing bug but are a test-rig artifact.
- **Restart the fleet AND the router between runs.** The mock's cache is in-process
  state (never persisted); a stale warm cache from a prior run silently changes the
  next run's cache-hit numbers. Kill both, relaunch, re-gate readiness, then start the
  next arm.

## A/B'ing `prefix-cache-split`

`prefix-cache-split` (`router/internal/policy/affinity/`) is the shared-marked-tree
policy: one tree whose runs record WHICH backends hold them, no threshold, a split
that grows the holder set under saturation, and tail-only TTL eviction. See
[../../router/docs/cache-affinity-redesign.md](../../router/docs/cache-affinity-redesign.md)
§9 for the design and the numbers.

Operationally it differs from the other policies in three ways worth knowing before
you run it:

- **`--max-node-concurrency` is MANDATORY.** Startup fails naming the flag if it is
  unset. It is both the gateway's admission cap and the limit the split guard is
  measured against, so it has to mean one thing; set it to the backends' real vLLM
  `--max-num-seqs`. Every other capacity source in the shipped config reads 1.
- **Two extra knobs:** `--cache-split-guard` (default `0.20`) and `--cache-tail-ttl`
  (default `5m`).
- **It never returns 429 itself.** Admission stays with the gateway, so a
  `429 all_backends_at_capacity` means every healthy backend was at the cap — which
  is exactly the acceptance criterion, and is why `router_saturation_rejects_total`
  is the metric to watch.

```bash
# Arm B: same fleet and replay as the Validated Standard Recipe, new policy.
/tmp/wllm-router \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003,http://127.0.0.1:9004 \
  --policy prefix-cache-split --max-node-concurrency 32
```

Read these after each arm (`curl -s http://127.0.0.1:29000/metrics | grep ...`):

| Metric | What it tells you |
|---|---|
| `router_route_decisions_total` | **Aggregate by label**, never enumerate members — the tiers are `cache`, `split`, `overflow`, `load`, `other`. Anton's bar is "absolute majority routed by cache". |
| `router_cache_avg_copies` | Mean backends holding each block, target ~1.0. A high value at LOW utilisation is the one to investigate; under oversubscription it rises legitimately because bouncing sessions really are on several nodes. |
| `router_cache_splits_total` | The holder set grew under saturation. Should track load peaks, not be constant. |
| `router_cache_overflows_total` | Idle capacity used without marking. These are the requests the reference design would have 429'd while backends sat under their limit. |
| `router_saturation_rejects_total` | The only rejections. Compare against `router_backend_inflight`: a reject while any backend is under the cap would be a bug. |
| `router_cache_tree_runs` / `_tail_set` / `_blocks_expired_total` | Whether the TTL is right for the workload's session lifetime. |

**A short smoke run will not discriminate the policies.** With a large shared system
prompt, `prefix-cache-candidates`'s 0.5 threshold is satisfied anyway and both arms
score the same. The regime where they diverge is a MODEST shared prefix inside large
requests (0 vs 192 cache decisions out of ~200 in one measured run), and saturation
with skewed load, where tiers 2/3 actually fire. Use the long `--total 30000` recipe.

Offline, `go test ./router/internal/policy/affinity/ -run TestFleet -v` replays the
same workload against the real policy in-process in about a second, and prints the
verdict, the A/B against `least-outstanding`, and the cross-check against the
reference simulator's own 3-tier ladder. Start there before booting a fleet.
