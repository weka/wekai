# Router Mock-Testing Flow

End-to-end loop for developing and evaluating the router (`wekai router serve`)
against a GPU-less mock vLLM fleet, driven by a real captured-traffic replay — no
hardware, no real model.

```
mock-vllm fleet (N instances)  <--  wekai router serve  <--  wekai benchmark auto (replay-v3)
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
# 1. Build (once). There is no router binary: the router is `wekai router serve`.
go build -o /tmp/mock-vllm ./router/cmd/mock-vllm
go build -o /tmp/wekai-local .

# 2. Fleet: 4 instances, ×10 aggregate rates, admission cap stays at the backend default
/tmp/mock-vllm --instances 4 --base-port 9001 --model-id <model-id> \
  --block-size-tokens 256 --block-capacity 7952 --max-concurrency 256 \
  --chars-per-token 3.729 --output-kv-multiplier 1.0 \
  --cold-input-tps 640000 --cached-input-tps 1330000 --output-tps 740 --base-latency 10ms

# 3. Router: the per-node cap under test lives HERE, not on the mock instances.
#    Endpoints are pipe-separated, the same syntax the client uses for a
#    multi-endpoint model.
/tmp/wekai-local router serve \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends 'http://127.0.0.1:9001|http://127.0.0.1:9002|http://127.0.0.1:9003|http://127.0.0.1:9004' \
  --max-node-concurrency 32

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
/tmp/wekai-local router serve \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends 'http://127.0.0.1:9001|http://127.0.0.1:9002' \
  --max-node-concurrency <N>   # optional: enables the concurrency split signal
  --rebalance-ratio <R>        # optional: enables the imbalance split signal
```

- **Routes, not just `--backends`.** `--backends` is shorthand for
  `--route '* => a|b|c'`. Rules are first-match-wins, so a mixed fleet is
  expressible: `--route 'llama => http://a:8000|http://b:8000'` with
  `--default https://api.anthropic.com` sends specific models to a local fleet
  and everything else to a hosted API. A pool of one endpoint is a plain proxy;
  several get prefix affinity. Same code path either way.
- **Endpoint kind is discovered, once.** An endpoint serving `vllm:` metrics at
  `/metrics` is treated as a vLLM instance: probed actively and eligible for
  upstream metric aggregation. Anything else falls back to passive health —
  still served, health inferred from traffic. The probe never repeats; use
  `--passive-health` to skip it for an upstream already known not to be vLLM.
  Discovery keys on metric names, NOT wire format, so a vLLM fronted with an
  Anthropic API is still recognised.
- **`--vllm-metrics`** aggregates upstream counters into router-level totals on
  the metrics listener. Only discovered vLLM endpoints are scraped, and totals
  accumulate deltas so they never rewind when a pod restarts.

- Metrics (`/metrics`), the live KV map (`/router-viz`, `/router-viz/data`), backend
  listing (`/workers`), and readiness (`/readiness`) all live on `--metrics-listen` /
  `--listen` respectively — see [../viz/index.md](../viz/index.md) for the KV map.
- **There is no `--policy` flag.** The router has ONE routing flow; what varies is
  which split signals are enabled, and each is turned on by setting its own value.
  `refused` (the backend's own 429) is always on and needs no configuration, so a bare
  `--backends` router is a valid deployment — see
  [Signals](#signals-what-replaced-the-policies) below.
- `--max-node-concurrency N` enables the **concurrency signal**: a backend at or above
  N router-leased in-flight requests is treated as saturated without waiting for it to
  say so. Use it to test a lower ceiling than the real fleet's own
  `WEKA_MAX_CONCURRENT_REQUESTS` without restarting vLLM. It is no longer an admission
  filter in the gateway and no longer mandatory.
- **Reading a 429 in a run:** `all_backends_saturated` means nothing could take work;
  `split_guard_blocked` means capacity existed and the guard refused to spend it on a
  duplicate copy of the prefix. Neither is an error — the second is the policy working
  as designed. The 503s mean something else entirely (`no_healthy_backends` is an
  outage, `router_at_capacity` is the router's own shed).
- `/readiness` reflects backend HEALTH only — a saturated router still answers
  ready=true and sheds with 429.

### Which signals to enable for a test run

`--help` is authoritative on what each flag does; `policy/affinity/signal.go` is
authoritative on how. What matters when SIZING A RUN:

- The `refused` signal is always on and needs nothing. **A bare `--backends` router is
  a valid arm**, and it is the only configuration where a routing mistake is not
  masked by the router's own guess — worth running whenever a change touches the
  refusal path.
- `--max-node-concurrency` is what makes a run comparable to the historical arms
  below, and what lets you test a lower ceiling than the fleet's real one.
- `--rebalance-ratio` trades locality for evenness. Leave it off unless that trade is
  what you are measuring; a fleet where affinity is working is supposed to look
  imbalanced.

Watch `router_signal_fired_total{signal=...}` to see which one is actually driving
decisions. An opt-in signal firing far more often than `refused` is predicting
saturation the backends do not have.

**Measured (4x32, `--total 30000`, client concurrency 128, 58-day capture):** the
`refused` signal ALONE, with no router-side limit and the mock fleet 429ing at its own
`--max-concurrency 32`, landed within noise of the concurrency signal — avg copies
1.078 vs 1.085, 5m42s vs 6m6s, 30000/30000 and zero errors either way.

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
- **The guard is an absolute rule, and the policy DOES reject.** A split onto a
  backend whose in-flight is at or above `max-node-concurrency * (1 - guard)`
  (25.6 at 32/0.20) is refused, and if nothing clears the guard the request gets
  `429 split_guard_blocked` — even though idle capacity exists. There is no
  serve-anyway path: a guarded split is the only way a backend is ever recorded as
  holding a prefix. Watch `router_cache_guard_rejects_total`; it is distinct from
  `router_saturation_rejects_total`, which means zero idle slots fleet-wide.
- **The replay client waits 429s out** (10ms doubling to a 3s cap, 30s total
  budget, jittered) rather than recording them as errors, so a run measures the
  fleet and not the harness. `429 backoff` in the run summary is the cost.

The A/B is now between SIGNALS, not policies — same fleet and replay, different
capacity source:

```bash
# Arm A: predict saturation. Mock keeps its own --max-concurrency 256.
/tmp/wekai-local router serve \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends 'http://127.0.0.1:9001|http://127.0.0.1:9002|http://127.0.0.1:9003|http://127.0.0.1:9004' \
  --max-node-concurrency 32

# Arm B: discover it. No router-side limit at all; the mock fleet 429s at its own
# cap instead (--max-concurrency 32), so the `refused` signal drives everything.
/tmp/wekai-local router serve \
  --listen :8080 --metrics-listen 127.0.0.1:29000 \
  --backends 'http://127.0.0.1:9001|http://127.0.0.1:9002|http://127.0.0.1:9003|http://127.0.0.1:9004'
```

Arm B is the one worth running when a change touches the refusal path: it is the only
configuration where a routing mistake is not masked by the router's own guess. Note
the two "32"s are NOT the same instant — the router's lease is taken before the
upstream request is issued and released only after the response body is fully copied
(LB-4), while the backend counts from arrival to end of generation, so the router's
window strictly contains the backend's and a virtual 32 bites earlier than a real one.

Read these after each arm (`curl -s http://127.0.0.1:29000/metrics | grep ...`):

| Metric | What it tells you |
|---|---|
| `router_route_decisions_total` | **Aggregate by label**, never enumerate members — the tiers are `cache`, `split`, `load`, `other` (`overflow` is retired and reads 0). Anton's bar is "absolute majority routed by cache". |
| all `router_cache_*` / `router_signal_fired_total` | Carry a `pool` label. One router may front several pools whose trees are unrelated; summing across them describes nothing. Aggregate by pool. |
| `router_cache_avg_copies` | Mean backends holding each block, target ~1.0. **Cheap to read** — a running `blocks x holders` sum divided by blocks, not a tree walk (the O(tree) call on the same ticker is the per-backend `router_cache_entries`/`_tokens` pair). Validated against per-backend ground truth in the fleet sim, so it can be trusted rather than corroborated. |
| `router_cache_splits_total` | The holder set grew under saturation, guarded. The ONLY way a holder is ever added. |
| `router_cache_guard_rejects_total` | 429s caused by the guard: idle capacity existed but every backend was inside the guard band. This is the price of holding avg_copies near 1.0 — read the two together. |
| `router_cache_shallow_anchors_total` | Requests whose own holders were unavailable so only a shared ancestor matched. These used to be served and marked, unguarded, and were the entire source of duplication (~50-60% of cache decisions at 100% utilisation, ~0% below it). They now fall through to the guard. |
| `router_saturation_rejects_total` | Rejections with zero idle slots fleet-wide. Distinct from the guard rejects above. |
| `router_cache_tree_runs` / `_tail_set` / `_blocks_expired_total` | Whether the TTL is right for the workload's session lifetime. |

**Sizing the arm decides what `avg_copies` measures.** The recipe above sets client
concurrency to the fleet's total capacity (128 = 4 x 32), which pins utilisation at
100% by construction. Under the previous serve-anyway ladder that alone drove
`avg_copies` to 1.68 on this replay — the duplication was a property of the offered
load, not of the policy, and the same code read 1.05 at concurrency 96. The strict
ladder holds ~1.01-1.09 at both. If you want to measure the POLICY, run below
saturation; if you want to measure the GUARD, pin it at 100%.

**A short smoke run will not discriminate anything.** The regimes where behaviour
diverges are a MODEST shared prefix inside large requests, and saturation with skewed
load, where the split and reject tiers actually fire. Use the long `--total 30000`
recipe.

Offline, `go test ./router/internal/policy/affinity/ -run TestFleet -v` replays the
same workload against the real flow in-process in about a second, and prints the
verdict, the A/B against `least-outstanding`, and the cross-check against the
reference simulator's own 3-tier ladder. Start there before booting a fleet.
`-run TestCopiesVsUtilisation -v` sweeps duplication against offered load, and
`-run TestShallowAnchorModes` A/Bs the ladder against the retired serve-anyway one.

**Note on the `least-outstanding` A/B arm.** It has no signals of its own and the
gateway no longer caps anything, so that arm now runs unbounded — it reached 206%
utilisation in the fleet sim. The hit-rate comparison stands; throughput between the
two arms does not.

## Mock surfaces: testing every deployment shape offline

`mock-vllm --surface` builds the three shapes the router must tell apart, so no
permutation needs a real backend:

| surface | serves | what it tests |
|---|---|---|
| `vllm` (default) | OpenAI routes, `/health`, `vllm:` metrics | an ordinary fleet member |
| `anthropic` | `/v1/messages` AND `vllm:` metrics | a vLLM fronted with an Anthropic API — must NOT be misclassified as a hosted API and have its metrics left unread |
| `hosted` | messages only; no `/metrics`, `/v1/models` or `/health` | discovery must fall back to passive health |

The `anthropic` surface runs the same engine and flattens to the same prompt the
OpenAI path builds, so a conversation produces identical block hashes through
either surface — otherwise cache behaviour would differ by wire format and the
mock would misrepresent the one thing it exists to model.

**Validated end to end** (2 instances, 6 requests sharing a prefix, 20
max_tokens each): affinity kept all six on ONE instance, leaving the other at
zero. That instance reported `generation_tokens_total` 120 (6 x 20),
`prefix_cache_queries_total` 96 (6 x 16 blocks) and `prefix_cache_hits_total` 80
(5 of 6 requests hitting the shared prefix; the first is cold). The router's
aggregated totals matched the sum exactly.
