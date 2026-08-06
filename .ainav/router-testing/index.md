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
  (default) is the load-only baseline for A/B.
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
