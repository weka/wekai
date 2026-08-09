# Mock Fleet Calibration

How to set the mock vLLM's latency-model rate flags
(`--cold-input-tps`/`--cached-input-tps`/`--output-tps`/`--base-latency`) and tokenizer
ratio (`--chars-per-token`) so a mock run reproduces a real fleet's cache-hit behavior,
plus how to compress wall-clock time for fast iteration without distorting the result.

## Calibration facts (2026-08-06, DeepSeek-V2-Lite golden replay, 4×TP2 real fleet)

> **Superseded once already** — the first fitted numbers here (444ms / 16,174 /
> 33,297 / 74 tok/s) assumed independent PER-REQUEST prefill rates. `8764e9a` changed
> `--cold-input-tps`/`--cached-input-tps` to model real vLLM's prefill throughput
> correctly: a per-INSTANCE resource shared across concurrent requests via processor
> sharing (N concurrent prefills each drain at 1/N of the instance rate — see
> [../architecture/index.md](../architecture/index.md)). The table below is the
> re-fit against that corrected model; the old per-request numbers no longer apply to
> any current mock build.

Real fleet's effective service rates, fitted from per-request data:

| Rate | Value | Scope |
|------|-------|-------|
| Base latency | 100 ms | per request |
| Cold (uncached) input | 64,000 tok/s | per INSTANCE, aggregate (processor-shared) |
| Cached input | 133,000 tok/s | per INSTANCE, aggregate (processor-shared) |
| Output (decode) | 74 tok/s | per request (decode is NOT contended — see architecture doc) |

- **Cached input is NOT free** — cached:cold ≈ 2:1, not ∞:1. A cache hit still costs a
  KV read; only recompute is skipped. Setting `--cached-input-tps` far higher than
  `--cold-input-tps` but still finite is load-bearing for matching real behavior.
- These rates reproduce the real fleet's wall time at ×1 (i.e. set them literally, no
  speedup, and a replay takes as long against the mock as it would against real
  hardware).
- **Cache-hit accuracy depends on run LENGTH, not just rate** — see "Depth matters"
  below; do not judge calibration off a short run.

## Speedup for fast iteration

Speedup = uniformly multiply the two aggregate prefill rates AND the output rate by K,
divide the base latency by K. This preserves the RELATIVE cost ratios (cold vs cached
vs output) that drive routing/caching decisions while shrinking wall-clock run time.

- **×10**: cold 640,000 tok/s, cached 1,330,000 tok/s, output 740 tok/s, base 10ms —
  the standard speedup for routing-improvement work (see
  [index.md](index.md#validated-standard-recipe-routing-improvement-work)'s recipe).
  Needs a LONG run (`--total 30000`, ~6 min) to be cache-realistic — see below.
- **×100**: the validated fast loop, ~1 minute. Detects a routing regression reliably
  but UNDERSTATES its size — see the matrix below. Use it for go/no-go, never for a
  number you intend to cite.
- **×1000**: pointless. Elapsed barely moves (50s vs 60s at ×100) because the
  bottleneck is the harness, not the backends. Nothing above ×100 buys anything.

**Speedup does not scale `--cache-tail-ttl`, and it should.** The TTL is wall-clock
and purely a memory bound, but leaving it at 5m while multiplying rates by K means the
SIMULATED retention window grows by K — at ×100 a one-minute run evicts nothing, so
the tree only accumulates. Divide it like the base latency: `--cache-tail-ttl 3s` at
×100. This changes no routing decision (eviction only removes childless runs whose
sessions are long dead) but it does move `router_cache_avg_copies`, which is a mean
over whatever is currently in the tree: dead sessions' runs are overwhelmingly
single-holder, so retaining them dilutes the mean toward 1.0.

### Validated: what a speedup can and cannot tell you

Two configurations, three speedups, `--total 30000`, 4 instances, client concurrency
128, mock `--max-concurrency 32` in every arm so only the router config varies.

- **good** — no router-side limit at all; the `refused` signal (the backend's own 429)
  and the default `--cache-split-guard 0.20`.
- **bad** — `--rebalance-ratio 0.05 --cache-split-guard 0.01`: an imbalance signal that
  fires constantly, and a guard too low to stop the splits it provokes.

| speedup | config | elapsed | `avg_copies` | splits | guard 429s | server cache |
|---|---|---|---|---|---|---|
| ×10 | good | 5m51s | 1.069 | 415 | 18,283 | 61.6% |
| ×10 | **bad** | 5m57s | **1.679** | **10,042** | **0** | **38.4%** |
| ×100 | good | 1m0s | 1.021 | 42 | 244 | — |
| ×100 | **bad** | 1m6s | **1.254** | **2,991** | 53 | — |
| ×1000 | good | 53s | 1.029 | 18 | 14 | — |
| ×1000 | **bad** | 50s | **1.269** | **3,652** | 36 | — |

All six completed 30000/30000 with zero client errors.

**Direction survives the speedup; magnitude does not.** The good-to-bad gap on
`avg_copies` is +0.610 at ×10 but only +0.233 at ×100 and +0.240 at ×1000 — a fast run
understates the damage by about 2.6x. It would have caught this regression instantly;
it would also have told you a broken guard costs 0.23 when it really costs 0.61.

**The sharpest discriminator is not `avg_copies` — it is guard 429s inverting.** At
×10 the healthy config rejects 18,283 requests and the broken one rejects **zero**. A
guard that never refuses anything is a guard that is not doing its job, and that reads
as a clean binary. Below ×10 the same inversion is present (244 vs 53) but the
absolute counts are too small to trust. Read `router_cache_guard_rejects_total` and
`router_cache_avg_copies` together, always: a misconfiguration lands in one or the
other depending on where the guard sits, and either one alone misses half the failure
space.

**Why magnitude shrinks: the fleet is not saturated above ×10.** The speedup
compresses backend service time but not the harness. The router's lease spans dispatch
to body-fully-copied, milliseconds at ×100, so most of the client's in-flight requests
sit in client-side work rather than router leases; router-side in-flight per backend
stays well under the cap and peak throughput of ~1900 req/s says the driver is the
bottleneck. The sampled workload shifts with it — a ×100 run walks ~1050 sessions
instead of ~1730, at nearly twice the requests per session. Everything the split guard
exists to do happens under saturation, so a fast run exercises it only weakly.

**Sanity anchor.** The `bad` arm at ×10 lands at `avg_copies` 1.679 with 38.4% server
cache — within noise of the numbers this router produced BEFORE the guard existed
(1.675 / 38.4%). Setting the guard to 0.01 effectively removes it and the original
defect returns to two decimal places, which is the clearest confirmation available
that the guard is the mechanism that fixed it. It doubles as a regression fixture: if
a future change makes the `good` arm look like the `bad` one, the guard has stopped
working.

**Use it as:** ×100 for go/no-go on every change; ×10 before citing any number;
`go test ./router/internal/policy/affinity/ -run TestFleet` (about a second, real flow,
per-backend ground truth) before either.

## Depth matters: run length changes what a run is valid evidence for

At the standard ×10 speedup, a SHORT run (`--total 4800`) under-thrashes the cache —
cache-hit levels read ~15-20 points optimistic vs. what a longer run settles at, because
compressing wall-clock time shortens re-reference distances between requests sharing a
prefix, so more of them land inside the cache's retention window before it would
naturally evict them in real time. This is fundamental to time compression, not a bug —
don't chase it by tuning rates further.

A LONG run at the same ×10 speedup (`--total 30000`, ~6 min) reached warm 71.2% /
cached 38.7%, with the warm/cached SPREAD matching the real fleet's shape (real
4,800-request reference: 51.3%/35.6%, capped run). So:

- **Short runs (`--total 4800`) are WORKLOAD-realistic only** — good for exercising
  routing logic and getting a fast signal, not for citing an absolute cache-hit number.
- **Long runs (`--total 30000` at ×10, ~6 min) are CACHE-realistic** — the run has had
  enough depth for the cache's retention dynamics to settle into the real fleet's
  shape. Use this length whenever a run's cache-hit numbers are going to be cited or
  compared against real-fleet data.

Same-depth validation (2026-08-06, standard recipe on both arms, `--total 30000`,
c128/no-hot/cap-32): mock ×10 warm 71.2% / cached 38.7% in 353s vs REAL fleet
warm 67.8% / cached 41.1% in 1,911s — both axes within ~3.5 points at 5.4× less
wall time. Caveat for long REAL runs driven from a laptop: ~18% of requests died
client-side (mid-stream EOFs on the laptop↔cluster path under sustained 128-way
SSE; router logged only 200s/rare 502s and backends were clean) — shares are
computed over successes so comparisons stand, but for clean long real runs prefer
driving wekai from inside the cluster (it is baked into the vLLM image).

## Sizing `--block-capacity`

Set per-instance `--block-capacity` to the real fleet's reported `GPU KV cache size`
(tokens) ÷ `--block-size-tokens`, so the mock's eviction pressure — not just its
latency — matches the real backend it's standing in for.
