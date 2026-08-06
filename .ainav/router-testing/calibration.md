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
- **×100**: fast logic-smoke loop — use for iterating on routing LOGIC itself
  (does it route at all, does it 429 correctly, does the tree render), not for any
  cache-hit-rate claim.

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
