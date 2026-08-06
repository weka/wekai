# Mock Fleet Calibration

How to set the mock vLLM's latency-model rate flags
(`--cold-input-tps`/`--cached-input-tps`/`--output-tps`/`--base-latency`) and tokenizer
ratio (`--chars-per-token`) so a mock run reproduces a real fleet's cache-hit behavior,
plus how to compress wall-clock time for fast iteration without distorting the result.

## Calibration facts (2026-08-06, DeepSeek-V2-Lite golden replay, 4×TP2 real fleet)

Real fleet's effective service rates, fitted from per-request data:

| Rate | Value |
|------|-------|
| Base latency | 444 ms |
| Cold (uncached) input | 16,174 tok/s |
| Cached input | 33,297 tok/s |
| Output (decode) | 74 tok/s |

- **Cached input is NOT free** — cached:cold ≈ 2:1, not ∞:1. A cache hit still costs a
  KV read; only recompute is skipped. Setting `--cached-input-tps` far higher than
  `--cold-input-tps` but still finite is load-bearing for matching real behavior.
- At these rates, the mock reproduces the real workload mix: warm share 48.9% (mock) vs
  50.3% (real) on the golden replay.
- **Cache-hit levels stay ~15 points optimistic under ANY speedup** (see below) — this
  is fundamental, not a bug: compressing wall-clock time shortens re-reference
  distances between requests that share a prefix, so more of them land inside the
  cache's retention window than would in real time. Don't chase this gap by tuning
  rates further; it's a property of time compression, not miscalibration.

## Speedup for fast iteration

Speedup = uniformly multiply the three tok/s rates and divide the base latency by the
same factor. This preserves the RELATIVE cost ratios (cold vs cached vs output) that
drive routing/caching decisions while shrinking wall-clock run time.

- **×10** (realistic reproduction): reproduces the real workload mix above in ~2 min
  for the golden replay.
- **×100** (fast logic-smoke loop): ~15s for 4,800 requests — use this for iterating on
  routing logic itself, not for cache-hit-rate claims (the time-compression optimism
  above gets worse as speedup increases).

## Sizing `--block-capacity`

Set per-instance `--block-capacity` to the real fleet's reported `GPU KV cache size`
(tokens) ÷ `--block-size-tokens`, so the mock's eviction pressure — not just its
latency — matches the real backend it's standing in for.
