# Architecture Pointers

## `kvcache` — the shared prefix-cache trie engine

`kvcache/` (module root, stdlib-only leaf — `router/hack` fences this so neither the
benchmark CLI nor the router can pull in anything heavier). One chain-hashed trie
(`kvcache.Trie`) is the SAME implementation used by three different consumers, so
prediction and ground truth can never drift from each other by construction:

- **Router's cache-aware policy prediction** (`router/internal/policy/cache`) —
  `Query`/`Commit` only, LRU eviction, no pinning (the router never has a real
  in-flight request to protect).
- **Mock vLLM's ground truth** (`router/internal/mockvllm`) — `RecordAndPin`/`Unpin`,
  because a live server actually admits requests: blocks must stay pinned for as long
  as the request holding them is in flight, with LRU eviction applying only to
  unpinned blocks.
- **Benchmark's token/cache estimator** — `EstimateTokens`/`ChunkContent`, fixed at a
  4.0 chars/token ratio (the mock's own `--chars-per-token` flag is a DELIBERATE,
  separate override of this default for calibration — see
  [../router-testing/calibration.md](../router-testing/calibration.md)).

Block hashing is chain-hashed (a real vLLM parent-hash chain analog): a match requires
the FULL ancestor chain, not just an equal leaf hash — two prompts whose second block
is byte-identical but whose first block differs do not cross-credit that second block.

## `router/internal/policy/cache` — prefix-cache-aware routing

Two policies share one `trieStore` (per-backend-URL trie, created on backend add,
discarded on drop — never reassigned to another backend):

- **`Policy`** (`prefix-cache-aware`) — picks the single backend with the highest
  predicted-hit fraction above `CacheThreshold`, tie-broken by load; spills to a
  load-based fallback once the fleet is measurably imbalanced (`BalanceAbsThreshold`/
  `BalanceRelThreshold`) or when a request has no routable prefix at all.
- **`ThresholdPolicy`** (`prefix-cache-candidates`) — filters candidates to those
  predicted to hold the prefix, then picks among that filtered set.

Both implement `viz.DataSource` (the merged-tree walk backing `/router-viz` — see
[../viz/index.md](../viz/index.md)) and both are POLICY-LEVEL only: candidate
FILTERING (health, dialect match, `--max-node-concurrency`) happens once, upstream, in
`gateway.Server.candidates()` — every policy inherits it without needing to know it
exists.

## `router/internal/mockvllm` — the mock vLLM engine

`Engine` owns one server instance's live `kvcache.Trie`, admission state
(`MaxConcurrency`), and counters. `Admit()` is the single admission decision:
reserves a concurrency slot AND pins the request's blocks in the SAME step (mirrors a
real vLLM scheduler — a sequence is admitted and its KV blocks reference-counted
together). `AppendOutputBlocks` models decode-KV: on completion, generated tokens
become cached blocks of the same sequence, appended to its chain — occupying capacity
and evictable exactly like prompt blocks (`--output-kv-multiplier`).

`tokenize.go` reimplements `kvcache.ChunkContent`'s chunking locally so
`--chars-per-token` can differ from `kvcache`'s fixed 4.0 default without touching the
shared package.

## `router/hack` — architectural fence tests

Mechanical invariant checks that run as ordinary Go tests, because the defects that
motivated the router rewrite were invariants nobody enforced, not subtle algorithms:

- **`TestOnlyLeaseMutatesInflight`** — only `internal/lease` may mutate a backend's
  in-flight counter (every load-based decision in the router reads it).
- **`TestProductionCodeUsesTheClockAbstraction`** — every time-based DECISION goes
  through `clock.Clock`, not a raw `time.Now()`/`time.Sleep()` (opt out per-line with
  `//clockexempt: <reason>` for pure latency MEASUREMENT, which isn't a decision).
- **`TestNoDeadMetrics`** — every collector declared in `internal/metrics` must be
  referenced somewhere outside that package, or it's a silent permanent zero.
- **`TestCoreDoesNotImportDialects`** — `registry`/`lease`/`policy`/`policy/cache`/
  `proxy`/`circuit`/`health` never import a dialect package (wire-format knowledge is
  fenced to `dialect/*` and `gateway`).
- **`TestAuthIsEnforcedInExactlyOnePlace`** — exactly one `subtle.ConstantTimeCompare`
  call site in the whole binary.
- **`TestNoArgvDump`** — nothing outside `cmd/*/main.go` references `os.Args` (a raw
  argv dump can leak a secret passed as a flag before logging/redaction exists).

Run these like any other test (`go test ./router/hack/...`); they're part of the
standard `go test ./router/...` sweep, not a separate step.
