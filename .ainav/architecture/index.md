# Architecture Pointers

## `kvcache` — the shared prefix-cache trie engine

`kvcache/` (module root, stdlib-only leaf — `router/hack` fences this so neither the
benchmark CLI nor the router can pull in anything heavier). One chain-hashed trie
(`kvcache.Trie`) is the SAME implementation used by three different consumers, so
prediction and ground truth can never drift from each other by construction:

- **Benchmark's offline cache simulation** (`benchmark/`) — `RecordAndCount`, one
  infinite cache, no eviction. The ROUTER no longer uses `kvcache.Trie` at all: the
  routing flow owns a single shared marked tree (`policy/affinity/tree.go`) instead of
  N per-backend tries, and it could not live in `kvcache` regardless — it needs
  `clock.Clock` for TTL, which the stdlib-only fence forbids.
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

## `router/internal/policy/affinity` — the routing flow

There is ONE routing flow, not a set of policies. `router/internal/policy/cache`
(`prefix-cache-aware`, `prefix-cache-candidates`, and the per-backend `trieStore`) is
gone, along with `round-robin` and `random`; the parts worth keeping came back as
signals. `--policy` no longer exists.

- **`tree.go` / `markset.go`** — one shared prefix tree, sharded 16 ways, whose runs
  record WHICH backends hold them in a growable bitset. Replaces N per-backend tries:
  "who holds this" is a property of the tree rather than something reconstructed by
  querying every candidate.
- **`policy.go`** — the ladder: `cache` (a usable backend holds the DEEPEST marked
  run) → `split` (no holder usable; grow the holder set, guarded) → `reject`
  (429 `split_guard_blocked`) → `load` (nothing marked anywhere). No threshold, and
  no serve-anyway path: a guarded split is the only way a backend is ever recorded as
  holding a prefix.
- **`signal.go`** — what varies between deployments. A signal answers "can this
  backend take more work, and if not, what in-flight level counts as *as loaded as
  it*" for the split guard. It never routes.
  - `refused` — ALWAYS ON, no configuration. The backend's own 429, the only ground
    truth about a vLLM's capacity the router gets. Latched against the in-flight
    count it happened at, so it clears when load falls rather than on a timer.
  - `concurrency` — opt-in via `--max-node-concurrency`. Predicts saturation instead
    of discovering it a round trip late.
  - `imbalance` — opt-in via `--rebalance-ratio`. `(inflight - fleetMin)/inflight >
    ratio`. Off by default: a fleet where affinity is working is SUPPOSED to look
    imbalanced.

Capacity is judged here and nowhere else. `gateway.Server.candidates()` filters for
HEALTH ONLY (health, dialect match) — it used to apply the concurrency cap too, which
split "is this backend full" across two components that could disagree.

`least-outstanding` survives as the flow's selector — the tie-break among equals — not
as a policy.

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
- **`TestCoreDoesNotImportDialects`** — `registry`/`lease`/`policy`/`policy/affinity`/
  `proxy`/`circuit`/`health` never import a dialect package (wire-format knowledge is
  fenced to `dialect/*` and `gateway`).
- **`TestAuthIsEnforcedInExactlyOnePlace`** — exactly one `subtle.ConstantTimeCompare`
  call site in the whole binary.
- **`TestNoArgvDump`** — nothing outside `cmd/*/main.go` references `os.Args` (a raw
  argv dump can leak a secret passed as a flag before logging/redaction exists).

Run these like any other test (`go test ./router/hack/...`); they're part of the
standard `go test ./router/...` sweep, not a separate step.
