# Cache-affinity routing redesign — handoff spec

**Status: IMPLEMENTED**, as a new policy `prefix-cache-split`
(`router/internal/policy/affinity/`) rather than as changes to
`prefix-cache-candidates`. Both existing cache policies are untouched. See
**§9 Outcome** for what was built, what was measured, and which of §7's open
questions are now answered — read that first; §1–§8 below are the original
gap analysis, preserved as written.

Two deliberate divergences from the reference simulator were taken. They have
since been **reviewed with Anton and one of them reversed** — see §9.7, which
supersedes §9.2's first half and the measurements in §9.4/§9.5 that depended on
it. Read §9.7 before trusting any number in §9.2-§9.5.

**Purpose of this document:** the current `prefix-cache-candidates` policy
(`router/internal/policy/cache/threshold.go`) was built and shipped, then an
architecture review surfaced that it diverges substantially from what the
architect actually intends. This document is the handoff: it records his
intent verbatim, the gap against what exists today, and the primary design
changes needed to close it. It is meant to be a **self-contained starting
point** for whoever implements this — no prior conversation context is
assumed.

---

## 1. Source material

### 1.1 The architect's notes (verbatim)

This is Anton's own writing, unedited, because paraphrasing loses precision
on a design like this:

> What I would expect from "routing by load" is a single request going via
> load-split, but then every following request will have cache path split
> based routing falling under cache, so i would expect absolute majority to
> be routed by by cache and not by load
>
> Few more thoughts on it
>
> - Ignore any optimization ideas that are done not for the worst case
> - concurrency limit on vllm is ultimate signal . ultimate means that if we
>   throw out everything else system behaves well
>   - definition of well = fully loaded. all nodes were able to reach
>     maximum concurrency before router returned 429 to caller
> - split is done by signals
> - signals might be expanded, to such as disbalance, 429(ultimate one), but
>   its not worth adding more signals beyond 429 until 429 behaves best
>   possible way
>
> this also means, that if being fully loaded is acceptable - then loading
> single server, because everything was routed to single server, until it
> reaches full capacity is also acceptable. System does not have to behave
> better under little load then under full load. If full load is
> unacceptable performance - then concurrency is not set right. concurrency
> is the only reasonable layer of admission
>
> only after above is working well, adding more signals can bring
> improvements for under loaded systems overall (system == multiple vllm
> instances)
>
> but signals will do exactly same split logic - instead of splitting onto
> new node by block hash. Splitting means this hash present on both nodes,
> select randomally any of them prioritizing less loaded. If all loaded -
> split again. We had 2 nodes - mark it as existing on 4, and so on
>
> If such logic leads to a point, that whole system marked as a hash, it
> means one of two things:
>
> - all requests indeed have such shared prefix in majority of requests
> - system saturated and we start splitting everything, next solves:
>
> limitation of split(whatever is the signal)
>
> - never split onto node that has within 20% of own signal , consider is as
>   unsuitable
>   - i.e if signal is the ultimate one, we know at this point that we
>     already have 32 requests in the air by inflight atomic, so if
>     candidate has 30 in the air - we are not splitting onto itl. This
>     should prevent marking absolutely every node as having absolutely
>     every hash
>
> make sure every node in prefix tree path marked on split, not only the
> last hit
>
> we might add "merge signals", when hashes removed only from part of nodes
> but i dont believe it will be needed and easier to consider tree nodes
> locations permanent until expiration of tails
>
> expiration of tails means we never evict from the midle, eviction done by
> tails
> for this - maintaining list of tails, to avoid traversing whole tree
> tails wont grow to a size it is problematic to traverse them
> every tail eviction immediately checks its parent for eligibility of
> eviction as well , allowing to throw out long chains
> on propogation-up of eviction - never evict if tree node has other
> children. last child will cleanup
>
> ttl should be 30min or 5min (we need to test memory usage), but overall i
> believe 5 minutes will make it work near perefect

### 1.2 The simulator

Anton produced `kv-router-sim.html` (a single-file JS simulator, no
dependencies), copied into this repo alongside this document at
`router/docs/kv-router-sim.html` so it travels with the handoff. It
implements the above concretely and lets you play with
load/concurrency/guard/TTL/node-count sliders against a 1-hour synthetic
agentic-session replay — open it directly in a browser, no build step.
**Read the `<script>` tag in that file before writing any code** — it is a
working reference implementation of every mechanism described below,
including the eviction cascade and the split guard, and its
`renderVerdict()` function *is* the acceptance criterion (see §6).

Key functions in the simulator, referenced throughout this doc:

- `walk(blocks, len)` — read-only lookup: walks the request's block-hash
  path down the shared tree, returns the deepest **marked** run and how many
  blocks matched.
- `commit(blocks, len, node)` — inserts the path and marks the serving node
  on **every** run it touches, not just the last one.
- `route(blocks, len)` — the decision: least-loaded among the marked set if
  any marked node is under its concurrency limit; otherwise **split** onto
  an unmarked node that clears the guard; otherwise reject (429).
- `evictTail(run, depth)` / `sweepTails()` — TTL-based tail-only eviction
  with upward cascade.

---

## 2. Current state (what exists today)

Two cache-aware policies live in `router/internal/policy/cache/`:

- **`cache.go`** (`Policy`, config name `prefix-cache-aware`) — the older
  one. Picks a single best-scoring backend, with a load-imbalance spill
  guard (`BalanceAbsThreshold`/`BalanceRelThreshold`, comparing max vs. min
  raw in-flight across candidates).
- **`threshold.go`** (`ThresholdPolicy`, config name
  `prefix-cache-candidates`) — the one this document is about, and the one
  actually deployed. Filters candidates to those whose predicted-hit
  fraction exceeds a threshold, then picks among that filtered set.

Both sit on top of `kvcache.Trie` (`kvcache/kvcache.go`): **one independent
trie per backend**, each modeling only what that one backend has been sent.
There is no shared structure across backends.

### 2.1 `ThresholdPolicy.Select` today, precisely

```go
// per candidate c:
cached, total := trie[c].Query(rr.Units)   // total = ENTIRE current request's estimated tokens
frac := cached / total
if frac > CacheThreshold /* default 0.5 */ { hot = append(hot, c) }

switch len(hot) {
case 0:   return leastLoadedOfAll(cands)                    // no affinity signal, ignore cache
case 1:   if hot[0].Inflight() < MaxPending /* default 32, absolute */ {
              return hot[0]
          }
          return leastLoadedOfAll(cands)                    // abandon cache entirely for this request
default:  return leastLoadedAmong(hot)                       // tie-break among the hot set
}
```

Config (`ThresholdConfig`, shared with `prefix-cache-aware`'s knobs):
`CacheThreshold = 0.5`, `MaxPending = 32`, trie bounds
`kvcache.RouterConfig() = {MaxNodes: 100_000, MaxTokens: 2_000_000,
EvictBudget: 64}`.

`kvcache.Trie` chunks raw content into 1024-byte units (~256 estimated
tokens each, 4-bytes/token heuristic), hashed with SHA-256, stored in a
per-backend radix trie with size-bounded LRU eviction (evicts the
globally least-recently-used **leaf**; a node with children is never a
leaf so this already respects "never evict from the middle," just via
recency-since-touched rather than time-since-touched).

### 2.2 Known adjacent bug (already identified, not yet fixed)

`registry.Backend.Capacity` — the denominator behind `NormalizedLoad` and
the value that *should* represent each backend's real vLLM concurrency
ceiling — defaults to **1** in the current deployment (`-backends=<urls>`
only carries URLs; there's no flag to set per-backend capacity, so it falls
back to `cfg.MaxInflightPerBackend`, default 1). It is never wired to vLLM's
actual `--max-num-seqs`. This matters a great deal for this redesign — see
§4.4.

---

## 3. Why this needed a redesign, not a tuning pass

Two observations from the review session made clear the gap isn't
parametric:

1. **Predicted-fraction telemetry showed avg ≈ 15%, and only ~46% of
   decisions classified as `cache`** — far below what Anton expects ("absolute
   majority... routed by cache"). Root-caused to §4.2 below: the 0.5
   threshold is measured against the *entire current request*, which for a
   growing multi-turn conversation shrinks the shared portion's fraction
   over time even though the backend still holds every token it's ever seen.
2. Re-reading the simulator's algorithm side by side with `threshold.go`
   shows the two aren't the same design tuned differently — one has a
   threshold gate and a hard abandon-on-saturation fallback with no
   admission concept; the other has neither, and instead grows the
   candidate set under pressure and treats real backend concurrency as the
   sole admission gate.

---

## 4. Primary design changes required

Each item: current behavior → desired behavior → why it matters → what has
to change.

### 4.1 Shared, cross-backend-marked prefix tree (replaces N private tries)

- **Current**: one independent `kvcache.Trie` per backend. Router queries
  every candidate separately to indirectly reconstruct "who has this."
- **Desired** (simulator's `marks: Set<nodeId>` per tree run): one tree.
  Each run/segment carries the set of backend IDs known to hold it. A single
  `walk()` finds the deepest matched, marked segment and reads its holder
  set directly — O(1) lookup instead of O(candidates × trie depth).
- **Why**: this is the foundation the other four changes build on. "Which
  nodes hold this, and how many" needs to be a property of the tree itself,
  not something reconstructed by querying every candidate and comparing
  independently-computed fractions.
- **Implementation direction**: replace `kvcache.Trie` (or add a sibling
  type — see §7 open question on whether the offline benchmark simulator
  keeps using the old per-cache `Trie`) with a structure whose node carries
  a mark set, not just a single owner. `commit()` in the simulator marks
  **every** node along the matched path, not just the terminus — carry that
  over exactly; it's explicit in Anton's notes ("make sure every node in
  prefix tree path marked on split, not only the last hit").

### 4.2 Drop the whole-request-fraction threshold; deepest-match-wins instead

- **Current**: a candidate must have `cached/total > 0.5` where `total` is
  the *entire current request's* estimated size.
- **Desired**: no threshold at all. Walk the request's blocks down the
  tree; the deepest node that has any mark defines the candidate set,
  regardless of what fraction of the whole request that segment represents.
  A 5%-of-request match with a real, walked-from-root hit is exactly as
  valid a cache signal as a 95% match.
- **Why**: this is the direct explanation for the ~15%-avg / ~46%-cache
  numbers observed. Long-running conversations accumulate tokens; a fixed
  shared prefix (system prompt, tool definitions) becomes a shrinking
  fraction of an ever-growing total even though the backend serving that
  conversation still holds everything it's ever seen. The threshold
  structurally penalizes exactly the sessions you'd most want pinned.
- **This is almost certainly the single highest-leverage, cheapest change**
  in this whole document — it doesn't require the shared-tree rewrite to
  attempt, though it's most correctly expressed once §4.1 lands (querying
  N private tries for "deepest match, not fraction" is still possible, just
  less direct).

### 4.3 Proactive split with a relative guard (replaces hard abandon)

- **Current**: when the sole hot candidate exceeds `MaxPending` (absolute,
  32), cache affinity is abandoned entirely for that request — fallback
  goes to least-loaded of *all* backends, which may well be cold.
- **Desired** (`route()`'s split branch): when every marked candidate is at
  its own concurrency limit, don't give up — **split**: find a node outside
  the marked set whose `inflight < own_signal × (1 − guard)` (guard = 20%
  by default, and both sides of the comparison use the *same* signal —
  Anton's example: "we already have 32 in the air... if candidate has 30 in
  the air, we are not splitting onto it"), route there, and mark it on
  every node along the matched path — so next time, that node is *also* a
  legitimate holder. The marked set for a hot prefix grows under pressure
  instead of being thrown away.
- **Why**: this is what produces Anton's "if the whole system ends up
  marked as one hash, that's fine — either the workload genuinely shares
  that prefix, or the system is saturated and this is exactly how it should
  spread load." Our current fallback can't converge to that; it's binary.
- **Guard is relative, not absolute**: `MaxPending=32` today is a fixed
  number independent of what any candidate's real capacity is. Anton's
  guard scales with the reference node's own signal, so it behaves
  correctly regardless of what that signal's absolute value is.

### 4.4 Real per-backend concurrency as the sole admission signal

- **Current**: no per-backend concurrency ceiling actually reaches routing
  decisions (`Capacity` defaults to 1, unrelated to vLLM's `--max-num-seqs`
  — see §2.2). Admission/shedding (503) is driven by the *router's own*
  global semaphore (`MaxConcurrentRequests`), not by "every backend
  individually saturated."
- **Desired**: "concurrency limit on vLLM is the ultimate signal... if we
  throw out everything else, system behaves well." A 429/reject should
  happen if and only if *every* candidate — marked or not, after the split
  attempt — is at its real concurrency ceiling. The simulator's verdict
  logic (§6) *is* the definition of "behaves well": every node reaches max
  concurrency before the first rejection, with as close to zero prematurely
  idle capacity as possible at the moment of rejection.
- **Why**: none of §4.1–4.3 can be evaluated as correct without this. The
  split guard's threshold, the "am I saturated" check that triggers a
  split, and the final reject-or-not decision are all expressed in terms of
  this signal.
- **What has to change**: `Backend.Capacity` needs to reflect each
  backend's actual concurrency ceiling (plumb `--max-num-seqs` or
  equivalent through discovery/static config — this is the pre-existing bug
  in §2.2, now load-bearing rather than cosmetic). The router's
  rejection/admission path needs to be reconsidered against "reject only
  when no candidate — including split candidates — clears the guard,"
  which is a different flow from today's global-semaphore 503.

### 4.5 Tail-only, TTL-based eviction (replaces per-backend size-bounded LRU)

- **Current**: per-backend, size-bounded (`MaxNodes: 100_000`,
  `MaxTokens: 2_000_000`), evicts the globally least-recently-used leaf.
  Already respects "never evict a node with children" as an invariant (the
  LRU list only ever contains leaves — see the comment on
  `Trie.evictLocked`), just via recency-since-touched rather than
  age-since-touched, and per-backend rather than shared.
- **Desired**: TTL-based (~5 minutes, pending memory testing — Anton
  flagged 30 min as the alternative to test), tail-set-only, on the
  **shared** tree from §4.1. An explicit maintained set of tails avoids
  tree traversal. Evicting a tail immediately re-checks its parent for
  eligibility (both marks-empty and children-empty), cascading upward
  through dead chains in one pass; a node with any remaining child is never
  touched — the last child triggers the cleanup.
- **Why**: on a shared tree, "evict" is more subtle than on N private ones
  — removing one backend's mark from a run doesn't delete the run if other
  backends still hold it. This needs the mark-removal and structural
  eviction to be separate steps, cascading correctly. This is lower
  priority than §4.1–4.4 (it doesn't change routing correctness, only
  memory bounds and prediction staleness), but the shared-tree rewrite
  needs *some* eviction story before it ships, and it should be this one
  rather than porting the old per-trie LRU.

---

## 5. Suggested implementation staging

Not gospel — validate against Anton before committing, per §7 — but ordered
by leverage-to-cost:

1. **§4.2 (drop the whole-request threshold)** first. Cheapest, and
   probably fixes most of the "not enough cache routing" symptom on its
   own, even against the current per-backend-trie structure. Good validation
   step before the bigger rewrite.
2. **§4.1 (shared marked tree)**. This is the real rewrite. Everything
   after this point should be built directly against it, not retrofitted
   onto per-backend tries.
3. **§4.3 (split-with-relative-guard)**. Needs §4.1 to have a "mark an
   additional node" operation to build on.
4. **§4.4 (real concurrency signal + admission rework)**. Can technically
   be done in parallel with 1–3 (it's a separate data-plumbing problem —
   `Capacity` wiring), but the *admission* half (reject only when nothing,
   including split candidates, clears the guard) depends on §4.3 existing.
5. **§4.5 (tail-TTL eviction)**. Last — needed before the shared tree can
   run unbounded in production, but doesn't block correctness validation of
   1–4 using a short-lived process or a generous bound as a placeholder.

---

## 6. Acceptance criteria — borrow the simulator's verdict logic

The simulator's `renderVerdict()` (in the HTML file, §1.2) already encodes
what "correct" means for this design, and it's worth lifting directly
rather than re-deriving:

- **FAIL**: any 429 issued before *every* node has reached its own max
  concurrency at least once.
- **PASS**: every node reached max concurrency before the first 429, and
  mean idle capacity across all rejections is under ~5%.
- **MARGINAL**: saturated first (so not a correctness bug), but a
  non-trivial fraction of rejections still found idle capacity elsewhere —
  in the simulator this is attributed to sessions with a single-candidate
  set that the guard wouldn't let split onto anything.

Recommend building the equivalent as an integration-level check (real
router, real or simulated backends) — track "first-429 timestamp" vs.
"all-nodes-maxed timestamp" and idle-capacity-at-rejection, the same way the
simulator's `stats` object does — rather than only unit-testing individual
routing decisions. Unit tests can and should still cover the per-decision
mechanics (deepest-match lookup, split-guard arithmetic, eviction cascade)
independently.

Existing router metrics that are relevant starting points (see
`router/internal/metrics/metrics.go`): `router_route_decisions_total`,
`router_cache_predicted_fraction`, `router_worker_load_{avg,max,min}`,
`router_backend_inflight`. New metrics will likely be needed for: splits
(count, and which node was split onto), 429s split by
justified-vs-premature (mirroring the simulator's `rejHard`/`rejPremature`),
and tail-set size / eviction count if §4.5 lands.

---

## 7. Open questions to validate before/while implementing

- Does `prefix-cache-aware` (`cache.go`, the older sibling policy) get
  retired, redesigned alongside this, or left alone as a separate,
  intentionally-different strategy? This document only covers
  `prefix-cache-candidates`.
- Does the offline benchmark simulator (`kvcache.Trie.RecordAndCount`,
  used by `benchmark/`) need to move onto the new shared-tree structure too,
  or does it keep using today's `Trie` (it models a single infinite cache,
  which is a different problem than "N backends, each with marks")? Package
  doc in `kvcache/kvcache.go` describes both consumers — re-read it before
  deciding whether they still belong in one type.
- Exact TTL value: Anton flagged 5 vs. 30 minutes as untested against real
  memory usage — needs a decision informed by actual measurement, not a
  guess.
- Chunking granularity: current `kvcache` units are 1024 bytes (~256 tokens
  estimated at 4 bytes/token); the simulator uses 256-token blocks directly
  (matching vLLM's real block size more closely, 16 tokens × 16 = 256).
  Worth checking whether closing that gap improves match fidelity, since
  Anton's design is block-hash-based like vLLM's own prefix cache, not an
  approximated byte-window.
- How does §4.4's per-backend concurrency ceiling get discovered/configured
  in practice — a new flag/field, or inferred from health checks, or
  required in static backend config? The static `-backends=urls` flag has
  no capacity field today (see §2.2); Kubernetes discovery
  (`router/internal/discovery/k8s`) may have more to work with via
  annotations, worth checking before designing the config surface.
- Relationship to the router's own global admission
  (`MaxConcurrentRequests` / `concurrencyMiddleware`, 503-on-router-full):
  does that stay as an outer, router-memory-protecting bound, with the new
  per-backend-concurrency logic operating as an inner layer, or does one
  subsume the other?

---

## 8. Where to look in the codebase

- `router/internal/policy/cache/threshold.go` — the policy this redesign
  replaces/reworks.
- `router/internal/policy/cache/cache.go` — the sibling policy; also home
  to `trieStore`, the per-backend trie lifecycle helper shared by both
  (`AddBackend`/`DropBackend`/`Flush`/`Stats`) — whatever replaces
  per-backend tries will need an equivalent lifecycle story.
- `kvcache/kvcache.go` — the current trie implementation and its package
  doc, which explains the two existing consumers (router, online;
  benchmark, offline) and why they share one type today.
- `router/internal/registry/backend.go` — `Capacity`/`NormalizedLoad`,
  relevant to §4.4.
- `router/internal/gateway/middleware.go` — `concurrencyMiddleware`, the
  router's current (different-purpose) admission gate, relevant to the
  open question in §7.
- `router/internal/metrics/metrics.go` — existing observability to extend
  per §6.
- `router/docs/rewrite/requirements.md` and `design.md` — house style for
  numbered requirements and design docs, if this document's structure needs
  to be extended into that format.

---

## 9. Outcome

Implemented as `--policy prefix-cache-split`, a new policy in
`router/internal/policy/affinity/`, sharing no mutable state with
`policy/cache`. `prefix-cache-aware` and `prefix-cache-candidates` are
unchanged and still selectable. Writing a new policy rather than reworking an
existing one was the explicit call: the decision ladders are different enough
that a shared implementation would have been two policies wearing one coat.

### 9.1 What was built

| §  | Change | Where |
|----|--------|-------|
| 4.1 | Shared, cross-backend-marked prefix tree | `affinity/tree.go`, `affinity/markset.go` |
| 4.2 | No threshold — deepest available run anchors | `affinity/policy.go`, tier 1 |
| 4.3 | Split with relative guard, growing the holder set | `affinity/policy.go`, tier 2 |
| 4.4 | Per-backend concurrency as the sole admission signal | already in the gateway; see 9.3 |
| 4.5 | Tail-only TTL eviction with upward cascade | `affinity/tree.go`, `sweep`/`evictTail` |

Holder sets are a `markSet` — a growable `[]uint64` word slice behind
`Has/Add/Remove/Intersects/Count/Clone/Each`, so there is no ceiling on fleet
size and no bitwise code outside that one file. The tree is sharded 16 ways by
first block hash (lossless: a walk always starts at the root child keyed by
block 0). Model isolation is structural, via one root sentinel per model key,
because the gateway filters candidates by `DialectID` and never by `Model`.

Two invariants are asserted under randomised operation sequences, because
future changes could silently break either:

- the tail set is exactly the childless runs;
- a descendant's holders are always a subset of its parent's — this is what
  makes "deepest marked run" also mean "smallest, most specific pool".

### 9.2 The two deliberate divergences from the simulator

**Marking and routing are separate decisions (a fourth tier).** The
simulator rejects when every holder is saturated and nothing clears the guard,
which is why its own `renderVerdict()` reports MARGINAL: at a 20% guard, every
backend between 80% and 100% of its limit is idle-but-unusable. Anton's stated
reason for the guard is to stop every backend being marked as holding every
prefix — that is about *marking*, not about refusing to serve. So a fourth
tier, `overflow`, routes to idle capacity **without** recording a new holder.
The guard keeps its real job and premature rejection becomes impossible.

The measured trade is real and is reported, not hidden. Under saturation the
reference ladder holds a markedly better hit rate (92.3% vs 71.1%) because it
refuses to serve cold, at the cost of 18,952 of 21,963 rejections landing while
capacity was idle, against zero for this policy. Total work completed is within
1% either way, so this is not a throughput argument in either direction: it is
client-visible 429s versus cache hit rate. On this fleet, KV is backed by WEKA,
so a cold or mispredicted route costs a read from WEKA rather than a full
prefill recompute — which is what makes serving the better side of that trade
here. **If that ever stops being true, the fourth tier is the thing to
reconsider first.**

**Admission stays in the gateway; the policy never rejects.**
`gateway.candidates()` already filters to healthy backends under
`--max-node-concurrency`, and the handler already emits
`429 all_backends_at_capacity` exactly when none are left. That *is* §4.4's
"reject only when every candidate is at its ceiling", already shipped and
tested — no gateway change was needed. One consequence to flag in review:
because saturated backends are filtered out before `Select` runs, the guard's
reference signal is the concurrency **limit** rather than a specific saturated
node's in-flight count. That is equivalent to the simulator in practice, where
a split only ever fires once the pool's least-loaded member has already reached
`conc`.

### 9.3 §7's open questions, answered

- **Does `prefix-cache-aware` get retired?** No. Untouched, still selectable.
  This work is purely additive.
- **Does the offline benchmark move to the new structure?** No. `kvcache.Trie`
  keeps both existing consumers. The new tree lives in the router and could not
  live in `kvcache` regardless: it needs `clock.Clock` for TTL, and
  `kvcache/fence_test.go` pins that package as a stdlib-only leaf.
- **TTL: 5 or 30 minutes?** Default 5m, `--cache-tail-ttl`. At 8 nodes and 150
  live sessions the tree settles at ~400 runs with ~26k blocks expired over 15
  simulated minutes, so 5m is comfortably bounded and 30m was not needed.
  `router_cache_tree_runs` / `router_cache_tail_set` /
  `router_cache_blocks_expired_total` are the evidence for revisiting it.
- **Chunking granularity?** Unchanged at `kvcache.DefaultChunkBytes` (1024
  bytes, ~256 estimated tokens), which already matches the simulator's
  256-token blocks closely. Deliberately not bundled with this change so that
  a granularity experiment is separable from a routing one. Note
  `cfg.Cache.ChunkBytes` is still validated but unplumbed — a pre-existing gap.
- **How does per-backend concurrency get configured?** Via the existing
  `--max-node-concurrency`. It is now **mandatory** for this policy: startup
  fails naming the flag. With the shipped defaults every other capacity source
  reads 1 (`--backends` carries no capacity field, `MaxInflightPerBackend`
  defaults to 1, and `Backend.Capacity` clamps below 1 up to 1), so a policy
  that quietly accepted the default would compute its guard against a
  meaningless number. Real per-backend capacity plumbing (§2.2) remains a
  separate follow-up.
- **Relationship to `MaxConcurrentRequests`?** Unchanged. It stays the outer,
  router-memory-protecting bound (503 `router_at_capacity`); per-backend
  concurrency is the inner layer (429 `all_backends_at_capacity`). Neither
  subsumes the other.

### 9.4 Measured

Offline fleet simulation (`affinity/fleetsim_test.go`): the simulator's
workload and latency model ported into Go, driving the real policy code, with
per-backend ground truth so prediction is checked rather than assumed. 8 nodes
x 32 concurrency, 15 simulated minutes.

| | moderate load (150 sessions) | saturated (420 sessions) |
|---|---|---|
| peak utilisation | ~80% | 100% |
| prefix hit rate | **93.9%** | 71.2% |
| vs `least-outstanding` | 72.7% (**+21.2 pts**) | 68.0% (+3.1 pts) |
| premature 429s | 0 | **0** |
| mean holders per block | 1.05 | 3.4 |

Against the mock fleet, on a workload where a modest shared prefix is a small
*fraction* of each request — the shape that defeats the 0.5 threshold:

| policy | cache decisions | load decisions |
|---|---|---|
| `prefix-cache-candidates` | 0 | 201 |
| `prefix-cache-split` | **192** | 9 |

Tier 2 confirmed end to end: warming a prefix on one backend and then
saturating it produced 8 splits that grew the holder set across the fleet, with
zero rejections and zero overflows.

Routing cost: `BenchmarkWalk` is ~134 ns/op with zero allocations, parallel,
over a 75-block path across 64 backends — against the NFR-2 p99 budget of
250 us.

### 9.5 Hazards accepted, and what watches them

**Holder sets on hot runs only ever grow.** A run under continuous traffic
never reaches its idle TTL, and TTL is the only mechanism that removes a
holder. This does *not* degrade routing — the subset invariant means a request
walks past a widely-held ancestor to its own narrower run — but it does mean
real KV duplication. Measured `avgCopies` is ~1.05 where affinity is
achievable and ~3.4 under 164% oversubscription; the latter is a property of
the offered load, since sessions that cannot stay on one backend genuinely are
on several. Watch `router_cache_avg_copies`: a high value at *low* utilisation
is the one that needs investigating. The fix, if it is ever needed, is
per-backend mark budgets sized to real KV — which is why the subset invariant
is tested now, since such eviction would have to drop marks tail-first or
`walk` starts lying.

**Concurrent first-touch marks more backends than necessary.** Commit happens
only after the upstream accepts, so every request for a brand-new prefix
arriving inside that round-trip still sees an unmarked tree and spreads by
load. Observed directly against the mock fleet: a burst of 24 concurrent
clients on one novel prefix produced ~25 `load` decisions before converging,
after which 376 of 400 requests routed by cache. One round per novel prefix,
self-correcting, and cheap given WEKA-backed KV. The fix — reserve at dispatch,
confirm on accept — conflicts with CACHE-9 and was deliberately not built. The
trigger for revisiting is `avg_copies` rising with client concurrency at fixed
offered load.

### 9.6 Where to look

- `router/internal/policy/affinity/markset.go` — holder sets.
- `router/internal/policy/affinity/tree.go` — the shared tree, walk/commit/
  split, slots, tail-TTL eviction.
- `router/internal/policy/affinity/policy.go` — the four-tier ladder.
- `router/internal/policy/affinity/snapshot.go` — `/router-viz`; the tree is
  natively the shape the page wants, so the per-poll merge step is gone.
- `router/internal/policy/affinity/fleetsim_test.go` — the fleet simulation,
  the verdict gate, the A/B, and the reference-ladder cross-check.
---

## 9.7 Reversal: the fourth tier is gone

§9.2 recorded two divergences from the reference simulator and noted neither had
been reviewed. The first — "marking and routing are separate decisions", the
`overflow` tier — has now been reviewed and **reversed**. Anton's call,
verbatim: *"there is no MUST serve. if we cannot split by 20% rule - we dont
split. there should not be any sort of 'lets fallback anyway'. only by split"*,
and on the trade: *"making it around 1.05 (even if not all nodes will reach 100%
concurrency and client will start receiving 429s) is the goal."*

The second divergence (admission lives in the gateway) stands.

### What was actually wrong

The `overflow` tier was not the main leak, and §9.5's hazard note pointed at the
wrong mechanism. `walk` returns the deepest run held by *a candidate*, and tier 1
fired on that — not on the deepest MARKED run. So the ordinary saturated case
(session's tail held by A, A at the cap and filtered out by the gateway, shared
system prompt held by everyone) was answered by **tier 1**, routed to any backend
merely under the limit, and then `commit` marked it on the whole path including
the session's private tail. The guard never saw those decisions. `overflow`
fired 0 times on the mock fleet while this path fired on ~50-60% of all cache
decisions.

`router_cache_shallow_anchors_total` now counts them. Below saturation they are
~0% of cache decisions and duplication sits at 1.01; at 100% utilisation they
were ~50-60% and duplication reached 2.5 (4 nodes) or 3.4 (8 nodes).

### What the numbers actually were

Two intermediate fixes were built and rejected, which is worth recording because
the obvious one is wrong. Suppressing the *mark* on a shallow anchor (either by
guarding it, or by requiring the deepest anchor and letting overflow serve) drove
`avg_copies` to 1.01 — and left real duplication **unchanged at 2.7-4.5**,
measured against per-backend ground truth in `fleetsim_test.go`. The backend
caches what it serves whether or not the tree records it. Those fixes only made
the gauge lie. Note the corollary: the gauge was honest all along, reading 2.52
against a ground truth of 2.74.

Only refusing to *serve* moves the real number. Full strict ladder, 8x32 fleet,
420 sessions, versus the shipped serve-anyway ladder:

| | serve-anyway | strict |
|---|---|---|
| avg copies (tree) | 3.41 | **1.06** |
| avg copies (ground truth) | 4.19 | **1.05** |
| prefix hit rate | 73.2% | **93.0%** |
| accepted | 9093 | 9091 |

The trade §9.2 assumed — 429s bought at the price of hit rate — does not exist at
these loads. A request that lands warm finishes sooner and frees its slot sooner,
so accepted work comes out level while the hit rate rises ~20 points. Our ladder
now matches the reference simulator it was ported from (hit 92.6% vs 92.3%,
copies 1.01 vs 1.01), which is the cross-check in
`TestMatchesReferenceLadderAtSaturation`.

Against the mock fleet on the 58-day capture, 4x32, `--total 30000`, client
concurrency 128 (= fleet capacity, so utilisation is pinned at 100%):

| | serve-anyway | strict + client backoff |
|---|---|---|
| `router_cache_avg_copies` | 1.675 | **1.080** |
| server-reported cache hit | 38.4% | **59.2%** |
| warm input | 71.4% | 71.0% |
| completed / errors | 30000 / 1 | 30000 / **0** |
| elapsed | 5m48s | 5m54s |
| peak req/s | 316.08 | 318.21 |
| TTFT p50 / p95 | 0.2s / 0.3s | 0.3s / **1.8s** |
| guard 429s | — | 16306, all absorbed by backoff |

Same work in the same wall time, 21 points more of it served from the backends'
own KV, and duplication down from 1.68 to 1.08.

### What it costs

- **16306 429s** on that run, every one issued while idle capacity existed. That
  is the guard working as specified, not a failure.
- **TTFT p95 0.3s -> 1.8s.** Sessions concentrate on their holder, so per-instance
  prefill contention rises where it used to be spread. Measured server-side —
  the client's backoff wait is excluded from TTFT by construction.
- **The reference `renderVerdict()` would score this FAIL**: the first 429 now
  precedes all-nodes-maxed. `TestFleetVerdictDuplicationHoldsAtSaturation`
  asserts the duplication bar instead and logs this rather than asserting it.
- **Clients must retry.** A client that treats 429 as fatal will see a run
  collapse; the replay client previously did, reporting 68% errors against a
  healthy fleet. It now backs off (10ms doubling to a 3s cap, 30s budget,
  jittered) — see `benchmark/replay_router_post.go`.

### Where to look

- `affinity/policy.go` — the ladder is now cache / split / reject / load.
  `Config.Ladder` keeps `LadderServeAnyway` for A/B only; it is not reachable
  from any flag.
- `policy.ErrSplitGuardBlocked` -> `gateway.go` -> `429 split_guard_blocked`,
  distinct from `all_backends_at_capacity` (which means zero idle slots).
- `router_cache_guard_rejects_total`, `router_cache_shallow_anchors_total`,
  `router_cache_shallow_anchor_blocks_total` are the new signals.
- `affinity/copies_sweep_test.go` — the utilisation sweep, the ground-truth
  duplication measurement, and the ladder A/B.

### Still open

`avg_copies` is measured against the tree, and the tree is a prediction. The
ground-truth cross-check lives only in the fleet simulation; nothing validates it
against a real fleet's reported `cached_tokens`. The strict ladder makes the tree
*conservative* (it can now only under-report, since a holder is added solely by a
guarded split), which is the safe direction, but a production
prediction-vs-observed check is still unbuilt.

---

- `grafana/dashboard.json` in the `wllm` repo — panel 503 now aggregates
  `sum by (decision)` (it previously named cache/load/other individually and so
  would have silently dropped the split and overflow tiers), plus new panels
  for the affinity mechanism, KV duplication, admission, and tree size.
