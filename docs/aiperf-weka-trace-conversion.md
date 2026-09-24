# Converting `replay-v3` captures to AIPerf's `weka_trace` format

`wekai router convert-aiperf-weka-trace` turns a `replay-v3` capture (wekai's
own tree-aware replay format, produced by `wekai router replay-prepare`) into
one JSON file per session in the format AIPerf (`agentx-harness`) expects from
its native `weka_trace` custom-dataset-type. This lets the same captured
agentic-coding traffic be benchmarked through AIPerf, not just wekai's own
`wekai router replay` tool.

```bash
wekai router convert-aiperf-weka-trace \
  --out <output-dir> \
  [--block-size 64] \
  [--include-ephemeral-probes] \
  <replay-v3.jsonl>

# The output directory feeds straight into AIPerf:
aiperf profile \
  --input-file <output-dir> --custom-dataset-type weka_trace \
  --model <model> --tokenizer <tokenizer> \
  --endpoint-type chat --streaming \
  --url <server>
```

This is a pure offline data transformation — no HTTP calls, no capture, no
replay. The two formats disagree on almost everything (session tree shape,
cache-block bookkeeping, timing fields), so the conversion has to rebuild
those from what `replay-v3` actually preserves rather than pass values
through.

## Why this needed real investigation, not a field-by-field copy

AIPerf already has a native `weka_trace` loader
(`aiperf.dataset.loader.weka_trace.WekaTraceLoader`) — the *same* loader used
behind `--public-dataset semianalysis_cc_traces_weka_062126`. So the target
schema was never in question. What took the work was that the source and
target formats aren't just differently-shaped versions of the same
information — **the source is a partially-evidenced, sometimes-forest-shaped
graph, and the target is a rigid, fixed two-level hierarchy the source data
was never guaranteed to fit.**

- The source (`replay-v3`) mixes two confidence levels of parent-child edges:
  some are genuinely detected agent spawns (matched via seed-hash against a
  spawn registry), others are pure "nearest thing in time" filler wekai's own
  producer inserts when it can't find real evidence. Even the real edges don't
  reliably form one tree rooted at `main` — a session can have several
  independent anchor points (a forest), because `main` is a persona label,
  not necessarily the graph's actual root.
- The target (`weka_trace`) is not a general DAG. It requires exactly one root
  sequence, plus subagents that must be flat, one level deep, each anchored
  at exactly one point in the root's sequence. No subagent-of-a-subagent.

Every fidelity decision below is a consequence of coercing the first shape
into the second.

## Source format (`replay-v3`)

Reference: `benchmark/replay_router.go` (struct source of truth),
`cli/command_router_replay_prepare.go` (producer + repair logic).

```
line 1:   {"_schema":"replay-v3", "name":..., "summary": {...}}
line 2+:  one session per line:
{
  "session_id": "...",
  "start_ts": "...",
  "instances": [
    {
      "instance_id": "...",
      "role": "main" | "sub-agent" | "helper-or-isolated" | "orphan-sub-agent" | "ephemeral (no system)",
      "parent_instance_id": "...",       // optional, CROSS-SESSION, informational only
      "parent_spawn_request_id": <uint>, // optional; when present, resolves to a request_id in THIS session
      "requests": [
        {
          "request_id": <uint>,          // corpus-global monotonic counter
          "ts": "RFC3339Nano",
          "model": "...", "stream": bool, "max_tokens": int,
          "input_tokens": int,           // total prompt tokens (ground truth, incl. cached)
          "prefill_tokens": int, "cache_read_tokens": int, "cache_creation_tokens": int,
          "output_tokens": int,
          "system_blocks": [{"type","hash","bytes","tokens","cache_control"?}, ...]?,
          "messages": [
            {"role":"user"|"assistant","hash":"sha256:...","block_types":[...],
             "bytes":int,"tokens":int,"cache_control"?,"seed_hash"?,
             "tool_use_ids"?, "tool_result_ids"?}
          ],
          "stop_reason": "end_turn"|"tool_use"|"max_tokens"|"",
          "upstream_latency_ms": 0,       // confirmed always 0 in the real corpus
          "total_ms": 0                   // confirmed always 0 in the real corpus
        }
      ]
    }
  ]
}
```

Role taxonomy, from wekai's own classifier (`cli/command_router_tree.go`):
`main` (the session's root persona — one per session after a dedup pass),
`sub-agent` (successfully linked to a parent spawn), `helper-or-isolated` /
`orphan-sub-agent` (root-level but **not confidently linkable** — the role
name is a direct admission of that), `ephemeral (no system)` (a tiny
8-in/1-out/`max_tokens=1` probe call — a router keep-alive ping, not real
traffic).

## Target format (`weka_trace`)

Reference: `src/aiperf/dataset/loader/weka_trace_models.py`,
`docs/tutorials/weka-trace.md` in `agentx-harness`. One JSON file per
session, written into a directory consumed via
`--input-file <dir> --custom-dataset-type weka_trace`:

```jsonc
{
  "id": "<session_id>",
  "models": ["..."],
  "block_size": 64,
  "hash_id_scope": "local",
  "tool_tokens": 0, "system_tokens": 0,
  "requests": [
    // root ("main") instance's requests, interleaved with WekaSubagentEntry
    // markers anchored at resolved parent_spawn_request_id
  ]
}
```

## Worked example

Real session `01ba8d66-61c9-48a1-83ce-877fce56ab78` — the clean case, where
everything resolves the way you'd hope.

**Source** (message/system-block bodies abbreviated):

```json
{
  "session_id": "01ba8d66-61c9-48a1-83ce-877fce56ab78",
  "start_ts": "2026-05-12T13:15:13.649569397Z",
  "instances": [
    {"role": "ephemeral (no system)", "requests": [
      {"request_id": 1704, "ts": "13:15:13.64...", "input_tokens": 8, "output_tokens": 1, "max_tokens": 1, "stop_reason": "max_tokens"}
    ]},
    {"role": "main", "parent_spawn_request_id": 1704, "requests": [
      {"request_id": 1707, "ts": "13:15:26.91...", "stream": true, "input_tokens": 362, "output_tokens": 14, "stop_reason": "end_turn"},
      {"request_id": 1708, "ts": "13:15:27.33...", "stream": true, "input_tokens": 362, "output_tokens": 14, "stop_reason": "end_turn"}
    ]},
    {"role": "helper-or-isolated", "parent_spawn_request_id": 1708, "requests": [
      {"request_id": 13, "ts": "13:18:55.56...", "input_tokens": 5203, "output_tokens": 72, "stop_reason": "end_turn"}
    ]}
  ]
}
```

**Target:**

```json
{
  "id": "01ba8d66-61c9-48a1-83ce-877fce56ab78",
  "models": ["claude-haiku-4-5-20251001"],
  "block_size": 64, "hash_id_scope": "local",
  "tool_tokens": 0, "system_tokens": 308,
  "requests": [
    {"t": 13.26, "type": "s", "in": 362, "out": 14, "hash_ids": [0,1,2,3,4,5,6], "stop": "end_turn"},
    {"t": 13.68, "type": "s", "in": 362, "out": 14, "hash_ids": [0,1,2,3,4,5,6], "stop": "end_turn", "think_time": 0.419},
    {"t": 221.92, "type": "subagent", "agent_id": "agent_001", "subagent_type": "helper-or-isolated",
     "duration_ms": 0, "total_tokens": 5275, "status": "completed",
     "requests": [{"t": 221.92, "type": "n", "in": 5203, "out": 72, "hash_ids": [7,8,...,89], "stop": "end_turn"}],
     "models": ["claude-haiku-4-5-20251001"], "tool_tokens": 0, "system_tokens": 4574}
  ]
}
```

What changed and why:

| What | Source | Target | Why |
|---|---|---|---|
| Ephemeral probe | separate instance, `in:8, out:1, max_tokens:1` | gone entirely | dropped by default — router keep-alive ping |
| Trace clock zero-point | `start_ts` = the (now-dropped) probe's own timestamp | `t:0` is `main`'s first request | the probe is gone, so the retained timeline starts later |
| Two identical `main` requests | same `system_blocks`/message hash both times | identical `hash_ids: [0..6]` both times | same content hash → same block ids, reused not re-minted |
| Idle gap between them | `total_ms: 0` both times (unusable) | `think_time: 0.419` | `api_time` omitted (not `0.0`) since the source latency fields are confirmed always 0; `think_time` falls back to the raw timestamp delta |
| `helper-or-isolated`, `parent_spawn_request_id: 1708` | linked to `main`'s 2nd request by number | nested `subagent` entry right after that request | `1708` resolved cleanly, 0 hops, to a request `main` itself owns |

## Tree reconstruction

1. **Drop `ephemeral (no system)` instances** by default (`--include-ephemeral-probes` to keep them). Logged per session and in aggregate.
2. **The `role == "main"` instance is always the root**, regardless of its own `parent_spawn_request_id`. This is not optional: wekai's producer runs an "implicit parent" pass that anchors *every* rootless instance — including `main` — to whatever request chronologically preceded it in the session. `main` carrying a parent pointer is normal, not a sign it should be treated as a subagent of something else.
3. **`parent_instance_id` is never used for tree-building — only `parent_spawn_request_id` is.** `parent_instance_id` is built from a *global*, cross-session lookup in wekai's producer and can point at an instance UUID belonging to a completely different session (confirmed directly on real data — see the worked example below). `parent_spawn_request_id`, by contrast, is guaranteed by the producer's own repair passes (`repairDanglingParents`, `breakParentCycles`) to resolve to a request in the *same* session, or be absent/zero.
4. **Anchor resolution**, for every non-root instance:
   - Resolves in one hop to a request the root owns → nested there directly.
   - Resolves to a request owned by *another* non-root instance (real multi-level nesting exists in the source, but `weka_trace`'s `WekaSubagentEntry.requests` has no nested-subagent variant) → walk `parent_spawn_request_id` transitively upward until reaching a root-owned request, and anchor there. This flattens nesting depth to a sibling anchor point; concurrency/ordering survives via each entry's own `t`.
   - No resolvable parent at all → anchor at the nearest-preceding root request by timestamp (or the very front, if it precedes the root's first request entirely).

### Worked example: the dangling / cross-session-reference case

Real session `0615cea7-1b54-4504-8a02-644834589d83`:

```json
{
  "session_id": "0615cea7-1b54-4504-8a02-644834589d83",
  "instances": [
    {"role": "helper-or-isolated", "requests": [{"request_id": 93, ...}]},
    {"role": "main", "parent_spawn_request_id": 93, "requests": [
      {"request_id": 94, "stop_reason": "tool_use"},
      {"request_id": 101, "stop_reason": "end_turn"}
    ]},
    {"role": "sub-agent",
     "parent_instance_id": "2521506f-7c0f-412a-8566-2b186c7af861:sha256:...",  // <- a DIFFERENT session's UUID
     "parent_spawn_request_id": 94,
     "requests": [{"request_id": 95}, ..., {"request_id": 100}]}
  ]
}
```

- `helper-or-isolated` (request 93) is the session's true chronological start and has no parent of its own. `main` itself carries `parent_spawn_request_id: 93` — confirming point 2 above directly: `main` is not guaranteed parentless.
- `sub-agent`'s `parent_instance_id` points at session `2521506f-...` — not this session at all. Its `parent_spawn_request_id: 94` correctly resolves, 0 hops, to `main`'s own request. This is exactly why `parent_instance_id` must never be trusted for structure.
- `93` has no `parent_spawn_request_id`, isn't `main`, and precedes `main`'s first request entirely → dangling fallback, anchored at the very front of the trace as its own `agent_001` subagent entry.

## `hash_ids` reconstruction

One namespace per session (`hash_id_scope: "local"`), built by walking every
retained instance's requests in session-wide chronological order (not per
instance), minting `ceil(tokens / block_size)` fresh sequential ids the first
time a block's hash is seen and reusing the prior ids whenever the exact
same hash recurs.

Two refinements exist on top of plain hash-string matching, both found by
actually running converted output through AIPerf's real loader rather than
just validating the JSON schema:

### 1. Cache-control-churn reuse

Anthropic's client resends the full message history verbatim on every call,
so a message that's genuinely unchanged between two requests of the same
instance should reuse its ids. But wekai's capture hash is computed over the
block's serialized form *including* its `cache_control` field, and
Anthropic's prompt-caching convention moves the cache breakpoint forward to
the newest block on nearly every multi-turn call — so the block that carried
the marker last time loses it this time (or vice versa), and the **same
underlying text gets a different hash**.

Confirmed on the real corpus: **62% of multi-request instances** show a
same-position, same-role message pair whose byte count matches within 2%,
whose `cache_control` presence differs, and whose hash differs. Left
unhandled, this mints a fresh, unearned id run for old content on the
majority of multi-turn conversations in the corpus.

Fix: when a hash lookup misses, check whether the same (instance, slot)
position's previous occurrence has a matching role, byte count within 10%,
and *differing* `cache_control` presence — if so, reuse those ids instead of
minting new ones. Gated tightly (role + position + tolerance + the specific
presence-flip signature) so a genuinely dynamic per-call block, which
typically carries no `cache_control` on either side, is never merged into a
false cache hit.

### 2. Slot reordering by majority-vote stability

AIPerf's own downstream chain detection (used to recognize a flattened
capture as one continuous conversation) compares `hash_ids` by **longest
common prefix starting at index 0**:

```python
def _hash_list_lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
```

A block that's genuinely per-call-dynamic (a timestamp preamble, no
`cache_control` ever) sitting at position 0 of every request's `hash_ids` —
which is exactly where a system prompt's first block naturally lands in real
wire order — silently defeats this check for the *entire* request, no
matter how well the rest of the content matches. Verified directly: a clean
2-turn `main` conversation was being split into two disconnected
`Conversation` objects by AIPerf's own loader purely because of this,
*even after* the cache-control-churn fix above closed the actual content gap
between the two requests (raw overlap went from 30% to 96%, and the split
persisted regardless, because position 0 still mismatched).

Two designs were tried and rejected before landing on the current one — both
are worth knowing, since each one's failure mode is exactly what the other
gets right:

1. **Classify a slot as stable the moment it's *ever* reused anywhere across
   the instance's whole history, then keep it at the front for every later
   request too.** Wrong: confirmed on the real corpus, a 1,122-request
   single conversation (no subagents at all) had a slot that coincidentally
   repeated once, early on, then changed on nearly every one of the
   following ~1,121 turns. "Ever" latched onto the one coincidence and kept
   the slot at position 0 throughout, breaking the prefix check on almost
   every transition and splitting one real conversation into **1,106**
   disconnected ones.
2. **Classify per occurrence instead of per slot — does this specific
   request's content match the immediately preceding request's, at this
   exact slot.** Wrong differently: a slot with no predecessor yet (a
   request's own first appearance) has nothing to compare against, so its
   own internal ordering falls back to real wire order — which can then
   differ from where a *later* request (whose own copy of that content DID
   have a predecessor to connect to, and got promoted) places the very same
   slot. Two requests that share content, ordered differently relative to
   each other, still fail the prefix check against each other even though
   the content genuinely matches.

**The fix**: a per-slot verdict that's consistent across every request of an
instance (fixing #2), based on how the slot behaves *on average* rather than
whether it *ever once* matched (fixing #1). For each (instance, kind,
position) slot, count its transitions (occurrences after the first) and how
many of those connect to their immediate predecessor (exact hash match, or
the cache-control-churn heuristic above); the slot is stable iff at least
half its transitions connect. Stable slots keep their relative order at the
front of every request's `hash_ids`; unstable slots go to the end, in every
request, uniformly — including a request where that particular transition
happened to connect, and including the slot's own first appearance (no
transitions yet, defaults to unstable, which is harmless since there's no
other request's copy of it to align with).

Measured on a 285-session real sample, before vs. after this fix:

| | Before | After |
|---|---:|---:|
| Sessions converting with zero excess fragmentation | 196/285 (68.8%) | 203/285 (71.2%) |
| Total excess disconnected conversations (corpus-wide) | 13,989 | 1,082 (**-92.3%**) |
| The 1,122-turn case above | 1 expected → 1,106 actual | 1 expected → 11 actual |
| Worst remaining case | 39 expected → 6,494 actual | 39 expected → 432 actual |

Not perfect — 81 of 285 sessions still show some excess fragmentation after
this fix, concentrated in a handful of large sessions. Those remaining cases
haven't been root-caused to the same level of confidence as the mechanism
above; they may be a residual variant of the same pattern, a different
mechanism entirely, or in some cases genuine compaction that AIPerf's loader
is correctly *supposed* to split. Worth a closer look before assuming the
job is fully done, but the dominant, corpus-wide mechanism is fixed.

## Known limitations (verified, not guessed)

These were found by loading real converted output through AIPerf's actual
`WekaTraceLoader`, not just validating the output JSON against the pydantic
schema — schema validity and loader-level fidelity are different questions,
and 100% of converted output passing schema validation says nothing about
either of the following.

**~45% of subagent anchors are a fallback placement, not a recovered fact.**
Breaking down a 285-session real sample's 4,343 subagent anchor
resolutions:

| Outcome | Share | What it means |
|---|---:|---|
| Clean, 0-hop anchor to a root request | 22% | a genuine `parent_spawn_request_id` match |
| Multi-level flattened (walked >1 hop to reach root) | 33% | a real link exists, but through another subagent; flattened to a root-level sibling |
| Dangling fallback (no resolvable link at all) | 45% | anchored by nearest-preceding-timestamp guess |

The dangling rate is a genuine property of the source, not a bug: `main`
starts first chronologically in only ~70% of sessions; the other ~30% have
a `helper-or-isolated` instance (wekai's own "couldn't confidently link"
label) starting earlier, forming its own disconnected chain that never
reaches `main` at any hop count.

**~4.4% of dangling-fallback anchors get silently dropped by AIPerf anyway.**
Of the 45% dangling-fallback instances, those anchored *before* the root's
first request (no preceding turn to hang a `SPAWN` branch on) are dropped by
AIPerf's own loader at load time (`_dropped_subagent_indices` in
`weka_trace.py`, also documented in `docs/tutorials/weka-trace.md`: "subagents
with no preceding parent turn are dropped"). Measured: 86 of 1,959 dangling
instances (4.4%) fall into this bucket. The remaining 95.6% survive with a
fabricated but harmless attachment point. This was a deliberate choice, not
an oversight: preserving 95.6% of otherwise-unplaceable real traffic was
judged worth a small, known, silent loss on the rest, rather than dropping
all of it at conversion time to avoid a partial loss at load time.

**A small amount of residual fragmentation remains after all three `hash_ids`
fixes above** — 81 of 285 real sessions (28%), totaling 1,082 excess
disconnected conversations corpus-wide, down from 13,989 before the
majority-vote fix. See "Slot reordering by majority-vote stability" above
for the measured before/after. Concentrated in a handful of large sessions;
not yet root-caused further.

## Verification approach

Schema validation (`WekaTrace.model_validate()` on every output file) is
necessary but not sufficient — it says the JSON is *shaped* correctly, not
that AIPerf's loader will *treat* it the way you'd expect. Two levels were
used together:

1. **Schema validation** against the real pydantic model, on every
   converted file (285/285 passing across the last full real-data run).
2. **Direct loader invocation** — `WekaTraceLoader.load_dataset()` +
   `.convert_to_conversations()`, called directly against real converted
   files, inspecting the actual `Conversation` objects produced (session
   ids, turn counts). This is what surfaced both the position -1 drop and
   the LCP-prefix fragmentation — neither is visible from schema validation
   or from a live `aiperf profile` run alone (a live run's aggregate request
   count doesn't distinguish "8 turns, 2 conversations" from "8 turns, 6
   disconnected conversations").

A live end-to-end `aiperf profile` run against the mock server
(`aiperf-mock-server`) was also exercised, useful for confirming the CLI
invocation shape works (`--custom-dataset-type weka_trace`, no explicit
`--fixed-schedule` needed — it auto-enables for trace datasets and passing it
explicitly triggers an unrelated validator error) but not precise enough on
its own to catch the fragmentation issue.
