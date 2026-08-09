# Architecture Pointers

Where things live and WHY they are shaped that way. The code is the source of truth
for what they do — this file exists to save you the search and to record the reasons
that are not visible from a function body.

## `kvcache` — the shared prefix-cache trie engine

`kvcache/` is a stdlib-only leaf, fenced by `router/hack` so neither the benchmark CLI
nor the router can pull anything heavier into it.

One `kvcache.Trie` implementation serves the benchmark's offline cache simulation, the
mock vLLM's ground truth, and the benchmark's token estimator — deliberately, so
prediction and ground truth cannot drift apart by construction.

**The router does not use it.** The routing flow owns a different structure (one
shared tree with per-run holder sets, not N private tries), and it could not live in
`kvcache` anyway: it needs `clock.Clock` for TTL eviction, which the stdlib-only fence
forbids.

Hashing is chain-hashed, a real vLLM parent-hash chain analog — a match requires the
full ancestor chain, so two prompts whose second block is byte-identical but whose
first differs do not cross-credit that block.

## `router/internal/policy/affinity` — the routing flow

**There is one routing flow, not a set of policies.** `--policy` does not exist;
`policy/cache` (`prefix-cache-aware`, `prefix-cache-candidates`), `round-robin` and
`random` were deleted, and the parts worth keeping came back as signals.

| file | what it owns |
|---|---|
| `tree.go`, `markset.go` | the shared prefix tree; runs record WHICH backends hold them |
| `policy.go` | the decision ladder and the split guard |
| `signal.go` | the signals, and the seam they plug into |
| `snapshot.go` | `viz.DataSource` for `/router-viz` |
| `fleetsim_test.go` | offline fleet replay against the real flow, with ground truth |

Three ideas explain the shape, and none of them is obvious from the code:

- **Holding is a property of the tree.** One tree whose runs carry a holder set,
  rather than N per-backend tries queried and compared. "Who holds this, and how many"
  is then a lookup, not a reconstruction.
- **A signal never routes.** It answers only "can this backend take more work, and if
  not, what in-flight level counts as *as loaded as it*" — the second half being what
  the split guard measures against. This is why enabling a signal cannot change the
  ladder, only the set it runs over.
- **Capacity is judged in exactly one place.** `gateway.Server.candidates()` filters
  for health alone. It used to apply a concurrency cap too, which split "is this
  backend full" across two components that could disagree — a router-side guess in
  front of a policy that re-derived the same thing and could not tell "saturated" from
  "does not exist".

`least-outstanding` survives as the flow's selector — the tie-break among equals — not
as a policy.

For the design record, the measurements behind it and the decisions that were reversed,
see [../../router/docs/cache-affinity-redesign.md](../../router/docs/cache-affinity-redesign.md).
For running it, see [../router-testing/](../router-testing/index.md).

## `router/internal/mockvllm` — the mock vLLM engine

A GPU-less stand-in that speaks the vLLM wire format, so router policy work needs no
hardware. Two modelling choices are load-bearing and easy to miss:

- `Admit()` reserves a concurrency slot AND pins the request's blocks in the same
  step, mirroring a real vLLM scheduler admitting a sequence and reference-counting
  its KV together. A rejected request touches no cache state at all.
- Decode KV is modelled: generated tokens become cached blocks appended to the
  sequence's own chain, occupying capacity and evicting like prompt blocks. Without
  this the mock's cache pressure is unrealistically low.

`tokenize.go` reimplements chunking locally so `--chars-per-token` can be calibrated
independently of `kvcache`'s fixed default — see
[../router-testing/calibration.md](../router-testing/calibration.md).

## `router/hack` — architectural fence tests

Mechanical invariant checks that run as ordinary Go tests, because the defects that
motivated the router rewrite were invariants nobody enforced rather than subtle
algorithms. They cover: only `lease` mutates in-flight; time-based DECISIONS go
through `clock.Clock`; no declared metric goes unemitted; core packages never import a
dialect; exactly one auth comparison site; no `os.Args` outside `main`.

Read the test names in `router/hack/` for the current list — they are written to be
self-explaining, and each failure message states the defect it is guarding against.
They run in the ordinary `go test ./router/...` sweep, not as a separate step.
