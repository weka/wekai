# Live KV Map (`/router-viz`)

A live, read-only visualization of what each backend's prefix cache currently holds —
served on the router's `--metrics-listen` address (never the inference listener), so it
shares the "internal-only" posture of `/metrics`.

- `GET /router-viz` — the page (self-contained HTML+JS, no external CDNs, polls ~1/s).
- `GET /router-viz/data` — the JSON it polls (`router/internal/viz.Snapshot`).

## What it shows

A single MERGED prefix tree across every backend — not a per-backend list. A prefix
shared by more than one backend appears ONCE, as a common-ancestor box, with sessions
branching below it wherever content actually diverges (radix-compressed: a run of
blocks that neither branches nor changes backend-presence collapses to one row).

Per-node badges, both counted in REAL blocks, not compressed rows:
- **`⊂N`** — total blocks in this row's subtree, including itself.
- **`dN`** — depth in blocks from the root through the END of this row (a 12-block
  compressed run advances depth by 12 in one step, not 1 per row).

Per-backend colored squares on the right of each box show presence. `avg_copies` is
the fleet's duplication metric (mean backends holding each distinct block, computed
before display compression) — target is CLOSE to 1.0 (~105%): a cache-aware router
should keep most content on one backend, not scatter it.

## Configuration is the page's UI controls, not query params

`?limit=`/`?max_children=`/`?max_depth=` exist on `/router-viz/data` but are the
mechanism the page's own UI inputs use, not the primary interface — default (no params
at all) is the FULL tree, unlimited in every dimension. An explicit huge value is
clamped (`viz.MaxParamValue`), not passed through unbounded.

## When there's no tree to show

`policy_active: false` in the JSON — the page shows a full-width banner in place of
the tiles/tree panel rather than an empty tree. With one routing flow this is now only
reachable in tests and in the nil-DataSource case; a shipped router always has a
tree.

## Code

`router/internal/viz/` — `viz.go` (types, handlers), `page.html` (embedded via
`embed.go`), `router/internal/policy/affinity/snapshot.go` (the `viz.DataSource`
implementation). The shared tree is natively the shape the page wants, so there is no
per-poll merge step — the second implementation that used to walk N per-backend tries
went with `policy/cache`.
