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

`router/internal/viz/` (types, handlers, embedded page) and
`router/internal/policy/affinity/snapshot.go` (the `viz.DataSource`). The shared tree
is natively the shape the page wants, so there is no per-poll merge step.

# Request timeline (benchmark report, standalone, live)

One renderer, three surfaces: `benchmark/timeline_assets/timeline.js` (+ `.css`,
`timeline_page.html`), embedded via `go:embed` in `benchmark/timeline_page.go`.

- Model (`benchmark/timeline.go`): `timelineSegments` turns a reqdata row into
  coloured segments in Go (wait -> prefill split by token source -> preempted ->
  decode; client-timestamp fallback). The JS only draws. `timelineKinds` is the
  legend/colour contract shipped to the page.
- Report: `visualize --timeline` / `visualize-merge --timeline` / `auto --timeline`
  embed each arm's `timeline` plus the renderer and a Metrics/Timeline switch;
  the view shares the report's `viewTMin/viewTMax` (draw() calls
  `tlSyncFromReport`). Time basis is ms since the arm's first request.
- Standalone: `wekai benchmark timeline <reqdata dir> -o file.html`.
- Live: `benchmark/live_view.go`, `benchmark auto --live-port N [--live-bind A]
  [--live-max-requests N]`. Endpoints: `/` page, `/api/arms`,
  `/api/data?arm=i&since=seq` (incremental; `reset` when the cursor fell off the
  bounded buffer), `/healthz`. Hooks (`liveArm.begin/complete/abandon`) are
  nil-safe, called where `requestDataRecord`s are written (auto.go, replay.go,
  replay_router.go). Default bind is loopback; kubectl port-forward reaches it.
- Layout: one row group per series_num (`r`, ordered by first request), overlapping
  requests of a series packed greedily into sub-rows (`u`, `subRowPacker`; live
  assigns at begin and keeps the row on completion). Instance = series_guid
  (`session:instance`), shown in the tooltip. `/api/data?arm=` accepts index or name.
- Rendering is virtual-scrolled on one canvas, so ~70k requests cost one screenful.
