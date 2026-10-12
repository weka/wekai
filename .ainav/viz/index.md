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
- Layout: one row group per concurrency SLOT (`r`, ordered by first request; the worker
  lane from `withSlot` in slot.go, recorded as `slot`/`slot_hot`; `series_num` is the
  SESSION the slot runs and only groups rows for old data without `slot`). A flag bit
  marks the first request of each session on a slot (boundary line), overlapping
  requests of a series packed greedily into sub-rows (`u`, `subRowPacker`; live
  assigns at begin and keeps the row on completion). Instance = series_guid
  (`session:instance`), shown in the tooltip. `/api/data?arm=` accepts index or name.
- Waits: `ready_ms`/`sent_ms` (unix ms) are stamped per request (router replay: due time
  or predecessor end -> gate opened, `eligibleAt`; synthetic/hermes: loop top -> gate
  opened). Client wait leads the bar (`cw`, included in `e`), then engine queue other,
  engine queue KV, unattributed (client e2e minus engine phases, never negative).
  Live in-flight requests show client wait growing until `liveArm.sent`.
- State line: live counts from `liveArm.state()` (in flight > waiting > think > idle;
  admitted-but-unstarted slots are idle) plus `admissionGate` (stat/window/limit/open/
  next tick, from the same ttftWindow the governor uses), in `/api/data.state` and
  `/api/state`; offline reconstructs it at the view's right edge. Think pauses are
  `think_pause_ms` per request (`tp`), shaded just left of the bar. `lateTolerance`
  (50ms) keeps timer jitter out of the realtime late counts.
- Perf: completed bars are painted into 512px tiles cached per view (scroll = blits),
  narrow bars collapse into runs, per-group requests are start-sorted for window
  search, wheel zoom previews the scaled old tiles and repaints after 150ms, hover is
  an overlay canvas. `?perf=1` on a standalone page prints timings; `?highlight_row=N`,
  `?scroll_row=N`, `?mouse=X,Y`, `?merge=1` pre-set state for headless screenshots.
- Rendering is virtual-scrolled on one canvas, so ~70k requests cost one screenful.
