// Package viz serves a live, read-only visualization of what each backend's
// prefix cache currently holds — the "green squares" KV block map anton
// asked for, on the metrics listener at /router-viz (page) and
// /router-viz/data (the JSON it polls).
//
// This package knows nothing about kvcache.Trie or the cache-aware
// policies: it depends only on the narrow DataSource interface, so wiring
// (router/cmd/wllm-router/main.go) can pass whichever concrete policy is
// active without viz importing router/internal/policy/cache, and a router
// run with a non-cache policy (round-robin, least-outstanding) can still
// serve the page — DataSource nil just means "no cache policy active",
// reported in the JSON rather than a broken page.
package viz

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

// DataSource is implemented by a cache-aware policy (see
// router/internal/policy/cache.Policy / ThresholdPolicy). Snapshot must be
// cheap and lock-safe to call on every poll (~1/sec): an implementation
// should copy out under its own locks and return before this package ever
// touches the result, so no lock is ever held while JSON is being encoded.
type DataSource interface {
	// Snapshot reports the full merged prefix tree by default (every field of
	// opts left at its zero value means unlimited) — anton's explicit ask:
	// the default view is everything, and any limiting is something the user
	// opts into (via the page's UI controls, which DataHandler turns into
	// these options; there is no default cap to opt out of).
	Snapshot(opts SnapshotOptions) Snapshot
}

// SnapshotOptions bounds a Snapshot. Every field's zero value means
// unlimited — this is a caller-requested reduction from the full tree, not
// a default the caller has to override.
type SnapshotOptions struct {
	MaxRows     int // total tree rows returned; 0 = unlimited
	MaxChildren int // children shown per node; 0 = unlimited
	MaxDepth    int // deepest row depth shown (root is depth 0); 0 = unlimited
}

// Snapshot is one point-in-time view of the fleet's prefix-cache state, as a
// single merged prefix TREE across every backend — not a per-backend list.
// A shared prefix (e.g. a common system prompt) appears ONCE, as one common
// ancestor row, with sessions diverging into separate branches below it,
// mirroring the reference kv-router-sim.html's radix-compressed tree view.
type Snapshot struct {
	GeneratedAt time.Time `json:"generated_at"`
	// PolicyActive is false when there is no cache-aware policy running (a
	// round-robin/least-outstanding router has nothing to visualize here).
	PolicyActive bool `json:"policy_active"`

	// Backends is the header row: identity and health/load for each known
	// backend, sorted by URL for a stable render. Every TreeNode.Present
	// slice is aligned to this same order.
	Backends []BackendMeta `json:"backends"`

	// Tree is a FLAT, parent-indexed view of the merged prefix tree — one
	// entry per compressed run of blocks (a maximal chain with no branch
	// and no change in which backends hold it), analogous to the
	// reference's buildView() output. Parent==-1 marks a root. Rendering
	// (the y-position/layout pass) happens client-side from this array, but
	// the compression and truncation happen here so the wire payload stays
	// small regardless of how large the underlying trie is.
	Tree []TreeNode `json:"tree"`

	// AvgCopies is the mean number of backends holding each distinct BLOCK
	// (computed before run-compression, so it reflects true block-level
	// duplication regardless of how the tree is compressed/capped for
	// display) — anton's duplication metric: the target direction is close
	// to 1.0 (e.g. ~1.05 = 105%), meaning the fleet is NOT scattering the
	// same content across many backends.
	AvgCopies float64 `json:"avg_copies"`

	// NodesShown/NodesTotal/Truncated report how much of the fleet's true
	// tree this snapshot actually renders — capping never happens silently
	// (Truncated=true means NodesShown < NodesTotal).
	NodesShown int  `json:"nodes_shown"`
	NodesTotal int  `json:"nodes_total"`
	Truncated  bool `json:"truncated"`
}

// TreeNode is one row of the compressed prefix tree: a run of RunLen
// consecutive blocks that neither branches nor changes which backends hold
// it, shown as one line the way the reference collapses a shared prefix
// into a single box rather than one row per block.
type TreeNode struct {
	// Hash is the run's FIRST block's hash (hex, kvcache.HexHash) — a stable
	// display identity; there is no human label since real traffic has no
	// simulator-assigned tags.
	Hash   string `json:"hash"`
	RunLen int    `json:"run_len"`
	Tokens int32  `json:"tokens"` // total estimated tokens across the run
	// SubtreeBlocks is the total REAL blocks (not compressed rows) in this
	// run's subtree, including its own RunLen — a long shared prefix that
	// compresses to one row still counts every block it represents, and a
	// row further down the tree adds its own descendants' blocks on top.
	SubtreeBlocks int `json:"subtree_blocks"`
	// BlockDepth is the number of REAL blocks from the root through the END
	// of this run, inclusive — a 12-block compressed run advances depth by
	// 12 in one row, not one per block. Distinct from Depth below, which
	// counts TREE ROWS (for layout), not blocks.
	BlockDepth int   `json:"block_depth"`
	Depth      int   `json:"depth"`
	Parent     int   `json:"parent"`   // index into Tree, -1 for a root
	Children   []int `json:"children"` // indices into Tree
	// Present is aligned to Snapshot.Backends: Present[i] is true iff
	// Backends[i] currently holds this run.
	Present []bool `json:"present"`
}

// BackendMeta is one backend's identity and best-effort health/load.
type BackendMeta struct {
	URL string `json:"url"`
	// Healthy is nil when the backend isn't reachable from the data source
	// (best-effort: a trie can be lazily created before the registry's add
	// hook has run) — distinct from a known-false healthy backend.
	Healthy  *bool `json:"healthy,omitempty"`
	Inflight int64 `json:"inflight"`
	// Nodes/Tokens are the backend's TRUE totals (kvcache.Trie.Stats), which
	// may exceed what the (possibly capped) Tree shows for it.
	Nodes  int64 `json:"nodes"`
	Tokens int64 `json:"tokens"`
}

// MaxParamValue is the hard ceiling an EXPLICITLY given ?limit=/
// ?max_children=/?max_depth= is clamped to — a safety bound against an
// absurd request, not a default: omitting a param entirely still means
// unlimited, not "clamped to this ceiling."
const MaxParamValue = 100_000

// DataHandler serves /router-viz/data: a JSON Snapshot from ds. By default —
// no query params at all — the response is the FULL tree, unlimited: the
// page's own UI controls are what set ?limit=/?max_children=/?max_depth=
// when a user explicitly asks to reduce the view; these are the mechanism
// the UI uses, not a surface an operator is expected to construct by hand.
// ds may be nil (no cache-aware policy active): the handler still returns
// 200 with PolicyActive:false rather than erroring, so the page always has
// something sensible to render.
func DataHandler(ds DataSource) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		opts := SnapshotOptions{
			MaxRows:     positiveIntParam(r, "limit"),
			MaxChildren: positiveIntParam(r, "max_children"),
			MaxDepth:    positiveIntParam(r, "max_depth"),
		}

		var snap Snapshot
		if ds != nil {
			snap = ds.Snapshot(opts)
		} else {
			//clockexempt: informational snapshot timestamp when no policy is active, not a routing/timing decision
			snap = Snapshot{GeneratedAt: time.Now(), PolicyActive: false}
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(snap)
	}
}

// positiveIntParam parses query param name as a positive int, clamped to
// MaxParamValue. An absent, malformed, zero, or negative value returns 0
// (unlimited) — there is no non-zero default to fall back to.
func positiveIntParam(r *http.Request, name string) int {
	q := r.URL.Query().Get(name)
	if q == "" {
		return 0
	}
	n, err := strconv.Atoi(q)
	if err != nil || n <= 0 {
		return 0
	}
	if n > MaxParamValue {
		return MaxParamValue
	}
	return n
}

// PageHandler serves /router-viz: the self-contained HTML+JS page (embedded
// at build time from page.html, no external CDNs) that polls
// /router-viz/data (relative to its own URL, so it works unmodified behind
// any path prefix) roughly once a second and re-renders.
func PageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write([]byte(pageHTML))
	}
}
