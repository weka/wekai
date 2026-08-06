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
	// Snapshot reports what each known backend currently holds, capped to at
	// most limit chains PER BACKEND (0 or negative = unlimited).
	Snapshot(limit int) Snapshot
}

// Snapshot is one point-in-time view of the fleet's prefix-cache state.
type Snapshot struct {
	GeneratedAt time.Time `json:"generated_at"`
	// PolicyActive is false when there is no cache-aware policy running (a
	// round-robin/least-outstanding router has nothing to visualize here).
	PolicyActive bool `json:"policy_active"`

	// Blocks is the GLOBAL column order every backend's Present slice is
	// aligned against: index i in every backend's Present corresponds to
	// Blocks[i]. Ordered so that one chain's blocks are always contiguous
	// (grouped by prefix-chain/tree order), which is what lets a session
	// read as one visual run rather than scattered columns.
	Blocks []BlockInfo `json:"blocks"`
	// Backends is sorted by URL for a stable, non-flickering render.
	Backends []BackendBlocks `json:"backends"`

	// AvgCopies is the mean number of backends holding each distinct block
	// in Blocks — anton's duplication metric (~1.0 = essentially no
	// duplication across the fleet; the target direction is close to 1.0,
	// e.g. ~1.05 = 105%, not scattered across many nodes).
	AvgCopies float64 `json:"avg_copies"`

	// ChainsShown/ChainsTotal/Truncated report how much of the fleet's true
	// state this snapshot actually reflects — capping never happens
	// silently (Truncated=true means ChainsShown < ChainsTotal).
	ChainsShown int  `json:"chains_shown"`
	ChainsTotal int  `json:"chains_total"`
	Truncated   bool `json:"truncated"`
}

// BlockInfo is one column: a single block's identity and where it sits
// within its session, for grouping/shading contiguous runs in the UI.
type BlockInfo struct {
	Hash    string `json:"hash"` // hex, kvcache.HexHash — for display/debugging only
	ChainID int    `json:"chain_id"`
	Pos     int    `json:"pos"`
	Tokens  int32  `json:"tokens"`
}

// BackendBlocks is one row: a backend's identity, health/load (best-effort —
// omitted fields default to their zero value when unavailable), and which
// columns of Snapshot.Blocks it currently holds.
type BackendBlocks struct {
	URL string `json:"url"`
	// Healthy is nil when the backend isn't reachable from the data source
	// (best-effort: a trie can be lazily created before the registry's add
	// hook has run) — distinct from a known-false healthy backend.
	Healthy  *bool `json:"healthy,omitempty"`
	Inflight int64 `json:"inflight"`
	// Nodes/Tokens are the backend's TRUE totals (kvcache.Trie.Stats), which
	// may exceed len(Present) when chains were capped — the per-backend
	// analog of Snapshot.Truncated.
	Nodes  int64 `json:"nodes"`
	Tokens int64 `json:"tokens"`
	// Present is parallel to Snapshot.Blocks: Present[i] is true iff this
	// backend currently holds Blocks[i].
	Present []bool `json:"present"`
}

// DefaultChainLimit bounds chains-per-backend when a request omits ?limit=,
// keeping the default poll cheap regardless of trie size.
const DefaultChainLimit = 60

// MaxChainLimit is the hard ceiling ?limit= is clamped to, so a client can
// ask for more detail without being able to force an unbounded walk.
const MaxChainLimit = 2000

// DataHandler serves /router-viz/data: a JSON Snapshot from ds, honoring an
// optional ?limit= query param (chains per backend). ds may be nil (no
// cache-aware policy active): the handler still returns 200 with
// PolicyActive:false rather than erroring, so the page always has something
// sensible to render.
func DataHandler(ds DataSource) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		limit := DefaultChainLimit
		if q := r.URL.Query().Get("limit"); q != "" {
			if n, err := strconv.Atoi(q); err == nil && n > 0 {
				limit = n
				if limit > MaxChainLimit {
					limit = MaxChainLimit
				}
			}
		}

		var snap Snapshot
		if ds != nil {
			snap = ds.Snapshot(limit)
		} else {
			//clockexempt: informational snapshot timestamp when no policy is active, not a routing/timing decision
			snap = Snapshot{GeneratedAt: time.Now(), PolicyActive: false}
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(snap)
	}
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
