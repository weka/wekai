package cache

import (
	"sort"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/viz"
)

// fetchCapPerBackend bounds how many chains are fetched from each backend's
// trie when building the merge tree. 0 (unlimited) ALWAYS — see the call
// site in snapshot.go: NodesTotal must report the fleet's true state
// regardless of what display reduction (SnapshotOptions) the caller asked
// for, so the fetch itself is never capped by the caller's own request.

// mergeNode is one exact block, built by inserting every backend's
// kvcache.Trie.Chains output into a single shared trie keyed by hash — the
// same shape a kvcache.Trie itself uses internally, rebuilt here across
// MULTIPLE backends so one tree can represent the whole fleet. A shared
// prefix (same hash chain served by more than one backend) becomes exactly
// one mergeNode with more than one entry in present, which is the whole
// mechanism behind "a shared prefix appears once as a common ancestor."
type mergeNode struct {
	hash     uint64
	tokens   int32
	children map[uint64]*mergeNode
	present  map[string]bool // backend URL -> true
}

func newMergeNode(hash uint64, tokens int32) *mergeNode {
	return &mergeNode{hash: hash, tokens: tokens, children: map[uint64]*mergeNode{}, present: map[string]bool{}}
}

// buildMergeTree inserts every backend's chains into one shared trie rooted
// at the returned virtual root (which represents no block itself). Also
// returns block-level AvgCopies, computed here — BEFORE run-compression —
// so the number reflects true per-block duplication regardless of how the
// tree gets compressed or capped for display afterward.
func buildMergeTree(chainsByURL map[string][]kvcache.Chain, urls []string) (root *mergeNode, avgCopies float64) {
	root = newMergeNode(0, 0)
	for _, url := range urls {
		for _, c := range chainsByURL[url] {
			n := root
			for i, h := range c.Hashes {
				child, ok := n.children[h]
				if !ok {
					var tok int32
					if i < len(c.Tokens) {
						tok = c.Tokens[i]
					}
					child = newMergeNode(h, tok)
					n.children[h] = child
				}
				child.present[url] = true
				n = child
			}
		}
	}

	var blocks, copiesSum int64
	var walk func(n *mergeNode)
	walk = func(n *mergeNode) {
		if n != root {
			blocks++
			copiesSum += int64(len(n.present))
		}
		for _, c := range n.children {
			walk(c)
		}
	}
	walk(root)
	if blocks > 0 {
		avgCopies = float64(copiesSum) / float64(blocks)
	}
	return root, avgCopies
}

// run is a maximal chain of merge nodes that neither branches (exactly one
// child) nor changes which backends hold it — one row in the rendered tree.
// This is the same radix compression the reference applies (a long shared
// prefix collapses to a single box instead of one row per block), extended
// with a presence-homogeneity check the reference didn't need: its "marks"
// live per-run by construction, but ours are derived from real per-block
// data, so a run may only compress through blocks that agree on which
// backends hold them — otherwise the row would show a wrong/blended
// presence pattern.
type run struct {
	hashes   []uint64
	tokens   []int32
	present  map[string]bool
	children []*run

	// blockDepth is filled TOP-DOWN, during compressFrom itself (it needs
	// the parent's depth to compute its own, the opposite direction from
	// subtreeSize/subtreeBlocks below): the number of REAL blocks from the
	// root through the END of this run, inclusive — a 12-block compressed
	// run advances depth by 12 in one row, not one per block. The UI's "d N"
	// badge: the deepest block this box represents is the Nth from root.
	blockDepth int

	// Filled bottom-up by computeSubtree, before any display capping:
	subtreeSize   int // this run plus every run (row) in its subtree — for ordering which branch survives a cap
	subtreeBlocks int // REAL blocks (sum of RunLen), not rows, this run plus its whole subtree — the UI's "⊂N" badge
}

// compressFrom walks n and its single-child, present-homogeneous
// descendants into one run, then starts a fresh run at each remaining
// branch point (children returned in sorted-by-hash order, for a
// deterministic tree that doesn't reshuffle between polls when nothing
// changed). parentBlockDepth is the parent run's own blockDepth (0 for a
// root, whose parent is the virtual, block-less merge root).
func compressFrom(n *mergeNode, parentBlockDepth int) *run {
	r := &run{present: n.present}
	cur := n
	for {
		r.hashes = append(r.hashes, cur.hash)
		r.tokens = append(r.tokens, cur.tokens)
		if len(cur.children) != 1 {
			break
		}
		var only *mergeNode
		for _, c := range cur.children {
			only = c
		}
		if !sameBackendSet(only.present, cur.present) {
			break
		}
		cur = only
	}
	r.blockDepth = parentBlockDepth + len(r.hashes)
	for _, c := range sortedChildren(cur) {
		r.children = append(r.children, compressFrom(c, r.blockDepth))
	}
	return r
}

func sameBackendSet(a, b map[string]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

func sortedChildren(n *mergeNode) []*mergeNode {
	out := make([]*mergeNode, 0, len(n.children))
	for _, c := range n.children {
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].hash < out[j].hash })
	return out
}

// computeSubtree fills subtreeSize (rows) and subtreeBlocks (real blocks,
// i.e. sum of RunLen) for r and its entire subtree, bottom-up, in one walk.
// subtreeBlocks is deliberately NOT the same number as subtreeSize: a
// single compressed row can represent many blocks (a long uncontested
// shared prefix collapses to ONE row but should still count all of its
// blocks toward the total), which is exactly the distinction anton asked
// the UI badge to make visible.
func (r *run) computeSubtree() (rows, blocks int) {
	rows, blocks = 1, len(r.hashes)
	for _, c := range r.children {
		cr, cb := c.computeSubtree()
		rows += cr
		blocks += cb
	}
	r.subtreeSize = rows
	r.subtreeBlocks = blocks
	return rows, blocks
}

// flattenTree turns root runs into the flat, parent-indexed viz.TreeNode
// array the page renders. By default (a zero opts) EVERY row is emitted —
// anton's explicit ask: unlimited unless the caller (the page's own UI
// controls, via DataHandler) opted into a smaller view. When a cap IS set
// and a node has more children than opts.MaxChildren, or the row/depth
// budget runs out, the rest are simply not emitted — which branch survives
// is decided by descending subtree size (the "busiest"/biggest branch
// first), the closest available proxy to the reference's traffic-sorted
// children since this package has no per-block hit counter to sort by.
func flattenTree(roots []*run, opts viz.SnapshotOptions, backends []string) []viz.TreeNode {
	sort.Slice(roots, func(i, j int) bool { return roots[i].subtreeSize > roots[j].subtreeSize })

	var nodes []viz.TreeNode
	fitsRows := func() bool { return opts.MaxRows <= 0 || len(nodes) < opts.MaxRows }

	var emit func(r *run, depth, parentIdx int)
	emit = func(r *run, depth, parentIdx int) {
		idx := len(nodes)
		present := make([]bool, len(backends))
		for i, url := range backends {
			present[i] = r.present[url]
		}
		var totalTok int32
		for _, t := range r.tokens {
			totalTok += t
		}
		nodes = append(nodes, viz.TreeNode{
			Hash: kvcache.HexHash(r.hashes[0]), RunLen: len(r.hashes), Tokens: totalTok,
			SubtreeBlocks: r.subtreeBlocks, BlockDepth: r.blockDepth,
			Depth: depth, Parent: parentIdx, Present: present,
		})

		// Depth cap: checked ONCE for this node's children collectively (they
		// all sit at the same depth+1), BEFORE reserving any child index —
		// deciding this inside the loop after childIdx were already recorded
		// would leave a dangling Children reference to a node never emitted.
		if opts.MaxDepth > 0 && depth+1 >= opts.MaxDepth {
			return
		}

		kids := append([]*run(nil), r.children...)
		sort.Slice(kids, func(i, j int) bool { return kids[i].subtreeSize > kids[j].subtreeSize })
		shown := 0
		for _, k := range kids {
			if opts.MaxChildren > 0 && shown >= opts.MaxChildren {
				break
			}
			if !fitsRows() {
				break
			}
			childIdx := len(nodes)
			nodes[idx].Children = append(nodes[idx].Children, childIdx)
			emit(k, depth+1, idx)
			shown++
		}
	}
	for _, r := range roots {
		if !fitsRows() {
			break
		}
		emit(r, 0, -1)
	}
	return nodes
}
