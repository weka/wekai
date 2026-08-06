package cache

import (
	"sort"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/viz"
)

// maxChildrenPerNode bounds how many branches a single tree row shows before
// the rest are simply not rendered (Snapshot.Truncated covers the "how much
// was left out" reporting at the whole-tree level) — the same idea as the
// reference kv-router-sim.html capping children per node, kept simpler here
// (no per-node "+N branches" ghost row) since the top-level NodesShown/
// NodesTotal/Truncated fields already say how much is missing.
const maxChildrenPerNode = 6

// fetchCapPerBackend bounds how many chains are fetched from each backend's
// trie when building the merge tree — see the call site in snapshot.go for
// why this is deliberately NOT the same knob as the caller's display limit.
const fetchCapPerBackend = 5000

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
	hashes      []uint64
	tokens      []int32
	present     map[string]bool
	children    []*run
	subtreeSize int // this run plus every run in its subtree; filled bottom-up for ordering
}

// compressFrom walks n and its single-child, present-homogeneous
// descendants into one run, then starts a fresh run at each remaining
// branch point (children returned in sorted-by-hash order, for a
// deterministic tree that doesn't reshuffle between polls when nothing
// changed).
func compressFrom(n *mergeNode) *run {
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
	for _, c := range sortedChildren(cur) {
		r.children = append(r.children, compressFrom(c))
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

func (r *run) computeSubtreeSize() int {
	size := 1
	for _, c := range r.children {
		size += c.computeSubtreeSize()
	}
	r.subtreeSize = size
	return size
}

// flattenTree turns root runs into the flat, parent-indexed, capped
// viz.TreeNode array the page renders. When a node has more children than
// maxChildrenPerNode, or the overall limit is reached, the rest are simply
// not emitted — which branch survives is decided by descending subtree
// size (the "busiest"/biggest branch first), the closest available proxy
// to the reference's traffic-sorted children since this package has no
// per-block hit counter to sort by.
func flattenTree(roots []*run, limit int, backends []string) []viz.TreeNode {
	sort.Slice(roots, func(i, j int) bool { return roots[i].subtreeSize > roots[j].subtreeSize })

	var nodes []viz.TreeNode
	var emit func(r *run, depth, parentIdx int)
	emit = func(r *run, depth, parentIdx int) {
		if limit > 0 && len(nodes) >= limit {
			return
		}
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
			Depth: depth, Parent: parentIdx, Present: present,
		})

		kids := append([]*run(nil), r.children...)
		sort.Slice(kids, func(i, j int) bool { return kids[i].subtreeSize > kids[j].subtreeSize })
		shown := 0
		for _, k := range kids {
			if shown >= maxChildrenPerNode || (limit > 0 && len(nodes) >= limit) {
				break
			}
			childIdx := len(nodes)
			nodes[idx].Children = append(nodes[idx].Children, childIdx)
			emit(k, depth+1, idx)
			shown++
		}
	}
	for _, r := range roots {
		if limit > 0 && len(nodes) >= limit {
			break
		}
		emit(r, 0, -1)
	}
	return nodes
}
