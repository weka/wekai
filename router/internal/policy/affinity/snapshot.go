package affinity

import (
	"sort"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/viz"
)

// Snapshot implements viz.DataSource for the live KV map at /router-viz.
//
// This tree needs far less work than the older policies' does. There, each
// backend owns a private trie and every poll rebuilds a merged cross-backend
// tree from N of them (buildMergeTree) purely so the page can show a shared
// prefix once with a presence set. Here the shared marked tree IS that
// structure: a run's markSet expands directly into TreeNode.Present, and runs
// are already radix-compressed. All that remains is a display pass — merging
// adjacent rows that carry the same holders, sizing subtrees, and capping.
//
// Lock discipline follows the older implementation: shard locks are held only
// long enough to copy the structure out into plain Go values, and nothing is
// held while the result is sized, capped, or JSON-encoded by viz.
func (p *Policy) Snapshot(opts viz.SnapshotOptions) viz.Snapshot {
	//clockexempt: a display timestamp, not a decision; the tree's own TTL uses the clock.
	snap := viz.Snapshot{GeneratedAt: time.Now(), PolicyActive: true}

	p.mu.RLock()
	backends := make([]*registry.Backend, 0, len(p.backends))
	for _, b := range p.backends {
		backends = append(backends, b)
	}
	p.mu.RUnlock()
	sort.Slice(backends, func(i, j int) bool { return backends[i].URL < backends[j].URL })

	// Slot per backend, index-aligned to the sorted list Present must match.
	slots := make([]int, len(backends))
	for i, b := range backends {
		slots[i] = p.tree.slotOrCreate(b.URL)
	}

	roots := p.tree.copyOut()
	per := p.tree.perBackend()

	snap.Backends = make([]viz.BackendMeta, len(backends))
	for i, b := range backends {
		healthy := b.Available()
		bs := per[b.URL]
		snap.Backends[i] = viz.BackendMeta{
			URL:      b.URL,
			Healthy:  &healthy,
			Inflight: b.Inflight(),
			Nodes:    bs.Runs,
			Tokens:   bs.Tokens,
		}
	}

	total := 0
	for _, r := range roots {
		rows, _ := r.computeSubtree()
		total += rows
	}
	snap.Tree = flattenTree(roots, opts, slots)
	snap.NodesShown = len(snap.Tree)
	snap.NodesTotal = total
	snap.Truncated = snap.NodesShown < snap.NodesTotal
	snap.AvgCopies = p.tree.stats().AvgCopies
	return snap
}

// vizRun is a lock-free copy of one display row.
type vizRun struct {
	hashes []uint64
	tokens []int32
	marks  markSet
	kids   []*vizRun

	blockDepth    int
	subtreeSize   int
	subtreeBlocks int
}

// copyOut lifts every shard's forest into plain values, merging adjacent runs
// that carry identical holders.
//
// The merge is a display concern, exactly as in the older policy. The tree
// itself can legitimately hold a parent and its only child with the same holder
// set — splitting a run leaves both halves marked identically, and if the branch
// that forced the split later expires there is nothing that re-joins them.
// Showing that as two rows would be noise.
func (t *tree) copyOut() []*vizRun {
	var out []*vizRun
	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.RLock()
		for _, root := range sh.roots {
			for _, k := range root.kids {
				out = append(out, copyRun(k, 0))
			}
		}
		sh.mu.RUnlock()
	}
	return out
}

func copyRun(r *run, parentBlockDepth int) *vizRun {
	v := &vizRun{marks: r.marks.Clone()}
	cur := r
	for {
		v.hashes = append(v.hashes, cur.hashes...)
		v.tokens = append(v.tokens, cur.tokens...)
		if len(cur.kids) != 1 {
			break
		}
		only := cur.kids[0]
		if !sameHolders(only.marks, cur.marks) {
			break
		}
		cur = only
	}
	v.blockDepth = parentBlockDepth + len(v.hashes)
	for _, k := range cur.kids {
		v.kids = append(v.kids, copyRun(k, v.blockDepth))
	}
	sort.Slice(v.kids, func(i, j int) bool { return v.kids[i].hashes[0] < v.kids[j].hashes[0] })
	return v
}

func sameHolders(a, b markSet) bool { return a.Subset(b) && b.Subset(a) }

// computeSubtree fills row and block totals bottom-up, before any capping, so
// a capped view still reports the true size of what it elided.
func (v *vizRun) computeSubtree() (rows, blocks int) {
	rows, blocks = 1, len(v.hashes)
	for _, k := range v.kids {
		kr, kb := k.computeSubtree()
		rows += kr
		blocks += kb
	}
	v.subtreeSize, v.subtreeBlocks = rows, blocks
	return rows, blocks
}

// flattenTree emits the flat, parent-indexed array the page renders. A zero
// opts means unlimited — capping is something the caller opts into, never a
// default. When a cap does bite, the surviving branch is the one with the
// larger subtree.
func flattenTree(roots []*vizRun, opts viz.SnapshotOptions, slots []int) []viz.TreeNode {
	sort.Slice(roots, func(i, j int) bool { return roots[i].subtreeSize > roots[j].subtreeSize })

	var nodes []viz.TreeNode
	fitsRows := func() bool { return opts.MaxRows <= 0 || len(nodes) < opts.MaxRows }

	var emit func(v *vizRun, depth, parentIdx int)
	emit = func(v *vizRun, depth, parentIdx int) {
		idx := len(nodes)
		present := make([]bool, len(slots))
		for i, s := range slots {
			present[i] = v.marks.Has(s)
		}
		var totalTok int32
		for _, tk := range v.tokens {
			totalTok += tk
		}
		nodes = append(nodes, viz.TreeNode{
			Hash: kvcache.HexHash(v.hashes[0]), RunLen: len(v.hashes), Tokens: totalTok,
			SubtreeBlocks: v.subtreeBlocks, BlockDepth: v.blockDepth,
			Depth: depth, Parent: parentIdx, Present: present,
		})

		// Checked once for the children collectively, before any child index is
		// reserved: deciding inside the loop would leave a Children entry
		// pointing at a row that never gets emitted.
		if opts.MaxDepth > 0 && depth+1 >= opts.MaxDepth {
			return
		}

		kids := append([]*vizRun(nil), v.kids...)
		sort.Slice(kids, func(i, j int) bool { return kids[i].subtreeSize > kids[j].subtreeSize })
		shown := 0
		for _, k := range kids {
			if opts.MaxChildren > 0 && shown >= opts.MaxChildren {
				break
			}
			if !fitsRows() {
				break
			}
			nodes[idx].Children = append(nodes[idx].Children, len(nodes))
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
