package cache

import (
	"sort"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/viz"
)

// snapshot builds a viz.Snapshot for the live KV prefix TREE at
// /router-viz — see Policy.Snapshot / ThresholdPolicy.Snapshot, the two
// thin exported entry points that both forward here, since the map of
// per-backend tries is the one thing they share (trieStore) and the walk
// is identical either way.
//
// Lock discipline matters here specifically because this runs on every
// browser poll (~1/sec): trieStore's own lock is held only long enough to
// copy out the url->trie and url->backend maps (a cheap map copy, not a
// walk), then released BEFORE any trie is walked. Each kvcache.Trie.Chains
// call below takes that ONE trie's own lock for the duration of ITS walk,
// never trieStore's — so a slow walk on one backend's trie can never block
// AddBackend/DropBackend/Commit on any other backend, and nothing is held
// while the tree gets merged/compressed/flattened or while the caller
// (viz.DataHandler) JSON-encodes the result — all of that happens on plain
// Go values with no lock involved.
func (s *trieStore) snapshot(opts viz.SnapshotOptions) viz.Snapshot {
	s.mu.RLock()
	urls := make([]string, 0, len(s.tries))
	tries := make(map[string]*kvcache.Trie, len(s.tries))
	healthy := make(map[string]bool, len(s.backends))
	inflight := make(map[string]int64, len(s.backends))
	for url, t := range s.tries {
		urls = append(urls, url)
		tries[url] = t
	}
	for url, b := range s.backends {
		healthy[url] = b.Available()
		inflight[url] = b.Inflight()
	}
	s.mu.RUnlock()
	sort.Strings(urls) // stable render: same backend order on every poll

	// Fetch each backend's chains independently (each call takes only that
	// trie's own RLock, per the doc above), UNLIMITED — deliberately not
	// opts, which governs only the DISPLAY reduction below. kvcache.Trie.Chains
	// always walks its ENTIRE trie to compute an accurate totalLeaves
	// regardless of the limit it's given — only the returned slice is capped
	// — so tying this fetch to the caller's (possibly small) display options
	// would silently under-report NodesTotal once run-compression is in the
	// mix (fewer chains fetched == fewer blocks merged == a smaller, wrong
	// "total"). NodesTotal must reflect the fleet's true state regardless of
	// what the user asked to see, so the fetch itself is never capped.
	chainsByURL := make(map[string][]kvcache.Chain, len(urls))
	for _, url := range urls {
		chains, _ := tries[url].Chains(0)
		chainsByURL[url] = chains
	}

	// Merge every backend's chains into ONE shared prefix trie: a session
	// shared by two backends becomes a single mergeNode with both URLs in
	// its present set, which is the entire mechanism behind "a shared
	// prefix appears once as a common ancestor, sessions diverge below it."
	mergeRoot, avgCopies := buildMergeTree(chainsByURL, urls)

	// Compress into runs (radix-style: a maximal non-branching,
	// presence-homogeneous chain becomes one row) and compute each root's
	// true subtree size before any capping, so ordering/truncation
	// decisions are based on the FULL tree, not an already-cut one.
	var roots []*run
	for _, c := range sortedChildren(mergeRoot) {
		r := compressFrom(c)
		r.computeSubtree()
		roots = append(roots, r)
	}
	nodesTotal := 0
	for _, r := range roots {
		nodesTotal += r.subtreeSize
	}

	treeNodes := flattenTree(roots, opts, urls)

	backends := make([]viz.BackendMeta, 0, len(urls))
	for _, url := range urls {
		nodes, tokens, _ := tries[url].Stats()
		bm := viz.BackendMeta{URL: url, Nodes: nodes, Tokens: tokens}
		if h, ok := healthy[url]; ok {
			hh := h
			bm.Healthy = &hh
			bm.Inflight = inflight[url]
		}
		backends = append(backends, bm)
	}

	return viz.Snapshot{
		//clockexempt: viz snapshot timestamp, purely informational, not a routing/timing decision
		GeneratedAt:  time.Now(),
		PolicyActive: true,
		Backends:     backends,
		Tree:         treeNodes,
		AvgCopies:    avgCopies,
		NodesShown:   len(treeNodes),
		NodesTotal:   nodesTotal,
		Truncated:    len(treeNodes) < nodesTotal,
	}
}
