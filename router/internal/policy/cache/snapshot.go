package cache

import (
	"sort"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/viz"
)

// snapshot builds a viz.Snapshot for the live KV block map at /router-viz —
// see Policy.Snapshot / ThresholdPolicy.Snapshot, the two thin exported
// entry points that both forward here, since the map of per-backend tries
// is the one thing they share (trieStore) and the walk is identical either
// way.
//
// Lock discipline matters here specifically because this runs on every
// browser poll (~1/sec): trieStore's own lock is held only long enough to
// copy out the url->trie and url->backend maps (a cheap map copy, not a
// walk), then released BEFORE any trie is walked. Each kvcache.Trie.Chains
// call below takes that ONE trie's own lock for the duration of ITS walk,
// never trieStore's — so a slow walk on one backend's trie can never block
// AddBackend/DropBackend/Commit on any other backend, and nothing is held
// while the caller (viz.DataHandler) JSON-encodes the result.
func (s *trieStore) snapshot(limit int) viz.Snapshot {
	s.mu.RLock()
	urls := make([]string, 0, len(s.tries))
	tries := make(map[string]*kvcache.Trie, len(s.tries))
	backends := make(map[string]bool, len(s.backends)) // healthy, by URL
	inflight := make(map[string]int64, len(s.backends))
	for url, t := range s.tries {
		urls = append(urls, url)
		tries[url] = t
	}
	for url, b := range s.backends {
		backends[url] = b.Available()
		inflight[url] = b.Inflight()
	}
	s.mu.RUnlock()
	sort.Strings(urls) // stable render: same backend order on every poll

	// Walk each backend's trie independently (each call takes only that
	// trie's own RLock, per the doc above) and remember which hashes it
	// holds, so the alignment pass below can look up presence by hash
	// without re-walking.
	chainsByURL := make(map[string][]kvcache.Chain, len(urls))
	presentByURL := make(map[string]map[uint64]bool, len(urls))
	chainsShown, chainsTotal := 0, 0
	for _, url := range urls {
		chains, total := tries[url].Chains(limit)
		present := make(map[uint64]bool, len(chains)*2)
		for _, c := range chains {
			for _, h := range c.Hashes {
				present[h] = true
			}
		}
		chainsByURL[url] = chains
		presentByURL[url] = present
		chainsShown += len(chains)
		chainsTotal += total
	}

	// Build the GLOBAL column order: walk every backend's chains, in the
	// same stable URL order, assigning a new column the first time a hash
	// is seen and reusing it every later time the same hash turns up (same
	// backend's other chains, or a different backend's chains). Because a
	// chain's hashes are appended in position order, one session's columns
	// are always contiguous; because a repeat reuses its existing column,
	// the SAME block on multiple backends lands in the SAME column on every
	// row — which is what makes duplication visible at a glance.
	colIndex := make(map[uint64]int)
	var colHashes []uint64
	var blocks []viz.BlockInfo
	chainID := 0
	for _, url := range urls {
		for _, c := range chainsByURL[url] {
			chainID++
			for pos, h := range c.Hashes {
				if _, seen := colIndex[h]; seen {
					continue
				}
				colIndex[h] = len(colHashes)
				colHashes = append(colHashes, h)
				var tok int32
				if pos < len(c.Tokens) {
					tok = c.Tokens[pos]
				}
				blocks = append(blocks, viz.BlockInfo{
					Hash: kvcache.HexHash(h), ChainID: chainID, Pos: pos, Tokens: tok,
				})
			}
		}
	}

	out := make([]viz.BackendBlocks, 0, len(urls))
	var copiesSum int64
	for _, url := range urls {
		present := make([]bool, len(colHashes))
		have := presentByURL[url]
		for i, h := range colHashes {
			if have[h] {
				present[i] = true
				copiesSum++
			}
		}
		nodes, tokens, _ := tries[url].Stats()
		bb := viz.BackendBlocks{URL: url, Nodes: nodes, Tokens: tokens, Present: present}
		if healthy, ok := backends[url]; ok {
			h := healthy
			bb.Healthy = &h
			bb.Inflight = inflight[url]
		}
		out = append(out, bb)
	}

	var avgCopies float64
	if len(colHashes) > 0 {
		avgCopies = float64(copiesSum) / float64(len(colHashes))
	}

	return viz.Snapshot{
		//clockexempt: viz snapshot timestamp, purely informational, not a routing/timing decision
		GeneratedAt:  time.Now(),
		PolicyActive: true,
		Blocks:       blocks,
		Backends:     out,
		AvgCopies:    avgCopies,
		ChainsShown:  chainsShown,
		ChainsTotal:  chainsTotal,
		Truncated:    chainsShown < chainsTotal,
	}
}
