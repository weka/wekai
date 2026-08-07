package affinity

import (
	"sync"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/clock"
)

// numShards partitions the forest by first block hash.
//
// What this buys and what it does not: a request's whole path lives in exactly
// one shard (a walk always starts at the root child keyed by block 0), so
// sharding is lossless and unrelated prefix families — different models,
// different system prompts, one-off sessions — stop contending. It does NOT
// spread a single dominant prefix: if 68% of traffic opens with the same system
// prompt, 68% of it lands in one shard by construction. That case is carried by
// the RWMutex and by keeping the critical sections short (walk is read-only;
// commit touches one path and returns), not by the sharding. The benchmark, not
// this comment, is the check.
const numShards = 16

// run is one radix-compressed segment: a maximal chain of blocks with no
// branch, plus the set of backends known to hold it.
type run struct {
	hashes []uint64
	tokens []int32

	parent *run
	// kids is sorted by kids[i].hashes[0] and binary-searched. A sorted slice
	// rather than a map for the reason kvcache gives at kvcache.go:188 — this
	// is the dominant allocation and a map costs ~2.5x the bytes per node.
	kids  []*run
	marks markSet

	lastUsed time.Time
	traffic  uint64
}

// blocks is the run's length in blocks.
func (r *run) blocks() int { return len(r.hashes) }

// find returns the child beginning with h, or nil.
func (r *run) find(h uint64) *run {
	lo, hi := 0, len(r.kids)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if r.kids[mid].hashes[0] < h {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(r.kids) && r.kids[lo].hashes[0] == h {
		return r.kids[lo]
	}
	return nil
}

// link inserts c into r.kids, keeping it sorted.
func (r *run) link(c *run) {
	h := c.hashes[0]
	lo, hi := 0, len(r.kids)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if r.kids[mid].hashes[0] < h {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	r.kids = append(r.kids, nil)
	copy(r.kids[lo+1:], r.kids[lo:])
	r.kids[lo] = c
	c.parent = r
}

// unlink removes the child beginning with h.
func (r *run) unlink(h uint64) {
	lo, hi := 0, len(r.kids)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if r.kids[mid].hashes[0] < h {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(r.kids) && r.kids[lo].hashes[0] == h {
		r.kids = append(r.kids[:lo], r.kids[lo+1:]...)
	}
}

// path is a request's block sequence, without materializing a slice.
//
// modelKey selects the root the walk starts from, so two models can never share
// a run: matching requires walking from a root, and they do not share one. The
// gateway filters candidates by DialectID and never by Model, so without this a
// router fronting two models on one dialect would credit one model's KV cache
// for the other's prompt. Keying by root rather than by folding the model into
// block 0's hash keeps every stored hash equal to the real block hash, which
// /router-viz renders directly.
type path struct {
	units    []kvcache.Unit
	modelKey uint64
}

func (p path) len() int { return len(p.units) }

func (p path) hash(i int) uint64 { return p.units[i].Hash }

func (p path) tokens(i int) int32 { return p.units[i].Tokens }

// shardKey spreads (model, first block) over the shards. splitmix64's
// finalizer, so two models whose keys differ in few bits do not land in
// lockstep the way a bare XOR would leave them.
func (p path) shardKey() uint64 {
	h := p.units[0].Hash ^ p.modelKey
	h ^= h >> 30
	h *= 0xbf58476d1ce4e5b9
	h ^= h >> 27
	h *= 0x94d049bb133111eb
	h ^= h >> 31
	return h
}

// shard is one independently-locked sub-forest.
type shard struct {
	mu sync.RWMutex
	// roots holds one sentinel per model key. A sentinel represents no block
	// and is never marked, evicted by TTL, or reported in stats.
	roots map[uint64]*run
	tails map[*run]struct{}

	runs   int64
	blocks int64
	// copies is the running sum of blocks x holders, so AvgCopies is a
	// division rather than a tree walk on every metrics tick.
	copies  int64
	expired int64
}

// tree is the shared, cross-backend-marked prefix forest.
type tree struct {
	clk clock.Clock
	ttl time.Duration

	shards [numShards]shard

	slotMu sync.RWMutex
	slots  map[string]int
	free   []int
	nextSl int
}

func newTree(clk clock.Clock, ttl time.Duration) *tree {
	t := &tree{clk: clk, ttl: ttl, slots: map[string]int{}}
	for i := range t.shards {
		t.shards[i].tails = map[*run]struct{}{}
		t.shards[i].roots = map[uint64]*run{}
	}
	return t
}

func (t *tree) shardFor(p path) *shard {
	return &t.shards[p.shardKey()%numShards]
}

// dropRoot removes a sentinel that has lost its last child, so churning through
// model names cannot leak one root per name. Caller holds the write lock.
func (s *shard) dropRoot(r *run) {
	for k, v := range s.roots {
		if v == r {
			delete(s.roots, k)
			return
		}
	}
}

// ---------------------------------------------------------------- slots

// slotFor allocates (or returns) the bit index for a backend URL.
func (t *tree) slotFor(url string) int {
	t.slotMu.Lock()
	defer t.slotMu.Unlock()
	if s, ok := t.slots[url]; ok {
		return s
	}
	var s int
	if n := len(t.free); n > 0 {
		s, t.free = t.free[n-1], t.free[:n-1]
	} else {
		s, t.nextSl = t.nextSl, t.nextSl+1
	}
	t.slots[url] = s
	return s
}

// slot reports a backend's bit index, if it has one.
func (t *tree) slot(url string) (int, bool) {
	t.slotMu.RLock()
	defer t.slotMu.RUnlock()
	s, ok := t.slots[url]
	return s, ok
}

// slotOrCreate is the request-path accessor: a read lock in the common case,
// allocating only when a backend reaches routing before the registry's add
// hook has run (the same lazy-create the older trieStore needs).
func (t *tree) slotOrCreate(url string) int {
	if s, ok := t.slot(url); ok {
		return s
	}
	return t.slotFor(url)
}

// dropBackend clears a backend's marks from every run and only then frees its
// slot for reuse.
//
// Order matters: freeing first would let the next backend to join inherit
// every prefix the dead one held. O(tree), but it runs on backend churn (a
// rollout), never on the request path.
func (t *tree) dropBackend(url string) {
	t.slotMu.Lock()
	s, ok := t.slots[url]
	if !ok {
		t.slotMu.Unlock()
		return
	}
	delete(t.slots, url)
	t.slotMu.Unlock()

	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.Lock()
		for _, root := range sh.roots {
			sh.clearSlot(root, s)
		}
		// A run left with no holders at all can never anchor a request again.
		// Reclaiming it here rather than waiting out the TTL keeps AvgCopies
		// meaningful (a ghost block would drag the mean below 1.0) and returns
		// a departing backend's memory at rollout speed.
		sh.evictOrphans()
		sh.mu.Unlock()
	}

	t.slotMu.Lock()
	t.free = append(t.free, s)
	t.slotMu.Unlock()
}

// clearSlot removes one backend from every run in the subtree. Caller holds
// the write lock.
func (s *shard) clearSlot(r *run, slot int) {
	if r.parent != nil && r.marks.Has(slot) {
		r.marks.Remove(slot)
		s.copies -= int64(r.blocks())
	}
	for _, k := range r.kids {
		s.clearSlot(k, slot)
	}
}

// evictOrphans removes every childless run that no backend holds, cascading
// upward. Caller holds the write lock.
func (s *shard) evictOrphans() {
	for {
		var dead []*run
		for r := range s.tails {
			if r.marks.Empty() {
				dead = append(dead, r)
			}
		}
		if len(dead) == 0 {
			return
		}
		for _, r := range dead {
			s.unlinkRun(r)
		}
	}
}

// ---------------------------------------------------------------- walk

// anchors is what one read-only walk yields.
//
// The mark sets are COPIES taken under the shard's read lock, not pointers into
// live runs. Returning *run would hand the caller a field that a concurrent
// commit may be writing, and the caller is a routing decision that runs on its
// own goroutine per request.
type anchors struct {
	// pool is the holder set of the deepest run held by at least one CANDIDATE
	// backend. Empty when no such run exists.
	pool markSet
	// held is the holder set of the deepest run held by anyone at all,
	// candidate or not. Non-empty held with an empty pool is the signal that
	// holders exist but are all saturated or gone — the condition a split
	// answers.
	held markSet
	// matched is how many blocks of the request were found in the tree.
	matched int
	// anchorBlocks is the block depth reached at the pool anchor, reported as
	// the affinity strength in place of a whole-request fraction.
	anchorBlocks int
}

// walk finds the anchors for p without mutating anything.
//
// A run is a valid anchor even when only partially matched: the holder holds
// the whole run, so it certainly holds the matched prefix of it.
func (t *tree) walk(p path, cands markSet) anchors {
	sh := t.shardFor(p)
	sh.mu.RLock()
	defer sh.mu.RUnlock()

	var a anchors
	root := sh.roots[p.modelKey]
	if root == nil {
		return a
	}
	var avail, marked *run
	cur, i, n := root, 0, p.len()
	availAt := 0
	for i < n {
		child := cur.find(p.hash(i))
		if child == nil {
			break
		}
		m := 0
		for m < child.blocks() && i+m < n && child.hashes[m] == p.hash(i+m) {
			m++
		}
		if !child.marks.Empty() {
			marked = child
			if child.marks.Intersects(cands) {
				avail, availAt = child, i+m
			}
		}
		cur, i = child, i+m
		if m < child.blocks() {
			break // diverged inside the run
		}
	}
	a.matched = i
	if avail != nil {
		a.pool, a.anchorBlocks = avail.marks.Clone(), availAt
	}
	if marked != nil {
		a.held = marked.marks.Clone()
	}
	return a
}

// ---------------------------------------------------------------- commit

// commit inserts p and marks slot on EVERY run along the path, not only the
// deepest.
//
// Marking the whole path is explicit in Anton's notes ("make sure every node in
// prefix tree path marked on split, not only the last hit") and is what makes a
// split converge: the next request for this prefix finds the new backend as a
// legitimate holder at whatever depth it anchors. It is also what keeps
// marks(descendant) subset-of marks(ancestor) true, which walk relies on.
//
// Returns how many blocks the backend already held, for the predicted-hit
// metric.
func (t *tree) commit(p path, slot int) int {
	sh := t.shardFor(p)
	sh.mu.Lock()
	defer sh.mu.Unlock()

	now := t.clk.Now()
	root := sh.roots[p.modelKey]
	if root == nil {
		root = &run{}
		sh.roots[p.modelKey] = root
	}
	cur, i, n := root, 0, p.len()
	hit, broken := 0, false

	for i < n {
		child := cur.find(p.hash(i))
		if child == nil {
			// Extending a private tail in place, rather than chaining a new
			// run per turn, is what keeps a long single-session context as one
			// run: a 200-turn conversation would otherwise cost 200 runs.
			if cur.parent != nil && len(cur.kids) == 0 &&
				cur.marks.Count() == 1 && cur.marks.Has(slot) {
				for k := i; k < n; k++ {
					cur.hashes = append(cur.hashes, p.hash(k))
					cur.tokens = append(cur.tokens, p.tokens(k))
				}
				added := int64(n - i)
				sh.blocks += added
				sh.copies += added // exactly one holder, by the test above
				cur.lastUsed, cur.traffic = now, cur.traffic+1
				return hit
			}
			nr := &run{lastUsed: now, traffic: 1}
			for k := i; k < n; k++ {
				nr.hashes = append(nr.hashes, p.hash(k))
				nr.tokens = append(nr.tokens, p.tokens(k))
			}
			nr.marks.Add(slot)
			cur.link(nr)
			delete(sh.tails, cur) // no-op unless cur was childless
			sh.tails[nr] = struct{}{}
			sh.runs++
			sh.blocks += int64(nr.blocks())
			sh.copies += int64(nr.blocks())
			return hit
		}

		m := 0
		for m < child.blocks() && i+m < n && child.hashes[m] == p.hash(i+m) {
			m++
		}
		if m < child.blocks() {
			child = sh.splitRun(child, m)
		}
		if !broken {
			if child.marks.Has(slot) {
				hit += m
			} else {
				broken = true
			}
		}
		if !child.marks.Has(slot) {
			child.marks.Add(slot)
			sh.copies += int64(child.blocks())
		}
		child.lastUsed, child.traffic = now, child.traffic+1
		cur, i = child, i+m
	}
	return hit
}

// splitRun cuts r after m blocks and returns the new upper half, which takes
// r's place under its parent. Both halves start from the same holder set.
// Caller holds the write lock.
func (s *shard) splitRun(r *run, m int) *run {
	// Unlink FIRST, while the parent's sorted index still agrees with r's
	// current first hash. Re-slicing r.hashes changes the key the parent has r
	// filed under, and the binary search in unlink would then fail to find it —
	// leaving a detached duplicate in kids and corrupting every later lookup.
	parent := r.parent
	parent.unlink(r.hashes[0])

	upper := &run{
		hashes:   r.hashes[:m:m],
		tokens:   r.tokens[:m:m],
		marks:    r.marks.Clone(),
		lastUsed: r.lastUsed,
		traffic:  r.traffic,
	}
	r.hashes = r.hashes[m:]
	r.tokens = r.tokens[m:]

	parent.link(upper)
	upper.link(r)

	s.runs++
	// Block total is unchanged, and copies is unchanged too: the same blocks
	// are held by the same backends, just described by two runs instead of one.
	return upper
}

// ---------------------------------------------------------------- eviction

// sweep evicts every tail idle for longer than the TTL.
//
// Tails only: the middle of the tree is never touched. Evicting a tail
// re-checks its parent, so a dead chain unwinds in one pass, but a run with any
// remaining child is left alone — the last child cleans up. Returns the number
// of blocks freed.
func (t *tree) sweep() int64 {
	now := t.clk.Now()
	var freed int64
	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.Lock()
		var dead []*run
		for r := range sh.tails {
			if now.Sub(r.lastUsed) > t.ttl {
				dead = append(dead, r)
			}
		}
		for _, r := range dead {
			freed += sh.evictTail(r, now, t.ttl)
		}
		sh.mu.Unlock()
	}
	return freed
}

// maxCascade bounds one eviction's upward unwind so a pathological chain
// cannot hold the shard lock unboundedly. Anything left over is caught by the
// next sweep.
const maxCascade = 4096

// unlinkRun detaches a childless run and updates every counter. Returns the
// parent, which may itself have just become a tail. Caller holds the write
// lock.
func (s *shard) unlinkRun(r *run) *run {
	n := int64(r.blocks())
	s.copies -= n * int64(r.marks.Count())
	s.blocks -= n
	s.runs--
	s.expired += n

	p := r.parent
	p.unlink(r.hashes[0])
	delete(s.tails, r)
	r.parent = nil

	if p.parent == nil {
		// The parent is a model root sentinel. It is not a run and never
		// becomes a tail; drop it once it holds nothing.
		if len(p.kids) == 0 {
			s.dropRoot(p)
		}
		return nil
	}
	if len(p.kids) == 0 {
		s.tails[p] = struct{}{}
	}
	return p
}

// evictTail removes r and cascades upward through the dead chain. Caller holds
// the write lock.
func (s *shard) evictTail(r *run, now time.Time, ttl time.Duration) int64 {
	var freed int64
	for depth := 0; depth < maxCascade; depth++ {
		if r == nil || r.parent == nil || len(r.kids) > 0 {
			return freed
		}
		freed += int64(r.blocks())
		p := s.unlinkRun(r)
		if p == nil || len(p.kids) > 0 || now.Sub(p.lastUsed) <= ttl {
			return freed
		}
		r = p
	}
	return freed
}

// ---------------------------------------------------------------- stats

// treeStats is what the metrics tick and /router-viz report.
type treeStats struct {
	Runs      int64
	Blocks    int64
	Tails     int64
	Expired   int64
	AvgCopies float64
}

func (t *tree) stats() treeStats {
	var st treeStats
	var copies int64
	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.RLock()
		st.Runs += sh.runs
		st.Blocks += sh.blocks
		st.Tails += int64(len(sh.tails))
		st.Expired += sh.expired
		copies += sh.copies
		sh.mu.RUnlock()
	}
	if st.Blocks > 0 {
		st.AvgCopies = float64(copies) / float64(st.Blocks)
	}
	return st
}

// backendStats is one backend's share of the tree.
type backendStats struct {
	Runs   int64
	Blocks int64
	Tokens int64
}

// perBackend totals what each backend is modelled as holding.
//
// Computed by walking the forest rather than maintained incrementally: it is
// read on the ~1s metrics tick and by /router-viz, and keeping per-slot
// counters correct through split, in-place tail extension, eviction cascade and
// slot reuse would put four more update sites on the request path to serve a
// number nothing on the request path reads. The older policy pays the same cost
// (kvcache.Trie.Chains walks the entire trie per backend, per poll).
func (t *tree) perBackend() map[string]backendStats {
	bySlot := map[int]*backendStats{}
	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.RLock()
		for _, root := range sh.roots {
			collectPerSlot(root, bySlot)
		}
		sh.mu.RUnlock()
	}

	t.slotMu.RLock()
	defer t.slotMu.RUnlock()
	out := make(map[string]backendStats, len(t.slots))
	for url, s := range t.slots {
		if bs, ok := bySlot[s]; ok {
			out[url] = *bs
		} else {
			out[url] = backendStats{}
		}
	}
	return out
}

func collectPerSlot(r *run, out map[int]*backendStats) {
	if r.parent != nil && !r.marks.Empty() {
		blocks := int64(r.blocks())
		var tokens int64
		for _, tk := range r.tokens {
			tokens += int64(tk)
		}
		r.marks.Each(func(slot int) {
			bs := out[slot]
			if bs == nil {
				bs = &backendStats{}
				out[slot] = bs
			}
			bs.Runs++
			bs.Blocks += blocks
			bs.Tokens += tokens
		})
	}
	for _, k := range r.kids {
		collectPerSlot(k, out)
	}
}

// flush discards every run and every mark, keeping slot assignments. Backs the
// same operational reset the older policies expose.
func (t *tree) flush() {
	for i := range t.shards {
		sh := &t.shards[i]
		sh.mu.Lock()
		sh.roots = map[uint64]*run{}
		sh.tails = map[*run]struct{}{}
		sh.runs, sh.blocks, sh.copies = 0, 0, 0
		sh.mu.Unlock()
	}
}
