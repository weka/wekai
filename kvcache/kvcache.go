// Package kvcache models what prefix of a request a KV cache has already seen.
//
// It is the single implementation shared by three consumers with genuinely
// different needs:
//
//   - benchmark, offline and single-cache: walk the trie, credit the matched
//     prefix, insert the novel tail, and accumulate an aggregate hit ratio. It
//     models an INFINITE cache, which is correct for measuring how much reuse a
//     workload contains.
//   - the router, online and one instance per backend: ask "how much of this
//     request does this backend already hold" WITHOUT mutating anything, then
//     record the answer against exactly one backend. It must be bounded, because
//     a real vLLM node evicts under memory pressure and an unbounded model grows
//     without limit while drifting optimistic.
//   - a live server that actually ADMITS requests (router/internal/mockvllm today;
//     conceivably a real one later): needs everything the router's model needs,
//     PLUS the guarantee that a block cannot be evicted out from under a request
//     that is still in flight. RecordAndPin/Unpin cover this — see their docs.
//
// All three are served by one type. A zero Config means unbounded — the
// simulator's semantics — so the offline path behaves exactly as it did before
// extraction.
//
// # Eviction is pluggable in spirit, not (yet) in code
//
// evictLocked below is LRU-among-unpinned-leaves. The router is expected to move
// to TTL/tail eviction instead (the current LRU is considered a defect there),
// while a live server wants LRU-among-unpinned to stay — those are legitimately
// different policies for different consumers of the same tree. Pinning is
// deliberately orthogonal to which discipline runs: a pinned node is pulled out
// of the eviction candidate set entirely (see pinChain/unpinChain), so it is
// protected under LRU today and would be equally protected under a future
// TTL-ordered candidate set without any change to the pinning code. Swapping the
// discipline itself (replacing evictLocked's LRU-tail walk with a TTL-tail walk)
// is intentionally NOT done here — that is the router's change to make, on its
// own schedule, against its own consumer.
//
// # What this is not
//
// This PREDICTS residency; it does not observe it. vLLM exposes no API to ask
// which blocks a node holds. A push-based feed exists (ZMQ BlockStored /
// BlockRemoved via --kv-events-config, carrying a `medium` tier label) and
// LMCache's controller offers POST /lookup, but neither is enabled in our
// deployments. Anything built on this is an approximation, and its accuracy
// should be measured against the worker's reported cached_tokens rather than
// assumed.
package kvcache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// Unit is one hashed segment of a request's prefix, with an estimated token
// count. Deliberately just numbers: no prompt content reaches this package, so
// nothing built on it can leak prompt text.
type Unit struct {
	Hash   uint64
	Tokens int32
}

// DefaultChunkBytes is the granularity at which raw content is segmented.
//
// vLLM hashes fixed 16-token blocks (~64 bytes); this is 1024 (~256 estimated
// tokens). The coarser window is defensible because the engine RANKS rather than
// bills: a 1024-byte unit rounds the true shared prefix down to a unit boundary,
// so the per-request under-estimate averages ~128 tokens — about 1.6% of a
// typical 8k-token prompt, applied roughly equally to every candidate, so it
// almost never changes an arg-max. Finer units cost 16x the nodes and 16x the
// units per request.
const DefaultChunkBytes = 1024

// EstimateTokens is the 4-bytes-per-token heuristic, clamped to at least 1 for
// non-empty content.
//
// Deliberately crude, and that crudeness must be restated wherever it influences
// a decision: nothing here tokenizes, so every token count in any consumer of
// this package is an estimate.
func EstimateTokens(byteLen int) int32 {
	if byteLen <= 0 {
		return 0
	}
	if n := byteLen / 4; n >= 1 {
		return int32(n)
	}
	return 1
}

// HashContent is sha256(tag || 0x00 || content) truncated to 64 bits.
//
// SHA-256 rather than a fast non-cryptographic hash on purpose: prompt content
// is attacker-influenced, and a cheap-to-collide hash would let one caller craft
// content that collides with another's prefix and steal its affinity. Hashing a
// 32 KiB prompt costs ~16us with SHA-NI and happens once per request.
//
// The tag is length-prefixed so that (tag, content) is unambiguous. Without it,
// tag="user\x00" + content="X" and tag="user" + content="\x00X" hash identically,
// and both fields are caller-controlled.
func HashContent(tag string, content []byte) uint64 {
	h := sha256.New()
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(tag)))
	_, _ = h.Write(lenBuf[:])
	_, _ = h.Write([]byte(tag))
	_, _ = h.Write(content)
	var sum [sha256.Size]byte
	return binary.BigEndian.Uint64(h.Sum(sum[:0]))
}

// HashLabel folds an opaque precomputed hash label into a trie key.
//
// It exists for callers that already carry hashes as strings — wekai's redacted
// captures store them as "sha256:<16 hex chars>", which is exactly 64 bits, so
// that form is parsed losslessly. Anything else is hashed.
func HashLabel(s string) uint64 {
	hexPart := s
	if i := strings.LastIndexByte(s, ':'); i >= 0 {
		hexPart = s[i+1:]
	}
	if len(hexPart) == 16 {
		if v, err := strconv.ParseUint(hexPart, 16, 64); err == nil {
			return v
		}
	}
	return HashContent("label", []byte(s))
}

// ChunkContent splits content into fixed byte windows and returns one Unit per
// window. Two inputs sharing a leading byte prefix yield identical leading
// units, which is what lets the trie credit the shared part as cached.
//
// Only the first window carries the tag, so a multi-window segment does not
// re-anchor mid-way.
func ChunkContent(tag string, content []byte, chunkBytes int) []Unit {
	if chunkBytes <= 0 {
		chunkBytes = DefaultChunkBytes
	}
	if len(content) == 0 {
		return []Unit{{Hash: HashContent(tag, nil), Tokens: 1}}
	}
	out := make([]Unit, 0, (len(content)+chunkBytes-1)/chunkBytes)
	for off := 0; off < len(content); off += chunkBytes {
		end := off + chunkBytes
		if end > len(content) {
			end = len(content)
		}
		chunk := content[off:end]
		t := tag
		if off > 0 {
			t = "\x01cont" // distinct from any first-chunk tag, including ""
		}
		out = append(out, Unit{Hash: HashContent(t, chunk), Tokens: EstimateTokens(len(chunk))})
	}
	return out
}

// Config bounds a Trie. Zero values mean unbounded, which is the offline
// simulator's semantics.
type Config struct {
	// MaxNodes bounds structure size. 0 = unbounded.
	MaxNodes int64
	// MaxTokens bounds modelled content in estimated tokens. 0 = unbounded. For a
	// router this is the binding constraint in practice: at 1024-byte units, 2M
	// tokens is reached at roughly 7,800 nodes.
	MaxTokens int64
	// EvictBudget caps eviction work per mutation so a burst cannot spike tail
	// latency. Any excess carries to the next mutation.
	EvictBudget int
}

// Bounded reports whether eviction is active.
func (c Config) Bounded() bool { return c.MaxNodes > 0 || c.MaxTokens > 0 }

// RouterConfig is the bounded configuration a router should use.
func RouterConfig() Config {
	return Config{MaxNodes: 100_000, MaxTokens: 2_000_000, EvictBudget: 64}
}

// node is one prefix unit. Children are a sorted slice rather than a map: the
// overwhelming majority of nodes have a single child, and an allocated Go map
// costs roughly 250 bytes against 96 for this — which matters when a router holds
// one trie per backend.
type node struct {
	key    uint64
	tokens int32
	parent *node
	kids   []*node

	// Intrusive LRU links. ONLY UNPINNED leaves are in the list; see
	// evictLocked and pinChain/unpinChain.
	prev, next *node
	inLRU      bool

	// pins counts in-flight requests currently holding this node (see
	// RecordAndPin/Unpin). A node with pins > 0 is never in the LRU list,
	// regardless of whether it is a leaf, so evictLocked's "the tail is
	// always evictable" invariant needs no change to respect pinning.
	pins int32
}

func (n *node) find(key uint64) *node {
	lo, hi := 0, len(n.kids)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if n.kids[mid].key < key {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(n.kids) && n.kids[lo].key == key {
		return n.kids[lo]
	}
	return nil
}

func (n *node) addKid(c *node) {
	lo, hi := 0, len(n.kids)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if n.kids[mid].key < c.key {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	n.kids = append(n.kids, nil)
	copy(n.kids[lo+1:], n.kids[lo:])
	n.kids[lo] = c
}

func (n *node) removeKid(key uint64) {
	for i, k := range n.kids {
		if k.key == key {
			n.kids = append(n.kids[:i], n.kids[i+1:]...)
			return
		}
	}
}

// Trie is one cache's model. Safe for concurrent use.
type Trie struct {
	cfg Config

	mu          sync.RWMutex
	root        node
	head, tail  *node // head = most recently used; unpinned leaves only
	nodes       int64
	tokens      int64
	anomalies   int64
	pinnedNodes int64 // nodes currently protected from eviction by >=1 Pin

	// Aggregates across observed requests, for the offline hit ratio. Atomic so a
	// snapshot read needs no lock.
	obsCached atomic.Int64
	obsTotal  atomic.Int64
}

func New(cfg Config) *Trie {
	if cfg.Bounded() && cfg.EvictBudget <= 0 {
		cfg.EvictBudget = 64
	}
	return &Trie{cfg: cfg}
}

// Query reports how many of the request's estimated tokens this cache has likely
// seen, and the request's total.
//
// Pure: no allocation, no LRU movement, no counter changes. That purity is what
// makes it safe for a router to run against every candidate, and it is deliberate
// that a query does not refresh recency — warmth must follow what was actually
// sent, or every model is refreshed by every request and eviction order stops
// meaning anything.
func (t *Trie) Query(units []Unit) (cached, total int) {
	for i := range units {
		total += int(units[i].Tokens)
	}
	if len(units) == 0 {
		return 0, 0
	}
	t.mu.RLock()
	n := &t.root
	for i := range units {
		c := n.find(units[i].Hash)
		if c == nil {
			break
		}
		// Credit the QUERIED unit's estimate, not the stored node's. They are
		// normally identical, but crediting the stored count lets cached exceed
		// total whenever a hash maps to a different estimate, which pushes a
		// matched fraction above 1.0 and makes any threshold built on it useless.
		cached += int(units[i].Tokens)
		n = c
	}
	t.mu.RUnlock()
	return cached, total
}

// Commit records that this cache served the request.
func (t *Trie) Commit(units []Unit) {
	if len(units) == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.commitLocked(units)
}

// RecordAndCount walks the matched prefix, credits it, inserts the novel tail,
// and returns (cached, total) for this call.
//
// This is the offline simulator's entry point: one call that both measures and
// learns. A router must not use it — querying every candidate this way would
// teach each of them every request until every model converged to the union, at
// which point the prediction carries no information.
func (t *Trie) RecordAndCount(units []Unit) (cached, total int) {
	for i := range units {
		total += int(units[i].Tokens)
	}
	if len(units) == 0 {
		return 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	n := &t.root
	i := 0
	for ; i < len(units); i++ {
		c := n.find(units[i].Hash)
		if c == nil {
			break
		}
		cached += int(units[i].Tokens)
		n = c
	}
	t.insertFrom(n, units, i)
	return cached, total
}

// Pin is a handle to one admitted request's reserved chain of blocks — both
// the reused prefix and any newly inserted tail — held for as long as the
// request is in flight so eviction cannot reclaim something it still needs.
// This is vLLM's real behavior: a running sequence's KV blocks are
// reference-counted and unevictable until the sequence finishes. Call Unpin
// exactly once, whatever the outcome (success, error, or client
// disconnect), or the blocks leak as permanently unevictable.
type Pin struct{ leaf *node }

// RecordAndPin is RecordAndCount plus pinning: it walks the matched prefix,
// credits it, inserts the novel tail (evicting unpinned LRU blocks if
// bounded — pinned blocks are never eviction candidates, see pinChain), and
// pins every node on the resulting root-to-leaf chain.
//
// Use this instead of RecordAndCount for any caller that has a real
// in-flight request to protect (RecordAndCount itself stays as-is: the
// offline simulator never has anything "in flight" to protect, since it
// processes one request fully before starting the next). Use this instead of
// Commit for the same reason the router doesn't use RecordAndCount: Commit is
// a fire-and-forget record of service, with nothing held afterward for a
// caller to release.
func (t *Trie) RecordAndPin(units []Unit) (cached, total int, pin *Pin) {
	for i := range units {
		total += int(units[i].Tokens)
	}
	if len(units) == 0 {
		return 0, 0, &Pin{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	n := &t.root
	i := 0
	for ; i < len(units); i++ {
		c := n.find(units[i].Hash)
		if c == nil {
			break
		}
		cached += int(units[i].Tokens)
		n = c
	}
	leaf := t.insertFrom(n, units, i)
	t.pinChain(leaf)
	return cached, total, &Pin{leaf: leaf}
}

// Unpin releases a Pin. Any of its blocks no longer held by another
// in-flight request's Pin become eviction candidates again, and a bounded
// trie takes the opportunity to catch up on any eviction that pinning had
// deferred.
func (t *Trie) Unpin(p *Pin) {
	if p == nil || p.leaf == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.unpinChain(p.leaf)
	if t.cfg.Bounded() {
		t.evictLocked(t.cfg.EvictBudget)
	}
}

// PinnedNodes reports how many nodes are currently protected from eviction by
// at least one held Pin.
func (t *Trie) PinnedNodes() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.pinnedNodes
}

// Chain is one root-to-leaf path through the trie: the ordered sequence of
// block hashes (and their estimated token counts) from the root to a single
// leaf, i.e. everything one session/request currently has resident, in the
// order it was requested.
type Chain struct {
	Hashes []uint64
	Tokens []int32
}

// Chains enumerates up to limit root-to-leaf paths (limit<=0 means
// unlimited) for read-only introspection — a live visualization, an
// operator debugging a fleet, anything that wants to SEE what's resident
// rather than query a specific prefix. Nothing in routing or the cache
// estimator uses this; Query/Commit/RecordAndCount/RecordAndPin remain the
// only mutating or decision-making entry points.
//
// Returns the chains found (capped at limit) and totalLeaves, the true
// total the trie holds regardless of the cap, so a caller that truncates
// can report how much was left out rather than truncating silently.
//
// Pure: RLock only, no mutation, no LRU movement, no pin/counter changes.
// The walk always visits every node to compute an accurate totalLeaves —
// for a router-scale trie (tens of thousands of nodes) that is a real but
// bounded cost, acceptable for an introspection endpoint polled at human
// (~1/sec) rates; short-circuiting once limit chains are found while still
// counting the rest cheaply would need a separate leaf-count fast path if
// this is ever called at higher frequency or against a much larger trie.
func (t *Trie) Chains(limit int) (chains []Chain, totalLeaves int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var walk func(n *node, hashes []uint64, tokens []int32)
	walk = func(n *node, hashes []uint64, tokens []int32) {
		if len(n.kids) == 0 {
			if n == &t.root {
				return // an entirely empty trie: the root itself is not a chain
			}
			totalLeaves++
			if limit <= 0 || len(chains) < limit {
				chains = append(chains, Chain{
					Hashes: append([]uint64(nil), hashes...),
					Tokens: append([]int32(nil), tokens...),
				})
			}
			return
		}
		for _, c := range n.kids {
			walk(c, append(hashes, c.key), append(tokens, c.tokens))
		}
	}
	walk(&t.root, nil, nil)
	return chains, totalLeaves
}

func (t *Trie) commitLocked(units []Unit) {
	n := &t.root
	i := 0
	for ; i < len(units); i++ {
		c := n.find(units[i].Hash)
		if c == nil {
			break
		}
		n = c
	}
	if n != &t.root && n.inLRU {
		t.lruMoveToFront(n)
	}
	t.insertFrom(n, units, i)
}

// insertFrom appends the novel tail starting at units[i], evicts if bounded,
// and returns the final node reached (n itself, if the tail was empty). The
// return value lets a caller that needs the whole chain — pinChain, in
// particular — find it without re-walking; every existing caller is free to
// ignore it, so this is additive.
func (t *Trie) insertFrom(n *node, units []Unit, i int) *node {
	inserted := 0
	for ; i < len(units); i++ {
		n = t.insertLocked(n, units[i])
		inserted++
	}
	if t.cfg.Bounded() {
		// The budget must cover the work this call created, or a call inserting
		// more nodes than the budget removes leaves a permanent net gain and the
		// caps become advisory rather than real.
		t.evictLocked(inserted + t.cfg.EvictBudget)
	}
	return n
}

func (t *Trie) insertLocked(parent *node, u Unit) *node {
	if parent != &t.root && parent.inLRU {
		t.lruRemove(parent) // the parent stops being a leaf
	}
	c := &node{key: u.Hash, tokens: u.Tokens, parent: parent}
	parent.addKid(c)
	t.lruPushFront(c)
	t.nodes++
	t.tokens += int64(u.Tokens)
	return c
}

// evictLocked drops least-recently-used leaves until within budget.
//
// The LRU list contains ONLY leaves, by construction, so the tail is always
// evictable: eviction never scans for a victim and never unlinks a subtree.
//
// Maintaining that on insert is what makes it true. The tempting alternative —
// keep every node in the list and argue the tail must be a leaf because a commit
// touches ancestors before descendants — is backwards: touching root-side nodes
// first makes the DEEPEST node most-recently-used, so the tail trends toward
// nodes WITH children and eviction silently stops making progress.
func (t *Trie) evictLocked(budget int) {
	for budget > 0 && t.overBudget() {
		v := t.tail
		if v == nil {
			return
		}
		if len(v.kids) != 0 {
			t.anomalies++ // unreachable unless the LRU discipline is broken
			return
		}
		p := v.parent
		p.removeKid(v.key)
		t.lruRemove(v)
		t.nodes--
		t.tokens -= int64(v.tokens)
		if p != &t.root && len(p.kids) == 0 && p.pins == 0 {
			// The parent just became a leaf. It goes to the TAIL, not the front: a
			// prefix whose whole subtree has gone should be reclaimed promptly, not
			// treated as freshly used.
			//
			// The pins==0 guard matters: p can be a DIFFERENT in-flight request's
			// leaf (that request's chain ends exactly here, while v was a longer,
			// already-finished request's now-evicted continuation). A pinned p must
			// stay out of the LRU list no matter how it lost its children — pinChain
			// already removed it when it was first pinned, and unpinChain is the
			// only path allowed to put it back.
			t.lruPushBack(p)
		}
		budget--
	}
}

func (t *Trie) overBudget() bool {
	return (t.cfg.MaxNodes > 0 && t.nodes > t.cfg.MaxNodes) ||
		(t.cfg.MaxTokens > 0 && t.tokens > t.cfg.MaxTokens)
}

func (t *Trie) lruPushFront(n *node) {
	n.inLRU = true
	n.prev, n.next = nil, t.head
	if t.head != nil {
		t.head.prev = n
	}
	t.head = n
	if t.tail == nil {
		t.tail = n
	}
}

func (t *Trie) lruPushBack(n *node) {
	n.inLRU = true
	n.next, n.prev = nil, t.tail
	if t.tail != nil {
		t.tail.next = n
	}
	t.tail = n
	if t.head == nil {
		t.head = n
	}
}

func (t *Trie) lruRemove(n *node) {
	if !n.inLRU {
		return
	}
	if n.prev != nil {
		n.prev.next = n.next
	} else {
		t.head = n.next
	}
	if n.next != nil {
		n.next.prev = n.prev
	} else {
		t.tail = n.prev
	}
	n.prev, n.next, n.inLRU = nil, nil, false
}

func (t *Trie) lruMoveToFront(n *node) {
	if t.head == n {
		return
	}
	t.lruRemove(n)
	t.lruPushFront(n)
}

// pinChain walks leaf up to (excluding) the root, incrementing each node's
// pin count and pulling any that just became pinned (0 -> 1) out of the LRU
// list. Ancestors reached this way include both the request's reused prefix
// and its newly inserted tail — the whole chain a real sequence would hold —
// which is why this must be called with the node insertFrom returned, not
// just the tail.
func (t *Trie) pinChain(leaf *node) {
	for n := leaf; n != nil && n != &t.root; n = n.parent {
		if n.pins == 0 {
			if n.inLRU {
				t.lruRemove(n)
			}
			t.pinnedNodes++
		}
		n.pins++
	}
}

// unpinChain is pinChain's inverse: decrements each node's pin count and, for
// any that just became unpinned (1 -> 0) AND are still a leaf, returns them
// to the LRU as most-recently-used (they were, after all, just released from
// active use). A node that gained children while pinned (another request's
// chain grew through it) is correctly left out of the LRU here too — the
// same invariant insertLocked maintains for any other interior node.
func (t *Trie) unpinChain(leaf *node) {
	for n := leaf; n != nil && n != &t.root; n = n.parent {
		if n.pins == 0 {
			continue // defensive: guards against an unpaired Unpin, not expected in normal use
		}
		n.pins--
		if n.pins == 0 {
			t.pinnedNodes--
			if len(n.kids) == 0 {
				t.lruPushFront(n)
			}
		}
	}
}

// Observe accumulates one request's outcome into the aggregate ratio. It inserts
// nothing; use RecordAndCount or Commit for that.
func (t *Trie) Observe(cached, total int) {
	if total <= 0 {
		return
	}
	t.obsCached.Add(int64(cached))
	t.obsTotal.Add(int64(total))
}

// Ratio is cached/total across all observed requests, or 0 if none.
func (t *Trie) Ratio() float64 {
	total := t.obsTotal.Load()
	if total <= 0 {
		return 0
	}
	return float64(t.obsCached.Load()) / float64(total)
}

// Reset clears the model. It does not clear the observed aggregates.
//
// Caution for a caller that also pins (router/internal/mockvllm; the router
// itself never does, so this is currently unreachable): Reset does not know
// about outstanding Pins and does not force-unpin them. A later Unpin against
// a Pin issued before Reset walks nodes that are now orphaned — detached from
// the fresh root — so it neither corrupts the reset tree nor panics, but it
// silently decrements pinnedNodes for blocks that were already discarded.
// Don't call Reset while requests holding Pins are in flight.
func (t *Trie) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.root = node{}
	t.head, t.tail = nil, nil
	t.nodes, t.tokens = 0, 0
}

// Stats reports current size and any detected invariant violations, which must
// stay at zero.
func (t *Trie) Stats() (nodes, tokens, anomalies int64) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.nodes, t.tokens, t.anomalies
}

// Nodes is the current node count.
func (t *Trie) Nodes() int64 {
	n, _, _ := t.Stats()
	return n
}

// Estimator wraps a Trie with a chunk size, for callers that have raw content
// rather than precomputed units.
type Estimator struct {
	trie       *Trie
	chunkBytes int
}

func NewEstimator(chunkBytes int, cfg Config) *Estimator {
	if chunkBytes <= 0 {
		chunkBytes = DefaultChunkBytes
	}
	return &Estimator{trie: New(cfg), chunkBytes: chunkBytes}
}

// Observe chunks content, walks and extends the trie, accumulates the aggregate
// ratio, and returns this request's cached fraction in [0,1].
func (e *Estimator) Observe(content string) float64 {
	units := ChunkContent("chunk", []byte(content), e.chunkBytes)
	cached, total := e.trie.RecordAndCount(units)
	e.trie.Observe(cached, total)
	if total <= 0 {
		return 0
	}
	return float64(cached) / float64(total)
}

// Insert extends the trie with content so later requests can match it. The
// per-request ratio is not meaningful for this call.
func (e *Estimator) Insert(content string) {
	e.trie.RecordAndCount(ChunkContent("chunk", []byte(content), e.chunkBytes))
}

func (e *Estimator) Ratio() float64 { return e.trie.Ratio() }
func (e *Estimator) Nodes() int     { return int(e.trie.Nodes()) }
func (e *Estimator) Trie() *Trie    { return e.trie }

// UnitsFromLabels builds units from precomputed hash labels and token counts,
// for callers whose hashes were computed elsewhere (e.g. a redacted capture).
func UnitsFromLabels(hashes []string, tokens []int) []Unit {
	out := make([]Unit, 0, len(hashes))
	for i, h := range hashes {
		tk := 0
		if i < len(tokens) {
			tk = tokens[i]
		}
		out = append(out, Unit{Hash: HashLabel(h), Tokens: int32(tk)})
	}
	return out
}

// HexHash renders a key for logging or for comparison with wekai's string form.
func HexHash(h uint64) string {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], h)
	return hex.EncodeToString(b[:])
}
