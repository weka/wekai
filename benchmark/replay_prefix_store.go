package benchmark

import (
	"sort"
	"time"

	"github.com/weka/wekai/kvcache"
)

// PrefixStore holds just what the offline cache simulations read from a
// replay: per request its time, model, server token counts and cacheable
// prefix as kvcache units. A replay request carries its conversation's whole
// history, so materialising every request's prefix costs the sum of all
// history lengths; instead a request is stored as "the first Shared units of
// its predecessor in the same instance, then Tail". Consecutive turns share
// all but the last message or so, which makes the tails short.
//
// Rebuilding a prefix walks predecessors, so every prefixCheckpointEvery-th
// request of a chain stores its whole prefix and bounds that walk.
type PrefixStore struct {
	Reqs []PrefixRequest
	// Sessions counts appended batches; a request's Session is its batch's
	// 0-based ordinal, so a batch per replay session makes it a session id.
	Sessions int
	models   map[string]string
}

const prefixCheckpointEvery = 32

// PrefixRequest is one request of a PrefixStore.
type PrefixRequest struct {
	Ts              int64 // unix nanos; 0 when unparseable
	Model           string
	Session         int32
	InputTokens     int
	CacheReadTokens int
	// AnyTokens reports a non-zero per-unit token count, as opposed to a
	// file without per-block tokens.
	AnyTokens bool

	parent int32 // predecessor index, -1 for none
	shared int32 // leading units taken from the predecessor
	length int32 // full prefix length
	depth  int32 // hops to the nearest stored-in-full ancestor
	tail   []kvcache.Unit
}

// Len is the request's prefix length in units.
func (r *PrefixRequest) Len() int { return int(r.length) }

// PrefixBatch is a store fragment built independently, e.g. from one session
// by a worker, then appended with PrefixStore.Append. Its parent links are
// local to the batch.
type PrefixBatch struct{ reqs []PrefixRequest }

// AddChain appends requests that follow one another, in order, such as one
// instance's requests, linking each to the previous.
func (b *PrefixBatch) AddChain(reqs []RouterReplayRequest) {
	var prev []kvcache.Unit
	prevIdx := int32(-1)
	prevDepth := int32(0)
	for _, req := range reqs {
		hashes, tokens := BuildReplayRequestPrefix(req)
		units := kvcache.UnitsFromLabels(hashes, tokens)
		r := PrefixRequest{
			Ts:              parseReplayTs(req.Ts),
			Model:           req.Model,
			InputTokens:     req.InputTokens,
			CacheReadTokens: req.CacheReadTokens,
			parent:          -1,
			length:          int32(len(units)),
		}
		for _, u := range units {
			if u.Tokens > 0 {
				r.AnyTokens = true
				break
			}
		}
		p := 0
		if prevIdx >= 0 && prevDepth+1 < prefixCheckpointEvery {
			for p < len(prev) && p < len(units) && prev[p] == units[p] {
				p++
			}
		}
		if p > 0 {
			r.parent, r.shared, r.depth = prevIdx, int32(p), prevDepth+1
		}
		r.tail = append([]kvcache.Unit(nil), units[p:]...)
		prev, prevIdx, prevDepth = units, int32(len(b.reqs)), r.depth
		b.reqs = append(b.reqs, r)
	}
}

// Append adds a batch, rebasing its parent links.
func (s *PrefixStore) Append(b *PrefixBatch) {
	if s.models == nil {
		s.models = map[string]string{}
	}
	off := int32(len(s.Reqs))
	sess := int32(s.Sessions)
	s.Sessions++
	for _, r := range b.reqs {
		r.Session = sess
		if r.parent >= 0 {
			r.parent += off
		}
		if m, ok := s.models[r.Model]; ok {
			r.Model = m
		} else {
			s.models[r.Model] = r.Model
		}
		s.Reqs = append(s.Reqs, r)
	}
}

// Units writes request i's prefix into buf, reallocating it if short, and
// returns it.
func (s *PrefixStore) Units(i int, buf []kvcache.Unit) []kvcache.Unit {
	r := &s.Reqs[i]
	if cap(buf) < int(r.length) {
		buf = make([]kvcache.Unit, r.length)
	}
	buf = buf[:r.length]
	limit := r.length
	for cur := r; limit > 0; cur = &s.Reqs[cur.parent] {
		if limit > cur.shared {
			copy(buf[cur.shared:limit], cur.tail[:limit-cur.shared])
			limit = cur.shared
		}
		if limit == 0 {
			break
		}
	}
	return buf
}

// ChronologicalOrder returns request indices sorted by time, ties in store
// order.
func (s *PrefixStore) ChronologicalOrder() []int32 {
	idx := make([]int32, len(s.Reqs))
	for i := range idx {
		idx[i] = int32(i)
	}
	sort.SliceStable(idx, func(a, b int) bool { return s.Reqs[idx[a]].Ts < s.Reqs[idx[b]].Ts })
	return idx
}

func parseReplayTs(ts string) int64 {
	t, err := time.Parse(time.RFC3339Nano, ts)
	if err != nil {
		return 0
	}
	return t.UnixNano()
}

// SimulateReplayCacheStore is SimulateReplayCache over the store requests
// named by idx, in that order.
func SimulateReplayCacheStore(s *PrefixStore, idx []int32) ReplayCacheReport {
	r := ReplayCacheReport{Requests: len(idx)}
	if len(idx) == 0 {
		return r
	}
	trie := newPrefixTrie()
	anyBlockToken, hadPrefix := false, false
	var buf []kvcache.Unit
	for _, i := range idx {
		req := &s.Reqs[i]
		if req.length > 0 {
			hadPrefix = true
			buf = s.Units(int(i), buf)
			cached, total := trie.t.RecordAndCount(buf)
			trie.ObserveRequest(cached, total)
			r.SimCachedTokens += cached
			r.SimTotalTokens += total
			anyBlockToken = anyBlockToken || req.AnyTokens
		}
		r.GTCachedTokens += req.CacheReadTokens
		r.GTTotalTokens += req.InputTokens
	}
	if r.SimTotalTokens > 0 {
		r.SimRatio = float64(r.SimCachedTokens) / float64(r.SimTotalTokens)
	}
	if r.GTTotalTokens > 0 {
		r.GTRatio = float64(r.GTCachedTokens) / float64(r.GTTotalTokens)
	}
	r.BlockTokensAllZero = hadPrefix && !anyBlockToken
	return r
}

// SimulateRetentionStore is SimulateRetention over the store requests named
// by idx, in that order.
func SimulateRetentionStore(s *PrefixStore, idx []int32, windows []time.Duration) RetentionReport {
	sim := newRetentionSim(windows)
	var buf []kvcache.Unit
	for _, i := range idx {
		buf = s.Units(int(i), buf)
		sim.observe(s.Reqs[i].Ts, buf)
	}
	return sim.report()
}
