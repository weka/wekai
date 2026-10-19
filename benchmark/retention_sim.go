package benchmark

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/weka/wekai/kvcache"
)

// Retention simulation: the same prefix matching as SimulateReplayCache, but
// with a time-to-live on cached content, scored for several TTLs in one pass.
//
// Model: every trie node remembers when it was last used. A use, hit or write,
// refreshes it, as a prompt cache's TTL does. Under a TTL w a request's cached
// prefix is its leading run of nodes, each idle no longer than w; a node behind
// an expired one cannot be reused, since KV is a prefix. So node i of a walk is
// a hit for exactly the windows >= the largest idle gap among nodes 0..i, and
// one walk scores every window. Refreshing does not depend on the window: under
// any TTL the node is either still cached or written again at this moment.

// DefaultRetentionWindows spans the API's 5-minute and 1-hour TTLs out to a
// week, enough to see where longer retention stops paying.
const DefaultRetentionWindows = "5m,15m,30m,1h,2h,4h,8h,12h,1d,2d,3d,5d,7d"

// ParseRetentionWindows parses a comma-separated list of durations; besides
// time.ParseDuration units it accepts a "d" (24h) suffix. The result is sorted
// and deduplicated. An empty string yields no windows.
func ParseRetentionWindows(s string) ([]time.Duration, error) {
	var out []time.Duration
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		var d time.Duration
		if n, ok := strings.CutSuffix(f, "d"); ok {
			days, err := strconv.ParseFloat(n, 64)
			if err != nil {
				return nil, fmt.Errorf("retention window %q: %w", f, err)
			}
			d = time.Duration(days * float64(24*time.Hour))
		} else {
			var err error
			if d, err = time.ParseDuration(f); err != nil {
				return nil, fmt.Errorf("retention window %q: %w", f, err)
			}
		}
		if d <= 0 {
			return nil, fmt.Errorf("retention window %q must be positive", f)
		}
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	uniq := out[:0]
	for _, d := range out {
		if len(uniq) == 0 || uniq[len(uniq)-1] != d {
			uniq = append(uniq, d)
		}
	}
	return uniq, nil
}

// FormatRetentionWindow renders a window compactly: 5m, 1h, 2d, 1h30m.
func FormatRetentionWindow(d time.Duration) string {
	if d%(24*time.Hour) == 0 {
		return fmt.Sprintf("%dd", d/(24*time.Hour))
	}
	s := d.String() // e.g. 1h30m0s, 5m0s, 1h0m0s
	if strings.HasSuffix(s, "m0s") {
		s = strings.TrimSuffix(s, "0s")
	}
	if strings.HasSuffix(s, "h0m") {
		s = strings.TrimSuffix(s, "0m")
	}
	return s
}

// RetentionReport is the result of SimulateRetention.
type RetentionReport struct {
	Windows []time.Duration
	// Cached[i] is the tokens served from cache under TTL Windows[i].
	Cached []int64
	// CachedInfinite is the never-evicting cache's figure; it equals
	// SimulateReplayCache's SimCachedTokens for the same requests.
	CachedInfinite int64
	Total          int64
}

// Ratio is Cached[i]/Total.
func (r RetentionReport) Ratio(i int) float64 {
	if r.Total == 0 {
		return 0
	}
	return float64(r.Cached[i]) / float64(r.Total)
}

// InfiniteRatio is CachedInfinite/Total.
func (r RetentionReport) InfiniteRatio() float64 {
	if r.Total == 0 {
		return 0
	}
	return float64(r.CachedInfinite) / float64(r.Total)
}

// Add accumulates o into r; both must use the same windows.
func (r *RetentionReport) Add(o RetentionReport) {
	if r.Windows == nil {
		r.Windows = o.Windows
		r.Cached = make([]int64, len(o.Windows))
	}
	for i := range o.Cached {
		r.Cached[i] += o.Cached[i]
	}
	r.CachedInfinite += o.CachedInfinite
	r.Total += o.Total
}

type retentionEdge struct {
	parent int32
	hash   uint64
}

// SimulateRetention scores reqs, sorted by timestamp, under each TTL in
// windows (ascending).
func SimulateRetention(reqs []RouterReplayRequest, windows []time.Duration) RetentionReport {
	sim := newRetentionSim(windows)
	for _, req := range reqs {
		hashes, tokens := BuildReplayRequestPrefix(req)
		sim.observe(parseReplayTs(req.Ts), kvcache.UnitsFromLabels(hashes, tokens))
	}
	return sim.report()
}

// retentionSim is SimulateRetention's state, fed one request at a time.
type retentionSim struct {
	windows []time.Duration
	// byBucket[k] collects tokens whose path's largest idle gap first fits
	// windows[k]; the last bucket fits none but the infinite cache.
	byBucket []int64
	total    int64
	edges    map[retentionEdge]int32
	last     []int64 // per node, last use; node 0 is the root
	prevTs   int64
}

func newRetentionSim(windows []time.Duration) *retentionSim {
	return &retentionSim{
		windows:  windows,
		byBucket: make([]int64, len(windows)+1),
		edges:    map[retentionEdge]int32{},
		last:     []int64{0},
	}
}

// observe scores one request at ts (unix nanos; 0 = unknown, taken as the
// previous request's time) and records its prefix.
func (r *retentionSim) observe(ts int64, units []kvcache.Unit) {
	if ts < r.prevTs {
		ts = r.prevTs // unknown, or a tie sorted out of order; never go back
	}
	r.prevTs = ts

	parent := int32(0)
	var maxGap int64
	bucket := 0
	matching := true
	for _, u := range units {
		tk := int64(u.Tokens)
		r.total += tk
		e := retentionEdge{parent, u.Hash}
		if matching {
			if id, ok := r.edges[e]; ok {
				if gap := ts - r.last[id]; gap > maxGap {
					maxGap = gap
					for bucket < len(r.windows) && int64(r.windows[bucket]) < maxGap {
						bucket++
					}
				}
				r.byBucket[bucket] += tk
				r.last[id] = ts
				parent = id
				continue
			}
			matching = false
		}
		id := int32(len(r.last))
		r.last = append(r.last, ts)
		r.edges[e] = id
		parent = id
	}
}

func (r *retentionSim) report() RetentionReport {
	rep := RetentionReport{Windows: r.windows, Cached: make([]int64, len(r.windows)), Total: r.total}
	var run int64
	for k := range r.windows {
		run += r.byBucket[k]
		rep.Cached[k] = run
	}
	rep.CachedInfinite = run + r.byBucket[len(r.windows)]
	return rep
}
