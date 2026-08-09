// Package vllmmetrics aggregates upstream vLLM counters into router-level
// counters that only ever go up.
//
// A client scraping one vLLM instance gets a plain Prometheus counter. Behind
// a router there is no single instance to scrape: there are N endpoints, they
// come and go, and each restarts its counters from zero when its pod is
// replaced. Summing the raw values would make the result jump BACKWARDS on
// every restart or scale-down, and a counter that decreases breaks rate() and
// increase() — silently, producing plausible-looking graphs that are wrong.
//
// So only DELTAS are accumulated:
//
//   - current >= previous  -> add current - previous
//   - current <  previous  -> the upstream restarted; add current, a delta from 0
//   - first ever sighting  -> add current, so a pod already serving before the
//     router started still contributes its history
//
// Totals are summed across endpoints under the upstream metric name and labels
// and never decrease, including when an endpoint leaves for good. That makes
// the router's /metrics a drop-in for scraping vLLM directly.
//
// Ported from the Rust vllm-router's vllm_metrics.rs, whose delta scheme this
// is; the reasoning above is its, and it is right.
package vllmmetrics

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Defaults, matching the reference implementation.
const (
	DefaultInterval = 30 * time.Second
	DefaultTimeout  = 10 * time.Second
	// maxMissingCycles is how long per-endpoint state survives after an
	// endpoint stops being registered. Keeping it briefly means an endpoint
	// that flaps out of the registry is not mistaken for a counter reset;
	// dropping it eventually bounds memory under pod churn. Accumulated totals
	// are never affected either way.
	maxMissingCycles = 120
)

// DefaultNames are the upstream counters aggregated unless told otherwise.
// Anything else in a scrape is ignored: a router that re-exported every vLLM
// series would multiply cardinality by the fleet size for no one's benefit.
var DefaultNames = []string{"vllm:prompt_tokens_by_source_total"}

// Config configures the aggregator.
type Config struct {
	Interval time.Duration
	Timeout  time.Duration
	// Names are the counter names to track, matched before the label set.
	Names []string
}

// Aggregator scrapes endpoints and holds the accumulated totals.
type Aggregator struct {
	cfg    Config
	client *http.Client

	mu sync.Mutex
	// totals is keyed by the full series (name plus labels), summed across
	// every endpoint that ever reported it.
	totals map[string]float64
	// prev is the last value seen per endpoint per series, which is what makes
	// the delta scheme possible.
	prev map[string]map[string]float64
	// missing counts consecutive cycles an endpoint went unregistered.
	missing map[string]int
}

func New(cfg Config) *Aggregator {
	if cfg.Interval <= 0 {
		cfg.Interval = DefaultInterval
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	if len(cfg.Names) == 0 {
		cfg.Names = DefaultNames
	}
	return &Aggregator{
		cfg:     cfg,
		client:  &http.Client{Timeout: cfg.Timeout},
		totals:  map[string]float64{},
		prev:    map[string]map[string]float64{},
		missing: map[string]int{},
	}
}

// Interval is how often Run scrapes.
func (a *Aggregator) Interval() time.Duration { return a.cfg.Interval }

// Run scrapes on the configured interval until ctx is done. endpoints is called
// each cycle so the fleet can change underneath.
func (a *Aggregator) Run(ctx context.Context, endpoints func() []string) {
	t := time.NewTicker(a.cfg.Interval)
	defer t.Stop()
	a.ScrapeAll(ctx, endpoints())
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			a.ScrapeAll(ctx, endpoints())
		}
	}
}

// ScrapeAll scrapes every endpoint once, concurrently, and folds the results in.
//
// Best effort by design: one unreachable endpoint must not stop the others from
// contributing, and a scrape that fails contributes nothing rather than being
// treated as a counter reset to zero.
func (a *Aggregator) ScrapeAll(ctx context.Context, endpoints []string) {
	type result struct {
		endpoint string
		series   map[string]float64
	}
	results := make(chan result, len(endpoints))
	var wg sync.WaitGroup
	for _, ep := range endpoints {
		wg.Add(1)
		go func(ep string) {
			defer wg.Done()
			s, err := a.scrapeOne(ctx, ep)
			if err != nil {
				return // best effort; absence is not a reset
			}
			results <- result{ep, s}
		}(ep)
	}
	wg.Wait()
	close(results)

	seen := map[string]bool{}
	a.mu.Lock()
	defer a.mu.Unlock()
	for r := range results {
		seen[r.endpoint] = true
		a.missing[r.endpoint] = 0
		prev := a.prev[r.endpoint]
		if prev == nil {
			prev = map[string]float64{}
			a.prev[r.endpoint] = prev
		}
		for key, cur := range r.series {
			last, had := prev[key]
			switch {
			case !had || cur < last:
				// First sighting, or the upstream restarted: the delta is the
				// whole current value. Treating a restart as a decrease is the
				// bug this scheme exists to avoid.
				a.totals[key] += cur
			default:
				a.totals[key] += cur - last
			}
			prev[key] = cur
		}
	}

	// Age out endpoints that have stopped reporting. Totals they contributed
	// stay: the work happened.
	for ep := range a.prev {
		if seen[ep] {
			continue
		}
		a.missing[ep]++
		if a.missing[ep] > maxMissingCycles {
			delete(a.prev, ep)
			delete(a.missing, ep)
		}
	}
}

func (a *Aggregator) scrapeOne(ctx context.Context, endpoint string) (map[string]float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/metrics", nil)
	if err != nil {
		return nil, err
	}
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("scrape %s: status %d", endpoint, resp.StatusCode)
	}
	return a.parse(resp.Body), nil
}

// parse reads Prometheus text exposition, keeping only the configured names.
//
// A deliberately small parser: this reads one well-known producer's output and
// needs only `name{labels} value` lines. Pulling in a full expfmt decoder to
// ignore all but a couple of series would be a lot of dependency for a
// substring match.
func (a *Aggregator) parse(r io.Reader) map[string]float64 {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		sp := strings.LastIndexByte(line, ' ')
		if sp < 0 {
			continue
		}
		series, raw := line[:sp], line[sp+1:]
		name := series
		if i := strings.IndexByte(series, '{'); i >= 0 {
			name = series[:i]
		}
		if !a.tracked(name) {
			continue
		}
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			continue
		}
		out[series] += v
	}
	return out
}

func (a *Aggregator) tracked(name string) bool {
	for _, want := range a.cfg.Names {
		if name == want {
			return true
		}
	}
	return false
}

// WriteTo renders the accumulated totals in Prometheus text exposition, so the
// router's /metrics can carry them alongside its own.
func (a *Aggregator) WriteTo(w io.Writer) error {
	a.mu.Lock()
	keys := make([]string, 0, len(a.totals))
	for k := range a.totals {
		keys = append(keys, k)
	}
	vals := make(map[string]float64, len(a.totals))
	for k, v := range a.totals {
		vals[k] = v
	}
	a.mu.Unlock()

	// Sorted so the output is stable, and so all series of one metric are
	// adjacent — required for the HELP/TYPE headers below to be valid.
	sort.Strings(keys)
	lastName := ""
	for _, k := range keys {
		name := k
		if i := strings.IndexByte(k, '{'); i >= 0 {
			name = k[:i]
		}
		if name != lastName {
			if _, err := fmt.Fprintf(w, "# HELP %s Aggregated across the fleet by the router; monotonic across upstream restarts.\n# TYPE %s counter\n", name, name); err != nil {
				return err
			}
			lastName = name
		}
		if _, err := fmt.Fprintf(w, "%s %g\n", k, vals[k]); err != nil {
			return err
		}
	}
	return nil
}
