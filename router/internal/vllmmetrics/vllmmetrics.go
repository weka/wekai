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
// and never decrease FOR THE LIFETIME OF THE PROCESS, including when an
// endpoint leaves for good.
//
// Two things that qualifies, both of which have produced a decreasing counter
// in production and neither of which this scheme can fix from the inside:
//
//   - The totals are in memory. A router restart takes them with it, and the
//     next scrape reports whatever the fleet's first sighting adds — which,
//     right after a restart, is one scrape's worth of the fleet rather than its
//     history. The exported series drops. Scraping several router REPLICAS
//     through one Service address has the same effect continuously, since each
//     replica accumulates independently and a round-robin scrape alternates
//     between them.
//   - One endpoint must be one process. The delta is computed against what THIS
//     ADDRESS reported last time, so pointing the router at a Service or a load
//     balancer that fronts N pods means consecutive scrapes read different
//     processes: every lower reading looks like a restart and contributes a
//     whole counter, every higher one contributes a fictitious delta, and the
//     total is arbitrary. This is detected and warned about (see loadBalanced)
//     rather than silently produced, because the resulting number looks
//     entirely plausible — it is the right order of magnitude and it moves in
//     the right direction most of the time.
//
// Ported from the Rust vllm-router's vllm_metrics.rs, whose delta scheme this
// is; the reasoning above is its, and it is right as far as it goes.
package vllmmetrics

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/router/internal/registry"
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
	// deadAfter is how many consecutive barren cycles — scrape failed, or
	// succeeded and carried none of the tracked series — retire an endpoint from
	// scraping for good.
	//
	// It exists because the endpoint set is now the LIVE registry rather than a
	// list probed once at startup, which is what makes discovered pods work.
	// Without a latch, an endpoint that is not a vLLM at all would be asked for
	// /metrics every interval forever — precisely the retry-forever shape
	// serve.probeIsVLLM's doc criticises. Two cycles, so a single restart window
	// or one dropped connection does not retire a real backend.
	deadAfter = 2
	// resetsBeforeWarning is how many apparent counter restarts from ONE address
	// are tolerated before saying out loud that the address is probably not one
	// process. Real pods do restart, so a couple mean nothing; a steady stream
	// from the same address means every scrape is reading a different pod.
	resetsBeforeWarning = 5
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
	// barren counts consecutive cycles an endpoint gave nothing; dead retires it
	// once that passes deadAfter. An endpoint the registry holds is not
	// necessarily a vLLM, and asking one forever is what this avoids.
	barren map[string]int
	dead   map[string]bool
	// lastOK and lastAsked are the previous cycle's outcome, RECORDED rather
	// than derived. Deriving it from prev/barren/missing looked possible and was
	// wrong: an endpoint that fails never enters prev at all, so a count taken
	// over prev reports full coverage at the moment coverage is lost — which is
	// precisely the reading this exists to prevent.
	lastOK, lastAsked int
	// resets counts apparent counter restarts per endpoint, and warned records
	// that we have already said something about it. A steady stream from one
	// address means the address is not one process.
	resets map[string]int
	warned map[string]bool
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
		barren:  map[string]int{},
		dead:    map[string]bool{},
		resets:  map[string]int{},
		warned:  map[string]bool{},
	}
}

// Retire marks an endpoint as never worth scraping — the caller already knows it
// is not a vLLM, so it should not cost two cycles to rediscover that.
func (a *Aggregator) Retire(endpoint string) {
	a.mu.Lock()
	a.dead[endpoint] = true
	a.mu.Unlock()
}

// live filters the caller's endpoint set to those still worth asking.
func (a *Aggregator) live(endpoints []string) []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		if !a.dead[ep] {
			out = append(out, ep)
		}
	}
	return out
}

// Coverage is how many endpoints answered on the last cycle and how many were
// asked. It is what separates "aggregating, and the fleet is idle" from
// "aggregating, and nothing is reachable" — two states whose totals are
// identical and whose meanings are opposite.
func (a *Aggregator) Coverage() (contributing, asked int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.lastOK, a.lastAsked
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
	endpoints = a.live(endpoints)
	results := make(chan result, len(endpoints))
	var wg sync.WaitGroup
	for _, ep := range endpoints {
		wg.Add(1)
		go func(ep string) {
			defer wg.Done()
			s, err := a.scrapeOne(ctx, ep)
			if err != nil {
				results <- result{ep, nil} // barren, and absence is not a reset
				return
			}
			results <- result{ep, s}
		}(ep)
	}
	wg.Wait()
	close(results)

	seen := map[string]bool{}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.lastAsked, a.lastOK = len(endpoints), 0
	for r := range results {
		seen[r.endpoint] = true
		a.missing[r.endpoint] = 0
		if len(r.series) > 0 {
			a.lastOK++
		}

		// An endpoint that gave nothing — unreachable, or reachable and serving
		// none of the tracked series, which is what a hosted API or a sidecar
		// does — is retired after deadAfter cycles rather than asked forever.
		if len(r.series) == 0 {
			a.barren[r.endpoint]++
			if a.barren[r.endpoint] >= deadAfter {
				a.dead[r.endpoint] = true
				slog.Info("no longer scraping this endpoint for vLLM counters: it has served "+
					"none of them on consecutive attempts, so it is either not a vLLM or not "+
					"reachable from here",
					"endpoint", r.endpoint, "counters", a.cfg.Names)
			}
			continue
		}
		a.barren[r.endpoint] = 0

		prev := a.prev[r.endpoint]
		if prev == nil {
			prev = map[string]float64{}
			a.prev[r.endpoint] = prev
		}
		reset := false
		for key, cur := range r.series {
			last, had := prev[key]
			switch {
			case !had:
				// First sighting: the delta is the whole current value, so a pod
				// already serving before the router started still contributes its
				// history.
				a.totals[key] += cur
			case cur < last:
				// The upstream restarted. Treating that as a decrease is the bug
				// this scheme exists to avoid.
				a.totals[key] += cur
				reset = true
			default:
				a.totals[key] += cur - last
			}
			prev[key] = cur
		}
		if reset {
			a.noteReset(r.endpoint)
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

// noteReset records an apparent counter restart and, once they stop looking
// like coincidence, says what they almost certainly mean. Caller holds the lock.
//
// A pod restarting resets its counters, which is normal and is exactly what the
// delta scheme handles. The same address resetting over and over is a different
// thing: it means each scrape is reading a DIFFERENT process, which happens when
// the configured endpoint is a Service or a load balancer rather than a pod. No
// delta scheme can work through that — there is no such thing as "what this
// address reported last time" when the address is several processes — and the
// resulting totals are arbitrary while looking entirely reasonable.
//
// Worth saying loudly for a second reason: an endpoint like that also defeats
// prefix affinity, since the router believes it is routing to one backend whose
// KV cache it is modelling, and the traffic lands wherever the load balancer
// feels like. The counters are the visible symptom of a deeper misconfiguration.
func (a *Aggregator) noteReset(endpoint string) {
	a.resets[endpoint]++
	if a.resets[endpoint] < resetsBeforeWarning || a.warned[endpoint] {
		return
	}
	a.warned[endpoint] = true
	slog.Warn("this endpoint's vLLM counters keep restarting, which usually means the address "+
		"is a Service or load balancer in front of several pods rather than one pod. Aggregated "+
		"upstream totals through such an address are not meaningful, and prefix affinity is not "+
		"either: point the router at the pods (a pods: selector, or their individual addresses).",
		"endpoint", endpoint, "resets", a.resets[endpoint])
}

func (a *Aggregator) scrapeOne(ctx context.Context, endpoint string) (map[string]float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, registry.ResolveURL(endpoint, "/metrics"), nil)
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

// Render writes the accumulated totals in Prometheus text exposition, so the
// router's /metrics can carry them alongside its own.
// Deliberately not named WriteTo: that name belongs to io.WriterTo, whose
// signature returns the byte count, and a method that looks like an interface
// it does not satisfy misleads every reader and every type switch.
func (a *Aggregator) Render(w io.Writer) error {
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
