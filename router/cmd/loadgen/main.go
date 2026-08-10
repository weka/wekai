// Command loadgen drives load at a router and reports what it achieved.
//
// It exists to measure the ROUTER, not a fleet: pointed at backends configured
// for zero service time, the only thing between a request and its response is
// the routing decision, the proxy hop and the accounting. Whatever throughput
// ceiling appears is the router's own.
//
// It runs inside the cluster on purpose. Driving this from a laptop through
// `kubectl port-forward` measures the port-forward — a single userspace TCP
// relay that saturates long before the router does.
//
// Traffic is prefix-heavy by construction, because that is what makes routing
// cost what it costs: the affinity flow walks a radix tree over the request's
// blocks, so a benchmark of unique random prompts would measure a code path
// nobody runs. Each virtual session repeats a long shared prefix and appends a
// turn, which is the shape of real agent traffic.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

type result struct {
	concurrency int
	rps         float64
	p50, p99    time.Duration
	errors      int64
	statuses    map[int]int64
}

func run(args []string) error {
	fs := flag.NewFlagSet("loadgen", flag.ContinueOnError)
	target := fs.String("target", "http://router:8080", "router base URL")
	model := fs.String("model", "mock-vllm", "model name to send")
	steps := fs.String("concurrency-steps", "1,2,4,8,16,32,64,128,256",
		"comma-separated concurrency levels, run in order until throughput stops improving")
	dur := fs.Duration("step-duration", 10*time.Second, "how long to hold each concurrency level")
	warmup := fs.Duration("warmup", 3*time.Second, "load applied before measuring, so the prefix tree is populated")
	sessions := fs.Int("sessions", 64, "distinct conversation prefixes in circulation")
	prefixTokens := fs.Int("prefix-tokens", 2048, "approximate tokens of shared prefix per session")
	plateau := fs.Float64("plateau", 1.05,
		"stop once a step fails to beat the best throughput by this factor — the point where the router is the bottleneck")
	if err := fs.Parse(args); err != nil {
		return err
	}

	var levels []int
	for _, s := range strings.Split(*steps, ",") {
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(s), "%d", &n); err != nil || n <= 0 {
			return fmt.Errorf("bad concurrency step %q", s)
		}
		levels = append(levels, n)
	}

	corpus := buildSessions(*sessions, *prefixTokens)
	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        4096,
			MaxIdleConnsPerHost: 4096,
			MaxConnsPerHost:     0,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	fmt.Printf("warmup %s against %s\n", *warmup, *target)
	drive(client, *target, *model, corpus, 16, *warmup)

	var best result
	var all []result
	stalled := 0
	for _, c := range levels {
		r := drive(client, *target, *model, corpus, c, *dur)
		all = append(all, r)
		fmt.Printf("concurrency %4d   %8.0f req/s   p50 %-8s p99 %-8s errors %d\n",
			r.concurrency, r.rps, r.p50.Round(time.Microsecond), r.p99.Round(time.Microsecond), r.errors)
		if r.rps > best.rps*(*plateau) {
			best = r
			stalled = 0
			continue
		}
		if r.rps > best.rps {
			best = r
		}
		// Two consecutive steps that fail to beat the best by the margin, not
		// one: throughput at these rates is noisy enough that a single dip is
		// not evidence of a ceiling, and stopping on it reports a plateau that
		// is really a bad sample.
		stalled++
		if stalled >= 2 {
			fmt.Printf("plateau: %.0f req/s at concurrency %d did not beat %.0f at %d\n",
				r.rps, r.concurrency, best.rps, best.concurrency)
			break
		}
	}

	fmt.Println()
	fmt.Println("=== BOTTLENECK ===")
	fmt.Printf("peak throughput      %.0f req/s\n", best.rps)
	fmt.Printf("at concurrency       %d\n", best.concurrency)
	fmt.Printf("latency p50 / p99    %s / %s\n",
		best.p50.Round(time.Microsecond), best.p99.Round(time.Microsecond))
	fmt.Printf("per-request cost     %s (wall clock / throughput, all overheads included)\n",
		time.Duration(float64(time.Second)/best.rps).Round(time.Nanosecond))
	if best.errors > 0 {
		fmt.Printf("errors               %d %v\n", best.errors, best.statuses)
	}
	// Machine-readable, so a task can diff two runs.
	enc, _ := json.Marshal(map[string]any{
		"peak_rps": best.rps, "concurrency": best.concurrency,
		"p50_us": best.p50.Microseconds(), "p99_us": best.p99.Microseconds(),
		"errors": best.errors,
	})
	fmt.Printf("RESULT %s\n", enc)
	return nil
}

// drive holds one concurrency level for d and reports what it achieved.
func drive(client *http.Client, target, model string, corpus []string, conc int, d time.Duration) result {
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()

	var done atomic.Int64
	var errs atomic.Int64
	var mu sync.Mutex
	lat := make([]time.Duration, 0, 1<<16)
	statuses := map[int]int64{}

	start := time.Now() //clockexempt: a benchmark measures real elapsed time
	var wg sync.WaitGroup
	for i := range conc {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			// Each worker owns a slice of the sessions, so a prefix stays with
			// one stream of requests the way a real conversation does.
			rng := rand.New(rand.NewPCG(uint64(worker), 0x9E3779B97F4A7C15))
			local := make([]time.Duration, 0, 1024)
			localStatus := map[int]int64{}
			for ctx.Err() == nil {
				body := requestBody(model, corpus[rng.IntN(len(corpus))], rng)
				t0 := time.Now() //clockexempt: a benchmark measures real elapsed time
				code, err := post(ctx, client, target+"/v1/chat/completions", body)
				el := time.Since(t0) //clockexempt: a benchmark measures real elapsed time
				if err != nil {
					if ctx.Err() != nil {
						break // the step ended mid-flight; not a failure
					}
					errs.Add(1)
					continue
				}
				localStatus[code]++
				if code != http.StatusOK {
					errs.Add(1)
					continue
				}
				local = append(local, el)
				done.Add(1)
			}
			mu.Lock()
			lat = append(lat, local...)
			for k, v := range localStatus {
				statuses[k] += v
			}
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start) //clockexempt: a benchmark measures real elapsed time

	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	return result{
		concurrency: conc,
		rps:         float64(done.Load()) / elapsed.Seconds(),
		p50:         pct(lat, 0.50),
		p99:         pct(lat, 0.99),
		errors:      errs.Load(),
		statuses:    statuses,
	}
}

func pct(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := int(float64(len(sorted)-1) * q)
	return sorted[i]
}

func post(ctx context.Context, c *http.Client, url string, body []byte) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
	if err != nil {
		return 0, err
	}
	// Drain before closing: an undrained body cannot be reused, and a benchmark
	// that opens a fresh connection per request measures the dial, not the
	// router.
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode, nil
}

// buildSessions makes n distinct long prefixes. Distinct at the very front, so
// they diverge immediately and land on different tree branches the way separate
// conversations do.
func buildSessions(n, tokens int) []string {
	out := make([]string, n)
	// Deterministic: two runs must send the same corpus or their numbers are
	// not comparable, which is the entire point of recording a peak.
	rng := rand.New(rand.NewPCG(42, 42))
	const charsPerToken = 4
	for i := range n {
		var sb strings.Builder
		fmt.Fprintf(&sb, "session-%04d ", i)
		for sb.Len() < tokens*charsPerToken {
			fmt.Fprintf(&sb, "%016x ", rng.Uint64())
		}
		out[i] = sb.String()
	}
	return out
}

// requestBody builds one turn: the session's shared prefix plus a short unique
// tail, so consecutive requests hit the cached prefix and extend it — the
// pattern the affinity tree is built for.
func requestBody(model, prefix string, rng *rand.Rand) []byte {
	b, _ := json.Marshal(map[string]any{
		"model":      model,
		"max_tokens": 1,
		"messages": []map[string]string{
			{"role": "user", "content": prefix + fmt.Sprintf("turn-%08x", rng.Uint32())},
		},
	})
	return b
}
