package cli

import (
	"fmt"
	"sort"
	"strings"
)

// Context size per request: the request's whole input as the server billed it
// (input_tokens, which in a replay already includes cached and cache-written
// tokens), so it is the context the model had to hold.

// contextBucketBounds are the histogram's upper bounds in tokens, exclusive;
// the last bucket is open-ended.
var contextBucketBounds = []int{
	1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 100_000, 128_000,
	150_000, 200_000, 300_000, 500_000, 1_000_000,
}

var contextPercentiles = []float64{0.50, 0.90, 0.95, 0.99}

type contextBucket struct {
	Lo, Hi   int // Hi 0 = open-ended
	Requests int
	Tokens   int64
}

type contextSizeStats struct {
	Model       string
	Requests    int
	Tokens      int64
	Percentiles []int // per contextPercentiles
	Max         int
	Mean        float64
}

type contextSizeReport struct {
	Buckets []contextBucket
	Total   contextSizeStats
	Models  []contextSizeStats
}

func buildContextSizeReport(models []string, byModel map[string][]int) *contextSizeReport {
	rep := &contextSizeReport{}
	lo := 0
	for _, hi := range contextBucketBounds {
		rep.Buckets = append(rep.Buckets, contextBucket{Lo: lo, Hi: hi})
		lo = hi
	}
	rep.Buckets = append(rep.Buckets, contextBucket{Lo: lo})

	var all []int
	for _, m := range models {
		sizes := byModel[m]
		rep.Models = append(rep.Models, contextStats(m, sizes))
		all = append(all, sizes...)
		for _, n := range sizes {
			b := sort.SearchInts(contextBucketBounds, n+1)
			rep.Buckets[b].Requests++
			rep.Buckets[b].Tokens += int64(n)
		}
	}
	rep.Total = contextStats("Total", all)
	return rep
}

// contextStats sorts sizes in place.
func contextStats(model string, sizes []int) contextSizeStats {
	s := contextSizeStats{Model: model, Requests: len(sizes)}
	if len(sizes) == 0 {
		return s
	}
	sort.Ints(sizes)
	for _, n := range sizes {
		s.Tokens += int64(n)
	}
	for _, p := range contextPercentiles {
		s.Percentiles = append(s.Percentiles, sizes[int(p*float64(len(sizes)-1))])
	}
	s.Max = sizes[len(sizes)-1]
	s.Mean = float64(s.Tokens) / float64(len(sizes))
	return s
}

func fmtTokensShort(n int) string {
	switch {
	case n >= 1_000_000 && n%1_000_000 == 0:
		return fmt.Sprintf("%dM", n/1_000_000)
	case n >= 999_500:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%dk", (n+500)/1_000)
	}
	return fmt.Sprint(n)
}

func printContextSizeText(rep *contextSizeReport) {
	t := rep.Total
	if t.Requests == 0 {
		return
	}
	fmt.Println("  Context size per request (input tokens incl. cached)")
	fmt.Println()
	fmt.Printf("    %-13s %9s %7s %7s %10s  %s\n", "tokens", "requests", "%", "cum%", "%tokens", "")
	const barWidth = 40
	maxReq := 0
	for _, b := range rep.Buckets {
		maxReq = max(maxReq, b.Requests)
	}
	cum := 0
	for _, b := range rep.Buckets {
		cum += b.Requests
		label := fmtTokensShort(b.Lo) + "-" + fmtTokensShort(b.Hi)
		if b.Hi == 0 {
			label = ">=" + fmtTokensShort(b.Lo)
		}
		bar := strings.Repeat("#", (b.Requests*barWidth+maxReq-1)/max(maxReq, 1))
		fmt.Printf("    %-13s %9d %6.2f%% %6.2f%% %9.2f%%  %s\n", label, b.Requests,
			100*float64(b.Requests)/float64(t.Requests), 100*float64(cum)/float64(t.Requests),
			100*shareOf(b.Tokens, t.Tokens), bar)
	}
	fmt.Println()

	fmt.Printf("    %-26s %9s %8s %8s %8s %8s %8s %8s\n", "model", "requests", "mean", "p50", "p90", "p95", "p99", "max")
	row := func(s contextSizeStats) {
		if s.Requests == 0 {
			return
		}
		fmt.Printf("    %-26s %9d %8s", s.Model, s.Requests, fmtTokensShort(int(s.Mean)))
		for _, p := range s.Percentiles {
			fmt.Printf(" %8s", fmtTokensShort(p))
		}
		fmt.Printf(" %8s\n", fmtTokensShort(s.Max))
	}
	for _, s := range rep.Models {
		if s.Model == "" {
			s.Model = "(unknown)"
		}
		row(s)
	}
	row(t)
	fmt.Println()
}

type jsonContextBucket struct {
	MinTokens int   `json:"min_tokens"`
	MaxTokens int   `json:"max_tokens,omitempty"` // exclusive; absent = open-ended
	Requests  int   `json:"requests"`
	Tokens    int64 `json:"tokens"`
}

type jsonContextStats struct {
	Model    string  `json:"model"`
	Requests int     `json:"requests"`
	Mean     float64 `json:"mean"`
	P50      int     `json:"p50"`
	P90      int     `json:"p90"`
	P95      int     `json:"p95"`
	P99      int     `json:"p99"`
	Max      int     `json:"max"`
}

type jsonContextSize struct {
	Histogram []jsonContextBucket `json:"histogram"`
	Models    []jsonContextStats  `json:"models"`
	Total     jsonContextStats    `json:"total"`
}

func toJSONContextSize(rep *contextSizeReport) *jsonContextSize {
	if rep == nil {
		return nil
	}
	conv := func(s contextSizeStats) jsonContextStats {
		j := jsonContextStats{Model: s.Model, Requests: s.Requests, Mean: s.Mean, Max: s.Max}
		if len(s.Percentiles) == 4 {
			j.P50, j.P90, j.P95, j.P99 = s.Percentiles[0], s.Percentiles[1], s.Percentiles[2], s.Percentiles[3]
		}
		return j
	}
	out := &jsonContextSize{Total: conv(rep.Total)}
	for _, b := range rep.Buckets {
		out.Histogram = append(out.Histogram, jsonContextBucket{MinTokens: b.Lo, MaxTokens: b.Hi, Requests: b.Requests, Tokens: b.Tokens})
	}
	for _, s := range rep.Models {
		out.Models = append(out.Models, conv(s))
	}
	return out
}
