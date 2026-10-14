package benchmark

// sglang's cache-source metrics engine (see cache_metrics.go for the shared
// sampler/record-type/report machinery this plugs into).
//
// The four reported bands are:
//   - gpu (device) = Δcached_tokens_total{cache_source="device"}  (GPU-resident
//     radix-cache hit)
//   - host    = Δcached_tokens_total{cache_source="host"}    (CPU/DRAM
//     HiCache L2 hit)
//   - storage = Δcached_tokens_total{cache_source="storage"} (WekaFS L3 hit)
//   - compute = max(0, Δprompt_tokens_total − (device+host+storage+total))
//     (recomputed from scratch; not itself a cache_source label; "total" is
//     SGLang's untiered fallback series, see sglangCompute)
//
// sglang also emits a cache_source="total" label, but only as a backward-compat
// fallback when the per-tier breakdown (cached_tokens_details) is unavailable —
// see python/sglang/srt/observability/metrics_collector.py's report_cache_source:
// "total" and the device/host/storage labels are emitted from mutually exclusive
// branches, never as siblings. A deployment reporting the per-tier breakdown (the
// normal, desired case — L3/HiCache enabled) never emits "total" at all, so
// summing device+host+storage+total (sglangCompute) handles both shapes
// uniformly: whichever shape a deployment is in, the other side of the sum
// stays 0, so the two never double count.
//
// Unlike vLLM eligibility (which speculatively samples any chat/completions
// endpoint on the theory that it might be vLLM behind the scenes — see
// vllmEngine.Speculative), sglang sampling only ever starts when the spec says
// type=openai_sglang outright (sglangEngine.Speculative = false): there is no
// bare-host/default shape in this codebase that plausibly resolves to sglang,
// so guessing would only add failed-probe traffic against every non-sglang
// endpoint. A sampler that starts therefore keeps polling for the life of the
// run — the operator asserted the server is sglang, so a server still loading
// weights or briefly unreachable must not cost the rest of the run's samples.
// An explicit --sglang-metrics-url override is operator intent too, and
// bypasses this gate the same way vLLM's (speculative) override does — see
// startMetricsSampler in cache_metrics.go.

import (
	"bufio"
	"io"
	"regexp"
	"strconv"
	"strings"
)

const (
	sglangCachedTokensTotalFamily = "sglang:cached_tokens_total"
	sglangPromptTokensTotalFamily = "sglang:prompt_tokens_total"
	sglangFamilyHint              = sglangCachedTokensTotalFamily + "/" + sglangPromptTokensTotalFamily

	// sglangPromptTokensKey is the totals/prev map key for the (unlabeled)
	// prompt_tokens_total family, kept distinct from the cache_source label
	// values ("device"/"host"/"storage"/"total") it's differenced against.
	sglangPromptTokensKey = "prompt_tokens_total"
)

var sglangCacheSourceLabelRe = regexp.MustCompile(`\bcache_source="([^"]*)"`)

// parseSGLangCounters scans Prometheus text exposition for the
// sglang:cached_tokens_total family (labeled by cache_source, summed across
// all other label combinations — model_name, engine/rank) and the
// sglang:prompt_tokens_total family (summed across all its labels into one
// value), returning cumulative values keyed by "device", "host", "storage",
// "total", and sglangPromptTokensKey. Missing families simply produce no
// entries for their keys; malformed lines are skipped.
func parseSGLangCounters(r io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		if rest, ok := strings.CutPrefix(line, sglangCachedTokensTotalFamily); ok {
			// The next byte must open the label set — cached_tokens_total is
			// always labeled by cache_source, so a bare value (or a sibling
			// series like _created sharing the prefix) is rejected.
			if len(rest) == 0 || rest[0] != '{' {
				continue
			}
			closeIdx := strings.LastIndex(rest, "}")
			if closeIdx < 0 {
				continue
			}
			fields := strings.Fields(rest[closeIdx+1:])
			if len(fields) == 0 {
				continue
			}
			v, err := strconv.ParseFloat(fields[0], 64)
			if err != nil {
				continue
			}
			m := sglangCacheSourceLabelRe.FindStringSubmatch(rest[1:closeIdx])
			if m == nil {
				continue
			}
			out[m[1]] += v
			continue
		}
		if rest, ok := strings.CutPrefix(line, sglangPromptTokensTotalFamily); ok {
			// Unlike cached_tokens_total, this family may or may not carry
			// labels (e.g. model_name) — accept either shape, and sum
			// everything into one key regardless of label values.
			if len(rest) == 0 || (rest[0] != '{' && rest[0] != ' ') {
				continue
			}
			valueField := rest
			if rest[0] == '{' {
				closeIdx := strings.LastIndex(rest, "}")
				if closeIdx < 0 {
					continue
				}
				valueField = rest[closeIdx+1:]
			}
			fields := strings.Fields(valueField)
			if len(fields) == 0 {
				continue
			}
			v, err := strconv.ParseFloat(fields[0], 64)
			if err != nil {
				continue
			}
			out[sglangPromptTokensKey] += v
			continue
		}
	}
	return out, sc.Err()
}

// sglangCompute derives the recomputed-from-scratch token count from the
// accumulated counter totals: prompt_tokens_total minus every cached token,
// clamped at 0. SGLang emits cached_tokens_total per tier (device/host/storage,
// each series appearing only once it is >0) and emits cache_source="total"
// only as a fallback when no per-tier details exist — so "total" is treated as
// untiered cached tokens and ADDED to the tiers rather than used alone (using
// "total" alone reads as ~0 cached whenever tiers are present). In the
// fallback case the tiers are absent and the sum is just "total", so the two
// shapes never double count.
func sglangCompute(totals map[string]float64) float64 {
	cached := totals["device"] + totals["host"] + totals["storage"] + totals["total"]
	compute := totals[sglangPromptTokensKey] - cached
	if compute < 0 {
		compute = 0
	}
	return compute
}

// sglangEngine is the cacheEngine value for sglang — see cache_metrics.go for
// what each field means.
var sglangEngine = cacheEngine{
	Name:        "sglang",
	SpecTypes:   []string{"openai_sglang"},
	Speculative: false,
	FamilyHint:  sglangFamilyHint,
	FlagHint:    "type=openai_sglang",
	Parse:       parseSGLangCounters,
	Bands:       []string{"compute", "gpu", "host", "storage"},
	Derive: func(totals map[string]float64) map[string]float64 {
		device := totals["device"]
		host := totals["host"]
		storage := totals["storage"]
		return map[string]float64{
			"compute": sglangCompute(totals),
			"gpu":     device,
			"host":    host,
			"storage": storage,
		}
	},
}
