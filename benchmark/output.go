package benchmark

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// FormatText formats the benchmark results as human-readable text
func (r *BenchmarkResult) FormatText() string {
	var sb strings.Builder

	// Header
	sb.WriteString(strings.Repeat("=", 80) + "\n")
	sb.WriteString(fmt.Sprintf("Benchmark Results\n"))
	sb.WriteString(strings.Repeat("=", 80) + "\n\n")

	sb.WriteString(fmt.Sprintf("Documentation: %s\n", r.DocsDir))
	sb.WriteString(fmt.Sprintf("Question: %s\n", r.Question))
	sb.WriteString(fmt.Sprintf("Cycles: %d (repeats all series to test long-term cache)\n", r.NumCycles))
	sb.WriteString(fmt.Sprintf("Series per cycle: %d (each series has unique GUID)\n", r.NumSeries))
	sb.WriteString(fmt.Sprintf("Concurrency: %d\n", r.Concurrency))
	sb.WriteString(fmt.Sprintf("Total Duration: %v\n", r.TotalDuration))
	if r.TimedOut {
		sb.WriteString("Status: PARTIAL RESULTS (stopped by timeout or signal)\n")
	}
	sb.WriteString("\n")

	// Per-model results
	for i, model := range r.Models {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(formatModelResult(&model))
	}

	// Summary comparison table
	if len(r.Models) > 1 {
		sb.WriteString("\n")
		sb.WriteString(strings.Repeat("=", 80) + "\n")
		sb.WriteString("Summary Comparison\n")
		sb.WriteString(strings.Repeat("=", 80) + "\n\n")
		sb.WriteString(formatComparisonTable(r.Models))
	}

	return sb.String()
}

// formatModelResult formats results for a single model
func formatModelResult(m *ModelBenchmarkResult) string {
	var sb strings.Builder

	sb.WriteString(strings.Repeat("-", 80) + "\n")
	sb.WriteString(fmt.Sprintf("Model: %s\n", m.ModelDisplayName))
	sb.WriteString(strings.Repeat("-", 80) + "\n")

	// Per-request details
	sb.WriteString("Request Details:\n")
	sb.WriteString(fmt.Sprintf("  %-6s %-8s %-8s %-15s %-15s %-12s %-12s\n",
		"Cycle", "Series", "Request", "TTFT", "Response Time", "Cached", "Status"))
	sb.WriteString(fmt.Sprintf("  %s\n", strings.Repeat("-", 85)))

	for _, req := range m.Requests {
		status := "Success"
		if req.Error != nil {
			errMsg := fmt.Sprintf("%v", req.Error)
			if len(errMsg) > 128 {
				errMsg = errMsg[:128] + "..."
			}
			status = fmt.Sprintf("Failed: %s", errMsg)
		}
		cachedStr := fmt.Sprintf("%d", req.UsageData.CachedTokens.Count)
		sb.WriteString(fmt.Sprintf("  %-6d %-8d %-8d %-15v %-15v %-12s %s\n",
			req.CycleNum,
			req.SeriesNum,
			req.RequestNum,
			req.TimeToFirstToken,
			req.TotalResponseTime,
			cachedStr,
			status))
	}
	sb.WriteString("\n")

	// Overall statistics — derive cycles, series, requests-per-series from data
	successful := m.TotalRequests - m.FailedRequests
	cycles := make(map[int]struct{})
	seriesGUIDs := make(map[string]struct{})
	for _, req := range m.Requests {
		cycles[req.CycleNum] = struct{}{}
		seriesGUIDs[req.SeriesGUID] = struct{}{}
	}
	numCycles := len(cycles)
	numSeries := len(seriesGUIDs)
	reqsPerSeries := 0
	if numSeries > 0 && numCycles > 0 {
		reqsPerSeries = m.TotalRequests / (numSeries * numCycles)
	}
	sb.WriteString(fmt.Sprintf("Total Requests: %d (%d cycles × %d series × %d req/series, concurrency %d)\n",
		m.TotalRequests, numCycles, numSeries, reqsPerSeries, m.Concurrency))
	sb.WriteString(fmt.Sprintf("Successful: %d | Failed: %d\n", successful, m.FailedRequests))
	sb.WriteString(fmt.Sprintf("Explicit Cached: %d | Implicit Cached: %d (50%%+ TTFT improvement)\n",
		m.CachedRequests, m.ImplicitCachedRequests))

	// Cache ratio: first request per unique series is never cached, all repeats are expected cached
	actualCached := m.CachedRequests + m.ImplicitCachedRequests
	uniqueSeries := make(map[string]struct{})
	for _, req := range m.Requests {
		if req.Error == nil {
			uniqueSeries[req.SeriesGUID] = struct{}{}
		}
	}
	expectedCached := successful - len(uniqueSeries)
	if successful > 0 {
		actualPct := float64(actualCached) / float64(successful) * 100
		expectedPct := float64(expectedCached) / float64(successful) * 100
		sb.WriteString(fmt.Sprintf("Cache ratio: %.1f%% (%.1f%% expected)\n", actualPct, expectedPct))
	}
	sb.WriteString("\n")

	// Throughput
	// Req/s = reqs / wall_time (system throughput)
	// Prefill tok/s = input_tokens / (sum_TTFT / concurrency) = system prefill rate
	// Decode tok/s = output_tokens / (sum_decode / concurrency) = system decode rate
	// Effective phase wall time = thread-time / concurrency
	if m.WallDuration > 0 && m.Concurrency > 0 {
		type catStats struct {
			reqs, inputTok, outputTok int
			sumTTFT, sumDecode        float64 // thread-seconds
		}
		var all, hit, miss catStats
		for _, req := range m.Requests {
			if req.Error != nil {
				continue
			}
			ttft := req.TimeToFirstToken.Seconds()
			decode := max(req.TotalResponseTime.Seconds()-ttft, 0)
			s := catStats{1, req.UsageData.InputTokens.Count, req.UsageData.OutputTokens.Count, ttft, decode}
			all.reqs += s.reqs
			all.inputTok += s.inputTok
			all.outputTok += s.outputTok
			all.sumTTFT += s.sumTTFT
			all.sumDecode += s.sumDecode
			if req.UsageData.CachedTokens.Count > 0 || req.IsImplicitCached {
				hit.reqs += s.reqs
				hit.inputTok += s.inputTok
				hit.outputTok += s.outputTok
				hit.sumTTFT += s.sumTTFT
				hit.sumDecode += s.sumDecode
			} else {
				miss.reqs += s.reqs
				miss.inputTok += s.inputTok
				miss.outputTok += s.outputTok
				miss.sumTTFT += s.sumTTFT
				miss.sumDecode += s.sumDecode
			}
		}
		wallSecs := m.WallDuration.Seconds()
		conc := float64(m.Concurrency)
		sb.WriteString(fmt.Sprintf("Throughput (%.1fs wall / %d concurrency):\n", wallSecs, m.Concurrency))
		sb.WriteString(fmt.Sprintf("  %-12s %8s %15s %15s\n", "", "Req/s", "Prefill tok/s", "Decode tok/s"))
		for _, row := range []struct {
			label string
			s     catStats
		}{
			{"Cache-miss", miss},
			{"Cache-hit", hit},
			{"Total", all},
		} {
			rps, itps, otps := "-", "-", "-"
			if row.s.reqs > 0 {
				rps = fmt.Sprintf("%.2f", float64(row.s.reqs)/wallSecs)
				if row.s.sumTTFT > 0 {
					itps = fmt.Sprintf("%.0f", float64(row.s.inputTok)/(row.s.sumTTFT/conc))
				}
				if row.s.sumDecode > 0 {
					otps = fmt.Sprintf("%.0f", float64(row.s.outputTok)/(row.s.sumDecode/conc))
				}
			}
			sb.WriteString(fmt.Sprintf("  %-12s %8s %15s %15s  (%d reqs)\n", row.label, rps, itps, otps, row.s.reqs))
		}
		// Phase time breakdown — percentages of total thread-time (prefill+decode overlap on GPU)
		totalThreadTime := all.sumTTFT + all.sumDecode
		if totalThreadTime > 0 {
			prefillPct := all.sumTTFT / totalThreadTime * 100
			decodePct := all.sumDecode / totalThreadTime * 100
			sb.WriteString(fmt.Sprintf("  Thread-time split: %.0f%% prefill (%.0fs) / %.0f%% decode (%.0fs), total %.0fs over %d slots\n",
				prefillPct, all.sumTTFT, decodePct, all.sumDecode, totalThreadTime, m.Concurrency))
		}
		sb.WriteString("\n")
	}

	// Collect TTFT and response time durations split by cache status
	var ttftAll, ttftHit, ttftMiss []time.Duration
	var rtAll, rtHit, rtMiss []time.Duration
	for _, req := range m.Requests {
		if req.Error != nil {
			continue
		}
		isCached := req.UsageData.CachedTokens.Count > 0 || req.IsImplicitCached
		ttftAll = append(ttftAll, req.TimeToFirstToken)
		rtAll = append(rtAll, req.TotalResponseTime)
		if isCached {
			ttftHit = append(ttftHit, req.TimeToFirstToken)
			rtHit = append(rtHit, req.TotalResponseTime)
		} else {
			ttftMiss = append(ttftMiss, req.TimeToFirstToken)
			rtMiss = append(rtMiss, req.TotalResponseTime)
		}
	}

	// Percentile helper
	pct := func(sorted []time.Duration, p float64) time.Duration {
		if len(sorted) == 0 {
			return 0
		}
		idx := int(float64(len(sorted)-1) * p)
		return sorted[idx]
	}
	sortDurations := func(d []time.Duration) {
		sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
	}
	sortDurations(ttftAll)
	sortDurations(ttftHit)
	sortDurations(ttftMiss)
	sortDurations(rtAll)
	sortDurations(rtHit)
	sortDurations(rtMiss)

	fmtDur := func(d time.Duration) string {
		if d == 0 {
			return "-"
		}
		return d.Truncate(time.Millisecond).String()
	}

	// TTFT percentiles: cache-miss / cache-hit / total
	sb.WriteString(fmt.Sprintf("%-14s %10s %10s %10s %10s %10s\n", "", "Avg", "Min", "p50", "p95", "p99"))
	for _, row := range []struct {
		label string
		d     []time.Duration
	}{
		{"TTFT miss", ttftMiss},
		{"TTFT hit", ttftHit},
		{"TTFT total", ttftAll},
		{"RespTime miss", rtMiss},
		{"RespTime hit", rtHit},
		{"RespTime total", rtAll},
	} {
		if len(row.d) == 0 {
			sb.WriteString(fmt.Sprintf("  %-12s %10s %10s %10s %10s %10s\n", row.label, "-", "-", "-", "-", "-"))
			continue
		}
		var sum time.Duration
		for _, v := range row.d {
			sum += v
		}
		avg := sum / time.Duration(len(row.d))
		sb.WriteString(fmt.Sprintf("  %-12s %10s %10s %10s %10s %10s\n",
			row.label, fmtDur(avg), fmtDur(row.d[0]), fmtDur(pct(row.d, 0.50)), fmtDur(pct(row.d, 0.95)), fmtDur(pct(row.d, 0.99))))
	}
	sb.WriteString("\n")

	// Token usage — single line
	sb.WriteString(fmt.Sprintf("Tokens: %d input, %d output, %d cached\n", m.TotalInputTokens, m.TotalOutputTokens, m.TotalCachedTokens))
	sb.WriteString("\n")

	return sb.String()
}

// formatComparisonTable creates a comparison table across all models
func formatComparisonTable(models []ModelBenchmarkResult) string {
	var sb strings.Builder

	// Header — all tok/s are system throughput (tokens / wall_time)
	sb.WriteString(fmt.Sprintf("%-30s %-9s %-10s %-10s %-7s %-7s %-7s %-12s %-12s\n",
		"Model", "Reqs o/f", "Min TTFT", "Avg TTFT", "Req/s", "Miss/s", "Hit/s",
		"Prefill t/s", "Decode t/s"))
	sb.WriteString(strings.Repeat("-", 120) + "\n")

	// Data rows
	for _, model := range models {
		successfulRequests := model.TotalRequests - model.FailedRequests
		requestsStr := fmt.Sprintf("%d/%d", successfulRequests, model.FailedRequests)

		allRps, missRps, hitRps, prefillTps, decodeTps := "-", "-", "-", "-", "-"
		if model.WallDuration > 0 && model.Concurrency > 0 {
			wallSecs := model.WallDuration.Seconds()
			conc := float64(model.Concurrency)
			var cacheHitReqs, totalIn, totalOut int
			var sumTTFT, sumDecode float64
			for _, req := range model.Requests {
				if req.Error != nil {
					continue
				}
				totalIn += req.UsageData.InputTokens.Count
				totalOut += req.UsageData.OutputTokens.Count
				ttft := req.TimeToFirstToken.Seconds()
				decode := max(req.TotalResponseTime.Seconds()-ttft, 0)
				sumTTFT += ttft
				sumDecode += decode
				if req.UsageData.CachedTokens.Count > 0 || req.IsImplicitCached {
					cacheHitReqs++
				}
			}
			cacheMissReqs := successfulRequests - cacheHitReqs
			allRps = fmt.Sprintf("%.2f", float64(successfulRequests)/wallSecs)
			if cacheMissReqs > 0 {
				missRps = fmt.Sprintf("%.2f", float64(cacheMissReqs)/wallSecs)
			}
			if cacheHitReqs > 0 {
				hitRps = fmt.Sprintf("%.2f", float64(cacheHitReqs)/wallSecs)
			}
			if sumTTFT > 0 {
				prefillTps = fmt.Sprintf("%.0f", float64(totalIn)/(sumTTFT/conc))
			}
			if sumDecode > 0 {
				decodeTps = fmt.Sprintf("%.0f", float64(totalOut)/(sumDecode/conc))
			}
		}

		sb.WriteString(fmt.Sprintf("%-30s %-9s %-10v %-10v %-7s %-7s %-7s %-12s %-12s\n",
			truncate(model.ModelDisplayName, 30),
			requestsStr,
			model.MinTTFT,
			model.AvgTTFT,
			allRps,
			missRps,
			hitRps,
			prefillTps,
			decodeTps))
	}

	return sb.String()
}

// truncate truncates a string to a maximum length
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// cacheRatio computes actual and expected cache percentages for a model.
// First request per unique series GUID is never expected cached; all repeats
// (from numRequests>1 or subsequent cycles) are expected cached.
func cacheRatio(m ModelBenchmarkResult) map[string]any {
	successful := m.TotalRequests - m.FailedRequests
	actualCached := m.CachedRequests + m.ImplicitCachedRequests
	uniqueSeries := make(map[string]struct{})
	for _, req := range m.Requests {
		if req.Error == nil {
			uniqueSeries[req.SeriesGUID] = struct{}{}
		}
	}
	expectedCached := successful - len(uniqueSeries)
	result := map[string]any{
		"actual_cached":   actualCached,
		"expected_cached": expectedCached,
		"total":           successful,
	}
	if successful > 0 {
		result["actual_pct"] = float64(actualCached) / float64(successful) * 100
		result["expected_pct"] = float64(expectedCached) / float64(successful) * 100
	}
	return result
}

// FormatJSON formats the benchmark results as JSON
func (r *BenchmarkResult) FormatJSON() (string, error) {
	// Create a JSON-friendly structure
	output := map[string]any{
		"documentation":     r.DocsDir,
		"question":          r.Question,
		"num_cycles":        r.NumCycles,
		"num_series":        r.NumSeries,
		"total_duration":    r.TotalDuration.String(),
		"total_duration_ms": r.TotalDuration.Milliseconds(),
		"timed_out":         r.TimedOut,
		"models":            make([]map[string]any, len(r.Models)),
	}

	for i, model := range r.Models {
		modelData := map[string]any{
			"model_name":               model.ModelName,
			"model_display_name":       model.ModelDisplayName,
			"total_requests":           model.TotalRequests,
			"cached_requests":          model.CachedRequests,
			"implicit_cached_requests": model.ImplicitCachedRequests,
			"failed_requests":          model.FailedRequests,
			"ttft": map[string]any{
				"min_ms": model.MinTTFT.Milliseconds(),
				"avg_ms": model.AvgTTFT.Milliseconds(),
				"max_ms": model.MaxTTFT.Milliseconds(),
				"min":    model.MinTTFT.String(),
				"avg":    model.AvgTTFT.String(),
				"max":    model.MaxTTFT.String(),
			},
			"response_time": map[string]any{
				"avg_ms": model.AvgResponseTime.Milliseconds(),
				"min_ms": model.MinResponseTime.Milliseconds(),
				"max_ms": model.MaxResponseTime.Milliseconds(),
				"avg":    model.AvgResponseTime.String(),
				"min":    model.MinResponseTime.String(),
				"max":    model.MaxResponseTime.String(),
			},
			"tokens": map[string]any{
				"input":  model.TotalInputTokens,
				"output": model.TotalOutputTokens,
				"cached": model.TotalCachedTokens,
			},
			"total_cost":          model.TotalCost,
			"wall_duration_ms":    model.WallDuration.Milliseconds(),
			"wall_duration":       model.WallDuration.String(),
			"concurrency":         model.Concurrency,
			"successful_requests": model.TotalRequests - model.FailedRequests,
			"cache_ratio":         cacheRatio(model),
			"requests":            make([]map[string]any, len(model.Requests)),
		}

		// Compute throughput stats (same logic as text formatter)
		if model.WallDuration > 0 && model.Concurrency > 0 {
			type catStats struct {
				reqs, inputTok, outputTok int
				sumTTFT, sumDecode        float64
			}
			var all, hit, miss catStats
			for _, req := range model.Requests {
				if req.Error != nil {
					continue
				}
				ttft := req.TimeToFirstToken.Seconds()
				decode := max(req.TotalResponseTime.Seconds()-ttft, 0)
				s := catStats{1, req.UsageData.InputTokens.Count, req.UsageData.OutputTokens.Count, ttft, decode}
				all.reqs += s.reqs
				all.inputTok += s.inputTok
				all.outputTok += s.outputTok
				all.sumTTFT += s.sumTTFT
				all.sumDecode += s.sumDecode
				if req.UsageData.CachedTokens.Count > 0 || req.IsImplicitCached {
					hit.reqs += s.reqs
					hit.inputTok += s.inputTok
					hit.outputTok += s.outputTok
					hit.sumTTFT += s.sumTTFT
					hit.sumDecode += s.sumDecode
				} else {
					miss.reqs += s.reqs
					miss.inputTok += s.inputTok
					miss.outputTok += s.outputTok
					miss.sumTTFT += s.sumTTFT
					miss.sumDecode += s.sumDecode
				}
			}
			wallSecs := model.WallDuration.Seconds()
			conc := float64(model.Concurrency)

			computeRow := func(s catStats) map[string]any {
				row := map[string]any{
					"requests": s.reqs,
				}
				if s.reqs > 0 {
					row["req_per_sec"] = float64(s.reqs) / wallSecs
					if s.sumTTFT > 0 {
						row["prefill_tok_per_sec"] = float64(s.inputTok) / (s.sumTTFT / conc)
					}
					if s.sumDecode > 0 {
						row["decode_tok_per_sec"] = float64(s.outputTok) / (s.sumDecode / conc)
					}
				}
				return row
			}

			throughput := map[string]any{
				"wall_secs":  wallSecs,
				"cache_miss": computeRow(miss),
				"cache_hit":  computeRow(hit),
				"total":      computeRow(all),
			}

			totalThreadTime := all.sumTTFT + all.sumDecode
			if totalThreadTime > 0 {
				throughput["phase_split"] = map[string]any{
					"prefill_pct":         all.sumTTFT / totalThreadTime * 100,
					"decode_pct":          all.sumDecode / totalThreadTime * 100,
					"prefill_thread_secs": all.sumTTFT,
					"decode_thread_secs":  all.sumDecode,
					"total_thread_secs":   totalThreadTime,
				}
			}

			modelData["throughput"] = throughput
		}

		for j, req := range model.Requests {
			reqData := map[string]any{
				"cycle_num":          req.CycleNum,
				"series_num":         req.SeriesNum,
				"series_guid":        req.SeriesGUID,
				"request_num":        req.RequestNum,
				"ttft_ms":            req.TimeToFirstToken.Milliseconds(),
				"ttft":               req.TimeToFirstToken.String(),
				"response_time_ms":   req.TotalResponseTime.Milliseconds(),
				"response_time":      req.TotalResponseTime.String(),
				"cached_tokens":      req.UsageData.CachedTokens.Count,
				"input_tokens":       req.UsageData.InputTokens.Count,
				"output_tokens":      req.UsageData.OutputTokens.Count,
				"is_implicit_cached": req.IsImplicitCached,
				"success":            req.Error == nil,
			}
			if req.Error != nil {
				reqData["error"] = req.Error.Error()
			}
			modelData["requests"].([]map[string]any)[j] = reqData
		}

		output["models"].([]map[string]any)[i] = modelData
	}

	// Marshal to JSON with indentation
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return string(data), nil
}
