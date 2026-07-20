package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/weka/wekai/llm"
)

// RouterAnalyzeCommand scans capture files and emits per-model analytics.
type RouterAnalyzeCommand struct {
	Format string `short:"f" long:"format" choice:"text" choice:"json" default:"text" description:"Output format: text or json"`
	User   string `long:"user" description:"Filter to a single user (matches the capture's top-level \"user\" field). When set, emits a per-model cost breakdown for just that user."`
	ByUser bool   `long:"by-user" description:"Emit a per-user spend breakdown (USD cost from server-reported usage) instead of / in addition to the per-model cache analysis."`
	Args   struct {
		Path string `positional-arg-name:"path" description:"Directory of *.jsonl files or single .jsonl file" required:"yes"`
	} `positional-args:"yes"`
}

// analyzeRecord represents a normalized capture record for analysis.
type analyzeRecord struct {
	ID          uint64
	Ts          time.Time
	ModelIn     string
	User        string
	Request     redactedRequest
	Response    redactedResponse
	IsPlainJSON bool
}

// modelStats holds aggregated statistics for a single model.
type modelStats struct {
	Model             string
	RequestCount      int
	InputTokens       int
	OutputTokens      int
	CacheReadTokens   int
	CacheCreateTokens int
	Records           []analyzeRecord
}

// modelAnalysisResult holds the final calculated results.
type modelAnalysisResult struct {
	Model                    string
	RequestCount             int
	InputTokens              int
	OutputTokens             int
	PrefixCacheTokens        int64
	TotalInputTokens         int64
	CacheHitRates            map[string]float64
	InfiniteRetentionHitRate float64
}

type captureRecordRaw struct {
	ID       uint64      `json:"id"`
	Ts       string      `json:"ts"`
	ModelIn  string      `json:"model_in"`
	User     string      `json:"user"`
	Request  captureBody `json:"request"`
	Response captureBody `json:"response"`
}

func (c *RouterAnalyzeCommand) Execute(args []string) error {
	path := c.Args.Path
	files, err := c.collectFiles(path)
	if err != nil {
		return err
	}

	var records []analyzeRecord
	rawCount, redactedCount, plainCount := 0, 0, 0
	var minTime, maxTime time.Time

	for _, f := range files {
		r, raw, red, plain, err := c.parseFile(f)
		if err != nil {
			return fmt.Errorf("parse %s: %w", f, err)
		}
		records = append(records, r...)
		rawCount += raw
		redactedCount += red
		plainCount += plain
		for _, rec := range r {
			if minTime.IsZero() || rec.Ts.Before(minTime) {
				minTime = rec.Ts
			}
			if rec.Ts.After(maxTime) {
				maxTime = rec.Ts
			}
		}
	}

	if len(records) == 0 {
		return fmt.Errorf("no records found")
	}

	// Per-user spend path: --user (filter to one user) or --by-user (all users).
	if c.User != "" || c.ByUser {
		users := c.aggregateUsers(records, c.User)
		if c.Format == "json" {
			return c.outputUsersJSON(users)
		}
		return c.outputUsersText(users, c.User)
	}

	byModel := c.groupByModel(records)
	retentionWindows := []time.Duration{5 * time.Minute, 15 * time.Minute, 30 * time.Minute, time.Hour, 6 * time.Hour}
	results := make(map[string]*modelAnalysisResult)
	for model, stats := range byModel {
		results[model] = c.analyzeModel(stats, retentionWindows)
	}

	if c.Format == "json" {
		return c.outputJSON(results, files, len(records), plainCount, minTime, maxTime, rawCount, redactedCount)
	}
	return c.outputText(results, files, len(records), plainCount, minTime, maxTime, rawCount, redactedCount)
}

func (c *RouterAnalyzeCommand) collectFiles(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		if !strings.HasSuffix(path, ".jsonl") {
			return nil, fmt.Errorf("expected .jsonl file or directory: %s", path)
		}
		return []string{path}, nil
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			files = append(files, filepath.Join(path, e.Name()))
		}
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no .jsonl files found in %s", path)
	}
	sort.Strings(files)
	return files, nil
}

func (c *RouterAnalyzeCommand) parseFile(path string) ([]analyzeRecord, int, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer f.Close()

	var records []analyzeRecord
	rawCount, redactedCount, plainCount := 0, 0, 0
	// Use bufio.NewReader + ReadBytes('\n') because raw capture lines can
	// exceed bufio.Scanner's line-size cap. Same approach as
	// command_router_tree.go and command_router_redact.go.
	br := bufio.NewReaderSize(f, 1<<20)

	for {
		line, readErr := br.ReadBytes('\n')
		if len(line) > 0 && line[len(line)-1] == '\n' {
			line = line[:len(line)-1]
		}
		if len(line) == 0 {
			if readErr == io.EOF {
				return records, rawCount, redactedCount, plainCount, nil
			}
			if readErr != nil {
				return records, rawCount, redactedCount, plainCount, readErr
			}
			continue
		}
		var rec captureRecordRaw
		if err := json.Unmarshal(line, &rec); err != nil {
			if readErr == io.EOF {
				return records, rawCount, redactedCount, plainCount, nil
			}
			if readErr != nil {
				return records, rawCount, redactedCount, plainCount, readErr
			}
			continue
		}
		ts, _ := time.Parse(time.RFC3339Nano, rec.Ts)
		var req redactedRequest
		var resp redactedResponse
		isPlain := false

		var reqProbe map[string]interface{}
		isRedacted := false
		if err := json.Unmarshal(rec.Request.Body, &reqProbe); err == nil {
			if schema, ok := reqProbe["_schema"].(string); ok && schema == "req-v1" {
				isRedacted = true
			}
		}

		if isRedacted {
			redactedCount++
			json.Unmarshal(rec.Request.Body, &req)
			json.Unmarshal(rec.Response.Body, &resp)
			if resp.PlainJSON != nil {
				isPlain = true
			}
		} else {
			rawCount++
			var reqBodyStr, respBodyStr string
			_ = json.Unmarshal(rec.Request.Body, &reqBodyStr)
			_ = json.Unmarshal(rec.Response.Body, &respBodyStr)
			// BuildRedactedPair runs token allocation using the response's
			// usage, so analyzing a raw file gives per-block token counts
			// byte-identical to what the live --capture redacted path would
			// produce.
			reqRaw, respRaw := BuildRedactedPair([]byte(reqBodyStr), []byte(respBodyStr))
			_ = json.Unmarshal(reqRaw, &req)
			_ = json.Unmarshal(respRaw, &resp)
		}

		if resp.PlainJSON != nil {
			isPlain = true
			plainCount++
		}
		records = append(records, analyzeRecord{
			ID: rec.ID, Ts: ts, ModelIn: rec.ModelIn, User: rec.User,
			Request: req, Response: resp, IsPlainJSON: isPlain,
		})
		if readErr == io.EOF {
			return records, rawCount, redactedCount, plainCount, nil
		}
		if readErr != nil {
			return records, rawCount, redactedCount, plainCount, readErr
		}
	}
}

// userModelSpend holds one user's spend on one model.
type userModelSpend struct {
	Model        string
	Requests     int
	InputTokens  int
	OutputTokens int
	CacheRead    int
	CacheCreate  int
	Cost         float64
	Priced       bool // false => model not in registry, cost is 0/unknown
}

// userSpend holds a single user's aggregated spend across all models.
type userSpend struct {
	User     string
	Requests int
	Cost     float64
	ByModel  map[string]*userModelSpend
}

// recordCost computes the USD cost of one record from its server-reported
// usage, using registry pricing for the request's model_in. Returns
// (cost, priced) where priced is false when the model has no registry entry
// (cost is then 0 and the caller should surface it as "unpriced").
func recordCost(r analyzeRecord) (float64, bool) {
	if r.Response.Usage == nil {
		return 0, true
	}
	if llm.LookupModelByIdentifier == nil {
		// No model registry hook registered (see llm.LookupModelByIdentifier) —
		// cost is unknown, not zero-priced, so report unpriced rather than $0.
		return 0, false
	}
	info, ok := llm.LookupModelByIdentifier(r.ModelIn)
	if !ok {
		return 0, false
	}
	usage := llm.Usage{
		InputTokens:              r.Response.Usage.InputTokens,
		CacheReadInputTokens:     r.Response.Usage.CacheReadInputTokens,
		CacheCreationInputTokens: r.Response.Usage.CacheCreationInputTokens,
		OutputTokens:             r.Response.Usage.OutputTokens,
	}
	return llm.CalculateCost(info, usage), true
}

// aggregateUsers builds per-user spend from server-reported usage. When
// userFilter is non-empty, only that user's records are included. count_tokens
// (PlainJSON) records carry no billable usage and are skipped.
func (c *RouterAnalyzeCommand) aggregateUsers(records []analyzeRecord, userFilter string) map[string]*userSpend {
	users := make(map[string]*userSpend)
	for _, r := range records {
		if r.IsPlainJSON {
			continue
		}
		user := r.User
		if user == "" {
			user = "(unknown)"
		}
		if userFilter != "" && user != userFilter {
			continue
		}
		u, ok := users[user]
		if !ok {
			u = &userSpend{User: user, ByModel: make(map[string]*userModelSpend)}
			users[user] = u
		}
		cost, priced := recordCost(r)
		u.Requests++
		u.Cost += cost

		ms, ok := u.ByModel[r.ModelIn]
		if !ok {
			ms = &userModelSpend{Model: r.ModelIn, Priced: true}
			u.ByModel[r.ModelIn] = ms
		}
		ms.Requests++
		ms.Cost += cost
		if !priced {
			ms.Priced = false
		}
		if r.Response.Usage != nil {
			ms.InputTokens += r.Response.Usage.InputTokens
			ms.OutputTokens += r.Response.Usage.OutputTokens
			ms.CacheRead += r.Response.Usage.CacheReadInputTokens
			ms.CacheCreate += r.Response.Usage.CacheCreationInputTokens
		}
	}
	return users
}

func (c *RouterAnalyzeCommand) outputUsersText(users map[string]*userSpend, userFilter string) error {
	var list []*userSpend
	for _, u := range users {
		list = append(list, u)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Cost > list[j].Cost })

	if userFilter != "" {
		if len(list) == 0 {
			fmt.Printf("No records found for user %q\n", userFilter)
			return nil
		}
		u := list[0]
		fmt.Printf("Spend for user %q\n", u.User)
		fmt.Printf("  requests:   %s\n", formatNumber(u.Requests))
		fmt.Printf("  total cost: $%.2f\n\n", u.Cost)
		c.printUserModelTable(u)
		return nil
	}

	var grandTotal float64
	for _, u := range list {
		grandTotal += u.Cost
	}
	fmt.Println("Per-user spend (USD, from server-reported usage)")
	fmt.Printf("  %-28s %10s %14s\n", "user", "requests", "cost")
	for _, u := range list {
		fmt.Printf("  %-28s %10s %13s\n", u.User, formatNumber(u.Requests), "$"+formatFloat(u.Cost))
	}
	fmt.Printf("  %-28s %10s %13s\n", "TOTAL", "", "$"+formatFloat(grandTotal))
	return nil
}

func (c *RouterAnalyzeCommand) printUserModelTable(u *userSpend) {
	var models []*userModelSpend
	for _, m := range u.ByModel {
		models = append(models, m)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Cost > models[j].Cost })
	fmt.Println("  Per-model breakdown:")
	fmt.Printf("    %-32s %8s %13s %13s %13s %12s\n", "model", "reqs", "input", "cache_read", "output", "cost")
	for _, m := range models {
		cost := "$" + formatFloat(m.Cost)
		if !m.Priced {
			cost = "(unpriced)"
		}
		name := m.Model
		if name == "" {
			name = "(no model)"
		}
		fmt.Printf("    %-32s %8s %13s %13s %13s %12s\n",
			name, formatNumber(m.Requests),
			formatNumber(m.InputTokens), formatNumber(m.CacheRead),
			formatNumber(m.OutputTokens), cost)
	}
}

func (c *RouterAnalyzeCommand) outputUsersJSON(users map[string]*userSpend) error {
	type jsonModel struct {
		Model        string  `json:"model"`
		Requests     int     `json:"requests"`
		InputTokens  int     `json:"input_tokens"`
		OutputTokens int     `json:"output_tokens"`
		CacheRead    int     `json:"cache_read_tokens"`
		CacheCreate  int     `json:"cache_creation_tokens"`
		Cost         float64 `json:"cost_usd"`
		Priced       bool    `json:"priced"`
	}
	type jsonUser struct {
		User     string      `json:"user"`
		Requests int         `json:"requests"`
		Cost     float64     `json:"cost_usd"`
		Models   []jsonModel `json:"models"`
	}
	var out []jsonUser
	for _, u := range users {
		ju := jsonUser{User: u.User, Requests: u.Requests, Cost: u.Cost}
		for _, m := range u.ByModel {
			ju.Models = append(ju.Models, jsonModel{
				Model: m.Model, Requests: m.Requests,
				InputTokens: m.InputTokens, OutputTokens: m.OutputTokens,
				CacheRead: m.CacheRead, CacheCreate: m.CacheCreate,
				Cost: m.Cost, Priced: m.Priced,
			})
		}
		sort.Slice(ju.Models, func(i, j int) bool { return ju.Models[i].Cost > ju.Models[j].Cost })
		out = append(out, ju)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Cost > out[j].Cost })
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(struct {
		Users []jsonUser `json:"users"`
	}{Users: out})
}

// formatFloat renders a USD amount with thousands separators and 2 decimals.
func formatFloat(f float64) string {
	whole := int(f)
	frac := int((f-float64(whole))*100 + 0.5)
	if frac == 100 {
		whole++
		frac = 0
	}
	return fmt.Sprintf("%s.%02d", formatNumber(whole), frac)
}

func (c *RouterAnalyzeCommand) groupByModel(records []analyzeRecord) map[string]*modelStats {
	byModel := make(map[string]*modelStats)
	for _, r := range records {
		if r.IsPlainJSON {
			continue
		}
		s, ok := byModel[r.ModelIn]
		if !ok {
			s = &modelStats{Model: r.ModelIn}
			byModel[r.ModelIn] = s
		}
		s.RequestCount++
		s.Records = append(s.Records, r)
		if r.Response.Usage != nil {
			s.InputTokens += r.Response.Usage.InputTokens + r.Response.Usage.CacheReadInputTokens + r.Response.Usage.CacheCreationInputTokens
			s.OutputTokens += r.Response.Usage.OutputTokens
			s.CacheReadTokens += r.Response.Usage.CacheReadInputTokens
			s.CacheCreateTokens += r.Response.Usage.CacheCreationInputTokens
		}
	}
	return byModel
}

func (c *RouterAnalyzeCommand) analyzeModel(stats *modelStats, windows []time.Duration) *modelAnalysisResult {
	sort.Slice(stats.Records, func(i, j int) bool {
		return stats.Records[i].Ts.Before(stats.Records[j].Ts)
	})

	result := &modelAnalysisResult{
		Model:         stats.Model,
		RequestCount:  stats.RequestCount,
		InputTokens:   stats.InputTokens,
		OutputTokens:  stats.OutputTokens,
		CacheHitRates: make(map[string]float64),
	}

	infiniteCached, infiniteTotal := c.simulateCache(stats.Records, 0)
	result.TotalInputTokens = infiniteTotal
	if infiniteTotal > 0 {
		result.InfiniteRetentionHitRate = float64(infiniteCached) / float64(infiniteTotal)
		result.PrefixCacheTokens = infiniteCached
	}

	for _, window := range windows {
		cached, total := c.simulateCache(stats.Records, window)
		if total > 0 {
			result.CacheHitRates[window.String()] = float64(cached) / float64(total)
		}
	}

	return result
}

// blockTokens prefers the pre-allocated per-block Tokens count (populated
// by BuildRedactedPair using the response's real usage) and falls back
// to bytes/4 for records built without a response pairing (old schema).
func blockTokens(bytes, tokens int) int64 {
	if tokens > 0 {
		return int64(tokens)
	}
	return int64((bytes + 3) / 4)
}

func (c *RouterAnalyzeCommand) computeInputTokens(req redactedRequest) int64 {
	var total int64
	for _, m := range req.Messages {
		total += blockTokens(m.Bytes, m.Tokens)
	}
	for _, s := range req.SystemBlocks {
		total += blockTokens(s.Bytes, s.Tokens)
	}
	if req.Tools != nil {
		total += blockTokens(req.Tools.Bytes, req.Tools.Tokens)
	}
	return total
}

// simulateCache returns (cachedTokens, totalTokens) for a set of records
// under a retention window. Match runs on messages[] prefix only; system
// and tools bytes/tokens are credited as cached whenever any prior
// request exists within the window (they're constant in a session).
func (c *RouterAnalyzeCommand) simulateCache(records []analyzeRecord, retention time.Duration) (cachedTokens, totalTokens int64) {
	type priorEntry struct {
		ts            time.Time
		messageHashes []string
	}
	var priors []priorEntry

	for _, rec := range records {
		var curHashes []string
		var curMsgTokens []int64
		for _, m := range rec.Request.Messages {
			curHashes = append(curHashes, m.Hash)
			curMsgTokens = append(curMsgTokens, blockTokens(m.Bytes, m.Tokens))
		}
		totalTokens += c.computeInputTokens(rec.Request)

		var windowStart time.Time
		if retention > 0 {
			windowStart = rec.Ts.Add(-retention)
		}

		bestLCP := 0
		hasPrior := false
		for _, p := range priors {
			if retention > 0 && p.ts.Before(windowStart) {
				continue
			}
			hasPrior = true
			lcp := c.longestCommonPrefix(curHashes, p.messageHashes)
			if lcp > bestLCP {
				bestLCP = lcp
			}
		}
		for i := 0; i < bestLCP; i++ {
			cachedTokens += curMsgTokens[i]
		}
		if hasPrior {
			for _, s := range rec.Request.SystemBlocks {
				cachedTokens += blockTokens(s.Bytes, s.Tokens)
			}
			if rec.Request.Tools != nil {
				cachedTokens += blockTokens(rec.Request.Tools.Bytes, rec.Request.Tools.Tokens)
			}
		}

		priors = append(priors, priorEntry{ts: rec.Ts, messageHashes: curHashes})
	}

	return cachedTokens, totalTokens
}

func (c *RouterAnalyzeCommand) longestCommonPrefix(a, b []string) int {
	max := len(a)
	if len(b) < max {
		max = len(b)
	}
	for i := 0; i < max; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return max
}

func (c *RouterAnalyzeCommand) outputText(results map[string]*modelAnalysisResult, files []string, totalRecs, plainCount int, minTime, maxTime time.Time, rawCount, redactedCount int) error {
	fmt.Println("Router capture analysis")
	fmt.Printf("  files:        %d (%s)\n", len(files), filepath.Base(files[0]))
	streamedCount := totalRecs - plainCount
	fmt.Printf("  records:      %d (%d streamed, %d count_tokens)\n", totalRecs, streamedCount, plainCount)
	duration := maxTime.Sub(minTime)
	fmt.Printf("  time range:   %s -> %s (%s)\n", minTime.Format("2006-01-02 15:04:05Z"), maxTime.Format("2006-01-02 15:04:05Z"), duration.Round(time.Minute))
	fmt.Printf("  format mix:   %d raw, %d redacted (converted on fly)\n", rawCount, redactedCount)

	var models []string
	for m := range results {
		models = append(models, m)
	}
	sort.Strings(models)

	fmt.Println("\nPer-model breakdown:")
	for _, model := range models {
		r := results[model]
		if r.RequestCount == 0 {
			continue
		}
		fmt.Printf("\n  %s (%d requests)\n", model, r.RequestCount)
		fmt.Printf("    Input tokens (incl. cached):   %s\n", formatNumber(r.InputTokens))
		fmt.Printf("    Output tokens:                    %s\n", formatNumber(r.OutputTokens))
		// Ratio is tokens-based and internally consistent. For the
		// absolute number we multiply the authoritative server-reported
		// total (r.InputTokens) by the ratio so the display never shows
		// cache-ready > total — per-block Tokens can drift on records
		// whose usage couldn't be parsed (they fall back to bytes/4).
		pct := 0.0
		if r.TotalInputTokens > 0 {
			pct = 100.0 * float64(r.PrefixCacheTokens) / float64(r.TotalInputTokens)
		}
		prefixTokens := int64(float64(r.InputTokens) * pct / 100.0)
		fmt.Printf("    Prefix cache-ready input:      %s tokens (%.2f%%)\n", formatNumber(int(prefixTokens)), pct)
		fmt.Println("    Expected cache hit rate by retention window:")

		retentionOrder := []string{"5m0s", "15m0s", "30m0s", "1h0m0s", "6h0m0s"}
		displayNames := []string{"5m", "15m", "30m", "1h", "6h"}
		for i := range retentionOrder {
			if i == 0 {
				fmt.Printf("      %-9s", displayNames[i])
			} else {
				fmt.Printf("%-10s", displayNames[i])
			}
		}
		fmt.Println()
		for i, key := range retentionOrder {
			rate := r.CacheHitRates[key]
			if i == 0 {
				fmt.Printf("      %.2f%%    ", rate*100)
			} else {
				fmt.Printf("%.2f%%     ", rate*100)
			}
		}
		fmt.Println()
	}

	return nil
}

func formatNumber(n int) string {
	s := fmt.Sprintf("%d", n)
	var result []byte
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			result = append(result, ',')
		}
		result = append(result, byte(c))
	}
	return string(result)
}

func (c *RouterAnalyzeCommand) outputJSON(results map[string]*modelAnalysisResult, files []string, totalRecs, plainCount int, minTime, maxTime time.Time, rawCount, redactedCount int) error {
	type jsonRetention struct {
		Window  string  `json:"window"`
		Seconds float64 `json:"seconds"`
		HitRate float64 `json:"hit_rate"`
	}
	type jsonModelResult struct {
		RequestCount             int             `json:"request_count"`
		InputTokens              int             `json:"input_tokens"`
		OutputTokens             int             `json:"output_tokens"`
		PrefixCacheTokens        int64           `json:"prefix_cache_tokens"`
		TotalInputTokens         int64           `json:"total_input_tokens"`
		PrefixCachePercent       float64         `json:"prefix_cache_percent"`
		InfiniteRetentionHitRate float64         `json:"infinite_retention_hit_rate"`
		RetentionWindows         []jsonRetention `json:"retention_windows"`
	}

	output := struct {
		Summary struct {
			Files        []string `json:"files"`
			TotalRecords int      `json:"total_records"`
			Streamed     int      `json:"streamed"`
			CountTokens  int      `json:"count_tokens"`
			TimeRange    struct {
				Start string `json:"start"`
				End   string `json:"end"`
			} `json:"time_range"`
			FormatMix struct {
				Raw      int `json:"raw"`
				Redacted int `json:"redacted"`
			} `json:"format_mix"`
		} `json:"summary"`
		Models map[string]jsonModelResult `json:"models"`
	}{}

	output.Summary.Files = files
	output.Summary.TotalRecords = totalRecs
	output.Summary.Streamed = totalRecs - plainCount
	output.Summary.CountTokens = plainCount
	output.Summary.TimeRange.Start = minTime.Format(time.RFC3339)
	output.Summary.TimeRange.End = maxTime.Format(time.RFC3339)
	output.Summary.FormatMix.Raw = rawCount
	output.Summary.FormatMix.Redacted = redactedCount
	output.Models = make(map[string]jsonModelResult)

	for model, r := range results {
		pct := 0.0
		if r.TotalInputTokens > 0 {
			pct = 100.0 * float64(r.PrefixCacheTokens) / float64(r.TotalInputTokens)
		}
		jr := jsonModelResult{
			RequestCount:             r.RequestCount,
			InputTokens:              r.InputTokens,
			OutputTokens:             r.OutputTokens,
			PrefixCacheTokens:        r.PrefixCacheTokens,
			TotalInputTokens:         r.TotalInputTokens,
			PrefixCachePercent:       pct,
			InfiniteRetentionHitRate: r.InfiniteRetentionHitRate,
		}

		retentionOrder := []time.Duration{5 * time.Minute, 15 * time.Minute, 30 * time.Minute, time.Hour, 6 * time.Hour}
		for _, window := range retentionOrder {
			if rate, ok := r.CacheHitRates[window.String()]; ok {
				jr.RetentionWindows = append(jr.RetentionWindows, jsonRetention{
					Window:  window.String(),
					Seconds: window.Seconds(),
					HitRate: rate,
				})
			}
		}

		output.Models[model] = jr
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(output)
}
