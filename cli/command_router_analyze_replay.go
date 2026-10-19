package cli

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/weka/wekai/benchmark"
)

// RouterAnalyzeReplayCommand simulates a replay or source capture offline
// and reports expected cache hit ratios per model.
type RouterAnalyzeReplayCommand struct {
	Format string `long:"format" choice:"text" choice:"json" default:"text" description:"Output format"`

	// Retention is the TTLs to score besides the infinite cache.
	Retention string `long:"retention" default:"5m,15m,30m,1h,2h,4h,8h,12h,1d,2d,3d,5d,7d" description:"Comma-separated cache TTLs to simulate (units s, m, h, d), each refreshed whenever cached content is used; an empty value skips the retention breakdown"`

	// --- Prefill/decode split estimate. Opt-in: with none of these set,
	// output is byte-identical to the plain cache-ratio report above.
	PrefillSplit            bool  `long:"prefill-split" description:"Additionally estimate the share of traffic and input tokens that would route through a dedicated prefill instance under dynamic prefill/decode segregation (see 'wekai router serve --prefill-split'). Opt-in: off by default, and adds an extra report section rather than changing the one above."`
	PrefillMinMissingBlocks int   `long:"prefill-min-missing-blocks" default:"32" description:"Same meaning and default as 'wekai router serve --prefill-min-missing-blocks': cold depth, in router blocks (~256 estimated tokens each), strictly above which a request is modeled as taking the prefill path. The sensitivity table below scores several thresholds regardless, which is the point of the report."`
	PrefillCacheMaxTokens   int64 `long:"prefill-cache-max-tokens" description:"Bound the simulated fleet-wide router-block cache to this many estimated tokens, modeling eviction (0 = unbounded/infinite retention, the default 'perfect affinity' lower bound). kvcache.RouterConfig()'s per-backend bound is 2,000,000 — a fleet-wide figure should scale that by node count."`

	contextSize *contextSizeReport
	weekly      *weeklyReport
	volume      *volumeReport

	Args struct {
		Path string `positional-arg-name:"PATH" description:"replay-v3 file, or source redacted-capture file/dir"`
	} `positional-args:"yes" required:"yes"`
}

// replayRequestWithModel pairs a RouterReplayRequest with its model name for
// grouping.
type replayRequestWithModel struct {
	req   benchmark.RouterReplayRequest
	model string
}

// modelReport pairs a model name with its cache simulation report.
type modelReport struct {
	Model     string
	Report    benchmark.ReplayCacheReport
	Retention benchmark.RetentionReport
}

func (c *RouterAnalyzeReplayCommand) Execute(args []string) error {
	path := c.Args.Path
	windows, err := benchmark.ParseRetentionWindows(c.Retention)
	if err != nil {
		return err
	}

	isReplay, err := c.isReplayV3(path)
	if err != nil {
		return fmt.Errorf("cannot read input: %w", err)
	}

	// The cache simulations read only each request's prefix units, time and
	// token counts, held compactly in a PrefixStore; full requests would keep
	// every turn's whole history and do not fit for a long capture.
	var store *benchmark.PrefixStore
	if isReplay {
		store, err = c.loadReplayStore(path)
	} else {
		store, err = c.loadSourceStore(path)
	}
	if err != nil {
		return err
	}

	if len(store.Reqs) == 0 {
		fmt.Println("No valid requests found.")
		return nil
	}

	// Chronological order, then grouped by model.
	byModel := map[string][]int32{}
	ctxByModel := map[string][]int{}
	for _, i := range store.ChronologicalOrder() {
		r := &store.Reqs[i]
		byModel[r.Model] = append(byModel[r.Model], i)
		ctxByModel[r.Model] = append(ctxByModel[r.Model], r.InputTokens)
	}
	models := make([]string, 0, len(byModel))
	for m := range byModel {
		models = append(models, m)
	}
	sort.Strings(models)

	c.contextSize = buildContextSizeReport(models, ctxByModel)
	c.weekly = buildWeeklyReport(store, isReplay)
	c.volume = buildVolumeReport(store)

	// Compute per-model + overall reports. Retention is scored per model
	// only: a cache never serves one model's KV to another, so its total is
	// the sum of the models'.
	var reports []modelReport
	var allReqs []int32
	var overallRetention benchmark.RetentionReport
	for _, m := range models {
		group := byModel[m]
		allReqs = append(allReqs, group...)
		r := modelReport{Model: m}
		done := make(chan struct{})
		go func() {
			defer close(done)
			if len(windows) > 0 {
				r.Retention = benchmark.SimulateRetentionStore(store, group, windows)
			}
		}()
		r.Report = benchmark.SimulateReplayCacheStore(store, group)
		<-done
		overallRetention.Add(r.Retention)
		reports = append(reports, r)
	}
	overall := benchmark.SimulateReplayCacheStore(store, allReqs)
	if len(windows) == 0 {
		overallRetention = benchmark.RetentionReport{}
	}

	var prefill *prefillSplitOutput
	if c.PrefillSplit {
		// The prefill estimate synthesises request content, so it needs the
		// full requests: loaded only for it, after the store is done with.
		store, byModel, allReqs = nil, nil, nil
		full, err := c.loadFullByModel(path, isReplay)
		if err != nil {
			return err
		}
		var perr error
		prefill, perr = c.computePrefillSplit(models, full)
		if perr != nil {
			return perr
		}
	}

	if c.Format == "json" {
		return c.outputJSON(reports, overall, overallRetention, prefill)
	}
	return c.outputText(reports, overall, overallRetention, prefill)
}

// loadFullByModel loads full requests grouped by model, each group in
// chronological order.
func (c *RouterAnalyzeReplayCommand) loadFullByModel(path string, isReplay bool) (map[string][]benchmark.RouterReplayRequest, error) {
	var reqs []replayRequestWithModel
	var err error
	if isReplay {
		reqs, err = c.collectReplayV3(path)
	} else {
		reqs, err = c.collectSourceCapture(path)
	}
	if err != nil {
		return nil, err
	}
	sort.SliceStable(reqs, func(i, j int) bool { return reqs[i].req.Ts < reqs[j].req.Ts })
	byModel := map[string][]benchmark.RouterReplayRequest{}
	for _, r := range reqs {
		byModel[r.model] = append(byModel[r.model], r.req)
	}
	return byModel, nil
}

// loadSourceStore builds the store from source capture records; they carry
// no instance structure, so each request stands alone.
func (c *RouterAnalyzeReplayCommand) loadSourceStore(path string) (*benchmark.PrefixStore, error) {
	reqs, err := c.collectSourceCapture(path)
	if err != nil {
		return nil, err
	}
	store := &benchmark.PrefixStore{}
	for _, r := range reqs {
		var b benchmark.PrefixBatch
		r.req.Model = r.model
		b.AddChain([]benchmark.RouterReplayRequest{r.req})
		store.Append(&b)
	}
	return store, nil
}

// loadReplayStore reads a replay file into a store, decoding session lines on
// all cores; sessions are appended in file order, so the result does not
// depend on scheduling.
func (c *RouterAnalyzeReplayCommand) loadReplayStore(path string) (*benchmark.PrefixStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)

	line, err := br.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("read header line: %w", err)
	}
	var hdr benchmark.RouterReplayHeader
	if err := json.Unmarshal(bytes.TrimSuffix(line, []byte("\n")), &hdr); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}
	if !benchmark.SupportedReplaySchema(hdr.Schema) {
		return nil, fmt.Errorf("expected replay-v3, replay-v4 or replay-v5, got %q", hdr.Schema)
	}

	type job struct {
		seq  int
		line []byte
	}
	type result struct {
		seq   int
		batch *benchmark.PrefixBatch
	}
	workers := runtime.NumCPU()
	jobs := make(chan job, workers)
	results := make(chan result, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				b := &benchmark.PrefixBatch{}
				var sess benchmark.RouterReplaySession
				if err := json.Unmarshal(j.line, &sess); err == nil {
					for _, inst := range sess.Instances {
						b.AddChain(inst.Requests)
					}
				}
				results <- result{j.seq, b}
			}
		}()
	}

	var readErr error
	go func() {
		defer func() { close(jobs); wg.Wait(); close(results) }()
		for seq := 0; ; {
			line, err := br.ReadBytes('\n')
			line = bytes.TrimSuffix(line, []byte("\n"))
			if len(line) > 0 && !benchmark.IsReplayIndexLine(line) {
				jobs <- job{seq, line}
				seq++
			}
			if err != nil {
				if err != io.EOF {
					readErr = err
				}
				return
			}
		}
	}()

	store := &benchmark.PrefixStore{}
	pending := map[int]*benchmark.PrefixBatch{}
	next := 0
	for r := range results {
		pending[r.seq] = r.batch
		for b, ok := pending[next]; ok; b, ok = pending[next] {
			store.Append(b)
			delete(pending, next)
			next++
		}
	}
	return store, readErr
}

// isReplayV3 peeks at line 1 of the file (or the first .jsonl file in a dir)
// to determine whether the input is a replay-v3 dataset.
func (c *RouterAnalyzeReplayCommand) isReplayV3(path string) (bool, error) {
	info, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	filePath := path
	if info.IsDir() {
		files, err := collectJSONLFiles(path)
		if err != nil {
			return false, err
		}
		if len(files) == 0 {
			return false, fmt.Errorf("no .jsonl files in %s", path)
		}
		filePath = files[0]
	}

	f, err := os.Open(filePath)
	if err != nil {
		return false, err
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)
	line, readErr := br.ReadBytes('\n')
	if readErr != nil && len(line) == 0 {
		return false, readErr
	}
	if len(line) > 0 && line[len(line)-1] == '\n' {
		line = line[:len(line)-1]
	}
	if len(line) == 0 {
		return false, nil
	}

	// Try replay-v3 header first.
	var hdr benchmark.RouterReplayHeader
	if json.Unmarshal(line, &hdr) == nil && benchmark.SupportedReplaySchema(hdr.Schema) {
		return true, nil
	}
	return false, nil
}

// collectReplayV3 reads a replay-v3 JSONL file and collects all RouterReplayRequest.
func (c *RouterAnalyzeReplayCommand) collectReplayV3(path string) ([]replayRequestWithModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)

	// Line 1: header.
	line, readErr := br.ReadBytes('\n')
	if readErr != nil {
		return nil, fmt.Errorf("read header line: %w", readErr)
	}
	if len(line) > 0 && line[len(line)-1] == '\n' {
		line = line[:len(line)-1]
	}
	var hdr benchmark.RouterReplayHeader
	if err := json.Unmarshal(line, &hdr); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}
	if !benchmark.SupportedReplaySchema(hdr.Schema) {
		return nil, fmt.Errorf("expected replay-v3, replay-v4 or replay-v5, got %q", hdr.Schema)
	}

	var out []replayRequestWithModel
	for {
		line, readErr := br.ReadBytes('\n')
		if len(line) > 0 {
			if line[len(line)-1] == '\n' {
				line = line[:len(line)-1]
			}
			if len(line) > 0 && !benchmark.IsReplayIndexLine(line) {
				var sess benchmark.RouterReplaySession
				if err := json.Unmarshal(line, &sess); err == nil {
					for _, inst := range sess.Instances {
						for _, req := range inst.Requests {
							out = append(out, replayRequestWithModel{req: req, model: req.Model})
						}
					}
				}
			}
		}
		if readErr == io.EOF {
			return out, nil
		}
		if readErr != nil {
			return out, readErr
		}
	}
}

// collectSourceCapture reads source redacted-capture file(s) and builds
// RouterReplayRequest entries for every request/response pair with Usage.
func (c *RouterAnalyzeReplayCommand) collectSourceCapture(path string) ([]replayRequestWithModel, error) {
	files, err := collectJSONLFiles(path)
	if err != nil {
		return nil, err
	}

	var out []replayRequestWithModel
	for _, f := range files {
		recs, err := c.parseSourceFile(f)
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", f, err)
		}
		out = append(out, recs...)
	}
	return out, nil
}

func (c *RouterAnalyzeReplayCommand) parseSourceFile(path string) ([]replayRequestWithModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []replayRequestWithModel
	br := bufio.NewReaderSize(f, 1<<20)

	for {
		line, readErr := br.ReadBytes('\n')
		if len(line) > 0 && line[len(line)-1] == '\n' {
			line = line[:len(line)-1]
		}
		if len(line) == 0 {
			if readErr == io.EOF {
				return out, nil
			}
			if readErr != nil {
				return out, readErr
			}
			continue
		}

		var raw captureRecordRaw
		if err := json.Unmarshal(line, &raw); err != nil {
			if readErr == io.EOF {
				return out, nil
			}
			if readErr != nil {
				return out, readErr
			}
			continue
		}

		// Parse redacted request and response bodies.
		var req redactedRequest
		var resp redactedResponse
		if perr := json.Unmarshal(raw.Request.Body, &req); perr != nil {
			if readErr == io.EOF {
				return out, nil
			}
			if readErr != nil {
				return out, readErr
			}
			continue
		}
		if perr := json.Unmarshal(raw.Response.Body, &resp); perr != nil {
			if readErr == io.EOF {
				return out, nil
			}
			if readErr != nil {
				return out, readErr
			}
			continue
		}

		// Skip count_tokens endpoint records (PlainJSON present).
		if resp.PlainJSON != nil {
			if readErr == io.EOF {
				return out, nil
			}
			if readErr != nil {
				return out, readErr
			}
			continue
		}

		// Skip records without usage.
		if resp.Usage == nil {
			if readErr == io.EOF {
				return out, nil
			}
			if readErr != nil {
				return out, readErr
			}
			continue
		}

		// Map into RouterReplayRequest using buildReplayRequest as reference.
		// Source-capture total = input + cache_read + cache_creation (distinct
		// from replay-v3 where InputTokens is already total).
		rr := benchmark.RouterReplayRequest{
			RequestID:           raw.ID,
			Ts:                  raw.Ts,
			Model:               raw.ModelIn,
			MaxTokens:           req.MaxTokens,
			Stream:              req.Stream,
			InputTokens:         resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens,
			PrefillTokens:       resp.Usage.InputTokens,
			CacheReadTokens:     resp.Usage.CacheReadInputTokens,
			CacheCreationTokens: resp.Usage.CacheCreationInputTokens,
			OutputTokens:        resp.Usage.OutputTokens,
			Temperature:         req.Temperature,
			TopP:                req.TopP,
			TopK:                req.TopK,
			Thinking:            req.Thinking,
			StopReason:          resp.StopReason,
		}
		for _, b := range req.SystemBlocks {
			rr.SystemBlocks = append(rr.SystemBlocks, benchmark.RouterReplaySystemBlock{
				Type:         b.Type,
				Hash:         b.Hash,
				Bytes:        b.Bytes,
				Tokens:       b.Tokens,
				CacheControl: b.CacheControl,
			})
		}
		if req.Tools != nil {
			rr.Tools = &benchmark.RouterReplayToolsSpec{
				Count:  req.Tools.Count,
				Bytes:  req.Tools.Bytes,
				Tokens: req.Tools.Tokens,
				Hash:   req.Tools.Hash,
			}
		}
		for _, m := range req.Messages {
			rr.Messages = append(rr.Messages, benchmark.RouterReplayMessage{
				Role:          m.Role,
				Hash:          m.Hash,
				BlockTypes:    m.BlockTypes,
				Bytes:         m.Bytes,
				Tokens:        m.Tokens,
				CacheControl:  m.CacheControl,
				ToolUseIDs:    m.ToolUseIDs,
				ToolResultIDs: m.ToolResultIDs,
				SeedHash:      m.SeedHash,
			})
		}

		model := raw.ModelIn
		if model == "" {
			model = "(unknown)"
		}
		out = append(out, replayRequestWithModel{req: rr, model: model})

		if readErr == io.EOF {
			return out, nil
		}
		if readErr != nil {
			return out, readErr
		}
	}
}

// ---- text output ----

func (c *RouterAnalyzeReplayCommand) outputText(reports []modelReport, overall benchmark.ReplayCacheReport, overallRetention benchmark.RetentionReport, prefill *prefillSplitOutput) error {
	fmt.Println("Router replay cache simulation")
	fmt.Printf("  requests:     %d\n", overall.Requests)
	fmt.Println()

	printReport := func(name string, rep benchmark.ReplayCacheReport) {
		fmt.Printf("  %s\n", name)
		fmt.Printf("    Requests:            %d\n", rep.Requests)
		if rep.BlockTokensAllZero {
			fmt.Println("    Warning:  No per-block token data — simulated ratio is hash-count-based only.")
		}
		gtPct := 0.0
		if rep.GTTotalTokens > 0 {
			gtPct = 100.0 * float64(rep.GTCachedTokens) / float64(rep.GTTotalTokens)
		}
		fmt.Printf("    Ground-truth ratio:  %.2f%% (cached %s / total %s)\n",
			gtPct, formatNumber(rep.GTCachedTokens), formatNumber(rep.GTTotalTokens))
		fmt.Printf("    Simulated ratio:     %.2f%% (cached %s / total %s)\n",
			rep.SimRatio*100, formatNumber(rep.SimCachedTokens), formatNumber(rep.SimTotalTokens))
		fmt.Println()
	}

	if len(reports) == 0 {
		printReport("(no data)", overall)
		return nil
	}

	fmt.Println("  Per-model breakdown:")
	fmt.Println()
	for _, r := range reports {
		label := r.Model
		if label == "" {
			label = "(unknown)"
		}
		printReport("  "+label, r.Report)
	}

	printReport("Total", overall)

	if len(overallRetention.Windows) > 0 {
		printRetentionText(reports, overallRetention)
	}

	if c.contextSize != nil {
		printContextSizeText(c.contextSize)
	}
	printVolumeText(c.volume)
	printWeeklyText(c.weekly)

	if prefill != nil {
		c.printPrefillSplitText(prefill)
	}
	return nil
}

// ---- JSON output ----

type jsonReplayReport struct {
	Model              string               `json:"model"`
	Requests           int                  `json:"requests"`
	SimCachedTokens    int                  `json:"sim_cached_tokens"`
	SimTotalTokens     int                  `json:"sim_total_tokens"`
	SimRatio           float64              `json:"sim_ratio"`
	GTCachedTokens     int                  `json:"gt_cached_tokens"`
	GTTotalTokens      int                  `json:"gt_total_tokens"`
	GTRatio            float64              `json:"gt_ratio"`
	BlockTokensAllZero bool                 `json:"block_tokens_all_zero,omitempty"`
	Retention          *jsonRetentionReport `json:"retention,omitempty"`
}

func (c *RouterAnalyzeReplayCommand) outputJSON(reports []modelReport, overall benchmark.ReplayCacheReport, overallRetention benchmark.RetentionReport, prefill *prefillSplitOutput) error {
	type result struct {
		Models       []jsonReplayReport `json:"models"`
		Overall      jsonReplayReport   `json:"overall"`
		ContextSize  *jsonContextSize   `json:"context_size,omitempty"`
		Weekly       []jsonWeek         `json:"weekly,omitempty"`
		Volume       []jsonVolumeRow    `json:"volume_by_context,omitempty"`
		PrefillSplit *jsonPrefillSplit  `json:"prefill_split,omitempty"`
	}
	out := result{
		Models:  make([]jsonReplayReport, 0, len(reports)),
		Overall: toJSONReport("(total)", overall),
	}
	out.Overall.Retention = toJSONRetention(overallRetention)
	out.ContextSize = toJSONContextSize(c.contextSize)
	out.Weekly = toJSONWeekly(c.weekly)
	out.Volume = toJSONVolume(c.volume)
	for _, r := range reports {
		jr := toJSONReport(r.Model, r.Report)
		jr.Retention = toJSONRetention(r.Retention)
		out.Models = append(out.Models, jr)
	}
	if prefill != nil {
		out.PrefillSplit = toJSONPrefillSplit(prefill)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func toJSONReport(model string, rep benchmark.ReplayCacheReport) jsonReplayReport {
	return jsonReplayReport{
		Model:              model,
		Requests:           rep.Requests,
		SimCachedTokens:    rep.SimCachedTokens,
		SimTotalTokens:     rep.SimTotalTokens,
		SimRatio:           rep.SimRatio,
		GTCachedTokens:     rep.GTCachedTokens,
		GTTotalTokens:      rep.GTTotalTokens,
		GTRatio:            rep.GTRatio,
		BlockTokensAllZero: rep.BlockTokensAllZero,
	}
}

// ---- retention breakdown ----

// retentionSufficiency is the share of the infinite cache's hits that a
// window must keep to be called sufficient.
var retentionSufficiency = []float64{0.99, 0.999}

// printRetentionText prints one row per model and the total: simulated cache
// ratio under each TTL, then the infinite cache's, then which TTL first keeps
// 99% / 99.9% of the infinite cache's hits.
func printRetentionText(reports []modelReport, total benchmark.RetentionReport) {
	fmt.Println("  Simulated ratio by cache retention (TTL refreshed on every use; per model)")
	fmt.Println()
	fmt.Printf("    %-26s", "model")
	for _, w := range total.Windows {
		fmt.Printf(" %6s", benchmark.FormatRetentionWindow(w))
	}
	fmt.Printf(" %6s   %-10s\n", "inf", "99%/99.9%")
	row := func(name string, r benchmark.RetentionReport) {
		fmt.Printf("    %-26s", name)
		for i := range r.Windows {
			fmt.Printf(" %6.2f", 100*r.Ratio(i))
		}
		fmt.Printf(" %6.2f   %s\n", 100*r.InfiniteRatio(), sufficientWindows(r))
	}
	for _, r := range reports {
		if len(r.Retention.Windows) == 0 || r.Retention.Total == 0 {
			continue
		}
		label := r.Model
		if label == "" {
			label = "(unknown)"
		}
		row(label, r.Retention)
	}
	row("Total", total)
	fmt.Println()

	fmt.Println("  Total by retention")
	fmt.Printf("    %-6s %8s %20s %14s %14s\n", "TTL", "ratio", "cached tokens", "of infinite", "gain vs prev")
	prev := int64(0)
	for i, w := range total.Windows {
		fmt.Printf("    %-6s %7.2f%% %20s %13.2f%% %14s\n",
			benchmark.FormatRetentionWindow(w), 100*total.Ratio(i),
			formatNumber(int(total.Cached[i])), 100*shareOf(total.Cached[i], total.CachedInfinite),
			signedTokens(total.Cached[i]-prev, i == 0))
		prev = total.Cached[i]
	}
	fmt.Printf("    %-6s %7.2f%% %20s %13.2f%% %14s\n", "inf", 100*total.InfiniteRatio(),
		formatNumber(int(total.CachedInfinite)), 100.0, signedTokens(total.CachedInfinite-prev, false))
	fmt.Printf("    total input tokens: %s\n", formatNumber(int(total.Total)))
	fmt.Println()
}

// sufficientWindows names the first window keeping each retentionSufficiency
// share of the infinite cache's hits, ">max" when none does.
func sufficientWindows(r benchmark.RetentionReport) string {
	var parts []string
	for _, frac := range retentionSufficiency {
		name := ">" + benchmark.FormatRetentionWindow(r.Windows[len(r.Windows)-1])
		for i, w := range r.Windows {
			if shareOf(r.Cached[i], r.CachedInfinite) >= frac {
				name = benchmark.FormatRetentionWindow(w)
				break
			}
		}
		parts = append(parts, name)
	}
	return strings.Join(parts, "/")
}

func shareOf(a, b int64) float64 {
	if b == 0 {
		return 1
	}
	return float64(a) / float64(b)
}

func signedTokens(d int64, first bool) string {
	if first {
		return ""
	}
	return "+" + formatNumber(int(d))
}

type jsonRetentionWindow struct {
	Window       string  `json:"window"`
	Seconds      float64 `json:"seconds"`
	CachedTokens int64   `json:"cached_tokens"`
	Ratio        float64 `json:"ratio"`
}

type jsonRetentionReport struct {
	Windows        []jsonRetentionWindow `json:"windows"`
	CachedInfinite int64                 `json:"cached_infinite"`
	InfiniteRatio  float64               `json:"infinite_ratio"`
	TotalTokens    int64                 `json:"total_tokens"`
}

func toJSONRetention(r benchmark.RetentionReport) *jsonRetentionReport {
	if len(r.Windows) == 0 {
		return nil
	}
	out := &jsonRetentionReport{CachedInfinite: r.CachedInfinite, InfiniteRatio: r.InfiniteRatio(), TotalTokens: r.Total}
	for i, w := range r.Windows {
		out.Windows = append(out.Windows, jsonRetentionWindow{
			Window: benchmark.FormatRetentionWindow(w), Seconds: w.Seconds(),
			CachedTokens: r.Cached[i], Ratio: r.Ratio(i),
		})
	}
	return out
}
