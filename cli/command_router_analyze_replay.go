package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/weka/wekai/benchmark"
)

// RouterAnalyzeReplayCommand simulates a replay or source capture offline
// and reports expected cache hit ratios per model.
type RouterAnalyzeReplayCommand struct {
	Format string `long:"format" choice:"text" choice:"json" default:"text" description:"Output format"`
	Args   struct {
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
	Model  string
	Report benchmark.ReplayCacheReport
}

func (c *RouterAnalyzeReplayCommand) Execute(args []string) error {
	path := c.Args.Path

	isReplay, err := c.isReplayV3(path)
	if err != nil {
		return fmt.Errorf("cannot read input: %w", err)
	}

	var reqs []replayRequestWithModel
	if isReplay {
		reqs, err = c.collectReplayV3(path)
	} else {
		reqs, err = c.collectSourceCapture(path)
	}
	if err != nil {
		return err
	}

	if len(reqs) == 0 {
		fmt.Println("No valid requests found.")
		return nil
	}

	// Sort by timestamp (stable) so simulation order matches chronological order.
	sort.SliceStable(reqs, func(i, j int) bool {
		return reqs[i].req.Ts < reqs[j].req.Ts
	})

	// Group by model.
	byModel := map[string][]benchmark.RouterReplayRequest{}
	for _, r := range reqs {
		byModel[r.model] = append(byModel[r.model], r.req)
	}

	// Compute per-model + overall reports.
	var reports []modelReport
	var allReqs []benchmark.RouterReplayRequest

	models := make([]string, 0, len(byModel))
	for m := range byModel {
		models = append(models, m)
	}
	sort.Strings(models)

	for _, m := range models {
		groupReqs := byModel[m]
		allReqs = append(allReqs, groupReqs...)
		reports = append(reports, modelReport{Model: m, Report: benchmark.SimulateReplayCache(groupReqs)})
	}
	overall := benchmark.SimulateReplayCache(allReqs)

	if c.Format == "json" {
		return c.outputJSON(reports, overall)
	}
	return c.outputText(reports, overall)
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
	if json.Unmarshal(line, &hdr) == nil && hdr.Schema == "replay-v3" {
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
	if hdr.Schema != "replay-v3" {
		return nil, fmt.Errorf("expected replay-v3, got %q", hdr.Schema)
	}

	var out []replayRequestWithModel
	for {
		line, readErr := br.ReadBytes('\n')
		if len(line) > 0 {
			if line[len(line)-1] == '\n' {
				line = line[:len(line)-1]
			}
			if len(line) > 0 {
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

func (c *RouterAnalyzeReplayCommand) outputText(reports []modelReport, overall benchmark.ReplayCacheReport) error {
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
	return nil
}

// ---- JSON output ----

type jsonReplayReport struct {
	Model              string  `json:"model"`
	Requests           int     `json:"requests"`
	SimCachedTokens    int     `json:"sim_cached_tokens"`
	SimTotalTokens     int     `json:"sim_total_tokens"`
	SimRatio           float64 `json:"sim_ratio"`
	GTCachedTokens     int     `json:"gt_cached_tokens"`
	GTTotalTokens      int     `json:"gt_total_tokens"`
	GTRatio            float64 `json:"gt_ratio"`
	BlockTokensAllZero bool    `json:"block_tokens_all_zero,omitempty"`
}

func (c *RouterAnalyzeReplayCommand) outputJSON(reports []modelReport, overall benchmark.ReplayCacheReport) error {
	type result struct {
		Models  []jsonReplayReport `json:"models"`
		Overall jsonReplayReport   `json:"overall"`
	}
	out := result{
		Models:  make([]jsonReplayReport, 0, len(reports)),
		Overall: toJSONReport("(total)", overall),
	}
	for _, r := range reports {
		out.Models = append(out.Models, toJSONReport(r.Model, r.Report))
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
