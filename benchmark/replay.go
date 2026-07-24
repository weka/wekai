package benchmark

// Replay mode: each series walks one conversation from a dataset. The series
// slot stays alive only for the duration of that conversation — when the last
// assistant turn completes, the worker pulls the next conversation from the
// shared queue and starts fresh. When the queue drains, the worker exits.
//
// Metrics (TTFT, response time, tokens, cache hit) are recorded per-request
// using the same autoState.stream / rdw pipeline as the synthetic benchmark,
// so live progress, hill-climber, and final summary all keep working.

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/weka/wekai/config"
	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// replayQueue hands out conversations to series workers.
type replayQueue struct {
	mu      sync.Mutex
	convs   []Conversation
	nextIdx int
	total   int
}

func newReplayQueue(convs []Conversation) *replayQueue {
	return &replayQueue{convs: convs, total: len(convs)}
}

// Pull returns the next conversation, or ok=false when the queue is drained.
func (q *replayQueue) Pull() (Conversation, int, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.nextIdx >= len(q.convs) {
		return Conversation{}, 0, false
	}
	c := q.convs[q.nextIdx]
	idx := q.nextIdx
	q.nextIdx++
	return c, idx, true
}

// Remaining is the number of conversations not yet pulled.
func (q *replayQueue) Remaining() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.convs) - q.nextIdx
}

func (q *replayQueue) Total() int { return q.total }

// runReplaySeriesLoop is the replay-mode body of a series goroutine. It loops
// until the queue drains, the context is cancelled, or cfg.Total is reached.
//
// Each iteration:
//   - pulls one Conversation from the queue
//   - builds a fresh Chat instance with the conversation's system prompt cached
//   - walks the conversation's turns, buffering human/tool content as user
//     input; on each 'gpt' turn, sends the buffered content as a Request,
//     discards the dataset's original gpt value, and lets the server's actual
//     response stay in chat history (drift is intentional — we're measuring
//     throughput, not evaluating output quality)
//   - emits one benchmark request per gpt turn, with full metric capture
//   - increments st.seriesReplayCompleted on conversation end
func runReplaySeriesLoop(
	benchCtx context.Context,
	cfg AutoBenchmarkConfig,
	st *autoState,
	rdw *requestDataWriter,
	queue *replayQueue,
	endpointOverride string,
	updateSnap func(*autoState),
	gate *concurrencyGate,
) {
	reqTimeout := cfg.RequestTimeout
	if reqTimeout == 0 {
		reqTimeout = 5 * time.Minute
	}

	for {
		select {
		case <-benchCtx.Done():
			return
		default:
		}
		if cfg.Total > 0 && st.totalEmitted.Load() >= int64(cfg.Total) {
			return
		}

		conv, convIdx, ok := queue.Pull()
		if !ok {
			return
		}
		seriesNum := convIdx + 1
		seriesGUID := conv.ID

		fullyWalked := runReplayConversation(benchCtx, cfg, st, rdw, conv, convIdx, seriesNum, seriesGUID, endpointOverride, reqTimeout, gate)
		// The conversation's slot is retired — its context no longer counts
		// toward the active dataset.
		st.datasetTracker.Reset(seriesNum)
		if fullyWalked {
			st.seriesReplayCompleted.Add(1)
		}
		updateSnap(st)
	}
}

// runReplayConversation processes one Conversation end-to-end, emitting one
// benchmark request per gpt turn. Errors on individual requests are recorded
// but don't abort the series — the next turn still runs.
//
// convIdx is this conversation's index into cfg.replayConversations
// (== seriesNum-1, passed explicitly rather than re-derived).
//
// Returns true if the whole conversation was walked; false if a stop signal
// (--total reached, context cancel) cut it short mid-walk.
func runReplayConversation(
	benchCtx context.Context,
	cfg AutoBenchmarkConfig,
	st *autoState,
	rdw *requestDataWriter,
	conv Conversation,
	convIdx int,
	seriesNum int,
	seriesGUID string,
	endpointOverride string,
	reqTimeout time.Duration,
	gate *concurrencyGate,
) bool {
	// Per-turn TTFT capture: each Request() call uses a fresh pair of mutexed
	// closures so the values don't leak across turns within the conversation.
	var ttftMu sync.Mutex
	var ttft time.Duration
	var firstTokenReceived bool
	var startTime time.Time

	resetTTFT := func(t time.Time) {
		ttftMu.Lock()
		defer ttftMu.Unlock()
		startTime = t
		ttft = 0
		firstTokenReceived = false
	}
	captureFirstToken := func(s string) {
		ttftMu.Lock()
		defer ttftMu.Unlock()
		if !firstTokenReceived && len(s) > 0 {
			ttft = time.Since(startTime)
			firstTokenReceived = true
		}
	}

	chatParams := &llm.ChatParams{
		ResponseCallback: captureFirstToken,
		ThinkingCallback: captureFirstToken,
		APIKeys:          config.GetAPIKeys(),
		EndpointOverride: endpointOverride,
		MaxTokens:        cfg.MaxOutputTokens,
	}
	chatGetter := config.GetChatGetter(cfg.Model, chatParams)
	chat := chatGetter.GetChat()

	// Conversation history builder for the content-level cache estimator.
	// We accumulate system + user + assistant text across turns so Observe()
	// can detect repeated prefixes within this conversation and across others.
	var history strings.Builder

	// Walk turns, seed system prompt from the first system turn if present.
	// By default prepend <ignore>RUN_GUID: uuid</ignore>, where the UUID is
	// fixed for the whole benchmark run. This prevents cross-run cache reuse
	// while still letting conversations that share a system prompt (e.g.
	// hermes has only ~6 unique system prompts across 7k rows) collide on
	// the stamped prefix and hit the server's prefix cache within a single
	// run. --replay-no-stamp disables this entirely for bitwise-faithful
	// replay.
	firstIdx := 0
	originalSys := ""
	if len(conv.Turns) > 0 && conv.Turns[0].From == "system" {
		originalSys = conv.Turns[0].Value
		firstIdx = 1
	}
	var sysContent string
	if !cfg.ReplayNoStamp {
		sysContent = fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>\n\n%s",
			cfg.RunID, originalSys)
	} else {
		sysContent = originalSys
	}
	if sysContent != "" {
		chat.SetSystemBlocks([]llm.SystemBlock{{ID: "root", Content: sysContent, Cache: true}})
		history.WriteString(sysContent)
		history.WriteString("\x00")
	}

	isFirstRequest := true
	var coldStartTTFT time.Duration
	var pending strings.Builder
	turnNum := 0

	flush := func(gptIdx int) bool {
		userContent := strings.TrimSpace(pending.String())
		pending.Reset()
		if userContent == "" {
			return true
		}

		select {
		case <-benchCtx.Done():
			return false
		default:
		}
		if cfg.Total > 0 && st.totalEmitted.Load() >= int64(cfg.Total) {
			return false
		}
		st.totalEmitted.Add(1)

		// Gate acquisition — cold for the first request of each conversation.
		if isFirstRequest {
			if err := gate.AcquireCold(benchCtx); err != nil {
				return false
			}
		} else {
			if err := gate.Acquire(benchCtx); err != nil {
				return false
			}
		}

		turnNum++
		requestNum := int(st.totalCompleted.Load()) + 1

		// Observe the accumulated history (including this turn's user message)
		// with the content-level estimator BEFORE the server responds. The
		// recite-instruction tail (below) is deliberately NOT part of what the
		// estimator/history see — it's per-request boilerplate, not
		// conversation content, and would otherwise skew the cache-ratio
		// estimate every single turn (recite-every-turn is the default).
		history.WriteString(userContent)
		ratio := st.estimator.Observe(history.String())

		reqCtx, reqCancel := context.WithTimeout(benchCtx, reqTimeout)
		resetTTFT(time.Now())
		response, err := chat.Request(reqCtx, llm.TextParts(userContent), nil)
		totalTime := time.Since(startTime)
		reqCancel()
		gate.Release()

		// Build RequestMetrics compatible with the synthetic path's recorder.
		metrics := RequestMetrics{
			RequestNum:        requestNum,
			SeriesNum:         seriesNum,
			CycleNum:          turnNum,
			SeriesGUID:        seriesGUID,
			TimeToFirstToken:  ttft,
			TotalResponseTime: totalTime,
			Error:             err,
		}
		if response != nil {
			metrics.Response = response.Content
			if strings.TrimSpace(metrics.Response) == "" {
				metrics.Response = response.Thinking
			}
			metrics.UsageData = tools.ExecutionUsageData{
				InputTokens:     tools.TokenUsage{Count: response.Usage.PromptTokens},
				OutputTokens:    tools.TokenUsage{Count: response.Usage.CompletionTokens},
				CachedTokens:    tools.TokenUsage{Count: response.Usage.CachedTokens},
				ReasoningTokens: tools.TokenUsage{Count: response.Usage.ReasoningTokens},
				RequestCount:    1,
			}
		}
		if err == nil && strings.TrimSpace(metrics.Response) == "" {
			metrics.IsEmpty = true
			metrics.Error = fmt.Errorf("empty response from model")
		}

		// Append the server's actual assistant response to the history so future
		// turns and future conversations with the same context can be matched.
		// We only extend on success; on error/empty we drop the turn.
		if metrics.Error == nil && metrics.Response != "" {
			history.WriteString("\x00")
			history.WriteString(metrics.Response)
			st.estimator.Insert(history.String())
		}

		metrics.LocalCacheRatio = ratio

		recordReplayRequest(cfg, st, rdw, metrics, isFirstRequest, &coldStartTTFT)
		isFirstRequest = false
		return true
	}

	for i := firstIdx; i < len(conv.Turns); i++ {
		t := conv.Turns[i]
		if t.From == "gpt" {
			if !flush(i) {
				return false
			}
			continue
		}
		if pending.Len() > 0 {
			pending.WriteString("\n\n")
		}
		pending.WriteString(t.Value)
	}
	// Trailing non-gpt content (if any) is discarded — there's no assistant
	// response to measure for it.
	return true
}

// recordReplayRequest mirrors the synthetic path's metric-recording block:
// cold-start tracking, implicit cache detection, per-request JSONL write,
// completion stream update, and atomic counter bumps. Keep logic aligned with
// the inline block in runSingleModelBenchmark (auto.go).
func recordReplayRequest(
	cfg AutoBenchmarkConfig,
	st *autoState,
	rdw *requestDataWriter,
	metrics RequestMetrics,
	isCold bool,
	coldStartTTFT *time.Duration,
) {
	if cfg.PrintResponses {
		st.printMu.Lock()
		ttftStr := formatDur(metrics.TimeToFirstToken)
		if metrics.Error != nil {
			fmt.Printf("\n\u2501\u2501\u2501 [%s] s%d t%d TTFT:%s ERROR: %v \u2501\u2501\u2501\n",
				shortModelName(cfg.Model), metrics.SeriesNum, metrics.CycleNum, ttftStr, metrics.Error)
		} else {
			fmt.Printf("\n\u2501\u2501\u2501 [%s] s%d t%d TTFT:%s total:%s in:%d out:%d \u2501\u2501\u2501\n%s\n",
				shortModelName(cfg.Model), metrics.SeriesNum, metrics.CycleNum, ttftStr,
				formatDur(metrics.TotalResponseTime),
				metrics.UsageData.InputTokens.Count, metrics.UsageData.OutputTokens.Count,
				metrics.Response)
		}
		st.printMu.Unlock()
	}

	if isCold && metrics.Error == nil {
		st.coldStartTTFTCount.Add(1)
		if metrics.TimeToFirstToken > 0 {
			*coldStartTTFT = metrics.TimeToFirstToken
			st.earlyColdMu.Lock()
			if len(st.earlyColdTTFTs) < maxEarlyCold {
				st.earlyColdTTFTs = append(st.earlyColdTTFTs, metrics.TimeToFirstToken)
			}
			st.earlyColdMu.Unlock()
		}
	}

	isErr := metrics.Error != nil
	explicitCache := metrics.UsageData.CachedTokens.Count > 0

	// UUID validation tallies (--replay-inject-uuids only). metrics.ExpectedUUIDs
	// is nil/empty for every request when the feature is off (default), so this
	// block — and the val* counters it touches — is fully inert then.
	uuidExpectedCount := len(metrics.ExpectedUUIDs)
	uuidFoundCount := 0
	for _, found := range metrics.UUIDFound {
		if found {
			uuidFoundCount++
		}
	}
	uuidLeakedCount := len(metrics.LeakedUUIDs)
	if uuidExpectedCount > 0 {
		st.valReqs.Add(1)
		st.valUUIDChecks.Add(int64(uuidExpectedCount))
		st.valUUIDFound.Add(int64(uuidFoundCount))
		if metrics.ExactMatch {
			st.valExactMatchReqs.Add(1)
		}
		if missCount := uuidExpectedCount - uuidFoundCount; missCount > 0 {
			st.valPresenceMissUUIDs.Add(int64(missCount))
			st.valPresenceMissReqs.Add(1)
		}
		if uuidLeakedCount > 0 {
			st.valCrossContamUUIDs.Add(int64(uuidLeakedCount))
			st.valCrossContamReqs.Add(1)
		}
	}

	earlyColdBaseline := st.earlyColdStartTTFT()

	st.mu.Lock()
	curConc := st.concurrency
	st.mu.Unlock()
	if curConc < 1 {
		curConc = 1
	}
	recentColdBaseline := st.stream.RecentColdTTFT(curConc)
	if recentColdBaseline == 0 {
		recentColdBaseline = *coldStartTTFT
	}

	var implicitCache bool
	if recentColdBaseline > 0 {
		hitThresh := time.Duration(float64(recentColdBaseline) * cfg.TTFTHitThreshold)
		implicitCache = !isCold && metrics.TimeToFirstToken > 0 &&
			metrics.TimeToFirstToken <= hitThresh
	}

	ttftDegraded := earlyColdBaseline > 0 && metrics.TimeToFirstToken > 0 &&
		metrics.TimeToFirstToken >= earlyColdBaseline*time.Duration(cfg.TTFTDegradationFactor)
	if ttftDegraded {
		st.ttftDegradedCount.Add(1)
	}

	cacheHit := !isCold && !ttftDegraded && implicitCache
	if cacheHit && metrics.TimeToFirstToken > 0 {
		st.earlyHitMu.Lock()
		if len(st.earlyHitTTFTs) < maxEarlyHit {
			st.earlyHitTTFTs = append(st.earlyHitTTFTs, metrics.TimeToFirstToken)
		}
		st.earlyHitMu.Unlock()
	}
	serverCacheConfirmed := explicitCache

	if rdw != nil {
		reqEnd := time.Now()
		reqStart := reqEnd.Add(-metrics.TotalResponseTime)
		errMsg := ""
		if metrics.Error != nil {
			errMsg = metrics.Error.Error()
		}
		rec := requestDataRecord{
			StartTime:            reqStart,
			EndTime:              reqEnd,
			TTFT:                 float64(metrics.TimeToFirstToken.Milliseconds()),
			ResponseMs:           float64(metrics.TotalResponseTime.Milliseconds()),
			Model:                cfg.Model,
			SeriesGUID:           metrics.SeriesGUID,
			SeriesNum:            metrics.SeriesNum,
			RequestNum:           metrics.RequestNum,
			CacheHit:             cacheHit,
			ServerCacheConfirmed: serverCacheConfirmed,
			IsColdStart:          isCold,
			InputTokens:          metrics.UsageData.InputTokens.Count,
			OutputTokens:         metrics.UsageData.OutputTokens.Count,
			CachedTokens:         metrics.UsageData.CachedTokens.Count,
			IsError:              isErr,
			ErrorMessage:         errMsg,
			IsEmpty:              metrics.IsEmpty,
			LocalCacheRatio:      metrics.LocalCacheRatio,
			UUIDExpected:         uuidExpectedCount,
			UUIDFound:            uuidFoundCount,
			UUIDLeaked:           uuidLeakedCount,
			UUIDExactMatch:       metrics.ExactMatch,
		}
		// Raw detail lists only on a miss or a leak — mirrors the
		// failed-request-only policy on PromptText/ResponseText/RawResponseTail
		// above (avoid bloating every row with data that matters only when
		// something's wrong).
		if uuidFoundCount < uuidExpectedCount || uuidLeakedCount > 0 {
			rec.ExpectedUUIDsRaw = metrics.ExpectedUUIDs
			rec.FoundMask = metrics.UUIDFound
			rec.LeakedUUIDsRaw = metrics.LeakedUUIDs
		}
		if writeErr := rdw.write(rec); writeErr != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to write request data: %v\n", writeErr)
		}
	}

	st.stream.Add(completionRecord{
		completedAt:          time.Now(),
		ttft:                 metrics.TimeToFirstToken,
		isError:              isErr,
		isColdStart:          isCold,
		cacheHit:             cacheHit,
		serverCacheConfirmed: serverCacheConfirmed,
		inputTokens:          metrics.UsageData.InputTokens.Count,
		outputTokens:         metrics.UsageData.OutputTokens.Count,
		cachedTokens:         metrics.UsageData.CachedTokens.Count,
		localCacheRatio:      metrics.LocalCacheRatio,
	})

	if !isErr {
		st.datasetTracker.Update(metrics.SeriesNum,
			int64(metrics.UsageData.InputTokens.Count+metrics.UsageData.CachedTokens.Count))
	}

	st.totalCompleted.Add(1)
	if isErr {
		st.totalErrors.Add(1)
		st.consecutiveFailures.Add(1)
		if cfg.PrintErrorsThreshold > 0 {
			nowNs := time.Now().UnixNano()
			last := st.lastErrorPrintNs.Load()
			if nowNs-last >= int64(cfg.PrintErrorsThreshold) &&
				st.lastErrorPrintNs.CompareAndSwap(last, nowNs) {
				fmt.Fprintf(os.Stderr, "[%s] s%d t%d error: %v\n",
					shortModelName(cfg.Model), metrics.SeriesNum, metrics.CycleNum, metrics.Error)
			}
		}
	} else {
		st.consecutiveFailures.Store(0)
	}
}
