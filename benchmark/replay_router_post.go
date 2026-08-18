package benchmark

// Direct-HTTP poster for tree-aware router replay. Bypasses llm.Chat
// because Chat builds the body itself from accumulated chat history; it
// can't accept a pre-built multi-turn conversation with mixed text /
// tool_use / tool_result blocks (which is what the replay needs to send).
//
// Endpoint: attempt-then-fallback — <base>+leaf verbatim first, /v1 inserted
// on 404 (see replayPoster). Body construction is in replay_router_wire.go;
// this file handles HTTP transport, headers, SSE parsing, and metrics.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// replayPoster owns the per-instance HTTP plumbing.
type replayPoster struct {
	model  string
	apiKey string
	// Attempt-then-fallback endpoint resolution: the FIRST attempt honors
	// the spec's base URL exactly (base + endpoint leaf — whatever path
	// prefix the operator gave: /v1, /v2, a proxy prefix, anything). If it
	// 404s, the same request retries once with "/v1" inserted (so bare-root
	// specs work against real vLLM without the operator knowing the API
	// path). The first 2xx latches its form for the rest of the run; only
	// 404 ever triggers the fallback — other 4xx/5xx are real request
	// errors on a valid path.
	epPrimary  string
	epFallback string
	epMu       sync.Mutex
	epResolved string // latched endpoint; "" until the first success
	epFellBack bool
	apiType    string // "anthropic", "openai", or "openai_vllm"
	client     *http.Client
	// retryBudget is the total time a request may spend waiting out 429s
	// before the shed stands as an error. Defaults to retry429Budget; a field
	// rather than a constant so a test can exercise the give-up path without
	// waiting 30 real seconds.
	retryBudget time.Duration
	runID       string
	dryRun      bool
	estimator   *cacheEstimator
	dryRates    struct {
		coldTPS   int
		warmTPS   int
		outputTPS int
	}

	// outputRatio and forceOutput implement --replay-output-ratio /
	// --replay-natural-output. Set directly on the poster after construction
	// (see runRouterReplayInstance) rather than threaded through
	// newReplayPoster, to avoid touching its many existing call sites.
	outputRatio float64
	forceOutput bool
	// limitContext: skip requests whose capture-recorded prompt tokens exceed it
	// (chars~=tokens*4 convention). 0 = off.
	limitContext int
	// replayCharsPerToken: --replay-chars-per-token. When > 0, synthesized
	// replay content is sized off each block's captured Tokens count
	// (tokens * replayCharsPerToken chars) instead of its captured Bytes
	// count, so the serving tokenizer's counts land near the original
	// capture's. 0 = byte-faithful sizing (default).
	replayCharsPerToken float64

	// UUID cache-coherency injection (--verify, router path —
	// see replay_router_uuid.go). Set directly on the poster after
	// construction, same rationale as outputRatio/forceOutput above.
	// uuidEnabled gates everything: false leaves do()/dryDo() byte-for-byte
	// identical to before this feature existed.
	//
	// Both fields below are global and read-only for the poster's lifetime,
	// which matters because with multi-endpoint routing a single replayPoster
	// is SHARED across every session whose requests land on that endpoint
	// (see endpointPicker in auto.go). There is no such thing as "this
	// poster's session": the caller passes the session's own view in per
	// request, so nothing session-specific is ever cached here.
	uuidEnabled bool
	// dumper writes the verbatim exchange when --dump-dir is set; nil
	// otherwise, and nil is the whole of the feature's cost when off.
	dumper *requestDumper
	// registry is the live marker set, shared by every poster in the run. It
	// answers "is this UUID one of ours, right now" for leak scoring, and its
	// refcounts are what make a block shared by concurrent sessions visible
	// without a corpus pass. See replay_uuid_registry.go.
	registry *uuidRegistry
}

// buildInjection returns this call's *uuidInjection, or nil when injection is
// off, the session has no qualifying turns, or this request has no qualifying
// turn visible in its message history yet.
//
// su is the session's own turn view (see buildSessionUUIDs), passed in per
// call rather than cached on the poster because a poster may be shared across
// sessions under multi-endpoint routing.
//
// Every VISIBLE qualifying turn in req.Messages gets stamped into StampByHash
// (keeping every turn's marker warm in KV as later requests repeat it) — see
// uuidInjection's doc. The recite WINDOW is separate and bounded: the first
// (visible) turn plus up to 3 most-recent turns EXCLUDING the current turn
// (the highest-index turn visible in THIS request), deduplicated and capped
// at 4 (see the package doc in replay_router_uuid.go for the design rationale
// and edge cases at turns 1-3).
func (p *replayPoster) buildInjection(req RouterReplayRequest, su *sessionUUIDs) *uuidInjection {
	if !p.uuidEnabled || su == nil || len(su.uuids) == 0 {
		return nil
	}
	stampByHash := map[string]turnStamp{}
	var visible []int // turn indices visible in this request, in first-appearance order
	seenTurn := map[int]bool{}
	for _, m := range req.Messages {
		t, ok := su.hashToTurn[m.Hash]
		if !ok || t < 0 || t >= len(su.uuids) {
			continue
		}
		stampByHash[m.Hash] = turnStamp{Idx: t, UUID: su.uuids[t], Label: fmt.Sprintf("turn-%d", t+1)}
		if !seenTurn[t] {
			seenTurn[t] = true
			visible = append(visible, t)
		}
	}
	if len(visible) == 0 {
		return nil
	}

	// The window is sized by THIS request's own captured output budget — see
	// reciteCapacity. Three cases, and the first is the one that matters:
	//
	//   capacity 0  the response cannot carry even one id. No ask is made, no
	//               scoring happens, and the request is counted as excluded.
	//   capacity 1  the first turn's id alone. The oldest marker is the one a
	//               fleet is likeliest to have lost, so a single-id budget is
	//               spent on the strongest probe available.
	//   capacity n  the first turn, plus the n-1 most recent EXCLUDING the
	//               current one, up to reciteMaxRecent.
	capacity := reciteCapacity(pickMaxTokens(req, p.outputRatio))
	if capacity < 1 {
		// Still returned, with no recite ask: the inline markers stay in the
		// prompt so the turn keeps its identity in KV for later requests that
		// CAN afford to ask about it.
		return &uuidInjection{StampByHash: stampByHash, BudgetShort: true}
	}

	first := visible[0]
	window := []int{first}
	if capacity > 1 {
		// Everything visible except the current turn is a candidate; the most
		// recent of those fill whatever the budget has left.
		var recent []int
		if len(visible) > 1 {
			recent = visible[:len(visible)-1]
		}
		room := capacity - 1
		if room > reciteMaxRecent {
			room = reciteMaxRecent
		}
		if len(recent) > room {
			recent = recent[len(recent)-room:]
		}
		for _, t := range recent {
			if t != first {
				window = append(window, t)
			}
		}
	}

	labels := make([]string, len(window))
	reciteUUIDs := make([]string, len(window))
	for i, t := range window {
		labels[i] = fmt.Sprintf("turn-%d", t+1)
		reciteUUIDs[i] = su.uuids[t]
	}

	return &uuidInjection{
		StampByHash:  stampByHash,
		ReciteLabels: labels,
		ReciteUUIDs:  reciteUUIDs,
	}
}

func newReplayPoster(modelSpec string, keys llm.APIKeys, endpointOverride string, runID string, dryRun bool, coldTPS, warmTPS, outputTPS int, estimator *cacheEstimator, dispatched *atomic.Int64) (*replayPoster, error) {
	if !llm.IsDynamicModel(modelSpec) {
		return nil, fmt.Errorf("router-replay requires a dynamic/ model spec; got %q", modelSpec)
	}
	dyn, err := llm.ParseDynamicModel(modelSpec)
	if err != nil {
		return nil, fmt.Errorf("parse model spec: %w", err)
	}

	// Accepted target types: anthropic (original), openai, openai_vllm.
	// openai_vllm is treated identically to openai in the replay path — both
	// use /v1/chat/completions. The distinction matters for the Chat path
	// (max_tokens vs max_completion_tokens) but replay requests carry their
	// own max_tokens so it's irrelevant here.
	switch dyn.Type {
	case "anthropic":
		// OK — existing behaviour.
	case "openai", "openai_vllm":
		// OK — new path.
	default:
		return nil, fmt.Errorf("router-replay supports type=anthropic, type=openai, or type=openai_vllm (got %q)", dyn.Type)
	}

	base := ""
	if endpointOverride != "" {
		base = endpointOverride
	} else if len(dyn.BaseURLs) > 0 {
		base = dyn.BaseURLs[0]
	} else {
		return nil, fmt.Errorf("no base URL in model spec")
	}
	base = strings.TrimRight(base, "/")

	// API key selection: Anthropic targets use x-api-key (or dummy-key for
	// local endpoints); OpenAI targets use Bearer auth with the OpenAI key
	// (or dummy-key for local endpoints).
	apiKey := keys.Anthropic
	if dyn.Type == "openai" || dyn.Type == "openai_vllm" {
		apiKey = keys.OpenAI
	}
	if apiKey == "" {
		apiKey = "dummy-key"
	}

	// Endpoint leaf: /messages for Anthropic, /chat/completions for OpenAI.
	// The primary attempt appends it to the operator's base verbatim; the
	// fallback inserts /v1 (see the struct comment for the contract).
	leaf := "/messages"
	if dyn.Type == "openai" || dyn.Type == "openai_vllm" {
		leaf = "/chat/completions"
	}
	epPrimary := base + leaf
	epFallback := base + "/v1" + leaf

	model := dyn.Model
	if model == "" && !dryRun {
		discovered, derr := cachedDiscoverModelName(base)
		if derr != nil {
			return nil, fmt.Errorf("model=... not set in %q and model discovery from %s failed: %w", modelSpec, base, derr)
		}
		model = discovered
		logDiscoveredModelOnce(base, model)
	}
	if model == "" && dryRun {
		model = "dry-run"
	}
	return &replayPoster{
		model:      model,
		epPrimary:  epPrimary,
		epFallback: epFallback,
		apiKey:     apiKey,
		apiType:    dyn.Type,
		runID:      runID,
		dryRun:     dryRun,
		estimator:  estimator,
		dryRates: struct {
			coldTPS   int
			warmTPS   int
			outputTPS int
		}{coldTPS, warmTPS, outputTPS},
		client:      &http.Client{Transport: newInflightTransport(nil, dispatched)},
		retryBudget: retry429Budget,
	}, nil
}

var (
	discoveredOnceMu sync.Mutex
	discoveredOnce   = map[string]bool{}   // base -> printed-already
	discoveredModel  = map[string]string{} // base -> model id (discovery cache)
)

// cachedDiscoverModelName memoizes discoverModelName per base URL. Every
// series instance builds its own poster; without this cache each retired or
// spawned instance fired its own /v1/models GET, and a fast series churn
// (e.g. --limit-context retiring oversized sessions with no HTTP) flooded
// the router with hundreds of concurrent discovery calls at startup,
// shedding 503s off its request-cap and cascading into per-request errors.
func cachedDiscoverModelName(base string) (string, error) {
	discoveredOnceMu.Lock()
	if m, ok := discoveredModel[base]; ok {
		discoveredOnceMu.Unlock()
		return m, nil
	}
	discoveredOnceMu.Unlock()
	m, err := discoverModelName(base)
	if err != nil {
		return "", err
	}
	discoveredOnceMu.Lock()
	discoveredModel[base] = m
	discoveredOnceMu.Unlock()
	return m, nil
}

// logDiscoveredModelOnce prints the auto-discovered model name once per
// endpoint per process. Each agent instance creates its own replayPoster,
// so the discovery line would otherwise repeat once per instance.
func logDiscoveredModelOnce(base, model string) {
	discoveredOnceMu.Lock()
	defer discoveredOnceMu.Unlock()
	if discoveredOnce[base] {
		return
	}
	discoveredOnce[base] = true
	fmt.Fprintf(os.Stderr, "[router-replay] %s: discovered model %q from the models listing\n", base, model)
}

// discoverModelName returns the first model id from the endpoint's models
// listing. Used when --model 'http://...,type=anthropic' omits model=... so
// the operator doesn't have to remember the exact id served by each port.
// Same attempt-then-fallback contract as the replay endpoint itself: honor
// the operator's path first (<base>/models), insert /v1 only on 404.
func discoverModelName(base string) (string, error) {
	id, status, err := fetchFirstModelID(base + "/models")
	if err != nil && status == http.StatusNotFound {
		id, _, err = fetchFirstModelID(base + "/v1/models")
	}
	return id, err
}

// fetchFirstModelID GETs an OpenAI-style models listing and returns the
// first id, the HTTP status (0 on transport error), and any error.
func fetchFirstModelID(url string) (string, int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return "", resp.StatusCode, fmt.Errorf("status %d from %s", resp.StatusCode, url)
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", resp.StatusCode, err
	}
	if len(body.Data) == 0 {
		return "", resp.StatusCode, fmt.Errorf("no models returned by %s", url)
	}
	return body.Data[0].ID, resp.StatusCode, nil
}

// endpointAttempts returns the URL(s) to try for a request: the latched
// endpoint once resolved, else primary then /v1 fallback.
//
// Resolution is shared PROCESS-WIDE, keyed by the primary form, not kept per
// poster. A poster is built per replay instance, so a per-poster answer meant
// every instance re-probed from scratch and paid a 404 before falling back —
// and because this is called inside the 429 backoff loop, a poster whose first
// request kept getting shed re-probed on every iteration. The two compound: the
// 404s stayed invisible on an idle fleet and became 16.5% of all requests once
// the fleet saturated and retries began.
//
// Against a bare endpoint only. An operator who writes /v1 themselves never
// sees it, which is how seven arms and a million requests passed over it.
func (p *replayPoster) endpointAttempts() []string {
	p.epMu.Lock()
	if p.epResolved != "" {
		defer p.epMu.Unlock()
		return []string{p.epResolved}
	}
	p.epMu.Unlock()

	// Someone else already answered this question.
	if known := lookupResolvedEndpoint(p.epPrimary); known != "" {
		p.epMu.Lock()
		if p.epResolved == "" {
			p.epResolved = known
			p.epFellBack = known == p.epFallback && p.epFallback != p.epPrimary
		}
		p.epMu.Unlock()
		return []string{known}
	}
	// Genuinely first: probe both. Deliberately no single-flight, so a
	// high-concurrency launch is never serialised behind one resolver; the
	// duplicate probes last only until the first latch.
	return []string{p.epPrimary, p.epFallback}
}

// Endpoint resolutions shared across every poster in the process, keyed by the
// primary form so two different bases never see each other's answer.
var (
	epSharedMu sync.RWMutex
	epShared   = map[string]string{}
)

func lookupResolvedEndpoint(primary string) string {
	epSharedMu.RLock()
	defer epSharedMu.RUnlock()
	return epShared[primary]
}

func shareResolvedEndpoint(primary, resolved string) {
	epSharedMu.Lock()
	if _, ok := epShared[primary]; !ok {
		epShared[primary] = resolved
	}
	epSharedMu.Unlock()
}

// Endpoint resolutions logged so far, PACKAGE-scoped: replayPoster is
// constructed per replay instance (per series), so a per-poster sync.Once
// printed "[router-replay] endpoint resolved" once per SERIES — thousands
// of lines per soak. Log once per distinct resolved endpoint instead; a
// genuine endpoint change still logs.
var (
	epLogMu   sync.Mutex
	epLogSeen = map[string]bool{}
)

// latchEndpoint records the first endpoint form that returned 2xx and logs
// the resolution once per distinct endpoint (process-wide). Concurrent
// first requests may each probe both forms in parallel — benign duplicate
// probes by design (no single-flight, so a high-concurrency launch is
// never serialized behind one resolver); the first success wins and later
// latches are no-ops.
func (p *replayPoster) latchEndpoint(url string) {
	p.epMu.Lock()
	if p.epResolved == "" {
		p.epResolved = url
		p.epFellBack = url == p.epFallback && p.epFallback != p.epPrimary
	}
	resolved, fellBack := p.epResolved, p.epFellBack
	p.epMu.Unlock()
	shareResolvedEndpoint(p.epPrimary, resolved)
	epLogMu.Lock()
	seen := epLogSeen[resolved]
	if !seen {
		epLogSeen[resolved] = true
	}
	epLogMu.Unlock()
	if !seen {
		note := ""
		if fellBack {
			note = " (fallback /v1 applied)"
		}
		fmt.Fprintf(os.Stderr, "[router-replay] endpoint resolved: %s%s\n", resolved, note)
	}
}

// 429 backoff schedule.
//
// A router under a per-backend concurrency cap sheds with 429 and expects the
// caller to come back — vLLM slots free continuously as requests finish, so the
// wait is usually milliseconds, not seconds. Recording that shed as a fatal
// error (what this client did before) makes a run measure the harness: the
// benchmark reported ~68% "errors" and stalled at active=7 against a router
// whose backends were healthy and whose fleet was simply full.
//
// Starting at 10ms rather than honouring the router's Retry-After: 1 is
// deliberate. Retry-After is a coarse, fixed hint; the real recovery time is
// one request completion away, and a fixed one-second floor would turn a 20ms
// wait into a 1s one on every shed and understate the fleet's throughput.
const (
	retry429Initial = 10 * time.Millisecond
	retry429Max     = 3 * time.Second
	retry429Budget  = 30 * time.Second
)

// backoff429 returns how long to wait before the next attempt, and false when
// the total budget is spent and the 429 should stand as an error.
//
// Jittered by +/-30%. Without it, a fleet-wide shed puts every waiting client
// on the same schedule and they return in a synchronised burst, which sheds
// them again — the backoff would convert one overload into a standing wave.
// backoff429 decides whether to wait again, and for how long.
//
// `spent` is ELAPSED WALL TIME since the request began, not the sum of the
// sleeps. That distinction is the whole of a defect measured on hardware: the
// budget bounded sleeping only, while each attempt also sat inside the router
// for up to --retry-time-limit before being refused. The two composed as
//
//	total = 30s sleep budget + (retries + 1) x 10s router hold
//
// which put 2,514 failures on exactly 200/210/220/230/240s, variance under a
// second, from a 30s budget. Nothing summed to those numbers because the second
// term is a PRODUCT, with a multiplier that varies as exponential backoff fits
// more or fewer attempts into the sleep allowance.
//
// Measuring elapsed makes the budget a bound again: 30s means 30s, whatever the
// server does with each attempt.
func backoff429(base, spent, budget time.Duration) (time.Duration, bool) {
	if budget <= 0 {
		budget = retry429Budget
	}
	if spent >= budget {
		return 0, false
	}
	wait := time.Duration(float64(base) * (0.7 + 0.6*rand.Float64()))
	if remaining := budget - spent; wait > remaining {
		wait = remaining
	}
	return wait, true
}

// sendOnce POSTs bodyBytes to url with the poster's auth/accept headers.
func (p *replayPoster) sendOnce(ctx context.Context, url string, bodyBytes []byte, stream bool) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiType == "anthropic" {
		httpReq.Header.Set("Accept", "application/json")
		httpReq.Header.Set("X-Api-Key", p.apiKey)
		httpReq.Header.Set("Anthropic-Version", "2023-06-01")
		if stream {
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	} else {
		// OpenAI-compatible: Bearer auth, no Anthropic-Version header.
		httpReq.Header.Set("Accept", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		if stream {
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	}
	return p.client.Do(httpReq)
}

// do issues one request and returns its metrics. Honors ctx for
// cancellation / deadline. Dispatches to the appropriate body builder
// and response parser based on p.apiType. isLastRequest is whether req is
// the final request of the calling instance's request list — see
// buildInjection.
func (p *replayPoster) do(
	ctx context.Context,
	req RouterReplayRequest,
	docs string,
	turnIdx int,
	sessionID string,
	instanceID string,
	seriesNum int,
	st *autoState,
	su *sessionUUIDs,
) RequestMetrics {
	startTime := time.Now()
	// The configured limit, read from the deadline the caller set, so the error
	// can name it rather than leaving "context deadline exceeded" to stand for
	// any of several.
	reqLimit := time.Duration(0)
	if dl, ok := ctx.Deadline(); ok {
		reqLimit = dl.Sub(startTime)
	}
	gotResponse := false

	sessionIdx := seriesNum - 1
	inj := p.buildInjection(req, su)

	var bodyBytes []byte
	var canonical string
	var err error
	switch p.apiType {
	case "openai", "openai_vllm":
		bodyBytes, canonical, err = buildOpenAIChatCompletionsBody(req, docs, p.model, stampFor(p, req), p.outputRatio, p.forceOutput, p.replayCharsPerToken, inj)
	default:
		bodyBytes, canonical, err = buildAnthropicMessagesBody(req, docs, p.model, stampFor(p, req), p.outputRatio, p.forceOutput, p.replayCharsPerToken, inj)
	}
	// --limit-context uses the capture's production-measured token counts
	// (usage.input_tokens + cache read/creation), not a chars heuristic:
	// the replay data records the real prompt size of every request.
	if p.limitContext > 0 {
		promptTokens := req.InputTokens + req.CacheReadTokens + req.CacheCreationTokens
		if promptTokens > p.limitContext {
			return RequestMetrics{Skipped: true}
		}
	}
	if err != nil {
		return RequestMetrics{
			RequestNum:        int(st.totalCompleted.Load()) + 1,
			SeriesNum:         seriesNum,
			CycleNum:          turnIdx,
			SeriesGUID:        sessionID + ":" + instanceID,
			Error:             fmt.Errorf("build body: %w", err),
			TotalResponseTime: time.Since(startTime),
		}
	}

	var localCacheRatio float64
	if p.estimator != nil {
		localCacheRatio = p.estimator.Observe(canonical)
	}

	// Attempt-then-fallback: try the operator's path verbatim; ONLY a 404
	// (path-level error) triggers one retry of the same request with /v1
	// inserted. Transport errors and non-404 statuses return as-is.
	//
	// Wrapped in the 429 backoff loop: a router that sheds under load expects
	// the client to come back, and a benchmark that instead records the shed as
	// a fatal error measures the harness rather than the fleet. See
	// backoff429.
	var resp *http.Response
	var attemptURL string
	var retryWait time.Duration
	var retries int
	attemptStart := startTime
	backoff := retry429Initial

	for {
		attemptStart = time.Now()
		attempts := p.endpointAttempts()
		for i, u := range attempts {
			resp, err = p.sendOnce(ctx, u, bodyBytes, req.Stream)
			if err != nil {
				err = classifyDeadline(err, reqLimit, time.Since(startTime), gotResponse, &RequestMetrics{})
				return RequestMetrics{
					RequestNum:        int(st.totalCompleted.Load()) + 1,
					SeriesNum:         seriesNum,
					CycleNum:          turnIdx,
					SeriesGUID:        sessionID + ":" + instanceID,
					Error:             err,
					Retries429:        retries,
					RetryWait:         retryWait,
					TotalResponseTime: time.Since(startTime),
				}
			}
			attemptURL = u
			if resp.StatusCode == http.StatusNotFound && i+1 < len(attempts) {
				io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
				resp.Body.Close()
				continue
			}
			break
		}

		// Latch as soon as the path is PROVEN, which a 429 does as well as a
		// 200: it means the router recognised the route and applied capacity
		// logic to it. Only a 404 says the path is wrong.
		//
		// Latching solely on success meant a fleet that stayed saturated never
		// resolved at all, so every request re-probed the operator's bare base
		// first and ate a 404 before the real attempt — doubling inbound
		// requests exactly when the fleet could least afford it, and filling the
		// router's log with 404s on a path nobody configured. Visible as a
		// perfect alternation of passthrough 404 and chat 429.
		if resp.StatusCode != http.StatusNotFound {
			p.latchEndpoint(attemptURL)
		}

		if resp.StatusCode != http.StatusTooManyRequests {
			break
		}
		wait, ok := backoff429(backoff, time.Since(startTime), p.retryBudget)
		if !ok {
			// Budget exhausted: keep this 429 and let it be recorded as the
			// error it now genuinely is.
			break
		}
		io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		resp.Body.Close()

		select {
		case <-ctx.Done():
			return RequestMetrics{
				RequestNum:        int(st.totalCompleted.Load()) + 1,
				SeriesNum:         seriesNum,
				CycleNum:          turnIdx,
				SeriesGUID:        sessionID + ":" + instanceID,
				Error:             ctx.Err(),
				Retries429:        retries,
				RetryWait:         retryWait,
				TotalResponseTime: time.Since(startTime),
			}
		case <-time.After(wait):
		}
		retryWait += wait
		retries++
		backoff = min(2*backoff, retry429Max)
	}
	defer resp.Body.Close()

	m := RequestMetrics{
		RequestNum: int(st.totalCompleted.Load()) + 1,
		SeriesNum:  seriesNum,
		CycleNum:   turnIdx,
		SeriesGUID: sessionID + ":" + instanceID,
	}

	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		m.Error = fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		if resp.StatusCode == http.StatusTooManyRequests {
			// Only reachable once the whole budget is spent, so say so: a 429
			// here means the fleet stayed saturated throughout, not that it
			// shed once.
			//
			// ELAPSED is stated as well as the sleeping, because the two differ
			// by however long the server held each attempt and the difference is
			// the finding. A shed that says "retry shortly" and arrives three and
			// a half minutes in is a slow failure wearing a fast failure's
			// words, and only the elapsed figure shows it.
			m.Error = fmt.Errorf("status 429 after %v elapsed (%v of it sleeping) over %d retries: %s",
				time.Since(startTime).Round(time.Millisecond),
				retryWait.Round(time.Millisecond), retries, strings.TrimSpace(string(body)))
		}
		m.Retries429, m.RetryWait = retries, retryWait
		m.TotalResponseTime = time.Since(startTime)
		return m
	}
	gotResponse = true
	// With --dump-dir, the consumers read through a tee so the capture is the
	// bytes that actually arrived — SSE framing and all — rather than a
	// re-rendering of what the parser made of them. A parser bug and a model
	// behaviour are indistinguishable in parsed output and obvious here.
	respReader := io.Reader(resp.Body)
	var respCapture bytes.Buffer
	if p.dumper != nil {
		respReader = io.TeeReader(resp.Body, &respCapture)
	}
	// Timed from attemptStart, not startTime: TTFT and the response time the
	// consumers compute must describe the attempt the server actually ran, so
	// that backoff cannot make a healthy fleet look slow. The client-side wait
	// is added back into TotalResponseTime below, where it belongs.
	if p.apiType == "openai" || p.apiType == "openai_vllm" {
		if req.Stream {
			consumeOpenAISSE(respReader, attemptStart, &m)
		} else {
			consumeOpenAIPlain(respReader, attemptStart, &m)
		}
	} else {
		if req.Stream {
			consumeSSE(respReader, attemptStart, &m)
		} else {
			consumePlain(respReader, attemptStart, &m)
		}
	}

	if m.TotalResponseTime == 0 {
		m.TotalResponseTime = time.Since(attemptStart)
	}
	// End to end, as the caller experienced it: server time plus every 429 it
	// waited out. TTFT is folded the same way and for the same reason — a
	// request that took seventeen attempts did not have a fast first token,
	// whatever the seventeenth attempt measured.
	//
	// A request that never produced a token keeps a zero TTFT rather than
	// inheriting the wait: it has no time-to-first-token to report, and
	// inventing one would put backoff into a statistic about tokens.
	m.Retries429, m.RetryWait = retries, retryWait
	m.TotalResponseTime += retryWait
	// One classification point for every path out of here: the consumers know
	// the error but not which deadline was set, and do() knows both.
	m.Error = classifyDeadline(m.Error, reqLimit, time.Since(startTime), gotResponse, &m)
	if m.TimeToFirstToken > 0 {
		m.TimeToFirstToken += retryWait
	}
	if m.Error == nil && strings.TrimSpace(m.Response) == "" && m.UsageData.OutputTokens.Count == 0 {
		m.IsEmpty = true
		m.Error = fmt.Errorf("empty response from model")
	}
	m.LocalCacheRatio = localCacheRatio

	// UUID cache-coherency validation (--verify, router path).
	// consumeOpenAISSE/consumeOpenAIPlain/consumeSSE/consumePlain already
	// merge reasoning/thinking into m.Response (see their doc comments), so
	// a single Contains-scan of m.Response covers both — thinking is passed
	// as "" here, mirroring the dataset path's own call shape.
	//
	// Two independent checks, mirroring the cache-coherency eval's two
	// reported tests: per-UUID PRESENCE (Contains anywhere in the response,
	// scored against inj.ReciteUUIDs — this request's recite WINDOW, not the
	// session's full turn history) and output CONFORMITY (the FIRST LINE of
	// the response is exactly the ordered, comma-joined window UUID list —
	// see firstLineConformity/matchesExpectedUUIDList). Cross-contamination
	// uses findLeakedUUIDsByOwner (replay_uuid.go), an O(response) reverse-
	// map scan — NOT FindLeakedUUIDs, whose O(population) iteration over
	// every session's UUID set no longer fits once turns (not sessions) are
	// the stamping unit.
	//
	// Gated on inj.Recite, NOT just inj != nil: buildInjection returns a
	// non-nil *uuidInjection whenever ANY qualifying turn is visible in this
	// request (it always stamps every visible turn inline, so those stamps
	// stay warm in KV across the session — see the package doc in
	// replay_router_uuid.go), but only inj.Recite (reciteEveryRequest ||
	// isLastRequest) says the model was actually ASKED to recite this turn.
	// Scoring a non-recite turn would count "the model didn't volunteer the
	// UUID list" as a PRESENCE_MISS/conformity failure even though nothing
	// asked it to.
	// Contamination is scored on EVERY completed request, including ones
	// carrying no marker of their own.
	//
	// A leak needs no cooperation from the model and no marker in the prompt:
	// it is a live marker in the response that this request did not send.
	// Scoring it behind the recite ask made coverage a function of what the
	// model was asked for — 18% of a measured run was never checked, because
	// a request whose visible history holds no qualifying user turn produced
	// no injection at all — and the resulting zero read as if it covered the
	// run. Requests carrying nothing of their own are in fact the cleanest
	// signal available: own is empty, so any live marker in the response is
	// unambiguously another session's.
	if p.uuidEnabled && m.Error == nil && !m.IsEmpty {
		var own map[string]bool
		if inj != nil {
			own = make(map[string]bool, len(inj.StampByHash))
			for _, st := range inj.StampByHash {
				own[st.UUID] = true
			}
		}
		m.LeakChecked = true
		m.LeakedUUIDs = findLeakedUUIDs(m.Response, "", own, p.registry)
		// Contamination is unaffected by the output budget: a leaked marker
		// arrives whether or not anything was asked for.
		m.ReciteBudgetShort = inj != nil && inj.BudgetShort
	}

	// Presence and conformity stay gated on inj.Recite, which is the right
	// gate for them: buildInjection returns non-nil whenever any qualifying
	// turn is visible (it always stamps them inline so they stay warm in KV),
	// but only Recite says the model was ASKED to repeat this turn. Scoring a
	// turn nobody asked about would count "the model did not volunteer a
	// list" as a presence miss.
	if inj != nil && !inj.BudgetShort && m.Error == nil && !m.IsEmpty {
		m.ConvIdx = sessionIdx
		m.ExpectedUUIDs = append([]string(nil), inj.ReciteUUIDs...)
		m.UUIDFound = make([]bool, len(inj.ReciteUUIDs))
		for i, u := range inj.ReciteUUIDs {
			m.UUIDFound[i] = strings.Contains(m.Response, u)
		}
		m.ExactMatch = firstLineConformity(m.Response, inj.ReciteUUIDs)
		// Attribute every miss while the injection is still in hand: only here
		// is it known which markers this request actually sent.
		cls := classifyRecite(m.Response, inj, m.UUIDFound)
		m.MissSubstituted, m.MissAbsent = cls.Substituted, cls.Absent
		m.ReciteEchoedTags, m.ReciteNoIDs = cls.EchoedTags, cls.NoIDs
	}

	// Written after scoring, so the derived verdict and the bytes it came from
	// land together: a PRESENCE_MISS beside a request that never carried the
	// marker is a different defect from one beside a response that ignored it.
	if p.dumper != nil {
		p.dumper.dump(dumpMeta{
			Series: seriesNum, Instance: instanceID, Turn: turnIdx,
			URL: attemptURL, Status: resp.StatusCode, TTFTMillis: m.TimeToFirstToken.Milliseconds(),
			InputTok: m.UsageData.InputTokens.Count, OutputTok: m.UsageData.OutputTokens.Count,
			Expected: m.ExpectedUUIDs, Found: m.UUIDFound, Leaked: m.LeakedUUIDs,
			ExactMatch: m.ExactMatch, LeakChecked: m.LeakChecked,
			MissSubstituted: m.MissSubstituted, MissAbsent: m.MissAbsent,
			ReciteEchoedTags: m.ReciteEchoedTags, ReciteNoIDs: m.ReciteNoIDs,
			BudgetShort: m.ReciteBudgetShort,
			Error:       errString(m.Error),
		}, bodyBytes, respCapture.Bytes(), mergedResponse(m.Response, respCapture.Bytes()))
	}
	return m
}

// buildReplayUsage builds ExecutionUsageData from a replayed provider response,
// enforcing the net-of-cache contract auto.go:283 depends on: InputTokens MUST
// EXCLUDE server-cached tokens, because auto.go reconstructs the full prompt as
// full = InputTokens + CachedTokens. Storing gross prompt tokens here double-counts
// the cached volume in the `in` denominator (deflating scached/in). ALL replay
// consumers (OpenAI + Anthropic, streaming + plain) MUST go through this one helper
// so the contract can never drift between paths again.
//
//	grossPrompt = total prompt tokens the server processed, INCLUDING cached
//	              (OpenAI: prompt_tokens; Anthropic: input + cache_read + cache_creation)
//	cached      = subset served from cache without recompute
//	              (OpenAI: prompt_tokens_details.cached_tokens; Anthropic: cache_read_input_tokens)
func buildReplayUsage(grossPrompt, cached, output int) tools.ExecutionUsageData {
	net := grossPrompt - cached
	if net < 0 {
		net = 0
	}
	return tools.ExecutionUsageData{
		InputTokens:  tools.TokenUsage{Count: net},
		OutputTokens: tools.TokenUsage{Count: output},
		CachedTokens: tools.TokenUsage{Count: cached},
		RequestCount: 1,
	}
}

// consumeSSE reads an Anthropic streaming response, capturing TTFT on the
// first content delta and usage numbers from message_start /
// message_delta events.
func consumeSSE(body io.Reader, startTime time.Time, m *RequestMetrics) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1<<16), 16<<20)
	var firstToken sync.Once
	var resp strings.Builder
	for scanner.Scan() {
		line := scanner.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data: "):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var ev map[string]json.RawMessage
		if err := json.Unmarshal(data, &ev); err != nil {
			continue
		}
		var evType string
		if t, ok := ev["type"]; ok {
			_ = json.Unmarshal(t, &evType)
		}
		switch evType {
		case "message_start":
			if msg, ok := ev["message"]; ok {
				var ms struct {
					Usage struct {
						InputTokens              int `json:"input_tokens"`
						CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
						CacheReadInputTokens     int `json:"cache_read_input_tokens"`
						OutputTokens             int `json:"output_tokens"`
					} `json:"usage"`
				}
				if err := json.Unmarshal(msg, &ms); err == nil {
					m.UsageData = buildReplayUsage(ms.Usage.InputTokens+ms.Usage.CacheReadInputTokens+ms.Usage.CacheCreationInputTokens, ms.Usage.CacheReadInputTokens, ms.Usage.OutputTokens)
				}
			}
		case "content_block_delta":
			if delta, ok := ev["delta"]; ok {
				var d struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
					Thinking    string `json:"thinking"`
				}
				if err := json.Unmarshal(delta, &d); err == nil {
					if d.Text != "" || d.Thinking != "" || d.PartialJSON != "" {
						firstToken.Do(func() {
							m.TimeToFirstToken = time.Since(startTime)
						})
					}
					resp.WriteString(d.Text)
					resp.WriteString(d.Thinking)
				}
			}
		case "message_delta":
			if usage, ok := ev["usage"]; ok {
				var u struct {
					InputTokens              int `json:"input_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					OutputTokens             int `json:"output_tokens"`
				}
				if err := json.Unmarshal(usage, &u); err == nil {
					if u.InputTokens != 0 {
						m.UsageData.InputTokens.Count = u.InputTokens + u.CacheCreationInputTokens // net of cache_read
						m.UsageData.CachedTokens.Count = u.CacheReadInputTokens
					}
					if u.OutputTokens != 0 {
						m.UsageData.OutputTokens.Count = u.OutputTokens
					}
				}
			}
			if delta, ok := ev["delta"]; ok {
				var d struct {
					StopReason string `json:"stop_reason"`
				}
				_ = json.Unmarshal(delta, &d)
			}
		case "error":
			if data, ok := ev["error"]; ok {
				m.Error = fmt.Errorf("anthropic stream error: %s", string(data))
			}
		}
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		if m.Error == nil {
			m.Error = err
		}
	}
	m.TotalResponseTime = time.Since(startTime)
	m.Response = resp.String()
}

// ---- OpenAI SSE consumption ----

// consumeOpenAISSE reads an OpenAI-compatible streaming chat/completions
// response (SSE with data: {} lines). Captures TTFT on the first content
// delta and usage from the final chunk carrying the "usage" field.
//
// OpenAI SSE format (standard):
//
//	data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
//	data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
//	data: [DONE]
//
// Some servers (sglang) omit the usage field entirely; we tolerate that
// gracefully.
func consumeOpenAISSE(body io.Reader, startTime time.Time, m *RequestMetrics) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1<<16), 16<<20)
	var firstToken sync.Once
	var resp strings.Builder

	for scanner.Scan() {
		line := scanner.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data: "):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"` // vLLM uses "reasoning"
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens        int `json:"prompt_tokens"`
				CompletionTokens    int `json:"completion_tokens"`
				TotalTokens         int `json:"total_tokens"`
				PromptTokensDetails *struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details,omitempty"`
			} `json:"usage,omitempty"`
		}
		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}

		// Capture usage from the final chunk.
		if chunk.Usage != nil {
			cached := 0
			if chunk.Usage.PromptTokensDetails != nil {
				cached = chunk.Usage.PromptTokensDetails.CachedTokens
			}
			m.UsageData = buildReplayUsage(chunk.Usage.PromptTokens, cached, chunk.Usage.CompletionTokens)
		}

		// Capture content delta & TTFT.
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			content := delta.Content
			reasoning := delta.ReasoningContent
			if reasoning == "" {
				reasoning = delta.Reasoning
			}
			if content != "" || reasoning != "" {
				firstToken.Do(func() {
					m.TimeToFirstToken = time.Since(startTime)
				})
			}
			resp.WriteString(content)
			resp.WriteString(reasoning)
		}
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		if m.Error == nil {
			m.Error = err
		}
	}
	m.TotalResponseTime = time.Since(startTime)
	m.Response = resp.String()
}

// consumeOpenAIPlain reads a non-streaming OpenAI chat/completions response.
// Like consumeOpenAISSE, this merges message.reasoning_content (or
// message.reasoning, the vLLM field name) into m.Response ahead of
// message.content — a non-streaming reasoning-model response carries its
// full reasoning text on the message object rather than as incremental
// deltas, and skipping it here would make a UUID recited/leaked only in the
// reasoning channel invisible to the presence/leak scan.
func consumeOpenAIPlain(body io.Reader, startTime time.Time, m *RequestMetrics) {
	b, err := io.ReadAll(body)
	if err != nil {
		m.Error = err
		return
	}
	m.TimeToFirstToken = time.Since(startTime)
	var resp struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				Reasoning        string `json:"reasoning"` // vLLM uses "reasoning"
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			TotalTokens         int `json:"total_tokens"`
			PromptTokensDetails *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details,omitempty"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		m.Error = err
		return
	}
	if len(resp.Choices) > 0 {
		msg := resp.Choices[0].Message
		reasoning := msg.ReasoningContent
		if reasoning == "" {
			reasoning = msg.Reasoning
		}
		m.Response = reasoning + msg.Content
	}
	cached := 0
	if resp.Usage.PromptTokensDetails != nil {
		cached = resp.Usage.PromptTokensDetails.CachedTokens
	}
	m.UsageData = buildReplayUsage(resp.Usage.PromptTokens, cached, resp.Usage.CompletionTokens)
}

// consumePlain reads a non-streaming Anthropic response. Like consumeSSE,
// this merges "thinking" content blocks into m.Response alongside "text"
// blocks — a non-streaming extended-thinking response carries its thinking
// block(s) as ordinary entries in the content array (field "thinking", not
// "text"), and skipping them here would make a UUID recited/leaked only in
// the thinking channel invisible to the presence/leak scan.
func consumePlain(body io.Reader, startTime time.Time, m *RequestMetrics) {
	b, err := io.ReadAll(body)
	if err != nil {
		m.Error = err
		return
	}
	m.TimeToFirstToken = time.Since(startTime)
	var resp struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		m.Error = err
		return
	}
	var sb strings.Builder
	for _, c := range resp.Content {
		switch c.Type {
		case "text":
			sb.WriteString(c.Text)
		case "thinking":
			sb.WriteString(c.Thinking)
		}
	}
	m.Response = sb.String()
	m.UsageData = buildReplayUsage(resp.Usage.InputTokens+resp.Usage.CacheReadInputTokens+resp.Usage.CacheCreationInputTokens, resp.Usage.CacheReadInputTokens, resp.Usage.OutputTokens)
}

// dryRunDurations returns the synthetic time-to-first-token (prefill) and total
// response time for a request, given its cold/warm input token split and output
// token count, at the configured per-second rates.
// A rate <= 0 OR a token count <= 0 contributes 0 duration.
func dryRunDurations(coldTokens, warmTokens, outputTokens, coldTPS, warmTPS, outputTPS int) (ttft, total time.Duration) {
	if coldTPS > 0 && coldTokens > 0 {
		ttft += time.Duration(float64(coldTokens) / float64(coldTPS) * float64(time.Second))
	}
	if warmTPS > 0 && warmTokens > 0 {
		ttft += time.Duration(float64(warmTokens) / float64(warmTPS) * float64(time.Second))
	}
	total = ttft
	if outputTPS > 0 && outputTokens > 0 {
		total += time.Duration(float64(outputTokens) / float64(outputTPS) * float64(time.Second))
	}
	return
}

// dryDo returns a synthetic RequestMetrics for dry-run mode without making any
// HTTP requests. It computes TTFT and total duration from the cold/warm token
// split (derived from the cache estimator ratio) and output tokens using the
// configured per-second rates, then sleeps for that duration (ctx-aware) to
// simulate real passage of time.
func (p *replayPoster) dryDo(
	ctx context.Context,
	req RouterReplayRequest,
	docs string,
	turnIdx int,
	sessionID string,
	instanceID string,
	seriesNum int,
	st *autoState,
	su *sessionUUIDs,
) RequestMetrics {
	// Build canonical string for estimator and compute ratio. Injection is
	// threaded through purely so the canonical text (and therefore the
	// cache-ratio estimate) stays consistent with a real do() call for the
	// same request — dry-run never makes a real request, so there's no
	// response to validate.
	inj := p.buildInjection(req, su)
	var canonical string
	switch p.apiType {
	case "openai", "openai_vllm":
		_, canonical, _ = buildOpenAIChatCompletionsBody(req, docs, p.model, stampFor(p, req), p.outputRatio, p.forceOutput, p.replayCharsPerToken, inj)
	default:
		_, canonical, _ = buildAnthropicMessagesBody(req, docs, p.model, stampFor(p, req), p.outputRatio, p.forceOutput, p.replayCharsPerToken, inj)
	}
	var ratio float64
	if p.estimator != nil {
		ratio = p.estimator.Observe(canonical)
	}

	total := req.InputTokens
	full := int64(total) + int64(req.CacheReadTokens)
	warm := int64(float64(full)*ratio + 0.5)
	if warm > full {
		warm = full
	}
	cold := int(full) - int(warm)
	if cold < 0 {
		cold = 0
	}
	cachedForTiming := int(warm)
	ttft, totalDur := dryRunDurations(cold, cachedForTiming, req.OutputTokens, p.dryRates.coldTPS, p.dryRates.warmTPS, p.dryRates.outputTPS)

	select {
	case <-time.After(totalDur):
	case <-ctx.Done():
	}

	return RequestMetrics{
		RequestNum:        int(st.totalCompleted.Load()) + 1,
		SeriesNum:         seriesNum,
		CycleNum:          turnIdx,
		SeriesGUID:        sessionID + ":" + instanceID,
		TimeToFirstToken:  ttft,
		TotalResponseTime: totalDur,
		LocalCacheRatio:   ratio,
		UsageData: tools.ExecutionUsageData{
			InputTokens:  tools.TokenUsage{Count: total},
			OutputTokens: tools.TokenUsage{Count: req.OutputTokens},
			CachedTokens: tools.TokenUsage{Count: 0},
			RequestCount: 1,
		},
		Error:   nil,
		IsEmpty: false,
	}
}

// stampFor is the content stamp for one request: the pass's, when the corpus is
// being replayed more than once, else the run's.
//
// Per REQUEST rather than on the poster because a poster is shared across a
// series and outlives any one pass. Uniform within a pass by construction —
// every session on that pass carries the same string — which is what keeps
// cross-session prefix sharing identical to the first pass.
func stampFor(p *replayPoster, req RouterReplayRequest) string {
	if req.passStamp != "" {
		return req.passStamp
	}
	return p.runID
}

// classifyDeadline turns a bare "context deadline exceeded" into something that
// says which limit fired, what it was set to, how far the request got and how
// long it ran.
//
// The bare string is unusable for the one job an error has here. Across two
// realtime arms every client-visible failure was that one sentence, and
// classifying them retrospectively — from response times, because the message
// gave nothing — showed the two fleets were not failing the same way at all:
// one died at the configured cap having streamed for five minutes, the other
// died before a first token in a tight cluster nowhere near any cap. Comparing
// the two error rates as though they counted the same event was wrong, and
// nothing in the output said so.
//
// Phase is the part that separates them. A request that hung before any bytes
// came back and one cut mid-generation at 4m59s are the same string today and
// mean opposite things about where to look.
func classifyDeadline(err error, limit time.Duration, elapsed time.Duration, gotResponse bool, m *RequestMetrics) error {
	if err == nil {
		return nil
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	phase := "before the response headers arrived"
	switch {
	case m.TimeToFirstToken > 0:
		phase = fmt.Sprintf("mid-stream, %s after its first token", (elapsed - m.TimeToFirstToken).Round(time.Millisecond))
	case gotResponse:
		phase = "after headers, before any token"
	}
	// The elapsed time is stated even though the limit is, because they differ
	// whenever something OTHER than this deadline cut the request — and a gap
	// between them is the signal that the limit named here is not the one that
	// fired.
	return fmt.Errorf("exceeded request timeout %s after %s: died %s",
		limit.Round(time.Second), elapsed.Round(time.Millisecond), phase)
}

// errString renders an error for the dump metadata without a nil check at
// every call site.
func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
