package benchmark

// --replay-exact-tokens: exact token-targeted sizing for router-replay
// content, using the serving endpoint's own tokenizer (POST /tokenize) as
// an oracle instead of a fixed chars/token ratio (--replay-chars-per-token,
// see sizeBudget in replay_router_wire.go).
//
// The replay filler content is a deterministic slice of the embedded corpus
// (bench_doc.txt — see synthText/hashOffset in replay_router_wire.go), so
// we can afford to index the WHOLE corpus once per run: tokenize it in
// fixed-size chunks through /tokenize, and treat the cumulative token count
// as piecewise-linear between chunk boundaries (token density is assumed
// locally uniform, reasonable for natural-language corpus text; chunks are
// tokenized independently of one another, so a handful of tokens near each
// boundary may differ from a single full-corpus tokenizer pass — negligible
// next to realistic per-block budgets of tens to thousands of tokens).
//
// A per-block char length is then found by binary search (corpusTokenIndex
// .charsForTokens): the smallest L such that the indexed token count of
// docs[offset:offset+L] (wrapping at the corpus boundary, exactly like
// synthText's own fill loop) reaches the block's captured target token
// count.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/weka/wekai/llm"
)

// tokenizeChunkChars is the production grid resolution used to index the
// corpus: large enough that indexing a multi-MB corpus takes tens, not
// thousands, of /tokenize calls (one-time cost per run); small enough that
// within-chunk linear interpolation stays close to the true tokenizer.
const tokenizeChunkChars = 32 * 1024

// tokenizeHTTPClient is shared by all /tokenize calls (index build + the
// startup probe). A generous timeout: chunk tokenization is cheap (a 32KB
// chunk took <1s against a live vLLM endpoint) but a heavily loaded server
// under a concurrent benchmark run may queue the request behind inference
// traffic.
var tokenizeHTTPClient = &http.Client{Timeout: 30 * time.Second}

// tokenizeCount calls the target's POST /tokenize with prompt and returns
// its token count. Same attempt-then-fallback contract as
// discoverModelName/newReplayPoster (replay_router_post.go): try
// <base>/tokenize first — vLLM serves /tokenize at the server root, not
// under /v1 (verified against a live endpoint: POST /v1/tokenize 404s,
// POST /tokenize 200s) — and insert /v1 only on 404, for deployments that
// front it behind a /v1-prefixed proxy.
func tokenizeCount(base, model, apiKey, prompt string) (int, error) {
	n, status, err := doTokenize(base+"/tokenize", model, apiKey, prompt)
	if err != nil && status == http.StatusNotFound {
		n, _, err = doTokenize(base+"/v1/tokenize", model, apiKey, prompt)
	}
	return n, err
}

// doTokenize issues one POST /tokenize and returns (count, HTTP status,
// error). status is 0 on a transport-level failure (no response received),
// matching fetchFirstModelID's convention so callers can distinguish "got a
// 404, try the fallback path" from "couldn't even connect."
//
// Response shape (verified live): {"count": N, "max_model_len": ...,
// "tokens": [...ids...], "token_strs": null}. "model" is accepted but
// optional — vLLM falls back to its single loaded model when omitted. We
// prefer "count" and fall back to len("tokens") if count is ever absent.
func doTokenize(url, model, apiKey, prompt string) (int, int, error) {
	reqBody, err := json.Marshal(map[string]interface{}{
		"model":  model,
		"prompt": prompt,
	})
	if err != nil {
		return 0, 0, err
	}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return 0, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" && apiKey != "dummy-key" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := tokenizeHTTPClient.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return 0, resp.StatusCode, fmt.Errorf("status %d from %s", resp.StatusCode, url)
	}
	var body struct {
		Count  int   `json:"count"`
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return 0, resp.StatusCode, err
	}
	n := body.Count
	if n == 0 && len(body.Tokens) > 0 {
		n = len(body.Tokens)
	}
	return n, resp.StatusCode, nil
}

// corpusTokenIndex is a piecewise-linear approximation of "cumulative
// tokens consumed by docs[0:charOffset]" for a fixed corpus, built by
// tokenizing it in chunkChars-sized chunks through a target's /tokenize.
type corpusTokenIndex struct {
	docsLen    int
	chunkChars int
	// cum[i] = cumulative token count at char offset i*chunkChars.
	// len(cum) == ceil(docsLen/chunkChars) + 1 (cum[0] == 0 always).
	cum []float64
}

// buildCorpusTokenIndex tokenizes docs in chunkChars-sized chunks against
// base/model and returns the resulting index. This is the ONE-TIME
// indexing pass — callers build it once per run (see buildReplayTokenIndex
// / AutoBenchmarkConfig.replayTokenIndex) and share the pointer; calling it
// per-request or per-instance would multiply /tokenize traffic by the
// request count.
func buildCorpusTokenIndex(base, model, apiKey, docs string, chunkChars int) (*corpusTokenIndex, error) {
	if chunkChars <= 0 {
		return nil, fmt.Errorf("chunkChars must be > 0")
	}
	n := len(docs)
	idx := &corpusTokenIndex{docsLen: n, chunkChars: chunkChars, cum: []float64{0}}
	total := 0
	for start := 0; start < n; start += chunkChars {
		end := start + chunkChars
		if end > n {
			end = n
		}
		count, err := tokenizeCount(base, model, apiKey, docs[start:end])
		if err != nil {
			return nil, fmt.Errorf("tokenize corpus chunk [%d:%d]: %w", start, end, err)
		}
		total += count
		idx.cum = append(idx.cum, float64(total))
	}
	return idx, nil
}

// tokensUpTo returns the interpolated cumulative token count at char offset
// charOffset, clamped to [0, docsLen]. Exact (no interpolation error) at
// grid boundaries (multiples of chunkChars), linearly interpolated between
// them.
func (idx *corpusTokenIndex) tokensUpTo(charOffset int) float64 {
	if idx == nil || idx.docsLen == 0 {
		return 0
	}
	if charOffset <= 0 {
		return 0
	}
	if charOffset >= idx.docsLen {
		return idx.cum[len(idx.cum)-1]
	}
	gridIdx := charOffset / idx.chunkChars
	gridStart := gridIdx * idx.chunkChars
	lo := idx.cum[gridIdx]
	hi := idx.cum[gridIdx+1]
	frac := float64(charOffset-gridStart) / float64(idx.chunkChars)
	return lo + frac*(hi-lo)
}

// tokensInSlice returns the (interpolated) token count of docs[off:off+n],
// wrapping at docsLen exactly like synthText's own fill loop (so the token
// estimate matches what synthText will actually emit for the same off/n).
func (idx *corpusTokenIndex) tokensInSlice(off, n int) float64 {
	if idx == nil || n <= 0 || idx.docsLen == 0 {
		return 0
	}
	var total float64
	remaining := n
	for remaining > 0 {
		room := idx.docsLen - off
		if room <= 0 {
			off = 0
			room = idx.docsLen
		}
		take := remaining
		if take > room {
			take = room
		}
		total += idx.tokensUpTo(off+take) - idx.tokensUpTo(off)
		remaining -= take
		off += take
	}
	return total
}

// charsForTokens binary-searches for the smallest char length L such that
// tokensInSlice(off, L) reaches targetTokens. tokensInSlice is monotone
// non-decreasing in L (tokensUpTo is a cumulative sum of non-negative
// chunk deltas), so the search always converges. The upper bound grows
// geometrically from a generous starting guess, so an unusually
// low-tokens-per-char corpus region — or a target far beyond one corpus
// pass, which tokensInSlice satisfies by wrapping — still resolves.
func (idx *corpusTokenIndex) charsForTokens(off int, targetTokens int) int {
	if idx == nil || targetTokens <= 0 || idx.docsLen == 0 {
		return 0
	}
	lo, hi := 0, targetTokens*8+1024 // realistic chars/token is ~2-6; generous headroom
	for idx.tokensInSlice(off, hi) < float64(targetTokens) {
		next := hi * 2
		if next <= hi { // int overflow guard
			break
		}
		hi = next
	}
	for lo < hi {
		mid := lo + (hi-lo)/2
		if idx.tokensInSlice(off, mid) < float64(targetTokens) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// resolveTokenizeTarget extracts the (base URL, model, api key) that
// --replay-exact-tokens probes/indexes against: modelSpec's first base
// URL, mirroring newReplayPoster's own resolution (replay_router_post.go) —
// a dynamic model spec is required, and the model name is discovered from
// /models when the spec omits model=....
func resolveTokenizeTarget(modelSpec string, keys llm.APIKeys) (base, model, apiKey string, err error) {
	if !llm.IsDynamicModel(modelSpec) {
		return "", "", "", fmt.Errorf("router-replay requires a dynamic/ model spec; got %q", modelSpec)
	}
	dyn, err := llm.ParseDynamicModel(modelSpec)
	if err != nil {
		return "", "", "", fmt.Errorf("parse model spec: %w", err)
	}
	if len(dyn.BaseURLs) == 0 {
		return "", "", "", fmt.Errorf("no base URL in model spec")
	}
	base = strings.TrimRight(dyn.BaseURLs[0], "/")
	apiKey = keys.Anthropic
	if dyn.Type == "openai" || dyn.Type == "openai_vllm" {
		apiKey = keys.OpenAI
	}
	if apiKey == "" {
		apiKey = "dummy-key"
	}
	model = dyn.Model
	if model == "" {
		discovered, derr := discoverModelName(base)
		if derr != nil {
			return "", "", "", fmt.Errorf("model=... not set in %q and model discovery from %s failed: %w", modelSpec, base, derr)
		}
		model = discovered
	}
	return base, model, apiKey, nil
}

// buildReplayTokenIndex resolves the first target endpoint from modelSpec
// and builds the corpus token index against it. Called once at startup by
// RunAutoBenchmark when --replay-exact-tokens is set; the resulting index
// is shared (via AutoBenchmarkConfig.replayTokenIndex) by every poster the
// run constructs — this is the ONLY corpus-indexing pass for the whole run,
// regardless of how many series/instances/endpoints it fans out to.
func buildReplayTokenIndex(modelSpec string, keys llm.APIKeys, docs string) (*corpusTokenIndex, error) {
	base, model, apiKey, err := resolveTokenizeTarget(modelSpec, keys)
	if err != nil {
		return nil, err
	}
	return buildCorpusTokenIndex(base, model, apiKey, docs, tokenizeChunkChars)
}
