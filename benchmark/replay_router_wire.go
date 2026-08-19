package benchmark

// Wire reconstruction for tree-aware router replay.
//
// The replay-v3 file carries a per-request structured spec (system_blocks,
// tools, messages) lifted directly from the redacted capture. The original
// content is gone, but the schema preserved enough metadata — block types,
// per-block sizes (bytes + tokens), tool_use / tool_result ids,
// cache_control flags, and a hash per block — that we can rebuild an HTTP
// body whose shape and size match the original wire form.
//
// Determinism: every synthesized block is keyed by its original hash, so
// two requests that referenced the same block (same hash) emit byte-
// identical bytes for it. That preserves the server's prefix-cache hit
// pattern as the original capture experienced it.
//
// Content source: bytes are drawn from the embedded docs corpus, sliced
// at an offset derived from the hash. For sizes larger than the corpus
// we wrap around. Tools are synthesized as N valid tool definitions
// proportioned to fill the original tools.bytes total.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// verboseOutputInstruction is appended as a system block/message in every
// replay mode. Retargeting max_tokens upward (e.g. via --replay-output-ratio)
// is a no-op unless the model is actually nudged to keep generating instead
// of stopping at its natural response length, so this instruction is injected
// to make the cap load-bearing.
const verboseOutputInstruction = "Provide a thorough, detailed response and keep elaborating rather than stopping early."

// replayLengthAsk names the CURRENT request's output budget to the model, in
// words. The generic keep-elaborating instruction holds ~50% conformity on
// large budgets and stronger generic wording measured WORSE — a model has no
// way to know how much elaboration is wanted, so it guesses. The budget is
// known per request, and the only cache-safe place for per-request text is
// the tail, after the shared history: a varying instruction in the system
// slot would fork every request's prefix at the top and destroy the replay's
// sharing structure.
//
// The ask names TWICE the captured length (1 token ~ 0.75 English words, so
// words = 1.5 x tokens ~ 2x the true size). Overshooting is free — the server
// clamps at max_tokens — and the model delivers a consistent fraction of
// whatever figure it is given. The curve measured at 300 requests: asking the
// exact length yields 84.8% conformity, twice yields 90.5%, four times drops
// to 72.6% — a number too far past plausible gets discounted the same way
// "write infinitely" does. Two is the peak, hard-coded rather than exposed:
// it compensates for how models discount length asks, which is not a property
// an operator tunes per run.
//
// Below ~16 tokens no ask is made: "write about 6 words" reads as a trick,
// and tiny budgets conform by clamping anyway.
func replayLengthAsk(maxTokens int) string {
	if maxTokens < 16 {
		return ""
	}
	words := maxTokens * 3 / 2
	return fmt.Sprintf("\n\nWrite a response of at least %d words. Keep elaborating with relevant"+
		" detail until you reach that length — do not stop short of it.", words)
}

// buildAnthropicMessagesBody reconstructs the body of a POST /v1/messages
// request that matches the original capture's shape and size as closely
// as possible. Returns the marshaled JSON bytes and the canonical text
// string (all synthesized content concatenated in generation order) for
// feeding into the content-level cache estimator.
//
// inj carries the UUID cache-coherency injection (--verify,
// router path — see replay_router_uuid.go); nil means "no injection",
// leaving the body byte-for-byte identical to before this feature existed.
func buildAnthropicMessagesBody(req RouterReplayRequest, docs string, modelName string, runID string, outputRatio float64, forceVolume bool, charsPerToken float64, inj *uuidInjection) ([]byte, string, error) {
	var stampByHash map[string]turnStamp
	if inj != nil {
		stampByHash = inj.StampByHash
	}
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": pickMaxTokens(req, outputRatio),
		"stream":     req.Stream,
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.TopK != nil {
		body["top_k"] = *req.TopK
	}
	if len(req.Thinking) > 0 {
		body["thinking"] = json.RawMessage(req.Thinking)
	}

	var systemArr []map[string]interface{}
	if len(req.SystemBlocks) > 0 {
		systemArr = buildSystem(effectiveSystemBlocks(req.SystemBlocks), docs, charsPerToken)
	}
	if runID != "" {
		stamp := map[string]interface{}{
			"type": "text",
			"text": fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID),
		}
		systemArr = append([]map[string]interface{}{stamp}, systemArr...)
	}

	// Both modes carry the instruction — see the openai path's comment.
	systemArr = append(systemArr, map[string]interface{}{
		"type": "text",
		"text": verboseOutputInstruction,
	})
	if len(systemArr) > 0 {
		body["system"] = systemArr
	}
	if req.Tools != nil && req.Tools.Count > 0 {
		body["tools"] = buildTools(req.Tools, docs, charsPerToken)
	}
	var msgs []map[string]interface{}
	if len(req.Messages) > 0 {
		msgs = buildMessages(req.Messages, docs, charsPerToken, stampByHash)
	}
	tail := ""
	if inj != nil {
		tail = replayReciteWindowInstruction(inj.ReciteLabels)
	}
	tail += replayLengthAsk(pickMaxTokens(req, outputRatio))
	if tail != "" {
		msgs = appendTailMessageAnthropic(msgs, tail)
	}
	if len(msgs) > 0 {
		body["messages"] = msgs
	}

	// Collect canonical text for the cache estimator. Uses the SAME
	// buildMessageContent(m, docs, stampByHash) call as buildMessages above
	// so the cache-estimate canonical matches the wire body exactly
	// (including any inline turn-stamp markers).
	var canonical strings.Builder
	if runID != "" {
		canonical.WriteString(fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID))
	}
	for _, b := range effectiveSystemBlocks(req.SystemBlocks) {
		canonical.WriteString(synthText(b.Hash, sizeBudget(b.Bytes, b.Tokens, charsPerToken), docs))
	}
	if req.Tools != nil && req.Tools.Count > 0 {
		n := req.Tools.Count
		if n <= 0 {
			n = 1
		}
		const perToolOverhead = 120
		totalOverhead := perToolOverhead * n
		descBudget := sizeBudget(req.Tools.Bytes, req.Tools.Tokens, charsPerToken) - totalOverhead
		if descBudget < n*4 {
			descBudget = n * 4
		}
		perDesc := descBudget / n
		for i := 0; i < n; i++ {
			canonical.WriteString(synthText(fmt.Sprintf("%s:tool:%d", req.Tools.Hash, i), perDesc, docs))
		}
	}
	for _, m := range req.Messages {
		blocks := buildMessageContent(m, docs, charsPerToken, stampByHash)
		for _, blk := range blocks {
			if t, ok := blk["text"]; ok {
				canonical.WriteString(fmt.Sprintf("%v", t))
			}
			if t, ok := blk["content"]; ok {
				canonical.WriteString(fmt.Sprintf("%v", t))
			}
			if t, ok := blk["thinking"]; ok {
				canonical.WriteString(fmt.Sprintf("%v", t))
			}
			if t, ok := blk["input"]; ok {
				if mm, ok2 := t.(map[string]interface{}); ok2 {
					if p, ok3 := mm["_padding"]; ok3 {
						canonical.WriteString(fmt.Sprintf("%v", p))
					}
				}
			}
		}
	}
	if inj != nil {
		canonical.WriteString(replayReciteWindowInstruction(inj.ReciteLabels))
	}

	bodyBytes, err := json.Marshal(body)
	return bodyBytes, canonical.String(), err
}

// pickMaxTokens picks the per-request max_tokens cap. When outputRatio > 0
// (--replay-output-ratio) the cap is retargeted to outputRatio * InputTokens
// (the FULL prompt), so the target output:input ratio is defined against the
// full prompt. Sizing off only the new/uncached input was considered and
// rejected: with heavy prefix caching the new input is a small fraction of the
// prompt, which collapses output. This overrides the recorded output_tokens,
// which otherwise pins max_tokens to what the model produced in the original
// capture (making the model stop early on replay). Otherwise falls back to the
// original precedence: output_tokens, then max_tokens, then 1 as a guard.
func pickMaxTokens(req RouterReplayRequest, outputRatio float64) int {
	if outputRatio > 0 && req.InputTokens > 0 {
		n := int(math.Round(float64(req.InputTokens) * outputRatio))
		if n < 1 {
			n = 1
		}
		return n
	}
	if req.OutputTokens > 0 {
		return req.OutputTokens
	}
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}
	return 1
}

// effectiveSystemBlocks drops the per-request billing/preamble header block
// (SystemBlocks[0] when <200B) before it is emitted on the wire or fed to the
// cache estimator. That header carries a ~unique hash per request; unlike
// Anthropic's breakpoint cache (whose breakpoints sit on the SHARED system
// blocks that follow), it poisons vLLM's sequential prefix-block hashing —
// denying the GPU/weka cache a stable key for the shared system prompt behind
// it. Mirrors BuildReplayRequestPrefix (replay_router.go:730), which already
// skips exactly this block for the offline prefix definition.
func effectiveSystemBlocks(blocks []RouterReplaySystemBlock) []RouterReplaySystemBlock {
	if len(blocks) > 0 && blocks[0].Bytes < 200 {
		return blocks[1:]
	}
	return blocks
}

// buildSystem rebuilds the system array as a list of text blocks sized to
// match each original system block's bytes (or, with charsPerToken > 0, its
// captured token count). cache_control is preserved when present so the
// server caches at the same boundaries.
func buildSystem(blocks []RouterReplaySystemBlock, docs string, charsPerToken float64) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(blocks))
	for _, b := range blocks {
		entry := map[string]interface{}{
			"type": "text",
			"text": synthText(b.Hash, sizeBudget(b.Bytes, b.Tokens, charsPerToken), docs),
		}
		if b.CacheControl != "" {
			entry["cache_control"] = map[string]string{"type": b.CacheControl}
		}
		out = append(out, entry)
	}
	return out
}

// buildTools rebuilds an Anthropic tools array of Count entries that
// canonically marshals to approximately the original Bytes (or, with
// charsPerToken > 0, sized off the captured Tokens instead). Each tool has
// a stable name derived from the tools.hash + its index, and a description
// padded with docs to round out the total size.
func buildTools(spec *RouterReplayToolsSpec, docs string, charsPerToken float64) []map[string]interface{} {
	n := spec.Count
	if n <= 0 {
		n = 1
	}
	// Reserve approximate JSON overhead per tool (object braces, fixed
	// fields, brackets/commas): ~120 bytes. The remainder is description
	// padding distributed evenly.
	const perToolOverhead = 120
	totalOverhead := perToolOverhead * n
	descBudget := sizeBudget(spec.Bytes, spec.Tokens, charsPerToken) - totalOverhead
	if descBudget < n*4 {
		descBudget = n * 4
	}
	perDesc := descBudget / n
	tools := make([]map[string]interface{}, 0, n)
	for i := 0; i < n; i++ {
		t := map[string]interface{}{
			"name":        fmt.Sprintf("tool_%s_%03d", shortHashHex(spec.Hash), i),
			"description": synthText(fmt.Sprintf("%s:tool:%d", spec.Hash, i), perDesc, docs),
			"input_schema": map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		}
		tools = append(tools, t)
	}
	return tools
}

// buildMessages rebuilds the messages array — preserving role and block
// shape of each message. For text blocks we synthesize content sized to a
// share of the message's total Bytes. For tool_use and tool_result blocks
// we preserve the original ids verbatim so the conversation maintains a
// valid reference graph that matches what the original capture sent.
// stampByHash is inj.StampByHash (nil when --verify is off) —
// threaded down to buildMessageContent, which appends each qualifying
// turn's inline UUID marker to its own synthesized content.
func buildMessages(msgs []RouterReplayMessage, docs string, charsPerToken float64, stampByHash map[string]turnStamp) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(msgs))
	for _, m := range msgs {
		content := buildMessageContent(m, docs, charsPerToken, stampByHash)
		entry := map[string]interface{}{
			"role":    roleOrUser(m.Role),
			"content": content,
		}
		// Anthropic's cache_control sits on individual content blocks, not
		// on the message — but the redacted form aggregated message-level
		// cache_control. Apply it to the last content block of this
		// message (which is the canonical anchor point in the original).
		if m.CacheControl != "" && len(content) > 0 {
			last := content[len(content)-1]
			last["cache_control"] = map[string]string{"type": m.CacheControl}
			content[len(content)-1] = last
		}
		out = append(out, entry)
	}
	return out
}

func roleOrUser(role string) string {
	if role == "user" || role == "assistant" || role == "system" {
		return role
	}
	return "user"
}

// appendTailMessageAnthropic appends text (the windowed recite ask — see
// replayReciteWindowInstruction) to the end of an Anthropic messages array,
// preserving strict user/
// assistant role alternation (Anthropic rejects consecutive same-role
// messages, and "system" is not a valid role inside `messages` at all):
//   - if the array is empty, or the last message is role "assistant", a
//     NEW role="user" message carrying the text is appended (valid
//     alternation; a fresh assistant turn should never have text
//     injected into it after the fact).
//   - if the last message is role "user" (the common case — router-replay
//     requests carry history up to, but not including, the response being
//     generated), the text is appended as an additional content block on
//     THAT message instead of a new one.
func appendTailMessageAnthropic(msgs []map[string]interface{}, text string) []map[string]interface{} {
	if text == "" {
		return msgs
	}
	newUserMsg := map[string]interface{}{
		"role":    "user",
		"content": []map[string]interface{}{{"type": "text", "text": text}},
	}
	if len(msgs) == 0 {
		return append(msgs, newUserMsg)
	}
	last := msgs[len(msgs)-1]
	if last["role"] != "user" {
		return append(msgs, newUserMsg)
	}
	content, _ := last["content"].([]map[string]interface{})
	content = append(content, map[string]interface{}{"type": "text", "text": text})
	last["content"] = content
	msgs[len(msgs)-1] = last
	return msgs
}

// buildMessageContent materializes per-block content for one message.
// block_types lists the kinds in order; we use tool_use_ids and
// tool_result_ids to populate ids on the matching blocks (in order of
// appearance in block_types).
//
// stampByHash is inj.StampByHash (nil when --verify is off, or
// when this message isn't a qualifying turn). When m.Hash is a key in
// stampByHash, the labeled marker "\n\n[turn-N id: <uuid>]" is appended to
// the LAST "text"-type block's synthesized content, at a position wholly
// determined by m.Hash (same seed synthText already keys on) — so two
// requests carrying the same turn emit byte-identical content for it,
// preserving the within-session cache-hit property. Only count==1
// (genuinely per-session) messages are ever present in stampByHash — see
// isQualifyingUserTurn — so a cross-session-shared message is never
// perturbed.
func buildMessageContent(m RouterReplayMessage, docs string, charsPerToken float64, stampByHash map[string]turnStamp) []map[string]interface{} {
	nText := 0
	for _, t := range m.BlockTypes {
		if t == "text" {
			nText++
		}
	}
	textBudget := sizeBudget(m.Bytes, m.Tokens, charsPerToken)
	if textBudget < 0 {
		textBudget = 0
	}
	// Subtract a rough per-block JSON overhead per non-text block so we
	// don't massively overshoot the total bytes.
	const otherBlockOverhead = 80
	for _, t := range m.BlockTypes {
		if t != "text" {
			textBudget -= otherBlockOverhead
		}
	}
	if textBudget < nText*4 {
		textBudget = nText * 4
	}
	perText := 0
	if nText > 0 {
		perText = textBudget / nText
	}

	tuIdx, trIdx := 0, 0
	out := make([]map[string]interface{}, 0, len(m.BlockTypes))
	for i, t := range m.BlockTypes {
		blockSeed := fmt.Sprintf("%s:block:%d", m.Hash, i)
		switch t {
		case "text":
			out = append(out, map[string]interface{}{
				"type": "text",
				"text": synthText(blockSeed, perText, docs),
			})
		case "tool_use":
			id := ""
			if tuIdx < len(m.ToolUseIDs) {
				id = m.ToolUseIDs[tuIdx]
				tuIdx++
			}
			out = append(out, map[string]interface{}{
				"type":  "tool_use",
				"id":    id,
				"name":  fmt.Sprintf("tool_%s", shortHashHex(blockSeed)),
				"input": map[string]interface{}{"_padding": synthText(blockSeed+":input", 32, docs)},
			})
		case "tool_result":
			id := ""
			if trIdx < len(m.ToolResultIDs) {
				id = m.ToolResultIDs[trIdx]
				trIdx++
			}
			out = append(out, map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": id,
				"content":     synthText(blockSeed+":result", perText, docs),
			})
		case "thinking":
			out = append(out, map[string]interface{}{
				"type":     "thinking",
				"thinking": synthText(blockSeed+":thinking", perText, docs),
			})
		case "image":
			// We can't fake images cheaply; replace with a small text
			// placeholder so the request still validates.
			out = append(out, map[string]interface{}{
				"type": "text",
				"text": "[image placeholder]",
			})
		default:
			out = append(out, map[string]interface{}{
				"type": "text",
				"text": synthText(blockSeed+":fallback", perText, docs),
			})
		}
	}
	if stamp, ok := stampByHash[m.Hash]; ok {
		marker := "\n\n[" + stamp.Label + " id: " + stamp.UUID + "]"
		for i := len(out) - 1; i >= 0; i-- {
			if out[i]["type"] == "text" {
				out[i]["text"] = fmt.Sprintf("%v", out[i]["text"]) + marker
				break
			}
		}
	}
	return out
}

// synthText returns a deterministic string of approximately n bytes drawn
// from docs at an offset chosen by sha256(seed). Same seed → same bytes,
// so two replays of the same original block produce identical content and
// the server's prefix cache treats them identically.
//
// We slice the docs corpus starting at a hash-derived offset; for n
// larger than len(docs)-offset we wrap. A short hash-prefixed header
// makes the output visually distinct from neighbouring blocks for
// debugging purposes without affecting size determinism.
func synthText(seed string, n int, docs string) string {
	if n <= 0 {
		return ""
	}
	if len(docs) == 0 {
		// Fallback: repeat the hash so we always have *something* of the
		// right size.
		base := seed
		if base == "" {
			base = "x"
		}
		s := strings.Repeat(base, (n/len(base))+1)
		return s[:n]
	}
	off := hashOffset(seed, len(docs))
	var b strings.Builder
	b.Grow(n)
	for b.Len() < n {
		take := n - b.Len()
		room := len(docs) - off
		if room <= 0 {
			off = 0
			room = len(docs)
		}
		if take > room {
			take = room
		}
		b.WriteString(docs[off : off+take])
		off += take
	}
	return b.String()[:n]
}

// sizeBudget picks the byte budget for a content-sizing site. By default
// (charsPerToken == 0, or the block carries no captured Tokens count) it
// falls back to the capture's raw byte count — byte-faithful sizing.
// With --replay-chars-per-token set and tokens > 0, it instead returns
// round(tokens * charsPerToken): the serving tokenizer's count on synthetic
// text runs ~3-4 chars/token, so byte-faithful replay bodies under-tokenize
// relative to the original (production) capture; sizing off the captured
// token count lands the replay's token count near the original's.
func sizeBudget(bytes int, tokens int, charsPerToken float64) int {
	if charsPerToken > 0 && tokens > 0 {
		return int(math.Round(float64(tokens) * charsPerToken))
	}
	return bytes
}

func hashOffset(seed string, mod int) int {
	if mod <= 0 {
		return 0
	}
	h := sha256.Sum256([]byte(seed))
	v := binary.BigEndian.Uint64(h[:8])
	return int(v % uint64(mod))
}

func shortHashHex(seed string) string {
	h := sha256.Sum256([]byte(seed))
	return fmt.Sprintf("%x", h[:4])
}

// buildOpenAITools rebuilds an OpenAI function-tool array of Count entries
// sized to approximately spec.Bytes, mirroring buildTools but using the
// OpenAI {"type":"function","function":{...}} wrapper shape (~150 bytes/tool
// overhead vs ~120 for Anthropic).
func buildOpenAITools(spec *RouterReplayToolsSpec, docs string, charsPerToken float64) []map[string]interface{} {
	n := spec.Count
	if n <= 0 {
		n = 1
	}
	const perToolOverhead = 150
	totalOverhead := perToolOverhead * n
	descBudget := sizeBudget(spec.Bytes, spec.Tokens, charsPerToken) - totalOverhead
	if descBudget < n*4 {
		descBudget = n * 4
	}
	perDesc := descBudget / n
	tools := make([]map[string]interface{}, 0, n)
	for i := 0; i < n; i++ {
		tools = append(tools, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        fmt.Sprintf("tool_%s_%03d", shortHashHex(spec.Hash), i),
				"description": synthText(fmt.Sprintf("%s:tool:%d", spec.Hash, i), perDesc, docs),
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		})
	}
	return tools
}

// ---- OpenAI /v1/chat/completions body builder ----

// buildOpenAIChatCompletionsBody translates a router-replay request
// (originally captured in Anthropic /v1/messages format) into an
// OpenAI-compatible /v1/chat/completions body. The same deterministic synth
// content is used, so prefix-cache patterns are preserved. Returns the
// marshaled JSON bytes and a canonical text string for the cache estimator.
//
// Key translations:
//   - System blocks become "system"-role messages prepended to the messages
//     array. Anthropic's bare-string system is not supported in the replay
//     path (the capture always uses content-block arrays).
//   - Per-block content arrays are collapsed into plain strings where the
//     message only has text blocks (the common case). Multi-block messages
//     (text + tool_use, or text + tool_result) are downgraded to text-only
//     with a warning because OpenAI's tool-use wire format is structurally
//     different (tool_calls delta vs content blocks with ids). Tool-use
//     translation is deferred to a follow-up.
//   - Anthropic-specific fields (top_k, thinking) are dropped.
//   - Stream options with include_usage are set so we get token counts in
//     the final SSE chunk.
//
// inj carries the UUID cache-coherency injection (--verify,
// router path — see replay_router_uuid.go); nil means "no injection".
func buildOpenAIChatCompletionsBody(req RouterReplayRequest, docs string, modelName string, runID string, outputRatio float64, forceVolume bool, charsPerToken float64, inj *uuidInjection) ([]byte, string, error) {
	var stampByHash map[string]turnStamp
	if inj != nil {
		stampByHash = inj.StampByHash
	}
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": pickMaxTokens(req, outputRatio),
		"stream":     req.Stream,
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.Stream {
		body["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	}

	// Build messages array: system blocks first, then user/assistant messages.
	messages := make([]map[string]interface{}, 0)

	// System blocks become system-role messages at the front of the array.
	if len(req.SystemBlocks) > 0 {
		sysBlocks := buildSystem(effectiveSystemBlocks(req.SystemBlocks), docs, charsPerToken)
		for _, b := range sysBlocks {
			text := ""
			if t, ok := b["text"]; ok {
				text = fmt.Sprintf("%v", t)
			}
			messages = append(messages, map[string]interface{}{
				"role":    "system",
				"content": text,
			})
		}
	}
	if runID != "" {
		stamp := map[string]interface{}{
			"role":    "system",
			"content": fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID),
		}
		messages = append([]map[string]interface{}{stamp}, messages...)
	}

	// The keep-generating instruction rides in both modes, so forced and
	// unforced runs send byte-identical prompts and differ only by
	// ignore_eos — the clean A/B for whether prompting alone can hold output
	// at the captured budget.
	messages = append(messages, map[string]interface{}{
		"role":    "system",
		"content": verboseOutputInstruction,
	})
	if forceVolume {
		// --force-output-volume: vLLM ignores the stop token, so the
		// (possibly retargeted) budget is filled deterministically — 100%
		// conformity, at the cost of padding every response past its natural
		// end with degenerate output. The default leaves the prompt in
		// charge: ~90% conformity, genuine text throughout.
		body["ignore_eos"] = true
	}

	// Convert conversation messages. For text-only messages we produce a
	// plain-string content field (which OpenAI and most OpenAI-compatible
	// servers handle). Mixed-block messages (tool_use, tool_result, thinking)
	// are flattened to text: we extract the synthesized text content from
	// each block and join them with newlines, and log a one-time warning so
	// the operator knows tool fidelity is lost.
	if len(req.Messages) > 0 {
		openaiMsgs := buildOpenAIMessages(req.Messages, docs, charsPerToken, stampByHash)
		messages = append(messages, openaiMsgs...)
	}

	tailAsk := ""
	if inj != nil {
		tailAsk = replayReciteWindowInstruction(inj.ReciteLabels)
	}
	tailAsk += replayLengthAsk(pickMaxTokens(req, outputRatio))
	if tailAsk != "" {
		messages = appendTailMessageOpenAI(messages, tailAsk)
	}

	body["messages"] = messages
	if req.Tools != nil && req.Tools.Count > 0 {
		body["tools"] = buildOpenAITools(req.Tools, docs, charsPerToken)
	}

	// Collect canonical text for the cache estimator (system blocks +
	// messages). Uses the SAME buildMessageContent(m, docs, stampByHash)
	// call as buildOpenAIMessages above so the cache-estimate canonical
	// matches the wire body exactly (including any inline turn-stamp
	// markers).
	var canonical strings.Builder
	if runID != "" {
		canonical.WriteString(fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID))
	}
	for _, b := range effectiveSystemBlocks(req.SystemBlocks) {
		canonical.WriteString(synthText(b.Hash, sizeBudget(b.Bytes, b.Tokens, charsPerToken), docs))
	}
	for _, m := range req.Messages {
		blocks := buildMessageContent(m, docs, charsPerToken, stampByHash)
		for _, blk := range blocks {
			if t, ok := blk["text"]; ok {
				canonical.WriteString(fmt.Sprintf("%v", t))
			}
			if t, ok := blk["content"]; ok {
				canonical.WriteString(fmt.Sprintf("%v", t))
			}
			if t, ok := blk["thinking"]; ok {
				canonical.WriteString(fmt.Sprintf("%v", t))
			}
		}
	}
	if inj != nil {
		canonical.WriteString(replayReciteWindowInstruction(inj.ReciteLabels))
	}

	bodyBytes, err := json.Marshal(body)
	return bodyBytes, canonical.String(), err
}

// appendTailMessageOpenAI appends text (the windowed recite ask — see
// replayReciteWindowInstruction) to the end of an OpenAI messages array.
// Unlike Anthropic, OpenAI has no
// strict role-alternation requirement, but we still fold the text into the
// last message's content when that message is one the model would read as
// its own turn's input (user/system/tool) rather than always creating a
// new message — keeping the shape close to a real client's behavior. If
// the last message is "assistant" (or there are no messages at all), a new
// role="user" message carrying the text is appended instead.
func appendTailMessageOpenAI(msgs []map[string]interface{}, text string) []map[string]interface{} {
	if text == "" {
		return msgs
	}
	newUserMsg := map[string]interface{}{"role": "user", "content": text}
	if len(msgs) == 0 {
		return append(msgs, newUserMsg)
	}
	last := msgs[len(msgs)-1]
	role, _ := last["role"].(string)
	if role == "assistant" {
		return append(msgs, newUserMsg)
	}
	existing, _ := last["content"].(string)
	if existing != "" {
		last["content"] = existing + "\n\n" + text
	} else {
		last["content"] = text
	}
	msgs[len(msgs)-1] = last
	return msgs
}

// buildOpenAIMessages converts router-replay messages into OpenAI chat
// messages. tool_use blocks become tool_calls on the assistant message;
// tool_result blocks become separate role="tool" messages. Orphaned
// tool_result blocks (no matching prior tool_call) are folded into user text.
// stampByHash is threaded through to buildMessageContent (see its doc).
func buildOpenAIMessages(msgs []RouterReplayMessage, docs string, charsPerToken float64, stampByHash map[string]turnStamp) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(msgs))
	seenToolCallIDs := map[string]bool{}

	for _, m := range msgs {
		blocks := buildMessageContent(m, docs, charsPerToken, stampByHash)
		role := roleOrUser(m.Role)

		if role == "assistant" {
			var textParts []string
			var toolCalls []map[string]interface{}
			for _, blk := range blocks {
				switch blk["type"] {
				case "tool_use":
					id := fmt.Sprintf("%v", blk["id"])
					name := fmt.Sprintf("%v", blk["name"])
					argBytes, _ := json.Marshal(blk["input"])
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   id,
						"type": "function",
						"function": map[string]interface{}{
							"name":      name,
							"arguments": string(argBytes),
						},
					})
					seenToolCallIDs[id] = true
				case "text":
					textParts = append(textParts, fmt.Sprintf("%v", blk["text"]))
				case "thinking":
					textParts = append(textParts, fmt.Sprintf("%v", blk["thinking"]))
				default:
					if t, ok := blk["text"]; ok {
						textParts = append(textParts, fmt.Sprintf("%v", t))
					}
				}
			}
			msg := map[string]interface{}{
				"role":    "assistant",
				"content": strings.Join(textParts, "\n"),
			}
			if len(toolCalls) > 0 {
				msg["tool_calls"] = toolCalls
			}
			out = append(out, msg)
		} else {
			var textParts []string
			var toolMsgs []map[string]interface{}
			for _, blk := range blocks {
				switch blk["type"] {
				case "tool_result":
					id := fmt.Sprintf("%v", blk["tool_use_id"])
					if id != "" && seenToolCallIDs[id] {
						toolMsgs = append(toolMsgs, map[string]interface{}{
							"role":         "tool",
							"tool_call_id": id,
							"content":      fmt.Sprintf("%v", blk["content"]),
						})
					} else {
						// Orphan: fold content into user text to keep the request valid.
						textParts = append(textParts, fmt.Sprintf("%v", blk["content"]))
					}
				case "text":
					textParts = append(textParts, fmt.Sprintf("%v", blk["text"]))
				case "thinking":
					textParts = append(textParts, fmt.Sprintf("%v", blk["thinking"]))
				default:
					if t, ok := blk["text"]; ok {
						textParts = append(textParts, fmt.Sprintf("%v", t))
					}
				}
			}
			// Emit valid tool messages first, then user text if any (or always
			// when no tool messages were emitted).
			out = append(out, toolMsgs...)
			if len(textParts) > 0 || len(toolMsgs) == 0 {
				out = append(out, map[string]interface{}{
					"role":    role,
					"content": strings.Join(textParts, "\n"),
				})
			}
		}
	}
	return out
}
