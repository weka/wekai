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
	"strings"
)

// buildAnthropicMessagesBody reconstructs the body of a POST /v1/messages
// request that matches the original capture's shape and size as closely
// as possible. Returns the marshaled JSON bytes and the canonical text
// string (all synthesized content concatenated in generation order) for
// feeding into the content-level cache estimator.
func buildAnthropicMessagesBody(req RouterReplayRequest, docs string, modelName string, runID string) ([]byte, string, error) {
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": pickMaxTokens(req),
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
		systemArr = buildSystem(effectiveSystemBlocks(req.SystemBlocks), docs)
	}
	if runID != "" {
		stamp := map[string]interface{}{
			"type": "text",
			"text": fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID),
		}
		systemArr = append([]map[string]interface{}{stamp}, systemArr...)
	}
	if len(systemArr) > 0 {
		body["system"] = systemArr
	}
	if req.Tools != nil && req.Tools.Count > 0 {
		body["tools"] = buildTools(req.Tools, docs)
	}
	if len(req.Messages) > 0 {
		body["messages"] = buildMessages(req.Messages, docs)
	}

	// Collect canonical text for the cache estimator.
	var canonical strings.Builder
	if runID != "" {
		canonical.WriteString(fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID))
	}
	for _, b := range effectiveSystemBlocks(req.SystemBlocks) {
		canonical.WriteString(synthText(b.Hash, b.Bytes, docs))
	}
	if req.Tools != nil && req.Tools.Count > 0 {
		n := req.Tools.Count
		if n <= 0 {
			n = 1
		}
		const perToolOverhead = 120
		totalOverhead := perToolOverhead * n
		descBudget := req.Tools.Bytes - totalOverhead
		if descBudget < n*4 {
			descBudget = n * 4
		}
		perDesc := descBudget / n
		for i := 0; i < n; i++ {
			canonical.WriteString(synthText(fmt.Sprintf("%s:tool:%d", req.Tools.Hash, i), perDesc, docs))
		}
	}
	for _, m := range req.Messages {
		blocks := buildMessageContent(m, docs)
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

	bodyBytes, err := json.Marshal(body)
	return bodyBytes, canonical.String(), err
}

// pickMaxTokens applies the same precedence used elsewhere: original
// output_tokens, then original max_tokens, then 1 as a guard.
func pickMaxTokens(req RouterReplayRequest) int {
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
// match each original system block's bytes. cache_control is preserved
// when present so the server caches at the same boundaries.
func buildSystem(blocks []RouterReplaySystemBlock, docs string) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(blocks))
	for _, b := range blocks {
		entry := map[string]interface{}{
			"type": "text",
			"text": synthText(b.Hash, b.Bytes, docs),
		}
		if b.CacheControl != "" {
			entry["cache_control"] = map[string]string{"type": b.CacheControl}
		}
		out = append(out, entry)
	}
	return out
}

// buildTools rebuilds an Anthropic tools array of Count entries that
// canonically marshals to approximately the original Bytes. Each tool has
// a stable name derived from the tools.hash + its index, and a description
// padded with docs to round out the total size.
func buildTools(spec *RouterReplayToolsSpec, docs string) []map[string]interface{} {
	n := spec.Count
	if n <= 0 {
		n = 1
	}
	// Reserve approximate JSON overhead per tool (object braces, fixed
	// fields, brackets/commas): ~120 bytes. The remainder is description
	// padding distributed evenly.
	const perToolOverhead = 120
	totalOverhead := perToolOverhead * n
	descBudget := spec.Bytes - totalOverhead
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
func buildMessages(msgs []RouterReplayMessage, docs string) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(msgs))
	for _, m := range msgs {
		content := buildMessageContent(m, docs)
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

// buildMessageContent materializes per-block content for one message.
// block_types lists the kinds in order; we use tool_use_ids and
// tool_result_ids to populate ids on the matching blocks (in order of
// appearance in block_types).
func buildMessageContent(m RouterReplayMessage, docs string) []map[string]interface{} {
	nText := 0
	for _, t := range m.BlockTypes {
		if t == "text" {
			nText++
		}
	}
	textBudget := m.Bytes
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
func buildOpenAITools(spec *RouterReplayToolsSpec, docs string) []map[string]interface{} {
	n := spec.Count
	if n <= 0 {
		n = 1
	}
	const perToolOverhead = 150
	totalOverhead := perToolOverhead * n
	descBudget := spec.Bytes - totalOverhead
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
func buildOpenAIChatCompletionsBody(req RouterReplayRequest, docs string, modelName string, runID string) ([]byte, string, error) {
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": pickMaxTokens(req),
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
		sysBlocks := buildSystem(effectiveSystemBlocks(req.SystemBlocks), docs)
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

	// Convert conversation messages. For text-only messages we produce a
	// plain-string content field (which OpenAI and most OpenAI-compatible
	// servers handle). Mixed-block messages (tool_use, tool_result, thinking)
	// are flattened to text: we extract the synthesized text content from
	// each block and join them with newlines, and log a one-time warning so
	// the operator knows tool fidelity is lost.
	if len(req.Messages) > 0 {
		openaiMsgs := buildOpenAIMessages(req.Messages, docs)
		messages = append(messages, openaiMsgs...)
	}

	body["messages"] = messages
	if req.Tools != nil && req.Tools.Count > 0 {
		body["tools"] = buildOpenAITools(req.Tools, docs)
	}

	// Collect canonical text for the cache estimator (system blocks + messages).
	var canonical strings.Builder
	if runID != "" {
		canonical.WriteString(fmt.Sprintf("<ignore>RUN_GUID: %s</ignore>", runID))
	}
	for _, b := range effectiveSystemBlocks(req.SystemBlocks) {
		canonical.WriteString(synthText(b.Hash, b.Bytes, docs))
	}
	for _, m := range req.Messages {
		blocks := buildMessageContent(m, docs)
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

	bodyBytes, err := json.Marshal(body)
	return bodyBytes, canonical.String(), err
}

// buildOpenAIMessages converts router-replay messages into OpenAI chat
// messages. tool_use blocks become tool_calls on the assistant message;
// tool_result blocks become separate role="tool" messages. Orphaned
// tool_result blocks (no matching prior tool_call) are folded into user text.
func buildOpenAIMessages(msgs []RouterReplayMessage, docs string) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(msgs))
	seenToolCallIDs := map[string]bool{}

	for _, m := range msgs {
		blocks := buildMessageContent(m, docs)
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
