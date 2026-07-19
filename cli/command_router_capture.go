package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
)

// --- Redacted request schema (_schema: "req-v1") ---

type redactedRequest struct {
	Schema        string                `json:"_schema"`
	OriginalBytes int                   `json:"original_bytes"`
	Model         string                `json:"model,omitempty"`
	MaxTokens     int                   `json:"max_tokens,omitempty"`
	Stream        bool                  `json:"stream,omitempty"`
	Thinking      json.RawMessage       `json:"thinking,omitempty"`
	Temperature   *float64              `json:"temperature,omitempty"`
	TopP          *float64              `json:"top_p,omitempty"`
	TopK          *int                  `json:"top_k,omitempty"`
	SystemBlocks  []redactedSystemBlock `json:"system_blocks,omitempty"`
	Tools         *redactedToolsInfo    `json:"tools,omitempty"`
	Messages      []redactedMessage     `json:"messages,omitempty"`
	ExtraKeys     []string              `json:"extra_keys,omitempty"`
	ParseError    string                `json:"parse_error,omitempty"`
}

type redactedSystemBlock struct {
	Type         string `json:"type"`
	Hash         string `json:"hash"`
	Bytes        int    `json:"bytes"`
	Tokens       int    `json:"tokens,omitempty"`
	CacheControl string `json:"cache_control,omitempty"`
}

type redactedToolsInfo struct {
	Count  int    `json:"count"`
	Bytes  int    `json:"bytes"`
	Tokens int    `json:"tokens,omitempty"`
	Hash   string `json:"hash"`
}

type redactedMessage struct {
	Role          string   `json:"role"`
	Hash          string   `json:"hash"`
	BlockTypes    []string `json:"block_types"`
	Bytes         int      `json:"bytes"`
	Tokens        int      `json:"tokens,omitempty"`
	CacheControl  string   `json:"cache_control,omitempty"`
	ToolUseIDs    []string `json:"tool_use_ids,omitempty"`
	ToolResultIDs []string `json:"tool_result_ids,omitempty"`
	// SeedHash is sha256Short of the concatenated text of a user message
	// after stripping <system-reminder>...</system-reminder> wrappers. Set
	// only when the message has text content and no tool_result blocks —
	// i.e., it's a candidate spawn-seed for an agent instance.
	SeedHash string `json:"seed_hash,omitempty"`
}

// --- Redacted response schema (_schema: "resp-v1") ---

type redactedResponse struct {
	Schema        string                `json:"_schema"`
	OriginalBytes int                   `json:"original_bytes"`
	MessageID     string                `json:"message_id,omitempty"`
	Usage         *redactedUsage        `json:"usage,omitempty"`
	StopReason    string                `json:"stop_reason,omitempty"`
	StopSequence  string                `json:"stop_sequence,omitempty"`
	OutputBlocks  []redactedOutputBlock `json:"output_blocks,omitempty"`
	PlainJSON     json.RawMessage       `json:"plain_json,omitempty"`
	ParseError    string                `json:"parse_error,omitempty"`
}

type redactedUsage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	OutputTokens             int `json:"output_tokens"`
}

type redactedOutputBlock struct {
	Type         string `json:"type"`
	Bytes        int    `json:"bytes"`
	Tokens       int    `json:"tokens,omitempty"`
	ToolNameHash string `json:"tool_name_hash,omitempty"`
	ToolUseID    string `json:"tool_use_id,omitempty"`
	// PromptHash is sha256Short of the tool's input.prompt string when
	// present. Same primitive as ToolNameHash, applied to a string we
	// would otherwise discard. Lets a downstream spawn-seed (a sub-agent's
	// first user message) be matched against the spawning tool call.
	PromptHash string `json:"prompt_hash,omitempty"`
}

// sha256Short returns "sha256:<first 16 hex chars>" — short enough for a
// compact capture file, long enough that collisions across a session are
// astronomically unlikely.
func sha256Short(data []byte) string {
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:])[:16]
}

// systemReminderRE matches <system-reminder>...</system-reminder> blocks
// (including newlines) plus any trailing whitespace, so that a spawn-seed
// hashed on the child side strips identically to what the parent emitted.
var systemReminderRE = regexp.MustCompile(`(?s)<system-reminder>.*?</system-reminder>\s*`)

// stripSystemReminders removes any <system-reminder>...</system-reminder>
// blocks from s and trims the result. The router doesn't add these — Claude
// Code wraps spawn prompts client-side — so stripping makes the seed text
// byte-equal to the prompt the parent passed via Agent.input.prompt.
func stripSystemReminders(s string) string {
	return strings.TrimSpace(systemReminderRE.ReplaceAllString(s, ""))
}

// BuildRedactedRequest parses an Anthropic /v1/messages request body into
// the req-v1 schema and returns its JSON form. Per-block token counts are
// NOT populated here (no access to the paired response's usage). Use
// BuildRedactedPair when the response is available so tokens can be
// allocated proportionally from server-reported input_tokens.
func BuildRedactedRequest(bodyBytes []byte) json.RawMessage {
	result := buildRedactedRequestStruct(bodyBytes)
	out, _ := json.Marshal(result)
	return out
}

// BuildRedactedPair builds both sides and allocates per-block token counts
// from the response's server-reported usage, spread over blocks by byte
// share. The returned RawMessages are byte-identical to the separate
// Build*/Build* versions for fields other than the new Tokens fields.
func BuildRedactedPair(reqBytes, respBytes []byte) (json.RawMessage, json.RawMessage) {
	req := buildRedactedRequestStruct(reqBytes)
	resp := buildRedactedResponseStruct(respBytes)
	allocateTokensFromUsage(&req, &resp)
	reqOut, _ := json.Marshal(req)
	respOut, _ := json.Marshal(resp)
	return reqOut, respOut
}

func buildRedactedRequestStruct(bodyBytes []byte) redactedRequest {
	result := redactedRequest{Schema: "req-v1", OriginalBytes: len(bodyBytes)}
	if len(bodyBytes) == 0 {
		return result
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(bodyBytes, &raw); err != nil {
		result.ParseError = err.Error()
		return result
	}

	known := map[string]bool{
		"model": true, "max_tokens": true, "stream": true,
		"thinking": true, "temperature": true, "top_p": true, "top_k": true,
		"system": true, "tools": true, "messages": true, "metadata": true,
	}
	for k := range raw {
		if !known[k] {
			result.ExtraKeys = append(result.ExtraKeys, k)
		}
	}
	if len(result.ExtraKeys) > 1 {
		// Stable ordering for diffs/testing.
		sortStrings(result.ExtraKeys)
	}

	if v, ok := raw["model"]; ok {
		_ = json.Unmarshal(v, &result.Model)
	}
	if v, ok := raw["max_tokens"]; ok {
		_ = json.Unmarshal(v, &result.MaxTokens)
	}
	if v, ok := raw["stream"]; ok {
		_ = json.Unmarshal(v, &result.Stream)
	}
	if v, ok := raw["thinking"]; ok {
		result.Thinking = v
	}
	if v, ok := raw["temperature"]; ok {
		var f float64
		if err := json.Unmarshal(v, &f); err == nil {
			result.Temperature = &f
		}
	}
	if v, ok := raw["top_p"]; ok {
		var f float64
		if err := json.Unmarshal(v, &f); err == nil {
			result.TopP = &f
		}
	}
	if v, ok := raw["top_k"]; ok {
		var i int
		if err := json.Unmarshal(v, &i); err == nil {
			result.TopK = &i
		}
	}
	if v, ok := raw["system"]; ok {
		result.SystemBlocks = parseSystemBlocks(v)
	}
	if v, ok := raw["tools"]; ok {
		result.Tools = parseToolsSummary(v)
	}
	if v, ok := raw["messages"]; ok {
		result.Messages = parseMessages(v)
	}

	return result
}

func parseSystemBlocks(raw json.RawMessage) []redactedSystemBlock {
	// Plain string form.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []redactedSystemBlock{{Type: "text", Hash: sha256Short([]byte(s)), Bytes: len(s)}}
	}
	// Array form.
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil
	}
	out := make([]redactedSystemBlock, 0, len(arr))
	for _, item := range arr {
		var b map[string]json.RawMessage
		if err := json.Unmarshal(item, &b); err != nil {
			continue
		}
		sb := redactedSystemBlock{Type: "text"}
		if t, ok := b["type"]; ok {
			_ = json.Unmarshal(t, &sb.Type)
		}
		canonical, _ := json.Marshal(b)
		sb.Hash = sha256Short(canonical)
		sb.Bytes = len(canonical)
		sb.CacheControl = extractCacheControl(b)
		out = append(out, sb)
	}
	return out
}

func parseToolsSummary(raw json.RawMessage) *redactedToolsInfo {
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil || len(arr) == 0 {
		return nil
	}
	// Canonical marshal of the decoded structure gives a stable hash across
	// key-order variations in the source.
	canonical, _ := json.Marshal(arr)
	return &redactedToolsInfo{
		Count: len(arr),
		Bytes: len(canonical),
		Hash:  sha256Short(canonical),
	}
}

func parseMessages(raw json.RawMessage) []redactedMessage {
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil
	}
	out := make([]redactedMessage, 0, len(arr))
	for _, m := range arr {
		if msg := parseOneMessage(m); msg != nil {
			out = append(out, *msg)
		}
	}
	return out
}

func parseOneMessage(raw json.RawMessage) *redactedMessage {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}
	rm := &redactedMessage{}
	if v, ok := m["role"]; ok {
		_ = json.Unmarshal(v, &rm.Role)
	}

	var (
		blockHashes   []string
		cacheControls []string
		// seedTexts collects raw text of any text blocks; if the message is
		// a user message with only text content (no tool_result), we'll
		// concatenate, strip system-reminder wrappers, and hash → SeedHash.
		seedTexts  []string
		hasNonText bool
	)
	if content, ok := m["content"]; ok {
		var s string
		if err := json.Unmarshal(content, &s); err == nil {
			// Plain-string content: treat as one text block.
			rm.BlockTypes = []string{"text"}
			blockHashes = []string{sha256Short([]byte(s))}
			seedTexts = append(seedTexts, s)
		} else {
			var blocks []json.RawMessage
			if err := json.Unmarshal(content, &blocks); err == nil {
				for _, b := range blocks {
					bt, h, cc, toolUseID, toolResultID := parseContentBlock(b)
					rm.BlockTypes = append(rm.BlockTypes, bt)
					blockHashes = append(blockHashes, h)
					if cc != "" {
						cacheControls = append(cacheControls, cc)
					}
					if toolUseID != "" {
						rm.ToolUseIDs = append(rm.ToolUseIDs, toolUseID)
					}
					if toolResultID != "" {
						rm.ToolResultIDs = append(rm.ToolResultIDs, toolResultID)
					}
					if bt == "text" {
						var bb struct {
							Text string `json:"text"`
						}
						if err := json.Unmarshal(b, &bb); err == nil {
							seedTexts = append(seedTexts, bb.Text)
						}
					} else {
						hasNonText = true
					}
				}
			}
		}
	}

	// SeedHash: hash the concatenated text content of a user-role message
	// (system-reminder wrappers stripped) so a sub-agent's first turn can
	// be matched by hash equality against its parent's spawn prompt. Only
	// set when the message is user-role and contains no non-text blocks
	// (tool_result and friends signal this is not a spawn seed).
	if rm.Role == "user" && !hasNonText && len(seedTexts) > 0 {
		joined := strings.Join(seedTexts, "\n")
		stripped := stripSystemReminders(joined)
		if stripped != "" {
			rm.SeedHash = sha256Short([]byte(stripped))
		}
	}

	if len(blockHashes) > 0 {
		rm.Hash = sha256Short([]byte(strings.Join(blockHashes, "|")))
	}
	if len(cacheControls) > 0 {
		rm.CacheControl = cacheControls[0]
		for _, cc := range cacheControls[1:] {
			if cc != cacheControls[0] {
				log.Printf("capture: message has mixed cache_control values %v", cacheControls)
				break
			}
		}
	}
	canonical, _ := json.Marshal(m)
	rm.Bytes = len(canonical)
	return rm
}

// parseContentBlock returns (blockType, blockHash, cacheControl, toolUseID, toolResultID).
// toolUseID is set for tool_use blocks, toolResultID for tool_result blocks
// (the ID the result references via tool_use_id).
func parseContentBlock(raw json.RawMessage) (string, string, string, string, string) {
	var b map[string]json.RawMessage
	if err := json.Unmarshal(raw, &b); err != nil {
		return "unknown", sha256Short(raw), "", "", ""
	}
	var blockType string
	if t, ok := b["type"]; ok {
		_ = json.Unmarshal(t, &blockType)
	}
	cacheControl := extractCacheControl(b)

	var toolUseID, toolResultID string
	switch blockType {
	case "tool_use":
		if id, ok := b["id"]; ok {
			_ = json.Unmarshal(id, &toolUseID)
		}
		// Tool name is hashed as part of the canonical block JSON below —
		// no separate storage needed since the block's hash covers it.
	case "tool_result":
		if id, ok := b["tool_use_id"]; ok {
			_ = json.Unmarshal(id, &toolResultID)
		}
	}

	canonical, _ := json.Marshal(b)
	return blockType, sha256Short(canonical), cacheControl, toolUseID, toolResultID
}

func extractCacheControl(block map[string]json.RawMessage) string {
	cc, ok := block["cache_control"]
	if !ok {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(cc, &m); err != nil {
		return ""
	}
	if t, ok := m["type"].(string); ok {
		return t
	}
	return ""
}

// BuildRedactedResponse handles three response shapes:
//   - plain-JSON (e.g. /v1/messages/count_tokens returns {"input_tokens": N})
//   - plain-JSON Anthropic non-streaming message response
//   - SSE stream of message_start / content_block_* / message_delta events
func BuildRedactedResponse(bodyBytes []byte) json.RawMessage {
	result := buildRedactedResponseStruct(bodyBytes)
	out, _ := json.Marshal(result)
	return out
}

func buildRedactedResponseStruct(bodyBytes []byte) redactedResponse {
	result := redactedResponse{Schema: "resp-v1", OriginalBytes: len(bodyBytes)}
	if len(bodyBytes) == 0 {
		return result
	}

	trimmed := strings.TrimLeft(string(bodyBytes), " \t\r\n")
	if strings.HasPrefix(trimmed, "{") {
		var probe map[string]json.RawMessage
		if err := json.Unmarshal(bodyBytes, &probe); err == nil {
			// count_tokens response: compact object keyed only on token counts.
			if _, hasInput := probe["input_tokens"]; hasInput && probe["content"] == nil && probe["usage"] == nil {
				result.PlainJSON = append([]byte(nil), bodyBytes...)
				return result
			}
			// Full non-streaming Anthropic response.
			if _, hasID := probe["id"]; hasID {
				if _, hasType := probe["type"]; hasType {
					parsePlainMessageResponse(bodyBytes, &result)
					return result
				}
			}
		}
	}

	parseSSEResponse(bodyBytes, &result)
	return result
}

// allocateTokensFromUsage spreads the server-reported input/output token
// counts across the redacted request's blocks and the response's output
// blocks by byte share. Writes per-block Tokens fields in place.
//
// The input pool includes cached tokens (input_tokens + cache_read +
// cache_creation) since we're allocating across the full input that was
// tokenized regardless of what hit cache. For Kimi/vLLM where cache
// fields are absent, the totals reduce to just input_tokens / output_tokens
// and allocation still works.
func allocateTokensFromUsage(req *redactedRequest, resp *redactedResponse) {
	if resp.Usage == nil {
		return
	}
	totalInputTokens := resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens
	if totalInputTokens > 0 {
		var totalBytes int64
		for _, m := range req.Messages {
			totalBytes += int64(m.Bytes)
		}
		for _, s := range req.SystemBlocks {
			totalBytes += int64(s.Bytes)
		}
		if req.Tools != nil {
			totalBytes += int64(req.Tools.Bytes)
		}
		if totalBytes > 0 {
			for i := range req.Messages {
				req.Messages[i].Tokens = int(int64(req.Messages[i].Bytes) * int64(totalInputTokens) / totalBytes)
			}
			for i := range req.SystemBlocks {
				req.SystemBlocks[i].Tokens = int(int64(req.SystemBlocks[i].Bytes) * int64(totalInputTokens) / totalBytes)
			}
			if req.Tools != nil {
				req.Tools.Tokens = int(int64(req.Tools.Bytes) * int64(totalInputTokens) / totalBytes)
			}
		}
	}

	if resp.Usage.OutputTokens > 0 && len(resp.OutputBlocks) > 0 {
		var totalOut int64
		for _, b := range resp.OutputBlocks {
			totalOut += int64(b.Bytes)
		}
		if totalOut > 0 {
			for i := range resp.OutputBlocks {
				resp.OutputBlocks[i].Tokens = int(int64(resp.OutputBlocks[i].Bytes) * int64(resp.Usage.OutputTokens) / totalOut)
			}
		} else {
			// No per-block bytes (e.g., tool-only responses): split evenly.
			each := resp.Usage.OutputTokens / len(resp.OutputBlocks)
			for i := range resp.OutputBlocks {
				resp.OutputBlocks[i].Tokens = each
			}
		}
	}
}

func parsePlainMessageResponse(body []byte, result *redactedResponse) {
	var msg struct {
		ID           string          `json:"id"`
		Content      json.RawMessage `json:"content"`
		StopReason   string          `json:"stop_reason"`
		StopSequence string          `json:"stop_sequence"`
		Usage        struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &msg); err != nil {
		result.ParseError = err.Error()
		return
	}
	result.MessageID = msg.ID
	result.StopReason = msg.StopReason
	result.StopSequence = msg.StopSequence
	result.Usage = &redactedUsage{
		InputTokens:              msg.Usage.InputTokens,
		CacheCreationInputTokens: msg.Usage.CacheCreationInputTokens,
		CacheReadInputTokens:     msg.Usage.CacheReadInputTokens,
		OutputTokens:             msg.Usage.OutputTokens,
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(msg.Content, &blocks); err == nil {
		for _, b := range blocks {
			result.OutputBlocks = append(result.OutputBlocks, outputBlockFromPlain(b))
		}
	}
}

func outputBlockFromPlain(raw json.RawMessage) redactedOutputBlock {
	var b struct {
		Type  string                 `json:"type"`
		Text  string                 `json:"text"`
		Name  string                 `json:"name"`
		ID    string                 `json:"id"`
		Input map[string]interface{} `json:"input"`
	}
	_ = json.Unmarshal(raw, &b)
	ob := redactedOutputBlock{Type: b.Type, Bytes: len(b.Text)}
	if b.Type == "tool_use" {
		ob.ToolNameHash = sha256Short([]byte(b.Name))
		ob.ToolUseID = b.ID
		// Hash input.prompt (if the tool supplies one). Same primitive as
		// ToolNameHash applied to a string we currently discard. Generic
		// across tool names — any tool that names its prompt field "prompt"
		// gets a matchable hash with no Agent-specific gating.
		// stripSystemReminders matches what SeedHash does on the child side
		// so the two hashes survive Claude Code's spawn-prompt wrapping.
		if p, ok := b.Input["prompt"].(string); ok {
			stripped := stripSystemReminders(p)
			if stripped != "" {
				ob.PromptHash = sha256Short([]byte(stripped))
			}
		}
	}
	if ob.Bytes == 0 {
		// For non-text blocks, take the whole block's JSON length as a proxy.
		ob.Bytes = len(raw)
	}
	return ob
}

func parseSSEResponse(body []byte, result *redactedResponse) {
	lines := strings.Split(string(body), "\n")
	currentBlockIdx := -1
	// inputAccum holds incremental partial_json bytes per block so we can
	// parse out input.prompt at content_block_stop. Keyed by block index;
	// only populated for tool_use blocks. Bounded to one prompt's worth of
	// JSON per active block — released when the block stops.
	inputAccum := map[int]*strings.Builder{}
	finalizeBlock := func(idx int) {
		bld, ok := inputAccum[idx]
		if !ok {
			return
		}
		delete(inputAccum, idx)
		if idx < 0 || idx >= len(result.OutputBlocks) {
			return
		}
		if result.OutputBlocks[idx].Type != "tool_use" || bld.Len() == 0 {
			return
		}
		var input map[string]interface{}
		if err := json.Unmarshal([]byte(bld.String()), &input); err != nil {
			return
		}
		if p, ok := input["prompt"].(string); ok {
			stripped := stripSystemReminders(p)
			if stripped != "" {
				result.OutputBlocks[idx].PromptHash = sha256Short([]byte(stripped))
			}
		}
	}
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[len("data: "):]
		if strings.TrimSpace(data) == "[DONE]" {
			continue
		}
		var ev map[string]json.RawMessage
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		var eventType string
		if t, ok := ev["type"]; ok {
			_ = json.Unmarshal(t, &eventType)
		}
		switch eventType {
		case "message_start":
			if m, ok := ev["message"]; ok {
				var ms struct {
					ID    string `json:"id"`
					Usage struct {
						InputTokens              int `json:"input_tokens"`
						CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
						CacheReadInputTokens     int `json:"cache_read_input_tokens"`
						OutputTokens             int `json:"output_tokens"`
					} `json:"usage"`
				}
				if err := json.Unmarshal(m, &ms); err == nil {
					result.MessageID = ms.ID
					result.Usage = &redactedUsage{
						InputTokens:              ms.Usage.InputTokens,
						CacheCreationInputTokens: ms.Usage.CacheCreationInputTokens,
						CacheReadInputTokens:     ms.Usage.CacheReadInputTokens,
						OutputTokens:             ms.Usage.OutputTokens,
					}
				}
			}
		case "content_block_start":
			if idx, ok := ev["index"]; ok {
				_ = json.Unmarshal(idx, &currentBlockIdx)
			}
			if cb, ok := ev["content_block"]; ok {
				var b struct {
					Type string `json:"type"`
					Name string `json:"name"`
					ID   string `json:"id"`
				}
				_ = json.Unmarshal(cb, &b)
				ob := redactedOutputBlock{Type: b.Type}
				if b.Type == "tool_use" {
					ob.ToolNameHash = sha256Short([]byte(b.Name))
					ob.ToolUseID = b.ID
					// Begin accumulating partial_json so we can hash input.prompt
					// at content_block_stop. Allocated only for tool_use blocks.
					inputAccum[currentBlockIdx] = &strings.Builder{}
				}
				// Keep OutputBlocks indexed by currentBlockIdx so that deltas
				// (which reference index) line up even if events arrive out of order.
				for len(result.OutputBlocks) <= currentBlockIdx {
					result.OutputBlocks = append(result.OutputBlocks, redactedOutputBlock{})
				}
				result.OutputBlocks[currentBlockIdx] = ob
			}
		case "content_block_delta":
			if idx, ok := ev["index"]; ok {
				_ = json.Unmarshal(idx, &currentBlockIdx)
			}
			if delta, ok := ev["delta"]; ok {
				var d struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
					Thinking    string `json:"thinking"`
				}
				if err := json.Unmarshal(delta, &d); err == nil {
					add := len(d.Text) + len(d.PartialJSON) + len(d.Thinking)
					if currentBlockIdx >= 0 && currentBlockIdx < len(result.OutputBlocks) {
						result.OutputBlocks[currentBlockIdx].Bytes += add
					}
					if d.PartialJSON != "" {
						if bld, ok := inputAccum[currentBlockIdx]; ok {
							bld.WriteString(d.PartialJSON)
						}
					}
				}
			}
		case "content_block_stop":
			idx := currentBlockIdx
			if i, ok := ev["index"]; ok {
				_ = json.Unmarshal(i, &idx)
			}
			finalizeBlock(idx)
		case "message_delta":
			if delta, ok := ev["delta"]; ok {
				var d struct {
					StopReason   string `json:"stop_reason"`
					StopSequence string `json:"stop_sequence"`
				}
				if err := json.Unmarshal(delta, &d); err == nil {
					if d.StopReason != "" {
						result.StopReason = d.StopReason
					}
					if d.StopSequence != "" {
						result.StopSequence = d.StopSequence
					}
				}
			}
			if usage, ok := ev["usage"]; ok {
				var u struct {
					InputTokens              int `json:"input_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					OutputTokens             int `json:"output_tokens"`
				}
				if err := json.Unmarshal(usage, &u); err == nil {
					if result.Usage == nil {
						result.Usage = &redactedUsage{}
					}
					// message_delta.usage is final; override message_start values.
					if u.InputTokens != 0 {
						result.Usage.InputTokens = u.InputTokens
					}
					if u.CacheCreationInputTokens != 0 {
						result.Usage.CacheCreationInputTokens = u.CacheCreationInputTokens
					}
					if u.CacheReadInputTokens != 0 {
						result.Usage.CacheReadInputTokens = u.CacheReadInputTokens
					}
					if u.OutputTokens != 0 {
						result.Usage.OutputTokens = u.OutputTokens
					}
				}
			}
		}
	}
	// Fallback: finalize any blocks that didn't see a content_block_stop
	// (truncated capture). Iterate over a snapshot of remaining keys; the
	// finalizer deletes from the map as it runs.
	for idx := range inputAccum {
		finalizeBlock(idx)
	}
}

func sortStrings(s []string) {
	// Small, avoids importing "sort" at the call site for this one use.
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

// Prevent fmt import from being marked unused when we strip debug paths.
var _ = fmt.Sprintf
