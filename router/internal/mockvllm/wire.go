package mockvllm

import (
	"encoding/json"
	"strconv"
	"strings"
)

// This file defines the OpenAI-compatible wire types this server accepts and
// emits. Field names and shapes are pinned to what
// benchmark/replay_router_post.go actually sends and parses (verified by
// reading buildOpenAIChatCompletionsBody, consumeOpenAIPlain, and
// consumeOpenAISSE) so that code path needs zero changes to run against this
// server instead of a real vLLM worker.

// chatMessage mirrors one OpenAI chat message. Content is decoded leniently:
// the replay poster always sends a plain string, but a curl/manual client may
// send the block-array form, so both are accepted.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// extractText pulls flat text out of a message's content field, whether it is
// a plain string (the replay poster's shape) or an array of
// {"type":"text","text":...} / {"content":...} blocks (general OpenAI
// clients). Unrecognized shapes contribute nothing rather than erroring — a
// mock backend should never fail a request over content it merely can't
// interpret for hashing purposes.
func extractText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []map[string]any
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var sb strings.Builder
		for _, b := range blocks {
			if t, ok := b["text"].(string); ok {
				sb.WriteString(t)
				sb.WriteByte('\n')
			} else if c, ok := b["content"].(string); ok {
				sb.WriteString(c)
				sb.WriteByte('\n')
			}
		}
		return sb.String()
	}
	return ""
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// chatCompletionRequest is POST /v1/chat/completions' body.
type chatCompletionRequest struct {
	Model         string          `json:"model"`
	Messages      []chatMessage   `json:"messages"`
	Stream        bool            `json:"stream"`
	StreamOptions *streamOptions  `json:"stream_options"`
	MaxTokens     int             `json:"max_tokens"`
	Tools         json.RawMessage `json:"tools"`
}

// promptBytes builds the canonical byte stream this request's prompt hashes
// over: tools (if any) first, then each message as "role:text\n" in order.
// vLLM hashes the flattened, chat-templated token sequence; this is the same
// idea at byte granularity — deterministic, order-sensitive, and stable
// across requests that repeat the same messages verbatim, which is exactly
// what lets two requests sharing a system prompt or conversation prefix land
// on the same trie path and be credited as cached.
func (r *chatCompletionRequest) promptBytes() []byte {
	var sb strings.Builder
	if len(r.Tools) > 0 && string(r.Tools) != "null" {
		sb.Write(r.Tools)
		sb.WriteByte('\n')
	}
	for _, m := range r.Messages {
		sb.WriteString(m.Role)
		sb.WriteByte(':')
		sb.WriteString(extractText(m.Content))
		sb.WriteByte('\n')
	}
	return []byte(sb.String())
}

func (r *chatCompletionRequest) wantsUsage() bool {
	return r.StreamOptions != nil && r.StreamOptions.IncludeUsage
}

// completionRequest is POST /v1/completions' body (the legacy, non-chat
// endpoint). Not exercised by the current benchmark (which only speaks
// chat/completions), but cheap to support for router-generality and manual
// testing. Prompt accepts either an OpenAI string or a []string (first
// element used, matching typical single-prompt usage).
type completionRequest struct {
	Model         string          `json:"model"`
	Prompt        json.RawMessage `json:"prompt"`
	Stream        bool            `json:"stream"`
	StreamOptions *streamOptions  `json:"stream_options"`
	MaxTokens     int             `json:"max_tokens"`
}

func (r *completionRequest) promptText() string {
	var s string
	if err := json.Unmarshal(r.Prompt, &s); err == nil {
		return s
	}
	var arr []string
	if err := json.Unmarshal(r.Prompt, &arr); err == nil && len(arr) > 0 {
		return arr[0]
	}
	return ""
}

func (r *completionRequest) wantsUsage() bool {
	return r.StreamOptions != nil && r.StreamOptions.IncludeUsage
}

// ---- Response types ----

type usageDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// usage matches what consumeOpenAIPlain / consumeOpenAISSE parse:
// prompt_tokens, completion_tokens, total_tokens, and
// prompt_tokens_details.cached_tokens.
type usage struct {
	PromptTokens        int           `json:"prompt_tokens"`
	CompletionTokens    int           `json:"completion_tokens"`
	TotalTokens         int           `json:"total_tokens"`
	PromptTokensDetails *usageDetails `json:"prompt_tokens_details,omitempty"`
}

func buildUsage(promptTokens, cachedTokens, completionTokens int) usage {
	return usage{
		PromptTokens:        promptTokens,
		CompletionTokens:    completionTokens,
		TotalTokens:         promptTokens + completionTokens,
		PromptTokensDetails: &usageDetails{CachedTokens: cachedTokens},
	}
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      *chatMsgOut `json:"message,omitempty"`
	Delta        *chatMsgOut `json:"delta,omitempty"`
	FinishReason *string     `json:"finish_reason"`
}

type chatMsgOut struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type chatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *usage       `json:"usage,omitempty"`
}

type completionChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

type completionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   *usage             `json:"usage,omitempty"`
}

type modelsResponse struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

// syntheticTokens produces n whitespace-separated placeholder tokens
// ("tok0 tok1 ..."), for a completion body whose exact content the router
// benchmark never inspects (only usage counters and cache behavior matter to
// it) — same convention router/internal/testutil/mockvllm's SSE script uses
// ("t%d").
func syntheticTokens(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = "tok" + strconv.Itoa(i)
	}
	return out
}
