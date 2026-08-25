// Package openai implements the router's one dialect (API-3). It covers the
// OpenAI-compatible surface and Anthropic's Messages surface, because the
// backends this router fronts serve both and a request's routable content is the
// same either way: a system prompt, a tool set, and a conversation.
//
// One dialect rather than two is a deliberate trade. The alternative — a second
// Dialect implementation — buys a cleaner name and costs the thing that matters:
// candidate filtering is per dialect ID, so two dialects partition one fleet into
// two, and a backend registered under one could not serve a request that arrived
// on the other's path. The ID stays "openai" for the same reason it is not worth
// renaming: it labels metrics and tags every registered backend.
//
// What genuinely differs between the surfaces is handled where it differs, and
// nowhere else:
//
//   - Anthropic carries the system prompt in a top-level `system` field rather
//     than as a message, and its first block is a near-unique per-request
//     preamble that must be skipped — see billingBlockBytes. That skip is an
//     Anthropic artifact and is applied to that field ONLY, never to an OpenAI
//     body's leading system message (API-11).
//   - The streams end differently; NewStreamScanner accepts both terminals.
//   - The usage envelopes are shaped differently; ExtractUsage reads both.
package openai

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/jsonscan"
)

const ID = "openai"

type Dialect struct{}

func New() Dialect { return Dialect{} }

func (Dialect) ID() string { return ID }

// Routes is the wire surface. It matches v1 so existing clients and vLLM workers
// need no change (COMPAT-1), plus Anthropic's Messages surface.
//
// /v1/messages is claimed rather than left to the passthrough tier because that
// tier extracts no units — it is for paths whose shape the router does not know,
// and what it can offer them is load balancing. A dialect that knows this shape
// must say so here: claiming the path is the whole of what makes Anthropic-format
// traffic cache routed, and a fleet fronted for Claude-shaped clients is
// otherwise getting none of the prefix affinity the router exists to provide.
func (Dialect) Routes() []dialect.Route {
	return []dialect.Route{
		{Pattern: "POST /v1/chat/completions", Class: "chat", Stream: true},
		{Pattern: "POST /v1/completions", Class: "completions", Stream: true},
		{Pattern: "POST /v1/embeddings", Class: "embeddings"},
		{Pattern: "POST /v1/responses", Class: "responses", Stream: true},
		{Pattern: "POST /v1/rerank", Class: "rerank"},
		{Pattern: "POST /rerank", Class: "rerank"},
		{Pattern: "POST /generate", Class: "generate", Stream: true},
		{Pattern: "POST /inference/v1/generate", Class: "generate", Stream: true},
		{Pattern: "POST /v1/messages", Class: ClassMessages, Stream: true},
		{Pattern: "POST /v1/messages/count_tokens", Class: ClassCountTokens},
		{Pattern: "GET /v1/models", Class: "models"},
		{Pattern: "GET /get_model_info", Class: "models"},
	}
}

// Route classes referred to by name elsewhere in this package. The rest are
// metrics labels only.
const (
	// ClassMessages is Anthropic's Messages surface.
	ClassMessages = "messages"
	// ClassCountTokens is its token counter, which shares the request shape
	// exactly and is told apart from a generation only by the path.
	ClassCountTokens = "count_tokens"
)

// Introspect reads `model` and `stream` in one partial scan, without building a
// typed model of the body (GW-6, API-15).
func (Dialect) Introspect(body []byte) dialect.Introspection {
	var in dialect.Introspection
	if len(body) == 0 {
		return in
	}
	_ = jsonscan.Fields(body, func(k, v []byte) bool {
		switch string(k) {
		case "model":
			if s, ok := jsonscan.String(v); ok {
				in.Model = string(s)
			}
		case "stream":
			if b, ok := jsonscan.Bool(v); ok {
				in.Stream = b
			}
		}
		return true
	})
	return in
}

// billingBlockBytes is the size below which Anthropic's FIRST system block is
// treated as a per-request billing header rather than as content.
//
// Anthropic clients put a short, near-unique preamble in system block 0. Hashed
// like anything else it gives every request a different first block, so two turns
// of one conversation share no prefix at all and the affinity tree has nothing to
// walk. Skipping it is what makes the rest of the system prompt the shared prefix
// it actually is.
//
// The rule and the threshold are wekai's own, not a second opinion: the replay
// corpus is built by BuildReplayRequestPrefix under the identical `i == 0 &&
// Bytes < 200` skip. A router that disagreed would route against a definition of
// "prefix" that the measurement does not share, and every cache number taken from
// a replay would be describing a different router (API-11, CU-3).
const billingBlockBytes = 200

// ExtractUnits builds the routable prefix for cache-affinity policies.
//
// Ordering follows wekai's BuildReplayRequestPrefix — system, then tools, then
// messages in order — because that is the order vLLM's own sequential prefix
// hashing sees, so a shared system prompt forms a shared trie prefix.
func (Dialect) ExtractUnits(body []byte, class string, dst []kvcache.Unit) ([]kvcache.Unit, bool) {
	if len(body) == 0 {
		return dst, false
	}
	// A token count runs no forward pass and leaves no KV behind, though its body
	// is byte-for-byte the shape of the generation it is counting. Returning units
	// would commit the chosen backend as a holder of a prefix it has never seen,
	// and the next real request for that prefix would be sent there to collect a
	// hit that cannot exist. The path is the only thing that tells the two apart,
	// which is why the class is consulted here at all.
	if class == ClassCountTokens {
		return dst, false
	}
	// Segments accumulate in cache order; kvcache.ChunkContent does the windowing
	// and hashing, shared with the benchmark tooling so there is one implementation.
	var units []kvcache.Unit
	add := func(tag string, content []byte) {
		units = append(units, kvcache.ChunkContent(tag, content, kvcache.DefaultChunkBytes)...)
	}

	var messages, tools, systemSpan, promptSpan, inputSpan []byte
	_ = jsonscan.Fields(body, func(k, v []byte) bool {
		switch string(k) {
		case "messages":
			messages = v
		case "tools", "functions":
			tools = v
		case "system":
			// Anthropic only: OpenAI has no top-level system field, so its
			// presence is what identifies the shape without sniffing the path.
			systemSpan = v
		case "prompt":
			promptSpan = v
		case "input":
			inputSpan = v
		}
		return true
	})

	switch {
	case len(messages) > 0:
		// Anthropic's system prompt comes first, because that is where the cache
		// order puts it: the backend sees system, then tools, then the turns.
		for i, blk := range systemBlocks(systemSpan) {
			if i == 0 && len(blk) < billingBlockBytes {
				continue
			}
			add("system", blk)
		}
		// /v1/chat/completions. Leading system messages first so they anchor the
		// shared prefix even if a client reorders them.
		var sys, rest [][2][]byte
		_ = jsonscan.Array(messages, func(elem []byte) bool {
			var role, content []byte
			if err := jsonscan.Fields(elem, func(mk, mv []byte) bool {
				switch string(mk) {
				case "role":
					role, _ = jsonscan.String(mv)
				case "content":
					// A string content decodes; a structured content array is
					// hashed as its raw span, which is stable and sufficient.
					if s, ok := jsonscan.String(mv); ok {
						content = s
					} else {
						content = mv
					}
				}
				return true
			}); err != nil {
				// A non-object element (e.g. a bare string in `messages`). Hash its
				// raw span rather than collapsing it to the empty-content constant.
				add("elem", elem)
				return true
			}
			if content == nil {
				// No `content` key at all — an assistant turn carrying only
				// tool_calls, which is every turn of an agentic loop. Hashing
				// "role with empty content" yields a constant per role, so
				// unrelated conversations would appear to share a long prefix and
				// the router would confidently send a request to a backend holding
				// none of it. Hash the whole element instead: it is stable and it
				// actually distinguishes the turns.
				content = elem
			}
			pair := [2][]byte{role, content}
			if string(role) == "system" || string(role) == "developer" {
				sys = append(sys, pair)
			} else {
				rest = append(rest, pair)
			}
			return true
		})
		for _, m := range sys {
			add(string(m[0]), m[1])
		}
		if len(tools) > 0 {
			add("tools", tools)
		}
		for _, m := range rest {
			add(string(m[0]), m[1])
		}
	case len(promptSpan) > 0:
		// /v1/completions
		if s, ok := jsonscan.String(promptSpan); ok {
			add("prompt", s)
		} else {
			add("prompt", promptSpan)
		}
	case len(inputSpan) > 0:
		// /v1/embeddings — no KV cache to speak of, but the shape is supported.
		if s, ok := jsonscan.String(inputSpan); ok {
			add("input", s)
		} else {
			add("input", inputSpan)
		}
	default:
		return dst, false
	}

	if len(units) == 0 {
		return dst, false
	}
	return append(dst, units...), true
}

// systemBlocks splits Anthropic's `system` field into its blocks, in order.
//
// Both shapes appear in real traffic: a bare string in the simple form, and an
// array of content blocks once a client starts marking cache breakpoints. The
// array elements are hashed as their raw spans, which is stable and keeps the
// `cache_control` marker part of the block's identity — two requests that
// disagree about where the cache breaks do not share that block.
func systemBlocks(span []byte) [][]byte {
	if len(span) == 0 {
		return nil
	}
	if s, ok := jsonscan.String(span); ok {
		return [][]byte{s}
	}
	var out [][]byte
	if err := jsonscan.Array(span, func(elem []byte) bool {
		out = append(out, elem)
		return true
	}); err != nil {
		return nil
	}
	return out
}

// NewStreamScanner detects the terminal marker of either surface this dialect
// serves: OpenAI closes a stream with `data: [DONE]`, Anthropic with
// `event: message_stop` (API-8).
//
// Both are accepted rather than chosen by route class because the scanner is
// built once per attempt, before the response exists. The marker decides only
// whether a stream that failed had already finished, so an unrecognised terminal
// costs nothing on the happy path and counts every late failure on that surface
// as an upstream abort.
func (Dialect) NewStreamScanner() dialect.StreamScanner {
	return &dialect.LineScanner{Markers: [][]byte{
		[]byte("data: [DONE]"),
		[]byte("event: message_stop"),
	}}
}

type errEnvelope struct {
	Error errBody `json:"error"`
}

type errBody struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}

// WriteError renders the OpenAI error envelope. v1 replied to auth failures with
// bare text naming a hard-coded third-party URL over plaintext HTTP, which no
// OpenAI SDK can parse — clients saw a transport error instead of the cause.
func (Dialect) WriteError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Del("Content-Length")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errEnvelope{Error: errBody{
		Message: msg,
		Type:    typeForStatus(status),
		Code:    code,
	}})
}

func typeForStatus(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "invalid_request_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status >= 500:
		return "api_error"
	default:
		return "invalid_request_error"
	}
}

// ExtractUsage reads token accounting from a response body. It is the closed
// loop on prefix-cache prediction: CachedTokens over PromptTokens is what
// router_cache_observed_fraction records.
//
// Both surfaces are read, because a router serving Anthropic-format traffic that
// understood only OpenAI's envelope would emit no observation at all — the
// prediction would go unchecked on exactly the traffic it was making.
//
// OpenAI: prompt_tokens is the whole prompt and
// prompt_tokens_details.cached_tokens is the part that hit. The detail requires
// the worker to run with --enable-prompt-tokens-details; absent that flag the
// field is simply missing, which is reported as a zero fraction rather than a
// misleading one.
//
// Anthropic: input_tokens EXCLUDES everything the cache accounted for, so the
// prompt is input_tokens + cache_creation_input_tokens + cache_read_input_tokens
// and only the read half was a hit. Reading input_tokens as the denominator is
// the mistake this shape invites, and it yields fractions above 1.0 on precisely
// the well-cached requests the metric exists to show.
func (Dialect) ExtractUsage(body []byte) (dialect.Usage, bool) {
	var u dialect.Usage
	var inputTokens, cacheCreate, cacheRead int
	var anthropic bool
	found := false
	_ = jsonscan.Fields(body, func(k, v []byte) bool {
		if string(k) != "usage" {
			return true
		}
		found = true
		_ = jsonscan.Fields(v, func(uk, uv []byte) bool {
			switch string(uk) {
			case "prompt_tokens":
				u.PromptTokens, _ = jsonscan.Int(uv)
			case "total_tokens":
				u.TotalTokens, _ = jsonscan.Int(uv)
			case "prompt_tokens_details":
				_ = jsonscan.Fields(uv, func(dk, dv []byte) bool {
					if string(dk) == "cached_tokens" {
						u.CachedTokens, _ = jsonscan.Int(dv)
					}
					return true
				})
			case "input_tokens":
				inputTokens, _ = jsonscan.Int(uv)
				anthropic = true
			case "cache_creation_input_tokens":
				cacheCreate, _ = jsonscan.Int(uv)
				anthropic = true
			case "cache_read_input_tokens":
				cacheRead, _ = jsonscan.Int(uv)
				anthropic = true
			}
			return true
		})
		return false // usage found; stop scanning the top level
	})
	// OpenAI's fields win when both are somehow present: a body carrying
	// prompt_tokens has already answered the question directly.
	if anthropic && u.PromptTokens == 0 {
		u.PromptTokens = inputTokens + cacheCreate + cacheRead
		u.CachedTokens = cacheRead
	}
	return u, found
}

// Credential accepts both forms the ecosystem uses (AUTH-3).
func (Dialect) Credential(h http.Header) (string, bool) {
	if v := h.Get("Authorization"); v != "" {
		const p = "bearer "
		if len(v) > len(p) && strings.EqualFold(v[:len(p)], p) {
			if tok := strings.TrimSpace(v[len(p):]); tok != "" {
				return tok, true
			}
		}
		return "", false
	}
	if tok := strings.TrimSpace(h.Get("X-Api-Key")); tok != "" {
		return tok, true
	}
	return "", false
}
