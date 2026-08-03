// Package openai implements the OpenAI-compatible dialect — the only one
// shipped in v2.0 (API-3).
//
// Adding Anthropic must require no change to any core package. When it lands,
// note that wekai's prefix builder is already Anthropic-native: it orders units
// system blocks -> tools -> messages, and its `i == 0 && Bytes < 200` skip exists
// because Anthropic's per-request billing header block carries a near-unique hash
// that poisons prefix-block hashing. That skip is an Anthropic artifact and MUST
// NOT be applied here (API-11).
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
// need no change (COMPAT-1).
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
		{Pattern: "GET /v1/models", Class: "models"},
		{Pattern: "GET /get_model_info", Class: "models"},
	}
}

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

// ExtractUnits builds the routable prefix for cache-affinity policies.
//
// Ordering follows wekai's BuildReplayRequestPrefix — system, then tools, then
// messages in order — because that is the order vLLM's own sequential prefix
// hashing sees, so a shared system prompt forms a shared trie prefix.
//
// wekai's `i == 0 && Bytes < 200` skip is deliberately NOT applied. That rule
// exists because Anthropic emits a small, near-unique per-request billing block
// as system block 0, which would poison every downstream prefix hash. OpenAI has
// no such block, and skipping a genuine leading system message here would discard
// exactly the shared content this policy exists to exploit (API-11, CU-3).
func (Dialect) ExtractUnits(body []byte, class string, dst []kvcache.Unit) ([]kvcache.Unit, bool) {
	if len(body) == 0 {
		return dst, false
	}
	// Segments accumulate in cache order; kvcache.ChunkContent does the windowing
	// and hashing, shared with the benchmark tooling so there is one implementation.
	var units []kvcache.Unit
	add := func(tag string, content []byte) {
		units = append(units, kvcache.ChunkContent(tag, content, kvcache.DefaultChunkBytes)...)
	}

	var messages, tools, promptSpan, inputSpan []byte
	_ = jsonscan.Fields(body, func(k, v []byte) bool {
		switch string(k) {
		case "messages":
			messages = v
		case "tools", "functions":
			tools = v
		case "prompt":
			promptSpan = v
		case "input":
			inputSpan = v
		}
		return true
	})

	switch {
	case len(messages) > 0:
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

// NewStreamScanner detects OpenAI's terminal marker. Anthropic's is
// `event: message_stop`, which is exactly why this is dialect-provided (API-8).
func (Dialect) NewStreamScanner() dialect.StreamScanner {
	return &dialect.LineScanner{Marker: []byte("data: [DONE]")}
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

// ExtractUsage reads token accounting from a response body.
//
// cached_tokens lives at usage.prompt_tokens_details.cached_tokens and requires
// the worker to run with --enable-prompt-tokens-details; absent that flag the
// field is simply missing, which is reported as ok=false rather than zero.
func (Dialect) ExtractUsage(body []byte) (dialect.Usage, bool) {
	var u dialect.Usage
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
			}
			return true
		})
		return false // usage found; stop scanning the top level
	})
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
