package prefill

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strconv"
)

// RejectHeader carries the recompute token count on the engine's refusal, an
// HTTP 503. The status is not the signal — a 503 is also what an overloaded or
// restarting backend says — so a refusal is recognised by this header, with the
// body's error type as the fallback.
const RejectHeader = "X-Weka-Prefill-Reject"

// maxRejectionBody bounds how much of a 503 is read to recognise a refusal by
// its body; the engine's is a few hundred bytes.
const maxRejectionBody = 64 << 10

// RejectionFromResponse recognises the engine's large-prefill refusal in a 503
// and returns the recompute it names. When it is not a refusal the body is left
// readable from the start.
func RejectionFromResponse(resp *http.Response) (tokens int64, ok bool) {
	if h := resp.Header.Get(RejectHeader); h != "" {
		n, _ := strconv.ParseInt(h, 10, 64)
		return max(n, 0), true
	}
	data, _ := io.ReadAll(io.LimitReader(resp.Body, maxRejectionBody))
	if n, ok := ParseRejection(data); ok {
		return n, true
	}
	resp.Body = &replayBody{Reader: io.MultiReader(bytes.NewReader(data), resp.Body), Closer: resp.Body}
	return 0, false
}

// replayBody is a response body with an already-read prefix put back.
type replayBody struct {
	io.Reader
	io.Closer
}

// BypassKey is the vllm_xargs entry that exempts a request from the engine's
// large-prefill rejection. The router sets it on a prefill job and on the
// decode resubmission that follows, so a rejection can never loop.
const BypassKey = "weka_prefill"

// RejectionType is the error type of the engine's refusal.
const RejectionType = "large_prefill_rejected"

var bypassMarker = []byte(`"` + BypassKey + `"`)

// HasBypass reports whether a request body already carries a truthy bypass.
// The substring test keeps the common case — no bypass anywhere — from parsing
// a body that can be megabytes.
func HasBypass(body []byte) bool {
	if !bytes.Contains(body, bypassMarker) {
		return false
	}
	var doc struct {
		XArgs map[string]any `json:"vllm_xargs"`
	}
	if json.Unmarshal(body, &doc) != nil {
		return false
	}
	switch v := doc.XArgs[BypassKey].(type) {
	case bool:
		return v
	case float64:
		return v != 0
	case string:
		return v != "" && v != "0" && v != "false"
	}
	return false
}

// WithBypass returns body with vllm_xargs.weka_prefill=1 merged into any
// existing vllm_xargs. A body that is not a JSON object is returned unchanged.
func WithBypass(body []byte) []byte {
	doc, ok := parseObject(body)
	if !ok {
		return body
	}
	return encode(doc, body)
}

// JobBody turns a request into its prefill-only form: it generates one token,
// does not stream, and carries the bypass. Fields that only make sense for a
// longer or streamed completion are dropped, because vLLM validates them
// against the rewritten ones and would answer 400 — stream_options without
// stream, min_tokens above max_tokens, a thinking budget above max_tokens.
//
// max_completion_tokens is set only where the client sent it; OpenAI chat and
// completions and Anthropic messages all take max_tokens.
func JobBody(body []byte) []byte {
	doc, ok := parseObject(body)
	if !ok {
		return body
	}
	doc.fields["max_tokens"] = json.RawMessage("1")
	if _, has := doc.fields["max_completion_tokens"]; has {
		doc.fields["max_completion_tokens"] = json.RawMessage("1")
	}
	if _, has := doc.fields["stream"]; has {
		doc.fields["stream"] = json.RawMessage("false")
	}
	for _, k := range []string{"stream_options", "min_tokens", "thinking"} {
		delete(doc.fields, k)
	}
	return encode(doc, body)
}

type object struct {
	fields map[string]json.RawMessage
	xargs  map[string]json.RawMessage
}

func parseObject(body []byte) (object, bool) {
	var o object
	if json.Unmarshal(body, &o.fields) != nil || o.fields == nil {
		return o, false
	}
	o.xargs = map[string]json.RawMessage{}
	if raw, ok := o.fields["vllm_xargs"]; ok {
		// A vllm_xargs that is not an object is the client's error to have
		// made; replace it rather than forward a body the engine would refuse
		// for a reason the router introduced.
		_ = json.Unmarshal(raw, &o.xargs)
		if o.xargs == nil {
			o.xargs = map[string]json.RawMessage{}
		}
	}
	return o, true
}

func encode(o object, orig []byte) []byte {
	o.xargs[BypassKey] = json.RawMessage("1")
	xa, err := json.Marshal(o.xargs)
	if err != nil {
		return orig
	}
	o.fields["vllm_xargs"] = xa
	out, err := json.Marshal(o.fields)
	if err != nil {
		return orig
	}
	return out
}

// rejectionPrompt reads the token count from the message when param is absent,
// as in an Anthropic-style envelope that does not carry it.
var rejectionPrompt = regexp.MustCompile(`prefill of (\d+) tokens`)

// ParseRejection recognises the engine's large-prefill refusal and returns the
// recompute it names. The count comes from error.param, falling back to the
// message; a refusal that names neither still reports ok with 0 tokens, so the
// caller can place it by the prompt size it estimates itself.
func ParseRejection(body []byte) (tokens int64, ok bool) {
	if !bytes.Contains(body, []byte(RejectionType)) {
		return 0, false
	}
	var doc struct {
		Error struct {
			Type    string          `json:"type"`
			Message string          `json:"message"`
			Param   json.RawMessage `json:"param"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &doc) != nil || doc.Error.Type != RejectionType {
		return 0, false
	}
	if p := bytes.Trim(doc.Error.Param, `"`); len(p) > 0 {
		if n, err := strconv.ParseInt(string(p), 10, 64); err == nil && n > 0 {
			return n, true
		}
	}
	if m := rejectionPrompt.FindStringSubmatch(doc.Error.Message); m != nil {
		if n, err := strconv.ParseInt(m[1], 10, 64); err == nil {
			return n, true
		}
	}
	return 0, true
}
