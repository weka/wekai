// Package pd is vanilla vLLM prefill/decode disaggregation over NIXL.
//
// A static subset of a pool's backends are prefill (P) nodes and the rest are
// decode (D) nodes. Every request goes to a P node first, as a copy limited to
// one token that asks the engine to leave its KV for a remote decoder; the
// engine answers with kv_transfer_params naming where it is. The request itself
// then goes to a D node carrying those params, which pulls the blocks it does
// not already hold and streams the answer.
package pd

import (
	"encoding/json"
	"strings"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

// Set is the P/D split of one pool.
type Set struct {
	prefill map[string]struct{}
	M       *metrics.PDMetrics
}

// New builds a set from the prefill backends' URLs.
func New(pool string, prefillURLs []string) *Set {
	s := &Set{prefill: map[string]struct{}{}, M: metrics.ForPD(pool)}
	for _, u := range prefillURLs {
		s.prefill[Normalize(u)] = struct{}{}
	}
	return s
}

// Normalize makes backend URLs comparable.
func Normalize(u string) string { return strings.TrimRight(strings.TrimSpace(u), "/") }

// IsPrefill reports whether b is a P node.
func (s *Set) IsPrefill(b *registry.Backend) bool {
	_, ok := s.prefill[Normalize(b.URL)]
	return ok
}

// Role is b's role, for display: "prefill" or "decode".
func (s *Set) Role(b *registry.Backend) string {
	if s.IsPrefill(b) {
		return "prefill"
	}
	return "decode"
}

// PrefillNodes keeps the P nodes of cands.
func (s *Set) PrefillNodes(cands []*registry.Backend) []*registry.Backend {
	return s.filter(cands, true)
}

// DecodeNodes keeps the D nodes of cands.
func (s *Set) DecodeNodes(cands []*registry.Backend) []*registry.Backend {
	return s.filter(cands, false)
}

func (s *Set) filter(cands []*registry.Backend, want bool) []*registry.Backend {
	out := make([]*registry.Backend, 0, len(cands))
	for _, b := range cands {
		if s.IsPrefill(b) == want {
			out = append(out, b)
		}
	}
	return out
}

// Params is the kv_transfer_params object of a request or response, verbatim.
type Params = json.RawMessage

var prefillParams = json.RawMessage(`{"do_remote_decode":true,"do_remote_prefill":false,` +
	`"remote_engine_id":null,"remote_block_ids":null,"remote_host":null,"remote_port":null}`)

// PrefillBody turns a request into its prefill leg: one token, not streamed,
// asking to leave its KV for a remote decoder. A body that is not a JSON object
// is returned unchanged.
func PrefillBody(body []byte) []byte {
	doc, ok := parse(body)
	if !ok {
		return body
	}
	doc["max_tokens"] = json.RawMessage("1")
	if _, has := doc["max_completion_tokens"]; has {
		doc["max_completion_tokens"] = json.RawMessage("1")
	}
	doc["stream"] = json.RawMessage("false")
	delete(doc, "stream_options")
	doc["kv_transfer_params"] = prefillParams
	return encode(doc, body)
}

// DecodeBody sets the request's kv_transfer_params to the prefill leg's answer.
func DecodeBody(body []byte, params Params) []byte {
	doc, ok := parse(body)
	if !ok {
		return body
	}
	doc["kv_transfer_params"] = params
	return encode(doc, body)
}

// ParseParams extracts the top-level kv_transfer_params object from a prefill
// leg's response. ok is false when it is absent, null or not an object.
func ParseParams(resp []byte) (Params, bool) {
	var doc struct {
		P json.RawMessage `json:"kv_transfer_params"`
	}
	if json.Unmarshal(resp, &doc) != nil {
		return nil, false
	}
	p := strings.TrimSpace(string(doc.P))
	if !strings.HasPrefix(p, "{") {
		return nil, false
	}
	return doc.P, true
}

func parse(body []byte) (map[string]json.RawMessage, bool) {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil || m == nil {
		return nil, false
	}
	return m, true
}

func encode(doc map[string]json.RawMessage, orig []byte) []byte {
	out, err := json.Marshal(doc)
	if err != nil {
		return orig
	}
	return out
}
