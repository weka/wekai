package pd

import (
	"encoding/json"
	"testing"

	"github.com/weka/wekai/router/internal/registry"
)

func field(t *testing.T, body []byte, k string) string {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatal(err)
	}
	return string(m[k])
}

func TestPrefillBodyAsksForRemoteDecodeAndOneToken(t *testing.T) {
	in := []byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true},` +
		`"max_tokens":512,"max_completion_tokens":512,"messages":[]}`)
	out := PrefillBody(in)
	for k, want := range map[string]string{"max_tokens": "1", "max_completion_tokens": "1", "stream": "false", "model": `"m"`} {
		if got := field(t, out, k); got != want {
			t.Errorf("%s = %s, want %s", k, got, want)
		}
	}
	if got := field(t, out, "stream_options"); got != "" {
		t.Errorf("stream_options kept: %s", got)
	}
	var kv map[string]any
	_ = json.Unmarshal([]byte(field(t, out, "kv_transfer_params")), &kv)
	if kv["do_remote_decode"] != true || kv["do_remote_prefill"] != false || kv["remote_engine_id"] != nil {
		t.Errorf("kv_transfer_params = %v", kv)
	}
	if field(t, in, "max_tokens") != "512" {
		t.Error("input body was modified")
	}
}

func TestPrefillBodyOmitsMaxCompletionTokensUnlessSent(t *testing.T) {
	if got := field(t, PrefillBody([]byte(`{"model":"m"}`)), "max_completion_tokens"); got != "" {
		t.Errorf("max_completion_tokens = %s", got)
	}
}

func TestDecodeBodyKeepsTheRequestAndCarriesTheParams(t *testing.T) {
	in := []byte(`{"model":"m","stream":true,"max_tokens":8,"kv_transfer_params":{"old":1}}`)
	out := DecodeBody(in, Params(`{"remote_engine_id":"e1"}`))
	if field(t, out, "stream") != "true" || field(t, out, "max_tokens") != "8" {
		t.Errorf("request fields changed: %s", out)
	}
	if got := field(t, out, "kv_transfer_params"); got != `{"remote_engine_id":"e1"}` {
		t.Errorf("kv_transfer_params = %s", got)
	}
}

func TestParseParams(t *testing.T) {
	if p, ok := ParseParams([]byte(`{"id":"x","kv_transfer_params":{"a":[1,2]}}`)); !ok || string(p) != `{"a":[1,2]}` {
		t.Errorf("got %s %v", p, ok)
	}
	for _, bad := range []string{`{}`, `{"kv_transfer_params":null}`, `{"kv_transfer_params":"x"}`, `nope`} {
		if _, ok := ParseParams([]byte(bad)); ok {
			t.Errorf("%s parsed", bad)
		}
	}
}

func TestSetPartitionsByURLIgnoringTrailingSlash(t *testing.T) {
	s := New("t", []string{"http://p1:8000/"})
	p1 := &registry.Backend{URL: "http://p1:8000"}
	d1 := &registry.Backend{URL: "http://d1:8000"}
	all := []*registry.Backend{p1, d1}
	if got := s.PrefillNodes(all); len(got) != 1 || got[0] != p1 {
		t.Errorf("P = %v", got)
	}
	if got := s.DecodeNodes(all); len(got) != 1 || got[0] != d1 {
		t.Errorf("D = %v", got)
	}
}
