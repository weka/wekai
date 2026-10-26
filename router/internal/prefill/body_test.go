package prefill

import (
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, b)
	}
	return m
}

func TestJobBody(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]any
	}{
		{
			name: "openai chat streaming",
			in: `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,` +
				`"stream_options":{"include_usage":true},"max_tokens":4096,"temperature":0.2}`,
			want: map[string]any{
				"model": "m", "messages": []any{map[string]any{"role": "user", "content": "hi"}},
				"stream": false, "max_tokens": 1.0, "temperature": 0.2,
				"vllm_xargs": map[string]any{"weka_prefill": 1.0},
			},
		},
		{
			name: "openai chat with max_completion_tokens",
			in:   `{"model":"m","messages":[],"max_completion_tokens":900,"min_tokens":8}`,
			want: map[string]any{
				"model": "m", "messages": []any{},
				"max_completion_tokens": 1.0, "max_tokens": 1.0,
				"vllm_xargs": map[string]any{"weka_prefill": 1.0},
			},
		},
		{
			name: "completions without max_tokens",
			in:   `{"model":"m","prompt":"abc"}`,
			want: map[string]any{
				"model": "m", "prompt": "abc", "max_tokens": 1.0,
				"vllm_xargs": map[string]any{"weka_prefill": 1.0},
			},
		},
		{
			name: "anthropic messages with thinking",
			in: `{"model":"m","max_tokens":8000,"stream":true,` +
				`"thinking":{"type":"enabled","budget_tokens":4000},` +
				`"messages":[{"role":"user","content":"hi"}]}`,
			want: map[string]any{
				"model": "m", "max_tokens": 1.0, "stream": false,
				"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
				"vllm_xargs": map[string]any{"weka_prefill": 1.0},
			},
		},
		{
			name: "existing vllm_xargs are kept",
			in:   `{"model":"m","prompt":"x","vllm_xargs":{"other":"keep","weka_prefill":0}}`,
			want: map[string]any{
				"model": "m", "prompt": "x", "max_tokens": 1.0,
				"vllm_xargs": map[string]any{"other": "keep", "weka_prefill": 1.0},
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := decode(t, JobBody([]byte(c.in))); !reflect.DeepEqual(got, c.want) {
				t.Errorf("JobBody =\n%v\nwant\n%v", got, c.want)
			}
		})
	}
}

func TestJobBodyLeavesUnparseableBodyAlone(t *testing.T) {
	for _, in := range []string{``, `not json`, `[1,2]`, `null`} {
		if got := string(JobBody([]byte(in))); got != in {
			t.Errorf("JobBody(%q) = %q, want it unchanged", in, got)
		}
		if got := string(WithBypass([]byte(in))); got != in {
			t.Errorf("WithBypass(%q) = %q, want it unchanged", in, got)
		}
	}
}

func TestWithBypassKeepsTheRequestOtherwiseIntact(t *testing.T) {
	in := `{"model":"m","stream":true,"max_tokens":512,"vllm_xargs":{"a":1}}`
	want := map[string]any{
		"model": "m", "stream": true, "max_tokens": 512.0,
		"vllm_xargs": map[string]any{"a": 1.0, "weka_prefill": 1.0},
	}
	if got := decode(t, WithBypass([]byte(in))); !reflect.DeepEqual(got, want) {
		t.Errorf("WithBypass = %v, want %v", got, want)
	}
}

func TestHasBypass(t *testing.T) {
	cases := map[string]bool{
		`{"prompt":"x"}`:                                     false,
		`{"vllm_xargs":{"weka_prefill":1}}`:                  true,
		`{"vllm_xargs":{"weka_prefill":true}}`:               true,
		`{"vllm_xargs":{"weka_prefill":0}}`:                  false,
		`{"vllm_xargs":{"other":1}}`:                         false,
		`{"prompt":"mentions \"weka_prefill\" in its text"}`: false,
		`{"vllm_xargs":{"weka_prefill":"1"}}`:                true,
		`broken "weka_prefill"`:                              false,
	}
	for in, want := range cases {
		if got := HasBypass([]byte(in)); got != want {
			t.Errorf("HasBypass(%s) = %v, want %v", in, got, want)
		}
	}
	if !HasBypass(WithBypass([]byte(`{"prompt":"x"}`))) {
		t.Error("a body rewritten by WithBypass does not read back as bypassed")
	}
	if !HasBypass(JobBody([]byte(`{"prompt":"x"}`))) {
		t.Error("a prefill job body does not read back as bypassed")
	}
}

func TestParseRejection(t *testing.T) {
	cases := []struct {
		name   string
		body   string
		tokens int64
		ok     bool
	}{
		{"openai envelope", `{"error":{"type":"large_prefill_rejected","message":"prefill of 90000 tokens exceeds WEKA_PREFILL_REJECT_TOKENS=8192","param":"90000","code":400}}`, 90000, true},
		{"numeric param", `{"error":{"type":"large_prefill_rejected","param":123456}}`, 123456, true},
		{"anthropic envelope", `{"type":"error","error":{"type":"large_prefill_rejected","message":"prefill of 70000 tokens exceeds WEKA_PREFILL_REJECT_TOKENS=8192"}}`, 70000, true},
		{"names no count", `{"error":{"type":"large_prefill_rejected","message":"too big"}}`, 0, true},
		{"other 400", `{"error":{"type":"invalid_request_error","message":"bad"}}`, 0, false},
		{"type only in the message", `{"error":{"type":"invalid_request_error","message":"large_prefill_rejected"}}`, 0, false},
		{"not json", `large_prefill_rejected`, 0, false},
	}
	for _, c := range cases {
		tokens, ok := ParseRejection([]byte(c.body))
		if tokens != c.tokens || ok != c.ok {
			t.Errorf("%s: ParseRejection = (%d, %v), want (%d, %v)", c.name, tokens, ok, c.tokens, c.ok)
		}
	}
}

func TestRejectionFromResponse(t *testing.T) {
	mk := func(h, body string) *http.Response {
		r := &http.Response{StatusCode: 503, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}
		if h != "" {
			r.Header.Set(RejectHeader, h)
		}
		return r
	}
	if n, ok := RejectionFromResponse(mk("70000", "")); !ok || n != 70000 {
		t.Errorf("header form = (%d, %v)", n, ok)
	}
	if n, ok := RejectionFromResponse(mk("", `{"error":{"type":"large_prefill_rejected","param":"5"}}`)); !ok || n != 5 {
		t.Errorf("body fallback = (%d, %v)", n, ok)
	}
	other := mk("", `{"error":{"type":"overloaded"}}`)
	if _, ok := RejectionFromResponse(other); ok {
		t.Error("an ordinary 503 was taken for a rejection")
	}
	if b, _ := io.ReadAll(other.Body); string(b) != `{"error":{"type":"overloaded"}}` {
		t.Errorf("body not restored: %q", b)
	}
}
