package jsonscan_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/weka/wekai/router/internal/jsonscan"
)

func fieldMap(t *testing.T, body string) map[string]string {
	t.Helper()
	out := map[string]string{}
	if err := jsonscan.Fields([]byte(body), func(k, v []byte) bool {
		out[string(k)] = string(v)
		return true
	}); err != nil {
		t.Fatalf("Fields(%s): %v", body, err)
	}
	return out
}

func TestFieldsExtractsTopLevelSpans(t *testing.T) {
	got := fieldMap(t, `{"model":"m-1","stream":true,"n":3,"messages":[{"role":"user"}],"opts":{"a":{"b":1}}}`)
	want := map[string]string{
		"model":    `"m-1"`,
		"stream":   `true`,
		"n":        `3`,
		"messages": `[{"role":"user"}]`,
		"opts":     `{"a":{"b":1}}`,
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("field %q = %q, want %q", k, got[k], w)
		}
	}
}

// Structural skipping must be string-and-escape aware. A brace inside a string
// value that changed the depth count would be a routing bug — and potentially a
// cross-tenant affinity bug, since prompt content is attacker-influenced.
func TestSkippingIgnoresBracesInsideStrings(t *testing.T) {
	body := `{"prompt":"} ] { [ \" } ]","model":"m-2"}`
	got := fieldMap(t, body)
	if got["model"] != `"m-2"` {
		t.Fatalf("model = %q, want \"m-2\" — braces inside a string broke depth tracking", got["model"])
	}
}

func TestStringEscapeFreeDoesNotAllocate(t *testing.T) {
	span := []byte(`"hello world"`)
	got, ok := jsonscan.String(span)
	if !ok {
		t.Fatal("String failed")
	}
	if string(got) != "hello world" {
		t.Fatalf("got %q", got)
	}
	// Escape-free path must return a sub-slice of the input, not a copy.
	if &got[0] != &span[1] {
		t.Error("escape-free String allocated instead of sub-slicing")
	}
}

func TestStringWithEscapes(t *testing.T) {
	cases := map[string]string{
		`"a\nb"`:         "a\nb",
		`"a\tb"`:         "a\tb",
		`"q\"q"`:         `q"q`,
		`"back\\"`:       `back\`,
		`"sl\/"`:         "sl/",
		`"\u006d"`:       "m",
		`"\u00e9"`:       "\u00e9",
		`"\ud83d\ude00"`: "\U0001F600",
	}
	for in, want := range cases {
		got, ok := jsonscan.String([]byte(in))
		if !ok || string(got) != want {
			t.Errorf("String(%s) = %q,%v want %q", in, got, ok, want)
		}
	}
}

func TestBoolAndInt(t *testing.T) {
	if v, ok := jsonscan.Bool([]byte("true")); !ok || !v {
		t.Error("Bool(true)")
	}
	if v, ok := jsonscan.Bool([]byte("false")); !ok || v {
		t.Error("Bool(false)")
	}
	if _, ok := jsonscan.Bool([]byte("1")); ok {
		t.Error("Bool(1) should fail")
	}
	if v, ok := jsonscan.Int([]byte("1234")); !ok || v != 1234 {
		t.Errorf("Int = %d,%v", v, ok)
	}
	if _, ok := jsonscan.Int([]byte("12a")); ok {
		t.Error("Int(12a) should fail")
	}
}

func TestArrayIteratesElements(t *testing.T) {
	var got []string
	err := jsonscan.Array([]byte(`[{"a":1},"two",3,[4]]`), func(e []byte) bool {
		got = append(got, string(e))
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{`{"a":1}`, `"two"`, `3`, `[4]`}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("elem %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestEarlyStop(t *testing.T) {
	n := 0
	err := jsonscan.Fields([]byte(`{"a":1,"b":2,"c":3}`), func(k, v []byte) bool {
		n++
		return false // stop after the first
	})
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("visited %d fields, want 1 — early stop ignored", n)
	}
}

func TestNonObjectRejected(t *testing.T) {
	for _, in := range []string{`[1,2]`, `"str"`, `42`, ``, `   `} {
		if err := jsonscan.Fields([]byte(in), func(k, v []byte) bool { return true }); err == nil {
			t.Errorf("Fields(%q) = nil error, want error", in)
		}
	}
}

// FuzzFieldsAgreesWithEncodingJSON is the correctness backstop: for any input
// that encoding/json accepts as an object, every key we report must exist with
// an equivalent value. A structural-skip bug is a routing bug, so this runs in
// CI with a persisted corpus rather than only locally.
func FuzzFieldsAgreesWithEncodingJSON(f *testing.F) {
	f.Add(`{"model":"m","stream":true}`)
	f.Add(`{"a":{"b":[1,2,{"c":"}"}]},"d":null}`)
	f.Add(`{"s":"esc \" \\ \n end","n":-1.5e10}`)
	f.Add(`{}`)
	f.Add(`{"k":[]}`)

	f.Fuzz(func(t *testing.T, s string) {
		b := []byte(s)
		var ref map[string]json.RawMessage
		if json.Unmarshal(b, &ref) != nil {
			return // only compare on inputs encoding/json accepts as an object
		}
		if ref == nil {
			// A literal `null` unmarshals into a map without error but is not an
			// object; rejecting it is correct, so it is not a comparable case.
			return
		}
		got := map[string]string{}
		rawKeysValid := true
		if err := jsonscan.Fields(b, func(k, v []byte) bool {
			// Keys must be valid UTF-8 for this comparison to mean anything.
			// encoding/json coerces invalid bytes in a key to U+FFFD, and U+FFFD
			// is itself valid UTF-8 — so by the time `ref` exists the evidence is
			// gone, and the check has to happen on OUR raw bytes. We return raw
			// bytes by design (see TestKeysAreRawBytesNotUTF8Coerced); that
			// difference cannot affect routing, because every key the router
			// matches is an ASCII literal.
			if !utf8.Valid(k) {
				rawKeysValid = false
				return false
			}
			got[string(k)] = string(v)
			return true
		}); err != nil {
			t.Fatalf("Fields rejected valid object %q: %v", s, err)
		}
		if !rawKeysValid {
			return
		}
		if len(got) != len(ref) {
			t.Fatalf("key count %d != %d for %q\ngot %v", len(got), len(ref), s, got)
		}
		for k, rv := range ref {
			gv, ok := got[k]
			if !ok {
				t.Fatalf("missing key %q in %q", k, s)
			}
			// Compare semantically where possible: our span is raw, so re-parse
			// both. Some valid JSON does not round-trip through `any` — 1e1000
			// overflows float64 — so fall back to comparing the raw spans, which
			// is the stronger claim anyway.
			var a, c any
			errA := json.Unmarshal([]byte(gv), &a)
			errC := json.Unmarshal(rv, &c)
			if errA != nil || errC != nil {
				if strings.TrimSpace(gv) != strings.TrimSpace(string(rv)) {
					t.Fatalf("key %q: raw span %q != %q (input %q)", k, gv, rv, s)
				}
				continue
			}
			if !jsonEqual(a, c) {
				t.Fatalf("key %q: got %v, want %v (input %q)", k, a, c, s)
			}
		}
	})
}

// TestEscapedKeysAreDecoded is a security regression test for a real bug the
// fuzzer above surfaced.
//
// A client can write {"\u0073tream": true}. Every JSON parser — including the
// backend's — reads that key as "stream". If the router compared raw key bytes
// it would not see the key at all, so it would treat the request as
// non-streaming while the backend streamed, and the same trick would hide
// "model" from model-aware filtering. Keys are therefore unescaped before the
// caller compares them.
func TestEscapedKeysAreDecoded(t *testing.T) {
	cases := []struct{ body, wantKey string }{
		{`{"\u0073tream":true}`, "stream"},
		{`{"\u006d\u006fdel":"m"}`, "model"},
		{`{"mod\u0065l":"m"}`, "model"},
	}
	for _, c := range cases {
		got := fieldMap(t, c.body)
		if _, ok := got[c.wantKey]; !ok {
			t.Errorf("body %s: key %q not seen (keys: %v) — an escaped key can hide a "+
				"routing-relevant field from the router while the backend honours it",
				c.body, c.wantKey, keysOf(got))
		}
	}
}

func keysOf(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// TestKeysAreRawBytesNotUTF8Coerced documents a deliberate difference from
// encoding/json, also found by the fuzzer.
//
// encoding/json replaces invalid UTF-8 in an object key with U+FFFD; we pass the
// bytes through (escapes are decoded, but invalid bytes are not coerced), because
// coercing would mean allocating on every key. Unlike the escaped-key case above
// this cannot affect routing: every key the router matches is an ASCII literal,
// so a key containing invalid UTF-8 simply never matches.
func TestKeysAreRawBytesNotUTF8Coerced(t *testing.T) {
	body := []byte("{\"\xac\":0,\"model\":\"m\"}")
	var keys []string
	if err := jsonscan.Fields(body, func(k, v []byte) bool {
		keys = append(keys, string(k))
		return true
	}); err != nil {
		t.Fatal(err)
	}
	if len(keys) != 2 {
		t.Fatalf("keys = %q, want 2", keys)
	}
	if keys[0] != "\xac" {
		t.Errorf("key[0] = %q, want the raw byte 0xac (no U+FFFD coercion)", keys[0])
	}
	if keys[1] != "model" {
		t.Errorf("key[1] = %q, want \"model\" — a preceding invalid key must not disturb scanning", keys[1])
	}
}

func jsonEqual(a, b any) bool {
	ab, err1 := json.Marshal(a)
	bb, err2 := json.Marshal(b)
	return err1 == nil && err2 == nil && string(ab) == string(bb)
}

func BenchmarkFieldsExtractModelAndStream(b *testing.B) {
	body := makeBody(32 << 10)
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var model []byte
		var stream bool
		_ = jsonscan.Fields(body, func(k, v []byte) bool {
			switch string(k) {
			case "model":
				model, _ = jsonscan.String(v)
			case "stream":
				stream, _ = jsonscan.Bool(v)
			}
			return true
		})
		_, _ = model, stream
	}
}

func makeBody(n int) []byte {
	filler := make([]byte, 0, n)
	for len(filler) < n {
		filler = append(filler, "the quick brown fox jumps over the lazy dog "...)
	}
	out := []byte(`{"model":"llama-3-70b","stream":true,"messages":[{"role":"user","content":"`)
	out = append(out, filler[:n]...)
	return append(out, `"}]}`...)
}

// TestMalformedArrayTerminates is the regression test for a remote DoS.
//
// `{"messages":[}}` was accepted by Fields (a bare depth counter treats "[}" as a
// complete value), and Array then looped forever on it: skipValue returned
// without consuming the '}', so every iteration invoked fn with an empty element
// and made no progress. ExtractUnits appends per element, so it was an unbounded
// allocation too — a 15-byte body drove the process to its memory limit.
func TestMalformedArrayTerminates(t *testing.T) {
	bodies := []string{
		`{"messages":[}}`, `{"messages":[}]}`, `{"a":[,}`, `{"a":[]}`,
		`{"a":{]}`, `{"a":[[}]}`, `{"a":[1,}`, `{"a":[:]}`,
	}
	for _, body := range bodies {
		done := make(chan struct{})
		go func(b string) {
			defer close(done)
			_ = jsonscan.Fields([]byte(b), func(k, v []byte) bool {
				n := 0
				_ = jsonscan.Array(v, func([]byte) bool { n++; return n < 100000 })
				if n >= 100000 {
					t.Errorf("%s: Array made %d callbacks — no forward progress", b, n)
				}
				return true
			})
		}(body)
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("%s: hung", body)
		}
	}
}

// Mismatched bracket types must be rejected, not accepted as a complete value.
func TestMismatchedBracketsRejected(t *testing.T) {
	for _, in := range []string{`{"a":[}}`, `{"a":{]}`, `{"a":[[]}`} {
		var got []byte
		_ = jsonscan.Fields([]byte(in), func(k, v []byte) bool { got = v; return true })
		if len(got) > 0 && jsonAcceptable(got) {
			t.Errorf("%s: accepted %q as a value", in, got)
		}
	}
}

func jsonAcceptable(b []byte) bool {
	var v any
	return json.Unmarshal(b, &v) != nil // true when encoding/json REJECTS it
}

// FuzzArrayTerminates: any input, Array must terminate and make progress.
func FuzzArrayTerminates(f *testing.F) {
	for _, s := range []string{`[1,2]`, `[}`, `[`, `[{"a":1},"b"]`, `[[[]]]`, `[,,]`, `[:]`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		n := 0
		_ = jsonscan.Array([]byte(s), func([]byte) bool { n++; return n < 10000 })
		if n >= 10000 {
			t.Fatalf("no forward progress on %q", s)
		}
	})
}
