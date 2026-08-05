// Package jsonscan is a structural JSON scanner that extracts raw byte spans
// for chosen keys and skips everything else.
//
// It exists so the router never builds a typed model of an OpenAI or Anthropic
// request body (GW-6, API-15, NG-4). v1 carried 3,823 lines of protocols/spec.rs
// plus 1,224 of validation.rs to deserialize bodies it then re-serialized
// unchanged — coupling the router to a schema that evolves upstream, for no
// benefit. Routing needs three scalars and a prefix; that is what this reads.
//
// It never allocates for the escape-free case, never unescapes unless asked, and
// returns sub-slices of the caller's buffer.
package jsonscan

import (
	"errors"
	"unicode/utf8"
)

var (
	ErrNotObject = errors.New("jsonscan: not a JSON object")
	ErrMalformed = errors.New("jsonscan: malformed JSON")
)

// Fields walks the top-level object, invoking fn with each key (unquoted, raw)
// and the raw span of its value. Return false from fn to stop early — which is
// how a caller reads `model` and `stream` and then abandons a 32 KiB body.
func Fields(b []byte, fn func(key, value []byte) bool) error {
	i := skipSpace(b, 0)
	if i >= len(b) || b[i] != '{' {
		return ErrNotObject
	}
	i++
	for {
		i = skipSpace(b, i)
		if i >= len(b) {
			return ErrMalformed
		}
		if b[i] == '}' {
			return nil
		}
		if b[i] == ',' {
			i++
			continue
		}
		if b[i] != '"' {
			return ErrMalformed
		}
		ks, ke, next, ok := scanString(b, i)
		if !ok {
			return ErrMalformed
		}
		i = skipSpace(b, next)
		if i >= len(b) || b[i] != ':' {
			return ErrMalformed
		}
		i = skipSpace(b, i+1)
		vs := i
		ve, ok := skipValue(b, i)
		if !ok {
			return ErrMalformed
		}
		i = ve

		// Keys MUST be unescaped before the caller compares them.
		//
		// This is a security property, not tidiness. A client can write
		// {"stream": true} — which every JSON parser, including the
		// backend's, reads as "stream". If we compared raw bytes we would not
		// see the key, so the router would believe the request is
		// non-streaming while the backend streams. The same trick hides
		// "model". Escaped keys are vanishingly rare in real traffic, so the
		// allocation this costs is off the hot path.
		key := b[ks:ke]
		if indexByte(key, '\\') >= 0 {
			dec, ok := unescape(key)
			if !ok {
				return ErrMalformed
			}
			key = dec
		}
		if !fn(key, b[vs:ve]) {
			return nil
		}
	}
}

// Array iterates a raw array span, invoking fn with each element's raw span.
func Array(v []byte, fn func(elem []byte) bool) error {
	i := skipSpace(v, 0)
	if i >= len(v) || v[i] != '[' {
		return ErrMalformed
	}
	i++
	for {
		i = skipSpace(v, i)
		if i >= len(v) {
			return ErrMalformed
		}
		if v[i] == ']' {
			return nil
		}
		if v[i] == ',' {
			i++
			continue
		}
		es := i
		ee, ok := skipValue(v, i)
		// A value that consumes nothing means the input is malformed at this
		// position. Without this check the loop spins forever on input like "[}",
		// invoking fn with an empty element each time — and callers append per
		// element, so it is an unbounded allocation too. A 15-byte request body
		// was enough to drive the process to its memory limit.
		if !ok || ee <= es {
			return ErrMalformed
		}
		i = ee
		if !fn(v[es:ee]) {
			return nil
		}
	}
}

// String decodes a JSON string span. The escape-free path — overwhelmingly the
// common case — returns a sub-slice with no allocation.
//
// Escapes are fully decoded, including \uXXXX and surrogate pairs. Decoding
// rather than preserving escapes verbatim also means two semantically identical
// prompts hash identically, so escaping style cannot split cache affinity.
func String(span []byte) ([]byte, bool) {
	if len(span) < 2 || span[0] != '"' || span[len(span)-1] != '"' {
		return nil, false
	}
	body := span[1 : len(span)-1]
	if indexByte(body, '\\') < 0 {
		return body, true
	}
	return unescape(body)
}

// unescape decodes the inner bytes of a JSON string.
func unescape(body []byte) ([]byte, bool) {
	out := make([]byte, 0, len(body))
	for i := 0; i < len(body); i++ {
		c := body[i]
		if c != '\\' {
			out = append(out, c)
			continue
		}
		i++
		if i >= len(body) {
			return nil, false
		}
		switch body[i] {
		case 'n':
			out = append(out, '\n')
		case 't':
			out = append(out, '\t')
		case 'r':
			out = append(out, '\r')
		case 'b':
			out = append(out, '\b')
		case 'f':
			out = append(out, '\f')
		case '/':
			out = append(out, '/')
		case '"':
			out = append(out, '"')
		case '\\':
			out = append(out, '\\')
		case 'u':
			r, consumed, ok := decodeUnicodeEscape(body, i)
			if !ok {
				return nil, false
			}
			out = utf8.AppendRune(out, r)
			i += consumed
		default:
			return nil, false
		}
	}
	return out, true
}

// decodeUnicodeEscape decodes \uXXXX at body[i] == 'u', joining a surrogate
// pair when one follows. It returns how many bytes past 'u' were consumed.
func decodeUnicodeEscape(body []byte, i int) (r rune, consumed int, ok bool) {
	// body[i] == 'u'; its four hex digits are body[i+1:i+5].
	if i+5 > len(body) {
		return 0, 0, false
	}
	hi, ok := hex4(body[i+1 : i+5])
	if !ok {
		return 0, 0, false
	}
	switch {
	case hi >= 0xD800 && hi <= 0xDBFF:
		// High surrogate. A low surrogate may follow as a further \uXXXX,
		// occupying the six bytes at i+5.
		if i+11 <= len(body) && body[i+5] == '\\' && body[i+6] == 'u' {
			if lo, ok := hex4(body[i+7 : i+11]); ok && lo >= 0xDC00 && lo <= 0xDFFF {
				return 0x10000 + (rune(hi)-0xD800)<<10 + (rune(lo) - 0xDC00), 10, true
			}
		}
		// Lone surrogate: encode U+FFFD, matching encoding/json.
		return utf8.RuneError, 4, true
	case hi >= 0xDC00 && hi <= 0xDFFF:
		return utf8.RuneError, 4, true // lone low surrogate
	default:
		return rune(hi), 4, true
	}
}

func hex4(b []byte) (uint32, bool) {
	if len(b) < 4 {
		return 0, false
	}
	var v uint32
	for i := 0; i < 4; i++ {
		v <<= 4
		switch c := b[i]; {
		case c >= '0' && c <= '9':
			v |= uint32(c - '0')
		case c >= 'a' && c <= 'f':
			v |= uint32(c-'a') + 10
		case c >= 'A' && c <= 'F':
			v |= uint32(c-'A') + 10
		default:
			return 0, false
		}
	}
	return v, true
}

// Bool decodes a JSON boolean span.
func Bool(span []byte) (bool, bool) {
	switch {
	case len(span) == 4 && string(span) == "true":
		return true, true
	case len(span) == 5 && string(span) == "false":
		return false, true
	}
	return false, false
}

// Int decodes a non-negative JSON integer span.
func Int(span []byte) (int, bool) {
	if len(span) == 0 {
		return 0, false
	}
	n := 0
	for _, c := range span {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

func skipSpace(b []byte, i int) int {
	for i < len(b) {
		switch b[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// scanString returns the inner bounds of a string plus the index after it.
// String scanning is escape-aware, which is the one real correctness hazard in
// structural skipping: a brace or bracket inside a string must not affect depth.
func scanString(b []byte, i int) (start, end, next int, ok bool) {
	if i >= len(b) || b[i] != '"' {
		return 0, 0, 0, false
	}
	i++
	start = i
	for i < len(b) {
		switch b[i] {
		case '\\':
			i += 2
			continue
		case '"':
			return start, i, i + 1, true
		}
		i++
	}
	return 0, 0, 0, false
}

// skipValue returns the index just past the value beginning at i.
func skipValue(b []byte, i int) (int, bool) {
	if i >= len(b) {
		return 0, false
	}
	switch b[i] {
	case '"':
		_, _, next, ok := scanString(b, i)
		return next, ok
	case '{', '[':
		// Track the opener types, not just a depth count. A bare counter accepts
		// "[}" and "{]" as complete values, which is what let a malformed array
		// reach Array and spin.
		var stack []byte
		for i < len(b) {
			switch c := b[i]; c {
			case '"':
				_, _, next, ok := scanString(b, i)
				if !ok {
					return 0, false
				}
				i = next
				continue
			case '{', '[':
				stack = append(stack, c)
			case '}', ']':
				if len(stack) == 0 {
					return 0, false
				}
				want := byte('}')
				if stack[len(stack)-1] == '[' {
					want = ']'
				}
				if c != want {
					return 0, false
				}
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					return i + 1, true
				}
			}
			i++
		}
		return 0, false
	default:
		// Number, true, false, null: run to the next structural byte. A value
		// cannot START with a structural byte — treating that as a zero-length
		// scalar is what produced the non-progress loop.
		switch b[i] {
		case ',', '}', ']', ':':
			return 0, false
		}
		start := i
		for i < len(b) {
			switch b[i] {
			case ',', '}', ']', ' ', '\t', '\n', '\r':
				return i, true
			}
			i++
		}
		if i == start {
			return 0, false
		}
		return i, true
	}
}

func indexByte(b []byte, c byte) int {
	for i := 0; i < len(b); i++ {
		if b[i] == c {
			return i
		}
	}
	return -1
}
