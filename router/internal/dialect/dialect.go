// Package dialect isolates every piece of wire-format knowledge in the router.
//
// Nothing else may know what an OpenAI or Anthropic request looks like. The
// registry, lease, policy, proxy, circuit and health packages are dialect-blind,
// and hack/ enforces that with an import fence (API-1). That fence is the whole
// point: v1's protocols/ grew to ~5,000 lines of typed schema woven through the
// routing path, which pinned the router to one upstream version and made adding
// a second wire format a rewrite rather than a package.
//
// A Dialect covers exactly seven concerns (API-2). Adding an eighth means the
// abstraction is leaking.
package dialect

import (
	"net/http"
	"sync"

	"github.com/weka/wekai/kvcache"
)

// Route is one endpoint a dialect claims.
type Route struct {
	// Pattern is a Go 1.22+ ServeMux pattern, e.g. "POST /v1/chat/completions".
	Pattern string
	// Class is the route class used for metrics labels and extraction rules.
	Class string
	// Stream reports whether this route can produce a streaming response.
	Stream bool
}

// Introspection is the set of routing-relevant scalars read from a body.
type Introspection struct {
	Model  string
	Stream bool
}

// Usage is the token accounting a response reports.
//
// CachedTokens is the closed-loop signal for prefix-cache prediction accuracy
// (RES-3). Note vLLM only populates it when started with
// --enable-prompt-tokens-details, and reports local and external cache hits
// combined, so it validates how much hit but not which memory tier.
type Usage struct {
	PromptTokens int
	CachedTokens int
	TotalTokens  int
}

// StreamScanner detects a dialect's stream-terminal marker.
//
// It must be correct across arbitrary chunk boundaries. v1 scanned each chunk
// for `data: [DONE]` with a 12-byte sliding window, which silently missed the
// marker whenever it straddled two chunks (STR-N3).
type StreamScanner interface {
	// Feed reports whether the terminal marker has now been seen.
	Feed(p []byte) bool
}

// Dialect is the complete wire-format contract.
type Dialect interface {
	// (0) identity
	ID() string
	// (1) the routes it claims
	Routes() []Route
	// (2) routing-relevant scalars, without deserializing the body (GW-6)
	Introspect(body []byte) Introspection
	// (3) prefix units for cache-affinity policies, appended to dst
	ExtractUnits(body []byte, class string, dst []kvcache.Unit) ([]kvcache.Unit, bool)
	// (4) stream terminal framing
	NewStreamScanner() StreamScanner
	// (5) error envelope, rendered in the INBOUND dialect (API-9)
	WriteError(w http.ResponseWriter, status int, code, msg string)
	// (6) usage extraction from a response body
	ExtractUsage(body []byte) (Usage, bool)
	// (7) recognized inbound credential form
	Credential(h http.Header) (token string, ok bool)
}

var (
	mu         sync.RWMutex
	registered = map[string]Dialect{}
)

// Register adds a dialect. Called only from cmd/, so the set of compiled-in
// dialects is explicit at the wiring layer rather than a package side effect.
func Register(d Dialect) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := registered[d.ID()]; dup {
		panic("dialect: duplicate registration for " + d.ID())
	}
	registered[d.ID()] = d
}

func Lookup(id string) (Dialect, bool) {
	mu.RLock()
	defer mu.RUnlock()
	d, ok := registered[id]
	return d, ok
}

// All returns every registered dialect.
func All() []Dialect {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]Dialect, 0, len(registered))
	for _, d := range registered {
		out = append(out, d)
	}
	return out
}

// Reset clears the registry. Tests only.
func Reset() {
	mu.Lock()
	defer mu.Unlock()
	registered = map[string]Dialect{}
}

// LineScanner finds a line-oriented terminal marker across chunk boundaries.
//
// It carries at most maxCarry bytes of a partial line. Beyond that it stops
// matching rather than growing without bound: a pathological unterminated line
// degrades to "terminal not seen", which is safe because the load lease is bound
// to the response body being finished, never to this marker (API-N3).
type LineScanner struct {
	Marker []byte

	carry  []byte
	capped bool
	seen   bool
}

const maxCarry = 8 << 10

func (s *LineScanner) Feed(p []byte) bool {
	for len(p) > 0 {
		j := indexByte(p, '\n')
		if j < 0 {
			if !s.capped {
				if len(s.carry)+len(p) <= maxCarry {
					s.carry = append(s.carry, p...)
				} else {
					s.capped = true
					s.carry = s.carry[:0]
				}
			}
			return s.seen
		}
		line := p[:j]
		if len(s.carry) > 0 {
			s.carry = append(s.carry, line...)
			line = s.carry
		}
		if !s.capped && hasPrefix(trimCR(line), s.Marker) {
			s.seen = true
		}
		s.carry = s.carry[:0]
		s.capped = false
		p = p[j+1:]
	}
	return s.seen
}

func trimCR(b []byte) []byte {
	for len(b) > 0 && b[len(b)-1] == '\r' {
		b = b[:len(b)-1]
	}
	return b
}

func hasPrefix(b, p []byte) bool {
	if len(b) < len(p) {
		return false
	}
	for i := range p {
		if b[i] != p[i] {
			return false
		}
	}
	return true
}

func indexByte(b []byte, c byte) int {
	for i := 0; i < len(b); i++ {
		if b[i] == c {
			return i
		}
	}
	return -1
}
