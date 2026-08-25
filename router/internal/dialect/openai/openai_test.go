package openai_test

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/dialect/openai"
)

func extract(t *testing.T, body string) []kvcache.Unit {
	t.Helper()
	u, ok := openai.New().ExtractUnits([]byte(body), "chat", nil)
	if !ok {
		return nil
	}
	return u
}

func same(a, b []kvcache.Unit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Hash != b[i].Hash {
			return false
		}
	}
	return true
}

// A message with no `content` key — every assistant turn in a tool-calling loop —
// must not hash to a constant that depends only on the role. Otherwise unrelated
// conversations appear to share a long prefix and the router confidently sends a
// request to a backend holding none of it.
func TestContentlessMessagesAreDistinguished(t *testing.T) {
	a := extract(t, `{"messages":[
	  {"role":"system","content":"shared"},
	  {"role":"assistant","tool_calls":[{"function":{"name":"alpha"}}]}]}`)
	b := extract(t, `{"messages":[
	  {"role":"system","content":"shared"},
	  {"role":"assistant","tool_calls":[{"function":{"name":"beta"}}]}]}`)

	if len(a) == 0 || len(b) == 0 {
		t.Fatal("no units extracted")
	}
	if same(a, b) {
		t.Error("two different tool calls produced identical units — they would " +
			"appear to share a prefix that does not exist on any backend")
	}
	if a[0].Hash != b[0].Hash {
		t.Error("the shared system prompt should still produce a shared first unit")
	}
}

func TestIdenticalRequestsProduceIdenticalUnits(t *testing.T) {
	body := `{"messages":[{"role":"system","content":"you are helpful"},
	                      {"role":"user","content":"hello there"}]}`
	for i := 0; i < 20; i++ {
		if !same(extract(t, body), extract(t, body)) {
			t.Fatal("extraction is not deterministic")
		}
	}
}

func TestSharedSystemPromptSharesPrefix(t *testing.T) {
	sys := strings.Repeat("system instructions. ", 200)
	a := extract(t, fmt.Sprintf(`{"messages":[{"role":"system","content":%q},{"role":"user","content":"one"}]}`, sys))
	b := extract(t, fmt.Sprintf(`{"messages":[{"role":"system","content":%q},{"role":"user","content":"two"}]}`, sys))
	if len(a) < 2 || len(b) < 2 {
		t.Fatalf("expected several units, got %d and %d", len(a), len(b))
	}
	shared := 0
	for i := 0; i < len(a) && i < len(b) && a[i].Hash == b[i].Hash; i++ {
		shared++
	}
	if shared < 2 {
		t.Errorf("only %d shared units for an identical 4KB system prompt", shared)
	}
	if same(a, b) {
		t.Error("different user turns produced identical units")
	}
}

// Malformed and hostile bodies must not panic or hang.
func TestMalformedBodiesAreSafe(t *testing.T) {
	for _, body := range []string{
		`{"messages":[}}`, `{"messages":[}]}`, `{"messages":["bare string"]}`,
		`{"messages":[{}]}`, `{"messages":[]}`, `{"messages":null}`,
		`{}`, ``, `not json`, `{"messages":[{"role":"user"}]}`,
	} {
		done := make(chan struct{})
		go func(b string) {
			defer close(done)
			defer func() {
				if v := recover(); v != nil {
					t.Errorf("%s: panicked: %v", b, v)
				}
			}()
			_, _ = openai.New().ExtractUnits([]byte(b), "chat", nil)
		}(body)
		select {
		case <-done:
		case <-timeoutAfter():
			t.Fatalf("%s: hung", body)
		}
	}
}

func timeoutAfter() <-chan time.Time { return time.After(5 * time.Second) }

// extractAs runs extraction for a named route class, which is what tells a
// generation apart from a token count on an identical body.
func extractAs(t *testing.T, class, body string) ([]kvcache.Unit, bool) {
	t.Helper()
	return openai.New().ExtractUnits([]byte(body), class, nil)
}

// The whole point of claiming /v1/messages: two turns of one Anthropic
// conversation must share a prefix, so the second lands on the backend already
// holding the first one's KV.
func TestAnthropicTurnsShareTheirSystemPrefix(t *testing.T) {
	system := strings.Repeat("a shared system prompt. ", 40)
	turn := func(user string) string {
		return fmt.Sprintf(`{"model":"m","system":%q,"messages":[{"role":"user","content":%q}]}`,
			system, user)
	}
	a, ok := extractAs(t, openai.ClassMessages, turn("first"))
	if !ok || len(a) == 0 {
		t.Fatal("an Anthropic body yielded no units; /v1/messages would route by load alone")
	}
	b, _ := extractAs(t, openai.ClassMessages, turn("second and quite different"))
	if a[0].Hash != b[0].Hash {
		t.Error("two turns sharing a system prompt produced different first units, " +
			"so the affinity tree has nothing to walk")
	}
	if same(a, b) {
		t.Error("different user turns produced identical unit sequences")
	}
}

// Anthropic's top-level `system` must land in the same position vLLM hashes it:
// before the tools and before the turns.
func TestAnthropicSystemComesBeforeToolsAndMessages(t *testing.T) {
	system := strings.Repeat("S", 400)
	tools := `[{"name":"` + strings.Repeat("T", 400) + `"}]`
	withTools := fmt.Sprintf(`{"system":%q,"tools":%s,"messages":[{"role":"user","content":%q}]}`,
		system, tools, strings.Repeat("U", 400))
	noTools := fmt.Sprintf(`{"system":%q,"messages":[{"role":"user","content":%q}]}`,
		system, strings.Repeat("U", 400))

	a, _ := extractAs(t, openai.ClassMessages, withTools)
	b, _ := extractAs(t, openai.ClassMessages, noTools)
	if len(a) == 0 || len(b) == 0 {
		t.Fatal("no units")
	}
	if a[0].Hash != b[0].Hash {
		t.Error("the system block is not first; adding tools must not disturb the " +
			"prefix ahead of them")
	}
}

// Anthropic clients put a short, near-unique preamble in system block 0. Hashing
// it would give every request a different first block and destroy the shared
// prefix entirely — which is the one thing this route was claimed to provide.
func TestAnthropicBillingBlockIsSkipped(t *testing.T) {
	shared := strings.Repeat("the real, shared system prompt. ", 30)
	body := func(preamble string) string {
		return fmt.Sprintf(`{"system":[{"type":"text","text":%q},{"type":"text","text":%q}],
		  "messages":[{"role":"user","content":"hi"}]}`, preamble, shared)
	}
	a, ok := extractAs(t, openai.ClassMessages, body("req-id 8f21a"))
	if !ok || len(a) == 0 {
		t.Fatal("no units")
	}
	b, _ := extractAs(t, openai.ClassMessages, body("req-id c40e7"))
	if !same(a, b) {
		t.Error("two requests differing only in the tiny first system block produced " +
			"different prefixes; the billing header was not skipped")
	}

	// The skip is by SIZE, not by position alone: a genuine first block that is
	// large is content, and dropping it would discard the shared prefix instead.
	big := strings.Repeat("a large genuine first block. ", 30)
	c, _ := extractAs(t, openai.ClassMessages,
		fmt.Sprintf(`{"system":[{"type":"text","text":%q}],"messages":[{"role":"user","content":"hi"}]}`, big))
	if len(c) == 0 {
		t.Error("a large first system block was skipped; only the small per-request " +
			"header should be")
	}
}

// The skip is an Anthropic artifact of the `system` FIELD. An OpenAI body's
// leading system message is genuine shared content (API-11).
func TestTheBillingSkipDoesNotTouchOpenAISystemMessages(t *testing.T) {
	u := extract(t, `{"messages":[{"role":"system","content":"short"},
	  {"role":"user","content":"hello"}]}`)
	if len(u) < 2 {
		t.Fatalf("got %d units; a short OpenAI system message must still be hashed", len(u))
	}
}

// A token count runs no forward pass and leaves no KV behind, though its body is
// the shape of the generation it is counting. Committing units for it would
// record a holder for a prefix the backend has never seen.
func TestCountTokensYieldsNoUnits(t *testing.T) {
	body := `{"model":"m","system":"` + strings.Repeat("S", 400) +
		`","messages":[{"role":"user","content":"hello"}]}`
	if _, ok := extractAs(t, openai.ClassCountTokens, body); ok {
		t.Error("a token count produced routable units; the backend would be " +
			"committed as holding a prefix it never processed")
	}
	if _, ok := extractAs(t, openai.ClassMessages, body); !ok {
		t.Error("the same body on the generation route must still yield units")
	}
}

// A stream whose terminal marker goes unrecognised makes every late failure look
// like an upstream abort.
func TestStreamScannerAcceptsBothTerminals(t *testing.T) {
	for _, tc := range []struct{ name, frame string }{
		{"openai", "data: [DONE]\n"},
		{"anthropic", "event: message_stop\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sc := openai.New().NewStreamScanner()
			if sc.Feed([]byte("data: {\"partial\":true}\n")) {
				t.Fatal("terminal reported before it arrived")
			}
			if !sc.Feed([]byte(tc.frame)) {
				t.Errorf("terminal %q not recognised", strings.TrimSpace(tc.frame))
			}
		})
	}
}

// Anthropic's input_tokens EXCLUDES what the cache accounted for, so reading it
// as the denominator reports fractions above 1.0 on exactly the well-cached
// requests router_cache_observed_fraction exists to show.
func TestExtractUsageReadsBothEnvelopes(t *testing.T) {
	for _, tc := range []struct {
		name           string
		body           string
		prompt, cached int
	}{
		{"openai", `{"usage":{"prompt_tokens":1000,"prompt_tokens_details":{"cached_tokens":750}}}`, 1000, 750},
		{"anthropic", `{"usage":{"input_tokens":100,"cache_creation_input_tokens":150,` +
			`"cache_read_input_tokens":750,"output_tokens":20}}`, 1000, 750},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u, ok := openai.New().ExtractUsage([]byte(tc.body))
			if !ok {
				t.Fatal("no usage found")
			}
			if u.PromptTokens != tc.prompt || u.CachedTokens != tc.cached {
				t.Errorf("prompt=%d cached=%d, want %d/%d",
					u.PromptTokens, u.CachedTokens, tc.prompt, tc.cached)
			}
			if frac := float64(u.CachedTokens) / float64(u.PromptTokens); frac > 1 {
				t.Errorf("observed fraction %.2f exceeds 1.0", frac)
			}
		})
	}
}
