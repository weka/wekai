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
