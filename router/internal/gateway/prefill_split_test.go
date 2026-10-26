package gateway_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/weka/wekai/router/internal/dialect/openai"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/internal/policy/affinity"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/prefill"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/routing"
)

// These run the real mock engine, with its shared KV tier and its
// large-prefill rejection, behind the real gateway. A request the engine
// refuses must be prefilled on a node taken out of decode routing and then
// served, and the caller must not see any of it.

const rejectAt = 2000

// heavyPrompt recomputes well over rejectAt tokens on a cold fleet.
func heavyPrompt(seed string) string { return seed + " " + strings.Repeat("lorem ipsum ", 3000) }

// call is one request a backend received.
type call struct {
	node      int
	path      string
	maxTokens float64
	stream    bool
	bypass    bool
}

type splitFleet struct {
	gw      http.Handler
	srv     *httptest.Server
	backend []*registry.Backend
	mgr     *prefill.Manager

	mu    sync.Mutex
	calls []call
}

func (f *splitFleet) callsOf(pred func(call) bool) (out []call) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, c := range f.calls {
		if pred(c) {
			out = append(out, c)
		}
	}
	return out
}

// newSplitFleet builds n mock engines sharing one tier. With split false the
// gateway has no prefill manager.
func newSplitFleet(t *testing.T, n int, split bool, surface mockvllm.Surface) *splitFleet {
	t.Helper()
	f := &splitFleet{}
	tier := mockvllm.NewTier()
	reg := registry.New(registry.Options{})
	for i := range n {
		eng := mockvllm.NewEngine(mockvllm.Config{
			Tier: tier, PrefillRejectTokens: rejectAt, DefaultMaxTokens: 4,
		})
		t.Cleanup(eng.Close)
		srv := mockvllm.NewServer(eng)
		srv.Surface = surface
		h := srv.Handler()
		node := i
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				body, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(body))
				var doc struct {
					MaxTokens float64        `json:"max_tokens"`
					Stream    bool           `json:"stream"`
					XArgs     map[string]any `json:"vllm_xargs"`
				}
				_ = json.Unmarshal(body, &doc)
				f.mu.Lock()
				f.calls = append(f.calls, call{node: node, path: r.URL.Path, maxTokens: doc.MaxTokens,
					stream: doc.Stream, bypass: doc.XArgs["weka_prefill"] != nil})
				f.mu.Unlock()
			}
			h.ServeHTTP(w, r)
		}))
		t.Cleanup(ts.Close)
		b, err := reg.Add(registry.Spec{URL: ts.URL, Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		f.backend = append(f.backend, b)
	}

	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg())}
	if split {
		f.mgr = prefill.New(prefill.Config{PoolName: t.Name()})
		p.Split = f.mgr
	}
	tbl, err := routing.NewTable([]routing.Rule{{Pool: p}})
	if err != nil {
		t.Fatal(err)
	}
	gw := gateway.New(gateway.Config{MaxBodyBytes: 64 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New())
	f.gw = gw
	f.srv = httptest.NewServer(gw)
	t.Cleanup(f.srv.Close)
	return f
}

func (f *splitFleet) post(t *testing.T, path, body string) (int, string) {
	t.Helper()
	resp, err := http.Post(f.srv.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func chatBody(content string, stream bool) string {
	b, _ := json.Marshal(map[string]any{
		"model": "m", "stream": stream, "max_tokens": 4,
		"stream_options": map[string]any{"include_usage": true},
		"messages":       []map[string]any{{"role": "user", "content": content}},
	})
	if !stream {
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		delete(m, "stream_options")
		b, _ = json.Marshal(m)
	}
	return string(b)
}

func isPrefillJob(c call) bool { return c.bypass && c.maxTokens == 1 }
func isDecode(c call) bool     { return c.bypass && c.maxTokens != 1 }

func TestHeavyPrefillRunsOnADrainedNodeThenDecodesElsewhere(t *testing.T) {
	cases := []struct {
		name   string
		path   string
		body   string
		stream bool
		sfc    mockvllm.Surface
		sse    bool
	}{
		{name: "chat", path: "/v1/chat/completions", body: chatBody(heavyPrompt("a"), false)},
		{name: "chat streaming", path: "/v1/chat/completions", body: chatBody(heavyPrompt("b"), true), stream: true, sse: true},
		{name: "completions", path: "/v1/completions", body: `{"model":"m","prompt":` + mustJSON(heavyPrompt("c")) + `,"max_tokens":4}`},
		{name: "anthropic messages", path: "/v1/messages", sfc: mockvllm.SurfaceAnthropic,
			body: `{"model":"m","max_tokens":4,"messages":[{"role":"user","content":` + mustJSON(heavyPrompt("d")) + `}]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newSplitFleet(t, 3, true, c.sfc)
			status, body := f.post(t, c.path, c.body)
			if status != http.StatusOK {
				t.Fatalf("client saw %d: %s", status, body)
			}
			if strings.Contains(body, prefill.RejectionType) {
				t.Fatalf("client saw the rejection: %s", body)
			}
			if c.sse && !strings.Contains(body, "[DONE]") {
				t.Errorf("a streaming request did not get its stream back:\n%s", body)
			}

			jobs := f.callsOf(isPrefillJob)
			if len(jobs) != 1 {
				t.Fatalf("prefill jobs = %d, want 1 (calls %+v)", len(jobs), f.calls)
			}
			if jobs[0].stream {
				t.Error("the prefill job streamed")
			}
			decodes := f.callsOf(isDecode)
			if len(decodes) != 1 {
				t.Fatalf("decode resubmissions = %d, want 1 (calls %+v)", len(decodes), f.calls)
			}
			if decodes[0].node == jobs[0].node {
				t.Errorf("decode ran on the prefill node %d", jobs[0].node)
			}
			if decodes[0].stream != c.stream {
				t.Errorf("decode stream = %v, want the client's %v", decodes[0].stream, c.stream)
			}
			if got := f.mgr.ModeOf(f.backend[jobs[0].node]); got == prefill.Decode {
				t.Errorf("prefill node is still in decode mode")
			}

			st := f.mgr.State()
			if st.Rejections != 1 || st.Jobs != 1 || st.Conversions != 1 || st.BypassFallbacks != 0 {
				t.Errorf("state = %+v, want 1 rejection, 1 job, 1 conversion, 0 fallbacks", st)
			}
		})
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestDecodeRoutingAvoidsThePrefillNodeAfterwards(t *testing.T) {
	f := newSplitFleet(t, 3, true, mockvllm.SurfaceVLLM)
	if status, body := f.post(t, "/v1/chat/completions", chatBody(heavyPrompt("x"), false)); status != 200 {
		t.Fatalf("heavy: %d %s", status, body)
	}
	pnode := f.callsOf(isPrefillJob)[0].node
	f.mu.Lock()
	f.calls = nil
	f.mu.Unlock()

	for i := range 12 {
		status, body := f.post(t, "/v1/chat/completions", chatBody("short question "+strings.Repeat("q", i), false))
		if status != 200 {
			t.Fatalf("light request %d: %d %s", i, status, body)
		}
	}
	if n := len(f.callsOf(func(c call) bool { return c.node == pnode })); n != 0 {
		t.Errorf("%d decode requests were routed to the prefill node", n)
	}
}

func TestSplitOffPassesTheRejectionThrough(t *testing.T) {
	f := newSplitFleet(t, 2, false, mockvllm.SurfaceVLLM)
	status, body := f.post(t, "/v1/chat/completions", chatBody(heavyPrompt("y"), false))
	if status != http.StatusServiceUnavailable || !strings.Contains(body, prefill.RejectionType) {
		t.Errorf("with the split off the client got %d %s, want the engine's 503", status, body)
	}
	if n := len(f.callsOf(isPrefillJob)); n != 0 {
		t.Errorf("%d prefill jobs with the split off", n)
	}
}

func TestClientBypassIsNeverRejectedOrSplit(t *testing.T) {
	f := newSplitFleet(t, 2, true, mockvllm.SurfaceVLLM)
	body := `{"model":"m","max_tokens":4,"vllm_xargs":{"weka_prefill":1},"messages":[{"role":"user","content":` +
		mustJSON(heavyPrompt("z")) + `}]}`
	if status, b := f.post(t, "/v1/chat/completions", body); status != 200 {
		t.Fatalf("%d %s", status, b)
	}
	if st := f.mgr.State(); st.Rejections != 0 || st.Jobs != 0 {
		t.Errorf("a bypassed request was split: %+v", st)
	}
}

func TestNoPrefillNodeFallsBackToTheOriginalWithBypass(t *testing.T) {
	f := newSplitFleet(t, 1, true, mockvllm.SurfaceVLLM)
	status, body := f.post(t, "/v1/chat/completions", chatBody(heavyPrompt("w"), false))
	if status != http.StatusOK {
		t.Fatalf("client saw %d: %s", status, body)
	}
	if n := len(f.callsOf(isPrefillJob)); n != 0 {
		t.Errorf("%d prefill jobs on a one-node fleet", n)
	}
	if n := len(f.callsOf(isDecode)); n != 1 {
		t.Errorf("bypassed resends = %d, want 1 (calls %+v)", n, f.calls)
	}
	if st := f.mgr.State(); st.Rejections != 1 || st.BypassFallbacks != 1 {
		t.Errorf("state = %+v, want 1 rejection and 1 bypass fallback", st)
	}
}

// A backend that refuses even a bypassed request must not make the router
// loop: the refusal is the caller's after one prefill job and one resubmission.
func TestRejectionOfABypassedRequestDoesNotLoop(t *testing.T) {
	var mu sync.Mutex
	n := 0
	reject := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		mu.Lock()
		n++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set(prefill.RejectHeader, "9000")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"type":"large_prefill_rejected","param":"9000","code":503}}`))
	})
	reg := registry.New(registry.Options{})
	for range 3 {
		ts := httptest.NewServer(reject)
		t.Cleanup(ts.Close)
		b, err := reg.Add(registry.Spec{URL: ts.URL, Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
	}
	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg()),
		Split: prefill.New(prefill.Config{PoolName: t.Name()})}
	tbl, err := routing.NewTable([]routing.Rule{{Pool: p}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(gateway.New(gateway.Config{MaxBodyBytes: 1 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New()))
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(chatBody("hello", false)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	mu.Lock()
	defer mu.Unlock()
	if n != 3 {
		t.Errorf("backends saw %d requests, want 3 (original, prefill job, resubmission)", n)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want the backend's 503 once the bypass was refused too", resp.StatusCode)
	}
}

func TestOtherServiceUnavailablePassesThroughIntact(t *testing.T) {
	const msg = `{"error":{"type":"overloaded","message":"busy","code":503}}`
	srvB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(msg))
	}))
	t.Cleanup(srvB.Close)
	reg := registry.New(registry.Options{})
	b, err := reg.Add(registry.Spec{URL: srvB.URL, Prov: registry.ProvStatic, Capacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	b.SetHealth(registry.Healthy)
	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg()),
		Split: prefill.New(prefill.Config{PoolName: t.Name()})}
	tbl, _ := routing.NewTable([]routing.Rule{{Pool: p}})
	srv := httptest.NewServer(gateway.New(gateway.Config{MaxBodyBytes: 1 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New()))
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody("hi", false)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 503 || string(got) != msg {
		t.Errorf("got %d %q, want the backend's own 503 byte for byte", resp.StatusCode, got)
	}
	if st := p.Split.State(); st.Rejections != 0 {
		t.Errorf("an unrelated 503 was counted as a rejection: %+v", st)
	}
}

func affinityCfg() affinity.Config { return affinity.Config{} }

// A rejection is a routing signal. However many a backend sends, it stays
// healthy: no breaker opens and no failure is recorded against it.
func TestRejectionsNeverCountAgainstBackendHealth(t *testing.T) {
	reject := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		w.Header().Set(prefill.RejectHeader, "9000")
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(reject.Close)
	reg := registry.New(registry.Options{})
	b, err := reg.Add(registry.Spec{URL: reject.URL, Prov: registry.ProvStatic, Capacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	b.SetHealth(registry.Healthy)
	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg()),
		Split: prefill.New(prefill.Config{PoolName: t.Name()})}
	tbl, _ := routing.NewTable([]routing.Rule{{Pool: p}})
	srv := httptest.NewServer(gateway.New(gateway.Config{MaxBodyBytes: 1 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New()))
	t.Cleanup(srv.Close)

	for range 60 {
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody("hi", false)))
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
	}
	if !b.Available() {
		t.Error("the backend was ejected by rejections")
	}
	if n := b.Failed.Load(); n != 0 {
		t.Errorf("%d failures recorded against a backend that only sent rejections", n)
	}
}
