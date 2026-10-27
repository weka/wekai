package gateway_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/weka/wekai/router/internal/dialect/openai"
	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/mockvllm"
	"github.com/weka/wekai/router/internal/pd"
	"github.com/weka/wekai/router/internal/pool"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
	"github.com/weka/wekai/router/internal/routing"
)

// These run the real mock engine behind the real gateway with prefill/decode
// disaggregation on: the first nP backends are P nodes, the rest D nodes.

type pdCall struct {
	node      int
	path      string
	requestID string
	body      map[string]json.RawMessage
}

func (c pdCall) isPrefillLeg() bool {
	var kv struct {
		RemoteDecode bool `json:"do_remote_decode"`
	}
	_ = json.Unmarshal(c.body["kv_transfer_params"], &kv)
	return kv.RemoteDecode
}

type pdFleet struct {
	srv     *httptest.Server
	nP      int
	servers []*mockvllm.Server
	backend []*registry.Backend
	// failP makes P legs fail: 1 answers 500, 2 answers 200 without params.
	failP atomic.Int32

	mu    sync.Mutex
	calls []pdCall
}

func (f *pdFleet) callsOf(pred func(pdCall) bool) (out []pdCall) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, c := range f.calls {
		if pred(c) {
			out = append(out, c)
		}
	}
	return out
}

func newPDFleet(t *testing.T, nP, nD int, rewrite string, surface ...mockvllm.Surface) *pdFleet {
	t.Helper()
	f := &pdFleet{nP: nP}
	reg := registry.New(registry.Options{})
	var pURLs []string
	for i := range nP + nD {
		eng := mockvllm.NewEngine(mockvllm.Config{DefaultMaxTokens: 4})
		t.Cleanup(eng.Close)
		ms := mockvllm.NewServer(eng)
		if len(surface) > 0 {
			ms.Surface = surface[0]
		}
		ms.EngineID = "engine-" + string(rune('a'+i))
		f.servers = append(f.servers, ms)
		h := ms.Handler()
		node := i
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				raw, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(raw))
				c := pdCall{node: node, path: r.URL.Path, requestID: r.Header.Get("X-Request-Id")}
				_ = json.Unmarshal(raw, &c.body)
				f.mu.Lock()
				f.calls = append(f.calls, c)
				f.mu.Unlock()
				if c.isPrefillLeg() {
					switch f.failP.Load() {
					case 1:
						http.Error(w, "boom", http.StatusInternalServerError)
						return
					case 2:
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(`{"choices":[]}`))
						return
					}
				}
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
		if i < nP {
			pURLs = append(pURLs, ts.URL)
		}
	}
	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg()),
		PD: pd.New("default", pURLs), PDFlow: mustFlow(t, affinityCfg())}
	tbl, err := routing.NewTable([]routing.Rule{{Pool: p, RewriteModel: rewrite}})
	if err != nil {
		t.Fatal(err)
	}
	gw := gateway.New(gateway.Config{MaxBodyBytes: 64 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New())
	f.srv = httptest.NewServer(gw)
	t.Cleanup(f.srv.Close)
	return f
}

func (f *pdFleet) post(t *testing.T, path, body string) (int, string) {
	t.Helper()
	resp, err := http.Post(f.srv.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func (f *pdFleet) isP(node int) bool { return node < f.nP }

func TestPDRequestRunsPrefillLegThenDecodeLegWithItsParams(t *testing.T) {
	cases := []struct {
		name   string
		path   string
		body   string
		stream bool
	}{
		{"chat", "/v1/chat/completions", chatBody("hello there", false), false},
		{"chat streaming", "/v1/chat/completions", chatBody("hello there", true), true},
		{"completions", "/v1/completions", `{"model":"m","prompt":"hello there","max_tokens":4}`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newPDFleet(t, 1, 2, "")
			status, body := f.post(t, c.path, c.body)
			if status != http.StatusOK {
				t.Fatalf("client saw %d: %s", status, body)
			}
			if c.stream && !strings.Contains(body, "[DONE]") {
				t.Errorf("stream not preserved:\n%s", body)
			}
			if len(f.calls) != 2 {
				t.Fatalf("calls = %d, want a prefill leg and a decode leg", len(f.calls))
			}
			pl, dl := f.calls[0], f.calls[1]
			if !pl.isPrefillLeg() || !f.isP(pl.node) {
				t.Fatalf("first leg is not a prefill leg on a P node: %+v", pl)
			}
			if f.isP(dl.node) || dl.isPrefillLeg() {
				t.Fatalf("second leg is not a decode leg on a D node: %+v", dl)
			}

			var kv map[string]json.RawMessage
			_ = json.Unmarshal(pl.body["kv_transfer_params"], &kv)
			for _, k := range []string{"do_remote_prefill", "remote_engine_id", "remote_block_ids", "remote_host", "remote_port"} {
				if _, ok := kv[k]; !ok {
					t.Errorf("prefill leg kv_transfer_params lacks %s", k)
				}
			}
			if string(kv["do_remote_prefill"]) != "false" || string(kv["remote_engine_id"]) != "null" {
				t.Errorf("prefill leg kv_transfer_params = %s", pl.body["kv_transfer_params"])
			}
			if string(pl.body["max_tokens"]) != "1" || string(pl.body["stream"]) != "false" {
				t.Errorf("prefill leg max_tokens=%s stream=%s", pl.body["max_tokens"], pl.body["stream"])
			}
			if _, has := pl.body["stream_options"]; has {
				t.Error("prefill leg kept stream_options")
			}

			if string(dl.body["max_tokens"]) != "4" {
				t.Errorf("decode leg max_tokens = %s, want the client's 4", dl.body["max_tokens"])
			}
			if c.stream && string(dl.body["stream"]) != "true" {
				t.Errorf("decode leg stream = %s", dl.body["stream"])
			}
			var got map[string]any
			_ = json.Unmarshal(dl.body["kv_transfer_params"], &got)
			if got["remote_engine_id"] != "engine-a" || got["do_remote_prefill"] != true {
				t.Errorf("decode leg did not get the P node's params: %s", dl.body["kv_transfer_params"])
			}
			recv := f.servers[dl.node].KVTransfersReceived()
			if len(recv) != 1 || !bytes.Contains(recv[0], []byte("engine-a")) {
				t.Errorf("D node received %s", recv)
			}

			if pl.requestID == "" || pl.requestID != dl.requestID {
				t.Errorf("request ids differ across legs: %q vs %q", pl.requestID, dl.requestID)
			}
		})
	}
}

func TestPDLegsNeverCrossNodeSets(t *testing.T) {
	f := newPDFleet(t, 2, 3, "")
	for i := range 30 {
		status, body := f.post(t, "/v1/chat/completions", chatBody("q "+strings.Repeat("x", i), false))
		if status != http.StatusOK {
			t.Fatalf("request %d: %d %s", i, status, body)
		}
	}
	for _, c := range f.calls {
		if c.isPrefillLeg() != f.isP(c.node) {
			t.Fatalf("leg %+v ran on the wrong set (nP=%d)", c, f.nP)
		}
	}
	if n := len(f.calls); n != 60 {
		t.Errorf("calls = %d, want 2 per request", n)
	}
}

// Turns of one conversation share a prefix, so affinity must keep landing them
// on the same P node and the same D node.
func TestPDAffinityKeepsASessionOnItsPAndDNodes(t *testing.T) {
	f := newPDFleet(t, 3, 3, "")
	type nodes struct{ p, d map[int]bool }
	sessions := map[int]*nodes{}
	for turn := range 4 {
		for s := range 6 {
			msgs := []map[string]any{{"role": "system", "content": strings.Repeat("session-"+string(rune('A'+s))+" ", 200)}}
			for k := 0; k <= turn; k++ {
				msgs = append(msgs, map[string]any{"role": "user", "content": "question " + string(rune('a'+k)) + strings.Repeat(" detail", 50)},
					map[string]any{"role": "assistant", "content": "answer " + string(rune('a'+k)) + strings.Repeat(" detail", 50)})
			}
			b, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 4, "messages": msgs})
			before := len(f.calls)
			if status, body := f.post(t, "/v1/chat/completions", string(b)); status != http.StatusOK {
				t.Fatalf("session %d turn %d: %d %s", s, turn, status, body)
			}
			n := sessions[s]
			if n == nil {
				n = &nodes{p: map[int]bool{}, d: map[int]bool{}}
				sessions[s] = n
			}
			for _, c := range f.calls[before:] {
				if c.isPrefillLeg() {
					n.p[c.node] = true
				} else {
					n.d[c.node] = true
				}
			}
		}
	}
	for s, n := range sessions {
		if len(n.p) != 1 || len(n.d) != 1 {
			t.Errorf("session %d moved: P nodes %v, D nodes %v", s, n.p, n.d)
		}
	}
}

func TestPDPrefillFailureFallsBackToDecodeWithoutParams(t *testing.T) {
	for name, mode := range map[string]int32{"error status": 1, "no kv_transfer_params": 2} {
		t.Run(name, func(t *testing.T) {
			f := newPDFleet(t, 1, 2, "")
			f.failP.Store(mode)
			status, body := f.post(t, "/v1/chat/completions", chatBody("hello there", false))
			if status != http.StatusOK {
				t.Fatalf("client saw %d: %s", status, body)
			}
			dl := f.callsOf(func(c pdCall) bool { return !c.isPrefillLeg() })
			if len(dl) != 1 || f.isP(dl[0].node) {
				t.Fatalf("decode calls = %+v", dl)
			}
			if _, has := dl[0].body["kv_transfer_params"]; has {
				t.Errorf("fallback decode carried kv_transfer_params: %s", dl[0].body["kv_transfer_params"])
			}
			if string(dl[0].body["max_tokens"]) != "4" {
				t.Errorf("fallback decode max_tokens = %s", dl[0].body["max_tokens"])
			}
		})
	}
}

func TestPDRewritesTheModelOnBothLegs(t *testing.T) {
	f := newPDFleet(t, 1, 1, "served-model")
	if status, body := f.post(t, "/v1/chat/completions", chatBody("hi", false)); status != http.StatusOK {
		t.Fatalf("%d %s", status, body)
	}
	if len(f.calls) != 2 {
		t.Fatalf("calls = %d", len(f.calls))
	}
	for _, c := range f.calls {
		if string(c.body["model"]) != `"served-model"` {
			t.Errorf("leg on node %d sent model %s", c.node, c.body["model"])
		}
	}
}

func TestPDMessagesAreServedOnDecodeNodesOnly(t *testing.T) {
	f := newPDFleet(t, 1, 2, "", mockvllm.SurfaceAnthropic)
	status, body := f.post(t, "/v1/messages", `{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}`)
	if status != http.StatusOK {
		t.Fatalf("%d %s", status, body)
	}
	if len(f.calls) != 1 || f.isP(f.calls[0].node) {
		t.Fatalf("calls = %+v, want one on a D node", f.calls)
	}
	if _, has := f.calls[0].body["kv_transfer_params"]; has {
		t.Error("messages request carried kv_transfer_params")
	}
}
