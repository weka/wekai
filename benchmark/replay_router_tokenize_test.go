package benchmark

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/weka/wekai/llm"
)

// fakeWordTokenizeServer implements a simple, deterministic tokenizer for
// POST /tokenize: token count = number of whitespace-separated fields in
// prompt (i.e. "tokens = words"). Mirrors the live response shape verified
// against vLLM: {"count": N, ...}.
func fakeWordTokenizeServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			http.NotFound(w, r)
			return
		}
		var body struct {
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		count := len(strings.Fields(body.Prompt))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"count":         count,
			"max_model_len": 163840,
			"tokens":        make([]int, count),
			"token_strs":    nil,
		})
	}))
}

// TestTokenizeCountBasic verifies the /tokenize request/response round-trip
// against the fake word-tokenizer.
func TestTokenizeCountBasic(t *testing.T) {
	ts := fakeWordTokenizeServer(t)
	defer ts.Close()

	n, err := tokenizeCount(ts.URL, "m", "", "one two three four")
	if err != nil {
		t.Fatalf("tokenizeCount: %v", err)
	}
	if n != 4 {
		t.Errorf("tokenizeCount = %d, want 4", n)
	}
}

// TestTokenizeCountV1Fallback verifies the attempt-then-fallback contract:
// a 404 at <base>/tokenize (root) retries once at <base>/v1/tokenize,
// mirroring discoverModelName/newReplayPoster's endpoint resolution. Real
// vLLM serves /tokenize at the root (verified live: /v1/tokenize 404s,
// /tokenize 200s) but some deployments front it behind a /v1 prefix.
func TestTokenizeCountV1Fallback(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/tokenize" {
			var body struct {
				Prompt string `json:"prompt"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"count": len(strings.Fields(body.Prompt))})
			return
		}
		// Anything else (including root /tokenize) 404s, forcing the fallback.
		http.NotFound(w, r)
	}))
	defer ts.Close()

	n, err := tokenizeCount(ts.URL, "m", "", "a b c")
	if err != nil {
		t.Fatalf("tokenizeCount with /v1 fallback: %v", err)
	}
	if n != 3 {
		t.Errorf("tokenizeCount (v1 fallback) = %d, want 3", n)
	}
}

// TestTokenizeCountErrorPropagates verifies a server that always errors
// surfaces that error rather than silently returning 0 — the fail-fast
// contract --replay-exact-tokens depends on.
func TestTokenizeCountErrorPropagates(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer ts.Close()

	_, err := tokenizeCount(ts.URL, "m", "", "hello")
	if err == nil {
		t.Fatal("tokenizeCount: expected error from a 500-ing server, got nil")
	}
}

// TestCorpusTokenIndexRoundTrip is the core exact-mode test: build a small
// index against the fake word-tokenizer over a corpus with a UNIFORM word
// width (so chunk-grid interpolation is exact, isolating the binary search
// itself from interpolation noise), then verify charsForTokens's result
// round-trips through the SAME tokenizer to within the target ± a small
// tolerance — mirroring how sizer.budget's caller (synthText) would use it.
func TestCorpusTokenIndexRoundTrip(t *testing.T) {
	ts := fakeWordTokenizeServer(t)
	defer ts.Close()

	// "wxyz " is 5 chars/word; 2000 words = 10000 chars total.
	docs := strings.Repeat("wxyz ", 2000)
	const chunkChars = 100 // 20 words/chunk — small enough for a fast test

	idx, err := buildCorpusTokenIndex(ts.URL, "m", "", docs, chunkChars)
	if err != nil {
		t.Fatalf("buildCorpusTokenIndex: %v", err)
	}

	cases := []struct {
		name   string
		off    int
		target int
	}{
		{"offset 0, small target", 0, 50},
		{"offset mid-corpus, small target", 1234, 30},
		{"offset near end, wraps around", 9900, 40}, // 9900+L exceeds docsLen=10000 for any L>100
		{"large target, multiple corpus wraps", 0, 5000},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			length := idx.charsForTokens(c.off, c.target)
			if length <= 0 {
				t.Fatalf("charsForTokens(%d, %d) = %d, want > 0", c.off, c.target, length)
			}
			got := wrappedTokenCount(t, ts.URL, docs, c.off, length)
			diff := got - c.target
			if diff < 0 {
				diff = -diff
			}
			if diff > 2 {
				t.Errorf("charsForTokens(%d, %d) -> %d chars -> retokenized to %d tokens, want within ±2 of %d",
					c.off, c.target, length, got, c.target)
			}
		})
	}
}

// wrappedTokenCount extracts docs[off:off+length] with wraparound — mirroring
// synthText's own fill loop exactly — and retokenizes it through the fake
// server, giving the "actual" token count a real client would observe for
// content sized by charsForTokens.
func wrappedTokenCount(t *testing.T, base, docs string, off, length int) int {
	t.Helper()
	var b strings.Builder
	b.Grow(length)
	for b.Len() < length {
		take := length - b.Len()
		room := len(docs) - off
		if room <= 0 {
			off = 0
			room = len(docs)
		}
		if take > room {
			take = room
		}
		b.WriteString(docs[off : off+take])
		off += take
	}
	n, err := tokenizeCount(base, "m", "", b.String()[:length])
	if err != nil {
		t.Fatalf("tokenizeCount (verification): %v", err)
	}
	return n
}

// TestCorpusTokenIndexEmptyDocs verifies the degenerate empty-corpus case
// doesn't panic and resolves to zero-length budgets.
func TestCorpusTokenIndexEmptyDocs(t *testing.T) {
	ts := fakeWordTokenizeServer(t)
	defer ts.Close()

	idx, err := buildCorpusTokenIndex(ts.URL, "m", "", "", 100)
	if err != nil {
		t.Fatalf("buildCorpusTokenIndex (empty docs): %v", err)
	}
	if got := idx.charsForTokens(0, 50); got != 0 {
		t.Errorf("charsForTokens on empty corpus = %d, want 0", got)
	}
	if got := idx.tokensInSlice(0, 100); got != 0 {
		t.Errorf("tokensInSlice on empty corpus = %v, want 0", got)
	}
}

// TestBuildCorpusTokenIndexErrorPropagates verifies a broken /tokenize
// endpoint fails the index build (not a silent empty/degraded index) — the
// property RunAutoBenchmark's --replay-exact-tokens fail-fast startup check
// depends on.
func TestBuildCorpusTokenIndexErrorPropagates(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server not ready", http.StatusServiceUnavailable)
	}))
	defer ts.Close()

	_, err := buildCorpusTokenIndex(ts.URL, "m", "", strings.Repeat("x", 500), 100)
	if err == nil {
		t.Fatal("buildCorpusTokenIndex: expected error from a failing /tokenize, got nil")
	}
}

// TestResolveTokenizeTarget covers the modelSpec -> (base, model, apiKey)
// resolution used by --replay-exact-tokens' startup probe: explicit
// model=... is honored verbatim; a spec without model=... falls back to
// /models discovery (same convention as newReplayPoster); a non-dynamic
// spec and a spec pointing at a server with no discoverable model both
// error.
func TestResolveTokenizeTarget(t *testing.T) {
	t.Run("non-dynamic spec errors", func(t *testing.T) {
		_, _, _, err := resolveTokenizeTarget("gpt-4", llm.APIKeys{})
		if err == nil {
			t.Fatal("expected error for a non-dynamic model spec")
		}
	})

	t.Run("explicit model is honored verbatim", func(t *testing.T) {
		spec := "dynamic/http://127.0.0.1:9999,type=openai,model=my-explicit-model"
		base, model, apiKey, err := resolveTokenizeTarget(spec, llm.APIKeys{OpenAI: "sk-test"})
		if err != nil {
			t.Fatalf("resolveTokenizeTarget: %v", err)
		}
		if base != "http://127.0.0.1:9999" {
			t.Errorf("base = %q, want %q", base, "http://127.0.0.1:9999")
		}
		if model != "my-explicit-model" {
			t.Errorf("model = %q, want %q", model, "my-explicit-model")
		}
		if apiKey != "sk-test" {
			t.Errorf("apiKey = %q, want %q", apiKey, "sk-test")
		}
	})

	t.Run("model discovery when spec omits model=", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/models" {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]string{{"id": "discovered-model"}},
			})
		}))
		defer ts.Close()

		spec := fmt.Sprintf("dynamic/%s,type=openai", ts.URL)
		_, model, _, err := resolveTokenizeTarget(spec, llm.APIKeys{})
		if err != nil {
			t.Fatalf("resolveTokenizeTarget: %v", err)
		}
		if model != "discovered-model" {
			t.Errorf("model = %q, want %q (discovered)", model, "discovered-model")
		}
	})

	t.Run("discovery failure propagates", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.NotFound(w, r) // /models and /v1/models both 404
		}))
		defer ts.Close()

		spec := fmt.Sprintf("dynamic/%s,type=openai", ts.URL)
		_, _, _, err := resolveTokenizeTarget(spec, llm.APIKeys{})
		if err == nil {
			t.Fatal("expected error when model discovery fails")
		}
	})
}

// TestBuildReplayTokenIndexEndToEnd exercises the full startup path
// RunAutoBenchmark uses for --replay-exact-tokens: resolve the target from
// a model spec, then build the corpus index against it.
func TestBuildReplayTokenIndexEndToEnd(t *testing.T) {
	ts := fakeWordTokenizeServer(t)
	defer ts.Close()

	spec := fmt.Sprintf("dynamic/%s,type=openai,model=m", ts.URL)
	idx, err := buildReplayTokenIndex(spec, llm.APIKeys{}, strings.Repeat("wxyz ", 100))
	if err != nil {
		t.Fatalf("buildReplayTokenIndex: %v", err)
	}
	if idx.docsLen != 500 {
		t.Errorf("idx.docsLen = %d, want 500", idx.docsLen)
	}

	t.Run("fails fast when /tokenize is broken", func(t *testing.T) {
		broken := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "down", http.StatusInternalServerError)
		}))
		defer broken.Close()
		spec := fmt.Sprintf("dynamic/%s,type=openai,model=m", broken.URL)
		if _, err := buildReplayTokenIndex(spec, llm.APIKeys{}, strings.Repeat("x", 200)); err == nil {
			t.Fatal("expected error when the target's /tokenize is broken")
		}
	})
}

// TestReplaySizerBudgetModePriority verifies replaySizer.budget's priority
// chain: exact (tokenIndex set, tokens>0, docsLen matches) beats ratio
// (charsPerToken>0) beats byte-faithful (the bytes fallback); a docsLen
// mismatch against the index defensively falls through instead of
// computing an offset against the wrong corpus length.
func TestReplaySizerBudgetModePriority(t *testing.T) {
	ts := fakeWordTokenizeServer(t)
	defer ts.Close()

	docs := strings.Repeat("wxyz ", 200) // 1000 chars
	idx, err := buildCorpusTokenIndex(ts.URL, "m", "", docs, 100)
	if err != nil {
		t.Fatalf("buildCorpusTokenIndex: %v", err)
	}

	t.Run("exact mode used when index set and tokens>0", func(t *testing.T) {
		s := replaySizer{charsPerToken: 3.4, tokenIndex: idx}
		got := s.budget("seed-a", 999, 40, len(docs))
		want := idx.charsForTokens(hashOffset("seed-a", len(docs)), 40)
		if got != want {
			t.Errorf("budget() = %d, want %d (exact mode via charsForTokens)", got, want)
		}
		// Sanity: exact-mode result should differ from what ratio mode would
		// have produced, proving exact (not ratio) actually ran.
		ratioOnly := sizeBudget(999, 40, 3.4)
		if got == ratioOnly && got == 999 {
			t.Errorf("exact-mode result %d indistinguishable from both ratio and byte fallback; test is not discriminating", got)
		}
	})

	t.Run("falls back to ratio when tokenIndex is nil", func(t *testing.T) {
		s := replaySizer{charsPerToken: 3.4}
		if got, want := s.budget("seed-b", 999, 40, len(docs)), sizeBudget(999, 40, 3.4); got != want {
			t.Errorf("budget() = %d, want %d (ratio fallback)", got, want)
		}
	})

	t.Run("falls back to bytes when neither mode is configured", func(t *testing.T) {
		s := replaySizer{}
		if got := s.budget("seed-c", 999, 40, len(docs)); got != 999 {
			t.Errorf("budget() = %d, want 999 (byte-faithful fallback)", got)
		}
	})

	t.Run("falls back to bytes when Tokens==0 even with both modes configured", func(t *testing.T) {
		s := replaySizer{charsPerToken: 3.4, tokenIndex: idx}
		if got := s.budget("seed-d", 999, 0, len(docs)); got != 999 {
			t.Errorf("budget() = %d, want 999 (Tokens==0 always falls back to bytes)", got)
		}
	})

	t.Run("docsLen mismatch defensively falls back instead of misusing the index", func(t *testing.T) {
		s := replaySizer{charsPerToken: 3.4, tokenIndex: idx}
		mismatchedDocsLen := len(docs) + 12345
		got := s.budget("seed-e", 999, 40, mismatchedDocsLen)
		want := sizeBudget(999, 40, 3.4) // ratio, since exact mode is guarded off
		if got != want {
			t.Errorf("budget() with mismatched docsLen = %d, want %d (ratio fallback, not a bogus exact-mode offset)", got, want)
		}
	})
}
