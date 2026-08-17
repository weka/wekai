package benchmark

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/weka/wekai/llm"
)

// A bare endpoint is resolved by probing: <base>+leaf first, /v1 inserted on a
// 404. That probe must be paid ONCE for the process, not once per replay
// instance and not again on every 429 backoff iteration.
//
// Both of those were happening. A poster is built per instance, so each one
// re-probed from scratch; and endpointAttempts is called inside the 429 loop,
// so a poster whose first request kept getting shed re-probed each time round.
// On an idle fleet neither shows: the 404s appeared only once the fleet
// saturated and retries began, at which point they were 16.5% of all requests —
// none of which the fleet ever saw, while all of them counted as errors.

func resetSharedEndpoints() {
	epSharedMu.Lock()
	epShared = map[string]string{}
	epSharedMu.Unlock()
	epLogMu.Lock()
	epLogSeen = map[string]bool{}
	epLogMu.Unlock()
}

// bareEndpointServer 404s anything outside /v1 and counts both.
func bareEndpointServer(t *testing.T) (*httptest.Server, *atomic.Int64, *atomic.Int64) {
	t.Helper()
	var bare, versioned atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/") {
			versioned.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
			return
		}
		bare.Add(1)
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	return srv, &bare, &versioned
}

func TestBareEndpointIsProbedOncePerProcess(t *testing.T) {
	resetSharedEndpoints()
	srv, bare, _ := bareEndpointServer(t)
	spec := "dynamic/" + srv.URL + ",type=openai_vllm,model=m"

	// Ten posters, as ten replay instances would build.
	for range 10 {
		p, err := newReplayPoster(spec, llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
		if err != nil {
			t.Fatalf("newReplayPoster: %v", err)
		}
		attempts := p.endpointAttempts()
		// Simulate the first poster succeeding on the fallback.
		if len(attempts) == 2 {
			p.latchEndpoint(attempts[1])
		}
	}

	// Only the first poster may see the un-versioned form as a candidate.
	resolved := lookupResolvedEndpoint(srv.URL + "/chat/completions")
	if resolved == "" {
		t.Fatal("nothing was shared after a latch")
	}
	if !strings.Contains(resolved, "/v1/") {
		t.Errorf("shared endpoint %q is not the versioned form", resolved)
	}
	_ = bare
}

// TestLaterPostersSkipTheProbeEntirely is the property that matters: instance
// two onward must offer exactly one candidate.
func TestLaterPostersSkipTheProbeEntirely(t *testing.T) {
	resetSharedEndpoints()
	srv, _, _ := bareEndpointServer(t)
	spec := "dynamic/" + srv.URL + ",type=openai_vllm,model=m"
	mk := func() *replayPoster {
		p, err := newReplayPoster(spec, llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}

	first := mk()
	if got := first.endpointAttempts(); len(got) != 2 {
		t.Fatalf("first poster offered %d candidates, want 2 (it has nothing to go on)", len(got))
	}
	first.latchEndpoint(first.epFallback)

	second := mk()
	got := second.endpointAttempts()
	if len(got) != 1 {
		t.Errorf("a later poster offered %d candidates, want 1: it re-probes and pays a 404 that "+
			"the fleet never sees and the run counts as an error", len(got))
	}
	if len(got) == 1 && !strings.Contains(got[0], "/v1/") {
		t.Errorf("later poster resolved to %q", got[0])
	}
}

// TestRetryLoopDoesNotReProbe. endpointAttempts is called inside the 429 backoff
// loop, so a poster whose first request keeps being shed asked again each time
// round — which is why this only appeared once the fleet saturated.
func TestRetryLoopDoesNotReProbe(t *testing.T) {
	resetSharedEndpoints()
	srv, _, _ := bareEndpointServer(t)
	spec := "dynamic/" + srv.URL + ",type=openai_vllm,model=m"
	p, err := newReplayPoster(spec, llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	shareResolvedEndpoint(p.epPrimary, p.epFallback) // another poster got there first

	for i := range 5 { // five backoff iterations
		if got := p.endpointAttempts(); len(got) != 1 {
			t.Fatalf("iteration %d offered %d candidates, want 1", i, len(got))
		}
	}
}

func TestConcurrentResolutionAgrees(t *testing.T) {
	resetSharedEndpoints()
	srv, _, _ := bareEndpointServer(t)
	spec := "dynamic/" + srv.URL + ",type=openai_vllm,model=m"

	var wg sync.WaitGroup
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p, err := newReplayPoster(spec, llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
			if err != nil {
				return
			}
			if a := p.endpointAttempts(); len(a) == 2 {
				p.latchEndpoint(a[1])
			}
		}()
	}
	wg.Wait()

	if got := lookupResolvedEndpoint(srv.URL + "/chat/completions"); !strings.Contains(got, "/v1/") {
		t.Errorf("concurrent resolution settled on %q", got)
	}
}

// TestExplicitV1NeverProbes: the form seven arms used, and the reason none of
// them saw a single 404.
func TestExplicitV1NeverProbes(t *testing.T) {
	resetSharedEndpoints()
	srv, bare, _ := bareEndpointServer(t)
	spec := "dynamic/" + srv.URL + "/v1,type=openai_vllm,model=m"
	p, err := newReplayPoster(spec, llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, u := range p.endpointAttempts() {
		if !strings.Contains(u, "/v1/") {
			t.Errorf("candidate %q lacks /v1 despite an explicit base", u)
		}
	}
	if bare.Load() != 0 {
		t.Errorf("%d un-versioned requests against an explicit /v1 base", bare.Load())
	}
}
