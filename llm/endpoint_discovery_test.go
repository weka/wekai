package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// --- Pure-function tests: no network, no mocked LLM flow. shouldProbeForV1
// is an HTTP URL-shape utility, not part of the LLM chat flow — table tests
// on it are exactly the kind of pure data-transformation test the project's
// "no mock flows" policy allows.

func TestShouldProbeForV1(t *testing.T) {
	cases := []struct {
		name string
		base string
		want bool
	}{
		{"bare host:port with trailing slash", "http://localhost:8000/", true},
		{"bare https host with trailing slash", "https://10.71.0.4:8000/", true},
		{"already has /v1", "http://localhost:8000/v1/", false},
		{"already has /v1 no trailing slash normalization needed", "http://localhost:8000/v1", false},
		{"explicit custom path", "http://localhost:8000/custom/path/", false},
		{"explicit custom path no trailing slash", "http://localhost:8000/custom", false},
		{"root path only, no trailing slash", "http://localhost:8000", true},
		{"malformed URL", "http://[::1", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := shouldProbeForV1(c.base)
			if got != c.want {
				t.Errorf("shouldProbeForV1(%q) = %v, want %v", c.base, got, c.want)
			}
		})
	}
}

// --- HTTP utility tests for probeModelsEndpoint. An httptest server serving
// /v1/models is a plain HTTP fixture for this function, not a mocked LLM
// provider — probeModelsEndpoint never touches the Chat/Manager interfaces.

func newModelsServer(t *testing.T, id string) *httptest.Server {
	t.Helper()
	var hits int
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(modelsListResponse{
			Data: []struct {
				ID string `json:"id"`
			}{{ID: id}},
		})
	})
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r) // mirrors real vLLM: bare /models 404s
	})
	return httptest.NewServer(mux)
}

func TestProbeModelsEndpoint_Success(t *testing.T) {
	srv := newModelsServer(t, "test-model-1")
	defer srv.Close()

	id, ok := probeModelsEndpoint(srv.URL + "/v1/")
	if !ok {
		t.Fatalf("expected ok=true")
	}
	if id != "test-model-1" {
		t.Errorf("model id = %q, want %q", id, "test-model-1")
	}
}

func TestProbeModelsEndpoint_NotFound(t *testing.T) {
	srv := newModelsServer(t, "test-model-1")
	defer srv.Close()

	// Bare (no /v1) base — hits the 404 handler.
	_, ok := probeModelsEndpoint(srv.URL + "/")
	if ok {
		t.Fatalf("expected ok=false for 404 endpoint")
	}
}

// --- resolveDynamicEndpoint: proves the single-probe-per-endpoint contract.

func TestResolveDynamicEndpoint_BareURLDiscoversV1AndModel(t *testing.T) {
	resetEndpointDiscoveryCacheForTest()
	srv := newModelsServer(t, "auto-discovered-model")
	defer srv.Close()

	res := resolveDynamicEndpoint(srv.URL+"/", true)
	if res.base != srv.URL+"/v1/" {
		t.Errorf("base = %q, want %q", res.base, srv.URL+"/v1/")
	}
	if res.model != "auto-discovered-model" {
		t.Errorf("model = %q, want %q", res.model, "auto-discovered-model")
	}
}

func TestResolveDynamicEndpoint_ExplicitV1PathSkipsPathProbe(t *testing.T) {
	resetEndpointDiscoveryCacheForTest()
	srv := newModelsServer(t, "explicit-v1-model")
	defer srv.Close()

	res := resolveDynamicEndpoint(srv.URL+"/v1/", true)
	if res.base != srv.URL+"/v1/" {
		t.Errorf("base = %q, want unchanged %q", res.base, srv.URL+"/v1/")
	}
	if res.model != "explicit-v1-model" {
		t.Errorf("model = %q, want %q", res.model, "explicit-v1-model")
	}
	// Exactly one probe: shouldProbeForV1 is false (path already "/v1/"), so
	// only the model-discovery probe against the given base fires.
	if got := DiscoveryProbeCount.Load(); got != 1 {
		t.Errorf("probe count = %d, want 1", got)
	}
}

func TestResolveDynamicEndpoint_ExplicitCustomPathLeftAlone(t *testing.T) {
	resetEndpointDiscoveryCacheForTest()

	res := resolveDynamicEndpoint("http://example.invalid/custom/", false)
	if res.base != "http://example.invalid/custom/" {
		t.Errorf("base = %q, want unchanged", res.base)
	}
	if res.model != "" {
		t.Errorf("model = %q, want empty (wantModel=false)", res.model)
	}
	if got := DiscoveryProbeCount.Load(); got != 0 {
		t.Errorf("probe count = %d, want 0 (no probe expected: custom path + wantModel=false)", got)
	}
}

// TestResolveDynamicEndpoint_MemoizedOncePerEndpoint is the load-bearing
// test for the "discovery happens ONCE per resolved spec, NOT per request"
// requirement: GetChatGetter (and therefore resolveDynamicEndpoint) is
// called fresh per request on several benchmark code paths, so this proves
// the network probe fires exactly once per distinct endpoint even when
// resolveDynamicEndpoint is hammered concurrently, simulating a
// concurrency>1 benchmark run against the same target.
func TestResolveDynamicEndpoint_MemoizedOncePerEndpoint(t *testing.T) {
	resetEndpointDiscoveryCacheForTest()
	srv := newModelsServer(t, "concurrent-model")
	defer srv.Close()

	const callers = 200
	var wg sync.WaitGroup
	results := make([]resolvedEndpoint, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = resolveDynamicEndpoint(srv.URL+"/", true)
		}(i)
	}
	wg.Wait()

	for i, res := range results {
		if res.base != srv.URL+"/v1/" || res.model != "concurrent-model" {
			t.Fatalf("caller %d got %+v, want base=%s/v1/ model=concurrent-model", i, res, srv.URL)
		}
	}

	// One endpoint, bare URL, wantModel=true, no error → exactly one HTTP
	// probe (/v1/models) total, regardless of 200 concurrent callers.
	if got := DiscoveryProbeCount.Load(); got != 1 {
		t.Errorf("probe count = %d, want exactly 1 for %d concurrent callers against one endpoint", got, callers)
	}
}

// TestResolveDynamicEndpoint_MemoizedAcrossManySequentialCalls simulates the
// realistic shape of the bug this guards against: a benchmark issuing many
// sequential requests (each re-resolving the same spec via GetChatGetter),
// not just concurrent ones.
func TestResolveDynamicEndpoint_MemoizedAcrossManySequentialCalls(t *testing.T) {
	resetEndpointDiscoveryCacheForTest()
	srv := newModelsServer(t, "sequential-model")
	defer srv.Close()

	for i := 0; i < 500; i++ {
		res := resolveDynamicEndpoint(srv.URL+"/", true)
		if res.model != "sequential-model" {
			t.Fatalf("call %d: model = %q, want %q", i, res.model, "sequential-model")
		}
	}
	if got := DiscoveryProbeCount.Load(); got != 1 {
		t.Errorf("probe count = %d, want exactly 1 after 500 sequential resolutions", got)
	}
}

func TestResolveDynamicEndpoint_DistinctEndpointsProbedIndependently(t *testing.T) {
	resetEndpointDiscoveryCacheForTest()
	srv1 := newModelsServer(t, "model-a")
	defer srv1.Close()
	srv2 := newModelsServer(t, "model-b")
	defer srv2.Close()

	res1 := resolveDynamicEndpoint(srv1.URL+"/", true)
	res2 := resolveDynamicEndpoint(srv2.URL+"/", true)

	if res1.model != "model-a" || res2.model != "model-b" {
		t.Fatalf("got model-a=%q model-b=%q", res1.model, res2.model)
	}
	// Two distinct endpoints → two probes total (one each), still not one
	// per call — repeat both to confirm the count doesn't grow further.
	resolveDynamicEndpoint(srv1.URL+"/", true)
	resolveDynamicEndpoint(srv2.URL+"/", true)
	if got := DiscoveryProbeCount.Load(); got != 2 {
		t.Errorf("probe count = %d, want 2 (one per distinct endpoint)", got)
	}
}
