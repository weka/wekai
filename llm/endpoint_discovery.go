package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// discoveryHTTPClient is used only for endpoint autodiscovery probes. Short
// timeout: a live LLM server should answer /v1/models almost instantly, and
// we don't want an unreachable/firewalled endpoint to stall spec resolution
// for the whole benchmark run (resolveDynamicEndpoint is on the request hot
// path — see its doc comment).
var discoveryHTTPClient = &http.Client{Timeout: 5 * time.Second}

// DiscoveryProbeCount counts actual HTTP discovery requests fired by
// resolveDynamicEndpoint (both the /v1-path probe and any follow-up model
// list fetch). It exists so tests can prove memoization: for N calls against
// the same endpoint this must land on a small constant, not grow with N.
var DiscoveryProbeCount atomic.Int64

// shouldProbeForV1 reports whether base — a normalized endpoint URL with a
// trailing slash, e.g. "http://host:8000/" — is a bare host:port with no
// path, and therefore a candidate for /v1 autodiscovery. Endpoints that
// already carry an explicit path, including one that already ends in
// "/v1/", are left exactly as the user wrote them.
func shouldProbeForV1(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	return strings.Trim(u.Path, "/") == ""
}

// modelsListResponse is the OpenAI/vLLM-compatible shape returned by a
// "/models" endpoint: {"data":[{"id":"..."},...]}.
type modelsListResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// probeModelsEndpoint GETs base+"models" (base must already end in "/") with
// a short timeout. ok=true only on a 200 response with a non-empty "data"
// array; any other outcome (network error, non-200, unparseable body, empty
// list) returns ok=false with no error — callers treat that as "endpoint
// doesn't support discovery here" and fall back, not as fatal.
func probeModelsEndpoint(base string) (modelID string, ok bool) {
	n := DiscoveryProbeCount.Add(1)
	modelsURL := base + "models"
	fmt.Fprintf(os.Stderr, "Endpoint autodiscovery: probe #%d GET %s\n", n, modelsURL)
	resp, err := discoveryHTTPClient.Get(modelsURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Endpoint autodiscovery: probe #%d %s failed: %v\n", n, modelsURL, err)
		return "", false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "Endpoint autodiscovery: probe #%d %s returned status %d\n", n, modelsURL, resp.StatusCode)
		return "", false
	}
	var parsed modelsListResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		fmt.Fprintf(os.Stderr, "Endpoint autodiscovery: probe #%d %s returned unparseable body: %v\n", n, modelsURL, err)
		return "", false
	}
	if len(parsed.Data) == 0 || parsed.Data[0].ID == "" {
		fmt.Fprintf(os.Stderr, "Endpoint autodiscovery: probe #%d %s returned no models\n", n, modelsURL)
		return "", false
	}
	fmt.Fprintf(os.Stderr, "Endpoint autodiscovery: probe #%d %s -> model %q\n", n, modelsURL, parsed.Data[0].ID)
	return parsed.Data[0].ID, true
}

// resolvedEndpoint is the outcome of autodiscovery for one raw base URL.
type resolvedEndpoint struct {
	base  string // effective base URL to use for requests (with /v1/ appended if discovered)
	model string // discovered model id; "" if not found/not attempted
}

// endpointDiscoveryEntry gates one rawBase's discovery behind a sync.Once so
// concurrent first-callers block and share a single result instead of each
// firing their own probe.
type endpointDiscoveryEntry struct {
	once sync.Once
	val  resolvedEndpoint
}

var (
	endpointDiscoveryMu      sync.Mutex
	endpointDiscoveryEntries = map[string]*endpointDiscoveryEntry{}
)

// resolveDynamicEndpoint resolves the effective base URL (autodiscovering a
// /v1 path segment for bare host:port endpoints) and, when wantModel is
// true, the first model id the endpoint advertises.
//
// This sits on the benchmark request hot path: GetChatGetter — and thus
// ParseDynamicModel/resolveDynamicEndpoint — is called fresh per request in
// several benchmark modes (see benchmark/embed.go, throughput.go), so a
// single benchmark run can call this thousands of times for the very same
// endpoint. The sync.Once per rawBase key guarantees the underlying network
// probe(s) fire exactly once per process per distinct endpoint, no matter
// how many times or how concurrently this is called — subsequent (and
// concurrently-blocked) callers just read the memoized result.
func resolveDynamicEndpoint(rawBase string, wantModel bool) resolvedEndpoint {
	endpointDiscoveryMu.Lock()
	entry, ok := endpointDiscoveryEntries[rawBase]
	if !ok {
		entry = &endpointDiscoveryEntry{}
		endpointDiscoveryEntries[rawBase] = entry
	}
	endpointDiscoveryMu.Unlock()

	entry.once.Do(func() {
		base := rawBase
		var model string

		if shouldProbeForV1(rawBase) {
			v1Base := strings.TrimRight(rawBase, "/") + "/v1/"
			if id, ok := probeModelsEndpoint(v1Base); ok {
				base = v1Base
				model = id
			}
			// else: endpoint didn't answer /v1/models — leave base as
			// rawBase; fall through below to try rawBase's own "models"
			// path directly (covers non-OpenAI-shaped servers that still
			// expose "models" at the root).
		}

		if wantModel && model == "" {
			if id, ok := probeModelsEndpoint(base); ok {
				model = id
			}
		}

		entry.val = resolvedEndpoint{base: base, model: model}
	})
	return entry.val
}

// resetEndpointDiscoveryCacheForTest clears memoized discovery state. Test-only.
func resetEndpointDiscoveryCacheForTest() {
	endpointDiscoveryMu.Lock()
	defer endpointDiscoveryMu.Unlock()
	endpointDiscoveryEntries = map[string]*endpointDiscoveryEntry{}
	DiscoveryProbeCount.Store(0)
}

// errModelAutodiscoveryFailed is returned (as a panic value, matching the
// panic-on-config-error style already used throughout getDynamicChatGetter)
// when a dynamic model spec has no model= and autodiscovery could not find
// one for a type that has no other way to guess it.
func errModelAutodiscoveryFailed(dynType, base string) error {
	return fmt.Errorf("model autodiscovery failed for %s endpoint %s: /v1/models (or /models) did not return a usable model list; specify model=<id> explicitly in the dynamic model spec", dynType, base)
}
