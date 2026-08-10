package gateway

import (
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

// Config is what the HTTP surface itself needs — nothing about routing, which
// arrives through Router.
//
// It is declared here rather than shared with the rest of the router because
// this is the whole of it: seven knobs, all about the listener's own behaviour.
// The previous arrangement passed one 40-field config struct through every
// layer, so a package's real dependencies were invisible and the file+env+flag
// loader behind it had to exist before anything could be constructed at all.
type Config struct {
	// APIKey, when set, is required on every inference and admin request.
	// Empty means the listener is unauthenticated, which is logged loudly at
	// startup rather than assumed intentional.
	APIKey string

	// RequireAuthForProbes extends auth to /liveness and /readiness. Off by
	// default so a kubelet probe needs no credential; v1 required one, which is
	// why its probes were configured as exec.
	RequireAuthForProbes bool

	// MaxBodyBytes bounds a request body. The body is buffered whole so a retry
	// can replay it (REL-4), so this is the real memory bound per in-flight
	// request, not a formality.
	MaxBodyBytes int64

	// MaxConcurrentRequests bounds in-flight requests router-wide, protecting
	// the router's own memory. Distinct from any per-backend capacity signal:
	// this sheds 503 router_at_capacity, those shed 429. 0 disables.
	MaxConcurrentRequests int

	// PathAllowlist restricts which upstream paths may be proxied. Empty allows
	// the dialect's own routes only.
	PathAllowlist []string

	// CORSOrigins are the origins permitted to call the inference listener.
	// A wildcard combined with an API key is rejected at construction (SEC-10).
	CORSOrigins []string

	// DefaultCapacity is applied to a backend added at runtime through the
	// admin endpoints, which carry no capacity of their own.
	DefaultCapacity int64
}

// Redacted returns the config with nothing secret in it, for
// /get_server_info. The API key is the only secret here, and it is reported as
// a boolean rather than a length: a length is a hint (SEC-6, CFG-7).
func (c Config) Redacted() map[string]any {
	return map[string]any{
		"api_key_set":             c.APIKey != "",
		"require_auth_for_probes": c.RequireAuthForProbes,
		"max_body_bytes":          c.MaxBodyBytes,
		"max_concurrent_requests": c.MaxConcurrentRequests,
		"path_allowlist":          c.PathAllowlist,
		"cors_origins":            c.CORSOrigins,
	}
}

// Target is one routing destination: a set of interchangeable endpoints and the
// flow that chooses among them.
type Target struct {
	// Name identifies the pool, for logs, admin output and the `pool` metric
	// label.
	Name string
	// Registry holds the pool's endpoints.
	Registry *registry.Registry
	// Selector is the pool's routing flow.
	Selector proxy.Selector
	// RewriteModel, when set, replaces the request's model field before
	// forwarding — the `as <model>` half of a route rule, and what lets a
	// client's name for a model differ from the backend's.
	RewriteModel string
	// StripAuth drops inbound credentials before forwarding, for upstreams that
	// are unauthenticated and would reject or log them.
	StripAuth bool
	// Credential, when set, authenticates the ROUTER to this pool's upstreams
	// instead of forwarding the caller's own. It is what lets one router front
	// a hosted API the caller pays for alongside an internal fleet the router
	// holds the key to.
	Credential string
	// ForwardClientCredential passes the caller's own credential upstream,
	// which a hosted API the user pays for requires and an internal one must
	// never receive.
	ForwardClientCredential bool
}

// Router maps a request's model to the pool that serves it.
//
// The gateway deliberately knows nothing about HOW that mapping is configured —
// pattern matching, a catch-all, or a single implicit pool. It asks a question
// and gets a destination.
type Router interface {
	// Route returns the target for a model name. ok is false when nothing
	// matches, which the caller turns into a 404 naming the model rather than a
	// 503, because no amount of retrying will help.
	Route(model string) (Target, bool)
	// Targets returns every configured target, for admin and readiness.
	Targets() []Target
}
