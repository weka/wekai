// Package routing maps a request's model to the pool that serves it.
//
// This is the outer half of the router. The inner half — choosing among a
// pool's interchangeable endpoints — is the affinity flow; this decides which
// pool the question is even asked of.
//
// The two used to be different programs. `wekai router serve` matched models to
// single upstreams and had no notion of a fleet; `wllm-router` fronted one fleet
// and had no notion of a model. A Table is both: rules select a pool, the pool
// selects an endpoint.
package routing

import (
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/weka/wekai/router/internal/gateway"
	"github.com/weka/wekai/router/internal/pool"
)

// Rule is one routing rule: which model names it claims, and where they go.
type Rule struct {
	// Patterns are lowercased substrings of the model name. Empty means
	// catch-all — the shape a router has when it fronts one fleet and serves
	// every model from it.
	Patterns []string
	Pool     *pool.Pool
	// RewriteModel replaces the request's model before forwarding, so a
	// client's name for a model can differ from the backend's.
	RewriteModel string
	// AutoModel is the model discovered from the pool's own upstreams, filled
	// in asynchronously once a backend is healthy enough to answer.
	//
	// It is a live pointer rather than a value because discovery finishes after
	// the table is built: a pool's backends are commonly still loading weights
	// when the router starts. Reading it per request is what makes a late answer
	// take effect at all — the previous arrangement snapshotted the value at
	// construction, so a discovery that completed one second after startup
	// updated a field nothing ever read again.
	AutoModel *atomic.Pointer[string]
	// StripAuth drops inbound credentials, for an upstream that is
	// unauthenticated and would otherwise receive someone else's key.
	StripAuth bool
	// Credential authenticates the router to this pool's upstreams, replacing
	// whatever the caller sent.
	Credential string
	// ForwardClientCredential passes the caller's credential upstream.
	ForwardClientCredential bool
}

func (r Rule) matches(model string) bool {
	if len(r.Patterns) == 0 {
		return true
	}
	lower := strings.ToLower(model)
	for _, p := range r.Patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// String renders the rule the way it is logged at startup.
func (r Rule) String() string {
	pat := "*"
	if len(r.Patterns) > 0 {
		pat = strings.Join(r.Patterns, ",")
	}
	out := fmt.Sprintf("%s => pool %q (%d endpoints)", pat, r.Pool.Name,
		len(r.Pool.Registry.Snapshot().Backends))
	if r.RewriteModel != "" {
		out += " as " + r.RewriteModel
	} else if r.AutoModel != nil {
		if m := r.AutoModel.Load(); m != nil {
			out += " as " + *m + " (auto-discovered)"
		}
	}
	if r.StripAuth {
		out += " (strip-auth)"
	}
	return out
}

// Table resolves models to pools. First matching rule wins, which is what makes
// rule order meaningful and a catch-all last.
type Table struct {
	rules   []Rule
	targets []gateway.Target
}

// NewTable builds a table. It rejects an empty rule set rather than defaulting
// to something: a router with nowhere to send traffic is a configuration
// mistake, not a mode.
func NewTable(rules []Rule) (*Table, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("routing: no rules configured")
	}
	seen := map[string]bool{}
	t := &Table{rules: rules}
	for _, r := range rules {
		if r.Pool == nil {
			return nil, fmt.Errorf("routing: rule %q has no pool", strings.Join(r.Patterns, ","))
		}
		if seen[r.Pool.Name] {
			continue // several rules may share one pool; report it once
		}
		seen[r.Pool.Name] = true
		t.targets = append(t.targets, r.target())
	}
	return t, nil
}

func (r Rule) target() gateway.Target {
	// An explicit `as <model>` always wins: the operator said it, and discovery
	// does not second-guess it.
	rewrite := r.RewriteModel
	if rewrite == "" && r.AutoModel != nil {
		if m := r.AutoModel.Load(); m != nil {
			rewrite = *m
		}
	}
	return gateway.Target{
		Name:                    r.Pool.Name,
		Registry:                r.Pool.Registry,
		Selector:                r.Pool.Flow,
		RewriteModel:            rewrite,
		StripAuth:               r.StripAuth,
		Credential:              r.Credential,
		ForwardClientCredential: r.ForwardClientCredential,
	}
}

// Route implements gateway.Router.
func (t *Table) Route(model string) (gateway.Target, bool) {
	for _, r := range t.rules {
		if r.matches(model) {
			return r.target(), true
		}
	}
	return gateway.Target{}, false
}

// Targets implements gateway.Router, returning each distinct pool once.
func (t *Table) Targets() []gateway.Target { return t.targets }

// Rules exposes the configured rules for startup logging.
func (t *Table) Rules() []Rule { return t.rules }

// NormalizePatterns lowercases and trims a comma-separated pattern list,
// mapping "*" to the empty (catch-all) set.
func NormalizePatterns(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "*" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(raw, ",") {
		if p = strings.ToLower(strings.TrimSpace(p)); p != "" {
			out = append(out, p)
		}
	}
	return out
}
