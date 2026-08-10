// Package registry owns the set of backends and publishes immutable,
// deterministically-ordered snapshots of it.
//
// The ordering guarantee is not cosmetic. v1 returned worker lists straight out
// of a DashMap, in unspecified order that varied between calls, which alone
// degenerated round-robin into random and made every `index 0` fallback pin
// traffic to an arbitrary backend that changed per request (WRK-N1, LB-N3).
package registry

import (
	"fmt"
	"net"
	"net/url"
	"strings"
	"sync/atomic"

	"github.com/weka/wekai/router/internal/circuit"
)

// Kind distinguishes a leaf inference worker from a child router.
//
// Hierarchical routing is deferred (post-v2.0), but the field ships now as a
// no-op default so it stays additive rather than becoming a refactor (HIER-1).
type Kind uint8

const (
	KindWorker Kind = iota
	KindRouter
)

func (k Kind) String() string {
	if k == KindRouter {
		return "router"
	}
	return "worker"
}

// Provenance records who put a backend here. Discovery may only ever touch
// entries it owns, so a reconcile pass can never delete a statically-declared
// backend (HIER-19, WRK-7).
type Provenance uint8

const (
	ProvStatic Provenance = iota
	ProvDiscovered
)

func (p Provenance) String() string {
	if p == ProvDiscovered {
		return "discovered"
	}
	return "static"
}

// HealthModel is declared, never inferred. A backend with no health endpoint is
// a configuration (`passive`), not a special case (API-16).
type HealthModel uint8

const (
	HealthActive HealthModel = iota
	HealthPassive
)

// Health is the observed state. A new backend starts Unknown and becomes
// eligible only after its first successful check (HLT-5, SD-N3).
type Health int32

const (
	Unknown Health = iota
	Healthy
	Unhealthy
)

func (h Health) String() string {
	switch h {
	case Healthy:
		return "healthy"
	case Unhealthy:
		return "unhealthy"
	}
	return "unknown"
}

// Gauge is the minimal metrics surface a Backend needs. Satisfied by
// prometheus.Gauge. Kept as an interface so this package stays dependency-free
// and so tests need no metrics registry.
//
// Resolving the child gauge once, here, rather than calling WithLabelValues on
// the request path, is R5: that call takes a lock and a map lookup, which at
// target load would be ~40k resolutions/second for a label that never changes.
type Gauge interface {
	Inc()
	Dec()
}

type nopGauge struct{}

func (nopGauge) Inc() {}
func (nopGauge) Dec() {}

// Spec is the desired configuration of a backend. It is the input to Add and to
// discovery reconciliation; Backend is the live object.
type Spec struct {
	URL       string
	Kind      Kind
	DialectID string
	Health    HealthModel
	Prov      Provenance
	Model     string
	Locality  string
	// Capacity is the denominator of normalized load (HIER-5). For a leaf it
	// is max_inflight_per_worker; <1 is treated as 1.
	Capacity int64
}

// Backend is a routable destination.
//
// Membership is immutable once published in a Snapshot; the mutable per-backend
// state below is atomic, so a policy holding a snapshot sees a stable
// membership list with live counters — which is exactly what WRK-4 requires.
//
// Kind, Model and Locality are ALSO atomic, even though they look like static
// identity, because Add's update path (re-registration and discovery
// reconciliation) can rewrite them on a *Backend already shared across
// published snapshots and read concurrently on the request path (gateway
// filtering, admin serialization). URL, DialectID, HealthMod and Prov are
// genuinely set once at construction and never touched by Add's update path,
// so they stay plain fields.
type Backend struct {
	URL       string // canonical
	DialectID string
	HealthMod HealthModel
	Prov      Provenance

	kind     atomic.Int32
	model    atomic.Pointer[string]
	locality atomic.Pointer[string]

	CB            *circuit.Breaker
	InflightGauge Gauge

	capacity atomic.Int64
	inflight atomic.Int64
	draining atomic.Bool
	health   atomic.Int32

	Served, Failed atomic.Uint64
}

func (b *Backend) Kind() Kind     { return Kind(b.kind.Load()) }
func (b *Backend) SetKind(k Kind) { b.kind.Store(int32(k)) }

func (b *Backend) Model() string {
	if p := b.model.Load(); p != nil {
		return *p
	}
	return ""
}
func (b *Backend) SetModel(m string) { b.model.Store(&m) }

func (b *Backend) Locality() string {
	if p := b.locality.Load(); p != nil {
		return *p
	}
	return ""
}
func (b *Backend) SetLocality(l string) { b.locality.Store(&l) }

func (b *Backend) Inflight() int64 { return b.inflight.Load() }

func (b *Backend) Capacity() int64 {
	if c := b.capacity.Load(); c > 0 {
		return c
	}
	return 1
}

func (b *Backend) SetCapacity(c int64) { b.capacity.Store(c) }

// NormalizedLoad is what least-outstanding compares (HIER-5).
//
// Comparing raw in-flight counts across heterogeneous backends is HIER-N1: a
// child router fronting 40 GPUs with 40 requests in flight would look "more
// loaded" than an idle leaf at 1, starving an entire healthy subtree. Dividing
// by capacity makes the comparison meaningful. For a fleet of uniform leaves
// with capacity 1 this degenerates to the raw count, at no cost.
func (b *Backend) NormalizedLoad() float64 {
	return float64(b.inflight.Load()) / float64(b.Capacity())
}

func (b *Backend) Health() Health     { return Health(b.health.Load()) }
func (b *Backend) SetHealth(h Health) { b.health.Store(int32(h)) }
func (b *Backend) Draining() bool     { return b.draining.Load() }

// Available reports whether this backend may receive new traffic.
//
// Note it reads CB.State(), which is read-only, and never CB.Allow(), which
// mutates. Filtering must not consume a half-open probe token for a backend it
// does not go on to select (R2).
func (b *Backend) Available() bool {
	return b.Health() == Healthy && !b.Draining() && b.CB.State() != circuit.Open
}

// AddInflight is the in-flight counter mutator.
//
// It is exported only because Go has no friend packages: the sole legitimate
// caller is internal/lease. Nothing else — not health checks, not admin
// endpoints, not discovery, not flush_cache — may call it (LB-6). This is
// enforced mechanically by hack/lease_fence_test.go rather than by convention,
// because "one increment, three decrements, plus a health checker that zeroes
// every counter every ten cycles" is the exact defect that motivated the
// rewrite (LB-N1, LB-N2, HLT-N5).
func (b *Backend) AddInflight(d int64) int64 { return b.inflight.Add(d) }

// StoreInflight is likewise lease-only; it exists to clamp a detected
// underflow rather than let the counter wrap (LB-5).
func (b *Backend) StoreInflight(v int64) { b.inflight.Store(v) }

// Canonical normalizes a backend URL so that two inputs denoting the same
// backend compare equal (WRK-1): lowercase scheme and host, explicit port, no
// path, no trailing slash, IPv6 correctly bracketed (SD-9).
func Canonical(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", fmt.Errorf("empty backend URL")
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("parse %q: %w", raw, err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		// Rejecting file://, gopher:// and friends here is the SSRF guard for
		// the admin API (SEC-5).
		return "", fmt.Errorf("unsupported scheme %q (want http or https)", u.Scheme)
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return "", fmt.Errorf("missing host in %q", raw)
	}
	port := u.Port()
	if port == "" {
		if scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}
	// The PATH is part of the identity, not decoration. An endpoint can live
	// behind a base path — Google's OpenAI-compatible surface is
	// /v1beta/openai, and a vLLM behind an ingress prefix is the same shape —
	// and dropping it silently rewrites the backend to a different service on
	// the same host. Two paths on one host are two backends.
	path := strings.TrimRight(u.EscapedPath(), "/")
	return scheme + "://" + net.JoinHostPort(host, port) + path, nil
}
