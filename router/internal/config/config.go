// Package config loads and validates the router's configuration.
//
// Exactly one Config struct, one set of defaults, and one precedence order:
// flag > env > file > default (CFG-1, CFG-3). v1 had three divergent default
// sets across the Rust CLI, the PyO3 binding and the Python CLI — the default
// policy was cache_aware in one and RoundRobin in another — and accepted knobs
// it silently discarded (CFG-N3, CFG-N4).
package config

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type Backend struct {
	URL      string `json:"url"`
	Kind     string `json:"kind,omitempty"`    // worker | router
	Dialect  string `json:"dialect,omitempty"` // openai
	Health   string `json:"health,omitempty"`  // active | passive
	Model    string `json:"model,omitempty"`
	Locality string `json:"locality,omitempty"`
	Capacity int64  `json:"capacity,omitempty"`
}

type Config struct {
	Listen        string    `json:"listen"`
	MetricsListen string    `json:"metrics_listen"`
	MaxBodyBytes  int64     `json:"max_body_bytes"`
	Backends      []Backend `json:"backends"`

	// APIKey is intentionally NOT settable by flag: a flag value is visible in
	// `ps` output to every user on the host (CFG-8). Env or file only.
	APIKey     string `json:"-"`
	APIKeyFile string `json:"api_key_file"`
	// UpstreamCredential is sent to backends; distinct from APIKey, which
	// authenticates clients TO us and is never forwarded (AUTH-9).
	UpstreamCredential string `json:"-"`

	// PathAllowlist gates which paths are served at all, and is orthogonal to auth:
	// a listed path still requires a credential, and an unlisted path 404s even for
	// a caller holding a valid key. Set via FORWARD_PATH_ALLOWLIST (v1's name).
	//
	// An empty list serves every path, with auth still applied to every path — the
	// same semantics as v1. Note this does NOT implement AUTH-8, which asked for
	// empty to mean deny-by-default; see the AUTH-8 row in docs/rewrite for why
	// that was not adopted.
	PathAllowlist []string `json:"path_allowlist"`
	CORSOrigins   []string `json:"cors_origins"`

	// RequireAuthForProbes makes /readiness and /health require a credential.
	//
	// Default false, and that default is load-bearing for Kubernetes. A kubelet
	// httpGet probe cannot authenticate without embedding the secret in the pod
	// spec in plaintext, so requiring auth here makes the pod permanently
	// NotReady — which is precisely why v1's README told operators to replace
	// httpGet with an exec probe, an instruction that cannot be followed on a
	// distroless image with no shell. When probes are unauthenticated they
	// disclose only a boolean; the backend counts require a credential (AUTH-6).
	RequireAuthForProbes bool `json:"require_auth_for_probes"`

	Cache CacheConfig `json:"cache"`

	// MaxConcurrentRequests caps in-flight requests; 0 disables the cap. Each
	// in-flight request can hold up to MaxBodyBytes buffered for retry, so this is
	// what actually bounds memory.
	MaxConcurrentRequests int   `json:"max_concurrent_requests"`
	MaxAttempts           int   `json:"max_attempts"`
	StreamBufferBytes     int   `json:"stream_buffer_bytes"`
	MaxInflightPerBackend int64 `json:"max_inflight_per_backend"`

	// MaxNodeConcurrency approximates, AT THE ROUTER, the per-backend
	// concurrency level at which vLLM itself would 429 — so a lower ceiling
	// can be tested (e.g. 32 against a real fleet running
	// WEKA_MAX_CONCURRENT_REQUESTS=48) without restarting any backend. 0
	// disables this (today's behavior): a backend whose router-side in-flight
	// lease count is >= this value is excluded from candidate selection for
	// every policy alike, and if every healthy backend is at cap the router
	// itself returns 429 rather than 503 (distinguishable from "no healthy
	// backends" and from the existing router-wide MaxConcurrentRequests shed).
	// Single-router deployment is assumed: the router's own lease count is
	// authoritative only when it is the sole source of a backend's load.
	MaxNodeConcurrency int64 `json:"max_node_concurrency"`

	// RebalanceRatio enables the imbalance split signal by being set. A backend
	// is treated as unable to take more work while
	// (inflight - fleetMin) / inflight exceeds it, so 0.5 means "rebalance once
	// the gap is more than half the busier side". 0 (the default) leaves the
	// signal off, because a fleet where prefix affinity is working is SUPPOSED
	// to look imbalanced — turning this on trades locality for evenness.
	//
	// It replaces the retired prefix-cache-aware policy's paired
	// balance_abs_threshold / balance_rel_threshold, which needed retuning per
	// deployment because one half was an absolute request count.
	RebalanceRatio float64 `json:"rebalance_ratio"`

	HealthInterval Duration `json:"health_interval"`
	HealthTimeout  Duration `json:"health_timeout"`
	HealthPath     string   `json:"health_path"`
	DrainDeadline  Duration `json:"drain_deadline"`
	RequestTimeout Duration `json:"request_timeout"`
	IdleTimeout    Duration `json:"idle_timeout"`

	LogLevel  string `json:"log_level"`
	LogFormat string `json:"log_format"`

	Discovery Discovery `json:"discovery"`
}

// CacheConfig tunes prefix-cache-aware routing.
//
// Residency is PREDICTED, not observed: vLLM exposes no residency query, and
// neither its KV event stream nor LMCache's /lookup is enabled here. The
// prediction's accuracy is measured against the worker's reported cached_tokens
// rather than assumed (RES-3).
type CacheConfig struct {
	// SplitGuard keeps a split from landing on a backend nearly as loaded as
	// the saturated holders it is relieving, which is what stops every backend
	// from ending up marked as holding every prefix. A candidate qualifies
	// while its in-flight is below MaxNodeConcurrency * (1 - SplitGuard).
	SplitGuard float64 `json:"split_guard"`
	// TailTTL is how long a leaf of the shared prefix tree may go untouched
	// before eviction. Eviction is tail-only: the middle is never removed.
	TailTTL Duration `json:"tail_ttl"`
	// RefusalTTL is how long a backend's own 429 keeps it out of its prefixes.
	// A hint that saves the next request a wasted round trip, not a health
	// verdict — a success from that backend clears it immediately.
	RefusalTTL Duration `json:"refusal_ttl"`
}

// Discovery configures Kubernetes-based backend discovery. Disabled by default:
// the router must run correctly from a static list alone (SD-10).
type Discovery struct {
	Enabled bool `json:"enabled"`
	// Mode is "endpointslice" (recommended) or "pod" (v1 parity).
	Mode      string `json:"mode"`
	Namespace string `json:"namespace"`
	Service   string `json:"service"`
	Selector  string `json:"selector"`
	Port      int    `json:"port"`
	PortName  string `json:"port_name"`
	Scheme    string `json:"scheme"`
	// Kubeconfig is a local-development fallback; in-cluster config is preferred.
	Kubeconfig     string   `json:"kubeconfig"`
	ResyncInterval Duration `json:"resync_interval"`
}

func Default() Config {
	return Config{
		Listen:                ":8080",
		MetricsListen:         "127.0.0.1:29000",
		MaxBodyBytes:          64 << 20,
		MaxConcurrentRequests: 256,
		MaxAttempts:           2,
		StreamBufferBytes:     64 << 10,
		MaxInflightPerBackend: 1,
		HealthInterval:        Duration(10 * time.Second),
		HealthTimeout:         Duration(5 * time.Second),
		HealthPath:            "/health",
		DrainDeadline:         Duration(60 * time.Second),
		RequestTimeout:        Duration(600 * time.Second),
		IdleTimeout:           Duration(300 * time.Second),
		LogLevel:              "info",
		LogFormat:             "json",
		Cache: CacheConfig{
			SplitGuard: 0.20,
			TailTTL:    Duration(5 * time.Minute),
			RefusalTTL: Duration(2 * time.Second),
		},
		Discovery: Discovery{
			Mode:           "endpointslice",
			Scheme:         "http",
			ResyncInterval: Duration(5 * time.Minute),
		},
	}
}

// Load applies defaults, then the config file, then env, then flags.
func Load(args []string, getenv func(string) string) (Config, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	cfg := Default()

	// A pre-pass finds --config without consuming the real flag set, so the file
	// can be layered underneath flags.
	var configPath string
	pre := flag.NewFlagSet("pre", flag.ContinueOnError)
	pre.SetOutput(devNull{})
	pre.StringVar(&configPath, "config", getenv("WLLM_CONFIG"), "")
	_ = pre.Parse(args)

	if configPath != "" {
		raw, err := os.ReadFile(configPath)
		if err != nil {
			return cfg, fmt.Errorf("read config %q: %w", configPath, err)
		}
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		// An unknown key is a hard error, not a warning (CFG-6).
		dec.DisallowUnknownFields()
		if err := dec.Decode(&cfg); err != nil {
			return cfg, fmt.Errorf("parse config %q: %w", configPath, err)
		}
	}

	applyEnv(&cfg, getenv)

	fs := flag.NewFlagSet("wllm-router", flag.ContinueOnError)
	fs.StringVar(&configPath, "config", configPath, "path to a JSON config file")
	fs.StringVar(&cfg.Listen, "listen", cfg.Listen, "inference listener address")
	fs.StringVar(&cfg.MetricsListen, "metrics-listen", cfg.MetricsListen, "metrics listener address")
	fs.Int64Var(&cfg.MaxBodyBytes, "max-body-bytes", cfg.MaxBodyBytes, "maximum request body size")
	fs.IntVar(&cfg.MaxConcurrentRequests, "max-concurrent-requests", cfg.MaxConcurrentRequests, "in-flight request cap (0 disables)")
	fs.IntVar(&cfg.MaxAttempts, "max-attempts", cfg.MaxAttempts, "total upstream attempts including the first")
	fs.Int64Var(&cfg.MaxNodeConcurrency, "max-node-concurrency", cfg.MaxNodeConcurrency,
		"enables the concurrency split signal (0 = off): the router's own guess at the backends' vLLM "+
			"--max-num-seqs. A backend at or above this many in-flight requests is treated as saturated "+
			"without waiting for it to say so, which saves a wasted round trip per saturation event. The "+
			"backend's own 429 remains the ultimate signal and backstops this one, which is why it is optional")
	fs.Float64Var(&cfg.RebalanceRatio, "rebalance-ratio", cfg.RebalanceRatio,
		"enables the imbalance split signal (0 = off): a backend is treated as saturated while "+
			"(inflight - fleetMin) / inflight exceeds this. 0.5 means rebalance once the gap is more than "+
			"half the busier side. Off by default: a fleet where affinity is working is supposed to look "+
			"imbalanced, so this trades locality for evenness")
	fs.Var(&cfg.Cache.RefusalTTL, "cache-refusal-ttl",
		"how long a backend's own 429 keeps it out of its prefixes; cleared early by any success from it")
	fs.Float64Var(&cfg.Cache.SplitGuard, "cache-split-guard", cfg.Cache.SplitGuard,
		"prefix-cache-split only: a saturated prefix is split onto a backend whose in-flight is below "+
			"max-node-concurrency * (1 - this). Higher values keep the holder set tighter at the cost of "+
			"splitting less readily")
	fs.Var(&cfg.Cache.TailTTL, "cache-tail-ttl",
		"prefix-cache-split only: how long a leaf of the shared prefix tree may go untouched before "+
			"eviction, e.g. 5m. Eviction is tail-only; the middle of the tree is never removed")
	fs.Var(&cfg.HealthInterval, "health-interval", "health check interval, e.g. 10s")
	fs.Var(&cfg.HealthTimeout, "health-timeout", "per-check timeout, e.g. 5s")
	fs.StringVar(&cfg.HealthPath, "health-path", cfg.HealthPath, "backend health endpoint path")
	fs.Var(&cfg.DrainDeadline, "drain-deadline", "graceful drain deadline, e.g. 60s")
	fs.StringVar(&cfg.LogLevel, "log-level", cfg.LogLevel, "debug | info | warn | error")
	fs.StringVar(&cfg.LogFormat, "log-format", cfg.LogFormat, "json | text")
	fs.StringVar(&cfg.APIKeyFile, "api-key-file", cfg.APIKeyFile, "file containing the inbound API key")
	fs.BoolVar(&cfg.Discovery.Enabled, "discovery", cfg.Discovery.Enabled, "enable Kubernetes backend discovery")
	fs.StringVar(&cfg.Discovery.Mode, "discovery-mode", cfg.Discovery.Mode, "endpointslice | pod")
	fs.StringVar(&cfg.Discovery.Namespace, "discovery-namespace", cfg.Discovery.Namespace, "namespace to watch")
	fs.StringVar(&cfg.Discovery.Service, "discovery-service", cfg.Discovery.Service, "Service name (endpointslice mode)")
	fs.StringVar(&cfg.Discovery.Selector, "discovery-selector", cfg.Discovery.Selector, "pod label selector (pod mode)")
	fs.IntVar(&cfg.Discovery.Port, "discovery-port", cfg.Discovery.Port, "backend port")
	fs.StringVar(&cfg.Discovery.PortName, "discovery-port-name", cfg.Discovery.PortName, "named port on the EndpointSlice")
	fs.StringVar(&cfg.Discovery.Kubeconfig, "kubeconfig", cfg.Discovery.Kubeconfig, "kubeconfig path for local development")
	var backends, allowlist, origins string
	fs.StringVar(&backends, "backends", "", "comma-separated backend URLs")
	fs.StringVar(&allowlist, "path-allowlist", "", "comma-separated allowed paths")
	fs.StringVar(&origins, "cors-origins", "", "comma-separated allowed CORS origins")
	// Note: no -api-key flag exists. That is CFG-8 enforced by construction
	// rather than by a check that could be forgotten.

	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	if backends != "" {
		cfg.Backends = nil
		for _, u := range splitList(backends) {
			cfg.Backends = append(cfg.Backends, Backend{URL: u})
		}
	}
	if allowlist != "" {
		cfg.PathAllowlist = splitList(allowlist)
	}
	if origins != "" {
		cfg.CORSOrigins = splitList(origins)
	}

	if cfg.APIKeyFile != "" {
		b, err := os.ReadFile(cfg.APIKeyFile)
		if err != nil {
			return cfg, fmt.Errorf("read api key file: %w", err)
		}
		cfg.APIKey = strings.TrimSpace(string(b))
	}
	return cfg, cfg.Validate()
}

func applyEnv(cfg *Config, getenv func(string) string) {
	set := func(dst *string, key string) {
		if v := strings.TrimSpace(getenv(key)); v != "" {
			*dst = v
		}
	}
	set(&cfg.Listen, "WLLM_LISTEN")
	set(&cfg.MetricsListen, "WLLM_METRICS_LISTEN")
	set(&cfg.APIKey, "WLLM_API_KEY")
	set(&cfg.APIKeyFile, "WLLM_API_KEY_FILE")
	set(&cfg.UpstreamCredential, "WLLM_UPSTREAM_CREDENTIAL")
	set(&cfg.LogLevel, "WLLM_LOG_LEVEL")
	set(&cfg.LogFormat, "WLLM_LOG_FORMAT")
	set(&cfg.HealthPath, "WLLM_HEALTH_PATH")
	set(&cfg.Discovery.Mode, "WLLM_DISCOVERY_MODE")
	set(&cfg.Discovery.Namespace, "WLLM_DISCOVERY_NAMESPACE")
	set(&cfg.Discovery.Service, "WLLM_DISCOVERY_SERVICE")
	set(&cfg.Discovery.Selector, "WLLM_DISCOVERY_SELECTOR")
	set(&cfg.Discovery.PortName, "WLLM_DISCOVERY_PORT_NAME")
	set(&cfg.Discovery.Scheme, "WLLM_DISCOVERY_SCHEME")
	if v := strings.TrimSpace(getenv("WLLM_DISCOVERY_ENABLED")); v != "" {
		cfg.Discovery.Enabled = v == "1" || strings.EqualFold(v, "true")
	}
	if v := strings.TrimSpace(getenv("WLLM_DISCOVERY_PORT")); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Discovery.Port = n
		}
	}
	if v := strings.TrimSpace(getenv("WLLM_BACKENDS")); v != "" {
		cfg.Backends = nil
		for _, u := range splitList(v) {
			cfg.Backends = append(cfg.Backends, Backend{URL: u})
		}
	}
	// FORWARD_PATH_ALLOWLIST, not WLLM_PATH_ALLOWLIST: this deliberately breaks the
	// WLLM_ prefix to keep v1's name. The migration is "swap the image, keep the
	// env", and under a renamed variable the allowlist would simply be off — every
	// path served instead of the listed few, with nothing in the logs to say so.
	// Auth still applies, so it is a wider surface rather than an open router, but
	// silence is the problem. Matching v1 costs a prefix; not matching costs a
	// misconfiguration nobody sees.
	if v := strings.TrimSpace(getenv("FORWARD_PATH_ALLOWLIST")); v != "" {
		cfg.PathAllowlist = splitList(v)
	}
	if v := strings.TrimSpace(getenv("WLLM_CORS_ORIGINS")); v != "" {
		cfg.CORSOrigins = splitList(v)
	}
}

// Validate aggregates every problem into one report rather than failing on the
// first, so a misconfigured deployment needs one round trip to fix (CFG-5).
func (c Config) Validate() error {
	var errs []error

	if c.MaxBodyBytes <= 0 {
		errs = append(errs, errors.New("max_body_bytes must be > 0: an unbounded body limit is a memory-exhaustion DoS"))
	}
	// v1's health checker could fall behind indefinitely because nothing
	// required the timeout to be shorter than the interval (HLT-2).
	if c.HealthTimeout >= c.HealthInterval {
		errs = append(errs, fmt.Errorf("health_timeout (%s) must be < health_interval (%s)",
			c.HealthTimeout, c.HealthInterval))
	}
	if c.HealthInterval <= 0 {
		errs = append(errs, errors.New("health_interval must be > 0"))
	}
	if c.MaxAttempts < 1 {
		errs = append(errs, errors.New("max_attempts must be >= 1"))
	}
	if c.MaxNodeConcurrency < 0 {
		errs = append(errs, errors.New("max_node_concurrency must be >= 0 (0 disables the cap)"))
	}
	if c.StreamBufferBytes <= 0 {
		errs = append(errs, errors.New("stream_buffer_bytes must be > 0"))
	}
	if !strings.HasPrefix(c.HealthPath, "/") {
		errs = append(errs, fmt.Errorf("health_path %q must start with /", c.HealthPath))
	}
	for _, p := range c.PathAllowlist {
		if !strings.HasPrefix(p, "/") {
			// v1 accepted these and they silently matched nothing, locking out
			// the whole service.
			errs = append(errs, fmt.Errorf("path_allowlist entry %q must start with /", p))
		}
	}
	for _, o := range c.CORSOrigins {
		if o == "*" && c.APIKey != "" {
			errs = append(errs, errors.New(`cors_origins "*" cannot be combined with an API key (SEC-10)`))
		}
	}
	if c.RebalanceRatio < 0 || c.RebalanceRatio >= 1 {
		errs = append(errs, fmt.Errorf("rebalance_ratio %v must be in [0,1); 0 disables the imbalance signal", c.RebalanceRatio))
	}
	{
		if c.Cache.SplitGuard <= 0 || c.Cache.SplitGuard >= 1 {
			errs = append(errs, fmt.Errorf("cache.split_guard %v must be in (0,1)", c.Cache.SplitGuard))
		}
		if time.Duration(c.Cache.TailTTL) <= 0 {
			errs = append(errs, errors.New("cache.tail_ttl must be > 0"))
		}
		if time.Duration(c.Cache.RefusalTTL) <= 0 {
			errs = append(errs, errors.New("cache.refusal_ttl must be > 0"))
		}
	}
	if c.Discovery.Enabled {
		switch c.Discovery.Mode {
		case "endpointslice":
			if c.Discovery.Service == "" {
				errs = append(errs, errors.New("discovery.service is required in endpointslice mode"))
			}
		case "pod":
			if c.Discovery.Selector == "" {
				errs = append(errs, errors.New("discovery.selector is required in pod mode"))
			}
			if c.Discovery.Port <= 0 {
				errs = append(errs, errors.New("discovery.port is required in pod mode"))
			}
		default:
			errs = append(errs, fmt.Errorf("discovery.mode %q must be endpointslice or pod", c.Discovery.Mode))
		}
		if c.Discovery.Namespace == "" {
			errs = append(errs, errors.New("discovery.namespace is required when discovery is enabled"))
		}
	}
	if !c.Discovery.Enabled && len(c.Backends) == 0 {
		errs = append(errs, errors.New("no backends configured and discovery is disabled: the router would have nowhere to route"))
	}
	for i, b := range c.Backends {
		if strings.TrimSpace(b.URL) == "" {
			errs = append(errs, fmt.Errorf("backends[%d]: empty url", i))
		}
		if b.Kind != "" && b.Kind != "worker" && b.Kind != "router" {
			errs = append(errs, fmt.Errorf("backends[%d]: kind %q must be worker or router", i, b.Kind))
		}
		if b.Health != "" && b.Health != "active" && b.Health != "passive" {
			errs = append(errs, fmt.Errorf("backends[%d]: health %q must be active or passive", i, b.Health))
		}
	}
	return errors.Join(errs...)
}

// Redacted returns the effective config with secrets removed, for
// /get_server_info (CFG-7, SEC-6).
func (c Config) Redacted() Config {
	out := c
	if out.APIKey != "" {
		out.APIKey = "[redacted]"
	}
	if out.UpstreamCredential != "" {
		out.UpstreamCredential = "[redacted]"
	}
	return out
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

type devNull struct{}

func (devNull) Write(p []byte) (int, error) { return len(p), nil }
