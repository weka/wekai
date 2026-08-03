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
	Policy        string    `json:"policy"`
	MaxBodyBytes  int64     `json:"max_body_bytes"`
	Backends      []Backend `json:"backends"`

	// APIKey is intentionally NOT settable by flag: a flag value is visible in
	// `ps` output to every user on the host (CFG-8). Env or file only.
	APIKey     string `json:"-"`
	APIKeyFile string `json:"api_key_file"`
	// UpstreamCredential is sent to backends; distinct from APIKey, which
	// authenticates clients TO us and is never forwarded (AUTH-9).
	UpstreamCredential string `json:"-"`

	// PathAllowlist is a strict allowlist. An EMPTY list means no exemptions,
	// which is a deliberate inversion of v1's "empty means allow everything" —
	// there, a config typo silently opened the router (AUTH-8, AUTH-N2).
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
	// CacheThreshold is the matched fraction below which affinity is not worth
	// overriding load balance.
	CacheThreshold float64 `json:"cache_threshold"`
	// Spill guard: prefer affinity until the fleet is measurably imbalanced.
	BalanceAbsThreshold int64   `json:"balance_abs_threshold"`
	BalanceRelThreshold float64 `json:"balance_rel_threshold"`
	// Per-backend model bounds. MaxTokens is the binding one in practice.
	MaxNodes  int64 `json:"max_nodes_per_backend"`
	MaxTokens int64 `json:"max_tokens_per_backend"`
	// ChunkBytes is the prefix-unit granularity; see prefix.DefaultChunkBytes.
	ChunkBytes int `json:"chunk_bytes"`
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

// ValidPolicies is the closed set. An unknown name fails startup with this list
// rather than silently falling back, which is how v1 lost `consistent_hash` in
// one of its two divergent name tables.
var ValidPolicies = []string{
	"least-outstanding", "round-robin", "random", "prefix-cache-aware",
	"prefix-cache-candidates",
}

func Default() Config {
	return Config{
		Listen:                ":8080",
		MetricsListen:         "127.0.0.1:29000",
		Policy:                "least-outstanding",
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
			CacheThreshold:      0.5,
			BalanceAbsThreshold: 32,
			BalanceRelThreshold: 1.5,
			MaxNodes:            100_000,
			MaxTokens:           2_000_000,
			ChunkBytes:          1024,
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
	fs.StringVar(&cfg.Policy, "policy", cfg.Policy,
		"routing policy: "+strings.Join(ValidPolicies, ", "))
	fs.Int64Var(&cfg.MaxBodyBytes, "max-body-bytes", cfg.MaxBodyBytes, "maximum request body size")
	fs.IntVar(&cfg.MaxConcurrentRequests, "max-concurrent-requests", cfg.MaxConcurrentRequests, "in-flight request cap (0 disables)")
	fs.IntVar(&cfg.MaxAttempts, "max-attempts", cfg.MaxAttempts, "total upstream attempts including the first")
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
	set(&cfg.Policy, "WLLM_POLICY")
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
	if v := strings.TrimSpace(getenv("WLLM_PATH_ALLOWLIST")); v != "" {
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

	if !contains(ValidPolicies, c.Policy) {
		errs = append(errs, fmt.Errorf("unknown policy %q; valid: %s",
			c.Policy, strings.Join(ValidPolicies, ", ")))
	}
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
	if c.Policy == "prefix-cache-aware" || c.Policy == "prefix-cache-candidates" {
		if c.Cache.CacheThreshold <= 0 || c.Cache.CacheThreshold > 1 {
			errs = append(errs, fmt.Errorf("cache.cache_threshold %v must be in (0,1]", c.Cache.CacheThreshold))
		}
		if c.Cache.MaxTokens <= 0 || c.Cache.MaxNodes <= 0 {
			errs = append(errs, errors.New("cache.max_tokens_per_backend and max_nodes_per_backend must be > 0"))
		}
		if c.Cache.ChunkBytes <= 0 {
			errs = append(errs, errors.New("cache.chunk_bytes must be > 0"))
		}
	}
	if c.Policy == "prefix-cache-aware" {
		if c.Cache.BalanceRelThreshold < 1 {
			errs = append(errs, fmt.Errorf("cache.balance_rel_threshold %v must be >= 1", c.Cache.BalanceRelThreshold))
		}
	}
	if c.Policy == "prefix-cache-candidates" {
		if c.Cache.BalanceAbsThreshold <= 0 {
			errs = append(errs, fmt.Errorf("cache.balance_abs_threshold %v must be > 0 (used as the pending-tasks threshold)", c.Cache.BalanceAbsThreshold))
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
