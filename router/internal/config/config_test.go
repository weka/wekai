package config_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/config"
)

// TestConfigMapDurationsParse is a regression test for a bug found while writing
// the Kubernetes ConfigMap: a plain time.Duration field cannot unmarshal "10s"
// from JSON, only a raw nanosecond count. The manifest would have failed at
// startup, and the alternative — writing 600000000000 in a ConfigMap — is where
// operators make mistakes.
func TestConfigMapDurationsParse(t *testing.T) {
	body := `{
      "listen": ":8080",
      "backends": [{"url": "http://w:8000"}],
      "health_interval": "10s",
      "health_timeout": "5s",
      "drain_deadline": "90s",
      "request_timeout": "10m",
      "idle_timeout": "5m",
      "discovery": {"resync_interval": "1h30m"}
    }`
	cfg := mustLoadFile(t, body)

	for _, c := range []struct {
		name string
		got  time.Duration
		want time.Duration
	}{
		{"health_interval", cfg.HealthInterval.D(), 10 * time.Second},
		{"health_timeout", cfg.HealthTimeout.D(), 5 * time.Second},
		{"drain_deadline", cfg.DrainDeadline.D(), 90 * time.Second},
		{"request_timeout", cfg.RequestTimeout.D(), 10 * time.Minute},
		{"idle_timeout", cfg.IdleTimeout.D(), 5 * time.Minute},
		{"discovery.resync_interval", cfg.Discovery.ResyncInterval.D(), 90 * time.Minute},
	} {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// A nanosecond count must still work, so a config written against a plain
// time.Duration field does not silently change meaning.
func TestNumericDurationsStillParse(t *testing.T) {
	cfg := mustLoadFile(t, `{"backends":[{"url":"http://w:8000"}],"health_interval":15000000000}`)
	if got := cfg.HealthInterval.D(); got != 15*time.Second {
		t.Errorf("health_interval = %v, want 15s", got)
	}
}

func TestDurationRoundTripsAsString(t *testing.T) {
	var d config.Duration
	if err := d.Set("2m30s"); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `"2m30s"` {
		t.Errorf("marshalled to %s, want \"2m30s\" — an unreadable form defeats the purpose", b)
	}
	var back config.Duration
	if err := json.Unmarshal(b, &back); err != nil || back != d {
		t.Errorf("round trip failed: %v %v", back, err)
	}
}

func TestBadDurationIsReported(t *testing.T) {
	var d config.Duration
	if err := json.Unmarshal([]byte(`"ten seconds"`), &d); err == nil {
		t.Error("expected an error for an unparseable duration")
	}
	if err := json.Unmarshal([]byte(`{}`), &d); err == nil {
		t.Error("expected an error for a non-scalar duration")
	}
}

// CFG-6: an unknown key is a hard error, not a warning. A typo in a ConfigMap
// should fail loudly rather than silently leave a default in place.
func TestUnknownKeyIsRejected(t *testing.T) {
	_, err := load(t, `{"listen":":8080","polcy":"random"}`)
	if err == nil {
		t.Fatal("expected an error for an unknown key")
	}
	if !strings.Contains(err.Error(), "polcy") {
		t.Errorf("error should name the offending key, got: %v", err)
	}
}

// CFG-5: validation aggregates every problem into one report, so a misconfigured
// deployment takes one round trip to fix rather than N.
func TestValidationAggregatesAllProblems(t *testing.T) {
	_, err := load(t, `{
      "rebalance_ratio": 2,
      "max_body_bytes": 0,
      "health_interval": "1s",
      "health_timeout": "5s",
      "max_attempts": 0,
      "path_allowlist": ["missing-slash"],
      "backends": [{"url": "http://w:8000", "kind": "sidecar"}]
    }`)
	if err == nil {
		t.Fatal("expected validation errors")
	}
	msg := err.Error()
	for _, want := range []string{
		"rebalance_ratio",
		"max_body_bytes",
		"health_timeout",
		"max_attempts",
		"must start with /",
		"must be worker or router",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("aggregated error is missing %q:\n%s", want, msg)
		}
	}
}

// HLT-2: a timeout at or above the interval lets checks fall behind forever,
// which is how v1's health state went stale indefinitely.
func TestTimeoutMustBeLessThanInterval(t *testing.T) {
	if _, err := load(t, `{"backends":[{"url":"http://w:8000"}],"health_interval":"5s","health_timeout":"5s"}`); err == nil {
		t.Error("timeout == interval should be rejected")
	}
	if _, err := load(t, `{"backends":[{"url":"http://w:8000"}],"health_interval":"5s","health_timeout":"1s"}`); err != nil {
		t.Errorf("timeout < interval should be accepted: %v", err)
	}
}

// A router with no backends and no discovery has nowhere to route. Failing at
// startup beats serving 503 to every request.
func TestNoBackendsAndNoDiscoveryIsRejected(t *testing.T) {
	if _, err := load(t, `{"listen":":8080"}`); err == nil {
		t.Error("expected an error when neither backends nor discovery is configured")
	}
}

func TestDiscoveryValidation(t *testing.T) {
	cases := []struct {
		name string
		body string
		ok   bool
	}{
		{"endpointslice ok", `{"discovery":{"enabled":true,"mode":"endpointslice","namespace":"ns","service":"vllm"}}`, true},
		{"endpointslice without service", `{"discovery":{"enabled":true,"mode":"endpointslice","namespace":"ns"}}`, false},
		{"pod ok", `{"discovery":{"enabled":true,"mode":"pod","namespace":"ns","selector":"app=vllm","port":8000}}`, true},
		{"pod without port", `{"discovery":{"enabled":true,"mode":"pod","namespace":"ns","selector":"app=vllm"}}`, false},
		{"no namespace", `{"discovery":{"enabled":true,"mode":"endpointslice","service":"vllm"}}`, false},
		{"bad mode", `{"discovery":{"enabled":true,"mode":"magic","namespace":"ns"}}`, false},
	}
	for _, c := range cases {
		_, err := load(t, c.body)
		if c.ok && err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
		}
		if !c.ok && err == nil {
			t.Errorf("%s: expected an error", c.name)
		}
	}
}

// CFG-8: the API key must not be settable by flag, because a flag value is
// visible in `ps` to anything on the host.
func TestAPIKeyIsNotAFlag(t *testing.T) {
	_, err := config.Load([]string{"-api-key", "leaked"}, func(string) string { return "" })
	if err == nil {
		t.Fatal("-api-key was accepted as a flag; it must be env or file only")
	}
}

func TestAPIKeyFromFileAndEnv(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "key")
	if err := os.WriteFile(path, []byte("  file-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(
		[]string{"-api-key-file", path, "-backends", "http://w:8000"},
		func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "file-key" {
		t.Errorf("APIKey = %q, want the trimmed file contents", cfg.APIKey)
	}

	cfg2, err := config.Load([]string{"-backends", "http://w:8000"}, func(k string) string {
		if k == "WLLM_API_KEY" {
			return "env-key"
		}
		return ""
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg2.APIKey != "env-key" {
		t.Errorf("APIKey = %q, want env-key", cfg2.APIKey)
	}
}

// CFG-1: flag beats env beats file beats default.
//
// Carried on log_level since --policy was retired: the router has one routing
// flow, so there is no policy name left to layer.
func TestPrecedenceFlagOverEnvOverFileOverDefault(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"log_level":"warn","backends":[{"url":"http://w:8000"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	noEnv := func(string) string { return "" }
	envDebug := func(k string) string {
		if k == "WLLM_LOG_LEVEL" {
			return "debug"
		}
		return ""
	}

	base, err := config.Load([]string{"-backends", "http://w:8000"}, noEnv)
	if err != nil {
		t.Fatal(err)
	}
	if base.LogLevel != "info" {
		t.Errorf("default log_level = %q, want info", base.LogLevel)
	}

	fromFile, err := config.Load([]string{"-config", path}, noEnv)
	if err != nil {
		t.Fatal(err)
	}
	if fromFile.LogLevel != "warn" {
		t.Errorf("file log_level = %q, want warn", fromFile.LogLevel)
	}

	fromEnv, err := config.Load([]string{"-config", path}, envDebug)
	if err != nil {
		t.Fatal(err)
	}
	if fromEnv.LogLevel != "debug" {
		t.Errorf("env log_level = %q, want debug", fromEnv.LogLevel)
	}

	fromFlag, err := config.Load([]string{"-config", path, "-log-level", "error"}, envDebug)
	if err != nil {
		t.Fatal(err)
	}
	if fromFlag.LogLevel != "error" {
		t.Errorf("flag log_level = %q, want error", fromFlag.LogLevel)
	}
}

// SEC-6/CFG-7: /get_server_info must not leak secrets.
func TestRedactedRemovesSecrets(t *testing.T) {
	cfg := config.Default()
	cfg.APIKey = "inbound-secret"
	cfg.UpstreamCredential = "upstream-secret"
	b, err := json.Marshal(cfg.Redacted())
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"inbound-secret", "upstream-secret"} {
		if strings.Contains(string(b), s) {
			t.Errorf("Redacted() leaked %q", s)
		}
	}
}

// SEC-10: a wildcard CORS origin combined with credentials is rejected.
func TestWildcardCORSWithAPIKeyRejected(t *testing.T) {
	_, err := config.Load([]string{"-backends", "http://w:8000", "-cors-origins", "*"},
		func(k string) string {
			if k == "WLLM_API_KEY" {
				return "secret"
			}
			return ""
		})
	if err == nil {
		t.Error("expected an error for wildcard CORS with an API key")
	}
}

func mustLoadFile(t *testing.T, body string) config.Config {
	t.Helper()
	cfg, err := load(t, body)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return cfg
}

func load(t *testing.T, body string) (config.Config, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return config.Load([]string{"-config", path}, func(string) string { return "" })
}

// TestNoSignalConfiguredStillStarts is the inverse of what this file used to
// assert. max_node_concurrency was MANDATORY, because it was simultaneously the
// gateway's admission filter and the split guard's reference and so had to mean
// one thing. It is now one opt-in signal alongside the backend's own 429, which
// is always on and needs no configuration, so a bare --backends deployment is
// valid rather than a silent misconfiguration.
func TestNoSignalConfiguredStillStarts(t *testing.T) {
	if _, err := load(t, `{"backends":[{"url":"http://w:8000"}]}`); err != nil {
		t.Errorf("a router with no optional signals configured should start: %v", err)
	}
}

// TestSignalKnobsAreValidated: all have defaults, so reaching these errors
// means an operator set something meaningless on purpose.
func TestSignalKnobsAreValidated(t *testing.T) {
	const base = `{"backends":[{"url":"http://w:8000"}],`
	for _, tc := range []struct{ frag, want string }{
		{`"cache":{"split_guard":0,"tail_ttl":"5m","refusal_ttl":"2s"}}`, "split_guard"},
		{`"cache":{"split_guard":1,"tail_ttl":"5m","refusal_ttl":"2s"}}`, "split_guard"},
		{`"cache":{"split_guard":0.2,"tail_ttl":"0s","refusal_ttl":"2s"}}`, "tail_ttl"},
		{`"cache":{"split_guard":0.2,"tail_ttl":"5m","refusal_ttl":"0s"}}`, "refusal_ttl"},
		{`"rebalance_ratio":1,"cache":{"split_guard":0.2,"tail_ttl":"5m","refusal_ttl":"2s"}}`, "rebalance_ratio"},
		{`"rebalance_ratio":-0.1,"cache":{"split_guard":0.2,"tail_ttl":"5m","refusal_ttl":"2s"}}`, "rebalance_ratio"},
	} {
		_, err := load(t, base+tc.frag)
		if err == nil {
			t.Errorf("%s should be rejected", tc.frag)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error should name %q:\n%s", tc.frag, tc.want, err)
		}
	}
}

// TestRetiredPolicyKeyIsRejected: the router has one routing flow, so `policy`
// is not merely ignored — DisallowUnknownFields means a config still naming one
// fails loudly rather than silently routing differently than its author asked.
func TestRetiredPolicyKeyIsRejected(t *testing.T) {
	_, err := load(t, `{"backends":[{"url":"http://w:8000"}],"policy":"prefix-cache-aware"}`)
	if err == nil {
		t.Fatal("a config naming a retired policy should fail startup, not be ignored")
	}
	if !strings.Contains(err.Error(), "policy") {
		t.Errorf("the error must name the offending key:\n%s", err)
	}
}
