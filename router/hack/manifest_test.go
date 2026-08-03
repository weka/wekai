package hack_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"sigs.k8s.io/yaml"

	"github.com/weka/wekai/router/internal/config"
)

// TestShippedConfigMapLoads asserts that the config.json embedded in the
// Kubernetes manifest is actually accepted by the loader.
//
// This exists because the first version of that ConfigMap did not load: durations
// were written as "10s", which a plain time.Duration field cannot unmarshal from
// JSON. Nothing in the Go test suite touched the manifest, so the failure would
// only have appeared on deploy. A manifest that cannot be loaded is a broken
// deliverable, and this keeps the two from drifting.
func TestShippedConfigMapLoads(t *testing.T) {
	root := repoRoot(t)
	path := filepath.Join(root, "deploy", "k8s", "deployment.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}

	body := extractConfigJSON(t, raw)
	if body == "" {
		t.Fatal("no ConfigMap with a config.json key found in deploy/k8s/deployment.yaml")
	}

	// Guard the guard: if the manifest ever stops using human-readable durations,
	// this test silently stops covering the bug it was written for.
	if !strings.Contains(body, `"health_interval": "`) {
		t.Error("the ConfigMap no longer expresses health_interval as a duration string, " +
			"so this test no longer covers the parse path it exists for")
	}

	tmp := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(tmp, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load([]string{"-config", tmp}, func(string) string { return "" })
	if err != nil {
		t.Fatalf("the shipped ConfigMap does not load:\n%v\n\n--- config.json ---\n%s", err, body)
	}

	// A few properties the manifest depends on being true.
	if cfg.MetricsListen == "127.0.0.1:29000" {
		t.Error("metrics_listen is loopback-only, so the Prometheus scrape annotation " +
			"in the manifest can never reach it")
	}
	if !cfg.Discovery.Enabled {
		t.Error("the manifest ships RBAC for discovery but discovery is disabled in the ConfigMap")
	}
	if cfg.HealthTimeout.D() >= cfg.HealthInterval.D() {
		t.Error("health_timeout >= health_interval in the shipped ConfigMap")
	}
}

// extractConfigJSON pulls the config.json value out of the first ConfigMap
// document that has one.
func extractConfigJSON(t *testing.T, raw []byte) string {
	t.Helper()
	for _, doc := range strings.Split(string(raw), "\n---") {
		doc = strings.TrimSpace(doc)
		if doc == "" {
			continue
		}
		var obj struct {
			Kind string            `json:"kind"`
			Data map[string]string `json:"data"`
		}
		if err := yaml.Unmarshal([]byte(doc), &obj); err != nil {
			// Not every document is a ConfigMap; a parse failure on one is only
			// interesting if it is the one we want, so keep looking.
			continue
		}
		if obj.Kind == "ConfigMap" {
			if v, ok := obj.Data["config.json"]; ok {
				return v
			}
		}
	}
	return ""
}

// TestManifestsAreValidYAML catches a broken manifest before kubectl does.
func TestManifestsAreValidYAML(t *testing.T) {
	root := repoRoot(t)
	dir := filepath.Join(root, "deploy", "k8s")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read deploy/k8s: %v", err)
	}
	seen := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for i, doc := range strings.Split(string(raw), "\n---") {
			doc = strings.TrimSpace(doc)
			if doc == "" {
				continue
			}
			var obj map[string]any
			if err := yaml.Unmarshal([]byte(doc), &obj); err != nil {
				t.Errorf("%s document %d is not valid YAML: %v", e.Name(), i, err)
				continue
			}
			if obj["kind"] == nil || obj["apiVersion"] == nil {
				t.Errorf("%s document %d is missing kind or apiVersion", e.Name(), i)
			}
			seen++
		}
	}
	if seen == 0 {
		t.Error("no manifest documents found under deploy/k8s")
	}
}
