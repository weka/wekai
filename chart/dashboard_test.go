package chart_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// The dashboard shipped in the chart had gone stale without anything noticing:
// it charted only the model-routing accounting metrics, so an operator
// installing the chart got a dashboard that said nothing about whether backend
// routing — the thing the router exists to do — was working at all. A dashboard
// is documentation that rots silently, because nothing compiles it.
//
// This compiles it. Every metric a panel queries has to be a metric the code
// actually exports.
func TestDashboardOnlyQueriesMetricsThatExist(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("router", "grafana", "dashboard.json"))
	if err != nil {
		t.Fatalf("read dashboard: %v", err)
	}
	var dash struct {
		Panels []struct {
			Title   string `json:"title"`
			Targets []struct {
				Expr string `json:"expr"`
			} `json:"targets"`
		} `json:"panels"`
	}
	if err := json.Unmarshal(raw, &dash); err != nil {
		t.Fatalf("dashboard is not valid JSON: %v", err)
	}

	declared := declaredMetrics(t)
	// Prometheus derives these from a histogram; the base name is what the code
	// declares.
	suffixes := []string{"_bucket", "_sum", "_count"}
	metricRE := regexp.MustCompile(`\b(?:wekai_)?router_[a-z0-9_]+`)

	queried := map[string][]string{}
	for _, p := range dash.Panels {
		for _, tgt := range p.Targets {
			for _, m := range metricRE.FindAllString(tgt.Expr, -1) {
				for _, s := range suffixes {
					m = strings.TrimSuffix(m, s)
				}
				queried[m] = append(queried[m], p.Title)
			}
		}
	}
	if len(queried) == 0 {
		t.Fatal("no metrics found in any panel; the dashboard queries nothing")
	}

	var missing []string
	for m, panels := range queried {
		if !declared[m] {
			missing = append(missing, m+" (panel: "+panels[0]+")")
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("dashboard queries %d metric(s) the code does not export, so those "+
			"panels render empty:\n  %s", len(missing), strings.Join(missing, "\n  "))
	}
}

// The router's own routing metrics are the ones with no chart at all if this
// drifts, so require the dashboard to cover the decisive few by name.
func TestDashboardCoversBackendRouting(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("router", "grafana", "dashboard.json"))
	if err != nil {
		t.Fatalf("read dashboard: %v", err)
	}
	body := string(raw)
	for _, want := range []string{
		"router_cache_avg_copies",          // duplication, the number the design targets
		"router_route_decisions_total",     // which tier of the ladder served the request
		"router_cache_splits_total",        // what drives duplication up
		"router_cache_guard_rejects_total", // the trade: a 429 instead of a copy
		"router_signal_fired_total",        // why a backend was called saturated
		"router_backend_inflight",          // load, which affinity deliberately skews
		"router_backends_total",            // a pool that quietly shrank
	} {
		if !strings.Contains(body, want) {
			t.Errorf("dashboard has no panel using %q; backend routing is unobservable "+
				"without it, which is how the previous dashboard went stale", want)
		}
	}
}

// declaredMetrics collects every metric name declared in the Go source, which is
// the authority the dashboard is checked against.
func declaredMetrics(t *testing.T) map[string]bool {
	t.Helper()
	nameRE := regexp.MustCompile(`"((?:wekai_)?router_[a-z0-9_]+)"`)
	out := map[string]bool{}
	err := filepath.Walk("..", func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range nameRE.FindAllStringSubmatch(string(b), -1) {
			out[m[1]] = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk source: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("found no metric declarations in the source tree")
	}
	return out
}
