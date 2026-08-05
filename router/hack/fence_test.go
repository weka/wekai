// Package hack holds mechanical invariant checks that run as ordinary tests.
//
// These exist because the defects that motivated the v2 rewrite were not
// subtle algorithms — they were invariants that no tool enforced, so they
// drifted. A grep in CI is cheap; a corrupt load counter cost v1 every
// load-based routing decision it ever made.
package hack_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// repoRoot resolves the module root from this test's location.
func repoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// hack/ -> router/ (the router subtree root inside the wekai module)
	return filepath.Dir(wd)
}

// goFiles walks the Go source we own, reporting paths relative to the root.
func goFiles(t *testing.T, root string, includeTests bool) []string {
	t.Helper()
	var out []string
	for _, dir := range []string{"internal", "cmd"} {
		base := filepath.Join(root, dir)
		if _, err := os.Stat(base); os.IsNotExist(err) {
			continue
		}
		err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() || !strings.HasSuffix(path, ".go") {
				return nil
			}
			if !includeTests && strings.HasSuffix(path, "_test.go") {
				return nil
			}
			rel, _ := filepath.Rel(root, path)
			out = append(out, rel)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// calledSelectors returns the set of `X.Name(` method names invoked in a file.
func calledSelectors(t *testing.T, path string) map[string][]int {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	found := map[string][]int{}
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		name := sel.Sel.Name
		found[name] = append(found[name], fset.Position(call.Pos()).Line)
		return true
	})
	return found
}

// TestOnlyLeaseMutatesInflight enforces LB-6, the single most important
// invariant in the router.
//
// v1's in-flight counter was incremented on one code path and decremented on
// three, and the health checker zeroed every counter every ten cycles
// (LB-N1, LB-N2, HLT-N5). Every load-sensitive decision downstream — power of
// two, the cache-aware imbalance guard, the operator-facing load gauges — was
// therefore reading noise. The fix is that exactly one package may touch the
// counter, and that is checked here rather than trusted.
func TestOnlyLeaseMutatesInflight(t *testing.T) {
	root := repoRoot(t)
	const (
		mutatorPkg = "internal/lease"
		declFile   = "internal/registry/backend.go"
	)
	mutators := []string{"AddInflight", "StoreInflight"}

	for _, rel := range goFiles(t, root, false) {
		dir := filepath.ToSlash(filepath.Dir(rel))
		if dir == mutatorPkg || filepath.ToSlash(rel) == declFile {
			continue // the one legitimate caller, and the declaration itself
		}
		calls := calledSelectors(t, filepath.Join(root, rel))
		for _, m := range mutators {
			if lines, ok := calls[m]; ok {
				t.Errorf("%s:%v calls %s — only %s may mutate in-flight load (LB-6). "+
					"Acquire a lease.Lease instead.", rel, lines, m, mutatorPkg)
			}
		}
	}
}

// TestProductionCodeUsesTheClockAbstraction enforces AC-0.2: every time-based
// DECISION must be driven by a clock.Clock, so circuit windows, health
// hysteresis, drain deadlines and cache TTLs are testable without sleeps.
//
// Latency measurement is a legitimate exception — a histogram observation is not
// a decision, and faking it would measure nothing. Rather than exempting whole
// files, a call site may opt out with a `//clockexempt: <reason>` comment on the
// same line or the line above. The exemption then travels with the code and has
// to state why, so a decision cannot quietly acquire a wall clock.
func TestProductionCodeUsesTheClockAbstraction(t *testing.T) {
	root := repoRoot(t)
	banned := []string{"time.Now(", "time.Since(", "time.Sleep(", "time.Tick("}

	for _, rel := range goFiles(t, root, false) {
		slashed := filepath.ToSlash(rel)
		switch {
		case strings.HasPrefix(slashed, "internal/clock"):
			continue // the implementation of the abstraction itself
		case strings.HasPrefix(slashed, "internal/testutil/"):
			continue // test infrastructure must exercise real timing
		}

		src, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(src), "\n")
		for i, line := range lines {
			for _, b := range banned {
				if !strings.Contains(line, b) {
					continue
				}
				if exempted(lines, i) {
					continue
				}
				t.Errorf("%s:%d calls %s) without a //clockexempt: reason — "+
					"take a clock.Clock instead (AC-0.2)", rel, i+1, b)
			}
		}
	}
}

// exempted reports whether line i carries a clockexempt marker, on itself or on
// the line immediately above.
func exempted(lines []string, i int) bool {
	const marker = "//clockexempt:"
	if strings.Contains(lines[i], marker) {
		return true
	}
	return i > 0 && strings.Contains(lines[i-1], marker)
}

// TestNoArgvDump guards CFG-N2. v1 printed `DEBUG: Raw args: {:?}` to stdout
// before logging was configured, dumping the full command line — including any
// secret passed as a flag — with no way to suppress it by log level.
func TestNoArgvDump(t *testing.T) {
	root := repoRoot(t)
	for _, rel := range goFiles(t, root, false) {
		if filepath.ToSlash(rel) == "cmd/wllm-router/main.go" {
			continue // main legitimately reads os.Args to parse them
		}
		src, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(src), "os.Args") {
			t.Errorf("%s references os.Args outside main (CFG-2, CFG-N2)", rel)
		}
	}
}

// TestNoDeadMetrics enforces OBS-5.
//
// Every collector declared in internal/metrics must be referenced somewhere else.
// This was claimed in a comment and never implemented, and four metrics had
// silently rotted: router_load_accounting_errors_total (the canary release gate —
// permanently zero, indistinguishable from healthy), router_cache_observed_fraction
// (so prediction accuracy was unmeasurable), and the two cache-size gauges. It is
// the exact anti-pattern v1 was criticised for, where a whole tokenizer metric
// family was registered for a subsystem the routing path never called.
func TestNoDeadMetrics(t *testing.T) {
	root := repoRoot(t)
	src, err := os.ReadFile(filepath.Join(root, "internal", "metrics", "metrics.go"))
	if err != nil {
		t.Fatal(err)
	}

	var declared []string
	for _, line := range strings.Split(string(src), "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.Contains(trimmed, "= prometheus.New") {
			continue
		}
		name := strings.TrimSpace(strings.SplitN(trimmed, "=", 2)[0])
		// Both declaration forms: inside a `var (...)` block, and a standalone
		// `var X = ...`. Missing the second form is how this fence first shipped
		// unable to fail.
		name = strings.TrimPrefix(name, "var ")
		name = strings.TrimSpace(name)
		if name != "" && name[0] >= 'A' && name[0] <= 'Z' {
			declared = append(declared, name)
		}
	}
	if len(declared) == 0 {
		t.Fatal("parsed no collectors from internal/metrics/metrics.go")
	}

	for _, name := range declared {
		used := false
		for _, rel := range goFiles(t, root, false) {
			if strings.HasPrefix(filepath.ToSlash(rel), "internal/metrics/") {
				continue
			}
			b, err := os.ReadFile(filepath.Join(root, rel))
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(b), "metrics."+name) {
				used = true
				break
			}
		}
		if !used {
			t.Errorf("metric %s is declared and registered but never emitted — it will "+
				"read as a flat zero forever, which is worse than absent", name)
		}
	}
}

// TestCoreDoesNotImportDialects enforces API-1.
//
// The claim that adding a second wire format needs no core change is only true if
// nothing in the core knows about wire formats. This was asserted in a package
// comment and never checked. `go list -deps` is transitive, so indirect leakage
// is caught too.
func TestCoreDoesNotImportDialects(t *testing.T) {
	core := []string{
		"registry", "lease", "policy", "policy/cache",
		"proxy", "circuit", "health",
	}
	for _, pkg := range core {
		out, err := exec.Command("go", "list", "-deps",
			"github.com/weka/wekai/router/internal/"+pkg).Output()
		if err != nil {
			t.Fatalf("go list -deps %s: %v", pkg, err)
		}
		for _, dep := range strings.Split(string(out), "\n") {
			if strings.Contains(dep, "wekai/router/internal/dialect/") {
				t.Errorf("core package %s depends on %s — a dialect must not reach "+
					"the routing core (API-1)", pkg, dep)
			}
		}
	}
}

// TestAuthIsEnforcedInExactlyOnePlace enforces AUTH-4.
//
// v1 layered a global auth middleware AND called authorize_request at the top of
// all ~28 handlers; two mechanisms for one invariant is two things to drift.
func TestAuthIsEnforcedInExactlyOnePlace(t *testing.T) {
	root := repoRoot(t)
	sites := 0
	for _, rel := range goFiles(t, root, false) {
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatal(err)
		}
		sites += strings.Count(string(b), "subtle.ConstantTimeCompare")
	}
	if sites != 1 {
		t.Errorf("found %d credential-comparison sites, want exactly 1", sites)
	}
}
