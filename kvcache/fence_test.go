package kvcache_test

import (
	"os/exec"
	"strings"
	"testing"
)

// kvcache is shared by the benchmark tooling and the router, so it must stay a
// leaf: no wire formats, no router internals, nothing but the standard library.
// A dependency creeping in here would couple the benchmark CLI to the router.
func TestKvcacheDependsOnlyOnStdlib(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/weka/wekai/kvcache").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	for _, dep := range strings.Split(string(out), "\n") {
		dep = strings.TrimSpace(dep)
		if dep == "" || !strings.Contains(dep, ".") {
			continue // stdlib packages have no dot in their first path element
		}
		if strings.HasPrefix(dep, "github.com/weka/wekai/kvcache") {
			continue
		}
		t.Errorf("kvcache depends on %s; it must remain a stdlib-only leaf so both "+
			"the benchmark CLI and the router can hold it without coupling", dep)
	}
}
