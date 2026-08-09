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
		if dep == "" {
			continue
		}
		// Stdlib packages have no dot in their FIRST path element. Testing the
		// whole path instead was equivalent until Go 1.26, which gave the FIPS
		// 140-3 internals versioned import paths — crypto/sha256 now pulls in
		// crypto/internal/entropy/v1.0.0, whose dots are in the version
		// segment. That read as a third-party dependency and failed this fence
		// on a package that had not changed.
		if first, _, _ := strings.Cut(dep, "/"); !strings.Contains(first, ".") {
			continue
		}
		if strings.HasPrefix(dep, "github.com/weka/wekai/kvcache") {
			continue
		}
		t.Errorf("kvcache depends on %s; it must remain a stdlib-only leaf so both "+
			"the benchmark CLI and the router can hold it without coupling", dep)
	}
}
