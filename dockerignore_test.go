package main

import (
	"os"
	"strings"
	"testing"
)

// TestDockerignoreKeepsWhatTheImagesBuild guards a failure that is invisible
// until someone builds an image.
//
// .dockerignore once excluded main.go, cli/, benchmark/, llm/, config/, chart/
// and tools/. That was correct while the router was a separate small binary
// under router/cmd/ needing only router/ and kvcache/. Once the router became
// `wekai router serve` — the whole binary, built from the ROOT package — the
// same file left the build context with no Go files at all, and BOTH images
// failed with "no Go files in /src".
//
// Nothing caught it: the release path builds through Dagger, which filters the
// directory itself rather than applying .dockerignore, so only a local
// `docker build` hit it.
func TestDockerignoreKeepsWhatTheImagesBuild(t *testing.T) {
	raw, err := os.ReadFile(".dockerignore")
	if err != nil {
		t.Fatalf("read .dockerignore: %v", err)
	}
	var patterns []string
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			patterns = append(patterns, strings.Trim(line, "/"))
		}
	}

	// Everything `go build .` needs to reach the container. main.go is the
	// entrypoint; the rest are packages it imports transitively.
	required := []string{
		"main.go", "cli", "router", "benchmark", "kvcache", "llm", "config", "tools",
	}
	for _, need := range required {
		for _, p := range patterns {
			if p == need {
				t.Errorf(".dockerignore excludes %q, which `go build .` needs — the image "+
					"build will fail with \"no Go files in /src\" or a missing import. "+
					"The router is the whole wekai binary now, not a package under router/.", need)
			}
		}
	}
}
