package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestWorkflowsDoNotInvokeMake guards a failure that only appears on main.
//
// The Makefile was removed in favour of the Taskfile, but release.yml still ran
// `make test`. Nothing caught it: a workflow is not compiled and not linted,
// and the release job is the one job that does NOT run on a pull request — so
// the first sign of trouble was a failed release after the merge.
//
// Every build command in a workflow has to be one this repository can still
// run. Lives here rather than beside the workflows because the go tool skips
// dot-directories, so a test under .github/ would never run at all.
func TestWorkflowsDoNotInvokeMake(t *testing.T) {
	if _, err := os.Stat("Makefile"); err == nil {
		t.Skip("a Makefile exists again; `make` in a workflow is fine")
	}
	runMake := regexp.MustCompile(`(?m)^\s*(?:run:|-)\s+.*\bmake\b.*$`)
	for _, f := range workflowFiles(t) {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for _, line := range runMake.FindAllString(string(b), -1) {
			t.Errorf("%s runs make, but this repository has no Makefile — and a workflow "+
				"only fails on the branch that runs it:\n\t%s", f, strings.TrimSpace(line))
		}
	}
}

// TestReleaseWorkflowGatesOnTests: a release that ships untested code is worse
// than one that fails to build.
func TestReleaseWorkflowGatesOnTests(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(".github", "workflows", "release.yml"))
	if err != nil {
		t.Fatalf("read release.yml: %v", err)
	}
	body := string(b)
	if !strings.Contains(body, "run: task test") {
		t.Error("release.yml does not run `task test`; the release gate would ship " +
			"whatever was merged without running the suite")
	}
	// Any `task` invocation needs task on the runner, which is not preinstalled.
	if strings.Contains(body, "run: task ") && !strings.Contains(body, "arduino/setup-task") {
		t.Error("release.yml runs task without installing it first; the step fails " +
			"with 'task: command not found'")
	}
}

// TestWorkflowTasksExist: every `task <name>` a workflow runs has to be a target
// the Taskfile actually declares, or CI fails with a target-not-found the same
// way `make test` did.
func TestWorkflowTasksExist(t *testing.T) {
	tf, err := os.ReadFile("Taskfile.yaml")
	if err != nil {
		t.Fatalf("read Taskfile.yaml: %v", err)
	}
	declared := map[string]bool{}
	for _, m := range regexp.MustCompile(`(?m)^  ([a-z][a-z0-9:_-]*):`).FindAllStringSubmatch(string(tf), -1) {
		declared[m[1]] = true
	}
	if len(declared) == 0 {
		t.Fatal("parsed no targets from Taskfile.yaml; this guard is checking nothing")
	}

	invoke := regexp.MustCompile(`run:\s+task\s+([a-z][a-z0-9:_-]*)`)
	for _, f := range workflowFiles(t) {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for _, m := range invoke.FindAllStringSubmatch(string(b), -1) {
			if !declared[m[1]] {
				t.Errorf("%s runs `task %s`, which Taskfile.yaml does not declare", f, m[1])
			}
		}
	}
}

func workflowFiles(t *testing.T) []string {
	t.Helper()
	dir := filepath.Join(".github", "workflows")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	var out []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".yml") || strings.HasSuffix(e.Name(), ".yaml") {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	if len(out) == 0 {
		t.Fatalf("no workflow files in %s; this guard is checking nothing", dir)
	}
	return out
}
