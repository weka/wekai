// Package chart holds render tests for the wekai Helm chart. There is no Go
// code here — the tests shell out to `helm template` and assert on the rendered
// manifests, which is the only way to cover template logic. They skip when helm
// isn't installed, matching how the node-based report tests behave.
package chart

import (
	"os/exec"
	"strings"
	"testing"
)

func render(t *testing.T, args ...string) string {
	t.Helper()
	helm, err := exec.LookPath("helm")
	if err != nil {
		t.Skip("helm not installed; chart render test skipped")
	}
	// The chart lives in ./wekai; this file deliberately sits OUTSIDE it so it
	// is not swept into the packaged chart.
	base := []string{"template", "t", "wekai", "--set", "endpoint=http://x:8000"}
	out, err := exec.Command(helm, append(base, args...)...).CombinedOutput()
	if err != nil {
		t.Fatalf("helm template failed: %v\n%s", err, out)
	}
	return string(out)
}

// saveRequestDataArg extracts the --save-request-data value from the rendered
// pod command.
func saveRequestDataArg(t *testing.T, manifest string) string {
	t.Helper()
	const flag = "--save-request-data="
	i := strings.Index(manifest, flag)
	if i < 0 {
		t.Fatal("rendered manifest has no --save-request-data flag")
	}
	rest := manifest[i+len(flag):]
	return strings.TrimSpace(strings.SplitN(rest, "\n", 2)[0])
}

// TestResultsDataPath covers where run data lands. The same PVC is normally
// shared across purposes, so wekai writes into a subdirectory rather than
// dropping run directories loose at the volume root.
//
// The nil-vs-empty distinction is the point: Helm's `default` collapses them,
// but here they must differ — an EXPLICIT empty string selects the flat layout,
// while an absent or null key keeps the default.
func TestResultsDataPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"default", nil, "/results/wekai-requests-data"},
		{"explicit empty writes at the volume root", []string{"--set", "resultsSubPath="}, "/results"},
		{"null falls back to the default", []string{"--set", "resultsSubPath=null"}, "/results/wekai-requests-data"},
		{"custom sub-path", []string{"--set", "resultsSubPath=teamA/soak"}, "/results/teamA/soak"},
		{"stray slashes are normalised", []string{"--set", "resultsSubPath=/teamA/soak/"}, "/results/teamA/soak"},
		{"custom mount path", []string{"--set", "resultsMountPath=/mnt/weka", "--set", "resultsSubPath=runs"}, "/mnt/weka/runs"},
		{"mount path trailing slash", []string{"--set", "resultsMountPath=/mnt/weka/", "--set", "resultsSubPath=runs"}, "/mnt/weka/runs"},
		{"both empty", []string{"--set", "resultsMountPath=/mnt/weka", "--set", "resultsSubPath="}, "/mnt/weka"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := saveRequestDataArg(t, render(t, tc.args...)); got != tc.want {
				t.Errorf("--save-request-data = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestResultsVolumeMountIsNotSubPath: the whole volume must stay mounted at
// resultsMountPath. Using a volumeMount subPath would hide the rest of a shared
// PVC from kubectl exec/cp, which is the opposite of what namespacing wekai's
// own writes is for.
func TestResultsVolumeMountIsNotSubPath(t *testing.T) {
	manifest := render(t)
	i := strings.Index(manifest, "- name: results")
	if i < 0 {
		t.Fatal("no results volumeMount in the rendered manifest")
	}
	block := manifest[i:min(i+300, len(manifest))]
	if strings.Contains(block, "subPath:") {
		t.Errorf("results volumeMount uses subPath, which hides the rest of a shared PVC:\n%s", block)
	}
	if !strings.Contains(block, "mountPath: /results") {
		t.Errorf("results volume should mount the whole volume at /results:\n%s", block)
	}
}

// TestResultsSubPathAppliesToEphemeralToo: the layout must not depend on which
// storage backs the path, so a run is reproducible whether or not a PVC is set.
func TestResultsSubPathAppliesToEphemeralToo(t *testing.T) {
	ephemeral := render(t)
	if !strings.Contains(ephemeral, "emptyDir: {}") {
		t.Fatal("expected an emptyDir when no claim is configured")
	}
	withPVC := render(t, "--set", "resultsClaim=shared-scratch")
	if !strings.Contains(withPVC, "claimName: shared-scratch") {
		t.Fatal("expected the configured claim to be mounted")
	}
	if a, b := saveRequestDataArg(t, ephemeral), saveRequestDataArg(t, withPVC); a != b {
		t.Errorf("data path differs by backing store: emptyDir %q vs PVC %q", a, b)
	}
}

// TestSleepMessagePointsAtTheData: the pod parks after the run so results can be
// pulled off, and the message has to name the directory they are actually in.
func TestSleepMessagePointsAtTheData(t *testing.T) {
	manifest := render(t)
	want := saveRequestDataArg(t, manifest)
	if !strings.Contains(manifest, "results under "+want+" stay explorable") {
		t.Errorf("sleep message does not point at %q", want)
	}
}

// renderRouter templates the router chart. It is a separate chart from
// ./wekai for a reason worth stating: the benchmark image embeds a multi-GB
// replay artifact, and a router that never reads it should not pay to pull it.
func renderRouter(t *testing.T, args ...string) string {
	t.Helper()
	helm, err := exec.LookPath("helm")
	if err != nil {
		t.Skip("helm not installed; chart render test skipped")
	}
	base := []string{"template", "t", "router"}
	out, err := exec.Command(helm, append(base, args...)...).CombinedOutput()
	if err != nil {
		t.Fatalf("helm template failed: %v\n%s", err, out)
	}
	return string(out)
}

// TestRouterChartUsesTheReplaylessImage is the tripwire for the mistake this
// split exists to prevent: pointing the router chart back at the benchmark
// image would silently add gigabytes to every router pull.
func TestRouterChartUsesTheReplaylessImage(t *testing.T) {
	out := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000")
	if !strings.Contains(out, "quay.io/weka.io/wekai-router:") {
		t.Errorf("rendered router deployment does not use the wekai-router image:\n%s", out)
	}
	if strings.Contains(out, "quay.io/weka.io/wekai:") ||
		strings.Contains(out, "wekai-benchmark") {
		t.Errorf("router chart references a replay-carrying image:\n%s", out)
	}
}

// TestRouterChartRunsTheSubcommand: the router is `wekai router serve`, not a
// standalone binary. If the args stop saying so, the pod runs the image's
// default and serves nothing.
func TestRouterChartRunsTheSubcommand(t *testing.T) {
	out := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000")
	for _, want := range []string{`- "router"`, `- "serve"`} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered args missing %s:\n%s", want, out)
		}
	}
}

// TestRouterChartRBACIsOptedInto: discovery needs a ServiceAccount, Role and
// RoleBinding, and granting them unconditionally would hand every router
// deployment namespace read it does not use. The pod rule is gated separately
// again, because pod-mode discovery is the only thing that needs it.
func TestRouterChartRBACIsOptedInto(t *testing.T) {
	off := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000")
	for _, kind := range []string{"kind: Role", "kind: RoleBinding", "kind: ServiceAccount"} {
		if strings.Contains(off, kind) {
			t.Errorf("%s is created with discovery disabled; RBAC must be opted into", kind)
		}
	}

	on := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000",
		"--set", "discovery.enabled=true")
	for _, kind := range []string{"kind: Role", "kind: RoleBinding", "kind: ServiceAccount"} {
		if !strings.Contains(on, kind) {
			t.Errorf("%s missing when discovery is enabled:\n%s", kind, on)
		}
	}
	if strings.Contains(on, "kind: ClusterRole") {
		t.Error("discovery is namespace-scoped; a ClusterRole grants more than it needs")
	}
	if strings.Contains(on, `resources: ["pods"]`) {
		t.Error("pod read granted in endpointslice mode, which does not use it")
	}

	podMode := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000",
		"--set", "discovery.enabled=true", "--set", "discovery.mode=pod")
	if !strings.Contains(podMode, `resources: ["pods"]`) {
		t.Error("pod-mode discovery did not grant pod read, so it cannot work")
	}
}
