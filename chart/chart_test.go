// Package chart holds render tests for the wekai Helm chart. There is no Go
// code here — the tests shell out to `helm template` and assert on the rendered
// manifests, which is the only way to cover template logic. They skip when helm
// isn't installed, matching how the node-based report tests behave.
package chart

import (
	"os"
	"os/exec"
	"path/filepath"
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
// Pod discovery needs API access, and needing it is not a detail an operator
// should have to discover from a runtime permissions error that names nothing
// about the chart. RBAC is created by default, and can be declined.
//
// It stays a Role and never a ClusterRole: discovery only ever searches the
// router's own namespace, so cluster-wide pod read would be a standing
// privilege for a capability that does not exist.
func TestRouterChartRBACIsOnByDefaultAndNamespaced(t *testing.T) {
	out := renderRouter(t, "--set-string", "router.routes[0]=* => pods:app=vllm")
	for _, want := range []string{"kind: ServiceAccount", "kind: Role", "kind: RoleBinding"} {
		if !strings.Contains(out, want) {
			t.Errorf("%s is not created by default; a pods: route would fail at runtime "+
				"with a permissions error that names nothing about this chart", want)
		}
	}
	if strings.Contains(out, "kind: ClusterRole") {
		t.Error("chart creates a ClusterRole; discovery is namespace-scoped, so " +
			"cluster-wide pod read is a standing privilege for nothing")
	}
	if !strings.Contains(out, "automountServiceAccountToken: true") {
		t.Error("the API token is not mounted, so discovery has no credential")
	}

	off := renderRouter(t, "--set-string", "router.routes[0]=* => http://vllm:8000",
		"--set", "discovery.enabled=false")
	for _, unwanted := range []string{"kind: Role", "kind: RoleBinding"} {
		if strings.Contains(off, unwanted) {
			t.Errorf("%s is still created with discovery.enabled=false; a router with "+
				"static endpoints needs no API access at all", unwanted)
		}
	}
	if !strings.Contains(off, "automountServiceAccountToken: false") {
		t.Error("a router that does not discover should not hold a Kubernetes credential")
	}
}
func TestRouterChartPassesRoutesVerbatim(t *testing.T) {
	const route = "sonnet,big => http://a:8000|http://b:8000 as Qwen/Qwen3-32B"
	// A values FILE, not --set: helm's --set splits on commas, so a route with
	// several model patterns is truncated at the first one. That is a helm CLI
	// property rather than a chart defect, and it is why the docs tell operators
	// to put routes in values.yaml.
	dir := t.TempDir()
	vals := filepath.Join(dir, "values.yaml")
	if err := os.WriteFile(vals, []byte("router:\n  routes:\n    - \""+route+"\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	out := renderRouter(t, "-f", vals)
	if !strings.Contains(out, `- "--route=`+route+`"`) {
		t.Errorf("route did not survive templating verbatim; wanted %q in:\n%s", route, out)
	}
}

// TestRouterChartStructuredRoutesSurviveSet is the FIX for helm --set splitting
// on commas, not merely a note about it.
//
// A route written as one string is truncated at its first model pattern when
// given via --set: it renders without error and fails at runtime, which is the
// worst shape a config bug can take. Given as lists, no single --set value
// contains a comma, so the same route arrives intact.
func TestRouterChartStructuredRoutesSurviveSet(t *testing.T) {
	out := renderRouter(t,
		"--set", "router.routes[0].patterns[0]=fast",
		"--set", "router.routes[0].patterns[1]=small",
		"--set", "router.routes[0].endpoints[0]=http://a:8000",
		"--set", "router.routes[0].endpoints[1]=http://b:8000",
		"--set", "router.routes[0].as=Qwen/Qwen3-32B")
	const want = `- "--route=fast,small => http://a:8000|http://b:8000 as Qwen/Qwen3-32B"`
	if !strings.Contains(out, want) {
		t.Errorf("structured route did not assemble; wanted\n  %s\ngot:\n%s", want, out)
	}

	// The string form still works, so existing values files keep rendering.
	str := renderRouter(t, "--set-string", "router.routes[0]=* => http://only:8000")
	if !strings.Contains(str, `- "--route=* => http://only:8000"`) {
		t.Errorf("string-form route stopped working:\n%s", str)
	}
}

// TestRouterChartProbesPathsTheBinaryServes is the regression test for a
// CrashLoopBackOff found only by deploying.
//
// The chart probed /healthz and /livez, which the router does not serve. That
// alone would be a clean 404, but the router PROXIES any unclaimed path — so
// the probe was forwarded to a backend, answered 404 by something that knows
// nothing about it, and read as "the router is unhealthy". A probe must be
// answered by the router itself.
func TestRouterChartProbesPathsTheBinaryServes(t *testing.T) {
	out := renderRouter(t, "--set", "router.backends[0]=http://a:8000")
	for _, want := range []string{"path: /readiness", "path: /liveness"} {
		if !strings.Contains(out, want) {
			t.Errorf("probe %q missing; a probe on a path the router does not serve is "+
				"proxied to a backend and fails:\n%s", want, out)
		}
	}
	for _, bad := range []string{"path: /healthz", "path: /livez"} {
		if strings.Contains(out, bad) {
			t.Errorf("chart still probes %s", bad)
		}
	}
}

// TestRouterChartCaptureIsOffByDefault: capture writes to /data, and enabling
// it with nowhere writable crashed the pod at startup. Off unless asked for.
func TestRouterChartCaptureIsOffByDefault(t *testing.T) {
	out := renderRouter(t, "--set", "router.backends[0]=http://a:8000")
	if strings.Contains(out, "--capture=") {
		t.Errorf("capture is enabled by default; it writes to /data and needs a volume:\n%s", out)
	}
	if strings.Contains(out, "mountPath: /data") {
		t.Error("a /data volume is mounted with no capture and no PVC, for nothing to write")
	}
}

// TestRouterChartCaptureGetsWritableStorage: with capture on and no PVC the
// chart must mount an emptyDir, which its own comment always claimed it did.
// Records are then lost with the pod — acceptable for a test, which is why
// capture defaults off rather than silently looking like it retains data.
func TestRouterChartCaptureGetsWritableStorage(t *testing.T) {
	noPVC := renderRouter(t,
		"--set", "router.backends[0]=http://a:8000",
		"--set", "router.capture=redacted")
	if !strings.Contains(noPVC, "emptyDir: {}") || !strings.Contains(noPVC, "mountPath: /data") {
		t.Errorf("capture without a PVC has nowhere writable, so the pod crashes on "+
			"startup:\n%s", noPVC)
	}

	withPVC := renderRouter(t,
		"--set", "router.backends[0]=http://a:8000",
		"--set", "router.capture=redacted",
		"--set", "datastore.sharedPvc.enabled=true")
	if !strings.Contains(withPVC, "persistentVolumeClaim:") {
		t.Errorf("capture with a PVC enabled did not use it:\n%s", withPVC)
	}
	if strings.Contains(withPVC, "emptyDir: {}") {
		t.Error("a PVC was requested but an emptyDir was mounted; capture would not survive the pod")
	}
}

// TestRouterChartDiscoveryFlags: pod discovery is configured through the chart,
// and it is the case a Service cannot cover — several DaemonSets on different
// Everything about a discovered pool travels in the route string. The port
// used to be a router-wide --discover-port flag, which by construction could
// not describe a router fronting two fleets on different ports — the very case
// pod discovery exists for.
func TestRouterChartDiscoveryTravelsInTheRoute(t *testing.T) {
	// The selector is a MAP, which is the only form --set can express: a label
	// selector is comma-separated, and --set splits on commas, so the string
	// form silently arrives truncated at its first label.
	out := renderRouter(t,
		"--set-string", "router.routes[0].patterns[0]=fast",
		"--set-string", "router.routes[0].pods.app=vllm",
		"--set-string", "router.routes[0].pods.tier=prod",
		"--set-string", "router.routes[0].port=http",
		// A migration: static and discovered endpoints in one pool.
		"--set-string", "router.routes[1].patterns[0]=slow",
		"--set-string", "router.routes[1].endpoints[0]=http://legacy:8000",
		"--set-string", "router.routes[1].pods.app=vllm-cpu",
		"--set-string", "router.routes[1].port=9000")
	for _, want := range []string{
		`- "--route=fast => pods:app=vllm,tier=prod:http"`,
		`- "--route=slow => http://legacy:8000|pods:app=vllm-cpu:9000"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered manifest missing %q:\n%s", want, out)
		}
	}
	// The flags these replaced must be gone, not merely unused: leaving them
	// renderable would give two ways to say one thing and let them disagree.
	for _, gone := range []string{"--discover-namespace", "--discover-port", "--discover-port-name"} {
		if strings.Contains(out, gone) {
			t.Errorf("chart still emits %s; discovery config belongs in the route", gone)
		}
	}
}
func TestRouterChartCredentials(t *testing.T) {
	out := renderRouter(t,
		"--set-string", "router.routes[0].patterns[0]=llama",
		"--set-string", "router.routes[0].endpoints[0]=http://inner:8080",
		"--set-string", "router.routes[0].using=/etc/router/inner-key",
		"--set-string", "router.routes[1].patterns[0]=*",
		"--set-string", "router.routes[1].endpoints[0]=https://api.anthropic.com",
		"--set-string", "router.routes[1].using=client",
		"--set", "router.secretMounts[0].name=inner-key",
		"--set", "router.secretMounts[0].secretName=router-inner-key",
		"--set", "router.secretMounts[0].mountPath=/etc/router",
		"--set", "router.apiKeyFile=/etc/router/inbound")

	for _, want := range []string{
		`- "--route=llama => http://inner:8080 using /etc/router/inner-key"`,
		`- "--route=* => https://api.anthropic.com using client"`,
		`- "--api-key-file=/etc/router/inbound"`,
		"secretName: router-inner-key",
		"mountPath: /etc/router",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered manifest missing %q:\n%s", want, out)
		}
	}
	// A secret set as a value lands in the Deployment spec, readable by anyone
	// who can describe the pod; the file form must win when both are given.
	if strings.Contains(out, `--api-key=`) {
		t.Error("apiKeyFile given but --api-key also rendered, putting the secret in the pod spec")
	}
}

// TestRouterChartShipsNoDefaultLimits guards a trap rather than a bug.
//
// Shipping default limits alongside default requests makes the natural
// override — setting only resources.requests, which is what an operator sizing
// a pod actually does — a hard rejection: Helm MERGES the maps, so the chart's
// old limits survive underneath the new requests and the pod is refused for
// requesting more than it is allowed. The API error names the limit, not the
// merge, so nothing in it points at the chart.
func TestRouterChartShipsNoDefaultLimits(t *testing.T) {
	out := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000")
	if strings.Contains(out, "limits:") {
		t.Errorf("router chart ships default resource limits; a requests-only override "+
			"then merges into requests above limits and the pod is rejected:\n%s", out)
	}
	// Requests are still expressed, because they are the scheduling floor.
	for _, want := range []string{"requests:", `cpu: "4"`, "memory: 8Gi"} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered deployment is missing %q from its resource requests:\n%s", want, out)
		}
	}
}

// Both halves must be overridable on their own, and either must be able to go
// away entirely.
func TestRouterChartResourcesAreIndependentlyOverridable(t *testing.T) {
	t.Run("requests only, limits stay absent", func(t *testing.T) {
		out := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000",
			"--set", "resources.requests.cpu=8")
		if !strings.Contains(out, "cpu: 8") {
			t.Errorf("cpu request override did not take:\n%s", out)
		}
		if !strings.Contains(out, "memory: 8Gi") {
			t.Errorf("overriding cpu dropped the default memory request; Helm merges "+
				"maps, so the untouched key must survive:\n%s", out)
		}
		if strings.Contains(out, "limits:") {
			t.Errorf("a requests-only override introduced limits:\n%s", out)
		}
	})

	t.Run("limits can be added", func(t *testing.T) {
		out := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000",
			"--set", "resources.limits.memory=16Gi")
		if !strings.Contains(out, "limits:") || !strings.Contains(out, "memory: 16Gi") {
			t.Errorf("limits override did not take:\n%s", out)
		}
	})

	t.Run("resources can be dropped entirely", func(t *testing.T) {
		out := renderRouter(t, "--set", "router.routes[0]=* => http://vllm:8000",
			"--set", "resources.requests=null")
		// The container's block specifically: the RBAC Role has a
		// `resources: ["pods"]` of its own, and matching that instead would
		// make this assertion permanently true for the wrong reason.
		if strings.Contains(out, "\n          resources:") {
			t.Errorf("emptying every section must omit the container resources block "+
				"rather than emit an empty one:\n%s", out)
		}
	})
}
