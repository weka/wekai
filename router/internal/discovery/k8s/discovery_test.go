package k8s_test

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	k8sdisc "github.com/weka/wekai/router/internal/discovery/k8s"
	"github.com/weka/wekai/router/internal/registry"
)

const ns = "inference"

func ptr[T any](v T) *T { return &v }

func slice(name string, port int32, addrs ...string) *discoveryv1.EndpointSlice {
	eps := make([]discoveryv1.Endpoint, 0, len(addrs))
	for _, a := range addrs {
		eps = append(eps, discoveryv1.Endpoint{
			Addresses:  []string{a},
			Conditions: discoveryv1.EndpointConditions{Ready: ptr(true)},
		})
	}
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    map[string]string{discoveryv1.LabelServiceName: "vllm"},
		},
		AddressType: discoveryv1.AddressTypeIPv4,
		Ports:       []discoveryv1.EndpointPort{{Name: ptr("http"), Port: ptr(port)}},
		Endpoints:   eps,
	}
}

// run starts a Discoverer and waits until the registry reaches want, so tests do
// not race the informer.
func run(t *testing.T, cfg k8sdisc.Config, objs []runtimeObject, want []string) (*registry.Registry, context.CancelFunc) {
	t.Helper()
	var conv []any
	for _, o := range objs {
		conv = append(conv, o)
	}
	client := fake.NewSimpleClientset(toRuntime(conv)...)
	reg := registry.New(registry.Options{})

	d, err := k8sdisc.New(cfg, client, reg, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = d.Run(ctx) }()

	waitFor(t, reg, want)
	return reg, cancel
}

func waitFor(t *testing.T, reg *registry.Registry, want []string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	var got []string
	for time.Now().Before(deadline) {
		got = urls(reg)
		if slices.Equal(got, want) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("registry never reached the expected set:\n got %v\nwant %v", got, want)
}

func urls(reg *registry.Registry) []string {
	snap := reg.Snapshot()
	out := make([]string, 0, len(snap.Backends))
	for _, b := range snap.Backends {
		if !b.Draining() {
			out = append(out, b.URL)
		}
	}
	return out
}

func TestEndpointSliceDiscovery(t *testing.T) {
	cfg := k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm", PortName: "http"}
	_, cancel := run(t, cfg,
		[]runtimeObject{slice("vllm-abc", 8000, "10.0.0.2", "10.0.0.1")},
		[]string{"http://10.0.0.1:8000", "http://10.0.0.2:8000"}, // sorted by the registry
	)
	defer cancel()
}

// TestApplySameSetTwiceIsIdentical guards SD-N2. v1 reconciled into additive
// secondary indexes, so a repeated apply duplicated URLs and weighted one worker
// N times in selection. Full reconciliation makes this a no-op by construction.
func TestApplySameSetTwiceIsIdentical(t *testing.T) {
	reg := registry.New(registry.Options{})
	desired := []registry.Spec{
		{URL: "http://10.0.0.2:8000"},
		{URL: "http://10.0.0.1:8000"},
	}
	var first []string
	for pass := 0; pass < 5; pass++ {
		if _, err := reg.ReconcileDiscovered(desired); err != nil {
			t.Fatal(err)
		}
		got := urls(reg)
		if first == nil {
			first = got
			continue
		}
		if !slices.Equal(got, first) {
			t.Fatalf("pass %d changed the registry:\n%v\n%v", pass, first, got)
		}
	}
	if len(first) != 2 {
		t.Fatalf("registry holds %d backends after repeated apply, want 2: %v", len(first), first)
	}
}

// SD-N3: a discovered endpoint must not be routable until its first successful
// health check, or the router sends traffic to a pod that has not loaded its model.
func TestDiscoveredEndpointIsNotRoutableUntilHealthy(t *testing.T) {
	cfg := k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm", PortName: "http"}
	reg, cancel := run(t, cfg,
		[]runtimeObject{slice("vllm-abc", 8000, "10.0.0.1")},
		[]string{"http://10.0.0.1:8000"},
	)
	defer cancel()

	snap := reg.Snapshot()
	b := snap.Backends[0]
	if b.Health() != registry.Unknown {
		t.Errorf("health = %v, want unknown on discovery", b.Health())
	}
	if b.Available() {
		t.Error("discovered endpoint is routable before its first health check")
	}
	if n := len(snap.Available()); n != 0 {
		t.Errorf("available candidates = %d, want 0", n)
	}
}

func TestNotReadyAndTerminatingEndpointsExcluded(t *testing.T) {
	es := slice("vllm-abc", 8000, "10.0.0.1")
	es.Endpoints = append(es.Endpoints,
		discoveryv1.Endpoint{
			Addresses:  []string{"10.0.0.2"},
			Conditions: discoveryv1.EndpointConditions{Ready: ptr(false)},
		},
		discoveryv1.Endpoint{
			Addresses: []string{"10.0.0.3"},
			Conditions: discoveryv1.EndpointConditions{
				Ready: ptr(true), Terminating: ptr(true),
			},
		},
	)
	cfg := k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm", PortName: "http"}
	_, cancel := run(t, cfg, []runtimeObject{es}, []string{"http://10.0.0.1:8000"})
	defer cancel()
}

// SD-9: IPv6 addresses must be bracketed, or every URL is malformed.
func TestIPv6AddressesAreBracketed(t *testing.T) {
	es := slice("vllm-v6", 8000, "2001:db8::1")
	es.AddressType = discoveryv1.AddressTypeIPv6
	cfg := k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm", PortName: "http"}
	_, cancel := run(t, cfg, []runtimeObject{es}, []string{"http://[2001:db8::1]:8000"})
	defer cancel()
}

// A configured port name that does not appear on the slice is a configuration
// error. Routing to an arbitrary other port would be worse than routing nowhere.
func TestMissingNamedPortYieldsNoBackends(t *testing.T) {
	cfg := k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm", PortName: "grpc"}
	_, cancel := run(t, cfg, []runtimeObject{slice("vllm-abc", 8000, "10.0.0.1")}, nil)
	defer cancel()
}

func TestLabelsAndAnnotationsAreHonoured(t *testing.T) {
	es := slice("vllm-abc", 8000, "10.0.0.1")
	es.Labels[k8sdisc.LabelKind] = "router"
	es.Labels[k8sdisc.LabelDialect] = "openai"
	es.Labels[k8sdisc.LabelModel] = "llama-3-70b"
	es.Annotations = map[string]string{k8sdisc.AnnotationCapacity: "40"}

	cfg := k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm", PortName: "http"}
	reg, cancel := run(t, cfg, []runtimeObject{es}, []string{"http://10.0.0.1:8000"})
	defer cancel()

	b := reg.Snapshot().Backends[0]
	if b.Kind() != registry.KindRouter {
		t.Errorf("kind = %v, want router", b.Kind())
	}
	if b.Model() != "llama-3-70b" {
		t.Errorf("model = %q", b.Model())
	}
	if b.Capacity() != 40 {
		t.Errorf("capacity = %d, want 40 from the annotation", b.Capacity())
	}
}

func TestPodModeDiscovery(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vllm-0", Namespace: ns,
			Labels: map[string]string{"app": "vllm"},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning, PodIP: "10.1.0.5",
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
	notReady := pod.DeepCopy()
	notReady.Name = "vllm-1"
	notReady.Status.PodIP = "10.1.0.6"
	notReady.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}}

	cfg := k8sdisc.Config{Mode: k8sdisc.ModePod, Namespace: ns, Selector: "app=vllm", Port: 8000}
	_, cancel := run(t, cfg, []runtimeObject{pod, notReady}, []string{"http://10.1.0.5:8000"})
	defer cancel()
}

// HIER-19: discovery may never override or delete a statically configured
// backend. With topology coming from both config and discovery, this is what
// stops a reconcile pass from deleting a pinned cross-cluster backend.
func TestStaticBackendSurvivesDiscovery(t *testing.T) {
	reg := registry.New(registry.Options{})
	st, err := reg.Add(registry.Spec{
		URL: "http://10.0.0.1:8000", Prov: registry.ProvStatic, Model: "pinned",
	})
	if err != nil {
		t.Fatal(err)
	}

	conflicts, err := reg.ReconcileDiscovered([]registry.Spec{
		{URL: "http://10.0.0.1:8000", Model: "discovered"},
		{URL: "http://10.0.0.9:8000"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(conflicts, []string{"http://10.0.0.1:8000"}) {
		t.Fatalf("conflicts = %v, want the static URL reported", conflicts)
	}
	if st.Model() != "pinned" {
		t.Errorf("static backend was overwritten: model = %q", st.Model())
	}

	// A later pass that omits it must not drain it either.
	if _, err := reg.ReconcileDiscovered([]registry.Spec{{URL: "http://10.0.0.9:8000"}}); err != nil {
		t.Fatal(err)
	}
	if st.Draining() {
		t.Error("discovery drained a statically configured backend")
	}
}

func TestConfigValidation(t *testing.T) {
	cases := []struct {
		name string
		cfg  k8sdisc.Config
		ok   bool
	}{
		{"endpointslice ok", k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns, Service: "vllm"}, true},
		{"endpointslice without service", k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Namespace: ns}, false},
		{"pod ok", k8sdisc.Config{Mode: k8sdisc.ModePod, Namespace: ns, Selector: "app=vllm", Port: 8000}, true},
		{"pod without selector", k8sdisc.Config{Mode: k8sdisc.ModePod, Namespace: ns, Port: 8000}, false},
		{"pod with bad selector", k8sdisc.Config{Mode: k8sdisc.ModePod, Namespace: ns, Selector: "!!!", Port: 8000}, false},
		{"pod without port", k8sdisc.Config{Mode: k8sdisc.ModePod, Namespace: ns, Selector: "app=vllm"}, false},
		{"no namespace", k8sdisc.Config{Mode: k8sdisc.ModeEndpointSlice, Service: "vllm"}, false},
		{"unknown mode", k8sdisc.Config{Mode: "magic", Namespace: ns}, false},
	}
	for _, c := range cases {
		_, err := k8sdisc.New(c.cfg, fake.NewSimpleClientset(), registry.New(registry.Options{}), nil)
		if c.ok && err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
		}
		if !c.ok && err == nil {
			t.Errorf("%s: expected a validation error", c.name)
		}
	}
}

func TestRemovedEndpointIsDrainedNotDeleted(t *testing.T) {
	reg := registry.New(registry.Options{})
	if _, err := reg.ReconcileDiscovered([]registry.Spec{
		{URL: "http://10.0.0.1:8000"}, {URL: "http://10.0.0.2:8000"},
	}); err != nil {
		t.Fatal(err)
	}
	b, ok := reg.Snapshot().Get("http://10.0.0.2:8000")
	if !ok {
		t.Fatal("setup failed")
	}
	b.AddInflight(1) // a request is in flight
	defer b.AddInflight(-1)

	if _, err := reg.ReconcileDiscovered([]registry.Spec{{URL: "http://10.0.0.1:8000"}}); err != nil {
		t.Fatal(err)
	}
	if !b.Draining() {
		t.Error("removed endpoint is not draining")
	}
	if b.Available() {
		t.Error("draining endpoint still accepts new traffic")
	}
	if _, still := reg.Snapshot().Get("http://10.0.0.2:8000"); !still {
		t.Error("endpoint was hard-deleted while a request was in flight")
	}
}

func BenchmarkReconcile256(b *testing.B) {
	reg := registry.New(registry.Options{})
	desired := make([]registry.Spec, 256)
	for i := range desired {
		desired[i] = registry.Spec{URL: fmt.Sprintf("http://10.0.%d.%d:8000", i/256, i%256)}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := reg.ReconcileDiscovered(desired); err != nil {
			b.Fatal(err)
		}
	}
}

// TestPodPortComesFromThePodNotTheFlag covers what a Service cannot express and
// why pod discovery exists: a fleet run as several DaemonSets, one per GPU
// topology or model, each listening on a different port. A Service maps ONE
// port, so covering them needs one Service per set — and the router loses the
// single label selector that made them one pool.
func TestPodPortComesFromThePodNotTheFlag(t *testing.T) {
	mk := func(name, ip string, port int32) *corev1.Pod {
		p := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns,
				Labels: map[string]string{"app": "vllm"}},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning, PodIP: ip,
				Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
			},
		}
		if port > 0 {
			p.Spec.Containers = []corev1.Container{{Ports: []corev1.ContainerPort{
				{Name: "http", ContainerPort: port}}}}
		}
		return p
	}

	// Three DaemonSets, three ports, ONE label selector. The configured Port is
	// only the floor for a pod that declares nothing.
	_, cancel := run(t, k8sdisc.Config{
		Mode: k8sdisc.ModePod, Namespace: ns, Selector: "app=vllm",
		Port: 8000, Scheme: "http",
	}, []runtimeObject{
		mk("set-a-0", "10.1.0.1", 8001),
		mk("set-b-0", "10.1.0.2", 8002),
		mk("set-c-0", "10.1.0.3", 0), // declares none: falls back to the flag
	}, []string{
		"http://10.1.0.1:8001",
		"http://10.1.0.2:8002",
		"http://10.1.0.3:8000",
	})
	cancel()
}
