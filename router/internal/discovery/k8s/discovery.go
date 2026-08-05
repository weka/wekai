// Package k8s discovers backends from the Kubernetes API.
//
// Two v1 defects shape this design:
//
//   - v1 discovered only Pod IPs. That breaks the moment workers sit behind a
//     Service or a headless Service, and a pod IP reused by a new pod is
//     indistinguishable from the old one. EndpointSlice is the recommended mode
//     here (SD-1), with pod-label selection retained for parity (SD-2).
//   - v1 reconciled incrementally into additive secondary indexes, so applying
//     the same observed set twice duplicated URLs and weighted one worker N times
//     in selection. Here every event triggers a FULL reconcile of the desired set,
//     which makes double application a no-op by construction (SD-7, SD-N2).
//
// Discovery only ever proposes. The registry decides admission and health decides
// eligibility, so a freshly discovered pod is not routable until its first
// successful health check (SD-4, SD-N3).
package k8s

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	discoveryv1client "k8s.io/client-go/kubernetes/typed/discovery/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/registry"
)

// Labels and annotations a workload uses to describe itself to the router.
const (
	// LabelKind marks a backend as a leaf worker or a child router. Hierarchical
	// routing is deferred, but reading the label now costs nothing and keeps it
	// additive (HIER-1, HIER-16).
	LabelKind = "wllm.weka.io/backend-kind"
	// LabelDialect declares the wire format. Never probed (API-6).
	LabelDialect = "wllm.weka.io/backend-dialect"
	// LabelModel is the model identifier, used by the (currently no-op) model
	// candidate filter.
	LabelModel = "wllm.weka.io/model"
	// AnnotationCapacity overrides the backend's concurrency denominator.
	AnnotationCapacity = "wllm.weka.io/capacity"
)

type Mode string

const (
	// ModeEndpointSlice watches EndpointSlices for a named Service. Recommended.
	ModeEndpointSlice Mode = "endpointslice"
	// ModePod watches Pods matching a label selector, for v1 parity.
	ModePod Mode = "pod"
)

type Config struct {
	Mode      Mode
	Namespace string
	// Service is required for ModeEndpointSlice.
	Service string
	// Selector is required for ModePod, e.g. "app=vllm,tier=inference".
	Selector string
	// Port is the backend port. For EndpointSlice mode a named port on the slice
	// wins if PortName is set; otherwise this is used.
	Port     int
	PortName string
	Scheme   string
	// ResyncInterval bounds how long a missed event can go unnoticed. The
	// informer is watch-based, so this is a safety net rather than a poll loop.
	ResyncInterval time.Duration
	// DefaultCapacity applies when a workload does not annotate its own.
	DefaultCapacity int64
	// DefaultDialect applies when a workload does not label its own.
	DefaultDialect string
}

func (c Config) validate() error {
	switch c.Mode {
	case ModeEndpointSlice:
		if c.Service == "" {
			return fmt.Errorf("discovery: service name is required in %s mode", c.Mode)
		}
	case ModePod:
		if c.Selector == "" {
			return fmt.Errorf("discovery: selector is required in %s mode", c.Mode)
		}
		if _, err := labels.Parse(c.Selector); err != nil {
			return fmt.Errorf("discovery: invalid selector %q: %w", c.Selector, err)
		}
	default:
		return fmt.Errorf("discovery: unknown mode %q (want %s or %s)",
			c.Mode, ModeEndpointSlice, ModePod)
	}
	if c.Namespace == "" {
		return fmt.Errorf("discovery: namespace is required")
	}
	if c.Mode == ModePod && c.Port <= 0 {
		return fmt.Errorf("discovery: port is required in %s mode", c.Mode)
	}
	return nil
}

// Client is the narrow slice of the Kubernetes API this package uses.
//
// Deliberately not kubernetes.Interface: that interface is satisfied by
// *kubernetes.Clientset, whose package imports a typed client for every API group
// in Kubernetes. Depending on only the two groups we actually read keeps them out
// of the shipped binary. *kubernetes.Clientset and the fake clientset both satisfy
// this, so tests are unaffected.
type Client interface {
	DiscoveryV1() discoveryv1client.DiscoveryV1Interface
	CoreV1() corev1client.CoreV1Interface
}

// clients bundles the two typed clients the production path builds directly,
// avoiding kubernetes.Clientset entirely.
type clients struct {
	discovery discoveryv1client.DiscoveryV1Interface
	core      corev1client.CoreV1Interface
}

func (c clients) DiscoveryV1() discoveryv1client.DiscoveryV1Interface { return c.discovery }
func (c clients) CoreV1() corev1client.CoreV1Interface                { return c.core }

// Discoverer watches the API and reconciles the registry.
type Discoverer struct {
	cfg    Config
	client Client
	reg    *registry.Registry
	log    *slog.Logger
}

// NewInClusterOrKubeconfig builds a client from the in-cluster service account,
// falling back to a kubeconfig path for local development.
func NewInClusterOrKubeconfig(kubeconfig string) (Client, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		if kubeconfig == "" {
			return nil, fmt.Errorf("not running in-cluster and no kubeconfig given: %w", err)
		}
		cfg, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			return nil, fmt.Errorf("build kubeconfig: %w", err)
		}
	}
	dc, err := discoveryv1client.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("discovery.k8s.io client: %w", err)
	}
	cc, err := corev1client.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("core client: %w", err)
	}
	return clients{discovery: dc, core: cc}, nil
}

func New(cfg Config, client Client, reg *registry.Registry, log *slog.Logger) (*Discoverer, error) {
	if cfg.Mode == "" {
		cfg.Mode = ModeEndpointSlice
	}
	if cfg.Scheme == "" {
		cfg.Scheme = "http"
	}
	if cfg.ResyncInterval <= 0 {
		cfg.ResyncInterval = 5 * time.Minute
	}
	if cfg.DefaultCapacity <= 0 {
		cfg.DefaultCapacity = 1
	}
	if cfg.DefaultDialect == "" {
		cfg.DefaultDialect = "openai"
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if log == nil {
		log = slog.Default()
	}
	return &Discoverer{cfg: cfg, client: client, reg: reg, log: log}, nil
}

// Run watches until ctx is cancelled.
//
// A shared informer gives watch-expiry re-list with capped backoff for free, so
// there is no poll loop to get wrong (SD-3).
func (d *Discoverer) Run(ctx context.Context) error {
	// A targeted informer rather than informers.NewSharedInformerFactory.
	//
	// The generic factory is convenient but transitively imports a typed informer
	// for every resource in the Kubernetes API — 71 informer packages and 56 API
	// group packages, measured. We watch exactly one resource type, and the
	// binary size difference is the whole NFR-8 image budget.
	lw, objType := d.listerWatcher(ctx)
	informer := cache.NewSharedIndexInformer(lw, objType, d.cfg.ResyncInterval, cache.Indexers{})

	// Every event triggers a full reconcile from the informer's store rather than
	// an incremental patch. That is what makes repeated application idempotent.
	reconcile := func(any) { d.reconcileFrom(informer.GetStore()) }
	if _, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    reconcile,
		UpdateFunc: func(_, obj any) { reconcile(obj) },
		DeleteFunc: reconcile,
	}); err != nil {
		return fmt.Errorf("add event handler: %w", err)
	}

	go informer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), informer.HasSynced) {
		return fmt.Errorf("discovery: cache sync failed or context cancelled")
	}
	d.log.Info("kubernetes discovery synced",
		"mode", d.cfg.Mode, "namespace", d.cfg.Namespace,
		"service", d.cfg.Service, "selector", d.cfg.Selector)
	// Reconcile once after sync so a router that starts with an already-populated
	// cluster does not wait for the first event.
	d.reconcileFrom(informer.GetStore())

	<-ctx.Done()
	return nil
}

// listerWatcher builds a namespace- and label-scoped ListWatch for the configured
// mode, plus the object type the informer stores.
func (d *Discoverer) listerWatcher(ctx context.Context) (cache.ListerWatcher, runtime.Object) {
	if d.cfg.Mode == ModeEndpointSlice {
		api := d.client.DiscoveryV1().EndpointSlices(d.cfg.Namespace)
		return &cache.ListWatch{
			ListFunc: func(opts metav1.ListOptions) (runtime.Object, error) {
				d.tweak(&opts)
				return api.List(ctx, opts)
			},
			WatchFunc: func(opts metav1.ListOptions) (watch.Interface, error) {
				d.tweak(&opts)
				return api.Watch(ctx, opts)
			},
		}, &discoveryv1.EndpointSlice{}
	}
	api := d.client.CoreV1().Pods(d.cfg.Namespace)
	return &cache.ListWatch{
		ListFunc: func(opts metav1.ListOptions) (runtime.Object, error) {
			d.tweak(&opts)
			return api.List(ctx, opts)
		},
		WatchFunc: func(opts metav1.ListOptions) (watch.Interface, error) {
			d.tweak(&opts)
			return api.Watch(ctx, opts)
		},
	}, &corev1.Pod{}
}

func (d *Discoverer) tweak(opts *metav1.ListOptions) {
	switch d.cfg.Mode {
	case ModeEndpointSlice:
		opts.LabelSelector = discoveryv1.LabelServiceName + "=" + d.cfg.Service
	default:
		opts.LabelSelector = d.cfg.Selector
	}
}

// reconcileFrom builds the complete desired set and hands it to the registry.
func (d *Discoverer) reconcileFrom(store cache.Store) {
	var desired []registry.Spec
	switch d.cfg.Mode {
	case ModeEndpointSlice:
		for _, obj := range store.List() {
			es, ok := obj.(*discoveryv1.EndpointSlice)
			if !ok {
				continue
			}
			desired = append(desired, d.specsFromSlice(es)...)
		}
	default:
		for _, obj := range store.List() {
			pod, ok := obj.(*corev1.Pod)
			if !ok {
				continue
			}
			if s, ok := d.specFromPod(pod); ok {
				desired = append(desired, s)
			}
		}
	}

	conflicts, err := d.reg.ReconcileDiscovered(desired)
	if err != nil {
		d.log.Error("discovery reconcile failed", "err", err)
		return
	}
	for _, url := range conflicts {
		// A discovered endpoint colliding with a static backend is ignored, not
		// merged: static wins (HIER-19). Surfacing it is how an operator finds a
		// duplicated declaration.
		metrics.DiscoveryConflicts.WithLabelValues(url).Inc()
		d.log.Warn("discovered endpoint collides with a statically configured backend; ignoring",
			"backend", url)
	}
}

func (d *Discoverer) specsFromSlice(es *discoveryv1.EndpointSlice) []registry.Spec {
	port := d.portFromSlice(es)
	if port <= 0 {
		return nil
	}
	var out []registry.Spec
	for _, ep := range es.Endpoints {
		// Not-ready endpoints are excluded. Terminating ones are too: they are
		// draining and must not receive new traffic (SD-5).
		if ep.Conditions.Ready != nil && !*ep.Conditions.Ready {
			continue
		}
		if ep.Conditions.Terminating != nil && *ep.Conditions.Terminating {
			continue
		}
		for _, addr := range ep.Addresses {
			out = append(out, registry.Spec{
				// JoinHostPort brackets IPv6 correctly (SD-9).
				URL:       d.cfg.Scheme + "://" + net.JoinHostPort(addr, strconv.Itoa(port)),
				Kind:      kindFrom(es.Labels),
				DialectID: dialectFrom(es.Labels, d.cfg.DefaultDialect),
				Model:     es.Labels[LabelModel],
				Capacity:  capacityFrom(es.Annotations, d.cfg.DefaultCapacity),
			})
		}
	}
	return out
}

// portFromSlice prefers a named port when PortName is configured, so a Service
// exposing several ports routes to the right one.
func (d *Discoverer) portFromSlice(es *discoveryv1.EndpointSlice) int {
	if d.cfg.PortName != "" {
		for _, p := range es.Ports {
			if p.Name != nil && *p.Name == d.cfg.PortName && p.Port != nil {
				return int(*p.Port)
			}
		}
		// A configured name that does not appear is a configuration error, not a
		// reason to silently route to an arbitrary port.
		return 0
	}
	if d.cfg.Port > 0 {
		return d.cfg.Port
	}
	for _, p := range es.Ports {
		if p.Port != nil {
			return int(*p.Port)
		}
	}
	return 0
}

func (d *Discoverer) specFromPod(pod *corev1.Pod) (registry.Spec, bool) {
	if pod.Status.PodIP == "" || pod.DeletionTimestamp != nil {
		return registry.Spec{}, false
	}
	if pod.Status.Phase != corev1.PodRunning {
		return registry.Spec{}, false
	}
	ready := false
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			ready = true
		}
	}
	if !ready {
		return registry.Spec{}, false
	}
	return registry.Spec{
		URL:       d.cfg.Scheme + "://" + net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(d.cfg.Port)),
		Kind:      kindFrom(pod.Labels),
		DialectID: dialectFrom(pod.Labels, d.cfg.DefaultDialect),
		Model:     pod.Labels[LabelModel],
		Capacity:  capacityFrom(pod.Annotations, d.cfg.DefaultCapacity),
	}, true
}

func kindFrom(l map[string]string) registry.Kind {
	if l[LabelKind] == "router" {
		return registry.KindRouter
	}
	return registry.KindWorker
}

func dialectFrom(l map[string]string, def string) string {
	if v := l[LabelDialect]; v != "" {
		return v
	}
	return def
}

func capacityFrom(a map[string]string, def int64) int64 {
	if v := a[AnnotationCapacity]; v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return def
}
