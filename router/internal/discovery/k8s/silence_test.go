package k8s_test

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/client-go/kubernetes/fake"

	k8sdisc "github.com/weka/wekai/router/internal/discovery/k8s"
	"github.com/weka/wekai/router/internal/registry"
)

// A selector that matches nothing has to say so.
//
// This is taken from a live incident: a router was configured with the label
// key `wllm-pool` while the fleet was labelled `bench/pool`, so discovery
// matched zero pods. The pool sat empty, the pod never became ready, and the
// only visible symptom was a kubelet reporting "Readiness probe failed: HTTP
// probe failed with statuscode: 503". Nothing in the router's logs mentioned
// the selector, the namespace, or that the count was zero — it said
// "kubernetes discovery synced" and stopped there. The diagnosis took a walk
// across three components to reach a typo the router could have named at
// startup.
func TestEmptyDiscoveryIsReported(t *testing.T) {
	var logs syncBuf
	log := slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))

	reg := registry.New(registry.Options{})
	client := fake.NewSimpleClientset() // no pods at all
	d, err := k8sdisc.New(k8sdisc.Config{
		Mode: k8sdisc.ModePod, Namespace: "default", Selector: "wllm-pool=typo",
		Port: 8000, Scheme: "http",
	}, client, reg, log)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = d.Run(ctx) }()

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && !strings.Contains(logs.String(), "matched NO endpoints") {
		time.Sleep(10 * time.Millisecond)
	}
	out := logs.String()
	if !strings.Contains(out, "matched NO endpoints") {
		t.Fatalf("discovery matched nothing and said nothing about it; the only symptom "+
			"an operator gets is a 503 readiness probe:\n%s", out)
	}
	// The two facts needed to find the mistake without leaving the log line.
	for _, want := range []string{"wllm-pool=typo", "default"} {
		if !strings.Contains(out, want) {
			t.Errorf("the warning omits %q, so it does not say what to compare against "+
				"the pods' labels:\n%s", want, out)
		}
	}
}

type syncBuf struct {
	mu  sync.Mutex
	buf strings.Builder
}

func (b *syncBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuf) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
