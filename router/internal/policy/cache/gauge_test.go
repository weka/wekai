package cache_test

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	cachepolicy "github.com/weka/wekai/router/internal/policy/cache"
	"github.com/weka/wekai/router/internal/registry"
)

func TestPublishGaugesReflectsCommits(t *testing.T) {
	r := registry.New(registry.Options{})
	b, _ := r.Add(registry.Spec{URL: "http://g:8000", Capacity: 1})
	p := cachepolicy.New(cachepolicy.DefaultConfig(), policy.LeastOutstanding{})
	p.AddBackend(b)
	p.Commit(b, &policy.RoutingRequest{Units: []kvcache.Unit{
		{Hash: 1, Tokens: 10}, {Hash: 2, Tokens: 20},
	}})
	p.PublishGauges()
	if got := testutil.ToFloat64(metrics.CacheEntries.WithLabelValues("http://g:8000")); got != 2 {
		t.Fatalf("cache_entries = %v, want 2 — the gauge does not reflect commits", got)
	}
	if got := testutil.ToFloat64(metrics.CacheTokens.WithLabelValues("http://g:8000")); got != 30 {
		t.Fatalf("cache_tokens = %v, want 30", got)
	}
}
