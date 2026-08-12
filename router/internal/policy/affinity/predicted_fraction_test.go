package affinity

import (
	"math"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/kvcache"
	"github.com/weka/wekai/router/internal/metrics"
)

// router_cache_predicted_fraction exists to be read against
// router_cache_observed_fraction, which comes from the backend's own
// usage.prompt_tokens_details.cached_tokens and is therefore a TOKEN share. The
// predicted side must be one too.
//
// It was a BLOCK share — matched blocks over total blocks — and the two were
// plotted on one Grafana panel. Blocks here are variable-sized, so on agentic
// traffic the two differ severalfold and the only closed loop the router has on
// whether prefix prediction works was comparing two different quantities.

// hist reads a histogram child's running sum and count, for delta assertions:
// the collectors are package-level and shared across the whole test binary, so
// absolute values are meaningless.
func hist(t *testing.T, o prometheus.Observer) (sum float64, count uint64) {
	t.Helper()
	m, ok := o.(prometheus.Metric)
	if !ok {
		t.Fatalf("observer %T is not a prometheus.Metric; cannot read it back", o)
	}
	var pb dto.Metric
	if err := m.Write(&pb); err != nil {
		t.Fatalf("read histogram: %v", err)
	}
	return pb.Histogram.GetSampleSum(), pb.Histogram.GetSampleCount()
}

// unevenUnits builds a request whose blocks differ wildly in size, which is the
// only shape that can tell a token fraction from a block fraction.
func unevenUnits(tokens ...int32) []kvcache.Unit {
	u := make([]kvcache.Unit, len(tokens))
	for i, tk := range tokens {
		u[i] = kvcache.Unit{Hash: 0xF00D0000 | uint64(i+1), Tokens: tk}
	}
	return u
}

func TestPredictedFractionIsTokenWeighted(t *testing.T) {
	p, _ := newTestPolicy(t)
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}

	// One fat leading block and three thin ones: 1000 of 1030 tokens, but only
	// 1 of 4 blocks. A block share would report 0.25 here; the truth is 0.97.
	u := unevenUnits(1000, 10, 10, 10)
	p.Commit(f[0], req(u[:1])) // f[0] holds the fat block, and nothing else

	obs := metrics.CachePredictedFraction.WithLabelValues(DefaultPoolName)
	sum0, n0 := hist(t, obs)

	got := route(t, p, f, req(u))
	if got != f[0] {
		t.Fatalf("routed to %s, want the holder %s — the fixture needs a tier-1 cache hit to "+
			"observe the metric at all", got.URL, f[0].URL)
	}

	sum1, n1 := hist(t, obs)
	if n1 != n0+1 {
		t.Fatalf("the routing decision recorded %d observations, want exactly 1", n1-n0)
	}
	observed := sum1 - sum0

	wantTokens := 1000.0 / 1030.0
	wantBlocks := 0.25
	if math.Abs(observed-wantBlocks) < 1e-9 {
		t.Fatalf("predicted fraction = %.4f, which is matched BLOCKS over total blocks. It is "+
			"plotted against router_cache_observed_fraction, a token share from the backend's "+
			"own usage accounting, so it has to be token-weighted: want %.4f", observed, wantTokens)
	}
	if math.Abs(observed-wantTokens) > 1e-6 {
		t.Errorf("predicted fraction = %.6f, want %.6f (1000 of 1030 tokens resident)",
			observed, wantTokens)
	}
}

// TestPredictedFractionMatchesCover ties the metric to the shared scorer rather
// than to a second copy of the arithmetic, so the router, the offline replay
// analyzer and the dashboards cannot drift apart.
func TestPredictedFractionMatchesCover(t *testing.T) {
	p, _ := newTestPolicy(t)
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}

	u := unevenUnits(40, 900, 30, 30)
	p.Commit(f[0], req(u[:2])) // two blocks held: 940 of 1000 tokens

	obs := metrics.CachePredictedFraction.WithLabelValues(DefaultPoolName)
	sum0, _ := hist(t, obs)
	if got := route(t, p, f, req(u)); got != f[0] {
		t.Fatalf("routed to %s, want the holder %s", got.URL, f[0].URL)
	}
	sum1, _ := hist(t, obs)

	want := kvcache.Cover(u, 2).TokenFraction()
	if got := sum1 - sum0; math.Abs(got-want) > 1e-6 {
		t.Errorf("predicted fraction = %.6f, want kvcache.Cover(units, 2).TokenFraction() = %.6f",
			got, want)
	}
}
