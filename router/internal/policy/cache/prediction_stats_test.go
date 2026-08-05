package cache

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/wekai/router/internal/metrics"
)

// histogramSumCount reads a Histogram's cumulative _sum/_count. Histograms
// only accumulate (Observe never overwrites), and these are shared
// package-level collectors across every test in the package, so callers
// compare deltas rather than absolute values.
func histogramSumCount(t *testing.T, h prometheus.Histogram) (sum float64, count uint64) {
	t.Helper()
	var m dto.Metric
	if err := h.Write(&m); err != nil {
		t.Fatal(err)
	}
	return m.GetHistogram().GetSampleSum(), m.GetHistogram().GetSampleCount()
}

func TestPublishPredictionStatsComputesAvgMaxMin(t *testing.T) {
	avgSumBefore, avgCountBefore := histogramSumCount(t, metrics.CachePredictionAvg)
	maxSumBefore, maxCountBefore := histogramSumCount(t, metrics.CachePredictionMax)
	minSumBefore, minCountBefore := histogramSumCount(t, metrics.CachePredictionMin)

	publishPredictionStats([]float64{1.0, 0.5, 0.0})

	if sum, count := histogramSumCount(t, metrics.CachePredictionAvg); count != avgCountBefore+1 || sum != avgSumBefore+0.5 {
		t.Errorf("avg histogram: sum=%v count=%v, want sum=%v count=%v", sum, count, avgSumBefore+0.5, avgCountBefore+1)
	}
	if sum, count := histogramSumCount(t, metrics.CachePredictionMax); count != maxCountBefore+1 || sum != maxSumBefore+1.0 {
		t.Errorf("max histogram: sum=%v count=%v, want sum=%v count=%v", sum, count, maxSumBefore+1.0, maxCountBefore+1)
	}
	if sum, count := histogramSumCount(t, metrics.CachePredictionMin); count != minCountBefore+1 || sum != minSumBefore+0.0 {
		t.Errorf("min histogram: sum=%v count=%v, want sum=%v count=%v", sum, count, minSumBefore+0.0, minCountBefore+1)
	}
}

// A cold call with nothing computable must Observe nothing at all — no
// misleading zero recorded into the distribution.
func TestPublishPredictionStatsNoopOnEmpty(t *testing.T) {
	_, countBefore := histogramSumCount(t, metrics.CachePredictionAvg)

	publishPredictionStats(nil)

	if _, count := histogramSumCount(t, metrics.CachePredictionAvg); count != countBefore {
		t.Errorf("count changed from %v to %v on an empty slice; want unchanged", countBefore, count)
	}
}
