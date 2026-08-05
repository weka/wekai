package cache

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/wekai/router/internal/metrics"
)

func TestPublishPredictionStatsComputesAvgMaxMin(t *testing.T) {
	publishPredictionStats([]float64{1.0, 0.5, 0.0})

	if avg := testutil.ToFloat64(metrics.CachePredictionAvg); avg != 0.5 {
		t.Errorf("avg = %v, want 0.5", avg)
	}
	if max := testutil.ToFloat64(metrics.CachePredictionMax); max != 1.0 {
		t.Errorf("max = %v, want 1.0", max)
	}
	if min := testutil.ToFloat64(metrics.CachePredictionMin); min != 0.0 {
		t.Errorf("min = %v, want 0.0", min)
	}
}

// A cold call with nothing computable must leave the last real reading in
// place, not report a coincidental zero indistinguishable from "every
// candidate scored 0".
func TestPublishPredictionStatsNoopOnEmpty(t *testing.T) {
	publishPredictionStats([]float64{0.7, 0.3})
	before := testutil.ToFloat64(metrics.CachePredictionAvg)

	publishPredictionStats(nil)

	if after := testutil.ToFloat64(metrics.CachePredictionAvg); after != before {
		t.Errorf("avg changed from %v to %v on an empty slice; want unchanged", before, after)
	}
}
