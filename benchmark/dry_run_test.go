package benchmark

import (
	"testing"
	"time"
)

func TestDryRunDurations(t *testing.T) {
	tests := []struct {
		name           string
		coldTokens     int
		warmTokens     int
		outputTokens   int
		coldTPS        int
		warmTPS        int
		outputTPS      int
		wantTTFT       time.Duration
		wantTotalExtra time.Duration // additional duration beyond ttft
	}{
		{
			name:       "1M cold at 1M tps = 1s ttft",
			coldTokens: 1_000_000,
			coldTPS:    1_000_000,
			wantTTFT:   time.Second,
		},
		{
			name:         "warm component adds to ttft",
			coldTokens:   500_000,
			warmTokens:   5_000_000,
			outputTokens: 100_000,
			coldTPS:      1_000_000,
			warmTPS:      10_000_000,
			outputTPS:    100_000,
			// ttft = 500k/1M + 5M/10M = 0.5s + 0.5s = 1s
			wantTTFT: time.Second,
			// total = ttft + 100k/100k = 1s + 1s = 2s
			wantTotalExtra: time.Second,
		},
		{
			name:         "output component adds to total beyond ttft",
			coldTokens:   1_000_000,
			outputTokens: 200_000,
			coldTPS:      1_000_000,
			outputTPS:    100_000,
			// ttft = 1M/1M = 1s
			wantTTFT: time.Second,
			// total = 1s + 200k/100k = 3s
			wantTotalExtra: 2 * time.Second,
		},
		{
			name:      "zero tokens = zero duration",
			coldTPS:   1_000_000,
			warmTPS:   10_000_000,
			outputTPS: 100_000,
			wantTTFT:  0,
		},
		{
			name:       "zero tps = zero duration",
			coldTokens: 1_000_000,
			coldTPS:    0,
			wantTTFT:   0,
		},
		{
			name:       "negative tps = zero duration",
			coldTokens: 1_000_000,
			coldTPS:    -1,
			wantTTFT:   0,
		},
		{
			name:         "mixed: some zero rates",
			coldTokens:   1_000_000,
			warmTokens:   5_000_000,
			outputTokens: 100_000,
			coldTPS:      1_000_000,
			warmTPS:      0,
			outputTPS:    100_000,
			// ttft = 1M/1M + 5M/0(skipped) = 1s
			wantTTFT: time.Second,
			// total = 1s + 100k/100k = 2s
			wantTotalExtra: time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTTFT, gotTotal := dryRunDurations(
				tt.coldTokens, tt.warmTokens, tt.outputTokens,
				tt.coldTPS, tt.warmTPS, tt.outputTPS,
			)
			if gotTTFT != tt.wantTTFT {
				t.Errorf("ttft = %v, want %v", gotTTFT, tt.wantTTFT)
			}
			wantTotal := tt.wantTTFT + tt.wantTotalExtra
			if gotTotal != wantTotal {
				t.Errorf("total = %v, want %v", gotTotal, wantTotal)
			}
		})
	}
}
