package cli

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func TestResolveGarbageChars(t *testing.T) {
	tests := []struct {
		name              string
		garbageCharacters int
		garbageTokens     int
		wantChars         int
		wantWarningSubstr string // empty = no warning expected
	}{
		{
			name:      "neither set -> default garbage chars",
			wantChars: defaultGarbageChars,
		},
		{
			name:      "default constant is 213000 (preserves ~50k token budget with <ignore> filler)",
			wantChars: 213000,
		},
		{
			name:              "only --garbage-characters set",
			garbageCharacters: 250000,
			wantChars:         250000,
		},
		{
			name:              "only deprecated --garbage-tokens set -> *4 chars + deprecation warning",
			garbageTokens:     100000,
			wantChars:         400000,
			wantWarningSubstr: "deprecated",
		},
		{
			name:              "both set -> --garbage-characters wins + precedence warning",
			garbageCharacters: 250000,
			garbageTokens:     999,
			wantChars:         250000,
			wantWarningSubstr: "ignored",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			got := resolveGarbageChars(tt.garbageCharacters, tt.garbageTokens, &buf)
			if got != tt.wantChars {
				t.Errorf("resolveGarbageChars(%d, %d) = %d, want %d", tt.garbageCharacters, tt.garbageTokens, got, tt.wantChars)
			}
			warned := buf.Len() > 0
			wantWarn := tt.wantWarningSubstr != ""
			if warned != wantWarn {
				t.Errorf("resolveGarbageChars(%d, %d) warning output = %q, wantWarning=%v", tt.garbageCharacters, tt.garbageTokens, buf.String(), wantWarn)
			}
			if wantWarn && !strings.Contains(buf.String(), tt.wantWarningSubstr) {
				t.Errorf("resolveGarbageChars(%d, %d) warning = %q, want substring %q", tt.garbageCharacters, tt.garbageTokens, buf.String(), tt.wantWarningSubstr)
			}
		})
	}
}

// TestFindLeakedUUIDs moved to benchmark.FindLeakedUUIDs's own test
// (benchmark/replay_uuid_test.go) — the function itself moved to the
// benchmark package (benchmark/replay_uuid.go) so both this CLI and
// dataset-replay UUID validation share one implementation.

func TestColdWarmCycleCounts(t *testing.T) {
	tests := []struct {
		name       string
		cyclesSeen map[int]bool
		wantCold   int
		wantWarm   int
	}{
		{
			name:       "empty (no surviving requests)",
			cyclesSeen: map[int]bool{},
			wantCold:   0,
			wantWarm:   0,
		},
		{
			name:       "default 2-cycle run: cold=1 cycle, warm=1 cycle",
			cyclesSeen: map[int]bool{1: true, 2: true},
			wantCold:   1,
			wantWarm:   1,
		},
		{
			name:       "--total 16-cycle run: cold=1 cycle, warm=15 cycles",
			cyclesSeen: map[int]bool{1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true, 8: true, 9: true, 10: true, 11: true, 12: true, 13: true, 14: true, 15: true, 16: true},
			wantCold:   1,
			wantWarm:   15,
		},
		{
			name:       "cold only (a --total run too short to reach cycle 2)",
			cyclesSeen: map[int]bool{1: true},
			wantCold:   1,
			wantWarm:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCold, gotWarm := coldWarmCycleCounts(tt.cyclesSeen)
			if gotCold != tt.wantCold || gotWarm != tt.wantWarm {
				t.Errorf("coldWarmCycleCounts(%v) = (%d, %d), want (%d, %d)", tt.cyclesSeen, gotCold, gotWarm, tt.wantCold, tt.wantWarm)
			}
		})
	}
}

func TestFormatCycleDistribution(t *testing.T) {
	tests := []struct {
		name                      string
		coldCount, warmCount      int
		nCold, nWarm              int
		wantAbsCold, wantNormCold string
		wantAbsWarm, wantNormWarm string
	}{
		{
			name:        "spec example: 1 cold + 3 warm cycles, misses uniform across cycles -> 25% abs / 50% norm cold",
			coldCount:   1,
			warmCount:   3,
			nCold:       1,
			nWarm:       3,
			wantAbsCold: "25.0%", wantNormCold: "50.0%",
			wantAbsWarm: "75.0%", wantNormWarm: "50.0%",
		},
		{
			name:        "default 2-cycle run: abs == norm (equal cycle cardinality)",
			coldCount:   2,
			warmCount:   6,
			nCold:       1,
			nWarm:       1,
			wantAbsCold: "25.0%", wantNormCold: "25.0%",
			wantAbsWarm: "75.0%", wantNormWarm: "75.0%",
		},
		{
			name:        "warm-biased loss surfaces after norm even when abs looks cold-heavy",
			coldCount:   10,
			warmCount:   10,
			nCold:       1,
			nWarm:       9,
			wantAbsCold: "50.0%", wantNormCold: "90.0%",
			wantAbsWarm: "50.0%", wantNormWarm: "10.0%",
		},
		{
			name:        "zero counts -> abs n/a, norm n/a",
			coldCount:   0,
			warmCount:   0,
			nCold:       1,
			nWarm:       1,
			wantAbsCold: "n/a", wantNormCold: "n/a",
			wantAbsWarm: "n/a", wantNormWarm: "n/a",
		},
		{
			name:        "nWarm == 0 (run too short to reach cycle 2) -> norm n/a, abs still computed",
			coldCount:   3,
			warmCount:   0,
			nCold:       1,
			nWarm:       0,
			wantAbsCold: "100.0%", wantNormCold: "n/a",
			wantAbsWarm: "0.0%", wantNormWarm: "n/a",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatCycleDistribution(tt.coldCount, tt.warmCount, tt.nCold, tt.nWarm)
			wantSubstrs := []string{
				fmt.Sprintf("cold %d (%s abs / %s norm)", tt.coldCount, tt.wantAbsCold, tt.wantNormCold),
				fmt.Sprintf("warm %d (%s abs / %s norm)", tt.warmCount, tt.wantAbsWarm, tt.wantNormWarm),
			}
			for _, s := range wantSubstrs {
				if !strings.Contains(got, s) {
					t.Errorf("formatCycleDistribution(%d, %d, %d, %d) = %q, want substring %q", tt.coldCount, tt.warmCount, tt.nCold, tt.nWarm, got, s)
				}
			}
		})
	}
}
