package cli

import (
	"bytes"
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
