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

func TestFindLeakedUUIDs(t *testing.T) {
	seriesUUIDs := [][]string{
		{"uuid-0-a", "uuid-0-b"}, // series 0
		{"uuid-1-a", "uuid-1-b"}, // series 1
		{"uuid-2-a", "uuid-2-b"}, // series 2
	}

	t.Run("no leak: response only contains own series uuids", func(t *testing.T) {
		resp := "uuid-0-a,uuid-0-b"
		got := findLeakedUUIDs(resp, "", 0, seriesUUIDs)
		if len(got) != 0 {
			t.Errorf("expected no leaks, got %v", got)
		}
	})

	t.Run("single leak from another series in response", func(t *testing.T) {
		resp := "uuid-0-a,uuid-1-a"
		got := findLeakedUUIDs(resp, "", 0, seriesUUIDs)
		if len(got) != 1 {
			t.Fatalf("expected 1 leak, got %v", got)
		}
		if !strings.Contains(got[0], "uuid-1-a") || !strings.Contains(got[0], "series=1") {
			t.Errorf("leak entry %q missing uuid or owning series", got[0])
		}
	})

	t.Run("multiple leaks from multiple series, deterministic order", func(t *testing.T) {
		resp := "uuid-0-a,uuid-1-a,uuid-2-b"
		got1 := findLeakedUUIDs(resp, "", 0, seriesUUIDs)
		got2 := findLeakedUUIDs(resp, "", 0, seriesUUIDs)
		if len(got1) != 2 {
			t.Fatalf("expected 2 leaks, got %v", got1)
		}
		// Order must be deterministic (series-index order) across repeated calls.
		for i := range got1 {
			if got1[i] != got2[i] {
				t.Errorf("non-deterministic leak order: %v vs %v", got1, got2)
			}
		}
	})

	t.Run("leak detected in thinking, not just response", func(t *testing.T) {
		got := findLeakedUUIDs("uuid-0-a", "I recall uuid-2-a from earlier", 0, seriesUUIDs)
		if len(got) != 1 || !strings.Contains(got[0], "uuid-2-a") {
			t.Errorf("expected leak of uuid-2-a via thinking, got %v", got)
		}
	})

	t.Run("own series never reported as a leak", func(t *testing.T) {
		got := findLeakedUUIDs("uuid-1-a,uuid-1-b", "", 1, seriesUUIDs)
		if len(got) != 0 {
			t.Errorf("expected no leaks for own series, got %v", got)
		}
	})
}
