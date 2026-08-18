package benchmark

import (
	"strings"
	"testing"
)

// TestResponseIsGarbage pins the corruption test to signals the serving stack
// alone can produce. The counter's value is that firing at all means "read the
// capture" — which it stops meaning the first time ordinary prose trips it.
func TestResponseIsGarbage(t *testing.T) {
	for _, c := range []struct {
		name string
		resp string
		want bool
	}{
		{"plain prose", "The whale, in Chapter 32, is classified at length.", false},
		{"tabs newlines and CR are formatting", "a\tb\nc\r\nd", false},
		{"unicode prose is not garbage", "naïve café — привет, 鯨", false},
		{"replacement char is decode corruption", "the id is ���", true},
		{"NUL byte", "abc\x00def", true},
		{"stray control char", "abc\x07def", true},
		{"empty is fine", "", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := responseIsGarbage(c.resp); got != c.want {
				t.Errorf("responseIsGarbage(%q) = %v, want %v", c.resp, got, c.want)
			}
		})
	}
}

// TestProgressLineVerifySegment: the periodic line is most people's only view
// of a running benchmark, so what it shows — and refuses to show — is part of
// the measurement's honesty.
func TestProgressLineVerifySegment(t *testing.T) {
	base := func() *displaySnapshot {
		return &displaySnapshot{series: 1, concurrency: 1, totalCompleted: 10}
	}

	t.Run("off means absent, not zero", func(t *testing.T) {
		if line := renderModelOneLiner(base()); strings.Contains(line, "guid=") {
			t.Errorf("verify segment rendered with verification off:\n%s", line)
		}
	})

	t.Run("clean run shows the rate and no BAD block", func(t *testing.T) {
		s := base()
		s.verifyOn = true
		s.verifyChecks, s.verifyFound = 81, 80
		line := renderModelOneLiner(s)
		if !strings.Contains(line, "guid=98.8%") {
			t.Errorf("want guid=98.8%% in:\n%s", line)
		}
		if strings.Contains(line, "BAD(") {
			t.Errorf("BAD block rendered with nothing bad to report:\n%s", line)
		}
	})

	t.Run("bad signals surface with their counts", func(t *testing.T) {
		s := base()
		s.verifyOn = true
		s.verifyChecks, s.verifyFound = 100, 90
		s.verifyLeaked, s.verifyGarbage, s.verifyAbsent = 1, 2, 3
		line := renderModelOneLiner(s)
		if !strings.Contains(line, "BAD(leak=1 garbage=2 lost=3)") {
			t.Errorf("want BAD(leak=1 garbage=2 lost=3) in:\n%s", line)
		}
	})

	t.Run("enabled but nothing scored yet reads 100, not panic", func(t *testing.T) {
		s := base()
		s.verifyOn = true
		if line := renderModelOneLiner(s); !strings.Contains(line, "guid=100.0%") {
			t.Errorf("want guid=100.0%% before any scoring in:\n%s", line)
		}
	})

	t.Run("DONE line carries the segment too", func(t *testing.T) {
		s := base()
		s.verifyOn = true
		s.verifyChecks, s.verifyFound = 10, 10
		s.termReason = "Timeout"
		if line := renderModelOneLiner(s); !strings.Contains(line, "guid=100.0%") {
			t.Errorf("final line dropped the verify segment:\n%s", line)
		}
	})
}
