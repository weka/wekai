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
		s.verifyLeaked, s.verifyAbsent = 1, 3
		s.verifyGarbage = 7
		s.verifyGarbagePostEOS, s.verifyGarbageTail, s.verifyGarbageBabble, s.verifyGarbageMid = 3, 1, 1, 2
		line := renderModelOneLiner(s)
		if !strings.Contains(line, "gbg(eos=3 tail=1 babble=1 mid=2)") {
			t.Errorf("want the garbage classes inline in:\n%s", line)
		}
		// tail+babble+mid = 1+1+2: everything except the proven post-EOS class.
		if !strings.Contains(line, "BAD(leak=1 gbg-noneos=4 lost=3)") {
			t.Errorf("BAD must carry all garbage except the proven post-EOS class:\n%s", line)
		}
	})

	t.Run("proven post-EOS garbage alone does not raise BAD", func(t *testing.T) {
		// Only the class with proof is excluded: a visible stop token before
		// the corruption shows the model had finished and ignore_eos pushed
		// it onward. A run whose only garbage is that stays clean at a
		// glance.
		s := base()
		s.verifyOn = true
		s.verifyChecks, s.verifyFound = 100, 100
		s.verifyGarbage = 5
		s.verifyGarbagePostEOS = 5
		line := renderModelOneLiner(s)
		if !strings.Contains(line, "gbg(eos=5 tail=0 babble=0 mid=0)") {
			t.Errorf("classes missing:\n%s", line)
		}
		if strings.Contains(line, "BAD(") {
			t.Errorf("BAD raised for provably post-EOS garbage:\n%s", line)
		}
	})

	t.Run("unproven tail garbage raises BAD", func(t *testing.T) {
		// tail merely RESEMBLES the post-EOS event; without the marker there
		// is no proof, and corruption without an explanation is bad.
		s := base()
		s.verifyOn = true
		s.verifyChecks, s.verifyFound = 100, 100
		s.verifyGarbage = 3
		s.verifyGarbageTail = 3
		line := renderModelOneLiner(s)
		if !strings.Contains(line, "BAD(leak=0 gbg-noneos=3 lost=0)") {
			t.Errorf("tail garbage must count as BAD:\n%s", line)
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

// TestClassifyGarbagePositions pins the situating fields, with fixtures shaped
// like the two events observed live: babble immediately after a literal EOS
// marker, and a repeating �\t cycle running to the end of the budget.
func TestClassifyGarbagePositions(t *testing.T) {
	t.Run("literal EOS marker located before the garbage", func(t *testing.T) {
		g := classifyGarbage("aaa-a42<｜end▁of▁sentence｜>7€\uFFFD#@/ uncertain")
		if g.EOSByte < 0 || g.EOSMarker != "<｜end▁of▁sentence｜>" {
			t.Fatalf("EOS marker not found: %+v", g)
		}
		if g.EOSByte > g.FirstByte {
			t.Errorf("EOS at byte %d should precede the first bad rune at byte %d", g.EOSByte, g.FirstByte)
		}
	})
	t.Run("repeated stop attempts are counted", func(t *testing.T) {
		// The observed shape: answer, EOS, forced continuation, EOS again,
		// garbage. The verdict names the first; the count explains why the
		// excerpt can show a different one.
		g := classifyGarbage("guid<｜end▁of▁sentence｜>中文继续<｜end▁of▁sentence｜>\uFFFD more")
		if g.EOSCount != 2 {
			t.Errorf("EOSCount = %d, want 2", g.EOSCount)
		}
		if g.EOSByte != 4 {
			t.Errorf("EOSByte = %d, want the FIRST marker's position (4)", g.EOSByte)
		}
	})
	t.Run("tail babble runs to the end", func(t *testing.T) {
		g := classifyGarbage("coherent prose then \uFFFD\t\uFFFD\t\uFFFD\t")
		if !g.tailBabble() {
			t.Errorf("garbage to the end of the response must read as tail babble: %+v", g)
		}
	})
	t.Run("clean text after the last bad rune is counted", func(t *testing.T) {
		g := classifyGarbage("x\uFFFDy" + strings.Repeat("clean ", 20))
		if g.tailBabble() {
			t.Errorf("120 clean runes after the corruption is not a tail: %+v", g)
		}
		if g.CleanAfter < 100 {
			t.Errorf("CleanAfter = %d, want the full clean tail counted", g.CleanAfter)
		}
	})
}

// TestTailGuidBabble pins the reclassifier against the shape observed live: a
// 4102-rune tail of recombined uuid fragments that earned MID-RESPONSE — the
// verdict reserved for corruption the harness's own ignore_eos cannot explain
// — because invented guids are valid UTF-8 and read as "clean text resumed".
func TestTailGuidBabble(t *testing.T) {
	own := map[string]bool{"11111111-1111-4111-8111-111111111111": true}

	t.Run("novel guid spam is babble", func(t *testing.T) {
		tail := "8b7a4f2e-9c3d-4f5a-8b2c-1e6d9f0a3b5c 8b7a4f2e-9c3d-4f5a-8b2c-1e6d9f0a3b5d 8b7a4f2e-9c3d-4f5a-8b2c-1e6d9f0a3b5e"
		if _, _, dense := tailGuidBabble(tail, own); !dense {
			t.Error("a tail that is nothing but invented uuid shapes must classify as babble")
		}
	})
	t.Run("prose is not babble", func(t *testing.T) {
		if _, _, dense := tailGuidBabble("and the whale surfaced once more, spouting into the dawn, while the crew watched in silence from the rigging above", own); dense {
			t.Error("plain prose misclassified as guid babble")
		}
	})
	t.Run("legitimate recall of own markers is not babble", func(t *testing.T) {
		// Every id real, none invented: novelty is the discriminator, so a
		// correct recitation must never land in the babble bucket.
		tail := "11111111-1111-4111-8111-111111111111 11111111-1111-4111-8111-111111111111 11111111-1111-4111-8111-111111111111"
		if _, novel, dense := tailGuidBabble(tail, own); dense || novel != 0 {
			t.Errorf("own-marker recall classified as babble (novel=%d dense=%v)", novel, dense)
		}
	})
}
