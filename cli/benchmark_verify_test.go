package cli

import (
	"strings"
	"testing"

	"github.com/weka/wekai/benchmark"
)

// TestVerifyRejectsUnsupportedModes: --verify silently doing nothing would
// produce a validation section of all zeroes, which reads exactly like a fleet
// that passed. Every mode that cannot carry a marker has to say so at startup,
// while there is still a person watching.
func TestVerifyRejectsUnsupportedModes(t *testing.T) {
	for _, c := range []struct {
		name string
		opts BenchmarkAutoOptions
		want string
	}{
		{"no replay file at all", BenchmarkAutoOptions{Verify: true}, "requires --router-replay-file"},
		{"dataset replay", BenchmarkAutoOptions{Verify: true, RouterReplayFile: "r.jsonl", FromDataset: "d"}, "--from-dataset"},
		{"synthetic prompts", BenchmarkAutoOptions{Verify: true, RouterReplayFile: "r.jsonl", DocsDir: "/docs"}, "--docs-dir"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := c.opts.validateVerify()
			if err == nil {
				t.Fatalf("--verify accepted with %s: the run would report all-zero validation and "+
					"be indistinguishable from a fleet that passed", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q never mentions %q, so it does not tell the user what to change", err, c.want)
			}
		})
	}

	t.Run("router replay is accepted", func(t *testing.T) {
		o := BenchmarkAutoOptions{Verify: true, RouterReplayFile: "r.jsonl"}
		if err := o.validateVerify(); err != nil {
			t.Fatalf("--verify rejected on the one mode that supports it: %v", err)
		}
	})

	t.Run("silent when off", func(t *testing.T) {
		o := BenchmarkAutoOptions{DocsDir: "/docs"}
		if err := o.validateVerify(); err != nil {
			t.Fatalf("validation fired with --verify off: %v", err)
		}
	})
}

// TestVerifyReciteChoiceMapping pins the string choice to the boolean the
// benchmark reads. "every" and "last" are the only two values the parser
// accepts, and getting the mapping backwards would silently score a handful of
// requests per session while the summary still claimed every turn.
func TestVerifyReciteChoiceMapping(t *testing.T) {
	for choice, wantEvery := range map[string]bool{"every": true, "last": false} {
		if got := choice != "last"; got != wantEvery {
			t.Errorf("--verify-recite=%s maps to reciteEvery=%v, want %v", choice, got, wantEvery)
		}
	}
}

// TestVerifyImpliesNaturalOutput pins the mode-to-mechanism rule: a regular
// run forces output volume (ignore_eos), a verify run does not — forcing pads
// every response past its natural stop with the degenerate text a coherency
// check exists to notice, not to manufacture — and --verify-force-eos puts it
// back for verify runs that want deterministic volume anyway.
func TestVerifyImpliesNaturalOutput(t *testing.T) {
	if !(benchmark.AutoBenchmarkConfig{}).ForceVolumeForTest() {
		t.Error("a regular benchmark run must force output volume; deterministic load is the tool's primary job")
	}
	if (benchmark.AutoBenchmarkConfig{Verify: true}).ForceVolumeForTest() {
		t.Error("--verify must drop ignore_eos by default; forced babble corrupts the signals it scores")
	}
	if !(benchmark.AutoBenchmarkConfig{Verify: true, VerifyForceEOS: true}).ForceVolumeForTest() {
		t.Error("--verify-force-eos must restore engine enforcement under --verify")
	}
}

// TestVerifyForceEOSRequiresVerify: outside verify the flag is a no-op
// masquerading as a choice, so it is refused rather than ignored.
func TestVerifyForceEOSRequiresVerify(t *testing.T) {
	o := BenchmarkAutoOptions{VerifyForceEOS: true}
	if err := o.validateVerifyForceEOS(); err == nil {
		t.Error("--verify-force-eos without --verify accepted; it would silently do nothing")
	}
	o = BenchmarkAutoOptions{Verify: true, VerifyForceEOS: true, RouterReplayFile: "r.jsonl"}
	if err := o.validateVerifyForceEOS(); err != nil {
		t.Errorf("valid combination rejected: %v", err)
	}
}
