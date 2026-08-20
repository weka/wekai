package cli

import (
	"os"
	"testing"

	"github.com/jessevdk/go-flags"
)

// Cycling the corpus is ON by default, and it is a boolean VALUE rather than a
// bare switch so a run can turn it off.
//
// The default is the way round it is because both ways a run falls short of its
// dataset produce a number that reads as the fleet's. A long arm drains the
// corpus and the slots empty out hours in; a session count above the corpus size
// leaves the surplus slots empty from the first minute. Neither shows up as an
// error — throughput simply comes in low, against a session count the run keeps
// printing.
//
// Asserted through the real parser rather than by reading the struct tag,
// because a `default:` tag only takes effect when the command is parsed.
func TestReplayReuseSessionsDefaultsOn(t *testing.T) {
	// Neutralize the env override so the tag default is what is under test.
	// Both calls are needed: t.Setenv registers the restore-on-cleanup that
	// keeps a developer's own environment out of the next test, and Unsetenv
	// then removes the variable the parser would otherwise read.
	t.Setenv("BENCHMARK_REPLAY_REUSE_SESSIONS", "unused")
	os.Unsetenv("BENCHMARK_REPLAY_REUSE_SESSIONS")

	parse := func(t *testing.T, args ...string) BenchmarkAutoOptions {
		t.Helper()
		var c BenchmarkAutoOptions
		if _, err := flags.ParseArgs(&c, args); err != nil {
			t.Fatalf("parse %v: %v", args, err)
		}
		return c
	}

	if c := parse(t); c.ReplayReuseSessions != "true" {
		t.Errorf("no flags: ReplayReuseSessions = %q, want \"true\": a run that is not told otherwise "+
			"should outlast its corpus rather than quietly stop offering the load it reports",
			c.ReplayReuseSessions)
	}

	// The off switch has to be a VALUE, not the flag's absence — that is what
	// lets a deployment template render the flag unconditionally and still turn
	// it off, and what makes the default reachable in reverse.
	if c := parse(t, "--replay-reuse-sessions=false"); c.ReplayReuseSessions != "false" {
		t.Errorf("--replay-reuse-sessions=false parsed as %q; a single sweep of the corpus must stay "+
			"reachable, and it is also the only terminator a run with no --timeout and no --total has",
			c.ReplayReuseSessions)
	}

	if c := parse(t, "--replay-reuse-sessions=true"); c.ReplayReuseSessions != "true" {
		t.Errorf("--replay-reuse-sessions=true parsed as %q", c.ReplayReuseSessions)
	}

	// The bare switch keeps meaning what it meant when it was one, so every
	// existing invocation still asks for the same thing.
	if c := parse(t, "--replay-reuse-sessions"); c.ReplayReuseSessions != "true" {
		t.Errorf("bare --replay-reuse-sessions parsed as %q, want \"true\"", c.ReplayReuseSessions)
	}
}

// TestReplayReuseSessionsReachesTheConfig: the flag is a string with three
// reachable states and the run takes a bool, so the conversion is where a
// default can be lost. Only an explicit "false" turns cycling off.
func TestReplayReuseSessionsReachesTheConfig(t *testing.T) {
	for _, tc := range []struct {
		flag string
		want bool
	}{
		{"", true}, // unparsed zero value: the config must not read it as off
		{"true", true},
		{"false", false},
	} {
		c := BenchmarkAutoOptions{ReplayReuseSessions: tc.flag}
		if got := c.ReuseSessions(); got != tc.want {
			t.Errorf("flag %q -> reuse %v, want %v", tc.flag, got, tc.want)
		}
	}
}
