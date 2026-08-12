package cli

import (
	"testing"

	"github.com/jessevdk/go-flags"
)

// The router imposes NO global in-flight limit unless asked.
//
// Capacity is the fleet's answer to give: a vLLM returns 429 when it is full,
// the routing flow walks this prefix's other holders, and a 429 reaches the
// client only once nothing can take the request. A router-side ceiling
// short-circuits that with a number that knows nothing about the fleet — it
// refuses work the backends could still do, identically whether they are idle
// or saturated.
//
// Asserted through the real flag parser rather than by reading the struct tag,
// because a `default:` tag only takes effect when the command is parsed — a
// value set anywhere else would leave the help text saying one thing and the
// router doing another.
func TestMaxConcurrentRequestsIsUnlimitedByDefault(t *testing.T) {
	var c RouterServeCommand
	p := flags.NewParser(&c, flags.Default)
	if _, err := p.ParseArgs([]string{"--backends", "http://a:8000"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.MaxConcurrent != 0 {
		t.Errorf("MaxConcurrent = %d with no flag given, want 0 (no limit). A default ceiling "+
			"sheds 503 router_at_capacity while the fleet still has room, and the shed looks "+
			"identical whether the backends are idle or saturated.", c.MaxConcurrent)
	}

	// And it must still be settable, for the case the cap actually addresses:
	// the ROUTER's own memory, since each in-flight request may hold up to
	// max-body-bytes buffered for retry.
	var capped RouterServeCommand
	p2 := flags.NewParser(&capped, flags.Default)
	if _, err := p2.ParseArgs([]string{"--backends", "http://a:8000", "--max-concurrent-requests", "64"}); err != nil {
		t.Fatalf("parse with the flag: %v", err)
	}
	if capped.MaxConcurrent != 64 {
		t.Errorf("MaxConcurrent = %d with --max-concurrent-requests 64, want 64", capped.MaxConcurrent)
	}
}
