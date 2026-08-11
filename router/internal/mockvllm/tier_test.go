package mockvllm

import (
	"testing"
	"time"
)

// The shared tier is what makes prefilling on one instance worth anything to
// another, so these tests are about exactly one thing: that a block computed on
// instance A is found by instance B, that B pays the external rate for it
// rather than the cold one, and that none of this happens unless a Tier was
// asked for.

func tierFleet(t *testing.T, n int, tier *Tier) []*Engine {
	t.Helper()
	out := make([]*Engine, n)
	for i := range out {
		out[i] = NewEngine(Config{
			BlockSizeTokens:  16,
			CharsPerToken:    4,
			ColdInputTPS:     1_000,
			ExternalInputTPS: 10_000,
			CachedInputTPS:   100_000,
			Tier:             tier,
		})
		t.Cleanup(out[i].Close)
	}
	return out
}

func TestTierMakesOneInstancesPrefillVisibleToAnother(t *testing.T) {
	tier := NewTier()
	f := tierFleet(t, 2, tier)
	prompt := "a prompt long enough to span several blocks: " + repeat("content ", 200)
	units := f[0].Tokenize(prompt)

	// Instance 0 serves it cold and publishes.
	rel, res, ok := f[0].Admit(units)
	if !ok {
		t.Fatal("admit on instance 0")
	}
	if res.Local != 0 || res.External != 0 {
		t.Fatalf("first request found %d local and %d external blocks, want a cold miss",
			res.Local, res.External)
	}
	f[0].PublishToTier(units)
	rel()

	// Instance 1 has never seen it, and must still find it — externally.
	rel1, res1, ok := f[1].Admit(units)
	if !ok {
		t.Fatal("admit on instance 1")
	}
	defer rel1()
	if res1.Local != 0 {
		t.Errorf("instance 1 reported %d LOCAL blocks for a prompt it has never served; the tier "+
			"is not its own cache", res1.Local)
	}
	if res1.External != res1.Total {
		t.Errorf("instance 1 found %d of %d blocks in the shared tier, want all of them: instance "+
			"0 computed and published every one", res1.External, res1.Total)
	}
	if res1.Cold() != 0 {
		t.Errorf("instance 1 still has %d cold blocks; nothing is left to recompute", res1.Cold())
	}
}

func TestTierCostsLessThanRecomputeAndMoreThanLocal(t *testing.T) {
	e := NewEngine(Config{
		BlockSizeTokens: 16, CharsPerToken: 4,
		ColdInputTPS: 1_000, ExternalInputTPS: 10_000, CachedInputTPS: 100_000,
	})
	t.Cleanup(e.Close)

	cold := e.PrefillWork(Residency{Total: 1000})
	external := e.PrefillWork(Residency{External: 1000, Total: 1000})
	local := e.PrefillWork(Residency{Local: 1000, Total: 1000})

	if !(cold > external && external > local) {
		t.Errorf("prefill work: cold %v, external %v, local %v — the three tiers must be strictly "+
			"ordered, or the mock cannot express what a shared KV tier buys", cold, external, local)
	}
	if want := time.Second; cold != want {
		t.Errorf("1000 cold tokens at 1000 tok/s took %v, want %v", cold, want)
	}
	if want := 100 * time.Millisecond; external != want {
		t.Errorf("1000 external tokens at 10000 tok/s took %v, want %v", external, want)
	}
}

func TestTierAbsentMeansIndependentInstances(t *testing.T) {
	f := tierFleet(t, 2, nil)
	units := f[0].Tokenize("some prompt " + repeat("x", 4000))

	rel, _, ok := f[0].Admit(units)
	if !ok {
		t.Fatal("admit on instance 0")
	}
	f[0].PublishToTier(units) // a no-op without a tier, and must not panic
	rel()

	rel1, res, ok := f[1].Admit(units)
	if !ok {
		t.Fatal("admit on instance 1")
	}
	defer rel1()
	if res.External != 0 || res.Cold() != res.Total {
		t.Errorf("without a Tier configured, instance 1 found %d external blocks and only %d cold "+
			"of %d: instances must be independent by default, which is what plain vLLM is",
			res.External, res.Cold(), res.Total)
	}
}

// TestTierPublishesOnlyWhatWasServed guards the rule that a refused request
// advertises nothing. A 429'd request never reached the engine's cache, so the
// fleet must not be told its blocks are available.
func TestTierPublishesOnlyWhatWasServed(t *testing.T) {
	tier := NewTier()
	e := NewEngine(Config{
		BlockSizeTokens: 16, CharsPerToken: 4, MaxConcurrency: 1,
		ColdInputTPS: 1_000, ExternalInputTPS: 10_000, Tier: tier,
	})
	t.Cleanup(e.Close)

	hold, _, ok := e.Admit(e.Tokenize("first " + repeat("y", 4000)))
	if !ok {
		t.Fatal("first admit")
	}
	defer hold()

	refused := e.Tokenize("second " + repeat("z", 4000))
	if _, _, ok := e.Admit(refused); ok {
		t.Fatal("second admit succeeded at MaxConcurrency 1")
	}
	if n := tier.Query(refused); n != 0 {
		t.Errorf("the shared tier holds %d blocks of a REFUSED request; nothing computed it, so "+
			"another instance would load KV that does not exist", n)
	}
}

func repeat(s string, n int) string {
	out := make([]byte, 0, len(s)*n)
	for range n {
		out = append(out, s...)
	}
	return string(out)
}
