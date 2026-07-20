package cli

import (
	"math"
	"testing"

	"github.com/weka/wekai/llm"
)

// rec builds a minimal analyzeRecord with a server-reported usage for tests.
func rec(user, model string, in, cacheRead, cacheCreate, out int) analyzeRecord {
	return analyzeRecord{
		User:    user,
		ModelIn: model,
		Response: redactedResponse{
			Usage: &redactedUsage{
				InputTokens:              in,
				CacheReadInputTokens:     cacheRead,
				CacheCreationInputTokens: cacheCreate,
				OutputTokens:             out,
			},
		},
	}
}

// testModelRegistry is a minimal stand-in for the ResolveModel/
// LookupModelByIdentifier hooks a real embedding application (e.g. wekai)
// registers from its static registry — core ships none of its own (see
// llm.LookupModelByIdentifier), so tests exercising the hook-dependent path
// install one locally.
var testModelRegistry = map[string]llm.ModelInfo{
	"claude-opus-4-8": {
		Provider: llm.ProviderAnthropic, ModelIdentifier: "claude-opus-4-8",
		InputCostPerMillion: 15, OutputCostPerMillion: 75,
		CachedCostPerMillion: 1.5, CacheTokens5MinCostPerMillion: 18.75,
	},
	"claude-haiku-4-5": {
		Provider: llm.ProviderAnthropic, ModelIdentifier: "claude-haiku-4-5",
		InputCostPerMillion: 1, OutputCostPerMillion: 5,
		CachedCostPerMillion: 0.1, CacheTokens5MinCostPerMillion: 1.25,
	},
}

func testLookupModelByIdentifier(id string) (llm.ModelInfo, bool) {
	info, ok := testModelRegistry[llm.NormalizeModelIdentifier(id)]
	return info, ok
}

// installTestModelRegistry registers the test registry as the
// LookupModelByIdentifier hook for the duration of the test, restoring
// whatever was previously registered on cleanup.
func installTestModelRegistry(t *testing.T) {
	t.Helper()
	prev := llm.LookupModelByIdentifier
	llm.LookupModelByIdentifier = testLookupModelByIdentifier
	t.Cleanup(func() { llm.LookupModelByIdentifier = prev })
}

// TestRecordCostMatchesRegistryPricing verifies recordCost folds all four
// token buckets through the same pricing CalculateCost uses, and that a raw
// dated model id (claude-...-YYYYMMDD) still resolves to a price.
func TestRecordCostMatchesRegistryPricing(t *testing.T) {
	installTestModelRegistry(t)

	r := rec("u", "claude-opus-4-8", 1_000_000, 2_000_000, 500_000, 100_000)
	got, priced := recordCost(r)
	if !priced {
		t.Fatalf("claude-opus-4-8 should be priced (added to registry)")
	}
	info, ok := llm.LookupModelByIdentifier("claude-opus-4-8")
	if !ok {
		t.Fatalf("claude-opus-4-8 missing from registry")
	}
	want := llm.CalculateCost(info, llm.Usage{
		InputTokens: 1_000_000, CacheReadInputTokens: 2_000_000,
		CacheCreationInputTokens: 500_000, OutputTokens: 100_000,
	})
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("recordCost=%v want=%v", got, want)
	}

	// A dated haiku id must still resolve (date suffix stripped).
	if _, p := recordCost(rec("u", "claude-haiku-4-5-20251001", 10, 0, 0, 5)); !p {
		t.Fatalf("dated haiku id should resolve to a price")
	}

	// Unknown model => unpriced, zero cost.
	if cost, p := recordCost(rec("u", "totally-unknown-model", 10, 0, 0, 5)); p || cost != 0 {
		t.Fatalf("unknown model should be unpriced with zero cost, got cost=%v priced=%v", cost, p)
	}
}

// TestAggregateUsersGroupsAndFilters checks per-user/per-model grouping,
// the userFilter, that count_tokens (PlainJSON) records are skipped, and that
// totals add up.
func TestAggregateUsersGroupsAndFilters(t *testing.T) {
	plain := analyzeRecord{User: "alice", ModelIn: "claude-opus-4-8", IsPlainJSON: true}
	records := []analyzeRecord{
		rec("alice", "claude-opus-4-8", 100, 0, 0, 50),
		rec("alice", "claude-opus-4-8", 200, 0, 0, 10),
		rec("alice", "claude-sonnet-4-6", 100, 0, 0, 20),
		rec("bob", "claude-opus-4-8", 100, 0, 0, 50),
		plain, // must be skipped
	}

	// No filter: both users present, plain record skipped.
	all := (&RouterAnalyzeCommand{}).aggregateUsers(records, "")
	if len(all) != 2 {
		t.Fatalf("want 2 users, got %d", len(all))
	}
	alice := all["alice"]
	if alice.Requests != 3 {
		t.Fatalf("alice requests=%d want 3 (plain skipped)", alice.Requests)
	}
	if len(alice.ByModel) != 2 {
		t.Fatalf("alice should have 2 models, got %d", len(alice.ByModel))
	}
	if alice.ByModel["claude-opus-4-8"].Requests != 2 {
		t.Fatalf("alice opus requests=%d want 2", alice.ByModel["claude-opus-4-8"].Requests)
	}

	// per-model costs must sum to the user total.
	var sum float64
	for _, m := range alice.ByModel {
		sum += m.Cost
	}
	if math.Abs(sum-alice.Cost) > 1e-9 {
		t.Fatalf("alice per-model sum=%v != total=%v", sum, alice.Cost)
	}

	// Filter to bob only.
	only := (&RouterAnalyzeCommand{}).aggregateUsers(records, "bob")
	if len(only) != 1 || only["bob"] == nil {
		t.Fatalf("filter should yield only bob, got %v", only)
	}
	if only["bob"].Requests != 1 {
		t.Fatalf("bob requests=%d want 1", only["bob"].Requests)
	}
}

func TestFormatFloatThousands(t *testing.T) {
	cases := map[float64]string{
		0:       "0.00",
		1.5:     "1.50",
		901.097: "901.10", // rounds to cents
		1234.5:  "1,234.50",
		1000000: "1,000,000.00",
		9.999:   "10.00", // carry into whole
	}
	for in, want := range cases {
		if got := formatFloat(in); got != want {
			t.Errorf("formatFloat(%v)=%q want %q", in, got, want)
		}
	}
}
