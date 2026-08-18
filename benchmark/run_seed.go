package benchmark

import (
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
)

// The run's identity, and the one number that reproduces it.
//
// A run is separated from every other run by its RUN_GUID stamp, which is
// prepended to every prompt so a new run is cold for its own prefixes whatever
// the fleet served before. That stamp was drawn from crypto/rand and was the
// only thing making runs distinct, while a separate --seed governed coherency
// markers — so passing a seed reproduced the markers and nothing else, and two
// runs at one seed were still different runs.
//
// The seed now produces the stamp, and the stamp produces everything derived
// from it. One printed number reproduces a run exactly.
//
// The cost of that is real and belongs in the caller's face rather than in a
// doc comment: two runs at the same seed emit IDENTICAL content, so the second
// starts warm on the first's prefixes and its cache hit rate describes the
// leftovers rather than the fleet. That is the one thing this repo's replay
// invariant exists to prevent, which is why an explicit seed warns and the
// default draws a fresh one.
func resolveRunSeed(seed int64) (int64, error) {
	if seed != 0 {
		return seed, nil
	}
	var b [8]byte
	if _, err := crand.Read(b[:]); err != nil {
		return 0, fmt.Errorf("draw run seed: %w", err)
	}
	// Cleared sign bit: the seed is printed for a human to copy back into
	// --seed, and a leading minus invites being dropped or mis-shell-quoted.
	return int64(binary.LittleEndian.Uint64(b[:]) >> 1), nil
}

// runIDFromSeed derives the run stamp, deliberately NOT UUID-shaped.
//
// The stamp is prepended to every prompt as <ignore>RUN_GUID: ...</ignore>,
// and --verify asks the model to repeat "the id" for a tagged turn. A
// UUID-shaped stamp sitting above every marker is a plausible wrong answer to
// that question — it is the first UUID in the context, and a model asked for
// "the first id" has every reason to reach for it. A plain hex digest cannot
// be confused with a marker, and cannot be matched by uuidRe either, so it can
// never be picked up by the contamination scan.
func runIDFromSeed(seed int64) string {
	var s [8]byte
	binary.LittleEndian.PutUint64(s[:], uint64(seed))
	sum := sha256.Sum256(append([]byte("wekai-run-id:"), s[:]...))
	return "run" + hex.EncodeToString(sum[:12])
}

// formatUUID renders 16 bytes canonically, with the version and variant
// nibbles set, so the result is a well-formed v4-shaped value rather than
// arbitrary hex — and so uuidRe matches it on the way back out of a response.
func formatUUID(b []byte) string {
	out := make([]byte, 16)
	copy(out, b)
	out[6] = (out[6] & 0x0f) | 0x40
	out[8] = (out[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", out[0:4], out[4:6], out[6:8], out[8:10], out[10:16])
}

// applyRunSeed fixes cfg.Seed and the run stamp it produces, and says out loud
// what an explicit seed costs.
//
// A run is comparable to another only if it started cold for its own prefixes.
// Reusing a seed reproduces the content exactly, so the second run starts warm
// on the first's and its cache hit rate describes what was left behind — the
// one explanation this repo's replay invariant forbids reaching for. Nobody
// should discover that from a doc comment after the numbers look good.
func applyRunSeed(cfg *AutoBenchmarkConfig) error {
	seed, err := resolveRunSeed(cfg.Seed)
	if err != nil {
		return err
	}
	if cfg.Seed != 0 {
		fmt.Fprintf(os.Stderr,
			"warning: --seed=%d reproduces this run's content exactly. Any earlier run with this "+
				"seed left the same prefixes on the fleet, so cache hit rate here measures those "+
				"leftovers, not a cold start. Omit --seed unless you are deliberately replaying "+
				"identical content.\n", seed)
	}
	cfg.Seed = seed
	cfg.RunID = runIDFromSeed(seed)
	fmt.Printf("Run seed: %d (run-id %s) — pass --seed=%d to reproduce this run's content and markers\n",
		seed, cfg.RunID, seed)
	return nil
}
