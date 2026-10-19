package benchmark

import (
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/weka/wekai/kvcache"
)

// storeTestChains builds instances whose turns extend the previous one, with
// the previous turn's last message rewritten (as a moved cache_control does),
// occasional compactions, and more turns than prefixCheckpointEvery.
func storeTestChains(rng *rand.Rand) [][]RouterReplayRequest {
	t0 := time.Date(2026, 5, 12, 8, 0, 0, 0, time.UTC)
	var chains [][]RouterReplayRequest
	for c := 0; c < 6; c++ {
		var msgs []string
		var chain []RouterReplayRequest
		for turn := 0; turn < 100; turn++ {
			if len(msgs) > 0 {
				msgs[len(msgs)-1] += "'"
			}
			if rng.Intn(25) == 0 {
				msgs = msgs[:1] // compaction
			}
			msgs = append(msgs, fmt.Sprintf("c%d-t%d-a", c%3, turn), fmt.Sprintf("c%d-t%d-u", c, turn))
			r := RouterReplayRequest{
				Ts:    t0.Add(time.Duration(rng.Intn(400)) * time.Minute).Format(time.RFC3339Nano),
				Model: fmt.Sprint("m", c%2), InputTokens: 10 * len(msgs), CacheReadTokens: 5,
			}
			for _, h := range msgs {
				r.Messages = append(r.Messages, RouterReplayMessage{Hash: h, Tokens: 1 + rng.Intn(50)})
			}
			chain = append(chain, r)
		}
		chains = append(chains, chain)
	}
	return chains
}

func TestPrefixStoreMatchesSliceSimulations(t *testing.T) {
	chains := storeTestChains(rand.New(rand.NewSource(1)))
	var s PrefixStore
	var flat []RouterReplayRequest
	for _, ch := range chains {
		var b PrefixBatch
		b.AddChain(ch)
		s.Append(&b)
		flat = append(flat, ch...)
	}

	var buf []kvcache.Unit
	for i, req := range flat {
		hashes, tokens := BuildReplayRequestPrefix(req)
		want := kvcache.UnitsFromLabels(hashes, tokens)
		buf = s.Units(i, buf)
		if !reflect.DeepEqual(append([]kvcache.Unit{}, buf...), want) {
			t.Fatalf("request %d: units differ", i)
		}
		if d := s.Reqs[i].depth; d >= prefixCheckpointEvery {
			t.Fatalf("request %d: depth %d", i, d)
		}
	}

	order := s.ChronologicalOrder()
	sorted := append([]RouterReplayRequest(nil), flat...)
	sort.SliceStable(sorted, func(a, b int) bool { return sorted[a].Ts < sorted[b].Ts })
	windows := []time.Duration{5 * time.Minute, time.Hour, 24 * time.Hour}
	if got, want := SimulateReplayCacheStore(&s, order), SimulateReplayCache(sorted); got != want {
		t.Fatalf("cache sim: got %+v, want %+v", got, want)
	}
	got, want := SimulateRetentionStore(&s, order, windows), SimulateRetention(sorted, windows)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("retention: got %+v, want %+v", got, want)
	}
}
