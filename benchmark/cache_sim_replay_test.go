package benchmark

import "testing"

func TestSimulateReplayCache(t *testing.T) {
	// Empty input — zero report, no panic.
	empty := SimulateReplayCache(nil)
	if empty.Requests != 0 || empty.SimCachedTokens != 0 || empty.SimTotalTokens != 0 ||
		empty.GTCachedTokens != 0 || empty.GTTotalTokens != 0 {
		t.Errorf("empty input: got %+v, want all-zero", empty)
	}
	if empty.SimRatio != 0 || empty.GTRatio != 0 {
		t.Errorf("empty input: ratios should be zero, got SimRatio=%f GTRatio=%f", empty.SimRatio, empty.GTRatio)
	}

	// Helper: make a request with given system/message hashes+tokens,
	// and ground-truth usage fields.
	makeReq := func(sysBlocks [][2]string, msgBlocks [][2]string, cacheRead, inputTotal int) RouterReplayRequest {
		r := RouterReplayRequest{
			Ts:              "2025-01-01T00:00:00Z",
			CacheReadTokens: cacheRead,
			InputTokens:     inputTotal,
		}
		for i, sb := range sysBlocks {
			r.SystemBlocks = append(r.SystemBlocks, RouterReplaySystemBlock{
				Type: "text", Hash: sb[0], Bytes: len(sb[1]), Tokens: len(sb[1]),
			})
			_ = i
		}
		for _, mb := range msgBlocks {
			r.Messages = append(r.Messages, RouterReplayMessage{
				Role: "user", Hash: mb[0], Bytes: len(mb[1]), Tokens: len(mb[1]),
			})
		}
		return r
	}

	t.Run("identical requests → SimRatio ≈ 0.5 over the pair", func(t *testing.T) {
		reqs := []RouterReplayRequest{
			makeReq(nil, [][2]string{{"a", "hello world"}}, 0, 100),
			makeReq(nil, [][2]string{{"a", "hello world"}}, 100, 100),
		}
		rep := SimulateReplayCache(reqs)
		if rep.Requests != 2 {
			t.Fatalf("expected 2 requests, got %d", rep.Requests)
		}
		if rep.SimTotalTokens <= 0 {
			t.Fatal("SimTotalTokens should be > 0")
		}
		// Two identical requests: first pays all, second is fully cached.
		// Expected SimRatio = total cached / total for both ≈ 0.5
		expected := 0.5
		if rep.SimRatio < expected-0.1 || rep.SimRatio > expected+0.1 {
			t.Errorf("SimRatio = %f, want ~%f", rep.SimRatio, expected)
		}
		// Ground-truth: 0 + 100 = 100 cached / 200 total = 0.5
		if rep.GTTotalTokens != 200 {
			t.Errorf("GTTotalTokens = %d, want 200", rep.GTTotalTokens)
		}
		if rep.GTCachedTokens != 100 {
			t.Errorf("GTCachedTokens = %d, want 100", rep.GTCachedTokens)
		}
		if rep.GTRatio < 0.49 || rep.GTRatio > 0.51 {
			t.Errorf("GTRatio = %f, want ~0.5", rep.GTRatio)
		}
	})

	t.Run("shared-prefix diverging-suffix → partial cache", func(t *testing.T) {
		reqs := []RouterReplayRequest{
			makeReq(nil, [][2]string{{"a", "hello"}, {"b", "world"}}, 0, 50),
			makeReq(nil, [][2]string{{"a", "hello"}, {"c", "x"}}, 30, 50),
		}
		rep := SimulateReplayCache(reqs)
		// First msg "a":"hello" is cached for the second request.
		// So SimRatio should be > 0 (at least first message cached) and < 0.5 (second message not cached).
		if rep.SimRatio <= 0 {
			t.Errorf("SimRatio = %f, expected > 0 (shared prefix)", rep.SimRatio)
		}
		if rep.SimRatio >= 0.5 {
			t.Errorf("SimRatio = %f, expected < 0.5 (only first msg cached)", rep.SimRatio)
		}
	})

	t.Run("disjoint requests → SimRatio == 0", func(t *testing.T) {
		reqs := []RouterReplayRequest{
			makeReq(nil, [][2]string{{"x", "aaa"}, {"y", "bbb"}}, 0, 10),
			makeReq(nil, [][2]string{{"p", "ccc"}, {"q", "ddd"}}, 0, 10),
		}
		rep := SimulateReplayCache(reqs)
		if rep.SimRatio != 0 {
			t.Errorf("SimRatio = %f, want 0 (disjoint)", rep.SimRatio)
		}
	})

	t.Run("ground-truth sums correct", func(t *testing.T) {
		reqs := []RouterReplayRequest{
			{CacheReadTokens: 10, InputTokens: 100},
			{CacheReadTokens: 30, InputTokens: 200},
			{CacheReadTokens: 0, InputTokens: 50},
		}
		rep := SimulateReplayCache(reqs)
		if rep.GTCachedTokens != 40 {
			t.Errorf("GTCachedTokens = %d, want 40", rep.GTCachedTokens)
		}
		if rep.GTTotalTokens != 350 {
			t.Errorf("GTTotalTokens = %d, want 350", rep.GTTotalTokens)
		}
		if rep.GTRatio < 0.11 || rep.GTRatio > 0.12 {
			t.Errorf("GTRatio = %f, want ~0.114", rep.GTRatio)
		}
	})

	t.Run("block tokens all zero warning", func(t *testing.T) {
		// Requests with tokens=0 in all blocks — BlockTokensAllZero should be true.
		reqs := []RouterReplayRequest{
			{
				Ts: "2025-01-01T00:00:00Z",
				SystemBlocks: []RouterReplaySystemBlock{
					{Type: "text", Hash: "h1", Bytes: 100, Tokens: 0},
				},
				Messages: []RouterReplayMessage{
					{Role: "user", Hash: "m1", Bytes: 50, Tokens: 0},
				},
				InputTokens: 100,
			},
		}
		rep := SimulateReplayCache(reqs)
		if !rep.BlockTokensAllZero {
			t.Errorf("BlockTokensAllZero = false, want true (all block tokens are 0)")
		}
	})
}
