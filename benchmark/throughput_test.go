package benchmark

import (
	"math"
	"testing"
	"time"
)

func TestDisplayHitRateServerCache(t *testing.T) {
	const tol = 1e-9

	t.Run("server reports cached tokens", func(t *testing.T) {
		s := newCompletionStream(50)
		// cold start with cached tokens — included in token ratio, excluded from heuristic hits
		s.Add(completionRecord{
			completedAt:  time.Now(),
			isColdStart:  true,
			cachedTokens: 900,
			inputTokens:  1000,
		})
		// warm miss by heuristic, but server reports cache
		s.Add(completionRecord{
			completedAt:  time.Now(),
			isColdStart:  false,
			cacheHit:     false,
			cachedTokens: 800,
			inputTokens:  1000,
		})
		// warm hit
		s.Add(completionRecord{
			completedAt:  time.Now(),
			isColdStart:  false,
			cacheHit:     true,
			cachedTokens: 950,
			inputTokens:  1000,
		})

		cm := s.CacheMetrics(10, 1)
		if !cm.serverReported {
			t.Fatal("expected serverReported=true")
		}
		// inputTokens is net-of-cache; full prompt = inputTokens + cachedTokens.
		// serverTokenRate = cached / full = (900+800+950) / ((1000+1000+1000)+(900+800+950))
		//                 = 2650 / 5650
		wantRate := 2650.0 / 5650.0
		if math.Abs(cm.serverTokenRate-wantRate) > tol {
			t.Errorf("serverTokenRate = %f, want ~%f", cm.serverTokenRate, wantRate)
		}
		if cm.serverTokenRate > 1.0 {
			t.Errorf("serverTokenRate = %f must never exceed 1.0", cm.serverTokenRate)
		}
		if math.Abs(cm.DisplayHitRate()-wantRate) > tol {
			t.Errorf("DisplayHitRate() = %f, want ~%f (server token rate)", cm.DisplayHitRate(), wantRate)
		}
		// sanity: heuristic hitRate would be 1/3 (only 1 hit out of 3 total, cold counts as miss)
		wantHeuristic := 1.0 / 3.0
		if math.Abs(cm.hitRate-wantHeuristic) > tol {
			t.Errorf("hitRate (heuristic) = %f, want ~%f", cm.hitRate, wantHeuristic)
		}
	})

	t.Run("no server cache reported — heuristic fallback", func(t *testing.T) {
		s := newCompletionStream(50)
		s.Add(completionRecord{
			completedAt: time.Now(),
			isColdStart: true,
			inputTokens: 1000,
		})
		s.Add(completionRecord{
			completedAt: time.Now(),
			isColdStart: false,
			cacheHit:    false,
			inputTokens: 1000,
		})
		s.Add(completionRecord{
			completedAt: time.Now(),
			isColdStart: false,
			cacheHit:    true,
			inputTokens: 1000,
		})

		cm := s.CacheMetrics(10, 1)
		if cm.serverReported {
			t.Fatal("expected serverReported=false")
		}
		if cm.DisplayHitRate() != cm.hitRate {
			t.Errorf("DisplayHitRate() = %f, want hitRate = %f", cm.DisplayHitRate(), cm.hitRate)
		}
	})
}

// TestWarmTokensIncludeServerCache reproduces the multi-backend anomaly where an
// aggressively-caching backend reported near-zero "warm" tokens despite serving
// more requests. Providers subtract server-cached tokens out of prompt_tokens, so
// inputTokens is net-of-cache; the cold/warm split must add the cached tokens back
// to reflect full prompt volume, otherwise it silently depends on each backend's
// server-cache behavior and is not comparable across backends.
//
// With the ratio-based bucketing, warm/cold is driven by localCacheRatio: a warm
// request with localCacheRatio=1.0 means 100% of the full prompt was cached by the
// content-level estimator. This matches "the series has already submitted this prefix".
func TestWarmTokensIncludeServerCache(t *testing.T) {
	const tol = 1e-9

	// addBackend feeds nWarm warm requests after one cold start. fullPrompt is the
	// logical prompt size of every request; cachedPerWarm is how much of each warm
	// request the server served from its KV cache (so net input = fullPrompt-cached).
	// localRatioWarm is the content-level cache ratio for warm requests (1.0 = fully warm).
	addBackend := func(nWarm, fullPrompt, cachedPerWarm int, localRatioWarm float64) *completionStream {
		s := newCompletionStream(10_000)
		// Cold start: localCacheRatio=0 (first request, nothing cached yet).
		s.Add(completionRecord{completedAt: time.Now(), isColdStart: true, inputTokens: fullPrompt, localCacheRatio: 0})
		for i := 0; i < nWarm; i++ {
			s.Add(completionRecord{
				completedAt:     time.Now(),
				isColdStart:     false,
				inputTokens:     fullPrompt - cachedPerWarm, // net-of-cache, as providers report it
				cachedTokens:    cachedPerWarm,
				localCacheRatio: localRatioWarm,
			})
		}
		return s
	}

	// "weka": more requests, almost the whole prompt served from cache (ratio=1.0 = fully warm).
	weka := addBackend(600, 1000, 990, 1.0)
	// "vanilla": fewer requests, little server caching (ratio=1.0 too — still a repeated series).
	vanilla := addBackend(500, 1000, 100, 1.0)

	wt := weka.TokenTotals()
	vt := vanilla.TokenTotals()

	// The core anomaly: despite serving MORE requests, weka's warm must not be
	// smaller than vanilla's. Warm reflects full prompt volume, not net-of-cache.
	if wt.inputWarm <= vt.inputWarm {
		t.Errorf("warm regression: weka warm=%d (600 reqs) should exceed vanilla warm=%d (500 reqs)", wt.inputWarm, vt.inputWarm)
	}
	// With localCacheRatio=1.0: warm = round(full * 1.0) = full = inputTokens + cachedTokens = 1000.
	if want := int64(600 * 1000); wt.inputWarm != want {
		t.Errorf("weka warm = %d, want %d (full prompt = net + cached)", wt.inputWarm, want)
	}

	// Server-cache fraction must stay in [0,1] even when the cache covers ~all of
	// the prompt (the old cached/net formula blew past 100%).
	cm := weka.CacheMetrics(10_000, 1)
	if cm.serverTokenRate > 1.0 || cm.serverTokenRate < 0 {
		t.Errorf("serverTokenRate = %f out of [0,1]", cm.serverTokenRate)
	}
	// cached/(net+cached) over the warm window = 990/1000 = 0.99 (cold contributes
	// 1000 net, 0 cached; with 600 warm it's ~0.99).
	wantRate := float64(600*990) / float64(600*1000+1000)
	if math.Abs(cm.serverTokenRate-wantRate) > tol {
		t.Errorf("serverTokenRate = %f, want ~%f", cm.serverTokenRate, wantRate)
	}

	// Summary "Server cache: uncached = total - cached" must stay non-negative.
	total := wt.inputCold + wt.inputWarm
	if total-wt.cached < 0 {
		t.Errorf("uncached = %d negative: total=%d cached=%d", total-wt.cached, total, wt.cached)
	}
}

func TestDecodeRateFromDelta(t *testing.T) {
	cases := []struct {
		name               string
		warmOutputTokens   int
		decodeOutputTokens int
		warmWall           time.Duration
		decodeWall         time.Duration
		wantApprox         float64 // expected value; 0 means expect exactly 0
		wantZero           bool
	}{
		{
			name:               "basic case",
			warmOutputTokens:   4,   // 4 requests * 1 token each (warm phase max_tokens=1)
			decodeOutputTokens: 400, // 4 requests * ~100 tokens each
			warmWall:           1 * time.Second,
			decodeWall:         2 * time.Second,
			// (400 - 4) / 1.0 = 396
			wantApprox: 396,
		},
		{
			name:               "net tokens <= 0 yields zero",
			warmOutputTokens:   100,
			decodeOutputTokens: 100,
			warmWall:           1 * time.Second,
			decodeWall:         2 * time.Second,
			wantZero:           true,
		},
		{
			name:               "decodeWall == warmWall yields zero",
			warmOutputTokens:   4,
			decodeOutputTokens: 400,
			warmWall:           2 * time.Second,
			decodeWall:         2 * time.Second,
			// dt == 0, decodeRateFromDelta returns 0
			wantZero: true,
		},
		{
			name:               "decodeWall < warmWall yields zero",
			warmOutputTokens:   4,
			decodeOutputTokens: 400,
			warmWall:           3 * time.Second,
			decodeWall:         2 * time.Second,
			// dt < 0, decodeRateFromDelta returns 0
			wantZero: true,
		},
		{
			name:               "large pool",
			warmOutputTokens:   64,     // 64 requests * 1 token each
			decodeOutputTokens: 256000, // 64 requests * ~4000 tokens each
			warmWall:           5 * time.Second,
			decodeWall:         10 * time.Second,
			// (256000 - 64) / 5.0 = 51187.2
			wantApprox: 51187.2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := decodeRateFromDelta(tc.warmOutputTokens, tc.decodeOutputTokens, tc.warmWall, tc.decodeWall)
			if tc.wantZero {
				if got != 0 {
					t.Errorf("expected 0, got %f", got)
				}
				return
			}
			// Allow 0.01% relative tolerance for floating-point.
			diff := got - tc.wantApprox
			if diff < 0 {
				diff = -diff
			}
			if tc.wantApprox > 0 && diff/tc.wantApprox > 0.0001 {
				t.Errorf("got %f, want ~%f (diff %f)", got, tc.wantApprox, diff)
			}
		})
	}
}

func TestTokenTotalsServerCached(t *testing.T) {
	s := newCompletionStream(50)
	s.Add(completionRecord{completedAt: time.Now(), inputTokens: 500, cachedTokens: 100})
	s.Add(completionRecord{completedAt: time.Now(), inputTokens: 500, cachedTokens: 0})
	s.Add(completionRecord{completedAt: time.Now(), inputTokens: 500, cachedTokens: 250})
	// error record — must NOT be counted
	s.Add(completionRecord{completedAt: time.Now(), isError: true, cachedTokens: 999})
	tt := s.TokenTotals()
	if tt.cached != 350 {
		t.Errorf("cached = %d, want 350", tt.cached)
	}
}

func TestEffectiveColdPrefillConcurrency(t *testing.T) {
	cases := []struct {
		name     string
		general  int
		override int
		want     int
	}{
		{
			name:     "override 0 returns general",
			general:  16,
			override: 0,
			want:     16,
		},
		{
			name:     "positive override returns override",
			general:  16,
			override: 4,
			want:     4,
		},
		{
			name:     "negative override returns general",
			general:  16,
			override: -1,
			want:     16,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := effectiveColdPrefillConcurrency(tc.general, tc.override)
			if got != tc.want {
				t.Errorf("effectiveColdPrefillConcurrency(%d, %d) = %d, want %d", tc.general, tc.override, got, tc.want)
			}
		})
	}
}

// TestAddLocalCacheRatioBucketing verifies ratio-based warm/cold bucketing:
// ratio=0 → all cold, ratio=1 → all warm, GlobalLocalCacheRate = token-weighted mean.
func TestAddLocalCacheRatioBucketing(t *testing.T) {
	s := newCompletionStream(100)
	// Cold: ratio=0, full=1000
	s.Add(completionRecord{
		completedAt:     time.Now(),
		inputTokens:     1000,
		cachedTokens:    0,
		localCacheRatio: 0.0,
	})
	// Warm: ratio=1, full=1000
	s.Add(completionRecord{
		completedAt:     time.Now(),
		inputTokens:     10,
		cachedTokens:    990,
		localCacheRatio: 1.0,
	})

	tt := s.TokenTotals()
	// ratio=0 → warm=0, cold=1000
	// ratio=1 → warm=1000, cold=0
	if tt.inputCold != 1000 {
		t.Errorf("inputCold=%d, want 1000", tt.inputCold)
	}
	if tt.inputWarm != 1000 {
		t.Errorf("inputWarm=%d, want 1000", tt.inputWarm)
	}

	rate := s.GlobalLocalCacheRate()
	// warm/(warm+cold) = 1000/2000 = 0.5
	if math.Abs(rate-0.5) > 1e-9 {
		t.Errorf("GlobalLocalCacheRate()=%f, want 0.5", rate)
	}
}
