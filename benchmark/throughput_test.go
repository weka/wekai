package benchmark

import (
	"math"
	"strings"
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

// TestNoCacheDataObserved locks down the distinction noCacheDataObserved
// exists to draw: a genuinely absent cache signal (server not caching, or not
// reporting it) versus a cache that is working but happens to show 0% right
// now (e.g. still warming up, or a heuristic miss offset by a nonzero server
// report).
func TestNoCacheDataObserved(t *testing.T) {
	cases := []struct {
		name string
		cm   cacheMetrics
		min  int
		want bool
	}{
		{"neither signal ever fired, enough records", cacheMetrics{count: 20, hitRate: 0, serverReported: false}, 10, true},
		{"below minCount — too early to tell", cacheMetrics{count: 5, hitRate: 0, serverReported: false}, 10, false},
		{"server reported even though heuristic is 0", cacheMetrics{count: 20, hitRate: 0, serverReported: true}, 10, false},
		{"heuristic found hits", cacheMetrics{count: 20, hitRate: 0.4, serverReported: false}, 10, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := noCacheDataObserved(c.cm, c.min); got != c.want {
				t.Errorf("noCacheDataObserved(%+v, %d) = %v, want %v", c.cm, c.min, got, c.want)
			}
		})
	}
}

// TestNoCacheEverObservedUnitCases locks down noCacheEverObserved's own
// all-time semantics in isolation, mirroring TestNoCacheDataObserved's table
// shape for the windowed function it replaces as the cacheWarning signal.
func TestNoCacheEverObservedUnitCases(t *testing.T) {
	cases := []struct {
		name      string
		tt        tokenTotals
		nonErrors int64
		min       int
		want      bool
	}{
		{"neither signal ever fired, enough records", tokenTotals{inputWarm: 0, cached: 0, cacheHits: 0}, 20, 10, true},
		{"below minCount — too early to tell", tokenTotals{inputWarm: 0, cached: 0, cacheHits: 0}, 5, 10, false},
		{"server reported cached tokens", tokenTotals{inputWarm: 0, cached: 500, cacheHits: 0}, 20, 10, false},
		{"heuristic-only warm tokens, no confirmed hits — must still warn", tokenTotals{inputWarm: 500, cached: 0, cacheHits: 0}, 20, 10, true},
		{"confirmed cache hits, no estimator warm — must not warn", tokenTotals{inputWarm: 0, cached: 0, cacheHits: 5}, 20, 10, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := noCacheEverObserved(c.tt, c.nonErrors, c.min); got != c.want {
				t.Errorf("noCacheEverObserved(%+v, %d, %d) = %v, want %v", c.tt, c.nonErrors, c.min, got, c.want)
			}
		})
	}
}

// TestNoCacheEverObservedSurvivesColdStartWindow is the regression test for
// the sticky-latch bug: a windowed check (noCacheDataObserved/CacheMetrics)
// can see an all-cold-start window first — e.g. right after a series
// scale-up, or at a pass boundary under --replay-reuse-sessions — and that
// window alone genuinely has hitRate==0 and serverReported==false, because a
// cold start is always a structural miss. The old code latched cacheWarning
// true the moment that happened and never cleared it, even once later
// windows showed the server caching normally. noCacheEverObserved reads the
// all-time running counters instead, so once real cache activity has
// accumulated anywhere in the run, it must report false — regardless of what
// an early window looked like.
func TestNoCacheEverObservedSurvivesColdStartWindow(t *testing.T) {
	s := newCompletionStream(500)
	const minCount = 10

	// An all-cold-start window: minCount requests, each a structural miss,
	// server reporting nothing yet. This is exactly the window shape that
	// used to latch cacheWarning permanently.
	for i := 0; i < minCount; i++ {
		s.Add(completionRecord{
			completedAt: time.Now(),
			isColdStart: true,
			inputTokens: 1000,
		})
	}

	tt := s.TokenTotals()
	if !noCacheEverObserved(tt, minCount, minCount) {
		t.Fatal("expected noCacheEverObserved=true right after an all-cold-start window with no cache signal yet")
	}

	// The run continues and the server starts reporting real cache hits —
	// both the TTFT heuristic (localCacheRatio>0, driving globalInputWarm)
	// and explicit server-reported cached_tokens.
	for i := 0; i < minCount; i++ {
		s.Add(completionRecord{
			completedAt:          time.Now(),
			isColdStart:          false,
			cacheHit:             true,
			serverCacheConfirmed: true,
			cachedTokens:         800,
			inputTokens:          200,
			localCacheRatio:      0.8,
		})
	}

	tt = s.TokenTotals()
	if noCacheEverObserved(tt, 2*minCount, minCount) {
		t.Error("expected noCacheEverObserved=false once the run-wide counters show real cache activity — " +
			"the warning must not stay latched from the earlier all-cold-start window")
	}
}

// TestNoCacheEverObservedFailureMode1EstimatorWarmNeverConfirmed is the
// regression test for failure mode 1: a repeated prompt can make the
// client-side content estimator (localCacheRatio, driving globalInputWarm)
// report warm reuse even though the server never actually caches anything —
// cacheHit stays false and cachedTokens stays 0 on every record. Before this
// fix, tt.inputWarm alone was enough to suppress the warning, so this exact
// shape — real 0% server-reported hit rate, estimator convinced otherwise —
// went unflagged. The fix drops inputWarm from the check entirely, so the
// warning must still fire here.
func TestNoCacheEverObservedFailureMode1EstimatorWarmNeverConfirmed(t *testing.T) {
	s := newCompletionStream(500)
	const minCount = 10

	for i := 0; i < minCount; i++ {
		s.Add(completionRecord{
			completedAt:     time.Now(),
			isColdStart:     i == 0,
			cacheHit:        false,
			cachedTokens:    0,
			inputTokens:     1000,
			localCacheRatio: 0.9, // estimator convinced this prompt is mostly cached...
		})
	}

	tt := s.TokenTotals()
	if tt.inputWarm == 0 {
		t.Fatal("test setup bug: expected globalInputWarm to be nonzero from the high localCacheRatio")
	}
	if !noCacheEverObserved(tt, minCount, minCount) {
		t.Error("expected noCacheEverObserved=true: estimator-only warm reuse with cacheHit never true and " +
			"cachedTokens always 0 must not suppress the warning")
	}
}

// TestNoCacheEverObservedFailureMode2TTFTConfirmedWithoutEstimatorWarm is the
// regression test for failure mode 2: a server that is genuinely fast via
// real caching (cacheHit true via the TTFT heuristic) but with the
// content-level estimator seeing no repeated/warm prompt volume
// (localCacheRatio 0, so globalInputWarm never grows) and no explicit
// cachedTokens reported either. Before this fix, tt.inputWarm == 0 and
// tt.cached == 0 here meant the warning fired right next to real, confirmed
// cache activity — the exact "sticky latch" bug this function exists to
// avoid. The fix's tt.cacheHits counter is incremented from cacheHit
// directly, so the warning must not fire.
func TestNoCacheEverObservedFailureMode2TTFTConfirmedWithoutEstimatorWarm(t *testing.T) {
	s := newCompletionStream(500)
	const minCount = 10

	for i := 0; i < minCount; i++ {
		s.Add(completionRecord{
			completedAt:     time.Now(),
			isColdStart:     i == 0,
			cacheHit:        i != 0, // every non-cold request is a confirmed TTFT hit
			cachedTokens:    0,
			inputTokens:     1000,
			localCacheRatio: 0, // estimator sees no repeated content at all
		})
	}

	tt := s.TokenTotals()
	if tt.inputWarm != 0 {
		t.Fatal("test setup bug: expected globalInputWarm to stay zero with localCacheRatio always 0")
	}
	if tt.cacheHits == 0 {
		t.Fatal("test setup bug: expected globalCacheHits to be nonzero from the confirmed TTFT hits")
	}
	if noCacheEverObserved(tt, minCount, minCount) {
		t.Error("expected noCacheEverObserved=false: confirmed TTFT cache hits with no estimator-detected " +
			"warm reuse and no server-reported cachedTokens must not trigger the warning")
	}
}

// TestCacheWarningMessage locks down that the warning names the actual
// server-launch flag for vLLM/SGLang, rather than a generic message that
// leaves the operator to guess between "server doesn't cache at all" and
// "server caches but wasn't launched with the flag that reports it" — the
// distinction that prompted this message in the first place (see
// cacheWarningMessage's doc comment for the exact flags and why both a
// client-side opt-in and a server-side flag are required).
func TestCacheWarningMessage(t *testing.T) {
	cases := []struct {
		name         string
		spec         string
		wantContains string
	}{
		{"vllm names its flag", "dynamic/http://h:1/v1,type=openai_vllm,model=m", "--enable-prompt-tokens-details"},
		{"sglang names its flag", "dynamic/http://h:1/v1,type=openai_sglang,model=m", "--enable-cache-report"},
		{"plain openai gets the generic message", "dynamic/http://h:1/v1,type=openai,model=m", "may not support"},
		{"non-dynamic model gets the generic message", "gpt-4", "may not support"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := cacheWarningMessage(c.spec)
			if !strings.Contains(got, c.wantContains) {
				t.Errorf("cacheWarningMessage(%q) = %q, want it to contain %q", c.spec, got, c.wantContains)
			}
		})
	}
}

// TestIsOpenAISGLangSpec locks down the exact gate the no-usage-ever warning
// (noUsageWarningMessage) uses: it must fire only for type=openai_sglang,
// never for vLLM, plain openai, or a non-dynamic model spec.
func TestIsOpenAISGLangSpec(t *testing.T) {
	cases := []struct {
		spec string
		want bool
	}{
		{"dynamic/http://h:1/v1,type=openai_sglang,model=m", true},
		{"dynamic/http://h:1/v1,type=openai_vllm,model=m", false},
		{"dynamic/http://h:1/v1,type=openai,model=m", false},
		{"dynamic/http://h:1/v1,model=m", false}, // default type
		{"gpt-4", false},
	}
	for _, c := range cases {
		t.Run(c.spec, func(t *testing.T) {
			if got := isOpenAISGLangSpec(c.spec); got != c.want {
				t.Errorf("isOpenAISGLangSpec(%q) = %v, want %v", c.spec, got, c.want)
			}
		})
	}
}

// TestNoUsageWarningMessage is a light smoke test — the exact wording isn't
// load-bearing, but it must name the actual symptom (the ingest chart) and
// the actual mitigating flag, mirroring TestCacheWarningMessage's rationale.
func TestNoUsageWarningMessage(t *testing.T) {
	got := noUsageWarningMessage()
	for _, want := range []string{"usage", "ingest", "--enable-cache-report"} {
		if !strings.Contains(strings.ToLower(got), strings.ToLower(want)) {
			t.Errorf("noUsageWarningMessage() = %q, want it to mention %q", got, want)
		}
	}
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
