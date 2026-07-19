package benchmark

import (
	"strings"
	"testing"
	"time"
)

func TestMatchesExpectedUUIDList(t *testing.T) {
	expected := []string{"uuid-a", "uuid-b", "uuid-c"}
	tests := []struct {
		name     string
		response string
		want     bool
	}{
		// PASS: only separator whitespace differs.
		{"exact comma-joined", "uuid-a,uuid-b,uuid-c", true},
		{"comma-space joined (DeepSeek style)", "uuid-a, uuid-b, uuid-c", true},
		{"trailing separator then newline", "uuid-a, uuid-b, uuid-c,\n", true},
		{"surrounding whitespace on elements", "  uuid-a ,uuid-b,  uuid-c  ", true},
		// FAIL: content differs, not just whitespace.
		{"extra trailing chatty token", "uuid-a, uuid-b, uuid-c, done", false},
		{"missing a uuid", "uuid-a, uuid-b", false},
		{"reordered pair", "uuid-a, uuid-c, uuid-b", false},
		{"duplicated uuid (interior empty not dropped analog)", "uuid-a, uuid-b, uuid-b, uuid-c", false},
		{"interior empty from doubled separator", "uuid-a,, uuid-b, uuid-c", false},
		{"chatty prefix", "The UUIDs are: uuid-a, uuid-b, uuid-c", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matchesExpectedUUIDList(tt.response, expected); got != tt.want {
				t.Errorf("matchesExpectedUUIDList(%q) = %v, want %v", tt.response, got, tt.want)
			}
		})
	}
}

func TestComputeStampsPerSeries(t *testing.T) {
	tests := []struct {
		name         string
		garbageChars int
		want         int
	}{
		{"default 400000 chars ~49 stamps", 400000, 49}, // 400000/8192 = 48.828... rounds to 49
		{"legacy 100000-char budget", 100000, 12},       // 100000/8192 = 12.207 rounds to 12
		{"exact multiple", 16384, 2},                    // 16384/8192 = 2, already >= 2
		{"zero garbage floors to 2", 0, 2},
		{"tiny garbage floors to 2", 100, 2},
		{"one interval rounds up to 2 minimum", 8192, 2}, // 8192/8192 = 1 -> floored to 2
		// Exercise the constant itself (interval-relative, not a magic number): a budget
		// of exactly 5*stampIntervalChars must yield exactly 5 stamps.
		{"five intervals exactly", 5 * stampIntervalChars, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeStampsPerSeries(tt.garbageChars)
			if got != tt.want {
				t.Errorf("computeStampsPerSeries(%d) = %d, want %d", tt.garbageChars, got, tt.want)
			}
			if got < 2 {
				t.Errorf("computeStampsPerSeries(%d) = %d, must always be >= 2", tt.garbageChars, got)
			}
		})
	}

	// Guard the load-preservation contract at the default budget: the OLD default
	// (--garbage-tokens 100000 -> Repeat("A", 100000*4) = 400000 chars) must map to 49
	// stamps under the 8192-char interval.
	if got := computeStampsPerSeries(400000); got != 49 {
		t.Errorf("default 400000-char budget: computeStampsPerSeries = %d, want 49", got)
	}
}

func TestComputeMaxOutputTokens(t *testing.T) {
	tests := []struct {
		name     string
		numUUIDs int
	}{
		{"single uuid", 1},
		{"default-ish 49 uuids", 49},
		{"large 196 uuids", 196},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeMaxOutputTokens(tt.numUUIDs, 3.0)
			if got <= 0 {
				t.Fatalf("computeMaxOutputTokens(%d, 3.0) = %d, want > 0", tt.numUUIDs, got)
			}
			// Must be roughly 3x the naive expected-token estimate (uuid string is
			// 36 chars, +1 for the separating comma, /4 chars-per-token).
			expectedChars := tt.numUUIDs*expectedUUIDStrLen + (tt.numUUIDs - 1)
			expectedTokens := (expectedChars + 3) / 4 // ceil div by 4
			wantMin := expectedTokens * 3
			if got != wantMin {
				t.Errorf("computeMaxOutputTokens(%d, 3.0) = %d, want %d", tt.numUUIDs, got, wantMin)
			}
		})
	}

	// Monotonic: more uuids never yields a smaller budget.
	prev := 0
	for n := 1; n <= 50; n++ {
		got := computeMaxOutputTokens(n, 3.0)
		if got < prev {
			t.Fatalf("computeMaxOutputTokens not monotonic at n=%d: got %d < prev %d", n, got, prev)
		}
		prev = got
	}

	// Multiplier is configurable (--max-output-multiplier): a higher multiplier must
	// yield a strictly larger budget for the same UUID count, giving reasoning models
	// headroom to think before answering.
	if lo, hi := computeMaxOutputTokens(49, 3.0), computeMaxOutputTokens(49, 6.0); hi <= lo {
		t.Errorf("computeMaxOutputTokens(49, 6.0) = %d, want > computeMaxOutputTokens(49, 3.0) = %d", hi, lo)
	}
}

func TestBuildCoherencySeriesPrompt(t *testing.T) {
	garbageBlock := strings.Repeat("A", stampIntervalChars)

	t.Run("two uuids: one garbage block between them", func(t *testing.T) {
		uuids := []string{"uuid-0", "uuid-1"}
		got := buildCoherencySeriesPrompt(uuids, garbageBlock)
		want := "<request>uuid-0 " + garbageBlock + " uuid-1</request>"
		if got != want {
			t.Errorf("buildCoherencySeriesPrompt() mismatch:\ngot:  %q\nwant: %q", got, want)
		}
	})

	t.Run("three uuids: two garbage blocks, interleaved", func(t *testing.T) {
		uuids := []string{"uuid-0", "uuid-1", "uuid-2"}
		got := buildCoherencySeriesPrompt(uuids, garbageBlock)
		want := "<request>uuid-0 " + garbageBlock + " uuid-1 " + garbageBlock + " uuid-2</request>"
		if got != want {
			t.Errorf("buildCoherencySeriesPrompt() mismatch:\ngot:  %q\nwant: %q", got, want)
		}
		// Sanity: N uuids -> N-1 garbage blocks, and the prompt starts/ends correctly.
		if strings.Count(got, garbageBlock) != len(uuids)-1 {
			t.Errorf("expected %d garbage blocks, got %d", len(uuids)-1, strings.Count(got, garbageBlock))
		}
		if !strings.HasPrefix(got, "<request>"+uuids[0]) {
			t.Errorf("prompt must start with the first uuid right after <request>")
		}
		if !strings.HasSuffix(got, uuids[len(uuids)-1]+"</request>") {
			t.Errorf("prompt must end with the last uuid right before </request>")
		}
	})
}

func TestModelSpecifiesMaxTokens(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantSet   bool
		wantError bool
	}{
		{"plain static model, no params", "anthropic/haiku", false, false},
		{"static model with max_tokens", "zai/glm-4.6,max_tokens=200", true, false},
		{"static model with unrelated param only", "anthropic/haiku", false, false},
		{"dynamic model, no max_tokens", "dynamic/http://localhost:8000/v1,type=openai", false, false},
		{"dynamic model with max_tokens", "dynamic/http://localhost:8000/v1,type=openai,max_tokens=128", true, false},
		{"openrouter model, no max_tokens", "openrouter/anthropic/claude-3.5-sonnet", false, false},
		{"openrouter model with max_tokens", "openrouter/anthropic/claude-3.5-sonnet,max_tokens=2048", true, false},
		{"malformed static params -> error", "model,max_tokens=notanumber", false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := modelSpecifiesMaxTokens(tt.model)
			if tt.wantError {
				if err == nil {
					t.Fatalf("modelSpecifiesMaxTokens(%q) expected error, got nil", tt.model)
				}
				return
			}
			if err != nil {
				t.Fatalf("modelSpecifiesMaxTokens(%q) unexpected error: %v", tt.model, err)
			}
			if got != tt.wantSet {
				t.Errorf("modelSpecifiesMaxTokens(%q) = %v, want %v", tt.model, got, tt.wantSet)
			}
		})
	}
}

// TestSeededUUIDGenerationDeterministic verifies that the same --seed always
// produces the same UUID sequence — the load-bearing property behind
// "identical seed => identical prompts" (RunCacheCoherencyEval derives its
// UUID reader identically to this).
func TestSeededUUIDGenerationDeterministic(t *testing.T) {
	genN := func(seed int64, n int) []string {
		gen := newUUIDGenerator(seed)
		out := make([]string, n)
		for i := range out {
			out[i] = gen()
		}
		return out
	}

	a := genN(42, 10)
	b := genN(42, 10)
	if len(a) != len(b) {
		t.Fatalf("length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("uuid[%d] mismatch for same seed: %q vs %q", i, a[i], b[i])
		}
	}

	c := genN(43, 10)
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Errorf("different seeds produced identical uuid sequences")
	}
}

// TestBuildCoherencySharedSeriesPrompt verifies the --shared-prefix-per-series prompt
// shape: a byte-identical leading shared prefix reused across peers, with only the
// per-series UUID stamps trailing it (never inside the shared prefix, so block 0 of the
// KV hash chain is genuinely shared).
func TestBuildCoherencySharedSeriesPrompt(t *testing.T) {
	sharedPrefix := "[group 0] " + strings.Repeat("A", 64)
	uuidsA := []string{"uuid-a0", "uuid-a1"}
	uuidsB := []string{"uuid-b0", "uuid-b1"}

	promptA := buildCoherencySharedSeriesPrompt(uuidsA, sharedPrefix)
	promptB := buildCoherencySharedSeriesPrompt(uuidsB, sharedPrefix)

	wantA := "<request>[group 0] " + strings.Repeat("A", 64) + " uuid-a0 uuid-a1</request>"
	if promptA != wantA {
		t.Errorf("promptA = %q, want %q", promptA, wantA)
	}

	// Two series in the same group must share a byte-identical LEADING run — the whole
	// shared prefix — diverging only where their unique stamps begin.
	if !strings.HasPrefix(promptA, "<request>"+sharedPrefix+" ") ||
		!strings.HasPrefix(promptB, "<request>"+sharedPrefix+" ") {
		t.Errorf("both prompts must start with <request>+sharedPrefix; got\nA=%q\nB=%q", promptA, promptB)
	}
	commonLen := len("<request>" + sharedPrefix + " ")
	if promptA[:commonLen] != promptB[:commonLen] {
		t.Errorf("shared leading run diverged within group:\nA=%q\nB=%q", promptA[:commonLen], promptB[:commonLen])
	}
	// The unique stamps must appear in the tail, after the shared prefix.
	if idx := strings.Index(promptA, "uuid-a0"); idx < commonLen {
		t.Errorf("per-series UUID leaked into the shared prefix region (idx %d < %d)", idx, commonLen)
	}
}

// TestResetPrefixCacheURL verifies the vLLM admin URL is derived correctly from an
// OpenAI-compatible dynamic-model base URL — vLLM mounts /reset_prefix_cache at the
// server root, not under /v1, so the conventional trailing "v1/" must be stripped.
func TestResetPrefixCacheURL(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		want    string
	}{
		{"conventional /v1/ base", "http://localhost:8000/v1/", "http://localhost:8000/reset_prefix_cache"},
		{"no trailing slash on v1", "http://localhost:8000/v1", "http://localhost:8000/reset_prefix_cache"},
		{"already at server root", "http://localhost:8000/", "http://localhost:8000/reset_prefix_cache"},
		{"root, no trailing slash", "http://localhost:8000", "http://localhost:8000/reset_prefix_cache"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resetPrefixCacheURL(tt.baseURL); got != tt.want {
				t.Errorf("resetPrefixCacheURL(%q) = %q, want %q", tt.baseURL, got, tt.want)
			}
		})
	}
}

// TestResolveResetPrefixCacheURL verifies --reset-every-n only resolves a URL for
// dynamic/ vLLM endpoint specs — the reset_prefix_cache admin endpoint has no
// equivalent on hosted providers (anthropic/, openai/, openrouter/, etc.), so those
// must error out (the CLI layer turns that into a no-op-with-warning).
func TestResolveResetPrefixCacheURL(t *testing.T) {
	url, err := resolveResetPrefixCacheURL("dynamic/http://localhost:8000/v1,type=openai_vllm")
	if err != nil {
		t.Fatalf("unexpected error for dynamic/ model: %v", err)
	}
	if want := "http://localhost:8000/reset_prefix_cache"; url != want {
		t.Errorf("resolveResetPrefixCacheURL() = %q, want %q", url, want)
	}

	if _, err := resolveResetPrefixCacheURL("anthropic/sonnet"); err == nil {
		t.Errorf("expected error for non-dynamic model, got nil")
	}
}

// TestAbortDelay verifies --abort-delay-ms's two modes: a positive fixed value is
// used verbatim; 0 draws a delay bounded by the live cold-TTFT window estimate.
func TestAbortDelay(t *testing.T) {
	window := newAbortWindowEstimator()

	if got, want := abortDelay(500, window), 500*time.Millisecond; got != want {
		t.Errorf("abortDelay(500, ...) = %v, want %v", got, want)
	}

	// 0 = random in [0, window). Sample repeatedly and check bounds against the
	// (untouched, so still default-seeded) window.
	upper := time.Duration(defaultAbortWindowMs) * time.Millisecond
	for i := 0; i < 100; i++ {
		got := abortDelay(0, window)
		if got < 0 || got > upper {
			t.Fatalf("abortDelay(0, ...) = %v, want in [0, %v]", got, upper)
		}
	}

	// observe() should shift the window's mean, and therefore future random delays'
	// upper bound.
	window.observe(100)
	if got := window.windowMs(); got != 100 {
		t.Errorf("windowMs() after single observe(100) = %v, want 100", got)
	}
}
