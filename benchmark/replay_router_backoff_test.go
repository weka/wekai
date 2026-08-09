package benchmark

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/weka/wekai/llm"
)

// A router enforcing a per-backend concurrency cap sheds with 429 and expects
// the caller back; vLLM frees a slot every time a request finishes, so the wait
// is usually milliseconds. Recording the shed as a fatal error made a run
// measure the harness instead of the fleet — against a saturated mock fleet the
// replay reported ~68% errors and stalled with 7 requests in flight while every
// backend was healthy.

func TestBackoff429Schedule(t *testing.T) {
	t.Run("waits stay within the jitter band", func(t *testing.T) {
		for _, base := range []time.Duration{
			retry429Initial, 100 * time.Millisecond, retry429Max,
		} {
			for range 200 {
				got, ok := backoff429(base, 0, retry429Budget)
				if !ok {
					t.Fatalf("base=%v: budget reported spent at zero elapsed", base)
				}
				lo := time.Duration(float64(base) * 0.7)
				hi := time.Duration(float64(base) * 1.3)
				if got < lo || got > hi {
					t.Fatalf("base=%v: wait %v outside [%v, %v]", base, got, lo, hi)
				}
			}
		}
	})

	t.Run("doubling from the initial wait reaches the cap and stops", func(t *testing.T) {
		// The schedule the caller runs: 10ms doubling, clamped at 3s.
		d := retry429Initial
		steps := 0
		for d < retry429Max {
			d = min(2*d, retry429Max)
			steps++
			if steps > 64 {
				t.Fatal("doubling never reached the cap")
			}
		}
		if d != retry429Max {
			t.Fatalf("cap = %v, want %v", d, retry429Max)
		}
	})

	t.Run("the budget is a hard ceiling", func(t *testing.T) {
		if _, ok := backoff429(retry429Initial, retry429Budget, retry429Budget); ok {
			t.Error("a request that has already spent the whole budget was told to wait again")
		}
		// The final wait is clipped so total wait can never exceed the budget.
		spent := retry429Budget - 50*time.Millisecond
		got, ok := backoff429(retry429Max, spent, retry429Budget)
		if !ok {
			t.Fatal("a request with budget left was refused a wait")
		}
		if spent+got > retry429Budget {
			t.Errorf("total wait %v exceeds the %v budget", spent+got, retry429Budget)
		}
	})

	t.Run("a zero budget falls back to the default", func(t *testing.T) {
		if _, ok := backoff429(retry429Initial, 0, 0); !ok {
			t.Error("an unset budget must behave as the default, not as no retries at all")
		}
	})
}

// shedNTimes serves 429 for the first n requests and a valid completion after
// that, counting attempts.
func shedNTimes(n int) (*httptest.Server, func() int) {
	var mu sync.Mutex
	seen := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen++
		shed := seen <= n
		mu.Unlock()
		if shed {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"code":"split_guard_blocked"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}],`+
			`"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
	}))
	return ts, func() int {
		mu.Lock()
		defer mu.Unlock()
		return seen
	}
}

func backoffTestReq() RouterReplayRequest {
	return RouterReplayRequest{
		RequestID: 1, Stream: false, OutputTokens: 10,
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "h1", Bytes: 40, BlockTypes: []string{"text"}},
		},
	}
}

func backoffTestPoster(t *testing.T, base string, budget time.Duration) *replayPoster {
	t.Helper()
	p, err := newReplayPoster(
		fmt.Sprintf("dynamic/%s/v1,type=openai_vllm,model=m", base),
		llm.APIKeys{OpenAI: "sk-test"}, "", "", false, 0, 0, 0, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}
	p.retryBudget = budget
	return p
}

func TestReplayWaitsOut429sRatherThanFailing(t *testing.T) {
	ts, attempts := shedNTimes(3)
	defer ts.Close()

	p := backoffTestPoster(t, ts.URL, retry429Budget)
	st := &autoState{stream: newCompletionStream(200)}

	m := p.do(context.Background(), backoffTestReq(), strings.Repeat("x", 400), 1, "s", "i", 1, st)

	if m.Error != nil {
		t.Fatalf("request failed despite the fleet recovering: %v", m.Error)
	}
	if got := attempts(); got != 4 {
		t.Errorf("server saw %d attempts, want 4 (3 shed + 1 served)", got)
	}
	if m.Retries429 != 3 {
		t.Errorf("Retries429 = %d, want 3", m.Retries429)
	}
	if m.RetryWait <= 0 {
		t.Error("RetryWait not recorded, so the cost of the backoff is invisible in the run")
	}
	// 10 + 20 + 40 ms nominal, jittered; generous upper bound to stay stable
	// on a loaded machine.
	if m.RetryWait > time.Second {
		t.Errorf("RetryWait = %v for three sheds, want the ~70ms schedule", m.RetryWait)
	}
	// TTFT must describe the attempt the server ran, not the waiting; the wait
	// belongs to TotalResponseTime.
	if m.TimeToFirstToken >= m.RetryWait {
		t.Errorf("TimeToFirstToken %v absorbed the %v backoff; it must time the served attempt only",
			m.TimeToFirstToken, m.RetryWait)
	}
	if m.TotalResponseTime < m.RetryWait {
		t.Errorf("TotalResponseTime %v excludes the %v the caller spent waiting",
			m.TotalResponseTime, m.RetryWait)
	}
}

func TestReplayGivesUpAfterTheRetryBudget(t *testing.T) {
	// Never recovers. A short budget keeps the test fast; the schedule and the
	// ceiling arithmetic are covered by TestBackoff429Schedule.
	ts, attempts := shedNTimes(1 << 30)
	defer ts.Close()

	const budget = 200 * time.Millisecond
	p := backoffTestPoster(t, ts.URL, budget)
	st := &autoState{stream: newCompletionStream(200)}

	start := time.Now()
	m := p.do(context.Background(), backoffTestReq(), strings.Repeat("x", 400), 1, "s", "i", 1, st)
	elapsed := time.Since(start)

	if m.Error == nil {
		t.Fatal("a fleet that never recovers must eventually surface the 429")
	}
	if !strings.Contains(m.Error.Error(), "429") || !strings.Contains(m.Error.Error(), "backoff") {
		t.Errorf("error %q does not say the 429 survived the backoff", m.Error)
	}
	if m.RetryWait > budget {
		t.Errorf("waited %v, over the %v budget", m.RetryWait, budget)
	}
	if elapsed > 5*time.Second {
		t.Errorf("gave up after %v; the budget must bound it", elapsed)
	}
	if attempts() < 2 {
		t.Errorf("server saw %d attempts; the client did not retry at all", attempts())
	}
}

// TestReplayDoesNotRetryOtherStatuses pins the scope: only 429 means "come
// back". A 400 is the request's own fault and a 503 means no healthy backend,
// and retrying either would hide a real failure behind a 30-second wait.
func TestReplayDoesNotRetryOtherStatuses(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusServiceUnavailable, http.StatusBadGateway} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			var mu sync.Mutex
			seen := 0
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				seen++
				mu.Unlock()
				w.WriteHeader(status)
			}))
			defer ts.Close()

			p := backoffTestPoster(t, ts.URL, retry429Budget)
			st := &autoState{stream: newCompletionStream(200)}
			m := p.do(context.Background(), backoffTestReq(), strings.Repeat("x", 400), 1, "s", "i", 1, st)

			if m.Error == nil {
				t.Fatalf("status %d was not reported as an error", status)
			}
			mu.Lock()
			defer mu.Unlock()
			if seen != 1 {
				t.Errorf("server saw %d attempts for status %d, want 1", seen, status)
			}
			if m.Retries429 != 0 {
				t.Errorf("Retries429 = %d for status %d, want 0", m.Retries429, status)
			}
		})
	}
}
