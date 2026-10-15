package benchmark

import (
	"testing"
	"time"

	"github.com/weka/wekai/llm"
	"github.com/weka/wekai/tools"
)

// TestUsageCountsReported locks down the condition replay.go (dataset-replay)
// and bench_prompts.go (synthetic) both use to set RequestMetrics.UsageObserved:
// llm.Response.Usage is a plain value, always "present" at its zero value even
// when the server never reported usage, so "at least one count is nonzero" is
// the stand-in for "the usage object was present" that consumeOpenAISSE/
// consumeOpenAIPlain get from a nil check instead.
func TestUsageCountsReported(t *testing.T) {
	cases := []struct {
		name                                  string
		prompt, completion, cached, reasoning int
		want                                  bool
	}{
		{"all zero — usage genuinely absent", 0, 0, 0, 0, false},
		{"prompt tokens only", 10, 0, 0, 0, true},
		{"completion tokens only", 0, 5, 0, 0, true},
		{"cached tokens only", 0, 0, 100, 0, true},
		{"reasoning tokens only", 0, 0, 0, 20, true},
		{"everything populated", 100, 20, 50, 10, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := usageCountsReported(c.prompt, c.completion, c.cached, c.reasoning); got != c.want {
				t.Errorf("usageCountsReported(%d,%d,%d,%d) = %v, want %v",
					c.prompt, c.completion, c.cached, c.reasoning, got, c.want)
			}
		})
	}
}

// TestDatasetReplayUsageObservedWiring exercises the fix for the bug where
// replay.go's flush() populated metrics.UsageData from response.Usage but
// never set metrics.UsageObserved, so a dataset-replay run could never trip
// the openai_sglang no-usage warning no matter what the server reported.
//
// It mirrors flush()'s own construction of RequestMetrics from an *llm.Response
// (see the "if response != nil" block in replay.go) rather than calling
// through replayOneConversation's full request loop, since that closure
// captures an HTTP chat client and isn't independently testable — the
// construction logic itself, and its wiring into recordReplayRequest /
// trackUsageObserved, is what the bug was in.
func TestDatasetReplayUsageObservedWiring(t *testing.T) {
	buildMetrics := func(response *llm.Response) RequestMetrics {
		metrics := RequestMetrics{Response: "ok"}
		if response != nil {
			metrics.UsageData = tools.ExecutionUsageData{
				InputTokens:     tools.TokenUsage{Count: response.Usage.PromptTokens},
				OutputTokens:    tools.TokenUsage{Count: response.Usage.CompletionTokens},
				CachedTokens:    tools.TokenUsage{Count: response.Usage.CachedTokens},
				ReasoningTokens: tools.TokenUsage{Count: response.Usage.ReasoningTokens},
				RequestCount:    1,
			}
			metrics.UsageObserved = usageCountsReported(
				response.Usage.PromptTokens, response.Usage.CompletionTokens,
				response.Usage.CachedTokens, response.Usage.ReasoningTokens)
		}
		return metrics
	}

	cfg := AutoBenchmarkConfig{Model: "dynamic/http://h:1/v1,type=openai_sglang,model=m", MinStabilization: 1}

	t.Run("real usage data present — UsageObserved true, warning does not fire", func(t *testing.T) {
		st := newTestAutoState(4)
		var cold time.Duration
		resp := &llm.Response{Content: "hi", Usage: llm.UsageData{PromptTokens: 100, CompletionTokens: 20, CachedTokens: 80}}
		metrics := buildMetrics(resp)
		if !metrics.UsageObserved {
			t.Fatal("UsageObserved = false, want true: response carried nonzero usage counts")
		}
		recordReplayRequest(cfg, st, nil, metrics, true, &cold)

		noUsageObserved := st.sawResponseEver.Load() && !st.sawUsageEver.Load()
		if noUsageObserved {
			t.Error("no-usage warning would fire, want it suppressed: a response reported real usage")
		}
	})

	t.Run("usage genuinely absent — warning still fires", func(t *testing.T) {
		st := newTestAutoState(4)
		var cold time.Duration
		resp := &llm.Response{Content: "hi"} // zero-value Usage: server never reported it
		metrics := buildMetrics(resp)
		if metrics.UsageObserved {
			t.Fatal("UsageObserved = true, want false: response carried an all-zero usage")
		}
		recordReplayRequest(cfg, st, nil, metrics, true, &cold)

		noUsageObserved := st.sawResponseEver.Load() && !st.sawUsageEver.Load()
		if !noUsageObserved {
			t.Error("no-usage warning would not fire, want it to: no request ever reported usage")
		}
	})
}

// TestTrackUsageObservedSyntheticPath covers the second half of the same bug:
// the synthetic (non-replay) recording block in auto.go never called anything
// equivalent to recordReplayRequest's sawResponseEver/sawUsageEver update, so
// a synthetic run against a server that never reports usage could never
// surface the no-usage warning either — not because usage genuinely wasn't
// observed, but because nothing was tracking it for that run type. Both the
// synthetic block and recordReplayRequest now call the shared
// trackUsageObserved helper, so exercising it here covers both call sites.
func TestTrackUsageObservedSyntheticPath(t *testing.T) {
	t.Run("completed request with real usage latches both flags", func(t *testing.T) {
		st := newTestAutoState(4)
		usageObserved := usageCountsReported(100, 20, 0, 0)
		trackUsageObserved(st, false /* isErr */, false /* skipped */, usageObserved)
		if !st.sawResponseEver.Load() {
			t.Error("sawResponseEver = false, want true: request completed without error")
		}
		if !st.sawUsageEver.Load() {
			t.Error("sawUsageEver = false, want true: request reported real usage")
		}
	})

	t.Run("completed request with no usage leaves sawUsageEver false", func(t *testing.T) {
		st := newTestAutoState(4)
		usageObserved := usageCountsReported(0, 0, 0, 0)
		trackUsageObserved(st, false /* isErr */, false /* skipped */, usageObserved)
		if !st.sawResponseEver.Load() {
			t.Error("sawResponseEver = false, want true: request completed without error")
		}
		if st.sawUsageEver.Load() {
			t.Error("sawUsageEver = true, want false: no usage was ever reported")
		}
	})

	t.Run("errored request counts as neither", func(t *testing.T) {
		st := newTestAutoState(4)
		trackUsageObserved(st, true /* isErr */, false /* skipped */, true /* usageObserved */)
		if st.sawResponseEver.Load() {
			t.Error("sawResponseEver = true, want false: the only request errored")
		}
	})
}
