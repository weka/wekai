package benchmark

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/weka/wekai/llm"
)

// captureStderr runs fn with os.Stderr redirected and returns what was
// written. The contamination report goes to stderr by design — it must reach
// the operator's terminal even when stdout is a progress display — so the
// tests read it from where the operator would.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stderr = w
	done := make(chan string)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	fn()
	w.Close()
	os.Stderr = old
	return <-done
}

// TestContaminationStopsTheRunWithForensics is the contract of the default
// mode: a leaked marker prints everything a person needs to chase it — both
// sessions, the marker, the exact --replay-series-indices pair — and arms the
// stop. The eleven-hour run that motivated this saw one event in 65771
// responses, and its entire trace was the digit 1 in the final summary.
func TestContaminationStopsTheRunWithForensics(t *testing.T) {
	const stamp = "contam-run"
	other := buildSessionUUIDs(RouterReplaySession{
		SessionID: "victim-session",
		Instances: []RouterReplayInstance{{
			Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("their-turn", 40)}}},
		}},
	}, stamp)
	foreign := other.uuids[0]

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		fmt.Fprintf(w, `{"id":"c","object":"chat.completion","model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
			"as I recall: "+foreign)
	}))
	defer ts.Close()

	mineSess := RouterReplaySession{
		SessionID: "probe-session",
		fileIdx:   335,
		Instances: []RouterReplayInstance{{
			Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("my-turn", 40)}}},
		}},
	}
	mine := buildSessionUUIDs(mineSess, stamp)

	run := func(continueOn bool) (*autoState, string) {
		reg := newUUIDRegistry()
		reg.Acquire(other.uuids, uuidHolder{Series: 7, SessionID: "victim-session", FileIdx: 260})
		reg.Acquire(mine.uuids, uuidHolder{Series: 12, SessionID: "probe-session", FileIdx: 335})
		p := newContamTestPoster(t, ts.URL, reg, continueOn)
		st := &autoState{stream: newCompletionStream(200)}
		req := RouterReplayRequest{
			Stream: false, OutputTokens: 400,
			Messages: []RouterReplayMessage{userText("my-turn", 40)},
		}
		out := captureStderr(t, func() {
			m := p.do(context.Background(), req, strings.Repeat("docs ", 60), 1,
				"probe-session", "i1", 12, st, mine)
			if m.Error != nil {
				t.Fatalf("request failed: %v", m.Error)
			}
			if len(m.LeakedUUIDs) != 1 {
				t.Fatalf("LeakedUUIDs = %v, want the planted leak", m.LeakedUUIDs)
			}
		})
		return st, out
	}

	t.Run("default stops and prints everything", func(t *testing.T) {
		st, out := run(false)
		if !st.contaminationStop.Load() {
			t.Error("contaminationStop not armed: the run would sail on and churn the caches " +
				"the investigation needs intact")
		}
		for _, want := range []string{
			"CROSS_CONTAMINATION",
			foreign,                           // the leaked marker itself
			"victim-session",                  // whose it is
			"probe-session",                   // who received it
			"file-index=260",                  // owner's replay index
			"file-index=335",                  // this session's replay index
			"--replay-series-indices=260,335", // the copy-pastable repro
			"stopping the run",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("report is missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("opt-out continues but still prints in full", func(t *testing.T) {
		st, out := run(true)
		if st.contaminationStop.Load() {
			t.Error("contaminationStop armed despite --verify-continue-on-contamination")
		}
		for _, want := range []string{"CROSS_CONTAMINATION", foreign, "--replay-series-indices=260,335", "continuing"} {
			if !strings.Contains(out, want) {
				t.Errorf("opt-out report is missing %q:\n%s", want, out)
			}
		}
	})
}

// TestGarbagePrintsAsItHappens: each corrupted response produces one stderr
// line with the kind counts and the bytes in context, so the detector can be
// tuned against what it actually fires on rather than a total in the summary.
func TestGarbagePrintsAsItHappens(t *testing.T) {
	const stamp = "garbage-run"
	su := buildSessionUUIDs(RouterReplaySession{
		SessionID: "s",
		Instances: []RouterReplayInstance{{
			Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("turn-a", 40)}}},
		}},
	}, stamp)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		fmt.Fprintf(w, `{"id":"c","object":"chat.completion","model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
			"the whale \ufffd\ufffd surfaced")
	}))
	defer ts.Close()

	reg := newUUIDRegistry()
	reg.Acquire(su.uuids, uuidHolder{Series: 1})
	p := newContamTestPoster(t, ts.URL, reg, false)
	st := &autoState{stream: newCompletionStream(200)}
	req := RouterReplayRequest{
		Stream: false, OutputTokens: 400,
		Messages: []RouterReplayMessage{userText("turn-a", 40)},
	}
	out := captureStderr(t, func() {
		m := p.do(context.Background(), req, strings.Repeat("docs ", 60), 3, "s", "i1", 1, st, su)
		if !m.ResponseGarbage {
			t.Fatal("a response carrying U+FFFD was not flagged as garbage")
		}
	})
	for _, want := range []string{"GARBAGE", "2\u00d7U+FFFD", "turn=3", "whale"} {
		if !strings.Contains(out, want) {
			t.Errorf("garbage line is missing %q:\n%s", want, out)
		}
	}
	if st.contaminationStop.Load() {
		t.Error("garbage armed the contamination stop; it is a signal to read, not a leak")
	}
}

func newContamTestPoster(t *testing.T, url string, reg *uuidRegistry, continueOn bool) *replayPoster {
	t.Helper()
	p, err := newReplayPoster(fmt.Sprintf("dynamic/%s,type=openai,model=m", url),
		llm.APIKeys{OpenAI: "sk-test"}, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}
	p.uuidEnabled = true
	p.registry = reg
	p.continueOnContamination = continueOn
	return p
}
