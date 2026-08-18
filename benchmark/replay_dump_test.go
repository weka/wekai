package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/weka/wekai/llm"
)

// TestDumpCapturesTheWireNotTheParse drives a real exchange through do() and
// checks the three files against what actually crossed the wire.
//
// The capture exists to separate defects that produce identical numbers: a
// marker that never reached the prompt, an instruction the model ignored, and
// a parser that mis-read a good response all report the same PRESENCE_MISS. So
// the assertions are that the request file holds the marker that was sent,
// that the response file holds the SSE framing the server emitted rather than
// the parsed text, and that the metadata carries the verdict derived from them.
func TestDumpCapturesTheWireNotTheParse(t *testing.T) {
	const stamp = "dump-stamp"
	su := buildSessionUUIDs(RouterReplaySession{Instances: []RouterReplayInstance{{
		Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("turn-a", 40)}}},
	}}}, stamp)
	marker := su.uuids[0]

	// Streamed, so the raw capture and the parsed text differ visibly.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", "I recall "+marker)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	dir := t.TempDir()
	dumper, err := newRequestDumper(dir, 10)
	if err != nil {
		t.Fatalf("newRequestDumper: %v", err)
	}
	p, err := newReplayPoster(fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL),
		llm.APIKeys{OpenAI: "sk-test"}, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("newReplayPoster: %v", err)
	}
	p.uuidEnabled = true
	p.registry = newUUIDRegistry()
	p.registry.Acquire(su.uuids, 1)
	p.dumper = dumper

	req := RouterReplayRequest{
		Stream: true, OutputTokens: 400, // room for the recite ask
		Messages: []RouterReplayMessage{userText("turn-a", 40)},
	}
	m := p.do(context.Background(), req, strings.Repeat("dump-docs ", 50), 7, "s1", "inst-1", 3,
		&autoState{stream: newCompletionStream(200)}, su)
	if m.Error != nil {
		t.Fatalf("request failed: %v", m.Error)
	}

	base := filepath.Join(dir, "s003-inst-1-t007")
	reqBytes, err := os.ReadFile(base + ".request.json")
	if err != nil {
		t.Fatalf("request file: %v", err)
	}
	if !strings.Contains(string(reqBytes), marker) {
		t.Error("the request capture does not contain the marker that was sent, so it cannot " +
			"distinguish 'never injected' from 'model ignored it'")
	}

	rawBytes, err := os.ReadFile(base + ".response.raw")
	if err != nil {
		t.Fatalf("response file: %v", err)
	}
	raw := string(rawBytes)
	if !strings.Contains(raw, "data: ") || !strings.Contains(raw, "[DONE]") {
		t.Errorf("the response capture lost its SSE framing (%q); it is the parse, not the wire, "+
			"and a parser defect would be invisible in it", raw)
	}
	if !strings.Contains(raw, marker) {
		t.Error("the response capture does not contain what the server sent")
	}

	metaBytes, err := os.ReadFile(base + ".meta.json")
	if err != nil {
		t.Fatalf("meta file: %v", err)
	}
	var meta dumpMeta
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		t.Fatalf("meta is not valid json: %v", err)
	}
	if meta.Series != 3 || meta.Turn != 7 || meta.Instance != "inst-1" {
		t.Errorf("meta identifies s%d t%d %q, want s3 t7 inst-1", meta.Series, meta.Turn, meta.Instance)
	}
	if len(meta.Expected) != 1 || meta.Expected[0] != marker {
		t.Errorf("meta.Expected = %v, want the one marker asked for", meta.Expected)
	}
	if len(meta.Found) != 1 || !meta.Found[0] {
		t.Errorf("meta.Found = %v, but the response plainly contains the marker — the verdict "+
			"beside the bytes disagrees with the bytes", meta.Found)
	}
	if !meta.LeakChecked {
		t.Error("meta.LeakChecked = false; an absent leak list would then be ambiguous between " +
			"'scanned, clean' and 'never scanned'")
	}
}

// TestDumpStopsAtItsLimitOutLoud: a capture that quietly stops looks exactly
// like a run that quietly stopped making requests.
func TestDumpStopsAtItsLimitOutLoud(t *testing.T) {
	dir := t.TempDir()
	d, err := newRequestDumper(dir, 2)
	if err != nil {
		t.Fatalf("newRequestDumper: %v", err)
	}
	for i := 0; i < 5; i++ {
		d.dump(dumpMeta{Series: 1, Instance: "i", Turn: i}, []byte("req"), []byte("resp"))
	}
	got, _ := filepath.Glob(filepath.Join(dir, "*.request.json"))
	if len(got) != 2 {
		t.Errorf("wrote %d exchanges, want 2 (the limit)", len(got))
	}
}

// TestDumperOffIsFree: a nil dumper must be inert, not a panic — the capture is
// an observation, and an observation must never break what it observes.
func TestDumperOffIsFree(t *testing.T) {
	d, err := newRequestDumper("", 10)
	if err != nil || d != nil {
		t.Fatalf("newRequestDumper(\"\") = %v, %v; want nil, nil", d, err)
	}
	d.dump(dumpMeta{}, nil, nil) // must not panic
}
