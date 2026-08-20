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
	dumper, err := newRequestDumper(dumpAll, dir, 10)
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
	p.registry.Acquire(su.uuids, uuidHolder{Series: 1})
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
	d, err := newRequestDumper(dumpAll, dir, 2)
	if err != nil {
		t.Fatalf("newRequestDumper: %v", err)
	}
	for i := 0; i < 5; i++ {
		d.dump(dumpMeta{Series: 1, Instance: "i", Turn: i}, []byte("req"), []byte("resp"), []byte("merged"))
	}
	got, _ := filepath.Glob(filepath.Join(dir, "*.request.json"))
	if len(got) != 2 {
		t.Errorf("wrote %d exchanges, want 2 (the limit)", len(got))
	}
}

// TestDumperOffIsFree: a nil dumper must be inert, not a panic — the capture is
// an observation, and an observation must never break what it observes.
func TestDumperOffIsFree(t *testing.T) {
	d, err := newRequestDumper(dumpAll, "", 10)
	if err != nil || d != nil {
		t.Fatalf("newRequestDumper(dumpAll, \"\") = %v, %v; want nil, nil", d, err)
	}
	d.dump(dumpMeta{}, nil, nil, nil) // must not panic
}

// TestGarbageDumpKeepsTheCorruptExchangesAndNothingElse: the default-on
// capture is only affordable because it is selective, and only useful if the
// selection is the corrupt ones.
func TestGarbageDumpKeepsTheCorruptExchangesAndNothingElse(t *testing.T) {
	dir := t.TempDir()
	d, err := newRequestDumper(dumpGarbage, dir, 10)
	if err != nil {
		t.Fatalf("newRequestDumper: %v", err)
	}
	for i := 0; i < 5; i++ {
		d.dump(dumpMeta{Series: 1, Instance: "i", Turn: i, Garbage: i == 2}, []byte("req"), []byte("resp"), []byte("merged"))
	}
	got, _ := filepath.Glob(filepath.Join(dir, "*.request.json"))
	if len(got) != 1 {
		t.Fatalf("wrote %d exchanges, want only the one corrupt turn: %v", len(got), got)
	}
	if !strings.Contains(got[0], "t002") {
		t.Errorf("kept %q, want the corrupt turn t002", got[0])
	}
	for _, suffix := range []string{".response.raw", ".response.merged.txt", ".meta.json"} {
		if _, err := os.Stat(filepath.Join(dir, "s001-i-t002"+suffix)); err != nil {
			t.Errorf("garbage capture is missing %s: %v", suffix, err)
		}
	}
}

// TestGarbageDumpCreatesNoDirectoryUntilItHasSomething: a capture that is on by
// default must cost a clean run nothing — including an empty directory in
// /tmp, which reads as evidence of a run that produced garbage until someone
// opens it.
func TestGarbageDumpCreatesNoDirectoryUntilItHasSomething(t *testing.T) {
	d, err := newRequestDumper(dumpGarbage, "", 10)
	if err != nil {
		t.Fatalf("newRequestDumper: %v", err)
	}
	d.dump(dumpMeta{Series: 1, Instance: "i", Turn: 1}, []byte("req"), []byte("resp"), []byte("merged"))
	if dir, n, _ := d.Written(); dir != "" || n != 0 {
		t.Fatalf("a clean exchange created %q with %d written; want no directory at all", dir, n)
	}

	d.dump(dumpMeta{Series: 1, Instance: "i", Turn: 2, Garbage: true}, []byte("req"), []byte("resp"), []byte("merged"))
	dir, n, garbageOnly := d.Written()
	if dir == "" {
		t.Fatal("the first corrupt exchange did not resolve a directory")
	}
	defer os.RemoveAll(dir)
	if n != 1 || !garbageOnly {
		t.Errorf("Written() = %q, %d, %v; want one exchange in a garbage-only capture", dir, n, garbageOnly)
	}
	if _, err := os.Stat(filepath.Join(dir, "s001-i-t002.meta.json")); err != nil {
		t.Errorf("the corrupt exchange is not on disk: %v", err)
	}
}

// TestGarbageDumpLimitBoundsWhatIsWritten: the skipped exchanges must not
// consume the budget, or a long clean stretch spends it before the first
// corrupt response arrives — the capture would then be empty precisely when
// it was needed.
func TestGarbageDumpLimitBoundsWhatIsWritten(t *testing.T) {
	dir := t.TempDir()
	d, err := newRequestDumper(dumpGarbage, dir, 2)
	if err != nil {
		t.Fatalf("newRequestDumper: %v", err)
	}
	for i := 0; i < 50; i++ {
		d.dump(dumpMeta{Series: 1, Instance: "i", Turn: i}, []byte("req"), []byte("resp"), []byte("merged"))
	}
	for i := 50; i < 55; i++ {
		d.dump(dumpMeta{Series: 1, Instance: "i", Turn: i, Garbage: true}, []byte("req"), []byte("resp"), []byte("merged"))
	}
	got, _ := filepath.Glob(filepath.Join(dir, "*.request.json"))
	if len(got) != 2 {
		t.Errorf("wrote %d exchanges, want 2 (the limit, spent on garbage only): %v", len(got), got)
	}
}

// TestGarbageResponseIsCapturedWholeThroughDo drives a corrupt response through
// do() and checks that the exchange behind the verdict is on disk without
// anyone having asked for a capture.
//
// This is the point of the default: the count in the summary says two responses
// were corrupt, and every question that follows — what was in the prompt, where
// the corruption sat, what the model produced instead — is answerable only from
// the bytes, which are gone by the time the summary prints.
func TestGarbageResponseIsCapturedWholeThroughDo(t *testing.T) {
	const stamp = "garbage-stamp"
	su := buildSessionUUIDs(RouterReplaySession{Instances: []RouterReplayInstance{{
		Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("turn-a", 40)}}},
	}}}, stamp)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", "answer then ���")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	dir := t.TempDir()
	dumper, err := newRequestDumper(dumpGarbage, dir, 10)
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
	p.registry.Acquire(su.uuids, uuidHolder{Series: 1})
	p.dumper = dumper

	req := RouterReplayRequest{
		Stream: true, OutputTokens: 400,
		Messages: []RouterReplayMessage{userText("turn-a", 40)},
	}
	m := p.do(context.Background(), req, strings.Repeat("dump-docs ", 50), 4, "s1", "inst-1", 2,
		&autoState{stream: newCompletionStream(200)}, su)
	if m.Error != nil {
		t.Fatalf("request failed: %v", m.Error)
	}
	if !m.GarbageChecked || !m.ResponseGarbage {
		t.Fatalf("checked=%v garbage=%v; the response is plainly corrupt", m.GarbageChecked, m.ResponseGarbage)
	}

	base := filepath.Join(dir, "s002-inst-1-t004")
	for _, suffix := range []string{".request.json", ".response.raw", ".response.merged.txt", ".meta.json"} {
		if _, err := os.Stat(base + suffix); err != nil {
			t.Errorf("garbage capture is missing %s: %v", suffix, err)
		}
	}
	metaBytes, err := os.ReadFile(base + ".meta.json")
	if err != nil {
		t.Fatalf("meta file: %v", err)
	}
	var meta dumpMeta
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		t.Fatalf("meta is not valid json: %v", err)
	}
	if !meta.Garbage || meta.GarbageVerdict == "" {
		t.Errorf("meta records garbage=%v verdict=%q; the file must carry the verdict it was kept for",
			meta.Garbage, meta.GarbageVerdict)
	}
}
