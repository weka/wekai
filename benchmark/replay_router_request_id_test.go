package benchmark

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The X-Request-Id header is how a record (e.g. a cross-contamination hit) is
// matched to the server's own log lines; vLLM adopts it as its request id.
func TestSendOnceSetsXRequestID(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.Header.Get("X-Request-Id"))
	}))
	defer srv.Close()
	p := &replayPoster{apiType: "openai_vllm", client: srv.Client()}
	for _, id := range []string{"wk-s7-t3-0000abcd", ""} {
		resp, err := p.sendOnce(context.Background(), srv.URL, []byte("{}"), false, id)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	if len(got) != 2 || got[0] != "wk-s7-t3-0000abcd" || got[1] != "" {
		t.Fatalf("X-Request-Id seen by server = %q, want [wk-s7-t3-0000abcd, \"\"]", got)
	}
}
