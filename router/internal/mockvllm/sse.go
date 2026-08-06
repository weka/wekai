package mockvllm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/weka/wekai/kvcache"
)

// This file implements minimal OpenAI-compatible SSE streaming: a role-only
// opening delta, one content delta per synthetic output token (paced by
// OutputTokenInterval, the per-token duration Engine.Config.OutputTPS
// implies, after an initial TTFT delay standing in for prefill), a closing
// delta carrying finish_reason (and usage, iff the
// request set stream_options.include_usage — matching real vLLM, which omits
// usage on streamed responses unless asked), and the terminal "data: [DONE]"
// marker. This exact shape is what
// benchmark/replay_router_post.go:consumeOpenAISSE parses.

type chatChunk struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *usage       `json:"usage,omitempty"`
}

type completionChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   *usage             `json:"usage,omitempty"`
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", b)
	if flusher != nil {
		flusher.Flush()
	}
}

func writeSSEDone(w http.ResponseWriter, flusher http.Flusher) {
	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func startSSE(w http.ResponseWriter) http.Flusher {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	if flusher != nil {
		flusher.Flush()
	}
	return flusher
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, req chatCompletionRequest, modelID string, units []kvcache.Unit, cached, total, maxTok int, ttft time.Duration) {
	ctx := r.Context()
	flusher := startSSE(w)
	id := s.newID("chatcmpl")
	//clockexempt: cosmetic OpenAI wire-format timestamp, not a routing or timing decision
	created := time.Now().Unix()

	if !sleepCtx(ctx, ttft) {
		return
	}

	roleFinish := (*string)(nil)
	writeSSE(w, flusher, chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: modelID,
		Choices: []chatChoice{{Index: 0, Delta: &chatMsgOut{Role: "assistant"}, FinishReason: roleFinish}},
	})

	perToken := s.engine.OutputTokenInterval()
	tokens := syntheticTokens(maxTok)
	generated := 0
	for _, tok := range tokens {
		if !sleepCtx(ctx, perToken) {
			// Client gone mid-stream: stop generating, don't claim we sent it all.
			break
		}
		writeSSE(w, flusher, chatChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: modelID,
			Choices: []chatChoice{{Index: 0, Delta: &chatMsgOut{Content: tok + " "}, FinishReason: nil}},
		})
		generated++
	}

	finish := "stop"
	final := chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: modelID,
		Choices: []chatChoice{{Index: 0, Delta: &chatMsgOut{}, FinishReason: &finish}},
	}
	if req.wantsUsage() {
		u := buildUsage(total, cached, generated)
		final.Usage = &u
	}
	writeSSE(w, flusher, final)
	writeSSEDone(w, flusher)

	s.engine.RecordOutput(generated)
	// AppendOutputBlocks uses what was ACTUALLY streamed (generated may be
	// less than maxTok on an early client disconnect), matching real vLLM:
	// decode-KV is written for whatever was actually generated, not the
	// originally requested budget.
	s.engine.AppendOutputBlocks(units, generated, strings.Join(tokens[:generated], " "))
	s.coll.observe("success", cached, total, generated)
}

func (s *Server) streamCompletion(w http.ResponseWriter, r *http.Request, req completionRequest, modelID string, units []kvcache.Unit, cached, total, maxTok int, ttft time.Duration) {
	ctx := r.Context()
	flusher := startSSE(w)
	id := s.newID("cmpl")
	//clockexempt: cosmetic OpenAI wire-format timestamp, not a routing or timing decision
	created := time.Now().Unix()

	if !sleepCtx(ctx, ttft) {
		return
	}

	perToken := s.engine.OutputTokenInterval()
	tokens := syntheticTokens(maxTok)
	generated := 0
	for _, tok := range tokens {
		if !sleepCtx(ctx, perToken) {
			break
		}
		writeSSE(w, flusher, completionChunk{
			ID: id, Object: "text_completion", Created: created, Model: modelID,
			Choices: []completionChoice{{Index: 0, Text: tok + " ", FinishReason: nil}},
		})
		generated++
	}

	finish := "stop"
	final := completionChunk{
		ID: id, Object: "text_completion", Created: created, Model: modelID,
		Choices: []completionChoice{{Index: 0, Text: "", FinishReason: &finish}},
	}
	if req.wantsUsage() {
		u := buildUsage(total, cached, generated)
		final.Usage = &u
	}
	writeSSE(w, flusher, final)
	writeSSEDone(w, flusher)

	s.engine.RecordOutput(generated)
	s.engine.AppendOutputBlocks(units, generated, strings.Join(tokens[:generated], " "))
	s.coll.observe("success", cached, total, generated)
}
