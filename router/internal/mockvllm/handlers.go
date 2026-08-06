package mockvllm

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/weka/wekai/kvcache"
)

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	cfg := s.engine.Config()
	writeJSON(w, http.StatusOK, modelsResponse{
		Object: "list",
		Data:   []modelInfo{{ID: cfg.ModelID, Object: "model", OwnedBy: "mock-vllm"}},
	})
}

// admitOrReject tokenizes the prompt, then makes the single admission
// decision: reserve a concurrency slot AND pin those blocks together (see
// Engine.Admit). On rejection it writes the 429 and records it; nothing was
// touched in the cache model for a rejected request, matching a real 429
// where the backend did no prefill and cached nothing. Callers must invoke
// the returned release exactly once when ok.
func (s *Server) admitOrReject(w http.ResponseWriter, units []kvcache.Unit) (release func(), cached, total int, ok bool) {
	release, cached, total, ok = s.engine.Admit(units)
	if !ok {
		writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded",
			"server is at max concurrency; retry")
		s.coll.observe("rejected", 0, 0, 0)
	}
	return release, cached, total, ok
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}

	units := s.engine.Tokenize(string(req.promptBytes()))
	release, cached, total, ok := s.admitOrReject(w, units)
	if !ok {
		return
	}
	defer release()

	maxTok := s.engine.MaxTokensOrDefault(req.MaxTokens)
	work := s.engine.PrefillWork(cached, total)
	modelID := s.engine.Config().ModelID

	if req.Stream {
		s.streamChat(w, r, req, modelID, units, cached, total, maxTok, work)
		return
	}

	if !s.engine.AwaitTTFT(r.Context(), work) {
		return // client disconnected mid-prefill
	}
	if !sleepCtx(r.Context(), s.engine.DecodeDuration(maxTok)) {
		return // client disconnected mid-decode
	}
	s.engine.RecordOutput(maxTok)

	content := strings.Join(syntheticTokens(maxTok), " ")
	s.engine.AppendOutputBlocks(units, maxTok, content)
	finish := "stop"
	u := buildUsage(total, cached, maxTok)
	writeJSON(w, http.StatusOK, chatCompletionResponse{
		ID:     s.newID("chatcmpl"),
		Object: "chat.completion",
		//clockexempt: cosmetic OpenAI wire-format timestamp, not a routing or timing decision
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []chatChoice{{
			Index:        0,
			Message:      &chatMsgOut{Role: "assistant", Content: content},
			FinishReason: &finish,
		}},
		Usage: &u,
	})

	s.coll.observe("success", cached, total, maxTok)
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}

	units := s.engine.Tokenize(req.promptText())
	release, cached, total, ok := s.admitOrReject(w, units)
	if !ok {
		return
	}
	defer release()

	maxTok := s.engine.MaxTokensOrDefault(req.MaxTokens)
	work := s.engine.PrefillWork(cached, total)
	modelID := s.engine.Config().ModelID

	if req.Stream {
		s.streamCompletion(w, r, req, modelID, units, cached, total, maxTok, work)
		return
	}

	if !s.engine.AwaitTTFT(r.Context(), work) {
		return
	}
	if !sleepCtx(r.Context(), s.engine.DecodeDuration(maxTok)) {
		return
	}
	s.engine.RecordOutput(maxTok)

	text := strings.Join(syntheticTokens(maxTok), " ")
	s.engine.AppendOutputBlocks(units, maxTok, text)
	finish := "stop"
	u := buildUsage(total, cached, maxTok)
	writeJSON(w, http.StatusOK, completionResponse{
		ID:     s.newID("cmpl"),
		Object: "text_completion",
		//clockexempt: cosmetic OpenAI wire-format timestamp, not a routing or timing decision
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []completionChoice{{Index: 0, Text: text, FinishReason: &finish}},
		Usage:   &u,
	})

	s.coll.observe("success", cached, total, maxTok)
}

func (s *Server) newID(prefix string) string {
	return prefix + "-mock-" + strconv.FormatInt(s.nextID.Add(1), 10)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    typ,
			"code":    status,
		},
	})
}
