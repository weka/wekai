package mockvllm

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
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

// admitOrReject reserves a concurrency slot or writes a 429 and records the
// rejection. Callers must invoke the returned release exactly once when ok.
func (s *Server) admitOrReject(w http.ResponseWriter) (release func(), ok bool) {
	release, ok = s.engine.Admit()
	if !ok {
		writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded",
			"server is at max concurrency; retry")
		s.coll.observe("rejected", 0, 0, 0)
	}
	return release, ok
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	release, ok := s.admitOrReject(w)
	if !ok {
		return
	}
	defer release()

	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}

	cached, total := s.engine.Serve(s.engine.Tokenize(string(req.promptBytes())))
	maxTok := s.engine.MaxTokensOrDefault(req.MaxTokens)
	ttft, totalDur := s.engine.Latency(cached, total, maxTok)
	modelID := s.engine.Config().ModelID

	if req.Stream {
		s.streamChat(w, r, req, modelID, cached, total, maxTok, ttft)
		return
	}

	if !sleepCtx(r.Context(), totalDur) {
		return // client disconnected mid-"inference"
	}
	s.engine.RecordOutput(maxTok)

	content := strings.Join(syntheticTokens(maxTok), " ")
	finish := "stop"
	u := buildUsage(total, cached, maxTok)
	writeJSON(w, http.StatusOK, chatCompletionResponse{
		ID:      s.newID("chatcmpl"),
		Object:  "chat.completion",
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
	release, ok := s.admitOrReject(w)
	if !ok {
		return
	}
	defer release()

	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}

	cached, total := s.engine.Serve(s.engine.Tokenize(req.promptText()))
	maxTok := s.engine.MaxTokensOrDefault(req.MaxTokens)
	ttft, totalDur := s.engine.Latency(cached, total, maxTok)
	modelID := s.engine.Config().ModelID

	if req.Stream {
		s.streamCompletion(w, r, req, modelID, cached, total, maxTok, ttft)
		return
	}

	if !sleepCtx(r.Context(), totalDur) {
		return
	}
	s.engine.RecordOutput(maxTok)

	text := strings.Join(syntheticTokens(maxTok), " ")
	finish := "stop"
	u := buildUsage(total, cached, maxTok)
	writeJSON(w, http.StatusOK, completionResponse{
		ID:      s.newID("cmpl"),
		Object:  "text_completion",
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
