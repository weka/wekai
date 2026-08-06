package mockvllm

import (
	"net/http"
	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server wires an Engine to an OpenAI-compatible HTTP surface: /health,
// /v1/models, /v1/chat/completions, /v1/completions, and /metrics.
type Server struct {
	engine *Engine
	coll   *collectors
	reg    *prometheus.Registry
	nextID atomic.Int64
}

func NewServer(engine *Engine) *Server {
	coll := newCollectors(engine)
	reg := prometheus.NewRegistry()
	coll.register(reg)
	return &Server{engine: engine, coll: coll, reg: reg}
}

func (s *Server) Engine() *Engine { return s.engine }

// Handler builds the request router. Kept separate from ListenAndServe so
// tests can exercise it via httptest without opening a real socket.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /models", s.handleModels) // pre-/v1 discovery probe, see replay_router_post.go:discoverModelName
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	mux.Handle("GET /metrics", promhttp.HandlerFor(s.reg, promhttp.HandlerOpts{Registry: s.reg}))
	return mux
}
