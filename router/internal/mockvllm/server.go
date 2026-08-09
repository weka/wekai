package mockvllm

import (
	"net/http"
	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Surface selects which HTTP endpoints the mock exposes, so a test can build
// every deployment shape the router must discover without a real backend.
//
// The router decides two independent things about an endpoint: whether it is a
// vLLM instance (from vllm: metric names at /metrics) and whether active health
// probing can work. Those are independent of the request format, because a vLLM
// instance can be fronted with an Anthropic-shaped API and still serve its own
// metrics. Testing that required a mock that can be a vLLM WITHOUT the OpenAI
// surface, and a hosted API with neither.
type Surface int

const (
	// SurfaceVLLM is a real vLLM: OpenAI routes, /health, and vllm: metrics.
	SurfaceVLLM Surface = iota
	// SurfaceAnthropic is a vLLM fronted with an Anthropic-shaped API: it
	// serves /v1/messages and still exposes vllm: metrics, which is exactly the
	// case that must not be misclassified as a hosted API.
	SurfaceAnthropic
	// SurfaceHosted is a hosted API: messages only. No /metrics, no /v1/models,
	// no /health — so discovery must fall back to passive health.
	SurfaceHosted
)

// Server wires an Engine to an HTTP surface.
type Server struct {
	// Surface selects which endpoints are exposed; zero value is a real vLLM.
	Surface Surface

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
	if s.Surface != SurfaceHosted {
		// A hosted API exposes none of these; that is what makes it
		// indistinguishable from an outage to an active health probe, and why
		// the router must fall back to passive for it.
		mux.HandleFunc("GET /health", s.handleHealth)
		mux.HandleFunc("GET /v1/models", s.handleModels)
		mux.HandleFunc("GET /models", s.handleModels) // pre-/v1 discovery probe
		mux.Handle("GET /metrics", promhttp.HandlerFor(s.reg, promhttp.HandlerOpts{Registry: s.reg}))
	}
	if s.Surface != SurfaceAnthropic {
		mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
		mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	}
	if s.Surface != SurfaceVLLM {
		// Anthropic's messages endpoint. The engine is the same; only the wire
		// shape differs, which is the point — the router must route and measure
		// it identically.
		mux.HandleFunc("POST /v1/messages", s.handleMessages)
	}
	return mux
}
