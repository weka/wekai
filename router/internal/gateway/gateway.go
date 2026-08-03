// Package gateway is the inbound HTTP surface: the middleware chain, the route
// table, and the handlers.
//
// It is the only package that may import a dialect (API-1).
package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/weka/wekai/router/internal/config"
	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/obs"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

type Server struct {
	cfg     config.Config
	reg     *registry.Registry
	pol     proxy.Selector
	px      *proxy.Proxy
	dialect dialect.Dialect
	apiKey  []byte
	sem     chan struct{}

	handler http.Handler
}

func New(cfg config.Config, reg *registry.Registry, pol proxy.Selector, px *proxy.Proxy, d dialect.Dialect) *Server {
	s := &Server{cfg: cfg, reg: reg, pol: pol, px: px, dialect: d}
	if cfg.APIKey != "" {
		s.apiKey = []byte(cfg.APIKey)
	}
	if cfg.MaxConcurrentRequests > 0 {
		s.sem = make(chan struct{}, cfg.MaxConcurrentRequests)
	}
	s.handler = s.build()
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.handler.ServeHTTP(w, r) }

func (s *Server) build() http.Handler {
	mux := http.NewServeMux()

	// Dialect-claimed inference routes. The matched pattern carries the route
	// class and the dialect — never sniffed from the body (API-5, API-N1).
	for _, rt := range s.dialect.Routes() {
		route := rt
		mux.Handle(route.Pattern, s.inferenceHandler(route))
	}

	// Operational endpoints.
	mux.HandleFunc("GET /liveness", s.handleLiveness)
	mux.HandleFunc("GET /readiness", s.handleReadiness)
	mux.HandleFunc("GET /health", s.handleReadiness)
	mux.HandleFunc("GET /get_server_info", s.handleServerInfo)

	// Admin endpoints. Auth applies (AUTH-11).
	mux.HandleFunc("GET /workers", s.handleListWorkers)
	mux.HandleFunc("GET /list_workers", s.handleListWorkers)
	mux.HandleFunc("POST /workers", s.handleAddWorker)
	mux.HandleFunc("POST /add_worker", s.handleAddWorker)
	mux.HandleFunc("POST /remove_worker", s.handleRemoveWorker)
	mux.HandleFunc("GET /get_loads", s.handleGetLoads)

	// The middleware chain. Order is documented in middleware.go and is the
	// substance of GW-8/GW-9/GW-10/AUTH-4.
	return chain(mux,
		s.recoverMiddleware,
		requestIDMiddleware,
		accessLogMiddleware,
		s.corsMiddleware,
		s.bodyLimitMiddleware,
		s.authMiddleware,
		s.concurrencyMiddleware,
	)
}

// inferenceHandler proxies one request class to a selected backend.
func (s *Server) inferenceHandler(route dialect.Route) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		obs.SetRoute(ctx, route.Class, s.dialect.ID())

		// Buffer the body so a retry can replay it (REL-4). MaxBytesReader is
		// already armed, so this is bounded by max_body_bytes.
		var body []byte
		if r.Body != nil && r.Method != http.MethodGet {
			b, err := io.ReadAll(r.Body)
			if err != nil {
				// MaxBytesReader signals oversize here.
				s.dialect.WriteError(w, http.StatusRequestEntityTooLarge,
					"payload_too_large", "request body exceeds the configured limit")
				return
			}
			body = b
		}

		intro := s.dialect.Introspect(body)
		rr := &policy.RoutingRequest{
			RequestID:  obs.RequestID(ctx),
			RouteClass: route.Class,
			DialectID:  s.dialect.ID(),
			Model:      intro.Model,
			Stream:     intro.Stream,
		}
		if units, ok := s.dialect.ExtractUnits(body, route.Class, nil); ok {
			rr.Units = units
		}

		candidates := s.candidates()
		if len(candidates) == 0 {
			// Never route to a known-bad backend to avoid an error (HLT-11).
			s.dialect.WriteError(w, http.StatusServiceUnavailable,
				"no_healthy_backends", "no healthy backend is available")
			return
		}

		var accepted proxy.OnAccepted
		if c, ok := s.pol.(policy.Committer); ok {
			accepted = func(b *registry.Backend) { c.Commit(b, rr) }
		}
		res := s.px.Serve(w, r, candidates, s.pol, s.dialect, rr, body, accepted)
		if res.Err != nil && !res.Committed {
			s.dialect.WriteError(w, http.StatusBadGateway,
				"upstream_unavailable", "all upstream attempts failed")
		}
	})
}

// candidates filters the snapshot to backends eligible for new traffic.
//
// Filtering reads circuit State() only and never Allow(), so it cannot consume a
// half-open probe token for a backend it does not select (R2, LB-9).
func (s *Server) candidates() []*registry.Backend {
	snap := s.reg.Snapshot()
	out := make([]*registry.Backend, 0, len(snap.Backends))
	for _, b := range snap.Backends {
		if b.Available() && b.DialectID == s.dialect.ID() {
			out = append(out, b)
		}
	}
	return out
}

func (s *Server) handleLiveness(w http.ResponseWriter, r *http.Request) {
	// Process-up only, and the one endpoint exempt from auth (AUTH-5).
	w.Header().Set("Content-Type", "text/plain")
	_, _ = io.WriteString(w, "ok\n")
}

// handleReadiness fails when there is no healthy backend, so an orchestrator
// drains this instance through the ordinary path (HIER-7).
func (s *Server) handleReadiness(w http.ResponseWriter, r *http.Request) {
	n := len(s.candidates())
	body := map[string]any{"ready": n > 0}
	// The fleet size is operational detail, so it is disclosed only to an
	// authenticated caller. An unauthenticated kubelet probe gets the boolean it
	// needs and nothing more.
	if Authed(r.Context()) {
		body["healthy_backends"] = n
	}
	w.Header().Set("Content-Type", "application/json")
	if n == 0 {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	_ = json.NewEncoder(w).Encode(body)
}

// handleServerInfo reports the effective configuration with secrets redacted
// (CFG-7, SEC-6).
func (s *Server) handleServerInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"config":   s.cfg.Redacted(),
		"policy":   s.pol.Name(),
		"dialects": []string{s.dialect.ID()},
		// FR-RTR-01 is served by prediction, not by observed residency. Saying so
		// here is RES-4: nobody should mistake one for the other.
		"residency_source": "predicted",
		"residency_note": "KV-cache residency is predicted, not observed. Exact residency " +
			"requires vLLM --kv-events-config (ZMQ BlockStored/BlockRemoved) or an " +
			"LMCache cache-controller /lookup; neither is enabled in this deployment.",
	})
}

type workerView struct {
	URL      string  `json:"url"`
	Kind     string  `json:"kind"`
	Dialect  string  `json:"dialect"`
	Health   string  `json:"health"`
	Circuit  string  `json:"circuit"`
	Inflight int64   `json:"inflight"`
	Capacity int64   `json:"capacity"`
	Load     float64 `json:"normalized_load"`
	Draining bool    `json:"draining"`
	Prov     string  `json:"provenance"`
	Served   uint64  `json:"served"`
	Failed   uint64  `json:"failed"`
}

// handleListWorkers returns backends in deterministic snapshot order (WRK-8).
func (s *Server) handleListWorkers(w http.ResponseWriter, r *http.Request) {
	snap := s.reg.Snapshot()
	out := make([]workerView, 0, len(snap.Backends))
	for _, b := range snap.Backends {
		out = append(out, workerView{
			URL: b.URL, Kind: b.Kind.String(), Dialect: b.DialectID,
			Health: b.Health().String(), Circuit: b.CB.State().String(),
			Inflight: b.Inflight(), Capacity: b.Capacity(),
			Load: b.NormalizedLoad(), Draining: b.Draining(),
			Prov:   b.Prov.String(),
			Served: b.Served.Load(), Failed: b.Failed.Load(),
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"workers": out, "version": snap.Version})
}

func (s *Server) handleGetLoads(w http.ResponseWriter, r *http.Request) {
	snap := s.reg.Snapshot()
	loads := make(map[string]int64, len(snap.Backends))
	for _, b := range snap.Backends {
		loads[b.URL] = b.Inflight()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"loads": loads})
}

func (s *Server) handleAddWorker(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Query().Get("url")
	if url == "" {
		var req struct {
			URL      string `json:"url"`
			Kind     string `json:"kind"`
			Capacity int64  `json:"capacity"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err == nil {
			url = req.URL
		}
	}
	spec := registry.Spec{URL: url, Prov: registry.ProvStatic, Capacity: s.cfg.MaxInflightPerBackend}
	// Canonical() rejects non-http(s) schemes, which is the SSRF guard on this
	// endpoint (SEC-5).
	b, err := s.reg.Add(spec)
	if err != nil {
		s.dialect.WriteError(w, http.StatusBadRequest, "invalid_worker", err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(map[string]any{"url": b.URL, "health": b.Health().String()})
}

func (s *Server) handleRemoveWorker(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Query().Get("url")
	if err := s.reg.Remove(url); err != nil {
		s.dialect.WriteError(w, http.StatusNotFound, "unknown_worker", err.Error())
		return
	}
	// Removal is a graceful drain, not a hard delete (WRK-6).
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"draining": url})
}

// Timeouts for the inference listener.
func (s *Server) HTTPServer(addr string) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           s,
		ReadHeaderTimeout: 15 * time.Second,
		// No WriteTimeout: it would cap long streaming generations.
		IdleTimeout: 120 * time.Second,
	}
}
