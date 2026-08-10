// Package gateway is the inbound HTTP surface: the middleware chain, the route
// table, and the handlers.
//
// It is the only package that may import a dialect (API-1).
package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/weka/wekai/router/internal/dialect"
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/obs"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/proxy"
	"github.com/weka/wekai/router/internal/registry"
)

type Server struct {
	cfg     Config
	router  Router
	px      *proxy.Proxy
	dialect dialect.Dialect
	apiKey  []byte
	sem     chan struct{}

	handler http.Handler
}

// New builds the HTTP surface over a Router, which decides which pool serves a
// given model.
func New(cfg Config, rt Router, px *proxy.Proxy, d dialect.Dialect) *Server {
	s := &Server{cfg: cfg, router: rt, px: px, dialect: d}
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
		if route.Class == "models" {
			// Model listing is answered by the router, not proxied to one pool.
			// A router fronting several pools serves several models, and
			// forwarding the question to whichever pool matched the empty model
			// name would advertise a fraction of what it can actually serve —
			// so a client discovering models through the router would never see
			// the others exist.
			mux.Handle(route.Pattern, s.handleMergedModels())
			continue
		}
		mux.Handle(route.Pattern, s.inferenceHandler(route, true))
	}

	// Passthrough. Anything the dialect does not claim is still proxied to the
	// pool the request's model resolves to — by load, with no prefix affinity,
	// because there is nothing to be affine to: units are dialect knowledge and
	// this path has none.
	//
	// This is what lets one router front both a local vLLM fleet on
	// /v1/chat/completions AND a hosted API on its own paths — Anthropic's
	// /v1/messages, say — which is how captured traffic was collected in the
	// first place. Dropping it would have made the merge a regression.
	mux.Handle("/", s.inferenceHandler(dialect.Route{Pattern: "/", Class: "passthrough"}, false))

	// Operational endpoints.
	mux.HandleFunc("GET /liveness", s.handleLiveness)
	mux.HandleFunc("GET /readiness", s.handleReadiness)
	mux.HandleFunc("GET /health", s.handleReadiness)
	// Kubernetes-convention aliases. They exist because of the passthrough tier
	// below: an unclaimed path is PROXIED, so a probe pointed at /healthz would
	// be forwarded upstream and answered by a backend that knows nothing about
	// it — a 404 that reads as "the router is unhealthy" and CrashLoopBackOffs
	// the pod. A probe must never leave the router.
	mux.HandleFunc("GET /healthz", s.handleReadiness)
	mux.HandleFunc("GET /livez", s.handleLiveness)
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
func (s *Server) inferenceHandler(route dialect.Route, affine bool) http.Handler {
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
		if affine {
			if units, ok := s.dialect.ExtractUnits(body, route.Class, nil); ok {
				rr.Units = units
			}
		}

		target, ok := s.router.Route(intro.Model)
		if !ok {
			// Nothing serves this model. A 404 naming it, not a 503: no amount
			// of retrying will make a route appear, and telling the caller
			// which name failed is the difference between a typo found in
			// seconds and one found in a support thread.
			s.dialect.WriteError(w, http.StatusNotFound, "no_route_for_model",
				"no route matches model "+strconv.Quote(intro.Model))
			return
		}
		if target.RewriteModel != "" {
			body = rewriteModelField(body, target.RewriteModel)
			rr.Model = target.RewriteModel
		}
		auth := proxy.Auth{Credential: target.Credential,
			Forward: target.ForwardClientCredential, Strip: target.StripAuth}
		obs.SetTarget(ctx, target.Name, target.RewriteModel)

		candidates := s.candidates(target)
		if len(candidates) == 0 {
			// Never route to a known-bad backend to avoid an error (HLT-11).
			// Capacity is no longer judged here — the flow's signals do that and
			// answer 429 themselves — so an empty candidate set now means one
			// thing only: nothing is healthy.
			s.dialect.WriteError(w, http.StatusServiceUnavailable,
				"no_healthy_backends", "no healthy backend is available")
			return
		}

		var accepted proxy.OnAccepted
		if c, ok := target.Selector.(policy.Committer); ok {
			accepted = func(b *registry.Backend) { c.Commit(b, rr) }
		}
		var outcome proxy.OnOutcome
		if o, ok := target.Selector.(policy.Observer); ok {
			outcome = func(b *registry.Backend, status int) {
				// A 429 is the ultimate signal; anything the backend actually
				// served proves it is taking work again.
				if status == http.StatusTooManyRequests {
					o.OnRefused(b)
					return
				}
				if status >= 200 && status < 500 {
					o.OnAccepted(b)
				}
			}
		}
		res := s.px.Serve(w, r, candidates, target.Selector, s.dialect, rr, body, accepted, outcome, auth)
		if res.Backend != nil {
			obs.SetBackend(ctx, res.Backend.URL)
		}
		switch {
		case res.Committed || res.Err == nil:
		case errors.Is(res.Err, policy.ErrSplitGuardBlocked):
			// Declined on purpose: every backend holding this prefix is
			// saturated and none of the rest is far enough below to be worth a
			// duplicate copy. Idle capacity may well exist — that is what makes
			// this distinct from all_backends_saturated below.
			//
			// The WHY goes to the log, not to the caller. A client can do
			// exactly one thing with a 429 — back off and retry — and the
			// router's prefix bookkeeping is not the caller's business to
			// reason about, nor something to expose on a public endpoint.
			slog.Warn("rejected: every backend holding this prefix is saturated and none "+
				"of the rest is far enough below it to take a copy. Idle capacity may "+
				"exist; this is the split guard declining to duplicate the prefix.",
				"request_id", obs.RequestID(ctx), "pool", target.Name,
				"reason", "split_guard_blocked")
			writeBusy(w, s.dialect)
		case errors.Is(res.Err, policy.ErrAllBackendsSaturated):
			// Every healthy backend is saturated by some signal. Replaces the
			// old gateway-side all_backends_at_capacity, which reached the same
			// conclusion from a router-side concurrency filter before any
			// routing ran.
			metrics.SaturationRejects.Inc()
			slog.Warn("rejected: every healthy backend in the pool is saturated",
				"request_id", obs.RequestID(ctx), "pool", target.Name,
				"reason", "all_backends_saturated")
			writeBusy(w, s.dialect)
		default:
			s.dialect.WriteError(w, http.StatusBadGateway,
				"upstream_unavailable", "all upstream attempts failed")
		}
	})
}

// candidates filters the snapshot to backends eligible for new traffic:
// healthy, non-draining, closed-circuit, matching dialect. HEALTH ONLY.
//
// Capacity is deliberately not judged here any more. It used to be — a
// --max-node-concurrency filter sat in front of every policy — and that split
// the answer to "is this backend full" across two components: a router-side
// guess here, and a policy downstream that re-derived the same thing from
// in-flight counts and could not tell "saturated" from "does not exist". The
// judgement now lives in exactly one place, the flow's signals, where the
// backend's own 429 is available as ground truth and a configured limit is just
// one opt-in opinion alongside it.
//
// Filtering reads circuit State() only and never Allow(), so it cannot consume a
// half-open probe token for a backend it does not select (R2, LB-9).
func (s *Server) candidates(t Target) []*registry.Backend {
	snap := t.Registry.Snapshot()
	cands := make([]*registry.Backend, 0, len(snap.Backends))
	for _, b := range snap.Backends {
		if !b.Available() || b.DialectID != s.dialect.ID() {
			continue
		}
		cands = append(cands, b)
	}
	return cands
}

// poolByName resolves an admin request's ?pool=. Empty means the first target,
// which is the only one on a single-pool router — the common case, and the one
// where making an operator name it would be pure ceremony.
// handleMergedModels answers /v1/models by asking one live backend in EVERY
// pool and merging the results.
//
// One backend per pool, not all of them: endpoints within a pool are
// interchangeable and serve the same models, so asking more is redundant load.
// A pool with nothing healthy contributes nothing rather than failing the whole
// listing — a partial answer is more useful than none, and the unhealthy pool
// is visible in /readiness and the metrics.
func (s *Server) handleMergedModels() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		type model struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by,omitempty"`
			Pool    string `json:"pool,omitempty"`
		}
		var out []model
		seen := map[string]bool{}

		for _, t := range s.router.Targets() {
			cands := s.candidates(t)
			if len(cands) == 0 {
				continue
			}
			body, err := s.fetchModels(r.Context(), cands[0].URL, r, t)
			if err != nil {
				continue
			}
			for _, m := range body.Data {
				// A model served by two pools is listed once. Which pool wins
				// is rule order, the same thing that decides where a request
				// for it goes, so the label cannot disagree with the routing.
				if seen[m.ID] {
					continue
				}
				seen[m.ID] = true
				out = append(out, model{ID: m.ID, Object: orString(m.Object, "model"),
					OwnedBy: m.OwnedBy, Pool: t.Name})
			}
		}
		if out == nil {
			out = []model{}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": out})
	})
}

type modelsResponse struct {
	Data []struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	} `json:"data"`
}

// fetchModels asks one endpoint what it serves.
//
// Attempt-then-fallback on the path, because hosted providers do not agree:
// OpenAI and Anthropic answer <base>/v1/models, while Google's
// OpenAI-compatible surface is <base>/models with the /v1beta/openai prefix
// already in the base. Trying both is cheaper than making an operator encode
// which flavour each endpoint is.
//
// Credentials are forwarded from the CALLER's request rather than configured
// here. A hosted provider returns 401 to an unauthenticated listing, so without
// this a hosted pool silently contributes nothing to the merged result — which
// looks identical to it having no models.
func (s *Server) fetchModels(ctx context.Context, base string, from *http.Request, t Target) (*modelsResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	var lastErr error
	for _, path := range []string{"/v1/models", "/models"} {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, registry.ResolveURL(base, path), nil)
		if err != nil {
			return nil, err
		}
		if !t.StripAuth && from != nil {
			for _, h := range []string{"Authorization", "X-Api-Key", "Anthropic-Version"} {
				if v := from.Header.Get(h); v != "" {
					req.Header.Set(h, v)
				}
			}
			// Anthropic rejects a request without it, and no other provider
			// minds an extra header.
			if req.Header.Get("Anthropic-Version") == "" && req.Header.Get("X-Api-Key") != "" {
				req.Header.Set("Anthropic-Version", "2023-06-01")
			}
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			lastErr = fmt.Errorf("models %s%s: status %d", base, path, resp.StatusCode)
			continue
		}
		var out modelsResponse
		err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		return &out, nil
	}
	return nil, lastErr
}

func orString(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

// poolNames lists the configured pools for /get_server_info. It replaces the
// old "policy" field: there is one routing flow now, so what an operator
// actually needs to see is which pools exist.
func (s *Server) poolNames() []string {
	targets := s.router.Targets()
	out := make([]string, 0, len(targets))
	for _, t := range targets {
		out = append(out, t.Name)
	}
	return out
}

func (s *Server) poolByName(name string) (Target, bool) {
	targets := s.router.Targets()
	if len(targets) == 0 {
		return Target{}, false
	}
	if name == "" {
		return targets[0], true
	}
	for _, t := range targets {
		if t.Name == name {
			return t, true
		}
	}
	return Target{}, false
}

// stripInboundCredentials removes client credentials before forwarding, for a
// route marked as pointing at an unauthenticated upstream. Forwarding them
// there leaks a caller's key into someone else's logs.
func stripInboundCredentials(r *http.Request) {
	for _, h := range []string{"Authorization", "X-Api-Key", "Anthropic-Beta"} {
		r.Header.Del(h)
	}
}

// applyRouterCredential replaces the caller's credentials with the router's own
// for this pool.
//
// The caller's are REPLACED, not merged: forwarding both would send a user's
// key to an internal service that has no business seeing it, which is the leak
// this feature exists to prevent in the other direction.
//
// Both header styles are set because the two conventions coexist —
// Authorization: Bearer for OpenAI-compatible servers, x-api-key for
// Anthropic — and a pool is configured by URL, not by which flavour it speaks.
// Sending the key twice to a server that reads one is harmless; guessing wrong
// is a 401 an operator has to debug.
func applyRouterCredential(r *http.Request, cred string) {
	stripInboundCredentials(r)
	r.Header.Set("Authorization", "Bearer "+cred)
	r.Header.Set("X-Api-Key", cred)
	if r.Header.Get("Anthropic-Version") == "" {
		r.Header.Set("Anthropic-Version", "2023-06-01")
	}
}

// rewriteModelField replaces the JSON body's "model" so a client's name for a
// model can differ from the backend's. Returns the body unchanged on any parse
// failure: a route rewrite is not worth failing a request the upstream might
// well have accepted.
func rewriteModelField(body []byte, model string) []byte {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(body, &doc); err != nil {
		return body
	}
	enc, err := json.Marshal(model)
	if err != nil {
		return body
	}
	doc["model"] = enc
	out, err := json.Marshal(doc)
	if err != nil {
		return body
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
	// Readiness reflects backend HEALTH, and candidates() is now health-only, so
	// this is simply its size. A fully saturated router is still ready to
	// receive traffic — it sheds with 429, not by going NotReady.
	n := 0
	for _, t := range s.router.Targets() {
		n += len(s.candidates(t))
	}
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
		"pools":    s.poolNames(),
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
	Pool     string  `json:"pool"`
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
	var out []workerView
	version := uint64(0)
	for _, t := range s.router.Targets() {
		snap := t.Registry.Snapshot()
		version += snap.Version
		for _, b := range snap.Backends {
			out = append(out, workerView{
				Pool: t.Name,
				URL:  b.URL, Kind: b.Kind().String(), Dialect: b.DialectID,
				Health: b.Health().String(), Circuit: b.CB.State().String(),
				Inflight: b.Inflight(), Capacity: b.Capacity(),
				Load: b.NormalizedLoad(), Draining: b.Draining(),
				Prov:   b.Prov.String(),
				Served: b.Served.Load(), Failed: b.Failed.Load(),
			})
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"workers": out, "version": version})
}

func (s *Server) handleGetLoads(w http.ResponseWriter, r *http.Request) {
	loads := map[string]int64{}
	for _, t := range s.router.Targets() {
		for _, b := range t.Registry.Snapshot().Backends {
			loads[b.URL] = b.Inflight()
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"loads": loads})
}

func (s *Server) handleAddWorker(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Query().Get("url")
	kind := ""
	cap := int64(0)
	if url == "" {
		var req struct {
			URL      string `json:"url"`
			Kind     string `json:"kind"`
			Capacity int64  `json:"capacity"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err == nil {
			url, kind, cap = req.URL, req.Kind, req.Capacity
		}
	}
	target, ok := s.poolByName(r.URL.Query().Get("pool"))
	if !ok {
		s.dialect.WriteError(w, http.StatusNotFound, "unknown_pool",
			"no pool named "+strconv.Quote(r.URL.Query().Get("pool")))
		return
	}
	spec := registry.Spec{URL: url, Prov: registry.ProvStatic, Capacity: s.cfg.DefaultCapacity}
	if cap != 0 {
		spec.Capacity = cap
	}
	if kind == "router" {
		spec.Kind = registry.KindRouter
	}
	// Canonical() rejects non-http(s) schemes, which is the SSRF guard on this
	// endpoint (SEC-5).
	b, err := target.Registry.Add(spec)
	if err != nil {
		s.dialect.WriteError(w, http.StatusBadRequest, "invalid_worker", err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"pool": target.Name, "url": b.URL, "health": b.Health().String()})
}

func (s *Server) handleRemoveWorker(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Query().Get("url")
	// Removal searches every pool: an operator draining a node knows its URL,
	// not which pool it was filed under.
	removed := false
	for _, t := range s.router.Targets() {
		if err := t.Registry.Remove(url); err == nil {
			removed = true
		}
	}
	if !removed {
		s.dialect.WriteError(w, http.StatusNotFound, "unknown_worker",
			"no pool holds "+strconv.Quote(url))
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

// writeBusy answers a capacity rejection.
//
// One shape for every reason the router ran out of room, because a caller can
// act on exactly one thing — back off and retry — and the reason is internal
// bookkeeping. The specific cause is logged and counted; it is not published on
// what may well be a public endpoint.
func writeBusy(w http.ResponseWriter, d dialect.Dialect) {
	w.Header().Set("Retry-After", "1")
	d.WriteError(w, http.StatusTooManyRequests,
		"rate_limit_error", "server is busy; retry shortly")
}
