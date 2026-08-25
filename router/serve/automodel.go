package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/router/internal/registry"
)

// Auto-model discovery: a route with no explicit `as <model>` asks its own pool
// what it serves and rewrites the request's model to that answer.
//
// Single-model servers (vLLM, llama.cpp, SGLang) reject any name they do not
// serve — vLLM answers a request for the wrong model with a bare 404. Clients
// send the name they know; the operator should not have to look a checkpoint
// name up by hand and repeat it.
//
// Two things about WHERE this runs matter, and both were wrong before.
//
// It runs per POOL, not per configured endpoint. A pool's backends are
// interchangeable by definition, so the question is asked once of the pool and
// answered for all of it — including backends that arrive later from discovery,
// which a startup-time probe of a static list never saw.
//
// It runs AFTER health, not before. Probing the first endpoint in a list was
// wrong twice over: a pool whose first backend is dead never resolved even when
// every other backend could have answered, and at startup no backend has been
// probed at all, so the very first attempt was guaranteed to race the health
// checker. Discovery now waits for a backend the router would actually route
// to, then asks that one.
const (
	autoModelAuto  = "auto"  // rewrite only when the pool serves exactly one model
	autoModelOff   = "off"   // never probe; the client's model is forwarded untouched
	autoModelForce = "force" // rewrite to the first listed model, however many there are
)

const (
	// autoModelProbeTimeout bounds a single listing request. A backend that has
	// passed a health check answers in milliseconds.
	autoModelProbeTimeout = 3 * time.Second
	// autoModelPoll is how often the resolver re-checks for a healthy backend
	// while it has none. Matched to the unhealthy health-check interval: there
	// is no point looking more often than the health checker updates.
	autoModelPoll = 1 * time.Second
	// autoModelAttempts bounds retries against a HEALTHY pool. A backend that
	// passes health checks but cannot answer /v1/models is not going to start;
	// it is a hosted API without a listing, or an engine that does not serve
	// one. Retrying forever produced a goroutine and a log line every 15s for
	// the life of the process, which is noise, not resilience.
	//
	// Waiting for the pool to become healthy is NOT bounded by this — that wait
	// is the normal startup case and ends when weights finish loading.
	autoModelAttempts = 5
)

// resolveAutoModel fills slot with the model a pool serves, once the pool has a
// backend healthy enough to ask. It returns when it has an answer, when the
// answer is settled as "do not rewrite", or when ctx ends.
func resolveAutoModel(ctx context.Context, mode string, name string, reg *registry.Registry,
	cred string, slot *atomic.Pointer[string], log *slog.Logger) {
	if mode == autoModelOff {
		return
	}
	attempts := 0
	for {
		// Ask the POOL, not a node. Every backend the router would route to gets
		// a turn before the round counts as a failure: a pool whose first
		// backend cannot answer — dead, or a hosted endpoint with no listing —
		// still resolves from any sibling that can. Asking only one was the
		// original bug, and it is not hypothetical: an endpoint that answers
		// nothing at startup is classified passive, which marks it healthy, so
		// the broken one is exactly the one at the front of the list.
		avail := available(reg)
		if len(avail) == 0 {
			// Nothing routable yet. At startup this is the normal case, and a
			// pool fed by pod discovery may have no backends at all.
			if !sleep(ctx, autoModelPoll) {
				return
			}
			continue
		}

		var models []string
		var err error
		for _, b := range avail {
			models, err = listModels(ctx, b.URL, cred)
			if err == nil {
				break
			}
			log.Debug("auto-model: backend did not answer a model listing, trying the next",
				"pool", name, "backend", redactURL(b.URL), "err", err)
		}
		if err != nil {
			attempts++
			if attempts >= autoModelAttempts {
				log.Info("auto-model: giving up; no healthy backend in the pool answers a "+
					"model listing, so request models will be forwarded untouched. Use "+
					"'as <model>' to set one explicitly.",
					"pool", name, "backends", len(avail), "attempts", attempts, "err", err)
				return
			}
			if !sleep(ctx, autoModelPoll) {
				return
			}
			continue
		}

		picked := pickAutoModel(mode, models)
		if picked == "" {
			log.Info("auto-model: pool serves several models, so request models are "+
				"forwarded untouched (use 'as <model>' or --auto-model force to pin one)",
				"pool", name, "models", len(models))
			return
		}
		slot.Store(&picked)
		log.Info("auto-model: rewriting every matching request's model to what the pool serves",
			"pool", name, "model", picked)
		return
	}
}

// sleep waits d, reporting false if ctx ended first.
func sleep(ctx context.Context, d time.Duration) bool {
	select {
	case <-ctx.Done():
		return false
	case <-time.After(d):
		return true
	}
}

// available returns the backends the router would actually route to, which are
// the only ones worth asking. Health comes first by construction: Available() is
// false until the health checker has proved a backend up.
func available(reg *registry.Registry) []*registry.Backend {
	if reg == nil {
		return nil
	}
	var out []*registry.Backend
	for _, b := range reg.Snapshot().Backends {
		if b.Available() {
			out = append(out, b)
		}
	}
	return out
}

// redactURL strips any credential embedded in a backend URL before it reaches a
// log line. Rare, but a userinfo-bearing URL logged verbatim is a leaked secret.
func redactURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	return u.Redacted()
}

// listModels fetches the backend's model listing. The shape is the same on
// OpenAI-compatible servers and on Anthropic itself, so one parse covers both.
//
// cred is the pool's own credential, and the listing needs it: a fleet started
// with vLLM's --api-key answers an unauthenticated listing with 401, which
// discovery cannot tell from a backend that has no listing to give. It exhausts
// its attempts, the route's model reaches the backend unrewritten, and vLLM
// answers 404 for a model it does not serve — a backend error with nothing in it
// to point at the credential.
//
// The CALLER's credential is never an option here: this runs on a background
// goroutine at startup, where there is no caller, and a pool whose upstreams need
// a user's key cannot be probed by the router on anyone's behalf.
func listModels(ctx context.Context, backendURL, cred string) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, autoModelProbeTimeout)
	defer cancel()

	u := registry.ResolveURL(backendURL, "/v1/models")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if cred != "" {
		// Both styles, for the same reason the proxy sets both: a pool is
		// configured by URL, not by which credential convention it speaks.
		req.Header.Set("Authorization", "Bearer "+cred)
		req.Header.Set("X-Api-Key", cred)
		req.Header.Set("Anthropic-Version", "2023-06-01")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", u, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("GET %s: %s", u, resp.Status)
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("parse %s: %w", u, err)
	}
	out := make([]string, 0, len(payload.Data))
	for _, m := range payload.Data {
		if m.ID != "" {
			out = append(out, m.ID)
		}
	}
	return out, nil
}

// pickAutoModel applies the mode's selection policy. The empty string means
// "leave the client's model name alone".
//
// `auto` deliberately requires exactly one served model: that is the case where
// the pool has no choice to make and rewriting cannot route a request somewhere
// the operator did not intend. A multi-model upstream is left alone, so having
// discovery on by default cannot silently collapse a working fan-out.
func pickAutoModel(mode string, models []string) string {
	switch mode {
	case autoModelForce:
		if len(models) > 0 {
			return models[0]
		}
	case autoModelAuto:
		if len(models) == 1 {
			return models[0]
		}
	}
	return ""
}
