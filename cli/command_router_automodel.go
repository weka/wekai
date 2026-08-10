package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Auto-model discovery: a route with no explicit `as <model>` asks its own
// upstream what it serves (GET <upstream>/v1/models) and rewrites the body's
// `model` field to that answer.
//
// This exists because single-model servers (vLLM, llama.cpp, SGLang, …) reject
// any name they don't serve — vLLM answers `POST /v1/messages` with a bare 404
// when `model` isn't its one loaded checkpoint. Claude Code always sends
// `claude-*`, so pointing it at a local endpoint used to require the operator
// to look the checkpoint name up by hand and repeat it in `--default '… as …'`.
//
// The listing endpoint is shaped the same way on OpenAI-compatible servers and
// on Anthropic itself (`{"data":[{"id":…}]}`), so one parse covers both.
const (
	autoModelAuto  = "auto"  // rewrite only when the upstream serves exactly one model
	autoModelOff   = "off"   // never probe; `model` is forwarded untouched
	autoModelForce = "force" // rewrite to the first listed model, however many there are
)

const (
	// modelsProbeTimeout bounds a single listing request. Local upstreams answer
	// in milliseconds; a remote one that doesn't is not worth blocking startup for.
	modelsProbeTimeout = 3 * time.Second
	// autoModelRetryInterval paces background retries after a failed probe. The
	// upstream is commonly still loading weights when the router starts (or, in
	// k8s, not scheduled yet), so a failure at startup is not a permanent one.
	autoModelRetryInterval = 15 * time.Second
)

// joinPaths joins a base path with a leaf, tolerating a trailing or leading
// slash on either side.
func joinPaths(base, extra string) string {
	base = strings.TrimRight(base, "/")
	extra = "/" + strings.TrimLeft(extra, "/")
	return base + extra
}

// discoverUpstreamModels fetches <upstream>/v1/models and returns the served
// model IDs in the order the server listed them.
func discoverUpstreamModels(ctx context.Context, upstream *url.URL) ([]string, error) {
	u := *upstream
	u.Path = joinPaths(u.Path, "/v1/models")
	u.RawQuery = ""

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", u.String(), err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("GET %s: %s", u.String(), resp.Status)
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("parse %s: %w", u.String(), err)
	}

	ids := make([]string, 0, len(payload.Data))
	for _, m := range payload.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("%s listed no models", u.String())
	}
	return ids, nil
}

// pickAutoModel applies the mode's selection policy to a listing. The empty
// string means "leave the client's model name alone".
//
// `auto` deliberately requires exactly one served model: that is the case where
// the upstream has no choice to make and rewriting cannot route a request
// somewhere the operator didn't intend. A multi-model upstream (a real Anthropic
// endpoint, a multi-tenant gateway) is left alone, so enabling discovery by
// default cannot silently collapse a working fan-out onto one model.
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

// resolveAutoModels probes every rule that has no explicit `as <model>` and
// records what it found. Rules whose probe fails keep retrying in the
// background so a router that outraces its upstream still converges.
//
// Called before the listener starts, so the startup route lines already show
// whatever the fast path resolved.
func resolveAutoModels(rules []*routeRule, mode string) {
	if mode == autoModelOff {
		return
	}
	for _, r := range rules {
		if r.rewriteModel != "" {
			continue // operator was explicit; discovery has nothing to add
		}
		if probeAndApply(r, mode) {
			continue
		}
		go retryAutoModel(r, mode)
	}
}

// probeAndApply runs one probe and stores the result. It reports whether the
// probe itself succeeded — not whether a model was adopted, since a reachable
// multi-model upstream in `auto` mode is a settled answer ("don't rewrite"),
// not something to keep retrying.
func probeAndApply(r *routeRule, mode string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), modelsProbeTimeout)
	defer cancel()

	if len(r.endpoints) == 0 {
		// A discovery-only route has no endpoint to probe at startup: its pods
		// arrive from the watcher, later and possibly never. Reporting success
		// stops the retry loop rather than spinning on a rule that will never
		// have a static endpoint.
		return true
	}
	ep, err := url.Parse(r.endpoints[0])
	if err != nil {
		return false
	}
	// Endpoints in a pool are interchangeable by definition — they serve the
	// same model — so probing the first answers for all of them.
	models, err := discoverUpstreamModels(ctx, ep)
	if err != nil {
		log.Printf("auto-model: %s — probe failed (%v); retrying every %s", ep.Redacted(), err, autoModelRetryInterval)
		return false
	}
	picked := pickAutoModel(mode, models)
	if picked == "" {
		log.Printf("auto-model: %s serves %d models %v — leaving request models untouched (use 'as <model>' or --auto-model force to pin one)",
			ep.Redacted(), len(models), models)
		return true
	}
	r.autoModel.Store(&picked)
	log.Printf("auto-model: %s serves %q — rewriting every matching request's model to it", ep.Redacted(), picked)
	return true
}

func retryAutoModel(r *routeRule, mode string) {
	for {
		time.Sleep(autoModelRetryInterval)
		if probeAndApply(r, mode) {
			return
		}
	}
}
