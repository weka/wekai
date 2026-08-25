// Package obs sets up structured logging and the request-scoped context values
// that every log line and metric carries.
//
// Init must be the first call in main, before any other subsystem can emit
// output (CFG-10). v1 printed `DEBUG: Raw args: {:?}` — the entire command
// line, including anything passed as a secret flag — to stdout before logging
// was configured, so no log level could suppress it (CFG-N2).
package obs

import (
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
)

type Config struct {
	Level  string // debug | info | warn | error
	Format string // json | text
	Output io.Writer
}

func DefaultConfig() Config {
	return Config{Level: "info", Format: "json", Output: os.Stderr}
}

// Init installs the process logger and returns it.
func Init(cfg Config) *slog.Logger {
	if cfg.Output == nil {
		cfg.Output = os.Stderr
	}
	opts := &slog.HandlerOptions{Level: parseLevel(cfg.Level)}

	var h slog.Handler
	if strings.EqualFold(cfg.Format, "text") {
		h = slog.NewTextHandler(cfg.Output, opts)
	} else {
		// JSON is the default and is actually reachable, unlike v1's
		// json_format option which was hard-coded false at the call site.
		h = slog.NewJSONHandler(cfg.Output, opts)
	}
	l := slog.New(h)
	slog.SetDefault(l)
	return l
}

func parseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

type ctxKey int

const (
	keyRequestID ctxKey = iota
	keyRouteClass
	keyDialect
)

// WithRequestID stores the request id for the life of the request. The id is
// adopted from the inbound header when present and never re-minted on a forward
// hop, so one trace spans the whole path (GW-7, HIER-10, HIER-N5).
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, keyRequestID, id)
}

func RequestID(ctx context.Context) string { return str(ctx, keyRequestID) }

// RouteInfo is filled in by the handler and read by the access-log middleware.
//
// A plain context value cannot work here: the middleware sits OUTSIDE the mux, so
// the route class is not known when it builds its context, and a context derived
// inside the handler is invisible to the middleware that wrapped it. Without this
// indirection every metric and log line is labelled "unmatched", which makes the
// route dimension — the one operators slice by — worthless.
type RouteInfo struct {
	Class   string
	Dialect string

	// Pool, Backend and ModelOut are the routing OUTCOME, filled in once the
	// pool is chosen and the upstream has been picked. Capture reads them from
	// outside the mux for the same reason the access log reads Class from
	// there: it wraps the handler, so it cannot see a context derived inside
	// it, and without these a captured record cannot say where a request
	// actually went.
	Pool     string
	Backend  string
	ModelOut string

	// User is the caller identified by the leading path segment, when the
	// router runs with per-user prefixes. Empty otherwise.
	User string
}

// WithRouteHolder attaches a holder for a handler to populate, REUSING one an
// outer middleware already installed.
//
// Sharing is the point. Capture wraps the gateway from outside and the access
// log lives within it, so each would otherwise hold a holder the other never
// sees — and the one capture reads is the one no handler ever wrote to, which is
// how every captured record came to say it did not know which pool or backend
// served it.
func WithRouteHolder(ctx context.Context) (context.Context, *RouteInfo) {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		return ctx, ri
	}
	ri := &RouteInfo{}
	return context.WithValue(ctx, keyRouteClass, ri), ri
}

// SetUser records the caller a per-user path prefix named.
func SetUser(ctx context.Context, user string) {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		ri.User = user
	}
}

// SetRoute records the matched route class and dialect for this request.
func SetRoute(ctx context.Context, class, dialect string) {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		ri.Class, ri.Dialect = class, dialect
	}
}

// SetTarget records where this request was routed.
func SetTarget(ctx context.Context, pool, modelOut string) {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		ri.Pool, ri.ModelOut = pool, modelOut
	}
}

// SetBackend records which upstream actually served it, which is known only
// after selection and may differ from the first attempt after a retry.
func SetBackend(ctx context.Context, url string) {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		ri.Backend = url
	}
}

// Target reports the routing outcome, or the zero value if nothing routed.
func Target(ctx context.Context) RouteInfo {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		return *ri
	}
	return RouteInfo{}
}

func RouteClass(ctx context.Context) string {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		return ri.Class
	}
	return ""
}

func Dialect(ctx context.Context) string {
	if ri, ok := ctx.Value(keyRouteClass).(*RouteInfo); ok {
		return ri.Dialect
	}
	return ""
}

func str(ctx context.Context, k ctxKey) string {
	if v, ok := ctx.Value(k).(string); ok {
		return v
	}
	return ""
}

// Logger returns the default logger annotated with whatever request scope is
// present, so callers never have to thread a logger through signatures.
func Logger(ctx context.Context) *slog.Logger {
	l := slog.Default()
	if id := RequestID(ctx); id != "" {
		l = l.With("request_id", id)
	}
	if c := RouteClass(ctx); c != "" {
		l = l.With("route", c)
	}
	return l
}
