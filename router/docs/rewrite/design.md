# `wllm-router` v2 — Technical Design Document

**Module:** `github.com/weka/wllm-router` · **Go 1.25** · **Status:** design, pre-implementation
**Binds:** requirements spec §1–20 (`GW-*`…`NFR-*`, `OQ-1..9`) + `HIER-*` / `API-*` areas and resolved `OQ-10..15`.

> ## Amendments applied after review — read before implementing
>
> This document was reviewed and **corrected in place**. Where this banner conflicts with body text, the banner wins.
>
> | # | Change | Sections |
> |---|---|---|
> | **R1** | The LRU-tail-is-a-leaf argument was **inverted**. Replaced with a **leaves-only LRU**. | §D.4 (corrected inline) |
> | **R2** | Candidate filtering must not consume half-open circuit tokens. `Allow()` moves after selection. | §A.2 step 13, §B.6 (corrected inline) |
> | **R3** | Cache `Commit` moved to **after response headers arrive**, not before dispatch. | §A.2 steps 16–17 (corrected inline) |
> | **R4** | `pickBest` minimizes; `cache-usefulness` maximizes. Comparator is now parameterized so `LB-11`'s tie-break stays shared. | §C, §D.6 |
> | **R5** | Resolve the per-backend Prometheus child gauge once at registration, not per request. | §B.3 |
> | **C1** | **`queueing_penalty` is no longer the default.** `prefix-cache-aware` uses the `balance_abs_threshold`/`balance_rel_threshold` spill guard, per FR-RTR-01. | §D.6 (corrected inline) |
> | **C2** | No residency feed is available: `prefix-cache-aware` is **purely predictive**, indexed by **text prefix** (no tokenizer). FR-RTR-01 is *approximated, not satisfied*. | §D.5, §D.7 |
> | **CU-8** | `max_nodes_per_worker` default **500k → 100k** (500k breaks `NFR-5` at 64 backends). | §D.2 |
> | **HLT** | Health-check pool defaults to `min(256, max(32, N))`. | §F.2 |
> | — | **Hierarchical routing (§D.9, §F.4, §F.5) is DEFERRED** to post-v2.0. `Backend.kind` and `Backend.capacity` remain as no-op defaults so it stays additive. | — |
>
> **`plan.md` is authoritative** for product-requirement traceability (FR-RTR-01..05), the `RES-*` / `WEKA-*` / `LOAD-*` areas, milestones, and scope. §J's milestones are superseded by it.

---

## A. Architecture overview

### A.1 Directory layout

```
cmd/wllm-router/main.go        wiring, signal handling, listener lifecycle
internal/
  clock/          Clock interface + real/fake impls              AC-0.2
  config/         Config struct, loader, validation               CFG-1..CFG-11
  metrics/        every prometheus collector, declared once       OBS-3, OBS-4, OBS-5
  obs/            slog setup, access log, tracing shim            OBS-1, OBS-6, OBS-7
  jsonscan/       allocation-free partial JSON scanner            GW-6, API-15
  lease/          the load-accounting primitive                   LB-1..LB-7
  circuit/        sliding-window breaker + half-open semaphore    HLT-6..HLT-10
  registry/       Backend, canonicalization, COW Snapshot, drain  WRK-1..WRK-8, HIER-19
  health/         concurrent active checker; passive ingest       HLT-1..HLT-5, API-16
  discovery/k8s/  EndpointSlice + Pod informers                   SD-1..SD-10, HIER-16
  policy/         Policy iface, least-outstanding, rr, random     LB-8..LB-15, HIER-5
  policy/cache/   the one cache engine + two scorer presets       CACHE-*, CU-*, API-17
  cachetrie/      per-backend trie, LRU, Query/Commit             CU-1, CU-2, CU-5, CU-8
  dialect/        Dialect interface + registry (no wire knowledge) API-1, API-2
  dialect/openai/ the only shipped dialect                        API-3, API-11
  proxy/          ReverseProxy glue, retry, streaming relay       STR-*, REL-1..REL-6
  gateway/        mux, middleware chain, handlers, error envelopes GW-*, AUTH-*
  hier/           hop headers, node_state, forward semantics      HIER-1..HIER-19
  respmap/        bounded {id}→backend map                        GW-3, GW-4
hack/             CI checks: metric liveness, import fence, wekai drift
```

The import fence (`API-1`) is mechanical: `registry`, `lease`, `policy`, `policy/cache`, `cachetrie`, `proxy`, `circuit`, `health` must not transitively depend on `internal/dialect/...`. Only `gateway` and `cmd` may. Enforced by `hack/import_fence_test.go` (§G.3).

### A.2 Request path, end to end

1. **Accept.** `http.Server` on the inference listener (h2c-wrapped, `GW-16`).
2. **Recover** installs the panic→500 boundary (`REL-9`).
3. **Request id** taken from inbound `X-Request-Id` if ≤128 printable ASCII, else generated; stored in ctx. Never re-minted on a forward hop (`GW-7`, `HIER-10`/`HIER-N5`).
4. **Access-log/metrics** timer starts; deferred emit covers every subsequent outcome including 413/401/508 (`OBS-7`, `GW-N2`).
5. **Tracing** span started only if a tracer is installed; otherwise a nil-check branch, no allocation (`OBS-6`).
6. **CORS.** Preflight `OPTIONS` answered and returned here — before auth (`GW-10`, `GW-N3`, `SEC-10`).
7. **Body limit.** `r.Body = http.MaxBytesReader(w, r.Body, cfg.MaxBodyBytes)` — armed for every path including the catch-all (`GW-8`, `GW-9`, `GW-N1`). Nothing is read yet, so an unauthenticated client cannot force buffering.
8. **Auth.** The single enforcement site (`AUTH-4`). Constant-time compare, segment-boundary allowlist, admin paths never exemptible (`AUTH-1..AUTH-11`).
9. **Hierarchy ingest.** Parse `X-Wllm-Hops`, `X-Wllm-Via`, `X-Wllm-Deadline-Ms`, `X-Wllm-Attempts-Remaining`. Self-id in `Via` or hops > `max_hops` → `508` rendered in the inbound dialect (`HIER-2`). Deadline clamps `ctx` (`HIER-3`); attempts clamp the retry budget (`HIER-4`).
10. **Mux dispatch.** `http.ServeMux` with Go 1.22 method+wildcard patterns. **The matched pattern carries the dialect** — never sniffed from the body (`API-5`, `API-N1`).
11. **Body buffer.** Read up to `retry_buffer_bytes` (default 8 MiB) into a pooled buffer; set `req.GetBody`. Beyond that, the body streams and retries are disabled for this request (`REL-4`).
12. **Extraction.** One `jsonscan` pass yields `stream`, `model`, and the dialect's prefix units. Computed exactly once and reused by policy, metrics and logs (`CACHE-4`, `GW-6`).
13. **Candidate filter — side-effect-free (`R2`).** From the COW snapshot: healthy ∧ not draining ∧ `circuit.State() ∈ {Closed, HalfOpen}` ∧ `backend.Dialect == inbound dialect` ∧ (model label match, a no-op today) (`LB-9`, `API-7`, `API-13`). Empty → `503` with `no_healthy_backends` / `no_backend_for_dialect`.

    **Filtering reads `State()` only and never calls `Allow()`.** The original design filtered on "HalfOpen-*with-token*", which acquires a half-open semaphore token for *every* candidate while `Record()` releases only the *selected* one. With `HalfOpenMax = 1` a single filtering pass permanently exhausts a backend's probe budget: it can never be probed, so it can never close, so it never returns to rotation. That is v1's bug F1 — selection mutating circuit state as a side effect on every candidate — in a new form. `Allow()` is called exactly once, after selection, in step 15a.
14. **Select.** `policy.Select(ctx, candidates, req)`; timed into `router_routing_decision_duration_seconds` and nothing else (`OBS-8`, `NFR-2`).
15. **Lease.** `lease.Acquire(backend)` — the only increment site in the program (`LB-1`).
15a. **Circuit admission (`R2`).** `backend.CB.Allow()` exactly once, for the selected backend only. If denied, release the lease, drop that backend from the candidate set, and return to step 14. The returned token is released by the matching `Record()` in step 19.
16. **Dispatch.** `ReverseProxy.ServeHTTP` with a per-attempt `ResponseWriter` wrapper. Outbound request strips client credentials, injects the upstream credential, decrements the attempt budget (`AUTH-9`, `SEC-3`, `SEC-4`).
17. **Commit — after response headers arrive (`R3`).** Cache policies commit against the chosen backend only, in `ModifyResponse`, once the upstream has demonstrably *accepted* the request (`CACHE-9`, `CU-5`).

    The original design committed at step 16, before dispatch. A request that then failed at connect had already written its prefix into that backend's model, so the backend looked warm for a prefix it never received — permanently, and self-reinforcingly, since future requests would be steered toward it and would commit again. Committing on headers costs nothing: the `units` slice is already alive for the whole attempt.
18. **Relay.** `FlushInterval:-1`, 64 KiB pooled buffer, dialect stream scanner wrapping the response body (`STR-1`, `STR-5`, `API-8`).
19. **Outcome.** Classified for the breaker and passive health, releasing the step-15a token (`HLT-9`, `HLT-12`); usage sniffed for `cached_tokens` (`CU-13`, `API-10`, `RES-3`).
20. **Release.** `defer lease.Release()` fires after the body copy returns — i.e. after the stream completes or aborts (`LB-2`, `LB-4`).
21. **Retry** (only if nothing reached the client, attempts remain, and the failure is in `REL-2`): a *new* lease on a *different* candidate; step 14 re-runs on the reduced candidate set (`REL-1`, `REL-3`, `REL-5`, `LB-3`, `STR-9`). Because commit happens at step 17, a failed attempt leaves **no** trace in any cache model.

### A.3 Diagram

```
 client                    ROOT ROUTER (depth 0)
   │  POST /v1/chat/completions
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ recover │ reqid │ log/metrics │ trace │ CORS │ bodylimit │ AUTH │ hier│
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼  ServeMux (pattern ⇒ dialect=openai)
              ┌──────────────── handler ────────────────┐
              │ buffer body ─▶ jsonscan ─▶ units[]      │
              │        │                                │
              │        ▼                                │
              │  snapshot ─▶ filter(healthy,dialect,cb) │
              │        │                                │
              │        ▼   Select (NFR-2 budget)        │
              │   ┌─ policy ──────────────┐             │
              │   │ least-outstanding     │  inflight/capacity  ◀── HIER-5
              │   │ cache engine (Query×N)│  Σ named terms      ◀── API-17
              │   └───────────┬───────────┘             │
              │               ▼                          │
              │        lease.Acquire  ── the ONLY ++ ── LB-1
              │               ▼                          │
              │        cache.Commit(chosen)   ── CACHE-9 │
              │               ▼                          │
              │   ReverseProxy(attemptWriter) ── retry loop, REL-1..6
              └───────────────┬─────────────────────────┘
                              │  strip Authorization, inject upstream key
                              │  X-Request-Id (unchanged), X-Wllm-Via += self,
                              │  X-Wllm-Hops+1, X-Wllm-Deadline-Ms - elapsed - reserve,
                              │  X-Wllm-Attempts-Remaining - 1
        ┌─────────────────────┴──────────────────────┐
        ▼ kind=worker                                ▼ kind=router
   ┌──────────┐                            ┌────────────────────┐
   │ vLLM leaf│                            │ zone-b router  d=1 │
   └──────────┘                            │ (identical chain)  │
        ▲                                  └─────────┬──────────┘
        │  body bytes, 64 KiB pooled                 ▼
        │  FlushInterval -1                     vLLM leaves
        │  dialect StreamScanner ── data: [DONE] across chunk bounds
        └────────────────────────────────────────────────────────
   defer lease.Release()  ← fires after copy returns, LB-2/LB-4
   parent polls GET /v1/internal/node_state on the health interval,
   O(edges) total: each node polls only its own children.   HIER-6/HIER-N6
```

### A.4 `httputil.ReverseProxy` vs a hand-rolled proxy — **validate STR-10**, with conditions

I **validate** `STR-10`. Use one shared `*httputil.ReverseProxy` (not one per backend), with `Rewrite`, `ModifyResponse`, `ErrorHandler`, `BufferPool`, `FlushInterval: -1`.

What we get for free, all of which v1 hand-rolled and got wrong:

| Concern | `ReverseProxy` behaviour | Requirement |
|---|---|---|
| Backpressure | `io.CopyBuffer` with our 64 KiB pooled buffer; a slow client blocks the upstream read | `STR-1`, `STR-N1` |
| `Content-Type` | copied verbatim; never rewritten | `STR-2`, `STR-N2` |
| Non-2xx on a stream | relayed as-is, no SSE wrapping | `STR-3` |
| Per-chunk flush | `FlushInterval:-1` flushes after every `Write` | `STR-5` |
| Hop-by-hop | strips `Connection`-listed and the standard set both directions | `SEC-3` |
| Client cancel | `r.Context()` cancellation aborts the upstream round trip immediately | `GW-15`, `STR-7` |
| Trailers, 1xx, `Expect: 100-continue`, HTTP/2 | handled | `GW-16` |

The four constraints named in the brief are all satisfiable inside this shape:

- **`LB-4` lease-until-body-complete.** `ServeHTTP` returns *after* `copyResponse` completes. A `defer lse.Release()` in the calling handler is therefore exactly the right lifetime. No special work.
- **`API-8` dialect stream terminals.** `ModifyResponse` wraps `resp.Body` in a `scanReadCloser` holding the dialect's `StreamScanner`. Every byte flows through it before `copyResponse` sees it.
- **`HIER-3` deadline propagation.** Handled before `ServeHTTP` by clamping `ctx` and setting the header in `Rewrite`.
- **`HIER-4` / `REL-1` attempt budgets and retry.** This is the only genuinely awkward part and costs ~120 LOC of glue:

```go
var errRetryable = errors.New("retryable upstream outcome")

// per-attempt state hung off the outbound request context
type attempt struct {
    lse       *lease.Lease
    backend   *registry.Backend
    outcome   circuit.Outcome
    retryable bool
    committed *bool // shared across attempts: true once a byte reached the client
}
```

`ModifyResponse` classifies the status; if it is retryable **and** `!*committed` **and** attempts remain, it returns `errRetryable`, which makes `ReverseProxy` close the upstream body and call `ErrorHandler`. Our `ErrorHandler` **writes nothing** — it only records the outcome. The handler's `attemptWriter` swallows any write while `retryable` is set and flips `*committed = true` on the first byte that is actually passed through, which is precisely `REL-3` / `STR-9`.

**Dissent, narrowly:** we do *not* use `ReverseProxy`'s default `ErrorHandler` (it writes a bare 502 with no dialect envelope, violating `API-9`), and we do not use `Director` (it preserves inbound `X-Forwarded-*`; `Rewrite` clears them, which is the `SEC-9` default we want). A fully hand-rolled loop would be roughly the same LOC as the glue but would re-open every bug in the `STR-N*` list, so it is rejected.

---

## B. Core types and interfaces

### B.1 Backend

```go
package registry

type Kind uint8   // HIER-1
const (KindWorker Kind = iota; KindRouter)

type Provenance uint8 // HIER-19, WRK-7
const (ProvStatic Provenance = iota; ProvDiscovered)

type HealthModel uint8 // API-16
const (HealthActive HealthModel = iota; HealthPassive)

// Backend is dialect-agnostic: DialectID is an opaque string the core only
// ever compares for equality. No dialect package is imported here (API-1).
type Backend struct {
    URL       string        // canonical: scheme://host:port, no path/slash (WRK-1)
    Kind      Kind          // worker | router                      HIER-1
    DialectID string        // "openai"; declared, never probed     API-6
    Health    HealthModel   // active | passive                     API-16
    Prov      Provenance    // static wins over discovered          HIER-19
    Model     string        // label; equality filter, no-op today  NG-6/OQ-4
    Locality  string        // zone/rack                            HIER-15

    // Capacity is the denominator of normalized load (HIER-5).
    // Leaf: cfg.MaxInflightPerWorker (default 1).
    // Router child: subtree_capacity from node_state, ≥1, refreshed
    // on the health interval; degrades to 1 on error (HIER-6).
    capacity atomic.Int64

    inflight  atomic.Int64  // written ONLY by internal/lease (LB-6)
    draining  atomic.Bool   // WRK-6
    health    atomic.Int32  // Unknown|Healthy|Unhealthy (HLT-3/HLT-5)
    CB        *circuit.Breaker

    // R5: resolved once at registration; WithLabelValues on the request path
    // costs a lock + map lookup, i.e. 40k resolutions/s at NFR-1 load.
    InflightGauge prometheus.Gauge

    Served, Failed atomic.Uint64  // cumulative (WRK-5)
    LastTransition atomic.Int64   // unix nanos, from clock.Clock
}

func (b *Backend) Inflight() int64 { return b.inflight.Load() }
func (b *Backend) Capacity() int64 { c := b.capacity.Load(); if c < 1 { return 1 }; return c }
func (b *Backend) NormalizedLoad() float64 {                 // HIER-5, HIER-N1
    return float64(b.inflight.Load()) / float64(b.Capacity())
}
```

`inflight` is unexported and the only mutator lives in `internal/lease`; `hack/lease_fence_test.go` greps for `.inflight.Add(` outside that package (`LB-6`, `HLT-N5`).

### B.2 Registry and copy-on-write Snapshot

```go
type Snapshot struct {
    Version  uint64
    Backends []*Backend // sorted by URL, immutable after publish (WRK-3/WRK-4)
    byURL    map[string]*Backend
}

type Registry struct {
    mu   sync.Mutex             // serializes writers only
    cur  atomic.Pointer[Snapshot]
    hook func(added, removed []*Backend) // predictor lifecycle (CU-4/CU-12)
}

// Snapshot is wait-free: one atomic load. Every policy, admin handler and
// metric reads exclusively through this (WRK-3).
func (r *Registry) Snapshot() *Snapshot { return r.cur.Load() }

// Apply mutates under mu, builds a fresh sorted slice, publishes atomically.
// Idempotent: applying the same desired set twice yields an identical
// Snapshot.Backends (pointer-equal elements) and does NOT bump Version (SD-7).
func (r *Registry) Apply(mut func(w *writeView)) (changed bool)
```

`*Backend` pointers are **shared** across snapshots — only the slice and map are copied. Per-backend mutable state is atomic, so a policy holding a snapshot sees a stable *membership list* with live counters, which is exactly what `WRK-4` requires. Snapshot construction is O(N log N) at ~64–1000 backends, on the discovery/admin path only.

### B.3 The `Lease` — the load-accounting primitive

```go
package lease

// Lease is the single, symmetric, RAII-style lifecycle for in-flight load.
// Acquire is the ONLY increment site in the program (LB-1).
// Release is idempotent (LB-2) and underflow-safe (LB-5).
type Lease struct {
    b    *registry.Backend
    once sync.Once
}

func Acquire(b *registry.Backend) *Lease {
    b.AddInflight(+1)                 // package-internal accessor
    b.InflightGauge.Inc()             // R5: child gauge resolved at registration
    return &Lease{b: b}
}

// Release is safe to call any number of times from any number of goroutines.
// nil-receiver-safe so `defer lse.Release()` is valid before Acquire succeeds.
func (l *Lease) Release() {
    if l == nil {
        return
    }
    l.once.Do(func() {
        if n := l.b.AddInflight(-1); n < 0 {          // LB-5
            metrics.LoadAccountingErrors.Inc()
            slog.Error("in-flight underflow: this is a bug",
                "worker", l.b.URL, "value", n)
            l.b.StoreInflight(0)                       // clamp; never wrap
        }
        l.b.InflightGauge.Dec()                        // R5
    })
}
```

How it survives each hazard:

- **Retry to a different backend (`LB-3`, `REL-5`).** Each attempt owns its own `*Lease` inside the loop body. Leases are never transferred or reused:

```go
for att := 0; att < budget; att++ {
    b, err := pol.Select(ctx, candidates, rr)
    if err != nil { return err }
    lse := lease.Acquire(b)
    res := px.do(ctx, w, r, b, lse, &committed)
    lse.Release()                      // explicit, at the end of THIS attempt
    if !res.retryable || committed { return res.err }
    candidates = without(candidates, b) // REL-1: a different worker
}
```
  Because `Release` is `sync.Once`-guarded, the extra `defer lse.Release()` inside `px.do` (which fires on early return, panic, or error handler) is harmless — the *first* release wins and the second is a no-op. This is the structural fix for `LB-N1`: v1 had one increment and three decrements; here there are three *call sites* and one *effect*.

- **Client cancellation (`GW-15`).** Cancellation aborts `ReverseProxy.ServeHTTP`, which returns, which runs the deferred release inside `do`. The 100 ms budget is met because `http.Transport` cancels the round trip on ctx done with no polling.

- **Streaming until body complete (`LB-4`).** `ServeHTTP` does not return until `copyResponse` finishes, so the release is naturally at end-of-stream. On abort mid-stream, `copyResponse` returns an error and `ServeHTTP` still returns. There is no path where the lease outlives the goroutine that holds it — no `go func(){...}()` anywhere in the release path.

- **Panic (`REL-9`).** The deferred release runs during unwinding, before the recover boundary.

`AddInflight` is `func (b *Backend) AddInflight(d int64) int64` with a `//go:linkname`-free but *package-visibility-fenced* discipline: it is exported on `Backend` (Go has no friend packages) and guarded by the CI grep fence instead. That is the honest trade — documented at the declaration.

### B.4 Policy

```go
package policy

var ErrNoCandidates = errors.New("no candidates")

// RoutingRequest carries nothing dialect-specific. Units are pre-extracted
// opaque (hash, tokens) pairs; DialectID is an opaque string used only for
// metric labels — candidate filtering by dialect already happened upstream
// (API-1, API-7).
type RoutingRequest struct {
    RequestID string
    RouteClass string          // "chat", "completions", "embeddings", ...
    DialectID  string
    Model      string
    Stream     bool
    Units      []cachetrie.Unit // may be nil ⇒ cache policies decline (CU-11)
    Locality   string           // client hint, usually ""      HIER-15
    Deadline   time.Time
}

type Policy interface {
    Name() string
    Select(ctx context.Context, candidates []*registry.Backend, rr *RoutingRequest) (*registry.Backend, error)
}

// Committer is optional; only the cache engine implements it (CACHE-9, CU-5).
type Committer interface {
    Commit(b *registry.Backend, rr *RoutingRequest)
}
```

`policy` imports `registry`, `cachetrie`, `clock`, `metrics` — and no dialect. `cachetrie.Unit` is `struct{ Hash uint64; Tokens int32 }`: pure numbers, no prompt bytes, which is also what makes `HIER-13` trivially satisfiable if bid mode is ever built.

### B.5 Dialect — exactly the seven concerns of `API-2`

```go
package dialect

type Dialect interface {
    ID() string

    // (a) routes it claims: pattern ⇒ route class. Registered into ServeMux.
    Routes() []Route                                            // API-5

    // (b) prefix units for the cache policies, from a raw body, no
    //     deserialization and no re-serialization (GW-6, API-15, API-11).
    ExtractUnits(body []byte, class string, dst []cachetrie.Unit) ([]cachetrie.Unit, bool)

    // (c) streaming terminal / event framing, chunk-boundary correct.
    NewStreamScanner() StreamScanner                            // API-8, STR-4

    // (d) error envelope rendering, in the dialect of the INBOUND route.
    WriteError(w http.ResponseWriter, status int, code, msg string) // API-9

    // (e) usage / cached-token extraction from a response body.
    ExtractUsage(body []byte) (Usage, bool)                     // API-10, CU-13

    // (f) recognized inbound credential header form.
    Credential(h http.Header) (token string, ok bool)           // AUTH-3

    // (g) where the model identifier lives.
    Model(body []byte) (string, bool)
}

type StreamScanner interface {
    // Feed reports whether the dialect's terminal marker has been seen.
    // Correct across arbitrary chunk boundaries; carries a bounded
    // partial-line buffer (8 KiB cap, after which it degrades to "no
    // terminal seen" rather than growing).
    Feed(p []byte) (terminal bool)
}

type Route struct {
    Pattern string // Go 1.22 "POST /v1/chat/completions"
    Class   string
    Stream  bool   // may stream
}

type Usage struct{ PromptTokens, CachedTokens, Total int }

func Register(d Dialect)              // called from cmd/ only
func Lookup(id string) (Dialect, bool)
```

Seven methods, seven concerns. Adding Anthropic is a new package plus one `Register` call in `cmd/` (`API-3`); the compile-time proof is a `stubdialect` registered in `gateway`'s tests.

### B.6 CircuitBreaker

```go
package circuit

type State uint8
const (Closed State = iota; Open; HalfOpen)

type Outcome uint8
const (Success Outcome = iota; Failure)

// Classify is explicit and total (HLT-9, HLT-N4).
//   5xx, connection error, timeout        → Failure
//   429, 503                              → Failure (overload)
//   408, 425                              → Failure
//   other 4xx                             → Success (client's fault)
//   2xx, 3xx                              → Success
func Classify(status int, err error) Outcome

type Config struct {
    Window      time.Duration // default 30s, REALLY used (HLT-7, HLT-N2)
    Buckets     int           // default 30 ⇒ 1s resolution
    MinRequests int           // default 20
    FailureRate float64       // default 0.5
    OpenFor     time.Duration // default 30s before HalfOpen
    HalfOpenMax int32         // default 1 (HLT-8)
}

type bucket struct{ startSec int64; ok, fail uint32 }

type Breaker struct {
    cfg   Config
    clk   clock.Clock              // AC-0.2

    mu    sync.Mutex               // guards ring + state; ~40ns, off the hot read path
    ring  []bucket                 // fixed-size, index = (unixSec % Buckets)
    state State
    openedAt time.Time

    halfOpenTokens atomic.Int32    // semaphore, NOT a state check (HLT-N3)
}

// Allow is the hot path: one atomic in Closed, one CAS loop in HalfOpen.
func (b *Breaker) Allow() (permitted bool, token bool)
// Record must be called with token from Allow; releases the half-open token.
func (b *Breaker) Record(o Outcome, token bool)
func (b *Breaker) State() State
```

The ring is a fixed slice of `Buckets` entries; a bucket whose `startSec` is older than `now-Window` is treated as empty and lazily reset — so the sliding window is O(Buckets) to sum and allocation-free, and `Window` is genuinely read (`HLT-N2`). Half-open admission is a `CompareAndSwap` on `halfOpenTokens`, so 100 concurrent probes admit exactly `HalfOpenMax` (`HLT-N3`). Every transition logs old→new plus the triggering counters (`HLT-10`).

---

## C. Policies

All five share the same skeleton: single pass over `candidates`, reservoir tie-break, no allocation.

```go
// pickBest is the ONE tie-break implementation (LB-11, LB-N4).
// Single pass, uniform over the tied set, zero allocation.
//
// CORRECTED (R4): the comparator is parameterized. The original hard-coded
// `s < bestScore` with a +Inf seed, i.e. minimize — which the load policies
// want but `cache-usefulness` (which maximizes an expected-time-saved score)
// cannot use. Left as-is, that policy would have grown its own selection loop
// and quietly stopped enforcing LB-11's uniform tie-break.
type direction bool
const (minimize direction = false; maximize direction = true)

func pickBest(cands []*registry.Backend, dir direction, score func(*registry.Backend) float64) *registry.Backend {
    var best *registry.Backend
    bestScore, ties := math.Inf(1), 0
    if dir == maximize { bestScore = math.Inf(-1) }
    for _, c := range cands {
        s := score(c)
        better := s < bestScore
        if dir == maximize { better = s > bestScore }
        switch {
        case better:
            best, bestScore, ties = c, s, 1
        case s == bestScore:
            ties++
            if rand.IntN(ties) == 0 { best = c }   // math/rand/v2 reservoir
        }
    }
    return best
}
```

### C.1 `least-outstanding` (default, `LB-13` amended by `HIER-5`)

- **Score:** `float64(inflight) / float64(capacity)`. A leaf's capacity is `max_inflight_per_worker` (default 1) so the score degenerates to raw in-flight; a child router's capacity is its polled `subtree_capacity`, so a child fronting 40 GPUs with 40 in flight scores 1.0 against an idle leaf's 0.0 and a half-loaded leaf's 0.5 — it is chosen ahead of the saturated leaf and behind the idle one. That is exactly the `HIER-N1` fix.
- **Structure:** none. Reads `atomic.Int64`s off the snapshot.
- **Tie-break:** reservoir (above).
- **Complexity:** O(N), N atomic loads. At N=64 this is ~200 ns — two orders under the `NFR-2` p99 of 250 µs.

### C.2 `round-robin` (`LB-14`) — the starvation-free cursor

The obvious `cursor.Add(1) % len(candidates)` is **wrong** and I reject it explicitly. Two independent failures: (i) the modulus changes when the set resizes, so the cursor maps to a different member and the rotation restarts arbitrarily; (ii) a member that is briefly unhealthy loses its slot permanently once the set grows back, because index *k* now means someone else — a worker can be starved indefinitely while the counter marches on.

**Design: virtual-time (last-served sequence) selection.** Do not index the candidate slice at all.

```go
type roundRobin struct {
    mu   sync.Mutex
    seq  int64
    last map[string]int64 // canonical URL → sequence when last served
}

func (p *roundRobin) Select(_ context.Context, cands []*registry.Backend, _ *RoutingRequest) (*registry.Backend, error) {
    if len(cands) == 0 { return nil, ErrNoCandidates }
    p.mu.Lock()
    defer p.mu.Unlock()

    // Newcomers enter the rotation at the current front, not at the back:
    // they inherit the minimum last-served sequence among current candidates,
    // so they compete on equal terms immediately and are never starved.
    min := int64(math.MaxInt64)
    for _, c := range cands {
        if v, ok := p.last[c.URL]; ok && v < min { min = v }
    }
    if min == math.MaxInt64 { min = p.seq }
    for _, c := range cands {
        if _, ok := p.last[c.URL]; !ok { p.last[c.URL] = min }
    }

    // Select smallest last-served; ties broken by canonical URL for
    // determinism (LB-10) — the set is genuinely ordered, so a random
    // tie-break here would break the exactly-10-each guarantee.
    best := cands[0]
    for _, c := range cands[1:] {
        if lv, bv := p.last[c.URL], p.last[best.URL]; lv < bv || (lv == bv && c.URL < best.URL) {
            best = c
        }
    }
    p.seq++
    p.last[best.URL] = p.seq
    return best, nil
}
```

Why this is starvation-free across a changing set: `last[w]` is a per-worker fact, not a positional one. Removing and re-adding *w* does not change `last[w]`, so on return *w* has the oldest sequence in the set and is served next — it is *compensated*, not skipped. Shrink-then-grow visits every member (the `HIER`/`LB-14` acceptance test). Over 10N requests to N stable workers each is served exactly 10 times, because the selection is precisely "serve the least-recently-served", which is a strict rotation on a stable set.

- **Structure:** one map bounded by registry size; pruned against the snapshot version on each change (entries for URLs absent from the current snapshot are dropped, with a grace of one version so drain flaps don't lose position).
- **Complexity:** O(N) with one mutex. At N=64, ~1.5 µs including map lookups. Well inside budget; if it ever isn't, `last` can be moved into `*Backend` as an `atomic.Int64`, making it lock-free — noted as the escape hatch, not shipped.
- **Bonus:** this is a virtual-time scheduler, so `HIER-5` weighting is a one-line change (`p.last[best] = p.seq` → `+= 1/capacity`). Not shipped in v2.0 (`LB-20`: unconsumed knobs fail startup).

### C.3 `random` (`LB-15`)

`cands[rand.IntN(len(cands))]` using `math/rand/v2` (per-P state, no global lock). O(1). Chi-square over 100k draws across 8 workers.

### C.4 & C.5 `prefix-cache-aware` and `cache-usefulness`

Both are the same engine with different scorer presets — see §D.

---

## D. The cache engine

### D.1 OQ-3 — merge. One engine, two scorer presets.

**Decision: merge.** `internal/policy/cache` is a single policy type holding one `*cachetrie.Trie` per backend, one extractor (the dialect's), one eviction model, one Query/Commit split. The *only* difference between the two named policies is the term set, and `API-17` already mandates that scoring be a sum of named toggleable terms. Shipping two engines would duplicate the trie, the LRU, the lifecycle hooks and the fallback logic — roughly 400 LOC of the 8,000 budget — for zero behavioural difference.

Config maps names to presets:

| Policy name | Terms enabled | Extra behaviour |
|---|---|---|
| `cache-usefulness` | `predicted_time_saved`, `queueing_penalty` | none |
| `prefix-cache-aware` | `matched_fraction` (step at `cache_threshold`), `queueing_penalty` off | imbalance guard (`CACHE-3`) then fall back to `least-outstanding` |

`prefix-cache-aware`'s literal `CACHE-2`/`CACHE-3` semantics are preserved by a `matched_fraction` term that returns `+∞` for the single owner whose matched fraction ≥ `cache_threshold` and 0 otherwise, followed by the imbalance guard computed **over `candidates` only** (`CACHE-N4`, `LB-N7`). So the spec text is satisfied verbatim while the code is one path.

**Recommendation for GA:** expose both names in v2.0, default to `least-outstanding`, and use `CU-13`'s predicted-vs-observed histogram to decide before GA whether `prefix-cache-aware` is worth keeping as a public name. My expectation is that it is not.

### D.2 Per-backend trie

```go
package cachetrie

type Unit struct {
    Hash   uint64 // sha256(role \0 content)[:8], big-endian — wekai-identical
    Tokens int32  // len(bytes)/4, min 1  (CACHE-6)
}

type node struct {
    key      uint64
    tokens   int32
    nkids    int32
    kids     []child          // sorted by key; binary search
    parent   *node
    lru      link             // intrusive: prev, next *node
    lastUsed uint32           // coarse tick, for debugging/asserts only
}
type child struct{ key uint64; n *node }
```

**Bytes per node.** `key` 8 + `tokens` 4 + `nkids` 4 + `kids` slice header 24 + `parent` 8 + `lru` 16 + `lastUsed` 4 (+4 pad) = **72 B**, plus the backing array for `kids`: 16 B/child, and Go's size class rounds the node to 80 B. A single-child node therefore costs ~**96 B**.

I deliberately reject `map[uint64]*node` children (wekai's shape). An allocated Go map with one entry costs ~48 B of `hmap` plus a ~200 B bucket, i.e. ~250 B per node against 96 B — a 2.6× memory penalty on a structure where the overwhelming majority of nodes have exactly one child. The sorted-slice + binary search is also faster below ~16 children.

**This changes a default.** `CU-8` proposes 500k nodes/worker; at 96 B that is 48 MiB/worker, and 64 workers is 3.0 GiB — a direct `NFR-5` violation. Recommended amendment: **default `max_nodes_per_worker = 100_000` (~10 MiB/worker, 640 MiB at N=64)**. In practice the node cap is not the binding constraint anyway: at the default 1024-byte unit granularity, the 2M-est-token budget is reached at ≈ 2,000,000 / 256 ≈ **7,800 nodes**. The token budget binds first by more than an order of magnitude. The node cap exists purely as a hard backstop for pathological fine granularity.

**Concurrency: one `sync.RWMutex` per backend trie.** Query takes `RLock`, Commit and eviction take `Lock`.

Justification against the alternatives, measured against `NFR-2` (p99 ≤ 1 ms Select, 64 backends, 32 KiB prompt = 32 units):

- *Single `sync.Mutex` (wekai's choice).* wekai has one trie; we have 64, and Query runs against all 64 on the request path. Per-trie contention is already only 1/64 of write traffic, so a Mutex would probably be adequate — but Query is also concurrent *with itself* across in-flight requests against the same trie, and at 20k req/s (`NFR-1`) that is 20k lock acquisitions per second per trie serialized behind a ~1.5 µs walk. `RWMutex` makes the read side parallel for the cost of one extra word. Cheap insurance.
- *Sharded trie.* Rejected: a prefix walk is inherently sequential from the root, so there is no shard key. You would have to shard by *root child*, which buys nothing because the root's fan-out is dominated by a handful of system prompts.
- *Copy-on-write.* Rejected: a Commit rewrites the path to the root, so every request allocates `depth` nodes (32 × 96 B = 3 KiB/request, 60 MB/s of garbage at NFR-1 load) — a direct `NFR-3`/`NFR-4` GC-pressure problem. Eviction under COW is worse still.

**Query cost.** 64 backends × 32 units, each unit a binary search over a small `kids` slice: ~64 × 32 × ~30 ns ≈ **61 µs**, plus 64 `RLock`/`RUnlock` pairs (~1.5 µs). p99 comfortably under 1 ms with an order of magnitude of headroom, which is what the `CU-15` 2 ms hard deadline is there to catch when it isn't.

### D.3 Query / Commit split (`CU-5`, `CACHE-9`)

```go
// Query is pure and read-only. Safe for unbounded concurrency across all
// backends. Zero allocations: no slices are built, no LRU is touched, no
// counters move (CU-N2 — 1,000 Query calls leave state byte-identical).
func (t *Trie) Query(units []Unit) (predictedCachedTokens, totalTokens int) {
    for _, u := range units { totalTokens += int(u.Tokens) }
    t.mu.RLock()
    n := &t.root
    for _, u := range units {
        c := n.find(u.Hash)          // binary search, no alloc
        if c == nil { break }
        predictedCachedTokens += int(c.tokens)
        n = c
    }
    t.mu.RUnlock()
    return
}

// Commit is called on exactly one backend, after selection succeeded and
// the request was dispatched (CACHE-9, CU-N2, HIER-14).
func (t *Trie) Commit(units []Unit) {
    t.mu.Lock()
    n := &t.root
    i := 0
    for ; i < len(units); i++ {
        c := n.find(units[i].Hash)
        if c == nil { break }
        t.touch(c)                   // LRU move-to-front, O(1)
        n = c
    }
    for ; i < len(units); i++ {
        c := t.insert(n, units[i])   // links into LRU at MRU
        n = c
    }
    t.evictLocked(evictBudgetPerCommit) // amortized, same lock (CACHE-7)
    t.mu.Unlock()
}
```

**How Query stays allocation-free.** The `units` slice is built exactly once per request during extraction (`CACHE-4`) into a `[]Unit` drawn from a `sync.Pool` and returned to it in the handler's defer. `Query` returns two `int`s by value; no closure, no interface boxing (the scorer takes `(cached, total int, b *Backend)` as plain arguments, not a `func` value per candidate). Escape analysis confirms zero heap traffic; `BenchmarkQuery64x32` asserts `0 allocs/op` in CI.

Note the deliberate asymmetry: **Query does not touch the LRU.** Cache warmth must follow what was actually *sent*, not what was speculatively considered — otherwise every backend's LRU is refreshed by every request and eviction order becomes meaningless. This also makes the `CU-N2` purity test exact.

### D.4 Bounded LRU eviction (`CU-8`, `CACHE-5`, `CACHE-7`)

Intrusive doubly-linked list over nodes, embedded in the node (`lru link`), plus running totals `nodes int64` and `tokens int64`.

> **CORRECTED (R1).** The original argument here was inverted and is retained only as a warning. It claimed: *"`Commit` touches every ancestor before the matched node, therefore `lastUsed` is monotonically non-increasing from root to leaf, so the LRU tail is always a leaf."*
>
> That is backwards. Touching root-side nodes **first** makes the deepest node the *most* recently used, so `lastUsed` is non-**de**creasing from root to leaf and the tail trends toward nodes **with children**. Combined with the original `if v.nkids != 0 { return }` guard, the failure mode was: **eviction silently stops entirely**, the trie grows without bound (`NFR-5`, `CU-N3` violated), and the only signal is a metric nobody watches.
>
> Counterexample: a conversation committed once and never again, path root→A→B→C. Touch order A, B, C leaves A nearest the tail with `nkids == 1`, so eviction hits the guard and returns having freed nothing.

**The invariant, by construction: only leaves are in the LRU list.**

```go
// insert links the new node at MRU and unlinks its parent, which is no
// longer a leaf. touch() moves an existing LEAF to MRU; interior nodes are
// not in the list at all, so there is nothing to move.
func (t *Trie) insert(parent *node, u Unit) *node {
    if parent.nkids == 0 && parent != &t.root {
        t.lruRemove(parent)              // parent stops being a leaf
    }
    n := &node{key: u.Hash, tokens: u.Tokens, parent: parent}
    parent.addChild(n)
    t.lruPushFront(n)                    // new node is a leaf
    t.nodes++; t.tokens += int64(u.Tokens)
    return n
}

func (t *Trie) evictLocked(budget int) {
    for budget > 0 && (t.nodes > t.maxNodes || t.tokens > t.maxTokens) {
        v := t.lru.tail
        if v == nil || v == &t.root { return }
        // The list contains only leaves, so this holds by construction.
        // It is now a genuine assertion, not a silent stall.
        if v.nkids != 0 {
            panic("cachetrie: interior node in LRU list — invariant broken")
        }
        p := v.parent
        p.removeChild(v.key)
        t.lruRemove(v)
        t.nodes--; t.tokens -= int64(v.tokens)
        metrics.CacheEvictions.WithLabelValues(t.url).Inc()
        if p.nkids == 0 && p != &t.root {
            t.lruPushBack(p)             // parent became a leaf: evictable next
        }
        budget--
    }
}
```

Both operations stay O(1). The parent is relinked at the **tail**, not the front, so a prefix whose entire subtree has been evicted is itself reclaimed promptly rather than being treated as freshly used.

Note this also gives the semantically right ordering for free: a shared system prompt sits on an interior node, so it is not in the LRU list at all and cannot be evicted while any conversation still depends on it — it becomes evictable only once its last descendant is gone.

**Interaction with readers — `CACHE-N2` becomes structurally impossible.** There is **no evictor goroutine**. Eviction runs inline at the end of `Commit`, under the *same* write lock as insertion, with a per-commit work budget (default 64 nodes) so it cannot spike tail latency; if still over budget it simply continues on the next Commit. Readers hold `RLock`, so they cannot be concurrent with an unlink at all — there is no window in which a reader observes a partially-unlinked subtree, and no stop-the-world DFS to race with. This also kills `CACHE-N3` outright: the cache subsystem owns zero goroutines, so `goleak` on policy shutdown is trivially satisfied and `CACHE-8` is met vacuously.

`Trie.Reset()` backs `POST /flush_cache` and touches nothing outside the trie (`CACHE-13`).

Per-backend lifecycle: `Registry.hook` creates a trie on add and drops it on remove; prefixes are never reassigned (`CACHE-10`, `CU-4`, `CU-12`).

### D.5 Prefix-unit construction (`CU-3`, `API-11`)

The builder is structured with **Anthropic as the reference model and OpenAI as the adaptation**, because `BuildReplayRequestPrefix`'s ordering *is* the Anthropic Messages shape:

```go
// unitbuild.Canonical is the reference ordering, taken verbatim from
// wekai BuildReplayRequestPrefix (replay_router.go:731):
//   system blocks → tools → messages.
//
// skipTinyLeadingSystemBlock: Anthropic emits a per-request billing header
// as system block 0. It is small (<200 bytes) and near-unique per request,
// so including it poisons every downstream sequential prefix-block hash —
// the whole prefix diverges at unit 0 and nothing ever matches. This is an
// ANTHROPIC BILLING-HEADER ARTIFACT, NOT A UNIVERSAL RULE (API-11), and it
// MUST NOT be applied to OpenAI requests, which have no such block.
func Build(p Parts, skipTinyLeadingSystemBlock bool, dst []Unit) []Unit {
    for i, sb := range p.System {
        if skipTinyLeadingSystemBlock && i == 0 && len(sb) < 200 { continue }
        dst = append(dst, unit("system", sb))
    }
    if len(p.Tools) > 0 { dst = append(dst, unit("tools", p.Tools)) }
    for _, m := range p.Messages { dst = append(dst, unit(m.Role, m.Content)) }
    return dst
}
```

- `dialect/anthropic` (future) calls `Build(parts, true, dst)`; its `req-v1` golden test proves byte-identical reproduction (`API-12`).
- `dialect/openai` calls `Build(parts, false, dst)` — `Parts.System` holds the leading `role:"system"` messages, `Tools` the `tools`/`functions` array bytes, `Messages` the remainder in order. Unit test named `TestSkipTinyLeadingSystemBlock_NotAppliedToOpenAI` (`CU-3` asserts the behaviour *by name*, in both directions).

Hashing stays `sha256(role || 0x00 || content)[:8]` interpreted big-endian — bit-identical to wekai's `hashMessage` so the golden corpus test is meaningful, and collision-resistant, which matters: a non-cryptographic hash would let a client craft a prompt that collides with another tenant's prefix and steal its affinity. 32 KiB of SHA-256 with SHA-NI is ~16 µs, once per request during extraction, and therefore **outside** the `OBS-8`/`NFR-2` Select budget.

Units above the sub-1024-byte-message case: OpenAI messages larger than `cache_unit_bytes` are additionally split into fixed windows via wekai's `chunkPromptPrefixN` logic, so long single-message prompts still get sub-message resolution.

### D.6 Scoring as a sum of named terms (`API-17`, `CU-6`, `CU-7`)

```go
// Term returns a score in SECONDS. Positive is good (time saved);
// negative is bad (time cost). Terms are summed; each is individually
// toggleable by name in config, and a new term (per-token cost, a
// retry-after penalty) is added without touching Policy or any other term.
type Term interface {
    Name() string
    Score(b *registry.Backend, cachedTok, totalTok int) float64
}

type predictedTimeSaved struct{ perToken float64 } // 1/coldTPS - 1/warmTPS
func (t predictedTimeSaved) Score(_ *registry.Backend, cached, _ int) float64 {
    return float64(cached) * t.perToken
}

type queueingPenalty struct{ estServiceSec float64 }
func (t queueingPenalty) Score(b *registry.Backend, _, _ int) float64 {
    return -b.NormalizedLoad() * t.estServiceSec     // CU-7, HIER-5-aware
}
```

**Default constants**, labelled in config docs and `/get_server_info` as *measured on our hardware, not universal* (`CU-6`):

| Knob | Default | Note |
|---|---|---|
| `cold_tps` | 2,500 tok/s | prefill on a cold KV block |
| `warm_tps` | 20,000 tok/s | prefix-cache hit path |
| `output_tps` | 60 tok/s | unused by scoring in v2.0; recorded for OQ-6 |
| `est_service_time` | 2.0 s | mean wall time of one in-flight request |
| `score_epsilon` | 0.05 s | below this spread, fall back (`CU-11`) |

`perToken = 1/2500 − 1/20000 = 4.0e-4 − 5.0e-5 = 3.5e-4 s/token`.

> ## CORRECTED (C1) — `queueing_penalty` is NOT the default
>
> The worked examples below are arithmetically right and **rejected as the default behaviour**. They show a backend holding an 8,000-token warm prefix losing to a cold idle one, with the crossover at `inflight < 1.4` — *one queued request is enough to abandon an 8k-token cache hit.*
>
> **FR-RTR-01 requires the opposite:** route to the node holding the KV slice *"even if that server is already heavily loaded."* This is a product whose purpose is to demonstrate that cache tiering beats recompute; a policy that routes away from cache hits under mild load would systematically hide the effect being sold.
>
> **`prefix-cache-aware` (the FR-RTR-01/02 policy) therefore uses a threshold spill guard, not a continuous penalty** — reusing v1's operator-facing model:
>
> ```
> spill_to_least_loaded  ⟺  (max_load − min_load) > balance_abs_threshold   // default 32
>                       AND   max_load > min_load × balance_rel_threshold   // default 1.5
> ```
>
> Residency wins outright below the threshold; above it, traffic spills to less-loaded nodes. This is *less* machinery than the scoring below, not more.
>
> **v1's guard must not be ported — it was broken four ways, all fixed here:**
>
> | v1 defect | Fix |
> |---|---|
> | Fed by the corrupt load counter (one increment site, three decrement sites, zeroed every 10 health cycles) — the guard evaluated **noise** | `Lease` (§B.3) is the only writer. This guard is the first consumer of a load signal that is actually correct. |
> | `max_load` computed over **all** workers incl. unhealthy, so one dead worker holding stale load latched the guard permanently ON and silently disabled cache routing forever (`CACHE-N4`) | Computed over **`candidates` only** (`LB-9`) |
> | `min_by_key` tie-break always returned index 0 → 32-deep thundering herd on a cold fleet (`CACHE-N5`) | Shared reservoir tie-break, uniform over the tied set (`LB-11`), chi-square tested |
> | Divergent defaults: `abs=32, rel=1.1` in the policy vs `abs=64, rel=1.5` on the CLI | One default set (`CFG-3`): `abs=32, rel=1.5`, reported in `/get_server_info` |
>
> **Bound on FR-RTR-01:** `max_inflight_per_worker` (`REL-10`) still applies, so a resident node at its hard cap sheds rather than queueing without limit. That is the *only* limit on "even if heavily loaded" and must be documented as such.
>
> The continuous scoring below **ships as the separate, experimental `cache-usefulness` policy**, where it is also the path that FR-RTR-04 (per-tier rates) would light up once a residency feed exists. Both policies share one engine — one trie, one extractor, one eviction model — per `OQ-3`.

**Worked numeric example (`cache-usefulness` only) — a warm-but-saturated backend loses to a cold idle one.**

| | A (warm, saturated) | B (cold, idle) |
|---|---|---|
| predicted cached tokens | 8,000 | 0 |
| in-flight / capacity | 6 / 1 → 6.0 | 0 / 1 → 0.0 |
| `predicted_time_saved` | 8,000 × 3.5e-4 = **+2.80 s** | 0 × 3.5e-4 = **+0.00 s** |
| `queueing_penalty` | −6.0 × 2.0 = **−12.00 s** | −0.0 × 2.0 = **−0.00 s** |
| **total** | **−9.20 s** | **0.00 s** |

B wins by 9.20 s. This is `CU-7` satisfied numerically, and it is the exact scenario `LB-N4`/`CACHE-N5` describes — v1's cache-aware policy would have piled all 8,000-token-warm traffic onto A.

The term set is not degenerate in the other direction. Same A, but with only 1 in flight:

| | A (warm, light) | B (cold, idle) |
|---|---|---|
| `predicted_time_saved` | +2.80 s | +0.00 s |
| `queueing_penalty` | −1 × 2.0 = −2.00 s | −0.00 s |
| **total** | **+0.80 s** | **0.00 s** |

A wins. The crossover is `8000 × 3.5e-4 > inflight × 2.0` ⟹ `inflight < 1.4` — i.e. an 8k-token warm prefix is worth exactly one queued request. That is a strong, checkable statement about the model, and it is the number to revisit when `CU-13` produces real predicted-vs-observed data.

`Select` computes `Query` for every candidate, sums the enabled terms, and picks the max. If `max − min < score_epsilon`, or all `total == 0`, or `Units == nil`, it delegates to `least-outstanding` and increments `router_policy_fallback_total{policy,reason}` (`CU-11`). A `clock`-based deadline check at 2 ms aborts to the same fallback (`CU-15`).

### D.7 OQ-2 — prefix-unit granularity: **1024 bytes, configurable**

vLLM hashes fixed 16-token blocks (~64 bytes); wekai uses 1024-byte windows (~256 est. tokens). **Decision: ship 1024 B as the default (`cache_unit_bytes`), expose 64 B as the fine setting, and resolve empirically with an offline sweep against `wekai router analyze` traces before GA.**

The rationale is a bounded-error argument, and it is why 1024 is defensible today. The engine's job is to *rank* backends, not to predict token counts. A 1024-byte unit rounds the true shared prefix down to the nearest unit boundary, so the per-request under-estimate is uniform on [0, 1024) bytes ≈ [0, 256) tokens, mean 128 tokens. Against a typical 8k-token prompt that is a **1.6% mean error, applied roughly equally to every candidate** — it shifts all scores by a similar amount and therefore almost never changes the arg-max. At 3.5e-4 s/token, 128 tokens is 45 ms of modelled saving against a `queueing_penalty` quantum of 2.0 s: the granularity error is 2% of one queue slot.

Cost of going to 64 B: 16× the nodes (2M-token budget → ~125k nodes ≈ 12 MiB/worker, which is why the node cap stays as a backstop), 16× the units per request (32 KiB prompt → 512 units), and Query at 64 × 512 ≈ 1 ms — right at the `NFR-2` ceiling. So finer granularity is *affordable but not free*, and unproven. Ship 1024; let the sweep move it.

### D.8 CU-2 — vendored copy with provenance and drift check

```
internal/cachetrie/
  wekai_provenance.go   // constants only, no logic
  trie.go               // our implementation; header cites the origin
  golden_test.go        // reproduces cacheEstimator.Observe on a fixed corpus
  testdata/wekai_cache_sim.go.golden   // byte copy of the upstream file
```

`wekai_provenance.go`:
```go
// Package cachetrie is derived from github.com/weka/wekai
//   benchmark/cache_sim.go @ <commit-sha>
// and                       benchmark/replay_router.go:731 (BuildReplayRequestPrefix).
// Cross-repo import of package benchmark is forbidden (CU-N5): it would pull
// the entire benchmark harness and its transitive deps into a production proxy.
const (
    WekaiRepo      = "github.com/weka/wekai"
    WekaiCommit    = "<sha>"
    WekaiCacheSim  = "benchmark/cache_sim.go"
    WekaiCacheSimSHA256 = "<sha256 of the upstream file at that commit>"
)
```

CI (`hack/check-wekai-drift.sh`, non-blocking-on-network, blocking-on-mismatch) fetches `benchmark/cache_sim.go` from wekai `main`, hashes it, and compares against `WekaiCacheSimSHA256`. On mismatch it fails with a diff against `testdata/wekai_cache_sim.go.golden` and instructions: review the upstream change, decide whether it affects prediction semantics, update the pin and the golden file in the same PR. The **behavioural** guarantee is separately held by `golden_test.go`, which asserts our `Query`/`Commit` reproduce `cacheEstimator.Observe`'s ratios on a fixed corpus — so a cosmetic upstream refactor fails the hash check (a review prompt) while a semantic one also fails the golden test (a real regression).

### D.9 Opaque-mode hierarchy (`HIER-12a`, `HIER-N4`)

A parent keeps **one trie per child router**, exactly as it does per leaf. That trie models the **union of the child's leaves**: the parent commits a request's units to child C's trie when it forwards to C, so the trie records "somewhere under C, this prefix has been seen".

**Budget bound.** The parent's modelled budget for C is
`maxTokens(C) = min(cfg.max_tokens_per_backend, Σ_{leaves ℓ under C} budget(ℓ))`
where the sum is reported by C in `node_state.subtree_cache_tokens` (bottom-up aggregation, §F.5) and degrades to `cfg.max_tokens_per_backend` if absent. Without this bound the parent would model an unboundedly large cache and predict hits that no single leaf can hold (`HIER-N4`).

**Quantified optimism error.** Let C have *k* healthy leaves. The parent's `Query` returns the tokens present in the *union*. The request will land on exactly one leaf, chosen by C's own policy. Two extremes:

- C runs a cache policy with perfect affinity: the prefix is on the leaf C picks, so the parent's prediction is **exact**.
- C runs `least-outstanding` or `random`: the prefix is on the chosen leaf with probability ≈ 1/k, so the parent's prediction is optimistic by a factor of up to **k**.

Hence `E[actual] ∈ [predicted/k, predicted]`, and the parent's prediction is optimistic by a factor in `[1, k]` — the width of the band is exactly the child's internal affinity quality.

**Mitigation shipped:** a `subtree_affinity` scalar per child (default **0.5**, configurable) multiplying the `predicted_time_saved` term for `kind=router` backends only. This is deliberately a knob, not a model: `CU-13`'s predicted-vs-observed histogram, labelled by `backend_kind`, measures the real factor in production. Per `OQ-12`, that same measurement is the instrument that decides whether bid mode (`HIER-12b`) is ever built — if `subtree_affinity` calibrates near 1.0 the opaque model is fine and bid mode is unnecessary; if it calibrates near `1/k` the opaque model is nearly worthless at depth and bid mode earns its 2 ms fan-out.

`HIER-14` is structural: the parent's `Commit` targets its trie for C, and C's `Commit` independently targets its trie for the chosen leaf. There is no cross-tier commit path in the code.

---

## E. Gateway

### E.1 Middleware chain — exact ordering, outermost first

| # | Middleware | Satisfies | Why here |
|---|---|---|---|
| 1 | `Recover` | `REL-9` | must wrap everything, including later middleware, to convert any panic to 500 + `router_panics_total` |
| 2 | `RequestID` | `GW-7`, `HIER-10`, `HIER-N5` | must precede logging so every line has it; adopts inbound id unchanged so a hop preserves the trace |
| 3 | `AccessLog` + `RequestMetrics` | `OBS-1`, `OBS-3`, `OBS-7`, `GW-N2`, `OBS-N3` | wraps *below* everything so 413s, 401s, 508s and CORS preflights are all logged and counted, including on the catch-all |
| 4 | `Trace` | `OBS-6` | nil-tracer branch, zero allocation when disabled |
| 5 | **`CORS`** | **`GW-10`, `GW-N3`, `AUTH-N5`, `SEC-10`** | **must be OUTSIDE auth**; preflight `OPTIONS` carries no credential and is answered and returned here |
| 6 | `BodyLimit` | `GW-8`, `GW-9`, `GW-N1` | installs `MaxBytesReader` on *every* path including the catch-all. Placed before auth deliberately: it only arms the reader, it reads nothing, so it cannot be used to buffer memory pre-auth |
| 7 | **`Auth`** | **`AUTH-1..AUTH-11`, `AUTH-N1`** | **the single enforcement site in the binary.** Handlers never re-check |
| 8 | `HierIngest` | `HIER-2`, `HIER-3`, `HIER-4`, `HIER-17` | 508 on self-in-`Via` or hops > max; clamps ctx deadline; clamps attempt budget |
| 9 | `ConcurrencyLimit` | `REL-10` | per-route-class inbound cap |
| 10 | `mux.ServeHTTP` | `GW-1..GW-5`, `GW-11`, `API-5` | matched pattern determines route class **and dialect** |

The catch-all (`OQ-1`, default off) is registered as the `"/"` pattern *inside* the same mux, so it is by construction downstream of all ten (`GW-9`). There is no `fallback` escape hatch as in v1.

Two separate `http.Server`s carry reduced chains: the metrics listener (`127.0.0.1:29000`, `GW-13`, not reachable from the inference mux because it is a distinct `ServeMux` on a distinct socket), and the optional admin listener (`GW-12`, default same-listener).

Auth details worth stating: `subtle.ConstantTimeCompare` on equal-length padded buffers (`AUTH-1`, `SEC-1`); allowlist matching on segment boundaries with an explicit trailing-`/` subtree rule, so `/v1/mod` never matches `/v1/models` (`AUTH-7`, `AUTH-N3`); **empty allowlist serves all paths, with auth still applied to all paths** — v1's semantics, kept because the allowlist gates reachability while auth gates access and the two are independent (`AUTH-8` withdrawn; see the requirements row); admin paths need no hard-coded set to satisfy `AUTH-11`, because the allowlist has no power to exempt anything from auth — listing an admin path makes it reachable but never unauthenticated, and leaving it unlisted 404s it; no credential material in any log field, not even a prefix (`AUTH-10`).

### E.2 Routing-text / prefix-unit extraction without deserialization

**Technique: a hand-rolled, allocation-free structural JSON scanner — `internal/jsonscan`, ~300 LOC. No third-party library.**

```go
// Scanner walks a top-level JSON object, returning raw byte spans for a
// small set of wanted keys and structurally skipping everything else.
// It never allocates, never unescapes, and never builds a typed model
// (GW-6, GW-N4, API-15, NG-4).
type Scanner struct{ b []byte; i int }

func New(b []byte) *Scanner

// Next returns the next top-level key and the raw span of its value.
// Value spans are sub-slices of b — no copy.
func (s *Scanner) Next() (key, value []byte, ok bool)

// Array iterates a value known to be an array, yielding element spans.
func (s *Scanner) Array(value []byte, fn func(elem []byte) bool)

// String decodes a JSON string span, using the escape-free fast path
// (the overwhelmingly common case) and only allocating when \\ is present.
func String(span []byte) (string, bool)
```

Skipping is depth-counted with string-and-escape awareness — the only real correctness hazard, and it is fuzzed. The dialect's `ExtractUnits` drives it: scan the top level for `messages`, `system`, `tools`, `stream`, `model`; iterate `messages` as an array; per element scan for `role` and `content`; hash the role and the raw content span directly. `stream` and `model` fall out of the same pass, so the whole of step 12 is **one linear pass over the buffered body with zero allocations**.

Alternatives rejected: `encoding/json.Decoder.Token()` (allocates an `interface{}` per token; ~10× slower on a 32 KiB body); `tidwall/gjson` (builds a `Result` tree and its path syntax invites exactly the schema-coupling creep that produced v1's 5,000-LOC `protocols/`); `buger/jsonparser` (thin maintenance, no fuzz corpus). A stdlib-only hand-rolled scanner is ~300 LOC we fully own, which is the whole point of the rewrite.

Correctness assurance: `FuzzScannerAgreesWithEncodingJSON` asserts that for any input, if `encoding/json.Valid`, then every key/value span the scanner reports round-trips identically through `encoding/json`. That gives us confidence without paying `encoding/json`'s cost at runtime.

### E.3 Streaming relay

`BufferPool` backed by `sync.Pool` of 64 KiB `[]byte` (`STR-1`, `NFR-5`: `concurrent_streams × 64 KiB`). `FlushInterval: -1` (`STR-5`). `Content-Type` untouched (`STR-2`). Non-2xx relayed as-is (`STR-3`).

Terminal detection is dialect-provided (`API-8`) and chunk-boundary correct (`STR-4`, `STR-N3`, `API-N3`):

```go
type lineScanner struct {
    carry  []byte           // partial trailing line, capped at 8 KiB
    match  []byte           // "data: [DONE]"  |  "event: message_stop"
    capped bool
}

func (s *lineScanner) Feed(p []byte) bool {
    for len(p) > 0 {
        j := bytes.IndexByte(p, '\n')
        if j < 0 {
            if len(s.carry)+len(p) <= maxCarry { s.carry = append(s.carry, p...) } else { s.capped = true }
            return false
        }
        line := p[:j]
        if len(s.carry) > 0 {
            s.carry = append(s.carry, line...)
            line = s.carry
        }
        if !s.capped && bytes.HasPrefix(bytes.TrimRight(line, "\r"), s.match) {
            s.carry = s.carry[:0]
            return true
        }
        s.carry = s.carry[:0]
        p = p[j+1:]
    }
    return false
}
```

No fixed trailing-byte window anywhere. The 8 KiB carry cap means a pathological unterminated line degrades to "terminal not seen" rather than growing without bound; the `STR-8` idle timeout and the end-of-body signal both still release the lease, so `API-N3`'s slow-leak failure mode cannot occur — the lease lifetime is bound to `ServeHTTP` returning, never to a terminal marker. Terminal detection is used only for metrics (`router_stream_completed_total{terminal}`) and abort classification, which is deliberately the weakest possible coupling.

`STR-6`/`STR-7`: upstream disconnect surfaces as a copy error → `router_stream_aborted_total{reason}` + WARN with request id and worker; client disconnect cancels the ctx, which cancels the round trip within the 100 ms budget. `STR-8` idle timeout via a `SetReadDeadline`-style timer reset per chunk on the wrapped body.

---

## F. Registry, health, discovery, hierarchy

### F.1 Copy-on-write mechanics

Writers (admin API, discovery reconcile, drain expiry) serialize on `Registry.mu`, build a new `Snapshot` (new sorted slice, new map, **shared `*Backend` pointers**) and publish with one `atomic.Pointer.Store`. Readers do one `Load`. `Apply` compares the desired set against the current one and returns `changed=false` — publishing nothing — when identical, which is what makes `SD-7` reconciliation idempotent and stops informer resyncs from churning snapshots.

Drain (`WRK-6`, `WRK-N3`): `Remove` sets `draining` (immediately excludes from candidate filtering), starts a `clock`-driven timer for `drain_deadline` (default 60 s), and unlinks the entry when `inflight == 0` or the deadline fires. Predictor teardown (`CACHE-10`, `CU-12`) runs on unlink via `Registry.hook`.

### F.2 Health checker

```go
type Checker struct {
    reg  *registry.Registry
    clk  clock.Clock         // AC-0.2 — no sleeps in tests
    sem  chan struct{}       // bounded pool, default 32
    cfg  Config              // Interval, Timeout (Timeout < Interval, HLT-2)
}

func (c *Checker) Run(ctx context.Context)  // one goroutine, ctx-owned (CACHE-8/NFR-9)
```

Each tick: snapshot, then for every backend with `Health == HealthActive` spawn a check bounded by `sem`; `HealthPassive` backends are **never probed** (`API-16`) and derive state solely from `circuit.Classify` on proxied outcomes (`HLT-12`). Round wall time is `ceil(N/32) × timeout` — 200 workers at a 5 s timeout completes in ≤ 35 s, and the AC's `< 6 s` is met at pool 32 by `ceil(200/32)=7`… so the pool default is raised to **64**, giving `ceil(200/64)=4 × 5 s = 20 s` worst case and ~5 s typical. I flag this: the `HLT-N1` acceptance test (`200 workers, 5 s timeouts, round < 6 s`) requires the pool to be ≥ 200 *or* checks to be non-blocking. **Design choice: pool size defaults to `min(256, max(32, N))`**, i.e. effectively unbounded-but-capped, which meets the AC and still bounds fd usage. Hysteresis N=3 fail / M=2 pass (`HLT-3`); new backends start `Unknown` (`HLT-5`, `SD-N3`).

The checker writes health and circuit state only. It **never** touches `inflight` (`HLT-4`, `HLT-N5`, `LB-6`), enforced by the CI grep fence.

For `kind=router` backends the same round also polls `GET /v1/internal/node_state` and stores `capacity` (§F.5).

### F.3 Discovery

`k8s.io/client-go` shared informer on `discovery.k8s.io/v1` `EndpointSlice` filtered by `kubernetes.io/service-name` (`SD-1`, default), plus an optional Pod-label informer (`SD-2`). Informers give watch-expiry re-list with backoff for free (`SD-3`); we cap the rate limiter at 5 min. `conditions.ready == false` → excluded (`SD-5`). IPv6 addresses bracketed by `net.JoinHostPort` (`SD-9`).

Each event triggers a **full reconcile**, not an incremental patch: build the complete desired discovered-set from the informer's store and hand it to `Registry.Apply`. Full reconcile is what makes double-application a no-op (`SD-7`, `SD-N2`) — there is no additive index to corrupt (`WRK-N2`).

`kind` and `dialect` come from labels/annotations `wllm.weka.io/backend-kind` and `wllm.weka.io/backend-dialect` (`HIER-16`, `API-6`), defaulting to `worker`/`openai`.

**Provenance merge (`HIER-19`, `OQ-10`, `WRK-7`)** inside `Apply`:

```
for each desired discovered endpoint d:
    if existing, ok := byURL[d.URL]; ok && existing.Prov == ProvStatic:
        metrics.DiscoveryConflicts.WithLabelValues(d.URL).Inc()
        slog.Warn("discovered endpoint collides with static backend; ignoring", ...)
        continue
    upsert(d)  // Prov = ProvDiscovered
for each existing e where e.Prov == ProvDiscovered && e.URL ∉ desired:
    drain(e)
// static entries are never added, updated, or removed by this pass
```

Removal is always via drain (`SD-6`). Discovery only *proposes*; health decides eligibility (`SD-4`). RBAC documented as `get/list/watch` on `endpointslices` (+ `pods` for SD-2) in the configured namespaces (`SD-8`).

### F.4 Hierarchical forwarding

Set in `Rewrite`, per hop:

| Header | Rule | Req |
|---|---|---|
| `X-Request-Id` | copied byte-identical, never re-minted | `GW-7`, `HIER-10`, `HIER-N5` |
| `traceparent` | propagated | `HIER-10`, `OBS-6` |
| `X-Wllm-Hops` | `n+1`; inbound `> max_hops` (default 4) → `508` | `HIER-2` |
| `X-Wllm-Via` | `inbound + "," + self_node_id`; self already present → `508` | `HIER-2` |
| `X-Wllm-Deadline-Ms` | `remaining − elapsed − per_hop_reserve`; a child may only shrink it | `HIER-3` |
| `X-Wllm-Attempts-Remaining` | `k−1`; this node makes at most `k` attempts | `HIER-4`, `HIER-N2` |
| `Authorization` / `X-Api-Key` | **deleted**, then the configured upstream credential injected | `AUTH-9`, `SEC-4`, `HIER-9`, `HIER-N3` |

Startup rejects any backend URL canonicalizing to the node's own advertised address, and warns when `max_hops × per_hop_reserve > global_deadline` (`HIER-17`).

`HIER-4` is what keeps a partial outage from becoming a self-inflicted DoS: with `max_attempts=2` at the root, the root forwards `Attempts-Remaining: 1`, the mid-tier makes exactly one attempt and forwards `0`, and the leaf tier makes one. Total upstream calls across a depth-3 tree is bounded by `max_attempts`, not `max_attempts³`.

### F.5 `node_state` and its O(edges) bound

```
GET /v1/internal/node_state →
{ "node_id","role","depth","healthy_leaves","subtree_capacity",
  "subtree_inflight","subtree_cache_tokens","dialects":["openai"],"ready":true }
```

Each node polls **only its own direct children of `kind=router`**, on the health interval, in the same bounded pool as active checks. `subtree_capacity` is computed bottom-up:

```
subtree_capacity(self) = Σ over healthy backends b:
    b.Kind == worker ? max_inflight_per_worker : b.reported_subtree_capacity
```

So the root learns the entire tree's capacity through a chain of local reports, and the total poll count per interval is exactly the number of router edges — **O(edges), never O(nodes²)** (`HIER-6`, `HIER-N6`). Absence, error, or timeout degrades that child to `{leaf, capacity 1}` and never fails a request. `dialects` feeds the `API-13` candidate filter. `readiness` fails at zero healthy backends so a parent drains a child through the ordinary `HLT-3` path with no special-casing (`HIER-7`, `HIER-8`).

---

## G. Configuration and observability

### G.1 Config

**Recommendation: one `Config` struct, stdlib `flag`, one ~200-LOC reflective loader, JSON/YAML file, no Viper/Koanf.**

```go
type Config struct {
    Listen        string        `flag:"listen" env:"WLLM_LISTEN" default:":8080"`
    MetricsListen string        `flag:"metrics-listen" env:"WLLM_METRICS_LISTEN" default:"127.0.0.1:29000"`
    MaxBodyBytes  int64         `flag:"max-body-bytes" env:"WLLM_MAX_BODY_BYTES" default:"67108864"`
    Policy        string        `flag:"policy" env:"WLLM_POLICY" default:"least-outstanding"`
    APIKeyFile    string        `flag:"api-key-file" env:"WLLM_API_KEY_FILE"`
    APIKey        string        `flag:"-" env:"WLLM_API_KEY" secret:"true"`   // CFG-8: no flag, ever
    Health        HealthConfig
    Circuit       CircuitConfig
    Cache         CacheConfig
    Hier          HierConfig
    Backends      []BackendConfig
}

func Default() Config          // THE single source of defaults (CFG-3, CFG-N3)
func Load(args []string, env func(string) string) (Config, error)
```

`Load` order, giving `CFG-1` precedence for free:
1. `Default()`.
2. Config file, unmarshalled with `DisallowUnknownFields` — an unknown key is a hard error (`CFG-6`).
3. One reflective walk of the struct: for each field with an `env` tag present in the environment, parse and set.
4. `flag.FlagSet.Parse(args)` last, with each flag bound to the field's address so an explicitly-set flag wins. `flag:"-"` fields register no flag at all — that is `CFG-8` enforced by construction, not by a check.

Validation runs after load and **aggregates** every problem into one `errors.Join` report (`CFG-5`): `timeout >= interval` (`HLT-2`), body limit 0 or unbounded, circuit window 0, unknown policy name with the valid list, and — the interesting one — **any knob set that the selected policy does not consume** (`LB-20`, `CFG-N4`, `LB-N5`). That last check is implemented by having each policy declare `ConsumedKnobs() []string` and comparing against the set of fields the loader observed as explicitly set (tracked in a `map[string]bool` during steps 2–4).

`CFG-9`/`CFG-N2`: there is no argv printing anywhere; a `hack/no_argv_dump_test.go` greps for `os.Args` outside `main`. `CFG-10`: `obs.Init` is the first call in `main`, before config file reads emit anything. `CFG-7`: `/get_server_info` marshals the effective config with `secret:"true"` fields replaced by `"[redacted]"` and the `coldTPS/warmTPS` values labelled `measured_on: <hardware>`.

Format: JSON natively; YAML via `sigs.k8s.io/yaml`, which client-go already pulls in, so it is a zero-cost dependency (`K`).

### G.2 Metrics and the dead-metric check

Every collector is declared in `internal/metrics/metrics.go` as an exported package var, registered in one `init`. That single file is the whole `OBS-3`/`OBS-4` set plus `router_retries_total`, `router_panics_total`, `router_discovery_conflicts_total`, `router_cache_eviction_anomalies_total`. Labels are closed enums (`route`, `status` bucketed, `policy`, `reason`, `worker`, `state`, `kind`, `dialect`) so cardinality is bounded (`API-14`).

`OBS-5` dead-metric check, as a Go test (no shell):

```go
func TestNoDeadMetrics(t *testing.T) {
    // 1. Parse internal/metrics with go/ast; collect exported var names.
    // 2. Parse every other package in the module; count identifier
    //    references of the form metrics.<Name>.
    // 3. Fail listing any metric with zero references outside its
    //    declaration file. (OBS-N1: v1 registered tokenizer metrics.)
}
```

### G.3 The `API-1` import fence

```go
func TestCorePackagesDoNotImportDialects(t *testing.T) {
    core := []string{"registry", "lease", "policy", "policy/cache",
                     "cachetrie", "proxy", "circuit", "health"}
    for _, p := range core {
        out, err := exec.Command("go", "list", "-deps",
            "github.com/weka/wllm-router/internal/"+p).Output()
        // fail if any dep matches internal/dialect/
    }
}
```

`go list -deps` is transitive, so this catches indirect leakage too. Paired with `TestSecondDialectRegistersWithoutCoreChanges`, which registers a `stubdialect` and drives a full request through the gateway (`API-3`).

---

## H. Testing strategy

| Package | Named tests |
|---|---|
| `testutil/mockvllm` | `AC-0.1` |
| `clock` | `AC-0.2` |
| `gateway` | `TestPreflightSucceedsWithAuthEnabled` (`GW-N3`), `TestCatchAllBodyLimitReturns413WithBoundedRSS` (`GW-N1`), `TestCatchAllRequestIsAccessLogged` (`GW-N2`), `TestAllowlistSegmentBoundary_V1ModVsV1Models` (`AUTH-N3`), `TestEmptyAllowlistServesAllPathsUnderAuth` (`AUTH-8` withdrawn), `TestAdminNotExemptibleByAllowlist` (`AUTH-11`), `TestWorkerReceivesNoClientCredential` (`AUTH-N4`), `TestExactlyOneAuthCallSite` (`AUTH-N1`), `TestSecondDialectRegistersWithoutCoreChanges` (`API-3`), `TestDialectMatrix_InboundXBackend` (`API-7`) |
| `jsonscan` | `FuzzScannerAgreesWithEncodingJSON`, `BenchmarkExtract32KiB` (0 allocs) |
| `registry` | `TestCanonicalizationTable` (`WRK-1`), `TestRegisterTwiceYieldsOneEntryInAllIndices` (`WRK-N2`), `TestSnapshotOrderStableOver1000Mutations` (`WRK-N1`), `TestStaticWinsOverDiscovered` (`HIER-19`), `TestDrainHoldsInFlightStreamToCompletion` (`WRK-6`) |
| `lease` | **`TestLeasePropertyAllCountersReturnToZero`** — see below (`LB-7`) |
| `circuit` | `TestSlidingWindowOpenTriggerUsesWindowDuration` (`HLT-N2`), `TestHalfOpenAdmitsExactlyMaxUnder100Probes` (`HLT-N3`), `TestClassify429IsFailure` (`HLT-N4`) |
| `health` | `Test200WorkersRoundCompletesUnder6s` (`HLT-N1`), `TestPassiveBackendIsNeverProbed` (`API-16`), `TestCheckerNeverWritesInflight` (`HLT-N5`) |
| `policy` | `TestRoundRobinExactlyTenEachOverTenN` (`LB-N3`), `TestRoundRobinShrinkThenGrowVisitsEveryMember` (`LB-14`), `TestTieBreakChiSquare100k` (`LB-N4`), `TestLeastOutstandingCapacityNormalized` (`HIER-N1`), `TestZeroOneTwoCandidates` (`LB-12`) |
| `cachetrie` | `TestGoldenReproducesWekaiObserve` (`CU-2`), `TestQueryIsPure1000Calls` (`CU-N2`), `TestPerWorkerIsolation` (`CU-N1`), `TestEvictionBoundsNodesAndTokens` (`CU-N3`), **`TestLRUTailIsAlwaysALeaf`** (the D.4 invariant), **`TestConcurrentInsertersAndEvictor`** — see below (`CACHE-N2`), `TestSkipTinyLeadingSystemBlock_AnthropicOnly` (`CU-3`, `API-11`), `BenchmarkQuery64Backends32KiBPrompt` (`NFR-2`, 0 allocs) |
| `proxy` | **`TestTerminalMarkerSplitAtEveryByteOffset`** — see below (`STR-N3`, `API-8`), `TestNon2xxOnStreamPassesContentTypeThrough` (`STR-N2`), `TestSlowClientBoundsRSS` (`STR-N1`), `TestNoRetryAfterFirstByte` (`REL-3`), `TestClientDisconnectCancelsUpstreamWithin100ms` (`GW-15`) |
| `discovery/k8s` | `TestApplySameEndpointSliceTwiceIsIdentical` (`SD-N2`), `TestNewEndpointNotRoutableUntilFirstHealthPass` (`SD-N3`), `TestIPv6URLBracketing` (`SD-9`) |
| `hier` (integration) | `TestThreeTierRequestIDConstantAcrossHops`, `TestConfiguredCycleReturns508` (`HIER-2`), **`TestRetryAmplificationBounded`** — see below (`HIER-N2`), `TestTrafficSpreadsProportionalToSubtreeCapacity` (chi-square), `TestKillMidTierRouterMidStream` |
| every package | `TestMain` with `goleak` (`NFR-9`); full suite under `-race` (`NFR-10`) |

**`AC-0.1` mock vLLM worker.** A `httptest.Server` driven by a `Script` value: `{TTFTDelay, InterTokenDelay, Tokens, Status, Body, ContentType, AbortAfterBytes, ResetAfterBytes, SlowBodyRate, Usage{PromptTokens, CachedTokens}}`, plus a `Behaviour(func(*http.Request) Script)` hook for per-request programming and a `Calls() []Call` recorder that captures headers (so `AUTH-N4`, `HIER-*` header assertions, and `COMPAT-2` byte-identical-body assertions are all one helper). Also serves `/health`, `/v1/models`, `/get_server_info`, and `/v1/internal/node_state` so it can impersonate a child router.

**`LB-7` lease property test.**
```go
func TestLeasePropertyAllCountersReturnToZero(t *testing.T) {
    // 8 backends, 10,000 lifecycles, pseudo-random seed reported on failure.
    // Each lifecycle picks uniformly from:
    //   normal completion | client cancel | upstream 503 → retry to a
    //   different backend | timeout | body-read error | stream abort |
    //   panic in the handler (recovered) | double Release() | Release()
    //   from two goroutines concurrently.
    // Runs at concurrency 64 under -race.
    // Assert: every backend's Inflight()==0 AND
    //         metrics.LoadAccountingErrors == 0.
}
```
The `double Release` and `concurrent Release` cases are the ones that would have caught `LB-N1` directly.

**`CACHE-N2` race test.** 8 goroutines committing random unit sequences + 1 goroutine forcing budget pressure (shrinking `maxTokens` repeatedly) + 4 goroutines running `Query`, for 60 s under `-race`, with an invariant checker that periodically takes the write lock and walks the whole trie asserting: every node reachable from the root is in the LRU list exactly once; `nodes` and `tokens` totals match the walk; every node with `nkids==0` is LRU-reachable; the LRU tail has `nkids==0`. Because eviction is inline under the write lock, a failure here means a genuine bug in `evictLocked`, not a design race.

**`API-8` every-byte-offset test.**
```go
func TestTerminalMarkerSplitAtEveryByteOffset(t *testing.T) {
    for _, marker := range []string{"data: [DONE]\n", "event: message_stop\n"} {
        stream := prelude + marker
        for split := 0; split <= len(stream); split++ {
            sc := dialectFor(marker).NewStreamScanner()
            got := sc.Feed([]byte(stream[:split])) || sc.Feed([]byte(stream[split:]))
            require.True(t, got, "marker missed with split at offset %d", split)
        }
    }
}
```
Plus a three-way-split variant and a `\r\n` variant.

**`HIER-N2` retry-amplification test.** Build root → 2 mid → 4 leaves each (13 mock nodes, all in-process). Configure every leaf to return `503`. Send one client request with `max_attempts = 2`. Assert the **sum** of `Calls()` across all 12 non-root nodes equals **2**, not 8, and that `X-Wllm-Attempts-Remaining` observed at each tier is strictly decreasing. Then assert the client sees a single `503` with the `all_attempts_failed` code rendered in the OpenAI envelope.

---

## I. Migration and rollout

**Phase 1 — offline replay.** Point `wekai router analyze` traces at an in-process harness that drives v2's policy layer with no network. Compare `predicted_cached_fraction` against the traces' `cache_read_tokens` ground truth. This is where `OQ-2` (unit granularity) and the `subtree_affinity` default are actually settled, and where `coldTPS`/`warmTPS` defaults for `OQ-6` are measured. Gate: correlation ≥ 0.7 (`AC` for cache-usefulness).

**Phase 2 — shadow (`CU-14`).** Deploy v2 alongside the Rust router with `--policy=least-outstanding --shadow-policy=cache-usefulness`. `cache-usefulness` computes its choice and emits `router_shadow_agreement_total{agreed}` and its own predicted-fraction histogram, but routing is `least-outstanding`. Zero risk, real traffic, and it validates the `CU-13` instrument that `OQ-12` depends on.

**Phase 3 — traffic replay.** Use wekai's replay tooling (`wekai router analyze` capture → `replay_router` post) against v2 with mock leaves at measured latencies, asserting `NFR-1`, `NFR-3`, `NFR-4` and, critically, `router_load_accounting_errors_total == 0` under sustained chaos. Also run the Rust router through the identical harness so the p99 TTFT comparison in `AC-R1` is apples-to-apples.

**Phase 4 — canary cutover.** 1% → 10% → 50% → 100% behind the existing ingress, ≥ 72 h at each of the last two steps (`AC-R1`). Rollback is a DNS/ingress weight change; v2 shares no state with v1, so rollback is instant and lossless except for the in-memory `{id}`→backend map (`GW-3`, an accepted `OQ-5` limitation).

**Metric-name migration (`OQ-9`, `COMPAT-4`).** Ship `--emit-legacy-metrics` (default **off**), dual-emitting the v1 names for exactly one release, with two exclusions: `vllm_router_worker_load`, `max_load` and `min_load` are **not** dual-emitted. Those gauges were fed from the corrupt counter (`OBS-N2`); re-exporting them under v2 would give a broken dashboard a plausible-looking green light. The release notes carry a full old→new mapping table and mark those three as "removed, were incorrect — replace panel with `router_worker_inflight`".

---

## J. Implementation plan

Ordered so a runnable, useful binary exists at the end of M2.

| M | Milestone | Packages | LOC |
|---|---|---|---|
| **M0** | Skeleton + test infra | `clock` 80, `metrics` 300, `obs` 250, `testutil/mockvllm` (tests, unbudgeted) | 630 |
| **M1** | **Load accounting first** — the central bug | `lease` 120, `registry` 500, `circuit` 220 | 840 |
| **M2** | **Runnable proxy.** Static backends, `least-outstanding`, no cache, no hierarchy. Deployable. | `config` 450, `gateway` 750, `proxy` 600, `policy` (iface + LO/RR/random) 400, `dialect` 150, `dialect/openai` 400, `health` 300, `respmap` 150 | 3,200 |
| **M3** | Discovery | `discovery/k8s` 550 | 550 |
| **M4** | Cache engine | `jsonscan` 300, `cachetrie` 500, `policy/cache` 250, `dialect/openai` extraction +150 | 1,200 |
| **M5** | Hierarchy | `hier` 350, `registry`/`policy` deltas 150 | 500 |
| **M6** | Hardening: soak, benchmarks, drift/fence/dead-metric CI, container | — | 100 |
| | | **Total non-test** | **7,020** |

Roughly 980 LOC of headroom under `NFR-11`'s 8,000. The headroom is deliberate and is the first thing spent if the retry glue in `proxy` or the `jsonscan` fuzz-hardening runs long.

**Explicit risks.**

1. **The `ReverseProxy` retry glue (§A.4) is the highest-risk 120 LOC in the design.** The `ModifyResponse → errRetryable → ErrorHandler → attemptWriter` dance depends on `ReverseProxy` internals that are documented but subtle. *Mitigation:* build it in M2 with the `REL-*` integration tests written first; if it proves fragile, the fallback is a hand-rolled copy loop for the retry-eligible path only (~200 LOC), keeping `ReverseProxy` for the streaming path. Decide by end of M2, not later.
2. **`NFR-2` p99 ≤ 1 ms for the cache policy at 64 backends is a real constraint, not a formality.** The §D.2 estimate is 61 µs with 16× headroom, but that assumes a 32-unit prompt. A 512 KiB prompt at 1024 B granularity is 512 units → ~1 ms. *Mitigation:* the `CU-15` 2 ms hard deadline with fallback is not optional, and `BenchmarkQuery` gates CI at 20% regression.
3. **The `CU-8` default (500k nodes) is unachievable within `NFR-5` at N=64** (§D.2). Requires a requirements amendment to 100k. Raise this in review before implementation, not during.
4. **The `HLT-N1` acceptance test (200 workers, 5 s timeout, round < 6 s) conflicts with a small bounded pool** (§F.2). Resolved by `pool = min(256, max(32, N))`, which is arguably no longer "bounded"; the real bound is fds, and 256 concurrent health checks is 256 fds. Flag for review.
5. **Opaque-mode hierarchy prediction quality is unmeasured** (§D.9). The `subtree_affinity=0.5` default is a guess. If `CU-13` shows it near `1/k`, cache-aware routing at depth is worth little and `OQ-12` should be revisited early rather than at GA.
6. **`jsonscan` correctness against adversarial JSON.** A structural-skip bug is a routing bug, potentially a cross-tenant affinity bug. *Mitigation:* the differential fuzz target against `encoding/json` runs in CI with a persisted corpus, not just locally.

---

## K. Dependency decisions

| Concern | Choice | Justification |
|---|---|---|
| HTTP routing | **stdlib `net/http.ServeMux`** (Go 1.22 method + `{id}` wildcard patterns) | The whole surface is ~25 static patterns plus four `/v1/responses/{id}` shapes — exactly what the new `ServeMux` handles. `chi` would add a dependency for middleware chaining we write in 40 LOC and `URLParam` we get from `r.PathValue`. Precedence rules are well-defined and the conflict panic at registration is a feature. **Rejected:** chi, gorilla/mux, gin. |
| Reverse proxy | **stdlib `net/http/httputil`** | §A.4. |
| Config | **stdlib `flag` + `encoding/json` + `sigs.k8s.io/yaml`** | ~200 LOC of reflective glue beats Viper's dependency tree (~40 modules), its silent-precedence surprises, and its case-insensitive key handling which fights `CFG-6`'s unknown-key rejection. `sigs.k8s.io/yaml` is already in the graph via client-go, so YAML is free. **Rejected:** viper, koanf, kong. |
| Kubernetes | **`k8s.io/client-go` + `k8s.io/api` + `k8s.io/apimachinery`** | No alternative. Shared informers give `SD-3`'s watch-expiry/re-list/backoff for free. This is by far the largest dependency (and the main driver of the ~40 MiB `NFR-8` image budget); it is compiled out of nothing — mitigate by keeping it behind `internal/discovery/k8s` so `SD-10` (static-only) is a runtime path, not a build tag. |
| Metrics | **`prometheus/client_golang`** | The de-facto standard; `promhttp` handler; no realistic stdlib alternative. Use `promauto` off — explicit registration in one file is what makes the `OBS-5` check possible. |
| Logging | **stdlib `log/slog`** with `JSONHandler` | Mandated by `OBS-1`. Structured, allocation-conscious, no dependency. **Rejected:** zap, zerolog. |
| Tracing | **`go.opentelemetry.io/otel` (trace API only), optional** | `OBS-6`. The API package alone is small; SDK and exporters are pulled only when tracing is compiled in. Zero-overhead-when-disabled via a nil tracer check, not a no-op tracer (which still allocates a span context). |
| Randomness | **stdlib `math/rand/v2`** | Per-P state, no global mutex, `IntN` without modulo bias. `LB-15`, `LB-11`. |
| Testing | **stdlib `testing`** + `go.uber.org/goleak` + `github.com/stretchr/testify/require` | `goleak` is mandated (`CACHE-8`, `NFR-9`). `testify/require` only — never `assert` — to keep failures fatal and the dependency shallow. Fuzzing and benchmarking are stdlib. Property tests are hand-rolled loops with reported seeds, not `gopter`/`rapid`: a seeded loop is 20 LOC and debuggable. |
| Hashing | **stdlib `crypto/sha256`** | Bit-identical to wekai (`CU-2` golden test), collision-resistant against prompt-crafted affinity theft, and SHA-NI-accelerated. |

Direct non-stdlib dependencies: **6** (client-go trio, prometheus, otel-api, goleak, testify, sigs.k8s.io/yaml). `govulncheck` in CI (`SEC-8`); distroless static base, non-root, read-only rootfs (`SEC-7`).

---

## L. OQ-1 … OQ-9

| OQ | Decision | Rationale |
|---|---|---|
| **OQ-1** — keep the catch-all? | **Keep, behind `--enable-passthrough`, default off**, registered as the `"/"` pattern inside the same mux. | Future vLLM endpoints work for free when opted into, and being a normal mux pattern makes `GW-9`'s "identical middleware chain" structural rather than a promise. |
| **OQ-2** — prefix-unit granularity | **1024 bytes default, `cache_unit_bytes` configurable, 64 B available; settled empirically in migration Phase 1.** | The engine ranks, it doesn't bill: the mean granularity error is ~128 tokens ≈ 45 ms of modelled saving against a 2.0 s queue quantum (§D.7). 64 B costs 16× nodes and puts Query at the `NFR-2` ceiling for unproven gain. |
| **OQ-3** — do the two cache policies merge? | **Yes — one engine, two scorer presets** (`API-17` term sets). Both names exposed in v2.0; decide before GA whether `prefix-cache-aware` survives as a public name. | They share the trie, the extractor, the eviction model and the Query/Commit split; only the scoring differs, and `API-17` already requires pluggable scoring. Merging saves ~400 LOC of the 8,000. |
| **OQ-4** — multi-model routing | **NG-6 stands.** But `Backend.Model` and a model-equality candidate filter are plumbed now as a no-op default. | Zero cost today; lifting NG-6 later becomes a config change rather than a refactor of the candidate filter. |
| **OQ-5** — Responses-API affinity durability | **Accept the limitation.** In-memory bounded LRU (100k / 1 h), `404` with a distinguishable code on miss, `--responses-affinity=memory\|off`. | Sticky-hash only works with a stable worker set (which K8s discovery breaks) and a shared store adds a dependency to a proxy for a feature nobody has confirmed is in use. Revisit if `/v1/responses` traffic actually appears in the access logs. |
| **OQ-6** — source of coldTPS/warmTPS/outputTPS | **Static config in v2.0**, values measured in Phase 1 and recorded in the release notes. Emit an observed-TTFT-vs-predicted-cached-tokens histogram so a continuous estimator has training data. | A feedback loop that steers routing from routing-influenced measurements can oscillate; ship the open loop, instrument for the closed one. |
| **OQ-7** — ship `power-of-two`? | **No.** | `least-outstanding`'s O(64) scan is ~200 ns against a 250 µs budget (§C.1). P2C exists to avoid an O(N) scan we can afford three orders of magnitude of, and it would strictly worsen selection quality. Rejecting it also removes the `LB-N5` knob-discard temptation. |
| **OQ-8** — per-worker KV capacity discovery | **Unresolved; must be verified against the deployed vLLM build before M4.** Design for it: `CU-9` reads `vllm:num_gpu_blocks` / `vllm:gpu_cache_usage_perc` from the worker's `/metrics` on the health interval when present, else uses the fixed budget. Behind `--cache-budget-from-worker`, default off. | The plumbing is ~40 LOC and the fallback is the shipped behaviour, so this is cheap to build speculatively and safe if the metrics turn out to be absent. |
| **OQ-9** — metric-name migration | **Dual-emit v1 names for one release behind `--emit-legacy-metrics` (default off), excluding `vllm_router_worker_load` / `max_load` / `min_load`.** | Those three were fed by the corrupt counter (`OBS-N2`); re-exporting them would keep a broken dashboard looking healthy against a router that finally has a correct signal. Release notes carry the full mapping table and mark them removed-because-wrong. |

---

### Critical Files for Implementation

- `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/cache_sim.go` — the engine being extracted; source of `prefixTrie`, `hashMessage`, `estimateTokens`, `promptChunkBytes` and the `CU-2` provenance pin.
- `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/replay_router.go` — `BuildReplayRequestPrefix` at line 731; the `i == 0 && sb.Bytes < 200` skip and the system→tools→messages ordering that §D.5 must reproduce.
- `/Users/ofer.kiselovnahman/workspace/wekai/benchmark/replay_router_post.go` — `dryRunDurations` (line 663) and `dryDo` (line 682); the cold/warm TTFT cost model behind §D.6's `predicted_time_saved` term.
- `/Users/ofer.kiselovnahman/workspace/wllm-router/src/server.rs` lines 815–910 — the authoritative wire surface and the v1 middleware ordering that §E.1 deliberately inverts (CORS/auth, and the `fallback` that bypasses the chain).
- `/private/tmp/claude-502/-Users-ofer-kiselovnahman-workspace-wllm-router/45cd3c2b-50d4-4b8d-9fa9-e44e82d8a3f8/scratchpad/requirements.md` and `/Users/ofer.kiselovnahman/.claude/plans/help-me-rewrite-vllm-router-silly-creek.md` — the binding requirement IDs cited throughout.

**Three items need a requirements amendment before implementation starts** (all flagged in §J risks): `CU-8`'s 500k-node default (unachievable within `NFR-5` at N=64 — propose 100k), the `HLT-N1` acceptance criterion vs. a small bounded health pool, and `STR-10`'s "where possible" becoming a firm commitment with the §A.4 retry-glue caveat recorded.