# Deploying wllm-router

Two deliverables: a static Go binary and a Kubernetes-ready container image.

## Build

```bash
make -f Makefile.go.mk build          # ./wllm-router for the host platform
make -f Makefile.go.mk image          # container image for the host platform
make -f Makefile.go.mk image-multiarch  # linux/amd64 + linux/arm64, pushed
make -f Makefile.go.mk image-size     # size against the NFR-8 budget
make -f Makefile.go.mk verify         # lint + fences + full test suite
```

Image characteristics, as verified:

| Property | Value |
|---|---|
| Base | `gcr.io/distroless/static-debian12:nonroot` — no shell, no libc, no package manager |
| User | `nonroot` (uid 65532) |
| Read-only rootfs | compatible; the router writes nothing to disk |
| Compressed pull size | ~11 MiB |
| Uncompressed on disk | ~53 MB (see "Image size" below) |
| Binary | static, `CGO_ENABLED=0`, stripped |

### Image size

The uncompressed image exceeds `NFR-8`'s 40 MiB *should*-budget. Almost all of it
is `client-go`, which is irreducible if Kubernetes discovery is wanted. Two
reductions were already applied, taking the binary from 44.2 MB to 35.2 MB:

- A **targeted informer** rather than `informers.NewSharedInformerFactory`, which
  transitively imports a typed informer for every resource in the Kubernetes API
  (71 informer packages, 56 API-group packages measured).
- A **narrow client interface** with only `DiscoveryV1()` and `CoreV1()`, rather
  than `kubernetes.Interface`, whose implementation imports a typed client for
  every API group.

The compressed pull size — what actually costs anything at scale — is ~11 MiB.
Treat `NFR-8` as satisfied on pull size and knowingly exceeded on disk, or build
without discovery if the disk figure matters.

## Kubernetes

```bash
kubectl create namespace inference
kubectl -n inference create secret generic wllm-router-auth \
  --from-literal=api-key="$(openssl rand -hex 32)"
kubectl apply -f deploy/k8s/rbac.yaml
kubectl apply -f deploy/k8s/deployment.yaml
```

`deploy/k8s/deployment.yaml` ships a ConfigMap, Secret placeholder, Deployment,
Service and PodDisruptionBudget. `rbac.yaml` ships the ServiceAccount, Role and
RoleBinding.

### Probes

`GET /liveness`, `GET /readiness` and `GET /health` are reachable **without a
credential** by default, because a kubelet `httpGet` probe cannot authenticate —
presenting a credential would mean embedding the secret in the pod spec in
plaintext. This is deliberate and it is what makes `httpGet` probes usable at all:
v1 kept its health endpoints behind auth and its README therefore told operators
to switch to an `exec` probe, an instruction that cannot be followed on a
distroless image with no shell.

An unauthenticated probe is told only whether the router is ready. The healthy-
backend count requires a credential, so the fleet size is not disclosed. Set
`require_auth_for_probes: true` to lock probes down, accepting that you then need
probes that can authenticate.

`/readiness` returns `503` when no backend is healthy, so a router whose fleet has
gone away is removed from the Service rather than blackholing traffic.

### RBAC

Minimal and namespace-scoped — a `Role`, never a `ClusterRole`:

| Resource | Verbs | Needed for |
|---|---|---|
| `discovery.k8s.io/endpointslices` | get, list, watch | default discovery mode |
| `pods` | get, list, watch | only `--discovery-mode=pod`; delete otherwise |

### Graceful shutdown

On `SIGTERM` the router flips readiness, stops accepting new work, and lets
in-flight requests — including streams — finish up to `drain_deadline`. Verified
to exit 0.

`terminationGracePeriodSeconds` must exceed `drain_deadline` plus the `preStop`
sleep, or the kubelet will SIGKILL mid-generation. The manifest ships 90s against
a 60s drain and a 5s preStop.

The `preStop` sleep exists because endpoint removal is eventually consistent:
without it, the kubelet can begin shutdown before kube-proxy has stopped sending
new connections. It uses the `sleep` lifecycle action rather than `exec`, because
there is no shell in the image to exec.

### Discovery

Discovery only ever *proposes* backends. The registry decides admission and health
decides eligibility, so a freshly discovered pod is not routable until its first
successful health check — it cannot receive traffic before its model is loaded.

Workloads describe themselves with labels and annotations:

| Key | Purpose |
|---|---|
| `wllm.weka.io/backend-kind` | `worker` (default) or `router` |
| `wllm.weka.io/backend-dialect` | wire format; defaults to `openai` |
| `wllm.weka.io/model` | model identifier |
| `wllm.weka.io/capacity` (annotation) | concurrency denominator for load comparison |

A statically configured backend always wins over a discovered one with the same
URL; the collision is counted in `router_discovery_conflicts_total` and logged,
never silently merged.

## Configuration

Precedence: flag > environment > config file > default.

The API key is **never** settable by flag — a flag value is visible in `ps` to
anything on the host. Use `WLLM_API_KEY`, or `-api-key-file` with a mounted
Secret, which is what the manifest does so the key can be rotated without
changing the pod spec.

Unknown config keys are a hard startup error, and validation reports every
problem at once rather than failing on the first.

## Observability

Metrics are on a separate listener, default `127.0.0.1:29000`. **The manifest
overrides this to `0.0.0.0:29000`**, since loopback-only is unscrapable from
outside the pod. `/metrics` is deliberately not routable on the inference
listener.

The metric worth alerting on is `router_load_accounting_errors_total`. It must
stay at zero: any non-zero value means the in-flight accounting invariant is
broken, which is the defect this rewrite exists to fix.

Cache-affinity and load-balance quality are tracked by
`router_worker_load_{avg,max,min}` (fleet load spread across available
backends), `router_cache_prediction_{avg,max,min}` (predicted-hit-fraction
spread across a request's queried candidates), and
`router_route_decisions_total{decision="cache"|"load"|"other"}` (how
selections are actually being made). `router/deploy/watch-metrics.sh` polls
these from the command line; the "wllm-router — Cache & Load Routing" row in
`wllm`'s `grafana/dashboard.json` (the same dashboard the vLLM workers report
into) graphs them over time alongside worker-side stats.

## Local run

```bash
make -f Makefile.go.mk run   # against a backend at 127.0.0.1:8000
./wllm-router -version
```
