# Deploying the router

The router is `wekai router serve` — one subcommand of the `wekai` binary, not a
separate program. It ships as its own container image and Helm chart, published
from the same commit and under the same semver as the benchmark pair on every
release.

## Image

`Dockerfile.router` builds the same `wekai` binary as the benchmark image, minus
the embedded replay artifact. That artifact is several GB and exists so a
benchmark pod can replay captured traffic without a volume; a router never opens
it, so shipping it would make every replica pay pull time and node disk for
nothing.

```bash
make router-image            # quay.io/weka.io/wekai-router:<version>
make router-image-multiarch  # linux/amd64 + linux/arm64, pushed
make verify                  # gofmt + vet + full suite under -race
```

## Kubernetes

Use the chart at [`chart/router`](../../chart/router). It carries the
Deployment, Service, PodMonitor, optional Ingress and a Grafana dashboard, and
creates discovery RBAC only when discovery is enabled — a namespace-scoped Role,
never a ClusterRole, with pod read granted only in pod mode.

```bash
helm upgrade --install my-router oci://quay.io/weka.io/helm/wekai-router \
  --version <release> \
  --set 'router.routes[0]=* => http://vllm-a:8000|http://vllm-b:8000'
```

The raw manifests that used to live here are gone: they configured the router
through a JSON config file that no longer exists, and the chart supersedes them.

## Local development

See [`.ainav/router-testing/`](../../.ainav/router-testing/index.md) for the
mock-fleet loop — a GPU-less vLLM fleet driven by a real captured-traffic
replay, which is how routing changes are evaluated without hardware.

`watch-metrics.sh` tails the router's own metrics listener during a run.
