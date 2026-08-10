# wekai-router

Helm chart that deploys `wekai router serve` — the model-aware HTTP reverse
proxy for LLM traffic. Uses the replay-less `wekai-router` image: the same
binary as `wekai`, built without the embedded replay corpora, which keeps a
router pull small. `chart_test.go` guards that — pointing this chart back at the
benchmark image adds gigabytes to every pull.

## Quick install

```bash
helm upgrade --install wekai-router ./chart/router \
  -n default \
  --set ingress.enabled=true \
  --set domain=wekai-router.example.com \
  --set 'router.default=https://api.anthropic.com'
```

`ingress.className` defaults to empty so the cluster's default IngressClass
picks up the resource. Set it explicitly if you need a non-default controller:

```bash
  --set ingress.className=nginx
```

## Multi-route example

Routes use the syntax `<pattern> => <upstream>[ as <model>]`. Pattern is
comma-separated case-insensitive substrings of the model name, or `*` for
catch-all. First matching rule wins.

```bash
helm upgrade --install wekai-router ./chart/router \
  -n default \
  --set ingress.enabled=true \
  --set domain=wekai-router.example.com \
  --set 'router.routes[0]=claude => https://api.anthropic.com' \
  --set 'router.routes[1]=gpt,openai => https://api.openai.com' \
  --set 'router.default=https://api.anthropic.com'
```

When the `--set` form gets unwieldy, write a values file and pass `-f`:

```bash
helm upgrade --install wekai-router ./chart/router \
  -n default \
  -f my-router-values.yaml
```

## Capture to PVC

```bash
helm upgrade --install wekai-router ./chart/router \
  -n default \
  --set router.capture=redacted \
  --set datastore.sharedPvc.enabled=true
```

Captures land in `/data/router/capture/<mode>/` on the PVC and survive pod
restarts. Pair with `wekai router analyze` for offline analytics.

## Values

Key values (see `values.yaml` for the full list):

| Key | Default | Purpose |
|---|---|---|
| `imageRepository` | `quay.io/weka.io/wekai-router` | Replay-less router image |
| `imageTag` | `""` (falls back to `.Chart.AppVersion`) | Pin to a specific build |
| `replicaCount` | `1` | Cache-affinity state is per-pod and unshared; see `values.yaml` before scaling |
| `service.targetPort` | `25201` | Container listen port (`--listen`) |
| `router.routes` | `[]` | Repeated `--route` flags |
| `router.default` | `""` | Catch-all `--default` flag |
| `router.capture` | `""` | `""`, `"raw"`, or `"redacted"` |
| `ingress.enabled` | `false` | Expose externally |
| `ingress.className` | `""` | Leave empty to use the cluster default |
| `domain` | `""` | Ingress hostname (required when `ingress.enabled`) |
| `datastore.sharedPvc.enabled` | `false` | Mount a PVC at `/data` for capture output |
