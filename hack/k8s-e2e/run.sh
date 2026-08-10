#!/usr/bin/env bash
# End-to-end test of the SHIPPED artifacts in a real Kubernetes cluster.
#
# What this covers that nothing else does: the chart's own manifests, running
# the real image, probed by a real kubelet. Both bugs that reached a deployment
# — probes on paths the router does not serve, and capture with nowhere
# writable — were invisible to `go test` and to `helm template`, because
# rendering YAML does not tell you whether the pod it describes stays up.
#
# k3d (k3s in Docker) with `k3d image import`: no registry, so a locally built
# image is testable before it is pushed anywhere.
set -euo pipefail

CLUSTER=${CLUSTER:-wekai-e2e}
NS=${NS:-wekai-e2e}
TAG=e2e   # NOT :latest — that would default imagePullPolicy to Always and fail.
KEEP=${KEEP:-0}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for tool in k3d kubectl helm docker; do
  command -v "$tool" >/dev/null || { echo "need $tool on PATH"; exit 1; }
done

cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "==> KEEP=1, leaving cluster '$CLUSTER' up. Remove with: k3d cluster delete $CLUSTER"
    return
  fi
  echo "==> tearing down"
  k3d cluster delete "$CLUSTER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> building images"
docker build -q -f "$ROOT/Dockerfile.router"    -t "wekai-router:$TAG" "$ROOT" >/dev/null
docker build -q -f "$ROOT/Dockerfile.mock-vllm" -t "mock-vllm:$TAG"    "$ROOT" >/dev/null

echo "==> creating cluster"
k3d cluster delete "$CLUSTER" >/dev/null 2>&1 || true
k3d cluster create "$CLUSTER" --wait --timeout 120s >/dev/null

echo "==> importing images (no registry involved)"
k3d image import "wekai-router:$TAG" "mock-vllm:$TAG" -c "$CLUSTER" >/dev/null

kubectl create namespace "$NS" >/dev/null
kubectl config set-context --current --namespace="$NS" >/dev/null

echo "==> deploying the mock fleet"
kubectl apply -f "$ROOT/hack/k8s-e2e/mock-vllm.yaml" >/dev/null
kubectl rollout status statefulset/mock-vllm --timeout=120s 2>/dev/null

echo "==> helm install the router chart"
helm install router "$ROOT/chart/router" \
  --set imageRepository=wekai-router \
  --set imageTag="$TAG" \
  --set 'router.backends[0]=http://mock-vllm-0.mock-vllm:8000' \
  --set 'router.backends[1]=http://mock-vllm-1.mock-vllm:8000' \
  --set router.signals.maxNodeConcurrency=32 \
  --set service.port=8080 --set service.targetPort=8080 \
  --set replicaCount=1 >/dev/null

# The real assertion: does the kubelet's own probe pass? A router whose probes
# point at paths it does not serve CrashLoopBackOffs here and nowhere else.
echo "==> waiting for the router to become Ready (this is the probe test)"
if ! kubectl rollout status deploy/router --timeout=120s 2>/dev/null; then
  echo "!! router never became ready"
  kubectl describe deploy/router | tail -30
  kubectl logs deploy/router --tail=50 || true
  exit 1
fi

# Driven from inside the router pod with busybox wget. Deliberately not a
# curl image: pulling one needs a registry, and the point of this test is that
# it runs with nothing but locally built images.
echo "==> routing a request through it"
BODY='{"model":"e2e-model","max_tokens":8,"messages":[{"role":"user","content":"hello from k8s"}]}'
if ! kubectl exec deploy/router -- wget -q -O- \
      --header='Content-Type: application/json' \
      --post-data="$BODY" \
      http://127.0.0.1:8080/v1/chat/completions > /tmp/e2e-body 2>/tmp/e2e-err; then
  echo "!! request failed"; cat /tmp/e2e-err; kubectl logs deploy/router --tail=40; exit 1
fi
grep -q 'chat.completion' /tmp/e2e-body || {
  echo "!! response is not a completion:"; head -c 300 /tmp/e2e-body; exit 1; }

echo "==> checking the router's own metrics"
kubectl exec deploy/router -- wget -q -O- http://127.0.0.1:29000/metrics > /tmp/e2e-metrics 2>/dev/null || {
  echo "!! metrics listener unreachable"; exit 1; }
grep -q 'router_route_decisions_total' /tmp/e2e-metrics || {
  echo "!! router_route_decisions_total absent"; head -20 /tmp/e2e-metrics; exit 1; }

# Second install: the same fleet, found by LABEL rather than listed. This is the
# case a Service cannot express — each pod contributes its own containerPort —
# and it exercises the discovery RBAC, which only a real cluster can check.
echo "==> helm install a second router using pod discovery"
helm install router-disc "$ROOT/chart/router" \
  --set imageRepository=wekai-router \
  --set imageTag="$TAG" \
  --set-string 'router.routes[0]=* => pods:app=mock-vllm' \
  --set router.discovery.portName=http \
  --set discovery.enabled=true \
  --set service.port=8080 --set service.targetPort=8080 \
  --set replicaCount=1 >/dev/null

if ! kubectl rollout status deploy/router-disc --timeout=120s 2>/dev/null; then
  echo "!! discovery router never became ready"
  kubectl logs deploy/router-disc --tail=40 || true
  exit 1
fi

echo "==> routing through the discovered pool"
if ! kubectl exec deploy/router-disc -- wget -q -O- \
      --header='Content-Type: application/json' \
      --post-data="$BODY" \
      http://127.0.0.1:8080/v1/chat/completions > /tmp/e2e-disc 2>/tmp/e2e-disc-err; then
  echo "!! discovered-pool request failed"; cat /tmp/e2e-disc-err
  kubectl logs deploy/router-disc --tail=40; exit 1
fi
grep -q 'chat.completion' /tmp/e2e-disc || {
  echo "!! discovered pool did not serve a completion:"; head -c 300 /tmp/e2e-disc; exit 1; }

# Both pods must be found, on the port they declare rather than a default.
kubectl exec deploy/router-disc -- wget -q -O- http://127.0.0.1:29000/metrics > /tmp/e2e-disc-metrics 2>/dev/null || true
found=$(grep -c '^router_backend_inflight{' /tmp/e2e-disc-metrics || true)
if [ "${found:-0}" -lt 2 ]; then
  echo "!! discovery found $found backends, want 2 (one per mock pod)"
  kubectl logs deploy/router-disc --tail=30; exit 1
fi

echo
echo "PASS — chart deployed, probes passed, request routed, metrics served."
echo "PASS — pod-label discovery found $found backends and served through them."
grep -E 'router_route_decisions_total\{' /tmp/e2e-metrics | head -3
