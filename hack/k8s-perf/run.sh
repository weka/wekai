#!/usr/bin/env bash
# Measures the ROUTER's own throughput ceiling, in a real Kubernetes cluster.
#
# Everything downstream is neutralised: eight vLLM stand-ins with every latency
# term set to zero and concurrency unbounded, so a backend costs nothing but
# parsing. What is left between a request and its response is the routing
# decision, the proxy hop and the accounting. The ceiling this finds is the
# router's.
#
# The load generator runs INSIDE the cluster. Driving it through `kubectl
# port-forward` measures the port-forward — one userspace TCP relay that
# saturates long before the router does.
#
# Concurrency is stepped up until throughput stops improving. That plateau is
# the answer: the point where adding clients only adds queueing.
set -euo pipefail

CLUSTER=${CLUSTER:-wekai-perf}
NS=${NS:-wekai-perf}
TAG=perf
KEEP=${KEEP:-0}
STEPS=${STEPS:-1,2,4,8,16,32,64,128,256,512}
STEP_DURATION=${STEP_DURATION:-10s}
SESSIONS=${SESSIONS:-64}
PREFIX_TOKENS=${PREFIX_TOKENS:-2048}
ROUTER_CPU=${ROUTER_CPU:-2}
# The load generator must never be the bottleneck, or the number this reports is
# the client's ceiling wearing the router's name.
LOADGEN_CPU=${LOADGEN_CPU:-2}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-${TMPDIR:-/tmp}/wekai-perf-result.json}

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
docker build -q -f "$ROOT/Dockerfile.loadgen"   -t "loadgen:$TAG"      "$ROOT" >/dev/null

echo "==> creating cluster"
k3d cluster delete "$CLUSTER" >/dev/null 2>&1 || true
k3d cluster create "$CLUSTER" --wait --timeout 180s >/dev/null

echo "==> importing images (no registry involved)"
k3d image import "wekai-router:$TAG" "mock-vllm:$TAG" "loadgen:$TAG" -c "$CLUSTER" >/dev/null

kubectl create namespace "$NS" >/dev/null
kubectl config set-context --current --namespace="$NS" >/dev/null

echo "==> deploying 8 zero-latency backends"
sed "s/mock-vllm:perf/mock-vllm:$TAG/" "$ROOT/hack/k8s-perf/mock-fleet.yaml" | kubectl apply -f - >/dev/null
kubectl rollout status statefulset/mock-vllm --timeout=180s >/dev/null

BACKENDS=()
for i in $(seq 0 7); do
  BACKENDS+=(--set "router.backends[$i]=http://mock-vllm-$i.mock-vllm:8000")
done

echo "==> installing the router"
# One replica: this measures a router, not a fleet of them. Affinity state is
# per-pod, so two replicas would also measure a different routing regime.
#
# maxNodeConcurrency matches the backends' real ceiling in a production fleet;
# here the backends never refuse, so it only gives the split guard a reference.
helm install router "$ROOT/chart/router" \
  --set imageRepository=wekai-router \
  --set imageTag="$TAG" \
  "${BACKENDS[@]}" \
  --set router.signals.maxNodeConcurrency=48 \
  --set service.port=8080 --set service.targetPort=8080 \
  --set replicaCount=1 \
  --set resources.requests.cpu="$ROUTER_CPU" \
  --set resources.requests.memory=1Gi >/dev/null
kubectl rollout status deployment/router --timeout=180s >/dev/null

echo "==> driving load until throughput plateaus"
kubectl run loadgen --image="loadgen:$TAG" --image-pull-policy=IfNotPresent --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"loadgen","image":"loadgen:'"$TAG"'","imagePullPolicy":"IfNotPresent","args":[
      "-target=http://router:8080",
      "-model=mock-vllm",
      "-concurrency-steps='"$STEPS"'",
      "-step-duration='"$STEP_DURATION"'",
      "-sessions='"$SESSIONS"'",
      "-prefix-tokens='"$PREFIX_TOKENS"'"],
      "resources":{"requests":{"cpu":"'"$LOADGEN_CPU"'","memory":"512Mi"}}}]}}' >/dev/null
kubectl wait --for=condition=Ready pod/loadgen --timeout=120s >/dev/null 2>&1 || true
kubectl logs -f pod/loadgen 2>/dev/null | tee /tmp/loadgen.out
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/loadgen --timeout=600s >/dev/null 2>&1 || true

echo
echo "==> what the router itself reported"
# By role rather than by name: app.kubernetes.io/name comes from the chart
# name, not the release name, so selecting on "router" silently matched nothing
# and the run died after producing its result.
ROUTER_POD=$(kubectl get pod -l role=router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
# The router's own view, which is the half a client cannot see: how long the
# routing decision took, and which tier of the ladder served each request.
# Never fatal: the measurement is already in hand by this point, and losing it
# to a label or a port change would be absurd.
METRICS=""
[ -n "$ROUTER_POD" ] && METRICS=$(kubectl exec "$ROUTER_POD" -- wget -qO- http://127.0.0.1:29000/metrics 2>/dev/null || true)
if [ -n "$METRICS" ]; then
  echo "$METRICS" | grep -E '^router_(route_decisions_total|cache_avg_copies|cache_splits_total|cache_guard_rejects_total|saturation_rejects_total|routing_decision_duration_seconds_(sum|count))'
  # The headline for this harness: how long the routing DECISION takes, which
  # is the only part of a request that is purely the router's own work. Client
  # latency includes the proxy hop, the backend and the network; this does not.
  echo "$METRICS" | awk '
    /^router_routing_decision_duration_seconds_sum/  { sum = $2 }
    /^router_routing_decision_duration_seconds_count/{ cnt = $2 }
    END {
      if (cnt > 0)
        printf "\nrouting decision mean   %.1f µs over %d decisions\n", (sum / cnt) * 1e6, cnt
    }'
else
  echo "(metrics endpoint not reachable; router.metricsListen may differ)"
fi

if grep -q '^RESULT ' /tmp/loadgen.out; then
  grep '^RESULT ' /tmp/loadgen.out | tail -1 | sed 's/^RESULT //' > "$OUT"
  echo
  echo "==> result written to $OUT"
  cat "$OUT"
else
  echo "!! load generator produced no result"
  exit 1
fi
