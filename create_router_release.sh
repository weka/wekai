#!/bin/bash
COMMIT_ID_OF_REPO="$(git rev-parse --short HEAD)"
BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD | tr '[:upper:]/' '[:lower:]-' | sed 's/[^a-z0-9_.-]/-/g')"
IMAGE_TAG="${BRANCH_NAME}-${COMMIT_ID_OF_REPO}"
docker buildx build --builder scalar \
    --platform linux/amd64 \
    --ssh default \
    -f Dockerfile.wllm-router -t quay.io/weka.io/vllm-router:${IMAGE_TAG} --push --progress=plain . && \
    echo "build successful, pull with quay.io/weka.io/vllm-router:${IMAGE_TAG}"


