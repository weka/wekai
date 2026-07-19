# syntax=docker/dockerfile:1
#
# wekai-core replay image: the wekai-core binary (benchmark/router/eval
# command trees) plus one embedded router-replay JSONL artifact, so a
# deployment can run `wekai-core benchmark auto --router-replay-file
# /wekai/replay.jsonl ...` with no external volume or fetch step.
#
# The replay artifact is published separately (see wekai's
# `.dagger` push_replay / `task benchmark:push-replay`) as a minimal scratch
# image under the historical `wekai-benchmark` quay repo — that repo name is
# where replay artifacts have always lived and is kept for that source
# reference only; nothing built or tagged from *this* Dockerfile uses
# "benchmark" in its name.

# ARGs used in a FROM must be declared before the first FROM (global scope).
# Override with --build-arg REPLAY_IMAGE=... to embed a different capture.
ARG REPLAY_IMAGE=quay.io/weka.io/wekai-benchmark:replay-099a98c60fd7

# ---- Builder: compile the wekai-core binary ----
FROM golang:1.25.7-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64
RUN go build -o /out/wekai-core .

# ---- Replay artifact source ----
FROM ${REPLAY_IMAGE} AS replay

# ---- Runtime ----
FROM alpine:latest
LABEL org.opencontainers.image.title="wekai-core" \
      org.opencontainers.image.description="LLM benchmarking, router/proxy, and capture-replay toolkit" \
      org.opencontainers.image.source="https://github.com/weka/wekai-core"
RUN apk add --no-cache ca-certificates
COPY --from=builder /out/wekai-core /usr/local/bin/wekai-core
COPY --from=replay /replay.jsonl /wekai/replay.jsonl
ENTRYPOINT ["/usr/local/bin/wekai-core"]
