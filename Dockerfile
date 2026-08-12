# syntax=docker/dockerfile:1.26@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
#
# wekai replay image: the wekai binary (benchmark/router/eval
# command trees) plus one embedded router-replay JSONL artifact, so a
# deployment can run `wekai benchmark auto --router-replay-file
# /wekai/replay.jsonl ...` with no external volume or fetch step.
#
# The replay artifact is published separately (`task replay:push` /
# `dagger call push-replay`) as a minimal scratch image in the same wekai
# quay repo, distinguished by the replay- tag prefix.

# ARGs used in a FROM must be declared before the first FROM (global scope).
# Override with --build-arg REPLAY_IMAGE=... to embed a different capture.
ARG REPLAY_IMAGE=quay.io/weka.io/wekai:replay-24e7f15ba0ea

# ---- Builder: compile the wekai binary ----
FROM golang:1.25.7-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64
RUN go build -o /out/wekai .

# ---- Replay artifact source ----
FROM ${REPLAY_IMAGE} AS replay

# ---- Runtime ----
FROM alpine:latest
LABEL org.opencontainers.image.title="wekai" \
      org.opencontainers.image.description="LLM benchmarking, router/proxy, and capture-replay toolkit" \
      org.opencontainers.image.source="https://github.com/weka/wekai"
RUN apk add --no-cache ca-certificates
# --link: each COPY becomes an independent, content-addressed layer that
# doesn't depend on prior layer state (both destinations are plain
# directories with no special ownership/permissions set up by a prior RUN,
# so this is safe under BuildKit). This matters most for the replay layer,
# which can be several GB: with --link, rebuilding the builder stage (a Go
# source change) does not invalidate or recopy the replay layer — its
# digest is stable across rebuilds, so registries can reuse/cross-mount the
# existing blob instead of re-uploading it.
COPY --link --from=builder /out/wekai /usr/local/bin/wekai
COPY --link --from=replay /replay.jsonl /wekai/replay.jsonl
ENTRYPOINT ["/usr/local/bin/wekai"]
