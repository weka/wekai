# Go build targets for the v2 router. Included by the top-level Makefile, which
# still drives the Rust crate during the transition.
#
#   make -f router/Makefile.mk help
#
GO           ?= go
BINARY       := wllm-router
CMD          := ./cmd/wllm-router
IMAGE_REPO   ?= ghcr.io/weka/wllm-router
VERSION      ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT       ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo none)
BUILD_DATE   ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
PLATFORMS    ?= linux/amd64,linux/arm64

LDFLAGS := -s -w \
	-X main.version=$(VERSION) \
	-X main.commit=$(COMMIT) \
	-X main.date=$(BUILD_DATE)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the binary for the host platform
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags="$(LDFLAGS)" -o $(BINARY) $(CMD)

.PHONY: test
test: ## Run the full test suite with the race detector
	$(GO) test -race -count=1 ./...

.PHONY: test-short
test-short: ## Run tests without the race detector
	$(GO) test -count=1 ./...

.PHONY: fuzz
fuzz: ## Fuzz the JSON scanner (FUZZTIME=60s)
	$(GO) test ./internal/jsonscan/ -run=NONE \
		-fuzz=FuzzFieldsAgreesWithEncodingJSON -fuzztime=$(or $(FUZZTIME),60s)

.PHONY: bench
bench: ## Run benchmarks
	$(GO) test -run=NONE -bench=. -benchmem ./...

.PHONY: lint
lint: ## Vet, gofmt check, and the invariant fences
	$(GO) vet ./...
	@out=$$(gofmt -l internal cmd hack); \
		if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	$(GO) test -count=1 ./hack/

.PHONY: vulncheck
vulncheck: ## Scan dependencies for known vulnerabilities (SEC-8)
	$(GO) run golang.org/x/vuln/cmd/govulncheck@latest ./...

.PHONY: verify
verify: lint test ## Everything CI should gate on

.PHONY: image
image: ## Build the container image for the host platform
	docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(IMAGE_REPO):$(VERSION) -t $(IMAGE_REPO):latest .

.PHONY: image-multiarch
image-multiarch: ## Build and push a multi-arch image (requires buildx + login)
	docker buildx build --platform $(PLATFORMS) \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(IMAGE_REPO):$(VERSION) --push .

.PHONY: image-smoke
image-smoke: image ## Build the image and check the binary reports its version
	docker run --rm $(IMAGE_REPO):$(VERSION) -version

.PHONY: image-size
image-size: ## Report image size, uncompressed and compressed
	@# `docker image inspect --format {{.Size}}` is NOT used here: on a
	@# manifest-list image (which buildx produces by default, with an attestation
	@# manifest) it reports a misleading figure — 11 MiB for an image that is
	@# 53 MB on disk. `docker images` reports the honest uncompressed size.
	@printf 'uncompressed on disk : %s\n' \
		"$$(docker images $(IMAGE_REPO):$(VERSION) --format '{{.Size}}')"
	@printf 'compressed pull size : %.1f MiB\n' \
		"$$(docker save $(IMAGE_REPO):$(VERSION) | wc -c | awk '{print $$1/1048576}')"
	@printf 'NFR-8 budget         : 40 MiB (SHOULD) — see docs/rewrite/plan.md\n'

.PHONY: deploy
deploy: ## Apply the Kubernetes manifests
	kubectl apply -f deploy/k8s/rbac.yaml
	kubectl apply -f deploy/k8s/deployment.yaml

.PHONY: manifests-validate
manifests-validate: ## Server-side dry-run of the manifests
	kubectl apply --dry-run=server -f deploy/k8s/rbac.yaml
	kubectl apply --dry-run=server -f deploy/k8s/deployment.yaml

.PHONY: run
run: build ## Run locally against a backend at :8000
	./$(BINARY) -listen 127.0.0.1:8080 -metrics-listen 127.0.0.1:29000 \
		-backends http://127.0.0.1:8000 -log-format text

.PHONY: clean
clean: ## Remove build artifacts
	rm -f $(BINARY)
	$(GO) clean -testcache
