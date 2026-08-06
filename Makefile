# wekai — build, test and verification targets.
#
# CLAUDE.md documents `task build` / `task test`, but no Taskfile exists in the
# repo, so those commands do not run. This Makefile provides the real entry
# points; `make verify` is what CI gates on.
#
# Publishing stays with Dagger (.dagger/): `task replay:push` and friends are
# likewise aspirational — see .github/workflows/release.yml for what actually
# runs on a release.

GO         ?= go
BINARY     := wekai
ROUTER     := wllm-router
ROUTER_PKG := ./router/cmd/wllm-router

IMAGE_REPO ?= quay.io/weka.io/vllm-router
VERSION    ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo none)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
PLATFORMS  ?= linux/amd64,linux/arm64

# Directories holding first-party Go source. vendor/ is excluded deliberately:
# it is generated, not committed, and gofmt has no business rewriting it.
GO_DIRS := benchmark chart cli config kvcache llm router tools

LDFLAGS := -s -w \
	-X main.version=$(VERSION) \
	-X main.commit=$(COMMIT) \
	-X main.date=$(BUILD_DATE)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- verification --------------------------------------------------------------

.PHONY: verify
verify: fmt-check vet test-race ## Everything CI gates on

.PHONY: fmt-check
fmt-check: ## Fail if any first-party Go file is not gofmt'd
	@out=$$(gofmt -l $(GO_DIRS)); \
		if [ -n "$$out" ]; then \
			echo "not gofmt'd:"; echo "$$out"; \
			echo "run: make fmt"; exit 1; \
		fi

.PHONY: fmt
fmt: ## Format first-party Go source in place
	gofmt -w $(GO_DIRS)

.PHONY: vet
vet: ## go vet the whole module
	$(GO) vet ./...

.PHONY: test
test: ## Run the test suite (matches what release.yml gates on)
	$(GO) test ./... -timeout 300s

.PHONY: test-race
test-race: ## Run the test suite with the race detector
	$(GO) test -race -count=1 ./... -timeout 600s

# The router's invariant fences are ordinary tests under router/hack, so `make
# test` already runs them. This target exists to run them alone when iterating:
# they assert things like "only internal/lease may mutate in-flight load" and
# "no core package imports a dialect", which are cheap greps that would
# otherwise drift into comments nobody enforces.
.PHONY: fences
fences: ## Run only the router's invariant fences
	$(GO) test -count=1 ./router/hack/

.PHONY: fuzz
fuzz: ## Fuzz the router's JSON scanner (FUZZTIME=60s)
	$(GO) test ./router/internal/jsonscan/ -run=NONE \
		-fuzz=FuzzFieldsAgreesWithEncodingJSON -fuzztime=$(or $(FUZZTIME),60s)
	$(GO) test ./router/internal/jsonscan/ -run=NONE \
		-fuzz=FuzzArrayTerminates -fuzztime=$(or $(FUZZTIME),60s)

.PHONY: bench
bench: ## Run benchmarks
	$(GO) test -run=NONE -bench=. -benchmem ./kvcache/ ./router/...

.PHONY: vulncheck
vulncheck: ## Scan dependencies for known vulnerabilities
	$(GO) run golang.org/x/vuln/cmd/govulncheck@latest ./...

# --- builds --------------------------------------------------------------------

.PHONY: build
build: ## Build the wekai CLI
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags="$(LDFLAGS)" -o $(BINARY) .

.PHONY: router-build
router-build: ## Build the router binary
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags="$(LDFLAGS)" -o $(ROUTER) $(ROUTER_PKG)

.PHONY: router-image
router-image: ## Build the router container image
	docker build -f Dockerfile.wllm-router \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(IMAGE_REPO):$(VERSION) .

.PHONY: router-image-multiarch
router-image-multiarch: ## Build and push a multi-arch router image
	docker buildx build -f Dockerfile.wllm-router --platform $(PLATFORMS) \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(IMAGE_REPO):$(VERSION) --push .

.PHONY: router-image-smoke
router-image-smoke: router-image ## Build the image and check the binary reports its version
	docker run --rm $(IMAGE_REPO):$(VERSION) -version

.PHONY: router-run
router-run: router-build ## Run the router locally against a backend at :8000
	./$(ROUTER) -listen 127.0.0.1:8080 -metrics-listen 127.0.0.1:29000 \
		-backends http://127.0.0.1:8000 -log-format text

.PHONY: router-deploy
router-deploy: ## Apply the router's Kubernetes manifests
	kubectl apply -f router/deploy/k8s/rbac.yaml
	kubectl apply -f router/deploy/k8s/deployment.yaml

.PHONY: router-manifests-validate
router-manifests-validate: ## Server-side dry-run of the router manifests
	kubectl apply --dry-run=server -f router/deploy/k8s/rbac.yaml
	kubectl apply --dry-run=server -f router/deploy/k8s/deployment.yaml

.PHONY: router-image-size
router-image-size: ## Report router image size, uncompressed and compressed
	@# `docker image inspect --format {{.Size}}` is NOT used: on a manifest-list
	@# image (buildx default, with an attestation manifest) it reports a
	@# misleading figure — 11 MiB for an image that is 53 MB on disk.
	@printf 'uncompressed on disk : %s\n' \
		"$$(docker images $(IMAGE_REPO):$(VERSION) --format '{{.Size}}')"
	@printf 'compressed pull size : %.1f MiB\n' \
		"$$(docker save $(IMAGE_REPO):$(VERSION) | wc -c | awk '{print $$1/1048576}')"

.PHONY: clean
clean: ## Remove build artifacts
	rm -f $(BINARY) $(ROUTER)
	$(GO) clean -testcache
