// Command mock-vllm is a standalone, GPU-less stand-in for a vLLM OpenAI
// server. It models prefix-cache behavior (chained block hashing, LRU
// eviction of unpinned blocks, and in-flight-request pinning, via
// github.com/weka/wekai/kvcache — the same engine the router's
// prefix-cache-aware policy uses to predict residency), per-instance
// concurrency admission (HTTP 429 past --max-concurrency), and a token-rate
// latency model, so router routing/cache policies can be developed and
// evaluated against a fleet of these without any real GPUs or vLLM workers.
//
// The latency knobs are rates (tokens/sec), not per-token durations, on
// purpose: the intended workflow is running a real vLLM fleet through the
// router once, reading its actual prefill/decode throughput and its
// per-instance num_gpu_blocks, then setting THOSE numbers here (via
// --cold-input-tps / --cached-input-tps / --output-tps / --block-capacity)
// so a run against this fleet is calibrated against the real one it's
// standing in for, rather than an arbitrary latency shape.
//
// Two more calibration knobs close fidelity gaps found comparing a mock run
// against a real fleet: --chars-per-token (the mock's own byte-to-token
// ratio for every conversion it does, independent of kvcache's fixed 4.0
// used elsewhere in this module — real vLLM's actual tokenizer runs closer
// to 2.9-3.4 chars/token on dense agentic content) and
// --output-kv-multiplier (real vLLM writes decode KV into the same pool as
// prompt KV; this mock ignored that until now — a completed request appends
// ceil(output_tokens*multiplier/block-size-tokens) blocks to its own chain,
// occupying capacity and evictable exactly like prompt blocks).
//
// Run several independent instances (each with its own cache/counters) to
// form a fleet, either as separate processes on separate ports:
//
//	go run ./router/cmd/mock-vllm --port 9001 &
//	go run ./router/cmd/mock-vllm --port 9002 &
//	go run ./router/cmd/mock-vllm --port 9003 &
//	go run ./router/cmd/mock-vllm --port 9004 &
//
// or as one process serving all of them (identical behavior, fewer things to
// manage — every instance still gets its own Engine, so they never share
// cache state):
//
//	go run ./router/cmd/mock-vllm --instances 4 --base-port 9001
//
// Point the router's --backends at http://localhost:9001|..|9004 and
// its cache-aware policy, retry logic, and circuit breaker all exercise
// exactly as they would against real workers.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/weka/wekai/router/internal/mockvllm"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("mock-vllm", flag.ContinueOnError)
	port := fs.Int("port", 9000, "listen port (ignored, -base-port used instead, when -instances > 1)")
	instances := fs.Int("instances", 1, "number of independent instances to run in this process, each with its own cache/counters, on consecutive ports starting at -base-port")
	basePort := fs.Int("base-port", 9000, "starting port when -instances > 1")
	modelID := fs.String("model-id", "mock-vllm", "model id reported by /v1/models and echoed in responses (same across all instances, matching a real fleet serving one model)")
	blockSizeTokens := fs.Int("block-size-tokens", 16, "cache block size in tokens (vLLM's block_size analog; converted to bytes at -chars-per-token)")
	charsPerToken := fs.Float64("chars-per-token", 4.0, "bytes-to-tokens ratio used EVERYWHERE this engine converts content bytes to tokens: block segmentation, usage.prompt_tokens/cached_tokens, and (through those) the latency model's token counts. Real vLLM's actual tokenizer runs closer to 2.9-3.4 for dense agentic content; 4.0 matches this repo's historical flat estimate")
	blockCapacity := fs.Int64("block-capacity", 100_000, "cache capacity in blocks; 0 = unbounded (never evicts). Set to match the real fleet's reported num_gpu_blocks per instance for a calibrated comparison")
	maxConcurrency := fs.Int("max-concurrency", 64, "requests admitted at once before returning HTTP 429; 0 = unbounded")
	defaultMaxTokens := fs.Int("default-max-tokens", 128, "completion length used when a request omits max_tokens")
	baseLatency := fs.Duration("base-latency", 20*time.Millisecond, "fixed latency added to every request's TTFT, on top of the token-rate terms below")
	coldInputTPS := fs.Float64("cold-input-tps", 50_000, "tokens/sec for UNCACHED prompt tokens (prefill). Set from the real fleet's measured/estimated prefill throughput")
	cachedInputTPS := fs.Float64("cached-input-tps", 1_000_000, "tokens/sec for CACHED prompt tokens (cache read) — NOT free: only recompute is skipped, a cache hit still costs a KV read")
	outputTPS := fs.Float64("output-tps", 500, "tokens/sec decode, per request — also paces SSE chunk spacing")
	outputKVMultiplier := fs.Float64("output-kv-multiplier", 1.0, "models real vLLM writing decode KV into the same pool as prompt KV: on completion, ceil(output_tokens*multiplier/block-size-tokens) blocks are appended to the request's own chain, occupying capacity and evictable like prompt blocks. 0 disables this (outputs stay invisible to the cache, the historical behavior)")
	externalKVTPS := fs.Float64("external-kv-tps", 0, "tokens/sec for prompt tokens read from a FLEET-SHARED KV tier (LMCache over WEKA), and the switch that creates one: 0 (the default) gives independent per-instance caches, where prefilling on one instance does nothing for another. Set it and every instance in this process reads and writes one shared trie, so a block computed anywhere is loadable everywhere at this rate — between --cached-input-tps (local HBM) and --cold-input-tps (recompute). This is what the router's --prefill-split depends on, and turning it off is how you measure that dependency")
	logLevel := fs.String("log-level", "info", "debug, info, warn, or error")
	surface := fs.String("surface", "vllm", "which HTTP surface to expose: 'vllm' (OpenAI routes + /health + /metrics), "+
		"'anthropic' (a vLLM fronted with /v1/messages, still serving vllm: metrics — the case that must not be "+
		"misclassified as a hosted API), or 'hosted' (messages only: no /metrics, /v1/models or /health, so the "+
		"router must fall back to passive health)")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *instances < 1 {
		return fmt.Errorf("-instances must be >= 1, got %d", *instances)
	}

	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(*logLevel)); err != nil {
		return fmt.Errorf("invalid -log-level %q: %w", *logLevel, err)
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))

	// One Tier for the whole process, or none at all. Shared across instances is
	// the entire point — a per-instance "shared" tier would be an ordinary local
	// cache wearing a different name — which also means this flag only models a
	// shared tier for the instances THIS process runs. A fleet split across
	// several mock-vllm processes gets one tier per process.
	var tier *mockvllm.Tier
	if *externalKVTPS > 0 {
		tier = mockvllm.NewTier()
	}

	cfg := mockvllm.Config{
		ModelID:            *modelID,
		Tier:               tier,
		ExternalInputTPS:   *externalKVTPS,
		BlockSizeTokens:    *blockSizeTokens,
		CharsPerToken:      *charsPerToken,
		BlockCapacity:      *blockCapacity,
		MaxConcurrency:     *maxConcurrency,
		BaseLatency:        *baseLatency,
		ColdInputTPS:       *coldInputTPS,
		CachedInputTPS:     *cachedInputTPS,
		OutputTPS:          *outputTPS,
		OutputKVMultiplier: *outputKVMultiplier,
		DefaultMaxTokens:   *defaultMaxTokens,
	}

	// -instances <= 1 (the default) uses -port exactly as before; only
	// -instances > 1 switches to the -base-port-derived range. This keeps
	// every existing single-instance invocation byte-for-byte unchanged.
	ports := []int{*port}
	if *instances > 1 {
		ports = make([]int, *instances)
		for i := range ports {
			ports[i] = *basePort + i
		}
	}

	servers := make([]*http.Server, len(ports))
	for i, p := range ports {
		// Each instance gets its OWN Engine — independent cache, counters,
		// and admission state — sharing only the config VALUES, never any
		// state. That independence is what makes this a fleet rather than
		// one cache behind N listeners.
		engine := mockvllm.NewEngine(cfg)
		srv := mockvllm.NewServer(engine)
		switch *surface {
		case "vllm":
			srv.Surface = mockvllm.SurfaceVLLM
		case "anthropic":
			srv.Surface = mockvllm.SurfaceAnthropic
		case "hosted":
			srv.Surface = mockvllm.SurfaceHosted
		default:
			return fmt.Errorf("unknown -surface %q: want vllm, anthropic or hosted", *surface)
		}
		servers[i] = &http.Server{Addr: fmt.Sprintf(":%d", p), Handler: srv.Handler()}
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, len(servers))
	for _, hs := range servers {
		go func() {
			log.Info("mock-vllm listening", "addr", hs.Addr, "model", cfg.ModelID,
				"block_size_tokens", cfg.BlockSizeTokens, "chars_per_token", cfg.CharsPerToken,
				"block_capacity", cfg.BlockCapacity, "max_concurrency", cfg.MaxConcurrency,
				"cold_input_tps", cfg.ColdInputTPS, "cached_input_tps", cfg.CachedInputTPS,
				"output_tps", cfg.OutputTPS, "output_kv_multiplier", cfg.OutputKVMultiplier,
				"shared_kv_tier", cfg.Tier != nil, "external_kv_tps", cfg.ExternalInputTPS)
			errCh <- hs.ListenAndServe()
		}()
	}

	select {
	case err := <-errCh:
		// One instance failing to bind (e.g. a port already in use) is fatal
		// for the whole process — a partial fleet is a misleading fleet.
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-ctx.Done():
		log.Info("shutting down", "instances", len(servers))
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		var wg sync.WaitGroup
		errs := make([]error, len(servers))
		for i, hs := range servers {
			wg.Add(1)
			go func(i int, hs *http.Server) {
				defer wg.Done()
				errs[i] = hs.Shutdown(shutdownCtx)
			}(i, hs)
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				return err
			}
		}
		return nil
	}
}
