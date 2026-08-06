// Command mock-vllm is a standalone, GPU-less stand-in for a vLLM OpenAI
// server. It models prefix-cache behavior (chained block hashing + LRU
// eviction, via github.com/weka/wekai/kvcache — the same engine the router's
// prefix-cache-aware policy uses to predict residency), per-instance
// concurrency admission (HTTP 429 past --max-concurrency), and a configurable
// latency model, so router routing/cache policies can be developed and
// evaluated against a fleet of these without any real GPUs or vLLM workers.
//
// Run several on different ports to form a fleet:
//
//	go run ./router/cmd/mock-vllm --port 9001 &
//	go run ./router/cmd/mock-vllm --port 9002 &
//	go run ./router/cmd/mock-vllm --port 9003 &
//	go run ./router/cmd/mock-vllm --port 9004 &
//
// Point wllm-router's static backend list at http://localhost:9001..9004 and
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
	port := fs.Int("port", 9000, "listen port")
	modelID := fs.String("model-id", "mock-vllm", "model id reported by /v1/models and echoed in responses")
	blockSizeTokens := fs.Int("block-size-tokens", 16, "cache block size in tokens (vLLM's block_size analog; converted to bytes at 4 bytes/token)")
	blockCapacity := fs.Int64("block-capacity", 100_000, "cache capacity in blocks; 0 = unbounded (never evicts)")
	maxConcurrency := fs.Int("max-concurrency", 64, "requests admitted at once before returning HTTP 429; 0 = unbounded")
	defaultMaxTokens := fs.Int("default-max-tokens", 128, "completion length used when a request omits max_tokens")
	baseLatency := fs.Duration("base-latency", 20*time.Millisecond, "fixed latency added to every request's TTFT")
	prefillPerToken := fs.Duration("prefill-per-token", 200*time.Microsecond, "TTFT added per UNCACHED prompt token")
	decodePerToken := fs.Duration("decode-per-token", 20*time.Millisecond, "latency added per output token (also paces SSE chunks)")
	logLevel := fs.String("log-level", "info", "debug, info, warn, or error")

	if err := fs.Parse(args); err != nil {
		return err
	}

	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(*logLevel)); err != nil {
		return fmt.Errorf("invalid -log-level %q: %w", *logLevel, err)
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))

	cfg := mockvllm.Config{
		ModelID:          *modelID,
		BlockSizeTokens:  *blockSizeTokens,
		BlockCapacity:    *blockCapacity,
		MaxConcurrency:   *maxConcurrency,
		BaseLatency:      *baseLatency,
		PrefillPerToken:  *prefillPerToken,
		DecodePerToken:   *decodePerToken,
		DefaultMaxTokens: *defaultMaxTokens,
	}
	engine := mockvllm.NewEngine(cfg)
	srv := mockvllm.NewServer(engine)

	addr := fmt.Sprintf(":%d", *port)
	httpSrv := &http.Server{Addr: addr, Handler: srv.Handler()}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Info("mock-vllm listening", "addr", addr, "model", cfg.ModelID,
			"block_size_tokens", cfg.BlockSizeTokens, "block_capacity", cfg.BlockCapacity,
			"max_concurrency", cfg.MaxConcurrency)
		errCh <- httpSrv.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-ctx.Done():
		log.Info("shutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return httpSrv.Shutdown(shutdownCtx)
	}
}
