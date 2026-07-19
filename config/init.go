package config

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/go-logr/logr"
	"github.com/go-logr/zerologr"
	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
	"github.com/weka/go-weka-observability/instrumentation"
	"github.com/weka/wekai-core/llm"
)

// LoadDotEnv loads a .env file from the working directory if present. Safe
// to call multiple times.
func LoadDotEnv() error {
	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to load .env file: %w", err)
	}
	return nil
}

// getBaseLogger builds the base zerolog-backed logr.Logger: stderr when
// StderrLogs is set (or TmpDir is unusable), otherwise a log file under
// TmpDir/wekai-core.log.
func getBaseLogger() *logr.Logger {
	if Config.StderrLogs || Config.TmpDir == "" {
		l := zerolog.New(os.Stderr).
			Level(zerolog.InfoLevel).
			With().
			Timestamp().
			Caller().
			Logger()
		lr := zerologr.New(&l)
		return &lr
	}

	if err := os.MkdirAll(Config.TmpDir, 0755); err != nil {
		l := zerolog.New(os.Stderr).Level(zerolog.InfoLevel).With().Timestamp().Logger()
		lr := zerologr.New(&l)
		return &lr
	}
	logFilePath := filepath.Join(Config.TmpDir, "wekai-core.log")
	file, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		l := zerolog.New(os.Stderr).Level(zerolog.InfoLevel).With().Timestamp().Logger()
		lr := zerologr.New(&l)
		return &lr
	}

	l := zerolog.New(file).With().Timestamp().Logger()
	lr := zerologr.New(&l)
	return &lr
}

// Init is the unified initialization entry point for wekai-core commands:
// loads .env, loads API keys from the environment (unless the embedding
// application already populated Config.APIKeys — see the wekai-side
// migration notes), propagates LogHTTPRequests to the LOG_HTTP_REQUESTS env
// var the llm package reads directly, and sets up logging/OTel.
//
// mode is a free-form string identifying the command (e.g. "benchmark",
// "router_serve", "eval_cache_coherency") used only for OTel resource
// attribution.
func Init(ctx context.Context, mode string) (context.Context, func(context.Context) error, error) {
	if err := LoadDotEnv(); err != nil {
		// Non-critical — proceed without .env.
		_ = err
	}

	// Only load from env if the embedding application hasn't already
	// populated Config.APIKeys itself (e.g. from its own key-loading path).
	var zeroKeys llm.APIKeys
	if Config.APIKeys == zeroKeys {
		Config.APIKeys = LoadAPIKeys()
	}

	if Config.LogHTTPRequests {
		os.Setenv("LOG_HTTP_REQUESTS", "true")
	} else {
		os.Unsetenv("LOG_HTTP_REQUESTS")
	}

	baseLogger := getBaseLogger()
	ctx, logger := instrumentation.GetLoggerForContext(ctx, baseLogger, "")
	shutdown, err := instrumentation.SetupOTelSDK(ctx, "wekai-core", "v0.0.1", logger, "mode", mode)
	if err != nil {
		shutdown = func(context.Context) error { return nil }
	}

	if Config.TmpDir != "" {
		if err := os.MkdirAll(Config.TmpDir, os.ModePerm); err != nil {
			return ctx, shutdown, fmt.Errorf("failed to create temporary directory %s: %w", Config.TmpDir, err)
		}
	}

	return ctx, shutdown, nil
}
