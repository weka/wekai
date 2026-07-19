// Package config provides the minimal global configuration and
// environment-based API key loading needed by wekai-core's benchmark,
// router, and eval commands. It intentionally carries none of the
// bot/agent/datastore configuration surface (sandbox mode, job depth,
// datastore URI, etc.) — those concepts don't exist in wekai-core.
package config

import (
	"os"

	"github.com/weka/wekai-core/llm"
)

// ConfigStruct holds the subset of global configuration read by moved
// benchmark/router/eval commands. Applications embedding wekai-core (e.g.
// wekai itself) should populate the equivalent fields on their own config
// struct AND sync them here — see the wekai-side migration notes.
type ConfigStruct struct {
	TmpDir              string
	JSONOutput          bool
	LogFormat           string // "text" or "json" — reserved, not currently branched on (matches upstream wekai behavior)
	StderrLogs          bool
	LogHTTPRequests     bool
	GlobalModelOverride string
	APIKeys             llm.APIKeys
}

// Config is the single global configuration instance, mirroring wekai's own
// package-level `config.Config` pattern so CLI command Execute() methods can
// assign fields directly (`config.Config.TmpDir = ...`).
var Config ConfigStruct

// LoadAPIKeys reads all provider API keys from environment variables. This
// is the only place wekai-core reads API keys from the environment; the
// embedding application may instead inject keys it already loaded itself
// via Config.APIKeys directly.
func LoadAPIKeys() llm.APIKeys {
	return llm.APIKeys{
		OpenAI:     os.Getenv("OPENAI_API_KEY"),
		Anthropic:  os.Getenv("ANTHROPIC_API_KEY"),
		Gemini:     os.Getenv("GEMINI_API_KEY"),
		Groq:       os.Getenv("GROQ_API_KEY"),
		OpenRouter: os.Getenv("OPENROUTER_API_KEY"),
		ZAI:        os.Getenv("ZAI_TOKEN"),
	}
}

// GetChatGetter is the core equivalent of wekai's app.App.GetChatGetter: it
// applies GlobalModelOverride and fills in any unset ChatParams.APIKeys
// fields from Config.APIKeys before delegating to llm.GetChatGetter. Moved
// benchmark/cli code calls this instead of threading an *app.App through
// every function signature.
func GetChatGetter(model string, params *llm.ChatParams) *llm.ChatGetter {
	if params == nil {
		params = &llm.ChatParams{}
	}

	if Config.GlobalModelOverride != "" {
		model = Config.GlobalModelOverride
	}

	if params.APIKeys.OpenAI == "" {
		params.APIKeys.OpenAI = Config.APIKeys.OpenAI
	}
	if params.APIKeys.Anthropic == "" {
		params.APIKeys.Anthropic = Config.APIKeys.Anthropic
	}
	if params.APIKeys.Gemini == "" {
		params.APIKeys.Gemini = Config.APIKeys.Gemini
	}
	if params.APIKeys.Groq == "" {
		params.APIKeys.Groq = Config.APIKeys.Groq
	}
	if params.APIKeys.Internal1 == "" {
		params.APIKeys.Internal1 = Config.APIKeys.Internal1
	}
	if params.APIKeys.OpenRouter == "" {
		params.APIKeys.OpenRouter = Config.APIKeys.OpenRouter
	}
	if params.APIKeys.ZAI == "" {
		params.APIKeys.ZAI = Config.APIKeys.ZAI
	}

	return llm.GetChatGetter(model, params)
}

// GetAPIKeys returns the currently configured API keys.
func GetAPIKeys() llm.APIKeys {
	return Config.APIKeys
}
