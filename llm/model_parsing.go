package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// DynamicModelConfig holds configuration for a dynamically-specified model
type DynamicModelConfig struct {
	BaseURL         string   // Base URL for the API endpoint — always BaseURLs[0] (for backward compat, model fetching, display)
	BaseURLs        []string // All endpoint URLs; random selection happens per call when len > 1
	Type            string   // Client type: "openai", "anthropic", "gemini" (default: "openai")
	ContextSize     int      // Context size if specified (0 = use default)
	MaxTokens       int      // Maximum output tokens if specified (0 = use default)
	Model           string   // Model identifier to send to the API (optional, can be extracted from /models endpoint)
	Alias           string   // Optional short alias for display purposes (e.g., "local", "gpu")
	ReasoningEffort string   // "low", "medium", "high" — empty means no reasoning effort set
	Thinking        string   // Raw "thinking" value forwarded verbatim into the request body (e.g. "on"/"off"/"enabled"); empty means unset
}

// ParseDynamicModel parses a dynamic model identifier
// Format: dynamic/<endpoint>,type=<type>,context_size=<size>,max_tokens=<tokens>,model=<model>
// Example: dynamic/http://localhost:8000/v1,type=openai,context_size=202752,max_tokens=128,model=glm-4.6-fp8
func ParseDynamicModel(modelStr string) (*DynamicModelConfig, error) {
	if !strings.HasPrefix(modelStr, "dynamic/") {
		return nil, fmt.Errorf("dynamic model identifier must start with 'dynamic/'")
	}

	// Remove "dynamic/" prefix
	remainder := strings.TrimPrefix(modelStr, "dynamic/")

	// Split by comma to separate endpoint from parameters
	parts := strings.Split(remainder, ",")
	if len(parts) == 0 {
		return nil, fmt.Errorf("dynamic model identifier missing endpoint")
	}

	// Split first part on "|" to support multiple endpoints
	endpointPart := parts[0]
	rawURLs := strings.Split(endpointPart, "|")
	baseURLs := make([]string, 0, len(rawURLs))
	for _, u := range rawURLs {
		u = strings.TrimSpace(u)
		if u == "" {
			continue
		}
		if !strings.HasSuffix(u, "/") {
			u += "/"
		}
		baseURLs = append(baseURLs, u)
	}
	if len(baseURLs) == 0 {
		return nil, fmt.Errorf("dynamic model identifier missing endpoint")
	}

	config := &DynamicModelConfig{
		BaseURL:  baseURLs[0],
		BaseURLs: baseURLs,
		Type:     "openai", // Default type
	}

	// Parse optional parameters
	for i := 1; i < len(parts); i++ {
		param := strings.TrimSpace(parts[i])
		if param == "" {
			continue
		}

		kv := strings.SplitN(param, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid parameter format: %s (expected key=value)", param)
		}

		key := strings.TrimSpace(kv[0])
		value := strings.TrimSpace(kv[1])

		switch key {
		case "type":
			config.Type = value
		case "context_size":
			var size int
			_, err := fmt.Sscanf(value, "%d", &size)
			if err != nil {
				return nil, fmt.Errorf("invalid context_size value: %s", value)
			}
			config.ContextSize = size
		case "max_tokens":
			var tokens int
			_, err := fmt.Sscanf(value, "%d", &tokens)
			if err != nil {
				return nil, fmt.Errorf("invalid max_tokens value: %s", value)
			}
			config.MaxTokens = tokens
		case "model":
			config.Model = value
		case "alias":
			config.Alias = value
		case "reasoning_effort":
			config.ReasoningEffort = value
		case "thinking":
			config.Thinking = value
		default:
			return nil, fmt.Errorf("unknown parameter: %s", key)
		}
	}

	return config, nil
}

// IsDynamicModel checks if a model string represents a dynamic model
func IsDynamicModel(model string) bool {
	return strings.HasPrefix(model, "dynamic/")
}

// NormalizeModelSpec maps shorthand model specs to canonical ones so callers
// don't have to spell out boilerplate. Today it handles one case:
//   - A bare URL ("http://..." or "https://...") is promoted to a dynamic
//     spec by prepending "dynamic/" and defaulting type=openai_vllm (vLLM's
//     /v1 endpoints want max_tokens, not max_completion_tokens). If the
//     caller already wrote type=... in the suffix, we leave it alone.
//
// "dynamic/..." specs and all provider-prefixed specs pass through unchanged.
func NormalizeModelSpec(model string) string {
	if !strings.HasPrefix(model, "http://") && !strings.HasPrefix(model, "https://") {
		return model
	}
	spec := "dynamic/" + model
	// Inspect suffix params only (after the first comma) to see if the user
	// already supplied a type=; checking the whole string would false-match on
	// the URL path.
	hasType := false
	if idx := strings.Index(spec, ","); idx >= 0 {
		for _, p := range strings.Split(spec[idx+1:], ",") {
			if strings.HasPrefix(strings.TrimSpace(p), "type=") {
				hasType = true
				break
			}
		}
	}
	if !hasType {
		spec += ",type=openai_vllm"
	}
	return spec
}

// OpenRouterModelConfig holds configuration for an OpenRouter model
type OpenRouterModelConfig struct {
	Model       string   // Model identifier (e.g., "anthropic/claude-3.5-sonnet")
	ContextSize int      // Context size if specified (0 = use default)
	MaxTokens   int      // Maximum output tokens if specified (0 = use default)
	Provider    []string // Provider preference order (e.g., ["OpenAI", "Together"])
}

// ParseOpenRouterModel parses an OpenRouter model identifier
// Format: openrouter/<model>,context_size=<size>,max_tokens=<tokens>,provider=<provider1,provider2>
// Example: openrouter/anthropic/claude-3.5-sonnet,context_size=200000,max_tokens=2048,provider=OpenAI,Together
func ParseOpenRouterModel(modelStr string) (*OpenRouterModelConfig, error) {
	if !strings.HasPrefix(modelStr, "openrouter/") {
		return nil, fmt.Errorf("openrouter model identifier must start with 'openrouter/'")
	}

	// Remove "openrouter/" prefix
	remainder := strings.TrimPrefix(modelStr, "openrouter/")

	// Split by comma to separate model from parameters
	parts := strings.Split(remainder, ",")
	if len(parts) == 0 || parts[0] == "" {
		return nil, fmt.Errorf("openrouter model identifier missing model name")
	}

	config := &OpenRouterModelConfig{
		Model: parts[0],
	}

	// Parse optional parameters
	for i := 1; i < len(parts); i++ {
		param := strings.TrimSpace(parts[i])
		if param == "" {
			continue
		}

		kv := strings.SplitN(param, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid parameter format: %s (expected key=value)", param)
		}

		key := strings.TrimSpace(kv[0])
		value := strings.TrimSpace(kv[1])

		switch key {
		case "context_size":
			var size int
			_, err := fmt.Sscanf(value, "%d", &size)
			if err != nil {
				return nil, fmt.Errorf("invalid context_size value: %s", value)
			}
			config.ContextSize = size
		case "max_tokens":
			var tokens int
			_, err := fmt.Sscanf(value, "%d", &tokens)
			if err != nil {
				return nil, fmt.Errorf("invalid max_tokens value: %s", value)
			}
			config.MaxTokens = tokens
		case "provider":
			// Split provider value by comma to support multiple providers
			// e.g., provider=OpenAI,Together
			providerList := strings.Split(value, ",")
			for _, p := range providerList {
				p = strings.TrimSpace(p)
				if p != "" {
					config.Provider = append(config.Provider, p)
				}
			}
		default:
			return nil, fmt.Errorf("unknown parameter: %s", key)
		}
	}

	return config, nil
}

// IsOpenRouterModel checks if a model string represents an OpenRouter model
func IsOpenRouterModel(model string) bool {
	return strings.HasPrefix(model, "openrouter/")
}

// StaticModelConfig holds configuration for a static model with parameters
type StaticModelConfig struct {
	BaseModel string // Base model name (e.g., "zai/glm-4.6")
	MaxTokens int    // Maximum output tokens if specified (0 = use default)
}

// ParseStaticModelParams parses a static model identifier with optional parameters
// Format: <model>,max_tokens=<tokens>
// Example: zai/glm-4.6,max_tokens=2048
func ParseStaticModelParams(modelStr string) (*StaticModelConfig, error) {
	// Check if the model string contains parameters
	if !strings.Contains(modelStr, ",") {
		// No parameters, return base model as-is
		return &StaticModelConfig{
			BaseModel: modelStr,
			MaxTokens: 0, // Use default
		}, nil
	}

	// Split by comma to separate model from parameters
	parts := strings.SplitN(modelStr, ",", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid model format: %s", modelStr)
	}

	baseModel := strings.TrimSpace(parts[0])
	if baseModel == "" {
		return nil, fmt.Errorf("model name cannot be empty: %s", modelStr)
	}

	config := &StaticModelConfig{
		BaseModel: baseModel,
		MaxTokens: 0, // Default
	}

	// Parse parameters
	paramStr := strings.TrimSpace(parts[1])
	params := strings.Split(paramStr, ",")

	for _, param := range params {
		param = strings.TrimSpace(param)
		if param == "" {
			continue
		}

		kv := strings.SplitN(param, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid parameter format: %s (expected key=value)", param)
		}

		key := strings.TrimSpace(kv[0])
		value := strings.TrimSpace(kv[1])

		switch key {
		case "max_tokens":
			var tokens int
			_, err := fmt.Sscanf(value, "%d", &tokens)
			if err != nil {
				return nil, fmt.Errorf("invalid max_tokens value: %s", value)
			}
			config.MaxTokens = tokens
		default:
			return nil, fmt.Errorf("unknown parameter for static model: %s", key)
		}
	}

	return config, nil
}

// ValidateModel checks if a model name is valid and returns an error if not.
// Named (non-dynamic, non-OpenRouter) models are validated against the
// ResolveModel hook — see hooks.go. If no hook is registered, named models
// cannot be validated (core ships no static registry) and validation fails.
func ValidateModel(model string) error {
	if model == "" {
		return nil // Empty model is allowed, will use default
	}

	// Check if it's a dynamic model
	if IsDynamicModel(model) {
		// Validate dynamic model syntax
		_, err := ParseDynamicModel(model)
		if err != nil {
			return fmt.Errorf("invalid dynamic model: %w", err)
		}
		return nil
	}

	// Check if it's an OpenRouter model
	if IsOpenRouterModel(model) {
		// Validate OpenRouter model syntax
		_, err := ParseOpenRouterModel(model)
		if err != nil {
			return fmt.Errorf("invalid openrouter model: %w", err)
		}
		return nil
	}

	// Check registry for standard models (with optional parameters)
	staticConfig, err := ParseStaticModelParams(model)
	if err != nil {
		return fmt.Errorf("invalid model parameters: %w", err)
	}

	if ResolveModel == nil {
		return fmt.Errorf("no model registry configured: %s\n\nUse a dynamic (dynamic/...) or openrouter/... model spec, or embed wekai-core with a ResolveModel hook", staticConfig.BaseModel)
	}
	if _, ok := ResolveModel(staticConfig.BaseModel); !ok {
		return fmt.Errorf("unsupported model: %s", staticConfig.BaseModel)
	}

	return nil
}

// fetchModelFromEndpoint attempts to fetch the first model from an OpenAI-compatible /models endpoint
func fetchModelFromEndpoint(baseURL string) (string, error) {
	// Ensure trailing slash
	if !strings.HasSuffix(baseURL, "/") {
		baseURL += "/"
	}

	modelsURL := baseURL + "models"

	resp, err := sharedHTTPClient.Get(modelsURL)
	if err != nil {
		return "", fmt.Errorf("failed to fetch models from %s: %w", modelsURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("models endpoint returned status %d", resp.StatusCode)
	}

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse models response: %w", err)
	}

	if len(result.Data) == 0 {
		return "", fmt.Errorf("no models found at endpoint")
	}

	return result.Data[0].ID, nil
}
