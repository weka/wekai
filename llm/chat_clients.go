package llm

import (
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
)

// LLMConfig holds configuration for LLM client initialization
type LLMConfig struct {
	Model                  string
	MaxTokens              int
	ReasoningEffort        ReasoningEffort
	Thinking               string
	ForceToolCall          bool // Force tool_choice: "required" in API requests
	BaseURL                string
	APIKey                 string
	StreamResponseCallback func(string)
	StreamThinkingCallback func(string)
	ExtraBodyParams        map[string]interface{} // Extra parameters to include in the request body (e.g., OpenRouter provider preferences)
	UseCompatMaxTokens     bool                   // Use "max_tokens" instead of "max_completion_tokens" for vLLM and other OpenAI-compatible servers
	WebSearchEnabled       bool                   // Enable provider-native web search tool
}

// APIKeys holds all the API keys for different LLM providers
type APIKeys struct {
	OpenAI     string
	Anthropic  string
	Gemini     string
	Groq       string
	Internal1  string
	OpenRouter string
	ZAI        string
}

// ChatParams holds parameters for creating a ChatGetter
type ChatParams struct {
	MaxTokens        int
	ResponseCallback func(string) // each tool/chat can decide to use either of this, depending on what it intends to return to user
	ThinkingCallback func(string)
	ProgressCallback func(s string) // TODO: not yet used
	APIKeys          APIKeys
	WebSearchEnabled bool   // Enable provider-native web search tool
	EndpointOverride string // If non-empty, use this endpoint instead of random selection
}

// GetChatGetter creates a ChatGetter for the specified model
func GetChatGetter(model string, params *ChatParams) *ChatGetter {
	if params == nil {
		params = &ChatParams{}
	}

	// Check if this is a dynamic model
	if IsDynamicModel(model) {
		return getDynamicChatGetter(model, params)
	}

	// Check if this is an OpenRouter model
	if IsOpenRouterModel(model) {
		return getOpenRouterChatGetter(model, params)
	}

	// Parse static model parameters (e.g., max_tokens)
	staticConfig, err := ParseStaticModelParams(model)
	if err != nil {
		panic(fmt.Sprintf("failed to parse static model parameters: %v", err))
	}

	// Use the parsed max_tokens if specified and not already set in params
	if staticConfig.MaxTokens > 0 && params.MaxTokens == 0 {
		params.MaxTokens = staticConfig.MaxTokens
	}

	// Named models are resolved via the ResolveModel hook — wekai-core ships
	// no static registry of its own. See hooks.go.
	if ResolveModel == nil {
		panic("no model registry configured: " + staticConfig.BaseModel)
	}
	modelInfo, ok := ResolveModel(staticConfig.BaseModel)
	if !ok {
		panic("unsupported model: " + staticConfig.BaseModel)
	}

	// --- Parameter & Callback Setup ---
	thinkingCallback := params.ThinkingCallback
	if thinkingCallback == nil {
		thinkingCallback = func(s string) {} // No-op default
	}
	responseCallback := params.ResponseCallback
	if responseCallback == nil {
		responseCallback = func(s string) {} // No-op default
	}

	// --- API Key Validation ---
	var apiKey string
	switch modelInfo.Provider {
	case ProviderOpenAI:
		apiKey = params.APIKeys.OpenAI
		if apiKey == "" {
			panic("OpenAI API key is required for model: " + model)
		}
	case ProviderAnthropic:
		apiKey = params.APIKeys.Anthropic
		if apiKey == "" {
			panic("Anthropic API key is required for model: " + model)
		}
	case ProviderGemini:
		apiKey = params.APIKeys.Gemini
		if apiKey == "" {
			panic("Gemini API key is required for model: " + model)
		}
	case ProviderGroq:
		apiKey = params.APIKeys.Groq
		if apiKey == "" {
			panic("Groq API key is required for model: " + model)
		}
	case ProviderInternal1:
		apiKey = params.APIKeys.Internal1
		if apiKey == "" {
			panic("Internal1 API key is required for internal1 model: " + model)
		}
	case ProviderZAI:
		apiKey = params.APIKeys.ZAI
		if apiKey == "" {
			panic("ZAI API key is required for ZAI model: " + model)
		}
	default:
		panic("unknown provider for model: " + model) // Should not happen if registry is correct
	}

	// --- Max Tokens Calculation ---
	// Determine the effective max *output* tokens to request.
	// Priority: User override (if valid) > Model Default Output Budget > Model Max Output Tokens
	effectiveMaxTokens := 0
	modelMaxOutput := modelInfo.MaxTokens // MaxTokens now means max *output* tokens
	if modelMaxOutput <= 0 {
		// Fallback if MaxTokens isn't set in registry
		modelMaxOutput = 4096 // A reasonable default fallback
		fmt.Fprintf(os.Stderr, "Warning: MaxTokens (max output) not set for model %s, defaulting to %d\n", staticConfig.BaseModel, modelMaxOutput)
	}

	if params.MaxTokens > 0 {
		// User provided an override for max output tokens
		if params.MaxTokens <= modelMaxOutput {
			effectiveMaxTokens = params.MaxTokens // User value is within model's output limit
		} else {
			effectiveMaxTokens = modelMaxOutput // User value exceeds limit, cap it
			fmt.Fprintf(os.Stderr, "Warning: Requested MaxTokens (%d) exceeds model's max output limit (%d) for %s. Using model limit.\n", params.MaxTokens, modelMaxOutput, staticConfig.BaseModel)
		}
	} else if modelInfo.DefaultMaxTokens > 0 {
		// No user override, use model's default *output* budget if set (e.g., for Anthropic effort)
		effectiveMaxTokens = modelInfo.DefaultMaxTokens
		// Ensure default output budget doesn't exceed the model's max output (sanity check)
		if effectiveMaxTokens > modelMaxOutput {
			effectiveMaxTokens = modelMaxOutput
			fmt.Fprintf(os.Stderr, "Warning: DefaultMaxTokens (%d) exceeds model's max output limit (%d) for %s. Capping at model limit.\n", modelInfo.DefaultMaxTokens, modelMaxOutput, staticConfig.BaseModel)
		}
	} else {
		// No user override and no default output budget, use the model's max output token limit
		effectiveMaxTokens = modelMaxOutput
	}

	// --- LLM Config Construction ---
	llmConfig := LLMConfig{
		Model:                  modelInfo.ModelIdentifier,
		MaxTokens:              effectiveMaxTokens,        // Use the calculated effective value
		ReasoningEffort:        modelInfo.ReasoningEffort, // Use the default from the registry for this alias
		ForceToolCall:          modelInfo.ForceToolCall,   // Force tool_choice: "required" for models that need it
		BaseURL:                modelInfo.BaseURL,         // Will be empty string if not specified, handled by client
		APIKey:                 apiKey,
		StreamResponseCallback: responseCallback,
		StreamThinkingCallback: thinkingCallback,
		WebSearchEnabled:       params.WebSearchEnabled,
	}

	// --- Client Instantiation ---
	var chatFunc func() Chat
	switch modelInfo.ClientType {
	case ClientTypeOpenAI:
		chatFunc = func() Chat {
			return NewOpenAiLLMClient(llmConfig)
		}
	case ClientTypeOpenAIResponses:
		chatFunc = func() Chat {
			return NewOpenAiResponsesLLMClient(llmConfig)
		}
	case ClientTypeAnthropic:
		// Ensure Anthropic-specific MaxTokens logic based on ReasoningEffort is respected
		// The DefaultMaxTokens in modelInfo already handles the budgeting overrides.
		chatFunc = func() Chat {
			return NewAnthropicLLMClient(llmConfig)
		}
	case ClientTypeGeminiNative:
		// Native Gemini client might not use BaseURL or ReasoningEffort from LLMConfig
		// Ensure NewGeminiLLMClient handles the config appropriately.
		chatFunc = func() Chat {
			return NewGeminiLLMClient(llmConfig)
		}
	default:
		panic("unsupported client type for model: " + model) // Should not happen
	}

	return &ChatGetter{
		modelInfo: modelInfo,
		modelName: model, // Store the user-facing model name
		chatFunc:  chatFunc,
	}
}

// getDynamicChatGetter creates a ChatGetter for a dynamic model specification
func getDynamicChatGetter(model string, params *ChatParams) *ChatGetter {
	// Parse dynamic model config
	dynConfig, err := ParseDynamicModel(model)
	if err != nil {
		panic(fmt.Sprintf("failed to parse dynamic model: %v", err))
	}

	// Autodiscover each endpoint's effective base URL (appending /v1/ for a
	// bare host:port that answers /v1/models) and, for the primary endpoint
	// only, a model id if the spec didn't give one. resolveDynamicEndpoint
	// memoizes per raw endpoint via sync.Once, so however many times this
	// function runs for the same spec (it runs per-request on several
	// benchmark paths), the actual network probe fires exactly once per
	// distinct endpoint per process.
	wantModel := dynConfig.Model == ""
	resolvedURLs := make([]string, len(dynConfig.BaseURLs))
	var primary resolvedEndpoint
	for i, u := range dynConfig.BaseURLs {
		res := resolveDynamicEndpoint(u, wantModel && i == 0)
		resolvedURLs[i] = res.base
		if i == 0 {
			primary = res
		}
	}
	dynConfig.BaseURLs = resolvedURLs
	dynConfig.BaseURL = resolvedURLs[0]

	if wantModel {
		if primary.model != "" {
			dynConfig.Model = primary.model
		} else if dynConfig.Type == "anthropic" {
			// Anthropic-compatible servers reject unknown/placeholder model
			// ids outright, unlike some permissive single-model vLLM
			// deployments — silently defaulting would just trade a clear
			// startup error for a cryptic per-request one.
			panic(errModelAutodiscoveryFailed(dynConfig.Type, dynConfig.BaseURL))
		} else {
			fmt.Fprintf(os.Stderr, "Warning: Failed to autodiscover model from endpoint %s. Using 'default' as model identifier.\n", dynConfig.BaseURL)
			dynConfig.Model = "default"
		}
	}

	// Determine MaxTokens with priority: explicit max_tokens > context_size/2 > default
	maxTokens := 32768 // Default max output tokens
	if dynConfig.MaxTokens > 0 {
		// Use explicitly specified max_tokens
		maxTokens = dynConfig.MaxTokens
	} else if dynConfig.ContextSize > 0 {
		// Use a portion of context size for output (e.g., half)
		maxTokens = dynConfig.ContextSize / 2
	}

	contextWindow := 131072
	if dynConfig.ContextSize > 0 {
		contextWindow = dynConfig.ContextSize
	}

	// Create a synthetic ModelInfo for the dynamic model
	modelInfo := ModelInfo{
		Provider:             "dynamic",
		ModelIdentifier:      dynConfig.Model,
		MaxTokens:            maxTokens,
		BaseURL:              dynConfig.BaseURL,
		Description:          fmt.Sprintf("Dynamic model: %s at %s", dynConfig.Model, strings.Join(dynConfig.BaseURLs, ", ")),
		CachedCostPerMillion: 0.0, // Unknown costs for dynamic models
		InputCostPerMillion:  0.0,
		OutputCostPerMillion: 0.0,
		ContextWindow:        contextWindow,
	}

	// Determine client type based on config
	switch dynConfig.Type {
	case "openai", "openai_vllm":
		modelInfo.ClientType = ClientTypeOpenAI
	case "openai_responses":
		modelInfo.ClientType = ClientTypeOpenAIResponses
	case "anthropic":
		modelInfo.ClientType = ClientTypeAnthropic
	case "gemini_native":
		modelInfo.ClientType = ClientTypeGeminiNative
	default:
		panic(fmt.Sprintf("unsupported dynamic model type: %s", dynConfig.Type))
	}

	// --- Parameter & Callback Setup ---
	thinkingCallback := params.ThinkingCallback
	if thinkingCallback == nil {
		thinkingCallback = func(s string) {} // No-op default
	}
	responseCallback := params.ResponseCallback
	if responseCallback == nil {
		responseCallback = func(s string) {} // No-op default
	}

	// --- API Key Handling ---
	// Dynamic models always point to a user-supplied URL (dynConfig.BaseURL
	// is set by ParseDynamicModel), not the provider's canonical endpoint.
	// Many local/test deployments (vLLM, sglang, a self-hosted proxy) don't
	// validate the Authorization header, so requiring a real provider key
	// here makes the dynamic path unusable for those cases.
	// Policy: use the real key if present (some routed endpoints still
	// require one), otherwise fall back to a placeholder — exactly the same
	// treatment the openai branches have used since this code was written.
	var apiKey string
	switch dynConfig.Type {
	case "openai", "openai_vllm", "openai_responses":
		apiKey = params.APIKeys.OpenAI
	case "anthropic":
		apiKey = params.APIKeys.Anthropic
	case "gemini_native":
		apiKey = params.APIKeys.Gemini
	}
	if apiKey == "" {
		// Some endpoints don't validate the key; pass a placeholder so the
		// header is well-formed without forcing the user to set a dummy
		// env var (e.g. ANTHROPIC_API_KEY=dummy) just to hit localhost.
		apiKey = "dummy-key"
	}

	// --- Max Tokens Calculation ---
	effectiveMaxTokens := maxTokens
	if params.MaxTokens > 0 {
		if params.MaxTokens <= maxTokens {
			effectiveMaxTokens = params.MaxTokens
		} else {
			effectiveMaxTokens = maxTokens
			fmt.Fprintf(os.Stderr, "Warning: Requested MaxTokens (%d) exceeds dynamic model's max (%d). Using model limit.\n", params.MaxTokens, maxTokens)
		}
	}

	// --- LLM Config Template ---
	// BaseURL will be overridden per-call when multiple endpoints are configured.
	llmConfigTemplate := LLMConfig{
		Model:                  dynConfig.Model,
		MaxTokens:              effectiveMaxTokens,
		ReasoningEffort:        ReasoningEffort(dynConfig.ReasoningEffort),
		Thinking:               dynConfig.Thinking,
		BaseURL:                dynConfig.BaseURL,
		APIKey:                 apiKey,
		StreamResponseCallback: responseCallback,
		StreamThinkingCallback: thinkingCallback,
		UseCompatMaxTokens:     dynConfig.Type == "openai_vllm",
		WebSearchEnabled:       params.WebSearchEnabled,
	}

	// selectBaseURL returns a random endpoint when multiple are configured,
	// or the single endpoint otherwise.
	baseURLs := dynConfig.BaseURLs
	selectBaseURL := func() string {
		if params.EndpointOverride != "" {
			return params.EndpointOverride
		}
		if len(baseURLs) <= 1 {
			return llmConfigTemplate.BaseURL
		}
		return baseURLs[rand.IntN(len(baseURLs))]
	}

	// --- Client Instantiation ---
	var chatFunc func() Chat
	switch modelInfo.ClientType {
	case ClientTypeOpenAI:
		chatFunc = func() Chat {
			cfg := llmConfigTemplate
			cfg.BaseURL = selectBaseURL()
			return NewOpenAiLLMClient(cfg)
		}
	case ClientTypeOpenAIResponses:
		chatFunc = func() Chat {
			cfg := llmConfigTemplate
			cfg.BaseURL = selectBaseURL()
			return NewOpenAiResponsesLLMClient(cfg)
		}
	case ClientTypeAnthropic:
		chatFunc = func() Chat {
			cfg := llmConfigTemplate
			cfg.BaseURL = selectBaseURL()
			return NewAnthropicLLMClient(cfg)
		}
	case ClientTypeGeminiNative:
		chatFunc = func() Chat {
			cfg := llmConfigTemplate
			cfg.BaseURL = selectBaseURL()
			return NewGeminiLLMClient(cfg)
		}
	default:
		panic(fmt.Sprintf("unsupported client type for dynamic model: %s", modelInfo.ClientType))
	}

	return &ChatGetter{
		modelInfo: modelInfo,
		modelName: model, // Store the original dynamic model specification
		chatFunc:  chatFunc,
	}
}

// getOpenRouterChatGetter creates a ChatGetter for an OpenRouter model specification
func getOpenRouterChatGetter(model string, params *ChatParams) *ChatGetter {
	// Parse OpenRouter model config
	orConfig, err := ParseOpenRouterModel(model)
	if err != nil {
		panic(fmt.Sprintf("failed to parse OpenRouter model: %v", err))
	}

	// OpenRouter API endpoint
	baseURL := "https://openrouter.ai/api/v1/"

	// Determine MaxTokens based on max_tokens, context_size, or use a reasonable default
	maxTokens := 32768 // Default max output tokens
	if orConfig.MaxTokens > 0 {
		// Use explicitly specified max_tokens
		maxTokens = orConfig.MaxTokens
	} else if orConfig.ContextSize > 0 {
		// Use a portion of context size for output (e.g., half)
		maxTokens = orConfig.ContextSize / 2
	}

	// Create a synthetic ModelInfo for the OpenRouter model
	modelInfo := ModelInfo{
		Provider:             ProviderOpenRouter,
		ClientType:           ClientTypeOpenAI, // OpenRouter uses OpenAI-compatible API
		ModelIdentifier:      orConfig.Model,
		MaxTokens:            maxTokens,
		BaseURL:              baseURL,
		Description:          fmt.Sprintf("OpenRouter model: %s", orConfig.Model),
		CachedCostPerMillion: 0.0, // Unknown costs for OpenRouter models
		InputCostPerMillion:  0.0,
		OutputCostPerMillion: 0.0,
		ContextWindow:        orConfig.ContextSize,
	}

	// --- Parameter & Callback Setup ---
	thinkingCallback := params.ThinkingCallback
	if thinkingCallback == nil {
		thinkingCallback = func(s string) {} // No-op default
	}
	responseCallback := params.ResponseCallback
	if responseCallback == nil {
		responseCallback = func(s string) {} // No-op default
	}

	// --- API Key Handling ---
	apiKey := params.APIKeys.OpenRouter
	if apiKey == "" {
		panic("OpenRouter API key is required for OpenRouter models (set OPENROUTER_API_KEY environment variable)")
	}

	// --- Max Tokens Calculation ---
	effectiveMaxTokens := maxTokens
	if params.MaxTokens > 0 {
		if params.MaxTokens <= maxTokens {
			effectiveMaxTokens = params.MaxTokens
		} else {
			effectiveMaxTokens = maxTokens
			fmt.Fprintf(os.Stderr, "Warning: Requested MaxTokens (%d) exceeds OpenRouter model's max (%d). Using model limit.\n", params.MaxTokens, maxTokens)
		}
	}

	// --- Provider Preferences Setup ---
	var extraBodyParams map[string]interface{}
	if len(orConfig.Provider) > 0 {
		// Build the provider object according to OpenRouter API spec
		extraBodyParams = map[string]interface{}{
			"provider": map[string]interface{}{
				"order": orConfig.Provider,
			},
		}
	}

	// --- LLM Config Construction ---
	llmConfig := LLMConfig{
		Model:                  orConfig.Model,
		MaxTokens:              effectiveMaxTokens,
		ReasoningEffort:        ReasoningEffortNone, // OpenRouter models default to no reasoning effort
		BaseURL:                baseURL,
		APIKey:                 apiKey,
		StreamResponseCallback: responseCallback,
		StreamThinkingCallback: thinkingCallback,
		ExtraBodyParams:        extraBodyParams,
	}

	// --- Client Instantiation ---
	// OpenRouter uses OpenAI-compatible API
	chatFunc := func() Chat {
		return NewOpenAiLLMClient(llmConfig)
	}

	return &ChatGetter{
		modelInfo: modelInfo,
		modelName: model, // Store the original OpenRouter model specification
		chatFunc:  chatFunc,
	}
}
