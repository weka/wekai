package llm

// Model provider constants
const (
	ProviderOpenAI     = "openai"
	ProviderAnthropic  = "anthropic"
	ProviderGemini     = "gemini"
	ProviderGroq       = "groq"
	ProviderInternal1  = "internal1"
	ProviderOpenRouter = "openrouter"
	ProviderZAI        = "zai"
)

// ReasoningEffort defines levels of computational effort for LLM reasoning.
// Note: Currently primarily influences Anthropic token budgeting.
type ReasoningEffort string

const (
	ReasoningEffortNone   ReasoningEffort = "none"   // Default/no specific effort adjustment
	ReasoningEffortLow    ReasoningEffort = "low"    // Lower computational budget
	ReasoningEffortMedium ReasoningEffort = "medium" // Medium computational budget
	ReasoningEffortHigh   ReasoningEffort = "high"   // Higher computational budget
)

// ClientType identifies the underlying client implementation
type ClientType string

const (
	ClientTypeOpenAI          ClientType = "openai"
	ClientTypeOpenAIResponses ClientType = "openai_responses" // New Responses API client
	ClientTypeAnthropic       ClientType = "anthropic"
	ClientTypeGeminiNative    ClientType = "gemini_native" // Added native Gemini client type
)

// ModelInfo holds the configuration details for a specific model alias.
type ModelInfo struct {
	Provider                      string          // e.g., ProviderOpenAI, ProviderAnthropic
	ClientType                    ClientType      // Which client implementation to use (e.g., ClientTypeOpenAI)
	ModelIdentifier               string          // The actual model string for the API (e.g., "o4-mini", "claude-3-7-sonnet-latest")
	MaxTokens                     int             // Maximum *output* tokens the model can generate in a single response.
	DefaultMaxTokens              int             // Default *output* tokens if not overridden by ChatParams (used for budgeting like Anthropic effort). 0 means use MaxTokens.
	BaseURL                       string          // Optional Base URL override (e.g., for Groq, Gemini via OpenAI endpoint)
	ReasoningEffort               ReasoningEffort // Default reasoning effort for this model alias
	ForceToolCall                 bool            // Force tool_choice: "required" in Responses API (for models that otherwise ask clarifying questions)
	Description                   string          // User-facing description
	CachedCostPerMillion          float64         // Cost per million cached tokens (USD)
	InputCostPerMillion           float64         // Cost per million input tokens (USD)
	OutputCostPerMillion          float64         // Cost per million output tokens (USD)
	CacheTokens5MinCostPerMillion float64         // Cost per million cache tokens with 5-minute TTL (USD) - Anthropic specific
	ContextWindow                 int             // Maximum input context window in tokens (e.g., 200000 for Claude)
}
