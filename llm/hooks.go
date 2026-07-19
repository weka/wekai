package llm

// ResolveModel resolves a named (non-dynamic, non-OpenRouter) model
// identifier — e.g. "anthropic/sonnet-high" — to its ModelInfo. wekai-core
// ships no static model registry; it only understands dynamic model specs
// ("dynamic/...") and OpenRouter specs ("openrouter/...") natively.
//
// Applications embedding wekai-core with a named-model registry (aliases,
// pricing, defaults) should set this hook at init time. Left nil, any
// GetChatGetter call for a named model panics with "unsupported model".
var ResolveModel func(name string) (ModelInfo, bool)

// LookupModelByIdentifier finds a ModelInfo whose ModelIdentifier matches a
// raw provider-side model id (e.g. "claude-opus-4-8"), used by the router's
// `analyze` command to price captured traffic that was recorded by raw model
// id rather than by wekai-core's own alias. Left nil, cost is reported as
// 0/unknown for identifiers that don't come from a ChatGetter created in
// this process.
var LookupModelByIdentifier func(id string) (ModelInfo, bool)
