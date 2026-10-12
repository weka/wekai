package llm

// Engine-neutral per-request KV provenance.
//
// vLLM (--enable-request-kv-details) and SGLang (return_cached_tokens_details)
// both report where a request's prompt tokens came from, in different wire
// shapes. TokenSources is the one representation the rest of wekai consumes;
// the wire types below exist only to decode each engine's JSON into it.

// TokenSources splits a request's prompt tokens by where they came from.
//
//	GPU       served from the engine's on-GPU prefix cache
//	Host      served from host DRAM (SGLang HiCache L2); vLLM reports none
//	External  loaded from an external tier: vLLM's KV connector (weka,
//	          offload) or SGLang's storage backend (L3)
//	Computed  prefilled by the engine
type TokenSources struct {
	GPU      int `json:"gpu"`
	Host     int `json:"host,omitempty"`
	External int `json:"external"`
	Computed int `json:"computed"`
}

// Total is the prompt size the sources add up to.
func (s TokenSources) Total() int { return s.GPU + s.Host + s.External + s.Computed }

// TokenSourcesFromSGLang maps SGLang's tier split (device/host/storage) onto
// TokenSources. SGLang does not report computed tokens; they are the prompt
// remainder, so promptTokens must be the request's GROSS prompt size.
func TokenSourcesFromSGLang(d *CachedTokensDetails, promptTokens int) *TokenSources {
	if d == nil {
		return nil
	}
	s := &TokenSources{GPU: d.Device, Host: d.Host, External: d.Storage}
	if c := promptTokens - (d.Device + d.Host + d.Storage); c > 0 {
		s.Computed = c
	}
	return s
}

// KVSources is vLLM's usage.prompt_tokens_details.kv_sources.
type KVSources struct {
	GPUCache int `json:"gpu_cache"`
	External int `json:"external"`
	Computed int `json:"computed"`
}

// TokenSources converts the wire shape to the engine-neutral one.
func (k *KVSources) TokenSources() *TokenSources {
	if k == nil {
		return nil
	}
	return &TokenSources{GPU: k.GPUCache, External: k.External, Computed: k.Computed}
}

// KVRecompute is vLLM's usage.prompt_tokens_details.kv_recompute: work redone
// after the request's first admission.
type KVRecompute struct {
	NumPreemptions            int `json:"num_preemptions"`
	PreemptDiscardedTokens    int `json:"preempt_discarded_tokens"`
	NumExternalLoadFailures   int `json:"num_external_load_failures"`
	ExternalLoadRecomputeToks int `json:"external_load_recompute_tokens"`
}

// RequestTimings is vLLM's usage.vllm_request_timings: engine-side durations
// in seconds. The scheduler-time phases are QueueWaitKV (waiting because KV
// blocks could not be allocated), QueueWaitOther, KVFetch (waiting for an
// async external KV load), Prefill, Decode and PreemptedTime.
type RequestTimings struct {
	QueueWaitKV    float64 `json:"queue_wait_kv"`
	QueueWaitOther float64 `json:"queue_wait_other"`
	KVFetch        float64 `json:"kv_fetch"`
	Prefill        float64 `json:"prefill"`
	Decode         float64 `json:"decode"`
	PreemptedTime  float64 `json:"preempted_time"`
	NumPreemptions int     `json:"num_preemptions"`
}

// KVDetails is everything a response reported about where a request's prompt
// came from and where its engine time went. Nil means the engine reported
// nothing; individual members are nil when that part was absent.
type KVDetails struct {
	Sources   *TokenSources   `json:"sources,omitempty"`
	Recompute *KVRecompute    `json:"recompute,omitempty"`
	Timings   *RequestTimings `json:"timings,omitempty"`
}

// NewKVDetails assembles a KVDetails from vLLM's three wire members, or nil
// when none is present (the flag is off, or another engine).
func NewKVDetails(src *KVSources, rc *KVRecompute, tm *RequestTimings) *KVDetails {
	if src == nil && rc == nil && tm == nil {
		return nil
	}
	return &KVDetails{Sources: src.TokenSources(), Recompute: rc, Timings: tm}
}

// WithSGLang fills Sources from SGLang's tier split when the response carried
// no vLLM sources. promptTokens is the gross prompt size.
func (d *KVDetails) WithSGLang(c *CachedTokensDetails, promptTokens int) *KVDetails {
	s := TokenSourcesFromSGLang(c, promptTokens)
	if s == nil {
		return d
	}
	if d == nil {
		return &KVDetails{Sources: s}
	}
	if d.Sources == nil {
		d.Sources = s
	}
	return d
}
