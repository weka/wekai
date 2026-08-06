package mockvllm

import "github.com/weka/wekai/kvcache"

// This file reimplements kvcache.ChunkContent's chunking ONLY because that
// function's per-chunk token count always goes through kvcache.EstimateTokens,
// which is fixed at 4.0 bytes/token for every consumer of the shared package
// (the router's own cache prediction, the benchmark's estimator). This
// engine's whole point is an independently calibratable ratio
// (Config.CharsPerToken — real vLLM's actual tokenizer runs closer to
// 2.9-3.4 chars/token on dense agentic text), so it needs its own token-count
// math while still sharing kvcache's hashing (HashContent) for chain-hash
// compatibility with the rest of the module — a prompt this engine chunks
// and a prompt the router's own trie chunks (at kvcache's fixed ratio) will
// generally disagree on exact block boundaries, which is expected: they are
// deliberately calibrated to different tokenizer fidelities, and only this
// engine's OWN chain (built once here, matched against itself on repeat
// requests) needs internal consistency.

// tokensForBytes converts a byte length to an estimated token count using
// charsPerToken, mirroring kvcache.EstimateTokens' semantics (clamped to at
// least 1 for non-empty content) but with a caller-supplied ratio instead of
// kvcache's fixed 4.0.
func tokensForBytes(byteLen int, charsPerToken float64) int32 {
	if byteLen <= 0 {
		return 0
	}
	if charsPerToken <= 0 {
		charsPerToken = 4.0
	}
	if n := int32(float64(byteLen) / charsPerToken); n >= 1 {
		return n
	}
	return 1
}

// chunkContent splits content into fixed byte windows and returns one Unit
// per window, exactly like kvcache.ChunkContent — same tag/continuation
// convention (only the first window carries tag; later windows use
// "\x01cont" so a multi-window segment doesn't re-anchor mid-way), same
// hashing (kvcache.HashContent) — but with tokensForBytes(charsPerToken)
// instead of kvcache.EstimateTokens for the per-chunk token count.
func chunkContent(tag string, content []byte, chunkBytes int, charsPerToken float64) []kvcache.Unit {
	if chunkBytes <= 0 {
		chunkBytes = kvcache.DefaultChunkBytes
	}
	if len(content) == 0 {
		return []kvcache.Unit{{Hash: kvcache.HashContent(tag, nil), Tokens: 1}}
	}
	out := make([]kvcache.Unit, 0, (len(content)+chunkBytes-1)/chunkBytes)
	for off := 0; off < len(content); off += chunkBytes {
		end := off + chunkBytes
		if end > len(content) {
			end = len(content)
		}
		chunk := content[off:end]
		t := tag
		if off > 0 {
			t = "\x01cont"
		}
		out = append(out, kvcache.Unit{Hash: kvcache.HashContent(t, chunk), Tokens: tokensForBytes(len(chunk), charsPerToken)})
	}
	return out
}
