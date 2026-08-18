package benchmark

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// The readable view of a response, written beside the wire bytes.
//
// A streamed response is hundreds of SSE frames; nobody reads one to find out
// what the model said. The merged file is what it said, in order, with the
// same text scoring saw at the top so the two can be compared by eye.
//
// Tool calls are assembled here rather than taken from the parser, because the
// parser does not keep them: the streaming consumers read Anthropic's
// partial_json only to time first-token and never write it, and they do not
// look at OpenAI tool_calls deltas at all. That is defensible for a benchmark
// measuring latency and tokens, and useless for reading a transcript. They are
// therefore reconstructed from the capture and kept under their own heading —
// clearly not part of the text any scoring decision was made on, so the file
// cannot be mistaken for evidence of what the scorer read.

// mergedResponse renders scored text plus any tool calls found in raw.
func mergedResponse(scored string, raw []byte) []byte {
	var b bytes.Buffer
	b.WriteString(scored)
	calls := extractToolCalls(raw)
	if len(calls) == 0 {
		if !strings.HasSuffix(scored, "\n") {
			b.WriteString("\n")
		}
		return b.Bytes()
	}
	if !strings.HasSuffix(scored, "\n") {
		b.WriteString("\n")
	}
	b.WriteString("\n===== tool calls (reconstructed from the stream; NOT part of the text above) =====\n")
	for i, c := range calls {
		fmt.Fprintf(&b, "\n[%d] %s\n%s\n", i+1, c.Name, c.Arguments)
	}
	return b.Bytes()
}

type toolCall struct {
	Index     int
	Name      string
	Arguments string
}

// extractToolCalls handles both dialects and both transports. It is
// deliberately lenient: a frame it cannot parse is skipped rather than failing
// the dump, since this runs after the request is already measured and must
// never be able to affect it.
func extractToolCalls(raw []byte) []toolCall {
	byIndex := map[int]*toolCall{}
	get := func(i int) *toolCall {
		if c, ok := byIndex[i]; ok {
			return c
		}
		c := &toolCall{Index: i}
		byIndex[i] = c
		return c
	}

	consider := func(payload []byte) {
		// OpenAI, streamed and whole: choices[].delta|message.tool_calls[]
		var oa struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int `json:"index"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				Message struct {
					ToolCalls []struct {
						Index    int `json:"index"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(payload, &oa) == nil {
			for _, ch := range oa.Choices {
				for _, tc := range append(ch.Delta.ToolCalls, ch.Message.ToolCalls...) {
					c := get(tc.Index)
					if tc.Function.Name != "" {
						c.Name = tc.Function.Name
					}
					// Arguments arrive as fragments and mean nothing apart.
					c.Arguments += tc.Function.Arguments
				}
			}
		}

		// Anthropic streamed: content_block_start names it, content_block_delta
		// carries partial_json fragments keyed by the same index.
		var an struct {
			Type         string `json:"type"`
			Index        int    `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"content_block"`
			Delta struct {
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal(payload, &an) == nil {
			switch an.Type {
			case "content_block_start":
				if an.ContentBlock.Type == "tool_use" {
					get(an.Index).Name = an.ContentBlock.Name
				}
			case "content_block_delta":
				if an.Delta.PartialJSON != "" {
					if c, ok := byIndex[an.Index]; ok {
						c.Arguments += an.Delta.PartialJSON
					}
				}
			}
		}

		// Anthropic whole: content[] entries of type tool_use.
		var anWhole struct {
			Content []struct {
				Type  string          `json:"type"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			} `json:"content"`
		}
		if json.Unmarshal(payload, &anWhole) == nil {
			for i, blk := range anWhole.Content {
				if blk.Type == "tool_use" {
					c := get(1000 + i) // offset so it cannot collide with stream indices
					c.Name = blk.Name
					c.Arguments = string(blk.Input)
				}
			}
		}
	}

	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	sawSSE := false
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		sawSSE = true
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		consider([]byte(payload))
	}
	if !sawSSE {
		consider(raw) // non-streaming: the body is one JSON document
	}

	out := make([]toolCall, 0, len(byIndex))
	for _, c := range byIndex {
		if c.Name != "" || c.Arguments != "" {
			out = append(out, *c)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}
