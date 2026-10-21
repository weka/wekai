package cli

import "strings"

// treeInterner deduplicates the strings and string slices of parsed capture
// records. Every request carries the conversation's full history, so the same
// message hash, block-type list and tool ids recur once per later turn; over a
// multi-month corpus that repetition dominates the parsed heap. Records share
// the interned values, which must therefore be treated as read-only.
//
// Messages are shared too: a turn's history is the previous turn's history
// plus the new exchange, so requests of one conversation become prefixes of
// one shared backing array (see messages).
type treeInterner struct {
	strs   map[string]string
	slices map[string][]string
	// convs maps a conversation's first message hash to the longest history
	// seen for it; later requests that extend or repeat it slice into it.
	convs map[string][]redactedMessage
}

func newTreeInterner() *treeInterner {
	return &treeInterner{
		strs:   map[string]string{},
		slices: map[string][]string{},
		convs:  map[string][]redactedMessage{},
	}
}

func (in *treeInterner) str(s string) string {
	if s == "" {
		return ""
	}
	if v, ok := in.strs[s]; ok {
		return v
	}
	in.strs[s] = s
	return s
}

func (in *treeInterner) slice(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	key := strings.Join(s, "\x00")
	if v, ok := in.slices[key]; ok {
		return v
	}
	for i := range s {
		s[i] = in.str(s[i])
	}
	in.slices[key] = s
	return s
}

func (in *treeInterner) request(r *redactedRequest) {
	r.Schema = in.str(r.Schema)
	r.Model = in.str(r.Model)
	r.ExtraKeys = in.slice(r.ExtraKeys)
	for i := range r.SystemBlocks {
		b := &r.SystemBlocks[i]
		b.Type = in.str(b.Type)
		b.Hash = in.str(b.Hash)
		b.CacheControl = in.str(b.CacheControl)
	}
	if r.Tools != nil {
		r.Tools.Hash = in.str(r.Tools.Hash)
	}
	for i := range r.Messages {
		m := &r.Messages[i]
		m.Role = in.str(m.Role)
		m.Hash = in.str(m.Hash)
		m.CacheControl = in.str(m.CacheControl)
		m.SeedHash = in.str(m.SeedHash)
		m.BlockTypes = in.slice(m.BlockTypes)
		m.ToolUseIDs = in.slice(m.ToolUseIDs)
		m.ToolResultIDs = in.slice(m.ToolResultIDs)
	}
	r.Messages = in.messages(r.Messages)
}

// messages returns msgs, or an equal slice that shares its backing array with
// earlier requests of the same conversation. The result is capped (len ==
// cap), so an append by a caller copies instead of writing into another
// request's history.
func (in *treeInterner) messages(msgs []redactedMessage) []redactedMessage {
	if len(msgs) == 0 {
		return msgs
	}
	key := msgs[0].Hash
	prev := in.convs[key]
	p := 0
	for p < len(prev) && p < len(msgs) && messagesEqual(&prev[p], &msgs[p]) {
		p++
	}
	switch {
	case p == len(msgs):
		// A repeat (retry, count_tokens) or a prefix of what was seen.
		return prev[:p:p]
	case p == len(prev):
		// The usual next turn: extend the shared history in place. Older
		// requests keep their capped views, so a reallocation leaves them
		// on the old array, still valid.
		prev = append(prev, msgs[p:]...)
		in.convs[key] = prev
		return prev[:len(msgs):len(msgs)]
	default:
		// Diverged (compaction, an edited turn, another conversation with
		// the same opener): this history becomes the one to extend.
		own := append(make([]redactedMessage, 0, len(msgs)), msgs...)
		in.convs[key] = own
		return own[:len(msgs):len(msgs)]
	}
}

// messagesEqual compares two interned messages; interned slices that are equal
// share their first element, so comparing that pointer is enough.
func messagesEqual(a, b *redactedMessage) bool {
	return a.Role == b.Role && a.Hash == b.Hash && a.Bytes == b.Bytes &&
		a.Tokens == b.Tokens && a.CacheControl == b.CacheControl &&
		a.SeedHash == b.SeedHash &&
		sameSlice(a.BlockTypes, b.BlockTypes) &&
		sameSlice(a.ToolUseIDs, b.ToolUseIDs) &&
		sameSlice(a.ToolResultIDs, b.ToolResultIDs)
}

func sameSlice(a, b []string) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

func (in *treeInterner) response(r *redactedResponse) {
	r.Schema = in.str(r.Schema)
	r.StopReason = in.str(r.StopReason)
	for i := range r.OutputBlocks {
		b := &r.OutputBlocks[i]
		b.Type = in.str(b.Type)
		b.ToolNameHash = in.str(b.ToolNameHash)
	}
}
