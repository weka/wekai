package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// Verbatim request/response capture, for deciding whether a validation result
// is the fleet's behaviour or this code's.
//
// Every other output here is derived: the JSONL records counts, the summary
// records ratios, and a PRESENCE_MISS says only that an expected marker was
// not in the response. It cannot say whether the marker reached the prompt,
// whether the recite instruction was intelligible, or what the model said
// instead — and those are three different defects with the same number.
//
// So this writes what actually crossed the wire and nothing else: the exact
// bytes posted, the exact bytes received (tee'd off the response body, so SSE
// framing survives), and the scoring that was derived from them, side by side
// under one name. A disagreement between the three is then readable rather
// than inferred.
type requestDumper struct {
	dir   string
	limit int64
	n     atomic.Int64
	warn  sync.Once
}

func newRequestDumper(dir string, limit int) (*requestDumper, error) {
	if dir == "" {
		return nil, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create dump dir: %w", err)
	}
	if limit <= 0 {
		limit = defaultDumpLimit
	}
	return &requestDumper{dir: dir, limit: int64(limit)}, nil
}

// defaultDumpLimit bounds the capture because a replay prompt averages several
// hundred kilobytes, so an unbounded dump fills a disk during a run nobody is
// watching. Reaching it is announced rather than silently applied: a directory
// that stops growing looks exactly like a run that stopped making requests.
const defaultDumpLimit = 200

// dumpMeta is the derived view, written beside the bytes it was derived from.
type dumpMeta struct {
	Series     int      `json:"series"`
	Instance   string   `json:"instance"`
	Turn       int      `json:"turn"`
	URL        string   `json:"url"`
	Status     int      `json:"status,omitempty"`
	Error      string   `json:"error,omitempty"`
	TTFTMillis int64    `json:"ttft_ms"`
	InputTok   int      `json:"input_tokens"`
	OutputTok  int      `json:"output_tokens"`
	Expected   []string `json:"expected_uuids,omitempty"`
	Found      []bool   `json:"uuid_found,omitempty"`
	Leaked     []string `json:"leaked_uuids,omitempty"`
	ExactMatch bool     `json:"first_line_exact"`
	// LeakChecked distinguishes "scanned, found nothing" from "never scanned",
	// which the file would otherwise render identically as an absent list.
	LeakChecked bool `json:"leak_checked"`
}

// dump writes one exchange. Errors are reported once and never fail the
// request: a capture is an observation, and an observation must not be able to
// break the thing it observes.
func (d *requestDumper) dump(meta dumpMeta, request, response []byte) {
	if d == nil {
		return
	}
	if d.n.Add(1) > d.limit {
		d.warn.Do(func() {
			fmt.Fprintf(os.Stderr,
				"[dump] reached --dump-limit=%d exchanges; no further requests are being written to %s\n",
				d.limit, d.dir)
		})
		return
	}
	base := filepath.Join(d.dir, fmt.Sprintf("s%03d-%s-t%03d", meta.Series, sanitizeName(meta.Instance), meta.Turn))
	write := func(suffix string, b []byte) {
		if err := os.WriteFile(base+suffix, b, 0o644); err != nil {
			d.warn.Do(func() {
				fmt.Fprintf(os.Stderr, "[dump] cannot write %s: %v (capture disabled for the rest of the run)\n",
					base+suffix, err)
			})
			d.n.Store(d.limit + 1)
		}
	}
	write(".request.json", request)
	write(".response.raw", response)
	if b, err := json.MarshalIndent(meta, "", "  "); err == nil {
		write(".meta.json", append(b, '\n'))
	}
}

// sanitizeName keeps an instance id usable as a filename. Replay instance ids
// come from captured traffic, so nothing guarantees they are path-safe.
func sanitizeName(s string) string {
	if s == "" {
		return "none"
	}
	out := []rune(s)
	for i, r := range out {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
		default:
			out[i] = '_'
		}
	}
	if len(out) > 40 {
		out = out[:40]
	}
	return string(out)
}
