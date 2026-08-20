package benchmark

import (
	"crypto/sha256"
	"encoding/hex"
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
	mode  dumpMode
	limit int64
	n     atomic.Int64
	warn  sync.Once

	// dirMu guards the destination, which a default-on capture only resolves
	// when it first has something to write. See ensureDir.
	dirMu  sync.Mutex
	dir    string
	dirErr error
}

// dumpMode selects which exchanges reach the disk.
type dumpMode int

const (
	dumpOff dumpMode = iota
	// dumpAll writes every exchange (--dump-dir). It answers questions about
	// the run as a whole and costs a few hundred KB per request, so it is
	// something a reader asks for deliberately.
	dumpAll
	// dumpGarbage writes only the exchanges whose response carried
	// decode-level corruption, and is ON by default under --verify.
	//
	// Garbage is the one verdict that cannot be re-derived after the fact. A
	// count says two responses were corrupt; it cannot say what was in the
	// prompt that produced them, whether the corruption sat mid-answer or ran
	// to the end of the budget, or whether the same session produced the next
	// one. By the time a reader knows they want that, the run is over and the
	// bytes are gone — and a corrupt response is rare enough that keeping
	// every one of them costs nothing next to re-running the arm.
	dumpGarbage
)

// Written reports how many exchanges landed on disk, where, and whether the
// capture was the garbage-only one — the summary prints this so a reader can
// go from a number straight to the bytes behind it.
//
// A dumper that resolves its directory on demand reports none until it has
// written something, which is the honest answer: nothing exists yet.
func (d *requestDumper) Written() (dir string, n int64, garbageOnly bool) {
	if d == nil {
		return "", 0, false
	}
	n = d.n.Load()
	if n > d.limit {
		n = d.limit
	}
	d.dirMu.Lock()
	dir = d.dir
	d.dirMu.Unlock()
	return dir, n, d.mode == dumpGarbage
}

// newRequestDumper builds the capture for one run. An empty dir under
// dumpGarbage means "make one when there is something to put in it"; under
// dumpAll it means the flag was not given, so there is no capture at all.
func newRequestDumper(mode dumpMode, dir string, limit int) (*requestDumper, error) {
	if mode == dumpOff || (mode == dumpAll && dir == "") {
		return nil, nil
	}
	if dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("create dump dir: %w", err)
		}
	}
	if limit <= 0 {
		limit = defaultDumpLimit
	}
	return &requestDumper{mode: mode, dir: dir, limit: int64(limit)}, nil
}

// ensureDir resolves the destination, creating a fresh mktemp directory the
// first time one is needed and announcing it on stderr.
//
// Deferring it is what makes the garbage capture safe to leave on: a run that
// produces no garbage creates no directory, so the default never litters
// /tmp with empty evidence of nothing having gone wrong. The announcement is
// on stderr beside the [verify] GARBAGE line it belongs to, so the path is in
// front of whoever is watching the run rather than only in the summary hours
// later.
func (d *requestDumper) ensureDir() string {
	d.dirMu.Lock()
	defer d.dirMu.Unlock()
	if d.dir != "" || d.dirErr != nil {
		return d.dir
	}
	dir, err := os.MkdirTemp("", "wekai-garbage-")
	if err != nil {
		d.dirErr = err
		fmt.Fprintf(os.Stderr, "[dump] cannot create a capture directory: %v (capture disabled)\n", err)
		return ""
	}
	d.dir = dir
	fmt.Fprintf(os.Stderr, "[dump] capturing garbage exchanges to %s (--dump-garbage-dir to choose one)\n", dir)
	return dir
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
	// The miss attribution for THIS exchange, so a single file explains its own
	// result. Without it a reader has to re-derive from the bytes what the
	// summary already worked out — which is exactly the re-derivation this
	// capture exists to spare them.
	MissSubstituted  int    `json:"miss_substituted,omitempty"`
	MissAbsent       int    `json:"miss_absent,omitempty"`
	ReciteEchoedTags bool   `json:"recite_echoed_tags,omitempty"`
	ReciteNoIDs      bool   `json:"recite_no_ids,omitempty"`
	BudgetShort      bool   `json:"recite_budget_short,omitempty"`
	Garbage          bool   `json:"response_garbage,omitempty"`
	GarbageVerdict   string `json:"garbage_verdict,omitempty"`
}

// dump writes one exchange: the request, the raw response, the readable
// merge, and the verdict. Errors are reported once and never fail the
// request: a capture is an observation, and an observation must not be able to
// break the thing it observes.
func (d *requestDumper) dump(meta dumpMeta, request, response, merged []byte) {
	if d == nil {
		return
	}
	// The filter precedes the counter, so the limit bounds what is WRITTEN.
	// Counting the exchanges it skips would let a clean stretch of the run
	// exhaust a garbage capture's budget before the first corrupt response
	// ever arrived.
	if d.mode == dumpGarbage && !meta.Garbage {
		return
	}
	if d.n.Add(1) > d.limit {
		d.warn.Do(func() {
			dir, _, _ := d.Written()
			fmt.Fprintf(os.Stderr,
				"[dump] reached --dump-limit=%d exchanges; no further requests are being written to %s\n",
				d.limit, dir)
		})
		return
	}
	dir := d.ensureDir()
	if dir == "" {
		return
	}
	base := filepath.Join(dir, fmt.Sprintf("s%03d-%s-t%03d", meta.Series, sanitizeName(meta.Instance), meta.Turn))
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
	write(".response.merged.txt", merged)
	if b, err := json.MarshalIndent(meta, "", "  "); err == nil {
		write(".meta.json", append(b, '\n'))
	}
}

// sanitizeName keeps an instance id usable as a filename. Replay instance ids
// come from captured traffic, so nothing guarantees they are path-safe.
//
// Long ids are shortened to a prefix PLUS a hash of the whole id, never a bare
// prefix. Instance ids in this corpus share their first 40+ characters — the
// session uuid, then "::sha256:..." — so a plain truncation collides across
// every instance of a session, and each instance's t001 silently overwrites
// the last one's. A capture that loses exchanges without saying so is worse
// than none: the count in the summary and the files on disk stop agreeing.
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
	if len(out) <= 40 {
		return string(out)
	}
	sum := sha256.Sum256([]byte(s))
	return string(out[:31]) + "-" + hex.EncodeToString(sum[:4])
}
