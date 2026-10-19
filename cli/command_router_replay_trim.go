package cli

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/weka/wekai/benchmark"
)

// RouterReplayTrimCommand writes the newest part of a replay-v5 file as a
// replay of its own: the same sessions, byte for byte, with a header and index
// recomputed for what was kept. Sessions sit in the file by start time, so the
// kept ones are a contiguous run read through the index.
//
// A session is kept or dropped whole, by its start: one that started before
// the cut is dropped even if it runs into the window, since replaying its tail
// alone would replay turns without the history they were sent with.
type RouterReplayTrimCommand struct {
	Out          string  `short:"o" long:"out" required:"yes" description:"Output replay file"`
	Name         string  `long:"name" description:"Name embedded in the header (default: output file name without .jsonl)"`
	LastDays     float64 `long:"last-days" description:"Keep sessions that started within this many days of the capture's end"`
	LastSessions int     `long:"last-sessions" description:"Keep the N sessions that started last (counted after incomplete sessions are excluded)"`
	// Incomplete sessions are excluded by default: one still active when the
	// capture stopped is cut off mid-conversation, and replaying it as if
	// complete under-states session length and the reuse its next turns would
	// have had.
	IncompleteWindow time.Duration `long:"incomplete-window" default:"24h" description:"Exclude sessions whose last request falls within this window of the capture's end, as possibly unfinished; with the default this removes the last day's sessions entirely. 0 keeps every session"`

	Args struct {
		Src string `positional-arg-name:"SRC" description:"replay-v5 file (with an index trailer)"`
	} `positional-args:"yes" required:"yes"`
}

// trimSession is the part of a session line the header needs.
type trimSession struct {
	SessionID string `json:"session_id"`
	StartTs   string `json:"start_ts"`
	Instances []struct {
		ParentSpawnRequestID uint64 `json:"parent_spawn_request_id,omitempty"`
		FanOutSize           int    `json:"fan_out_size,omitempty"`
		Requests             []struct {
			RequestID uint64 `json:"request_id"`
			Ts        string `json:"ts"`
			TotalMs   int64  `json:"total_ms"`
		} `json:"requests"`
	} `json:"instances"`
}

func (c *RouterReplayTrimCommand) Execute(args []string) error {
	if (c.LastDays > 0) == (c.LastSessions > 0) {
		return errors.New("pass exactly one of --last-days or --last-sessions")
	}
	if c.LastDays < 0 || c.LastSessions < 0 {
		return errors.New("--last-days and --last-sessions must be positive")
	}
	if abs(c.Out) == abs(c.Args.Src) {
		return errors.New("--out must differ from the source file")
	}

	src, err := os.Open(c.Args.Src)
	if err != nil {
		return err
	}
	defer src.Close()

	br := bufio.NewReader(src)
	hline, err := br.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("read header: %w", err)
	}
	var hdr ReplayHeader
	if err := json.Unmarshal(hline, &hdr); err != nil {
		return fmt.Errorf("parse header: %w", err)
	}
	if hdr.Schema != "replay-v5" {
		return fmt.Errorf("replay-trim needs a replay-v5 file (with an index), got %q; rebuild it with replay-prepare", hdr.Schema)
	}
	index, err := benchmark.ReadReplayIndex(src)
	if err != nil {
		return err
	}
	if index == nil || len(index.Sessions) == 0 {
		return errors.New("source has no session index")
	}

	// The capture's end: when its last request finished.
	var captureEnd time.Time
	for _, s := range index.Sessions {
		if t, err := time.Parse(time.RFC3339Nano, s.EndTs); err == nil && t.After(captureEnd) {
			captureEnd = t
		}
	}
	if captureEnd.IsZero() {
		return errors.New("source index has no session end times")
	}

	filter := c.IncompleteWindow > 0
	activeSince := captureEnd.Add(-c.IncompleteWindow)
	var cutoff time.Time
	if c.LastDays > 0 {
		cutoff = captureEnd.Add(-time.Duration(c.LastDays * float64(24*time.Hour)))
	}

	var kept []benchmark.ReplayIndexSession
	incomplete := 0
	for _, s := range index.Sessions {
		start, err := time.Parse(time.RFC3339Nano, s.StartTs)
		if err != nil {
			continue
		}
		if c.LastDays > 0 && start.Before(cutoff) {
			continue
		}
		if filter {
			if end, err := time.Parse(time.RFC3339Nano, s.EndTs); err != nil || end.After(activeSince) {
				incomplete++
				continue
			}
		}
		kept = append(kept, s)
	}
	if c.LastSessions > 0 && len(kept) > c.LastSessions {
		kept = kept[len(kept)-c.LastSessions:]
	}
	if len(kept) == 0 {
		return errors.New("no sessions selected")
	}

	name := c.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(c.Out), ".jsonl")
	}
	sum, err := trimSummary(src, kept)
	if err != nil {
		return err
	}
	out := ReplayHeader{
		Schema:      hdr.Schema,
		Name:        name,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Source:      hdr.Source,
		Summary:     sum,
	}
	if err := writeTrimmed(c.Out, src, &out, kept); err != nil {
		return err
	}

	sel := fmt.Sprintf("last %d sessions", c.LastSessions)
	if c.LastDays > 0 {
		sel = fmt.Sprintf("sessions started since %s (last %gd)", cutoff.Format(time.RFC3339), c.LastDays)
	}
	fmt.Printf("source: %s (%d sessions, %d requests, %s -> %s)\n", hdr.Name, hdr.Summary.Sessions,
		hdr.Summary.Requests, hdr.Summary.StartTs, captureEnd.Format(time.RFC3339))
	fmt.Printf("kept:   %s", sel)
	if filter {
		fmt.Printf("; %d sessions excluded as incomplete (a request after %s; --incomplete-window=0 keeps them)", incomplete, activeSince.Format(time.RFC3339))
	}
	fmt.Printf("\nwrote %d sessions / %d instances / %d requests (%s -> %s) -> %s\n",
		sum.Sessions, sum.Instances, sum.Requests, sum.StartTs, sum.EndTs, c.Out)
	return nil
}

func abs(p string) string {
	a, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return a
}

// trimSummary recomputes the header summary over the kept sessions: counts,
// time span, max fan-out and the timing block. The spawn, fan-out-turn and
// repair counters come from replay-prepare's analysis of the capture and
// cannot be rebuilt from a replay, so they are left 0 rather than estimated.
func trimSummary(src *os.File, kept []benchmark.ReplayIndexSession) (ReplaySummary, error) {
	var sum ReplaySummary
	think := newThinkAccumulator()
	var first, last time.Time
	for _, e := range kept {
		buf := make([]byte, e.Length)
		if _, err := src.ReadAt(buf, e.Offset); err != nil {
			return sum, fmt.Errorf("read session %s: %w", e.SessionID, err)
		}
		var ts trimSession
		if err := json.Unmarshal(buf, &ts); err != nil {
			return sum, fmt.Errorf("parse session %s: %w", e.SessionID, err)
		}
		// annotateThink recomputes what replay-prepare stored, from the same
		// timestamps, to rebuild the timing summary for the kept requests.
		rs := ReplaySession{SessionID: ts.SessionID, StartTs: ts.StartTs}
		for _, inst := range ts.Instances {
			ri := ReplayInstance{ParentSpawnRequestID: inst.ParentSpawnRequestID}
			sum.MaxFanOutInOneTurn = max(sum.MaxFanOutInOneTurn, inst.FanOutSize)
			for _, r := range inst.Requests {
				ri.Requests = append(ri.Requests, ReplayRequest{RequestID: r.RequestID, Ts: r.Ts, TotalMs: r.TotalMs})
				if t, err := time.Parse(time.RFC3339Nano, r.Ts); err == nil {
					if first.IsZero() || t.Before(first) {
						first = t
					}
					if t.After(last) {
						last = t
					}
				}
			}
			sum.Requests += len(inst.Requests)
			rs.Instances = append(rs.Instances, ri)
		}
		sum.Instances += len(ts.Instances)
		sum.Sessions++
		annotateThink(&rs, think)
	}
	sum.StartTs = first.Format(time.RFC3339)
	sum.EndTs = last.Format(time.RFC3339)
	sum.Timing = think.summary()
	return sum, nil
}

// writeTrimmed writes the header, the kept session lines verbatim, and a fresh
// index, to a temporary file renamed into place once complete.
func writeTrimmed(dst string, src *os.File, hdr *ReplayHeader, kept []benchmark.ReplayIndexSession) error {
	tmp := dst + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer os.Remove(tmp)
	defer f.Close()
	bw := bufio.NewWriterSize(f, 1<<20)
	hb, err := json.Marshal(hdr)
	if err != nil {
		return err
	}
	if _, err := bw.Write(append(hb, '\n')); err != nil {
		return err
	}
	offset := int64(len(hb)) + 1
	index := make([]benchmark.ReplayIndexSession, 0, len(kept))
	for _, e := range kept {
		n := e.Length + 1 // with its newline
		if _, err := io.Copy(bw, io.NewSectionReader(src, e.Offset, n)); err != nil {
			return fmt.Errorf("copy session %s: %w", e.SessionID, err)
		}
		e.Offset = offset
		offset += n
		index = append(index, e)
	}
	ib, err := json.Marshal(&benchmark.ReplayIndex{Schema: benchmark.ReplayIndexSchema, Sessions: index})
	if err != nil {
		return err
	}
	if _, err := bw.Write(append(ib, '\n')); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}
