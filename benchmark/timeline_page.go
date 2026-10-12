package benchmark

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// The renderer is one JS/CSS pair shared by every surface: the report's
// Timeline view, the standalone page below, and the live page served during a
// run (live_view.go).
var (
	//go:embed timeline_assets/timeline.js
	timelineJS string
	//go:embed timeline_assets/timeline.css
	timelineCSS string
	//go:embed timeline_assets/timeline_page.html
	timelinePageHTML string
)

var timelinePageTpl = template.Must(template.New("timeline").Parse(timelinePageHTML))

// timelinePageConfig is the page's runtime configuration.
type timelinePageConfig struct {
	Live        bool           `json:"live"`
	Base        string         `json:"base,omitempty"` // URL prefix of the live API ("" = same directory)
	PollMs      int            `json:"pollMs,omitempty"`
	MaxRequests int            `json:"maxRequests,omitempty"`
	Kinds       []timelineKind `json:"kinds,omitempty"`
}

// renderTimelinePage renders the self-contained page. payload is embedded for
// a static page; a live page passes nil and fetches its data.
func renderTimelinePage(title, subtitle string, payload *timelinePayload, cfg timelinePageConfig) ([]byte, error) {
	data := template.JS("null")
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal timeline data: %w", err)
		}
		data = template.JS(b)
	}
	cb, err := json.Marshal(cfg)
	if err != nil {
		return nil, fmt.Errorf("marshal timeline config: %w", err)
	}
	var buf bytes.Buffer
	err = timelinePageTpl.Execute(&buf, map[string]any{
		"Title":    title,
		"Subtitle": subtitle,
		"CSS":      template.CSS(timelineCSS),
		"JS":       template.JS(timelineJS),
		"Config":   template.JS(cb),
		"Data":     data,
	})
	return buf.Bytes(), err
}

// timelineArmsFromDir reads every reqdata .jsonl in dir into one timeline arm
// each. Files without request rows are skipped.
func timelineArmsFromDir(dir string, maxElapsed time.Duration) ([]timelineArm, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("glob jsonl files: %w", err)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no .jsonl files found in %s", dir)
	}
	sort.Strings(files)
	var arms []timelineArm
	for _, f := range files {
		records, samples, err := readJSONLFile(f)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", f, err)
		}
		records, _ = truncateToElapsed(records, samples, maxElapsed)
		if len(records) == 0 {
			continue
		}
		name := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		if alias := resolveRecordsAlias(records); alias != "" {
			name = alias
		}
		arms = append(arms, buildTimelineArm(name, records))
	}
	if len(arms) == 0 {
		return nil, fmt.Errorf("no request rows in %s", dir)
	}
	return arms, nil
}

// GenerateTimeline writes a standalone timeline page for the reqdata in dir.
// out is the destination file; empty means a new dir/timeline[_vN].html.
func GenerateTimeline(dir, out string, maxElapsed time.Duration) (string, error) {
	arms, err := timelineArmsFromDir(dir, maxElapsed)
	if err != nil {
		return "", err
	}
	if out == "" {
		out = nextVersionedPath(dir, "timeline", ".html")
	}
	var parts []string
	for _, a := range arms {
		parts = append(parts, a.summary())
	}
	p := newTimelinePayload(arms...)
	page, err := renderTimelinePage("Request timeline", strings.Join(parts, " · "), &p, timelinePageConfig{})
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(out, page, 0o644); err != nil {
		return "", fmt.Errorf("write %s: %w", out, err)
	}
	return out, nil
}
