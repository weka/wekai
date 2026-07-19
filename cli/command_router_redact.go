package cli

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// RouterRedactCommand reads router raw-capture JSONL files and writes
// redacted copies using the same logic as the live `--capture redacted` path.
// This lets you test the redaction code against real captured data, and
// retroactively redact a raw capture you no longer want in the clear.
type RouterRedactCommand struct {
	Src string `short:"s" long:"src" required:"true" description:"Source raw-capture JSONL file or directory containing *.jsonl"`
	Dst string `short:"d" long:"dst" required:"true" description:"Destination directory for redacted JSONL output"`
}

func (c *RouterRedactCommand) Execute(args []string) error {
	info, err := os.Stat(c.Src)
	if err != nil {
		return fmt.Errorf("stat src: %w", err)
	}

	var files []string
	if info.IsDir() {
		entries, err := os.ReadDir(c.Src)
		if err != nil {
			return err
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
				files = append(files, filepath.Join(c.Src, e.Name()))
			}
		}
		if len(files) == 0 {
			return fmt.Errorf("no .jsonl files found in %s", c.Src)
		}
	} else {
		files = []string{c.Src}
	}

	if err := os.MkdirAll(c.Dst, 0o755); err != nil {
		return fmt.Errorf("mkdir dst: %w", err)
	}

	var grandTotal int
	for _, src := range files {
		dst := filepath.Join(c.Dst, filepath.Base(src))
		n, err := redactRawFile(src, dst)
		if err != nil {
			return fmt.Errorf("redact %s: %w", src, err)
		}
		fmt.Printf("%d records  %s -> %s\n", n, src, dst)
		grandTotal += n
	}
	fmt.Printf("total: %d records across %d file(s)\n", grandTotal, len(files))
	return nil
}

func redactRawFile(srcPath, dstPath string) (int, error) {
	in, err := os.Open(srcPath)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	out, err := os.Create(dstPath)
	if err != nil {
		return 0, err
	}
	defer out.Close()

	br := bufio.NewReaderSize(in, 1<<20)
	bw := bufio.NewWriterSize(out, 1<<20)
	defer bw.Flush()

	count := 0
	for lineNum := 1; ; lineNum++ {
		line, readErr := br.ReadBytes('\n')
		trimmed := bytes.TrimRight(line, "\r\n")
		if len(trimmed) > 0 {
			var rec captureRecord
			if err := json.Unmarshal(trimmed, &rec); err != nil {
				return count, fmt.Errorf("line %d parse: %w", lineNum, err)
			}
			// Convert Body from old string format if needed.
			// Old raw files stored body as a JSON string; new format uses json.RawMessage
			// that is a JSON string in raw mode. Ensure Body is a valid json.RawMessage.
			rec.Request.Body = normalizeBodyField(rec.Request.Body)
			rec.Response.Body = normalizeBodyField(rec.Response.Body)
			applyBodyRedaction(&rec)
			data, err := json.Marshal(&rec)
			if err != nil {
				return count, fmt.Errorf("line %d marshal: %w", lineNum, err)
			}
			if _, err := bw.Write(data); err != nil {
				return count, err
			}
			if err := bw.WriteByte('\n'); err != nil {
				return count, err
			}
			count++
		}
		if readErr == io.EOF {
			return count, nil
		}
		if readErr != nil {
			return count, readErr
		}
	}
}

// normalizeBodyField ensures the body is in json.RawMessage format.
// Handles old format where body was a plain string.
func normalizeBodyField(body json.RawMessage) json.RawMessage {
	if len(body) == 0 {
		return json.RawMessage("\"\"")
	}
	// Check if it's already a valid JSON-quoted string
	if len(body) >= 2 && body[0] == '"' {
		return body
	}
	// Old style: plain string that wasn't valid JSON - wrap it
	b, _ := json.Marshal(string(body))
	return b
}
