package benchmark

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// GenerateVisualizationMerged reads .jsonl files from multiple directories,
// creates a merged output directory with combined JSONL (using directory names
// as series aliases), generates per-source and merged CSVs, and produces a
// single HTML report.
func GenerateVisualizationMerged(dirs []string, outputDir string, concurrency int) (string, error) {
	if len(dirs) == 0 {
		return "", fmt.Errorf("no directories provided")
	}

	// Determine output directory
	if outputDir == "" {
		parent := filepath.Dir(dirs[0])
		outputDir = filepath.Join(parent, "merged")
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return "", fmt.Errorf("create output directory: %w", err)
	}

	// Create CSV subdirectories
	chunkedDir := filepath.Join(outputDir, "chunked_csv")
	fullDir := filepath.Join(outputDir, "full_csv")
	if err := os.MkdirAll(chunkedDir, 0o755); err != nil {
		return "", fmt.Errorf("create chunked_csv directory: %w", err)
	}
	if err := os.MkdirAll(fullDir, 0o755); err != nil {
		return "", fmt.Errorf("create full_csv directory: %w", err)
	}

	// For each input dir, read all JSONL files and write merged JSONL
	// with the directory basename as a prefix alias.
	var allRecords []taggedRecord

	for _, dir := range dirs {
		alias := filepath.Base(dir)
		files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
		if err != nil {
			return "", fmt.Errorf("glob jsonl files in %s: %w", dir, err)
		}
		if len(files) == 0 {
			fmt.Fprintf(os.Stderr, "warning: no .jsonl files in %s, skipping\n", dir)
			continue
		}

		var seriesRecords []taggedRecord
		outFile := filepath.Join(outputDir, alias+".jsonl")
		out, err := os.Create(outFile)
		if err != nil {
			return "", fmt.Errorf("create merged file %s: %w", outFile, err)
		}
		for _, f := range files {
			records, err := readJSONLFile(f)
			if err != nil {
				out.Close()
				return "", fmt.Errorf("read %s: %w", f, err)
			}
			enc := json.NewEncoder(out)
			for _, r := range records {
				if err := enc.Encode(r); err != nil {
					out.Close()
					return "", fmt.Errorf("write record to %s: %w", outFile, err)
				}
				tr := taggedRecord{source: alias, record: r}
				seriesRecords = append(seriesRecords, tr)
				allRecords = append(allRecords, tr)
			}
		}
		out.Close()

		// Write per-source full CSV
		if err := writeFullCSV(filepath.Join(fullDir, alias+".csv"), seriesRecords); err != nil {
			return "", fmt.Errorf("write full CSV for %s: %w", alias, err)
		}
		// Write per-source chunked CSVs
		if err := writeChunkedCSV(filepath.Join(chunkedDir, alias), seriesRecords); err != nil {
			return "", fmt.Errorf("write chunked CSV for %s: %w", alias, err)
		}
	}

	// Write combined full CSV
	if err := writeFullCSV(filepath.Join(fullDir, "merged.csv"), allRecords); err != nil {
		return "", fmt.Errorf("write merged full CSV: %w", err)
	}
	// Write combined chunked CSVs
	if err := writeChunkedCSV(filepath.Join(chunkedDir, "merged"), allRecords); err != nil {
		return "", fmt.Errorf("write merged chunked CSV: %w", err)
	}
	fmt.Fprintf(os.Stderr, "CSVs saved to: %s and %s\n", fullDir, chunkedDir)

	return GenerateVisualization(outputDir, concurrency)
}

type taggedRecord struct {
	source string
	record requestDataRecord
}

const csvMaxRows = 8000

var csvHeader = []string{
	"source", "time_offset_ms", "start_time", "end_time",
	"ttft_ms", "response_time_ms",
	"model", "series_guid", "series_num", "request_num",
	"cache_hit", "server_cache_confirmed", "is_cold_start",
	"input_tokens", "output_tokens", "cached_tokens", "local_cache_ratio",
	"is_error", "error_message", "is_empty",
}

func computeSourceStarts(records []taggedRecord) map[string]int64 {
	sourceStart := map[string]int64{}
	for _, tr := range records {
		t := tr.record.StartTime.UnixMilli()
		if prev, ok := sourceStart[tr.source]; !ok || t < prev {
			sourceStart[tr.source] = t
		}
	}
	return sourceStart
}

func recordToRow(tr taggedRecord, sourceStart map[string]int64) []string {
	r := tr.record
	offsetMs := r.StartTime.UnixMilli() - sourceStart[tr.source]
	return []string{
		tr.source,
		strconv.FormatInt(offsetMs, 10),
		r.StartTime.Format("2006-01-02T15:04:05.000Z07:00"),
		r.EndTime.Format("2006-01-02T15:04:05.000Z07:00"),
		strconv.FormatFloat(r.TTFT, 'f', 2, 64),
		strconv.FormatFloat(r.ResponseMs, 'f', 2, 64),
		r.Model,
		r.SeriesGUID,
		strconv.Itoa(r.SeriesNum),
		strconv.Itoa(r.RequestNum),
		strconv.FormatBool(r.CacheHit),
		strconv.FormatBool(r.ServerCacheConfirmed),
		strconv.FormatBool(r.IsColdStart),
		strconv.Itoa(r.InputTokens),
		strconv.Itoa(r.OutputTokens),
		strconv.Itoa(r.CachedTokens),
		strconv.FormatFloat(r.LocalCacheRatio, 'f', 6, 64),
		strconv.FormatBool(r.IsError),
		r.ErrorMessage,
		strconv.FormatBool(r.IsEmpty),
	}
}

// writeFullCSV writes all records into a single CSV file.
func writeFullCSV(path string, records []taggedRecord) error {
	sourceStart := computeSourceStarts(records)
	return writeSingleCSV(path, records, sourceStart)
}

// writeChunkedCSV writes records split into files of up to csvMaxRows rows.
// basePath is used without extension: produces basePath.csv or basePath_part1.csv, etc.
func writeChunkedCSV(basePath string, records []taggedRecord) error {
	sourceStart := computeSourceStarts(records)

	if len(records) <= csvMaxRows {
		return writeSingleCSV(basePath+".csv", records, sourceStart)
	}

	part := 1
	for i := 0; i < len(records); i += csvMaxRows {
		end := i + csvMaxRows
		if end > len(records) {
			end = len(records)
		}
		partPath := fmt.Sprintf("%s_part%d.csv", basePath, part)
		if err := writeSingleCSV(partPath, records[i:end], sourceStart); err != nil {
			return err
		}
		part++
	}
	return nil
}

func writeSingleCSV(path string, records []taggedRecord, sourceStart map[string]int64) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write(csvHeader); err != nil {
		return err
	}
	for _, tr := range records {
		if err := w.Write(recordToRow(tr, sourceStart)); err != nil {
			return err
		}
	}
	return nil
}

// FilterMergedDirs filters out directories with names starting with "merged"
// from a list, to avoid including output directories as input.
func FilterMergedDirs(dirs []string) []string {
	var filtered []string
	for _, d := range dirs {
		if !strings.HasPrefix(filepath.Base(d), "merged") {
			filtered = append(filtered, d)
		}
	}
	return filtered
}
