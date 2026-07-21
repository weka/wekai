package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeMixedJSONL(t *testing.T, dir, name string, records []requestDataRecord, samples []vllmMetricsSample) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, name+".jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range samples {
		if err := enc.Encode(s); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func benchFixtureData(alias string, base time.Time) ([]requestDataRecord, []vllmMetricsSample) {
	model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=" + alias
	var records []requestDataRecord
	for i := 0; i < 5; i++ {
		st := base.Add(time.Duration(i) * 20 * time.Second)
		records = append(records, requestDataRecord{
			StartTime:  st,
			EndTime:    st.Add(2 * time.Second),
			TTFT:       150,
			ResponseMs: 2000,
			Model:      model,
			SeriesNum:  1,
			RequestNum: i + 1,
		})
	}
	var samples []vllmMetricsSample
	for i := 0; i < 3; i++ {
		samples = append(samples, vllmMetricsSample{
			RecordType:          recordTypeVLLMMetricsSample,
			TS:                  base.Add(time.Duration(i) * 60 * time.Second),
			Model:               model,
			Sources:             vllmSourceCounters{Compute: int64(1000 * (i + 1)), LocalCache: int64(300 * i), ExternalCache: int64(100 * i)},
			ActiveDatasetTokens: int64(4000 * (i + 1)),
			ActiveSeries:        i + 1,
		})
	}
	return records, samples
}

func TestGenerateVisualizationWithCacheMixSamples(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	recA, smpA := benchFixtureData("mixA_weka", base)
	recB, smpB := benchFixtureData("mixB_hbm", base)
	writeMixedJSONL(t, dir, "a", recA, smpA)
	writeMixedJSONL(t, dir, "b", recB, smpB)

	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	for _, want := range []string{
		`"mix":`, `"adt":`, // per-series overlay data embedded
		"drawCacheMix", "showCacheMix", "HAS_CACHE_MIX", // overlay code paths
		"MIX_COMPUTE_COLOR", "active dataset (tokens)",
	} {
		if !strings.Contains(html, want) {
			t.Errorf("generated HTML missing %q", want)
		}
	}
	// Deltas, not cumulatives, are embedded: seg0 compute delta is 1000.
	if !strings.Contains(html, `"t0":`) {
		t.Errorf("mix segments not embedded")
	}
}

func TestGenerateVisualizationWithoutSamplesUnchanged(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	rec, _ := benchFixtureData("plain_run", base)
	writeMixedJSONL(t, dir, "a", rec, nil)

	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	html := string(b)
	// No samples => no overlay data keys in the embedded series JSON; the
	// runtime HAS_CACHE_MIX gate keeps the checkbox absent.
	if strings.Contains(html, `"mix":`) || strings.Contains(html, `"adt":`) {
		t.Errorf("sample-less dataset must not embed overlay data")
	}
}

func TestGenerateVisualizationMergedCarriesSamples(t *testing.T) {
	root := t.TempDir()
	base := time.Date(2026, 7, 21, 10, 0, 0, 0, time.UTC)
	dirA := filepath.Join(root, "armA")
	dirB := filepath.Join(root, "armB")
	recA, smpA := benchFixtureData("merged_weka", base)
	recB, smpB := benchFixtureData("merged_hbm", base)
	writeMixedJSONL(t, dirA, "reqs", recA, smpA)
	writeMixedJSONL(t, dirB, "reqs", recB, smpB)

	outDir := filepath.Join(root, "merged")
	htmlPath, err := GenerateVisualizationMerged([]string{dirA, dirB}, nil, outDir, 4)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}

	// Merged per-source JSONL must carry the sample rows through.
	mergedFiles, err := filepath.Glob(filepath.Join(outDir, "*.jsonl"))
	if err != nil || len(mergedFiles) != 2 {
		t.Fatalf("merged jsonl files: %v (%v)", mergedFiles, err)
	}
	for _, f := range mergedFiles {
		records, samples, err := readJSONLFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(records) != 5 || len(samples) != 3 {
			t.Errorf("%s: got %d records / %d samples, want 5/3", f, len(records), len(samples))
		}
	}

	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"mix":`) {
		t.Errorf("merged report missing cache-mix overlay data")
	}
}
