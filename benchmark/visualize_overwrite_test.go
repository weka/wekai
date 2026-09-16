package benchmark

import (
	"path/filepath"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// --overwrite: write dir/report.html in place instead of the default
// versioned report.html, report_v2.html, ... naming (see reportHTMLPath /
// nextVersionedPath in visualize.go). Exercises both the plain visualize
// path (GenerateVisualizationWithOverwrite) and the merge path
// (GenerateVisualizationMergedWithOverwrite).
// ---------------------------------------------------------------------------

// globReportHTMLs returns the base names of every report*.html file directly
// in dir, sorted by filepath.Glob's own (lexical) order.
func globReportHTMLs(t *testing.T, dir string) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, "report*.html"))
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(matches))
	for i, m := range matches {
		names[i] = filepath.Base(m)
	}
	return names
}

// TestGenerateVisualizationOverwriteWritesReportInPlace: with overwrite=true,
// two renders into the same directory produce only report.html -- no
// report_v2.html accumulates, and both calls return the same path.
func TestGenerateVisualizationOverwriteWritesReportInPlace(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("OnlyArm", base, 10, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, "overwrite-run-1")

	wantPath := filepath.Join(dir, "report.html")

	path1, err := GenerateVisualizationWithOverwrite(dir, 8, 0, true)
	if err != nil {
		t.Fatalf("first render: %v", err)
	}
	if path1 != wantPath {
		t.Fatalf("first render path = %q, want %q", path1, wantPath)
	}

	path2, err := GenerateVisualizationWithOverwrite(dir, 8, 0, true)
	if err != nil {
		t.Fatalf("second render: %v", err)
	}
	if path2 != wantPath {
		t.Fatalf("second render path = %q, want %q", path2, wantPath)
	}

	if got := globReportHTMLs(t, dir); len(got) != 1 || got[0] != "report.html" {
		t.Errorf("report*.html in %s = %v, want exactly [report.html]", dir, got)
	}
}

// TestGenerateVisualizationDefaultVersionsReport: without --overwrite
// (overwrite=false, today's default), a second render into the same
// directory produces report_v2.html alongside the untouched report.html --
// locks in the pre-existing versioned-naming behavior as unchanged.
func TestGenerateVisualizationDefaultVersionsReport(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("OnlyArm", base, 10, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, "overwrite-run-2")

	path1, err := GenerateVisualizationWithOverwrite(dir, 8, 0, false)
	if err != nil {
		t.Fatalf("first render: %v", err)
	}
	if want := filepath.Join(dir, "report.html"); path1 != want {
		t.Fatalf("first render path = %q, want %q", path1, want)
	}

	path2, err := GenerateVisualizationWithOverwrite(dir, 8, 0, false)
	if err != nil {
		t.Fatalf("second render: %v", err)
	}
	if want := filepath.Join(dir, "report_v2.html"); path2 != want {
		t.Fatalf("second render path = %q, want %q", path2, want)
	}

	got := globReportHTMLs(t, dir)
	if len(got) != 2 || got[0] != "report.html" || got[1] != "report_v2.html" {
		t.Errorf("report*.html in %s = %v, want [report.html report_v2.html]", dir, got)
	}
}

// TestGenerateVisualizationMergedOverwriteWritesReportInPlace: the merge
// path's --overwrite behaves identically to the plain visualize path -- two
// renders into the same merged output directory produce only report.html.
func TestGenerateVisualizationMergedOverwriteWritesReportInPlace(t *testing.T) {
	root := t.TempDir()
	dirA := filepath.Join(root, "a")
	dirB := filepath.Join(root, "b")
	outDir := filepath.Join(root, "merged")
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)

	recsA, samplesA := buildPublicFixtureArm("ArmA", base, 10, 0)
	recsB, samplesB := buildPublicFixtureArm("ArmB", base, 10, 0)
	writePublicFixtureFile(t, dirA, "a", recsA, samplesA, "merge-overwrite-run-a")
	writePublicFixtureFile(t, dirB, "b", recsB, samplesB, "merge-overwrite-run-b")

	wantPath := filepath.Join(outDir, "report.html")

	path1, err := GenerateVisualizationMergedWithOverwrite([]string{dirA, dirB}, []string{"arm-a", "arm-b"}, outDir, 8, 0, "", true)
	if err != nil {
		t.Fatalf("first merged render: %v", err)
	}
	if path1 != wantPath {
		t.Fatalf("first merged render path = %q, want %q", path1, wantPath)
	}

	path2, err := GenerateVisualizationMergedWithOverwrite([]string{dirA, dirB}, []string{"arm-a", "arm-b"}, outDir, 8, 0, "", true)
	if err != nil {
		t.Fatalf("second merged render: %v", err)
	}
	if path2 != wantPath {
		t.Fatalf("second merged render path = %q, want %q", path2, wantPath)
	}

	if got := globReportHTMLs(t, outDir); len(got) != 1 || got[0] != "report.html" {
		t.Errorf("report*.html in %s = %v, want exactly [report.html]", outDir, got)
	}
}

// TestGenerateVisualizationMergedDefaultVersionsReport: the merge path's
// default (overwrite=false) is byte-for-byte GenerateVisualizationMerged's
// pre-existing versioned-naming behavior.
func TestGenerateVisualizationMergedDefaultVersionsReport(t *testing.T) {
	root := t.TempDir()
	dirA := filepath.Join(root, "a")
	dirB := filepath.Join(root, "b")
	outDir := filepath.Join(root, "merged")
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)

	recsA, samplesA := buildPublicFixtureArm("ArmA", base, 10, 0)
	recsB, samplesB := buildPublicFixtureArm("ArmB", base, 10, 0)
	writePublicFixtureFile(t, dirA, "a", recsA, samplesA, "merge-version-run-a")
	writePublicFixtureFile(t, dirB, "b", recsB, samplesB, "merge-version-run-b")

	path1, err := GenerateVisualizationMerged([]string{dirA, dirB}, []string{"arm-a", "arm-b"}, outDir, 8, 0, "")
	if err != nil {
		t.Fatalf("first merged render: %v", err)
	}
	if want := filepath.Join(outDir, "report.html"); path1 != want {
		t.Fatalf("first merged render path = %q, want %q", path1, want)
	}

	path2, err := GenerateVisualizationMerged([]string{dirA, dirB}, []string{"arm-a", "arm-b"}, outDir, 8, 0, "")
	if err != nil {
		t.Fatalf("second merged render: %v", err)
	}
	if want := filepath.Join(outDir, "report_v2.html"); path2 != want {
		t.Fatalf("second merged render path = %q, want %q", path2, want)
	}
}
