package benchmark

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Baseline-arm selection for the merged/interactive report's summary %
// comparison (findBaselineIndex() in visualize.go). Rules, in order:
//   1. --baseline=<label> (visualize-merge only, threaded through as
//      generateVisualization's baselineLabel -> BASELINE_LABEL), matched
//      exactly against an arm's DISPLAYED name, always wins -- even over an
//      arm that would otherwise be the slowest. An unmatched --baseline is a
//      Go-side error from generateVisualization, not a silent fallback.
//   2. Otherwise the SLOWEST arm (fewest completed requests over the run's
//      full window) is the baseline -- the DEFAULT for every multi-arm
//      report, regardless of arm naming (classifyAlias is sort/color only).
//   3. A single-arm report has no baseline (nothing to compare against).
//
// These tests drive the actual generated report.html under node (same
// harness as the zoom-invariance/export tests above in
// visualize_public_test.go), reading back the already-populated summary
// table's own JS state (BASELINE_INDEX, DATA, sumRatios) rather than
// re-implementing the selection logic in Go.
// ---------------------------------------------------------------------------

// buildSpacedFixtureArm is buildPublicFixtureArm with the inter-request
// spacing parameterized, so a test can hold the completed-request COUNT
// fixed across two arms while still giving them different spanSec (and thus
// different mean req/s) -- needed to exercise the count-tie / rps-tiebreak
// path, which buildPublicFixtureArm's fixed 20s spacing can't produce.
func buildSpacedFixtureArm(alias string, base time.Time, nReq int, spacing time.Duration) []requestDataRecord {
	model := pubDynamicModel(alias)
	var records []requestDataRecord
	for i := 0; i < nReq; i++ {
		st := base.Add(time.Duration(i) * spacing)
		records = append(records, requestDataRecord{
			StartTime:    st,
			EndTime:      st.Add(2 * time.Second),
			TTFT:         120,
			ResponseMs:   1800,
			Model:        model,
			SeriesGUID:   "sess-" + pubSecretGUIDFrag + ":inst-" + alias + "-" + strconv.Itoa(i),
			SeriesNum:    i/2 + 1,
			RequestNum:   i + 1,
			InputTokens:  100,
			CachedTokens: 400,
			OutputTokens: 50,
		})
	}
	return records
}

type baselineProbeResult struct {
	BaselineIndex int      `json:"baselineIndex"`
	BaselineName  string   `json:"baselineName"`
	Names         []string `json:"names"`
	HasRatios     bool     `json:"hasRatios"`
	// Ratios[i][j] is the rendered ratio-to-baseline text for arm i,
	// metric j (empty string for the baseline's own row/no-comparison).
	Ratios [][]string `json:"ratios"`
}

// runBaselineProbe generates dir's interactive report.html (calling
// generateVisualization directly, same as GenerateVisualization does,
// but exposing baselineLabel -- the plain public entry point always passes
// "", so a --baseline test needs the unexported function), loads it under
// node, and reads back the summary panel's own JS state -- BASELINE_INDEX/
// DATA (the selection) and the already-rendered sumRatios cells (proof the
// selection actually drives the on-screen % comparison, not just an
// internal index).
func runBaselineProbe(t *testing.T, dir string, concurrency int, baselineLabel string) baselineProbeResult {
	t.Helper()
	nodeBin := nodeOrSkip(t)
	htmlPath, err := generateVisualization(dir, concurrency, false, 0, baselineLabel)
	if err != nil {
		t.Fatalf("generateVisualization: %v", err)
	}
	script := extractOuterScript(t, htmlPath)
	probe := `
const __names = DATA.map(d => d.name);
const __hasRatios = document.getElementById("summaryTable").className.indexOf("has-ratios") >= 0;
const __ratios = sumRatios.map(row => row.map(el => el.textContent));
const __result = {
  baselineIndex: BASELINE_INDEX,
  baselineName: BASELINE_INDEX >= 0 ? DATA[BASELINE_INDEX].name : "",
  names: __names,
  hasRatios: __hasRatios,
  ratios: __ratios,
};
console.log("===BASELINE_PROBE_START===");
console.log(JSON.stringify(__result));
console.log("===BASELINE_PROBE_END===");
`
	jsPath := filepath.Join(t.TempDir(), "baseline_probe.js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	rawOut, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil {
		t.Fatalf("node baseline probe failed: %v\n%s", err, rawOut)
	}
	out := string(rawOut)
	const startMarker = "===BASELINE_PROBE_START===\n"
	const endMarker = "\n===BASELINE_PROBE_END==="
	si := strings.Index(out, startMarker)
	ei := strings.Index(out, endMarker)
	if si < 0 || ei < 0 || ei <= si {
		t.Fatalf("could not locate baseline-probe markers in node output:\n%s", out)
	}
	var result baselineProbeResult
	if err := json.Unmarshal([]byte(out[si+len(startMarker):ei]), &result); err != nil {
		t.Fatalf("decode baseline-probe result: %v\nraw: %s", err, out[si+len(startMarker):ei])
	}
	return result
}

// TestBaselineSlowestArmIsFallback: with no --baseline override, the arm
// with fewer completed requests over the run's full window is the default
// baseline -- covers a comparison between two offload configs, neither a
// no-offload control, which previously got no ratio column at all.
func TestBaselineSlowestArmIsFallback(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	fastRecs, fastSamples := buildPublicFixtureArm("FastArm", base, 20, 0)
	slowRecs, slowSamples := buildPublicFixtureArm("SlowArm", base, 5, 0)
	writePublicFixtureFile(t, dir, "a", fastRecs, fastSamples, pubSecretRunID+"-fast")
	writePublicFixtureFile(t, dir, "b", slowRecs, slowSamples, pubSecretRunID+"-slow")

	res := runBaselineProbe(t, dir, 8, "")
	if res.BaselineIndex < 0 {
		t.Fatalf("expected a baseline to be selected (2 arms present, no --baseline), got BASELINE_INDEX=-1; names=%v", res.Names)
	}
	if res.BaselineName != "SlowArm" {
		t.Errorf("baseline = %q, want %q (the arm with fewer completed requests)", res.BaselineName, "SlowArm")
	}
	if !res.HasRatios {
		t.Errorf("summaryTable missing has-ratios class despite a baseline being selected")
	}
	// The non-baseline row (FastArm) must carry a non-empty ratio on at
	// least one metric -- proves the selection actually drives rendering,
	// not just the internal index.
	fastIdx := -1
	for i, n := range res.Names {
		if n == "FastArm" {
			fastIdx = i
		}
	}
	if fastIdx < 0 {
		t.Fatalf("FastArm missing from rendered DATA names %v", res.Names)
	}
	if !anyNonEmpty(res.Ratios[fastIdx]) {
		t.Errorf("FastArm's row has no rendered ratio-to-baseline text; ratios=%v", res.Ratios[fastIdx])
	}
}

// TestBaselineExplicitFlagOverrideWins: --baseline=<label> picks that arm as
// the baseline even when it is NOT the slowest arm present -- the explicit
// override takes precedence over the default slowest-arm rule. Arm naming
// (e.g. "hbm") plays no part any more -- classifyAlias is sort/color only.
func TestBaselineExplicitFlagOverrideWins(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	// FastArm completes MORE requests than SlowArm, so it would lose to
	// SlowArm under the default slowest-arm rule alone -- --baseline=FastArm
	// must still pick FastArm.
	fastRecs, fastSamples := buildPublicFixtureArm("FastArm", base, 20, 0)
	slowRecs, slowSamples := buildPublicFixtureArm("SlowArm", base, 5, 0)
	writePublicFixtureFile(t, dir, "a", fastRecs, fastSamples, pubSecretRunID+"-fast")
	writePublicFixtureFile(t, dir, "b", slowRecs, slowSamples, pubSecretRunID+"-slow")

	res := runBaselineProbe(t, dir, 8, "FastArm")
	if res.BaselineName != "FastArm" {
		t.Errorf("baseline = %q, want %q (--baseline must win over the default slowest-arm rule)", res.BaselineName, "FastArm")
	}
	if !res.HasRatios {
		t.Errorf("summaryTable missing has-ratios class despite an explicit --baseline")
	}
}

// TestBaselineFlagUnmatchedNameErrors: --baseline naming an arm that isn't
// present must fail the CLI command (generateVisualization returns an
// error), not silently render without the override.
func TestBaselineFlagUnmatchedNameErrors(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	fastRecs, fastSamples := buildPublicFixtureArm("FastArm", base, 20, 0)
	slowRecs, slowSamples := buildPublicFixtureArm("SlowArm", base, 5, 0)
	writePublicFixtureFile(t, dir, "a", fastRecs, fastSamples, pubSecretRunID+"-fast2")
	writePublicFixtureFile(t, dir, "b", slowRecs, slowSamples, pubSecretRunID+"-slow2")

	_, err := generateVisualization(dir, 8, false, 0, "NoSuchArm")
	if err == nil {
		t.Fatal("expected an error for a --baseline value matching no arm, got nil")
	}
	if !strings.Contains(err.Error(), "NoSuchArm") {
		t.Errorf("error %q does not name the unmatched --baseline value", err.Error())
	}
}

// TestBaselineTieBreaksOnMeanRps: two arms complete the SAME number of
// requests but over different spans (different mean req/s) -- the lower
// mean-rps arm is the baseline.
func TestBaselineTieBreaksOnMeanRps(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	// Same request COUNT (10) for both, but "Leisurely" is spread over 5x
	// the wall-clock time of "Brisk" -- same completed count, lower rps.
	briskRecs := buildSpacedFixtureArm("Brisk", base, 10, 5*time.Second)
	leisurelyRecs := buildSpacedFixtureArm("Leisurely", base, 10, 25*time.Second)
	writePublicFixtureFile(t, dir, "a", briskRecs, nil, pubSecretRunID+"-brisk")
	writePublicFixtureFile(t, dir, "b", leisurelyRecs, nil, pubSecretRunID+"-leisurely")

	res := runBaselineProbe(t, dir, 8, "")
	if res.BaselineName != "Leisurely" {
		t.Errorf("baseline = %q, want %q (equal completed-request counts, tie must break on lower mean req/s)", res.BaselineName, "Leisurely")
	}
}

// TestBaselineSingleArmNoComparison: a single-arm report has nothing to
// compare against -- no baseline, no ratio column.
func TestBaselineSingleArmNoComparison(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 9, 14, 9, 0, 0, 0, time.UTC)
	recs, samples := buildPublicFixtureArm("OnlyArm", base, 10, 0)
	writePublicFixtureFile(t, dir, "a", recs, samples, pubSecretRunID+"-only")

	res := runBaselineProbe(t, dir, 8, "")
	if res.BaselineIndex != -1 {
		t.Errorf("BASELINE_INDEX = %d, want -1 for a single-arm report (name=%q)", res.BaselineIndex, res.BaselineName)
	}
	if res.HasRatios {
		t.Errorf("summaryTable carries has-ratios class for a single-arm report")
	}
	if len(res.Ratios) != 1 || anyNonEmpty(res.Ratios[0]) {
		t.Errorf("single-arm report rendered a non-empty ratio cell; ratios=%v", res.Ratios)
	}
}

func anyNonEmpty(ss []string) bool {
	for _, s := range ss {
		if s != "" {
			return true
		}
	}
	return false
}
