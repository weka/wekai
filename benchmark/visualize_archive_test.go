package benchmark

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// extractReportScript pulls the single <script>...</script> block out of a
// generated report.html, the same way visualize_samples_test.go's JS-driving
// tests do (see e.g. TestSummaryPanelJS). Factored here because this file
// needs it three times.
func extractReportScript(t *testing.T, html string) string {
	t.Helper()
	start := strings.Index(html, "<script>")
	end := strings.Index(html, "</script>")
	if start < 0 || end < 0 || end <= start {
		t.Fatal("script block not found")
	}
	return html[start+len("<script>") : end]
}

// runNodeProbe writes reportDOMStub + the report's own script + probe to a
// temp .js file and runs it under node, failing the test unless it prints
// ALL_OK. Mirrors the exec pattern in visualize_samples_test.go's JS tests.
func runNodeProbe(t *testing.T, nodeBin, dir, name, script, probe string) {
	t.Helper()
	jsPath := filepath.Join(dir, name+".js")
	if err := os.WriteFile(jsPath, []byte(reportDOMStub+"\n"+script+"\n"+probe), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(nodeBin, jsPath).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL_OK") {
		t.Fatalf("node probe %q failed: %v\n%s", name, err, out)
	}
}

// ---------------------------------------------------------------------------
// TEST 1: series_guid interning with multiple instances per series_num.
//
// This is the regression test for a real bug: the original design keyed a
// series_num -> guid map on the false premise that series_guid is
// per-series. It is not -- seriesNum is per SESSION (replay_router.go:677)
// while SeriesGUID = sessionID + ":" + instanceID is per INSTANCE
// (replay_router.go:926), and one session runs every one of its instances
// under a single series_num. So N distinct GUIDs sharing one series_num is
// the NORMAL router-replay case, not an anomaly. The broken version
// collapsed them into a "|"-joined string, so no row carried its own GUID
// and a CSV row could no longer be joined back to its raw JSONL record.
//
// The fixture below has 3 series_num values, with 2/3/2 records carrying
// DISTINCT series_guids, plus one more record with NO guid at all (legacy
// data) to prove that case exports empty rather than a stray table entry.
// ---------------------------------------------------------------------------

func TestSeriesGUIDInterningJS(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 8, 1, 9, 0, 0, 0, time.UTC)
	model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=guidintern"

	// (seriesNum, guid) pairs in fixture/insertion order. "" means no guid at
	// all -- e.g. a record written before series_guid existed.
	pairs := []struct {
		sn   int
		guid string
	}{
		{1, "sess-1:inst-0"},
		{1, "sess-1:inst-1"},
		{2, "sess-2:inst-0"},
		{2, "sess-2:inst-1"},
		{2, "sess-2:inst-2"},
		{3, "sess-3:inst-0"},
		{3, "sess-3:inst-1"},
		{4, ""},
	}

	var records []requestDataRecord
	for i, p := range pairs {
		st := base.Add(time.Duration(i) * 5 * time.Second)
		records = append(records, requestDataRecord{
			StartTime:    st,
			EndTime:      st.Add(time.Second),
			Model:        model,
			SeriesGUID:   p.guid,
			SeriesNum:    p.sn,
			RequestNum:   i + 1,
			InputTokens:  10,
			CachedTokens: 0,
			OutputTokens: 5,
		})
	}
	writeMixedJSONL(t, dir, "a", records, nil)

	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}

	// Sanity: no raw "|"-joined collapse anywhere in the embedded RAW_DATA
	// (the exact shape the bug produced -- joining sibling instance GUIDs
	// with "|" into one string).
	html := string(b)
	for _, guid := range []string{"sess-1:inst-0|sess-1:inst-1", "sess-2:inst-0|sess-2:inst-1"} {
		if strings.Contains(html, guid) {
			t.Errorf("report embeds a pipe-joined guid collapse: %q", guid)
		}
	}

	// Direct check on the pure interning function itself, independent of the
	// JS/CSV path below -- keyed on the GUID string, never on series_num, so
	// records sharing a series_num keep their own distinct guid/index.
	t.Run("buildGuidTable direct", func(t *testing.T) {
		table, idx := buildGuidTable(records)
		wantTable := []string{
			"sess-1:inst-0", "sess-1:inst-1",
			"sess-2:inst-0", "sess-2:inst-1", "sess-2:inst-2",
			"sess-3:inst-0", "sess-3:inst-1",
		}
		if !reflect.DeepEqual(table, wantTable) {
			t.Fatalf("guid table = %v, want %v", table, wantTable)
		}
		for i, p := range pairs {
			if p.guid == "" {
				if idx[i] != -1 {
					t.Errorf("record %d (no guid): idx = %d, want -1", i, idx[i])
				}
				continue
			}
			if table[idx[i]] != p.guid {
				t.Errorf("record %d: idx %d resolves to %q, want %q", i, idx[i], table[idx[i]], p.guid)
			}
		}
	})

	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS series_guid CSV test skipped")
	}
	script := extractReportScript(t, html)

	probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
const rows = buildRequestsRows();
const header = rows[0].split(",");
const giIdx = header.indexOf("series_guid");
const snIdx = header.indexOf("series_num");
assert(giIdx >= 0 && snIdx >= 0, "series_guid/series_num columns present");

const dataRows = rows.slice(1).map(r => r.split(","));
const expectedGuids = [
  "sess-1:inst-0", "sess-1:inst-1",
  "sess-2:inst-0", "sess-2:inst-1", "sess-2:inst-2",
  "sess-3:inst-0", "sess-3:inst-1",
  "",
];
assert(dataRows.length === expectedGuids.length,
  "row count " + dataRows.length + ", want " + expectedGuids.length);

dataRows.forEach((cols, i) => {
  const got = cols[giIdx];
  assert(got === expectedGuids[i],
    "row " + i + " series_guid: want " + JSON.stringify(expectedGuids[i]) + ", got " + JSON.stringify(got));
  assert(got.indexOf("|") === -1, "row " + i + " series_guid must never contain a pipe-joined collapse: " + got);
});

// The last row (series_num=4) has no guid at all -- must export empty, not
// some stray table entry (e.g. an out-of-range index resolving to garbage).
assert(dataRows[dataRows.length - 1][giIdx] === "", "no-guid record exports empty series_guid");

// GuidTable holds exactly the distinct guids (first-seen order), and every
// record's own GuidIdx resolves back to ITS OWN guid -- never a table entry
// shared across the whole series_num group.
const s = DATA[0];
const distinct = expectedGuids.filter(g => g !== "");
const seen = [];
distinct.forEach(g => { if (seen.indexOf(g) < 0) seen.push(g); });
assert(JSON.stringify(s.guidTable) === JSON.stringify(seen),
  "guidTable = " + JSON.stringify(s.guidTable) + ", want " + JSON.stringify(seen));

s.records.forEach((r, i) => {
  if (expectedGuids[i] === "") {
    assert(r.gi === -1, "record " + i + " (no guid) must have gi=-1, got " + r.gi);
  } else {
    assert(s.guidTable[r.gi] === expectedGuids[i],
      "record " + i + " GuidIdx " + r.gi + " resolves to " + JSON.stringify(s.guidTable[r.gi]) +
      ", want " + JSON.stringify(expectedGuids[i]));
  }
});
console.log("ALL_OK");
`
	runNodeProbe(t, nodeBin, dir, "guid_intern_test", script, probe)
}

// ---------------------------------------------------------------------------
// TEST 2: CSV column contract, both exports.
//
// Pinned as literal, ordered, expected header slices so any reordering or
// accidental drop of a column fails loudly instead of silently shipping a
// misaligned export.
//
// WHY this is pinned exactly: the two exports (the report's in-report
// "download requests CSV" -- buildRequestsRows() in visualize.go -- and the
// on-disk merged CSV -- csvHeader/recordToRow in visualize_merge.go) were
// deliberately brought to column-order PARITY on tokens: both list
// input_tokens, output_tokens, cached_tokens in that order (an earlier
// version of the code disagreed on this ordering between the two exports).
// Separately, every NEW column the merge CSV gained (turn, cache_hit_ratio,
// and the whole uuid_* family) was deliberately APPENDED AFTER the original
// 20 columns rather than inserted next to its sibling field in
// requestDataRecord, specifically so an existing positional reader of
// merged.csv (source..is_empty at their original offsets) keeps working. A
// future edit that inserts a new column into the middle of either table
// -- shifting everything after it -- must fail one of these two checks.
// ---------------------------------------------------------------------------

func TestRequestCSVColumnContract(t *testing.T) {
	// Pinned literal expectations. Keep these as exact, ordered strings --
	// that is the entire point of this test.
	wantMergedHeader := []string{
		"source", "time_offset_ms", "start_time", "end_time",
		"ttft_ms", "response_time_ms",
		"model", "series_guid", "series_num", "request_num",
		"cache_hit", "server_cache_confirmed", "is_cold_start",
		"input_tokens", "output_tokens", "cached_tokens", "local_cache_ratio",
		"is_error", "error_message", "is_empty",
		"turn", "cache_hit_ratio",
		"uuid_expected", "uuid_found", "uuid_leaked", "uuid_exact_match",
		"expected_uuids_raw", "found_mask", "leaked_uuids_raw",
	}
	wantRequestsHeader := []string{
		"arm", "start_time", "series_guid", "series_num", "request_num", "turn",
		"ttft_ms", "response_time_ms", "cache_hit", "cache_hit_ratio", "server_cache_confirmed",
		"is_cold_start", "input_tokens", "output_tokens", "cached_tokens", "local_cache_ratio",
		"is_error", "is_empty",
		"uuid_expected", "uuid_found", "uuid_leaked", "uuid_exact_match",
		"expected_uuids_raw", "found_mask", "leaked_uuids_raw",
	}

	// Deliberate parity check called out in the doc above: both lists carry
	// input_tokens, output_tokens, cached_tokens consecutively and in that
	// order.
	assertTokenOrder := func(t *testing.T, header []string, label string) {
		t.Helper()
		wantSeq := []string{"input_tokens", "output_tokens", "cached_tokens"}
		var got []string
		for _, col := range header {
			for _, w := range wantSeq {
				if col == w {
					got = append(got, col)
				}
			}
		}
		if !reflect.DeepEqual(got, wantSeq) {
			t.Errorf("%s: token column order = %v, want %v", label, got, wantSeq)
		}
	}
	assertTokenOrder(t, wantMergedHeader, "pinned merged header")
	assertTokenOrder(t, wantRequestsHeader, "pinned requests header")

	dir := t.TempDir()
	base := time.Date(2026, 8, 2, 9, 0, 0, 0, time.UTC)
	rec, smp := benchFixtureData("csvcontract", base)
	writeMixedJSONL(t, dir, "a", rec, smp)

	// --- On-disk merged CSV (visualize_merge.go), fully end-to-end: run the
	// real merge pipeline and read the header line actually written to
	// disk, not just the in-memory csvHeader var. ---
	t.Run("merged_csv_on_disk", func(t *testing.T) {
		outDir := filepath.Join(dir, "merged")
		if _, err := GenerateVisualizationMerged([]string{dir}, []string{"arm-a"}, outDir, 4, 0); err != nil {
			t.Fatalf("merge: %v", err)
		}
		f, err := os.Open(filepath.Join(outDir, "full_csv", "merged.csv"))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		got, err := csv.NewReader(f).Read()
		if err != nil {
			t.Fatalf("read merged.csv header: %v", err)
		}
		if !reflect.DeepEqual(got, wantMergedHeader) {
			t.Fatalf("merged.csv header =\n%v\nwant\n%v", got, wantMergedHeader)
		}
		// Also pin the in-memory var directly: it is what recordToRow's
		// positions are keyed against, so a drift here is the root cause of
		// any drift on disk.
		if !reflect.DeepEqual(csvHeader, wantMergedHeader) {
			t.Fatalf("csvHeader var =\n%v\nwant\n%v", csvHeader, wantMergedHeader)
		}
	})

	// --- In-report CSV (visualize.go's buildRequestsRows), driven under
	// node exactly like the JSONL fixture would produce it. ---
	t.Run("in_report_csv_js", func(t *testing.T) {
		nodeBin, err := exec.LookPath("node")
		if err != nil {
			t.Skip("node not installed; JS requests-CSV header test skipped")
		}
		htmlPath, err := GenerateVisualization(dir, 4)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			t.Fatal(err)
		}
		script := extractReportScript(t, string(b))
		wantHeaderJSON, err := json.Marshal(wantRequestsHeader)
		if err != nil {
			t.Fatal(err)
		}
		probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
const want = ` + string(wantHeaderJSON) + `;
const rows = buildRequestsRows();
const got = rows[0].split(",");
assert(JSON.stringify(got) === JSON.stringify(want),
  "requests CSV header =\n" + JSON.stringify(got) + "\nwant\n" + JSON.stringify(want));
console.log("ALL_OK");
`
		runNodeProbe(t, nodeBin, dir, "requests_header_test", script, probe)
	})
}

// ---------------------------------------------------------------------------
// TEST 3: cache_hit_ratio is derived client-side and matches what the JSONL
// held; local_cache_ratio survives the 4-decimal rounding done at emit time.
//
// vizRecord no longer stores CacheHitRatio (see its field doc in
// visualize.go) -- the report's deriveCacheHitRatio() recomputes it as
// EXACTLY the same formula auto.go/replay.go use when first writing the
// JSONL's cache_hit_ratio field: cached/(input+cached) when either is
// non-zero, else 1 if cache_hit else 0 (the TTFT-heuristic-only path, where
// no usage was reported at all -- see deriveCacheHitRatio's own comment and
// auto.go's usageReported/cacheHitRatio computation around the
// runSingleModelBenchmark request loop). Each fixture row below sets
// CacheHitRatio to what that production formula would actually have
// written, so this test proves the client-side re-derivation reconstructs
// the stored value losslessly rather than merely computing "a" plausible
// number.
// ---------------------------------------------------------------------------

func TestCacheHitRatioAndLocalCacheRatioJS(t *testing.T) {
	nodeBin, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; JS cache_hit_ratio test skipped")
	}

	dir := t.TempDir()
	base := time.Date(2026, 8, 3, 9, 0, 0, 0, time.UTC)
	model := "dynamic/http://localhost:8000/v1,type=openai_vllm,alias=chr"

	type row struct {
		in, ca          int
		ch              bool
		cacheHitRatio   float64
		localCacheRatio float64
		label           string
	}
	rows := []row{
		// Normal partial hit: usage reported, ratio = ca/(in+ca).
		{in: 100, ca: 400, ch: true, cacheHitRatio: 400.0 / 500.0, localCacheRatio: 0.123456789, label: "partial-hit"},
		// cached == 0: usage reported (denom>0), ratio is simply 0.
		{in: 500, ca: 0, ch: false, cacheHitRatio: 0, localCacheRatio: 0, label: "zero-cached"},
		// The 0/0 case, cache_hit true: no usage reported at all (denom==0),
		// so the TTFT heuristic alone fired the hit -- ratio is 1, per
		// deriveCacheHitRatio's documented fallback and auto.go's
		// usageReported=false / cacheHit=>ratio=1 path.
		{in: 0, ca: 0, ch: true, cacheHitRatio: 1, localCacheRatio: 0, label: "zero-zero-hit"},
		// The 0/0 case, cache_hit false: same denom==0 fallback, but the
		// heuristic did not fire -- ratio is 0.
		{in: 0, ca: 0, ch: false, cacheHitRatio: 0, localCacheRatio: 0, label: "zero-zero-miss"},
	}

	var records []requestDataRecord
	for i, r := range rows {
		st := base.Add(time.Duration(i) * 10 * time.Second)
		records = append(records, requestDataRecord{
			StartTime: st, EndTime: st.Add(time.Second),
			Model: model, SeriesNum: 1, RequestNum: i + 1,
			InputTokens: r.in, CachedTokens: r.ca, OutputTokens: 5,
			CacheHit: r.ch, CacheHitRatio: r.cacheHitRatio,
			LocalCacheRatio: r.localCacheRatio,
		})
	}
	writeMixedJSONL(t, dir, "a", records, nil)

	htmlPath, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	b, err := os.ReadFile(htmlPath)
	if err != nil {
		t.Fatal(err)
	}
	script := extractReportScript(t, string(b))

	type expectRow struct {
		Label           string  `json:"label"`
		CacheHitRatio   float64 `json:"cacheHitRatio"`
		LocalCacheRatio float64 `json:"localCacheRatio"`
	}
	var expected []expectRow
	for _, r := range rows {
		expected = append(expected, expectRow{Label: r.label, CacheHitRatio: r.cacheHitRatio, LocalCacheRatio: round4(r.localCacheRatio)})
	}
	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		t.Fatal(err)
	}

	probe := `
function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }
const expected = ` + string(expectedJSON) + `;
const rows = buildRequestsRows();
const header = rows[0].split(",");
const chrIdx = header.indexOf("cache_hit_ratio");
const lcrIdx = header.indexOf("local_cache_ratio");
assert(chrIdx >= 0 && lcrIdx >= 0, "cache_hit_ratio/local_cache_ratio columns present");

const dataRows = rows.slice(1).map(r => r.split(","));
assert(dataRows.length === expected.length, "row count " + dataRows.length + ", want " + expected.length);

dataRows.forEach((cols, i) => {
  const exp = expected[i];
  const gotChr = parseFloat(cols[chrIdx]);
  assert(Math.abs(gotChr - exp.cacheHitRatio) < 1e-9,
    exp.label + ": derived cache_hit_ratio = " + gotChr + ", want " + exp.cacheHitRatio +
    " (must equal what the JSONL's cache_hit_ratio field held)");

  const gotLcr = parseFloat(cols[lcrIdx]);
  assert(Math.abs(gotLcr - exp.localCacheRatio) < 1e-9,
    exp.label + ": local_cache_ratio = " + gotLcr + ", want " + exp.localCacheRatio + " (4dp rounding)");
});

// Also verify directly through the rehydrated record + deriveCacheHitRatio,
// independent of the CSV string round-trip.
const s = DATA[0];
s.records.forEach((r, i) => {
  const exp = expected[i];
  const d = deriveCacheHitRatio(r);
  assert(Math.abs(d - exp.cacheHitRatio) < 1e-9,
    exp.label + ": deriveCacheHitRatio(r) = " + d + ", want " + exp.cacheHitRatio);
});
console.log("ALL_OK");
`
	runNodeProbe(t, nodeBin, dir, "cache_hit_ratio_test", script, probe)
}

// ---------------------------------------------------------------------------
// BONUS (cheap, pure Go, no node needed): sparse UUID encoding round-trip.
//
// vizRecord.MarshalJSON omits the UUID validation quad entirely (not even a
// zero placeholder) on a record with no UUID data, and additionally omits
// the raw-detail slot unless there was an actual miss/leak. This directly
// exercises that contract on the Go struct: a record with nothing to say
// encodes strictly shorter than one carrying the quad, which in turn encodes
// shorter than one that also carries miss/leak forensics -- and the content
// that IS emitted round-trips losslessly.
// ---------------------------------------------------------------------------

func TestVizRecordSparseUUIDEncoding(t *testing.T) {
	base := vizRecord{T: 0, TTFT: 1, ResponseMs: 2, InputTokens: 3, CachedTokens: 4, OutputTokens: 5}

	noUUID := base
	marshalLen := func(t *testing.T, r vizRecord) []any {
		t.Helper()
		buf, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		var arr []any
		if err := json.Unmarshal(buf, &arr); err != nil {
			t.Fatal(err)
		}
		return arr
	}

	arrNone := marshalLen(t, noUUID)
	if len(arrNone) != 16 {
		t.Errorf("no-UUID record: array len = %d, want 16 (positions 0-15 only)", len(arrNone))
	}

	cleanMatch := base
	cleanMatch.UUIDExpected = 2
	cleanMatch.UUIDFound = 2
	cleanMatch.UUIDExactMatch = true
	arrQuad := marshalLen(t, cleanMatch)
	if len(arrQuad) != 20 {
		t.Errorf("clean-match record: array len = %d, want 20 (quad appended, no raw detail)", len(arrQuad))
	}

	withMiss := base
	withMiss.UUIDExpected = 2
	withMiss.UUIDFound = 1
	withMiss.ExpectedUUIDsRaw = []string{"uuid-a", "uuid-b"}
	withMiss.FoundMask = []bool{true, false}
	withMiss.LeakedUUIDsRaw = []string{"uuid(series=9)"}
	arrMiss := marshalLen(t, withMiss)
	if len(arrMiss) != 21 {
		t.Fatalf("miss/leak record: array len = %d, want 21 (quad + raw detail slot)", len(arrMiss))
	}
	detail, ok := arrMiss[20].([]any)
	if !ok || len(detail) != 3 {
		t.Fatalf("slot 20 = %#v, want a 3-element [expected, foundMask, leaked] array", arrMiss[20])
	}
	gotExpected, _ := detail[0].([]any)
	gotMask, _ := detail[1].([]any)
	gotLeaked, _ := detail[2].([]any)
	if len(gotExpected) != 2 || gotExpected[0] != "uuid-a" || gotExpected[1] != "uuid-b" {
		t.Errorf("expected_uuids_raw round-trip = %v, want [uuid-a uuid-b]", gotExpected)
	}
	// FoundMask bools are encoded as 0/1 ints (see MarshalJSON's maskBits).
	if len(gotMask) != 2 || gotMask[0] != float64(1) || gotMask[1] != float64(0) {
		t.Errorf("found_mask round-trip = %v, want [1 0]", gotMask)
	}
	if len(gotLeaked) != 1 || gotLeaked[0] != "uuid(series=9)" {
		t.Errorf("leaked_uuids_raw round-trip = %v, want [uuid(series=9)]", gotLeaked)
	}
}
