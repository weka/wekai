package benchmark

import "encoding/json"

// A report arm's requests travel as one table: a row per request, every row
// the same positions, and a column map ("cols") naming them. The report's
// own fields and, with --timeline, the timeline's are columns of the same
// rows, so nothing is stored twice: the timeline reads start, latency, TTFT,
// tokens, series, turn and guid from the report's columns.
//
// A column is a name and a width: 1 is a scalar, >1 a fixed-width group laid
// out inline, 0 one slot holding a variable-length array. A column whose value
// is empty in every row of the arm is left out of the map and of every row;
// otherwise every row carries it, empty or not, and readers default an absent
// column (0, false, empty; gi -1).
type reportCol struct {
	Name  string
	Width int
}

func (c reportCol) MarshalJSON() ([]byte, error) { return json.Marshal([]any{c.Name, c.Width}) }

// timelineMeta is the per-arm part of the timeline that is not per request.
type timelineMeta struct {
	Span   float64     `json:"span"`
	Groups int         `json:"groups"`
	Rates  *rateSeries `json:"rates,omitempty"`
	TTFT   *ttftSeries `json:"ttft,omitempty"`
}

// colDef produces one column's value for row i: a float64 (scalar), a
// []float64 of exactly width (width > 1), or, when nested, any JSON array,
// nil when empty.
type colDef struct {
	name   string
	width  int     // >1: a fixed-width group; otherwise a scalar or, when nested, an array
	nested bool    // one slot holding a variable-length array
	def    float64 // a scalar's empty value
	get    func(i int) any
}

func (c colDef) col() reportCol {
	switch {
	case c.nested:
		return reportCol{Name: c.name, Width: 0}
	case c.width > 1:
		return reportCol{Name: c.name, Width: c.width}
	}
	return reportCol{Name: c.name, Width: 1}
}

func b01(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

func ints(v []int, width int) []float64 {
	out := make([]float64, width)
	for i := 0; i < len(v) && i < width; i++ {
		out[i] = float64(v[i])
	}
	return out
}

func floats(v []float64, width int) []float64 {
	out := make([]float64, width)
	copy(out, v)
	return out
}

func (c colDef) empty(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case float64:
		return x == c.def
	case []float64:
		for _, f := range x {
			if f != 0 {
				return false
			}
		}
		return true
	case []any:
		return len(x) == 0
	}
	return false
}

// buildReportTable lays out an arm's records (and, when tl is non-nil, the
// timeline entry for each record, tl[i] belonging to vr[i]) as columns + rows.
func buildReportTable(vr []vizRecord, tl []*timelineReq) ([]reportCol, [][]any) {
	r := func(i int) *vizRecord { return &vr[i] }
	defs := []colDef{
		{name: "t", get: func(i int) any { return r(i).T }},
		{name: "ttft", get: func(i int) any { return r(i).TTFT }},
		{name: "resp", get: func(i int) any { return r(i).ResponseMs }},
		{name: "err", get: func(i int) any { return b01(r(i).IsError) }},
		{name: "sn", get: func(i int) any { return float64(r(i).SeriesNum) }},
		{name: "rn", get: func(i int) any { return float64(r(i).RequestNum) }},
		{name: "ch", get: func(i int) any { return b01(r(i).CacheHit) }},
		{name: "in", get: func(i int) any { return float64(r(i).InputTokens) }},
		{name: "ca", get: func(i int) any { return float64(r(i).CachedTokens) }},
		{name: "out", get: func(i int) any { return float64(r(i).OutputTokens) }},
		{name: "turn", get: func(i int) any { return float64(r(i).Turn) }},
		{name: "scc", get: func(i int) any { return b01(r(i).ServerCacheConfirmed) }},
		{name: "cs", get: func(i int) any { return b01(r(i).IsColdStart) }},
		{name: "ie", get: func(i int) any { return b01(r(i).IsEmpty) }},
		{name: "lcr", get: func(i int) any { return r(i).LocalCacheRatio }},
		{name: "gi", def: -1, get: func(i int) any { return float64(r(i).GuidIdx) }},
		{name: "ue", get: func(i int) any { return float64(r(i).UUIDExpected) }},
		{name: "uf", get: func(i int) any { return float64(r(i).UUIDFound) }},
		{name: "ul", get: func(i int) any { return float64(r(i).UUIDLeaked) }},
		{name: "uem", get: func(i int) any { return b01(r(i).UUIDExactMatch) }},
		// [expected_uuids_raw, found_mask, leaked_uuids_raw] on a miss or leak.
		{name: "ux", nested: true, get: func(i int) any {
			x := r(i)
			if len(x.ExpectedUUIDsRaw) == 0 && len(x.FoundMask) == 0 && len(x.LeakedUUIDsRaw) == 0 {
				return nil
			}
			mask := make([]int, len(x.FoundMask))
			for k, v := range x.FoundMask {
				if v {
					mask[k] = 1
				}
			}
			return []any{x.ExpectedUUIDsRaw, mask, x.LeakedUUIDsRaw}
		}},
	}
	if tl != nil {
		q := func(i int) *timelineReq { return tl[i] }
		defs = append(defs,
			colDef{name: "r", get: func(i int) any { return float64(q(i).Row) }},
			colDef{name: "u", get: func(i int) any { return float64(q(i).Sub) }},
			colDef{name: "sl", get: func(i int) any { return float64(q(i).Slot) }},
			colDef{name: "tp", get: func(i int) any { return q(i).TP }},
			colDef{name: "rw", get: func(i int) any { return q(i).RW }},
			colDef{name: "r4", get: func(i int) any { return float64(q(i).R4) }},
			colDef{name: "cw", get: func(i int) any { return q(i).CW }},
			colDef{name: "x", get: func(i int) any { return float64(q(i).Flags) }},
			// The timeline's end-to-end time is resp + cw except when resp is
			// missing and the record's end time stood in; ef carries that value.
			colDef{name: "ef", get: func(i int) any {
				if e := q(i).E2E - q(i).CW; e != r(i).ResponseMs {
					return e
				}
				return 0.0
			}},
			colDef{name: "src", width: 4, get: func(i int) any { return ints(q(i).Src, 4) }},
			colDef{name: "rc", width: 4, get: func(i int) any { return ints(q(i).Rec, 4) }},
			colDef{name: "tm", width: 7, get: func(i int) any { return floats(q(i).Tm, 7) }},
			colDef{name: "g", nested: true, get: func(i int) any {
				s := q(i).Segs
				if len(s) == 0 {
					return nil
				}
				out := make([]any, len(s))
				for k, v := range s {
					out[k] = v
				}
				return out
			}},
		)
	}
	n := len(vr)
	var cols []reportCol
	var vals [][]any // vals[col][row]
	for _, d := range defs {
		col := make([]any, n)
		keep := false
		for i := 0; i < n; i++ {
			col[i] = d.get(i)
			if !keep && !d.empty(col[i]) {
				keep = true
			}
		}
		if !keep {
			continue
		}
		cols = append(cols, d.col())
		vals = append(vals, col)
	}
	rows := make([][]any, n)
	for i := 0; i < n; i++ {
		row := make([]any, 0, len(cols)+16)
		for c, col := range cols {
			v := vals[c][i]
			switch {
			case col.Width > 1:
				for _, f := range v.([]float64) {
					row = append(row, f)
				}
			case col.Width == 0 && v == nil:
				row = append(row, []any{})
			default:
				row = append(row, v)
			}
		}
		rows[i] = row
	}
	return cols, rows
}
