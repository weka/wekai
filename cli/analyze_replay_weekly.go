package cli

import (
	"fmt"
	"sort"
	"time"

	"github.com/weka/wekai/benchmark"
)

// Per-week trend: does context grow over the capture, and is that people
// working longer in one session or just the model mix moving to models with
// bigger windows? Weeks start Monday 00:00 UTC.

const weeklyTopModels = 4

type weekContext struct {
	Week     time.Time
	Requests int
	Sizes    contextSizeStats
	Over200k float64 // share of requests
	Over500k float64
	// Sessions started this week, by their first request.
	SessionsStarted int
	SessReqP50      int
	SessReqP90      int
	SessDurP50      time.Duration
	SessDurP90      time.Duration
	SessPeakP50     int // per-session peak context
	SessPeakP90     int
	// ModelP50 is the median context per top model, -1 when absent.
	ModelP50 []int
}

type weeklyReport struct {
	Weeks     []weekContext
	TopModels []string
	Sessions  bool // session columns are meaningful (replay input)
}

func weekStart(ts int64) time.Time {
	t := time.Unix(0, ts).UTC()
	d := (int(t.Weekday()) + 6) % 7 // Monday = 0
	return time.Date(t.Year(), t.Month(), t.Day()-d, 0, 0, 0, 0, time.UTC)
}

func buildWeeklyReport(store *benchmark.PrefixStore, sessions bool) *weeklyReport {
	type sess struct {
		first, last int64
		reqs, peak  int
	}
	perSess := make([]sess, store.Sessions)
	weekSizes := map[time.Time][]int{}
	weekModel := map[time.Time]map[string][]int{}
	modelReqs := map[string]int{}
	for i := range store.Reqs {
		r := &store.Reqs[i]
		if r.Ts == 0 {
			continue
		}
		w := weekStart(r.Ts)
		weekSizes[w] = append(weekSizes[w], r.InputTokens)
		if weekModel[w] == nil {
			weekModel[w] = map[string][]int{}
		}
		weekModel[w][r.Model] = append(weekModel[w][r.Model], r.InputTokens)
		modelReqs[r.Model]++
		s := &perSess[r.Session]
		if s.reqs == 0 || r.Ts < s.first {
			s.first = r.Ts
		}
		s.last = max(s.last, r.Ts)
		s.reqs++
		s.peak = max(s.peak, r.InputTokens)
	}

	rep := &weeklyReport{Sessions: sessions}
	for m := range modelReqs {
		rep.TopModels = append(rep.TopModels, m)
	}
	sort.Slice(rep.TopModels, func(a, b int) bool {
		ma, mb := rep.TopModels[a], rep.TopModels[b]
		if modelReqs[ma] != modelReqs[mb] {
			return modelReqs[ma] > modelReqs[mb]
		}
		return ma < mb
	})
	if len(rep.TopModels) > weeklyTopModels {
		rep.TopModels = rep.TopModels[:weeklyTopModels]
	}

	sessReqs := map[time.Time][]int{}
	sessDur := map[time.Time][]int{} // seconds
	sessPeak := map[time.Time][]int{}
	for _, s := range perSess {
		if s.reqs == 0 {
			continue
		}
		w := weekStart(s.first)
		sessReqs[w] = append(sessReqs[w], s.reqs)
		sessDur[w] = append(sessDur[w], int((s.last-s.first)/int64(time.Second)))
		sessPeak[w] = append(sessPeak[w], s.peak)
	}

	weeks := make([]time.Time, 0, len(weekSizes))
	for w := range weekSizes {
		weeks = append(weeks, w)
	}
	sort.Slice(weeks, func(a, b int) bool { return weeks[a].Before(weeks[b]) })
	for _, w := range weeks {
		sizes := weekSizes[w]
		wc := weekContext{Week: w, Requests: len(sizes)}
		var n200, n500 int
		for _, n := range sizes {
			if n >= 200_000 {
				n200++
			}
			if n >= 500_000 {
				n500++
			}
		}
		wc.Over200k = float64(n200) / float64(len(sizes))
		wc.Over500k = float64(n500) / float64(len(sizes))
		wc.Sizes = contextStats(w.Format("2006-01-02"), sizes)
		if r := sessReqs[w]; len(r) > 0 {
			wc.SessionsStarted = len(r)
			wc.SessReqP50, wc.SessReqP90 = pct(r, 0.5), pct(r, 0.9)
			d := sessDur[w]
			wc.SessDurP50 = time.Duration(pct(d, 0.5)) * time.Second
			wc.SessDurP90 = time.Duration(pct(d, 0.9)) * time.Second
			p := sessPeak[w]
			wc.SessPeakP50, wc.SessPeakP90 = pct(p, 0.5), pct(p, 0.9)
		}
		for _, m := range rep.TopModels {
			if v := weekModel[w][m]; len(v) > 0 {
				wc.ModelP50 = append(wc.ModelP50, pct(v, 0.5))
			} else {
				wc.ModelP50 = append(wc.ModelP50, -1)
			}
		}
		rep.Weeks = append(rep.Weeks, wc)
	}
	return rep
}

// pct sorts v in place and returns its p-quantile (nearest rank below).
func pct(v []int, p float64) int {
	sort.Ints(v)
	return v[int(p*float64(len(v)-1))]
}

func fmtDurShort(d time.Duration) string {
	switch {
	case d >= 24*time.Hour:
		return fmt.Sprintf("%.1fd", d.Hours()/24)
	case d >= time.Hour:
		return fmt.Sprintf("%.1fh", d.Hours())
	}
	return fmt.Sprintf("%dm", int(d.Minutes()))
}

func printWeeklyText(rep *weeklyReport) {
	if rep == nil || len(rep.Weeks) == 0 {
		return
	}
	fmt.Println("  Context size per week (weeks start Monday UTC; input tokens incl. cached)")
	fmt.Println()
	fmt.Printf("    %-10s %8s %6s %6s %6s %6s %6s %6s", "week", "requests", "mean", "p50", "p90", "p99", ">200k", ">500k")
	if rep.Sessions {
		fmt.Printf(" | %6s %7s %7s %7s %7s %7s %7s", "sess", "req p50", "req p90", "dur p50", "dur p90", "peak50", "peak90")
	}
	fmt.Println()
	for _, w := range rep.Weeks {
		s := w.Sizes
		fmt.Printf("    %-10s %8d %6s %6s %6s %6s %5.1f%% %5.1f%%", w.Week.Format("2006-01-02"), w.Requests,
			fmtTokensShort(int(s.Mean)), fmtTokensShort(s.Percentiles[0]), fmtTokensShort(s.Percentiles[1]),
			fmtTokensShort(s.Percentiles[3]), 100*w.Over200k, 100*w.Over500k)
		if rep.Sessions {
			if w.SessionsStarted > 0 {
				fmt.Printf(" | %6d %7d %7d %7s %7s %7s %7s", w.SessionsStarted, w.SessReqP50, w.SessReqP90,
					fmtDurShort(w.SessDurP50), fmtDurShort(w.SessDurP90),
					fmtTokensShort(w.SessPeakP50), fmtTokensShort(w.SessPeakP90))
			} else {
				fmt.Printf(" | %6d", 0)
			}
		}
		fmt.Println()
	}
	if rep.Sessions {
		fmt.Println("    sess = sessions started that week; req = requests per session; dur = first to last")
		fmt.Println("    request; peak = the session's largest request context. A session counts in the")
		fmt.Println("    week it started, so the last weeks are cut short by the capture's end.")
	}
	fmt.Println()

	if len(rep.TopModels) == 0 {
		return
	}
	fmt.Println("  Median context per week, busiest models (separates growth from model mix)")
	fmt.Println()
	fmt.Printf("    %-10s", "week")
	for _, m := range rep.TopModels {
		fmt.Printf(" %26s", m)
	}
	fmt.Println()
	for _, w := range rep.Weeks {
		fmt.Printf("    %-10s", w.Week.Format("2006-01-02"))
		for _, v := range w.ModelP50 {
			if v < 0 {
				fmt.Printf(" %26s", "-")
			} else {
				fmt.Printf(" %26s", fmtTokensShort(v))
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

type jsonWeek struct {
	Week            string         `json:"week"`
	Requests        int            `json:"requests"`
	Mean            float64        `json:"mean"`
	P50             int            `json:"p50"`
	P90             int            `json:"p90"`
	P99             int            `json:"p99"`
	Over200k        float64        `json:"share_over_200k"`
	Over500k        float64        `json:"share_over_500k"`
	SessionsStarted int            `json:"sessions_started,omitempty"`
	SessReqP50      int            `json:"session_requests_p50,omitempty"`
	SessReqP90      int            `json:"session_requests_p90,omitempty"`
	SessDurP50S     float64        `json:"session_duration_p50_s,omitempty"`
	SessDurP90S     float64        `json:"session_duration_p90_s,omitempty"`
	SessPeakP50     int            `json:"session_peak_context_p50,omitempty"`
	SessPeakP90     int            `json:"session_peak_context_p90,omitempty"`
	ModelP50        map[string]int `json:"model_p50,omitempty"`
}

func toJSONWeekly(rep *weeklyReport) []jsonWeek {
	if rep == nil {
		return nil
	}
	var out []jsonWeek
	for _, w := range rep.Weeks {
		j := jsonWeek{
			Week: w.Week.Format("2006-01-02"), Requests: w.Requests, Mean: w.Sizes.Mean,
			P50: w.Sizes.Percentiles[0], P90: w.Sizes.Percentiles[1], P99: w.Sizes.Percentiles[3],
			Over200k: w.Over200k, Over500k: w.Over500k, ModelP50: map[string]int{},
		}
		if rep.Sessions {
			j.SessionsStarted, j.SessReqP50, j.SessReqP90 = w.SessionsStarted, w.SessReqP50, w.SessReqP90
			j.SessDurP50S, j.SessDurP90S = w.SessDurP50.Seconds(), w.SessDurP90.Seconds()
			j.SessPeakP50, j.SessPeakP90 = w.SessPeakP50, w.SessPeakP90
		}
		for i, m := range rep.TopModels {
			if w.ModelP50[i] >= 0 {
				j.ModelP50[m] = w.ModelP50[i]
			}
		}
		out = append(out, j)
	}
	return out
}
