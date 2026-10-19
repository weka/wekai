package cli

import (
	"fmt"
	"sort"

	"github.com/weka/wekai/benchmark"
)

// Volume by context size: what share of the token volume, not of the request
// count, comes from requests whose context is at least a threshold. Input is
// the whole context, cached or not; output is generated tokens.

var volumeThresholds = []int{128_000, 200_000, 300_000, 500_000}

type volumeTotals struct {
	Requests      int
	Input, Output int64
}

func (v *volumeTotals) add(r *benchmark.PrefixRequest) {
	v.Requests++
	v.Input += int64(r.InputTokens)
	v.Output += int64(r.OutputTokens)
}

type volumeRow struct {
	Model string
	All   volumeTotals
	Over  []volumeTotals // per volumeThresholds
}

type volumeReport struct {
	Total  volumeRow
	Models []volumeRow
}

func buildVolumeReport(store *benchmark.PrefixStore) *volumeReport {
	byModel := map[string]*volumeRow{}
	rep := &volumeReport{Total: volumeRow{Model: "Total", Over: make([]volumeTotals, len(volumeThresholds))}}
	for i := range store.Reqs {
		r := &store.Reqs[i]
		m := byModel[r.Model]
		if m == nil {
			m = &volumeRow{Model: r.Model, Over: make([]volumeTotals, len(volumeThresholds))}
			byModel[r.Model] = m
		}
		for _, row := range []*volumeRow{&rep.Total, m} {
			row.All.add(r)
			for k, th := range volumeThresholds {
				if r.InputTokens >= th {
					row.Over[k].add(r)
				}
			}
		}
	}
	for _, m := range byModel {
		rep.Models = append(rep.Models, *m)
	}
	sort.Slice(rep.Models, func(a, b int) bool { return rep.Models[a].Model < rep.Models[b].Model })
	return rep
}

func printVolumeText(rep *volumeReport) {
	if rep == nil || rep.Total.All.Requests == 0 {
		return
	}
	t := rep.Total
	fmt.Println("  Volume by context size (share of the total held by requests at or above each size)")
	fmt.Println()
	fmt.Printf("    %-8s %10s %10s %10s\n", "context", "requests", "input", "output")
	for k, th := range volumeThresholds {
		o := t.Over[k]
		fmt.Printf("    >=%-6s %9.2f%% %9.2f%% %9.2f%%\n", fmtTokensShort(th),
			100*shareOf(int64(o.Requests), int64(t.All.Requests)), 100*shareOf(o.Input, t.All.Input),
			100*shareOf(o.Output, t.All.Output))
	}
	fmt.Printf("    totals:  %d requests, input %s, output %s tokens (input = whole context incl. cached)\n",
		t.All.Requests, formatNumber(int(t.All.Input)), formatNumber(int(t.All.Output)))
	fmt.Println()

	fmt.Println("  Input-token share at or above 200k / 500k, per model")
	fmt.Println()
	fmt.Printf("    %-26s %9s %14s %8s %8s %8s %8s\n", "model", "requests", "input tokens", "req>=200k", "in>=200k", "req>=500k", "in>=500k")
	k200, k500 := 1, 3
	row := func(r volumeRow) {
		name := r.Model
		if name == "" {
			name = "(unknown)"
		}
		fmt.Printf("    %-26s %9d %14s %8.1f%% %8.1f%% %8.1f%% %8.1f%%\n", name, r.All.Requests, formatNumber(int(r.All.Input)),
			100*shareOf(int64(r.Over[k200].Requests), int64(r.All.Requests)), 100*shareOf(r.Over[k200].Input, r.All.Input),
			100*shareOf(int64(r.Over[k500].Requests), int64(r.All.Requests)), 100*shareOf(r.Over[k500].Input, r.All.Input))
	}
	for _, m := range rep.Models {
		row(m)
	}
	row(t)
	fmt.Println()
}

type jsonVolumeThreshold struct {
	MinTokens    int     `json:"min_tokens"`
	RequestShare float64 `json:"request_share"`
	InputShare   float64 `json:"input_share"`
	OutputShare  float64 `json:"output_share"`
}

type jsonVolumeRow struct {
	Model        string                `json:"model"`
	Requests     int                   `json:"requests"`
	InputTokens  int64                 `json:"input_tokens"`
	OutputTokens int64                 `json:"output_tokens"`
	AtOrAbove    []jsonVolumeThreshold `json:"at_or_above"`
}

func toJSONVolume(rep *volumeReport) []jsonVolumeRow {
	if rep == nil {
		return nil
	}
	conv := func(r volumeRow) jsonVolumeRow {
		j := jsonVolumeRow{Model: r.Model, Requests: r.All.Requests, InputTokens: r.All.Input,
			OutputTokens: r.All.Output}
		for k, th := range volumeThresholds {
			o := r.Over[k]
			j.AtOrAbove = append(j.AtOrAbove, jsonVolumeThreshold{
				MinTokens: th, RequestShare: shareOf(int64(o.Requests), int64(r.All.Requests)),
				InputShare: shareOf(o.Input, r.All.Input), OutputShare: shareOf(o.Output, r.All.Output),
			})
		}
		return j
	}
	out := []jsonVolumeRow{conv(rep.Total)}
	for _, m := range rep.Models {
		out = append(out, conv(m))
	}
	return out
}
