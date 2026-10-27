package serve

import (
	"strings"
	"testing"
)

func TestValidatePD(t *testing.T) {
	routes := []Route{{Patterns: "*", Endpoints: []string{"http://p:8000", "http://d1:8000", "http://d2:8000"}}}
	cases := []struct {
		name string
		o    Options
		want string // substring of the error; empty is valid
	}{
		{"off", Options{Routes: routes}, ""},
		{"subset", Options{Routes: routes, PDPrefillBackends: []string{"http://p:8000/"}}, ""},
		{"with prefill split", Options{Routes: routes, PrefillSplit: true, PDPrefillBackends: []string{"http://p:8000"}}, "mutually exclusive"},
		{"not a backend", Options{Routes: routes, PDPrefillBackends: []string{"http://x:8000"}}, "not one of"},
		{"no decode left", Options{Routes: routes, PDPrefillBackends: []string{"http://p:8000", "http://d1:8000", "http://d2:8000"}}, "no decode backend"},
	}
	for _, c := range cases {
		err := c.o.validatePD()
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
			t.Errorf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

func TestPDBandOverrides(t *testing.T) {
	routes := []Route{{Patterns: "*", Endpoints: []string{"http://p:8000", "http://d1:8000"}}}
	pd := []string{"http://p:8000"}
	cases := []struct {
		name string
		o    Options
		want string
	}{
		{"decode band wider than global", Options{Routes: routes, PDPrefillBackends: pd, NodeConcurrency: 16, SoftNodeConcurrency: 8, PDDecodeMaxNodeConcurrency: 64, PDDecodeSoftNodeConcurrency: 32}, ""},
		{"soft inherited from global", Options{Routes: routes, PDPrefillBackends: pd, NodeConcurrency: 16, SoftNodeConcurrency: 8, PDPrefillMaxNodeConcurrency: 12}, ""},
		{"prefill soft at max", Options{Routes: routes, PDPrefillBackends: pd, PDPrefillSoftNodeConcurrency: 8, PDPrefillMaxNodeConcurrency: 8}, "prefill soft"},
		{"decode max below inherited soft", Options{Routes: routes, PDPrefillBackends: pd, NodeConcurrency: 16, SoftNodeConcurrency: 8, PDDecodeMaxNodeConcurrency: 6}, "decode soft"},
		{"soft without any max", Options{Routes: routes, PDPrefillBackends: pd, PDDecodeSoftNodeConcurrency: 4}, "needs a max"},
		{"negative", Options{Routes: routes, PDPrefillBackends: pd, PDDecodeMaxNodeConcurrency: -1}, "negative"},
		{"without pd", Options{Routes: routes, PDDecodeMaxNodeConcurrency: 32}, "needs --pd-prefill-backends"},
	}
	for _, c := range cases {
		err := c.o.validatePD()
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
			t.Errorf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

func TestPDRoleFlowsInheritAndOverride(t *testing.T) {
	o := Options{PDPrefillBackends: []string{"http://p:8000"}, NodeConcurrency: 16, SoftNodeConcurrency: 8,
		PDDecodeMaxNodeConcurrency: 64}
	dec, pre := o.roleFlowConfigs()
	if pre.NodeConcurrency != 16 || pre.SoftNodeConcurrency != 8 {
		t.Errorf("prefill band = %d..%d, want inherited 8..16", pre.SoftNodeConcurrency, pre.NodeConcurrency)
	}
	if dec.NodeConcurrency != 64 || dec.SoftNodeConcurrency != 8 {
		t.Errorf("decode band = %d..%d, want 8..64", dec.SoftNodeConcurrency, dec.NodeConcurrency)
	}
	if got, want := o.signalsLog(), "prefill=[refused,concurrency=8..16] decode=[refused,concurrency=8..64]"; got != want {
		t.Errorf("signalsLog = %q, want %q", got, want)
	}

	o.PDPrefillBackends = nil
	dec, pre = o.roleFlowConfigs()
	if dec.NodeConcurrency != 16 || pre.NodeConcurrency != 16 {
		t.Error("overrides applied without --pd-prefill-backends")
	}
	if got, want := o.signalsLog(), "refused,concurrency=8..16"; got != want {
		t.Errorf("signalsLog = %q, want %q", got, want)
	}
}
