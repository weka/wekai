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
