package cli

import (
	"os"
	"strings"
	"testing"
)

// The pods: token carries everything about the pool it describes, so a router
// fronting two fleets on different ports can say so. It used to need a global
// --discover-port flag, which by construction could only describe one of them.
func TestParsePodsToken(t *testing.T) {
	for _, tc := range []struct {
		in       string
		selector string
		port     int
		portName string
		wantErr  bool
	}{
		{in: "app=vllm", selector: "app=vllm"},
		{in: "app=vllm,tier=prod", selector: "app=vllm,tier=prod"},
		{in: "app=vllm:http", selector: "app=vllm", portName: "http"},
		{in: "app=vllm:8000", selector: "app=vllm", port: 8000},
		{in: "app=vllm,tier=prod:http", selector: "app=vllm,tier=prod", portName: "http"},
		// A label key may carry a DNS-subdomain prefix with a slash and dots.
		{in: "wekai.io/pool=perf:http", selector: "wekai.io/pool=perf", portName: "http"},
		{in: " app=vllm : http ", selector: "app=vllm", portName: "http"},
		{in: "app=vllm:", wantErr: true},
		{in: "app=vllm:70000", wantErr: true},
		{in: "", wantErr: true},
	} {
		t.Run(tc.in, func(t *testing.T) {
			sel, port, name, err := parsePodsToken(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parsePodsToken(%q) = (%q,%d,%q), want an error", tc.in, sel, port, name)
				}
				return
			}
			if err != nil {
				t.Fatalf("parsePodsToken(%q): %v", tc.in, err)
			}
			if sel != tc.selector || port != tc.port || name != tc.portName {
				t.Errorf("parsePodsToken(%q) = (%q,%d,%q), want (%q,%d,%q)",
					tc.in, sel, port, name, tc.selector, tc.port, tc.portName)
			}
		})
	}
}

// --backends is shorthand for "* => a|b|c" and must survive the syntax it is
// shorthand for: a comma separates endpoints, but inside a pods: token it
// separates labels.
func TestBackendsShorthandKeepsSelectorCommas(t *testing.T) {
	for _, tc := range []struct {
		in   []string
		want string
	}{
		{[]string{"http://a:8000", "http://b:8000"}, "http://a:8000|http://b:8000"},
		{[]string{"http://a:8000,http://b:8000"}, "http://a:8000|http://b:8000"},
		{[]string{"pods:app=vllm,tier=prod"}, "pods:app=vllm,tier=prod"},
		{[]string{"pods:app=vllm:http"}, "pods:app=vllm:http"},
		{[]string{"http://legacy:8000|pods:app=vllm,tier=prod"},
			"http://legacy:8000|pods:app=vllm,tier=prod"},
	} {
		if got := joinBackends(tc.in); got != tc.want {
			t.Errorf("joinBackends(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// The whole point: the tidiest possible form still reaches a discovered pool.
func TestBackendsShorthandBuildsADiscoveryRoute(t *testing.T) {
	c := &RouterServeCommand{Backends: []string{"pods:app=vllm,tier=prod:http"}}
	rules, err := c.buildRules()
	if err != nil {
		t.Fatalf("buildRules: %v", err)
	}
	if len(rules) != 1 {
		t.Fatalf("got %d rules, want 1", len(rules))
	}
	r := rules[0]
	if r.discoverSelector != "app=vllm,tier=prod" || r.discoverPortName != "http" {
		t.Errorf("selector=%q portName=%q port=%d, want app=vllm,tier=prod / http / 0",
			r.discoverSelector, r.discoverPortName, r.discoverPort)
	}
	if len(r.patterns) != 0 {
		t.Errorf("--backends must produce the catch-all rule, got patterns %v", r.patterns)
	}
}

// Discovery is configured entirely by the route, and this is the guard that it
// stays that way. Every knob it used to take as a flag is gone: the port and
// selector moved into the pods: token, the namespace became "this pod's"
// unconditionally, and the kubeconfig went with it — out of a pod there is no
// service account file to read a namespace from, so that flag could not do what
// it said.
func TestNoStandaloneDiscoveryFlags(t *testing.T) {
	src, err := os.ReadFile("command_router.go")
	if err != nil {
		t.Fatalf("read command_router.go: %v", err)
	}
	for _, gone := range []string{
		`long:"discover-namespace"`,
		`long:"discover-port"`,
		`long:"discover-port-name"`,
		`long:"discover-kubeconfig"`,
	} {
		if strings.Contains(string(src), gone) {
			t.Errorf("%s is back; discovery config belongs in the route token, or it "+
				"cannot describe a router fronting two fleets on different ports", gone)
		}
	}
}
