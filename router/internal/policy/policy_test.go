package policy_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

func mkBackends(t *testing.T, n int) []*registry.Backend {
	t.Helper()
	r := registry.New(registry.Options{})
	out := make([]*registry.Backend, 0, n)
	for i := 0; i < n; i++ {
		b, err := r.Add(registry.Spec{URL: fmt.Sprintf("http://w%02d:8000", i)})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		out = append(out, b)
	}
	return out
}

func allPolicies() []policy.Policy {
	return []policy.Policy{
		policy.LeastOutstanding{},
		policy.Random{},
		policy.NewRoundRobin(),
	}
}

// Every policy must behave for 0, 1 and 2 candidates (LB-12).
func TestZeroOneTwoCandidates(t *testing.T) {
	ctx := context.Background()
	rr := &policy.RoutingRequest{}
	for _, p := range allPolicies() {
		if _, err := p.Select(ctx, nil, rr); err != policy.ErrNoCandidates {
			t.Errorf("%s: empty candidates gave err=%v, want ErrNoCandidates", p.Name(), err)
		}
		for _, n := range []int{1, 2, 64} {
			bs := mkBackends(t, n)
			got, err := p.Select(ctx, bs, rr)
			if err != nil {
				t.Errorf("%s: n=%d err=%v", p.Name(), n, err)
				continue
			}
			if got == nil {
				t.Errorf("%s: n=%d returned nil backend", p.Name(), n)
			}
		}
	}
}

func TestLeastOutstandingPicksLowest(t *testing.T) {
	bs := mkBackends(t, 4)
	bs[0].AddInflight(5)
	bs[1].AddInflight(2)
	bs[2].AddInflight(9)
	// bs[3] stays at 0 and must win.
	got, err := policy.LeastOutstanding{}.Select(context.Background(), bs, &policy.RoutingRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[3] {
		t.Fatalf("selected %s, want %s", got.URL, bs[3].URL)
	}
}

// TestLeastOutstandingCapacityNormalized guards HIER-N1: a high-capacity
// backend with many in-flight requests must not lose to a saturated small one
// purely on raw counts.
func TestLeastOutstandingCapacityNormalized(t *testing.T) {
	bs := mkBackends(t, 2)
	small, big := bs[0], bs[1]
	small.SetCapacity(1)
	big.SetCapacity(40)

	small.AddInflight(1) // normalized 1.0 — saturated
	for i := 0; i < 10; i++ {
		big.AddInflight(1) // normalized 0.25 — plenty of headroom
	}

	got, err := policy.LeastOutstanding{}.Select(context.Background(), bs, &policy.RoutingRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if got != big {
		t.Fatalf("selected %s (raw inflight %d) over %s (raw inflight %d): "+
			"raw counts were compared instead of normalized load",
			got.URL, got.Inflight(), big.URL, big.Inflight())
	}
}

// TestTieBreakIsUniformNotFirstIndex guards LB-N4. v1's min_by_key returned the
// first minimum, so on a cold fleet all traffic piled onto candidate 0.
func TestTieBreakIsUniformNotFirstIndex(t *testing.T) {
	const (
		n     = 8
		draws = 100000
	)
	bs := mkBackends(t, n) // all at inflight 0: every candidate ties
	counts := map[string]int{}
	p := policy.LeastOutstanding{}
	for i := 0; i < draws; i++ {
		got, err := p.Select(context.Background(), bs, &policy.RoutingRequest{})
		if err != nil {
			t.Fatal(err)
		}
		counts[got.URL]++
	}

	if len(counts) != n {
		t.Fatalf("only %d of %d candidates ever selected — tie-break is not uniform "+
			"(this is the v1 index-0 thundering herd)", len(counts), n)
	}
	// Chi-square goodness of fit against uniform. 7 dof, 0.1% significance
	// critical value is 24.32; a first-index bias would score astronomically.
	expected := float64(draws) / float64(n)
	var chi2 float64
	for _, c := range counts {
		d := float64(c) - expected
		chi2 += d * d / expected
	}
	if chi2 > 24.32 {
		t.Fatalf("chi-square = %.2f exceeds 24.32 (7 dof, p=0.001): distribution %v", chi2, counts)
	}
	t.Logf("chi-square = %.2f over %d draws across %d candidates", chi2, draws, n)
}

func TestRandomIsUniform(t *testing.T) {
	const (
		n     = 8
		draws = 100000
	)
	bs := mkBackends(t, n)
	counts := map[string]int{}
	for i := 0; i < draws; i++ {
		got, err := policy.Random{}.Select(context.Background(), bs, &policy.RoutingRequest{})
		if err != nil {
			t.Fatal(err)
		}
		counts[got.URL]++
	}
	expected := float64(draws) / float64(n)
	var chi2 float64
	for _, c := range counts {
		d := float64(c) - expected
		chi2 += d * d / expected
	}
	if chi2 > 24.32 {
		t.Fatalf("chi-square = %.2f exceeds 24.32: %v", chi2, counts)
	}
}

// TestRoundRobinExactlyTenEachOverTenN is the LB-14 guarantee.
func TestRoundRobinExactlyTenEachOverTenN(t *testing.T) {
	for _, n := range []int{1, 2, 3, 7, 8, 13} {
		bs := mkBackends(t, n)
		p := policy.NewRoundRobin()
		counts := map[string]int{}
		for i := 0; i < 10*n; i++ {
			got, err := p.Select(context.Background(), bs, &policy.RoutingRequest{})
			if err != nil {
				t.Fatal(err)
			}
			counts[got.URL]++
		}
		for _, b := range bs {
			if counts[b.URL] != 10 {
				t.Errorf("n=%d: backend %s served %d times over %d requests, want exactly 10 (%v)",
					n, b.URL, counts[b.URL], 10*n, counts)
			}
		}
	}
}

// TestRoundRobinShrinkThenGrowVisitsEveryMember is the subtle case: an atomic
// cursor modulo len(candidates) fails here, because on regrowth index k denotes
// a different backend and the removed backend can be starved indefinitely.
func TestRoundRobinShrinkThenGrowVisitsEveryMember(t *testing.T) {
	bs := mkBackends(t, 4)
	p := policy.NewRoundRobin()

	for i := 0; i < 8; i++ { // warm up over the full set
		if _, err := p.Select(context.Background(), bs, &policy.RoutingRequest{}); err != nil {
			t.Fatal(err)
		}
	}
	shrunk := []*registry.Backend{bs[0], bs[1], bs[3]} // bs[2] goes away
	for i := 0; i < 30; i++ {
		if _, err := p.Select(context.Background(), shrunk, &policy.RoutingRequest{}); err != nil {
			t.Fatal(err)
		}
	}
	counts := map[string]int{}
	for i := 0; i < 4; i++ {
		got, err := p.Select(context.Background(), bs, &policy.RoutingRequest{})
		if err != nil {
			t.Fatal(err)
		}
		counts[got.URL]++
	}
	for _, b := range bs {
		if counts[b.URL] == 0 {
			t.Errorf("backend %s starved over a full cycle after shrink/grow: %v", b.URL, counts)
		}
	}
}

// A backend that returns after an absence must be compensated, not merely
// re-admitted: it holds the oldest last-served sequence, so it goes next.
func TestRoundRobinCompensatesReturningBackend(t *testing.T) {
	bs := mkBackends(t, 3)
	p := policy.NewRoundRobin()
	for i := 0; i < 9; i++ {
		if _, err := p.Select(context.Background(), bs, &policy.RoutingRequest{}); err != nil {
			t.Fatal(err)
		}
	}
	rest := []*registry.Backend{bs[1], bs[2]} // bs[0] removed
	for i := 0; i < 10; i++ {
		if _, err := p.Select(context.Background(), rest, &policy.RoutingRequest{}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := p.Select(context.Background(), bs, &policy.RoutingRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if got != bs[0] {
		t.Fatalf("selected %s, want the returning backend %s to be compensated first",
			got.URL, bs[0].URL)
	}
}

// A genuinely new backend must not be starved behind incumbents.
func TestRoundRobinNewcomerIsNotStarved(t *testing.T) {
	bs := mkBackends(t, 3)
	p := policy.NewRoundRobin()
	for i := 0; i < 300; i++ {
		if _, err := p.Select(context.Background(), bs, &policy.RoutingRequest{}); err != nil {
			t.Fatal(err)
		}
	}
	extra := mkBackends(t, 4)[3] // http://w03:8000, never seen by p
	grown := append(append([]*registry.Backend{}, bs...), extra)

	served := false
	for i := 0; i < len(grown); i++ {
		got, err := p.Select(context.Background(), grown, &policy.RoutingRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if got == extra {
			served = true
			break
		}
	}
	if !served {
		t.Fatal("newcomer not served within one full cycle of joining")
	}
}

func TestRoundRobinIsConcurrencySafe(t *testing.T) {
	bs := mkBackends(t, 8)
	p := policy.NewRoundRobin()
	const workers, per = 16, 200

	done := make(chan map[string]int, workers)
	for w := 0; w < workers; w++ {
		go func() {
			local := map[string]int{}
			for i := 0; i < per; i++ {
				got, err := p.Select(context.Background(), bs, &policy.RoutingRequest{})
				if err != nil {
					t.Error(err)
					break
				}
				local[got.URL]++
			}
			done <- local
		}()
	}
	total := map[string]int{}
	for w := 0; w < workers; w++ {
		for k, v := range <-done {
			total[k] += v
		}
	}
	// Under concurrency the exact-rotation guarantee relaxes, but the spread
	// must stay tight: no backend may be starved or overloaded.
	want := float64(workers*per) / float64(len(bs))
	for _, b := range bs {
		got := float64(total[b.URL])
		if math.Abs(got-want)/want > 0.10 {
			t.Errorf("backend %s served %.0f times, want ~%.0f (+/-10%%): %v",
				b.URL, got, want, total)
		}
	}
}

func BenchmarkLeastOutstanding64(b *testing.B) {
	bs := mkBackendsB(b, 64)
	p := policy.LeastOutstanding{}
	rr := &policy.RoutingRequest{}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Select(ctx, bs, rr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRoundRobin64(b *testing.B) {
	bs := mkBackendsB(b, 64)
	p := policy.NewRoundRobin()
	rr := &policy.RoutingRequest{}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Select(ctx, bs, rr); err != nil {
			b.Fatal(err)
		}
	}
}

func mkBackendsB(b *testing.B, n int) []*registry.Backend {
	b.Helper()
	r := registry.New(registry.Options{})
	out := make([]*registry.Backend, 0, n)
	for i := 0; i < n; i++ {
		be, err := r.Add(registry.Spec{URL: fmt.Sprintf("http://w%02d:8000", i)})
		if err != nil {
			b.Fatal(err)
		}
		be.SetHealth(registry.Healthy)
		out = append(out, be)
	}
	return out
}
