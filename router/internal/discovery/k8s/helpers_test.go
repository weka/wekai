package k8s_test

import "k8s.io/apimachinery/pkg/runtime"

// runtimeObject narrows what the test helpers accept, so a typo produces a
// compile error rather than a silently ignored object in the fake clientset.
type runtimeObject = runtime.Object

func toRuntime(in []any) []runtime.Object {
	out := make([]runtime.Object, 0, len(in))
	for _, o := range in {
		if ro, ok := o.(runtime.Object); ok {
			out = append(out, ro)
		}
	}
	return out
}
