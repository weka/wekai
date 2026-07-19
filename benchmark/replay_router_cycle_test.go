package benchmark

import (
	"testing"
)

func makeInst(id string, role string, parentSpawnRequestID uint64, requestIDs ...uint64) RouterReplayInstance {
	reqs := make([]RouterReplayRequest, len(requestIDs))
	for i, rid := range requestIDs {
		reqs[i] = RouterReplayRequest{RequestID: rid}
	}
	return RouterReplayInstance{
		InstanceID:           id,
		Role:                 role,
		ParentSpawnRequestID: parentSpawnRequestID,
		Requests:             reqs,
	}
}

func includeAll(role string) bool { return true }

func actionNames(actions []int) []string {
	out := make([]string, len(actions))
	for i, a := range actions {
		switch a {
		case instActWait:
			out[i] = "wait"
		case instActFire:
			out[i] = "fire"
		case instActDrop:
			out[i] = "drop"
		default:
			out[i] = "unknown"
		}
	}
	return out
}

func TestResolveInstanceActions_LinearTree(t *testing.T) {
	// root (req 1) -> A (parent=1, req 2) -> B (parent=2, req 3)
	instances := []RouterReplayInstance{
		makeInst("root", "main", 0, 1),
		makeInst("A", "sub-agent", 1, 2),
		makeInst("B", "sub-agent", 2, 3),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	expected := []string{"fire", "wait", "wait"}
	got := actionNames(actions)
	for i := range expected {
		if got[i] != expected[i] {
			t.Errorf("instance %d: expected %s, got %s", i, expected[i], got[i])
		}
	}
}

func TestResolveInstanceActions_FanOut(t *testing.T) {
	// root (req 1), 3 children all with parent=1
	instances := []RouterReplayInstance{
		makeInst("root", "main", 0, 1),
		makeInst("child1", "sub-agent", 1, 2),
		makeInst("child2", "sub-agent", 1, 3),
		makeInst("child3", "sub-agent", 1, 4),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	expected := []string{"fire", "wait", "wait", "wait"}
	got := actionNames(actions)
	for i := range expected {
		if got[i] != expected[i] {
			t.Errorf("instance %d: expected %s, got %s", i, expected[i], got[i])
		}
	}
}

func TestResolveInstanceActions_TwoCycle(t *testing.T) {
	// A->B, B->A (each ParentSpawnRequestID points to a request owned by the other)
	instances := []RouterReplayInstance{
		makeInst("A", "sub-agent", 20, 10),
		makeInst("B", "sub-agent", 10, 20),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 2 {
		t.Errorf("expected 2 promoted, got %d", promoted)
	}
	for i, a := range actions {
		if a != instActFire {
			t.Errorf("instance %d: expected fire, got %s", i, actionNames([]int{a})[0])
		}
	}
}

func TestResolveInstanceActions_ThreeCycle(t *testing.T) {
	instances := []RouterReplayInstance{
		makeInst("A", "sub-agent", 30, 10),
		makeInst("B", "sub-agent", 10, 20),
		makeInst("C", "sub-agent", 20, 30),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 3 {
		t.Errorf("expected 3 promoted, got %d", promoted)
	}
	for i, a := range actions {
		if a != instActFire {
			t.Errorf("instance %d: expected fire, got %s", i, actionNames([]int{a})[0])
		}
	}
}

func TestResolveInstanceActions_CycleWithDownstream(t *testing.T) {
	// A->B, B->A (cycle), C->B (downstream of cycle)
	instances := []RouterReplayInstance{
		makeInst("A", "sub-agent", 20, 10),
		makeInst("B", "sub-agent", 10, 20),
		makeInst("C", "sub-agent", 20, 30),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 3 {
		t.Errorf("expected 3 promoted, got %d (cycle members + downstream)", promoted)
	}
	for i, a := range actions {
		if a != instActFire {
			t.Errorf("instance %d: expected fire, got %s", i, actionNames([]int{a})[0])
		}
	}
}

func TestResolveInstanceActions_RoleFilterParentExcluded(t *testing.T) {
	// root("main", req 1) -> child("helper", parent=1)
	// filter: only "main" included
	includeRole := func(role string) bool { return role == "main" }
	instances := []RouterReplayInstance{
		makeInst("root", "main", 0, 1),
		makeInst("helper", "helper", 1, 2),
	}
	actions, promoted := resolveInstanceActions(instances, includeRole)
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	if actions[1] != instActDrop {
		t.Errorf("helper with excluded parent: expected drop, got %s", actionNames([]int{actions[1]})[0])
	}
}

func TestResolveInstanceActions_RoleFilterDeeperDescendant(t *testing.T) {
	// root("main", req 1) -> helper("helper", parent=1, req 2) -> sub("sub-agent", parent=2, req 3)
	// filter: only "main" and "sub-agent"
	includeRole := func(role string) bool { return role == "main" || role == "sub-agent" }
	instances := []RouterReplayInstance{
		makeInst("root", "main", 0, 1),
		makeInst("helper", "helper", 1, 2),
		makeInst("sub", "sub-agent", 2, 3),
	}
	actions, promoted := resolveInstanceActions(instances, includeRole)
	// helper is role-excluded (instActDrop), sub's parent (helper) is excluded
	// but helper has request 2 which IS mapped in reqOwnerAll, so sub gets
	// instActDrop (parent role-excluded).
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	if actions[2] != instActDrop {
		t.Errorf("sub (descendant of excluded parent): expected drop, got %s", actionNames([]int{actions[2]})[0])
	}
}

func TestResolveInstanceActions_DanglingParent(t *testing.T) {
	// instance with ParentSpawnRequestID not owned by anyone
	instances := []RouterReplayInstance{
		makeInst("orphan", "sub-agent", 999, 10),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	if actions[0] != instActFire {
		t.Errorf("orphan with dangling parent: expected fire, got %s", actionNames([]int{actions[0]})[0])
	}
}

func TestResolveInstanceActions_Empty(t *testing.T) {
	actions, promoted := resolveInstanceActions(nil, includeAll)
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	if len(actions) != 0 {
		t.Errorf("expected 0 actions, got %d", len(actions))
	}
}

func TestResolveInstanceActions_MixedRootsAndChildren(t *testing.T) {
	// root1, root2, child of root1 (parent=1), child of root2 (parent=20)
	instances := []RouterReplayInstance{
		makeInst("root1", "main", 0, 10),
		makeInst("root2", "main", 0, 20),
		makeInst("child1", "sub-agent", 10, 30),
		makeInst("child2", "sub-agent", 20, 40),
	}
	actions, promoted := resolveInstanceActions(instances, includeAll)
	if promoted != 0 {
		t.Errorf("expected 0 promoted, got %d", promoted)
	}
	expected := []string{"fire", "fire", "wait", "wait"}
	got := actionNames(actions)
	for i := range expected {
		if got[i] != expected[i] {
			t.Errorf("instance %d: expected %s, got %s", i, expected[i], got[i])
		}
	}
}
