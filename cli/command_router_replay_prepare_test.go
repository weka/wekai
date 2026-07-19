package cli

import (
	"testing"
)

func TestRepairDanglingParents(t *testing.T) {
	// Helper: an RFC3339Nano timestamp a bit after epoch. We use real
	// parseable strings so time.Parse(RFC3339Nano) works.
	const (
		T0 = "2025-01-01T00:00:00.000000000Z"
		T1 = "2025-01-01T00:00:01.000000000Z"
		T2 = "2025-01-01T00:00:02.000000000Z"
	)

	t.Run("dangling re-parented to timestamp predecessor", func(t *testing.T) {
		rs := &ReplaySession{
			SessionID: "s1",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 0, // root
					Requests: []ReplayRequest{
						{RequestID: 10, Ts: T0},
					},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 999, // dangling
					FanOutGroup:          "old-group",
					FanOutSize:           3,
					FanOutPosition:       2,
					Requests: []ReplayRequest{
						{RequestID: 20, Ts: T1},
					},
				},
			},
		}

		n := repairDanglingParents(rs)

		if n != 1 {
			t.Fatalf("expected 1 repaired, got %d", n)
		}
		b := &rs.Instances[1]
		if b.ParentSpawnRequestID != 10 {
			t.Fatalf("expected ParentSpawnRequestID=10 (latest before T1), got %d", b.ParentSpawnRequestID)
		}
		if b.FanOutGroup != "" || b.FanOutSize != 0 || b.FanOutPosition != 0 {
			t.Fatalf("fan-out fields not cleared: group=%q size=%d pos=%d",
				b.FanOutGroup, b.FanOutSize, b.FanOutPosition)
		}
	})

	t.Run("dangling and earliest becomes root 0", func(t *testing.T) {
		rs := &ReplaySession{
			SessionID: "s2",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 555, // dangling, and it's the earliest
					FanOutGroup:          "g",
					FanOutSize:           2,
					FanOutPosition:       1,
					Requests: []ReplayRequest{
						{RequestID: 30, Ts: T0},
					},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 0,
					Requests: []ReplayRequest{
						{RequestID: 40, Ts: T2},
					},
				},
			},
		}

		n := repairDanglingParents(rs)

		if n != 1 {
			t.Fatalf("expected 1 repaired, got %d", n)
		}
		a := &rs.Instances[0]
		if a.ParentSpawnRequestID != 0 {
			t.Fatalf("expected ParentSpawnRequestID=0 (no earlier request), got %d", a.ParentSpawnRequestID)
		}
		if a.FanOutGroup != "" || a.FanOutSize != 0 || a.FanOutPosition != 0 {
			t.Fatalf("fan-out fields not cleared")
		}
	})

	t.Run("valid parent untouched", func(t *testing.T) {
		rs := &ReplaySession{
			SessionID: "s3",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 0,
					Requests: []ReplayRequest{
						{RequestID: 50, Ts: T0},
					},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 50, // valid: points at request 50 in this session
					Requests: []ReplayRequest{
						{RequestID: 60, Ts: T1},
					},
				},
			},
		}

		n := repairDanglingParents(rs)

		if n != 0 {
			t.Fatalf("expected 0 repaired, got %d", n)
		}
		b := &rs.Instances[1]
		if b.ParentSpawnRequestID != 50 {
			t.Fatalf("valid parent was altered: got %d", b.ParentSpawnRequestID)
		}
	})

	t.Run("already root 0 untouched", func(t *testing.T) {
		rs := &ReplaySession{
			SessionID: "s4",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 0,
					Requests: []ReplayRequest{
						{RequestID: 70, Ts: T0},
					},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 0,
					Requests: []ReplayRequest{
						{RequestID: 80, Ts: T1},
					},
				},
			},
		}

		n := repairDanglingParents(rs)

		if n != 0 {
			t.Fatalf("expected 0 repaired, got %d", n)
		}
		if rs.Instances[0].ParentSpawnRequestID != 0 || rs.Instances[1].ParentSpawnRequestID != 0 {
			t.Fatal("root instances were altered")
		}
	})
}

func TestBreakParentCycles(t *testing.T) {
	t.Run("2-cycle both promoted", func(t *testing.T) {
		// A (request 10) → parent 20 (B's request)
		// B (request 20) → parent 10 (A's request)
		// Neither reaches a root → both promoted, return 2.
		rs := &ReplaySession{
			SessionID: "cyc-2",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 20, // owned by B
					Requests:             []ReplayRequest{{RequestID: 10}},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 10, // owned by A
					Requests:             []ReplayRequest{{RequestID: 20}},
				},
			},
		}

		n := breakParentCycles(rs)

		if n != 2 {
			t.Fatalf("expected 2 promoted, got %d", n)
		}
		if rs.Instances[0].ParentSpawnRequestID != 0 {
			t.Fatalf("inst-A ParentSpawnRequestID should be 0, got %d", rs.Instances[0].ParentSpawnRequestID)
		}
		if rs.Instances[1].ParentSpawnRequestID != 0 {
			t.Fatalf("inst-B ParentSpawnRequestID should be 0, got %d", rs.Instances[1].ParentSpawnRequestID)
		}
	})

	t.Run("3-cycle all promoted", func(t *testing.T) {
		// A(10)→B(20)→C(30)→A(10) — 3-cycle
		rs := &ReplaySession{
			SessionID: "cyc-3",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 30, // owned by C
					Requests:             []ReplayRequest{{RequestID: 10}},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 10, // owned by A
					Requests:             []ReplayRequest{{RequestID: 20}},
				},
				{
					InstanceID:           "inst-C",
					ParentSpawnRequestID: 20, // owned by B
					Requests:             []ReplayRequest{{RequestID: 30}},
				},
			},
		}

		n := breakParentCycles(rs)

		if n != 3 {
			t.Fatalf("expected 3 promoted, got %d", n)
		}
		for i, inst := range rs.Instances {
			if inst.ParentSpawnRequestID != 0 {
				t.Fatalf("inst[%d] ParentSpawnRequestID should be 0, got %d", i, inst.ParentSpawnRequestID)
			}
		}
	})

	t.Run("cycle with downstream child all promoted", func(t *testing.T) {
		// A(10) → B(20) → A(10)  [2-cycle between A and B]
		// C(30) → parent 20 (B's request)  [C depends on B, which is in cycle]
		// C can't anchor because B is in a cycle → all 3 promoted.
		rs := &ReplaySession{
			SessionID: "cyc-down",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 20, // owned by B
					Requests:             []ReplayRequest{{RequestID: 10}},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 10, // owned by A → cycle with A
					Requests:             []ReplayRequest{{RequestID: 20}},
				},
				{
					InstanceID:           "inst-C",
					ParentSpawnRequestID: 20, // owned by B (which is in cycle)
					Requests:             []ReplayRequest{{RequestID: 30}},
				},
			},
		}

		n := breakParentCycles(rs)

		if n != 3 {
			t.Fatalf("expected 3 promoted, got %d", n)
		}
		for i, inst := range rs.Instances {
			if inst.ParentSpawnRequestID != 0 {
				t.Fatalf("inst[%d] ParentSpawnRequestID should be 0, got %d", i, inst.ParentSpawnRequestID)
			}
		}
	})

	t.Run("clean DAG untouched", func(t *testing.T) {
		// Root(10, pid=0) → A(20, pid=10) → B(30, pid=20)
		// Pure chain starting from a root. Nothing to break.
		rs := &ReplaySession{
			SessionID: "dag",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-Root",
					ParentSpawnRequestID: 0, // true root
					Requests:             []ReplayRequest{{RequestID: 10}},
				},
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 10, // owned by Root
					Requests:             []ReplayRequest{{RequestID: 20}},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 20, // owned by A
					Requests:             []ReplayRequest{{RequestID: 30}},
				},
			},
		}

		n := breakParentCycles(rs)

		if n != 0 {
			t.Fatalf("expected 0 promoted, got %d", n)
		}
		if rs.Instances[0].ParentSpawnRequestID != 0 {
			t.Fatal("root ParentSpawnRequestID was altered")
		}
		if rs.Instances[1].ParentSpawnRequestID != 10 {
			t.Fatal("inst-A ParentSpawnRequestID was altered")
		}
		if rs.Instances[2].ParentSpawnRequestID != 20 {
			t.Fatal("inst-B ParentSpawnRequestID was altered")
		}
	})

	t.Run("already root untouched", func(t *testing.T) {
		rs := &ReplaySession{
			SessionID: "roots",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 0,
					Requests:             []ReplayRequest{{RequestID: 10}},
				},
				{
					InstanceID:           "inst-B",
					ParentSpawnRequestID: 0,
					Requests:             []ReplayRequest{{RequestID: 20}},
				},
			},
		}

		n := breakParentCycles(rs)

		if n != 0 {
			t.Fatalf("expected 0 promoted, got %d", n)
		}
		if rs.Instances[0].ParentSpawnRequestID != 0 || rs.Instances[1].ParentSpawnRequestID != 0 {
			t.Fatal("root instances were altered")
		}
	})

	t.Run("dangling parent treated as root untouched", func(t *testing.T) {
		// A (request 10) → parent 999 (not owned by any instance)
		// Dangling parents are treated as roots — returns 0.
		rs := &ReplaySession{
			SessionID: "dangling",
			Instances: []ReplayInstance{
				{
					InstanceID:           "inst-A",
					ParentSpawnRequestID: 999, // not in reqOwner
					Requests:             []ReplayRequest{{RequestID: 10}},
				},
			},
		}

		n := breakParentCycles(rs)

		if n != 0 {
			t.Fatalf("expected 0 promoted, got %d", n)
		}
		if rs.Instances[0].ParentSpawnRequestID != 999 {
			t.Fatal("dangling parent was altered — should stay as-is (already handled by repairDanglingParents)")
		}
	})
}
