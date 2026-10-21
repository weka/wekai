//go:build !darwin && !linux

package cli

func physicalMemory() uint64 { return 0 }
