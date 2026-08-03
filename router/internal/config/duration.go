package config

import (
	"encoding/json"
	"fmt"
	"time"
)

// Duration is a time.Duration that round-trips as a human-readable string.
//
// It exists because encoding/json cannot unmarshal "10s" into a time.Duration —
// only a raw nanosecond count — so a plain time.Duration field makes a ConfigMap
// either unreadable (600000000000) or invalid ("10s"). Both are bad: the first is
// where operators make mistakes, the second fails at startup.
//
// Numbers are still accepted and interpreted as nanoseconds, matching what
// encoding/json would have done, so a config written against the old shape does
// not silently change meaning.
type Duration time.Duration

// D returns the underlying time.Duration.
func (d Duration) D() time.Duration { return time.Duration(d) }

func (d Duration) String() string { return time.Duration(d).String() }

// Set implements flag.Value, so the same type serves flags and JSON.
func (d *Duration) Set(s string) error {
	v, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	*d = Duration(v)
	return nil
}

func (d Duration) MarshalJSON() ([]byte, error) { return json.Marshal(d.String()) }

func (d *Duration) UnmarshalJSON(b []byte) error {
	// String form: "10s", "5m", "1h30m".
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		v, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("invalid duration %q: %w", s, err)
		}
		*d = Duration(v)
		return nil
	}
	// Numeric form: nanoseconds, for compatibility with a plain time.Duration.
	var n int64
	if err := json.Unmarshal(b, &n); err != nil {
		return fmt.Errorf("duration must be a string like \"10s\" or a nanosecond count, got %s", b)
	}
	*d = Duration(n)
	return nil
}
