package llm

import "github.com/fatih/color"

// cyan/yellow back the GEMINI_DEBUG_RESPONSES debug logging in gemini.go.
// Defined locally (rather than imported from a shared utils package) so
// wekai-core has no dependency on the embedding application.
var (
	cyan   = color.New(color.FgCyan).PrintFunc()
	yellow = color.New(color.FgYellow).PrintFunc()
)

func init() {
	color.Output = color.Error
}
