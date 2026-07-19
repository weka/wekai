package benchmark

import (
	_ "embed"
	"fmt"
)

//go:embed bench_doc.txt
var embeddedBenchDoc string

const (
	// EmbeddedDocTotalBytes is the total size of the embedded document
	EmbeddedDocTotalBytes = 1234609
	// EmbeddedDocTotalTokens is the approximate token count (~4.04 bytes/token)
	EmbeddedDocTotalTokens = 305465
	// bytesPerToken is the ratio for truncation
	bytesPerToken = float64(EmbeddedDocTotalBytes) / float64(EmbeddedDocTotalTokens)
)

// GetEmbeddedDocs returns the embedded benchmark documentation content.
// If maxTokens > 0, truncates to approximately that many tokens.
// Returns the content formatted as ReadDirectoryContents would (with file markers).
func GetEmbeddedDocs(maxTokens int) (string, error) {
	doc := embeddedBenchDoc
	if maxTokens > 0 {
		if maxTokens > EmbeddedDocTotalTokens {
			return "", fmt.Errorf("requested tokens (%d) exceeds maximum %d", maxTokens, EmbeddedDocTotalTokens)
		}
		numBytes := int(float64(maxTokens) * bytesPerToken)
		if numBytes < len(doc) {
			doc = doc[:numBytes]
		}
	}
	// Format like ReadDirectoryContents would: wrap in file marker
	return fmt.Sprintf("---File: bench_doc.txt---\n%s\n", doc), nil
}
