package llm

// ContentPart represents a single part of multi-modal content in a message.
type ContentPart interface {
	PartType() string
}

// TextContent represents a text content part.
type TextContent struct {
	Text string
}

// PartType returns "text" for TextContent.
func (t *TextContent) PartType() string {
	return "text"
}

// ImageContent represents an image content part.
type ImageContent struct {
	BlobID   string // Reference to blob store (resolved before LLM call)
	MimeType string // "image/png", "image/jpeg", "image/gif", "image/webp"
	Data     []byte // Resolved inline data
	URL      string // Alternative: remote URL
}

// PartType returns "image" for ImageContent.
func (i *ImageContent) PartType() string {
	return "image"
}

// FileContent represents a file content part (e.g., PDF).
type FileContent struct {
	BlobID   string
	Name     string
	MimeType string // "application/pdf", etc.
	Data     []byte
}

// PartType returns "file" for FileContent.
func (f *FileContent) PartType() string {
	return "file"
}

// TextParts is a helper function for text-only callers (backward compatibility).
// It wraps a text message into a []ContentPart slice.
func TextParts(msg string) []ContentPart {
	return []ContentPart{&TextContent{Text: msg}}
}

// extractTextFromParts extracts and concatenates all text from a []ContentPart slice.
// Used as a fallback when a provider doesn't support multi-modal content in a given context.
func extractTextFromParts(parts []ContentPart) string {
	var text string
	for _, part := range parts {
		if tc, ok := part.(*TextContent); ok {
			if text != "" {
				text += " "
			}
			text += tc.Text
		}
	}
	return text
}
