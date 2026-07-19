package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/weka/go-weka-observability/instrumentation"
)

// shouldLogHTTPRequests checks if HTTP logging is enabled via environment variable
func shouldLogHTTPRequests() bool {
	// Check environment variable directly to avoid import cycle with config package
	return os.Getenv("LOG_HTTP_REQUESTS") == "true"
}

// LogHTTPRequest logs an outgoing HTTP request with headers and body
func LogHTTPRequest(ctx context.Context, req *http.Request, body []byte) {
	if !shouldLogHTTPRequests() {
		return
	}

	_, logger, end := instrumentation.GetLogSpan(ctx, "http_request")
	defer end()

	// Redact authorization header for security
	headers := make(map[string]string)
	for key, values := range req.Header {
		if strings.ToLower(key) == "authorization" {
			// Redact API key, show only prefix and suffix
			redacted := redactAPIKey(values[0])
			headers[key] = redacted
		} else {
			headers[key] = strings.Join(values, ", ")
		}
	}

	// Log the request with truncated body if too large
	const maxBodyLogSize = 1000 // 1KB for request body (structure only)
	bodyStr := string(body)
	bodySize := len(bodyStr)

	logFields := []interface{}{
		"method", req.Method,
		"url", req.URL.String(),
		"headers", headers,
		"body_size", bodySize,
	}

	if bodySize > maxBodyLogSize {
		truncatedBody := bodyStr[:maxBodyLogSize] + fmt.Sprintf("... (truncated, total size: %d bytes)", bodySize)
		logFields = append(logFields, "body", truncatedBody)
	} else {
		logFields = append(logFields, "body", bodyStr)
	}

	logger.Info("HTTP Request", logFields...)
}

// LogHTTPResponse logs an incoming HTTP response with status, headers, and body
func LogHTTPResponse(ctx context.Context, statusCode int, headers http.Header, body string, streaming bool) {
	if !shouldLogHTTPRequests() {
		return
	}

	_, logger, end := instrumentation.GetLogSpan(ctx, "http_response")
	defer end()

	// Convert headers to map for logging
	headerMap := make(map[string]string)
	for key, values := range headers {
		headerMap[key] = strings.Join(values, ", ")
	}

	// Log the response
	logFields := []interface{}{
		"status_code", statusCode,
		"headers", headerMap,
		"streaming", streaming,
	}

	// Include body size information
	bodySize := len(body)
	logFields = append(logFields, "body_size", bodySize)

	// For very large bodies, log a truncated version
	const maxBodyLogSize = 100000 // 100KB
	if bodySize > maxBodyLogSize {
		truncatedBody := body[:maxBodyLogSize] + fmt.Sprintf("\n... (truncated, total size: %d bytes)", bodySize)
		logFields = append(logFields, "body", truncatedBody)
	} else {
		logFields = append(logFields, "body", body)
	}

	logger.Info("HTTP Response", logFields...)
}

// redactAPIKey redacts an API key, showing only prefix and suffix
// Example: "Bearer sk-1234567890abcdef" -> "Bearer sk-***def"
func redactAPIKey(authHeader string) string {
	parts := strings.SplitN(authHeader, " ", 2)
	if len(parts) != 2 {
		return "***"
	}

	scheme := parts[0] // "Bearer"
	token := parts[1]

	if len(token) <= 6 {
		return scheme + " ***"
	}

	// Show first 3 and last 3 characters
	prefix := token[:3]
	suffix := token[len(token)-3:]

	return fmt.Sprintf("%s %s***%s", scheme, prefix, suffix)
}

// loggingResponseBody wraps an http.Response.Body to log the complete response
// after it has been fully read. This maintains streaming behavior while capturing
// the complete response for logging purposes.
type loggingResponseBody struct {
	body       io.ReadCloser
	ctx        context.Context
	statusCode int
	headers    http.Header
	buffer     *bytes.Buffer
	logged     bool
}

// NewLoggingResponseBody creates a new logging wrapper for an HTTP response body
func NewLoggingResponseBody(ctx context.Context, body io.ReadCloser, statusCode int, headers http.Header) io.ReadCloser {
	if !shouldLogHTTPRequests() {
		return body
	}

	return &loggingResponseBody{
		body:       body,
		ctx:        ctx,
		statusCode: statusCode,
		headers:    headers,
		buffer:     new(bytes.Buffer),
		logged:     false,
	}
}

// Read implements io.Reader interface
func (l *loggingResponseBody) Read(p []byte) (n int, err error) {
	n, err = l.body.Read(p)
	if n > 0 {
		// Accumulate data as it's read
		l.buffer.Write(p[:n])
	}

	// Log when we reach EOF (stream is complete)
	if err == io.EOF && !l.logged {
		LogHTTPResponse(l.ctx, l.statusCode, l.headers, l.buffer.String(), true)
		l.logged = true
	}

	return n, err
}

// Close implements io.Closer interface
func (l *loggingResponseBody) Close() error {
	// Log if we haven't already (in case Close is called before EOF)
	if !l.logged {
		LogHTTPResponse(l.ctx, l.statusCode, l.headers, l.buffer.String(), true)
		l.logged = true
	}
	return l.body.Close()
}
