package llm

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"
)

// sharedHTTPClient provides a properly configured HTTP client with connection pooling
// for all LLM providers to avoid connection reuse issues and EOF errors
var sharedHTTPClient = &http.Client{
	Timeout: 600 * time.Second,
	Transport: &http.Transport{
		// Connection pooling settings
		MaxIdleConns:        100,             // Maximum idle connections across all hosts
		MaxIdleConnsPerHost: 10,              // Maximum idle connections per host
		MaxConnsPerHost:     100,             // Maximum total connections per host
		IdleConnTimeout:     3 * time.Second, // Short timeout to avoid server-side connection closures
		// Many OpenAI-compatible servers have a 5s keep-alive timeout, so we use 3s to be safe

		// Dial settings
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,

		// TLS handshake timeout
		TLSHandshakeTimeout: 10 * time.Second,

		// Disable HTTP/2 for better connection reuse with streaming
		ForceAttemptHTTP2: false,

		// Response header timeout
		ResponseHeaderTimeout: 60 * time.Second,

		// Expect continue timeout
		ExpectContinueTimeout: 1 * time.Second,

		// TLS configuration
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	},
}
