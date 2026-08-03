package gateway

import "net/http"

// recoverForTest exposes the recover boundary so the external test package can
// assert that a genuine panic still becomes a 500 while a client disconnect does
// not count as one.
func (s *Server) RecoverForTest(next http.Handler) http.Handler { return s.recoverMiddleware(next) }
