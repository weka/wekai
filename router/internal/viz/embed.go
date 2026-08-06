package viz

import _ "embed"

// pageHTML is the entire /router-viz page — HTML, CSS, and JS in one file,
// no external CDNs, so it renders standalone behind any network policy the
// metrics listener sits behind.
//
//go:embed page.html
var pageHTML string
