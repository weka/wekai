package benchmark

import "regexp"

// Why a marker was not recited.
//
// A raw PRESENCE_MISS count is the sum of at least three unrelated things, and
// reporting it alone has already misled this project once: the first real run
// against a deep session reported 55 misses, every one of which turned out to
// be the model copying the tag names out of the instruction rather than the
// fleet losing anything. A number that large from a cause that mundane is only
// possible because nothing separated the causes.
//
// So each miss is attributed, and the attribution rests on evidence in the
// response rather than on inference:
//
//	substituted  the response carried a marker that IS in this request's own
//	             prompt but was not the one asked for. The model was reading
//	             tags and picked the wrong one, which means the tagged content
//	             was there to read. Strong evidence the fleet is intact.
//	absent       nothing was produced for it and no such substitute appeared.
//	             Consistent with a genuine loss, and also with the model
//	             simply not answering — the two are not separable from
//	             outside, and this count must not be read as if they were.
//
// Two further signals measure the ASK rather than the fleet. If they are not
// near zero the presence ratio is describing the instruction's clarity, and
// no conclusion about caching can be drawn from it at all.
type reciteOutcome struct {
	Substituted int
	Absent      int
	// EchoedTags: the response repeated turn names where guids were asked for.
	EchoedTags bool
	// NoIDs: the response contained no guid-shaped value anywhere.
	NoIDs bool
}

// tagEchoRe matches a bare turn name, which is what a response looks like when
// it copies the instruction's list instead of resolving it.
var tagEchoRe = regexp.MustCompile(`\[turn-\d+\]`)

// classifyRecite attributes every miss in one response.
//
// found is parallel to inj.ReciteUUIDs. own is every marker this request sent,
// which is what makes a substitution recognisable: a guid from the prompt that
// nobody asked for could only have come from the tagged content.
func classifyRecite(resp string, inj *uuidInjection, found []bool) reciteOutcome {
	var out reciteOutcome
	if inj == nil {
		return out
	}
	asked := make(map[string]bool, len(inj.ReciteUUIDs))
	for _, u := range inj.ReciteUUIDs {
		asked[u] = true
	}
	// A marker from this prompt that was not asked for, present in the
	// response: the model read a tag, just not the requested one.
	substitutePresent := false
	for _, st := range inj.StampByHash {
		if !asked[st.UUID] && containsUUID(resp, st.UUID) {
			substitutePresent = true
			break
		}
	}
	for i := range inj.ReciteUUIDs {
		if i < len(found) && found[i] {
			continue
		}
		if substitutePresent {
			out.Substituted++
		} else {
			out.Absent++
		}
	}
	if len(uuidRe.FindAllString(resp, 1)) == 0 {
		out.NoIDs = true
		// Only meaningful alongside NoIDs: a response that produced guids AND
		// mentions a turn name is not echoing, it is annotating.
		out.EchoedTags = tagEchoRe.MatchString(resp)
	}
	return out
}

// containsUUID is a plain substring test, matching how presence itself is
// scored — the two must agree, or a marker could be "found" by one and
// "substituted" by the other.
func containsUUID(resp, u string) bool {
	return len(u) > 0 && len(resp) >= len(u) && indexOf(resp, u) >= 0
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}

// responseIsGarbage reports decode-level corruption in a response: replacement
// characters, NUL bytes, or C0 control characters other than tab/newline/CR.
//
// This is the one bad signal besides contamination that needs no cooperation
// from the model and no marker in the prompt. A model can decline to recite
// and still be served intact context; it cannot emit U+FFFD on its own —
// json.Unmarshal put it there because the bytes off the wire were not valid
// UTF-8, which points at the serving stack, not at instruction-following.
//
// Deliberately narrow. Repetition loops, truncation and incoherent prose are
// all real degradations, but each has innocent causes (sampling, max_tokens,
// the corpus itself), and a "garbage" counter that fires on innocent causes
// gets ignored — this one firing at all is worth reading the capture.
func responseIsGarbage(resp string) bool {
	for _, r := range resp {
		switch {
		case r == '\uFFFD':
			return true
		case r == 0:
			return true
		case r < 0x20 && r != '\n' && r != '\t' && r != '\r':
			return true
		}
	}
	return false
}
