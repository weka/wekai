# wekai — AI-Assisted Development Guide

## Commit Convention (MANDATORY)

Every commit MUST follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` — **genuinely new functionality** → **minor** release
- `fix: ...` — bug fix, **and any tuning of existing behaviour** → **patch**
- `feat!: ...` / `fix!: ...` or a `BREAKING CHANGE:` footer → **major** release
- `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, `build:` — no release

**`feat:` means a capability that did not exist before.** Changing a default,
adjusting a threshold, or refining how something already works is a `fix:`,
however much the reasoning behind it changed — those ship as patches. Reaching
for `feat:` because a change felt significant inflates the minor version and
tells a reader to look for something new that is not there.

Releases are cut automatically: on every push to `main`,
`.github/workflows/release.yml` derives the next semver from the commit
types since the last tag, publishes the image and Helm chart under that
version, and creates a GitHub Release. A non-conforming commit message
means your change silently ships in someone else's release — always use
the correct type.

Scope is optional but encouraged: `feat(llm): ...`, `fix(chart): ...`,
`ci(release): ...`.

## Versioning

- **CI releases**: semver `vX.Y.Z` (from Conventional Commits, see above) —
  image tag, chart `version`/`appVersion`, and git tag all match.
- **Local/dev publishes** (`task app:push`, `task helm:push`): content-hash
  stamps `v999.0.0-<sha12>` of the source digest — deliberately sorted above
  any real semver so dev builds never masquerade as releases.
- The chart pins its image purely by propagation: `Chart.yaml` `appVersion`
  → template `imageTag | default .Chart.AppVersion`. Never hardcode a
  version in `values.yaml`.

## Replay Content and KV

**A wekai run generates unique content. KV is never reused between runs.**

So a fleet is cold for a new run's prefixes whatever it served before, and two
runs are comparable at equal elapsed time without resetting the backends. Cache
hit rate within a run comes from that run's own sessions extending their
prefixes, nothing else. Never explain a result — good or bad — by the fleet
being warm from earlier traffic.

## Comments and Docs

Describe the current design, never how it got there. A comment earns its place
by explaining what the code does and why it is that way — the trade-off taken,
the constraint respected, the failure a rule prevents. It does not earn its
place by narrating what the code used to do or which bug was fixed.

That belongs in the commit message, where `git log` and `git blame` put it in
front of whoever needs it.

## Verify Before You Commit (MANDATORY)

Run `task verify` — gofmt, `go vet`, and the whole suite under `-race`, exactly
what CI gates on — and only commit once it passes. Do it before every commit
that could change its result, and again between commits when you are landing a
series: a green tree at the end of five commits says nothing about whether
commits two through four build, and `git bisect` is worth nothing on a history
that does not.

A commit that cannot change the result does not need it. Editing this file,
a README, or a comment touches nothing gofmt, vet, or a test can see. Judge it
by what the diff can reach, not by how small it feels — a one-line change to a
struct tag or a default is exactly the kind that reaches everything.

`go test ./...` is not a substitute. It omits `-race`, and the races it omits
are the ones that pass locally and fail in CI, on whichever commit happened to
be building. If the suite is too slow to run at every commit, that is a bug in
the suite: find the test that is eating the time and fix it, rather than
skipping the check that would have caught the problem.

## Build & Test

- Binary is named `wekai` (main package at the repo root, so plain `go install github.com/weka/wekai@<vX>` works); module stays
  `github.com/weka/wekai`. `task build`, `task test`, `task docker:build`.
- Publishing: `task replay:push` / `task app:push` / `task helm:push`
  (Dagger module in `.dagger/`; `helm:push` publishes the image first by
  construction).
- Testing policy: no mocked LLM/Chat flows — pure unit tests on
  data-transformation functions, httptest servers for HTTP utilities, and
  real-endpoint e2e tests only.
