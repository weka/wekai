# wekai-core — AI-Assisted Development Guide

## Commit Convention (MANDATORY)

Every commit MUST follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` — new functionality → **minor** release
- `fix: ...` — bug fix → **patch** release
- `feat!: ...` / `fix!: ...` or a `BREAKING CHANGE:` footer → **major** release
- `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, `build:` — no release

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

## Build & Test

- Binary is named `wekai` (main package: `cmd/wekai/`); module stays
  `github.com/weka/wekai-core`. `task build`, `task test`, `task docker:build`.
- Publishing: `task replay:push` / `task app:push` / `task helm:push`
  (Dagger module in `.dagger/`; `helm:push` publishes the image first by
  construction).
- Testing policy: no mocked LLM/Chat flows — pure unit tests on
  data-transformation functions, httptest servers for HTTP utilities, and
  real-endpoint e2e tests only.
