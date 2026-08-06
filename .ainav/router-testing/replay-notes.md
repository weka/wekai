# Replay Benchmark Specifics (router-replay mode)

Mechanics of `wekai benchmark auto --router-replay-file` worth knowing before reading
run output or debugging a run that looks wrong. See README's "Replay benchmark"
section for the general capture → replay-prepare → replay pipeline; this file covers
router-mock-testing-relevant details and correctness fixes not yet in the README.

## replay-v3 file layout

- Source captures: `~/.wekai/router/capture/redacted/` (from `wekai router serve
  --capture redacted`).
- Compiled replay file: `~/.wekai/router/capture/replays/replay-<start>-to-<end>.jsonl`
  (default output of `wekai router replay-prepare`, overridable with `--out`).
- Format: line 1 is a JSON header (`_schema: "replay-v3"`, summary counts); each
  following line is one `ReplaySession` — a session's full instance tree (parent/child
  spawn linkage, fan-out groups) with per-request structured spec (system blocks,
  tools, messages) embedded so the replayer regenerates byte/token-faithful synthetic
  content from each block's content hash. One line per session keeps both the writer
  and the reader streaming with a bounded queue regardless of file size.

## `--limit-context` behavior

Skips (does not send) any request whose CAPTURE-recorded prompt tokens (usage input +
cache read/creation) exceed the limit — a client-side pre-check against the ORIGINAL
capture's numbers, not the model's live response. Once a session's prompt outgrows the
limit, every later turn in that instance would too (prompts only grow), so the whole
instance is RETIRED (not completed, not counted as an error) rather than re-checked
turn by turn.

**Skips return their emission budget** (fixed in `2e49f63`): a skip/retirement used to
consume a `--total` emission slot without ever completing, which meant the
`--total`-reached terminator (which waits on COMPLETED count) could never fire once
enough sessions retired — the run sat drained (`active=0`, `in_flight=0`) forever.
Fixed: skips decrement the emitted counter so another request can use the slot, and a
run genuinely attempts exactly `--total` requests.

## Model discovery is cached per endpoint

Each replay worker used to fire its own `/v1/models` discovery GET on first use. With
`--limit-context` retiring oversized sessions without any HTTP call, worker churn
through the queue at startup is fast enough that hundreds of concurrent discovery GETs
could flood the router past its concurrency cap, shedding 503s that cascaded into
per-request errors and instant run termination (fixed in `960b6a0`: one discovery GET
per endpoint per process, cached).

## Teardown records no phantom errors

Two run-termination artifacts used to inflate the error count on an otherwise-clean
run (fixed in `cdeac09`):

- At end-of-run (`--total` reached, or timeout), teardown closes done-channels en
  masse, unblocking many descendant instances at once. Each used to run model
  discovery BEFORE checking whether the run was already over — the simultaneous GET
  burst shed as mass 503s. Instances now check for run-end BEFORE firing discovery.
- In-flight requests aborted by shutdown (timeout or `--total` cancellation) used to be
  recorded as errors (`context deadline exceeded` at every timed run's cutoff).
  Shutdown-aborted requests are no longer recorded.

Verified: a `--total 30000` run completes with exactly 30000 attempted / 0 errors.
