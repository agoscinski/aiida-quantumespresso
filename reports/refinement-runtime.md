# Runtime refinement: mapped process failures keep their stored outputs

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

- **Track:** CORE runtime correction (`_resolve_placement_outputs`).
- **Owned files (this track only):** `src/aiida/engine/processes/graphs/run.py`
  (mapped-failure arm, core checkout `graph-benchmark`, detached at
  `upstream/graph-authoring-demo`), `tests/engine/processes/graphs/test_mapped_failure_outputs.py`
  (new), and this report. No production QE workflow file was added or modified; test
  collaborators are `ShellJob` substitutions, per the no-`WorkChain` constraint. Nothing was
  committed or pushed.
- **Unblocks:** QE blocker B2 (`reports/pw-migration.md`): mapped error arms nulled out success
  outputs, so a `501` base run could not hand its relaxed structure to the next iteration.

## 1. What was broken

`GraphRun._resolve_placement_outputs` resolved a mapped-failed `ProcessTask` to
`{'error': member}` only, discarding every output the child stored. A failed `CalcJob`
still stores `remote_folder` and any parsed files, but downstream wires (`ran.stdout`,
`ran.remote_folder`) read as `None`, so retries could not chain them and recovery tasks
requiring those inputs could never run. (`calcfunction` `Result` arms were unaffected:
they store nothing on failure.)

## 2. The fix

The mapped-failure arm now returns the stored **declared** outputs plus the synthesized
error member, built as a fresh mapping (the stored node gains no `error` link):

- available declared outputs (e.g. `stdout`, `remote_folder`) resolve to their nodes;
- absent declared arms stay absent and read as `None` downstream, as before;
- undeclared dynamic outputs (e.g. `ShellJob` `stderr`) stay out, exactly like the
  success path (`_placement_value` filters whole-namespace reads to declared ports), so
  branch/loop/map scope entry keeps validating unknown keys;
- unmapped failures still block: dependents never become ready and the graph exits
  `400 ERROR_TASK_FAILED`.

## 3. Observable behavior change (for the coordinator's changelog)

This is breaking for graphs that detect process failure through a **success arm**:
`task.output is None` after a mapped failure used to mean "it failed"; it now means
"it failed *and* stored nothing there". Failure detection must branch on
`task.error is not None`. Suggested `CHANGELOG.md` entry under `Unreleased` →
`Breaking changes`: "`@graph`: outputs a mapped-failed process task stored (e.g.
`CalcJob` `remote_folder`, parsed files) now resolve to those nodes instead of `None`;
use the `error` arm, not success-arm presence, to test for failure."

## 4. QE guidance from test discoveries

- Chain retries through **copy-style inputs** (`parent_folder`, or `nodes.<RemoteData>`
  for `ShellJob`), not the `remote_folder` **input** port: a `CalcJob` with a
  `remote_folder` input performs `_perform_import` (re-parses the referenced folder,
  never runs). Wiring a failed folder there re-parses the failure (`400`), which is
  correct import behavior but not a restart. The new retry test pins the copy pattern.
- Error-arm identity is status-based: map each distinct failure status the retry must
  distinguish (the tests map `400` and `303` side by side and assert each member).

## 5. Verification (exact)

New module `tests/engine/processes/graphs/test_mapped_failure_outputs.py` (6 tests,
`presto`, real `@graph` runs with `ShellJob` on localhost `false`/`cat`/`sh`):

- `test_graphs_build`, `test_failed_stdout_reaches_recovery` (stored `stdout` + `400`
  member reach recovery; failed child has `stdout`/`remote_folder`, no `error` link),
  `test_absent_output_stays_absent_with_correct_error` (`303` member, missing arm `None`),
  `test_retry_chains_failed_remote_folder` (retry's `prev` input **is** the failed
  `remote_folder` node; retry exits `0` with the file content),
  `test_unmapped_failure_still_blocks` (graph `400`, no further dispatch),
  `test_completed_task_reused_by_two_consumers` (one `cat` dispatch, shared node).
- Revert check: with the `run.py` fix stashed, `test_failed_stdout_reaches_recovery` and
  `test_retry_chains_failed_remote_folder` fail and the other 4 pass, as designed.
- Regression: `uv run pytest -m presto tests/engine/processes/graphs/ -q -p no:randomly -n 4`
  → **960 passed** in ~114 s. (`ir.py`/`tasks.py` show other-worker modifications; untouched.)
