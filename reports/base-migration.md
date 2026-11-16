# QE base-workflow graph migration

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

Owner: base workflows only — `aiida-quantumespresso/src/aiida_quantumespresso/workflows/graphs/base.py`
(+ `graphs/__init__.py`), `aiida-quantumespresso/tests/graphs/test_base.py`, this report.
Composites and integration are other workers' scope; their files were not touched.
(The `graphs/pdos.py`, `graphs/pw.py`, `tests/graphs/test_pdos.py`, `tests/graphs/test_pw.py`
present in the tree belong to other workers.)

## What was done

Re-expressed the six `*BaseWorkChain` restart loops (`pw`, `ph`, `neb`, `q2r`, `matdyn`,
`bands`) as native graph alternatives in a new `aiida_quantumespresso.workflows.graphs.base`
module. The original work chains are untouched. Science lives in `@task` collaborators
(restart mathematics ported from the `process_handler` methods); the `@graph` bodies own
the retry loops with inline counters and retry predicates, per the canonical recipe
(`tests/engine/processes/graphs/test_authoring_recipe.py`). Each loop graph ends in an
`ensure_*` guard (loud failure when the budget runs out) and a shared `summarize_restart`
report, so the root returns existing nodes.

## Coverage (no parity claimed)

### `pw_base_graph` — `PwCalculation` placed directly, `errors=PwErrors` (17 statuses)

| Original handler | Graph status |
|---|---|
| `setup` / `validate_kpoints` | Ported: k-points branch builds the mesh via `make_kpoints_mesh`, or wires explicit k-points. Missing both fails fast inside that task. |
| `prepare_process` (`max_seconds` capping) | **Not ported.** Scheduler options arrive via `task_metadata` and are constant across iterations; per-iteration capping has no graph equivalent. |
| `sanity_check_insufficient_bands` | Ported as `check_pw_sufficient_bands` on the success path (`nbnd` bump, charge-density restart). |
| `handle_known_unrecoverable_failure` (`exit_codes=[]`) | Omitted: with an empty exit-code list the original never fires either. |
| `handle_diagonalization_errors` | Ported as `handle_pw_diagonalization`. The tried-list is seeded once from the initial parameters and carried in the loop (the original keeps it in `ctx`); exhaustion requests no restart so the guard fails loudly (original aborts `ERROR_KNOWN_UNRECOVERABLE_FAILURE`). |
| `handle_out_of_walltime` | Ported as `handle_pw_walltime` (full restart, output structure carried). |
| `handle_ionic_interrupted_partial_trajectory` (503) | Ported inside `handle_pw_ionic_convergence` (scratch restart from output structure). |
| `handle_vcrelax_converged_except_final_scf` (501) | **Not ported.** Mapped-error children expose no success outputs, so "finish with outputs plus exit code" is unrepresentable; the graph fails loudly instead (safe direction, observable change documented). |
| `handle_electronic_convergence_stuck` (`STOPPED_BY_MONITOR`) | **Not ported.** The original gates on the exit *message*; the error arm carries only the enum member. `STOPPED_BY_MONITOR` (150) is excluded from `PwErrors`, so monitor stops fail loudly. |
| BFGS-history handler (601: trust-radius / `damp` / one-shot rattle) | Ported inside `handle_pw_ionic_convergence`, including the one-shot rattle via a conditional `rattle_task` placement and the upscale restore from the aliased initial parameters. Non-`relax`/`vc-relax` calculation types fall through to the generic ionic restart, as in the original. |
| `handle_relax_recoverable_ionic_convergence_error` (560) | Ported (charge-density restart). |
| `handle_vcrelax_recoverable_fft_significant_volume_contraction_error` (555) | Ported (`cell_factor` doubling, scratch restart). |
| `handle_relax_recoverable_electronic_convergence_error` (550) | Ported (mixing reduction, scratch restart from output structure). |
| `handle_electronic_convergence_not_reached` (410) | Ported (mixing reduction, full restart). |
| `handle_electronic_convergence_warning` (710) | **Not ported**, same reason as 501. |

### `ph_base_graph` — `PhCalculation` placed directly, `errors=PhErrors` (120, 400, 410, 462)

Ported: `recover=True` initialization (`seed_ph_parameters`), scheduler-walltime
`max_seconds` halving (**only when already present** — the scheduler walltime itself is
not visible to tasks), diagonalization switch to `cg` with give-up on repeat,
out-of-walltime recover restart with the recover latch (mirrors the never-cleared
`restart_calc`), and the full `nmix_ph` → `alpha_mix(1)` → `alpha_mix(20)` convergence
chain. `qpoints_distance` takes an **explicit `structure` input** instead of deriving
it from the parent-folder provenance. `only_initialization` folds into settings via
`prepare_ph_settings`.
**Not ported:** multi-recover output merging (`create_merged_output` walks the recover
chain through provenance and per-iteration outputs cannot be accumulated into one
container with current container types). The graph returns the last iteration outputs.

### `neb_base_graph` — `NebStaticWrapper` placed with `errors=NebErrors` (400, 410, 462–468, 502, 503)

Ported: diagonalization cycling on the nested `pw` parameters, interrupted-trajectory
full restart, `nstep_path` extension reading the **aliased initial parameters**, and
electronic mixing reduction. All statuses from the six `pw`/`neb` calculation types are
exercised through the same handler-task shapes as `pw`.
Wrapper note: the engine cannot close the required *nested-dynamic* `pw` namespace via
`task(..., inputs=...)` (the closed parent keeps its dynamic flag and the placement is
rejected), so `NebStaticWrapper` closes the shape at runtime and propagates the exit
status. Cost: one extra work-chain level per attempt. This is a graph-layer gap, not a
modeling choice.

### `q2r_base_graph`, `matdyn_base_graph`, `bands_base_graph` — single placement, no loop

Faithful: the originals define no process handlers, so any failure aborts without retry.
Each graph runs the calculation once and returns its outputs. Undeclared failures exit
with `ERROR_TASK_FAILED` (workchains abort with `ERROR_UNHANDLED_FAILURE`, 402).

## New static shapes (contracts for composite authors)

- `pseudos` is closed with an explicit single-kind example shape (`{'Si': UpfData}`).
  Structures with other kind names need a copy of the graph with entries added to the
  `task(...)` declaration. Handler tasks are kind-agnostic and reusable as-is.
- `Code` inputs are required. `CalcJob` scheduler options travel in `task_metadata`
  keyed by placement name at launch (every `CalcJob` placement needs an entry).
- Optional `Dict`/`KpointsData`/`RemoteData` inputs are wired as `X | None` (nullable);
  wiring `None` leaves the input out. `max_iterations: int = 5` bounds every loop
  (at most that many runs, matching the original budget semantics).
- Loop graphs return `{'<calc outputs…>', 'report': <summarize node>}`; absent optional
  outputs are skipped. Single-shot graphs return the calc outputs directly (no report).
- Failure vocabulary differs from work chains by design: budget exhaustion and
  unrecoverable handler states exit `400` (`ERROR_TASK_FAILED`) via the guard instead
  of `401`/`402`/handler-specific codes. The `report` records `attempts` + `status`
  (`finished`/`exhausted`); it is only emitted on paths that reach it.

## Graph-layer blockers encountered (authoring notes, verified while building)

- `task()` takes undecorated functions only: the `@calcfunction`s
  `create_kpoints_from_distance` / `rattle_structure` are mirrored as owned tasks
  (`make_kpoints_mesh`, `rattle_positions`); stored nodes carry the new labels.
- Function tasks cannot return ancestor nodes (`CREATE` cycle check): finalizers are
  pure-data tasks and the graph body rebinds structure/parent with no-else branches.
- Task modules must avoid `from __future__ import annotations`: pickled task executors
  resolve string return annotations in reconstructed globals and fail (`NameError`).
- `after=` needs a real placement, not a `None`-valued call; failing-fast validation
  was folded into `make_kpoints_mesh` instead of a separate validator task.
- Graph bodies see only inputs and locally bound names: no module constants/globals
  (literals spelled out, e.g. rattle `stdev = 0.01`), no whole-`TypedDict` wiring into
  task inputs (fields wired one by one), and `TypedDict` task outputs must be total
  (hence plain-data carries + branch selection).
- `run_get_node` on a single-output task returns the node itself (`ResultAndNode`),
  not an outputs dict; multi-output tasks and graphs return dicts.

## Test commands and results

Root environment only (`uv run`, never bare `python`/`pip`). The QE package is not
installed in the root env, so tests run with its source on `PYTHONPATH` plus its two
missing third-party deps overlaid (`--with` changes no tracked files):

Build (validates all six graphs without launching anything):

```
PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo python -c "
from aiida_quantumespresso.workflows.graphs import base
for name in ['pw_base_graph','ph_base_graph','neb_base_graph','q2r_base_graph','matdyn_base_graph','bands_base_graph']:
    print(name, '-> build():', getattr(base, name).build())
"
# pw_base_graph -> build(): None
# ph_base_graph -> build(): None
# neb_base_graph -> build(): None
# q2r_base_graph -> build(): None
# matdyn_base_graph -> build(): None
# bands_base_graph -> build(): None
```

Single representative runtime case, then the full module (skill workflow):

```
PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \
  pytest aiida-quantumespresso/tests/graphs/test_base.py -k "recovers_within_attempts or test_build" \
  --tb=short -p no:cacheprovider --no-cov -o addopts="" -q
# 2 passed, 19 deselected in ~17s

PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \
  pytest aiida-quantumespresso/tests/graphs/test_base.py \
  --tb=short -p no:cacheprovider --no-cov -o addopts="" -q
# 21 passed in ~55-58s
```

(`-o addopts=""` bypasses the repo-wide coverage/benchmark flags for this out-of-`tests/`
path; `-p no:cacheprovider` avoids cache writes in the shared dir.)

Lint: `uv run --with ruff ruff check` on the four owned files → `All checks passed!`;
`ruff format` reformatted 2 files (whitespace only), suite re-run green (21 passed).

What the 21 tests pin: `build()` for all six graphs; loop structure + handled statuses;
all `pw`/`ph`/`neb` handler tasks (claim, pass-through, cycling, exhaustion give-up,
mixing factors, cell-factor doubling, BFGS rattle-once-then-give-up, nmix/alpha walk,
nstep extension from initial inputs); k-points guards; rattle output; and a full
stub-loop runtime (`FlakyPwStep` shares `PwErrors`): recover in 3 runs with report
`{attempts: 3, status: finished}`, rattle-branch recovery in 2 runs, budget exhaustion
→ exit 400, unmapped fatal → exit 400 after exactly 1 run, plus provenance checks (no
`CREATE` links from the graph node, summary node identity, flat inputs).

## Known blockers / non-goals

1. Generic kind names for `pseudos` (static-ports design constraint) — per-structure
   graph copies needed; handler tasks reusable.
2. `NebStaticWrapper` extra provenance level (engine gap for nested-dynamic namespaces).
3. Considered-finished codes (501, 710) and message-gated monitor stops (150) fail
   loudly instead of their original semantics — safe direction, needs composite-level
   decision if silent-success parity is ever required.
4. `ph` multi-recover output merging not portable (provenance walk + container limits).
5. `max_seconds` scheduler capping not portable (constant `task_metadata`).
6. Real-`Code` execution (needs QE executables + `task_metadata` options) is
   untested here and belongs to integration; no daemon/submit coverage in this module.
7. `reports/` and `aiida-quantumespresso` paths are untracked in this worktree;
   nothing was committed or pushed, and no `pyproject`/CI/shared files were changed.
