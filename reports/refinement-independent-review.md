# Independent review: full nested QE workflow integration

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

Scope: verify that the production native graphs (`pw_relax_graph`,
`pw_bands_graph`, `pdos_from_scratch_serial`) run end to end through the
production `pw_base_graph` retry loop with only the external `CalcJob`s
substituted — i.e. real orchestration, real rebinding, real guards — rather
than the whole-base-subgraph `@task` stubs used in `test_pw.py`/`test_pdos.py`.
Owned files: `aiida-quantumespresso/tests/graphs/test_native_workflows.py`
(new) and this report. No production files were changed (see §5).

## 1. What the new tests prove

`test_native_workflows.py` (6 tests) patches **only** the external
`CalcJob` placements and resets every compiled template that embeds them, so
the production graph bodies recompile against the substitutes without being
copied:

- patched: `base.pw_calc` → `task(SuccessPwCalc | FullPwCalc,
  inputs={'pseudos': DataDict}, errors=PwErrors)`; `pdos.dos_calc` →
  `task(StubDosCalc)`; `pdos.projwfc_calc` → `task(StubProjwfcCalc)`;
- reset: `base.pw_base_graph`, `pw.pw_relax_graph`, `pw.pw_bands_graph`,
  `pw.init_relax_guarded`, and all six `pdos` stage/root handles;
- never replaced: `pw_base_graph` itself, or any `pw`/`pdos` production graph.

`SuccessPwCalc` is reused from `test_base.py` by file import (relax path);
`FullPwCalc` (a `PwCalculation` subclass that additionally fabricates
`output_band`, `number_of_electrons`, and `fermi_energy`) plus the
`DosCalculation`/`ProjwfcCalculation` subclasses cover the bands/PDOS paths.
All stubs use trivial `prepare_for_submission` plus a deterministic `parse` —
local fake executables, no QE install needed.

Coverage:

- relax convergence through the real base loop (1 stub run, status
  `converged`, output cell identical to the input cell);
- bands through the real base loop twice (SCF + bands), with SeekPath
  products present, `band_parameters.nbnd == 8`, and the bands run's
  `parent_folder` identical to the SCF run's `remote_folder`;
- serial PDOS from scratch (SCF + NSCF + DOS + PROJWFC), `report.mode ==
  'serial'`, window `emin/emax == -5.0/8.0` eV on every product, and actual
  dispatch order `dos_calc < sweep_remote < projwfc_calc` verified via
  `ctime` ordering;
- provenance shape on every runtime test: root node is a `GraphNode` with
  **no** `CREATE` links, **no** `WorkChainNode` descendant (asserted by node
  type name / `process_type` string — no `WorkChain` import), every remaining
  descendant a `CalcJobNode`/`CalcFunctionNode`/`GraphNode`, the report node
  identical to the stored summary result, and the general `DataDict` pseudos
  mapping reaching every pw `CalcJob` as `pseudos__<kind>` input links.

## 2. Commands and results

From the worktree root (`uv run` only, never bare `python`/`pip`):

- New file:
  `PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with
  aiida-pseudo pytest aiida-quantumespresso/tests/graphs/test_native_workflows.py -q`
  → **6 passed** (~100 s; PDOS serial dominates).
- Whole native-graphs suite:
  `PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with
  aiida-pseudo pytest aiida-quantumespresso/tests/graphs/ -q`
  → **100 passed** (94 pre-existing + 6 new), ~4 min.
- Lint: `uv run --with ruff ruff check <new file>` → clean;
  `ruff format --check` → clean (one import-sort + unused-import autofix
  applied, plus one manual `pseudos.keys()` → `pseudos` fix).

No `xfail`/`skip` (except the standard released-core `graph` guard shared by
all graph test files), no import fallbacks, no copies of production graphs.

## 3. Code review: hidden legacy coupling

- `grep` over `graphs/base.py`, `graphs/pw.py`, `graphs/pdos.py`,
  `graphs/protocols.py`: no `WorkChain` in any `import`/`from` line (only
  docstring references to the legacy work chains being re-expressed, which is
  intentional documentation). The new test file imports no `WorkChain` symbol
  or fixture either.
- Protocol helpers (`protocols.py`) return plain mappings, never builders;
  `test_protocols.py` already pins the no-`WorkChain`-import property and the
  YAML-file existence per family. Shortcoming (by design, documented in the
  module docstring): pseudo-family and meta-parameter (`conv_thr`,
  `etot_conv_thr`) resolution still lives in the legacy
  `PwBaseWorkChain.get_builder_from_protocol` and is deliberately not
  re-implemented — callers pass explicit `Dict` parameters. Not a defect for
  this track, but it bounds what "protocol support" means here.
- Scheduler options ride at launch under `task_metadata` keyed by placement
  name; the nested-scope forwarding was verified live (root stash reaches
  `loop_5_iteration_0__pw_base_graph__pw_calc`). One non-obvious requirement
  the new tests pin down: any graph embedding `pw_base_graph` must be
  launched with a `pw_calc` entry even when the caller never names a
  calculation placement — worth keeping in the user docs.

## 4. Scientific spot-checks (no errors found)

- Band-count rules (`resolve_bands_parameters`, `resolve_nscf_parameters`):
  `max(nbands, 0.5·nelect/·factor, 0.5·nelect+4)` with the
  `nbands_factor`/`SYSTEM.nbnd` mutual exclusion — matches the legacy
  branches these tasks mirror.
- Window extraction: bands min/max plus Fermi energy with the
  spin-resolved `max(fermi_up, fermi_down)` fallback — matches `inspect_nscf`.
- Serial PDOS order DOS → sweep → PROJWFC with unconditional `after` edges —
  matches `serial_clean` semantics; the `ctime` assertion pins it.
- `prepare_next_relax` mesh-regrowth predicate and the `summarize_*` reports
  behave as the existing unit tests pin them.
- Known, documented deviations (in `pw.py`/`pdos.py` docstrings, not
  regressions): exit codes collapse to graph `400`; `scf`-mode relax echoes
  the input structure where the legacy chain emits none; base exit `501`
  fails the step instead of triggering a Pulay rerun.

## 5. Production fixes: none

No production files were modified. One finding belongs to the **test stubs**,
not production: a fabricated `output_band` without an `occupations` array
makes the real `check_pw_sufficient_bands` claim an insufficient-bands
restart on every iteration (missing array → `ValueError` → `nbnd` growth),
exhausting the loop budget. Real `pw.x` parses always carry occupations, so
production is unaffected; the new `FullPwCalc` therefore attaches a two-band
structure with an empty top band (`occupations=[[1,0],[1,0]]`), which passes
the guard while keeping the `[-5, 8]` eV window.

## 6. Limits of this verification

- Single-kind silicon structures with hand-written minimal `UpfData`; the
  general `DataDict` mechanism (not every element) is what is exercised —
  two-kind coverage for the direct base loop already exists in `test_base.py`.
- Success-path stubs only: failure recovery inside the nested loop is covered
  by `test_base.py` at the direct-base level, not re-driven here.
- `presto` profile, deterministic fakes; wall-clock ~100 s for the new file,
  ~4 min for the whole `graphs/` directory.
