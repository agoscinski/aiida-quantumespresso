# QE composites refinement track

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

Native `@graph` orchestration for PW relax, PW bands, and PDOS. This track owns
only `aiida-quantumespresso/src/aiida_quantumespresso/workflows/graphs/pw.py`,
`pdos.py`, `graphs/protocols.py` (new), `tests/graphs/test_pw.py`,
`test_pdos.py`, `tests/graphs/test_protocols.py` (new), and this report. Shared
`graphs/__init__.py`, CI, and `pyproject.toml` were left untouched; nothing was
committed.

## What changed

- Removed all production `WorkChain` use. `PwBaseStep`, the `run_pw_base =
  task(PwBaseStep)` placement, and the `PdosWorkChain.get_protocol_filepath`
  import are gone from both graph modules. Every `pw.x` run now places the
  native `pw_base_graph` subgraph imported from `graphs.base` (owned by the
  base track).
- Removed the experimental per-run `options` ports (`options`,
  `scf_options`, `bands_options`, per-bundle `options`) instead of silently
  ignoring them. Scheduler options travel at launch under `task_metadata`
  keyed by placement name (`pw_calc` inside the base graph, `dos_calc` /
  `projwfc_calc` for post-processing). `protocols.task_metadata_for_options`
  forms the mapping; `pdos_protocol_task_metadata` lifts the DOS/PROJWFC
  scheduler options out of the PDOS protocol inputs.
- Replaced the `serial_clean` boolean (and its branch-dependent `after` hack)
  with explicit variants. `pdos_stage_parallel` runs DOS and PROJWFC
  concurrently; `pdos_stage_serial` orders DOS → sweep → PROJWFC with an
  unconditional `after` edge, and the from-scratch serial root sweeps the SCF
  remote after the stage. The compiled serial template carries the
  `('sweep_remote', 'projwfc_calc')` ordering dependency; the parallel
  template carries none. `clean_workdir` remains for the final sweep.
  Unqualified aliases (`pdos_stage`, `pdos_from_scratch`, `pdos_from_parent`)
  point at the parallel shapes so the shared `graphs/__init__.py` keeps
  importing.
- New `graphs/protocols.py`: plain mapping helpers only
  (`get_protocol_filepath`, `available_protocols`, `get_default_protocol`,
  `get_protocol_inputs`, `task_metadata_for_options`,
  `pdos_protocol_task_metadata`). They read the same YAML files as the legacy
  chains but import no `WorkChain` subclass: pseudo-family and
  meta-parameter (`conv_thr`/`etot_conv_thr`) resolution still lives in
  `PwBaseWorkChain.get_builder_from_protocol` and is deliberately not
  re-implemented. There is no `get_builder_from_protocol` here by design.
- New tests run the production graphs with stubbed externals instead of copied
  graphs. Runtime tests `monkeypatch` the module-level `pw_base_graph`,
  `dos_calc`, and `projwfc_calc` handles with `@task` stubs and reset each
  graph's `_template`, so the real retry loops, rebinding, window matching,
  guards, and serial ordering execute. No `WorkChain` appears in the new
  tests. Obsolete `StubBase`/`StubPwStep`/`StubPost` stand-ins, copied
  `stub_*` graphs, and the old import-fallback guard are removed.

## Verification

`build()` first, then one representative runtime case, then the full files,
per the authoring recipe (all with `uv run`, never bare `python`/`pip`):

- `ruff check` and `ruff format`: clean on all six owned files.
- Isolated run against the agreed base contract
  (`code, structure, pseudos: DataDict, parameters, settings?, kpoints?,
  kpoints_distance?, kpoints_force_parity?, parent_folder?` →
  `output_parameters/output_structure?/output_band?/remote_folder/report`):
  **65 passed** (`test_protocols.py` 9, `test_pw.py` unit + 7 production
  runtime, `test_pdos.py` unit + 7 production runtime across serial and
  parallel roots).
- Representative cases: `test_production_relax_converged`,
  `test_production_from_scratch_parallel`, and the serial-ordering
  dependency assertion all pass in the isolated run.

## Residual failures (exact, at HEAD)

The full in-tree suite cannot collect yet for reasons outside this track:

1. `graphs/base.py` (base track) is mid-migration and unimportable at HEAD:
   the `WorkChain`/`ToContext` imports were removed while
   `NebStaticWrapper(WorkChain)` and `ToContext` usages remain (`NameError`
   at import), and `pw_calc = task(PwCalculation, inputs={'pseudos':
   orm.DataDict}, ...)` is rejected by the core (`GraphTypeCheckError`:
   `` `pseudos` is already a declared port``). Every test module fails at
   collection through the shared `graphs/__init__.py` until the base track
   lands. This file was deliberately not touched.
2. The agreed `pw_base_graph` contract (`pseudos: DataDict`,
   `parent_folder`, node-compatible `parameters`) is not yet implemented by
   the base track (current signature still takes singular `pseudo: UpfData`
   with no `parent_folder`). Unpatched `build()` checks of the owned graphs
   will fail with ``'pseudos' is not an input of 'pw_base_graph'`` until the
   base worker updates the contract.
3. Whole-`DataDict` forwarding for the `pseudos` bundle depends on the
   namespace/core worker; the owned graphs already wire `pseudos: DataDict`
   straight through (no `Forward` bundles remain in `pdos.py`), minimizing
   exposure to that change.

The 65-test isolated result above was obtained by loading the real owned
files against a stub `base` module exposing `pw_base_graph` under the agreed
contract, without modifying any shared file (runner kept outside the repo
at `/tmp/run_owned_tests.py`).

## Supported features

- PW relax: success-path re-convergence (mesh-regrowth reruns, bounded
  budget, optional init relax), SCF-mode structure echo, SeekPath/nbands
  resolution for bands, `nbands_factor` rule, summary reports returning
  existing nodes.
- PDOS: from-scratch and from-parent roots in both serial and parallel
  shapes, Fermi-window extraction (including spin-polarized fallback),
  Fermi-relative ranges, `nbands_factor` from parent outputs, DOS/PROJWFC
  grid-equality enforcement before submission, prompt plus final remote
  sweeps.
- Protocols: YAML loading/merging/overrides for `pw_relax`, `pw_bands`,
  `pdos`, and `task_metadata` formation. Pseudo-family and threshold
  resolution are explicitly out of scope.
