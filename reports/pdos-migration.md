# PDOS graph migration

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

Native `@graph` rewrite of `PdosWorkChain`
(`aiida-quantumespresso/src/aiida_quantumespresso/workflows/pdos.py`)
as `aiida_quantumespresso.workflows.graphs.pdos`.
The legacy `WorkChain` is preserved untouched; the graphs orchestrate the
original base work chains and the original `dos.x` / `projwfc.x`
calculations instead of re-implementing them.

Run the checks from the parent core checkout:

```console
uv run --with plumpy pytest aiida-quantumespresso/tests/graphs/test_pdos.py -q
```

(`plumpy` is a transient overlay: the legacy `ProtocolMixin` still imports
it, while the core environment no longer ships it. The graphs themselves do
not need it.)

## What was built

Three graphs in `workflows/graphs/pdos.py`, all validated with `build()`:

| Graph | Legacy equivalent | Shape |
|---|---|---|
| `pdos_from_scratch` | `PdosWorkChain` **with** `scf` inputs | SCF step → shared stage, plus SCF sweeps |
| `pdos_from_parent` | `PdosWorkChain` **without** `scf` (`nscf.pw.parent_folder` given) | caller folder → shared stage |
| `pdos_stage` | `run_nscf` → `inspect_nscf` → DOS/PROJWFC → `results` | NSCF step → window → matched params → DOS + PROJWFC → report |

The legacy workflow runs SCF optionally, which is a dynamic (optional
namespace) input. Graphs only take static contracts, so the rewrite exposes
two static roots instead of one dynamic one, sharing `pdos_stage`. This is
the same trade the PW graphs make for optional inputs.

Science lives in `@task` collaborators plus the placing work chains; the
`@graph` bodies only wire names and literals:

- `PwBaseStep` / `run_pw_base` (shared with `workflows/graphs/pw.py`,
  sibling-owned) routes each base run through static ports and submits the
  real `PwBaseWorkChain`, so restarts, handlers, and error handling stay
  exactly where they were.
- `require_scf_parameters`, `resolve_nscf_parameters` (incl. the
  `nbands_factor` rule), `extract_nscf_window`, `resolve_post_windows`,
  `summarize_pdos`, `sweep_remote`, `sweep_stage` hold the remaining
  workchain logic: calculation-type/occupations checks, the band-count
  maximum, the bands-min/max + Fermi fallback, the DOS parameter assembly,
  the `emin`/`emax`/`deltae` equality check, the summary report, and the
  remote-folder cleaning.
- `dos_calc = task(DosCalculation)` and `projwfc_calc = task(ProjwfcCalculation)`
  place the original calculations directly.

## Input mapping

| Legacy `PdosWorkChain` input | Graph input | Notes |
|---|---|---|
| `structure` | `structure` (each root/stage) | injected into both base runs, as before |
| `scf.{pw.*, kpoints…}` | `scf` bundle + `scf_parameters` | `Forward(run_pw_base)` minus `structure`, `parent_folder`, `parameters`; the bundle carries `code`, `pseudos`, `kpoints`, `kpoints_distance`, `settings`, `options` |
| `nscf.{pw.*, kpoints…}` | `nscf` bundle + `nscf_parameters` (+ `parent_folder` / `parent_output_parameters` wired by the caller) | same shape; `parent_folder` is the SCF remote (scratch) or the caller folder (parent) |
| `dos.{code, parameters, settings}` | `dos_code`, `dos_parameters`, `dos_settings` | flat; `parent_folder` wired by the graph |
| `projwfc.{code, parameters, settings}` | `projwfc_code`, `projwfc_parameters`, `projwfc_settings` | flat; `parent_folder` wired by the graph |
| `nscf.pw.parent_folder` (no-SCF mode) | `parent_folder` of `pdos_from_parent` | direct input instead of a nested folder |
| `energy_range_vs_fermi` | `energy_range_vs_fermi: list \| None` | same pair-of-numbers semantics; wins over the window |
| `nbands_factor` | `nbands_factor: float \| None` | same maximum rule; from-parent runs also take `parent_output_parameters`, mirroring the legacy read of the parent creator outputs |
| `serial_clean`, `clean_workdir` | `serial_clean`, `clean_workdir` (default `False`) | sweep-only now, see limits |
| `dry_run` | — | dropped; tests use mocks instead (graphs reject `dry_run` metadata) |
| `kpoints_force_parity`, `clean_workdir`, `max_iterations`, `handler_overrides`, `vdw_table`, `hubbard_file`, `parallelization` inside `scf`/`nscf` | — | not exposed by the shared `PwBaseStep` (sibling-owned surface); see limits |

Scheduler options travel two ways, matching how the engine splits them:
base-run options ride in each bundle's `options` field (forwarded to
`pw.metadata.options` by the step); `dos.x`/`projwfc.x` options ride under
`task_metadata` at launch, keyed by the stable declaration names:

```python
run_get_node(
    pdos_from_scratch,
    ...,
    task_metadata={'dos_calc': {'options': {...}}, 'projwfc_calc': {'options': {...}}},
)
```

A compile-time test pins those two placement names across all three graphs.

## Output mapping

The legacy chain exposes whole namespaces (`scf__remote_folder`,
`nscf__output_band`, `dos__output_dos`, `projwfc__Dos`, …). The graphs
return flat leaves with the same grouping, plus a `report`:

- `scf_remote_folder`, `scf_output_parameters`, `scf_output_band | None`
  (scratch root only)
- `nscf_remote_folder`, `nscf_output_parameters`, `nscf_output_band`
- `dos_output_parameters`, `dos_output_dos`
- `projwfc_output_parameters`, `projwfc_dos` (legacy `Dos`), `projwfc_ldos`,
  `projwfc_pdos`, `projwfc_projections{,_up,_down}`, `projwfc_bands{,_up,_down}`
  (absent optionals stay out of the outputs)
- `report`: `{emin, emax, fermi_energy, nbands, stages, mode, status}` where
  `emin`/`emax` are the *covered* window, `stages` counts the PW base runs
  (2 from scratch, 1 from parent), `mode` follows `serial_clean`, and
  `status` is `'completed'` (the report is success-path only)

Provenance keeps the recipe shape: the root creates nothing (no `CREATE`
links), inputs stay flat on the stored node, task outputs are returned
unchanged, and the `report` is the summary task's own node.

## Static contracts where the legacy is dynamic

- `PwBaseWorkChain` cannot be placed directly: its `pw` namespace is
  dynamic *and* required (it nests the dynamic, required `pseudos`
  mapping), and the graph compiler only closes one dynamic level under a
  static parent. The shared `PwBaseStep` adapter routes each run through
  static ports at the cost of one extra `WorkChain` node per run.
  Pseudopotentials travel as an `orm.DataDict` (kind name → `UpfData`).
- The optional SCF namespace became two roots (see above).
- The `monitors` mappings stay unwired (dynamic, optional), matching the
  engine's "no monitors yet" limit.
- Validators run as tasks now: wrong `CONTROL.calculation`, non-`tetrahedra`
  occupations, missing `deltae`, malformed `energy_range_vs_fermi`, the
  `nbands_factor`/`SYSTEM.nbnd` clash, and grid inequality all fail their
  task (propagated as the task exception) instead of failing launch-time
  validation. The two legacy *warnings* (SCF + parent folder both given;
  range + explicit `emin`/`emax`) have no graph equivalent: the first is
  unrepresentable with two roots, the second resolves silently in favour of
  the range, as before.

## Restart / error / cleanup / protocol parity

- **Restart:** unchanged inside `PwBaseWorkChain` (same class, same
  handlers, same iteration budget). The graph adds no retry layer: a failed
  base run fails the graph, and dependents never start. Graph-level restart
  reuses finished tasks from the checkpoint, as for any graph.
- **Error:** graphs declare no exit codes. The legacy `401`–`404` (plus the
  restart `301`/`302`/`401`/`402`) collapse into the single graph `400
  ERROR_TASK_FAILED`. Stage attribution survives in the child exit statuses;
  the summary `report` is only emitted on success. Declared-failure routing
  (`errors=`) was deliberately not added: the base chain alone has dozens of
  exit statuses, so any partial mapping would be misleading.
- **Cleanup:** `serial_clean` sweeps each remote promptly after its
  calculation finishes, and `clean_workdir` sweeps the remaining remotes at
  the end through no-op-by-default tasks — *but neither serializes
  DOS before PROJWFC anymore*. Conditional ordering between two placements
  is inexpressible in the graph language (branch arms must bind the same
  names with the same wiring, and `after` cannot cross a branch boundary),
  so both calculations stay dispatch-ready and the engine may run them
  concurrently. Plan scratch-disk headroom of the legacy parallel mode; the
  serial flag still saves the steady-state footprint. The SCF sweep waits on
  the whole stage (`after=stage`), later but safer than the legacy
  post-NSCF sweep.
- **Protocols:** no `get_builder_from_protocol` equivalent. The protocol
  presets stay on the legacy work chain (whose `ProtocolMixin` still needs
  the `plumpy` overlay under core 3.0); the module only re-exposes
  `get_protocol_filepath()` for tooling. Graph launches take explicit
  inputs, as in the launch sketch above.

## Tests

`aiida-quantumespresso/tests/graphs/test_pdos.py` (33 tests):

- `build()` for all three graphs plus a placement pin for both CalcJobs —
  catches syntax, wiring, and contract errors without tasks.
- Unit runs of every `@task` through `run_get_node`: the three legacy
  energy-range cases (explicit / Fermi-relative with split Fermi /
  bands-default), wrong-calculation and wrong-occupations rejections, the
  `nbands_factor` rule and its mutual-exclusion/missing-parent failures,
  missing `deltae`, malformed ranges, grid match/mismatch, the summary
  content, and disabled sweeps.
- Mocked runtime of stub graphs that mirror the real bodies line for line
  except the placed steps (and the code-carrying bundles): from-scratch
  parallel incl. provenance shape (no root `CREATE` links, flat inputs,
  summary-node identity), serial mode, from-parent with `nbands_factor` and
  with `energy_range_vs_fermi`, and the three graph-level failures exiting
  `400`. Stubs fabricate data through `calcfunction` collaborators (work
  chains cannot create nodes inline) and never need codes or a cluster.

What the mocks do not cover: real `PwBaseWorkChain` runs (restarts,
handlers, `501` paths), real `dos.x`/`projwfc.x` parsing, scheduler options
under `task_metadata`, exit-status attribution beyond `400`, and concurrent
vs serial dispatch timing.

## Cross-worker notes

- Reuses `run_pw_base` / `PwBaseStep` from `workflows/graphs/pw.py`
  (sibling-owned): any change to that step's surface ripples into the
  `ScfBundle`/`NscfBundle` contracts and the explicit wiring here. The
  `build()` tests pin the joint.
- The shared `workflows/graphs/__init__.py` re-exports the sibling-owned
  `base` module; while that module is being reworked, its import-time
  failure would block these tests too, so the test module falls back to
  loading `pdos` straight from its file location. The fallback only
  activates when the plain import fails.
