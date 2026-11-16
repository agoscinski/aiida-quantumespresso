# PW relax / PW bands graph migration

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

- **Track:** native `@graph` orchestration for `PwRelaxWorkChain` and `PwBandsWorkChain` (`aiida-quantumespresso`).
- **Owned files (this track only):** `src/aiida_quantumespresso/workflows/graphs/pw.py`,
  `tests/graphs/test_pw.py`, `reports/pw-migration.md`. No other QE file, no shared `pyproject`/CI/`__init__`,
  was modified; nothing was committed or pushed.
- **Legacy sources preserved untouched:** `workflows/pw/relax.py`, `workflows/pw/bands.py`,
  `workflows/pw/base.py` and every `calcfunction`/utility they use.
- **Core under test:** parent checkout `graph-benchmark` worktree (fork `graph-authoring-demo-v3` line, `aiida-core
  3.0.0.dev0`); the canonical recipe is `tests/engine/processes/graphs/test_authoring_recipe.py` and the contract
  is `docs/source/topics/workflows/graphs.rst` at that checkout.
- **Environment note:** the legacy `workflows/protocols/utils.py` still does `from plumpy import PortNamespace`,
  but core 3.0 no longer depends on `plumpy`, so the parent `uv run` environment cannot import the QE workflow
  stack as-is. All commands below use a transient overlay that touches no shared file:
  `uv run --with plumpy ...`. The pinned graph CI (`reports/graph-layer-audit.md`, section 5) installs real QE
  test dependencies and will not need this overlay; the overlay is a local-development stand-in only.
- **No parity claimed.** This is an orchestration rewrite with documented gaps, not a drop-in replacement: no
  protocol builders, no `verdi` integration, no exit-code compatibility, no real `pw.x` execution.

## 1. What was built

`src/aiida_quantumespresso/workflows/graphs/pw.py`:

- `PwBaseStep(WorkChain)` — a static-port routing shim: one `PwBaseWorkChain` run with fully static ports
  (`structure`, `parameters`, `code`, `pseudos: DataDict`, `kpoints?`, `kpoints_distance?`, `settings?`,
  `parent_folder?`, `options?`), exposing the base outputs and failing with a single `401` when the base run
  fails. Placed once as `run_pw_base = task(PwBaseStep)` and reused by every graph.
- `@task` adapters with plain bodies, each delegating to the existing implementation (no science in any graph
  body): `prepare_next_relax` (mesh-regrowth predicate + `nbnd` carry, cf. `should_run_relax` success path),
  `prepare_kpoints_path` (explicit-vs-SeekPath branch, cf. `should_run_seekpath`), `resolve_bands_parameters`
  (`nbands_factor` rule, cf. `run_bands`), `ensure_structure` (missing-structure guard), `summarize_relax`
  (recipe-style value/target/attempts/status report).
- `@graph pw_relax_graph` — optional guarded init branch (`init_relax_guarded` subgraph, single-statement arms),
  then a bounded `while rerun and attempts < max_iterations` loop over `run_pw_base` + `prepare_next_relax`,
  then an `if/else` status and the summary. Returns `{output_structure, report}`.
- `@graph pw_bands_graph` — path adapter, SCF run, `nbnd` resolution, bands run with `parent_folder` chaining.
  Returns `{band_structure, scf_parameters, band_parameters, primitive_structure?, seekpath_parameters?}` where
  the SeekPath leaves stay out of the outputs when explicit k-points are given (legacy parity via nullable
  leaves).

`tests/graphs/test_pw.py` (25 tests): `build()` pins for both real graphs plus a placement-only graph over the
real `run_pw_base`; real adapter runs through `run_get_node` (converged/regrow/meta-off/`scf`-passthrough,
`nbands_factor` math, both-validators, SeekPath/explicit/both-modes, guard pass/fail, summary content); and
end-to-end orchestration runs of stub graphs that mirror the real bodies line for line but place a test-local
`StubBase` (convergence, exhaustion, meta-off, init on/off, zero budget, missing-structure failure, bands
SeekPath/explicit modes, report-node identity, flat inputs, no `CREATE` links on the graph node).

## 2. Design decisions and why

- **Shim instead of placing `PwBaseWorkChain` (blocker B1).** `task(PwBaseWorkChain)` is rejected: the exposed
  `pw` namespace is dynamic *and* required (it nests the dynamic, required `pseudos` mapping), and
  `_close_dynamic` only clears one dynamic level under a static parent — nested declarations can never clear a
  dynamic parent, so even `inputs={'pw': {'pseudos': {'Si': UpfData}}}` still fails with ``namespace `pw` is
  dynamic`` (reproduced against the real spec). The shim keeps every placed port static at the cost of one extra
  `WorkChain` node per step. `pseudos` travels as `DataDict` (one concrete node type per run; owned copies);
  scheduler options travel as a `Dict` into `pw.metadata.options`. The sibling `graphs/base.py` track initially
  hit this exact wall (`task(NebCalculation, inputs={'pw': ...})` failing the same way); coordination is still
  open on whether the shim pattern is adopted there too. No `graphs/base.py` API is used here.
- **No `errors=` mapping; `501` fails the graph (blocker B2).** Mapped error arms null out success outputs
  (verified by probe: a `WorkChain` task that stores outputs and exits with a mapped status reaches downstream
  tasks as `None`, with and without explicit `outputs=`). Mapping `501` would therefore lose the relaxed
  structure the legacy retry continues from, so the shim fails on *any* base failure and only success-path
  re-convergence is orchestrated.
- **Undecorated twins via `__wrapped__` (audit G8).** Engine forbids calling a decorated `calcfunction` inside a
  `@task` body; the adapters call the shared undecorated implementations, so the science is identical without
  duplication and without nested provenance nodes.
- **`PortModel` (not `TypedDict`) for task outputs with absent leaves.** Total `TypedDict`s reject `None` for
  required ports at runtime; `BandPath` is a `PortModel` with defaulted nullable fields instead. Graph-level
  `None` leaves stay out of the emitted outputs (covered by the explicit-kpoints stub test).
- **Passthroughs are value-copies.** A task cannot return its input (provenance cycle), so `ensure_structure`
  and the explicit-kpoints arm return `clone()`s — new node identities, documented.
- **Explicit flat contracts, no `Forward`.** The shim surface is small and fully routed, so every port is
  declared explicitly rather than forwarded.

## 3. Exact missing features and deviations

1. **Pulay-stress reruns (legacy `should_run_relax` `501` arm).** A `501` base run fails the graph with `400
   ERROR_TASK_FAILED` instead of starting another iteration. Blocked on B2 (usable outputs on mapped statuses).
2. **Exit codes.** Graphs expose only `400 ERROR_TASK_FAILED`; relax `400/401/402/403` and bands
   `201/202/402/403` collapse into it (plus shim `401` on the child). Outcomes are reported in `report`.
3. **Validators.** `nbands_factor`-vs-`nbnd` and k-points mutual exclusions raise `ValueError` inside tasks
   (task failures) instead of launch-time `ValidationError`s; direct `run_get_node(task, …)` re-raises while
   graph runs return a failed node (engine behavior, covered both ways in tests).
4. **`clean_workdir`, `dry_run`, monitors, `handler_overrides`, base `max_iterations`.** Not routed; the shim
   uses base defaults. Monitor maps additionally need static declarations per name (audit G4).
5. **Init shares the loop's routing.** `run_init_relax` shares code/pseudos/k-points/settings/options with the
   main loop; only `init_parameters` may differ (legacy has fully separate namespaces).
6. **`scf`-mode relax echoes the input structure** as `output_structure` (legacy emits none).
7. **`max_iterations=0` runs zero iterations** and reports `exhausted` (legacy always runs at least once).
8. **SeekPath extras dropped.** Only `primitive_structure`/`explicit_kpoints`/`parameters` cross the adapter;
   any other `seekpath_structure_analysis` products stay out.
9. **`band_structure` softens to absent** when the bands run emits none (legacy would except).
10. **No per-iteration outputs** (engine cap): only the final structure/parameters/report survive; trajectories
    and intermediate meshes are not outputs.
11. **`Computer` cannot cross a graph boundary** (not `Data`); the mocks mint their own localhost stand-in, and
    real scheduler targeting travels as the `options` dict.
12. **Loop cap / stored-only / serial loops** inherit engine limits (1000 iterations, no unstored runs).

## 4. Runtime-test limits

- Real `PwBaseWorkChain` runs need a `quantumespresso.pw` code, computers, and pseudopotentials: **no test here
  launches one.** `build()` validates the real wiring; runtime covers real adapters plus stub-placed orchestration
  graphs. The stub `StubBase` fabricates structures/parameters/`RemoteData`/bands locally and differs
  deliberately from `PwBaseStep` (no codes/pseudos); any drift between the stub graphs and the real graphs is a
  bug in the test file, while the real wiring is pinned by `build()`.
- `seekpath_structure_analysis` runs for real (SeekPath library, no cluster) in the SeekPath-mode tests.
- Profile: `presto` marker + `aiida_profile_clean`, mirroring the recipe module.

## 5. Reproduce

```console
uv run --with plumpy python -c "
from aiida_quantumespresso.workflows.graphs.pw import pw_relax_graph, pw_bands_graph
assert pw_relax_graph.build() is None
assert pw_bands_graph.build() is None
"
uv run --with plumpy pytest aiida-quantumespresso/tests/graphs/test_pw.py -q
```

## 6. Handoff notes

- Key symbols: `PwBaseStep`, `run_pw_base`, `prepare_next_relax`, `prepare_kpoints_path`,
  `resolve_bands_parameters`, `ensure_structure`, `summarize_relax`, `init_relax_guarded`, `pw_relax_graph`,
  `pw_bands_graph`, `BandPath`, `RelaxNext`, `RelaxOutputs`, `BandsOutputs`.
- If core later supports nested dynamic closing (B1) or usable mapped-error outputs (B2), the shim can shrink to
  a placement and the `501` arm can return as a retry predicate; the adapter/task split already isolates those
  changes.
- If the sibling `graphs/__init__.py` import ever fails again, the test module stubs `graphs.base` in
  `sys.modules` and still collects; the stub is a no-op while the real module imports.
