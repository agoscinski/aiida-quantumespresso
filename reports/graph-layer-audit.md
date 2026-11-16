# Graph-layer migration feasibility audit — `aiida-quantumespresso`

- **Audit scope:** can the current `WorkChain` layer be rewritten onto the `@graph`/`@task` API, and what must the integration CI pin to test it honestly?
- **Core under audit:** parent checkout at `3db2173f3c4761a4968595b2fdfeb7f3c63d8d32` (fork `agoscinski/aiida-core`, branch `graph-authoring-demo-v3`; verified pushed, branch head == SHA).
- **QE checkout under audit:** `rewrite` branch at `7568266` (`PpCalculation: PARENT_FOLDER_SYMLINK`).
- **Owned files (audit track only):** this report (`reports/graph-layer-audit.md`) and `.github/workflows/graph.yml`. No changes were made to existing CI, `pyproject.toml`, workflow sources, or tests; recommendations to the parent are listed, not applied. `tests/graphs/**` and any graph implementations belong to the implementation track and were intentionally not touched. Nothing was committed or pushed.
- **Method:** read the graph implementation (`src/aiida/engine/processes/graphs/`, `docs/source/topics/workflows/graphs.rst` at the pinned core commit), every module under QE `src/aiida_quantumespresso/workflows/`, QE `pyproject.toml`, `tests/conftest.py`, and the workflow test layout; confirmed each reported gap with a minimal `build()`-time reproduction against the local core checkout where feasible.

## 1. Version / dependency context (why a dedicated CI track is needed)

- QE `pyproject.toml` pins `dependencies = ['aiida_core[atomic_tools]~=2.8', ...]`. The graph API (`aiida.engine.graph`, `task`, `par_map`, `Forward`, `Result`/`Ok`/`Err`) exists only on the pinned fork revision. A plain `uv pip install -e '.[tests]'` therefore resolves to the **released 2.8 core and silently never exercises graphs**.
- The existing `nightly.yml` already installs `aiida-core@main` for a *released-core-compat* signal; that is a different question from *graph-migration* signal, which needs the exact fork SHA. Hence the dedicated `graph.yml` workflow (Section 5), which installs in two steps and then **fails loudly** unless `importlib.metadata.version('aiida-core')` is 3.x installed from the pinned `direct_url.json`.
- Python is unproblematic: core requires `>=3.10`, QE classifiers cover 3.10–3.14. `graph.yml` uses 3.12 (also the local dev interpreter); extend to a matrix later.

## 2. Migration surface inventory

Nine workflow entry points (`pyproject.toml` `[project.entry-points.'aiida.workflows']`):

| Entry point | Class | Role |
|---|---|---|
| `quantumespresso.pw.base` | `PwBaseWorkChain` (`BaseRestartWorkChain`, ~840 lines, ~10 `@process_handler`s) | restart loop, heaviest handler logic |
| `quantumespresso.ph.base` | `PhBaseWorkChain` (`BaseRestartWorkChain`, ~420 lines, 4 handlers + q-point merge) | restart loop + recover-chain merge |
| `quantumespresso.neb.base` | `NebBaseWorkChain` (`BaseRestartWorkChain`, ~320 lines, 4 handlers) | restart loop |
| `quantumespresso.pw.relax` | `PwRelaxWorkChain` (`WorkChain`, ~420 lines) | meta-convergence orchestrator over `PwBase` |
| `quantumespresso.pw.bands` | `PwBandsWorkChain` (`WorkChain`, ~300 lines) | seekpath → scf → bands orchestrator |
| `quantumespresso.pdos` | `PdosWorkChain` (`WorkChain`, ~650 lines) | scf → nscf → dos+projwfc (serial/parallel fan-out) |
| `quantumespresso.bands.base` / `matdyn.base` / `q2r.base` | thin `BaseRestartWorkChain`s (~60 lines each) | almost handler-free restart loops |

**Strategically important:** `task()` accepts `WorkChain` subclasses, and a `WorkChain` task keeps its own outline/handlers internally (CORE `lowering.py: task()` overloads; docs “Task kinds”). So the viable migration shape is **incremental**: keep the three `*BaseWorkChain` restart loops as `WorkChain` tasks while rewriting the orchestrators (`relax`, `bands`, `pdos`) as graphs first. Most handler-related gaps below then apply to phase 2 (rewriting the base chains themselves), not phase 1 — except where orchestrators *read* sub-process failure state, which they do pervasively (Section 3, G2/G3).

Supporting machinery that has no graph home: `get_builder_from_protocol` classmethods on every chain, `ProtocolMixin.get_protocol_inputs` / `recursive_merge` / `set_default_resources` (`workflows/protocols/utils.py`), and `prepare_process_inputs` (`utils/mapping.py`). `GraphHandle` exposes `get_launch_inputs`, not `get_builder` (CORE `lowering.py: GraphHandle`), so the whole protocol→builder pipeline must be re-homed into plain functions, with per-`CalcJob` scheduler options moving from `metadata.options` namespaces to launch-time `task_metadata` keyed by placement name (docs “Calculation job options”).

## 3. Concrete graph-layer gaps (blockers or API-breaking workarounds)

Each item names the QE pattern, the graph-side wall, evidence, and (where small) a minimal reproduction confirmed against the pinned core. Repros below are complete files run with the local checkout (`uv run python <file>`); all raise at `build()` time unless noted.

### G1 — Graphs cannot declare workflow exit codes (blocker for API parity)

`GraphProcess` defines exactly one exit code, `400 ERROR_TASK_FAILED` (CORE `process.py:101`, `_finish` at `process.py:269`). There is no `spec.exit_code` equivalent for `@graph`. The QE surface defines ~20 meaningful statuses consumed by callers and asserted in tests: `PwRelax` 400/401/402/403, `Pdos` 401–405 (incl. distinct DOS-vs-PROJWFC-vs-both failures), `PwBase` 201/202/310/501/710, `PhBase` 401 `ERROR_MERGING_QPOINTS`, validators returning `exit_codes.*.message` strings. A graph rewrite collapses all of these to 0-or-400. Preserving them needs either a core feature (custom graph exit codes) or an out-of-band side channel (a task output that callers must read instead of `exit_status`) — both breaking changes for `verdi process show` consumers and every `exit_status` assertion in `tests/workflows/`.

### G2 — Declared-failure handlers receive only the enum member, never the failed calculation's data (blocker for base-chain rewrite)

QE handlers routinely continue **from the failed calculation's outputs**: `handle_out_of_walltime` reads `outputs.remote_folder` and (when present) `outputs.output_structure` (`pw/base.py`, `PhBase` `handle_out_of_walltime`, `NebBase` equivalents); `sanity_check_insufficient_bands` reads `calculation.inputs.parameters` and `outputs.output_band`; `handle_relax_recoverable_*` reads `outputs.output_structure` / `output_trajectory`; `PhBase.create_merged_output` walks the recover chain via `inputs.parent_folder.creator`. In graphs, a child finishing with a mapped status “counts as completed, with its success outputs absent” (docs “Declared task failures”), and error tasks type their input as the enum itself (e.g. `recover_zero(error: DivisionErrors)`). There is no handle on the failed node, so `FROM_CHARGE_DENSITY` / `FULL` / `FROM_WAVE_FUNCTIONS` restarts — which need the previous `remote_folder` — cannot be expressed. The `while`-retry recipe (docs “Retrying declared failures”) retries with inline counters only.

### G3 — No priority-ordered, overlapping, or message-dispatched handlers (blocker for `PwBase`)

`PwBaseWorkChain` registers ~10 handlers with explicit priorities (585–602) over **overlapping** exit-code sets, plus an `ExitCode(0)` handler (`sanity_check_insufficient_bands`) that can force a restart of a *successful* calculation, plus a message-prefix dispatch (`handle_electronic_convergence_stuck` checks `ACCURACY_STUCK_MESSAGE_PREFIX` because `STOPPED_BY_MONITOR` is shared by all monitors, `pw/base.py`). Graph error dispatch is one-member→one-arm with `is`/`is None` tests (docs “Declared task failures”); there is no priority, no fallthrough, no exit-0 re-entry, and no access to `exit_message`. The diagonalization-rotation handlers additionally keep cross-attempt memory (`self.ctx.diagonalizations`, `self.ctx.rattled`) — re-expressable as inline state, but only once G2 is solved.

### G4 — Dynamic `monitors` mappings are not wirable (blocks protocol overrides)

`CalcJob.monitors` is a genuinely open-ended namespace (arbitrary keys), and QE protocols inject it via overrides, e.g. `{'pw': {'monitors': {'monitor1': {'entry_point': ...}}}}` (`tests/workflows/protocols/pw/test_base.py:306`; `protocols/utils.py` dynamic-marker logic). Graph dict syntax “only fills fixed namespaces” (docs “Known limitations”: “No monitors yet”), and `task_metadata` carries only `metadata`, never process inputs (CORE `tasks.py: _merge_task_metadata`). Confirmed:

```python
add_calc = task(ArithmeticAddCalculation)

@graph
def g(x: int, y: int) -> int:
    total = add_calc(x=x, y=y, monitors={'m1': x})  # GraphLanguageError:
    return total.sum                                # 'm1' is not a declared name here
```

A static declaration (`task(..., inputs={'monitors': {'m1': Dict}})` with a matching graph input) **does** build (verified), so the workaround is to freeze monitor names at authoring time and promote them to explicit graph inputs — a user-API change, and it cannot cover protocol overrides that invent names per launch.

### G5 — No `dry_run` (blocks `Pdos` test path)

Graph metadata rejects `dry_run` (docs “Known limitations”: “No dry runs”). `PdosWorkChain` threads a `dry_run` input through every submit (`pdos.py: setup/run_scf/run_nscf/run_dos_serial/...`) as its offline path. No equivalent exists; the rewrite must drop the feature or reimplement it as a launch wrapper outside the graph.

### G6 — No input/output validators; `PortModel` rejects pydantic validators (API change)

Docs “Known limitations”: “port types are checked, and `PortModel` fields reject pydantic validators; check values inside a task.” QE relies on validators heavily: cross-namespace checks (`validate_inputs`: scf-vs-parent_folder presence, DOS-vs-PROJWFC `emin/emax/deltae` equality; `validate_scf/nscf/dos/projwfc`; `nbands_factor`-vs-`nbnd` mutual exclusion; kpoints exclusivity), `jsonschema` parameter schemas (`get_parameter_schema`), and protocol-override key warnings (`protocols/utils.py: recursive_key_check`, incl. the dynamic-namespace `monitors`/`pseudos` carve-outs). All of these must move into `@task` bodies (one extra child process per check, and failures surface as task failures, not launch-time `ValidationError`s) or into un-validated launch wrappers.

### G7 — Graph-body subset rejects the orchestrators' Python (forces task-ification of pure logic)

The graph body allows only calls, `if`/`while`, inline arithmetic/comparison, and name/field wiring; everything else raises `GraphLanguageError` naming the source line. Confirmed rejections, each mirroring a QE orchestrator pattern:

- dict temporaries + `.get` (Pdos `_generate_dos_inputs/_generate_projwfc_inputs` dict surgery): `d = {'a': x}` → “outputs stay flat leaves; list each output by name instead of nesting them” (probe `g_dict_get`).
- `.get_dict()`/`.setdefault()`/attribute reads on nodes ( Relax `setup`, `run_bands` `nbnd` math, `should_run_relax` mesh comparison): same class of rejection — node payloads are only readable inside `@task`s.
- f-strings (all `report()` calls): “formatting is computation; build this text in a task instead” (probe `g_fstring`). Graphs have no `report()` at all.
- `try/except` (defensive output reads, e.g. `output_structure` `NotExistent` handling in `inspect_relax`): “only calls, branches, and loops belong here” (probe `g_try`).
- `while/else` → “`while` loops need a named check and no `else`” (probe `g_while_else`); input rebinding at top level → “graph inputs are recorded once at launch and cannot be reassigned” (probe `g_inline_return`); returning a bound literal at root → “graphs return existing nodes rather than creating data … create the value in a `@task`” (probe `g_inline_only`).

Concretely, `PwRelax.should_run_relax` (Pulay-stress check via `verify_convergence_stress(trajectory: TrajectoryData, ...)` in `utils/validation/trajectory.py:91`, plus k-mesh regrowth comparison that resubmits `create_kpoints_from_distance` with `store_provenance=False`) must become one or more `@task`s. That is feasible — tasks receive ORM types — but each former inline check becomes a stored child process, and unstored runs are unsupported (docs “Stored provenance only”), so the `store_provenance=False` kpoints probes (`relax.py: run_relax` mesh check, `ph/base.py: set_qpoints`) become permanent provenance nodes.

### G8 — `calcfunction`s cannot be placed directly (mechanical but provenance-visible)

`_check_task_contract` rejects decorated process functions: “tasks take plain functions, not decorated process functions” (CORE `tasks.py`). QE calls `create_kpoints_from_distance`, `seekpath_structure_analysis`, `rattle_structure`, `merge_ph_outputs` as `@calcfunction`s mid-workflow. Each needs an undecorated twin wrapped in `@task` (node type stays `CalcFunctionNode`, but link labels/call sites change and `metadata.call_link_label` values like `'seekpath'`, `'create_kpoints_from_distance'`, `'iteration_%02d'` must be re-expressed as placement names).

### G9 — No lifecycle hooks: `on_terminated` remote cleanup has no equivalent

`PwRelax.on_terminated` and `Pdos.on_terminated` (+ `clean_workchain_calcs`/`clean_calcjob_remote`) walk `called_descendants` to `_clean()` remote folders when `clean_workdir=True`. Graphs are plain functions — no `on_terminated`, no `ctx`. A finalizer `@task` could do the cleaning, but it would be a side-effectful local computation masquerading as science, and failure semantics (cleanup must run even on failure paths) cannot hook into graph termination. Defer or drop `clean_workdir` in phase 1.

### G10 — No per-iteration outputs; recover-chain merging is inexpressible

Docs: “a graph output holds one node, so a loop cannot return one result per iteration”. `PhBase.create_merged_output`/`_get_recover_chain` (`ph/base.py`) merges `output_parameters` across an *a priori unknown* number of recover-linked `PhCalculation`s via `merge_ph_outputs`. A graph loop has a static placement set and no access to `ctx.children` / creator-link walking, so variable-length recover chains cannot be merged inside the graph. (The suggested “copy values into one list node” loses node identity — unacceptable for `output_parameters` that downstream provenance expects.) Same wall hits any relax-trajectory-per-iteration output. A loop cap of 1000 iterations (CORE `ir.py:773`) is fine against protocol defaults (`max_iterations: 5`, `max_meta_convergence_iterations: 5`).

### G11 — Parallel fan-out limits: `par_map` excludes the Pdos shape

Pdos runs `dos.x` + `projwfc.x` in parallel via explicit `ToContext` double-submit (`run_pdos_parallel`), and serially with interleaved cleanup otherwise. `par_map` maps a **unary** callable over an owned `DataList`/`DataDict`, and “mapped `CalcJob` tasks and per-item metadata are not supported yet” (docs). The Pdos fan-out (two *different* `CalcJob`s over the *same* parent folder) is therefore not a `par_map`; it stays two explicit placements — which the graph model supports (independent tasks run without ordering edges), but the serial-clean variant (G9) does not transfer.

### G12 — Provenance and test churn (migration cost, not a layer gap)

Nested graphs run as scopes of one `GraphProcess` node (no intermediate process nodes); task call links use placement names with hierarchical labels (`multiply_add__multiply`, `loop_1_iteration_0__add`). QE regression tests assert exact link labels (`iteration_01`, `scf`, `bands`, `seekpath`, …), `serialize_builder` dicts, and `called_descendants` walks — all must be rewritten, and `tests/conftest.py` fixtures (`generate_workchain_pw`, `serialize_builder`, `pseudo_family`, …) assume `WorkChain` builders/specs. Budget for a near-total rewrite of `tests/workflows/` (14 files) alongside the source rewrite.

## 4. What migrates cleanly (phase-1 candidates)

- **Linear chains with static wiring:** `PwBands` (seekpath calcfunction → scf `WorkChain` task → bands `WorkChain` task) maps to straight-line placements + one `if` (`should_run_seekpath` → `bands_kpoints` presence check wires as `is None`, supported). `run_bands` `nbnd` arithmetic becomes either inline expressions (max-of-three over launch-known ints — inline `if` expressions are supported) or a small `@task`.
- **Optional-path orchestration:** `Pdos.should_run_scf`, `PwRelax.should_run_init_relax`, meta-convergence flag-off paths are presence/boolean branches — natively supported.
- **Thin base chains** (`bands`/`matdyn`/`q2r` bases, ~60 lines, no handlers) are the easiest full rewrites and good pilot targets: single `CalcJob` task in a bounded `while`, `Result`-less (plain failure → `ERROR_TASK_FAILED`), outputs passed straight through.
- **Bounded retry loops** where recovery needs no failed-node data (e.g. `NebBase.handle_neb_cycle_exceeded_nstep`, which only bumps `nstep_path` and restarts `FULL`) fit the documented `while` + inline-counter recipe.

## 5. Integration CI (`graph.yml`): what it does and its limits

What the workflow does (and why): two-step install (QE test deps first, then overwrite `aiida-core` with `git+https://github.com/agoscinski/aiida-core@3db2173…`), an anti-silence verification step (asserts version 3.x, `direct_url.json` contains the pinned SHA, `aiida.engine.{graph,task,par_map}` importable), then `pytest tests/graphs -v` when that directory exists and `pytest tests/workflows -q` as the legacy regression gate; `postgres:12` + `rabbitmq` services and `AIIDA_WARN_v3: 1` mirror `ci.yml`/`nightly.yml`.

Known limitations / accepted risks:

1. **Manual SHA bumps.** The fork revision is pinned by hash; core-side progress does not flow in. There is no Dependabot-for-SHA mechanism here — bumping is a deliberate PR action. Recommend the implementation track state the tested core SHA in every graph PR until QE floats a 3.x bound.
2. **No released-core signal.** `pyproject.toml` still says `~=2.8`, so PyPI installs never see graphs; this workflow is the *only* graph signal. Do not read it as release readiness.
3. **Guarded `tests/graphs` step.** The directory does not exist yet (owned by the implementation track). The step warns instead of failing so the regression leg still gates the fork; tighten to unconditional once the first graph tests land.
4. **Legacy scope is `tests/workflows` only** (14 files), not the full 74-file suite — enough to catch fork breakage on the migration surface without paying for parsers/calculations coverage that `ci.yml` already owns.
5. **Single Python (3.12).** Compatible with both bounds; widen to a matrix when the track matures.
6. **Provenance-store services** mirror `ci.yml` exactly (including its minimal `postgres` stanza); if graph tests need a live broker/profile beyond what `aiida.tools.pytest_fixtures` provides, align with `ci.yml` first and record the delta here.

## 6. Actionable recommendations (to the parent — proposed, not applied)

**Do not change yet** (kept out of this audit's files per the no-touch rule):

1. `pyproject.toml`: keep `aiida_core~=2.8` until a graph-based release exists; when the first graph ships, add an explicit lower bound / extra documenting the 3.x requirement rather than silently widening the range.
2. Existing CI (`ci.yml`/`nightly.yml`): no changes proposed. If `graph.yml` proves stable, the follow-up is to fold the pin-verification pattern into `nightly.yml` as a second axis — not to replace anything.

**Core feature requests** (ordered by how much QE migration they unblock), each traceable to a gap above:

1. Custom graph exit codes (G1) — without these, no QE orchestrator can be rewritten with API parity.
2. Failed-node data access in declared-failure arms, or “usable outputs on mapped statuses” for cases like 501 (G2) — the single biggest rewrite blocker after exit codes.
3. Dynamic `monitors` wiring for `CalcJob` tasks (G4); static declaration works (verified) but protocol overrides need dynamic keys.
4. Launch/validation hooks or `PortModel` validators for cross-namespace checks (G6).
5. `dry_run` / unstored-task support (G5, G7) for the Pdos offline path and `store_provenance=False` probes.
6. Per-iteration/collector outputs preserving node identity (G10) for recover-chain merges.
7. A finalizer/lifecycle hook for `clean_workdir` semantics (G9).
8. A `get_builder`-equivalent / documented protocol-helper pattern for `GraphHandle` (Section 2), so `get_builder_from_protocol` has somewhere to go.

**QE-side migration order** (for the implementation track): pilots `bands/matdyn/q2r` bases → `PwBands`/`Pdos` orchestrators over intact `WorkChain` base tasks → `PwRelax` meta-loop → base restart chains last (needs G1–G3 resolved). Rewrite `tests/workflows/` assertions (labels, builders, exit statuses) per workflow, not in one flag day.
