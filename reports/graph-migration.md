# Native QE graph rewrite — structured configuration iteration

## Verdict

**Native graphs use no WorkChains, but the rewrite is not feature-complete.**
This iteration first implements `task.md` in core, then adapts QE authoring to that
contract rather than adding further engine features for QE-specific behavior.
Legacy workflow entry points remain available and are not called by native graphs.

## Iteration 1: one focused core replacement

Core feature commit: **`062b26173b30ae29714dd6c2f63f1a6d4a5b839f`**.
Pushed integration branch: `agoscinski/aiida-core@refactor/structured-graph-metadata`.
QE's exact dependency pin: **`955f04745d114bf4c51b219de869e1bab14b7e64`**.
The integration head adds only the existing, separately committed DataDict builder
and file-staging fixes after the focused metadata feature.

The old exact-label, wildcard-default, and iteration-normalization commits
(`721c73b4d`, `184c05f71`, `be06c00fe`) are absent from this replacement branch's
ancestry. Namespace-forwarding (`fe808f614`) and failed-output (`bc362ddce`) commits
keep their original hashes. The tracked final tree was verified identical before
and after history consolidation.

A pre-existing unfinished rebase was discovered in the original core worktree.
It was left intact. The replacement was committed and pushed from a separate
worktree, `/tmp/aiida-structured-metadata`; no existing remote branch was forced.
The original worktree retains the validated source edits for local review/testing.

### Scope/placement contract

```python
# Ordinary Python aliases outside the graph; no second naming API:
scf = pw_base_graph
bands = pw_base_graph

# Corresponding native bands launch configuration:
task_metadata = {
    'scf': {'pw_calc': {'options': scf_options}},
    'bands': {'pw_calc': {'options': bands_options}},
}
```

Each actual nested graph call contributes one mapping level, keyed by the authored
callee name. Repeating a metadata-bearing alias in one authored scope requires
distinct Python aliases; generated suffixes are not configuration addresses.
Branch/loop/map control frames are transparent, and iterations reuse the same entry.
Runtime resolution follows persisted structural scope lineage, never provenance
label strings. There are no wildcards, normalization, ancestor defaults, or fallback
precedence. An authored `__` alias remains a literal valid name; an unknown flattened
provenance label fails like any other unknown path.

Configuration is checked before execution: missing entries, unknown paths,
non-mapping scopes/leaves, non-string keys, and invalid provided CalcJob metadata
values report actionable paths. Validation and selection preserve leaf contents.
Existing provenance label formatting is unchanged and tested independently.

**Validation:** 1021 core graph tests passed, including 25 focused configuration
regressions. Coverage includes direct CalcJobs, reused nested graphs, three-level
paths, loops, both branch arms, early rejection with unchanged process-node counts,
unchanged metadata contents, and two actual scheduler-state checkpoint round-trips
using real CalcJobs without resubmitting finished children. Ruff and mypy passed.
These are persisted GraphRun-state checks, not a new real-daemon interruption claim.

## Iteration 2: QE authoring and protocol migration

Only QE code changed for this iteration; no additional core feature was needed.
Native composites now call base graphs through authored aliases:

- Bands: `scf`, `bands`.
- Relaxation: `initial` inside `init_relax_guarded`, and `main` inside the loop.
- PDOS: `scf` at the root, `nscf` inside the serial/parallel stage.

Protocol helpers produce exactly the nested contract, with no old flat aliases or
wildcard keys. Empty leaf mappings are retained; only explicit `None` entries are
omitted. Invalid protocol metadata receives a boundary error rather than an internal
merge exception. The prior shared init/main scheduler-options rejection was removed:
a production nested run verifies initial walltime 1111 and main walltime 2222 on
the actual substituted external CalcJobs. SCF/bands and SCF/NSCF retry tests likewise
verify separate settings under reused native base graphs.

See [structured-metadata-migration.md](structured-metadata-migration.md) for exact
per-root mappings and the worker division. Tests compile production graphs and
execute their native base loops; they do not copy production graph bodies.

### Preserved contracts and earlier verified work

- PW/CP `pseudos` is one static DataDict input on core 3; NEB inherits it under
  static `pw`. Core 2.x retains the legacy namespace for released compatibility.
  Pseudos no longer require graph namespace forwarding. Plain mapping launches
  serialize to owned members; mixed concrete legacy/aiida-pseudo node classes in
  one homogeneous container are unsupported.
- **CalcJob monitor specification/runtime/provenance is unchanged.** Existing
  graph monitor forwarding remains. Opt-in structured accuracy-stuck diagnostics,
  not message prefixes, select native monitor recovery.
- Mapped failed-process outputs remain available next to the error arm; native
  recovery consumes stored restart folders and structures with missing-node guards.
- Serial PDOS remains DOS → sweep → PROJWFC; parallel PDOS uses independent calls.
- Cleanup after unexpected failure/cancellation still requires an external owner
  after termination. Successful-path tasks are not automatic termination hooks.
- PH merging remains external provenance-preserving post-processing; it does not
  retroactively replace the graph's stored outputs.

## QE validation

Initial full combined run: **191 passed**, 31 warnings, 376.13 seconds.
Subsequent empty-leaf/boundary corrections passed their focused tests; final full
suite and remote CI results will be recorded below after verification.

Commands from the original core worktree containing the validated source:

```sh
uv run pytest -m presto tests/engine/processes/graphs -n 4 -q
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/graphs -n 4 -q
```

The local core count includes pre-existing untracked graph tests; those files were
not committed. QE external CalcJobs are deterministic local substitutes, not real
QE scientific/scheduler validation. Both repositories' configured pre-commit hooks
passed. Dedicated graph CI installs pytest-xdist and runs all graph tests with four
workers; no graph tests are excluded.

## Remaining work

The result is not a production-compatible replacement release. Remaining QE work
includes complete protocol/output/optional-phase coverage, status 501/710 and other
recovery-policy cases, fixed-scheduler-limit `max_seconds` preparation, prepared-input
inspection, legacy numeric exits, and remaining legacy/core-3 API incompatibilities.
Cleanup and PH merged-result ownership still differ from legacy automatic behavior.
These are not evidence that builders, unrestricted graph-body Python, handler
registries, or a new monitoring engine are missing core prerequisites.

## Remote verification

Iteration 1 was pushed first and triggered core CI. Iteration 2 updates QE's pin to
the already-published exact revision. Current iteration-2 source push, final suite,
and CI observation are pending; old `ci-verification.json` results are historical
until replaced with this iteration's observations.
