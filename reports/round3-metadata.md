# Round 3: scope-aware `task_metadata` — core contract for QE

> Worker handoff: interim test counts and diagnoses below are superseded by the
> integrated validation in [graph-migration.md](graph-migration.md).

## Chosen contract (flat keys, smallest)

`task_metadata` stays `Mapping[str, Mapping]`. Values are always the `CalcJob`
`metadata` mapping (e.g. `{'options': {...}}`). Keys are one of:

- `pw_calc` — leaf placement name (existing; unchanged, applies everywhere uncovered).
- `scf__pw_base__pw_calc` — exact full hierarchical instance label (new; wins).
- `scf__*` — explicit scope rule (new; trailing `__*` only).

Precedence: **exact > longest matching scope > leaf**. Missing entries fail with
`GraphValidationError` naming placement + instance. Only trailing `__*` is a
scope rule; any other `*` (`scf*`, `scf_*`, `*`) fails loudly — no glob, no
ambiguous prefix fallback. No nested-dict form (`{'scf': {'pw_calc': ...}}`)
is accepted; the flat form keeps values uniformly metadata-shaped.

## Scope semantics QE needs

- Scope path is the `__`-joined ancestor placement path, e.g. `scf`, or
  `outer__scf` for deeper nesting. `scf__*` matches `scf__pw_base__pw_calc`
  and any deeper `scf__...` instance.
- Loops: matching strips `_iteration_N` segments, so `scf__*` covers
  `scf__loop_1_iteration_3__pw_calc`, and `myloop__*` covers
  `myloop_iteration_2__pw_calc`. No per-iteration keys exist by design.
- Retries (`_2` dedup from `__` in placement names) and map items are covered
  by their scope rule; exact retry labels (`scf__pw_calc_2`) also work.
- Longest scope wins: `outer__scf__*` beats `outer__*`.

## QE usage

```python
# Per-placement differentiation inside reused subgraphs:
task_metadata={
    'scf__pw_base__pw_calc': {'options': {'max_wallclock_seconds': 3600}},
    'nscf__pw_base__pw_calc': {'options': {'max_wallclock_seconds': 7200}},
}
# ... or one reusable rule per scope (loops/retries included):
task_metadata={
    'scf__*': {'options': {'max_wallclock_seconds': 3600}},
    'nscf__*': {'options': {'max_wallclock_seconds': 7200}},
}
```

Protocol workers should emit one `<scope>__*` rule per reusable
subgraph/scope plus exact overrides where one placement inside a scope differs.
Dedup scopes from calling one nested graph twice read `pw_base__*` vs
`pw_base_2__*`; prefer distinct wrapper graph names (`scf`/`nscf`) for readable
scope keys.

## Core verification (this round)

New `tests/engine/processes/graphs/test_scoped_task_metadata.py` (9 tests):
leaf-pair regression (`add_twice` 100s vs 200s), exact-label collision fix on a
real reused-nested graph (`scf`/`nscf` sharing `pw_base__pw_calc`, 111s vs 222s
observed on children via `get_option`), scope-rule equivalents, exact > scope >
leaf precedence, leaf fallback, dedup scopes, glob rejection, loop-iteration
matching (unit-level, no daemon), missing-entry instance naming. All pass;
existing `test_graphs.py` CalcJob metadata tests unaffected. Full instance
labels dispatch as `CALL_CALC` link labels (e.g. `scf__pw_base__pw_calc`).
