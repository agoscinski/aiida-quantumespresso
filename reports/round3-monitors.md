# Round 3 — QE monitors and recovery (native graphs)

> Worker handoff: interim test counts and diagnoses below are superseded by the
> integrated validation in [graph-migration.md](graph-migration.md).

## Scope

Owned and changed:

- `src/aiida_quantumespresso/workflows/graphs/base.py` — optional `monitors`
  forwarding through `pw_base_graph`, new `STOPPED_BY_MONITOR` (150) policy.
- `src/aiida_quantumespresso/tools/monitors/accuracy_stuck.py` — structured
  opt-in returning a machine-readable diagnostic.
- `src/aiida_quantumespresso/calculations/pw.py` — optional `monitor_diagnostics`
  output.
- `tests/graphs/test_base.py` — updated substitute shapes, error-enum and
  subgraph-contract expectations.
- `tests/graphs/test_monitors.py` — **new** monitor/recovery coverage.
- `tests/graphs/test_native_workflows.py` — one-line mechanical fix to the
  `_patch_nested` substitute shape (see §7); owned by another track, flagged here.

Not changed: the legacy `PwBaseWorkChain` handler (message-prefix dispatch
preserved byte-for-byte), `pyproject.toml`, any shared `__init__`, no
commits/pushes.

## Design: diagnostic reason, never message matching

Core already attaches `CalcJobMonitorResult.outputs` to the calculation
(`tasks.py`: `for label, output in monitor_result.outputs.items(): self.out(...)`),
so no core feature was needed. The contract is:

1. `accuracy_stuck.monitor(..., structured=True)` returns a
   `CalcJobMonitorResult` whose outputs hold
   `monitor_diagnostics = Dict({'reason': 'accuracy_stuck', ...})`.
   Default (`structured=False`) keeps the historical plain-string message, so
   the legacy work chain's `ACCURACY_STUCK_MESSAGE_PREFIX` match keeps working.
2. `PwCalculation` declares the optional output `monitor_diagnostics: Dict`.
   A failed run that a structured monitor stopped therefore stores a readable
   `run.monitor_diagnostics` node on the mapped 150 error arm, exactly like the
   already-consumed `run.remote_folder` / `run.output_structure`.
3. `handle_pw_monitor_stuck` claims `PwErrors.STOPPED_BY_MONITOR` **only** when
   `diagnostics['reason'] == 'accuracy_stuck'`. Any other reason, a missing
   diagnostic (legacy string-only stop), or any other error forwards the
   upstream restart decision untouched, so the loop exits and `ensure_pw_recovered`
   fails loudly instead of retrying the wrong recovery.
4. Retry mirrors the legacy `handle_electronic_convergence_stuck` `FULL` restart:
   `CONTROL.restart_mode = 'restart'`, charge-density flags dropped, parent
   rebound to the failed run's `remote_folder`; `relax`/`vc-relax` additionally
   reduce `IONS.upscale` by the legacy `delta_factor_upscale` (0.3, floor 1).
   `upscale <= 1` claims no restart (legacy reports unrecoverable failure; the
   graph has no fail-with-outputs contract, so it stays a loud
   `ensure_pw_recovered` failure rather than an invented outcome).

## Forwarding

- `pw_calc = task(PwCalculation, inputs={'pseudos': orm.DataDict,
  'monitors': orm.DataDict}, errors=PwErrors)` — container-forwarded dynamic
  namespaces, same pattern as the existing `pseudos` closing.
- `pw_base_graph(..., monitors: orm.DataDict | None = None, ...)` threads the
  mapping into both placements (initial and loop). `None`/absent leaves the
  `CalcJob` `monitors` namespace unwired (verified: unmonitored run succeeds in
  one attempt with no `monitors__*` incoming links on the child).

## Considered-finished codes 501/710: kept as loud failures

`ERROR_IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF` (501) and
`WARNING_ELECTRONIC_CONVERGENCE_NOT_REACHED` (710) finish *with* an exit code
while attaching outputs. A graph cannot finish-with-outputs-and-status, so
porting "considered finished" semantics would require inventing a new failure
contract. Per the audit steer (document instead of invent), `PwErrors` still
excludes 501/710 and the loop fails loudly via `ensure_*`. No silent success,
no fabricated nodes.

## Tests (`tests/graphs/test_monitors.py`, 14 tests, all passing)

- **Monitor result contract** (real `monitor()` with fake node/transport):
  legacy default returns the prefixed string; `structured=True` returns
  `CalcJobMonitorResult` with `monitor_diagnostics['reason'] == 'accuracy_stuck'`;
  improving accuracy returns `None` in both modes; `CalcJobMonitor(entry_point=
  'quantumespresso.accuracy_stuck', kwargs={'structured': True, ...})` validates,
  proving the opt-in passes core namespace validation.
- **Handler units**: relax upscale 100→30 with full-restart flags; `upscale <= 1`
  gives up; scf restarts from the parent with `upscale` untouched; other
  reason/missing diagnostic/wrong error never claim.
- **Production wiring** (real `pw_base_graph`, stubs subclassing `PwCalculation`
  that store a real `monitor_diagnostics` output and read `self.inputs.monitors`
  — no monkeypatched handler):
  - relax recovery in 2 attempts; `monitors__stuck` incoming links on both
    children prove forwarding reached the failed CalcJob; stored diagnostic
    reason asserted; retry shows reduced upscale and parent rebinding.
  - scf full restart with parameters otherwise unchanged.
  - absence: unmonitored run succeeds first try with no monitor links.
  - other-reason and bare-stop stubs fail persistently: budget exhausted with
    parameters untouched (no accidental upscale retry) and loud
    `ensure_pw_recovered` failure.

Also updated: `test_base.py` substitute shapes
(`{'pseudos', 'monitors'}`), `150 ∈ PwErrors` with 501/710 still excluded, and
the subgraph-contract signature (`monitors` after `parent_folder`).

Results: `test_monitors.py` 14 passed; `test_base.py` 29 passed;
`test_native_workflows.py` + `test_pw.py` + `test_protocols.py` + `test_pdos.py`
71 passed; `tests/tools/test_monitors.py` + `tests/calculations/test_pw.py`
51 passed. Pinned `ruff==0.13.2` check and format clean on all touched files.

## Handoff — what the coordinator needs for protocols/composites propagation

`pw_base_graph` already accepts `monitors`; composites (`graphs/pw.py`,
`graphs/pdos.py`) just need to thread it through. Precise args per placement:

```python
monitors = orm.DataDict({
    'stuck': orm.Dict({
        'entry_point': 'quantumespresso.accuracy_stuck',
        'kwargs': {'structured': True, 'min_repeats': 5},
    }),
})
initial = pw_base_graph(
    ...,  # existing args unchanged
    monitors=monitors,  # omit or pass None/empty DataDict to run unmonitored
)
```

Notes for the coordinator:

- `structured: True` is mandatory for recovery: without it the stop carries no
  `monitor_diagnostics` and the native handler deliberately ignores it.
- `min_repeats` (or any monitor kwarg) travels inside the per-monitor `Dict`;
  core validates kwargs against the monitor signature at launch.
- Known core limitation (other worker's track): every `pw_base_graph` scope
  reuses the inner placement name `pw_calc`, so one `task_metadata['pw_calc']`
  mapping applies in all SCF/NSCF scopes. Distinct inner placement names avoid
  it; scope-aware addressing, if pursued, belongs to that track, not here.
- `tests/graphs/test_native_workflows.py::_patch_nested` was updated to
  `inputs={'pseudos': _DataDict, 'monitors': _DataDict}` (one line). Any future
  placement-shape change in `base.py` needs the same mirror there; consider
  moving the substitute-shape construction next to the placement it mirrors.
