# Base graphs production-correctness review

- **Track:** native `@graph` alternatives to the six `*BaseWorkChain` restart loops
  (`aiida-quantumespresso`).
- **Owned files (this review only):** `src/aiida_quantumespresso/workflows/graphs/base.py`,
  `tests/graphs/test_base.py`, `reports/base-review.md`. No other QE file, no core file,
  no shared `pyproject`/CI was modified; nothing was committed or pushed.
- **Legacy sources preserved untouched:** `workflows/pw/base.py`, `workflows/ph/base.py`,
  `workflows/neb/base.py` (and `q2r`/`matdyn`/`bands` bases, which define no handlers).
- **Core under test:** parent checkout `graph-benchmark` worktree
  (`src/aiida/engine/processes/graphs/run.py`, `lowering.py`); the contract is the
  `errors=` mapped-failure semantics there.
- **Commands (parent env, never bare `python`/`pip`):**
  `PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo
  pytest aiida-quantumespresso/tests/graphs/test_base.py -q`.
- **Result:** 29 passed; `ruff check` and `ruff format --check` clean on both owned files.

## 1. Finding: mapped failures expose no success outputs downstream

A placement that finishes with a mapped error exposes only its `error` arm. In
`src/aiida/engine/processes/graphs/run.py`, `_is_mapped_failure` ("The success arms
of such a failure are absent by design, like the missing arm of a `Result` task, so
they read as `None` downstream") and `_resolve_placement_outputs` (a mapped failure
"synthesizes its member from the child exit status; absent success arms read as `None`
downstream like a `Result` arm") return only `{'error': ...}` for a mapped placement,
dropping every other output port — even outputs the failed child actually stored.

Proven by throwaway probe (deleted after use): a stub `WorkChain` sharing `PwErrors`
attached `output_parameters` + `output_structure` and then exited with a mapped ionic
status. Downstream the graph saw `error` present and `output_parameters`,
`output_structure`, `remote_folder` all `None`:

```text
ERROR_IS_NONE: False
PARAMS_IS_NONE: True
STRUCTURE_IS_NONE: True
REMOTE_IS_NONE: True
```

Legacy handlers read the failed calculation directly
(`calculation.outputs.remote_folder`, `calculation.outputs.output_structure`,
`calculation.outputs.output_parameters` — e.g. `workflows/pw/base.py`
`handle_out_of_walltime`, `handle_relax_recoverable_ionic_convergence_error`,
`handle_electronic_convergence_not_reached`; `workflows/neb/base.py` `set_restart_type`
callers; `workflows/ph/base.py` `prepare_process` via `restart_calc`). The graph can
never do this through an `errors=` placement. Consequence: **every** restart needing
the failed run's outputs retries degraded — the `is not None` rebinding guards silently
keep the old structure and drop the parent while the carried parameters still claim a
restart (walltime/ionic/electronic `pw`, all `neb` remote restarts, every `ph` recover
restart, including the recover latch retrying against a stale parent).

## 2. Finding: the handler chain dropped the restart decision

Independently of the masking, each handler built its plan from `_pass_*_plan(...)` with
fresh `restart=False`, `parent_mode='keep'`, `structure_from_output=False`,
`rattle_now=False` (and `recover=False` for `ph`). The final plan therefore carried only
the *last* handler's decision (`check_pw_sufficient_bands` / `handle_ph_convergence` /
`handle_neb_electronic_convergence`), which is a pass-through on every failure path.
`restart`, `parent_mode`, `structure_from_output`, `rattle_now`, and `recover` computed
by the matching handler never reached `finalize_*_carry` or the `is not None` rebinding
— the entire node-rebinding machinery (parent folder, output structure, rattle) was
dead code in production, and the loop retried on `run.error` alone with degraded inputs.

Exact evidence: the pre-fix `test_restartable_pw_rattle_branch_recovers` passed with
`{'attempts': 2, 'status': 'finished'}` while provenance showed `rattle_positions`
never ran (`CALLED_LABELS` contained no `rattle_positions`): with
`output_parameters` masked to `None`, the handler took the low-information `damp`
fallback, the restart flags were dropped downstream, and the stub succeeded on attempt
1 for reasons unrelated to rattling. The test name claimed rattle recovery; the run
proved damp-with-degraded-inputs recovery.

## 3. Fixes

- **Decision forwarding.** `_pass_pw_plan`, `_pass_ph_plan`, `_pass_neb_plan` now take
  the upstream decision (`restart`, `parent_mode`, plus `structure_from_output` /
  `rattle_now` / `recover` as applicable) and each non-leading handler forwards it,
  mutating only on its claimed error. Only one handler claims per failure (disjoint
  error codes; the bands check only claims on success), so the final plan carries the
  matching handler's decision. All pre-existing handler unit tests pass unchanged
  (forwarding parameters default to the old fresh-plan values).
- **Loud-failure guards.** New `require_pw_restart_nodes`, `require_ph_restart_nodes`,
  `require_neb_restart_nodes` tasks run before every `finalize_*_carry` (pre-loop and
  per-iteration, ordered via `after=` so a claimed restart cannot slip into the next
  placement) and raise `ValueError` naming the missing node when a claimed restart
  needs `remote_folder`/`output_structure` that the mapped arm hid. The graph then
  exits non-zero (observed: exit status 400,
  `The task 'require_pw_restart_nodes' did not finish successfully.`) with no outputs
  instead of retrying parent-less. Parameter-only restarts (diagonalization cycling)
  and success-path restarts (whose outputs are visible) pass through untouched.
- **Graph docstrings and the module docstring** now state the masking limitation, the
  chain-forwarding invariant, and which paths genuinely recover.
- **Tests.**
  - `restartable_pw` (the counter-driven mirror) threads decisions and guards like the
    production body.
  - `recover` mode now fails twice on diagonalization only (the one stub path that can
    genuinely recover) and asserts all guards passed; `rattle` and `always-fail` modes
    now assert loud failure (single placement, no `rattle_positions`, graph exit names
    the guard); `diag-fail` mode preserves budget-exhaustion coverage via `ensure_*`.
  - New bare-task unit tests for all three guards (idle/kept/honored pass; hidden
    nodes raise naming the node).
  - New **production-graph runtime tests** substitute a shape-compatible stub for the
    real `pw_calc` placement via `monkeypatch` (plus a `_template` reset, since
    `compile_graph` caches the template on the handle) and run the real
    `pw_base_graph` itself — no body copy: diagonalization recovery finishes in 3 real
    placements with `finished`; a first-try success returns outputs with 1 attempt; a
    walltime failure aborts at the guard with 1 placement. The stub failure is driven
    by the carried `parameters` (not a counter), and `InstalledCode`/`UpfData` are
    fabricated locally (no codes, clusters, or fixture files).

## 4. What genuinely recovers, per legacy comparison

- `pw` diagonalization cycling: parameters-only, no hidden nodes needed — recovers
  (proven end to end through the production graph).
- `pw` insufficient-bands growth: success-path only, outputs visible — recovers.
- `pw` walltime / electronic / ionic (incl. BFGS/rattle/cell-factor/interrupted)
  restarts: need `remote_folder` and/or `output_structure` — **abort loudly** at the
  guard. Note the BFGS rattle branch is additionally unreachable in production: the
  symmetry count comes from the masked `output_parameters`, so the handler always
  takes the damp fallback; the guard then aborts before any degraded retry.
- `neb` diagonalization cycling: recovers. Interrupted/`nstep`/electronic restarts:
  abort loudly (need `remote_folder`).
- `ph` diagonalization cycling (`cg` switch): parameter-only, recovers. Scheduler
  walltime / out-of-walltime / convergence recover restarts: abort loudly (need
  `remote_folder`; the recover latch can no longer bind a stale parent).
- `q2r`/`matdyn`/`bands`: single placement, unchanged; failures abort as before.
- Deliberately unported legacy behaviors are unchanged: message-gated
  `STOPPED_BY_MONITOR`, considered-finished `501`/`710` exits, scheduler
  `max_seconds` capping via `task_metadata`, and merged multi-recover `ph` outputs.

## 5. Residual gaps (not fixed here)

- Runtime substitution covers `pw_base_graph` only; `ph`/`neb` guards are covered by
  bare-task unit tests plus `build()`, not by substituted end-to-end runs.
- Diagonalization exhaustion still retries the same failing setup until
  `max_iterations` before `ensure_*` fails it; pre-existing behavior, left as is.
- The fixed `{'Si': UpfData}` pseudos shape still needs a graph copy per kind set;
  handler tasks themselves are kind-agnostic.
- Core framework note (no core change made): mapped `errors=` placements intentionally
  hide the failed child's success outputs, so any restart mathematics needing them
  cannot be expressed faithfully in-graph; loud failure is the honest behavior until
  and unless the framework offers failed-run output access.
