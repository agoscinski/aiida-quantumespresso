############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
############################################################################
# Round 3 — independent integration review (nested QE)

> Worker handoff: interim test counts and diagnoses below are superseded by the
> integrated validation in [graph-migration.md](graph-migration.md).

Owned files: `src/aiida_quantumespresso/workflows/graphs/pw.py`,
`src/aiida_quantumespresso/workflows/graphs/pdos.py`,
`src/aiida_quantumespresso/workflows/graphs/protocols.py` (targeted
monitor/metadata fixes only), `tests/graphs/test_pw.py` + `tests/graphs/test_pdos.py`
stub-shape compatibility, `tests/graphs/test_round3_integration.py` (new), this
report. No `WorkChain` placed or imported. No core edits, no commits/pushes.

This review does not accept the initial workers' "green" at face value: it
re-runs the production nested bases with a fail-once-then-succeed external
`CalcJob` and asserts independent scheduler options plus monitor threading
through retries.

## 1. Composites thread optional monitors

Shared `monitors: DataDict | None = None` added to every composite root, plus
per-stage overrides with the same kwargs shape that fall back to the shared
mapping:

- `init_relax_guarded(monitors)`, `pw_relax_graph(monitors, init_monitors)`,
  `pw_bands_graph(monitors, scf_monitors, bands_monitors)`.
- `pdos_stage_parallel/serial(monitors)`,
  `pdos_from_scratch_parallel/serial(monitors, scf_monitors, nscf_monitors)`,
  `pdos_from_parent_parallel/serial(monitors)`.
- Every `pw_base_graph` call forwards the resolved mapping; `None`/absent
  leaves the `CalcJob` `monitors` namespace unwired (same convention as the
  native base). No new `WorkChain`, no exit-code changes.

Module docstrings updated: the "no monitor wiring" honest-limit is replaced by
the forwarding contract. `build()` validates for all touched roots.

## 2. Protocol helper: explicit boundary + YAML `pw.monitors`, strict validation

Previously `monitors` in overrides raised "no monitor plumbing yet". Now:

- `_BASE_PW_KEYS` includes `monitors`; `_REJECTED_OVERRIDES` no longer lists it.
- YAML shape: `pw.monitors: {name: {entry_point, kwargs}}`. Only
  `quantumespresso.accuracy_stuck` is accepted; allowed kwargs are
  `min_repeats: int` and `structured: bool`. `structured` defaults to `True`
  and `False` is rejected (unstructured stops carry no diagnostics, so the
  native handler would ignore them). Unknown entry points, unknown
  `kwargs` keys, wrong types, and non-mapping entries all raise `ValueError`
  naming the path — never silently ignored.
- Conversion chain: YAML plain dict → per-monitor `orm.Dict` →
  `orm.DataDict`, via `_monitors_from_yaml` / `_coerce_monitor_entry`.
- Explicit boundary inputs: `monitors`, `scf_monitors`, `bands_monitors`,
  `nscf_monitors`, `init_monitors` (`DataDict | dict | None`, coerced by
  `_coerce_monitors_input`). Precedence per stage is explicit-per-stage >
  explicit-shared > YAML-per-section. From-parent PDOS rejects `scf_monitors`
  (no SCF consumer) and maps the NSCF choice onto its single `monitors` input.
- No huge duplicate code: one pair of coercion helpers serves all three
  families; `_resolve_base_section` returns `monitors` alongside
  `metadata_options`.

## 3. Scheduler metadata: `<scope>__*` rules covering first and retry runs

Problem: the helper emitted exact labels only
(`pw_base_graph__pw_calc`, `pw_base_graph_2__pw_calc`). Retries dispatch as
`pw_base_graph__loop_22_iteration_0__pw_calc` (observed in run logs), so an
exact-only entry misses retries and independent per-stage values fall back to
the shared leaf.

Fix: every family now emits both the exact aliases (kept for backward
compatibility — existing `test_protocols` assertions on
`pw_base_graph__pw_calc == pw_calc` still hold) and `<scope>__*` rules:

- bands: `pw_base_graph__*` (SCF) + `pw_base_graph_2__*` (bands), plus the two
  exact aliases and the leaf.
- PDOS from-scratch: `pw_base_graph__*` (SCF) +
  `{stage}__pw_base_graph__*` (NSCF, `stage = pdos_stage_serial|parallel`) plus
  exact aliases and leaves (`pw_calc`, `dos_calc`, `projwfc_calc`).
  From-parent emits only the stage rules.
- relax: `pw_base_graph__*`, `init_relax_guarded__*`, and
  `init_relax_guarded__pw_base_graph__*` plus the leaf (relax shares one mesh,
  settings, and scheduler options between init and main; differing
  `metadata_options` still raise).

Scope matching is iteration-insensitive (`_iteration_N` stripped), so one rule
covers the first attempt (`pw_base_graph__pw_calc`) and every retry
(`pw_base_graph__loop_22_iteration_0__pw_calc`). Longest-scope wins; leaf
remains the fallback.

## 4. Tests: `tests/graphs/test_round3_integration.py` (10 tests)

Substitution reuses `_patch_nested` from `test_native_workflows` (which already
carries the `{pseudos, monitors}` placement shape). The fail-once stub
`RetryWalltimeBandsPwCalc` exits 400 unless `CONTROL.restart_mode ==
'restart'`, storing the interrupted structure on failure and full SCF/bands
outputs (bands, electrons, Fermi energy) on recovery — so each of the SCF and
bands/NSCF stages fails once then recovers with no global state. Role
attribution uses `CONTROL.calculation` (`scf` vs `bands`/`nscf`), not
`parent_folder` presence (the SCF retry rebinds a parent and would otherwise
misclassify).

- `test_bands_retry_keeps_independent_scope_options`: 4 pw calcs (SCF fail +
  recover, bands fail + recover); SCF calcs all read 1111, bands calcs all
  read 2222 via `pw_base_graph__*` / `pw_base_graph_2__*`.
- `test_bands_exact_aliases_still_cover_first_attempt`: exact + scope keys
  together still finish with 4 calcs.
- `test_bands_monitors_reach_nested_calcs`: shared monitors reach all 4 nested
  calcs (`monitors__stuck` incoming link on each).
- `test_bands_per_stage_monitors`: shared + `bands_monitors` override, all 4
  calcs wired.
- `test_relax_monitors_thread_through`: relax forwards monitors into nested
  calcs.
- `test_pdos_retry_keeps_nscf_scope_options`: from-scratch serial, 4 pw calcs;
  SCF 1111 via `pw_base_graph__*`, NSCF 2222 via
  `pdos_stage_serial__pw_base_graph__*`; all 4 carry `monitors__stuck`.
- `test_protocol_bands_scope_rules_and_monitors`: resolver emits both scope
  rules and exact aliases; YAML `scf.pw.monitors` becomes a `DataDict` with
  the known entry point; explicit `monitors` boundary accepted; unknown
  entry point and unknown kwarg each raise.
- `test_protocol_rejects_unsupported_monitors_shape`: non-mapping entry and
  `structured: False` each raise.
- `test_protocol_pdos_scope_rules`: PDOS serial emits `pw_base_graph__*` and
  `pdos_stage_serial__pw_base_graph__*` plus exact alias equality.
- `test_graph_root_nodes_are_native`: root creates nothing, no `WorkChain`
  descendants.

Stub-shape compatibility: `tests/graphs/test_pw.py` and
`tests/graphs/test_pdos.py` `stub_pw_base` now declare
`monitors: DataDict | None = None` (ignored), mirroring the monitor worker's
`test_native_workflows` patch-shape fix. No semantic change to those suites.

## 5. Verification (exact evidence)

Pinned env: `PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools
--with aiida-pseudo pytest … -q`, presto profile.

- `test_round3_integration.py`: first full run 9 passed / 1 failed in 622 s.
  The failure was `test_bands_retry_keeps_independent_scope_options` asserting
  `{'bands': {1111, 2222}}`: root cause was test role attribution by
  `parent_folder` presence — the SCF retry rebinds a parent, so it
  misclassified as bands. Reproduced from the captured log
  (`pw_base_graph__loop_22_iteration_0__pw_calc` retry with 1111). Fix: classify
  by `CONTROL.calculation`. Isolated re-run of the fixed test: 1 passed in
  ~82 s. Composition with the 9 unaffected passes gives 10/10; no
  `xfail`/`skip` markers added (only the file-level released-core guard
  shared by all graph tests).
- `test_protocols.py`: 25 passed in ~132 s (includes the pre-existing exact-
  alias assertions, which still hold alongside the new scope rules).
- `test_pw.py + test_pdos.py`: 56 passed in ~46 s after the stub-shape fix
  (before: 14 failed with `'monitors' is not an input of 'pw_base_graph'`).
- `test_base.py + test_monitors.py`: 43 passed in ~182 s.
- `test_native_workflows.py`: 6 passed in ~119 s.
- `ruff check`: clean on all touched files; `ruff format` applied to
  `protocols.py` + `test_round3_integration.py`.

No combined-failure dismissal: the single round-3 failure was reproduced,
explained, and fixed as above; no other suite failed.

## 6. Known limits / semantic oversights checked

- Retries inside `pw_base_graph` loop iterations are covered by scope rules;
  per-iteration keys are intentionally absent (core normalizes `_iteration_N`).
- Relax init vs main share scheduler options by design (differing
  `metadata_options` raise); monitors may differ via `init_monitors`.
- Only the accuracy-stuck monitor is structurally wired; any other
  `entry_point` is rejected rather than passed through silently (core would
  accept it, but recovery semantics would be unclaimed).
- `structured: False` rejected: the native handler keys on
  `monitor_diagnostics['reason']`, which only structured mode stores.
- DOS/PROJWFC placements have no retry loops: exact keys suffice; scope rules
  emitted for symmetry.
- `DataDict` monitors take owned copies; `min_repeats` validated as `int`.
