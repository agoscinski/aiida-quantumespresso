# Native QE graphs — third refinement and validation

## Verdict

**Native, without WorkChains, but not feature-complete or legacy-API compatible.**
The updated `graph-layer-audit-response.md` is the checklist for this round.
Missing QE implementations are not presented as missing core authoring features.
Legacy workflow entry points remain available; native graphs never call them.

Required core: **`3e5947cc73ea3f6bb442b04b8d9dbbf3066ab9d2`**, pushed to
`agoscinski/aiida-core`, branch `refactor/graph-qe-round3`. This includes the
previous failed-output/namespace fixes and the narrowly scoped metadata improvement
below. The earlier core branch was preserved after a non-fast-forward push rejection;
no remote history was overwritten. Dedicated QE graph CI pins this exact revision.
Released core 2.8 does not provide these graph APIs; the stable dependency remains
unchanged for legacy entry points.

## Changes following the updated response

### 1. Real QE protocol-to-launch arguments

`graphs.protocols.get_inputs_from_protocol` now returns arguments usable with
`run_get_node`/`submit` for relax, bands, and matching serial/parallel/from-parent
PDOS roots. It does not construct builders or run legacy WorkChains.

Implemented and tested: raw-YAML defaults/overrides, explicit pseudos or installed
pseudo-family selection, recommended cutoffs, per-atom convergence thresholds,
electronic/spin/relax settings, k-point distances/settings, codes, optional monitor
inputs, and scheduler metadata. Tests use explicit pseudos or offline family
fixtures; no pseudo downloads are required. Unsupported options fail early.

This is resolution for the **supported subset**, not complete legacy protocol
parity: inner base budgets other than five, parallelization, distinct init/main
mesh/settings, automatic electronic type, and some optional phases are still
unsupported. Root `clean_workdir=True` for relax/bands and unrouted composite mesh
parity are rejected rather than silently discarded. PDOS success-path cleanup is
supported. Protocol YAML monitor conversion currently supports the accuracy-stuck
monitor; direct graph `DataDict` inputs are the lower-level monitor interface.

### 2. Monitor inputs and machine-readable recovery

PW base declares both `pseudos` and `monitors` as forwarded `DataDict` namespaces.
Relax/init/bands/PDOS pass optional shared or per-stage monitors into native bases.

The existing accuracy-stuck monitor supports **opt-in `structured=True`**, returning
`CalcJobMonitorResult` with an optional declared `monitor_diagnostics: Dict` output.
The diagnostic contains `reason='accuracy_stuck'`; human-readable text is not a
recovery dispatch key. Default monitor behavior remains a string for legacy callers.

Native recovery maps status 150, claims only the structured accuracy-stuck reason,
reduces `IONS.upscale` for the relevant relaxation case, and carries a full restart
folder. Other reasons, absent diagnostics, exhausted upscale, or missing required
restart nodes fail rather than being misclassified as accuracy-stuck recovery.
No arbitrary child-exit-message API or new monitoring engine was added.

### 3. Independent scheduler settings, including retries

Distinct leaf placement names already supported distinct metadata. The actual
problem was reused nested graphs sharing the inner name `pw_calc`.

Core now resolves `task_metadata` by:

1. Exact hierarchical execution label.
2. Longest matching explicit `<scope>__*` prefix (loop iteration indices normalized).
3. The exact leaf placement name, preserving existing callers.

No arbitrary glob syntax or dynamic per-attempt resource adaptation was added.
The protocol helper emits scoped defaults for SCF/bands and SCF/NSCF, so settings
remain distinct when either native base retries. Independent reviewers caught the
initial exact-label-only aliases losing stage settings on retries; this is fixed
and exercised through production nested graphs. The coordinator also removed an
unsafe numeric-suffix fallback that could silently reuse another placement's options.

### 4. Cleanup ownership: an explicit policy, not an automatic guarantee

`graphs.cleanup.cleanup_graph_workdirs(node)` is an ordinary external-owner helper
for a **terminated** graph. It handles stored descendants after success, mapped or
unmapped failure, and cancellation; it refuses a running root or active descendants.
It is idempotent and records expected filesystem/configuration failures per CalcJob
without changing process state or exit status. Unexpected programming errors can
still propagate. Cleanup is attempted once per folder; transport operations retain
their own timeout behavior.

Defensive guards refuse relative paths, filesystem/shared top-level roots, shared
temporary roots, and the computer's literal configured work folder. Real calculation
subdirectories below `/tmp` remain valid. These checks are **not a security sandbox**:
stored paths and computer configuration must still be trusted. Dry-run uses the
same refusal checks. Tests clean real temporary directories, not arbitrary fixture
paths, and check that unrelated directories remain intact.

**Policy decision:** PDOS sweeps cover the successful graph paths where they are
wired. A reusable sweep task is available for explicit handled-failure branches,
but such a task is not automatically inserted everywhere. Unexpected failure and
cancellation require an external owner to invoke the terminal helper after all
children terminate. We deliberately do **not** claim legacy automatic
`on_terminated` equivalence or cleanup after a crashed worker without owner action.
Relax/bands protocol resolution rejects an unsupported in-graph cleanup promise.

### 5. PH merging: implemented external post-processing, not a root-output rewrite

`graphs.ph.merge_ph_graph_outputs(node, last_calc=...)` walks a completed recovery
chain and calls the existing merge calcfunction with each original output Dict as
an explicit calculation input. Its tests verify direct provenance links, merge
math, q-point range checks, initialization/single-member cases, and missing outputs.
Disconnected PH branches require an explicit branch-end selector instead of a
silent latest-time choice.

The helper returns a post-processing Dict. The graph itself still publishes its
last-iteration output: no stored graph outputs are retroactively changed. This is
not full legacy result-contract parity. Owned node containers exist; their
copy/detachment semantics are not evidence that an in-graph solution is impossible.
No general collector API is requested on that basis.

## Worker division and independent verification

Four workers owned protocols, monitor/recovery, scoped metadata, and cleanup/PH.
Two further workers reviewed integration and cleanup/PH semantics. The coordinating
agent inspected and tested the combined tree, fixed integration issues, and reran
complete suites rather than accepting isolated runs or unexplained 'flaky' failures.

Corrections during review included retry metadata coverage, monitor plumbing through
all composites, rejection of silently ignored launch options, cleanup authentication
errors and dangerous roots, and ambiguous PH chain selection. Earlier worker handoff
reports retain their intermediate results; this report is authoritative.

## Validation

| Check | Observed result |
| --- | --- |
| Complete QE graph suite, `-n 4` | **181 passed**, 33 warnings, 348.23 seconds |
| Core graph suite, `-m presto -n 4` | **1006 passed**, 4 warnings, 110.84 seconds |
| Legacy monitor + PW calculation tests | **51 passed**, 1 warning, 4.40 seconds |
| Cleanup + PH helper tests | **37 passed**, 1 warning, 8.59 seconds |
| Legacy workflow suite on this core | **89 failed, 106 passed, 1 skipped**, 30.48 seconds |
| Repository pre-commit hooks | Core Ruff/mypy and QE Ruff lint/format passed on changed files |

```sh
uv run pytest -m presto tests/engine/processes/graphs -n 4 -q
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/graphs -n 4 -q
```

The local core suite includes existing untracked graph tests; they were not added
to the commits. The initial serial QE run exceeded its 480-second command budget
near completion; it is not counted as a pass. The combined parallel suite passed
before and after final boundary/safety corrections (169, then 181 tests).

### What these tests establish—and what they do not

Production graphs run through native PW bases with external CalcJobs substituted:
protocol-generated inputs, monitor forwarding/structured diagnostics, first attempts
and retries with independent scheduler options, relaxation/bands/serial PDOS,
remote-folder rebinding, general pseudo mappings, provenance, and no WorkChain
descendants. Core metadata tests execute actual local arithmetic CalcJobs.

These are **not scientific runs of QE executables**. Monitor tests exercise the
structured monitor result and production recovery wiring, not a complete real QE
scheduler-poll/kill cycle. Cleanup/PH tests include constructed terminated process
provenance; they are not daemon-crash tests. There are still deprecation/dependency
warnings, notably the legacy monitor's `CalcJobNode.attributes` access.

## Remaining work, accurately classified

- **QE implementation/compatibility:** complete protocol option coverage, optional
  bands relaxation and all outputs, status 501/710/legacy numeric exits, remaining
  handler cases, fixed-scheduler-limit `max_seconds` preparation, prepared-input
  inspection, and the legacy `orm.AbstractCode` protocol/core-3 failures.
- **Chosen ownership contract:** terminal cleanup and PH merged output currently
  require explicit external calls. Automatic legacy lifecycle/result behavior is
  not claimed. An automatic cancellation guarantee would need a specifically
  designed owner/termination mechanism, not merely another successful-path task.
- **Core composability:** shared nested metadata addressing is now implemented and
  tested. Runtime-adjustable scheduler settings should only be requested for a
  concrete policy that changes scheduler resources—not fixed-limit preparation.
- **Unverified framework path:** namespace forwarding in directly task-mapped
  `par_map` remains outside this work; QE's fixed DOS/PROJWFC fan-out does not need it.

Builders, protocol APIs, unrestricted graph-body Python, handler registries,
monitoring implementations, and decorated-calcfunction placement are not missing
prerequisites. No new broad core authoring wishlist follows from remaining QE work.

## Remote verification

The core source is pushed and QE CI is updated to the exact new pin. New QE source
push and CI observation are pending; previous `ci-verification.json` results belong
to the earlier revision until this section and that evidence file are updated.
