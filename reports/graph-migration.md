# Native Quantum ESPRESSO graphs — validated refinement

## Verdict

**The experimental graph implementation now uses zero WorkChains. It is still
not feature-complete.** Legacy workflow modules and entry points are retained for
compatibility; they are not called by the native graph implementation.

This refinement supersedes the earlier audit's blanket blocker statements.
Protocols, ordered handlers, scientific Python, fixed heterogeneous fan-out, and
serial PDOS do not require new graph language constructs. The audit response and
essential-features analysis informed the changes below.

Required core revision: **`d2045e4ff2932725796e04801df4727ba72d5db3`**, pushed to
`agoscinski/aiida-core`, branch `refactor/graph-qe-native`. Released core 2.8 and the
previous pinned core do not implement these refinements. QE's stable dependency
constraint and legacy entry points remain unchanged; dedicated graph CI explicitly
installs and verifies the new core revision.

## Source changes

- `graphs/base.py`: direct PW/PH/NEB/Q2R/Matdyn/Bands CalcJob placements. Removed
  `NebStaticWrapper`. General `pseudos: DataDict` replaces the Si-only contract;
  native PW base takes an initial `parent_folder` and returns existing calculation
  nodes, including the final remote folder, plus an explicit summary task output.
- `graphs/pw.py`: removed `PwBaseStep` and legacy `PwBaseWorkChain` submission.
  Relax, init-relax, and bands now orchestrate the native PW base subgraph.
- `graphs/pdos.py`: native PW base for SCF/NSCF; direct DOS/PROJWFC placements.
  Separate serial/parallel stage and root graphs. Serial execution orders
  **DOS → DOS-folder sweep → PROJWFC**; independent parallel placements need no
  `par_map`. Unqualified root names alias parallel roots.
- `graphs/protocols.py`: ordinary YAML protocol loading/merging/overrides and
  launch `task_metadata` helpers, without legacy class imports. These are **not
  yet fully resolved protocol-to-submission helpers**: pseudo-family selection,
  thresholds, all meta-parameters, and graph-input translation remain QE work.
- Scheduler options now use launch `task_metadata`, not old wrapper input ports.
  Experimental API changes are recorded in QE's `CHANGELOG.md`.

## Necessary core changes: implemented, not merely proposed

1. **Mapped failed-process outputs are retained.** The runtime synthesizes `error`
   alongside outputs actually stored by the child. Only genuinely missing outputs
   remain absent. Existing wires, presence checks, decision tasks, and loops now
   recover failed-run nodes; no new failure-record API was needed. Function tasks
   returning `Err` retain their existing absent-success-arm behavior.
2. **Typed container forwarding into dynamic input namespaces.** For example:

   ```python
   pw_calc = task(PwCalculation, inputs={'pseudos': orm.DataDict}, errors=PwErrors)
   neb_calc = task(NebCalculation, inputs={'pw': {'pseudos': orm.DataDict}}, errors=NebErrors)
   ```

   The graph contract has a static `DataDict` leaf; dispatch expands members into
   the process namespace and checks accepted node types. Recursive specialization
   also closes enclosing dynamic namespaces. No routing WorkChain is needed.
   Expansion itself preserves member objects; existing owned-container/process
   boundary rules can detach members into standalone nodes. Do not claim universal
   identity preservation across ownership boundaries.

Missing required mappings, wrong member types, and missing restart nodes fail
explicitly. Optional absent mappings stay unwired; an independent review fixed a
case that incorrectly inserted an empty nested namespace while dropping a wire.

## Work division and verification of workers

Four workers owned core failed-output correction, core namespace forwarding,
native bases, and native composites. Two subsequent independent workers reviewed
core semantics and full nested QE execution. The coordinating agent integrated
and verified their results rather than accepting isolated worker runs:

- Added the initially missing namespace-expansion dispatch hook.
- Removed the base track's temporary package-import spoof and all eight non-strict
  expected-failure markers. Their production tests now genuinely pass.
- Replaced dynamic test-file loading with a regular relative import.
- Corrected stale source explanations and misplaced changelog entries.
- Ran repository-configured pre-commit hooks, not just an unrelated Ruff config.
- Re-ran the combined QE suite and full selected core graph suite.

Worker handoff reports retain intermediate results for traceability. This report
supersedes their interim failures and the original migration/review conclusions.

## Validation results

| Check | Result |
| --- | --- |
| Combined QE graph suite | **100 passed**, 6 warnings, 242.69 seconds |
| Core graph suite, `-m presto -n 4` | **996 passed**, 4 warnings, 111.33 seconds |
| QE graph suite with isolated pytest configuration | **100 passed**, 1169 mostly SQLAlchemy/dependency warnings, 250.59 seconds |
| Focused mapped-output + forwarding regressions | **42 passed** in independent review |
| Core pre-commit on changed files | **Passed**, including Ruff and mypy |
| QE pre-commit on changed source/tests | **Passed** |
| Legacy workflow suite against new core | **89 failed, 106 passed, 1 skipped**, 27.74 seconds; protocol/core-3 compatibility remains red |

Commands from the parent core checkout:

```sh
uv run pytest -m presto tests/engine/processes/graphs -n 4 -q
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/graphs -q
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/workflows -q --tb=line
```

### What execution coverage proves

- Real production PW base graph: success, diagonalization/walltime/ionic recovery,
  failed-folder/structure rebinding by UUID, missing-node rejection, exhaustion,
  initial restart folder, and multi-kind Si/O pseudo mappings.
- Real production NEB base graph: direct CalcJob placement and diagonalization
  recovery through nested `pw.pseudos` forwarding.
- Full nested **relax/bands/serial-PDOS → native PW base → CalcJobs**, with only
  external CalcJob implementations substituted. No copied graph bodies and no
  replacement of the base graph in these integration tests. Assertions include
  absence of WorkChain descendants, root provenance, output identity, SCF→bands
  parent-folder identity, pseudo input links, and serial sweep ordering.
- Other composite tests substitute the base subgraph with deterministic function
  tasks; those test orchestration, not the complete nested stack.

These are deterministic local fake-executable/overridden-parser tests, **not real
QE scientific validation**. PH recovery/merging, daemon-interruption behavior, real
remote schedulers, and all CalcJob/parser output combinations are not verified.
The local core suite includes pre-existing untracked graph tests; only the new
mapping/failure tests and intended core source changes were committed.

## Remaining problems, classified accurately

### Framework execution guarantees / unresolved access

- **Failure/cancellation finalization:** success-path sweeps do not replace
  termination cleanup. Reliable cleanup requires defined finalization or an
  external resource owner; no graph guarantee was added here.
- **Actual child exit message:** output preservation does not expose it as a graph
  value. QE's message-gated accuracy-stuck recovery remains unresolved.
- **Per-attempt scheduler metadata:** launch metadata is fixed per placement.
  Only required for policies that actually change scheduler configuration;
  deriving QE `max_seconds` from a fixed scheduler limit is ordinary preparation.
- **Forwarded namespaces inside directly task-mapped `par_map` calls:** not
  covered by the new ProcessTask dispatch adapter. QE's fixed fan-out does not
  need this path. Do not infer general map support from direct-placement tests.

### QE implementation and compatibility work—not missing graph syntax

- Complete protocol-to-launch resolution and boundary validation helpers.
- PH provenance-preserving merge of recovered partial outputs.
- Monitor input forwarding and the monitor-specific recovery policy.
- All considered-finished exit states, optional bands relaxation phase, complete
  output/option coverage, and fixed-scheduler-limit `max_seconds` preparation.
- Recovery-handler coverage beyond the tested status paths; source translation
  alone is not evidence of equivalence.
- Prepared-input/dry-run inspection, legacy numeric exit-code compatibility, and
  `orm.AbstractCode` protocol usage incompatible with core 3.
- Distinct scheduler settings for subgraphs that reuse the same placement keys.

A mutable ProcessBuilder, general Python graph bodies, decorated-calcfunction
placement, a handler registry, or `par_map` for DOS/PROJWFC are **not prerequisites**.
The existing undecorated scientific implementations can be reused in task bodies.

## Remote verification

QE source revision **`cb94cf3565ff3379534819e2da06e270e62552a4`** was pushed to
`origin/rewrite`. The core dependency above was pushed before updating the CI pin.
A subsequent report-only commit records the results; it does not change executable
code. `ci-verification.json` contains the observed public API job/step conclusions.

| Run | Observed result |
| --- | --- |
| [QE stable-core CI 43910652437](https://github.com/agoscinski/aiida-quantumespresso/actions/runs/43910652437) | **Passed:** pre-commit and tests on Python 3.10–3.14. Graph tests deliberately skip on released core: this verifies stable compatibility, not native graph execution. |
| [QE native graph CI 43910652423](https://github.com/agoscinski/aiida-quantumespresso/actions/runs/43910652423) | New core installation/revision gate **passed**; graph-test step **passed**; legacy workflow regression step **failed**. Overall run is **red**, not a feature-complete migration gate. |
| [Core pre-commit 43909202414](https://github.com/agoscinski/aiida-core/actions/runs/43909202414) | **Passed.** |
| [Core docs build 43909202253](https://github.com/agoscinski/aiida-core/actions/runs/43909202253) | **Passed.** |
| [Broader core CI 43909202286](https://github.com/agoscinski/aiida-core/actions/runs/43909202286) | Still running at report observation; no full-core-CI success is claimed. Local graph regression results are separate. |

The configured `gh` credential was previously invalid; verification uses public
GitHub API job/step results. The local 89-failure legacy-suite breakdown is not a
claimed remote log transcript. Stable CI and native graph tests are green, but
legacy/core-3 compatibility and the outstanding behaviors listed above prevent a
feature-complete verdict.
