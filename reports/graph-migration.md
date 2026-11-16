# Quantum ESPRESSO graph migration — verification report

## Verdict

**Not feature-complete. Experimental alternatives, not a replacement release.**
All nine legacy workflow families have graph alternatives, but important restart,
protocol, validation, cleanup, output, and exit-code contracts are not preserved.
Existing workflow entry points deliberately still resolve to the legacy WorkChains.
Do not use the graph alternatives as production-compatible substitutes.

The implementation targets `agoscinski/aiida-core` commit
`3db2173f3c4761a4968595b2fdfeb7f3c63d8d32` (3.0.0.dev0), not released core 2.8.
QE's stable dependency constraint is unchanged. Dedicated `graph` CI installs and
verifies the exact experimental revision; ordinary CI tests the stable dependency.

## Work division and independent verification

Four worker agents owned disjoint tracks: six bases, PW relax/bands, PDOS, and
framework/CI audit. A subsequent worker reviewed base production correctness.
The coordinating agent reviewed integration, removed exception-swallowing test
import fallbacks, ran all combined tests, checked isolated pytest configuration,
and changed legacy imports from plumpy to AiiDA's public `ProcessState` and
`PortNamespace` (needed with core 3, compatible with core 2).

Review found two serious bugs despite the original 79 passing tests:

1. Later handlers discarded earlier restart decisions. Decisions now propagate.
2. Retry paths silently continued without failed-run nodes hidden by the graph
   framework. Guards now fail explicitly rather than run degraded restarts.

Three new tests execute the **production** PW base graph with an external-placement
substitute: first success, diagonalization recovery, and blocked walltime recovery.
This caught wiring defects that copied graph bodies cannot establish. See
[base-review.md](base-review.md) for evidence and exact remaining limits.

## Coverage by legacy workflow family

| Family | Graph implementation | Missing / changed behavior |
| --- | --- | --- |
| PW base | Native retry loop and handler tasks | Only parameter-only diagonalization and success-path bands growth can recover; failed-node-dependent paths explicitly fail. Fixed Si pseudo shape; no monitor-message handling, considered-finished 501/710, or scheduler-time cap. |
| PH base | Native retry loop and handler tasks | Parameter-only diagonalization supported; remote-folder recoveries blocked. No multi-recover output merge; explicit structure for q-point generation. |
| NEB base | Native retry loop over static-port WorkChain shim | Parameter-only diagonalization supported; remote-folder-dependent recoveries blocked. Fixed Si pseudo shape and extra workflow node. |
| Q2R base | Single CalcJob graph | No handlers in original; full API/output/cleanup parity and real external execution not verified. |
| Matdyn base | Single CalcJob graph | Same qualification as Q2R. |
| Bands base | Single CalcJob graph | Same qualification as Q2R. |
| PW relax | Native init/iteration/convergence orchestration | Places legacy PW base through routing shim; no 501 Pulay recovery, protocol builders, full init namespaces, or cleanup. Zero-budget and SCF output behavior differ. |
| PW bands | Native path → SCF → bands orchestration | Places legacy PW base; optional relax phase, full namespace forwarding/protocols/cleanup/legacy failure codes not preserved. |
| PDOS | SCF → NSCF → parallel DOS/PROJWFC, or parent restart | Places legacy PW base; serial cleanup ordering, all optional outputs/protocols/validators/exit codes not preserved. |

Detailed input/output maps and deviations: [base-migration.md](base-migration.md),
[pw-migration.md](pw-migration.md), [pdos-migration.md](pdos-migration.md).
These are worker handoff records; their original test totals/environment notes are
historical. This report and base-review supersede earlier recovery/parity claims.

## Problems with the graph authoring layer

### Highest priority

1. **Failed-calculation data is inaccessible.** `errors=` gives an error enum but
   masks all success outputs, even `remote_folder` and structures actually stored
   by a failed child. QE restart handlers need those nodes. Expose a typed failed-run
   record (outputs, exit status/message) without pretending the run succeeded.
2. **Nested dynamic namespaces cannot be closed adequately.** PW-base's `pw.pseudos`
   and NEB's nested input contract require routing WorkChains. Flat Si declarations
   in the base track are not general pseudopotential support. Recursive static shape
   specialization or typed owned mapping ports would remove this boilerplate.
3. **Workflow failure vocabulary collapses.** A root cannot preserve QE's distinct
   legacy exit statuses; most failures become `400 ERROR_TASK_FAILED`. A summary node
   is not equivalent to a workflow exit contract. Add typed graph-level failure exits.
4. **No reliable termination/failure cleanup equivalent.** Successful-path sweep tasks
   do not replace `on_terminated`; failed remote calculations may leave directories.
   Add structured finalization with explicit resource/provenance semantics.

### Authoring friction / contract gaps

- Conditional ordering (`after=` differing between branches), branch-local names,
  and restricted branch shapes made PDOS serial ordering awkward or unavailable in
  the current implementation. Distinguish deliberate language restrictions from
  unnecessarily restrictive compiler wiring checks.
- Scheduler `task_metadata` is placement-wide, not adjustable per iteration; restart
  handlers cannot adapt scheduler/time settings naturally.
- Static graph ports do not reproduce arbitrary monitor maps, protocol overrides,
  or namespace validators. Runtime task validation happens later than legacy launch
  validation. Graph `dry_run` is unsupported.
- Function tasks are calculation processes: they cannot call decorated calcfunctions
  or return existing input nodes as newly created outputs. The adapters use undecorated
  implementations and clones where needed. These are provenance constraints, not bugs;
  better authoring support for node selection would avoid unnecessary copies.
- Annotation-resolution and optional-output conventions are easy to get wrong;
  `TypedDict` must be total, while optional leaves need defaulted `PortModel` fields.
- Whole-port mappings/collections and per-iteration node accumulation are awkward;
  PH needs provenance-preserving output merging, not just value snapshots.
- Serial loops, a 1000-iteration cap, stored-only execution, and the experimental
  core dependency remain deployment constraints.

Framework source references and probes: [graph-layer-audit.md](graph-layer-audit.md).

## Local verification

From the parent core checkout:

```sh
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/graphs -q
```

**87 passed, 4 warnings** (127.96 seconds). Repeated with isolated QE pytest root,
`-c /dev/null --confcutdir tests`: **87 passed**, with additional SQLAlchemy warnings.
No real QE executable, remote scheduler, daemon restart, or full PH/NEB runtime was
validated. PW/PDOS composite runtime tests still use mirrored stub graphs; their
production bodies have build checks, not full production-placement runtime coverage.

Legacy workflows against experimental core, stopping at first failure:
**37 passed, 1 failed**: `tests/workflows/protocols/ph/test_base.py::test_default`
uses `orm.AbstractCode`, removed in core 3. This compatibility problem is not a graph
compiler defect, and must be migrated before a feature-complete core-3 release.
The dedicated CI intentionally includes this regression gate rather than hiding it.

## Remote verification

Pending push and GitHub Actions results. This section will be updated with the
pushed implementation commit, run URLs, conclusions, and any infrastructure limits.
