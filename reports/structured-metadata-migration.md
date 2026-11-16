# Structured metadata migration — verified iterations

## Iteration 1: core contract from `task.md`

Core branch: `refactor/structured-graph-metadata`.
Focused feature commit: `062b26173b30ae29714dd6c2f63f1a6d4a5b839f`.
Integration pin: `955f04745d114bf4c51b219de869e1bab14b7e64` (the feature followed
by the existing, separately committed builder and container-staging fixes).

The three exact-label / wildcard / iteration-normalization commits were replaced
by one feature commit. Namespace-forwarding (`fe808f614`) and failed-output
(`bc362ddce`) commits retain their original hashes and were not rewritten.
The final tracked tree was checked identical before and after consolidation.

An unrelated unfinished rebase existed in the original core worktree. It was not
aborted, continued, or force-pushed. The focused history was built and pushed from
`/tmp/aiida-structured-metadata`; the original rebase and its working copy remain
available for review.

### Naming and addressing

- Keys are existing authored callee names. Reuse a graph through ordinary Python
  aliases, e.g. `scf = pw_base_graph; nscf = pw_base_graph`.
- Each actual nested graph call contributes one mapping level. Controls (branches,
  loops, map bodies) contribute none. All iterations reuse the same entry.
- Repeating a metadata-bearing alias in one authored scope is rejected with alias
  guidance. No generated numeric suffix is required as a configuration address.
- No wildcards, label parsing, normalization, ancestor defaults, or fallback order.
  An authored name containing `__` is still an ordinary valid key; an unrecognized
  flattened provenance label fails as an unknown path, not via a syntax heuristic.
- Missing, unknown, and malformed paths fail before root/child creation where
  tested. Metadata leaf keys and values are checked against normal CalcJob ports,
  without filling or modifying the selected mapping.
- Provenance label formatting is unchanged. Structural scope lineage supplies the
  configuration path, and existing saved state restores the same path.

Verification: **1021 core graph tests passed**; the 25 focused metadata tests cover
both direct and repeated nested CalcJobs, three-level addressing, both branch arms,
loop reuse, invalid configuration, unchanged leaf contents, and two actual
`GraphRun.to_dict()/from_dict()` checkpoint restores with real CalcJobs and no
resubmission of completed jobs. Core Ruff/mypy pre-commit checks passed.

## Iteration 2: adapt QE authoring, not the engine again

No additional core feature was added for QE. Native graphs remain WorkChain-free;
static `pseudos: DataDict`, existing monitor forwarding, scientific preparation,
restart policies, and cleanup ownership are unchanged.

| Root | Structured `task_metadata` |
| --- | --- |
| PW bands | `{'scf': {'pw_calc': metadata}, 'bands': {'pw_calc': metadata}}` |
| PW relax | `{'init_relax_guarded': {'initial': {'pw_calc': metadata}}, 'main': {'pw_calc': metadata}}` |
| PDOS from scratch | `{'scf': {'pw_calc': metadata}, 'pdos_stage_serial': {'nscf': {'pw_calc': metadata}, 'dos_calc': metadata, 'projwfc_calc': metadata}}` |
| PDOS from parent | Same stage subtree, without the SCF entry |
| Native base directly | `{'pw_calc': metadata}` or the corresponding CalcJob name |

For parallel PDOS use the authored `pdos_stage_parallel` key instead of
`pdos_stage_serial`; these are concrete stage calls, not wildcard expressions.

The initial and main relaxation runs now accept independent protocol scheduler
settings, not just SCF/bands and SCF/NSCF. A production nested run verifies initial
walltime 1111 and main walltime 2222 on the actual substituted external CalcJobs.
Retry tests likewise verify independent stage settings through the real native
base loop. Compiled-production-graph checks pin the authored scope paths.

Protocol helpers emit only the chosen nested shape. Old label aliases, wildcard
entries, generated suffix keys, and leaf fallbacks are removed, not retained for
old tests. Empty metadata leaves are preserved (they request defaults); only
explicit `None` entries are omitted. Malformed protocol metadata receives an
explicit boundary error instead of an internal merge exception.

Final combined QE verification: **193 graph tests passed**, 30 warnings,
413.59 seconds. This includes two added boundary rejection cases, preservation of
empty metadata leaves, and a production initial/main run with distinct scheduler
options. Remote CI results are recorded in `graph-migration.md`.
Repository-configured QE Ruff checks passed. CI now installs pytest-xdist and runs
all graph tests with four workers; no graph tests are excluded.

## Scope and limits

This iteration changes scheduler configuration authoring and removes a previously
unnecessary init/main options restriction. It does not complete the remaining QE
science/recovery/output/inspection policies. Real QE executables and remote
schedulers were not used; external CalcJob substitutions exercise native production
orchestration. Checkpoint coverage above is persisted scheduler state, not a claim
of a new real-daemon interruption experiment.

Existing legacy/core-3 protocol compatibility failures remain a separate gate.
See `graph-migration.md` and `ci-verification.json` for observed CI outcomes rather
than treating a graph-test pass as full plugin compatibility.
