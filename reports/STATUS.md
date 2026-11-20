# Native QE graphs: status and next steps

## Verdict

Native graphs run without WorkChains, but **WorkChain parity is not complete**.
Keep legacy entry points unchanged. This is the current status document; archived
reports and their test counts are not current release evidence.

Only QE was changed in this iteration. No core features or convenience APIs were
added. Development checks use targeted subsets; full integration/compatibility CI
is deferred until the rewrite is ready for its final verification push.

## Port to the current graph language

The rewrite targets core `7896ff4e0ee7a42f9e7f2004344ca4f27ef5bdd6`
(`feature/graph-error-returns`), now the QE graph CI pin. The older revision
below describes historical checks, not the launch API of this port.

- Native graphs now use inline `.alias(...)` for stable stage placements and for
  repeated calls in the same scope. Launch and protocol helpers use
  `runtime_config`, e.g. `{'scf': {'pw_calc': {'metadata': {'options': options}}}}`.
  Existing graph-wired executables are unchanged and cannot be overridden by
  launch configuration.
- Core can bind dynamic monitor mappings on a CalcJob placement at launch.
  The port retains both PW placements and the graph-input monitor contract for
  now. A regression case exercises a launch-configured monitor on `pw_calc`.
  A single placement is a candidate if monitors become launch-only; this does
  not enable forwarding a graph-produced container into a dynamic namespace.
- PW, PH, and NEB return `Err(...)` directly if restart recovery fails, rather
  than dispatching an `ensure_*_recovered` task. Retry-budget exhaustion reports
  401 and an unhandled error reports 402. PW 501/710 still retain their usable
  outputs and distinct report statuses. Composite stage-specific exit statuses
  are still a follow-up.
- `PreparedRun` and `ExecutionEnvironment` are available for protocol and launch
  integration. The optional Slurm OOM retry policy does not replace QE restart
  logic, prepare PH cutoffs, or provide termination cleanup.

## Historical core selection and CI mismatch

The local core used for the checks below is:

```
6c40e797243034837ab8516582d11c86cccf9bc8
```

It retains mapped failure outputs, static-container serialization/staging fixes,
nested fixed-namespace specialization, and structured placement metadata. It has
no container-to-dynamic-namespace forwarding or wildcard metadata rules. Repeated
placements may use generated names such as `pw_calc_2`.

The previous `.github/workflows/graph.yml` pin was
`955f04745d114bf4c51b219de869e1bab14b7e64`. CI on that historical revision
is not evidence for the current implementation.

## Implemented in the current QE iteration

### Static monitor placement

The base graph uses two mutually exclusive handles:

```python
pw_calc = task(PwCalculation, errors=PwErrors)
pw_calc_monitored = task(
    PwCalculation, inputs={'monitors': {'stuck': orm.Dict}}, errors=PwErrors
)
```

A boundary task validates/extracts the single supported structured
`quantumespresso.accuracy_stuck` monitor under `stuck`. Empty/absent mappings use
the plain handle, with no monitor inputs. Unknown keys, wrong node types,
unrelated entry points, and unstructured stops are rejected. Recovery still
matches `monitor_diagnostics['reason']`, never message prefixes. Monitor-node
inputs are copied/normalized without mutating owned input members.

This is deliberately narrower than the legacy arbitrary-monitor namespace; it
is not general monitor parity. Core CalcJob monitoring is unchanged.

Both static placements need configuration, even though only one branch runs:

```python
runtime_config = {
    'scf': {
        'pw_calc': {'metadata': {'options': scf_options}},
        'pw_calc_monitored': {'metadata': {'options': scf_options}},
    },
    'bands': {
        'pw_calc': {'metadata': {'options': bands_options}},
        'pw_calc_monitored': {'metadata': {'options': bands_options}},
    },
}
```

QE protocol helpers generate both leaves, synchronize single-sided overrides,
and reject conflicting sibling settings. Nested scope configuration and empty
metadata leaves are retained; no flattened labels, wildcards, or scope defaults
were reintroduced.

### Recovery and composite acceptance

- PW statuses 501/710 return stored calculation outputs with distinct report
  statuses, rather than being retried as unhandled failures.
- Relax accepts 501 and applies the legacy Pulay-stress test (10% pressure
  tolerance); it rejects 710. The initial relaxation has the same acceptance list.
- Bands and PDOS require fully converged base outcomes. Explicit dependency guards
  prevent subsequent stages from launching after rejected outcomes.
- Malformed provided outcome reports fail closed instead of implying success.

These are report-status contracts, **not exact legacy numeric root exit codes**.

### Protocol preparation and relax outputs

- PW protocol launch helpers prepare `CONTROL.max_seconds` from 95% of the final
  fixed per-stage scheduler wallclock, preserving explicit `max_seconds` values.
  Conflicting/invalid limits fail actionably. Direct base launches still require
  callers to prepare such cutoffs themselves; PH preparation remains incomplete.
- Initial/main relaxation parameters, settings, k-point meshes/densities/parity,
  and monitors resolve independently for protocol launches. Direct launches keep
  their previous fallback defaults through `init_use_main_inputs=True`.
- Main explicit k-points no longer overwrite initial-stage mesh density.
- Relax returns the final calculation's parameters, remote folder, trajectory,
  bands, k-points, and occupations when present, preserving output-node identity.
  With zero iterations these optional outputs are absent; structure/report remain.

### PH recover-chain outputs

PH now folds recover outputs inside the production graph using the existing merge
mathematics. Each merge links the previous aggregate and the current original
parameter node directly; no container copies detach the inputs. Scratch restarts
reset the aggregate. Multi-run chains reject missing outputs or incomplete
requested q-point ranges. Single runs and initialization-only runs return the
original output node, and the last remote folder is exposed.

This creates a linear chain of pairwise merge calculations instead of the legacy
single N-way merge calculation. Original outputs remain reachable through normal
provenance links. External post-termination merge helpers remain available.

Review also corrected the PH recovery latch and omitted-settings preparation:
retry parameters now reflect the legacy `prepare_process` recovery behavior, and
omitting optional settings no longer returns an invalid absent task output.

### Legacy/core-3 compatibility

PW/PH protocol builders use the supported `orm.Code` base. Pseudo-family test
fixtures request fresh standalone nodes instead of deduplicating to owned
container members. Container-aware tests compare pseudo content rather than
requiring copy-on-insert members to retain caller UUIDs. Ownership invariants and
static `pseudos: DataDict` ports remain unchanged.

## Verification on the new core pin

- Core graph suite: 1,090 passed locally. GitHub Actions pre-commit and docs
  builds passed; the core test matrix is still running.
- QE graph suite: 271 passed. A separate PW budget regression passed and
  asserts exit status 401 without dispatching the success report task.
- QE legacy workflow suite: 198 passed, 1 skipped against the experimental
  core; 196 passed, 3 skipped against released core 2.9.4. The owned-container
  regression only applies to experimental core.

## Historical verification

Independent targeted checks on the earlier combined working tree:

- 11 selected input-resolution, malformed-report, scope-ordering, and graph-build
  checks passed (38.64 seconds).
- 3 selected relax output-identity/zero-budget and monitor-normalization checks
  passed (4.53 seconds).
- 12 selected outcome-classification, monitor-boundary, and protocol validation
  checks passed (3.71 seconds).
- 3 selected PH production-recover, pairwise-merge provenance, and scratch-reset
  regressions passed (15.88 seconds), with only the external calculation replaced.
  The production-recover test omits settings to cover the default-input fix.
- QE's configured Ruff lint/format pre-commit hooks passed on the initial parity
  checkpoint; subsequent PH changes are checked separately before committing.

Workers also ran focused unit/build subsets. An earlier worker reported 198
legacy workflow tests passed, 1 skipped, before long runs were deferred. That is
not a fresh final combined compatibility result. No all-green CI, scientific QE
executable, or remote-scheduler equivalence is claimed.

## Remaining work before parity

1. Complete exposed outputs and optional phases, especially PDOS's legacy
   namespaced/full output contract. Do not add side-products or a relaxation phase
   to bands merely for symmetry: the current legacy implementation has no such
   phase and its five named bands/SCF/path outputs are represented natively.
2. Broaden PH production recovery coverage (multiple recoveries, scratch reset,
   missing/incomplete outputs) and account explicitly for the pairwise provenance
   difference. Parent-derived structure/q-point preparation remains incomplete.
3. Define the caller-owned execution/finalization interface for cleanup after
   success, failure, and cancellation. `cleanup_graph_workdirs(node)` exists, but
   successful-path tasks are not termination hooks. Cleanup failure must not hide
   the original process outcome.
4. Complete remaining input/protocol/recovery policies and fixed-limit PH
   `max_seconds` preparation. Add production-graph regression cases for each
   meaningful policy, using external CalcJob substitutes rather than copied bodies.
5. Resolve or explicitly accept root numeric exit-status differences, the single
   supported monitor restriction, and prepared-input/dry-run differences.
6. Run QE graph, legacy, and stable-core compatibility CI using the new pin.
   Real QE executable and remote scheduler checks are required before claiming
   deployment/scientific parity.

No missing QE implementation above is, by itself, a request for a new core API.
