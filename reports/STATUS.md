# Native QE graphs: status and next steps

## Verdict

Native graphs run without WorkChains, but **WorkChain parity is not complete**.
Keep legacy entry points unchanged. This is the current status document; archived
reports and their test counts are not current release evidence.

Only QE was changed in this iteration. No core features or convenience APIs were
added. Development checks use targeted subsets; full integration/compatibility CI
is deferred until the rewrite is ready for its final verification push.

## Core selection and CI mismatch

The local core used for the checks below is:

```
6c40e797243034837ab8516582d11c86cccf9bc8
```

It retains mapped failure outputs, static-container serialization/staging fixes,
nested fixed-namespace specialization, and structured placement metadata. It has
no container-to-dynamic-namespace forwarding or wildcard metadata rules. Repeated
placements may use generated names such as `pw_calc_2`.

`.github/workflows/graph.yml` still pins
`955f04745d114bf4c51b219de869e1bab14b7e64`. CI on that historical revision does
not validate the current implementation. Before final CI, publish the agreed core
revision, update the pin to that published revision, and verify graph and legacy
gates. Do not point CI at an unpublished local hash.

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
task_metadata = {
    'scf': {
        'pw_calc': {'options': scf_options},
        'pw_calc_monitored': {'options': scf_options},
    },
    'bands': {
        'pw_calc': {'options': bands_options},
        'pw_calc_monitored': {'options': bands_options},
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

### Legacy/core-3 compatibility

PW/PH protocol builders use the supported `orm.Code` base. Pseudo-family test
fixtures request fresh standalone nodes instead of deduplicating to owned
container members. Container-aware tests compare pseudo content rather than
requiring copy-on-insert members to retain caller UUIDs. Ownership invariants and
static `pseudos: DataDict` ports remain unchanged.

## Verification so far

Independent targeted checks on the combined working tree:

- 11 selected input-resolution, malformed-report, scope-ordering, and graph-build
  checks passed (38.64 seconds).
- 3 selected relax output-identity/zero-budget and monitor-normalization checks
  passed (4.53 seconds).
- 12 selected outcome-classification, monitor-boundary, and protocol validation
  checks passed (3.71 seconds).
- QE's configured Ruff lint/format pre-commit hooks passed on all changed Python
  files and the new protocol-parity tests after formatting.

Workers also ran focused unit/build subsets. An earlier worker reported 198
legacy workflow tests passed, 1 skipped, before long runs were deferred. That is
not a fresh final combined compatibility result. No all-green CI, scientific QE
executable, or remote-scheduler equivalence is claimed.

## Remaining work before parity

1. Complete exposed outputs and optional phases, especially PDOS's legacy
   namespaced/full output contract and bands' optional relaxation phase. Do not
   add side-products to bands merely for symmetry: its legacy public result is
   the five named bands/SCF/path outputs, already represented natively.
2. Complete PH recovery-chain result ownership and merged parameters. Existing
   helpers merge original stored nodes with direct provenance links after graph
   termination; that does not replace the graph's stored last-run output.
3. Define the caller-owned execution/finalization interface for cleanup after
   success, failure, and cancellation. `cleanup_graph_workdirs(node)` exists, but
   successful-path tasks are not termination hooks. Cleanup failure must not hide
   the original process outcome.
4. Complete remaining input/protocol/recovery policies and fixed-limit PH
   `max_seconds` preparation. Add production-graph regression cases for each
   meaningful policy, using external CalcJob substitutes rather than copied bodies.
5. Resolve or explicitly accept root numeric exit-status differences, the single
   supported monitor restriction, and prepared-input/dry-run differences.
6. Publish/select the core revision, update the CI pin, then push QE to run full
   graph, legacy, and stable-core compatibility CI. Real QE executable and remote
   scheduler checks are required before claiming deployment/scientific parity.

No missing QE implementation above is, by itself, a request for a new core API.
