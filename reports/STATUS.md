# Native QE graphs: status and next steps

## Verdict

The native graph implementation runs without WorkChains, but is not a
production-compatible replacement for the legacy workflows. Keep legacy entry
points unchanged until the behavior and compatibility checks below are complete.

This is the single current status document. Earlier audits and worker reports were
archived outside the repository and removed from this branch's history. Their
feature requests and test totals are not current requirements or release evidence.

## Resolve the core revision mismatch first

The reviewed local aiida-core revision is:

```text
fee30f16bcf9988f37202f6c8adc98ec8fb44763
```

It contains:

- Nested fixed-namespace specialization fixes.
- Stored outputs retained alongside mapped process errors.
- Builder serialization and file-staging fixes for static containers.
- Structured `task_metadata`, resolved through actual graph scopes.

It does **not** contain container-to-dynamic-namespace forwarding, wildcard metadata
rules, or the restriction requiring distinct aliases for repeated CalcJob calls.
Repeated calls retain the existing generated names such as `pw_calc_2`.

QE's `.github/workflows/graph.yml` still pins:

```text
955f04745d114bf4c51b219de869e1bab14b7e64
```

That is a different experimental core history. Do not claim that passing CI on
that pin validates the reviewed local core. Publish the agreed core revision, then
update `CORE_REV` and rerun both the graph and legacy compatibility gates. Do not
change the pin to an unpublished local commit.

## 1. Finish the static monitor-input adaptation

Committed QE code still declares:

```python
pw_calc = task(PwCalculation, inputs={'monitors': orm.DataDict}, errors=PwErrors)
```

This requires the removed namespace-forwarding feature. The current uncommitted
`workflows/graphs/base.py` edit instead declares the known monitor statically:

```python
pw_calc = task(
    PwCalculation,
    inputs={'monitors': {'stuck': orm.Dict}},
    errors=PwErrors,
)
```

It extracts the `stuck` entry at the boundary and omits the monitor input when no
monitor is requested. Review and test that edit; it is not included in this
report/history cleanup.

Update the substituted task shapes in tests as well. For example,
`tests/graphs/test_monitors.py` still declares `inputs={'monitors': orm.DataDict}`.
Search the whole graph suite for remaining declarations using that removed adapter.

Required cases:

- No monitor: no monitor input links on the CalcJob.
- Supported structured `stuck` monitor: diagnostic output drives recovery.
- Missing or unrelated diagnostic: do not apply accuracy-stuck recovery.
- Unsupported mapping keys or wrong node types: actionable rejection.

CalcJob already implements monitoring. No new core monitor engine is needed.
Dispatch recovery on `monitor_diagnostics['reason']`, not exit-message prefixes.
Document the single-supported-monitor restriction rather than silently dropping
arbitrary protocol overrides.

## 2. Validate structured scheduler configuration against the selected core

QE already uses nested metadata, for example the bands workflow:

```python
task_metadata = {
    'scf': {'pw_calc': {'options': scf_options}},
    'bands': {'pw_calc': {'options': bands_options}},
}
```

Check protocol-generated paths against production graph placements, including
initial/main relaxation and serial/parallel PDOS. Preserve empty leaf mappings
when they intentionally request CalcJob defaults.

Verify distinct settings reach the actual SCF/bands, SCF/NSCF, and initial/main
CalcJobs, including retries. Do not reintroduce flattened-label aliases, wildcard
rules, fallback precedence, or an alias-renaming requirement merely for old tests.

Protocols, pseudo-family selection, threshold resolution, and launch-input
construction belong in QE, not in a new aiida-core builder API.

## 3. Repair legacy/core-3 compatibility

The previous legacy workflow gate was red. Archived results include removed
`orm.AbstractCode` usage and pseudo-family/group setup involving owned container
members. Reproduce failures against the selected core and classify them before
changing code; historical counts are not a current failure inventory.

Keep static `pseudos: DataDict` inputs separate from dynamic process namespaces.
Test ordinary mapping serialization, multi-kind structures, file staging, and
pseudo-family selection. Account explicitly for container copy/ownership rules;
do not assume an owned member can be grouped or linked like a standalone node.

## 4. Complete QE behavior, not WorkChain API imitation

Compare each native graph with its legacy workflow and record deliberate changes.
Prioritize concrete missing behavior:

- Status 501/710 and other considered-finished/recovery policies: define the desired
  outcome and usable-output contract, then test the real production graph.
- Complete optional phases, exposed options, and output coverage.
- Prepare QE `max_seconds` from a fixed scheduler limit where required. This does
  not require per-attempt scheduler metadata.
- Complete protocol-to-launch resolution and prepared-input inspection.
- Decide which terminal outcomes callers must distinguish. Exact legacy numeric
  exit statuses are compatibility decisions, not automatic core prerequisites.

Use external CalcJob substitutes to exercise production graphs, not copied graph
bodies. Add real QE executable and remote-scheduler validation before claiming
scientific or deployment equivalence.

## 5. Make cleanup and PH result ownership explicit

`cleanup_graph_workdirs(node)` exists for the caller to invoke after termination.
In-graph sweeps cover success and explicitly handled failure paths; they are not
automatic termination hooks. Specify which caller owns cleanup after unexpected
failure or cancellation, and test that ownership. Preserve the original process
outcome if cleanup fails.

PH helpers already reconstruct recover chains and call a merge calcfunction with
explicit original input nodes. The merge is external post-processing; it does not
replace the graph's stored `output_parameters`. Decide and document the caller's
result interface before claiming parity with legacy automatic merging.

Neither issue alone establishes a requirement for a new core lifecycle or collector
API. Request a core change only with a concrete case the existing mechanisms cannot
express under the agreed ownership contract.

## Validation and completion criteria

Historical QE evidence: 193 graph tests passed on the previous experimental pin,
with substituted external calculations; the legacy compatibility gate failed.
Those results do not validate the current core/monitor combination. No fresh QE
suite was run as part of this report cleanup.

Run from the parent core checkout, after aligning the dependency and test shapes:

```sh
uv run pytest -m presto tests/engine/processes/graphs -n 4 -q
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/graphs -n 4 -q
PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \
  pytest aiida-quantumespresso/tests/workflows -q --tb=short
```

Record the exact tested core/QE revisions and distinguish committed code from local
edits. Before changing legacy entry points, require passing graph tests, resolved
or explicitly accepted legacy compatibility differences, production-graph recovery
coverage, and defined cleanup/merged-result ownership. Run each repository's
configured pre-commit checks; report failures rather than assuming historical lint
or CI results still apply.
