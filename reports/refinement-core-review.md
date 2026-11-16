# CORE namespace forwarding review

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

Independent CORE review of `DataDict` container forwarding into dynamic namespaces:
declaration (`tasks.py`), closing/expansion/serialization (`ir.py`), and the
dispatch hook (`run.py`, read-only).
Covers flat/nested forwarding, task-produced containers, rejection paths, member
identity, and how mapped failed-process outputs relate to `Result` absent arms.

## Verdict

The semantics are sound and the dispatch hook is correctly placed.
One proven fix was needed (`_drop_leaf` creating parents on drop, below).
No `run.py` changes are needed: no coordinator action required.
No QE source, QE tests, changelogs, or commits touched.

## Semantic review

Declaration (`_declared_shape`, `tasks.py`).
A bare `DataDict` (or a one-tuple of it) declares a container-forwarded namespace;
anything else keeps its node-typed meaning.
Correct: the marker resolves against the live spec in `_forwarded_paths`, which
rejects unknown paths, static namespaces, and already declared ports loudly at
build time.

Closing (`_close_dynamic`, `ir.py`).
Forwarded paths close into a static `DataDict` leaf preserving the namespace's
`required` flag, so graph wiring type-checks one container per path.
Correct, including nested paths closing each enclosing parent.

Expansion (`TaskIR.expand_forwarded`, `ir.py`).
Runs at dispatch, after wiring resolved every reference to a node, which is the
only point the container contents exist.
Members are type-checked against the namespace's accepted types, static ports and
defaults pass through, and the input mapping is copied rather than mutated.
Missing/empty/wrong-typed wires raise `GraphValidationError` naming the path and
member.
Correct as written; provenance identity holds up to dispatch (same-session `is`
identity is pinned by tests).

Dispatch hook (`GraphRun._inputs_for`, `run.py`, read-only).
Expansion runs after scope-input and dependency wiring, before the child submits.
Verified by actual `run_get_node` graph runs: correct totals for flat, nested,
task-produced, full-optional, and empty-optional wiring.
No change needed.

Provenance.
Members arrive at the child as the container's entries, but the engine detaches
owned container children into standalone copies at the process boundary
(`detach_owned_inputs`, container spec).
This is pre-existing, spec'd behavior, identical for direct process runs, not
something forwarding introduces: graph-run tests therefore assert values and
totals, while session identity is pinned at the expansion level.
The docs added here deliberately promise values, not UUID preservation.

Serialization.
`forwarded` round-trips on `TaskIR` and at the compiled-template level for every
forwarding graph; pinned by tests.

## Proven fix

`_drop_leaf` used `_copied_parent`, which creates missing intermediate mappings.
Dropping a missing wire on an optional *nested* path therefore left a spurious
`{'sub': {}}` in the expanded inputs instead of leaving the input unwired.
Plumpy accepts the empty namespace so nothing failed loudly, but the child
launched with an explicitly wired empty namespace.
`_drop_leaf` now walks without creating parents (creation stays in
`_copied_parent` for the set path), proven by
`test_expand_drops_missing_optional_nested_wire`, which fails before and passes
after the fix.

## Rejection is pre-dispatch

A wrong-typed member raises `GraphValidationError` out of `run_get_node` with
zero `WorkChain`/`CalcFunction` nodes created: not even the graph node persists,
so no child ever dispatches.
One discovery shaped the tests: containers lock their element type at insert,
so a mixed `DataDict` cannot even be built; the rejection path takes a
homogeneously wrong-typed container.

## Mapped failed outputs vs `Result` absent arms

`_resolve_placement_outputs` preserves every stored declared output of a mapped
failure (e.g. `remote_folder`, parsed files) next to the synthesized `error`
member; only arms the run never stored read as `None`, like a `Result` arm.
The stored node is never mutated.
The guide previously said mapped failures complete "with its success outputs
absent"; the docs now state the preserved-available rule explicitly so the two
failure styles are not confused.

## Test coverage added

`tests/engine/processes/graphs/test_namespace_mapping.py` (all real `run_get_node`
graph runs unless noted): flat, nested, task-produced (`forward_chained`),
full-optional, and empty-optional forwarding with totals and child-input values;
wrong-member rejection with a node-count pre-dispatch assertion; template
serialization round trips for all five forwarding graphs; optional-nested
missing-wire expansion.
Module docstring updated: the dispatch hook it called "not wired here" is now
wired and covered.

## Docs

`docs/source/topics/workflows/graphs.rst`: new `namespace_forwarding` label under
dynamic namespaces with syntax, marker constraints, dispatch checks, and the
empty-optional rule; mapped-failure paragraph corrected to preserved-available.
One sentence per line, sentence-case headings kept.

## Boundaries and follow-ups

- Forwarding inside a mapped (`par_map`/`map`) callable is not wired: only
  `ProcessTask` dispatch expands. Untested, left for whoever needs it; nothing
  suggests silent corruption, but no test pins the failure mode either.
- `CalcJob` forwarding (`forward_calc`) stays build-only: shell coverage lives
  in `test_mapped_failure_outputs.py`, which was read for the available-vs-absent
  distinction but not modified.
- Fixtures use `WorkChain`s for the integration behind both shapes, per the CORE
  allowance; QE keeps none.
