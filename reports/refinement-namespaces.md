# Namespace forwarding: `DataDict` wires into dynamic namespaces

> Historical worker handoff / earlier audit. For integrated validation and current
> conclusions, see [graph-migration.md](graph-migration.md); interim failures and
> test totals below are not the final combined result.

- **Track:** core namespace forwarding (this track only).
- **Owned files (this track only):** `src/aiida/engine/processes/graphs/ir.py`,
  `src/aiida/engine/processes/graphs/tasks.py`,
  `tests/engine/processes/graphs/test_namespace_mapping.py`,
  `tests/engine/processes/graphs/programs/namespace_mapping.py`, and this report.
  `src/aiida/engine/processes/graphs/run.py` and `process.py` were deliberately **not**
  touched (see section 4); nothing was committed or pushed.
- **Goal:** remove the QE routing `WorkChain`s (`NebStaticWrapper`-style shims). The graph
  layer now accepts `task(PwCalculation, inputs={'pseudos': orm.DataDict})` and
  `task(NebCalculation, inputs={'pw': {'pseudos': orm.DataDict}})`: a `DataDict` wire is
  expanded into the declared process dynamic namespace before dispatch.

## 1. QE consumption (ready to use at declaration time)

```python
pw_calc = task(PwCalculation, inputs={'pseudos': orm.DataDict}, errors=PwErrors)
neb_calc = task(NebCalculation, inputs={'pw': {'pseudos': orm.DataDict}}, errors=NebErrors)


@graph
def pw_step(code, structure, parameters, pseudos: DataDict, kpoints, ...) -> ...:
    result = pw_calc(code=code, structure=structure, parameters=parameters, pseudos=pseudos, kpoints=kpoints)
    ...


@graph
def neb_step(code, images, parameters, pseudos: DataDict, pw_parameters: Dict, kpoints, ...) -> ...:
    result = neb_calc(
        code=code,
        images=images,
        parameters=parameters,
        pw={'parameters': pw_parameters, 'pseudos': pseudos, 'kpoints': kpoints},
    )
    ...
```

Verified in this worktree against the real specs: `PwCalculation.spec().inputs['pseudos']`
is a required dynamic namespace accepting `UpfData`, and
`NebCalculation.spec().inputs['pw']` is itself a required dynamic namespace nesting the
required dynamic `pw.pseudos`. Both declaration forms above build, and the closed contracts
validate, in a probe against the live QE classes. Rules for the QE worker:

- A bare `orm.DataDict` (or a one-tuple of it) at a dynamic-namespace path marks that
  namespace as container-forwarded. Anything else is unchanged: static entries keep node
  types (`inputs={'pseudos': {'Si': UpfData}}`), and that nested-static form now closes too
  (see section 2).
- The `DataDict` keys are the dynamic entry names (e.g. structure kind names); every member
  is checked at dispatch against the namespace's accepted types (`UpfData` for both QE cases
  above). A wrong member type fails the dispatch with the port path, key, and expected types.
- An empty `DataDict` on a required namespace (both QE `pseudos`) fails loudly instead of
  launching without entries. On a non-required namespace it leaves the input unwired.
- Required/static ports, defaults, and namespace validators are untouched: every other
  required port must still be wired, and the expanded mapping validates against the live
  process spec at submit. The child keeps its identity: no wrapper node, `PwCalculation` /
  `NebCalculation` run directly.
- Provenance: the child links directly to each member node; the `DataDict` wire itself
  leaves no link. Members keep their node identities (no copies are made).
- No generic-dict loophole: plain `dict` in `inputs=` is rejected at declaration, and a
  non-`DataDict` wire at a forwarded path is rejected at dispatch. Only `orm.DataDict`
  (identity-checked, no duck typing) marks a forwarded namespace.

## 2. What changed in core

`src/aiida/engine/processes/graphs/ir.py`:

- `TaskIR.forwarded: tuple[str, ...]` — dotted input paths container-forwarded by a bare
  `DataDict` declaration. Serialized in `to_dict`/`from_dict` (only when non-empty; version
  stays `0.1`). Old readers ignore the key and then fail loudly at port-close, never silently.
- `_forwarded_paths(process_class, declared)` — resolves each `DataDict` leaf in the
  declaration against the live spec. Forwards only dynamic namespaces; a static leaf, a
  static namespace, or an unknown path raises `GraphTypeCheckError` naming the fix.
- `_close_dynamic(..., forwarded=...)` — a forwarded path closes into a static
  `InputPort(valid_type=(DataDict,))` preserving the namespace's required flag and help, so
  build, routing, and `_check_types` work unchanged. Also fixes the recursive parent flag:
  a namespace whose child closed into a static contract now counts as declared, so nested
  declarations (`inputs={'pw': {'pseudos': ...}}`) fix every enclosing parent instead of
  leaving a dynamic `pw` behind (the old ``namespace `pw` is dynamic`` failure, i.e. QE
  blocker B1 in `reports/pw-migration.md`).
- `TaskIR.expand_forwarded(inputs)` — the dispatch adaptation (pure: returns a new mapping,
  never mutates its input). Pops each `DataDict` wire, validates members with `isinstance`
  against the live namespace's `valid_type`, and installs the member mapping under the
  namespace path. Raises `GraphValidationError` on a missing required wire, a non-`DataDict`
  wire, an empty required container, or a mistyped member. `_copied_parent` / `_drop_leaf`
  copy each nesting level so shared scope inputs are never mutated.

`src/aiida/engine/processes/graphs/tasks.py`: documented the `DataDict` marker on
`PortDeclaration` and `_declared_shape` (no behavior change; `DataDict` already shaped as a
leaf there).

Untouched on purpose: `lowering.py` (wiring/reference logic unchanged — forwarded leaves are
ordinary `DataDict` leaves at build time), `validate.py` (existing `_infer_node_types`
covers the leaf), `process.py`, `run.py`.

## 3. Tests

`tests/engine/processes/graphs/programs/namespace_mapping.py` (new fixture module):
`ItemGather` (runnable `WorkChain` with required dynamic `members`, optional dynamic
`extras`, required static `factor`, defaulted static `offset`), `NestedGather` (exposes it
under dynamic `sub`, mirroring `pw.pseudos`), `ItemCalcJob` (flat dynamic namespace,
mirroring open calculation ports), a `make_bundle` producer task, and six graphs covering
flat / optional / nested / chained / CalcJob / nested-static wirings.

`tests/engine/processes/graphs/test_namespace_mapping.py` (24 tests): six `build()` pins;
closed-contract shape assertions (forwarded leaf types, static parents); serialization
round trip; five misuse rejections (static leaf, nested static leaf, unknown path, plain
`dict`, non-node member); runtime expansion with stored nodes (flat + nested: member
identity via `is`/`uuid`, no caller mutation, live `spec.inputs.validate` green); wrong
element type, empty required, empty optional (dropped), non-container wire, and missing
wire failures; and two true `run_get_node` executions of the expanded inputs against the
real processes. Full graphs suite: 976 passed; the only failures
(`test_on_wait_registers_each_pending_task_once`, three checkpoint/boundary tests, two
broker-daemon tests, broker errors) reproduce identically on the pristine tree — collateral
of the other worker's uncommitted `run.py` edits / missing RabbitMQ, not this track.

## 4. Recommendation for the coordinator (tiny `run.py` adapter, not applied)

End-to-end graph runs need one hook where per-task inputs are assembled.
In `GraphRun._inputs_for` (`src/aiida/engine/processes/graphs/run.py`), just before the
final `return inputs`:

```python
        if isinstance(task, ProcessTask) and task.spec.forwarded:
            inputs = task.spec.expand_forwarded(inputs)
        return inputs
```

No new import needed (`ProcessTask` is already imported there). `run.py` is owned by the
runtime-output worker, so this track did not apply it. Rationale for this exact spot:
`_inputs_for` assembles routed graph inputs and dependency edges uniformly for root and
nested scopes, loops, and map items; `expand_forwarded` is pure and inputs are never
checkpointed (only labels/pks are), so scheduling and restarts are unaffected. The
`process.py:_submit` alternative would also work but splits input-shaping across two files.
After the hook lands, the QE worker can delete the routing shims and cover the full
`run_get_node` path; the adapter, validation, and execution contracts are already pinned by
this track's tests.
