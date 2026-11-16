# Round 3: QE graph cleanup policies and PH scientific merge

> Worker handoff: interim test counts and diagnoses below are superseded by the
> integrated validation in [graph-migration.md](graph-migration.md).

New files only: `src/aiida_quantumespresso/workflows/graphs/cleanup.py`,
`src/aiida_quantumespresso/workflows/graphs/ph.py`,
`tests/graphs/test_cleanup.py`, `tests/graphs/test_ph_merge.py`.
Existing `base.py`/`pdos.py` are untouched; the coordinator wires the new pieces.

## 1. Cleanup contract (decided, explicit)

Graphs have no `on_terminated` hook, so there is no automatic `finally` to claim. Cleanup therefore has two
explicit owners, split by which paths can fire:

- **`clean_workdir` in-graph sweeps** (`sweep_graph_remote` task in the new `cleanup.py`, same shape as the
  existing `sweep_remote`/`sweep_stage` in `pdos.py`): run on the **success path and on explicitly wired
  handled-failure branches only**. If a placement fails with an unmapped error, or the graph is cancelled, the
  dependent sweep tasks never execute — this is a dataflow fact, not a bug to patch around.
- **Terminal failure / cancellation**: the **external owner** (caller/coordinator) runs the ordinary
  `cleanup_graph_workdirs(node)` function **after** the graph node terminated. It covers success, mapped and
  unmapped failure, and killed roots uniformly, because it reads stored provenance instead of relying on task
  edges.

`BaseRestartWorkChain.on_terminated` and the QE relax/PDOS termination sweeps are the reference for this policy,
not a guarantee: they also only run when the daemon reaches termination, so nothing here claims cleanup after a
crashed worker either. The owner simply re-runs the idempotent helper once the node is known terminated.

## 2. `cleanup_graph_workdirs` semantics

- **Refuses running work**: raises `ValueError` naming the node when the root is not terminated, and when any
  descendant process is still active. Nothing is touched in either case (verified: folders intact).
- **Ordinary function, not a task**: never changes process state or exit statuses; cleanup problems can never
  overwrite the process outcome.
- **Idempotent**: folders already marked `is_cleaned`, and calculations with no `remote_folder` output, are
  reported as skipped; a second run cleans nothing new.
- **Bounded, caught errors**: one attempt per folder; `OSError`/`KeyError`/`ValueError` from the transport layer
  are recorded per calculation (`failed: [{pk, error}]`), never raised.
- **Scoped, plus a root/work-folder guard**: only `RemoteData` nodes reached as `remote_folder` outputs of
  the graph's own descendant `CalcJob` nodes are cleaned, and the helper takes no path argument so a caller
  cannot aim it elsewhere. Scoping alone is not a path guarantee — a descendant `RemoteData` may point anywhere,
  including `/tmp` — so `/`, `/home`, and the computer work folder are additionally refused per folder (recorded
  as `failed`, storage untouched) while siblings are still cleaned. Verified by tests for the unlinked decoy, the
  `/` refusal with per-folder outcome, the work-folder refusal, and legitimate `/tmp` folders still cleaning.
- **`dry_run=True`** reports what would be cleaned without touching storage.
- The `@task` wrapper `sweep_graph_remote(folder, enabled)` is the in-graph counterpart for SUCCESS and HANDLED
  failure edges: the coordinator places it with data/`after` edges where the graph decides to clean.

Tests (`tests/graphs/test_cleanup.py`, 13 tests): real temporary directories rooted on the localhost computer,
real stored `GraphNode`/`CalcJobNode`/`RemoteData` provenance with `CALL_CALC`/`CREATE` links, real cleaning
through the local transport — no remote scheduler, no calculation execution. One test runs a genuine `@graph`
for the success shape; failed/killed roots use the stored shapes a termination observer actually sees (finished
with exit status / killed). 22/22 pass together with `test_ph_merge.py`.

## 3. PH merge: inspection, decision, implementation

Inspected: `merge_ph_outputs` calcfunction (per-q-point irrep concatenation, wall-time summation, incomplete
dynamical matrices without `mode_symmetry` skipped, later outputs win) and `PhBaseWorkChain.create_merged_output`
/ `_get_recover_chain` (single child returns existing outputs, `ONLY_INITIALIZATION` short-circuits, expected
range from `INPUTPH` `start_q`/`last_q`, mismatch is `ERROR_MERGING_QPOINTS`, outside-chain parents stop the walk).

Decision: **no in-graph `@task` merge is wired, and no collector core feature is requested.** A recover chain has
a dynamic length, so its members cannot be static task inputs; `orm.List` holds plain values rather than nodes,
while `DataList`/`DataDict` hold owned copies (copy-on-insert with new identities and no provenance links), so
routing outputs through a container would detach the merge inputs from their provenance instead of keeping direct
`INPUT_CALC` links. The chain is finite stored provenance after termination, so the merge runs externally:

- `get_ph_recover_chain(node)` mirrors the legacy walk over graph descendants (last `ph.x` calculation back
  through `parent_folder` creators while `recover` holds; restarts from scratch and outside parents excluded).
- `merge_ph_graph_outputs(node)` calls the **existing** `merge_ph_outputs` calcfunction with one `output_<n>`
  input per chain member, so every member `Dict` keeps a direct `INPUT_CALC` link into the merged node.
  Single-member chains and `ONLY_INITIALIZATION` runs return the existing `output_parameters` with no new node;
  q-point coverage mismatch raises `ValueError` (the legacy `401` case), refusing to merge partial chains.
- `merge_ph_output_dicts` is the pure mapping-level port for tests; it duplicates the calcfunction body on
  purpose (that module is owned elsewhere) and equivalence is tested, not assumed.

This is **verified helpers, not end-to-end**: the merge is not wired into `ph_base_graph` (owned by the monitor
worker) and the report does not claim otherwise.

## 4. Integration recommendation (coordinator)

1. Wire `sweep_graph_remote` placements (or keep the existing `pdos.py` sweeps) for `clean_workdir` on success
   and handled-failure branches; do not expect them to cover unmapped failure/cancellation.
2. After a graph terminates — success, failure, or killed — call `cleanup_graph_workdirs(node)` from the
   coordinator/CLI layer and log the returned `cleaned`/`skipped`/`failed` groups. Retry-safe by idempotency.
3. After a `ph` graph terminates, call `merge_ph_graph_outputs(node)` to obtain the merged `output_parameters`,
   and `get_ph_recover_chain(node)` if the caller needs per-iteration attribution. No `base.py` change is needed
   for either call; if the merge should later surface as a graph output, that wiring belongs to the `ph_base_graph`
   owner.

## Verification

```
uv run pytest aiida-quantumespresso/tests/graphs/test_cleanup.py aiida-quantumespresso/tests/graphs/test_ph_merge.py -q
# 22 passed
uv run --with ruff ruff check <four new files>   # clean
uv run --with ruff ruff format --check <four new files>  # clean
```

No commits made. Existing `base.py`/`pdos.py`/report files untouched; `workflows/graphs/__init__.py` deliberately
not extended (coordinator imports from the new module paths).
