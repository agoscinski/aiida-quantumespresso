# Round 3 independent cleanup/PH review

> Worker handoff: interim test counts and diagnoses below are superseded by the
> integrated validation in [graph-migration.md](graph-migration.md).

Independent re-review of the nested-QE owners `workflows/graphs/cleanup.py` and `workflows/graphs/ph.py`, their
tests `tests/graphs/test_cleanup.py` / `tests/graphs/test_ph_merge.py`, and the updated audit response. This
report records what was checked, what was false, and what changed. No `base.py`/`pdos.py`/`protocols.py` or core
changes were made.

## 1. False documentation fixed (no impossible-feature claims)

- **Cleanup safety (was false):** the module claimed it "takes no path argument, so arbitrary paths (including
  the `/tmp` paths …) can never be addressed". Provenance scoping is real — only descendant `remote_folder`
  outputs are candidates — but a `RemoteData` can point at `/tmp`, `/`, or anywhere else, so scoping alone never
  addressed arbitrary paths. Fixed in `cleanup.py` and `reports/round3-cleanup-ph.md`: scoping plus an explicit
  narrow guard.
- **PH containers (was false):** the module claimed "neither `orm.List` nor any other container holds node
  references". `DataList`/`DataDict` do hold owned node references; the actual problem is ownership semantics
  (copy-on-insert with new identities and no provenance links) plus detachment complexity, which would split the
  merge inputs from their provenance. Fixed in `ph.py` and `reports/round3-cleanup-ph.md`: `List` holds values,
  `DataList`/`DataDict` hold owned copies, routing through them detaches provenance. No claim is made that an
  in-graph merge is impossible or that a core collector is required.

## 2. Cleanup policy (decided, explicit)

- `clean_workdir` in-graph sweeps (`sweep_graph_remote`) cover the success path and explicitly wired
  handled-failure branches only. They do not run after unmapped failures or cancellation because dependent tasks
  never execute. Success sweeps are not automatic `on_terminated` semantics.
- The external owner must call `cleanup_graph_workdirs(node)` after termination for unexpected
  failure/cancellation paths (and may also call it after success; it is idempotent). The helper only accepts
  terminated graphs and refuses active descendants before touching anything; cleanup errors are recorded per
  folder and never change process state or exit statuses (pinned by a new outcome-unchanged test).

## 3. Cleanup guard (protects without blanket-denying `/tmp`)

- `RemoteData._clean` removes whatever its path names, so the helper applies `_unsafe_reason` per folder before
  cleaning: `/`, `/home`, and the computer's own work folder are refused as `failed` entries with the reason,
  storage untouched, while sibling folders are still cleaned. Anything else — including legitimate `/tmp`-rooted
  test remotes — is allowed.
- `sweep_graph_remote` shares the same guard through `_clean_folder` and reports `False` without cleaning when
  refused.
- Tests: `/` refused per folder with the sibling still cleaned and the refused folder not marked cleaned;
  computer-work-folder refused with storage intact; `/tmp` stand-ins still clean; unlinked decoy still untouched;
  dry-run, idempotency, running-root/active-descendant refusals, error-recording, and outcome-unchanged coverage.

## 4. PH recover chain (no silent branch choice)

- The old selector (`max(ctime)`) silently chose one branch when a graph held several disconnected `ph.x`
  branches. The helper now resolves heads (members no other member recovered from): one head is walked as
  before, including restart-from-scratch exclusion; several heads raise `ValueError` naming the branch-end pks
  instead of choosing. `get_ph_recover_chain(node, last_calc=…)` and
  `merge_ph_graph_outputs(node, last_calc=…)` accept an explicit member node or pk to select a branch; an unknown
  selector raises a meaningful error.
- External merge on a terminated failed graph missing `output_parameters` raises `ValueError` naming the member
  instead of `AttributeError` or a partial merge; q-point coverage mismatch still raises. The merge returns the
  merged `Dict` only — single-member and `ONLY_INITIALIZATION` short-circuits return the existing outputs —
  with no claim of full graph-result parity.

## Verification

```
uv run pytest aiida-quantumespresso/tests/graphs/test_cleanup.py aiida-quantumespresso/tests/graphs/test_ph_merge.py -q
uv run --with ruff ruff check aiida-quantumespresso/src/aiida_quantumespresso/workflows/graphs/cleanup.py aiida-quantumespresso/src/aiida_quantumespresso/workflows/graphs/ph.py aiida-quantumespresso/tests/graphs/test_cleanup.py aiida-quantumespresso/tests/graphs/test_ph_merge.py
uv run --with ruff ruff format --check <same four files>
```

No commits made. Existing `base.py`/`pdos.py`/`protocols.py` and core untouched.
