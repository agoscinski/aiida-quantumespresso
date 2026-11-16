# Round 3 — native QE protocol launch helper (`graphs/protocols.py`)

> Worker handoff: interim test counts and diagnoses below are superseded by the
> integrated validation in [graph-migration.md](graph-migration.md).

Owned files: `src/aiida_quantumespresso/workflows/graphs/protocols.py`,
`tests/graphs/test_protocols.py`, this report. No changes to `graphs/base.py`,
`graphs/pw.py`, `graphs/pdos.py`, no `__init__`/CI/pyproject edits, no commits.

## 1. What was built

`get_inputs_from_protocol(kind, *, structure, ...)` is a real, plain-data
launch helper. It returns a flat mapping that splats directly into
`run_get_node`/`submit` for the matching production root:

| `kind` | production root(s) | helper output |
| --- | --- | --- |
| `'pw_relax'` | `pw_relax_graph` | `structure/parameters/init_parameters/run_init_relax/code/pseudos/kpoints_distance/kpoints_force_parity/settings/max_iterations/meta_convergence` + `task_metadata` |
| `'pw_bands'` | `pw_bands_graph` | `structure/scf_parameters/bands_parameters/scf_code/bands_code/pseudos/scf_kpoints_distance/bands_kpoints_distance/nbands_factor/settings…` + `task_metadata` |
| `'pdos'` | `pdos_from_scratch_{serial,parallel}`, `pdos_from_parent_{serial,parallel}` | `structure/pw_code/pseudos/scf_parameters?/nscf_parameters/dos_code/dos_parameters/projwfc_code/projwfc_parameters/…/clean_workdir` + `task_metadata` |

`parent_folder` selects the from-parent PDOS roots; `serial=True` selects the
serial roots (and their metadata alias prefixes). The pre-existing
`get_protocol_inputs` raw-YAML helper is unchanged.

Resolution mirrors `PwBaseWorkChain.get_builder_from_protocol` without
importing or calling it (or any `WorkChain`/`ProcessBuilder`):

* **Pseudos/cutoffs** — explicit `{kind: UpfData}` nodes shared by every
  section, or one `pseudo_family` label for all sections. Family cutoffs come
  from `RecommendedCutoffMixin.get_recommended_cutoffs`; families without
  cutoffs require both cutoffs in overrides; explicit pseudos require both
  cutoffs in overrides. The SPIN_ORBIT PseudoDojo-FR default swap is kept.
* **Thresholds** — `etot_conv_thr`/`conv_thr` scaled per atom from each
  section's `meta_parameters`.
* **K-points/settings** — distances and `settings` pass through as graph
  args; scheduler `options` lift into `task_metadata`.
* **Electron/spin** — `METAL`/`INSULATOR` handling and `get_magnetization`
  (reused from `workflows.protocols.utils`, a pure science helper) for
  `COLLINEAR`/`NON_COLLINEAR`/`SPIN_ORBIT`, including the
  `tot_magnetization` override rule.
* **Relax types** — `NONE`/`POSITIONS`/`SHAPE`/`POSITIONS_SHAPE`/
  `CELL`/`POSITIONS_CELL` map onto calculation modes, `cell_dofree`, and
  `FIXED_COORDS`, applied to both main and init parameters like the legacy
  builder. `VOLUME`/`POSITIONS_VOLUME` raise, like the legacy builder.

## 2. Fail fast, no silent drops, no over-claiming

The helper raises `ValueError` for: unknown kinds/protocols, missing codes,
unknown/legacy-only option keys anywhere in overrides (`dry_run`,
`serial_clean`, `monitors`, `parallelization`, inner `max_iterations ≠ 5`,
`bands.*.kpoints_*`, …), ORM nodes smuggled inside overrides (nodes travel
as keywords), conflicting shared relax inputs (init vs main mesh/settings),
conflicting pseudo families across sections, `AUTOMATIC` electronic type,
2D structures outside the x-y plane, malformed `energy_range_vs_fermi`, and
explicit-pseudos/family cutoff gaps. Caller overrides stay absolute-last for
parameters, exactly like the legacy builder.

Not resolved and stated as such (module docstring + §4): monitors, dry-run,
exit-code compatibility, `max_seconds` derivation, per-iteration scheduler
changes, PH/NEB/Q2R/matdyn/bands-CalcJob protocol families.

## 3. Collaborator contract: scoped `task_metadata` lookup

Current core (`src/aiida/engine/processes/graphs/tasks.py`,
`_merge_task_metadata`) looks up metadata by **leaf placement name only**
(`provided[placement]` with `placement = task.name`). Every nested
`pw_base_graph` therefore shares one `'pw_calc'` key, so SCF vs NSCF (PDOS)
and SCF vs bands (bands graph) cannot carry independent options today.

Implemented contract (core track, in-worktree as `tasks.py`/`process.py`
changes; `_submit` passes `instance=start.instance` into
`_merge_task_metadata` → `_resolve_task_metadata`), reported here for the
record:

1. The **exact hierarchical execution label** wins
   (`<scope-prefix>__<placement>`, e.g. `pw_base_graph__pw_calc`).
2. Then the longest matching **scope rule** `<scope>__*`, matched
   iteration-insensitively (`_iteration_N` suffixes normalized), so one
   entry covers a whole nested scope across loop passes and retries.
3. Then **leaf candidates**: the placement name (`pw_calc`), the instance
   leaf, and the retry-suffix-stripped leaf — preserving backward
   compatibility. Non-trailing `*` keys fail loudly.

The helper already emits both sides: leaf keys (worked before and after)
and exact hierarchical alias keys (honored by the new lookup). Extra keys
are ignored by old core, so the mapping is forward- and backward-compatible.
Callers may additionally pass `<scope>__*` rules through the helper's
`task_metadata` keyword (merged last). Where no stable hierarchical label
exists (relax: subgraphs sit inside branch/loop scopes, and exact
iteration-suffixed labels are runtime-dependent), only the leaf is emitted,
and differing per-scope options raise instead of being silently pinned to
one scope.

### Inspected graph placement names

From `compile_graph` IR dumps (`SubgraphTask` names) and observed runtime
dispatch labels (`test_launch_relax_production_graph` output):

| root | top-level subgraph placements | CalcJob alias keys emitted |
| --- | --- | --- |
| `pw_bands_graph` | `pw_base_graph` (SCF), `pw_base_graph_2` (bands) | `pw_base_graph__pw_calc`, `pw_base_graph_2__pw_calc` + leaf |
| `pdos_from_scratch_serial` | `pw_base_graph` (SCF), `pdos_stage_serial` | `pw_base_graph__pw_calc`, `pdos_stage_serial__pw_base_graph__pw_calc`, `pdos_stage_serial__dos_calc`, `pdos_stage_serial__projwfc_calc` + leaves |
| `pdos_from_scratch_parallel` | `pw_base_graph`, `pdos_stage_parallel` | same shape with `pdos_stage_parallel__` prefix |
| `pdos_from_parent_*` | `pdos_stage_parallel` / `pdos_stage_serial` | stage-prefixed keys only, no SCF alias |
| `pw_relax_graph` | inside branch/loop scopes (`init_relax_guarded`, `pw_base_graph`) | leaf `pw_calc` only |

Observed runtime labels confirm the nesting and the loop-iteration suffix:
`branch_1__init_relax_guarded__ensure_structure`,
`loop_5_iteration_0__pw_base_graph__pw_calc`,
`loop_5_iteration_0__pw_base_graph__branch_1__gen_kpoints`. Iteration-suffixed
labels are runtime-dependent, which is why relax metadata stays leaf-keyed
until the scope-prefix lookup exists.

## 4. Known limitations (not claimed as resolved)

* The relax graph shares one mesh/settings across init and main relax; the
  protocol's coarser init mesh (`0.3`) is documented and the main mesh wins.
  Explicitly conflicting overrides raise.
* The inner base restart budget is fixed at 5 (the graph default); only the
  outer meta-convergence budget is exposed.
* `DataDict` takes owned copies of the pseudo nodes (core container
  semantics): launched CalcJobs link content-identical clones, not the family
  members. Tests assert content-hash identity.
* PDOS `INSULATOR` still fails at the graph's tetrahedra validator, matching
  legacy behavior; from-parent `nbands_factor` needs
  `parent_output_parameters`, enforced by the graph task.
* Per-scope `pw` options that differ are carried under both leaf and alias
  keys; independent honoring uses the implemented exact-label/scope-rule
  lookup above. The relax root raises on differing scope options because it
  has no stable alias.

## 5. Verification

`tests/graphs/test_protocols.py` (25 tests): the 9 pre-existing raw-YAML
tests plus resolver unit tests (thresholds, cutoffs via explicit pseudos and
a fake offline `CutoffsPseudoPotentialFamily`, family-without-cutoffs and
missing-cutoffs failures, relax-type/spin/insulator mapping, all rejection
paths, caller non-mutation) and four production launches with generated
inputs — relax, bands, serial PDOS — running the real nested graphs with
only the external `CalcJob`s stubbed (stubs reused from
`test_native_workflows`, no pseudo downloads). The launches assert
threshold/cutoff propagation into dispatched CalcJob inputs, SCF/NSCF and
SCF/bands attribution, DOS-before-PROJWFC ordering, and report outcomes.
