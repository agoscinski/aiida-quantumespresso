############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Native ``@graph`` orchestration for PW relax and PW bands.

The graphs place the native :func:`pw_base_graph
<aiida_quantumespresso.workflows.graphs.base.pw_base_graph>` subgraph for every
``pw.x`` run. No ``WorkChain`` is placed or imported here. Science stays where it
was: restarts and error handling inside ``pw_base_graph``, k-point generation
inside ``create_kpoints_from_distance``, path generation inside
``seekpath_structure_analysis``. The graphs only wire names, branch on explicit
predicates, and loop with inline counters; every ``@task`` body delegates to the
existing implementations.

Run the checks with the pinned experimental core and QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_pw.py -q

and validate without launching anything with ``assert pw_relax_graph.build() is None``.

Scheduler options
-----------------

``CalcJob`` scheduler options travel at launch under ``task_metadata`` as a
strictly nested mapping, not through graph inputs. Each authored subgraph
callee alias adds one level; the ``CalcJob`` placement alias holds the normal
metadata mapping at the leaf. Branch and loop control frames are transparent:
iterations reuse their placement's configuration and add no key level. The
relax scopes are the initial wrapper plus its inner alias and the main-loop
alias; the bands scopes are ``scf`` and ``bands``. Each ``pw`` scope carries
both static placements (``pw_calc`` unmonitored plus ``pw_calc_monitored``)
with identical scheduler options::

    task_metadata = {
        'init_relax_guarded': {'initial': {'pw_calc': {'options': init_options},
                                           'pw_calc_monitored': {'options': init_options}}},
        'main': {'pw_calc': {'options': main_options},
                 'pw_calc_monitored': {'options': main_options}},
    }
    task_metadata = {
        'scf': {'pw_calc': {'options': scf_options},
                'pw_calc_monitored': {'options': scf_options}},
        'bands': {'pw_calc': {'options': bands_options},
                  'pw_calc_monitored': {'options': bands_options}},
    }

The experimental ``options`` ports of the previous revision were removed rather
than silently ignored: passing scheduler options through a graph ``Dict``
input would suggest per-iteration control that the native base graph does not
offer (its options are constant across iterations). Form the launch mapping
with :func:`task_metadata_for_options
<aiida_quantumespresso.workflows.graphs.protocols.task_metadata_for_options>`
or as a literal nested mapping; unknown paths, missing entries, and
non-mapping leaves fail loudly at launch.

Honest limits of this rewrite (see ``reports/STATUS.md``):

* ``pw_base_graph`` supplies the reusable restart loop; its
  contract is ``(code, structure, pseudos: DataDict, parameters, settings?,
  kpoints?, kpoints_distance?, kpoints_force_parity?, parent_folder?)`` returning
  ``output_parameters``/``output_structure?``/``output_band?``/``output_trajectory?``/
  ``remote_folder``/``report``. Pseudopotentials travel as a ``DataDict`` (kind name to ``UpfData``).
  Each composite below calls it only through the module-level aliases
  (``scf``/``bands``/``initial``/``main``), each containing the base's dual
  ``pw_calc``/``pw_calc_monitored`` placements (identical scheduler options on
  both leaves). Considered-finished base outcomes (``501``
  ``finished_ionic_convergence_reached_except_final_scf`` and ``710``
  ``finished_electronic_convergence_warning``) stay usable downstream: the
  runtime keeps the failed run's stored outputs alongside the synthesized error.
* Bands outcome (implemented report statuses, not exact numeric root parity):
  both the SCF and the bands base runs must finish with report
  status ``'finished'``. A considered-finished ``501``/``710`` outcome fails the
  graph via :func:`require_base_finished`, mirroring the legacy
  ``inspect_scf``/``inspect_bands`` ``is_finished_ok`` gates.
* Relax outcome (implemented report statuses, not exact numeric root parity): a ``501`` base outcome is accepted and reruns only when the legacy
  Pulay-stress pressure criterion requires it (``verify_convergence_stress`` on
  the step ``output_trajectory`` with ``CELL.press_conv_thr * 1.1`` tolerance
  against ``CELL.press``); a ``710`` outcome is rejected. This mirrors
  ``PwRelaxWorkChain.should_run_relax``/``inspect_relax``.
* Graphs declare no exit codes: legacy statuses (relax ``400/401/402/403``,
  bands ``201/202/402/403``) collapse into the single graph ``400
  ERROR_TASK_FAILED``. Outcomes are reported through the ``report`` output.
* Input validators moved into ``@task`` bodies and now surface as task failures,
  not launch-time ``ValidationError``s. There is no ``clean_workdir`` handling,
  no ``dry_run``. ``monitors`` is an optional shared ``DataDict`` forwarded into
  every nested base run through its authored alias (``initial`` for the initial
  relaxation, ``main`` for the main loop, ``scf``/``bands`` for the bands
  stages); per-stage overrides fall back to the shared mapping.
* ``scf``-mode relax echoes the input structure as ``output_structure`` where the
  legacy chain emits none.
* The ``initial``/``main`` aliases keep the two runs' configuration explicit. Each stage carries its own
  parameters, settings, k-points (explicit or density plus parity flag), and monitors; when
  ``init_use_main_inputs`` holds (the default, preserving direct launches) unset ``init_*`` graph
  inputs fall back to the matching main value, while the protocol helper launches with
  ``init_use_main_inputs=False`` so each stage carries its own resolved values with no
  inheritance (an absent ``init_*`` stays absent); the two stages' scheduler
  options likewise resolve independently per scope (``initial`` versus ``main``).
  Repeated calls in the same authored scope keep the core's generated names
  (such as ``pw_calc_2``); there is no distinct-alias requirement. Loop and
  branch control frames do not add configuration levels.
"""

import copy
import typing as t

from aiida import orm
from aiida.engine import PortModel, graph, task
from aiida.orm import BandsData, Code, DataDict, Dict, Float, KpointsData, RemoteData, StructureData, TrajectoryData

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance as _create_kpoints_calcfunction,
)
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis as _seekpath_calcfunction,
)
from aiida_quantumespresso.workflows.graphs.base import pw_base_graph
from aiida_quantumespresso.utils.validation.trajectory import verify_convergence_stress

# Authored callee aliases: repeated calls to the same subgraph are told apart by
# the name used at the call site, so each metadata-bearing placement gets its own
# module-level alias. The structured ``task_metadata`` descends one level per
# alias (``{'scf': {'pw_calc': ...}}``); loop iterations and branch arms share
# their placement's configuration and add no level of their own.
scf = pw_base_graph
bands = pw_base_graph
initial = pw_base_graph
main = pw_base_graph

# Undecorated twins of the ``calcfunction`` implementations above. Graph tasks run as calculation processes and so
# cannot call a decorated ``calcfunction`` (the engine forbids it); calling the shared undecorated implementation
# keeps the science identical without duplicating it and without adding nested provenance nodes.
_create_kpoints_from_distance = _create_kpoints_calcfunction.__wrapped__
_seekpath_structure_analysis = _seekpath_calcfunction.__wrapped__

__all__ = (
    'BandPath',
    'BandsOutputs',
    'InitInputs',
    'RelaxCarry',
    'RelaxNext',
    'RelaxOutputs',
    'ensure_structure',
    'init_relax_guarded',
    'prepare_kpoints_path',
    'prepare_next_relax',
    'pw_bands_graph',
    'pw_relax_graph',
    'require_base_finished',
    'require_relax_base',
    'resolve_bands_parameters',
    'resolve_init_inputs',
    'seed_relax_carry',
    'summarize_relax',
)


class RelaxNext(t.TypedDict):
    """Decision of :func:`prepare_next_relax`: rerun flag plus parameters for the next iteration."""

    rerun: bool
    parameters: Dict


class BandPath(PortModel):
    """K-point path of :func:`prepare_kpoints_path`, with unset leaves left out of the outputs."""

    primitive_structure: StructureData | None = None
    explicit_kpoints: KpointsData
    seekpath_parameters: Dict | None = None


class RelaxOutputs(t.TypedDict):
    """Outputs of :func:`pw_relax_graph`: the final structure and summary plus the final base run's outputs."""

    output_structure: StructureData
    output_parameters: Dict | None = None
    output_trajectory: TrajectoryData | None = None
    output_band: BandsData | None = None
    output_kpoints: KpointsData | None = None
    output_atomic_occupations: Dict | None = None
    remote_folder: RemoteData | None = None
    report: Dict


class RelaxCarry(PortModel):
    """Nullable seeds for the relax loop's final-output carry, all leaves absent."""

    output_parameters: Dict | None = None
    output_trajectory: TrajectoryData | None = None
    output_band: BandsData | None = None
    output_kpoints: KpointsData | None = None
    output_atomic_occupations: Dict | None = None
    remote_folder: RemoteData | None = None


class BandsOutputs(t.TypedDict):
    """Outputs of :func:`pw_bands_graph`; ``None`` leaves stay out of the emitted outputs."""

    band_structure: BandsData
    scf_parameters: Dict
    band_parameters: Dict
    primitive_structure: StructureData | None
    seekpath_parameters: Dict | None


class InitInputs(PortModel):
    """Effective initial-relax inputs resolved by :func:`resolve_init_inputs`, unset leaves left out."""

    parameters: Dict
    settings: Dict | None = None
    kpoints: KpointsData | None = None
    kpoints_distance: Float | None = None
    force_parity: bool
    monitors: DataDict | None = None


@task
def ensure_structure(candidate: StructureData | None) -> StructureData:
    """Return the candidate structure, failing loudly when the step produced none.

    A task cannot return its input (that would link the same node as both input and output of the task), so the
    structure is returned as a value-copy.

    :param candidate: the ``output_structure`` of a base run, possibly absent.
    :return: a copy of the structure, guaranteed present.
    :raises ValueError: if the step finished without an output structure.
    """
    if candidate is None:
        msg = 'The base run finished without an output structure.'
        raise ValueError(msg)
    return t.cast(StructureData, candidate.clone())


@task
def require_base_finished(report: Dict, stage: str) -> None:
    """Fail when a base run the bands graph depends on did not fully converge.

    Mirrors ``PwBandsWorkChain.inspect_scf``/``inspect_bands``: both require
    ``is_finished_ok``, so a considered-finished ``501``
    (``finished_ionic_convergence_reached_except_final_scf``) or ``710``
    (``finished_electronic_convergence_warning``) outcome must fail the graph
    instead of flowing into the band-count resolution or the band structure.

    :param report: the ``report`` of the base run.
    :param stage: ``'scf'`` or ``'bands'``, naming the failed stage.
    :raises ValueError: if the report status is not ``'finished'``.
    """
    status_data = report.get_dict()
    if 'status' not in status_data:
        msg = f"The {stage} base run report carries no 'status'; refusing to use it."
        raise ValueError(msg)
    status = status_data['status']
    if status != 'finished':
        msg = f"The {stage} base run finished with status '{status}'; the bands graph requires a fully converged run."
        raise ValueError(msg)
    return None


@task
def require_relax_base(report: Dict | None = None) -> None:
    """Fail when an initial-relax base run reports an unusable outcome.

    Mirrors ``PwRelaxWorkChain.inspect_init_relax``: ``501``
    (``finished_ionic_convergence_reached_except_final_scf``) is acceptable and
    flows into the output structure, while ``710``
    (``finished_electronic_convergence_warning``) must fail. The main loop needs
    no separate guard because :func:`prepare_next_relax` already rejects ``710``.

    :param report: the ``report`` of the base run, if threaded.
    :raises ValueError: if the report status is ``710`` or otherwise unexpected.
    """
    if report is None:
        return None
    status_data = report.get_dict()
    if 'status' not in status_data:
        msg = "The initial-relax base run report carries no 'status'; refusing to use it."
        raise ValueError(msg)
    status = status_data['status']
    if status in ('finished', 'finished_ionic_convergence_reached_except_final_scf'):
        return None
    if status == 'finished_electronic_convergence_warning':
        msg = 'The initial-relax base run reported electronic-convergence warning; rejecting it.'
        raise ValueError(msg)
    msg = f"Unexpected relax base report status '{status}'."
    raise ValueError(msg)


@task
def prepare_next_relax(
    parameters: Dict,
    output_parameters: Dict,
    structure_before: StructureData,
    structure_after: StructureData | None,
    kpoints_distance: Float | None,
    force_parity: bool,
    meta_convergence: bool,
    report: Dict | None = None,
    output_trajectory: TrajectoryData | None = None,
) -> RelaxNext:
    """Decide whether another relax iteration is needed and carry the parameters forward.

    Mirrors ``PwRelaxWorkChain.should_run_relax``: the number of bands of the finished run is
    carried into ``SYSTEM.nbnd``. A ``501`` (``finished_ionic_convergence_reached_except_final_scf``)
    outcome is accepted and reruns only when the legacy Pulay-stress pressure criterion requires it
    (``verify_convergence_stress`` on the step ``output_trajectory`` with a 10% tolerance over
    ``CELL.press_conv_thr`` against ``CELL.press``); a ``710``
    (``finished_electronic_convergence_warning``) outcome is rejected. Otherwise, when
    meta-convergence is on and the cell may still change volume, the k-point
    meshes before and after are regenerated with the undecorated twin of :func:`create_kpoints_from_distance
    <aiida_quantumespresso.calculations.functions.create_kpoints_from_distance.create_kpoints_from_distance>`; a
    grown mesh requests another iteration.

    :param parameters: the parameters the finished iteration ran with.
    :param output_parameters: the ``output_parameters`` of the finished base run.
    :param structure_before: the structure the finished iteration started from.
    :param structure_after: the ``output_structure`` of the finished iteration, if any.
    :param kpoints_distance: the mesh density, or ``None`` when k-points are explicit.
    :param force_parity: whether mesh generation preserves parity.
    :param meta_convergence: whether mesh-regrowth reruns are enabled at all.
    :param report: the ``report`` of the finished base run, if threaded.
    :param output_trajectory: the ``output_trajectory`` of the finished base run, needed for the ``501`` check.
    :return: whether to rerun and the parameters for the next iteration.
    :raises ValueError: if a relaxing calculation finished without an output structure,
        if a ``710`` outcome arrives, if a ``501`` outcome arrives without a trajectory,
        or if a provided report carries no ``'status'``.
    """
    if report is None:
        status = 'finished'
    else:
        status_data = report.get_dict()
        if 'status' not in status_data:
            msg = "The relax base run report carries no 'status'; refusing to use it."
            raise ValueError(msg)
        status = status_data['status']
    if status == 'finished_electronic_convergence_warning':
        msg = 'The relax base run reported electronic-convergence warning; the relax graph rejects it.'
        raise ValueError(msg)
    if status not in ('finished', 'finished_ionic_convergence_reached_except_final_scf'):
        msg = f"Unexpected relax base report status '{status}'."
        raise ValueError(msg)
    params = copy.deepcopy(parameters.get_dict())
    calculation = params.get('CONTROL', {}).get('calculation', 'scf')

    if structure_after is None:
        if calculation == 'scf' and status == 'finished':
            return {'rerun': False, 'parameters': Dict(params)}
        msg = 'A `relax`/`vc-relax` base run finished successfully but without an output structure.'
        raise ValueError(msg)

    params.setdefault('SYSTEM', {})['nbnd'] = output_parameters.get_dict()['number_of_bands']

    if status == 'finished_ionic_convergence_reached_except_final_scf':
        if not meta_convergence:
            return {'rerun': False, 'parameters': Dict(params)}
        if output_trajectory is None:
            msg = 'A `501` base run finished without an output trajectory for the Pulay-stress check.'
            raise ValueError(msg)
        threshold = params.get('CELL', {}).get('press_conv_thr', 0.5) * 1.1
        reference = params.get('CELL', {}).get('press', 0.0)
        if verify_convergence_stress(trajectory=output_trajectory, threshold=threshold, reference_pressure=reference):
            return {'rerun': False, 'parameters': Dict(params)}
        return {'rerun': True, 'parameters': Dict(params)}

    rerun = False
    cell_dofree = params.get('CELL', {}).get('cell_dofree')
    volume_cannot_change = calculation in ('scf', 'relax') or cell_dofree == 'shape'
    if meta_convergence and not volume_cannot_change and kpoints_distance is not None:
        parity = orm.Bool(force_parity)
        old_mesh, _ = _create_kpoints_from_distance(structure_before, kpoints_distance, parity).get_kpoints_mesh()
        new_mesh, _ = _create_kpoints_from_distance(structure_after, kpoints_distance, parity).get_kpoints_mesh()
        rerun = any(new > old for old, new in zip(old_mesh, new_mesh))
    return {'rerun': rerun, 'parameters': Dict(params)}


@task
def resolve_init_inputs(
    parameters: Dict,
    settings: Dict | None,
    kpoints: KpointsData | None,
    kpoints_distance: Float | None,
    force_parity: bool,
    monitors: DataDict | None,
    init_use_main_inputs: bool,
    init_parameters: Dict | None,
    init_settings: Dict | None,
    init_kpoints: KpointsData | None,
    init_kpoints_distance: Float | None,
    init_kpoints_force_parity: bool | None,
    init_monitors: DataDict | None,
) -> InitInputs:
    """Resolve the effective initial-relax inputs for one ``init_use_main_inputs`` mode.

    With ``init_use_main_inputs`` each unset ``init_*`` falls back to the matching main value, preserving direct
    launches; without it every ``init_*`` is used verbatim with no inheritance (an absent ``init_settings``,
    ``init_kpoints``, or ``init_monitors`` stays absent, an absent ``init_kpoints_distance`` means no density,
    and an absent parity flag resolves to ``False``). Returned nodes are value-copies: a task cannot return its
    input, so each resolved node is rebuilt rather than passed through.

    :param parameters: the ``pw.parameters`` of the main relax loop.
    :param settings: the ``pw.settings`` of the main relax loop, if any.
    :param kpoints: explicit k-points of the main relax loop, if any.
    :param kpoints_distance: mesh density of the main relax loop, if any.
    :param force_parity: parity flag of the main relax loop.
    :param monitors: monitors of the main relax loop, if any.
    :param init_use_main_inputs: whether unset ``init_*`` fall back to the matching main value.
    :param init_parameters: the requested initial parameters, if any.
    :param init_settings: the requested initial settings, if any.
    :param init_kpoints: explicit k-points of the initial relaxation, if any.
    :param init_kpoints_distance: mesh density of the initial relaxation, if any.
    :param init_kpoints_force_parity: parity flag of the initial relaxation, if any.
    :param init_monitors: monitors of the initial relaxation, if any.
    :return: the effective initial-relax inputs.
    :raises ValueError: if ``init_use_main_inputs`` is ``False`` and ``init_parameters`` is absent.
    """
    if init_use_main_inputs:
        eff_parameters = parameters if init_parameters is None else init_parameters
        eff_settings = settings if init_settings is None else init_settings
        eff_kpoints = kpoints if init_kpoints is None else init_kpoints
        eff_distance = kpoints_distance if init_kpoints_distance is None else init_kpoints_distance
        eff_parity = force_parity if init_kpoints_force_parity is None else init_kpoints_force_parity
        eff_monitors = monitors if init_monitors is None else init_monitors
    else:
        if init_parameters is None:
            msg = '`init_parameters` is required when `init_use_main_inputs=False`.'
            raise ValueError(msg)
        eff_parameters = init_parameters
        eff_settings = init_settings
        eff_kpoints = init_kpoints
        eff_distance = init_kpoints_distance
        eff_parity = False if init_kpoints_force_parity is None else init_kpoints_force_parity
        eff_monitors = init_monitors
    return InitInputs(
        parameters=Dict(eff_parameters.get_dict()),
        settings=Dict(eff_settings.get_dict()) if eff_settings is not None else None,
        kpoints=t.cast(KpointsData, eff_kpoints.clone()) if eff_kpoints is not None else None,
        kpoints_distance=Float(eff_distance.value) if eff_distance is not None else None,
        force_parity=bool(eff_parity),
        monitors=DataDict(dict(eff_monitors.items())) if eff_monitors is not None else None,
    )


@task
def seed_relax_carry() -> RelaxCarry:
    """Seed the relax loop's final-output carry with absent leaves.

    Loop-carried wires take their shape from the pre-loop binding, and a ``None`` literal seeds a
    ``None``-only wire that rejects the node rebound inside the loop (and vice versa for a node-first
    merge). Seeding from this task's omitted leaves gives ``None`` values with union shapes, so the
    ``is None`` keep arm first preserves nullability exactly like the ``parent`` carry in
    ``pw_base_graph``.

    :return: the empty carry whose leaves stay out of the emitted outputs.
    """
    return RelaxCarry()


@task
def prepare_kpoints_path(
    structure: StructureData,
    bands_kpoints: KpointsData | None,
    bands_kpoints_distance: Float | None,
) -> BandPath:
    """Resolve the structure and k-point path for the bands step.

    With explicit ``bands_kpoints`` the input structure passes through and the SeekPath leaves stay empty, matching
    the legacy ``should_run_seekpath`` branch. A task cannot return its input, so the explicit k-points travel as
    a value-copy. Otherwise the structure is normalized with the undecorated twin of
    :func:`seekpath_structure_analysis
    <aiida_quantumespresso.calculations.functions.seekpath_structure_analysis.seekpath_structure_analysis>`.

    :param structure: the input structure.
    :param bands_kpoints: explicit k-points for the bands step, if given.
    :param bands_kpoints_distance: minimum k-point distance for the SeekPath path, if given.
    :return: the primitive structure (or ``None``), the explicit k-points, and the SeekPath parameters (or ``None``).
    :raises ValueError: if both ``bands_kpoints`` and ``bands_kpoints_distance`` are given.
    """
    if bands_kpoints is not None and bands_kpoints_distance is not None:
        msg = 'Cannot specify both `bands_kpoints` and `bands_kpoints_distance`.'
        raise ValueError(msg)
    if bands_kpoints is not None:
        return BandPath(explicit_kpoints=t.cast(KpointsData, bands_kpoints.clone()))
    seekpath_kwargs: dict[str, Float] = {}
    if bands_kpoints_distance is not None:
        seekpath_kwargs['reference_distance'] = bands_kpoints_distance
    result = _seekpath_structure_analysis(structure, **seekpath_kwargs)
    return BandPath(
        primitive_structure=result['primitive_structure'],
        explicit_kpoints=result['explicit_kpoints'],
        seekpath_parameters=result['parameters'],
    )


@task
def resolve_bands_parameters(
    scf_output_parameters: Dict,
    bands_parameters: Dict,
    nbands_factor: Float | None,
) -> Dict:
    """Resolve the ``pw.parameters`` for the bands step, applying the ``nbands_factor`` rule.

    Mirrors ``PwBandsWorkChain.run_bands``: with ``nbands_factor`` the band count is the maximum of the SCF band
    count, ``0.5 * nelectrons * nbands_factor``, and ``0.5 * nelectrons + 4``; otherwise the SCF band count fills in
    an unset ``SYSTEM.nbnd``.

    :param scf_output_parameters: the ``output_parameters`` of the SCF base run.
    :param bands_parameters: the requested bands parameters.
    :param nbands_factor: the multiplier for the bands count, if given.
    :return: the resolved bands parameters.
    :raises ValueError: if both ``nbands_factor`` and ``SYSTEM.nbnd`` are specified.
    """
    params = copy.deepcopy(bands_parameters.get_dict())
    params.setdefault('CONTROL', {})
    params.setdefault('SYSTEM', {})
    params.setdefault('ELECTRONS', {})
    scf = scf_output_parameters.get_dict()

    if nbands_factor is not None:
        if 'nbnd' in params['SYSTEM']:
            msg = 'Cannot specify both `nbands_factor` and `bands.pw.parameters.SYSTEM.nbnd`.'
            raise ValueError(msg)
        factor = nbands_factor.value
        electrons_per_band = 1 if params['SYSTEM'].get('noncolin', False) else 2
        nbands = int(scf['number_of_bands'])
        nelectron = int(scf['number_of_electrons'])
        params['SYSTEM']['nbnd'] = max(
            int(nelectron / electrons_per_band * factor), int(nelectron / electrons_per_band) + 4, nbands
        )
    else:
        params['SYSTEM'].setdefault('nbnd', int(scf['number_of_bands']))
    return Dict(params)


@task
def summarize_relax(structure: StructureData, attempts: int, status: str, max_iterations: int) -> dict[str, str | int]:
    """Record the final structure identity, attempt count, and convergence status as one report.

    :param structure: the final structure of the relax graph.
    :param attempts: how many relax iterations ran.
    :param status: ``'converged'`` or ``'exhausted'``.
    :param max_iterations: the iteration budget the graph ran with.
    :return: the summary mapping stored as a single ``Dict``.
    """
    return {
        'structure_uuid': structure.uuid,
        'attempts': attempts,
        'status': status,
        'max_iterations': max_iterations,
    }


@graph
def init_relax_guarded(
    structure: StructureData,
    parameters: Dict,
    code: Code,
    pseudos: DataDict,
    kpoints: KpointsData | None = None,
    kpoints_distance: Float | None = None,
    kpoints_force_parity: bool = False,
    settings: Dict | None = None,
    parent_folder: RemoteData | None = None,
    monitors: DataDict | None = None,
    init_use_main_inputs: bool = True,
    init_parameters: Dict | None = None,
    init_settings: Dict | None = None,
    init_kpoints: KpointsData | None = None,
    init_kpoints_distance: Float | None = None,
    init_kpoints_force_parity: bool | None = None,
    init_monitors: DataDict | None = None,
) -> StructureData:
    """Run one initial relaxation and guarantee its output structure.

    The ``parameters``/``settings``/``kpoints``/``kpoints_distance``/``kpoints_force_parity``/``monitors``
    inputs carry the main-loop values; the ``init_*`` inputs override them per :func:`resolve_init_inputs`.
    With the default ``init_use_main_inputs=True`` each unset ``init_*`` falls back to the matching main
    value, so direct launches passing only the main values behave exactly as before; the protocol helper
    launches with ``init_use_main_inputs=False`` and passes each stage's independently resolved values with
    no inheritance.

    :param structure: the structure to relax.
    :param parameters: the main ``pw.parameters`` the initial relaxation falls back to.
    :param code: the ``quantumespresso.pw`` code.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param kpoints: explicit main k-points the initial relaxation falls back to, if any.
    :param kpoints_distance: main mesh density the initial relaxation falls back to, if any.
    :param kpoints_force_parity: main parity flag the initial relaxation falls back to.
    :param settings: the main ``pw.settings`` the initial relaxation falls back to, if any.
    :param parent_folder: the ``pw.parent_folder``, if any.
    :param monitors: main monitors the initial relaxation falls back to, if any.
    :param init_use_main_inputs: when ``True`` (default) unset ``init_*`` fall back to the matching main
        value; when ``False`` each ``init_*`` is used verbatim with no inheritance.
    :param init_parameters: the parameters of the initial relaxation, defaulting to ``parameters``.
    :param init_settings: the ``pw.settings`` of the initial relaxation, defaulting to ``settings``.
    :param init_kpoints: explicit k-points of the initial relaxation, defaulting to ``kpoints``.
    :param init_kpoints_distance: mesh density of the initial relaxation, defaulting to ``kpoints_distance``.
    :param init_kpoints_force_parity: parity flag of the initial relaxation, defaulting to
        ``kpoints_force_parity``.
    :param init_monitors: monitors of the initial relaxation, defaulting to ``monitors``.
    :return: the relaxed structure.
    """
    resolved = resolve_init_inputs(
        parameters=parameters,
        settings=settings,
        kpoints=kpoints,
        kpoints_distance=kpoints_distance,
        force_parity=kpoints_force_parity,
        monitors=monitors,
        init_use_main_inputs=init_use_main_inputs,
        init_parameters=init_parameters,
        init_settings=init_settings,
        init_kpoints=init_kpoints,
        init_kpoints_distance=init_kpoints_distance,
        init_kpoints_force_parity=init_kpoints_force_parity,
        init_monitors=init_monitors,
    )
    base = initial(
        code=code,
        structure=structure,
        pseudos=pseudos,
        parameters=resolved.parameters,
        settings=resolved.settings,
        kpoints=resolved.kpoints,
        kpoints_distance=resolved.kpoints_distance,
        kpoints_force_parity=resolved.force_parity,
        parent_folder=parent_folder,
        monitors=resolved.monitors,
    )
    checked = require_relax_base(report=base.report)
    return ensure_structure(candidate=base.output_structure, after=checked)


@graph
def pw_relax_graph(
    structure: StructureData,
    parameters: Dict,
    code: Code,
    pseudos: DataDict,
    max_iterations: int = 5,
    meta_convergence: bool = True,
    kpoints: KpointsData | None = None,
    kpoints_distance: Float | None = None,
    kpoints_force_parity: bool = False,
    settings: Dict | None = None,
    parent_folder: RemoteData | None = None,
    run_init_relax: bool = False,
    init_use_main_inputs: bool = True,
    init_parameters: Dict | None = None,
    init_settings: Dict | None = None,
    init_kpoints: KpointsData | None = None,
    init_kpoints_distance: Float | None = None,
    init_kpoints_force_parity: bool | None = None,
    monitors: DataDict | None = None,
    init_monitors: DataDict | None = None,
) -> RelaxOutputs:
    """Relax a structure through repeated native base runs, then report.

    An optional initial relaxation runs first when ``run_init_relax`` holds (it shares code, pseudopotentials,
    and parent folder with the main loop). With the default ``init_use_main_inputs=True`` the ``init_*`` inputs
    (``init_parameters``, ``init_settings``, ``init_kpoints``, ``init_kpoints_distance``,
    ``init_kpoints_force_parity``, ``init_monitors``) each default to the matching main value when unset, so
    direct launches stay unchanged. The protocol helper launches with ``init_use_main_inputs=False`` and passes
    each stage's independently resolved values, so no inheritance applies and an absent ``init_*`` stays absent
    (absent ``init_settings``/``init_kpoints``/``init_monitors`` mean none, absent ``init_kpoints_distance``
    means no density, and an absent parity flag resolves to ``False`` rather than the main flag). The main loop reruns while the mesh-regrowth predicate holds and the inline attempt budget
    lasts, then a summary task records the outcome. The root returns the existing final structure, the summary
    node, and the final base run's outputs (``output_parameters``, ``remote_folder``, ``output_trajectory``,
    ``output_band``, ``output_kpoints``, ``output_atomic_occupations``); unset leaves stay out of the emitted
    outputs when the loop never runs. Scheduler options ride at launch under ``task_metadata`` (see the module docstring); there is no
    per-iteration ``options`` port.

    :param structure: the structure to relax.
    :param parameters: the ``pw.parameters`` for the main relax loop.
    :param code: the ``quantumespresso.pw`` code.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param max_iterations: the meta-convergence iteration budget; at least one iteration always runs when positive.
    :param meta_convergence: when ``False``, every iteration is final and the loop still honors the budget.
    :param kpoints: explicit k-points, if any.
    :param kpoints_distance: mesh density regenerated from each new structure, if any.
    :param kpoints_force_parity: whether generated meshes preserve parity.
    :param settings: the ``pw.settings`` shared by every run, if any.
    :param parent_folder: the ``pw.parent_folder`` shared by every run, if any.
    :param run_init_relax: whether to run the initial relaxation.
    :param init_use_main_inputs: when ``True`` (default) unset ``init_*`` inputs fall back to the matching
        main value; when ``False`` each ``init_*`` is used verbatim with no inheritance.
    :param init_parameters: the parameters of the initial relaxation, defaulting to ``parameters``.
    :param init_settings: the ``pw.settings`` of the initial relaxation, defaulting to ``settings``.
    :param init_kpoints: explicit k-points of the initial relaxation, defaulting to ``kpoints``.
    :param init_kpoints_distance: mesh density of the initial relaxation, defaulting to ``kpoints_distance``.
    :param init_kpoints_force_parity: parity flag of the initial relaxation, defaulting to ``kpoints_force_parity``.
    :param monitors: optional shared monitors forwarded into every nested base run.
    :param init_monitors: optional initial-relax monitors, defaulting to ``monitors``.
    :return: the relaxed structure, the summary report, and the final base run's outputs: each
        ``is not None``-carried wire keeps the existing node identity, and unset leaves stay out of
        the emitted outputs when the loop never runs.
    """
    current = structure
    if run_init_relax:
        current = init_relax_guarded(
            structure=current,
            parameters=parameters,
            code=code,
            pseudos=pseudos,
            kpoints=kpoints,
            kpoints_distance=kpoints_distance,
            kpoints_force_parity=kpoints_force_parity,
            settings=settings,
            parent_folder=parent_folder,
            monitors=monitors,
            init_use_main_inputs=init_use_main_inputs,
            init_parameters=init_parameters,
            init_settings=init_settings,
            init_kpoints=init_kpoints,
            init_kpoints_distance=init_kpoints_distance,
            init_kpoints_force_parity=init_kpoints_force_parity,
            init_monitors=init_monitors,
        )
    params = parameters
    attempts = 0
    rerun = True
    seed = seed_relax_carry()
    final_output_parameters = seed.output_parameters
    final_output_trajectory = seed.output_trajectory
    final_output_band = seed.output_band
    final_output_kpoints = seed.output_kpoints
    final_output_atomic_occupations = seed.output_atomic_occupations
    final_remote_folder = seed.remote_folder
    while rerun and attempts < max_iterations:
        step = main(
            code=code,
            structure=current,
            pseudos=pseudos,
            parameters=params,
            settings=settings,
            kpoints=kpoints,
            kpoints_distance=kpoints_distance,
            kpoints_force_parity=kpoints_force_parity,
            parent_folder=parent_folder,
            monitors=monitors,
        )
        decision = prepare_next_relax(
            parameters=params,
            output_parameters=step.output_parameters,
            structure_before=current,
            structure_after=step.output_structure,
            kpoints_distance=kpoints_distance,
            force_parity=kpoints_force_parity,
            meta_convergence=meta_convergence,
            report=step.report,
            output_trajectory=step.output_trajectory,
        )
        if step.output_structure is not None:
            current = step.output_structure
        if step.output_parameters is None:
            final_output_parameters = final_output_parameters  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        else:
            final_output_parameters = step.output_parameters
        if step.output_trajectory is None:
            final_output_trajectory = final_output_trajectory  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        else:
            final_output_trajectory = step.output_trajectory
        if step.output_band is None:
            final_output_band = final_output_band  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        else:
            final_output_band = step.output_band
        if step.output_kpoints is None:
            final_output_kpoints = final_output_kpoints  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        else:
            final_output_kpoints = step.output_kpoints
        if step.output_atomic_occupations is None:
            final_output_atomic_occupations = final_output_atomic_occupations  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        else:
            final_output_atomic_occupations = step.output_atomic_occupations
        if step.remote_folder is None:
            final_remote_folder = final_remote_folder  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        else:
            final_remote_folder = step.remote_folder
        params = decision.parameters
        rerun = decision.rerun
        attempts = attempts + 1
    if rerun:
        status = 'exhausted'
    else:
        status = 'converged'
    report = summarize_relax(structure=current, attempts=attempts, status=status, max_iterations=max_iterations)
    return {
        'output_structure': current,
        'output_parameters': final_output_parameters,
        'output_trajectory': final_output_trajectory,
        'output_band': final_output_band,
        'output_kpoints': final_output_kpoints,
        'output_atomic_occupations': final_output_atomic_occupations,
        'remote_folder': final_remote_folder,
        'report': report,
    }


@graph
def pw_bands_graph(
    structure: StructureData,
    scf_parameters: Dict,
    bands_parameters: Dict,
    scf_code: Code,
    bands_code: Code,
    pseudos: DataDict,
    scf_kpoints: KpointsData | None = None,
    scf_kpoints_distance: Float | None = None,
    bands_kpoints: KpointsData | None = None,
    bands_kpoints_distance: Float | None = None,
    nbands_factor: Float | None = None,
    scf_settings: Dict | None = None,
    bands_settings: Dict | None = None,
    monitors: DataDict | None = None,
    scf_monitors: DataDict | None = None,
    bands_monitors: DataDict | None = None,
) -> BandsOutputs:
    """Compute a band structure along a SeekPath path (or explicit k-points) through SCF plus bands runs.

    The path adapter normalizes the structure unless explicit bands k-points are given; the SCF run feeds its remote
    folder and band count into the bands run. Scheduler options ride at launch under ``task_metadata`` (see the
    module docstring); there are no per-run ``options`` ports. The root returns the existing step outputs, leaving
    the SeekPath leaves out when explicit k-points were given.

    :param structure: the structure to compute the bands of.
    :param scf_parameters: the ``pw.parameters`` of the SCF run.
    :param bands_parameters: the requested ``pw.parameters`` of the bands run.
    :param scf_code: the ``quantumespresso.pw`` code of the SCF run.
    :param bands_code: the ``quantumespresso.pw`` code of the bands run.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param scf_kpoints: explicit SCF k-points, if any.
    :param scf_kpoints_distance: SCF mesh density, if any.
    :param bands_kpoints: explicit bands k-points, if any.
    :param bands_kpoints_distance: minimum k-point distance of the SeekPath path, if any.
    :param nbands_factor: multiplier of the SCF band count for the bands run, if any.
    :param scf_settings: the SCF ``pw.settings``, if any.
    :param bands_settings: the bands ``pw.settings``, if any.
    :param monitors: optional shared monitors forwarded into both nested base runs.
    :param scf_monitors: optional SCF monitors, defaulting to ``monitors``.
    :param bands_monitors: optional bands monitors, defaulting to ``monitors``.
    :return: the band structure, both parameter sets, and the SeekPath products when computed.
    """
    path = prepare_kpoints_path(
        structure=structure, bands_kpoints=bands_kpoints, bands_kpoints_distance=bands_kpoints_distance
    )
    if bands_kpoints is None:
        scf_structure = path.primitive_structure
    else:
        scf_structure = structure
    scf_run = scf(
        code=scf_code,
        structure=scf_structure,
        pseudos=pseudos,
        parameters=scf_parameters,
        settings=scf_settings,
        kpoints=scf_kpoints,
        kpoints_distance=scf_kpoints_distance,
        monitors=scf_monitors if scf_monitors is not None else monitors,
    )
    scf_checked = require_base_finished(report=scf_run.report, stage='scf')
    resolved = resolve_bands_parameters(
        scf_output_parameters=scf_run.output_parameters,
        bands_parameters=bands_parameters,
        nbands_factor=nbands_factor,
        after=scf_checked,
    )
    bands_run = bands(
        code=bands_code,
        structure=scf_structure,
        pseudos=pseudos,
        parameters=resolved,
        settings=bands_settings,
        kpoints=path.explicit_kpoints,
        parent_folder=scf_run.remote_folder,
        monitors=bands_monitors if bands_monitors is not None else monitors,
    )
    bands_checked = require_base_finished(report=bands_run.report, stage='bands')  # noqa: F841
    return {
        'band_structure': bands_run.output_band,
        'scf_parameters': scf_run.output_parameters,
        'band_parameters': bands_run.output_parameters,
        'primitive_structure': path.primitive_structure,
        'seekpath_parameters': path.seekpath_parameters,
    }
