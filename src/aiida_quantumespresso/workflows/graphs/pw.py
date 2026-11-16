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

``CalcJob`` scheduler options travel at launch under ``task_metadata``, keyed by
the ``pw.x`` placement name inside ``pw_base_graph`` (``'pw_calc'``), not through
graph inputs. The experimental ``options`` ports of the previous revision were
removed rather than silently ignored: passing scheduler options through a graph
``Dict`` input would suggest per-iteration control that the native base graph
does not offer (its options are constant across iterations). Form the launch
mapping with :func:`task_metadata_for_options
<aiida_quantumespresso.workflows.graphs.protocols.task_metadata_for_options>`::

    from aiida_quantumespresso.workflows.graphs.protocols import task_metadata_for_options

    task_metadata = task_metadata_for_options(pw_calc={'options': {'max_wallclock_seconds': 3600}})
    run_get_node(pw_relax_graph, ..., task_metadata=task_metadata)

Honest limits of this rewrite (see ``reports/refinement-composites.md``):

* ``pw_base_graph`` is owned by the sibling ``graphs/base.py`` track; its agreed
  contract is ``(code, structure, pseudos: DataDict, parameters, settings?,
  kpoints?, kpoints_distance?, kpoints_force_parity?, parent_folder?)`` returning
  ``output_parameters``/``output_structure?``/``output_band?``/``remote_folder``/
  ``report``. Pseudopotentials travel as a ``DataDict`` (kind name to ``UpfData``).
* A base run that exits ``501`` (ionic convergence reached except in final SCF)
  fails the step instead of triggering another iteration from the ``501``
  structure: mapped error arms null out success outputs, so the relaxed structure
  of a ``501`` run is unreachable downstream. Only success-path re-convergence
  (k-point mesh regrowth, bounded iterations) is orchestrated here.
* Graphs declare no exit codes: legacy statuses (relax ``400/401/402/403``,
  bands ``201/202/402/403``) collapse into the single graph ``400
  ERROR_TASK_FAILED``. Outcomes are reported through the ``report`` output.
* Input validators moved into ``@task`` bodies and now surface as task failures,
  not launch-time ``ValidationError``s. There is no ``clean_workdir`` handling,
  no ``dry_run``, and no monitor wiring.
* ``scf``-mode relax echoes the input structure as ``output_structure`` where the
  legacy chain emits none.
"""

import copy
import typing as t

from aiida import orm
from aiida.engine import PortModel, graph, task
from aiida.orm import BandsData, Code, DataDict, Dict, Float, KpointsData, RemoteData, StructureData

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance as _create_kpoints_calcfunction,
)
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis as _seekpath_calcfunction,
)
from aiida_quantumespresso.workflows.graphs.base import pw_base_graph

# Undecorated twins of the ``calcfunction`` implementations above. Graph tasks run as calculation processes and so
# cannot call a decorated ``calcfunction`` (the engine forbids it); calling the shared undecorated implementation
# keeps the science identical without duplicating it and without adding nested provenance nodes.
_create_kpoints_from_distance = _create_kpoints_calcfunction.__wrapped__
_seekpath_structure_analysis = _seekpath_calcfunction.__wrapped__

__all__ = (
    'BandPath',
    'BandsOutputs',
    'RelaxNext',
    'RelaxOutputs',
    'ensure_structure',
    'init_relax_guarded',
    'prepare_kpoints_path',
    'prepare_next_relax',
    'pw_bands_graph',
    'pw_relax_graph',
    'resolve_bands_parameters',
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
    """Outputs of :func:`pw_relax_graph`."""

    output_structure: StructureData
    report: Dict


class BandsOutputs(t.TypedDict):
    """Outputs of :func:`pw_bands_graph`; ``None`` leaves stay out of the emitted outputs."""

    band_structure: BandsData
    scf_parameters: Dict
    band_parameters: Dict
    primitive_structure: StructureData | None
    seekpath_parameters: Dict | None


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
def prepare_next_relax(
    parameters: Dict,
    output_parameters: Dict,
    structure_before: StructureData,
    structure_after: StructureData | None,
    kpoints_distance: Float | None,
    force_parity: bool,
    meta_convergence: bool,
) -> RelaxNext:
    """Decide whether another relax iteration is needed and carry the parameters forward.

    Mirrors ``PwRelaxWorkChain.should_run_relax`` for the success path: the number of bands of the finished run is
    carried into ``SYSTEM.nbnd``, and when meta-convergence is on and the cell may still change volume, the k-point
    meshes before and after are regenerated with the undecorated twin of :func:`create_kpoints_from_distance
    <aiida_quantumespresso.calculations.functions.create_kpoints_from_distance.create_kpoints_from_distance>`; a
    grown mesh requests another iteration. Pulay-stress reruns (base exit status ``501``) never reach this task:
    the base subgraph fails first, so the graph fails instead of retrying.

    :param parameters: the parameters the finished iteration ran with.
    :param output_parameters: the ``output_parameters`` of the finished base run.
    :param structure_before: the structure the finished iteration started from.
    :param structure_after: the ``output_structure`` of the finished iteration, if any.
    :param kpoints_distance: the mesh density, or ``None`` when k-points are explicit.
    :param force_parity: whether mesh generation preserves parity.
    :param meta_convergence: whether mesh-regrowth reruns are enabled at all.
    :return: whether to rerun and the parameters for the next iteration.
    :raises ValueError: if a relaxing calculation finished without an output structure.
    """
    params = copy.deepcopy(parameters.get_dict())
    calculation = params.get('CONTROL', {}).get('calculation', 'scf')

    if structure_after is None:
        if calculation == 'scf':
            return {'rerun': False, 'parameters': Dict(params)}
        msg = 'A `relax`/`vc-relax` base run finished successfully but without an output structure.'
        raise ValueError(msg)

    params.setdefault('SYSTEM', {})['nbnd'] = output_parameters.get_dict()['number_of_bands']

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
    settings: Dict | None = None,
    parent_folder: RemoteData | None = None,
) -> StructureData:
    """Run one initial relaxation and guarantee its output structure.

    :param structure: the structure to relax.
    :param parameters: the ``pw.parameters`` of the initial relaxation.
    :param code: the ``quantumespresso.pw`` code.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param kpoints: explicit k-points, if any.
    :param kpoints_distance: mesh density, if any.
    :param settings: the ``pw.settings``, if any.
    :param parent_folder: the ``pw.parent_folder``, if any.
    :return: the relaxed structure.
    """
    initial = pw_base_graph(
        code=code,
        structure=structure,
        pseudos=pseudos,
        parameters=parameters,
        settings=settings,
        kpoints=kpoints,
        kpoints_distance=kpoints_distance,
        parent_folder=parent_folder,
    )
    return ensure_structure(candidate=initial.output_structure)


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
    init_parameters: Dict | None = None,
) -> RelaxOutputs:
    """Relax a structure through repeated native base runs, then report.

    An optional initial relaxation runs first when ``run_init_relax`` holds (it shares code, pseudopotentials,
    k-points, settings, and parent folder with the main loop; only ``init_parameters`` may differ, defaulting to the
    main parameters). The main loop reruns while the mesh-regrowth predicate holds and the inline attempt budget
    lasts, then a summary task records the outcome. The root returns the existing final structure and the summary
    node. Scheduler options ride at launch under ``task_metadata`` (see the module docstring); there is no
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
    :param init_parameters: the parameters of the initial relaxation, defaulting to ``parameters``.
    :return: the relaxed structure and the summary report.
    """
    current = structure
    if run_init_relax:
        current = init_relax_guarded(
            structure=current,
            parameters=parameters if init_parameters is None else init_parameters,
            code=code,
            pseudos=pseudos,
            kpoints=kpoints,
            kpoints_distance=kpoints_distance,
            settings=settings,
            parent_folder=parent_folder,
        )
    params = parameters
    attempts = 0
    rerun = True
    while rerun and attempts < max_iterations:
        step = pw_base_graph(
            code=code,
            structure=current,
            pseudos=pseudos,
            parameters=params,
            settings=settings,
            kpoints=kpoints,
            kpoints_distance=kpoints_distance,
            kpoints_force_parity=kpoints_force_parity,
            parent_folder=parent_folder,
        )
        decision = prepare_next_relax(
            parameters=params,
            output_parameters=step.output_parameters,
            structure_before=current,
            structure_after=step.output_structure,
            kpoints_distance=kpoints_distance,
            force_parity=kpoints_force_parity,
            meta_convergence=meta_convergence,
        )
        if step.output_structure is not None:
            current = step.output_structure
        params = decision.parameters
        rerun = decision.rerun
        attempts = attempts + 1
    if rerun:
        status = 'exhausted'
    else:
        status = 'converged'
    report = summarize_relax(structure=current, attempts=attempts, status=status, max_iterations=max_iterations)
    return {'output_structure': current, 'report': report}


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
    :return: the band structure, both parameter sets, and the SeekPath products when computed.
    """
    path = prepare_kpoints_path(
        structure=structure, bands_kpoints=bands_kpoints, bands_kpoints_distance=bands_kpoints_distance
    )
    if bands_kpoints is None:
        scf_structure = path.primitive_structure
    else:
        scf_structure = structure
    scf = pw_base_graph(
        code=scf_code,
        structure=scf_structure,
        pseudos=pseudos,
        parameters=scf_parameters,
        settings=scf_settings,
        kpoints=scf_kpoints,
        kpoints_distance=scf_kpoints_distance,
    )
    resolved = resolve_bands_parameters(
        scf_output_parameters=scf.output_parameters, bands_parameters=bands_parameters, nbands_factor=nbands_factor
    )
    bands = pw_base_graph(
        code=bands_code,
        structure=scf_structure,
        pseudos=pseudos,
        parameters=resolved,
        settings=bands_settings,
        kpoints=path.explicit_kpoints,
        parent_folder=scf.remote_folder,
    )
    return {
        'band_structure': bands.output_band,
        'scf_parameters': scf.output_parameters,
        'band_parameters': bands.output_parameters,
        'primitive_structure': path.primitive_structure,
        'seekpath_parameters': path.seekpath_parameters,
    }
