############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Native ``@graph`` orchestration for the PDOS workflow.

This module rewrites the *orchestration* of :class:`PdosWorkChain
<aiida_quantumespresso.workflows.pdos.PdosWorkChain>` as ``@graph`` workflows. The legacy ``WorkChain`` is
preserved untouched; the graphs place the original base work chains (through :class:`PwBaseStep
<aiida_quantumespresso.workflows.graphs.pw.PwBaseStep>`, shared with the PW graphs) and the original ``dos.x``
and ``projwfc.x`` calculations as tasks instead of re-implementing them. Science stays where it was: restarts and
error handling inside ``PwBaseWorkChain``, the energy window in :func:`extract_nscf_window`, the band-count rule
in :func:`resolve_nscf_parameters`, and the DOS parameter assembly in :func:`resolve_post_windows`. The graphs
only wire names, pass whole namespaces between tasks, and sweep remote folders through dedicated tasks.

Run the checks with the pinned experimental core and QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_pdos.py -q

and validate without launching anything with ``assert pdos_from_scratch.build() is None``.

The legacy workflow runs SCF optionally, so the rewrite exposes two static roots instead of one dynamic one:
:func:`pdos_from_scratch` runs SCF then NSCF, while :func:`pdos_from_parent` starts from a caller-provided NSCF
parent folder. Both delegate the NSCF-to-report stage to :func:`pdos_stage`.

Honest limits of this rewrite (see ``reports/pdos-migration.md`` for the full record):

* ``PwBaseWorkChain`` cannot be placed directly: its ``pw`` namespace is dynamic *and* required (it nests the
  dynamic, required ``pseudos`` mapping), and the graph compiler only closes one dynamic level under a static
  parent. The shared :class:`PwBaseStep
  <aiida_quantumespresso.workflows.graphs.pw.PwBaseStep>` therefore routes each base run through fully static
  ports, at the cost of one extra ``WorkChain`` node per run. Pseudopotentials travel as an ``orm.DataDict``.
* Graphs declare no exit codes: the legacy statuses (``401``-``404`` plus the restart ``301``/``302``/``401``/
  ``402``) collapse into the single graph ``400 ERROR_TASK_FAILED``. Stage attribution survives in the child exit
  statuses; the summary ``report`` is only emitted on success.
* Input validators moved into ``@task`` bodies and now surface as task failures, not launch-time
  ``ValidationError``s. The equality of ``emin``/``emax``/``deltae`` across DOS and PROJWFC is enforced by
  :func:`resolve_post_windows`, which resolves both mappings from one window and returns them together.
* ``serial_clean`` sweeps remote folders promptly after each calculation and no longer serializes DOS before
  PROJWFC: the graph language merges branch arms by name and wiring, so conditional ordering between two
  placements is inexpressible and both calculations stay dispatch-ready. The engine may therefore run them
  concurrently; plan scratch-disk headroom of the legacy parallel mode. ``clean_workdir`` sweeps the remaining
  remotes at the end through no-op-by-default tasks instead of ``on_terminated``.
* There is no ``dry_run``, no monitor wiring, and no ``get_builder_from_protocol``: protocol presets stay on the
  legacy work chain until graph-specific protocol builders are implemented.
"""

import copy
import typing as t
from typing import Annotated, Any

from aiida.engine import Forward, graph, task
from aiida.orm import BandsData, Code, Dict, RemoteData, StructureData, XyData
from aiida.orm.nodes.data.array.projection import ProjectionData
from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.workflows.graphs.pw import run_pw_base

__all__ = (
    'MatchedParameters',
    'NscfBundle',
    'PdosFromParentOutputs',
    'PdosFromScratchOutputs',
    'PdosStageOutputs',
    'PdosWindow',
    'ScfBundle',
    'dos_calc',
    'extract_nscf_window',
    'pdos_from_parent',
    'pdos_from_scratch',
    'pdos_stage',
    'projwfc_calc',
    'require_scf_parameters',
    'resolve_nscf_parameters',
    'resolve_post_windows',
    'summarize_pdos',
    'sweep_remote',
    'sweep_stage',
)

dos_calc = task(DosCalculation)
projwfc_calc = task(ProjwfcCalculation)

ScfBundle = Annotated[Any, Forward(run_pw_base, exclude=('structure', 'parent_folder', 'parameters'))]
"""Inputs of one SCF base run: code, pseudos, k-points, settings, and options; ``structure`` and the resolved
``parameters`` are wired by the graph and ``parent_folder`` never applies."""

NscfBundle = Annotated[Any, Forward(run_pw_base, exclude=('structure', 'parent_folder', 'parameters'))]
"""Inputs of the NSCF base run: code, pseudos, k-points, settings, and options; ``structure``,
``parent_folder``, and the resolved ``parameters`` are wired by the graph."""


class PdosWindow(t.TypedDict):
    """Energy window of :func:`extract_nscf_window`: bands edges plus the Fermi energy, all in eV."""

    emin: float
    emax: float
    fermi: float


class PdosStageOutputs(t.TypedDict):
    """Outputs of :func:`pdos_stage`: the NSCF products, the DOS products, and the summary report."""

    nscf_remote_folder: RemoteData
    nscf_output_parameters: Dict
    nscf_output_band: BandsData
    dos_output_parameters: Dict
    dos_output_dos: XyData
    projwfc_output_parameters: Dict
    projwfc_dos: XyData
    projwfc_ldos: XyData | None
    projwfc_pdos: XyData | None
    projwfc_projections: ProjectionData | None
    projwfc_projections_up: ProjectionData | None
    projwfc_projections_down: ProjectionData | None
    projwfc_bands: BandsData | None
    projwfc_bands_up: BandsData | None
    projwfc_bands_down: BandsData | None
    report: Dict


class PdosFromScratchOutputs(PdosStageOutputs):
    """Outputs of :func:`pdos_from_scratch`: the stage outputs plus the SCF products."""

    scf_remote_folder: RemoteData
    scf_output_parameters: Dict
    scf_output_band: BandsData | None


class PdosFromParentOutputs(PdosStageOutputs):
    """Outputs of :func:`pdos_from_parent`: exactly the stage outputs (no SCF ran here)."""


@task
def require_scf_parameters(parameters: Dict) -> Dict:
    """Return a copy of the SCF parameters after checking the ``CONTROL.calculation`` type.

    Mirrors the ``validate_scf`` launch validator of the legacy work chain. The copy keeps the task output a
    freshly created node.

    :param parameters: the requested ``pw.parameters`` of the SCF run.
    :return: a copy of the same parameters, guaranteed to request an ``scf`` calculation.
    :raises ValueError: if ``CONTROL.calculation`` is not set to ``scf``.
    """
    if parameters.get_dict().get('CONTROL', {}).get('calculation', 'scf') != 'scf':
        msg = '`CONTROL.calculation` in `scf.pw.parameters` is not set to `scf`.'
        raise ValueError(msg)
    return Dict(parameters.get_dict())


@task
def resolve_nscf_parameters(
    parameters: Dict,
    parent_parameters: Dict | None = None,
    nbands_factor: float | None = None,
) -> Dict:
    """Resolve the ``pw.parameters`` of the NSCF run, applying the ``nbands_factor`` rule.

    Mirrors ``PdosWorkChain.run_nscf``: with ``nbands_factor`` the band count is the maximum of the parent band
    count, ``0.5 * nelectrons * nbands_factor``, and ``0.5 * nelectrons + 4``; otherwise the requested parameters
    pass through. The calculation type and the ``tetrahedra`` occupations are checked first, mirroring the
    ``validate_nscf`` launch validator.

    :param parameters: the requested ``pw.parameters`` of the NSCF run.
    :param parent_parameters: the ``output_parameters`` of the parent (SCF or caller-provided) run, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :return: the resolved NSCF parameters.
    :raises ValueError: if the calculation type or occupations are wrong, if both ``nbands_factor`` and
        ``SYSTEM.nbnd`` are specified, or if a factor is given without parent parameters.
    """
    params = copy.deepcopy(parameters.get_dict())
    if params.get('CONTROL', {}).get('calculation', 'scf') != 'nscf':
        msg = '`CONTROL.calculation` in `nscf.pw.parameters` is not set to `nscf`.'
        raise ValueError(msg)
    if not params.get('SYSTEM', {}).get('occupations', '').startswith('tetrahedra'):
        msg = '`SYSTEM.occupations` in `nscf.pw.parameters` is not set to one of the `tetrahedra` options.'
        raise ValueError(msg)
    if nbands_factor is not None:
        if 'nbnd' in params.get('SYSTEM', {}):
            msg = 'Cannot specify both `nbands_factor` and `nscf.pw.parameters.SYSTEM.nbnd`.'
            raise ValueError(msg)
        if parent_parameters is None:
            msg = '`nbands_factor` needs the parent `output_parameters`, but none were provided.'
            raise ValueError(msg)
        parent = parent_parameters.get_dict()
        nbands = int(parent['number_of_bands'])
        nelectron = int(parent['number_of_electrons'])
        params.setdefault('SYSTEM', {})['nbnd'] = max(
            int(0.5 * nelectron * nbands_factor), int(0.5 * nelectron) + 4, nbands
        )
    return Dict(params)


@task
def extract_nscf_window(output_parameters: Dict, output_band: BandsData | None = None) -> PdosWindow:
    """Extract the energy window of the NSCF run: bands edges plus the Fermi energy, all in eV.

    Mirrors ``PdosWorkChain.inspect_nscf``: the spin-resolved Fermi energies fall back to their maximum.

    :param output_parameters: the ``output_parameters`` of the NSCF base run.
    :param output_band: the ``output_band`` of the NSCF base run, if any.
    :return: the bands minimum and maximum and the Fermi energy.
    :raises ValueError: if the NSCF run produced no bands.
    """
    if output_band is None:
        msg = 'The NSCF run finished without an `output_band`; the DOS energy window needs its bands.'
        raise ValueError(msg)
    parsed = output_parameters.get_dict()
    if 'fermi_energy' in parsed:
        fermi = float(parsed['fermi_energy'])
    else:
        fermi = max(float(parsed['fermi_energy_up']), float(parsed['fermi_energy_down']))
    bands = output_band.get_array('bands')
    return {'emin': float(bands.min()), 'emax': float(bands.max()), 'fermi': fermi}


def _resolve_window(
    parameters: dict,
    emin: float,
    emax: float,
    fermi: float,
    energy_range_vs_fermi: list | None,
    namelist: str,
) -> dict:
    """Resolve one post-processing parameter mapping against the NSCF energy window.

    An explicit ``energy_range_vs_fermi`` wins over the window and over any requested ``emin``/``emax``;
    otherwise requested values are kept and only missing ones fall back to the window.

    :param parameters: the requested parameters mapping.
    :param emin: the NSCF bands minimum in eV.
    :param emax: the NSCF bands maximum in eV.
    :param fermi: the NSCF Fermi energy in eV.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param namelist: the namelist holding the energy range, ``'DOS'`` or ``'PROJWFC'``.
    :return: the resolved parameters mapping.
    :raises ValueError: if ``deltae`` is missing or ``energy_range_vs_fermi`` is not a pair of numbers.
    """
    params = copy.deepcopy(parameters)
    block = params.get(namelist, {})
    if 'deltae' not in block:
        msg = f'The `{namelist}` parameters need a `deltae` energy grid step.'
        raise ValueError(msg)
    if energy_range_vs_fermi is not None:
        if len(energy_range_vs_fermi) != 2 or not all(
            isinstance(value, (float, int)) for value in energy_range_vs_fermi
        ):
            msg = f'`energy_range_vs_fermi` should be a pair of numbers, but got: {energy_range_vs_fermi}'
            raise ValueError(msg)
        block['emin'] = float(energy_range_vs_fermi[0]) + fermi
        block['emax'] = float(energy_range_vs_fermi[1]) + fermi
    else:
        block.setdefault('emin', emin)
        block.setdefault('emax', emax)
    params[namelist] = block
    return params


class MatchedParameters(t.TypedDict):
    """Resolved post-processing parameters of :func:`resolve_post_windows`, keyed by calculation."""

    dos: Dict
    projwfc: Dict


@task
def resolve_post_windows(
    dos_parameters: Dict,
    projwfc_parameters: Dict,
    emin: float,
    emax: float,
    fermi: float,
    energy_range_vs_fermi: list | None = None,
) -> MatchedParameters:
    """Resolve the ``dos.x``/``projwfc.x`` parameters against the NSCF energy window, checking they match.

    Mirrors ``PdosWorkChain._generate_dos_inputs``/``_generate_projwfc_inputs`` plus the ``emin``/``emax``/
    ``deltae`` equality launch validator: both mappings resolve from the same window, and a mismatch between them
    fails here, before any submission. Returning both mappings threads a data edge into each calculation without
    an explicit ordering edge. The returned mappings are fresh nodes: tasks cannot return their inputs, which
    would close a provenance cycle.

    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param emin: the NSCF bands minimum in eV.
    :param emax: the NSCF bands maximum in eV.
    :param fermi: the NSCF Fermi energy in eV.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :return: both resolved mappings.
    :raises ValueError: if ``deltae`` is missing, ``energy_range_vs_fermi`` is malformed, or the resolved ``emin``,
        ``emax``, or ``deltae`` differ between the two inputs.
    """
    dos = _resolve_window(dos_parameters.get_dict(), emin, emax, fermi, energy_range_vs_fermi, 'DOS')
    projwfc = _resolve_window(projwfc_parameters.get_dict(), emin, emax, fermi, energy_range_vs_fermi, 'PROJWFC')
    for key in ('emin', 'emax', 'deltae'):
        if dos['DOS'].get(key, None) != projwfc['PROJWFC'].get(key, None):
            msg = f'The `{key}` parameter has to be equal for the `dos` and `projwfc` inputs.'
            raise ValueError(msg)
    return {'dos': Dict(dos), 'projwfc': Dict(projwfc)}


@task
def summarize_pdos(
    output_parameters: Dict, dos_parameters: Dict, fermi: float, stages: int, mode: str
) -> dict[str, float | int | str | None]:
    """Record the covered energy window, band count, run shape, and outcome as one report.

    :param output_parameters: the ``output_parameters`` of the NSCF base run.
    :param dos_parameters: the resolved ``dos.x`` parameters holding the covered window.
    :param fermi: the Fermi energy in eV.
    :param stages: how many PW base runs the graph placed.
    :param mode: ``'serial'`` or ``'parallel'``, following ``serial_clean``.
    :return: the summary mapping stored as a single ``Dict``.
    """
    window = dos_parameters.get_dict().get('DOS', {})
    return {
        'emin': window.get('emin', None),
        'emax': window.get('emax', None),
        'fermi_energy': fermi,
        'nbands': output_parameters.get_dict().get('number_of_bands', None),
        'stages': stages,
        'mode': mode,
        'status': 'completed',
    }


@task
def sweep_remote(folder: RemoteData | None = None, enabled: bool = False) -> bool:
    """Clean one remote folder when enabled, reporting whether anything was removed.

    Mirrors ``clean_calcjob_remote``; absent folders and disabled sweeps report ``False`` without touching storage.

    :param folder: the remote folder to clean, if any.
    :param enabled: whether cleaning is requested at all.
    :return: whether a remote folder was cleaned.
    """
    if not enabled or folder is None:
        return False
    try:
        folder._clean()
    except (OSError, KeyError):
        return False
    return True


@task
def sweep_stage(
    nscf_folder: RemoteData | None = None,
    dos_folder: RemoteData | None = None,
    projwfc_folder: RemoteData | None = None,
    enabled: bool = False,
) -> bool:
    """Clean the stage remote folders when enabled, reporting whether anything was removed.

    Runs after the DOS and PROJWFC placements through its data dependencies, so it doubles as the
    ``clean_workdir`` sweep of the legacy ``on_terminated`` step. Folders the serial sweeps already removed
    report ``False`` for their share.

    :param nscf_folder: the NSCF remote folder, if any.
    :param dos_folder: the DOS remote folder, if any.
    :param projwfc_folder: the PROJWFC remote folder, if any.
    :param enabled: whether cleaning is requested at all.
    :return: whether at least one remote folder was cleaned.
    """
    if not enabled:
        return False
    cleaned = False
    for folder in (nscf_folder, dos_folder, projwfc_folder):
        if folder is None:
            continue
        try:
            folder._clean()
        except (OSError, KeyError):
            continue
        cleaned = True
    return cleaned


@graph
def pdos_stage(
    structure: StructureData,
    parent_folder: RemoteData,
    nscf: NscfBundle,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    stages: int = 1,
    serial_clean: bool = False,
    clean_workdir: bool = False,
) -> PdosStageOutputs:
    """Run NSCF then DOS and PROJWFC from one parent folder, then report.

    The NSCF parameters resolve the band-count rule first; its bands window then feeds both post-processing
    parameter sets, which the matcher compares before either calculation submits. DOS and PROJWFC run as soon as
    their inputs are ready, so the engine may run them concurrently: ``serial_clean`` only sweeps remote folders
    promptly after each calculation and no longer serializes the pair (see the module limits). The final sweep
    covers the ``clean_workdir`` parity.

    :param structure: the structure the NSCF run computes.
    :param parent_folder: the SCF (or caller-provided) remote folder the NSCF run restarts from.
    :param nscf: the inputs of the NSCF base run, without ``structure``, ``parent_folder``, and ``parameters``.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param parent_output_parameters: the ``output_parameters`` of the parent run, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param stages: how many PW base runs the calling root placed.
    :param serial_clean: whether to sweep each remote folder promptly after its calculation.
    :param clean_workdir: whether to sweep the remaining remotes at the end.
    :return: the NSCF, DOS, and PROJWFC products plus the summary report.
    """
    resolved_nscf_parameters = resolve_nscf_parameters(
        parameters=nscf_parameters,
        parent_parameters=parent_output_parameters,
        nbands_factor=nbands_factor,
    )
    nscf_run = run_pw_base(
        structure=structure,
        parameters=resolved_nscf_parameters,
        code=nscf.code,
        pseudos=nscf.pseudos,
        kpoints=nscf.kpoints,
        kpoints_distance=nscf.kpoints_distance,
        settings=nscf.settings,
        options=nscf.options,
        parent_folder=parent_folder,
    )
    window = extract_nscf_window(output_parameters=nscf_run.output_parameters, output_band=nscf_run.output_band)
    matched = resolve_post_windows(
        dos_parameters=dos_parameters,
        projwfc_parameters=projwfc_parameters,
        emin=window.emin,
        emax=window.emax,
        fermi=window.fermi,
        energy_range_vs_fermi=energy_range_vs_fermi,
    )
    dos_run = dos_calc(
        code=dos_code,
        settings=dos_settings,
        parameters=matched.dos,
        parent_folder=nscf_run.remote_folder,
    )
    projwfc_run = projwfc_calc(
        code=projwfc_code,
        settings=projwfc_settings,
        parameters=matched.projwfc,
        parent_folder=nscf_run.remote_folder,
    )
    dos_swept = sweep_remote(folder=dos_run.remote_folder, enabled=serial_clean)  # noqa: F841
    projwfc_swept = sweep_remote(folder=projwfc_run.remote_folder, enabled=serial_clean)  # noqa: F841
    swept = sweep_stage(  # noqa: F841
        nscf_folder=nscf_run.remote_folder,
        dos_folder=dos_run.remote_folder,
        projwfc_folder=projwfc_run.remote_folder,
        enabled=clean_workdir,
    )
    report = summarize_pdos(
        output_parameters=nscf_run.output_parameters,
        dos_parameters=matched.dos,
        fermi=window.fermi,
        stages=stages,
        mode='serial' if serial_clean else 'parallel',
    )
    return {
        'nscf_remote_folder': nscf_run.remote_folder,
        'nscf_output_parameters': nscf_run.output_parameters,
        'nscf_output_band': nscf_run.output_band,
        'dos_output_parameters': dos_run.output_parameters,
        'dos_output_dos': dos_run.output_dos,
        'projwfc_output_parameters': projwfc_run.output_parameters,
        'projwfc_dos': projwfc_run.Dos,
        'projwfc_ldos': projwfc_run.Ldos,
        'projwfc_pdos': projwfc_run.Pdos,
        'projwfc_projections': projwfc_run.projections,
        'projwfc_projections_up': projwfc_run.projections_up,
        'projwfc_projections_down': projwfc_run.projections_down,
        'projwfc_bands': projwfc_run.bands,
        'projwfc_bands_up': projwfc_run.bands_up,
        'projwfc_bands_down': projwfc_run.bands_down,
        'report': report,
    }


@graph
def pdos_from_scratch(
    structure: StructureData,
    scf: ScfBundle,
    scf_parameters: Dict,
    nscf: NscfBundle,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    serial_clean: bool = False,
    clean_workdir: bool = False,
) -> PdosFromScratchOutputs:
    """Compute total and partial densities of states, running SCF first.

    Mirrors ``PdosWorkChain`` with the ``scf`` inputs given: the SCF base run feeds its remote folder and band
    count into the shared stage. The SCF remote is swept once the stage is placed when ``serial_clean`` holds,
    and again at the end when ``clean_workdir`` holds.

    Scheduler options for the ``dos.x``/``projwfc.x`` placements ride under ``task_metadata`` keyed
    ``'dos_calc'`` and ``'projwfc_calc'`` (the declaration names, stable across every graph); the base runs take theirs through the ``options`` field of each bundle.

    :param structure: the structure to compute the DOS of.
    :param scf: the inputs of the SCF base run, without ``structure``, ``parent_folder``, and ``parameters``.
    :param scf_parameters: the requested ``pw.parameters`` of the SCF run.
    :param nscf: the inputs of the NSCF base run, without ``structure``, ``parent_folder``, and ``parameters``.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param serial_clean: whether to sweep each remote folder promptly after its calculation.
    :param clean_workdir: whether to sweep the remaining remotes at the end.
    :return: the SCF products, the stage products, and the summary report.
    """
    checked_scf_parameters = require_scf_parameters(parameters=scf_parameters)
    scf_run = run_pw_base(
        structure=structure,
        parameters=checked_scf_parameters,
        code=scf.code,
        pseudos=scf.pseudos,
        kpoints=scf.kpoints,
        kpoints_distance=scf.kpoints_distance,
        settings=scf.settings,
        options=scf.options,
    )
    stage = pdos_stage(
        structure=structure,
        parent_folder=scf_run.remote_folder,
        nscf=nscf,
        nscf_parameters=nscf_parameters,
        dos_code=dos_code,
        dos_parameters=dos_parameters,
        projwfc_code=projwfc_code,
        projwfc_parameters=projwfc_parameters,
        dos_settings=dos_settings,
        projwfc_settings=projwfc_settings,
        parent_output_parameters=scf_run.output_parameters,
        energy_range_vs_fermi=energy_range_vs_fermi,
        nbands_factor=nbands_factor,
        stages=2,
        serial_clean=serial_clean,
        clean_workdir=clean_workdir,
    )
    scf_swept = sweep_remote(folder=scf_run.remote_folder, enabled=serial_clean, after=stage)
    scf_final = sweep_remote(folder=scf_run.remote_folder, enabled=clean_workdir, after=scf_swept)  # noqa: F841
    return {
        'scf_remote_folder': scf_run.remote_folder,
        'scf_output_parameters': scf_run.output_parameters,
        'scf_output_band': scf_run.output_band,
        'nscf_remote_folder': stage.nscf_remote_folder,
        'nscf_output_parameters': stage.nscf_output_parameters,
        'nscf_output_band': stage.nscf_output_band,
        'dos_output_parameters': stage.dos_output_parameters,
        'dos_output_dos': stage.dos_output_dos,
        'projwfc_output_parameters': stage.projwfc_output_parameters,
        'projwfc_dos': stage.projwfc_dos,
        'projwfc_ldos': stage.projwfc_ldos,
        'projwfc_pdos': stage.projwfc_pdos,
        'projwfc_projections': stage.projwfc_projections,
        'projwfc_projections_up': stage.projwfc_projections_up,
        'projwfc_projections_down': stage.projwfc_projections_down,
        'projwfc_bands': stage.projwfc_bands,
        'projwfc_bands_up': stage.projwfc_bands_up,
        'projwfc_bands_down': stage.projwfc_bands_down,
        'report': stage.report,
    }


@graph
def pdos_from_parent(
    structure: StructureData,
    parent_folder: RemoteData,
    nscf: NscfBundle,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    serial_clean: bool = False,
    clean_workdir: bool = False,
) -> PdosFromParentOutputs:
    """Compute total and partial densities of states, restarting NSCF from a caller-provided folder.

    Mirrors ``PdosWorkChain`` without the ``scf`` inputs (where the legacy chain reads
    ``nscf.pw.parent_folder``): the caller hands over the NSCF parent folder directly. ``parent_output_parameters``
    only matters with ``nbands_factor``, mirroring the legacy read of the parent creator outputs.

    Scheduler options for the ``dos.x``/``projwfc.x`` placements ride under ``task_metadata`` keyed
    ``'dos_calc'`` and ``'projwfc_calc'`` (the declaration names, stable across every graph); the base run takes its through the ``options`` field of the ``nscf`` bundle.

    :param structure: the structure to compute the DOS of.
    :param parent_folder: the remote folder the NSCF run restarts from.
    :param nscf: the inputs of the NSCF base run, without ``structure``, ``parent_folder``, and ``parameters``.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param parent_output_parameters: the ``output_parameters`` of the parent run, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param serial_clean: whether to sweep each remote folder promptly after its calculation.
    :param clean_workdir: whether to sweep the remaining remotes at the end.
    :return: the stage products and the summary report.
    """
    stage = pdos_stage(
        structure=structure,
        parent_folder=parent_folder,
        nscf=nscf,
        nscf_parameters=nscf_parameters,
        dos_code=dos_code,
        dos_parameters=dos_parameters,
        projwfc_code=projwfc_code,
        projwfc_parameters=projwfc_parameters,
        dos_settings=dos_settings,
        projwfc_settings=projwfc_settings,
        parent_output_parameters=parent_output_parameters,
        energy_range_vs_fermi=energy_range_vs_fermi,
        nbands_factor=nbands_factor,
        stages=1,
        serial_clean=serial_clean,
        clean_workdir=clean_workdir,
    )
    return {
        'nscf_remote_folder': stage.nscf_remote_folder,
        'nscf_output_parameters': stage.nscf_output_parameters,
        'nscf_output_band': stage.nscf_output_band,
        'dos_output_parameters': stage.dos_output_parameters,
        'dos_output_dos': stage.dos_output_dos,
        'projwfc_output_parameters': stage.projwfc_output_parameters,
        'projwfc_dos': stage.projwfc_dos,
        'projwfc_ldos': stage.projwfc_ldos,
        'projwfc_pdos': stage.projwfc_pdos,
        'projwfc_projections': stage.projwfc_projections,
        'projwfc_projections_up': stage.projwfc_projections_up,
        'projwfc_projections_down': stage.projwfc_projections_down,
        'projwfc_bands': stage.projwfc_bands,
        'projwfc_bands_up': stage.projwfc_bands_up,
        'projwfc_bands_down': stage.projwfc_bands_down,
        'report': stage.report,
    }


def get_protocol_filepath() -> t.Any:
    """Return the ``pathlib.Path`` to the legacy ``pdos.yaml`` protocol file.

    The graph takes explicit inputs and does not consume protocol presets itself; this helper only exposes the
    legacy file location for tooling that still reads it.
    """
    from aiida_quantumespresso.workflows.pdos import PdosWorkChain

    return PdosWorkChain.get_protocol_filepath()
