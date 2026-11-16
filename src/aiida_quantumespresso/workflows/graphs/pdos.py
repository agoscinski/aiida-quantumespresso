############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Native ``@graph`` orchestration for the PDOS workflow.

The graphs place the native :func:`pw_base_graph
<aiida_quantumespresso.workflows.graphs.base.pw_base_graph>` subgraph for the
SCF/NSCF runs and the ``dos.x``/``projwfc.x`` calculations as tasks. No
``WorkChain`` is placed or imported here. Science stays where it was: restarts
and error handling inside ``pw_base_graph``, the energy window in
:func:`extract_nscf_window`, the band-count rule in
:func:`resolve_nscf_parameters`, and the DOS parameter assembly in
:func:`resolve_post_windows`. The graphs only wire names and sweep remote
folders through dedicated tasks.

Run the checks with the pinned experimental core and QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_pdos.py -q

and validate without launching anything with ``assert pdos_from_scratch_parallel.build() is None``.

Serial versus parallel
----------------------

The legacy ``serial_clean`` flag is replaced by explicit variants. The
``*_parallel`` graphs run DOS and PROJWFC as soon as their inputs are ready, so
the engine may run them concurrently. The ``*_serial`` graphs order
DOS → cleanup → PROJWFC with an unconditional ``after`` edge, faithfully
serializing the pair; the DOS remote is swept before PROJWFC submits, and the
SCF remote (from-scratch roots) is swept after the stage. ``clean_workdir``
only controls the final sweep of the remaining remotes. There is no
branch-dependent ``after``: each variant hard-codes its own ordering.

Scheduler options
-----------------

``CalcJob`` scheduler options travel at launch under ``task_metadata`` as a
strictly nested mapping, not through graph inputs. Each authored subgraph
callee alias adds one level; the ``CalcJob`` placement alias holds the normal
metadata mapping at the leaf. The stage scopes nest the ``nscf`` base alias
plus the ``dos_calc``/``projwfc_calc`` placements; a from-scratch root nests
the ``scf`` base scope beside its stage scope::

    task_metadata = {
        'pdos_stage_serial': {
            'nscf': {'pw_calc': {'options': nscf_options}},
            'dos_calc': {'options': dos_options},
            'projwfc_calc': {'options': projwfc_options},
        },
    }
    task_metadata = {
        'scf': {'pw_calc': {'options': scf_options}},
        'pdos_stage_serial': {
            'nscf': {'pw_calc': {'options': nscf_options}},
            'dos_calc': {'options': dos_options},
            'projwfc_calc': {'options': projwfc_options},
        },
    }

Branch and loop control frames are transparent: the serial variant's ordered
sweeps add no key level. The experimental per-bundle ``options`` ports of the
previous revision were removed rather than silently ignored. Form the launch
mapping with :func:`task_metadata_for_options
<aiida_quantumespresso.workflows.graphs.protocols.task_metadata_for_options>`
or as a literal nested mapping; unknown paths, missing entries, and
non-mapping leaves fail loudly at launch.

Honest limits of this rewrite (see ``reports/refinement-composites.md``):

* Graphs declare no exit codes: the legacy statuses (``401``-``404`` plus the
  restart ``301``/``302``/``401``/``402``) collapse into the single graph ``400
  ERROR_TASK_FAILED``. Stage attribution survives in the child exit statuses;
  the summary ``report`` is only emitted on success.
* Input validators moved into ``@task`` bodies and now surface as task failures,
  not launch-time ``ValidationError``s. The equality of ``emin``/``emax``/
  ``deltae`` across DOS and PROJWFC is enforced by
  :func:`resolve_post_windows`, which resolves both mappings from one window
  and returns them together.
* There is no ``dry_run``. ``monitors`` is an optional shared ``DataDict`` forwarded
  into every nested base run through its authored alias (``scf`` for the SCF run,
  ``nscf`` for the NSCF run); per-stage overrides fall
  back to the shared mapping. Protocol presets live in
  :mod:`aiida_quantumespresso.workflows.graphs.protocols` as plain mappings;
  pseudo-family and meta-parameter resolution still needs the caller.
"""

import copy
import typing as t

from aiida.engine import graph, task
from aiida.orm import BandsData, Code, DataDict, Dict, Float, KpointsData, RemoteData, StructureData, XyData
from aiida.orm.nodes.data.array.projection import ProjectionData

from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.workflows.graphs.base import pw_base_graph

# Authored callee aliases: repeated calls to the same subgraph are told apart by
# the name used at the call site, so each metadata-bearing placement gets its own
# module-level alias. The structured ``task_metadata`` descends one level per
# alias (``{'scf': {'pw_calc': ...}}``); the ``dos_calc``/``projwfc_calc``
# calculation tasks keep their own stable names as leaves.
scf = pw_base_graph
nscf = pw_base_graph

__all__ = (
    'MatchedParameters',
    'PdosFromParentOutputs',
    'PdosFromScratchOutputs',
    'PdosStageOutputs',
    'PdosWindow',
    'dos_calc',
    'extract_nscf_window',
    'pdos_from_parent',
    'pdos_from_parent_parallel',
    'pdos_from_parent_serial',
    'pdos_from_scratch',
    'pdos_from_scratch_parallel',
    'pdos_from_scratch_serial',
    'pdos_stage',
    'pdos_stage_parallel',
    'pdos_stage_serial',
    'require_scf_parameters',
    'resolve_nscf_parameters',
    'resolve_post_windows',
    'summarize_pdos',
    'sweep_remote',
    'sweep_stage',
)

dos_calc = task(DosCalculation)
projwfc_calc = task(ProjwfcCalculation)


class PdosWindow(t.TypedDict):
    """Energy window of :func:`extract_nscf_window`: bands edges plus the Fermi energy, all in eV."""

    emin: float
    emax: float
    fermi: float


class PdosStageOutputs(t.TypedDict):
    """Outputs of the NSCF-to-report stage: the NSCF products, the DOS products, and the summary report."""

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
    """Outputs of the from-scratch roots: the stage outputs plus the SCF products."""

    scf_remote_folder: RemoteData
    scf_output_parameters: Dict
    scf_output_band: BandsData | None


class PdosFromParentOutputs(PdosStageOutputs):
    """Outputs of the from-parent roots: exactly the stage outputs (no SCF ran here)."""


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
    return {'emin': float(bands.min()), 'emax': float(bands.max()), 'fermi': float(fermi)}


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
    fails here, before any submission. Returning both mappings threads a data edge into each calculation. The
    returned mappings are fresh nodes: tasks cannot return their inputs, which would close a provenance cycle.

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
    :param mode: ``'serial'`` or ``'parallel'``.
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
    ``clean_workdir`` sweep. Folders the serial sweeps already removed report ``False`` for their share.

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
def pdos_stage_parallel(
    structure: StructureData,
    parent_folder: RemoteData,
    pw_code: Code,
    pseudos: DataDict,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    nscf_kpoints: KpointsData | None = None,
    nscf_kpoints_distance: Float | None = None,
    nscf_settings: Dict | None = None,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    stages: int = 1,
    clean_workdir: bool = False,
    monitors: DataDict | None = None,
) -> PdosStageOutputs:
    """Run NSCF then DOS and PROJWFC concurrently from one parent folder, then report.

    The NSCF parameters resolve the band-count rule first; its bands window then feeds both post-processing
    parameter sets, which the matcher compares before either calculation submits. DOS and PROJWFC run as soon as
    their inputs are ready, so the engine may run them concurrently. The final sweep covers ``clean_workdir``.

    :param structure: the structure the NSCF run computes.
    :param parent_folder: the SCF (or caller-provided) remote folder the NSCF run restarts from.
    :param pw_code: the ``quantumespresso.pw`` code of the NSCF run.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param nscf_kpoints: explicit NSCF k-points, if any.
    :param nscf_kpoints_distance: NSCF mesh density, if any.
    :param nscf_settings: the NSCF ``pw.settings``, if any.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param parent_output_parameters: the ``output_parameters`` of the parent run, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param stages: how many PW base runs the calling root placed.
    :param clean_workdir: whether to sweep the remotes at the end.
    :param monitors: optional monitors forwarded into the NSCF base run.
    :return: the NSCF, DOS, and PROJWFC products plus the summary report.
    """
    resolved_nscf_parameters = resolve_nscf_parameters(
        parameters=nscf_parameters,
        parent_parameters=parent_output_parameters,
        nbands_factor=nbands_factor,
    )
    nscf_run = nscf(
        code=pw_code,
        structure=structure,
        pseudos=pseudos,
        parameters=resolved_nscf_parameters,
        settings=nscf_settings,
        kpoints=nscf_kpoints,
        kpoints_distance=nscf_kpoints_distance,
        parent_folder=parent_folder,
        monitors=monitors,
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
        mode='parallel',
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
def pdos_stage_serial(
    structure: StructureData,
    parent_folder: RemoteData,
    pw_code: Code,
    pseudos: DataDict,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    nscf_kpoints: KpointsData | None = None,
    nscf_kpoints_distance: Float | None = None,
    nscf_settings: Dict | None = None,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    stages: int = 1,
    clean_workdir: bool = False,
    monitors: DataDict | None = None,
) -> PdosStageOutputs:
    """Run NSCF then DOS, sweep DOS, then PROJWFC from one parent folder, then report.

    The unconditional ``after`` edge from the DOS sweep into PROJWFC serializes the
    pair, matching the legacy ``serial_clean`` order DOS → cleanup → PROJWFC. The
    DOS remote is swept promptly before PROJWFC submits; the final sweep covers
    ``clean_workdir`` for the remaining remotes.

    :param structure: the structure the NSCF run computes.
    :param parent_folder: the SCF (or caller-provided) remote folder the NSCF run restarts from.
    :param pw_code: the ``quantumespresso.pw`` code of the NSCF run.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param nscf_kpoints: explicit NSCF k-points, if any.
    :param nscf_kpoints_distance: NSCF mesh density, if any.
    :param nscf_settings: the NSCF ``pw.settings``, if any.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param parent_output_parameters: the ``output_parameters`` of the parent run, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param stages: how many PW base runs the calling root placed.
    :param clean_workdir: whether to sweep the remaining remotes at the end.
    :param monitors: optional monitors forwarded into the NSCF base run.
    :return: the NSCF, DOS, and PROJWFC products plus the summary report.
    """
    resolved_nscf_parameters = resolve_nscf_parameters(
        parameters=nscf_parameters,
        parent_parameters=parent_output_parameters,
        nbands_factor=nbands_factor,
    )
    nscf_run = nscf(
        code=pw_code,
        structure=structure,
        pseudos=pseudos,
        parameters=resolved_nscf_parameters,
        settings=nscf_settings,
        kpoints=nscf_kpoints,
        kpoints_distance=nscf_kpoints_distance,
        parent_folder=parent_folder,
        monitors=monitors,
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
    dos_swept = sweep_remote(folder=dos_run.remote_folder, enabled=True)
    projwfc_run = projwfc_calc(
        code=projwfc_code,
        settings=projwfc_settings,
        parameters=matched.projwfc,
        parent_folder=nscf_run.remote_folder,
        after=dos_swept,
    )
    projwfc_swept = sweep_remote(folder=projwfc_run.remote_folder, enabled=True, after=projwfc_run)
    swept = sweep_stage(  # noqa: F841
        nscf_folder=nscf_run.remote_folder,
        dos_folder=dos_run.remote_folder,
        projwfc_folder=projwfc_run.remote_folder,
        enabled=clean_workdir,
        after=projwfc_swept,
    )
    report = summarize_pdos(
        output_parameters=nscf_run.output_parameters,
        dos_parameters=matched.dos,
        fermi=window.fermi,
        stages=stages,
        mode='serial',
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


# Backwards-compatible aliases: the unqualified stage ran the parallel shape.
pdos_stage = pdos_stage_parallel


@graph
def pdos_from_scratch_parallel(
    structure: StructureData,
    pw_code: Code,
    pseudos: DataDict,
    scf_parameters: Dict,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    scf_kpoints: KpointsData | None = None,
    scf_kpoints_distance: Float | None = None,
    nscf_kpoints: KpointsData | None = None,
    nscf_kpoints_distance: Float | None = None,
    scf_settings: Dict | None = None,
    nscf_settings: Dict | None = None,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    clean_workdir: bool = False,
    monitors: DataDict | None = None,
    scf_monitors: DataDict | None = None,
    nscf_monitors: DataDict | None = None,
) -> PdosFromScratchOutputs:
    """Compute total and partial densities of states, running SCF first, DOS/PROJWFC concurrently.

    The SCF base run feeds its remote folder and band count into the shared
    parallel stage. The SCF remote is swept at the end only when
    ``clean_workdir`` holds. Scheduler options ride at launch under
    ``task_metadata`` (see the module docstring).

    :param structure: the structure to compute the DOS of.
    :param pw_code: the ``quantumespresso.pw`` code of the SCF and NSCF runs.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param scf_parameters: the requested ``pw.parameters`` of the SCF run.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param scf_kpoints: explicit SCF k-points, if any.
    :param scf_kpoints_distance: SCF mesh density, if any.
    :param nscf_kpoints: explicit NSCF k-points, if any.
    :param nscf_kpoints_distance: NSCF mesh density, if any.
    :param scf_settings: the SCF ``pw.settings``, if any.
    :param nscf_settings: the NSCF ``pw.settings``, if any.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param clean_workdir: whether to sweep the remotes at the end.
    :param monitors: optional shared monitors forwarded into SCF and NSCF base runs.
    :param scf_monitors: optional SCF monitors, defaulting to ``monitors``.
    :param nscf_monitors: optional NSCF monitors, defaulting to ``monitors``.
    :return: the SCF products, the stage products, and the summary report.
    """
    checked_scf_parameters = require_scf_parameters(parameters=scf_parameters)
    scf_run = scf(
        code=pw_code,
        structure=structure,
        pseudos=pseudos,
        parameters=checked_scf_parameters,
        settings=scf_settings,
        kpoints=scf_kpoints,
        kpoints_distance=scf_kpoints_distance,
        monitors=scf_monitors if scf_monitors is not None else monitors,
    )
    stage = pdos_stage_parallel(
        structure=structure,
        parent_folder=scf_run.remote_folder,
        pw_code=pw_code,
        pseudos=pseudos,
        nscf_parameters=nscf_parameters,
        dos_code=dos_code,
        dos_parameters=dos_parameters,
        projwfc_code=projwfc_code,
        projwfc_parameters=projwfc_parameters,
        nscf_kpoints=nscf_kpoints,
        nscf_kpoints_distance=nscf_kpoints_distance,
        nscf_settings=nscf_settings,
        dos_settings=dos_settings,
        projwfc_settings=projwfc_settings,
        parent_output_parameters=scf_run.output_parameters,
        energy_range_vs_fermi=energy_range_vs_fermi,
        nbands_factor=nbands_factor,
        stages=2,
        clean_workdir=clean_workdir,
        monitors=nscf_monitors if nscf_monitors is not None else monitors,
    )
    scf_final = sweep_remote(folder=scf_run.remote_folder, enabled=clean_workdir, after=stage)  # noqa: F841
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
def pdos_from_scratch_serial(
    structure: StructureData,
    pw_code: Code,
    pseudos: DataDict,
    scf_parameters: Dict,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    scf_kpoints: KpointsData | None = None,
    scf_kpoints_distance: Float | None = None,
    nscf_kpoints: KpointsData | None = None,
    nscf_kpoints_distance: Float | None = None,
    scf_settings: Dict | None = None,
    nscf_settings: Dict | None = None,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    clean_workdir: bool = False,
    monitors: DataDict | None = None,
    scf_monitors: DataDict | None = None,
    nscf_monitors: DataDict | None = None,
) -> PdosFromScratchOutputs:
    """Compute total and partial densities of states, running SCF first, DOS before PROJWFC.

    Mirrors the legacy ``serial_clean`` order: SCF → NSCF → DOS → sweep DOS →
    PROJWFC. The SCF remote is swept after the stage (it outlives NSCF as its
    parent); the DOS remote is swept inside the serial stage before PROJWFC
    submits. The final sweep covers ``clean_workdir``.

    :param structure: the structure to compute the DOS of.
    :param pw_code: the ``quantumespresso.pw`` code of the SCF and NSCF runs.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param scf_parameters: the requested ``pw.parameters`` of the SCF run.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param scf_kpoints: explicit SCF k-points, if any.
    :param scf_kpoints_distance: SCF mesh density, if any.
    :param nscf_kpoints: explicit NSCF k-points, if any.
    :param nscf_kpoints_distance: NSCF mesh density, if any.
    :param scf_settings: the SCF ``pw.settings``, if any.
    :param nscf_settings: the NSCF ``pw.settings``, if any.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param clean_workdir: whether to sweep the remaining remotes at the end.
    :param monitors: optional shared monitors forwarded into SCF and NSCF base runs.
    :param scf_monitors: optional SCF monitors, defaulting to ``monitors``.
    :param nscf_monitors: optional NSCF monitors, defaulting to ``monitors``.
    :return: the SCF products, the stage products, and the summary report.
    """
    checked_scf_parameters = require_scf_parameters(parameters=scf_parameters)
    scf_run = scf(
        code=pw_code,
        structure=structure,
        pseudos=pseudos,
        parameters=checked_scf_parameters,
        settings=scf_settings,
        kpoints=scf_kpoints,
        kpoints_distance=scf_kpoints_distance,
        monitors=scf_monitors if scf_monitors is not None else monitors,
    )
    stage = pdos_stage_serial(
        structure=structure,
        parent_folder=scf_run.remote_folder,
        pw_code=pw_code,
        pseudos=pseudos,
        nscf_parameters=nscf_parameters,
        dos_code=dos_code,
        dos_parameters=dos_parameters,
        projwfc_code=projwfc_code,
        projwfc_parameters=projwfc_parameters,
        nscf_kpoints=nscf_kpoints,
        nscf_kpoints_distance=nscf_kpoints_distance,
        nscf_settings=nscf_settings,
        dos_settings=dos_settings,
        projwfc_settings=projwfc_settings,
        parent_output_parameters=scf_run.output_parameters,
        energy_range_vs_fermi=energy_range_vs_fermi,
        nbands_factor=nbands_factor,
        stages=2,
        clean_workdir=clean_workdir,
        monitors=nscf_monitors if nscf_monitors is not None else monitors,
    )
    scf_swept = sweep_remote(folder=scf_run.remote_folder, enabled=True, after=stage)
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


# Backwards-compatible alias: the unqualified from-scratch root ran the parallel shape.
pdos_from_scratch = pdos_from_scratch_parallel


@graph
def pdos_from_parent_parallel(
    structure: StructureData,
    parent_folder: RemoteData,
    pw_code: Code,
    pseudos: DataDict,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    nscf_kpoints: KpointsData | None = None,
    nscf_kpoints_distance: Float | None = None,
    nscf_settings: Dict | None = None,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    clean_workdir: bool = False,
    monitors: DataDict | None = None,
) -> PdosFromParentOutputs:
    """Compute total and partial densities of states from a caller-provided folder, concurrently.

    Mirrors the legacy chain without the ``scf`` inputs: the caller hands over
    the NSCF parent folder directly. ``parent_output_parameters`` only matters
    with ``nbands_factor``. Scheduler options ride at launch under
    ``task_metadata`` (see the module docstring).

    :param structure: the structure to compute the DOS of.
    :param parent_folder: the remote folder the NSCF run restarts from.
    :param pw_code: the ``quantumespresso.pw`` code of the NSCF run.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param nscf_kpoints: explicit NSCF k-points, if any.
    :param nscf_kpoints_distance: NSCF mesh density, if any.
    :param nscf_settings: the NSCF ``pw.settings``, if any.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param parent_output_parameters: the ``output_parameters`` of the parent run, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param clean_workdir: whether to sweep the remotes at the end.
    :param monitors: optional monitors forwarded into the NSCF base run.
    :return: the stage products and the summary report.
    """
    stage = pdos_stage_parallel(
        structure=structure,
        parent_folder=parent_folder,
        pw_code=pw_code,
        pseudos=pseudos,
        nscf_parameters=nscf_parameters,
        dos_code=dos_code,
        dos_parameters=dos_parameters,
        projwfc_code=projwfc_code,
        projwfc_parameters=projwfc_parameters,
        nscf_kpoints=nscf_kpoints,
        nscf_kpoints_distance=nscf_kpoints_distance,
        nscf_settings=nscf_settings,
        dos_settings=dos_settings,
        projwfc_settings=projwfc_settings,
        parent_output_parameters=parent_output_parameters,
        energy_range_vs_fermi=energy_range_vs_fermi,
        nbands_factor=nbands_factor,
        stages=1,
        clean_workdir=clean_workdir,
        monitors=monitors,
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


@graph
def pdos_from_parent_serial(
    structure: StructureData,
    parent_folder: RemoteData,
    pw_code: Code,
    pseudos: DataDict,
    nscf_parameters: Dict,
    dos_code: Code,
    dos_parameters: Dict,
    projwfc_code: Code,
    projwfc_parameters: Dict,
    nscf_kpoints: KpointsData | None = None,
    nscf_kpoints_distance: Float | None = None,
    nscf_settings: Dict | None = None,
    dos_settings: Dict | None = None,
    projwfc_settings: Dict | None = None,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    clean_workdir: bool = False,
    monitors: DataDict | None = None,
) -> PdosFromParentOutputs:
    """Compute total and partial densities of states from a caller-provided folder, DOS before PROJWFC.

    The serial stage orders DOS → sweep → PROJWFC with an unconditional
    ``after`` edge, matching the legacy ``serial_clean`` order.

    :param structure: the structure to compute the DOS of.
    :param parent_folder: the remote folder the NSCF run restarts from.
    :param pw_code: the ``quantumespresso.pw`` code of the NSCF run.
    :param pseudos: mapping of kind names onto pseudopotential nodes.
    :param nscf_parameters: the requested ``pw.parameters`` of the NSCF run.
    :param dos_code: the ``quantumespresso.dos`` code.
    :param dos_parameters: the requested ``dos.x`` parameters.
    :param projwfc_code: the ``quantumespresso.projwfc`` code.
    :param projwfc_parameters: the requested ``projwfc.x`` parameters.
    :param nscf_kpoints: explicit NSCF k-points, if any.
    :param nscf_kpoints_distance: NSCF mesh density, if any.
    :param nscf_settings: the NSCF ``pw.settings``, if any.
    :param dos_settings: the ``dos.x`` settings, if any.
    :param projwfc_settings: the ``projwfc.x`` settings, if any.
    :param parent_output_parameters: the ``output_parameters`` of the parent run, if any.
    :param energy_range_vs_fermi: the energy range around the Fermi level to cover, if any.
    :param nbands_factor: the multiplier of the parent band count, if any.
    :param clean_workdir: whether to sweep the remaining remotes at the end.
    :param monitors: optional monitors forwarded into the NSCF base run.
    :return: the stage products and the summary report.
    """
    stage = pdos_stage_serial(
        structure=structure,
        parent_folder=parent_folder,
        pw_code=pw_code,
        pseudos=pseudos,
        nscf_parameters=nscf_parameters,
        dos_code=dos_code,
        dos_parameters=dos_parameters,
        projwfc_code=projwfc_code,
        projwfc_parameters=projwfc_parameters,
        nscf_kpoints=nscf_kpoints,
        nscf_kpoints_distance=nscf_kpoints_distance,
        nscf_settings=nscf_settings,
        dos_settings=dos_settings,
        projwfc_settings=projwfc_settings,
        parent_output_parameters=parent_output_parameters,
        energy_range_vs_fermi=energy_range_vs_fermi,
        nbands_factor=nbands_factor,
        stages=1,
        clean_workdir=clean_workdir,
        monitors=monitors,
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


# Backwards-compatible alias: the unqualified from-parent root ran the parallel shape.
pdos_from_parent = pdos_from_parent_parallel
