############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Mocked tests for the native ``@graph`` PDOS orchestration.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_pdos.py -q

The real graphs (:mod:`aiida_quantumespresso.workflows.graphs.pdos`) are validated with ``build()`` and their
``@task`` collaborators run for real through ``run_get_node`` (all local: no codes, no cluster). Full
orchestration runs against the test-local :class:`StubPwStep` and :class:`StubPost` stand-ins, which mirror the
static routed surfaces of :class:`PwBaseStep
<aiida_quantumespresso.workflows.graphs.pw.PwBaseStep>` and the ``dos.x``/``projwfc.x`` calculations without
submitting them. The stub graphs below mirror the real graph bodies line for line except for the placed steps and
the code-carrying bundles; any drift between them is a bug in this file, while the real wiring is pinned by the
``build()`` tests. See ``reports/pdos-migration.md`` for what the mocks do not cover (real base runs, restart
handlers, exit codes, scheduler options).
"""

import typing as t

import numpy as np
import pytest

import aiida.engine

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)


from aiida_quantumespresso.workflows.graphs.pdos import (
    dos_calc,
    extract_nscf_window,
    pdos_from_parent,
    pdos_from_scratch,
    pdos_stage,
    projwfc_calc,
    require_scf_parameters,
    resolve_nscf_parameters,
    resolve_post_windows,
    summarize_pdos,
    sweep_remote,
    sweep_stage,
)

from aiida.common.links import LinkType
from aiida.engine import WorkChain, calcfunction, graph, run_get_node, task
from aiida.orm import BandsData, Code, Computer, Dict, KpointsData, QueryBuilder, RemoteData, StructureData, XyData

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


def _silicon() -> StructureData:
    """Return an unstored diamond silicon structure."""
    structure = StructureData(cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols='Si')
    return structure


def _scf_parameters() -> Dict:
    """Return SCF ``pw.parameters`` with ``tetrahedra``-free smearing, as the SCF step uses them."""
    return Dict(
        {
            'CONTROL': {'calculation': 'scf'},
            'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0},
            'ELECTRONS': {},
        }
    )


def _nscf_parameters() -> Dict:
    """Return NSCF ``pw.parameters`` with ``tetrahedra`` occupations, as the legacy inputs use them."""
    return Dict(
        {
            'CONTROL': {'calculation': 'nscf'},
            'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0, 'occupations': 'tetrahedra', 'nosym': True},
            'ELECTRONS': {},
        }
    )


def _dos_parameters() -> Dict:
    """Return ``dos.x`` parameters with only the grid step, as the legacy fixture uses them."""
    return Dict({'DOS': {'deltae': 0.01}})


def _projwfc_parameters() -> Dict:
    """Return ``projwfc.x`` parameters with only the grid step, as the legacy fixture uses them."""
    return Dict({'PROJWFC': {'deltae': 0.01, 'ngauss': 0, 'degauss': 0.01}})


def _stub_computer() -> Computer:
    """Return a stored and configured localhost computer for the stub remote folders."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import AuthInfo, User

    existing = QueryBuilder().append(Computer, filters={'label': 'stub-localhost'}).first(flat=True)
    if existing is not None:
        return existing
    computer = Computer(
        label='stub-localhost',
        hostname='localhost',
        transport_type='core.local',
        scheduler_type='core.direct',
        workdir='/tmp',
    )
    computer.store()
    try:
        computer.get_authinfo(User.collection.get_default())
    except NotExistent:
        AuthInfo(computer=computer, user=User.collection.get_default()).store()
    return computer


class StubPwOutputs(t.TypedDict):
    """Fabricated base-run products of :func:`_stub_pw_outputs`."""

    output_parameters: Dict
    output_band: BandsData


class StubPostOutputs(t.TypedDict):
    """Fabricated post-processing products of :func:`_stub_post_outputs`."""

    output_parameters: Dict
    output_dos: XyData
    Dos: XyData


@calcfunction
def _stub_pw_outputs(parameters: Dict) -> StubPwOutputs:
    """Fabricate base-run outputs from the input parameters: the band count echoes ``SYSTEM.nbnd``."""
    nbnd = parameters.get_dict().get('SYSTEM', {}).get('nbnd', 8)
    kpoints = KpointsData()
    kpoints.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    bands = BandsData()
    bands.set_kpointsdata(kpoints)
    bands.set_bands([[-5.0], [8.0]])
    return {
        'output_parameters': Dict({'number_of_bands': nbnd, 'number_of_electrons': 8, 'fermi_energy': 6.9}),
        'output_band': bands,
    }


@calcfunction
def _stub_post_outputs(parameters: Dict) -> StubPostOutputs:
    """Fabricate post-processing outputs echoing the resolved energy window."""
    params = parameters.get_dict()
    block = params.get('DOS', params.get('PROJWFC', {}))
    emin, emax = float(block.get('emin', -5.0)), float(block.get('emax', 8.0))
    output_dos = XyData()
    output_dos.set_x(np.linspace(emin, emax, num=2), 'energy', 'eV')
    output_dos.set_y([np.ones(2)], 'dos', 'states/eV')
    total_dos = XyData()
    total_dos.set_x(np.linspace(emin, emax, num=2), 'energy', 'eV')
    total_dos.set_y([np.ones(2)], 'dos', 'states/eV')
    return {
        'output_parameters': Dict({'emin': emin, 'emax': emax, 'deltae': block.get('deltae', 0.01)}),
        'output_dos': output_dos,
        'Dos': total_dos,
    }


@calcfunction
def _stub_remote_folder(path: str) -> RemoteData:
    """Return a ``RemoteData`` on the self-contained stub localhost computer."""
    return RemoteData(computer=_stub_computer(), remote_path=path)


class StubPwStep(WorkChain):
    """Mocked stand-in for ``PwBaseStep`` with the same routed static surface, minus codes and pseudos.

    Each run echoes the requested band count, fabricates ``output_parameters`` with a fixed Fermi energy, and emits
    a two-point ``output_band`` spanning the window the extract task reads.
    """

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        """Define the process specification."""
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=Dict)
        spec.input('settings', valid_type=Dict, required=False)
        spec.input('parent_folder', valid_type=RemoteData, required=False)
        spec.output('remote_folder', valid_type=RemoteData)
        spec.output('output_parameters', valid_type=Dict)
        spec.output('output_band', valid_type=BandsData)
        spec.outline(cls.produce)

    def produce(self):  # type: ignore[no-untyped-def]
        """Emit the mocked outputs."""
        produced = _stub_pw_outputs(self.inputs.parameters)
        self.out('remote_folder', _stub_remote_folder('/tmp/stub-pw'))
        self.out('output_parameters', produced['output_parameters'])
        self.out('output_band', produced['output_band'])


class StubPost(WorkChain):
    """Mocked stand-in for the ``dos.x``/``projwfc.x`` calculations, echoing the resolved window."""

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        """Define the process specification."""
        super().define(spec)
        spec.input('parameters', valid_type=Dict)
        spec.input('parent_folder', valid_type=RemoteData, required=False)
        spec.input('settings', valid_type=Dict, required=False)
        spec.output('output_parameters', valid_type=Dict)
        spec.output('output_dos', valid_type=XyData)
        spec.output('Dos', valid_type=XyData)
        spec.output('remote_folder', valid_type=RemoteData)
        spec.outline(cls.produce)

    def produce(self):  # type: ignore[no-untyped-def]
        """Emit the mocked outputs."""
        produced = _stub_post_outputs(self.inputs.parameters)
        self.out('remote_folder', _stub_remote_folder('/tmp/stub-post'))
        self.out('output_parameters', produced['output_parameters'])
        self.out('output_dos', produced['output_dos'])
        self.out('Dos', produced['Dos'])


stub_pw_step = task(StubPwStep)
stub_post = task(StubPost)


class StubStageOutputs(t.TypedDict):
    """Outputs of :func:`stub_pdos_stage`: the stubbed NSCF, DOS, and PROJWFC products plus the report."""

    nscf_remote_folder: RemoteData
    nscf_output_parameters: Dict
    nscf_output_band: BandsData
    dos_output_parameters: Dict
    dos_output_dos: XyData
    projwfc_output_parameters: Dict
    projwfc_dos: XyData
    report: Dict


class StubFromScratchOutputs(StubStageOutputs):
    """Outputs of :func:`stub_from_scratch`: the stage outputs plus the stubbed SCF products."""

    scf_remote_folder: RemoteData
    scf_output_parameters: Dict
    scf_output_band: BandsData


class StubFromParentOutputs(StubStageOutputs):
    """Outputs of :func:`stub_from_parent`: exactly the stage outputs (no SCF ran here)."""


@graph
def stub_pdos_stage(
    structure: StructureData,
    parent_folder: RemoteData,
    nscf_parameters: Dict,
    dos_parameters: Dict,
    projwfc_parameters: Dict,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    stages: int = 1,
    serial_clean: bool = False,
    clean_workdir: bool = False,
) -> StubStageOutputs:
    """Mirror :func:`pdos_stage` with the mocked steps and no codes."""
    resolved_nscf_parameters = resolve_nscf_parameters(
        parameters=nscf_parameters,
        parent_parameters=parent_output_parameters,
        nbands_factor=nbands_factor,
    )
    nscf_run = stub_pw_step(structure=structure, parameters=resolved_nscf_parameters, parent_folder=parent_folder)
    window = extract_nscf_window(output_parameters=nscf_run.output_parameters, output_band=nscf_run.output_band)
    matched = resolve_post_windows(
        dos_parameters=dos_parameters,
        projwfc_parameters=projwfc_parameters,
        emin=window.emin,
        emax=window.emax,
        fermi=window.fermi,
        energy_range_vs_fermi=energy_range_vs_fermi,
    )
    dos_run = stub_post(parameters=matched.dos, parent_folder=nscf_run.remote_folder)
    projwfc_run = stub_post(parameters=matched.projwfc, parent_folder=nscf_run.remote_folder)
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
        'report': report,
    }


@graph
def stub_from_scratch(
    structure: StructureData,
    scf_parameters: Dict,
    nscf_parameters: Dict,
    dos_parameters: Dict,
    projwfc_parameters: Dict,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    serial_clean: bool = False,
    clean_workdir: bool = False,
) -> StubFromScratchOutputs:
    """Mirror :func:`pdos_from_scratch` with the mocked steps and no codes."""
    checked_scf_parameters = require_scf_parameters(parameters=scf_parameters)
    scf_run = stub_pw_step(structure=structure, parameters=checked_scf_parameters)
    stage = stub_pdos_stage(
        structure=structure,
        parent_folder=scf_run.remote_folder,
        nscf_parameters=nscf_parameters,
        dos_parameters=dos_parameters,
        projwfc_parameters=projwfc_parameters,
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
        'report': stage.report,
    }


@graph
def stub_from_parent(
    structure: StructureData,
    parent_folder: RemoteData,
    nscf_parameters: Dict,
    dos_parameters: Dict,
    projwfc_parameters: Dict,
    parent_output_parameters: Dict | None = None,
    energy_range_vs_fermi: list | None = None,
    nbands_factor: float | None = None,
    serial_clean: bool = False,
    clean_workdir: bool = False,
) -> StubFromParentOutputs:
    """Mirror :func:`pdos_from_parent` with the mocked steps and no codes."""
    stage = stub_pdos_stage(
        structure=structure,
        parent_folder=parent_folder,
        nscf_parameters=nscf_parameters,
        dos_parameters=dos_parameters,
        projwfc_parameters=projwfc_parameters,
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
        'report': stage.report,
    }


def _base_children(node) -> list:
    """Return the called ``StubPwStep`` descendants of a graph node."""
    return [child for child in node.called_descendants if child.process_label == 'StubPwStep']


def test_build_stage_graph() -> None:
    """The real NSCF-to-report stage validates without launching anything."""
    assert pdos_stage.build() is None


def test_build_from_scratch_graph() -> None:
    """The real from-scratch graph validates without launching anything."""
    assert pdos_from_scratch.build() is None


def test_build_from_parent_graph() -> None:
    """The real from-parent graph validates without launching anything."""
    assert pdos_from_parent.build() is None


@graph
def place_post_only(code: Code, parameters: Dict, parent_folder: RemoteData) -> Dict:
    """Place the real post-processing calculations with required inputs only."""
    dos_run = dos_calc(code=code, parameters=parameters, parent_folder=parent_folder)
    projwfc_run = projwfc_calc(code=code, parameters=parameters, parent_folder=parent_folder)  # noqa: F841
    return dos_run.output_parameters


def test_build_post_placements() -> None:
    """The real ``dos.x``/``projwfc.x`` tasks expose static contracts a graph can place."""
    assert place_post_only.build() is None


def test_post_placements_need_task_metadata() -> None:
    """The calculation placements keep their names so one ``task_metadata`` entry covers every graph."""
    from aiida.engine.processes.graphs.ir import BranchControl, LoopControl, ProcessTask, SubgraphTask
    from aiida.engine.processes.graphs.lowering import compile_graph

    def _process_tasks(template) -> list:
        """Return every process task of a template, including nested bodies."""
        found = []

        def _visit(current) -> None:
            if isinstance(current, ProcessTask):
                found.append(current)
            elif isinstance(current, BranchControl):
                for region in (current.body, current.otherwise):
                    for nested in region.tasks:
                        _visit(nested)
            elif isinstance(current, (LoopControl, SubgraphTask)):
                for nested in current.body.tasks:
                    _visit(nested)

        for current in template.tasks:
            _visit(current)
        return found

    names = set()
    for target in (pdos_stage, pdos_from_scratch, pdos_from_parent):
        for placement in _process_tasks(compile_graph(target)):
            if placement.spec.identifier in ('DosCalculation', 'ProjwfcCalculation'):
                names.add(placement.name)
    assert names == {'dos_calc', 'projwfc_calc'}


def test_require_scf_parameters() -> None:
    """SCF parameters pass through after the calculation-type check."""
    outputs, node = run_get_node(require_scf_parameters, parameters=_scf_parameters())
    assert node.is_finished_ok
    assert outputs.get_dict()['CONTROL']['calculation'] == 'scf'


def test_require_scf_parameters_wrong_calculation_fails() -> None:
    """A non-``scf`` calculation fails the SCF parameter check."""
    with pytest.raises(ValueError, match='not set to `scf`'):
        run_get_node(require_scf_parameters, parameters=_nscf_parameters())


@pytest.mark.parametrize(
    ('parent', 'factor', 'expected'),
    [
        ({'number_of_bands': 8, 'number_of_electrons': 16}, None, None),
        ({'number_of_bands': 8, 'number_of_electrons': 10}, 1.5, 9),
        ({'number_of_bands': 8, 'number_of_electrons': 16}, 1.5, 12),
    ],
)
def test_resolve_nscf_parameters(parent, factor, expected) -> None:
    """The NSCF parameters resolve the band-count rule, mirroring the legacy ``run_nscf`` cases."""
    parent_parameters = Dict(parent) if parent is not None else None
    outputs, node = run_get_node(
        resolve_nscf_parameters,
        parameters=_nscf_parameters(),
        parent_parameters=parent_parameters,
        nbands_factor=factor,
    )
    assert node.is_finished_ok
    if expected is None:
        assert 'nbnd' not in outputs.get_dict()['SYSTEM']
    else:
        assert outputs.get_dict()['SYSTEM']['nbnd'] == expected


def test_resolve_nscf_parameters_wrong_calculation_fails() -> None:
    """SCF parameters fail the NSCF calculation-type check."""
    with pytest.raises(ValueError, match='not set to `nscf`'):
        run_get_node(
            resolve_nscf_parameters,
            parameters=_scf_parameters(),
            parent_parameters=None,
            nbands_factor=None,
        )


def test_resolve_nscf_parameters_wrong_occupations_fails() -> None:
    """Non-``tetrahedra`` occupations fail the NSCF check, mirroring the legacy validator."""
    parameters = Dict(
        {
            'CONTROL': {'calculation': 'nscf'},
            'SYSTEM': {'occupations': 'smearing', 'smearing': 'gaussian', 'degauss': 0.01},
            'ELECTRONS': {},
        }
    )
    with pytest.raises(ValueError, match='tetrahedra'):
        run_get_node(
            resolve_nscf_parameters,
            parameters=parameters,
            parent_parameters=None,
            nbands_factor=None,
        )


def test_resolve_nscf_parameters_mutual_exclusion_fails() -> None:
    """Both ``nbands_factor`` and ``SYSTEM.nbnd`` fail the resolution."""
    parameters = _nscf_parameters().get_dict()
    parameters['SYSTEM']['nbnd'] = 10
    with pytest.raises(ValueError, match='nbands_factor'):
        run_get_node(
            resolve_nscf_parameters,
            parameters=Dict(parameters),
            parent_parameters=Dict({'number_of_bands': 8, 'number_of_electrons': 16}),
            nbands_factor=1.5,
        )


def test_resolve_nscf_parameters_missing_parent_fails() -> None:
    """A factor without parent parameters fails, mirroring the legacy creator read."""
    with pytest.raises(ValueError, match='parent `output_parameters`'):
        run_get_node(
            resolve_nscf_parameters,
            parameters=_nscf_parameters(),
            parent_parameters=None,
            nbands_factor=1.5,
        )


def _bands(emin: float = -5.0, emax: float = 8.0) -> BandsData:
    """Return stored bands spanning the given window."""
    kpoints = KpointsData()
    kpoints.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    bands = BandsData()
    bands.set_kpointsdata(kpoints)
    bands.set_bands([[emin], [emax]])
    bands.store()
    return bands


def test_extract_window_fermi() -> None:
    """The window reads the bands edges and the Fermi energy."""
    outputs, node = run_get_node(
        extract_nscf_window,
        output_parameters=Dict({'number_of_bands': 8, 'fermi_energy': 6.9}),
        output_band=_bands(),
    )
    assert node.is_finished_ok
    assert outputs['emin'].value == pytest.approx(-5.0)
    assert outputs['emax'].value == pytest.approx(8.0)
    assert outputs['fermi'].value == pytest.approx(6.9)


def test_extract_window_spin_polarized_fermi() -> None:
    """Spin-resolved Fermi energies fall back to their maximum, mirroring ``inspect_nscf``."""
    outputs, node = run_get_node(
        extract_nscf_window,
        output_parameters=Dict({'number_of_bands': 8, 'fermi_energy_up': 6.9, 'fermi_energy_down': 5.9}),
        output_band=_bands(),
    )
    assert node.is_finished_ok
    assert outputs['fermi'].value == pytest.approx(6.9)


def test_extract_window_missing_bands_fails() -> None:
    """A missing ``output_band`` fails the extraction instead of inventing a window."""
    with pytest.raises(ValueError, match='without an `output_band`'):
        run_get_node(
            extract_nscf_window,
            output_parameters=Dict({'number_of_bands': 8, 'fermi_energy': 6.9}),
            output_band=None,
        )


@pytest.mark.parametrize(
    ('dos_emin', 'dos_emax', 'energy_range', 'expected'),
    [
        (-10.0, 10.0, None, (-10.0, 10.0)),
        (None, None, [-10.0, 10.0], (-3.1, 16.9)),
        (None, None, None, (-5.0, 8.0)),
    ],
)
def test_resolve_post_windows(dos_emin, dos_emax, energy_range, expected) -> None:
    """The DOS/PROJWFC parameters resolve explicit, Fermi-relative, and bands-default windows like the legacy chain."""
    dos = {'DOS': {'deltae': 0.01}}
    if dos_emin is not None:
        dos['DOS'].update({'emin': dos_emin, 'emax': dos_emax})
    projwfc = {'PROJWFC': dict(dos['DOS'])}
    outputs, node = run_get_node(
        resolve_post_windows,
        dos_parameters=Dict(dos),
        projwfc_parameters=Dict(projwfc),
        emin=-5.0,
        emax=8.0,
        fermi=6.9,
        energy_range_vs_fermi=energy_range,
    )
    assert node.is_finished_ok
    for label, namelist in (('dos', 'DOS'), ('projwfc', 'PROJWFC')):
        resolved = outputs[label].get_dict()[namelist]
        assert np.isclose(resolved['emin'], expected[0])
        assert np.isclose(resolved['emax'], expected[1])


def test_resolve_post_windows_missing_deltae_fails() -> None:
    """A missing ``deltae`` fails the resolution, mirroring the legacy parameter schema."""
    with pytest.raises(ValueError, match='deltae'):
        run_get_node(
            resolve_post_windows,
            dos_parameters=Dict({'DOS': {}}),
            projwfc_parameters=_projwfc_parameters(),
            emin=-5.0,
            emax=8.0,
            fermi=6.9,
            energy_range_vs_fermi=None,
        )


def test_resolve_post_windows_bad_range_fails() -> None:
    """A malformed ``energy_range_vs_fermi`` fails the resolution."""
    with pytest.raises(ValueError, match='pair of numbers'):
        run_get_node(
            resolve_post_windows,
            dos_parameters=_dos_parameters(),
            projwfc_parameters=_projwfc_parameters(),
            emin=-5.0,
            emax=8.0,
            fermi=6.9,
            energy_range_vs_fermi=[-10.0],
        )


def test_resolve_post_windows_mismatch_fails() -> None:
    """Unequal grids fail before any submission, mirroring the legacy validator."""
    dos = {'DOS': {'emin': -5.0, 'emax': 8.0, 'deltae': 0.01}}
    projwfc = {'PROJWFC': {'emin': -4.0, 'emax': 8.0, 'deltae': 0.01}}
    with pytest.raises(ValueError, match='has to be equal'):
        run_get_node(
            resolve_post_windows,
            dos_parameters=Dict(dos),
            projwfc_parameters=Dict(projwfc),
            emin=-5.0,
            emax=8.0,
            fermi=6.9,
            energy_range_vs_fermi=None,
        )


def test_summarize_pdos() -> None:
    """The summary records the window, band count, run shape, and outcome."""
    outputs, node = run_get_node(
        summarize_pdos,
        output_parameters=Dict({'number_of_bands': 12}),
        dos_parameters=Dict({'DOS': {'emin': -3.1, 'emax': 16.9, 'deltae': 0.01}}),
        fermi=6.9,
        stages=2,
        mode='parallel',
    )
    assert node.is_finished_ok
    assert outputs.get_dict() == {
        'emin': -3.1,
        'emax': 16.9,
        'fermi_energy': 6.9,
        'nbands': 12,
        'stages': 2,
        'mode': 'parallel',
        'status': 'completed',
    }


def test_sweep_remote() -> None:
    """Disabled sweeps and missing folders report ``False`` without touching storage."""
    outputs, node = run_get_node(sweep_remote, folder=None, enabled=False)
    assert node.is_finished_ok
    assert outputs.value is False


def test_sweep_stage_disabled() -> None:
    """The disabled stage sweep reports ``False``."""
    outputs, node = run_get_node(sweep_stage, nscf_folder=None, dos_folder=None, projwfc_folder=None, enabled=False)
    assert node.is_finished_ok
    assert outputs.value is False


def _stub_inputs(**overrides) -> dict:
    """Return the stub runtime inputs with the legacy-like defaults."""
    inputs: dict = {
        'structure': _silicon(),
        'scf_parameters': _scf_parameters(),
        'nscf_parameters': _nscf_parameters(),
        'dos_parameters': _dos_parameters(),
        'projwfc_parameters': _projwfc_parameters(),
    }
    inputs.update(overrides)
    return inputs


def test_stub_from_scratch_parallel() -> None:
    """The mocked from-scratch run finishes SCF, NSCF, and both post-processing steps, then reports."""
    inputs = _stub_inputs()
    outputs, node = run_get_node(stub_from_scratch, **inputs)
    structure = inputs['structure']
    assert node.is_finished_ok
    assert outputs['report'] == {
        'emin': -5.0,
        'emax': 8.0,
        'fermi_energy': 6.9,
        'nbands': 8,
        'stages': 2,
        'mode': 'parallel',
        'status': 'completed',
    }
    assert outputs['dos_output_parameters']['emin'] == -5.0
    assert outputs['dos_output_parameters']['emax'] == 8.0
    assert isinstance(outputs['dos_output_dos'], XyData)
    assert isinstance(outputs['projwfc_dos'], XyData)
    assert node.inputs.structure.uuid == structure.uuid
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    assert len(_base_children(node)) == 2
    summary_node = next(child for child in node.called_descendants if child.process_label == 'summarize_pdos')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


def test_stub_from_scratch_serial() -> None:
    """The mocked serial run reports the serial mode and survives the sweep tasks."""
    outputs, node = run_get_node(stub_from_scratch, serial_clean=True, clean_workdir=True, **_stub_inputs())
    assert node.is_finished_ok
    assert outputs['report']['mode'] == 'serial'
    assert outputs['report']['status'] == 'completed'


def test_stub_from_parent_nbands_factor() -> None:
    """The mocked from-parent run applies the band-count rule from the caller-provided parent outputs."""
    parent_folder = RemoteData(computer=_stub_computer(), remote_path='/tmp/stub-parent')
    parent_folder.store()
    inputs = _stub_inputs()
    del inputs['scf_parameters']
    inputs.update(
        {
            'parent_folder': parent_folder,
            'parent_output_parameters': Dict({'number_of_bands': 8, 'number_of_electrons': 10}),
            'nbands_factor': 1.5,
        }
    )
    outputs, node = run_get_node(stub_from_parent, **inputs)
    assert node.is_finished_ok
    assert outputs['nscf_output_parameters']['number_of_bands'] == 9
    assert outputs['report']['nbands'] == 9
    assert outputs['report']['stages'] == 1
    assert len(_base_children(node)) == 1


def test_stub_from_parent_energy_range() -> None:
    """The mocked from-parent run covers the Fermi-relative range, mirroring the legacy cases."""
    parent_folder = RemoteData(computer=_stub_computer(), remote_path='/tmp/stub-parent')
    parent_folder.store()
    inputs = _stub_inputs()
    del inputs['scf_parameters']
    inputs.update({'parent_folder': parent_folder, 'energy_range_vs_fermi': [-10.0, 10.0]})
    outputs, node = run_get_node(stub_from_parent, **inputs)
    assert node.is_finished_ok
    assert np.isclose(outputs['dos_output_parameters']['emin'], -3.1)
    assert np.isclose(outputs['dos_output_parameters']['emax'], 16.9)
    assert np.isclose(outputs['report']['emin'], -3.1)
    assert np.isclose(outputs['report']['emax'], 16.9)


def test_stub_from_scratch_window_mismatch_fails() -> None:
    """Diverging DOS/PROJWFC grids fail the mocked run with the single graph exit code."""
    dos = _dos_parameters().get_dict()
    dos['DOS'].update({'emin': -4.0, 'emax': 8.0})
    projwfc = _projwfc_parameters().get_dict()
    projwfc['PROJWFC'].update({'emin': -5.0, 'emax': 8.0})
    inputs = _stub_inputs()
    inputs.update({'dos_parameters': Dict(dos), 'projwfc_parameters': Dict(projwfc)})
    _outputs, node = run_get_node(stub_from_scratch, **inputs)
    assert not node.is_finished_ok
    assert node.exit_status == 400


def test_stub_from_scratch_nbands_exclusion_fails() -> None:
    """Both ``nbands_factor`` and ``SYSTEM.nbnd`` fail the mocked run."""
    nscf = _nscf_parameters().get_dict()
    nscf['SYSTEM']['nbnd'] = 10
    inputs = _stub_inputs()
    inputs.update({'nscf_parameters': Dict(nscf), 'nbands_factor': 1.5})
    _outputs, node = run_get_node(stub_from_scratch, **inputs)
    assert not node.is_finished_ok
    assert node.exit_status == 400


def test_stub_from_scratch_wrong_scf_calculation_fails() -> None:
    """A non-``scf`` SCF calculation fails the mocked run."""
    inputs = _stub_inputs()
    inputs.update({'scf_parameters': _nscf_parameters()})
    _outputs, node = run_get_node(stub_from_scratch, **inputs)
    assert not node.is_finished_ok
    assert node.exit_status == 400
