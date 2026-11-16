############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Production-graph tests for the native PDOS orchestration.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_pdos.py -q

``build()`` validates the real serial and parallel graphs without launching
anything. Runtime checks run the production graphs themselves with the external
``pw_base_graph`` subgraph and the ``dos.x``/``projwfc.x`` placements
substituted by deterministic ``@task`` stubs (no ``WorkChain``): substitutes
are swapped in via ``monkeypatch`` and the compiled templates are reset, so the
real NSCF-to-report wiring, window matching, and serial ordering execute.
``@task`` collaborators run for real through ``run_get_node`` (all local).
"""

import typing as t

import aiida.engine
import numpy as np
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida.common.links import LinkType
from aiida.engine import PortModel, run_get_node, task
from aiida.orm import (
    BandsData,
    Code,
    Computer,
    DataDict,
    Dict,
    Float,
    KpointsData,
    QueryBuilder,
    RemoteData,
    StructureData,
    XyData,
)
from aiida.orm.nodes.data.array.projection import ProjectionData

import aiida_quantumespresso.workflows.graphs.pdos as pdos_module
from aiida_quantumespresso.workflows.graphs.pdos import (
    extract_nscf_window,
    pdos_from_parent_parallel,
    pdos_from_parent_serial,
    pdos_from_scratch_parallel,
    pdos_from_scratch_serial,
    pdos_stage_parallel,
    pdos_stage_serial,
    require_scf_parameters,
    resolve_nscf_parameters,
    resolve_post_windows,
    summarize_pdos,
    sweep_remote,
    sweep_stage,
)

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


def _silicon() -> StructureData:
    """Return an unstored diamond silicon structure."""
    structure = StructureData(cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols='Si')
    return structure


def _scf_parameters() -> Dict:
    """Return SCF ``pw.parameters`` as the SCF step uses them."""
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

    existing = QueryBuilder().append(Computer, filters={'label': 'stub-pdos-computer'}).first(flat=True)
    if existing is not None:
        return existing
    computer = Computer(
        label='stub-pdos-computer',
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


def _stub_remote(path: str) -> RemoteData:
    """Return a stored ``RemoteData`` for the stub outputs."""
    return RemoteData(computer=_stub_computer(), remote_path=path)


class StubPwOutputs(t.TypedDict):
    """Fabricated base-run products of the ``pw_base_graph`` stub."""

    output_parameters: Dict
    output_band: BandsData
    remote_folder: RemoteData


class StubDosOutputs(t.TypedDict):
    """Fabricated post-processing products of the ``dos.x`` stub."""

    output_parameters: Dict
    output_dos: XyData
    remote_folder: RemoteData


class StubProjwfcOutputs(PortModel):
    """Fabricated post-processing products of the ``projwfc.x`` stub."""

    output_parameters: Dict
    Dos: XyData
    Ldos: XyData | None = None
    Pdos: XyData | None = None
    projections: ProjectionData | None = None
    projections_up: ProjectionData | None = None
    projections_down: ProjectionData | None = None
    bands: BandsData | None = None
    bands_up: BandsData | None = None
    bands_down: BandsData | None = None
    remote_folder: RemoteData


@task
def stub_pw_base(
    code: Code,
    structure: StructureData,
    pseudos: DataDict,
    parameters: Dict,
    settings: Dict | None = None,
    kpoints: KpointsData | None = None,
    kpoints_distance: Float | None = None,
    parent_folder: RemoteData | None = None,
    monitors: DataDict | None = None,
) -> StubPwOutputs:
    """Stand-in for ``pw_base_graph``: echo the band count, fabricate bands spanning the DOS window."""
    from aiida.orm import Float as OrmFloat  # noqa: F401  (keeps the stub import-local)

    del code, structure, pseudos, settings, kpoints, kpoints_distance, parent_folder, monitors
    nbnd = int(parameters.get_dict().get('SYSTEM', {}).get('nbnd', 8))
    kpoints_node = KpointsData()
    kpoints_node.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    bands = BandsData()
    bands.set_kpointsdata(kpoints_node)
    bands.set_bands([[-5.0], [8.0]])
    return {
        'output_parameters': Dict({'number_of_bands': nbnd, 'number_of_electrons': 8, 'fermi_energy': 6.9}),
        'output_band': bands,
        'remote_folder': _stub_remote('/tmp/stub-pw'),
    }


@task
def stub_dos(
    code: Code,
    parameters: Dict,
    parent_folder: RemoteData | None = None,
    settings: Dict | None = None,
) -> StubDosOutputs:
    """Stand-in for the ``dos.x`` calculation: echo the resolved energy window."""
    del code, parent_folder, settings
    params = parameters.get_dict()
    block = params.get('DOS', {})
    emin, emax = float(block.get('emin', -5.0)), float(block.get('emax', 8.0))
    output_dos = XyData()
    output_dos.set_x(np.linspace(emin, emax, num=2), 'energy', 'eV')
    output_dos.set_y([np.ones(2)], 'dos', 'states/eV')
    return {
        'output_parameters': Dict({'emin': emin, 'emax': emax, 'deltae': block.get('deltae', 0.01)}),
        'output_dos': output_dos,
        'remote_folder': _stub_remote('/tmp/stub-dos'),
    }


@task
def stub_projwfc(
    code: Code,
    parameters: Dict,
    parent_folder: RemoteData | None = None,
    settings: Dict | None = None,
) -> StubProjwfcOutputs:
    """Stand-in for the ``projwfc.x`` calculation: echo the resolved window as total DOS."""
    del code, parent_folder, settings
    params = parameters.get_dict()
    block = params.get('PROJWFC', {})
    emin, emax = float(block.get('emin', -5.0)), float(block.get('emax', 8.0))
    total_dos = XyData()
    total_dos.set_x(np.linspace(emin, emax, num=2), 'energy', 'eV')
    total_dos.set_y([np.ones(2)], 'dos', 'states/eV')
    return {
        'output_parameters': Dict({'emin': emin, 'emax': emax, 'deltae': block.get('deltae', 0.01)}),
        'Dos': total_dos,
        'remote_folder': _stub_remote('/tmp/stub-projwfc'),
    }


def _substitute_all(monkeypatch) -> None:
    """Replace the ``pw`` subgraph and both post-processing placements with ``@task`` stubs.

    The production graphs resolve placements by module-level name at
    compilation, so patching the module attributes redirects the real
    orchestration without copying its body. Compiled templates are cached on
    the graph handles, so they are reset (restored afterwards by
    ``monkeypatch``) to force recompilation against the substitutes.
    """
    monkeypatch.setattr(pdos_module, 'scf', stub_pw_base)
    monkeypatch.setattr(pdos_module, 'nscf', stub_pw_base)
    monkeypatch.setattr(pdos_module, 'dos_calc', stub_dos)
    monkeypatch.setattr(pdos_module, 'projwfc_calc', stub_projwfc)
    for handle in (
        pdos_module.pdos_stage_parallel,
        pdos_module.pdos_stage_serial,
        pdos_module.pdos_from_scratch_parallel,
        pdos_module.pdos_from_scratch_serial,
        pdos_module.pdos_from_parent_parallel,
        pdos_module.pdos_from_parent_serial,
    ):
        monkeypatch.setattr(handle, '_template', None)


def test_build_stage_graphs() -> None:
    """The real NSCF-to-report stages validate without launching anything."""
    assert pdos_stage_parallel.build() is None
    assert pdos_stage_serial.build() is None


def test_build_from_scratch_graphs() -> None:
    """The real from-scratch graphs validate without launching anything."""
    assert pdos_from_scratch_parallel.build() is None
    assert pdos_from_scratch_serial.build() is None


def test_build_from_parent_graphs() -> None:
    """The real from-parent graphs validate without launching anything."""
    assert pdos_from_parent_parallel.build() is None
    assert pdos_from_parent_serial.build() is None


def _calcjob_paths(template) -> set:
    """Return every ``CalcJob`` placement path of a compiled template.

    Each path descends one level per authored subgraph callee alias and ends
    at the ``CalcJob`` placement alias; branch and loop control frames are
    transparent and contribute no level of their own.
    """
    from aiida.engine import CalcJob
    from aiida.engine.processes.graphs.ir import BranchControl, LoopControl, ProcessTask, SubgraphTask

    found = set()

    def _visit(current, scope: tuple) -> None:
        if isinstance(current, ProcessTask):
            if issubclass(current.spec.process_class, CalcJob):
                found.add((*scope, current.name))
        elif isinstance(current, SubgraphTask):
            for nested in current.body.tasks:
                _visit(nested, (*scope, current.name))
        elif isinstance(current, BranchControl):
            for body in current.branches:
                for nested in body.tasks:
                    _visit(nested, scope)
        elif isinstance(current, LoopControl):
            for nested in current.body.tasks:
                _visit(nested, scope)

    for current in template.tasks:
        _visit(current, ())
    return found


def test_post_placements_keep_task_metadata_names() -> None:
    """The calculation placements keep their names so one nested scope entry covers every graph."""
    from aiida.engine.processes.graphs.ir import ProcessTask, SubgraphTask
    from aiida.engine.processes.graphs.lowering import compile_graph

    def _process_tasks(template) -> list:
        """Return every process task of a template, including nested bodies."""
        found = []

        def _visit(current) -> None:
            if isinstance(current, ProcessTask):
                found.append(current)
            elif isinstance(current, SubgraphTask):
                for nested in current.body.tasks:
                    _visit(nested)

        for current in template.tasks:
            _visit(current)
        return found

    names = set()
    for target in (pdos_stage_parallel, pdos_stage_serial, pdos_from_scratch_parallel, pdos_from_parent_parallel):
        for placement in _process_tasks(compile_graph(target)):
            if placement.spec.identifier in ('DosCalculation', 'ProjwfcCalculation'):
                names.add(placement.name)
    assert names == {'dos_calc', 'projwfc_calc'}


def test_stage_scopes_follow_authored_aliases() -> None:
    """The compiled PDOS stages address NSCF/DOS/PROJWFC through authored aliases.

    The checks compile the production graphs themselves (no copies): the NSCF
    base run descends through the ``nscf`` alias to the shared ``pw_calc``
    leaf, the post-processing placements keep their stable names, and no
    placement calls ``pw_base_graph`` directly or carries a generated suffix.
    """
    from aiida.engine.processes.graphs.lowering import compile_graph

    expected_stage = {('nscf', 'pw_calc'), ('dos_calc',), ('projwfc_calc',)}
    assert _calcjob_paths(compile_graph(pdos_stage_parallel)) == expected_stage
    assert _calcjob_paths(compile_graph(pdos_stage_serial)) == expected_stage


def test_roots_nest_stage_beside_scf_scope() -> None:
    """The compiled from-scratch roots nest the stage scope beside the SCF scope.

    The from-parent roots carry only the stage scope. Both variants address
    every ``CalcJob`` placement exactly once with no generated suffixes.
    """
    from aiida.engine.processes.graphs.lowering import compile_graph

    expected_stage = {('nscf', 'pw_calc'), ('dos_calc',), ('projwfc_calc',)}
    for root, stage in (
        (pdos_from_scratch_parallel, 'pdos_stage_parallel'),
        (pdos_from_scratch_serial, 'pdos_stage_serial'),
    ):
        assert _calcjob_paths(compile_graph(root)) == {
            ('scf', 'pw_calc'),
            *[(stage, *path) for path in expected_stage],
        }
    for root, stage in (
        (pdos_from_parent_parallel, 'pdos_stage_parallel'),
        (pdos_from_parent_serial, 'pdos_stage_serial'),
    ):
        assert _calcjob_paths(compile_graph(root)) == {(stage, *path) for path in expected_stage}


def test_serial_stage_orders_dos_before_projwfc() -> None:
    """The serial stage wires an unconditional ``after`` edge from the DOS sweep into PROJWFC."""
    from aiida.engine.processes.graphs.lowering import compile_graph

    parallel_edges = {(edge.source, edge.target) for edge in compile_graph(pdos_stage_parallel).dependencies}
    assert ('sweep_remote', 'projwfc_calc') not in parallel_edges
    serial_edges = {(edge.source, edge.target) for edge in compile_graph(pdos_stage_serial).dependencies}
    assert ('sweep_remote', 'projwfc_calc') in serial_edges


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


def _scratch_inputs(pw_code, dos_code, projwfc_code, **overrides) -> dict:
    """Return representative from-scratch inputs against the stub placements."""
    inputs: dict = {
        'structure': _silicon(),
        'pw_code': pw_code,
        'pseudos': DataDict(),
        'scf_parameters': _scf_parameters(),
        'nscf_parameters': _nscf_parameters(),
        'dos_code': dos_code,
        'dos_parameters': _dos_parameters(),
        'projwfc_code': projwfc_code,
        'projwfc_parameters': _projwfc_parameters(),
    }
    inputs.update(overrides)
    return inputs


def _codes(aiida_code_installed, suffix: str):
    """Return installed stub codes for the pw/dos/projwfc placements."""
    return (
        aiida_code_installed(label=f'test.pdos-pw-{suffix}', default_calc_job_plugin='quantumespresso.pw'),
        aiida_code_installed(label=f'test.pdos-dos-{suffix}', default_calc_job_plugin='quantumespresso.dos'),
        aiida_code_installed(label=f'test.pdos-projwfc-{suffix}', default_calc_job_plugin='quantumespresso.projwfc'),
    )


def test_production_from_scratch_parallel(monkeypatch, aiida_code_installed) -> None:
    """The production from-scratch parallel graph finishes SCF, NSCF, and both post-processing steps, then reports."""
    _substitute_all(monkeypatch)
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'par')
    inputs = _scratch_inputs(pw_code, dos_code, projwfc_code)
    structure = inputs['structure']
    outputs, node = run_get_node(pdos_from_scratch_parallel, **inputs)
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
    assert sum(1 for child in node.called_descendants if child.process_label == 'stub_pw_base') == 2
    summary_node = next(child for child in node.called_descendants if child.process_label == 'summarize_pdos')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


def test_production_from_scratch_serial(monkeypatch, aiida_code_installed) -> None:
    """The production serial root reports the serial mode and passes through the DOS sweep."""
    _substitute_all(monkeypatch)
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'ser')
    outputs, node = run_get_node(
        pdos_from_scratch_serial, clean_workdir=True, **_scratch_inputs(pw_code, dos_code, projwfc_code)
    )
    assert node.is_finished_ok
    assert outputs['report']['mode'] == 'serial'
    assert outputs['report']['status'] == 'completed'
    assert sum(1 for child in node.called_descendants if child.process_label == 'stub_pw_base') == 2
    assert sum(1 for child in node.called_descendants if child.process_label == 'sweep_remote') >= 2


def test_production_from_parent_nbands_factor(monkeypatch, aiida_code_installed) -> None:
    """The production from-parent graph applies the band-count rule from the caller-provided parent outputs."""
    _substitute_all(monkeypatch)
    parent_folder = _stub_remote('/tmp/stub-parent')
    parent_folder.store()
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'nbands')
    inputs = _scratch_inputs(pw_code, dos_code, projwfc_code)
    del inputs['scf_parameters']
    inputs.update(
        {
            'parent_folder': parent_folder,
            'parent_output_parameters': Dict({'number_of_bands': 8, 'number_of_electrons': 10}),
            'nbands_factor': 1.5,
        }
    )
    # The from-parent roots take no SCF inputs; drop the SCF-only keys.
    for key in ('pw_code', 'pseudos', 'scf_parameters'):
        pass
    outputs, node = run_get_node(pdos_from_parent_parallel, **inputs)
    assert node.is_finished_ok
    assert outputs['nscf_output_parameters']['number_of_bands'] == 9
    assert outputs['report']['nbands'] == 9
    assert outputs['report']['stages'] == 1
    assert sum(1 for child in node.called_descendants if child.process_label == 'stub_pw_base') == 1


def test_production_from_parent_energy_range(monkeypatch, aiida_code_installed) -> None:
    """The production from-parent graph covers the Fermi-relative range."""
    _substitute_all(monkeypatch)
    parent_folder = _stub_remote('/tmp/stub-parent')
    parent_folder.store()
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'erange')
    inputs = _scratch_inputs(pw_code, dos_code, projwfc_code)
    del inputs['scf_parameters']
    inputs.update({'parent_folder': parent_folder, 'energy_range_vs_fermi': [-10.0, 10.0]})
    outputs, node = run_get_node(pdos_from_parent_serial, **inputs)
    assert node.is_finished_ok
    assert np.isclose(outputs['dos_output_parameters']['emin'], -3.1)
    assert np.isclose(outputs['dos_output_parameters']['emax'], 16.9)
    assert np.isclose(outputs['report']['emin'], -3.1)
    assert np.isclose(outputs['report']['emax'], 16.9)


def test_production_from_scratch_window_mismatch_fails(monkeypatch, aiida_code_installed) -> None:
    """Diverging DOS/PROJWFC grids fail the production run with the single graph exit code."""
    _substitute_all(monkeypatch)
    dos = _dos_parameters().get_dict()
    dos['DOS'].update({'emin': -4.0, 'emax': 8.0})
    projwfc = _projwfc_parameters().get_dict()
    projwfc['PROJWFC'].update({'emin': -5.0, 'emax': 8.0})
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'mismatch')
    inputs = _scratch_inputs(pw_code, dos_code, projwfc_code)
    inputs.update({'dos_parameters': Dict(dos), 'projwfc_parameters': Dict(projwfc)})
    _outputs, node = run_get_node(pdos_from_scratch_parallel, **inputs)
    assert not node.is_finished_ok
    assert node.exit_status == 400


def test_production_from_scratch_nbands_exclusion_fails(monkeypatch, aiida_code_installed) -> None:
    """Both ``nbands_factor`` and ``SYSTEM.nbnd`` fail the production run."""
    _substitute_all(monkeypatch)
    nscf = _nscf_parameters().get_dict()
    nscf['SYSTEM']['nbnd'] = 10
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'excl')
    inputs = _scratch_inputs(pw_code, dos_code, projwfc_code)
    inputs.update({'nscf_parameters': Dict(nscf), 'nbands_factor': 1.5})
    _outputs, node = run_get_node(pdos_from_scratch_parallel, **inputs)
    assert not node.is_finished_ok
    assert node.exit_status == 400


def test_production_from_scratch_wrong_scf_calculation_fails(monkeypatch, aiida_code_installed) -> None:
    """A non-``scf`` SCF calculation fails the production run."""
    _substitute_all(monkeypatch)
    pw_code, dos_code, projwfc_code = _codes(aiida_code_installed, 'scf')
    inputs = _scratch_inputs(pw_code, dos_code, projwfc_code)
    inputs.update({'scf_parameters': _nscf_parameters()})
    _outputs, node = run_get_node(pdos_from_scratch_parallel, **inputs)
    assert not node.is_finished_ok
    assert node.exit_status == 400
