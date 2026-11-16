############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Production-graph tests for the native PW relax/bands orchestration.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_pw.py -q

``build()`` validates the real graphs without launching anything. Runtime
checks run the production graphs themselves with the external ``pw_base_graph``
subgraph substituted by a deterministic ``@task`` stub (no ``WorkChain``): the
stub is swapped in via ``monkeypatch`` and the compiled templates are reset, so
the real retry loop, rebinding, and guards execute. ``@task`` collaborators run
for real through ``run_get_node`` (all local: no codes, no cluster).
"""

import typing as t

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida.common.links import LinkType
from aiida.engine import run_get_node, task
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
)

import aiida_quantumespresso.workflows.graphs.pw as pw_module
from aiida_quantumespresso.workflows.graphs.pw import (
    ensure_structure,
    prepare_kpoints_path,
    prepare_next_relax,
    pw_bands_graph,
    pw_relax_graph,
    resolve_bands_parameters,
    summarize_relax,
)

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


def _silicon() -> StructureData:
    """Return an unstored primitive diamond silicon structure."""
    structure = StructureData(cell=[[2.715, 2.715, 0.0], [0.0, 2.715, 2.715], [2.715, 0.0, 2.715]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols='Si')
    return structure


def _cubic(element: str = 'Si', lattice: float = 10.0) -> StructureData:
    """Return an unstored single-atom cubic structure with the given lattice parameter."""
    structure = StructureData(
        cell=[[lattice, 0.0, 0.0], [0.0, lattice, 0.0], [0.0, 0.0, lattice]],
    )
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols=element)
    return structure


def _vc_relax_parameters() -> Dict:
    """Return ``vc-relax`` parameters with a mesh density, as the loop tests use them."""
    return Dict({'CONTROL': {'calculation': 'vc-relax'}, 'SYSTEM': {}, 'CELL': {}, 'DIRECTIVE': {}})


def _stub_computer() -> Computer:
    """Return a stored and configured localhost computer for stub remote folders."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import AuthInfo, User

    existing = QueryBuilder().append(Computer, filters={'label': 'stub-pw-computer'}).first(flat=True)
    if existing is not None:
        return existing
    computer = Computer(
        label='stub-pw-computer',
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
    """Fabricated products of the ``pw_base_graph`` stub."""

    output_parameters: Dict
    output_structure: StructureData
    output_band: BandsData
    remote_folder: RemoteData


def _stub_remote(path: str) -> RemoteData:
    """Return a stored ``RemoteData`` for the stub outputs."""
    return RemoteData(computer=_stub_computer(), remote_path=path)


@task
def stub_pw_base(
    code: Code,
    structure: StructureData,
    pseudos: DataDict,
    parameters: Dict,
    settings: Dict | None = None,
    kpoints: KpointsData | None = None,
    kpoints_distance: Float | None = None,
    kpoints_force_parity: bool = False,
    parent_folder: RemoteData | None = None,
) -> StubPwOutputs:
    """Deterministic stand-in for ``pw_base_graph``: echo the band count, scale the cell, fabricate the rest."""
    params = parameters.get_dict()
    scale = float(params.get('DIRECTIVE', {}).get('scale', 1.0))
    scaled = StructureData(
        cell=[[component * scale for component in vector] for vector in structure.cell],
    )
    for site in structure.sites:
        scaled.append_atom(position=tuple(site.position), symbols=site.kind_name)
    nbnd = int(params.get('SYSTEM', {}).get('nbnd', 8))
    nelectron = int(params.get('DIRECTIVE', {}).get('electrons', 8))
    output_parameters = Dict({'number_of_bands': nbnd, 'number_of_electrons': nelectron})
    kpoints_node = KpointsData()
    kpoints_node.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    bands = BandsData()
    bands.set_kpointsdata(kpoints_node)
    bands.set_bands([[0.0], [0.0]])
    return {
        'output_parameters': output_parameters,
        'output_structure': scaled,
        'output_band': bands,
        'remote_folder': _stub_remote('/tmp/stub-pw-base'),
    }


def _substitute_pw_base(monkeypatch) -> None:
    """Replace the native ``pw_base_graph`` placement with the ``@task`` stub.

    The production graphs resolve placements by module-level name at
    compilation, so patching ``pw_module.pw_base_graph`` redirects the real
    orchestration without copying its body. Compiled templates are cached on
    the graph handles, so they are reset (restored afterwards by
    ``monkeypatch``) to force recompilation against the substitute.
    """
    monkeypatch.setattr(pw_module, 'pw_base_graph', stub_pw_base)
    for handle in (pw_module.pw_relax_graph, pw_module.pw_bands_graph, pw_module.init_relax_guarded):
        monkeypatch.setattr(handle, '_template', None)


def test_build_relax_graph() -> None:
    """The real relax graph validates without launching anything."""
    assert pw_relax_graph.build() is None


def test_build_bands_graph() -> None:
    """The real bands graph validates without launching anything."""
    assert pw_bands_graph.build() is None


def test_native_placement_needs_task_metadata() -> None:
    """The relax graph places the native base subgraph under its agreed name."""
    from aiida.engine.processes.graphs.ir import LoopControl, SubgraphTask
    from aiida.engine.processes.graphs.lowering import compile_graph

    def _subgraphs(template) -> list:
        """Return every subgraph task of a template, including loop bodies."""
        found = []

        def _visit(current) -> None:
            if isinstance(current, SubgraphTask):
                found.append(current)
            if isinstance(current, LoopControl):
                for nested in current.body.tasks:
                    _visit(nested)

        for current in template.tasks:
            _visit(current)
        return found

    names = set()
    for target in (pw_relax_graph, pw_bands_graph):
        for placement in _subgraphs(compile_graph(target)):
            names.add(placement.name)
    assert 'pw_base_graph' in names


def test_prepare_next_converged() -> None:
    """An unchanged cell with a mesh density requests no rerun and carries the band count."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 12}),
        structure_before=_cubic(),
        structure_after=_cubic(),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=True,
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is False
    assert outputs['parameters'].get_dict()['SYSTEM']['nbnd'] == 12


def test_prepare_next_regrow() -> None:
    """A contracted cell with a mesh density requests another iteration."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 8}),
        structure_before=_cubic(lattice=20.0),
        structure_after=_cubic(lattice=10.0),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=True,
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is True


def test_prepare_next_meta_convergence_off() -> None:
    """A grown cell requests no rerun when meta-convergence is disabled."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 8}),
        structure_before=_cubic(lattice=20.0),
        structure_after=_cubic(lattice=10.0),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=False,
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is False


def test_prepare_next_scf_without_structure() -> None:
    """An ``scf`` step without output structure ends the loop instead of failing."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}}),
        output_parameters=Dict({'number_of_bands': 8}),
        structure_before=_cubic(),
        structure_after=None,
        kpoints_distance=None,
        force_parity=False,
        meta_convergence=True,
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is False


def test_prepare_next_relax_without_structure_fails() -> None:
    """A relaxing step without output structure fails the decision task."""
    with pytest.raises(ValueError, match='without an output structure'):
        run_get_node(
            prepare_next_relax,
            parameters=_vc_relax_parameters(),
            output_parameters=Dict({'number_of_bands': 8}),
            structure_before=_cubic(),
            structure_after=None,
            kpoints_distance=None,
            force_parity=False,
            meta_convergence=True,
        )


def test_resolve_bands_factor() -> None:
    """The ``nbands_factor`` rule takes the maximum of the three candidate counts."""
    outputs, node = run_get_node(
        resolve_bands_parameters,
        scf_output_parameters=Dict({'number_of_bands': 8, 'number_of_electrons': 16}),
        bands_parameters=Dict({'CONTROL': {}, 'SYSTEM': {}}),
        nbands_factor=Float(1.5),
    )
    assert node.is_finished_ok
    assert outputs.get_dict()['SYSTEM']['nbnd'] == 12


def test_resolve_bands_mutual_exclusion_fails() -> None:
    """Both ``nbands_factor`` and ``SYSTEM.nbnd`` fail the resolution task."""
    with pytest.raises(ValueError, match='nbands_factor'):
        run_get_node(
            resolve_bands_parameters,
            scf_output_parameters=Dict({'number_of_bands': 8, 'number_of_electrons': 16}),
            bands_parameters=Dict({'CONTROL': {}, 'SYSTEM': {'nbnd': 10}}),
            nbands_factor=Float(1.5),
        )


def test_resolve_bands_setdefault() -> None:
    """Without a factor the SCF band count fills an unset ``SYSTEM.nbnd``."""
    outputs, node = run_get_node(
        resolve_bands_parameters,
        scf_output_parameters=Dict({'number_of_bands': 8, 'number_of_electrons': 16}),
        bands_parameters=Dict({'CONTROL': {}, 'SYSTEM': {}}),
        nbands_factor=None,
    )
    assert node.is_finished_ok
    assert outputs.get_dict()['SYSTEM']['nbnd'] == 8


def test_prepare_path_explicit() -> None:
    """Explicit bands k-points pass through with empty SeekPath leaves."""
    kpoints = KpointsData()
    kpoints.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
    outputs, node = run_get_node(
        prepare_kpoints_path, structure=_silicon(), bands_kpoints=kpoints, bands_kpoints_distance=None
    )
    assert node.is_finished_ok
    assert outputs['explicit_kpoints'].get_kpoints().tolist() == kpoints.get_kpoints().tolist()
    assert 'primitive_structure' not in outputs
    assert 'seekpath_parameters' not in outputs


def test_prepare_path_seekpath() -> None:
    """Without explicit k-points the structure is normalized through SeekPath."""
    outputs, node = run_get_node(
        prepare_kpoints_path, structure=_silicon(), bands_kpoints=None, bands_kpoints_distance=Float(0.3)
    )
    assert node.is_finished_ok
    assert isinstance(outputs['primitive_structure'], StructureData)
    assert isinstance(outputs['explicit_kpoints'], KpointsData)
    assert isinstance(outputs['seekpath_parameters'], Dict)


def test_prepare_path_both_kpoints_fails() -> None:
    """Both k-point inputs fail the path task, mirroring the legacy validator."""
    kpoints = KpointsData()
    kpoints.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
    with pytest.raises(ValueError, match='bands_kpoints'):
        run_get_node(
            prepare_kpoints_path, structure=_silicon(), bands_kpoints=kpoints, bands_kpoints_distance=Float(0.3)
        )


def test_ensure_structure() -> None:
    """The guard passes structures through and fails on ``None``."""
    structure = _silicon()
    outputs, node = run_get_node(ensure_structure, candidate=structure)
    assert node.is_finished_ok
    assert outputs.cell == structure.cell

    with pytest.raises(ValueError, match='without an output structure'):
        run_get_node(ensure_structure, candidate=None)


def test_summarize_relax() -> None:
    """The summary records structure identity, attempts, and status."""
    structure = _silicon()
    outputs, node = run_get_node(summarize_relax, structure=structure, attempts=2, status='exhausted', max_iterations=2)
    assert node.is_finished_ok
    assert outputs.get_dict() == {
        'structure_uuid': structure.uuid,
        'attempts': 2,
        'status': 'exhausted',
        'max_iterations': 2,
    }


def test_production_relax_converged(monkeypatch, aiida_code_installed) -> None:
    """The production relax graph converges after one stub iteration and returns the summary node."""
    _substitute_pw_base(monkeypatch)
    structure = _silicon()
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=structure,
        parameters=_vc_relax_parameters(),
        code=aiida_code_installed(label='test.relax-pw', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=DataDict(),
        max_iterations=5,
        meta_convergence=True,
        kpoints_distance=Float(0.5),
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict()['attempts'] == 1
    assert outputs['report'].get_dict()['status'] == 'converged'
    assert outputs['output_structure'].cell == structure.cell
    assert node.inputs.structure.uuid == structure.uuid
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    assert sum(1 for child in node.called_descendants if child.process_label == 'stub_pw_base') == 1
    summary_node = next(child for child in node.called_descendants if child.process_label == 'summarize_relax')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


def test_production_relax_regrow_exhausted(monkeypatch, aiida_code_installed) -> None:
    """A contracting stub cell exhausts the production iteration budget."""
    _substitute_pw_base(monkeypatch)
    params = _vc_relax_parameters().get_dict()
    params['DIRECTIVE'] = {'scale': 0.5}
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=_silicon(),
        parameters=Dict(params),
        code=aiida_code_installed(label='test.relax-regrow', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=DataDict(),
        max_iterations=2,
        meta_convergence=True,
        kpoints_distance=Float(0.5),
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict() == {
        'structure_uuid': outputs['output_structure'].uuid,
        'attempts': 2,
        'status': 'exhausted',
        'max_iterations': 2,
    }
    assert sum(1 for child in node.called_descendants if child.process_label == 'stub_pw_base') == 2


def test_production_relax_no_meta_convergence(monkeypatch, aiida_code_installed) -> None:
    """A contracting stub cell still converges in one iteration when meta-convergence is off."""
    _substitute_pw_base(monkeypatch)
    params = _vc_relax_parameters().get_dict()
    params['DIRECTIVE'] = {'scale': 0.5}
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=_silicon(),
        parameters=Dict(params),
        code=aiida_code_installed(label='test.relax-nometa', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=DataDict(),
        max_iterations=5,
        meta_convergence=False,
        kpoints_distance=Float(0.5),
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict()['status'] == 'converged'
    assert outputs['report'].get_dict()['attempts'] == 1


def test_production_relax_init_branch(monkeypatch, aiida_code_installed) -> None:
    """The production initial relaxation runs only when requested."""
    _substitute_pw_base(monkeypatch)
    common = {
        'parameters': _vc_relax_parameters(),
        'code': aiida_code_installed(label='test.relax-init', default_calc_job_plugin='quantumespresso.pw'),
        'pseudos': DataDict(),
        'max_iterations': 5,
        'meta_convergence': True,
        'kpoints_distance': None,
    }
    _outputs, without = run_get_node(pw_relax_graph, structure=_silicon(), run_init_relax=False, **common)
    assert without.is_finished_ok
    assert sum(1 for child in without.called_descendants if child.process_label == 'stub_pw_base') == 1

    _outputs, with_init = run_get_node(pw_relax_graph, structure=_silicon(), run_init_relax=True, **common)
    assert with_init.is_finished_ok
    assert sum(1 for child in with_init.called_descendants if child.process_label == 'stub_pw_base') == 2


def test_production_relax_zero_budget(monkeypatch, aiida_code_installed) -> None:
    """A zero budget runs no iteration and reports exhaustion."""
    _substitute_pw_base(monkeypatch)
    structure = _silicon()
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=structure,
        parameters=_vc_relax_parameters(),
        code=aiida_code_installed(label='test.relax-zero', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=DataDict(),
        max_iterations=0,
        kpoints_distance=None,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict()['attempts'] == 0
    assert outputs['report'].get_dict()['status'] == 'exhausted'
    assert outputs['output_structure'].uuid == structure.uuid


def test_production_bands_seekpath(monkeypatch, aiida_code_installed) -> None:
    """The production bands graph normalizes through SeekPath and returns every product."""
    _substitute_pw_base(monkeypatch)
    outputs, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'DIRECTIVE': {'electrons': 16}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=aiida_code_installed(label='test.bands-scf', default_calc_job_plugin='quantumespresso.pw'),
        bands_code=aiida_code_installed(label='test.bands-bands', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=DataDict(),
        bands_kpoints_distance=Float(0.3),
        nbands_factor=Float(1.5),
    )
    assert node.is_finished_ok
    assert isinstance(outputs['primitive_structure'], StructureData)
    assert isinstance(outputs['seekpath_parameters'], Dict)
    assert isinstance(outputs['band_structure'], BandsData)
    assert outputs['band_parameters'].get_dict()['number_of_bands'] == 12
    assert sum(1 for child in node.called_descendants if child.process_label == 'stub_pw_base') == 2


def test_production_bands_explicit_kpoints(monkeypatch, aiida_code_installed) -> None:
    """Explicit bands k-points skip SeekPath and leave its outputs out."""
    _substitute_pw_base(monkeypatch)
    kpoints = KpointsData()
    kpoints.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    outputs, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=aiida_code_installed(label='test.bands-scf-exp', default_calc_job_plugin='quantumespresso.pw'),
        bands_code=aiida_code_installed(label='test.bands-bands-exp', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=DataDict(),
        bands_kpoints=kpoints,
    )
    assert node.is_finished_ok
    assert 'primitive_structure' not in outputs
    assert 'seekpath_parameters' not in outputs
    assert outputs['band_parameters'].get_dict()['number_of_bands'] == 8
