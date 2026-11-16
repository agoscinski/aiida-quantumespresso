############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Mocked tests for the native ``@graph`` PW relax/bands orchestration.

Run from the parent core checkout (the legacy ``ProtocolMixin`` still imports ``plumpy``, which the core
environment only provides as a transient overlay)::

    uv run --with plumpy pytest aiida-quantumespresso/tests/graphs/test_pw.py -q

The real graphs (:mod:`aiida_quantumespresso.workflows.graphs.pw`) are validated with ``build()`` and their
``@task`` adapters run for real through ``run_get_node`` (all local: no codes, no cluster). Full orchestration
runs against the test-local :class:`StubBase` stand-in, which mirrors the static routed surface of
:class:`PwBaseStep <aiida_quantumespresso.workflows.graphs.pw.PwBaseStep>` without submitting
``PwBaseWorkChain``. The stub graphs below mirror the real graph bodies line for line except for the placed step;
any drift between them is a bug in this file, while the real wiring is pinned by the ``build()`` tests. See
``reports/pw-migration.md`` for what the mocks do not cover (real base runs, ``501`` Pulay reruns, exit codes).
"""

import typing as t

import pytest

import aiida.engine

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida_quantumespresso.workflows.graphs.pw import (
    PwBaseStep,
    ensure_structure,
    prepare_kpoints_path,
    prepare_next_relax,
    pw_bands_graph,
    pw_relax_graph,
    resolve_bands_parameters,
    run_pw_base,
    summarize_relax,
)

from aiida.common.links import LinkType
from aiida.engine import WorkChain, calcfunction, graph, run_get_node, task
from aiida.orm import (
    BandsData,
    Bool,
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


@calcfunction
def _mock_structure(structure: StructureData, scale: Float) -> StructureData:
    """Return a copy of the structure with the cell scaled by ``scale``."""
    scaled = StructureData(cell=[[component * scale.value for component in vector] for vector in structure.cell])
    for site in structure.sites:
        scaled.append_atom(position=site.position, symbols=site.kind_name)
    return scaled


@calcfunction
def _mock_output_parameters(parameters: Dict) -> Dict:
    """Fabricate ``output_parameters`` from the input parameters."""
    directive = parameters.get_dict().get('DIRECTIVE', {})
    return Dict(
        {
            'number_of_bands': parameters.get_dict().get('SYSTEM', {}).get('nbnd', 8),
            'number_of_electrons': directive.get('electrons', 8),
        }
    )


@calcfunction
def _mock_remote_folder() -> RemoteData:
    """Return a ``RemoteData`` on a self-contained localhost stand-in."""
    existing = QueryBuilder().append(Computer, filters={'label': 'stub-graph-computer'}).first(flat=True)
    if existing is None:
        existing = Computer(
            label='stub-graph-computer',
            hostname='localhost',
            transport_type='core.local',
            scheduler_type='core.direct',
            workdir='/tmp',
        )
        existing.store()
    return RemoteData(computer=existing, remote_path='/tmp/stub')


@calcfunction
def _mock_bands() -> BandsData:
    """Return a minimal two-k-point band structure."""
    kpoints = KpointsData()
    kpoints.set_kpoints([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    bands = BandsData()
    bands.set_kpointsdata(kpoints)
    bands.set_bands([[0.0], [0.0]])
    return bands


class StubBase(WorkChain):
    """Mocked stand-in for ``PwBaseStep`` with the same static routing shape, minus codes and pseudos.

    ``Data`` outputs are fabricated by ``calcfunction`` collaborators (work chains cannot create nodes inline).
    ``emit_structure=False`` mimics steps that finish without an output structure. The ``marker`` input only
    exercises ``None`` wiring into optional ports.
    """

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        """Define the process specification."""
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=Dict)
        spec.input('scale', valid_type=Float, required=False)
        spec.input('emit_structure', valid_type=Bool, required=False)
        spec.input('kpoints', valid_type=KpointsData, required=False)
        spec.input('parent_folder', valid_type=RemoteData, required=False)
        spec.input('marker', valid_type=Dict, required=False)
        spec.output('output_structure', valid_type=StructureData, required=False)
        spec.output('output_parameters', valid_type=Dict)
        spec.output('remote_folder', valid_type=RemoteData)
        spec.output('output_band', valid_type=BandsData, required=False)
        spec.outline(cls.produce)

    def produce(self):  # type: ignore[no-untyped-def]
        """Emit the mocked outputs."""
        if self.inputs.get('emit_structure', Bool(True)).value:
            scale = self.inputs.scale if 'scale' in self.inputs else Float(1.0)
            self.out('output_structure', _mock_structure(self.inputs.structure, scale))
        self.out('output_parameters', _mock_output_parameters(self.inputs.parameters))
        self.out('remote_folder', _mock_remote_folder())
        self.out('output_band', _mock_bands())


stub_step = task(StubBase)


class StubRelaxOutputs(t.TypedDict):
    """Outputs of :func:`stub_relax_loop`."""

    output_structure: StructureData
    report: Dict


class StubBandsOutputs(t.TypedDict):
    """Outputs of :func:`stub_bands_path`."""

    band_structure: BandsData
    scf_parameters: Dict
    band_parameters: Dict
    primitive_structure: StructureData | None
    seekpath_parameters: Dict | None


@graph
def stub_init_guarded(
    structure: StructureData,
    parameters: Dict,
    scale: float = 1.0,
    emit_structure: bool = True,
    marker: Dict | None = None,
) -> StructureData:
    """Mirror ``init_relax_guarded`` with the mocked step."""
    initial = stub_step(
        structure=structure,
        parameters=parameters,
        scale=scale,
        emit_structure=emit_structure,
        marker=marker,
    )
    return ensure_structure(candidate=initial.output_structure)


@graph
def stub_relax_loop(
    structure: StructureData,
    parameters: Dict,
    max_iterations: int = 5,
    meta_convergence: bool = True,
    kpoints_distance: Float | None = None,
    scale: float = 1.0,
    emit_structure: bool = True,
    run_init: bool = False,
    marker: Dict | None = None,
) -> StubRelaxOutputs:
    """Mirror ``pw_relax_graph`` with the mocked step."""
    current = structure
    if run_init:
        current = stub_init_guarded(
            structure=current,
            parameters=parameters,
            scale=scale,
            emit_structure=emit_structure,
            marker=marker,
        )
    params = parameters
    attempts = 0
    rerun = True
    while rerun and attempts < max_iterations:
        step = stub_step(
            structure=current,
            parameters=params,
            scale=scale,
            emit_structure=emit_structure,
            marker=marker,
        )
        decision = prepare_next_relax(
            parameters=params,
            output_parameters=step.output_parameters,
            structure_before=current,
            structure_after=step.output_structure,
            kpoints_distance=kpoints_distance,
            force_parity=False,
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
def stub_bands_path(
    structure: StructureData,
    scf_parameters: Dict,
    bands_parameters: Dict,
    scf_kpoints: KpointsData | None = None,
    bands_kpoints: KpointsData | None = None,
    bands_kpoints_distance: Float | None = None,
    nbands_factor: Float | None = None,
) -> StubBandsOutputs:
    """Mirror ``pw_bands_graph`` with the mocked step."""
    path = prepare_kpoints_path(
        structure=structure, bands_kpoints=bands_kpoints, bands_kpoints_distance=bands_kpoints_distance
    )
    if bands_kpoints is None:
        scf_structure = path.primitive_structure
    else:
        scf_structure = structure
    scf = stub_step(structure=scf_structure, parameters=scf_parameters)
    resolved = resolve_bands_parameters(
        scf_output_parameters=scf.output_parameters, bands_parameters=bands_parameters, nbands_factor=nbands_factor
    )
    bands = stub_step(
        structure=scf_structure,
        parameters=resolved,
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


def _base_children(node) -> list:
    """Return the called ``StubBase`` descendants of a graph node."""
    return [child for child in node.called_descendants if child.process_label == 'StubBase']


def test_build_relax_graph() -> None:
    """The real relax graph validates without launching anything."""
    assert pw_relax_graph.build() is None


def test_build_bands_graph() -> None:
    """The real bands graph validates without launching anything."""
    assert pw_bands_graph.build() is None


@graph
def place_only(structure: StructureData, parameters: Dict, code: Code, pseudos: DataDict) -> Dict:
    """Place the real routing step with required inputs only."""
    done = run_pw_base(structure=structure, parameters=parameters, code=code, pseudos=pseudos)
    return done.output_parameters


def test_build_routed_step_placement() -> None:
    """The real routing step has a static contract a graph can place."""
    assert place_only.build() is None


def test_routing_step_spec() -> None:
    """The routing step routes the base inputs and exposes the base outputs."""
    spec = PwBaseStep.spec()
    assert set(spec.inputs) >= {
        'structure',
        'parameters',
        'code',
        'pseudos',
        'kpoints',
        'kpoints_distance',
        'settings',
        'parent_folder',
        'options',
    }
    assert spec.inputs['pseudos'].valid_type is DataDict
    assert not spec.inputs['kpoints'].required
    assert 'output_structure' in spec.outputs
    assert 'output_parameters' in spec.outputs
    assert 'remote_folder' in spec.outputs


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


def test_stub_relax_converged() -> None:
    """An unchanged mocked cell converges after one iteration and returns the summary node."""
    structure = _silicon()
    outputs, node = run_get_node(
        stub_relax_loop,
        structure=structure,
        parameters=_vc_relax_parameters(),
        max_iterations=5,
        meta_convergence=True,
        kpoints_distance=Float(0.5),
        scale=1.0,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict()['attempts'] == 1
    assert outputs['report'].get_dict()['status'] == 'converged'
    assert outputs['output_structure'].cell == structure.cell
    assert node.inputs.structure.uuid == structure.uuid
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    assert len(_base_children(node)) == 1
    summary_node = next(child for child in node.called_descendants if child.process_label == 'summarize_relax')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


def test_stub_relax_regrow_exhausted() -> None:
    """A contracting mocked cell exhausts the iteration budget."""
    outputs, node = run_get_node(
        stub_relax_loop,
        structure=_silicon(),
        parameters=_vc_relax_parameters(),
        max_iterations=2,
        meta_convergence=True,
        kpoints_distance=Float(0.5),
        scale=0.5,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict() == {
        'structure_uuid': outputs['output_structure'].uuid,
        'attempts': 2,
        'status': 'exhausted',
        'max_iterations': 2,
    }
    assert len(_base_children(node)) == 2


def test_stub_relax_no_meta_convergence() -> None:
    """A contracting mocked cell still converges in one iteration when meta-convergence is off."""
    outputs, node = run_get_node(
        stub_relax_loop,
        structure=_silicon(),
        parameters=_vc_relax_parameters(),
        max_iterations=5,
        meta_convergence=False,
        kpoints_distance=Float(0.5),
        scale=0.5,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict()['status'] == 'converged'
    assert outputs['report'].get_dict()['attempts'] == 1


def test_stub_relax_init_branch() -> None:
    """The initial relaxation runs only when requested."""
    common = {
        'structure': _silicon(),
        'parameters': _vc_relax_parameters(),
        'max_iterations': 5,
        'meta_convergence': True,
        'kpoints_distance': None,
        'scale': 1.0,
    }
    _outputs, without = run_get_node(stub_relax_loop, run_init=False, **common)
    assert without.is_finished_ok
    assert len(_base_children(without)) == 1

    _outputs, with_init = run_get_node(stub_relax_loop, run_init=True, **common)
    assert with_init.is_finished_ok
    assert len(_base_children(with_init)) == 2


def test_stub_relax_zero_budget() -> None:
    """A zero budget runs no iteration and reports exhaustion, unlike the legacy chain."""
    structure = _silicon()
    outputs, node = run_get_node(
        stub_relax_loop,
        structure=structure,
        parameters=_vc_relax_parameters(),
        max_iterations=0,
        kpoints_distance=None,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict()['attempts'] == 0
    assert outputs['report'].get_dict()['status'] == 'exhausted'
    assert outputs['output_structure'].uuid == structure.uuid


def test_stub_relax_missing_structure_fails() -> None:
    """A relaxing step without output structure fails the graph with the single graph exit code."""
    _outputs, node = run_get_node(
        stub_relax_loop,
        structure=_silicon(),
        parameters=_vc_relax_parameters(),
        max_iterations=2,
        emit_structure=False,
    )
    assert not node.is_finished_ok
    assert node.exit_status == 400


def test_stub_bands_seekpath() -> None:
    """The mocked bands path normalizes through SeekPath and returns every product."""
    outputs, node = run_get_node(
        stub_bands_path,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'DIRECTIVE': {'electrons': 16}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        bands_kpoints_distance=Float(0.3),
        nbands_factor=Float(1.5),
    )
    assert node.is_finished_ok
    assert isinstance(outputs['primitive_structure'], StructureData)
    assert isinstance(outputs['seekpath_parameters'], Dict)
    assert isinstance(outputs['band_structure'], BandsData)
    assert outputs['band_parameters'].get_dict()['number_of_bands'] == 12
    assert len(_base_children(node)) == 2


def test_stub_bands_explicit_kpoints() -> None:
    """Explicit bands k-points skip SeekPath and leave its outputs out."""
    kpoints = KpointsData()
    kpoints.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    outputs, node = run_get_node(
        stub_bands_path,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        bands_kpoints=kpoints,
    )
    assert node.is_finished_ok
    assert 'primitive_structure' not in outputs
    assert 'seekpath_parameters' not in outputs
    assert outputs['band_parameters'].get_dict()['number_of_bands'] == 8
