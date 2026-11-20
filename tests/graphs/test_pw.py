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
    TrajectoryData,
)

import aiida_quantumespresso.workflows.graphs.pw as pw_module
from aiida_quantumespresso.workflows.graphs.pw import (
    ensure_structure,
    prepare_kpoints_path,
    prepare_next_relax,
    pw_bands_graph,
    pw_relax_graph,
    require_base_finished,
    require_relax_base,
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
    output_kpoints: KpointsData
    output_atomic_occupations: Dict
    output_trajectory: TrajectoryData
    remote_folder: RemoteData
    report: Dict


def _stub_remote(path: str) -> RemoteData:
    """Return a stored ``RemoteData`` for the stub outputs."""
    return RemoteData(computer=_stub_computer(), remote_path=path)


@task
def pw_base_graph(
    code: Code,
    structure: StructureData,
    pseudos: DataDict,
    parameters: Dict,
    settings: Dict | None = None,
    kpoints: KpointsData | None = None,
    kpoints_distance: Float | None = None,
    kpoints_force_parity: bool = False,
    parent_folder: RemoteData | None = None,
    monitors: DataDict | None = None,
) -> StubPwOutputs:
    """Deterministic stand-in for ``pw_base_graph``: echo the band count, scale the cell, fabricate the rest."""
    import numpy as np

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
    trajectory = TrajectoryData()
    trajectory.set_trajectory(
        stepids=np.array([0]),
        cells=np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
        symbols=['Si'],
        positions=np.array([[[0.0, 0.0, 0.0]]]),
    )
    trajectory.set_array('stress', np.zeros((1, 3, 3)))
    report = Dict({'attempts': 1, 'status': 'finished'})
    kpoints_out = KpointsData()
    kpoints_out.set_kpoints([(0.0, 0.0, 0.0)])
    return {
        'output_parameters': output_parameters,
        'output_structure': scaled,
        'output_band': bands,
        'output_kpoints': kpoints_out,
        'output_atomic_occupations': Dict({'atomic_occupations': {'spin_up': [], 'spin_down': []}}),
        'output_trajectory': trajectory,
        'remote_folder': _stub_remote('/tmp/stub-pw-base'),
        'report': report,
    }


def _substitute_pw_base(monkeypatch) -> None:
    """Replace the native base placements with the ``@task`` stub.

    Inline aliases share one declaration; replace it and force recompilation.
    """
    monkeypatch.setattr(pw_module, 'pw_base_graph', pw_base_graph)
    for handle in (pw_module.pw_relax_graph, pw_module.pw_bands_graph, pw_module.init_relax_guarded):
        monkeypatch.setattr(handle, '_template', None)


def test_build_relax_graph() -> None:
    """The real relax graph validates without launching anything."""
    assert pw_relax_graph.build() is None


def test_build_relax_graph_output_contract() -> None:
    """The relax graph declares the legacy final-calculation outputs alongside structure and report."""
    from aiida.engine.processes.graphs.lowering import compile_graph

    assert set(compile_graph(pw_relax_graph).outputs) == {
        'output_structure',
        'output_parameters',
        'output_trajectory',
        'output_band',
        'output_kpoints',
        'output_atomic_occupations',
        'remote_folder',
        'report',
    }


def test_build_bands_graph() -> None:
    """The real bands graph validates without launching anything."""
    assert pw_bands_graph.build() is None


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


def test_relax_scopes_follow_authored_aliases() -> None:
    """The compiled relax graph addresses base runs through ``initial``/``main``.

    The check compiles the production graphs themselves (no copies): the
    initial branch nests the ``initial`` base alias inside the
    ``init_relax_guarded`` wrapper scope, the main loop places ``main`` with
    no generated suffix, and no placement calls ``pw_base_graph`` directly.
    Each base call site carries both the plain ``pw_calc`` and the monitored
    ``pw_calc_monitored`` (static ``monitors.stuck``) placements.
    """
    from aiida.engine.processes.graphs.lowering import compile_graph

    assert _calcjob_paths(compile_graph(pw_relax_graph)) == {
        ('init_relax_guarded', 'initial', 'pw_calc'),
        ('init_relax_guarded', 'initial', 'pw_calc_monitored'),
        ('main', 'pw_calc'),
        ('main', 'pw_calc_monitored'),
    }
    assert _calcjob_paths(compile_graph(pw_module.init_relax_guarded)) == {
        ('initial', 'pw_calc'),
        ('initial', 'pw_calc_monitored'),
    }


def test_bands_scopes_follow_authored_aliases() -> None:
    """The compiled bands graph addresses SCF/bands through ``scf``/``bands``.

        The check compiles the production graph itself (no copies): each stage
        descends exactly one level to the shared ``pw_calc``/``pw_calc_monitored``
    leaves with no generated numeric suffix and no direct ``pw_base_graph`` placement.
    """
    from aiida.engine.processes.graphs.lowering import compile_graph

    assert _calcjob_paths(compile_graph(pw_bands_graph)) == {
        ('scf', 'pw_calc'),
        ('scf', 'pw_calc_monitored'),
        ('bands', 'pw_calc'),
        ('bands', 'pw_calc_monitored'),
    }


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


def _converged_trajectory() -> TrajectoryData:
    """Return a trajectory whose pressure matches the default reference within tolerance."""
    import numpy as np

    trajectory = TrajectoryData()
    trajectory.set_trajectory(
        stepids=np.array([0]),
        cells=np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
        symbols=['Si'],
        positions=np.array([[[0.0, 0.0, 0.0]]]),
    )
    trajectory.set_array('stress', np.zeros((1, 3, 3)))
    return trajectory


def _stressed_trajectory(pressure: float = 10.0) -> TrajectoryData:
    """Return a trajectory whose pressure overshoots the default threshold."""
    import numpy as np

    trajectory = TrajectoryData()
    trajectory.set_trajectory(
        stepids=np.array([0]),
        cells=np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
        symbols=['Si'],
        positions=np.array([[[0.0, 0.0, 0.0]]]),
    )
    trajectory.set_array('stress', (np.eye(3) * pressure)[np.newaxis, :, :])
    return trajectory


def test_prepare_next_501_converged() -> None:
    """A ``501`` outcome within the Pulay pressure tolerance requests no rerun."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 9}),
        structure_before=_cubic(),
        structure_after=_cubic(),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=True,
        report=Dict({'attempts': 1, 'status': 'finished_ionic_convergence_reached_except_final_scf'}),
        output_trajectory=_converged_trajectory(),
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is False
    assert outputs['parameters'].get_dict()['SYSTEM']['nbnd'] == 9


def test_prepare_next_501_rerun() -> None:
    """A ``501`` outcome with Pulay stresses still present requests another iteration."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 9}),
        structure_before=_cubic(),
        structure_after=_cubic(),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=True,
        report=Dict({'attempts': 1, 'status': 'finished_ionic_convergence_reached_except_final_scf'}),
        output_trajectory=_stressed_trajectory(),
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is True


def test_prepare_next_501_meta_off_converged() -> None:
    """A ``501`` outcome with meta-convergence off requests no rerun without a stress check."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 8}),
        structure_before=_cubic(),
        structure_after=_cubic(),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=False,
        report=Dict({'attempts': 1, 'status': 'finished_ionic_convergence_reached_except_final_scf'}),
        output_trajectory=None,
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is False


def test_prepare_next_501_missing_trajectory_fails() -> None:
    """A ``501`` outcome without a trajectory fails loudly instead of retrying degraded."""
    with pytest.raises(ValueError, match='without an output trajectory'):
        run_get_node(
            prepare_next_relax,
            parameters=_vc_relax_parameters(),
            output_parameters=Dict({'number_of_bands': 8}),
            structure_before=_cubic(),
            structure_after=_cubic(),
            kpoints_distance=Float(0.5),
            force_parity=False,
            meta_convergence=True,
            report=Dict({'attempts': 1, 'status': 'finished_ionic_convergence_reached_except_final_scf'}),
            output_trajectory=None,
        )


def test_prepare_next_710_rejects() -> None:
    """A ``710`` outcome fails the relax decision instead of flowing downstream."""
    with pytest.raises(ValueError, match='electronic-convergence warning'):
        run_get_node(
            prepare_next_relax,
            parameters=_vc_relax_parameters(),
            output_parameters=Dict({'number_of_bands': 8}),
            structure_before=_cubic(),
            structure_after=_cubic(),
            kpoints_distance=Float(0.5),
            force_parity=False,
            meta_convergence=True,
            report=Dict({'attempts': 1, 'status': 'finished_electronic_convergence_warning'}),
            output_trajectory=_converged_trajectory(),
        )


def test_require_base_finished_ok() -> None:
    """A fully converged base report passes the bands guard."""
    _, node = run_get_node(require_base_finished, report=Dict({'attempts': 1, 'status': 'finished'}), stage='scf')
    assert node.is_finished_ok


def test_require_base_finished_rejects_501() -> None:
    """A considered-finished ``501`` report fails the bands guard."""
    with pytest.raises(ValueError, match='501|finished_ionic'):
        run_get_node(
            require_base_finished,
            report=Dict({'attempts': 1, 'status': 'finished_ionic_convergence_reached_except_final_scf'}),
            stage='scf',
        )


def test_require_base_finished_rejects_710() -> None:
    """A considered-finished ``710`` report fails the bands guard."""
    with pytest.raises(ValueError, match='710|finished_electronic'):
        run_get_node(
            require_base_finished,
            report=Dict({'attempts': 1, 'status': 'finished_electronic_convergence_warning'}),
            stage='bands',
        )


def test_require_relax_base_accepts_finished_and_501() -> None:
    """The initial-relax guard accepts fully converged and ``501`` reports."""
    _, ok_node = run_get_node(require_relax_base, report=Dict({'attempts': 1, 'status': 'finished'}))
    assert ok_node.is_finished_ok
    _, pulay_node = run_get_node(
        require_relax_base,
        report=Dict({'attempts': 1, 'status': 'finished_ionic_convergence_reached_except_final_scf'}),
    )
    assert pulay_node.is_finished_ok


def test_require_relax_base_rejects_710() -> None:
    """The initial-relax guard rejects a ``710`` report."""
    with pytest.raises(ValueError, match='electronic-convergence warning'):
        run_get_node(
            require_relax_base,
            report=Dict({'attempts': 1, 'status': 'finished_electronic_convergence_warning'}),
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
    stub = next(child for child in node.called_descendants if child.process_label == 'pw_base_graph')
    for name in (
        'output_parameters',
        'output_trajectory',
        'output_band',
        'output_kpoints',
        'output_atomic_occupations',
        'remote_folder',
    ):
        assert outputs[name].uuid == stub.outputs[name].uuid
    assert node.inputs.structure.uuid == structure.uuid
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    assert sum(1 for child in node.called_descendants if child.process_label == 'pw_base_graph') == 1
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
    assert sum(1 for child in node.called_descendants if child.process_label == 'pw_base_graph') == 2


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
    assert sum(1 for child in without.called_descendants if child.process_label == 'pw_base_graph') == 1

    _outputs, with_init = run_get_node(pw_relax_graph, structure=_silicon(), run_init_relax=True, **common)
    assert with_init.is_finished_ok
    assert sum(1 for child in with_init.called_descendants if child.process_label == 'pw_base_graph') == 2


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
    for name in (
        'output_parameters',
        'output_trajectory',
        'output_band',
        'output_kpoints',
        'output_atomic_occupations',
        'remote_folder',
    ):
        assert name not in outputs


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
    assert sum(1 for child in node.called_descendants if child.process_label == 'pw_base_graph') == 2


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


def test_require_base_finished_rejects_missing_status() -> None:
    """A provided bands report without ``'status'`` fails instead of defaulting to finished."""
    with pytest.raises(ValueError, match=r"carries no 'status'"):
        run_get_node(require_base_finished, report=Dict({'attempts': 1}), stage='scf')


def test_require_relax_base_rejects_missing_status() -> None:
    """A provided initial-relax report without ``'status'`` fails loudly."""
    with pytest.raises(ValueError, match=r"carries no 'status'"):
        run_get_node(require_relax_base, report=Dict({'attempts': 1}))


def test_require_relax_base_none_still_ok() -> None:
    """An absent initial-relax report stays a no-op for backwards compatibility."""
    _, node = run_get_node(require_relax_base, report=None)
    assert node.is_finished_ok


def test_prepare_next_relax_rejects_missing_status() -> None:
    """A provided relax report without ``'status'`` fails instead of assuming finished."""
    with pytest.raises(ValueError, match=r"carries no 'status'"):
        run_get_node(
            prepare_next_relax,
            parameters=_vc_relax_parameters(),
            output_parameters=Dict({'number_of_bands': 8}),
            structure_before=_cubic(),
            structure_after=_cubic(),
            kpoints_distance=Float(0.5),
            force_parity=False,
            meta_convergence=True,
            report=Dict({'attempts': 1}),
            output_trajectory=None,
        )


def test_prepare_next_relax_none_report_still_finished() -> None:
    """An absent relax report keeps the legacy finished default."""
    outputs, node = run_get_node(
        prepare_next_relax,
        parameters=_vc_relax_parameters(),
        output_parameters=Dict({'number_of_bands': 8}),
        structure_before=_cubic(),
        structure_after=_cubic(),
        kpoints_distance=Float(0.5),
        force_parity=False,
        meta_convergence=False,
        report=None,
        output_trajectory=None,
    )
    assert node.is_finished_ok
    assert outputs['rerun'].value is False


def test_relax_init_use_main_inputs_default_and_build() -> None:
    """Direct launches keep the ``True`` fallback default; both modes validate."""
    import inspect

    assert inspect.signature(pw_relax_graph).parameters['init_use_main_inputs'].default is True
    assert pw_relax_graph.build() is None


def test_resolve_init_inputs_falls_back_when_flag_set() -> None:
    """With the default mode unset ``init_*`` resolve to copies of the main values."""
    from aiida_quantumespresso.workflows.graphs.pw import resolve_init_inputs

    outputs, node = run_get_node(
        resolve_init_inputs,
        parameters=Dict({'CONTROL': {'calculation': 'relax'}}),
        settings=Dict({'cmdline': ['-nk', '2']}),
        kpoints=None,
        kpoints_distance=Float(0.15),
        force_parity=True,
        monitors=None,
        init_use_main_inputs=True,
        init_parameters=None,
        init_settings=None,
        init_kpoints=None,
        init_kpoints_distance=None,
        init_kpoints_force_parity=None,
        init_monitors=None,
    )
    assert node.is_finished_ok
    assert outputs['settings'].get_dict() == {'cmdline': ['-nk', '2']}
    assert outputs['kpoints_distance'].value == pytest.approx(0.15)
    assert outputs['force_parity'].value is True
    assert 'kpoints' not in outputs
    assert 'monitors' not in outputs


def test_resolve_init_inputs_verbatim_when_flag_unset() -> None:
    """Without inheritance an absent ``init_*`` stays absent instead of copying the main value."""
    from aiida_quantumespresso.workflows.graphs.pw import resolve_init_inputs

    outputs, node = run_get_node(
        resolve_init_inputs,
        parameters=Dict({'CONTROL': {'calculation': 'relax'}}),
        settings=Dict({'cmdline': ['-nk', '2']}),
        kpoints=None,
        kpoints_distance=Float(0.15),
        force_parity=True,
        monitors=None,
        init_use_main_inputs=False,
        init_parameters=Dict({'CONTROL': {'calculation': 'relax'}}),
        init_settings=None,
        init_kpoints=None,
        init_kpoints_distance=Float(0.3),
        init_kpoints_force_parity=None,
        init_monitors=None,
    )
    assert node.is_finished_ok
    assert 'settings' not in outputs
    assert outputs['kpoints_distance'].value == pytest.approx(0.3)
    assert outputs['force_parity'].value is False


def test_resolve_init_inputs_requires_parameters_when_flag_unset() -> None:
    """Without inheritance a missing ``init_parameters`` fails loudly."""
    from aiida_quantumespresso.workflows.graphs.pw import resolve_init_inputs

    with pytest.raises(ValueError, match='`init_parameters` is required'):
        run_get_node(
            resolve_init_inputs,
            parameters=Dict({'CONTROL': {'calculation': 'relax'}}),
            settings=None,
            kpoints=None,
            kpoints_distance=None,
            force_parity=False,
            monitors=None,
            init_use_main_inputs=False,
            init_parameters=None,
            init_settings=None,
            init_kpoints=None,
            init_kpoints_distance=None,
            init_kpoints_force_parity=None,
            init_monitors=None,
        )
