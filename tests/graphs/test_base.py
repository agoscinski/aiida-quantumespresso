"""Tests for the graph alternatives to the Quantum ESPRESSO ``base`` restart work chains.

Run from the repository root (``uv run`` from the root environment, never bare
``python``/``pip``)::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \
        pytest aiida-quantumespresso/tests/graphs/test_base.py -x --tb=short

``build()`` checks run without launching anything; the runtime checks use a ``presto``
profile and deterministic stubs that share :mod:`base` error codes and handler tasks,
mirroring ``tests/engine/processes/graphs/programs/matched_restart.py``. Two runtime
layers exist: ``restartable_pw`` mirrors the ``pw`` graph body line for line but
places a counter-driven stub (cheap handler coverage), while the ``*_real_wiring``
tests substitute a shape-compatible stub for the real ``pw_calc`` placement via
``monkeypatch`` and run the production ``pw_base_graph`` itself, so the retry loop,
rebinding, and ``require_*`` guards execute for real. A mapped failure exposes only
its ``error`` arm downstream, so any claimed restart needing the failed run's
``remote_folder``/``output_structure`` must fail loudly instead of retrying degraded.
"""

import io

import pytest

import aiida.engine

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida_quantumespresso.workflows.graphs import base
from aiida_quantumespresso.workflows.graphs.base import NebErrors, PhErrors, PwErrors

from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import calcfunction, graph, run_get_node, task
from aiida.engine.processes.workchains.workchain import WorkChain as WorkChainBase

pytestmark = pytest.mark.presto

PW_PARAMETERS = {
    'CONTROL': {'calculation': 'relax'},
    'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0},
    'ELECTRONS': {},
}

PH_PARAMETERS = {
    'INPUTPH': {'tr2_ph': 1.0e-14},
}

NEB_PARAMETERS = {
    'PATH': {'nstep_path': 10},
}

NEB_PW_PARAMETERS = {
    'CONTROL': {'calculation': 'scf'},
    'SYSTEM': {'ecutwfc': 30.0},
    'ELECTRONS': {},
}


def _silicon() -> orm.StructureData:
    """Return an unstored single-kind silicon structure for the example pseudos shape."""
    structure = orm.StructureData(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols='Si')
    return structure


def _occupied_bands() -> orm.BandsData:
    """Return a band structure whose top band is fully occupied (insufficient ``nbnd``)."""
    import numpy as np

    bands = orm.BandsData()
    bands.set_kpoints(np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
    bands.set_bands(
        np.array([[[0.0, 1.0], [0.0, 1.0]]]),
        occupations=np.array([[[1.0, 1.0], [1.0, 1.0]]]),
    )
    return bands


class FlakyPwStep(WorkChainBase):
    """Deterministic stub sharing the ``pw`` error codes so handler tasks run for real.

    Failure schedule (``mode``): ``recover`` fails twice on a diagonalization error
    before succeeding (parameter-only restarts need no hidden nodes, so this is the
    one stub path that genuinely recovers at runtime); ``diag-fail`` always fails on
    a diagonalization error; ``always-fail`` always runs out of walltime; ``rattle``
    attaches outputs then fails on ionic convergence with few symmetries, but the
    mapped error arm hides those outputs so the restart guard fails loudly instead
    of rattling; ``fatal`` fails with a status no handler claims.
    """

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        """Define the attempt/mode inputs and the success outputs."""
        super().define(spec)
        spec.input('attempt', valid_type=orm.Int)
        spec.input('mode', valid_type=orm.Str, default=lambda: orm.Str('recover'))
        spec.input('result', valid_type=orm.Float, default=lambda: orm.Float(1.5))
        spec.input('structure', valid_type=orm.StructureData)
        spec.input(
            'success_parameters',
            valid_type=orm.Dict,
            default=lambda: orm.Dict({'number_of_bands': 8, 'number_of_symmetries': 10}),
        )
        spec.input(
            'rattle_parameters',
            valid_type=orm.Dict,
            default=lambda: orm.Dict({'number_of_bands': 8, 'number_of_symmetries': 3}),
        )
        spec.output('value', valid_type=orm.Float, required=False)
        spec.output('output_parameters', valid_type=orm.Dict, required=False)
        spec.output('output_structure', valid_type=orm.StructureData, required=False)
        spec.output('remote_folder', valid_type=orm.RemoteData, required=False)
        for member in PwErrors:
            spec.exit_code(member.value.status, member.name, message=member.value.message)
        spec.exit_code(999, 'UNMAPPED_FATAL', message='Fatal failure no handler claims.')
        spec.outline(cls.run_step)

    def run_step(self):
        """Fail or succeed deterministically from the ``attempt`` and ``mode`` inputs."""
        mode = self.inputs.mode.value
        attempt = self.inputs.attempt.value
        if mode == 'fatal':
            return self.exit_codes.UNMAPPED_FATAL
        if mode == 'always-fail':
            return self.exit_codes.OUT_OF_WALLTIME
        if mode == 'diag-fail':
            return self.exit_codes.COMPUTING_CHOLESKY
        if mode == 'rattle':
            if attempt == 0:
                self.out('output_parameters', self.inputs.rattle_parameters)
                self.out('output_structure', self.inputs.structure)
                return self.exit_codes.IONIC_CONVERGENCE_NOT_REACHED
            self._succeed()
            return None
        if mode == 'recover':
            if attempt < 2:
                return self.exit_codes.COMPUTING_CHOLESKY
            self._succeed()
            return None
        msg = f'unknown stub mode {mode!r}.'
        raise ValueError(msg)

    def _succeed(self) -> None:
        """Attach the success outputs."""
        self.out('value', self.inputs.result)
        self.out('output_parameters', self.inputs.success_parameters)
        self.out('output_structure', self.inputs.structure)


flaky_pw = task(FlakyPwStep, errors=PwErrors)
"""The shared stub as a graph task; ``ran = flaky_pw(...)`` exposes ``ran.value`` and ``ran.error``."""


@graph
def restartable_pw(
    mode: str, max_attempts: int, structure: orm.StructureData, result: orm.Float, parameters: dict
) -> dict:
    """Retry the shared stub through the real ``pw`` handler tasks.

    Mirrors ``pw_base_graph`` but places the stub (which shares the error codes)
    instead of ``PwCalculation``: the stub takes an explicit ``attempt`` counter while
    the calculation is driven by the carried inputs. The ``require_pw_restart_nodes``
    guards are mirrored too, so loud-failure paths behave as in production.
    """
    tried = base.seed_diagonalizations(parameters=parameters)
    attempt = 0
    run = flaky_pw(attempt=attempt, mode=mode, result=result, structure=structure)
    plan_walltime = base.handle_pw_walltime(error=run.error, parameters=parameters, tried=tried)
    plan_diagonalization = base.handle_pw_diagonalization(
        error=run.error,
        parameters=plan_walltime.parameters,
        tried=plan_walltime.tried,
        restart=plan_walltime.restart,
        parent_mode=plan_walltime.parent_mode,
        structure_from_output=plan_walltime.structure_from_output,
        rattle_now=plan_walltime.rattle_now,
    )
    plan_electronic = base.handle_pw_electronic_convergence(
        error=run.error,
        parameters=plan_diagonalization.parameters,
        tried=plan_diagonalization.tried,
        restart=plan_diagonalization.restart,
        parent_mode=plan_diagonalization.parent_mode,
        structure_from_output=plan_diagonalization.structure_from_output,
        rattle_now=plan_diagonalization.rattle_now,
    )
    plan_ionic = base.handle_pw_ionic_convergence(
        error=run.error,
        parameters=plan_electronic.parameters,
        tried=plan_electronic.tried,
        rattled=False,
        output_parameters=run.output_parameters,
        initial_parameters=parameters,
        restart=plan_electronic.restart,
        parent_mode=plan_electronic.parent_mode,
        structure_from_output=plan_electronic.structure_from_output,
        rattle_now=plan_electronic.rattle_now,
    )
    plan_bands = base.check_pw_sufficient_bands(
        error=run.error,
        parameters=plan_ionic.parameters,
        tried=plan_ionic.tried,
        output_band=None,
        output_parameters=run.output_parameters,
        restart=plan_ionic.restart,
        parent_mode=plan_ionic.parent_mode,
        structure_from_output=plan_ionic.structure_from_output,
        rattle_now=plan_ionic.rattle_now,
    )
    # The stub takes no parent folder: only the structure branch is mirrored here, the
    # parent branches live in ``pw_base_graph`` where the calculation takes one. The
    # guard still reads ``run.remote_folder`` (always ``None`` on the mapped error arm)
    # so loud-failure paths behave as in production.
    if plan_ionic.rattle_now and run.output_structure is not None:
        rattled = base.rattle_task(structure=run.output_structure, stdev=0.01)
    else:
        rattled = structure
    guard = base.require_pw_restart_nodes(
        restart=plan_bands.restart,
        parent_mode=plan_bands.parent_mode,
        structure_from_output=plan_bands.structure_from_output,
        output_structure=run.output_structure,
        remote_folder=run.remote_folder,
    )
    final = base.finalize_pw_carry(
        parameters=plan_bands.parameters,
        tried=plan_bands.tried,
        restart=plan_bands.restart,
        parent_mode=plan_bands.parent_mode,
        rattle_now=plan_bands.rattle_now,
        rattled_before=False,
        after=guard,
    )
    if plan_bands.structure_from_output and run.output_structure is not None:
        structure = run.output_structure
    if plan_bands.rattle_now:
        structure = rattled
    restart = plan_bands.restart
    attempts = 1
    while (run.error is not None or restart) and attempts < max_attempts:
        attempts = attempts + 1
        attempt = attempt + 1
        run = flaky_pw(attempt=attempt, mode=mode, result=result, structure=structure)
        plan_walltime = base.handle_pw_walltime(error=run.error, parameters=final.parameters, tried=final.tried)
        plan_diagonalization = base.handle_pw_diagonalization(
            error=run.error,
            parameters=plan_walltime.parameters,
            tried=plan_walltime.tried,
            restart=plan_walltime.restart,
            parent_mode=plan_walltime.parent_mode,
            structure_from_output=plan_walltime.structure_from_output,
            rattle_now=plan_walltime.rattle_now,
        )
        plan_electronic = base.handle_pw_electronic_convergence(
            error=run.error,
            parameters=plan_diagonalization.parameters,
            tried=plan_diagonalization.tried,
            restart=plan_diagonalization.restart,
            parent_mode=plan_diagonalization.parent_mode,
            structure_from_output=plan_diagonalization.structure_from_output,
            rattle_now=plan_diagonalization.rattle_now,
        )
        plan_ionic = base.handle_pw_ionic_convergence(
            error=run.error,
            parameters=plan_electronic.parameters,
            tried=plan_electronic.tried,
            rattled=final.rattled,
            output_parameters=run.output_parameters,
            initial_parameters=parameters,
            restart=plan_electronic.restart,
            parent_mode=plan_electronic.parent_mode,
            structure_from_output=plan_electronic.structure_from_output,
            rattle_now=plan_electronic.rattle_now,
        )
        plan_bands = base.check_pw_sufficient_bands(
            error=run.error,
            parameters=plan_ionic.parameters,
            tried=plan_ionic.tried,
            output_band=None,
            output_parameters=run.output_parameters,
            restart=plan_ionic.restart,
            parent_mode=plan_ionic.parent_mode,
            structure_from_output=plan_ionic.structure_from_output,
            rattle_now=plan_ionic.rattle_now,
        )
        if plan_ionic.rattle_now and run.output_structure is not None:
            rattled = base.rattle_task(structure=run.output_structure, stdev=0.01)
        else:
            rattled = structure
        guard = base.require_pw_restart_nodes(
            restart=plan_bands.restart,
            parent_mode=plan_bands.parent_mode,
            structure_from_output=plan_bands.structure_from_output,
            output_structure=run.output_structure,
            remote_folder=run.remote_folder,
        )
        final = base.finalize_pw_carry(
            parameters=plan_bands.parameters,
            tried=plan_bands.tried,
            restart=plan_bands.restart,
            parent_mode=plan_bands.parent_mode,
            rattle_now=plan_bands.rattle_now,
            rattled_before=final.rattled,
            after=guard,
        )
        if plan_bands.structure_from_output and run.output_structure is not None:
            structure = run.output_structure
        if plan_bands.rattle_now:
            structure = rattled
        restart = plan_bands.restart
    checked = base.ensure_pw_recovered(error=run.error, restart=restart)
    if run.error is None and not restart:
        status = 'finished'
    else:
        status = 'exhausted'
    report = base.summarize_restart(attempts=attempts, status=status, after=checked)
    return {
        'value': run.value,
        'output_parameters': run.output_parameters,
        'report': report,
    }


def test_build_validates_all_base_graphs() -> None:
    """``build()`` validates every base graph without launching anything."""
    assert base.pw_base_graph.build() is None
    assert base.ph_base_graph.build() is None
    assert base.neb_base_graph.build() is None
    assert base.q2r_base_graph.build() is None
    assert base.matdyn_base_graph.build() is None
    assert base.bands_base_graph.build() is None


def test_pw_graph_places_calc_once_per_loop_iteration() -> None:
    """The ``pw`` retry loop places the calculation once and maps the handled failures."""
    from aiida.engine.processes.graphs.ir import LoopControl, ProcessTask
    from aiida.engine.processes.graphs.lowering import compile_graph

    template = compile_graph(base.pw_base_graph)
    (loop,) = [entry for entry in template.tasks if isinstance(entry, LoopControl)]
    assert loop.condition_port == 'restart' or loop.condition_port is not None
    placed = [
        entry
        for entry in (*template.tasks, *loop.body.tasks)
        if isinstance(entry, ProcessTask) and entry.spec.process_class is not None
    ]
    from aiida_quantumespresso.calculations.pw import PwCalculation

    calc_placements = [entry for entry in placed if issubclass(entry.spec.process_class, PwCalculation)]
    assert len(calc_placements) == 2  # one before the loop, one rebound inside it
    statuses = set()
    for entry in calc_placements:
        statuses.update(entry.spec.exit_codes.values())
    handled = {code.status for code in statuses}
    assert {410, 400, 462, 500} <= handled


@pytest.mark.usefixtures('aiida_profile_clean')
def test_seed_diagonalizations_lists_untried() -> None:
    """The seed excludes the current algorithm and keeps conjugate gradient last."""
    _, node = run_get_node(base.seed_diagonalizations, parameters=PW_PARAMETERS)
    assert node.is_finished_ok
    assert node.outputs.result == ['cg', 'paro']


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_walltime_claims_full_restart() -> None:
    """A walltime error restarts in full from the output structure."""
    outputs, node = run_get_node(
        base.handle_pw_walltime, error=PwErrors.OUT_OF_WALLTIME, parameters=PW_PARAMETERS, tried=['cg']
    )
    assert node.is_finished_ok
    assert outputs['restart'].value is True
    assert outputs['parent_mode'].value == 'remote'
    assert outputs['structure_from_output'].value is True
    assert outputs['parameters'].get_dict()['CONTROL']['restart_mode'] == 'restart'


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_diagonalization_cycles_then_gives_up() -> None:
    """Diagonalization errors cycle algorithms and stop once all were tried."""
    outputs, _ = run_get_node(
        base.handle_pw_diagonalization,
        error=PwErrors.COMPUTING_CHOLESKY,
        parameters=PW_PARAMETERS,
        tried=['cg', 'paro'],
    )
    assert outputs['restart'].value is True
    assert outputs['parameters'].get_dict()['ELECTRONS']['diagonalization'] == 'paro'
    assert outputs['tried'].get_list() == ['cg']

    exhausted, _ = run_get_node(
        base.handle_pw_diagonalization, error=PwErrors.COMPUTING_CHOLESKY, parameters=PW_PARAMETERS, tried=[]
    )
    assert exhausted['restart'].value is False


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_electronic_convergence_reduces_mixing() -> None:
    """The mixing beta is scaled by the documented factor with a full restart."""
    outputs, _ = run_get_node(
        base.handle_pw_electronic_convergence,
        error=PwErrors.ELECTRONIC_CONVERGENCE_NOT_REACHED,
        parameters=PW_PARAMETERS,
        tried=[],
    )
    assert outputs['parameters'].get_dict()['ELECTRONS']['mixing_beta'] == pytest.approx(0.7 * 0.8)
    assert outputs['parent_mode'].value == 'remote'


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_ionic_fft_doubles_cell_factor() -> None:
    """Volume-contraction failures double ``cell_factor`` and restart from scratch."""
    outputs, _ = run_get_node(
        base.handle_pw_ionic_convergence,
        error=PwErrors.RADIAL_FFT_SIGNIFICANT_VOLUME_CONTRACTION,
        parameters=PW_PARAMETERS,
        tried=[],
        rattled=False,
        output_parameters=None,
        initial_parameters=PW_PARAMETERS,
    )
    assert outputs['parameters'].get_dict()['CELL']['cell_factor'] == 4
    assert outputs['parent_mode'].value == 'none'
    assert outputs['structure_from_output'].value is True


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_ionic_bfgs_rattle_arms_once() -> None:
    """A low-symmetry BFGS failure rattles once, then gives up on repetition."""
    import copy

    parameters = copy.deepcopy(PW_PARAMETERS)
    outputs, _ = run_get_node(
        base.handle_pw_ionic_convergence,
        error=PwErrors.IONIC_CYCLE_BFGS_HISTORY_FAILURE,
        parameters=parameters,
        tried=[],
        rattled=False,
        output_parameters=orm.Dict({'number_of_symmetries': 3}),
        initial_parameters=parameters,
    )
    assert outputs['rattle_now'].value is True
    assert outputs['restart'].value is True

    repeated, _ = run_get_node(
        base.handle_pw_ionic_convergence,
        error=PwErrors.IONIC_CYCLE_BFGS_HISTORY_FAILURE,
        parameters=parameters,
        tried=[],
        rattled=True,
        output_parameters=orm.Dict({'number_of_symmetries': 3}),
        initial_parameters=parameters,
    )
    assert repeated['restart'].value is False


@pytest.mark.usefixtures('aiida_profile_clean')
def test_check_pw_sufficient_bands_grows_nbnd() -> None:
    """An occupied top band grows ``nbnd`` and restarts from the charge density."""
    outputs, _ = run_get_node(
        base.check_pw_sufficient_bands,
        error=None,
        parameters=PW_PARAMETERS,
        tried=[],
        output_band=_occupied_bands(),
        output_parameters=orm.Dict({'number_of_bands': 8}),
    )
    assert outputs['restart'].value is True
    assert outputs['parameters'].get_dict()['SYSTEM']['nbnd'] == 8 + max(int(8 * 0.05), 4)
    assert outputs['parameters'].get_dict()['ELECTRONS']['startingpot'] == 'file'


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_ph_convergence_walks_strategies() -> None:
    """The ``ph`` convergence handler raises ``nmix_ph``, then tunes ``alpha_mix``."""
    first, _ = run_get_node(
        base.handle_ph_convergence, error=PhErrors.CONVERGENCE_NOT_REACHED, parameters=PH_PARAMETERS
    )
    assert first['parameters'].get_dict()['INPUTPH']['nmix_ph'] == 8
    assert first['recover'].value is True

    second, _ = run_get_node(
        base.handle_ph_convergence,
        error=PhErrors.CONVERGENCE_NOT_REACHED,
        parameters={'INPUTPH': {'nmix_ph': 8}},
    )
    assert second['parameters'].get_dict()['INPUTPH']['alpha_mix(1)'] == pytest.approx(0.7 * 0.5)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_ph_scheduler_walltime_halves_max_seconds() -> None:
    """Scheduler walltime halves a present ``max_seconds`` and always recovers."""
    outputs, _ = run_get_node(
        base.handle_ph_scheduler_walltime,
        error=PhErrors.SCHEDULER_OUT_OF_WALLTIME,
        parameters={'INPUTPH': {'max_seconds': 100.0}},
    )
    assert outputs['parameters'].get_dict()['INPUTPH']['max_seconds'] == pytest.approx(50.0)
    assert outputs['recover'].value is True


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_neb_nstep_extends_from_initial() -> None:
    """The ``nstep_path`` extension reads the initial inputs, not the carried ones."""
    outputs, _ = run_get_node(
        base.handle_neb_nstep_exceeded,
        error=NebErrors.NEB_CYCLE_EXCEEDED_NSTEP,
        parameters=NEB_PARAMETERS,
        pw_parameters=NEB_PW_PARAMETERS,
        tried=[],
        initial_parameters=NEB_PARAMETERS,
    )
    assert outputs['parameters'].get_dict()['PATH']['nstep_path'] == 20
    assert outputs['parameters'].get_dict()['PATH']['restart_mode'] == 'restart'


@pytest.mark.usefixtures('aiida_profile_clean')
def test_make_kpoints_mesh_guards_missing_inputs() -> None:
    """Missing k-points inputs fail fast naming the offending value."""
    with pytest.raises(ValueError, match='neither `kpoints`'):
        run_get_node(base.make_kpoints_mesh, structure=_silicon(), distance=None, force_parity=False)
    _, node = run_get_node(base.make_kpoints_mesh, structure=_silicon(), distance=0.5, force_parity=False)
    assert node.is_finished_ok
    assert node.outputs.result.get_kpoints_mesh() is not None


@pytest.mark.usefixtures('aiida_profile_clean')
def test_rattle_positions_changes_sites() -> None:
    """Rattling keeps the kinds but moves the sites."""
    structure = _silicon()
    _, node = run_get_node(base.rattle_positions, structure=structure, stdev=0.01)
    assert node.is_finished_ok
    assert node.outputs.result.kinds[0].name == 'Si'
    assert node.outputs.result.sites[0].position != structure.sites[0].position


def _remote_folder() -> orm.RemoteData:
    """Return a stored ``RemoteData`` on a stub localhost computer."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import AuthInfo, Computer, QueryBuilder, User

    existing = QueryBuilder().append(Computer, filters={'label': 'stub-base-computer'}).first(flat=True)
    if existing is None:
        existing = Computer(
            label='stub-base-computer',
            hostname='localhost',
            transport_type='core.local',
            scheduler_type='core.direct',
            workdir='/tmp',
        )
        existing.store()
    try:
        existing.get_authinfo(User.collection.get_default())
    except NotExistent:
        AuthInfo(computer=existing, user=User.collection.get_default()).store()
    return orm.RemoteData(computer=existing, remote_path='/tmp/stub-base').store()


@pytest.mark.usefixtures('aiida_profile_clean')
def test_require_pw_restart_nodes_passes_through() -> None:
    """Idle, parameter-only, and honored restarts pass the ``pw`` guard."""
    _, idle = run_get_node(
        base.require_pw_restart_nodes,
        restart=False,
        parent_mode='remote',
        structure_from_output=True,
        output_structure=None,
        remote_folder=None,
    )
    assert idle.is_finished_ok
    _, kept = run_get_node(
        base.require_pw_restart_nodes,
        restart=True,
        parent_mode='keep',
        structure_from_output=False,
        output_structure=None,
        remote_folder=None,
    )
    assert kept.is_finished_ok
    _, honored = run_get_node(
        base.require_pw_restart_nodes,
        restart=True,
        parent_mode='remote',
        structure_from_output=True,
        output_structure=_silicon(),
        remote_folder=_remote_folder(),
    )
    assert honored.is_finished_ok


@pytest.mark.usefixtures('aiida_profile_clean')
def test_require_pw_restart_nodes_fails_loudly() -> None:
    """Claimed restarts needing hidden nodes fail naming the missing node."""
    with pytest.raises(ValueError, match='output_structure'):
        run_get_node(
            base.require_pw_restart_nodes,
            restart=True,
            parent_mode='remote',
            structure_from_output=True,
            output_structure=None,
            remote_folder=None,
        )
    with pytest.raises(ValueError, match='remote_folder'):
        run_get_node(
            base.require_pw_restart_nodes,
            restart=True,
            parent_mode='remote',
            structure_from_output=False,
            output_structure=None,
            remote_folder=None,
        )


@pytest.mark.usefixtures('aiida_profile_clean')
def test_require_ph_restart_nodes() -> None:
    """The ``ph`` guard passes idle and honored restarts, failing loudly otherwise."""
    _, idle = run_get_node(base.require_ph_restart_nodes, restart=False, parent_mode='remote', remote_folder=None)
    assert idle.is_finished_ok
    _, kept = run_get_node(base.require_ph_restart_nodes, restart=True, parent_mode='keep', remote_folder=None)
    assert kept.is_finished_ok
    _, honored = run_get_node(
        base.require_ph_restart_nodes, restart=True, parent_mode='remote', remote_folder=_remote_folder()
    )
    assert honored.is_finished_ok
    with pytest.raises(ValueError, match='remote_folder'):
        run_get_node(base.require_ph_restart_nodes, restart=True, parent_mode='remote', remote_folder=None)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_require_neb_restart_nodes() -> None:
    """The ``neb`` guard passes idle and honored restarts, failing loudly otherwise."""
    _, idle = run_get_node(base.require_neb_restart_nodes, restart=False, parent_mode='remote', remote_folder=None)
    assert idle.is_finished_ok
    _, kept = run_get_node(base.require_neb_restart_nodes, restart=True, parent_mode='keep', remote_folder=None)
    assert kept.is_finished_ok
    _, honored = run_get_node(
        base.require_neb_restart_nodes, restart=True, parent_mode='remote', remote_folder=_remote_folder()
    )
    assert honored.is_finished_ok
    with pytest.raises(ValueError, match='remote_folder'):
        run_get_node(base.require_neb_restart_nodes, restart=True, parent_mode='remote', remote_folder=None)


def _guard_children(node):
    """Return the ``require_pw_restart_nodes`` child tasks of a graph run."""
    return [child for child in node.called_descendants if child.process_label == 'require_pw_restart_nodes']


@pytest.mark.usefixtures('aiida_profile_clean')
def test_restartable_pw_recovers_within_attempts() -> None:
    """Two diagonalization failures retry parameter-only; the third attempt succeeds."""
    outputs, node = run_get_node(
        restartable_pw,
        mode='recover',
        max_attempts=5,
        structure=_silicon(),
        result=orm.Float(1.5),
        parameters=PW_PARAMETERS,
    )
    assert node.is_finished_ok
    assert outputs['value'].value == 1.5
    assert outputs['report'].get_dict() == {'attempts': 3, 'status': 'finished'}
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'FlakyPwStep') == 3
    guards = _guard_children(node)
    assert guards, 'the retry loop must pass through the restart-node guard'
    assert all(guard.is_finished_ok for guard in guards)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_restartable_pw_rattle_branch_fails_loudly() -> None:
    """A low-symmetry ionic failure cannot rattle: the outputs are hidden, so guard fails."""
    outputs, node = run_get_node(
        restartable_pw,
        mode='rattle',
        max_attempts=5,
        structure=_silicon(),
        result=orm.Float(1.5),
        parameters=PW_PARAMETERS,
    )
    assert not node.is_finished_ok
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'FlakyPwStep') == 1
    assert sum(1 for child in children if child.process_label == 'rattle_positions') == 0
    assert 'require_pw_restart_nodes' in (node.exit_message or '')
    guards = _guard_children(node)
    assert len(guards) == 1
    assert not guards[0].is_finished_ok


@pytest.mark.usefixtures('aiida_profile_clean')
def test_restartable_pw_walltime_fails_loudly() -> None:
    """A walltime failure claims a remote restart the mapped arm cannot honor."""
    outputs, node = run_get_node(
        restartable_pw,
        mode='always-fail',
        max_attempts=2,
        structure=_silicon(),
        result=orm.Float(1.5),
        parameters=PW_PARAMETERS,
    )
    assert not node.is_finished_ok
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'FlakyPwStep') == 1
    assert 'require_pw_restart_nodes' in (node.exit_message or '')
    guards = _guard_children(node)
    assert len(guards) == 1
    assert not guards[0].is_finished_ok


@pytest.mark.usefixtures('aiida_profile_clean')
def test_restartable_pw_exhausts_budget() -> None:
    """Persistent parameter-only failures exhaust the budget and fail loudly."""
    outputs, node = run_get_node(
        restartable_pw,
        mode='diag-fail',
        max_attempts=2,
        structure=_silicon(),
        result=orm.Float(1.5),
        parameters=PW_PARAMETERS,
    )
    assert node.exit_status == 400
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'FlakyPwStep') == 2


@pytest.mark.usefixtures('aiida_profile_clean')
def test_restartable_pw_fatal_failure_aborts() -> None:
    """An unmapped failure claims no handler and fails the graph without retrying."""
    outputs, node = run_get_node(
        restartable_pw,
        mode='fatal',
        max_attempts=5,
        structure=_silicon(),
        result=orm.Float(1.5),
        parameters=PW_PARAMETERS,
    )
    assert node.exit_status == 400
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'FlakyPwStep') == 1


@pytest.mark.usefixtures('aiida_profile_clean')
def test_restartable_pw_returns_existing_nodes() -> None:
    """The root returns task outputs and keeps flat inputs, like the recipe."""
    outputs, node = run_get_node(
        restartable_pw,
        mode='recover',
        max_attempts=5,
        structure=_silicon(),
        result=orm.Float(1.5),
        parameters=PW_PARAMETERS,
    )
    assert node.is_finished_ok
    assert node.inputs.structure is not None
    children = node.called_descendants
    summary_node = next(child for child in children if child.process_label == 'summarize_restart')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


@calcfunction
def _stub_pw_output_parameters() -> orm.Dict:
    """Fabricate ``output_parameters`` for the shape-compatible ``pw`` stubs."""
    return orm.Dict({'number_of_bands': 8})


@calcfunction
def _stub_pw_output_structure(structure: orm.StructureData) -> orm.StructureData:
    """Return a clone of the input structure for the shape-compatible ``pw`` stubs."""
    return structure.clone()


def _declare_shaped_pw(spec) -> None:
    """Declare the ``pw_calc`` call shape plus the full ``pw`` success surface."""
    spec.input('code', valid_type=orm.Code)
    spec.input('structure', valid_type=orm.StructureData)
    spec.input_namespace('pseudos', valid_type=base.UpfData, dynamic=True)
    spec.input('parameters', valid_type=orm.Dict)
    spec.input('settings', valid_type=orm.Dict, required=False)
    spec.input('kpoints', valid_type=orm.KpointsData, required=False)
    spec.input('parent_folder', valid_type=orm.RemoteData, required=False)
    spec.output('output_parameters', valid_type=orm.Dict, required=False)
    spec.output('output_structure', valid_type=orm.StructureData, required=False)
    spec.output('output_trajectory', valid_type=orm.TrajectoryData, required=False)
    spec.output('output_band', valid_type=orm.BandsData, required=False)
    spec.output('output_kpoints', valid_type=orm.KpointsData, required=False)
    spec.output('output_atomic_occupations', valid_type=orm.Dict, required=False)
    spec.output('remote_folder', valid_type=orm.RemoteData, required=False)
    for member in PwErrors:
        spec.exit_code(member.value.status, member.name, message=member.value.message)


class DiagFlakyPw(WorkChainBase):
    """Shape-compatible stub failing on diagonalization until ``cg`` is selected.

    The failure is driven by the carried ``parameters`` (not a counter), so the real
    ``pw_base_graph`` retry loop drives it exactly as it would ``PwCalculation``.
    """

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        """Declare the placement shape and the ``pw`` success outputs."""
        super().define(spec)
        _declare_shaped_pw(spec)
        spec.outline(cls.run_stub)

    def run_stub(self):
        """Fail until conjugate gradients are selected, then attach the outputs."""
        parameters = self.inputs.parameters.get_dict()
        if parameters.get('ELECTRONS', {}).get('diagonalization', 'david') != 'cg':
            return self.exit_codes.COMPUTING_CHOLESKY
        self.out('output_parameters', _stub_pw_output_parameters())
        self.out('output_structure', _stub_pw_output_structure(self.inputs.structure))
        return None


class WalltimeFlakyPw(WorkChainBase):
    """Shape-compatible stub failing on walltime until a restart is requested.

    The handler answers with ``restart_mode == 'restart'``, so the second attempt
    would recover if the guard let it through; the guard fails first, proving the
    production graph no longer retries parent-less.
    """

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        """Declare the placement shape and the ``pw`` success outputs."""
        super().define(spec)
        _declare_shaped_pw(spec)
        spec.outline(cls.run_stub)

    def run_stub(self):
        """Fail once on walltime, then attach the outputs."""
        parameters = self.inputs.parameters.get_dict()
        if parameters.get('CONTROL', {}).get('restart_mode') != 'restart':
            return self.exit_codes.OUT_OF_WALLTIME
        self.out('output_parameters', _stub_pw_output_parameters())
        self.out('output_structure', _stub_pw_output_structure(self.inputs.structure))
        return None


def _upf():
    """Return a minimal parsable pseudo for the example ``Si`` shape."""
    content = '<UPF version="2.0.1"><PP_HEADER\nelement="Si"\nz_valence="4.0"\n/></UPF>\n'
    return base.UpfData(io.BytesIO(content.encode('utf-8')), filename='Si.upf')


def _substitute_pw_calc(monkeypatch, stub):
    """Replace the real ``pw_calc`` placement with a shape-compatible stub task.

    The production graph resolves placements by module-level name at compilation, so
    patching :attr:`base.pw_calc` redirects the real retry loop without copying its
    body; the ``inputs`` closure keeps the static ``pseudos`` shape intact. The
    compiled template is cached on the graph handle, so it is reset (and restored
    afterwards by ``monkeypatch``) to force recompilation against the substitute.
    """
    shaped = task(stub, inputs={'pseudos': {'Si': base.UpfData}}, errors=PwErrors)
    monkeypatch.setattr(base, 'pw_calc', shaped)
    monkeypatch.setattr(base.pw_base_graph, '_template', None)
    return shaped


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_diag_recovers_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """The production graph recovers parameter-only across three real placements."""
    _substitute_pw_calc(monkeypatch, DiagFlakyPw)
    code = aiida_code_installed(label='test.pw-diag', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudo=_upf(),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints_distance=0.5,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict() == {'attempts': 3, 'status': 'finished'}
    assert outputs['output_parameters'].get_dict()['number_of_bands'] == 8
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'DiagFlakyPw') == 3
    guards = [child for child in children if child.process_label == 'require_pw_restart_nodes']
    assert guards, 'the production retry loop must pass through the restart-node guard'
    assert all(guard.is_finished_ok for guard in guards)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_success_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """A first-try success returns the stub outputs plus a one-attempt report."""
    _substitute_pw_calc(monkeypatch, DiagFlakyPw)
    code = aiida_code_installed(label='test.pw-success', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudo=_upf(),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {'diagonalization': 'cg'}},
        kpoints_distance=0.5,
    )
    assert node.is_finished_ok
    assert outputs['report'].get_dict() == {'attempts': 1, 'status': 'finished'}
    assert outputs['output_structure'] is not None
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'DiagFlakyPw') == 1


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_walltime_fails_loudly_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """The production graph aborts a walltime restart instead of retrying parent-less."""
    _substitute_pw_calc(monkeypatch, WalltimeFlakyPw)
    code = aiida_code_installed(label='test.pw-walltime', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudo=_upf(),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints_distance=0.5,
    )
    assert not node.is_finished_ok
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'WalltimeFlakyPw') == 1
    assert 'require_pw_restart_nodes' in (node.exit_message or '')
    guards = [child for child in children if child.process_label == 'require_pw_restart_nodes']
    assert len(guards) == 1
    assert not guards[0].is_finished_ok


def test_error_enums_cover_documented_statuses() -> None:
    """The error enums map the statuses the handlers claim, without the loud ones."""
    pw_statuses = {member.value.status for member in PwErrors}
    assert {400, 410, 462, 500, 542} <= pw_statuses
    assert 150 not in pw_statuses  # message-gated monitor stop stays unhandled
    assert 501 not in pw_statuses  # considered-finished would lose its outputs
    assert 710 not in pw_statuses
    ph_statuses = {member.value.status for member in PhErrors}
    assert {120, 400, 410, 462} <= ph_statuses
    neb_statuses = {member.value.status for member in NebErrors}
    assert {400, 410, 462, 502, 503} <= neb_statuses


def test_neb_wrapper_propagates_exit_status() -> None:
    """The static wrapper declares every mapped ``neb`` status for the error arms."""
    from aiida_quantumespresso.workflows.graphs.base import NebStaticWrapper

    spec = NebStaticWrapper.spec()
    declared = {code.status for code in spec.exit_codes.values()}
    assert {member.value.status for member in NebErrors} <= declared
    assert 'parent_folder' in spec.inputs
    assert 'remote_folder' in spec.outputs
