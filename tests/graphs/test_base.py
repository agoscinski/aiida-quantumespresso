"""Tests for the graph alternatives to the Quantum ESPRESSO ``base`` restart work chains.

Run from the repository root (``uv run`` from the root environment, never bare
``python``/``pip``)::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \
        pytest aiida-quantumespresso/tests/graphs/test_base.py -x --tb=short

``build()`` checks run without launching anything; the runtime checks use a ``presto``
profile and deterministic ``CalcJob`` stubs that share :mod:`base` error codes and
handler tasks. The stubs subclass the real calculations, execute trivially on
localhost, and decide their exit status from the carried parameters in an overridden
``parse`` (no ``WorkChain`` anywhere in this file): diagonalization failures until
``cg`` is selected, walltime failures until a restart is requested, ionic failures
until a charge-density restart arrives. Failed runs store the same outputs a real
failure would, so the production retry loop, rebinding, and ``require_*`` guards
execute for real. Scheduler options for the ``CalcJob`` placements travel under
``runtime_config`` as a nested mapping with one ``pw_calc``/``neb_calc`` scope:
the first attempt and every retry share the placement's configuration (loop
control is transparent, so no suffixed or per-iteration key exists).
``pseudos`` travel as a general ``DataDict`` mapping
kind names onto ``UpfData`` nodes (one entry per structure kind).
"""

import io

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida import orm
from aiida.engine import WorkChain, run_get_node, task

from aiida_quantumespresso.calculations.neb import NebCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.graphs import base
from aiida_quantumespresso.workflows.graphs.base import NebErrors, PhErrors, PwErrors

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


def _upf(element='Si'):
    """Return a minimal parsable pseudo for ``element``."""
    from aiida.plugins import DataFactory

    UpfData = DataFactory('pseudo.upf')
    content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
    return UpfData(io.BytesIO(content.encode('utf-8')), filename=f'{element}.upf')


def _silicon() -> orm.StructureData:
    """Return an unstored single-kind silicon structure."""
    structure = orm.StructureData(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols='Si')
    return structure


def _silicon_oxide() -> orm.StructureData:
    """Return an unstored two-kind structure for the general ``pseudos`` mapping."""
    structure = orm.StructureData(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols='Si')
    structure.append_atom(position=(2, 2, 2), symbols='O')
    return structure


def _pseudos(*elements: str) -> orm.DataDict:
    """Return a ``DataDict`` mapping kind names onto ``UpfData`` nodes."""
    return orm.DataDict({element: _upf(element) for element in elements})


def _kpoints() -> orm.KpointsData:
    """Return an explicit k-point mesh."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])
    return kpoints


def _images() -> orm.TrajectoryData:
    """Return a three-step single-kind trajectory for the ``neb`` inputs."""
    import numpy as np

    cell = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    trajectory = orm.TrajectoryData()
    trajectory.set_trajectory(
        stepids=np.array([0, 1, 2]),
        cells=np.array([cell, cell, cell], dtype=float),
        symbols=['Si'],
        positions=np.array([[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]]]),
        pbc=(True, True, True),
    )
    return trajectory


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


class StubPwCalc(PwCalculation):
    """Trivial-execution ``PwCalculation`` driven by the carried parameters.

    Subclasses select the failure schedule through ``MODE``: ``diag`` fails on
    diagonalization until ``cg`` is selected (parameter-only, needs no stored
    nodes); ``walltime`` fails on walltime until a restart is requested, storing
    the structure a real interrupted run would leave; ``walltime-bare`` fails on
    walltime storing no structure, for the absent-node guard path; ``ionic`` fails
    on ionic convergence until a charge-density restart arrives. ``success``
    attaches the outputs at once.
    """

    MODE = 'success'

    def prepare_for_submission(self, folder):
        """Submit a no-op job: the exit status comes from ``parse`` alone."""
        from aiida.common.datastructures import CalcInfo, CodeInfo

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = []
        return calcinfo

    def _attach_structure(self):
        """Attach the structure a real interrupted run would have stored."""
        self.out('output_structure', self.inputs.structure.clone())

    def _attach_success(self):
        """Attach the success outputs."""
        self.out('output_parameters', orm.Dict({'number_of_bands': 8, 'number_of_symmetries': 10}))
        self._attach_structure()

    def parse(self, retrieved_temporary_folder=None, existing_exit_code=None):
        """Fail or succeed from the carried parameters, storing real outputs."""
        from aiida.engine.processes.exit_code import ExitCode

        parameters = self.inputs.parameters.get_dict()
        if self.MODE == 'diag':
            if parameters.get('ELECTRONS', {}).get('diagonalization', 'david') != 'cg':
                return ExitCode(462, 'stub diagonalization failure')
            self._attach_success()
            return None
        if self.MODE == 'walltime':
            if parameters.get('CONTROL', {}).get('restart_mode') != 'restart':
                self._attach_success()
                return ExitCode(400, 'stub walltime failure')
            self._attach_success()
            return None
        if self.MODE == 'walltime-bare':
            if parameters.get('CONTROL', {}).get('restart_mode') != 'restart':
                self.out('output_parameters', orm.Dict({'number_of_bands': 8}))
                return ExitCode(400, 'stub bare walltime failure')
            self._attach_success()
            return None
        if self.MODE == 'ionic':
            if parameters.get('ELECTRONS', {}).get('startingpot') != 'file':
                self._attach_success()
                return ExitCode(500, 'stub ionic failure')
            self._attach_success()
            return None
        if self.MODE == 'considered-501':
            self._attach_success()
            return ExitCode(501, 'stub ionic converged except in final scf')
        if self.MODE == 'warning-710':
            self._attach_success()
            return ExitCode(710, 'stub electronic convergence warning')
        if self.MODE == 'success':
            self._attach_success()
            return None
        msg = f'unknown stub mode {self.MODE!r}.'
        raise ValueError(msg)


class DiagPwCalc(StubPwCalc):
    """Stub failing on diagonalization until ``cg`` is selected."""

    MODE = 'diag'


class WalltimePwCalc(StubPwCalc):
    """Stub failing on walltime until a restart is requested, keeping the structure."""

    MODE = 'walltime'


class BareWalltimePwCalc(StubPwCalc):
    """Stub failing on walltime storing no structure, for the absent-node guard path."""

    MODE = 'walltime-bare'


class IonicPwCalc(StubPwCalc):
    """Stub failing on ionic convergence until a charge-density restart arrives."""

    MODE = 'ionic'


class SuccessPwCalc(StubPwCalc):
    """Stub succeeding at once."""

    MODE = 'success'


class Considered501PwCalc(StubPwCalc):
    """Stub persistently considered-finished with 501, storing usable outputs."""

    MODE = 'considered-501'


class Warning710PwCalc(StubPwCalc):
    """Stub persistently considered-finished with 710, storing usable outputs."""

    MODE = 'warning-710'


class StubNebCalc(NebCalculation):
    """Trivial-execution ``NebCalculation`` driven by the nested ``pw`` parameters.

    Fails on diagonalization until ``cg`` is selected, then attaches the outputs a
    real run would store.
    """

    def prepare_for_submission(self, folder):
        """Submit a no-op job: the exit status comes from ``parse`` alone."""
        from aiida.common.datastructures import CalcInfo, CodeInfo

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = []
        return calcinfo

    def parse(self, retrieved_temporary_folder=None, existing_exit_code=None):
        """Fail on diagonalization until ``cg`` is selected, then attach outputs."""
        import numpy as np
        from aiida.engine.processes.exit_code import ExitCode

        pw_parameters = self.inputs.pw.parameters.get_dict()
        if pw_parameters.get('ELECTRONS', {}).get('diagonalization', 'david') != 'cg':
            return ExitCode(462, 'stub neb diagonalization failure')
        self.out('output_parameters', orm.Dict({'number_of_bands': 8}))
        self.out('output_trajectory', self.inputs.images.clone())
        mep = orm.ArrayData()
        mep.set_array('mep', np.array([[0.0, 0.0]]))
        self.out('output_mep', mep)
        return None


def _substitute(monkeypatch, stub, name, errors):
    """Replace a ``base`` placement with a stub task, recompiling the production graph.

    The production graph resolves placements by module-level name at compilation, so
    patching the attribute redirects the real retry loop without copying its body.
    The compiled template is cached on the graph handle, so it is reset (and restored
    afterwards by ``monkeypatch``) to force recompilation against the substitute.
    Pseudos wire directly through their static port; unmonitored runs use plain
    ``pw_calc`` while monitored runs use ``pw_calc_monitored`` (static
    ``monitors.stuck``). Both are patched to the same stub when the stub is a
    ``PwCalculation`` so the real retry loop executes in either shape.
    """
    if issubclass(stub, PwCalculation) and name == 'pw_calc':
        plain = task(stub, errors=errors)
        monitored = task(stub, inputs={'monitors': {'stuck': orm.Dict}}, errors=errors)
        monkeypatch.setattr(base, 'pw_calc', plain)
        monkeypatch.setattr(base, 'pw_calc_monitored', monitored)
        for handle in (base.pw_base_graph, base.neb_base_graph):
            monkeypatch.setattr(handle, '_template', None)
        return monitored
    shaped = task(stub, errors=errors)
    monkeypatch.setattr(base, name, shaped)
    for handle in (base.pw_base_graph, base.neb_base_graph):
        monkeypatch.setattr(handle, '_template', None)
    return shaped


def _pw_metadata() -> dict:
    """Return the launch metadata both ``pw`` placements need."""
    return {'pw_calc': {'metadata': {'options': {}}}, 'pw_calc_monitored': {'metadata': {'options': {}}}}


def _neb_metadata() -> dict:
    """Return the launch metadata the ``neb`` placement needs."""
    return {'neb_calc': {'metadata': {'options': {}}}}


def test_build_validates_all_base_graphs() -> None:
    """``build()`` validates every base graph without launching anything."""
    assert base.pw_base_graph.build() is None
    assert base.ph_base_graph.build() is None
    assert base.neb_base_graph.build() is None
    assert base.q2r_base_graph.build() is None
    assert base.matdyn_base_graph.build() is None
    assert base.bands_base_graph.build() is None


def test_no_workchain_in_base_module() -> None:
    """The native track places calculations directly: no ``WorkChain`` anywhere."""
    assert not hasattr(base, 'NebStaticWrapper')
    for handle in (
        base.pw_calc,
        base.pw_calc_monitored,
        base.ph_calc,
        base.neb_calc,
        base.q2r_calc,
        base.matdyn_calc,
        base.bands_calc,
    ):
        assert not issubclass(handle.process_class, WorkChain)


def test_pw_graph_places_calc_once_per_loop_iteration() -> None:
    """The ``pw`` retry loop places the calculation once per attempt in either monitor shape."""
    from aiida.engine.processes.graphs.ir import BranchControl, LoopControl, ProcessTask
    from aiida.engine.processes.graphs.lowering import compile_graph

    template = compile_graph(base.pw_base_graph)
    (loop,) = [entry for entry in template.tasks if isinstance(entry, LoopControl)]
    assert loop.condition_port == 'restart' or loop.condition_port is not None

    def _collect(tasks):
        found = []
        for entry in tasks:
            if isinstance(entry, ProcessTask) and entry.spec.process_class is not None:
                found.append(entry)
            elif isinstance(entry, BranchControl):
                for branch in entry.branches:
                    found.extend(_collect(branch.tasks))
            elif isinstance(entry, LoopControl):
                found.extend(_collect(entry.body.tasks))
        return found

    placed = _collect(template.tasks)
    calc_placements = [entry for entry in placed if issubclass(entry.spec.process_class, PwCalculation)]
    # One call site before the loop and one rebound inside it, each branching over the
    # plain ``pw_calc`` (no monitor inputs) and ``pw_calc_monitored`` (static ``monitors.stuck``).
    assert len(calc_placements) == 4
    assert {entry.name for entry in calc_placements} == {'pw_calc', 'pw_calc_monitored'}
    statuses = set()
    for entry in calc_placements:
        statuses.update(entry.spec.exit_codes.values())
    handled = {code.status for code in statuses}
    assert {410, 400, 462, 500} <= handled
    assert {501, 710} <= handled  # considered-finished keeps its stored outputs


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
    """The charge mixing beta is scaled by the documented factor with a full restart."""
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
    """Claimed restarts needing absent nodes fail naming the missing node."""
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


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_diag_recovers_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """The production graph recovers parameter-only across three real placements.

    The two-kind structure exercises the general ``DataDict`` pseudos mapping; the
    stub failure is driven by the carried ``parameters`` (not a counter), so the real
    ``pw_base_graph`` retry loop drives it exactly as it would ``PwCalculation``.
    """
    _substitute(monkeypatch, DiagPwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-diag', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon_oxide(),
        pseudos=_pseudos('Si', 'O'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 3, 'status': 'finished'}
    assert outputs['output_parameters'].get_dict()['number_of_bands'] == 8
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'DiagPwCalc') == 3
    guards = [child for child in children if child.process_label == 'require_pw_restart_nodes']
    assert guards, 'the production retry loop must pass through the restart-node guard'
    assert all(guard.is_finished_ok for guard in guards)
    summary_node = next(child for child in children if child.process_label == 'summarize_restart')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_success_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """A first-try success returns the stub outputs plus a one-attempt report."""
    _substitute(monkeypatch, SuccessPwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-success', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {'diagonalization': 'cg'}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 1, 'status': 'finished'}
    assert outputs['output_structure'] is not None
    assert outputs['remote_folder'] is not None
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'SuccessPwCalc') == 1


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_walltime_recovers_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """A walltime restart rebinds the failed run's folder and recovers."""
    _substitute(monkeypatch, WalltimePwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-walltime', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 2, 'status': 'finished'}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'WalltimePwCalc') == 2
    first, second = sorted(
        [child for child in node.called if child.process_label == 'WalltimePwCalc'],
        key=lambda child: child.ctime,
    )
    assert second.inputs.parent_folder.uuid == first.outputs.remote_folder.uuid


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_ionic_recovers_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """An ionic restart rebinds the output structure and recovers from scratch."""
    _substitute(monkeypatch, IonicPwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-ionic', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 2, 'status': 'finished'}
    assert outputs['output_structure'] is not None
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'IonicPwCalc') == 2


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_absent_structure_fails_loudly_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """A walltime restart without a stored structure aborts at the guard, not degraded."""
    _substitute(monkeypatch, BareWalltimePwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-bare', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert not node.is_finished_ok
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'BareWalltimePwCalc') == 1
    assert 'require_pw_restart_nodes' in (node.exit_message or '')
    guards = [child for child in children if child.process_label == 'require_pw_restart_nodes']
    assert len(guards) == 1
    assert not guards[0].is_finished_ok


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_exhausts_budget_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """Persistent parameter-only failures exhaust the budget and fail loudly."""
    _substitute(monkeypatch, DiagPwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-exhaust', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        max_iterations=2,
        runtime_config=_pw_metadata(),
    )
    assert node.exit_status == 401
    assert node.exit_message == 'The maximum number of iterations was exceeded.'
    assert outputs == {}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'DiagPwCalc') == 2
    assert not any(child.process_label == 'summarize_restart' for child in children)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_dict_parameters_and_parent_folder(monkeypatch, aiida_code_installed) -> None:
    """``orm.Dict`` parameters and an explicit parent folder wire through untouched."""
    _substitute(monkeypatch, SuccessPwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-dictparent', default_calc_job_plugin='quantumespresso.pw')
    parent = _remote_folder()
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters=orm.Dict({'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {'diagonalization': 'cg'}}),
        kpoints=_kpoints(),
        parent_folder=parent,
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 1, 'status': 'finished'}
    (child,) = [entry for entry in node.called if entry.process_label == 'SuccessPwCalc']
    assert child.inputs.parent_folder.uuid == parent.uuid


@pytest.mark.usefixtures('aiida_profile_clean')
def test_classify_pw_finished_maps_considered_statuses() -> None:
    """501/710 map to distinct finished statuses; anything else proceeds."""
    outputs, node = run_get_node(
        base.classify_pw_finished, error=PwErrors.IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['finished'].value is True
    assert outputs['status'].value == 'finished_ionic_convergence_reached_except_final_scf'
    outputs, node = run_get_node(base.classify_pw_finished, error=PwErrors.ELECTRONIC_CONVERGENCE_WARNING)
    assert node.is_finished_ok, node.exit_message
    assert outputs['finished'].value is True
    assert outputs['status'].value == 'finished_electronic_convergence_warning'
    for error in (None, PwErrors.OUT_OF_WALLTIME, PwErrors.STOPPED_BY_MONITOR):
        outputs, node = run_get_node(base.classify_pw_finished, error=error)
        assert node.is_finished_ok, node.exit_message
        assert outputs['finished'].value is False


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_considered_501_returns_usable_outputs(monkeypatch, aiida_code_installed) -> None:
    """A persistent 501 finishes at once with the stored outputs, not a retry.

    Mirrors the legacy ``handle_vcrelax_converged_except_final_scf``: convergence
    reached except in the final SCF is usable science, so the production graph
    returns the failed run's stored nodes with a distinct report status. Nothing
    here copies the graph body; the substitute only decides the exit status.
    """
    _substitute(monkeypatch, Considered501PwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-501', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {'diagonalization': 'cg'}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {
        'attempts': 1,
        'status': 'finished_ionic_convergence_reached_except_final_scf',
    }
    assert outputs['output_parameters'].get_dict()['number_of_bands'] == 8
    assert isinstance(outputs['output_structure'], orm.StructureData)
    children = [child for child in node.called if child.process_label == 'Considered501PwCalc']
    assert len(children) == 1
    assert children[0].exit_status == 501


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_warning_710_returns_usable_outputs(monkeypatch, aiida_code_installed) -> None:
    """A persistent 710 finishes at once with the stored outputs, not a retry.

    Mirrors the legacy ``handle_electronic_convergence_warning``: with
    ``scf_must_converge=False`` the unconverged result is the accepted outcome.
    """
    _substitute(monkeypatch, Warning710PwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.pw-710', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {'diagonalization': 'cg'}},
        kpoints=_kpoints(),
        runtime_config=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {
        'attempts': 1,
        'status': 'finished_electronic_convergence_warning',
    }
    assert outputs['output_parameters'].get_dict()['number_of_bands'] == 8
    children = [child for child in node.called if child.process_label == 'Warning710PwCalc']
    assert len(children) == 1
    assert children[0].exit_status == 710


@pytest.mark.usefixtures('aiida_profile_clean')
def test_neb_base_graph_diag_recovers_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """The production ``neb`` graph cycles nested diagonalization across three runs."""
    _substitute(monkeypatch, StubNebCalc, 'neb_calc', NebErrors)
    code = aiida_code_installed(label='test.neb-diag', default_calc_job_plugin='quantumespresso.neb')
    outputs, node = run_get_node(
        base.neb_base_graph,
        code=code,
        images=_images(),
        pseudos=_pseudos('Si'),
        parameters=dict(NEB_PARAMETERS),
        pw_parameters=dict(NEB_PW_PARAMETERS),
        kpoints=_kpoints(),
        runtime_config=_neb_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 3, 'status': 'finished'}
    assert outputs['output_parameters'] is not None
    assert outputs['output_trajectory'] is not None
    assert outputs['output_mep'] is not None
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'StubNebCalc') == 3
    guards = [child for child in children if child.process_label == 'require_neb_restart_nodes']
    assert guards, 'the production retry loop must pass through the restart-node guard'
    assert all(guard.is_finished_ok for guard in guards)


def test_error_enums_cover_documented_statuses() -> None:
    """The error enums map the statuses the handlers claim, without the loud ones."""
    pw_statuses = {member.value.status for member in PwErrors}
    assert {150, 400, 410, 462, 500, 542} <= pw_statuses
    # 150 is claimed only on a machine-readable `monitor_diagnostics` reason, never on the exit message.
    # 501/710 are considered-finished with usable outputs: the mapped failure keeps every
    # stored declared output alongside the synthesized error, so the graph returns the real
    # run outputs with a distinct report status instead of retrying or failing.
    assert {501, 710} <= pw_statuses
    ph_statuses = {member.value.status for member in PhErrors}
    assert {120, 400, 410, 462} <= ph_statuses
    neb_statuses = {member.value.status for member in NebErrors}
    assert {400, 410, 462, 502, 503} <= neb_statuses


def test_subgraph_contract() -> None:
    """The native ``pw`` base exposes the composite-track interface."""
    import inspect

    signature = inspect.signature(base.pw_base_graph)
    assert list(signature.parameters) == [
        'code',
        'structure',
        'pseudos',
        'parameters',
        'settings',
        'kpoints',
        'kpoints_distance',
        'kpoints_force_parity',
        'parent_folder',
        'monitors',
        'max_iterations',
    ]
    assert signature.parameters['pseudos'].annotation is orm.DataDict
    assert signature.parameters['monitors'].annotation == orm.DataDict | None
    assert signature.parameters['parent_folder'].annotation == orm.RemoteData | None
