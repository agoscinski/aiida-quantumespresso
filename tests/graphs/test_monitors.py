"""Tests for the native monitor wiring of ``pw_base_graph``.

Run from the repository root (``uv run`` from the root environment, never bare
``python``/``pip``)::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \
        pytest aiida-quantumespresso/tests/graphs/test_monitors.py -x --tb=short

``build()`` checks run without launching anything; the runtime checks use a ``presto``
profile and deterministic ``CalcJob`` stubs that subclass the real ``PwCalculation``,
execute trivially on localhost, and fail with ``STOPPED_BY_MONITOR`` (status 150)
from their carried inputs and ``monitors`` mapping in an overridden ``parse`` (no
``WorkChain`` anywhere in this file). The stubs stand in for the engine side of a
monitor kill: they store a real ``monitor_diagnostics`` output node exactly as the
engine would attach ``CalcJobMonitorResult.outputs``, and the production
``handle_pw_monitor_stuck`` task reads that stored node. No handler is monkeypatched.
"""

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida import orm
from aiida.engine import run_get_node, task

from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.tools.monitors.accuracy_stuck import (
    ACCURACY_STUCK_MESSAGE_PREFIX,
    ACCURACY_STUCK_REASON,
)
from aiida_quantumespresso.workflows.graphs import base
from aiida_quantumespresso.workflows.graphs.base import PwErrors

pytestmark = pytest.mark.presto


def _upf(element='Si'):
    """Return a minimal parsable pseudo for ``element``."""
    import io

    from aiida.plugins import DataFactory

    UpfData = DataFactory('pseudo.upf')
    content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
    return UpfData(io.BytesIO(content.encode('utf-8')), filename=f'{element}.upf')


def _silicon() -> orm.StructureData:
    """Return an unstored single-kind silicon structure."""
    structure = orm.StructureData(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols='Si')
    return structure


def _pseudos(*elements: str) -> orm.DataDict:
    """Return a ``DataDict`` mapping kind names onto ``UpfData`` nodes."""
    return orm.DataDict({element: _upf(element) for element in elements})


def _kpoints() -> orm.KpointsData:
    """Return an explicit k-point mesh."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])
    return kpoints


def _monitors(**kwargs) -> orm.DataDict:
    """Return a ``DataDict`` wiring the structured stuck monitor under ``stuck``."""
    monitor = {'entry_point': 'quantumespresso.accuracy_stuck', 'kwargs': {'structured': True, **kwargs}}
    return orm.DataDict({'stuck': orm.Dict(monitor)})


def _incoming_labels(node) -> list:
    """Return the incoming link labels of a stored node."""
    return [link.link_label for link in node.base.links.get_incoming().all()]


def _pw_metadata() -> dict:
    """Return the launch metadata the ``pw`` placement needs.

    The first attempt and every retry share one ``pw_calc`` scope: loop
    control is transparent, so no suffixed or per-iteration key exists.
    """
    return {'pw_calc': {'options': {}}}


class _MonitorStubBase(PwCalculation):
    """Trivial-execution ``PwCalculation`` failing with a monitor stop from its inputs.

    ``REASON`` is the diagnostic reason stored on the failed run, or `None` for a
    bare stop without diagnostics. Subclasses decide *when* to stop by overriding
    :meth:`should_stop`.
    """

    REASON: str | None = ACCURACY_STUCK_REASON

    def prepare_for_submission(self, folder):
        """Submit a no-op job: the exit status comes from ``parse`` alone."""
        from aiida.common.datastructures import CalcInfo, CodeInfo

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = []
        return calcinfo

    def should_stop(self) -> bool:
        """Return whether this run stops with ``STOPPED_BY_MONITOR``."""
        raise NotImplementedError

    def parse(self, retrieved_temporary_folder=None, existing_exit_code=None):
        """Stop with a diagnostic, or attach the success outputs a real run would store."""
        from aiida.engine.processes.exit_code import ExitCode

        self.out('output_parameters', orm.Dict({'number_of_bands': 8, 'number_of_symmetries': 10}))
        self.out('output_structure', self.inputs.structure.clone())
        if self.should_stop():
            if self.REASON is not None:
                self.out('monitor_diagnostics', orm.Dict({'reason': self.REASON}))
            return ExitCode(150, 'stub monitor stop')
        return None


class StuckRelaxPwCalc(_MonitorStubBase):
    """Stop as stuck until the ``upscale`` reduction arrives, then converge."""

    def should_stop(self) -> bool:
        """Stop while ``upscale`` is still at its default: any reduction recovers."""
        return self.inputs.parameters.get_dict().get('IONS', {}).get('upscale', 100) >= 100


class StuckScfPwCalc(_MonitorStubBase):
    """Stop a non-``relax`` run as stuck until the full restart rebinds the parent."""

    def should_stop(self) -> bool:
        """Stop on the first run; the restarted run inherits the parent folder and converges."""
        return 'parent_folder' not in self.inputs


class OtherReasonPwCalc(_MonitorStubBase):
    """Stop persistently with an unrelated diagnostic reason: never the stuck monitor."""

    REASON = 'other_monitor'

    def should_stop(self) -> bool:
        """Always stop so the graph must exhaust its budget without claiming recovery."""
        return True


class BareStopPwCalc(_MonitorStubBase):
    """Stop persistently with no diagnostic at all: a legacy string-only monitor stop."""

    REASON = None

    def should_stop(self) -> bool:
        """Always stop so the graph must exhaust its budget without claiming recovery."""
        return True


class ConditionalPwCalc(_MonitorStubBase):
    """Stop as stuck only when a monitor mapping actually reached the calculation."""

    def should_stop(self) -> bool:
        """Stop iff the forwarded ``monitors`` namespace arrived non-empty."""
        return bool(self.inputs.get('monitors', {}))


def _substitute(monkeypatch, stub):
    """Replace the ``pw_calc`` placement with a stub task, recompiling the production graph."""
    shaped = task(stub, inputs={'monitors': orm.DataDict}, errors=PwErrors)
    monkeypatch.setattr(base, 'pw_calc', shaped)
    monkeypatch.setattr(base.pw_base_graph, '_template', None)
    return shaped


def test_build_covers_monitor_wiring() -> None:
    """``build()`` validates the monitor-forwarding graph without launching anything."""
    assert base.pw_base_graph.build() is None


@pytest.mark.usefixtures('aiida_profile_clean')
def test_accuracy_stuck_legacy_default_stays_string() -> None:
    """The default monitor call still returns the historical string message."""
    from aiida_quantumespresso.tools.monitors.accuracy_stuck import monitor

    lines = '\n'.join('accuracy: 0.1000 Ry' for _ in range(6))
    result = monitor(_FakeNode('output.txt'), _FakeTransport(lines))
    assert isinstance(result, str)
    assert result.startswith(ACCURACY_STUCK_MESSAGE_PREFIX)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_accuracy_stuck_structured_returns_diagnostic_result() -> None:
    """The opt-in structured call returns a result carrying the machine-readable reason."""
    from aiida.engine.processes.calcjobs.monitors import CalcJobMonitorResult

    from aiida_quantumespresso.tools.monitors.accuracy_stuck import monitor

    lines = '\n'.join('accuracy: 0.1000 Ry' for _ in range(6))
    result = monitor(_FakeNode('output.txt'), _FakeTransport(lines), structured=True)
    assert isinstance(result, CalcJobMonitorResult)
    assert result.message.startswith(ACCURACY_STUCK_MESSAGE_PREFIX)
    diagnostics = result.outputs['monitor_diagnostics']
    assert isinstance(diagnostics, orm.Dict)
    assert diagnostics.get_dict()['reason'] == ACCURACY_STUCK_REASON
    assert diagnostics.get_dict()['reason'] == 'accuracy_stuck'


@pytest.mark.usefixtures('aiida_profile_clean')
def test_accuracy_stuck_structured_not_stuck_is_none() -> None:
    """Improving accuracy returns `None` in both legacy and structured mode."""
    from aiida_quantumespresso.tools.monitors.accuracy_stuck import monitor

    lines = '\n'.join(f'accuracy: {value:.4f} Ry' for value in [0.5, 0.4, 0.3, 0.2, 0.1])
    node, transport = _FakeNode('output.txt'), _FakeTransport(lines)
    assert monitor(node, transport) is None
    assert monitor(node, transport, structured=True) is None


@pytest.mark.usefixtures('aiida_profile_clean')
def test_accuracy_stuck_structured_kwargs_validate() -> None:
    """Core accepts the structured opt-in kwargs through the ``monitors`` namespace."""
    from aiida.engine.processes.calcjobs.monitors import CalcJobMonitor

    monitor = CalcJobMonitor(
        entry_point='quantumespresso.accuracy_stuck', kwargs={'structured': True, 'min_repeats': 3}
    )
    assert monitor.kwargs == {'structured': True, 'min_repeats': 3}

    from aiida.engine.processes.calcjobs.calcjob import validate_monitors

    assert (
        validate_monitors({'stuck': orm.Dict({'entry_point': monitor.entry_point, 'kwargs': monitor.kwargs})}, None)
        is None
    )


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_monitor_stuck_reduces_upscale() -> None:
    """A stuck ``relax`` run reduces ``upscale`` by the legacy factor and restarts fully."""
    parameters = {'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}}
    diagnostics = orm.Dict({'reason': ACCURACY_STUCK_REASON})
    outputs, _ = run_get_node(
        base.handle_pw_monitor_stuck,
        error=PwErrors.STOPPED_BY_MONITOR,
        parameters=parameters,
        tried=[],
        diagnostics=diagnostics,
    )
    assert outputs['restart'].value is True
    assert outputs['parent_mode'].value == 'remote'
    assert outputs['structure_from_output'].value is False
    restarted = outputs['parameters'].get_dict()
    assert restarted['IONS']['upscale'] == max(int(100 * 0.3), 1)
    assert restarted['CONTROL']['restart_mode'] == 'restart'


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_monitor_stuck_at_minimum_gives_up() -> None:
    """``upscale`` already at its minimum requests no restart: the loop fails loudly."""
    parameters = {'CONTROL': {'calculation': 'relax'}, 'IONS': {'upscale': 1}}
    diagnostics = orm.Dict({'reason': ACCURACY_STUCK_REASON})
    outputs, _ = run_get_node(
        base.handle_pw_monitor_stuck,
        error=PwErrors.STOPPED_BY_MONITOR,
        parameters=parameters,
        tried=[],
        diagnostics=diagnostics,
    )
    assert outputs['restart'].value is False


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_monitor_stuck_non_relax_full_restarts() -> None:
    """A stuck ``scf`` run fully restarts from the parent with ``upscale`` untouched."""
    parameters = {'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}
    diagnostics = orm.Dict({'reason': ACCURACY_STUCK_REASON})
    outputs, _ = run_get_node(
        base.handle_pw_monitor_stuck,
        error=PwErrors.STOPPED_BY_MONITOR,
        parameters=parameters,
        tried=[],
        diagnostics=diagnostics,
    )
    assert outputs['restart'].value is True
    assert outputs['parent_mode'].value == 'remote'
    restarted = outputs['parameters'].get_dict()
    assert restarted['CONTROL']['restart_mode'] == 'restart'
    assert 'IONS' not in restarted
    assert restarted['ELECTRONS'] == {}


@pytest.mark.usefixtures('aiida_profile_clean')
def test_handle_pw_monitor_stuck_ignores_other_diagnostics() -> None:
    """An unrelated reason, a missing diagnostic, or another error never claims recovery."""
    parameters = {'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}}
    for diagnostics in (orm.Dict({'reason': 'other_monitor'}), orm.Dict({}), None):
        outputs, _ = run_get_node(
            base.handle_pw_monitor_stuck,
            error=PwErrors.STOPPED_BY_MONITOR,
            parameters=parameters,
            tried=['cg'],
            diagnostics=diagnostics,
        )
        assert outputs['restart'].value is False
        assert outputs['parameters'].get_dict() == parameters
        assert outputs['tried'].get_list() == ['cg']

    idle, _ = run_get_node(
        base.handle_pw_monitor_stuck,
        error=PwErrors.OUT_OF_WALLTIME,
        parameters=parameters,
        tried=[],
        diagnostics=orm.Dict({'reason': ACCURACY_STUCK_REASON}),
    )
    assert idle['restart'].value is False


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_monitor_recovers_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """The production graph recovers a stuck ``relax`` run through forwarded monitors.

    The stub stops only while ``upscale`` is at its default, so the second attempt
    succeeding proves the handler's reduction drove the recovery. The asserted
    ``monitors__stuck`` incoming link proves the forwarded mapping reached the
    failed calculation; nothing here monkeypatches the handler.
    """
    _substitute(monkeypatch, StuckRelaxPwCalc)
    code = aiida_code_installed(label='test.pw-monitor', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        monitors=_monitors(),
        task_metadata=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 2, 'status': 'finished'}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'StuckRelaxPwCalc') == 2
    first, second = sorted(
        [child for child in node.called if child.process_label == 'StuckRelaxPwCalc'],
        key=lambda child: child.ctime,
    )
    assert 'monitors__stuck' in _incoming_labels(first)
    assert first.outputs.monitor_diagnostics.get_dict()['reason'] == ACCURACY_STUCK_REASON
    assert second.inputs.parameters.get_dict()['IONS']['upscale'] == max(int(100 * 0.3), 1)
    assert second.inputs.parent_folder.uuid == first.outputs.remote_folder.uuid
    assert 'monitors__stuck' in _incoming_labels(second)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_monitor_scf_restarts_unchanged(monkeypatch, aiida_code_installed) -> None:
    """A stuck ``scf`` run restarts from the parent folder with parameters untouched."""
    _substitute(monkeypatch, StuckScfPwCalc)
    code = aiida_code_installed(label='test.pw-monitor-scf', default_calc_job_plugin='quantumespresso.pw')
    parameters = {'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters=dict(parameters),
        kpoints=_kpoints(),
        monitors=_monitors(),
        task_metadata=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 2, 'status': 'finished'}
    children = node.called_descendants
    assert sum(1 for child in children if child.process_label == 'StuckScfPwCalc') == 2
    first, second = sorted(
        [child for child in node.called if child.process_label == 'StuckScfPwCalc'],
        key=lambda child: child.ctime,
    )
    restarted = second.inputs.parameters.get_dict()
    assert restarted['CONTROL']['restart_mode'] == 'restart'
    assert 'IONS' not in restarted  # no upscale reduction outside `relax`/`vc-relax`
    assert second.inputs.parent_folder.uuid == first.outputs.remote_folder.uuid


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_unmonitored_run_skips_monitor_stop(monkeypatch, aiida_code_installed) -> None:
    """Without forwarded monitors the conditional stub succeeds at once with no monitor inputs."""
    _substitute(monkeypatch, ConditionalPwCalc)
    code = aiida_code_installed(label='test.pw-monitor-absent', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        task_metadata=_pw_metadata(),
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 1, 'status': 'finished'}
    (child,) = [entry for entry in node.called if entry.process_label == 'ConditionalPwCalc']
    assert not any(label == 'monitors__stuck' or label == 'monitors' for label in _incoming_labels(child))


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_other_diagnostic_fails_loudly(monkeypatch, aiida_code_installed) -> None:
    """A stop with an unrelated diagnostic never triggers the stuck recovery.

    The stub stops persistently, so the loop exhausts its budget with parameters
    untouched instead of accidentally retrying the upscale recovery.
    """
    _substitute(monkeypatch, OtherReasonPwCalc)
    code = aiida_code_installed(label='test.pw-monitor-other', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        monitors=_monitors(),
        max_iterations=2,
        task_metadata=_pw_metadata(),
    )
    assert not node.is_finished_ok
    assert outputs == {}
    assert 'ensure_pw_recovered' in (node.exit_message or '')
    children = [child for child in node.called if child.process_label == 'OtherReasonPwCalc']
    assert len(children) == 2
    for child in children:
        assert 'upscale' not in child.inputs.parameters.get_dict().get('IONS', {})
        assert 'monitors__stuck' in _incoming_labels(child)


@pytest.mark.usefixtures('aiida_profile_clean')
def test_pw_base_graph_bare_stop_fails_loudly(monkeypatch, aiida_code_installed) -> None:
    """A stop without any diagnostic never triggers the stuck recovery either."""
    _substitute(monkeypatch, BareStopPwCalc)
    code = aiida_code_installed(label='test.pw-monitor-bare', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=_pseudos('Si'),
        parameters={'CONTROL': {'calculation': 'relax'}, 'SYSTEM': {}, 'ELECTRONS': {}},
        kpoints=_kpoints(),
        monitors=_monitors(),
        max_iterations=2,
        task_metadata=_pw_metadata(),
    )
    assert not node.is_finished_ok
    assert outputs == {}
    assert 'ensure_pw_recovered' in (node.exit_message or '')
    children = [child for child in node.called if child.process_label == 'BareStopPwCalc']
    assert len(children) == 2
    for child in children:
        assert 'monitor_diagnostics' not in child.outputs
        assert 'upscale' not in child.inputs.parameters.get_dict().get('IONS', {})


class _FakeNode:
    """Minimal ``CalcJobNode`` stand-in for testing the monitor function."""

    def __init__(self, filename):
        self._filename = filename

    @property
    def attributes(self):
        return {'output_filename': self._filename}


class _FakeTransport:
    """Minimal ``Transport`` stand-in that writes a fixed string to a local file."""

    def __init__(self, content):
        self._content = content

    def getfile(self, remote_path, local_path):
        with open(local_path, 'w', encoding='utf-8') as handle:
            handle.write(self._content)
