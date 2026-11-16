############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Tests for the post-termination graph cleanup helper.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_cleanup.py -q

The tests build real provenance (stored ``GraphNode``/``CalcJobNode``/``RemoteData`` nodes with ``CALL_CALC``
and ``CREATE`` links) rooted in real temporary directories on the localhost, and clean through the real local
transport: no remote scheduler is involved, and no calculation is executed. One test runs a genuine ``@graph``
to pin the terminated-success shape; failed and killed roots use the stored shapes a graph termination leaves
behind (finished with an exit status, killed), which is exactly what the external owner observes.
"""

from pathlib import Path

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida.common.links import LinkType
from aiida.common.processes import ProcessState
from aiida.engine import graph, run_get_node, task
from aiida.orm import CalcJobNode, Dict, GraphNode, RemoteData

from aiida_quantumespresso.workflows.graphs.cleanup import cleanup_graph_workdirs, sweep_graph_remote

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


@task
def echo_value(value: int) -> int:
    """Return the input, giving the genuine graph below one task."""
    return value


@graph
def tiny_success_graph(value: int) -> int:
    """Minimal graph that terminates successfully."""
    return echo_value(value=value)


def _write_files(path: Path, names: tuple = ('output.out',)) -> None:
    """Create real files in a directory that stands in for a remote folder."""
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (path / name).write_text('data', encoding='utf-8')


def _remote(computer, path: Path) -> RemoteData:
    """Return an unstored ``RemoteData`` rooted at a real directory."""
    return RemoteData(computer=computer, remote_path=str(path))


def _calc(computer, *, state=ProcessState.FINISHED, exit_status: int | None = 0) -> CalcJobNode:
    """Return an unstored ``CalcJobNode`` mimicking a ``pw.x`` run."""
    calc = CalcJobNode(computer=computer)
    calc.process_type = 'aiida.calculations:quantumespresso.pw'
    calc.set_process_state(state)
    calc.set_exit_status(exit_status)
    return calc


def _graph(pairs: tuple = (), *, state=ProcessState.FINISHED, exit_status: int | None = 0) -> GraphNode:
    """Return a stored ``GraphNode`` with ``(calc, folder)`` pairs linked as ``CALL_CALC``/``CREATE``.

    Links are added before each node is stored, in dependency order, so the provenance is real. A ``None``
    folder models a calculation that stored no ``remote_folder``.
    """
    node = GraphNode()
    node.set_process_state(state)
    node.set_exit_status(exit_status)
    node.store()
    for index, (calc, folder) in enumerate(pairs):
        calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label=f'call_{index + 1}')
        calc.store()
        if folder is not None:
            folder.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label='remote_folder')
            folder.store()
    return node


def test_genuine_success_graph_terminates() -> None:
    """A genuinely executed graph terminates and has nothing to clean."""
    _, node = run_get_node(tiny_success_graph, value=1)
    assert node.is_terminated
    assert node.is_finished_ok
    assert cleanup_graph_workdirs(node) == {'cleaned': [], 'skipped': [], 'failed': []}


def test_cleans_success_descendants(aiida_localhost, tmp_path) -> None:
    """Terminated success descendants have their real folders removed and are reported by pk."""
    computer = aiida_localhost
    first, second = tmp_path / 'run_1', tmp_path / 'run_2'
    _write_files(first)
    _write_files(second)
    node = _graph(((_calc(computer), _remote(computer, first)), (_calc(computer), _remote(computer, second))))
    result = cleanup_graph_workdirs(node)
    assert sorted(result['cleaned']) == sorted(child.pk for child in node.called)
    assert result['skipped'] == []
    assert result['failed'] == []
    assert not first.exists()
    assert not second.exists()
    assert all(child.outputs.remote_folder.is_cleaned for child in node.called)


def test_cleanup_is_idempotent(aiida_localhost, tmp_path) -> None:
    """A second run cleans nothing new and reports the already-cleaned calculation as skipped."""
    computer = aiida_localhost
    target = tmp_path / 'run'
    _write_files(target)
    node = _graph(((_calc(computer), _remote(computer, target)),))
    calc_pk = next(iter(node.called)).pk
    assert cleanup_graph_workdirs(node)['cleaned'] == [calc_pk]
    rerun = cleanup_graph_workdirs(node)
    assert rerun == {'cleaned': [], 'skipped': [calc_pk], 'failed': []}
    assert not target.exists()


def test_cleans_failed_and_killed_roots(aiida_localhost, tmp_path) -> None:
    """The helper is explicitly usable after terminal failure and after cancellation."""
    computer = aiida_localhost
    failed_dir, killed_dir = tmp_path / 'failed', tmp_path / 'killed'
    _write_files(failed_dir)
    _write_files(killed_dir)
    failed_calc = _calc(computer, exit_status=400)
    failed = _graph(((failed_calc, _remote(computer, failed_dir)),), exit_status=400)
    assert failed.is_finished
    assert not failed.is_finished_ok
    assert cleanup_graph_workdirs(failed)['cleaned'] == [failed_calc.pk]
    assert not failed_dir.exists()
    killed_calc = _calc(computer)
    killed = _graph(((killed_calc, _remote(computer, killed_dir)),), state=ProcessState.KILLED, exit_status=None)
    assert killed.is_killed
    assert cleanup_graph_workdirs(killed)['cleaned'] == [killed_calc.pk]
    assert not killed_dir.exists()


def test_cleans_unmapped_failure_descendant(aiida_localhost, tmp_path) -> None:
    """A descendant that finished with an unmapped exit status is still cleaned after termination."""
    computer = aiida_localhost
    target = tmp_path / 'run'
    _write_files(target)
    calc = _calc(computer, exit_status=501)
    node = _graph(((calc, _remote(computer, target)),), exit_status=400)
    assert cleanup_graph_workdirs(node)['cleaned'] == [calc.pk]
    assert not target.exists()


def test_refuses_running_root(aiida_localhost, tmp_path) -> None:
    """An unterminated root is refused before anything is touched, stored or not."""
    computer = aiida_localhost
    target = tmp_path / 'run'
    _write_files(target)
    with pytest.raises(ValueError, match='still `created`'):
        cleanup_graph_workdirs(GraphNode())
    running = _graph(((_calc(computer), _remote(computer, target)),), state=ProcessState.RUNNING, exit_status=None)
    assert not running.is_terminated
    with pytest.raises(ValueError, match=f'GraphNode<{running.pk}>'):
        cleanup_graph_workdirs(running)
    assert (target / 'output.out').exists()


def test_refuses_active_descendant(aiida_localhost, tmp_path) -> None:
    """A terminated root with a still-active descendant is refused and nothing is cleaned."""
    computer = aiida_localhost
    active_dir, done_dir = tmp_path / 'active', tmp_path / 'done'
    _write_files(active_dir)
    _write_files(done_dir)
    node = _graph(
        (
            (_calc(computer, state=ProcessState.RUNNING, exit_status=None), _remote(computer, active_dir)),
            (_calc(computer), _remote(computer, done_dir)),
        )
    )
    active = next(child for child in node.called if not child.is_terminated)
    with pytest.raises(ValueError, match=f'CalcJobNode<{active.pk}>'):
        cleanup_graph_workdirs(node)
    assert (active_dir / 'output.out').exists()
    assert (done_dir / 'output.out').exists()


def test_cleanup_errors_recorded_not_raised(aiida_localhost, tmp_path, monkeypatch) -> None:
    """A bounded transport error for one folder is recorded while the other folder is still cleaned."""
    import aiida.orm.utils.remote as remote_utils

    computer = aiida_localhost
    bad_dir, good_dir = tmp_path / 'bad', tmp_path / 'good'
    _write_files(bad_dir)
    _write_files(good_dir)
    bad, good = _calc(computer), _calc(computer)
    node = _graph(((bad, _remote(computer, bad_dir)), (good, _remote(computer, good_dir))))
    real_clean = remote_utils.clean_remote

    def _flaky(transport, path):
        if path == str(bad_dir):
            raise OSError('disk on fire')
        return real_clean(transport, path)

    monkeypatch.setattr(remote_utils, 'clean_remote', _flaky)
    result = cleanup_graph_workdirs(node)
    assert result['cleaned'] == [good.pk]
    assert result['skipped'] == []
    assert len(result['failed']) == 1
    assert result['failed'][0]['pk'] == bad.pk
    assert 'disk on fire' in result['failed'][0]['error']
    assert not good_dir.exists()
    assert (bad_dir / 'output.out').exists()


def test_never_cleans_unlinked_paths(aiida_localhost, tmp_path) -> None:
    """Folders that are not descendant ``remote_folder`` outputs are never touched, ``/tmp`` style or not."""
    computer = aiida_localhost
    decoy = tmp_path / 'stub-pw'
    _write_files(decoy)
    RemoteData(computer=computer, remote_path=str(decoy)).store()
    linked = tmp_path / 'linked'
    _write_files(linked)
    node = _graph(((_calc(computer), _remote(computer, linked)),))
    assert cleanup_graph_workdirs(node)['cleaned'] == [next(iter(node.called)).pk]
    assert (decoy / 'output.out').exists()
    assert not linked.exists()


def test_dry_run_reports_without_touching(aiida_localhost, tmp_path) -> None:
    """A dry run reports what would be cleaned and leaves storage alone."""
    computer = aiida_localhost
    target = tmp_path / 'run'
    _write_files(target)
    node = _graph(((_calc(computer), _remote(computer, target)),))
    calc_pk = next(iter(node.called)).pk
    assert cleanup_graph_workdirs(node, dry_run=True) == {'cleaned': [calc_pk], 'skipped': [], 'failed': []}
    assert (target / 'output.out').exists()


def test_rejects_non_graph_node() -> None:
    """Only a ``GraphNode`` is accepted, so callers cannot aim the helper at arbitrary nodes."""
    with pytest.raises(TypeError, match='needs a `GraphNode`'):
        cleanup_graph_workdirs(Dict())


def test_sweep_graph_remote_task(aiida_localhost, tmp_path) -> None:
    """The in-graph sweep task cleans when enabled and reports ``False`` otherwise without touching storage."""
    computer = aiida_localhost
    enabled_dir, disabled_dir = tmp_path / 'enabled', tmp_path / 'disabled'
    _write_files(enabled_dir)
    _write_files(disabled_dir)
    outputs, node = run_get_node(sweep_graph_remote, folder=_remote(computer, enabled_dir), enabled=True)
    assert node.is_finished_ok
    assert outputs.value is True
    assert not enabled_dir.exists()
    outputs, node = run_get_node(sweep_graph_remote, folder=_remote(computer, disabled_dir), enabled=False)
    assert node.is_finished_ok
    assert outputs.value is False
    assert (disabled_dir / 'output.out').exists()
    outputs, node = run_get_node(sweep_graph_remote, folder=None, enabled=True)
    assert node.is_finished_ok
    assert outputs.value is False


def test_missing_remote_folder_is_skipped(aiida_localhost) -> None:
    """A descendant without a ``remote_folder`` output is skipped, not failed."""
    node = _graph(((_calc(aiida_localhost), None),))
    assert cleanup_graph_workdirs(node) == {'cleaned': [], 'skipped': [next(iter(node.called)).pk], 'failed': []}


def test_unsafe_root_recorded_per_folder(aiida_localhost, tmp_path) -> None:
    """A folder pointing at `/` is refused per folder while the sibling folder is still cleaned."""
    computer = aiida_localhost
    good = tmp_path / 'good'
    _write_files(good)
    bad_calc, good_calc = _calc(computer), _calc(computer)
    node = _graph(((bad_calc, _remote(computer, Path('/'))), (good_calc, _remote(computer, good))))
    result = cleanup_graph_workdirs(node)
    assert result['cleaned'] == [good_calc.pk]
    assert result['skipped'] == []
    assert len(result['failed']) == 1
    assert result['failed'][0]['pk'] == bad_calc.pk
    assert 'filesystem root' in result['failed'][0]['error']
    assert not good.exists()
    assert not bad_calc.outputs.remote_folder.is_cleaned


def test_computer_workdir_refused(aiida_localhost, tmp_path) -> None:
    """A folder equal to the computer work folder is refused without touching storage."""
    import posixpath

    computer = aiida_localhost
    workdir = posixpath.normpath(str(tmp_path / 'workdir'))
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / 'output.out').write_text('data', encoding='utf-8')
    computer.set_workdir(workdir)
    computer.store()
    calc = _calc(computer)
    node = _graph(((calc, _remote(computer, Path(workdir))),))
    result = cleanup_graph_workdirs(node)
    assert result['cleaned'] == []
    assert result['skipped'] == []
    assert len(result['failed']) == 1
    assert 'work folder' in result['failed'][0]['error']
    assert (Path(workdir) / 'output.out').exists()


def test_tmp_folders_still_allowed(aiida_localhost, tmp_path) -> None:
    """Legitimate `/tmp`-rooted stand-in remotes are cleaned, not blanket-denied by the guard."""
    computer = aiida_localhost
    target = tmp_path / 'run'
    _write_files(target)
    node = _graph(((_calc(computer), _remote(computer, target)),))
    assert cleanup_graph_workdirs(node)['cleaned'] == [next(iter(node.called)).pk]
    assert not target.exists()


@pytest.mark.parametrize('path', ['/tmp', '/var/tmp', '/home'])
@pytest.mark.parametrize('dry_run', [False, True])
def test_shared_roots_refused_even_in_preview(aiida_localhost, path, dry_run) -> None:
    """A stored remote path is not evidence that a shared directory is safe to delete."""
    calc = _calc(aiida_localhost)
    node = _graph(((calc, _remote(aiida_localhost, Path(path))),))
    result = cleanup_graph_workdirs(node, dry_run=dry_run)
    assert result['cleaned'] == []
    assert result['failed'][0]['pk'] == calc.pk
    assert 'shared' in result['failed'][0]['error']
    assert not calc.outputs.remote_folder.is_cleaned


def test_relative_remote_path_refused(aiida_localhost) -> None:
    """Remote working directories need an absolute path before cleanup is allowed."""
    calc = _calc(aiida_localhost)
    node = _graph(((calc, _remote(aiida_localhost, Path('relative'))),))
    result = cleanup_graph_workdirs(node)
    assert result['cleaned'] == []
    assert 'absolute' in result['failed'][0]['error']


def test_missing_authentication_is_recorded(aiida_localhost, tmp_path, monkeypatch) -> None:
    """Expected authentication configuration errors do not overwrite the graph failure."""
    from aiida.common.exceptions import NotExistent

    calc = _calc(aiida_localhost, exit_status=400)
    node = _graph(((calc, _remote(aiida_localhost, tmp_path / 'run')),), exit_status=400)

    def missing_authinfo(self):
        raise NotExistent('test authentication record is missing')

    monkeypatch.setattr(RemoteData, 'get_authinfo', missing_authinfo)
    result = cleanup_graph_workdirs(node)
    assert result['cleaned'] == []
    assert 'NotExistent' in result['failed'][0]['error']
    assert node.exit_status == 400


def test_cleanup_leaves_process_outcome_unchanged(aiida_localhost, tmp_path) -> None:
    """Cleaning records folders but never modifies process state or exit statuses."""
    computer = aiida_localhost
    target = tmp_path / 'run'
    _write_files(target)
    calc = _calc(computer, exit_status=400)
    node = _graph(((calc, _remote(computer, target)),), exit_status=400)
    state_before, exit_before = node.process_state, node.exit_status
    calc_state, calc_exit = calc.process_state, calc.exit_status
    result = cleanup_graph_workdirs(node)
    assert result['cleaned'] == [calc.pk]
    assert (node.process_state, node.exit_status) == (state_before, exit_before)
    assert (calc.process_state, calc.exit_status) == (calc_state, calc_exit)
