############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Post-termination cleanup for the native QE graphs.

The native graphs in this package sweep remote folders through explicit ``@task`` placements (for example
``sweep_remote``/``sweep_stage`` in :mod:`aiida_quantumespresso.workflows.graphs.pdos`). Those tasks only run when
their data and ordering edges fire: on the success path and on explicitly wired handled-failure branches. They do
**not** run when a placement fails with an unmapped error or when the graph is cancelled, because the dependent
tasks never execute. Graphs also have no ``on_terminated`` hook, so there is no automatic ``finally`` to claim.

This module therefore splits cleanup into two owners with an explicit contract:

* **In-graph sweeps** (:func:`sweep_graph_remote`): a ``@task`` the coordinator wires with edges for the success
  path and for handled-failure branches. It covers ``clean_workdir`` on those paths only.
* **Terminal failure/cancellation** (:func:`cleanup_graph_workdirs`): an ordinary function the external owner
  (caller/coordinator) runs **after** the graph node terminated. It walks the descendant ``CalcJob`` nodes and
  cleans their remote folders. It refuses a running root and refuses active descendants, it is idempotent, and it
  records bounded per-folder errors instead of raising, so cleanup problems never overwrite the process outcome.
  The owner call is required for unexpected failure/cancellation paths: success sweeps never run there, and they
  are not automatic ``on_terminated`` semantics.

``BaseRestartWorkChain.on_terminated`` and the legacy relax/PDOS termination sweeps are the reference for this
policy, not a guarantee: they also only run when the daemon reaches termination, so nothing here claims cleanup
after a crashed worker either. The owner simply re-runs the idempotent helper once the node is known terminated.

Safety: the helper only ever cleans ``RemoteData`` nodes reached as ``remote_folder`` outputs of the graph's own
descendant ``CalcJob`` nodes. It takes no path argument, so a caller cannot aim it at an arbitrary path. That
scoping alone is not a path guarantee: a descendant ``RemoteData`` may legitimately point anywhere, including the
``/tmp`` folders used by the test suite. The helper therefore additionally refuses per folder (recorded as
``failed``, never raised) to clean the filesystem root ``/``, the shared ``/home`` directory, and the computer's
own work folder, leaving those folders untouched while still cleaning the remaining folders.
"""

import posixpath
import typing as t

from aiida.common.exceptions import ConfigurationError, NotExistent
from aiida.engine import task
from aiida.orm import CalcJobNode, GraphNode, ProcessNode, RemoteData

__all__ = (
    'CleanupFailure',
    'CleanupResult',
    'cleanup_graph_workdirs',
    'sweep_graph_remote',
)


class CleanupFailure(t.TypedDict):
    """One bounded cleanup failure: the calculation whose folder could not be cleaned, and why."""

    pk: int
    error: str


class CleanupResult(t.TypedDict):
    """Outcome of :func:`cleanup_graph_workdirs`: calculation pks grouped by what happened to their folders."""

    cleaned: list[int]
    skipped: list[int]
    failed: list[CleanupFailure]


def _state_name(node: ProcessNode) -> str:
    """Return the process-state value of a node, naming unstored nodes as created."""
    state = node.process_state
    return state.value if state is not None else 'created'


def _unsafe_reason(folder: RemoteData) -> str | None:
    """Return why a ``RemoteData`` folder must not be cleaned, or ``None`` when it may be cleaned.

    Refuse relative paths, filesystem roots, top-level shared directories, shared temporary roots,
    and the computer's configured work folder. Calculation-specific directories below temporary roots
    remain valid. This is a defensive guard, not a sandbox: callers must trust the stored remote paths.

    :param folder: the ``RemoteData`` that would be cleaned.
    :return: the refusal reason, or ``None`` when cleaning may proceed.
    """
    try:
        remote_path = folder.get_remote_path()
    except AttributeError:
        return 'refusing to clean: `remote_path` is not set.'
    if not posixpath.isabs(remote_path):
        return f'refusing to clean `{remote_path}`: the remote path must be absolute.'
    normalized = posixpath.normpath(remote_path)
    if normalized == '/':
        return f'refusing to clean `{remote_path}`: it is the filesystem root.'
    if normalized.count('/') == 1 or normalized == '/var/tmp':
        return f'refusing to clean `{remote_path}`: it is a shared top-level or temporary directory.'
    computer = folder.computer
    if computer is None:
        return 'refusing to clean: the computer is not set.'
    try:
        workdir = computer.get_workdir()
    except ValueError:
        workdir = None
    if workdir is not None and normalized == posixpath.normpath(workdir):
        return f'refusing to clean `{remote_path}`: it is the computer work folder.'
    return None


def _clean_folder(folder: RemoteData) -> str | None:
    """Clean one remote folder, returning ``None`` on success or the caught error message.

    Expected filesystem, configuration, and missing-authentication-record errors are recorded after
    one attempt per folder. Unexpected programming errors still propagate. Folders refused by :func:`_unsafe_reason` are reported as errors without
    touching storage.

    :param folder: the ``RemoteData`` whose content should be removed.
    :return: ``None`` when cleaned, otherwise the recorded error.
    """
    refused = _unsafe_reason(folder)
    if refused is not None:
        return f'ValueError: {refused}'
    try:
        folder._clean()
    except (OSError, KeyError, ValueError, ConfigurationError, NotExistent) as exception:
        return f'{type(exception).__name__}: {exception}'
    return None


def cleanup_graph_workdirs(node: GraphNode, *, dry_run: bool = False) -> CleanupResult:
    """Clean the remote folders of a terminated graph's descendant ``CalcJob`` nodes.

    This is an ordinary function, not a ``@task``: it runs outside the graph, owned by the caller, after the
    graph node terminated. It works uniformly after success, after mapped or unmapped failure, and after
    cancellation, because it reads stored provenance instead of relying on task edges that only fire on some
    paths. It never changes process state or exit statuses. Expected filesystem/configuration errors are
    recorded in the result; unexpected programming errors propagate without changing the process outcome.

    Folders refused by the root/work-folder guard are reported under ``failed`` with the refusal reason and
    are left untouched; the remaining folders are still processed. Process state and exit statuses are never
    modified.

    :param node: the terminated ``GraphNode`` whose descendant calculation folders should be cleaned.
    :param dry_run: when ``True``, report what would be cleaned without touching storage.
    :return: the pks of cleaned, skipped (missing or already cleaned folder), and failed calculations.
    :raises TypeError: if ``node`` is not a ``GraphNode``.
    :raises ValueError: if the root is not terminated, or if any descendant process is still active. The message
        names the offending node so the owner knows termination is incomplete.
    """
    if not isinstance(node, GraphNode):
        msg = f'`cleanup_graph_workdirs` needs a `GraphNode`, got `{type(node).__name__}`.'
        raise TypeError(msg)
    if not node.is_terminated:
        msg = (
            f'refusing to clean GraphNode<{node.pk}>: it is still `{_state_name(node)}`; '
            'run the helper again after termination.'
        )
        raise ValueError(msg)
    for descendant in node.called_descendants:
        if isinstance(descendant, ProcessNode) and not descendant.is_terminated:
            msg = (
                f'refusing to clean GraphNode<{node.pk}>: descendant {type(descendant).__name__}'
                f'<{descendant.pk}> is still `{_state_name(descendant)}`.'
            )
            raise ValueError(msg)

    result: CleanupResult = {'cleaned': [], 'skipped': [], 'failed': []}
    for descendant in node.called_descendants:
        if not isinstance(descendant, CalcJobNode):
            continue
        try:
            folder = descendant.outputs.remote_folder
        except AttributeError:
            result['skipped'].append(descendant.pk)
            continue
        if not isinstance(folder, RemoteData) or folder.is_cleaned:
            result['skipped'].append(descendant.pk)
            continue
        refused = _unsafe_reason(folder)
        if refused is not None:
            result['failed'].append({'pk': descendant.pk, 'error': f'ValueError: {refused}'})
            continue
        if dry_run:
            result['cleaned'].append(descendant.pk)
            continue
        error = _clean_folder(folder)
        if error is None:
            result['cleaned'].append(descendant.pk)
        else:
            result['failed'].append({'pk': descendant.pk, 'error': error})
    return result


@task
def sweep_graph_remote(folder: RemoteData | None = None, enabled: bool = False) -> bool:
    """Clean one remote folder when enabled, reporting whether anything was removed.

    This is the in-graph counterpart of :func:`cleanup_graph_workdirs` for the success path and for explicitly
    wired handled-failure branches: the coordinator places it with data or ``after`` edges so it runs exactly
    where the graph decides to clean. It does not cover unexpected failures or cancellation; those paths never
    execute dependent tasks, so the external owner calls :func:`cleanup_graph_workdirs` after termination.

    :param folder: the remote folder to clean, if any.
    :param enabled: whether cleaning is requested at all (the ``clean_workdir`` flag).
    :return: whether a remote folder was cleaned.
    """
    if not enabled or folder is None:
        return False
    return _clean_folder(folder) is None
