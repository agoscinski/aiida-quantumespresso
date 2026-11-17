############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Provenance-preserving merging of ``ph.x`` recover-chain outputs for the native graphs.

There are two merging routes, with different provenance folds over the same mathematics. The
native :func:`ph_base_graph <aiida_quantumespresso.workflows.graphs.base.ph_base_graph>` folds
recover runs incrementally in-graph through ``merge_ph_pair``: each step takes the previous
aggregate plus the current run's original ``output_parameters`` as direct inputs, so the final
``Dict`` reaches the members through a chain of pairwise merges; a ``PhMergeCheck`` carry
validates the multi-run chain at the end (missing originals or a parsed q-point count outside
the requested ``INPUTPH`` ``start_q``/``last_q`` range fails loudly, reset on a restart from
scratch, skipped for ``ONLY_INITIALIZATION`` and single runs, mirroring ``ERROR_MERGING_QPOINTS``).
This module's :func:`merge_ph_graph_outputs` is the external post-hoc route: it walks the stored
chain of a terminated graph and calls the existing ``merge_ph_outputs`` calcfunction
(:mod:`aiida_quantumespresso.calculations.functions.merge_ph_outputs`) once with one ``output_<n>``
input per member, so every member ``Dict`` links directly into a single merged node. The
post-hoc route stays for terminated graphs and explicit branch selection; neither route claims
full ``ph`` graph-result parity yet.

There is deliberately no container-forwarded in-graph ``@task`` merge beyond that pairwise fold.
A recover chain has a dynamic length, so its members cannot be wired as static task inputs in the
graph body. ``orm.List`` holds plain values rather than node references, while
``DataList``/``DataDict`` hold owned copies made by copy-on-insert with new identities and no
provenance links, so routing the per-iteration outputs through a container would detach the merge
inputs from their provenance instead of keeping direct ``INPUT_CALC`` links. No core collector
feature is requested for this: the chain is finite stored provenance, and the external merge runs
after termination, owned externally like cleanup.

Contract:

* :func:`get_ph_recover_chain` walks the stored descendants of a terminated graph node, mirroring
  ``PhBaseWorkChain._get_recover_chain``: from the last ``ph.x`` calculation back through the ``parent_folder``
  creators while each step ran with ``recover = True``. Calculations outside the chain (a restart from scratch
  after an unhandled failure recomputes its q-points) are excluded, exactly as in the legacy work chain. When
  the graph holds several disconnected ``ph.x`` branches, there is no single last calculation: the helper raises
  instead of silently choosing one, unless the caller passes ``last_calc`` explicitly.
* :func:`merge_ph_graph_outputs` merges that chain by calling the existing ``merge_ph_outputs`` calcfunction
  with one ``output_<n>`` input per member, so every member ``Dict`` keeps a direct ``INPUT_CALC`` link into
  the merged node. A single-member chain returns the existing ``output_parameters`` without creating a node,
  and an ``ONLY_INITIALIZATION`` run is returned unmerged, both mirroring ``create_merged_output``. The merge
  returns the merged ``Dict`` only; it is not full graph-result parity. A chain member without stored
  ``output_parameters`` (for example a failed graph) raises a meaningful ``ValueError`` instead of merging a
  partial chain.
* :func:`merge_ph_output_dicts` is the pure mapping-level merge for tests and for callers that already hold
  plain dictionaries. It duplicates the calcfunction body line for line on purpose: the calcfunction module is
  owned elsewhere, so the duplication is tested for equivalence in ``tests/graphs/test_ph_merge.py`` rather
  than refactored away here.
"""

import typing as t
from collections.abc import Mapping, Sequence

from aiida.orm import CalcJobNode, Dict, GraphNode, RemoteData

from aiida_quantumespresso.calculations.functions.merge_ph_outputs import (
    merge_ph_outputs as _merge_ph_outputs_calcfunction,
)

__all__ = (
    'PH_CALCULATION_PROCESS_TYPE',
    'get_ph_recover_chain',
    'merge_ph_graph_outputs',
    'merge_ph_output_dicts',
)

#: Process type identifying the ``ph.x`` calculations of a graph's recover chain.
PH_CALCULATION_PROCESS_TYPE = 'aiida.calculations:quantumespresso.ph'


def merge_ph_output_dicts(outputs: Sequence[Mapping[str, t.Any]]) -> dict[str, t.Any]:
    """Merge ``ph.x`` output-parameter mappings, in chain order.

    Pure mapping-level port of the ``merge_ph_outputs`` calcfunction body: wall times add up, the
    per-q-point irrep counts concatenate, dynamical matrices without ``mode_symmetry`` are skipped as
    incomplete, and later outputs win on any other shared key.

    :param outputs: the parsed output-parameter mappings, in the order the calculations ran.
    :return: the merged mapping.
    """
    merged: dict[str, t.Any] = {}
    total_walltime = 0
    number_of_qpoints = 0
    number_irreps: list = []
    for output in outputs:
        current = dict(output)
        num_irreps_per_q = current.pop('number_of_irr_representations_for_each_q', [])
        number_of_qpoints += len(num_irreps_per_q)
        number_irreps.extend(num_irreps_per_q)
        total_walltime += current.pop('wall_time_seconds', 0)
        for key, value in current.items():
            if 'dynamical_matrix_' in key and 'mode_symmetry' not in value:
                continue
            merged[key] = value
    merged['wall_time_seconds'] = total_walltime
    merged['number_of_irr_representations_for_each_q'] = number_irreps
    merged['number_of_qpoints'] = number_of_qpoints
    return merged


def _is_ph_calculation(node: t.Any) -> bool:
    """Return whether a node is a stored ``ph.x`` calculation."""
    return isinstance(node, CalcJobNode) and node.process_type == PH_CALCULATION_PROCESS_TYPE


def _recover_requested(node: CalcJobNode) -> bool:
    """Return whether a ``ph.x`` calculation ran with ``INPUTPH.recover = True``."""
    try:
        parameters = node.inputs.parameters.get_dict()
    except AttributeError:
        return False
    return bool(parameters.get('INPUTPH', {}).get('recover', False))


def _resolve_last_calc(children: list[CalcJobNode], last_calc: CalcJobNode | int | None) -> CalcJobNode:
    """Return the chain head to walk from, rejecting ambiguous disconnected branches.

    Heads are ``ph.x`` calculations no other ``ph.x`` calculation recovered from (that is, no other member's
    ``parent_folder`` was created by them). A single head means one recover chain, possibly preceded by
    restart-from-scratch ancestors that recomputed their q-points. Several heads mean disconnected branches with
    no single last calculation, so the helper raises instead of silently picking the latest ``ctime``.

    :param children: the graph's ``ph.x`` descendants.
    :param last_calc: an explicit selector: a member node or pk.
    :return: the calculation to walk back from.
    :raises ValueError: if the selector is unknown, or when branches are disconnected and no selector is given.
    """
    by_pk = {child.pk: child for child in children if child.pk is not None}
    if last_calc is not None:
        wanted: int | None = last_calc.pk if isinstance(last_calc, CalcJobNode) else int(last_calc)
        if wanted is None or wanted not in by_pk:
            msg = f'`last_calc` {wanted} is not a `ph.x` descendant of this graph; known members are {sorted(by_pk)}.'
            raise ValueError(msg)
        return by_pk[wanted]
    parent_pks: set[int] = set()
    for child in children:
        try:
            parent_folder = child.inputs.parent_folder
        except AttributeError:
            continue
        if not isinstance(parent_folder, RemoteData):
            continue
        creator = parent_folder.creator
        if creator is not None and creator.pk is not None and creator.pk in by_pk:
            parent_pks.add(creator.pk)
    heads = [child for child in children if child.pk not in parent_pks]
    if len(heads) == 1:
        return heads[0]
    pks = sorted(head.pk for head in heads if head.pk is not None)
    msg = (
        f'graph holds {len(heads)} disconnected `ph.x` branches ending in {pks}; pass `last_calc` '
        'to select which branch to merge instead of silently choosing one.'
    )
    raise ValueError(msg)


def get_ph_recover_chain(node: GraphNode, last_calc: CalcJobNode | int | None = None) -> list[CalcJobNode]:
    """Return the ``ph.x`` recover chain of a terminated graph, in the order the calculations ran.

    Starting from the last ``ph.x`` calculation, follow each calculation's ``parent_folder`` creator back while
    that creator is one of the graph's own ``ph.x`` calculations and the step ran with ``recover = True``. The
    walk stops at a restart from scratch (whose q-points were recomputed and must not be merged) and at a parent
    folder created outside the graph (for example the ``pw.x`` run the chain recovered from).

    When the graph holds several disconnected ``ph.x`` branches there is no single last calculation: the helper
    raises instead of silently choosing the latest ``ctime``. Pass ``last_calc`` (a member node or pk) to select
    the branch explicitly.

    :param node: the terminated ``GraphNode`` whose descendants hold the chain.
    :param last_calc: an explicit branch selector for graphs with disconnected ``ph.x`` branches.
    :return: the chain members, oldest first; empty when the graph ran no ``ph.x`` calculation.
    :raises TypeError: if ``node`` is not a ``GraphNode``.
    :raises ValueError: if the root is not terminated, if branches are disconnected and no selector is given,
        or if the selector names no member.
    """
    if not isinstance(node, GraphNode):
        msg = f'`get_ph_recover_chain` needs a `GraphNode`, got `{type(node).__name__}`.'
        raise TypeError(msg)
    if not node.is_terminated:
        state = node.process_state
        msg = (
            f'refusing to read the recover chain of GraphNode<{node.pk}>: it is still '
            f'`{state.value if state is not None else "created"}`; read the chain after termination.'
        )
        raise ValueError(msg)
    children = [
        descendant
        for descendant in node.called_descendants
        if isinstance(descendant, CalcJobNode) and _is_ph_calculation(descendant)
    ]
    if not children:
        return []
    children_pks = {child.pk for child in children}
    current = _resolve_last_calc(children, last_calc)
    chain = [current]
    while _recover_requested(current):
        try:
            parent_folder = current.inputs.parent_folder
        except AttributeError:
            break
        if not isinstance(parent_folder, RemoteData):
            break
        previous = parent_folder.creator
        if previous is None or previous.pk not in children_pks:
            break
        chain.insert(0, previous)
        current = previous
    return chain


def merge_ph_graph_outputs(node: GraphNode, last_calc: CalcJobNode | int | None = None) -> Dict:
    """Merge the ``ph.x`` recover-chain outputs of a terminated graph into one ``Dict``.

    This is an ordinary function, not a ``@task``: the external owner calls it after the graph terminated, when
    the dynamic chain is finite stored provenance. The merge itself runs through the existing
    ``merge_ph_outputs`` calcfunction with one ``output_<n>`` input per chain member, so provenance is
    preserved: every member ``Dict`` links directly into the merged node. The result is the merged ``Dict``
    only, not full graph-result parity: a terminated failed graph whose members miss ``output_parameters``
    raises a meaningful ``ValueError`` instead of returning a partial merge.

    :param node: the terminated ``GraphNode`` whose chain should be merged.
    :param last_calc: an explicit branch selector for graphs with disconnected ``ph.x`` branches.
    :return: the existing ``output_parameters`` for a single-member chain or an ``ONLY_INITIALIZATION`` run,
        otherwise the merged ``Dict`` created by the calcfunction.
    :raises ValueError: if the root is not terminated, if the graph ran no ``ph.x`` calculation, if a chain
        member exposes no ``output_parameters``, or if the parsed q-points do not cover the requested range
        (the legacy ``ERROR_MERGING_QPOINTS`` case).
    """
    chain = get_ph_recover_chain(node, last_calc=last_calc)
    if not chain:
        msg = f'GraphNode<{node.pk}> ran no `ph.x` calculation; nothing to merge.'
        raise ValueError(msg)
    last = chain[-1]
    try:
        output_parameters = last.outputs.output_parameters
    except AttributeError:
        msg = f'PhCalculation<{last.pk}> exposes no `output_parameters`; nothing to merge.'
        raise ValueError(msg)
    try:
        settings = last.inputs.settings.get_dict()
    except AttributeError:
        settings = {}
    if settings.get('ONLY_INITIALIZATION', False):
        return output_parameters
    if len(chain) == 1:
        return output_parameters
    member_outputs: list[Dict] = []
    for member in chain:
        try:
            member_outputs.append(member.outputs.output_parameters)
        except AttributeError:
            msg = (
                f'PhCalculation<{member.pk}> exposes no `output_parameters`; refusing to merge '
                f'a partial recover chain of GraphNode<{node.pk}>.'
            )
            raise ValueError(msg) from None
    inputs = {f'output_{index + 1}': output for index, output in enumerate(member_outputs)}
    try:
        parameters = last.inputs.parameters.get_dict().get('INPUTPH', {})
    except AttributeError:
        parameters = {}
    expected = parameters.get('last_q', None)
    if expected is not None:
        expected = expected - parameters.get('start_q', 1) + 1
    found = sum(len(output.get_dict().get('number_of_irr_representations_for_each_q', [])) for output in member_outputs)
    if expected is not None and found != expected:
        msg = (
            f'only {found} of {expected} q-points were parsed; refusing to merge a partial recover chain '
            f'of GraphNode<{node.pk}>.'
        )
        raise ValueError(msg)
    return _merge_ph_outputs_calcfunction(**inputs)
