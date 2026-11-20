############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Tests for the provenance-preserving ``ph.x`` recover-chain merge.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_ph_merge.py -q

The merge mathematics is checked at the pure mapping level and for equivalence with the existing
``merge_ph_outputs`` calcfunction. Chain walking and the external-owner merge run against real stored
provenance (a ``GraphNode`` with ``ph.x`` ``CalcJobNode`` descendants linked through ``parent_folder``
``RemoteData`` nodes): no scheduler and no calculation execution is involved.
"""

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida.common.links import LinkType
from aiida.common.processes import ProcessState
from aiida.engine import run_get_node
from aiida.orm import CalcJobNode, Dict, GraphNode, RemoteData

from aiida_quantumespresso.calculations.functions.merge_ph_outputs import (
    merge_ph_outputs as merge_ph_outputs_calcfunction,
)
from aiida_quantumespresso.workflows.graphs import base as graphs_base
from aiida_quantumespresso.workflows.graphs.ph import (
    PH_CALCULATION_PROCESS_TYPE,
    get_ph_recover_chain,
    merge_ph_graph_outputs,
    merge_ph_output_dicts,
)

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]

_OUTPUT_A = {
    'number_of_irr_representations_for_each_q': [3, 3],
    'wall_time_seconds': 10.0,
    'dynamical_matrix_1': {'mode_symmetry': [1], 'frequencies': [100.0]},
    'dynamical_matrix_2': {'frequencies': [200.0]},
    'convergence': 'first',
}

_OUTPUT_B = {
    'number_of_irr_representations_for_each_q': [3, 3],
    'wall_time_seconds': 20.0,
    'dynamical_matrix_3': {'mode_symmetry': [1], 'frequencies': [300.0]},
    'convergence': 'second',
}


def _stored_dict(data: dict) -> Dict:
    """Return a stored ``Dict``."""
    node = Dict(data)
    node.store()
    return node


def _ph_chain(computer, *, recover_second: bool = True, irreps_second: tuple = (3, 3), only_init_second: bool = False):
    """Return ``(graph, calc_first, calc_second)`` of a two-step ``ph.x`` recover chain.

    The first calculation recovers from a folder outside the graph (the ``pw.x`` parent), so the walk stops
    there; the second recovers from the first when ``recover_second`` holds. The requested range covers four
    q-points, two parsed per member.
    """
    node = GraphNode()
    node.set_process_state(ProcessState.FINISHED)
    node.store()
    outside = RemoteData(computer=computer, remote_path='/ph/pw-parent')
    outside.store()

    def _calc(
        label: str, parameters: Dict, parent_folder: RemoteData, irreps: tuple, settings: Dict | None = None
    ) -> CalcJobNode:
        calc = CalcJobNode(computer=computer)
        calc.process_type = PH_CALCULATION_PROCESS_TYPE
        calc.set_process_state(ProcessState.FINISHED)
        calc.set_exit_status(0)
        calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label=label)
        calc.base.links.add_incoming(parameters, link_type=LinkType.INPUT_CALC, link_label='parameters')
        calc.base.links.add_incoming(parent_folder, link_type=LinkType.INPUT_CALC, link_label='parent_folder')
        if settings is not None:
            calc.base.links.add_incoming(settings, link_type=LinkType.INPUT_CALC, link_label='settings')
        calc.store()
        output = _stored_dict(dict(_OUTPUT_A, number_of_irr_representations_for_each_q=list(irreps)))
        output.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label='output_parameters')
        output.store()
        folder = RemoteData(computer=computer, remote_path=f'/ph/{label}')
        folder.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label='remote_folder')
        folder.store()
        return calc

    first = _calc(
        'call_1',
        _stored_dict({'INPUTPH': {'recover': True, 'start_q': 1, 'last_q': 4}}),
        outside,
        (3, 3),
    )
    first_folder = first.outputs.remote_folder
    second = _calc(
        'call_2',
        _stored_dict({'INPUTPH': {'recover': recover_second, 'start_q': 1, 'last_q': 4}}),
        first_folder,
        irreps_second,
        _stored_dict({'ONLY_INITIALIZATION': True}) if only_init_second else None,
    )
    return node, first, second


def test_merge_ph_output_dicts() -> None:
    """The pure merge adds wall times, concatenates irreps, skips incomplete matrices, later keys win."""
    merged = merge_ph_output_dicts([_OUTPUT_A, _OUTPUT_B])
    assert merged['wall_time_seconds'] == 30.0
    assert merged['number_of_irr_representations_for_each_q'] == [3, 3, 3, 3]
    assert merged['number_of_qpoints'] == 4
    assert merged['dynamical_matrix_1'] == {'mode_symmetry': [1], 'frequencies': [100.0]}
    assert 'dynamical_matrix_2' not in merged
    assert merged['dynamical_matrix_3'] == {'mode_symmetry': [1], 'frequencies': [300.0]}
    assert merged['convergence'] == 'second'
    assert _OUTPUT_A['wall_time_seconds'] == 10.0


def test_merge_matches_calcfunction() -> None:
    """The pure helper and the existing calcfunction agree on the same fixture."""
    merged = merge_ph_output_dicts([_OUTPUT_A, _OUTPUT_B])
    outputs, node = run_get_node(merge_ph_outputs_calcfunction, output_1=Dict(_OUTPUT_A), output_2=Dict(_OUTPUT_B))
    assert node.is_finished_ok
    assert outputs.get_dict() == merged
    assert {link.link_label for link in node.base.links.get_incoming().all()} == {'output_1', 'output_2'}


def test_recover_chain_two_members(aiida_localhost) -> None:
    """The chain walks back through the recovering parent and merges with provenance links."""
    node, first, second = _ph_chain(aiida_localhost)
    assert [member.pk for member in get_ph_recover_chain(node)] == [first.pk, second.pk]
    merged = merge_ph_graph_outputs(node)
    assert merged.get_dict()['wall_time_seconds'] == 20.0
    assert merged.get_dict()['number_of_qpoints'] == 4
    assert merged.get_dict()['number_of_irr_representations_for_each_q'] == [3, 3, 3, 3]
    creator = merged.creator
    assert creator is not None
    incoming = {link.link_label: link.node.pk for link in creator.base.links.get_incoming().all()}
    assert incoming == {'output_1': first.outputs.output_parameters.pk, 'output_2': second.outputs.output_parameters.pk}


def test_restart_from_scratch_excluded(aiida_localhost) -> None:
    """A member that restarted from scratch recomputed its q-points and is not merged."""
    node, _, second = _ph_chain(aiida_localhost, recover_second=False)
    assert [member.pk for member in get_ph_recover_chain(node)] == [second.pk]
    merged = merge_ph_graph_outputs(node)
    assert merged.uuid == second.outputs.output_parameters.uuid


def test_single_member_returns_existing(aiida_localhost) -> None:
    """A lone ``ph.x`` run returns its stored outputs without creating a merge node."""
    node = GraphNode()
    node.set_process_state(ProcessState.FINISHED)
    node.store()
    calc = CalcJobNode(computer=aiida_localhost)
    calc.process_type = PH_CALCULATION_PROCESS_TYPE
    calc.set_process_state(ProcessState.FINISHED)
    calc.set_exit_status(0)
    calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label='call_1')
    calc.base.links.add_incoming(
        _stored_dict({'INPUTPH': {'recover': False, 'start_q': 1, 'last_q': 2}}),
        link_type=LinkType.INPUT_CALC,
        link_label='parameters',
    )
    calc.store()
    output = _stored_dict(dict(_OUTPUT_A))
    output.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label='output_parameters')
    output.store()
    assert get_ph_recover_chain(node) == [calc]
    assert merge_ph_graph_outputs(node).uuid == output.uuid


def test_coverage_mismatch_raises(aiida_localhost) -> None:
    """A partial chain (fewer parsed q-points than requested) refuses to merge."""
    node, _, _ = _ph_chain(aiida_localhost, irreps_second=(3,))
    with pytest.raises(ValueError, match='only 3 of 4 q-points were parsed'):
        merge_ph_graph_outputs(node)


def test_only_initialization_returns_unmerged(aiida_localhost) -> None:
    """An ``ONLY_INITIALIZATION`` run returns its outputs without merging."""
    node, _, second = _ph_chain(aiida_localhost, only_init_second=True)
    assert merge_ph_graph_outputs(node).uuid == second.outputs.output_parameters.uuid


def test_no_ph_calculations_raises(aiida_localhost) -> None:
    """A graph that ran no ``ph.x`` calculation has nothing to merge."""
    node = GraphNode()
    node.set_process_state(ProcessState.FINISHED)
    node.store()
    assert get_ph_recover_chain(node) == []
    with pytest.raises(ValueError, match='ran no `ph.x` calculation'):
        merge_ph_graph_outputs(node)


def test_refuses_running_root() -> None:
    """The chain is only read after termination."""
    with pytest.raises(ValueError, match='still `created`'):
        get_ph_recover_chain(GraphNode())
    with pytest.raises(TypeError, match='needs a `GraphNode`'):
        get_ph_recover_chain(Dict())


def _ph_branch(computer, node, label: str, recover: bool = True):
    """Append one standalone one-member ``ph.x`` branch rooted outside the graph."""
    outside = RemoteData(computer=computer, remote_path=f'/ph/pw-parent-{label}')
    outside.store()
    calc = CalcJobNode(computer=computer)
    calc.process_type = PH_CALCULATION_PROCESS_TYPE
    calc.set_process_state(ProcessState.FINISHED)
    calc.set_exit_status(0)
    calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label=label)
    calc.base.links.add_incoming(
        _stored_dict({'INPUTPH': {'recover': recover, 'start_q': 1, 'last_q': 1}}),
        link_type=LinkType.INPUT_CALC,
        link_label='parameters',
    )
    calc.base.links.add_incoming(outside, link_type=LinkType.INPUT_CALC, link_label='parent_folder')
    calc.store()
    output = _stored_dict(dict(_OUTPUT_A, number_of_irr_representations_for_each_q=[3]))
    output.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label='output_parameters')
    output.store()
    return calc


def test_disconnected_branches_raise(aiida_localhost) -> None:
    """Two disconnected ``ph.x`` branches raise instead of silently picking the latest ``ctime``."""
    node = GraphNode()
    node.set_process_state(ProcessState.FINISHED)
    node.store()
    first = _ph_branch(aiida_localhost, node, 'call_1')
    second = _ph_branch(aiida_localhost, node, 'call_2')
    with pytest.raises(ValueError, match='disconnected'):
        get_ph_recover_chain(node)
    with pytest.raises(ValueError, match='disconnected'):
        merge_ph_graph_outputs(node)
    assert {first.pk, second.pk} == {child.pk for child in node.called}


def test_explicit_last_calc_selects_branch(aiida_localhost) -> None:
    """An explicit ``last_calc`` selector merges the chosen disconnected branch."""
    node = GraphNode()
    node.set_process_state(ProcessState.FINISHED)
    node.store()
    first = _ph_branch(aiida_localhost, node, 'call_1')
    second = _ph_branch(aiida_localhost, node, 'call_2')
    assert get_ph_recover_chain(node, last_calc=second.pk) == [second]
    assert get_ph_recover_chain(node, last_calc=second) == [second]
    merged = merge_ph_graph_outputs(node, last_calc=second.pk)
    assert merged.uuid == second.outputs.output_parameters.uuid
    assert get_ph_recover_chain(node, last_calc=first.pk) == [first]
    with pytest.raises(ValueError, match='not a `ph.x` descendant'):
        get_ph_recover_chain(node, last_calc=123456789)


def test_missing_output_parameters_raise(aiida_localhost) -> None:
    """A failed chain member without ``output_parameters`` raises a meaningful error."""
    node = GraphNode()
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(400)
    node.store()
    calc = CalcJobNode(computer=aiida_localhost)
    calc.process_type = PH_CALCULATION_PROCESS_TYPE
    calc.set_process_state(ProcessState.FINISHED)
    calc.set_exit_status(410)
    calc.base.links.add_incoming(node, link_type=LinkType.CALL_CALC, link_label='call_1')
    calc.base.links.add_incoming(
        _stored_dict({'INPUTPH': {'recover': False, 'start_q': 1, 'last_q': 1}}),
        link_type=LinkType.INPUT_CALC,
        link_label='parameters',
    )
    calc.store()
    assert get_ph_recover_chain(node) == [calc]
    with pytest.raises(ValueError, match='exposes no `output_parameters`'):
        merge_ph_graph_outputs(node)


def test_base_merge_ph_pair_matches_pure_helper() -> None:
    """The in-graph incremental merge task equals the pure mapping merge."""
    outputs, node = run_get_node(graphs_base.merge_ph_pair, first=Dict(_OUTPUT_A), second=Dict(_OUTPUT_B))
    assert node.is_finished_ok
    assert outputs.get_dict() == merge_ph_output_dicts([_OUTPUT_A, _OUTPUT_B])
    incoming = {link.link_label for link in node.base.links.get_incoming().all()}
    assert incoming == {'first', 'second'}


def test_base_ph_run_recovered_reads_inputph() -> None:
    """The recover gate mirrors the legacy ``INPUTPH.recover`` check."""
    recovered, _ = run_get_node(graphs_base.ph_run_recovered, parameters={'INPUTPH': {'recover': True}})
    assert recovered.value is True
    scratch, _ = run_get_node(graphs_base.ph_run_recovered, parameters={'INPUTPH': {'recover': False}})
    assert scratch.value is False
    missing, _ = run_get_node(graphs_base.ph_run_recovered, parameters={})
    assert missing.value is False


def test_base_seed_ph_merged_is_empty() -> None:
    """The merge carry seeds empty so the first run rebinds with its identity."""
    outputs, node = run_get_node(graphs_base.seed_ph_merged)
    assert node.is_finished_ok
    assert dict(outputs) == {}


def test_base_ph_graph_builds_with_merge() -> None:
    """The production ``ph`` graph still validates with the merge wiring."""
    assert graphs_base.ph_base_graph.build() is None


def _ph_output_dict(irreps: tuple, total: int) -> Dict:
    """Return an unstored ``output_parameters``-shaped ``Dict``."""
    return Dict(dict(_OUTPUT_A, number_of_irr_representations_for_each_q=list(irreps), number_of_qpoints=total))


def test_base_seed_ph_merge_check_is_empty() -> None:
    """The merge validation seeds with no runs seen."""
    outputs, node = run_get_node(graphs_base.seed_ph_merge_check)
    assert node.is_finished_ok
    assert outputs == {'expected': 0, 'found': 0, 'runs': 0, 'missing': False, 'has_expected': False}


def _check_kwargs(check: dict) -> dict:
    """Return the scalar validation inputs for the merge-check tasks."""
    return {
        'expected': check['expected'],
        'found': check['found'],
        'runs': check['runs'],
        'missing': check['missing'],
        'has_expected': check['has_expected'],
    }


def test_base_ph_merge_check_success() -> None:
    """Two recover runs covering the requested range validate cleanly."""
    seed, _ = run_get_node(graphs_base.seed_ph_merge_check)
    first, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(seed)),
        output=_ph_output_dict((3, 3), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    assert (first['expected'], first['found'], first['runs'], first['missing']) == (4, 2, 1, False)
    second, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(first)),
        output=_ph_output_dict((3, 3), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    assert (second['expected'], second['found'], second['runs']) == (4, 4, 2)
    _, node = run_get_node(graphs_base.validate_ph_merge, **_check_kwargs(dict(second)), only_initialization=False)
    assert node.is_finished_ok


def test_base_ph_merge_check_range_incomplete() -> None:
    """A short parsed count fails the final validation instead of merging partially."""
    seed, _ = run_get_node(graphs_base.seed_ph_merge_check)
    first, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(seed)),
        output=_ph_output_dict((3, 3), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    short, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(first)),
        output=_ph_output_dict((3,), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    assert (short['expected'], short['found'], short['runs']) == (4, 3, 2)
    with pytest.raises(ValueError, match=r'only 3 of 4 q-points were parsed'):
        run_get_node(graphs_base.validate_ph_merge, **_check_kwargs(dict(short)), only_initialization=False)


def test_base_ph_merge_check_missing_output() -> None:
    """A missing original in a multi-run chain fails instead of merging partially."""
    seed, _ = run_get_node(graphs_base.seed_ph_merge_check)
    first, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(seed)),
        output=_ph_output_dict((3, 3), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    missing, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(first)),
        output=None,
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    assert missing['missing'].value is True and missing['runs'].value == 2
    with pytest.raises(ValueError, match='exposes no `output_parameters`'):
        run_get_node(graphs_base.validate_ph_merge, **_check_kwargs(dict(missing)), only_initialization=False)


def test_base_ph_merge_check_reset_and_skips() -> None:
    """A scratch restart resets the expectation; single runs and init runs skip the check."""
    seed, _ = run_get_node(graphs_base.seed_ph_merge_check)
    first, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(seed)),
        output=_ph_output_dict((3, 3), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    reset, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(first)),
        output=_ph_output_dict((3, 3), 2),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 2}},
        recovered=False,
        only_initialization=False,
    )
    assert (reset['expected'], reset['found'], reset['runs'], reset['missing']) == (2, 2, 1, False)
    _, node = run_get_node(graphs_base.validate_ph_merge, **_check_kwargs(dict(reset)), only_initialization=False)
    assert node.is_finished_ok
    _, single = run_get_node(graphs_base.validate_ph_merge, **_check_kwargs(dict(first)), only_initialization=False)
    assert single.is_finished_ok
    _, skipped = run_get_node(graphs_base.validate_ph_merge, **_check_kwargs(dict(seed)), only_initialization=True)
    assert skipped.is_finished_ok


def test_base_ph_merge_check_scratch_missing_resets() -> None:
    """A scratch run without outputs drops prior runs instead of flagging them partial."""
    seed, _ = run_get_node(graphs_base.seed_ph_merge_check)
    first, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(seed)),
        output=_ph_output_dict((3, 3), 4),
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=True,
        only_initialization=False,
    )
    reset, _ = run_get_node(
        graphs_base.update_ph_merge_check,
        **_check_kwargs(dict(first)),
        output=None,
        parameters={'INPUTPH': {'start_q': 1, 'last_q': 4}},
        recovered=False,
        only_initialization=False,
    )
    assert (reset['expected'].value, reset['found'].value) == (0, 0)
    assert (reset['runs'].value, reset['missing'].value, reset['has_expected'].value) == (1, True, False)


def test_base_finalize_ph_carry_sets_recover() -> None:
    """The latched carry launches the next run with ``INPUTPH.recover=True``."""
    params = {'INPUTPH': {'recover': False, 'max_seconds': 10.0}}
    latched, _ = run_get_node(
        graphs_base.finalize_ph_carry,
        parameters=params,
        restart=True,
        recover=True,
        parent_mode='remote',
        recover_before=False,
    )
    assert latched['recover'].value is True
    assert latched['parameters']['INPUTPH']['recover'] is True
    assert params['INPUTPH']['recover'] is False
    before, _ = run_get_node(
        graphs_base.finalize_ph_carry,
        parameters={'INPUTPH': {'recover': False}},
        restart=True,
        recover=False,
        parent_mode='remote',
        recover_before=True,
    )
    assert before['recover'].value is True
    assert before['parameters']['INPUTPH']['recover'] is True
    unlatched, _ = run_get_node(
        graphs_base.finalize_ph_carry,
        parameters={'INPUTPH': {'diagonalization': 'cg'}},
        restart=True,
        recover=False,
        parent_mode='keep',
        recover_before=False,
    )
    assert unlatched['recover'].value is False
    assert 'recover' not in unlatched['parameters']['INPUTPH']
