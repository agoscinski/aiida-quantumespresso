############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Runtime regression for the production ``ph_base_graph`` without real executables.

Run from the parent core checkout with the QE dependencies installed::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \\
        pytest aiida-quantumespresso/tests/graphs/test_ph_native.py -q

The production ``ph_base_graph`` retry loop, rebinding, guards, incremental
merge fold, and merge validation all execute for real. Only the external
``ph.x`` placement is substituted by a deterministic trivial-execution stub
that subclasses the real ``PhCalculation`` (no ``WorkChain`` anywhere): the
first run stores a partial two-q-point output and fails with walltime 400, the
recover run (its parent folder created by a ``ph`` calculation) stores the
remaining two q-points and succeeds. No production graph body is copied here;
the substitution patches only ``base.ph_calc`` and resets the compiled
template so the production body recompiles against the substitute.
"""

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida import orm
from aiida.engine import run_get_node, task

from aiida_quantumespresso.calculations.ph import PhCalculation
from aiida_quantumespresso.workflows.graphs import base
from aiida_quantumespresso.workflows.graphs.base import PhErrors

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]

PH_PARAMETERS = {'INPUTPH': {'tr2_ph': 1.0e-14, 'start_q': 1, 'last_q': 4}}

_PARTIAL = {
    'number_of_irr_representations_for_each_q': [3, 3],
    'wall_time_seconds': 10.0,
    'dynamical_matrix_1': {'mode_symmetry': [1], 'frequencies': [100.0]},
}

_REMAINING = {
    'number_of_irr_representations_for_each_q': [3, 3],
    'wall_time_seconds': 20.0,
    'dynamical_matrix_3': {'mode_symmetry': [1], 'frequencies': [300.0]},
}

_FULL = {
    'number_of_irr_representations_for_each_q': [3, 3, 3, 3],
    'wall_time_seconds': 30.0,
    'dynamical_matrix_1': {'mode_symmetry': [1], 'frequencies': [100.0]},
}


class RecoverPhCalc(PhCalculation):
    """Stub failing once on walltime, then recovering with the remaining q-points.

    The first run recovers from the seeded parent (a folder with no ``ph``
    creator) and stores a partial two-q-point output with exit 400; the recover
    run (its parent folder created by a ``ph`` calculation) stores the remaining
    two q-points and succeeds. Both runs store the folder a real run would leave
    via the automatic ``remote_folder`` output.
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
        """Store partial outputs once, then the remainder on the recover run."""
        from aiida.engine.processes.exit_code import ExitCode

        if self.inputs.parent_folder.creator is None:
            self.out('output_parameters', orm.Dict(dict(_PARTIAL)))
            return ExitCode(400, 'stub ph walltime failure')
        self.out('output_parameters', orm.Dict(dict(_REMAINING)))
        return None


class SuccessPhCalc(PhCalculation):
    """Stub succeeding at once with the full q-point range."""

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
        """Attach the full outputs."""
        self.out('output_parameters', orm.Dict(dict(_FULL)))
        return None


def _substitute(monkeypatch, stub):
    """Patch the production ``ph`` placement with a stub and force recompilation."""
    shaped = task(stub, errors=PhErrors)
    monkeypatch.setattr(base, 'ph_calc', shaped)
    monkeypatch.setattr(base.ph_base_graph, '_template', None)
    return shaped


def _qpoints() -> orm.KpointsData:
    """Return an explicit q-point mesh."""
    qpoints = orm.KpointsData()
    qpoints.set_kpoints_mesh([2, 2, 2])
    return qpoints


def test_ph_base_graph_recover_merges_real_wiring(monkeypatch, aiida_code_installed) -> None:
    """A walltime recover run folds both originals into one merged node for real."""
    _substitute(monkeypatch, RecoverPhCalc)
    code = aiida_code_installed(label='test.ph-recover', default_calc_job_plugin='quantumespresso.ph')
    parent = orm.RemoteData(computer=code.computer, remote_path='/tmp/ph-parent').store()
    outputs, node = run_get_node(
        base.ph_base_graph,
        code=code,
        parent_folder=parent,
        parameters=dict(PH_PARAMETERS),
        qpoints=_qpoints(),
        runtime_config={'ph_calc': {'metadata': {'options': {}}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 2, 'status': 'finished'}
    merged = outputs['output_parameters'].get_dict()
    assert merged['wall_time_seconds'] == 30.0
    assert merged['number_of_qpoints'] == 4
    assert merged['number_of_irr_representations_for_each_q'] == [3, 3, 3, 3]
    assert outputs['remote_folder'] is not None

    runs = sorted(
        [child for child in node.called if child.process_label == 'RecoverPhCalc'],
        key=lambda child: child.ctime,
    )
    assert len(runs) == 2
    assert runs[0].exit_status == 400
    assert runs[1].exit_status == 0
    assert runs[1].inputs.parent_folder.uuid == runs[0].outputs.remote_folder.uuid

    merges = [child for child in node.called_descendants if child.process_label == 'merge_ph_pair']
    assert len(merges) == 1
    incoming = {link.link_label: link.node.uuid for link in merges[0].base.links.get_incoming().all()}
    assert incoming['first'] == runs[0].outputs.output_parameters.uuid
    assert incoming['second'] == runs[1].outputs.output_parameters.uuid
    assert outputs['output_parameters'].uuid == merges[0].outputs.result.uuid
    assert outputs['output_parameters'].uuid not in {
        runs[0].outputs.output_parameters.uuid,
        runs[1].outputs.output_parameters.uuid,
    }
    guards = [child for child in node.called_descendants if child.process_label == 'require_ph_restart_nodes']
    assert guards, 'the production retry loop must pass through the restart-node guard'
    assert all(guard.is_finished_ok for guard in guards)


def test_ph_base_graph_only_initialization_returns_existing(monkeypatch, aiida_code_installed) -> None:
    """An ``ONLY_INITIALIZATION`` run returns its original node with no merge node."""
    _substitute(monkeypatch, SuccessPhCalc)
    code = aiida_code_installed(label='test.ph-init', default_calc_job_plugin='quantumespresso.ph')
    parent = orm.RemoteData(computer=code.computer, remote_path='/tmp/ph-parent-init').store()
    outputs, node = run_get_node(
        base.ph_base_graph,
        code=code,
        parent_folder=parent,
        parameters=dict(PH_PARAMETERS),
        qpoints=_qpoints(),
        only_initialization=True,
        runtime_config={'ph_calc': {'metadata': {'options': {}}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 1, 'status': 'finished'}
    (run,) = [child for child in node.called if child.process_label == 'SuccessPhCalc']
    assert outputs['output_parameters'].uuid == run.outputs.output_parameters.uuid
    assert not [child for child in node.called_descendants if child.process_label == 'merge_ph_pair'], (
        'ONLY_INITIALIZATION must skip the incremental merge entirely'
    )
