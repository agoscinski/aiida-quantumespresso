############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Independent integration for static ``CalcJob`` ``DataDict`` inputs.

Run from the parent core checkout with the QE dependencies installed::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \\
        pytest aiida-quantumespresso/tests/graphs/test_static_calcjob_inputs.py -q

What this proves: the production ``pw_base_graph``/``neb_base_graph`` handles wire
static pseudopotential ``DataDict`` containers directly, with no required dynamic
namespaces. Monitor forwarding remains unchanged. Provenance links the single ``pseudos`` container;
owned members are read through the stored container without detaching or
re-expanding, keeping their identity and repository content.
"""

import io

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida import orm
from aiida.engine import run_get_node, task

from aiida_quantumespresso.calculations.neb import NebCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.graphs import base
from aiida_quantumespresso.workflows.graphs.base import PwErrors

pytestmark = pytest.mark.presto


def _upf(element='Si'):
    """Return a minimal parsable pseudo for ``element``."""
    from aiida.plugins import DataFactory

    UpfData = DataFactory('pseudo.upf')
    content = f'<UPF version="2.0.1"><PP_HEADER\\nelement="{element}"\\nz_valence="4.0"\\n/></UPF>\\n'
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


def _incoming(node) -> dict:
    """Return the incoming link label to node mapping."""
    return {link.link_label: link.node for link in node.base.links.get_incoming().all()}


def _assert_no_required_dynamic(spec) -> None:
    """Assert the spec exposes no required dynamic input namespaces."""

    def visit(namespace) -> None:
        from aiida.engine.processes.ports import PortNamespace

        for child in namespace.values():
            if isinstance(child, PortNamespace):
                assert not (child.dynamic and child.required), (
                    f'static CalcJob inputs must not require a dynamic namespace `{child.name}`'
                )
                visit(child)

    visit(spec.inputs)


def test_pseudos_need_no_namespace_forwarding() -> None:
    """Pseudos need no forwarding, while the existing monitor namespace still does."""
    assert task(PwCalculation)._task_spec.forwarded == ()
    assert base.pw_calc._task_spec.forwarded == ('monitors',)
    assert base.neb_calc._task_spec.forwarded == ()
    assert base.ph_calc._task_spec.forwarded == ()


def test_actual_specs_have_no_required_dynamic_namespaces() -> None:
    """Actual ``PwCalculation``/``NebCalculation`` specs expose static containers."""
    from aiida.engine.processes.ports import InputPort

    pw_spec = PwCalculation.spec().inputs
    assert isinstance(pw_spec['pseudos'], InputPort)
    assert pw_spec['pseudos'].valid_type in (orm.DataDict, (orm.DataDict,))
    _assert_no_required_dynamic(PwCalculation.spec())

    neb_pw = NebCalculation.spec().inputs['pw']
    assert isinstance(neb_pw['pseudos'], InputPort)
    _assert_no_required_dynamic(NebCalculation.spec())


def test_build_validates_static_wiring() -> None:
    """``build()`` validates the static container wiring without launching anything."""
    assert base.pw_base_graph.build() is None
    assert base.neb_base_graph.build() is None


class StaticSuccessPwCalc(PwCalculation):
    """Trivial-execution stub succeeding at once with the success outputs."""

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
        """Attach the success outputs a real run would store."""
        self.out('output_parameters', orm.Dict({'number_of_bands': 8, 'number_of_symmetries': 10}))
        self.out('output_structure', self.inputs.structure.clone())
        return None


def _substitute(monkeypatch, stub, name, errors):
    """Replace a ``base`` placement with a plain static stub task."""
    shaped = task(stub, inputs={'monitors': orm.DataDict}, errors=errors)
    monkeypatch.setattr(base, name, shaped)
    for handle in (base.pw_base_graph, base.neb_base_graph):
        monkeypatch.setattr(handle, '_template', None)
    return shaped


@pytest.mark.usefixtures('aiida_profile_clean')
def test_static_pseudos_container_provenance(monkeypatch, aiida_code_installed) -> None:
    """The single ``pseudos`` container reaches the CalcJob with owned members intact."""
    _substitute(monkeypatch, StaticSuccessPwCalc, 'pw_calc', PwErrors)
    code = aiida_code_installed(label='test.static-pseudos', default_calc_job_plugin='quantumespresso.pw')
    pseudos = _pseudos('Si')
    outputs, node = run_get_node(
        base.pw_base_graph,
        code=code,
        structure=_silicon(),
        pseudos=pseudos,
        parameters={'CONTROL': {}, 'SYSTEM': {}, 'ELECTRONS': {'diagonalization': 'cg'}},
        kpoints=_kpoints(),
        task_metadata={'pw_calc': {'options': {}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict() == {'attempts': 1, 'status': 'finished'}
    (child,) = [entry for entry in node.called if entry.process_label == 'StaticSuccessPwCalc']
    incoming = _incoming(child)
    assert 'pseudos' in incoming
    assert not any(label.startswith('pseudos__') for label in incoming)
    container = incoming['pseudos']
    assert container.is_stored
    assert container.uuid == node.inputs.pseudos.uuid
    member = container['Si']
    assert member.is_stored
    assert member.base.repository.list_object_names()
    # Identity without detaching: the owned member is the staged pseudo.
    assert member.uuid == child.inputs.pseudos['Si'].uuid
