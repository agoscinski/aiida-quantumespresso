###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Tests for the static ``pseudos`` `DataDict` input contract on ``aiida-core`` 3.

On ``aiida-core`` 2.x (still pinned by ``pyproject.toml``) `BasePwCpInputGenerator` keeps the legacy dynamic
``pseudos`` namespace, so this module only covers the static contract and skips there.
"""

import pytest
from aiida import orm
from aiida.common import LinkType
from aiida.engine.processes.ports import InputPort, PortNamespace
from aiida.plugins import DataFactory

from aiida_quantumespresso.calculations.cp import CpCalculation
from aiida_quantumespresso.calculations.neb import NebCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation

pytestmark = pytest.mark.skipif(not hasattr(orm, 'DataDict'), reason='Static `pseudos` contract requires aiida-core 3.')


@pytest.mark.parametrize('calc_class', [PwCalculation, CpCalculation])
def test_pseudos_port_is_static(calc_class):
    """The ``pseudos`` port is a static `DataDict` input, not a dynamic namespace."""
    port = calc_class.spec().inputs['pseudos']
    assert isinstance(port, InputPort)
    assert port.valid_type is orm.DataDict
    assert port.serializer is orm.DataDict


def test_task_pw_plain_no_inputs():
    """A plain ``task(PwCalculation)`` builds from the native spec with no dynamic closure."""
    from aiida.engine.processes.graphs import task

    handle = task(PwCalculation)
    port = handle.spec().inputs['pseudos']
    assert isinstance(port, InputPort)
    assert port.valid_type is orm.DataDict
    # The lowered GraphIR declares no dynamic-namespace inputs: nothing is forwarded.
    assert handle._task_spec.forwarded == ()  # noqa: SLF001


def test_task_neb_pw_namespace_static():
    """A plain ``task(NebCalculation)`` inherits the static ``pw.pseudos`` input under a static ``pw``."""
    from aiida.engine.processes.graphs import task

    handle = task(NebCalculation)
    pw = handle.spec().inputs['pw']
    assert isinstance(pw, PortNamespace)
    assert pw.dynamic is False
    port = pw['pseudos']
    assert isinstance(port, InputPort)
    assert port.valid_type is orm.DataDict
    assert port.serializer is orm.DataDict
    # The lowered GraphIR declares no dynamic-namespace inputs: nothing is forwarded.
    assert handle._task_spec.forwarded == ()  # noqa: SLF001


def test_dict_launch_matches_datadict_launch(fixture_sandbox, generate_calc_job, generate_inputs_pw):
    """A plain `dict` launch prepares the same files as an explicit `DataDict` launch."""
    entry_point_name = 'quantumespresso.pw'

    inputs = generate_inputs_pw()
    source_pseudos = dict(inputs['pseudos'])
    calc_info_plain = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    with fixture_sandbox.open('aiida.in') as handle:
        content_plain = handle.read()

    # The serializer wraps the mapping without mutating the caller's dict or storing the sources.
    assert isinstance(inputs['pseudos'], dict)
    assert set(inputs['pseudos']) == set(source_pseudos)
    for upf in inputs['pseudos'].values():
        assert not upf.is_stored

    inputs_wrapped = generate_inputs_pw()
    container = orm.DataDict(inputs_wrapped['pseudos'])
    inputs_wrapped['pseudos'] = container
    calc_info_wrapped = generate_calc_job(fixture_sandbox, entry_point_name, inputs_wrapped)
    with fixture_sandbox.open('aiida.in') as handle:
        content_wrapped = handle.read()

    assert content_wrapped == content_plain

    def staged(calc_info):
        # Owned container copies get fresh UUIDs on insertion, so only staged paths are stable.
        return sorted((filename, dest) for _, filename, dest in calc_info.local_copy_list)

    assert staged(calc_info_wrapped) == staged(calc_info_plain) != []


def test_builder_serializes_static_pseudo_members(serialize_builder, generate_inputs_pw):
    """Builder regression serialization keeps the member values instead of an opaque container hash."""
    builder = PwCalculation.get_builder()
    builder._update(generate_inputs_pw())
    assert isinstance(builder.pseudos, orm.DataDict)
    result = serialize_builder(builder)
    assert result['pseudos'] == {key: f'{value.element}<md5={value.md5}>' for key, value in builder.pseudos.items()}


def test_process_inputs_hold_container(generate_inputs_pw):
    """The instantiated calculation holds the `DataDict` container at the process boundary."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager

    inputs = generate_inputs_pw()
    inputs['pseudos'] = orm.DataDict(inputs['pseudos'])
    runner = get_manager().get_runner()
    process = instantiate_process(runner, PwCalculation, **inputs)
    assert isinstance(process.inputs.pseudos, orm.DataDict)
    assert set(process.inputs.pseudos.keys()) == {'Si'}


def test_neb_datadict_launch(fixture_sandbox, generate_calc_job, generate_inputs_neb):
    """A `NebCalculation` launches with an explicit `DataDict` under the static ``pw`` namespace."""
    entry_point_name = 'quantumespresso.neb'

    inputs = generate_inputs_neb()
    inputs['pw']['pseudos'] = orm.DataDict(inputs['pw']['pseudos'])
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    content_list = fixture_sandbox.get_content_list()
    assert 'neb.dat' in content_list
    assert 'pw_1.in' in content_list
    kinds = set(inputs['pw']['pseudos'].keys())
    staged = sorted(dest for _, _, dest in calc_info.local_copy_list)
    assert staged == sorted(f'./pseudo/{kind}.upf' for kind in kinds)


def test_cp_datadict_launch(fixture_sandbox, generate_calc_job, generate_inputs_cp):
    """A `CpCalculation` inherits the static contract and prepares the same files."""
    entry_point_name = 'quantumespresso.cp'

    inputs = generate_inputs_cp()
    inputs['pseudos'] = orm.DataDict(inputs['pseudos'])
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    with fixture_sandbox.open('aiida.in') as handle:
        assert handle.read()
    assert sorted(dest for _, _, dest in calc_info.local_copy_list) == ['./pseudo/Si.upf']


def test_pseudos_missing(generate_calc_job, generate_inputs_pw, fixture_sandbox):
    """Omitting ``pseudos`` fails on the required static port."""
    inputs = generate_inputs_pw()
    inputs.pop('pseudos', None)
    with pytest.raises(ValueError, match=r"required value was not provided for 'pseudos'"):
        generate_calc_job(fixture_sandbox, 'quantumespresso.pw', inputs)


def test_pseudos_wrong_type(generate_calc_job, generate_inputs_pw, fixture_sandbox):
    """A non-`DataDict` node for ``pseudos`` fails port validation."""
    inputs = generate_inputs_pw()
    inputs['pseudos'] = orm.Dict({'Si': 'not-a-pseudo'})
    with pytest.raises(ValueError, match=r'pseudos'):
        generate_calc_job(fixture_sandbox, 'quantumespresso.pw', inputs)


def test_pseudos_empty(generate_calc_job, generate_inputs_pw, generate_inputs_cp, fixture_sandbox):
    """An empty `DataDict` is rejected for both ``pw.x`` and ``cp.x``."""
    for entry_point_name, inputs in (
        ('quantumespresso.pw', generate_inputs_pw()),
        ('quantumespresso.cp', generate_inputs_cp()),
    ):
        inputs['pseudos'] = orm.DataDict()
        with pytest.raises(ValueError, match=r'The `pseudos` input is empty'):
            generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_pseudos_wrong_member(generate_calc_job, generate_inputs_pw, fixture_sandbox):
    """A `DataDict` member that is not a pseudo is rejected."""
    inputs = generate_inputs_pw()
    inputs['pseudos'] = orm.DataDict({'Si': orm.Int(1)})
    with pytest.raises(ValueError, match=r'Invalid pseudopotential for kind `Si`'):
        generate_calc_job(fixture_sandbox, 'quantumespresso.pw', inputs)


def test_pseudos_kind_mismatch(generate_calc_job, generate_inputs_pw, generate_upf_data, fixture_sandbox):
    """A `DataDict` whose keys do not match the structure kinds is rejected."""
    inputs = generate_inputs_pw()
    inputs['pseudos'] = orm.DataDict({'X': generate_upf_data('X')})
    with pytest.raises(ValueError, match=r'The `pseudos` specified and structure kinds do not match'):
        generate_calc_job(fixture_sandbox, 'quantumespresso.pw', inputs)


def test_pseudos_accepts_legacy_upf(generate_calc_job, generate_inputs_pw, fixture_sandbox, filepath_fixtures):
    """A `DataDict` of legacy ``core.upf`` nodes validates and prepares files."""
    import pathlib

    legacy_upf = DataFactory('core.upf')
    filepath = pathlib.Path(filepath_fixtures) / 'pseudos' / 'Si.upf'
    inputs = generate_inputs_pw()
    inputs['pseudos'] = orm.DataDict({'Si': legacy_upf(file=str(filepath))})
    calc_info = generate_calc_job(fixture_sandbox, 'quantumespresso.pw', inputs)
    assert sorted(dest for _, _, dest in calc_info.local_copy_list) == ['./pseudo/Si.upf']


def test_pseudos_single_container_link(generate_upf_data):
    """Provenance records a single ``pseudos`` link to the owning `DataDict`, not per-kind links."""
    upf = generate_upf_data('Si')
    pseudos = orm.DataDict({'Si': upf})
    # Insertion owns a copy: the source keeps its identity and stays unstored.
    assert pseudos['Si'].uuid != upf.uuid
    assert not upf.is_stored

    node = orm.CalcJobNode()
    node.base.links.add_incoming(pseudos, LinkType.INPUT_CALC, 'pseudos')
    incoming = node.base.links.get_incoming(link_type=LinkType.INPUT_CALC).all()
    assert [(link.link_label, type(link.node).__name__) for link in incoming] == [('pseudos', 'DataDict')]
    assert set(incoming[0].node.keys()) == {'Si'}

    # Stored container children stay addressable by UUID with their repo files, which is the path the
    # execmanager uses to stage `local_copy_list` entries (`load_node` first, `_find_data_node` fallback).
    pseudos.store()
    child = pseudos['Si']
    assert orm.load_node(uuid=child.uuid).uuid == child.uuid
    assert child.base.repository.list_object_names() == ['Si.upf']
