############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Tests for the plain protocol helpers of the native graphs.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_protocols.py -q

The helpers return plain protocol settings and never import a ``WorkChain``.
"""

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida_quantumespresso.workflows.graphs import protocols

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


def test_protocol_filepaths_exist() -> None:
    """Every graph family resolves to an existing YAML file."""
    for kind in protocols.PROTOCOL_KINDS:
        assert protocols.get_protocol_filepath(kind).is_file()


def test_protocol_filepath_unknown_kind_fails() -> None:
    """An unknown kind fails naming the supported families."""
    with pytest.raises(ValueError, match='Unknown protocol kind'):
        protocols.get_protocol_filepath('phonons')


def test_available_protocols_lists_families() -> None:
    """Each family lists the three standard protocols with descriptions."""
    for kind in protocols.PROTOCOL_KINDS:
        available = protocols.available_protocols(kind)
        assert {'balanced', 'fast', 'stringent'} <= set(available)
        assert available['balanced']['description']


def test_get_protocol_inputs_defaults_and_overrides() -> None:
    """Protocol inputs merge defaults, the named protocol, and caller overrides as plain data."""
    balanced = protocols.get_protocol_inputs('pdos', protocol='balanced')
    assert balanced['nscf']['kpoints_distance'] == 0.10
    fast = protocols.get_protocol_inputs('pdos', protocol='fast')
    assert fast['nscf']['kpoints_distance'] == 0.5
    assert fast['dos']['parameters']['DOS']['deltae'] == 0.1
    assert isinstance(fast['dos']['parameters'], dict)

    overridden = protocols.get_protocol_inputs(
        'pdos', protocol='fast', overrides={'dos': {'parameters': {'DOS': {'deltae': 0.02}}}}
    )
    assert overridden['dos']['parameters']['DOS']['deltae'] == 0.02
    # Overrides merge; unrelated defaults survive.
    assert overridden['nscf']['kpoints_distance'] == 0.5


def test_get_protocol_inputs_alias_and_unknown() -> None:
    """Protocol aliases resolve while unknown names fail."""
    assert protocols.get_protocol_inputs('pdos', protocol='moderate')['nscf']['kpoints_distance'] == 0.10
    with pytest.raises(ValueError, match='not a valid protocol'):
        protocols.get_protocol_inputs('pdos', protocol='ultrafast')


def test_pw_protocol_inputs_expose_relax_bands_keys() -> None:
    """The PW families expose the keys the graphs consume."""
    relax = protocols.get_protocol_inputs('pw_relax', protocol='balanced')
    assert 'base_relax' in relax
    bands = protocols.get_protocol_inputs('pw_bands', protocol='fast')
    assert bands['bands_kpoints_distance'] == 0.1


def test_task_metadata_for_options_skips_empty() -> None:
    """Empty placements are left out of the launch mapping."""
    assert protocols.task_metadata_for_options() == {}
    assert protocols.task_metadata_for_options(pw_calc=None, dos_calc={}) == {}
    mapping = protocols.task_metadata_for_options(
        pw_calc={'options': {'max_wallclock_seconds': 3600}},
        dos_calc={'options': {'withmpi': True}},
    )
    assert mapping == {
        'pw_calc': {'options': {'max_wallclock_seconds': 3600}},
        'dos_calc': {'options': {'withmpi': True}},
    }


def test_pdos_protocol_task_metadata_lifts_options() -> None:
    """The PDOS helper lifts ``dos``/``projwfc`` scheduler options under placement names."""
    inputs = protocols.get_protocol_inputs('pdos', protocol='balanced')
    task_metadata = protocols.pdos_protocol_task_metadata(inputs)
    assert task_metadata['dos_calc']['options']['max_wallclock_seconds'] == 43200
    assert task_metadata['projwfc_calc']['options']['max_wallclock_seconds'] == 43200
    # The PW options stay out: the base graph takes them under its own key.
    assert 'pw_calc' not in task_metadata


def test_no_workchain_import() -> None:
    """The protocols module never imports a work chain subclass."""
    source = protocols.__file__
    assert source is not None
    with open(source, encoding='utf-8') as handle:
        text = handle.read()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            assert 'WorkChain' not in stripped
            assert 'workflows.pdos import' not in stripped
            assert 'workflows.pw' not in stripped and 'workflows.graphs' not in stripped
