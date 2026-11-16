############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Tests for the plain protocol helpers and the launch-input resolver of the native graphs.

Run from the parent core checkout with the QE dependencies installed::

    uv run pytest aiida-quantumespresso/tests/graphs/test_protocols.py -q

The raw-YAML helpers return plain protocol settings and never import a
``WorkChain``. ``get_inputs_from_protocol`` translates that nesting into
launch-ready native graph inputs plus ``task_metadata``; the launch tests run
the production ``pw_relax_graph`` / ``pw_bands_graph`` /
``pdos_from_scratch_serial`` with only the external ``CalcJob`` placements
substituted by the deterministic stubs reused from ``test_native_workflows``.
Explicit ``UpfData`` nodes and fake silicon-only families keep every test
offline: no pseudo is ever downloaded.
"""

import copy
import io

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida import orm
from aiida.engine import run_get_node

from aiida_quantumespresso.common.types import ElectronicType, RelaxType, SpinType
from aiida_quantumespresso.workflows.graphs import protocols
from aiida_quantumespresso.workflows.graphs.pdos import pdos_from_scratch_serial
from aiida_quantumespresso.workflows.graphs.pw import pw_bands_graph, pw_relax_graph

from .test_native_workflows import (
    FullPwCalc,
    StubDosCalc,
    StubProjwfcCalc,
    SuccessPwCalc,
    _patch_nested,
)

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


def _silicon() -> orm.StructureData:
    """Return an unstored two-site silicon structure."""
    structure = orm.StructureData(cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 2.715]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols='Si')
    return structure


def _upf(element: str = 'Si'):
    """Return an unstored parsable pseudo with a valence charge for spin handling."""
    from aiida.plugins import DataFactory

    UpfData = DataFactory('pseudo.upf')
    content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
    return UpfData(io.BytesIO(content.encode('utf-8')), filename=f'{element}.upf')


def _cutoff_overrides(ecutwfc: float = 30.0, ecutrho: float = 240.0, *sections: str) -> dict:
    """Return per-section cutoff overrides for explicit pseudos."""
    return {
        section: {'pw': {'parameters': {'SYSTEM': {'ecutwfc': ecutwfc, 'ecutrho': ecutrho}}}} for section in sections
    }


def _silicon_family(generate_silicon_pseudo_family, label: str = 'test/si-protocols'):
    """Create a silicon-only SSSP family with recommended cutoffs, offline."""
    from aiida_pseudo.groups.family import CutoffsPseudoPotentialFamily

    family = generate_silicon_pseudo_family(CutoffsPseudoPotentialFamily, label)
    family.set_cutoffs({'Si': {'cutoff_wfc': 30.0, 'cutoff_rho': 240.0}}, 'standard', unit='Ry')
    return family


def _pw_calcs(node) -> list:
    """Return the dispatched ``pw.x`` CalcJobs below a finished graph."""
    from aiida.orm import CalcJobNode

    return [
        child
        for child in node.called_descendants
        if isinstance(child, CalcJobNode) and child.process_label.endswith('PwCalc')
    ]


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


def test_launch_relax_explicit_pseudos(aiida_code_installed, generate_upf_data) -> None:
    """The relax resolver applies per-atom thresholds, cutoffs, and the relax type."""
    structure = _silicon()
    code = aiida_code_installed(label='test.proto-relax-pw', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=structure,
        code=code,
        pseudos={'Si': generate_upf_data('Si')},
        overrides=_cutoff_overrides(30.0, 240.0, 'base_relax', 'base_init_relax'),
    )
    parameters = launch['parameters'].get_dict()
    assert parameters['CONTROL']['calculation'] == 'vc-relax'
    assert parameters['CELL']['cell_dofree'] == 'all'
    assert parameters['SYSTEM']['ecutwfc'] == 30.0
    assert parameters['SYSTEM']['ecutrho'] == 240.0
    # Two sites scale the base balanced per-atom thresholds.
    assert parameters['ELECTRONS']['conv_thr'] == pytest.approx(2 * 0.2e-9)
    assert parameters['CONTROL']['etot_conv_thr'] == pytest.approx(2 * 1.0e-5)
    assert launch['init_parameters'].get_dict()['CONTROL']['etot_conv_thr'] == pytest.approx(2 * 1.0e-3)
    assert launch['kpoints_distance'].value == pytest.approx(0.15)
    assert launch['run_init_relax'] is True
    assert launch['pseudos']['Si'].z_valence == 4.0
    assert launch['task_metadata']['pw_calc']['options']['max_wallclock_seconds'] == 43200


def test_launch_relax_family_cutoffs(aiida_code_installed, generate_silicon_pseudo_family) -> None:
    """A fake installed family recommends cutoffs with no downloads."""
    family = _silicon_family(generate_silicon_pseudo_family)
    code = aiida_code_installed(label='test.proto-relax-fam', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol('pw_relax', structure=_silicon(), code=code, pseudo_family=family.label)
    parameters = launch['parameters'].get_dict()
    assert parameters['SYSTEM']['ecutwfc'] == 30.0
    assert parameters['SYSTEM']['ecutrho'] == 240.0
    # `DataDict` takes owned copies, so identity is by content, not UUID.
    assert (
        launch['pseudos']['Si'].base.caching.get_hash()
        == family.get_pseudos(structure=_silicon())['Si'].base.caching.get_hash()
    )


def test_launch_family_without_cutoffs_fails(aiida_code_installed, generate_silicon_pseudo_family) -> None:
    """A family that recommends no cutoffs fails fast unless overrides carry them."""
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    family = generate_silicon_pseudo_family(PseudoPotentialFamily, 'test/si-nocutoffs')
    code = aiida_code_installed(label='test.proto-relax-nocut', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(ValueError, match='cannot recommend cutoffs'):
        protocols.get_inputs_from_protocol('pw_relax', structure=_silicon(), code=code, pseudo_family=family.label)


def test_launch_explicit_pseudos_missing_cutoffs_fails(aiida_code_installed) -> None:
    """Explicit pseudos without cutoff overrides fail fast."""
    code = aiida_code_installed(label='test.proto-relax-ecut', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(ValueError, match='both `ecutwfc` and `ecutrho`'):
        protocols.get_inputs_from_protocol('pw_relax', structure=_silicon(), code=code, pseudos={'Si': _upf()})


def test_launch_rejects_unsupported_options(aiida_code_installed) -> None:
    """Unknown keys, dropped ports, and legacy flags fail instead of vanishing."""
    code = aiida_code_installed(label='test.proto-reject-pw', default_calc_job_plugin='quantumespresso.pw')
    base = {
        'structure': _silicon(),
        'code': code,
        'pseudos': {'Si': _upf()},
        'overrides': _cutoff_overrides(30.0, 240.0, 'base_relax', 'base_init_relax'),
    }
    with pytest.raises(ValueError, match='not a supported protocol option'):
        protocols.get_inputs_from_protocol('pw_relax', **{**base, 'overrides': {**base['overrides'], 'bogus_knob': 1}})
    with pytest.raises(ValueError, match='no `dry_run` mode'):
        protocols.get_inputs_from_protocol(
            'pdos',
            structure=_silicon(),
            pw_code=code,
            dos_code=code,
            projwfc_code=code,
            pseudos={'Si': _upf()},
            overrides={'dry_run': True, **_cutoff_overrides(30.0, 240.0, 'scf', 'nscf')},
        )
    with pytest.raises(ValueError, match='no port for it'):
        protocols.get_inputs_from_protocol(
            'pw_bands',
            structure=_silicon(),
            scf_code=code,
            bands_code=code,
            pseudos={'Si': _upf()},
            overrides={
                'bands': {'pw': {'parallelization': {'npool': 2}}},
                **_cutoff_overrides(30.0, 240.0, 'scf'),
            },
        )


def test_launch_rejects_nodes_in_overrides(aiida_code_installed) -> None:
    """ORM nodes inside overrides fail fast towards the keyword arguments."""
    from aiida.orm import KpointsData

    code = aiida_code_installed(label='test.proto-reject-node', default_calc_job_plugin='quantumespresso.pw')
    mesh = KpointsData()
    mesh.set_kpoints_mesh([2, 2, 2])
    with pytest.raises(ValueError, match='holds an ORM node'):
        protocols.get_inputs_from_protocol(
            'pw_relax',
            structure=_silicon(),
            code=code,
            pseudos={'Si': _upf()},
            overrides={'base_relax': {'kpoints': mesh}},
        )


def test_launch_relax_type_and_spin(aiida_code_installed) -> None:
    """Relax types map onto calculation modes; spin and insulator settings resolve."""
    code = aiida_code_installed(label='test.proto-relax-spin', default_calc_job_plugin='quantumespresso.pw')
    base = {
        'structure': _silicon(),
        'code': code,
        'pseudos': {'Si': _upf()},
        'overrides': _cutoff_overrides(30.0, 240.0, 'base_relax', 'base_init_relax'),
    }
    scf = protocols.get_inputs_from_protocol('pw_relax', relax_type=RelaxType.NONE, **base)
    assert scf['parameters'].get_dict()['CONTROL']['calculation'] == 'scf'
    assert 'CELL' not in scf['parameters'].get_dict()
    with pytest.raises(ValueError, match='not supported'):
        protocols.get_inputs_from_protocol('pw_relax', relax_type=RelaxType.VOLUME, **base)
    collinear = protocols.get_inputs_from_protocol('pw_relax', spin_type=SpinType.COLLINEAR, **base)
    system = collinear['parameters'].get_dict()['SYSTEM']
    assert system['nspin'] == 2
    assert system['starting_magnetization']['Si'] != 0
    insulator = protocols.get_inputs_from_protocol('pw_relax', electronic_type=ElectronicType.INSULATOR, **base)
    insulator_system = insulator['parameters'].get_dict()['SYSTEM']
    assert insulator_system['occupations'] == 'fixed'
    assert 'smearing' not in insulator_system
    with pytest.raises(ValueError, match='not supported'):
        protocols.get_inputs_from_protocol('pw_relax', electronic_type=ElectronicType.AUTOMATIC, **base)


def test_launch_inputs_do_not_mutate_callers(aiida_code_installed) -> None:
    """Resolving launch inputs leaves the structure and overrides untouched."""
    code = aiida_code_installed(label='test.proto-nomut', default_calc_job_plugin='quantumespresso.pw')
    structure = _silicon()
    before_sites = [(site.kind_name, tuple(site.position)) for site in structure.sites]
    overrides = _cutoff_overrides(30.0, 240.0, 'base_relax', 'base_init_relax')
    snapshot = copy.deepcopy(overrides)
    protocols.get_inputs_from_protocol(
        'pw_relax', structure=structure, code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    assert overrides == snapshot
    assert [(site.kind_name, tuple(site.position)) for site in structure.sites] == before_sites


def test_launch_bands_inputs(aiida_code_installed) -> None:
    """The bands resolver splits SCF/bands parameters and emits hierarchical metadata aliases."""
    code = aiida_code_installed(label='test.proto-bands-pw', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'bands'),
    )
    assert launch['scf_parameters'].get_dict()['CONTROL']['calculation'] == 'scf'
    assert launch['bands_parameters'].get_dict()['CONTROL']['calculation'] == 'bands'
    assert launch['bands_kpoints_distance'].value == pytest.approx(0.025)
    assert launch['nbands_factor'].value == pytest.approx(3.0)
    assert launch['scf_parameters'].get_dict()['SYSTEM']['ecutwfc'] == 30.0
    task_metadata = launch['task_metadata']
    assert task_metadata['pw_calc']['options']['max_wallclock_seconds'] == 43200
    assert task_metadata['pw_base_graph__pw_calc'] == task_metadata['pw_calc']
    assert task_metadata['pw_base_graph_2__pw_calc']['options']['max_wallclock_seconds'] == 43200


def test_launch_bands_kpoints_conflict(aiida_code_installed) -> None:
    """Explicit bands k-points conflict with the protocol distance."""
    from aiida.orm import KpointsData

    code = aiida_code_installed(label='test.proto-bands-k', default_calc_job_plugin='quantumespresso.pw')
    path = KpointsData()
    path.set_kpoints_mesh([2, 2, 2])
    with pytest.raises(ValueError, match='both `bands_kpoints` and `bands_kpoints_distance`'):
        protocols.get_inputs_from_protocol(
            'pw_bands',
            structure=_silicon(),
            scf_code=code,
            bands_code=code,
            pseudos={'Si': _upf()},
            bands_kpoints=path,
            overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'bands'),
        )


def test_launch_pdos_inputs(aiida_code_installed) -> None:
    """The PDOS resolver maps SCF/NSCF/DOS/PROJWFC sections onto the from-scratch root."""
    code = aiida_code_installed(label='test.proto-pdos-pw', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pdos',
        structure=_silicon(),
        pw_code=code,
        dos_code=code,
        projwfc_code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'nscf'),
    )
    nscf = launch['nscf_parameters'].get_dict()
    assert nscf['CONTROL']['calculation'] == 'nscf'
    assert nscf['SYSTEM']['occupations'].startswith('tetrahedra')
    assert 'smearing' not in nscf['SYSTEM']
    assert launch['nscf_kpoints_distance'].value == pytest.approx(0.10)
    assert launch['dos_parameters'].get_dict()['DOS']['deltae'] == pytest.approx(0.01)
    task_metadata = launch['task_metadata']
    assert task_metadata['dos_calc']['options']['max_wallclock_seconds'] == 43200
    assert task_metadata['pdos_stage_parallel__dos_calc'] == task_metadata['dos_calc']
    assert task_metadata['pdos_stage_parallel__pw_base_graph__pw_calc']['options']['max_wallclock_seconds'] == 43200
    serial = protocols.get_inputs_from_protocol(
        'pdos',
        structure=_silicon(),
        pw_code=code,
        dos_code=code,
        projwfc_code=code,
        pseudos={'Si': _upf()},
        serial=True,
        overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'nscf'),
    )
    assert 'pdos_stage_serial__dos_calc' in serial['task_metadata']


@pytest.mark.parametrize('kind', ['pw_relax', 'pw_bands'])
def test_launch_does_not_silently_ignore_cleanup(kind) -> None:
    """Unsupported in-graph cleanup is rejected before resolving any scientific inputs."""
    with pytest.raises(ValueError, match='cleanup_graph_workdirs'):
        protocols.get_inputs_from_protocol(kind, structure=_silicon(), overrides={'clean_workdir': True})


@pytest.mark.parametrize('kind, section', [('pw_bands', 'scf'), ('pdos', 'nscf')])
def test_launch_does_not_silently_ignore_mesh_parity(kind, section) -> None:
    """Composite mesh options without a consumer must not disappear at the launch boundary."""
    with pytest.raises(ValueError, match='kpoints_force_parity'):
        protocols.get_inputs_from_protocol(
            kind, structure=_silicon(), overrides={section: {'kpoints_force_parity': True}}
        )


def test_launch_pdos_requires_codes() -> None:
    """Each family names its missing codes."""
    with pytest.raises(ValueError, match='`code` is required'):
        protocols.get_inputs_from_protocol('pw_relax', structure=_silicon())
    with pytest.raises(ValueError, match='`scf_code` and `bands_code`'):
        protocols.get_inputs_from_protocol('pw_bands', structure=_silicon())
    with pytest.raises(ValueError, match='`pw_code`, `dos_code`, and `projwfc_code`'):
        protocols.get_inputs_from_protocol('pdos', structure=_silicon())


def test_launch_relax_production_graph(monkeypatch, aiida_code_installed) -> None:
    """Generated relax inputs launch the production relax graph through one stub run."""
    _patch_nested(monkeypatch, pw_stub=SuccessPwCalc)
    structure = _silicon()
    snapshot = copy.deepcopy(structure.cell)
    code = aiida_code_installed(label='test.proto-run-relax', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoff_overrides(30.0, 240.0, 'base_relax', 'base_init_relax')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax', structure=structure, code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    outputs, node = run_get_node(pw_relax_graph, **launch)
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict()['status'] == 'converged'
    assert structure.cell == snapshot
    pw_calcs = _pw_calcs(node)
    # The initial and the main relax run their own resolved thresholds.
    assert len(pw_calcs) == 2
    etot_values = sorted(calc.inputs.parameters.get_dict()['CONTROL']['etot_conv_thr'] for calc in pw_calcs)
    assert etot_values == pytest.approx([2 * 1.0e-5, 2 * 1.0e-3])
    for calc in pw_calcs:
        parameters = calc.inputs.parameters.get_dict()
        assert parameters['SYSTEM']['ecutwfc'] == 30.0
        assert parameters['SYSTEM']['ecutrho'] == 240.0


def test_launch_bands_production_graph(monkeypatch, aiida_code_installed) -> None:
    """Generated bands inputs launch the production bands graph with SeekPath products."""
    _patch_nested(monkeypatch, pw_stub=FullPwCalc)
    code = aiida_code_installed(label='test.proto-run-bands', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'bands'),
    )
    outputs, node = run_get_node(pw_bands_graph, **launch)
    assert node.is_finished_ok, node.exit_message
    assert isinstance(outputs['band_structure'], orm.BandsData)
    assert isinstance(outputs['primitive_structure'], orm.StructureData)
    pw_calcs = _pw_calcs(node)
    assert len(pw_calcs) == 2
    calculations = sorted(calc.inputs.parameters.get_dict()['CONTROL']['calculation'] for calc in pw_calcs)
    assert calculations == ['bands', 'scf']


def test_launch_bands_per_scope_options(monkeypatch, aiida_code_installed) -> None:
    """Hierarchical metadata aliases reach the matching scope: SCF and bands differ."""
    _patch_nested(monkeypatch, pw_stub=FullPwCalc)
    code = aiida_code_installed(label='test.proto-run-scoped', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'bands'),
        task_metadata={
            'pw_base_graph__pw_calc': {'options': {'max_wallclock_seconds': 1111}},
            'pw_base_graph_2__pw_calc': {'options': {'max_wallclock_seconds': 2222}},
        },
    )
    outputs, node = run_get_node(pw_bands_graph, **launch)
    assert node.is_finished_ok, node.exit_message
    assert isinstance(outputs['band_structure'], orm.BandsData)
    by_role = {}
    for calc in _pw_calcs(node):
        role = 'bands' if 'parent_folder' in calc.inputs else 'scf'
        by_role[role] = calc.get_option('max_wallclock_seconds')
    assert by_role == {'scf': 1111, 'bands': 2222}


def test_launch_pdos_production_graph(monkeypatch, aiida_code_installed) -> None:
    """Generated PDOS inputs launch the production serial graph with DOS cleanup ordered before PROJWFC."""
    _patch_nested(monkeypatch, pw_stub=FullPwCalc, dos_stub=StubDosCalc, projwfc_stub=StubProjwfcCalc)
    pw_code = aiida_code_installed(label='test.proto-run-pdos-pw', default_calc_job_plugin='quantumespresso.pw')
    dos_code = aiida_code_installed(label='test.proto-run-pdos-dos', default_calc_job_plugin='quantumespresso.dos')
    projwfc_code = aiida_code_installed(
        label='test.proto-run-pdos-projwfc', default_calc_job_plugin='quantumespresso.projwfc'
    )
    launch = protocols.get_inputs_from_protocol(
        'pdos',
        structure=_silicon(),
        pw_code=pw_code,
        dos_code=dos_code,
        projwfc_code=projwfc_code,
        pseudos={'Si': _upf()},
        serial=True,
        overrides=_cutoff_overrides(30.0, 240.0, 'scf', 'nscf'),
    )
    outputs, node = run_get_node(pdos_from_scratch_serial, **launch)
    assert node.is_finished_ok, node.exit_message
    assert outputs['report']['status'] == 'completed'
    assert outputs['report']['mode'] == 'serial'
    assert outputs['dos_output_dos'].is_stored
    nscf_calcs = [
        calc for calc in _pw_calcs(node) if calc.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'nscf'
    ]
    assert nscf_calcs
    for calc in nscf_calcs:
        assert calc.inputs.parameters.get_dict()['SYSTEM']['ecutwfc'] == 30.0
