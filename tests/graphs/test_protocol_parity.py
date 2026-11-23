############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Protocol parity for the native QE graphs: fixed-scheduler ``max_seconds`` preparation.

Run from the parent core checkout with the QE dependencies installed::

    PYTHONPATH=aiida-quantumespresso/src uv run --with aiida-pseudo --with qe-tools \\
        pytest aiida-quantumespresso/tests/graphs/test_protocol_parity.py -q

The legacy ``PwBaseWorkChain.prepare_process`` derives ``CONTROL.max_seconds`` as
``0.95 * metadata.options.max_wallclock_seconds`` on every attempt when the
parameters carry no explicit value. Scheduler options are fixed at launch for the
native graphs, so :func:`get_inputs_from_protocol
<aiida_quantumespresso.workflows.graphs.protocols.get_inputs_from_protocol>`
prepares the same cutoff per ``pw`` placement up front. These tests pin that
contract against the production graphs with only the external ``CalcJob``
placements substituted: per-stage wallclocks resolve independently, an explicit
``max_seconds`` always wins, per-stage ``pw.settings`` reach the matching
``CalcJob`` (and stay constant across base-loop retries), and monitor mappings
outside the single supported entry fail fast at the boundary.
"""

import io

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida import orm
from aiida.engine import DeferredGraphInput, ResolvedGraphInput

from aiida_quantumespresso.workflows.graphs import protocols
from aiida_quantumespresso.workflows.graphs.pdos import pdos_from_scratch_serial
from aiida_quantumespresso.workflows.graphs.pw import pw_bands_graph, pw_relax_graph

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]

_WALLCLOCK = 43200
_EXPECTED = _WALLCLOCK * 0.95


def _silicon() -> orm.StructureData:
    """Return an unstored two-site silicon structure."""
    structure = orm.StructureData(cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 2.715]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols='Si')
    return structure


def _upf(element: str = 'Si'):
    """Return an unstored parsable pseudo with a valence charge."""
    from aiida.plugins import DataFactory

    UpfData = DataFactory('pseudo.upf')
    content = f'<UPF version="2.0.1"><PP_HEADER\\nelement="{element}"\\nz_valence="4.0"\\n/></UPF>\\n'
    return UpfData(io.BytesIO(content.encode('utf-8')), filename=f'{element}.upf')


def _cutoffs(*sections: str) -> dict:
    """Return per-section cutoff overrides for explicit pseudos."""
    return {section: {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}} for section in sections}


def test_relax_prepares_max_seconds_per_placement(aiida_code_installed) -> None:
    """Both relax placements seed ``0.95`` of the protocol wallclock."""
    code = aiida_code_installed(label='test.parity-relax-default', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
    )
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)


def test_relax_per_stage_wallclocks_stay_independent(aiida_code_installed) -> None:
    """Initial and main wallclocks derive independent cutoffs."""
    code = aiida_code_installed(label='test.parity-relax-scoped', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('base_relax', 'base_init_relax')
    overrides['base_relax']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 2222}}
    overrides['base_init_relax']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 1111}}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax', structure=_silicon(), code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(2222 * 0.95)
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(1111 * 0.95)


def test_relax_preserves_explicit_max_seconds(aiida_code_installed) -> None:
    """An explicit ``CONTROL.max_seconds`` always wins over the derived cutoff."""
    code = aiida_code_installed(label='test.parity-relax-explicit', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('base_relax', 'base_init_relax')
    overrides['base_relax']['pw']['parameters']['CONTROL'] = {'max_seconds': 1234.0}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax', structure=_silicon(), code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(1234.0)
    # The untouched stage still derives its cutoff.
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)


def test_relax_global_options_seed_both_cutoffs(aiida_code_installed) -> None:
    """The ``options`` keyword merges into every placement before the cutoff is derived."""
    code = aiida_code_installed(label='test.parity-relax-global', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
        options={'max_wallclock_seconds': 3600},
    )
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(3600 * 0.95)
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(3600 * 0.95)


def test_relax_explicit_runtime_config_wallclock_wins(aiida_code_installed) -> None:
    """Explicit ``runtime_config`` wallclocks feed the cutoff, not the stale section defaults."""
    code = aiida_code_installed(label='test.parity-relax-tm', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
        runtime_config={'main': {'pw_calc': {'metadata': {'options': {'max_wallclock_seconds': 999}}}}},
    )
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(999 * 0.95)
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)


def test_bands_prepares_max_seconds_per_stage(aiida_code_installed) -> None:
    """SCF and bands wallclocks derive independent cutoffs."""
    code = aiida_code_installed(label='test.parity-bands-scoped', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('scf', 'bands')
    overrides['scf']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 1111}}
    overrides['bands']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 2222}}
    launch = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides=overrides,
    )
    assert launch['scf_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(1111 * 0.95)
    assert launch['bands_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(2222 * 0.95)


def test_pdos_prepares_max_seconds_for_pw_stages(aiida_code_installed) -> None:
    """SCF and NSCF cutoffs derive from their own scopes; DOS/PROJWFC carry no ``CONTROL`` cutoff."""
    code = aiida_code_installed(label='test.parity-pdos-pw', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pdos',
        structure=_silicon(),
        pw_code=code,
        dos_code=code,
        projwfc_code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('scf', 'nscf'),
    )
    assert launch['scf_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)
    assert launch['nscf_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)
    assert 'max_seconds' not in launch['dos_parameters'].get_dict().get('DOS', {})
    assert 'max_seconds' not in launch['projwfc_parameters'].get_dict().get('PROJWFC', {})


def test_ensure_max_seconds_without_wallclock_is_noop() -> None:
    """Placements without a scheduler wallclock get no cutoff."""
    parameters: dict = {'CONTROL': {'calculation': 'scf'}}
    protocols._ensure_max_seconds(parameters, {})
    assert 'max_seconds' not in parameters['CONTROL']
    protocols._ensure_max_seconds(parameters, {'withmpi': True})
    assert 'max_seconds' not in parameters['CONTROL']


def test_monitor_boundary_rejects_unknown_mappings(aiida_code_installed) -> None:
    """Monitor mappings outside the single supported entry fail fast at the launch boundary."""
    code = aiida_code_installed(label='test.parity-monitors', default_calc_job_plugin='quantumespresso.pw')
    base = {
        'structure': _silicon(),
        'code': code,
        'pseudos': {'Si': _upf()},
        'overrides': _cutoffs('base_relax', 'base_init_relax'),
    }
    with pytest.raises(ValueError, match='only the accuracy-stuck monitor'):
        protocols.get_inputs_from_protocol(
            'pw_relax',
            **base,
            monitors={'stuck': {'entry_point': 'quantumespresso.unknown', 'kwargs': {}}},
        )
    with pytest.raises(ValueError, match='is not supported; expected one of'):
        protocols.get_inputs_from_protocol(
            'pw_relax',
            **base,
            monitors={'stuck': {'entry_point': 'quantumespresso.accuracy_stuck', 'bogus_kwarg': 1}},
        )
    with pytest.raises(ValueError, match='must be a `DataDict` or a plain mapping'):
        protocols.get_inputs_from_protocol('pw_relax', **base, monitors='stuck')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='only the `stuck` monitor key'):
        protocols.get_inputs_from_protocol(
            'pw_relax',
            **base,
            monitors={'watchdog': {'entry_point': 'quantumespresso.accuracy_stuck', 'kwargs': {}}},
        )


def test_production_graphs_build() -> None:
    """Every production graph under test validates without launching anything."""
    assert pw_relax_graph.build() is None
    assert pw_bands_graph.build() is None
    assert pdos_from_scratch_serial.build() is None


def test_production_relax_carries_cutoff_and_options(aiida_code_installed) -> None:
    """Generated relax launch inputs carry per-placement cutoffs and options to the graph boundary."""
    code = aiida_code_installed(label='test.parity-run-relax', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('base_relax', 'base_init_relax')
    overrides['base_relax']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 2222}}
    overrides['base_init_relax']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 1111}}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax', structure=_silicon(), code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    # Both static placements carry the same options; the launch mapping below is what the
    # production graph consumes.
    assert (
        launch['runtime_config']['init_relax_guarded']['initial']['pw_calc']['metadata']['options'][
            'max_wallclock_seconds'
        ]
        == 1111
    )
    assert launch['runtime_config']['main']['pw_calc']['metadata']['options']['max_wallclock_seconds'] == 2222
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(2222 * 0.95)
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(1111 * 0.95)
    assert pw_relax_graph.build() is None


def test_production_bands_per_stage_settings_and_cutoffs(aiida_code_installed) -> None:
    """Distinct SCF/bands settings and cutoffs resolve to the matching production scopes."""
    code = aiida_code_installed(label='test.parity-run-bands', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('scf', 'bands')
    overrides['scf']['pw']['settings'] = {'cmdline': ['-nk', '2']}
    overrides['scf']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 1111}}
    overrides['bands']['pw']['settings'] = {'cmdline': ['-nk', '4']}
    overrides['bands']['pw']['metadata'] = {'options': {'max_wallclock_seconds': 2222}}
    launch = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides=overrides,
    )
    # Both static placements carry the same per-stage options; asserted at the launch
    # boundary the production graph consumes unchanged.
    assert launch['scf_settings'].get_dict() == {'cmdline': ['-nk', '2']}
    assert launch['bands_settings'].get_dict() == {'cmdline': ['-nk', '4']}
    assert launch['scf_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(1111 * 0.95)
    assert launch['bands_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(2222 * 0.95)
    assert launch['runtime_config']['scf']['pw_calc']['metadata']['options']['max_wallclock_seconds'] == 1111
    assert launch['runtime_config']['bands']['pw_calc']['metadata']['options']['max_wallclock_seconds'] == 2222
    assert pw_bands_graph.build() is None


def test_prepared_bands_preview_shows_resolved_and_deferred_inputs(aiida_code_installed) -> None:
    """Preview a protocol launch without running SCF or guessing its output folder."""
    code = aiida_code_installed(label='test.preview-bands', default_calc_job_plugin='quantumespresso.pw')
    prepared = protocols.prepare_run_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('scf', 'bands'),
    )

    preview = pw_bands_graph.build(prepared_run=prepared)

    placements = {placement.address: placement for placement in preview.placements}
    for scope in ('scf', 'bands'):
        placement = placements[f'{scope}.pw_calc']
        assert placement.inputs['code'] == ResolvedGraphInput(code)
        assert placement.inputs['metadata.options.max_wallclock_seconds'] == ResolvedGraphInput(_WALLCLOCK)
        assert any(region.endswith(':loop') for region in placement.regions)
    assert isinstance(placements['bands.pw_calc'].inputs['parent_folder'], DeferredGraphInput)
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0


def test_production_pdos_serial_carries_nscf_cutoff(aiida_code_installed) -> None:
    """Generated PDOS launch inputs prepare the NSCF cutoff for the production serial graph."""
    pw_code = aiida_code_installed(label='test.parity-run-pdos-pw', default_calc_job_plugin='quantumespresso.pw')
    dos_code = aiida_code_installed(label='test.parity-run-pdos-dos', default_calc_job_plugin='quantumespresso.dos')
    projwfc_code = aiida_code_installed(
        label='test.parity-run-pdos-projwfc', default_calc_job_plugin='quantumespresso.projwfc'
    )
    launch = protocols.get_inputs_from_protocol(
        'pdos',
        structure=_silicon(),
        pw_code=pw_code,
        dos_code=dos_code,
        projwfc_code=projwfc_code,
        pseudos={'Si': _upf()},
        serial=True,
        overrides=_cutoffs('scf', 'nscf'),
    )
    # Both static placements carry the same options; the cutoff is asserted at the launch boundary.
    assert launch['nscf_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)
    assert launch['scf_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(_EXPECTED)
    assert pdos_from_scratch_serial.build() is None


def test_conflicting_pw_leaves_fail_fast(aiida_code_installed) -> None:
    """Divergent ``pw_calc``/``pw_calc_monitored`` overrides fail instead of desynchronizing."""
    code = aiida_code_installed(label='test.parity-conflict', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(ValueError, match=r'conflicting `pw_calc` and `pw_calc_monitored`'):
        protocols.get_inputs_from_protocol(
            'pw_relax',
            structure=_silicon(),
            code=code,
            pseudos={'Si': _upf()},
            overrides=_cutoffs('base_relax', 'base_init_relax'),
            runtime_config={
                'main': {
                    'pw_calc': {'metadata': {'options': {'max_wallclock_seconds': 111}}},
                    'pw_calc_monitored': {'metadata': {'options': {'max_wallclock_seconds': 222}}},
                }
            },
        )


def test_normalize_pw_leaves_single_sided_and_conflict() -> None:
    """Single-sided overrides duplicate onto the sibling; conflicting siblings name the scope."""
    single = protocols._normalize_pw_leaves({'scope': {'pw_calc_monitored': {'metadata': {'options': {'a': 1}}}}})
    assert single['scope'] == {'pw_calc': {'metadata': {'options': {'a': 1}}}}
    with pytest.raises(ValueError, match=r'runtime_config\.scope'):
        protocols._normalize_pw_leaves(
            {
                'scope': {
                    'pw_calc': {'metadata': {'options': {'a': 1}}},
                    'pw_calc_monitored': {'metadata': {'options': {'a': 2}}},
                }
            }
        )


def test_ensure_max_seconds_rejects_invalid_wallclock() -> None:
    """Non-numeric wallclocks fail fast instead of silently skipping the cutoff."""
    for bad in ('soon', [3600], {'seconds': 3600}, True, float('nan'), 0, -100):
        with pytest.raises(ValueError, match='max_wallclock_seconds'):
            protocols._ensure_max_seconds({'CONTROL': {}}, {'max_wallclock_seconds': bad}, where='scf.pw_calc')
    parameters: dict = {'CONTROL': {}}
    protocols._ensure_max_seconds(parameters, {'max_wallclock_seconds': 3600}, where='scf.pw_calc')
    assert parameters['CONTROL']['max_seconds'] == pytest.approx(3600 * 0.95)


def test_relax_init_and_main_inputs_stay_independent(aiida_code_installed) -> None:
    """Per-stage meshes, settings, and parity resolve without leaking across stages."""
    code = aiida_code_installed(label='test.parity-relax-split', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('base_relax', 'base_init_relax')
    overrides['base_relax']['kpoints_distance'] = 0.2
    overrides['base_init_relax']['kpoints_distance'] = 0.4
    overrides['base_relax']['kpoints_force_parity'] = True
    overrides['base_init_relax']['pw']['settings'] = {'cmdline': ['-nk', '1']}
    overrides['base_relax']['pw']['settings'] = {'cmdline': ['-nk', '2']}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax', structure=_silicon(), code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    assert launch['kpoints_distance'].value == pytest.approx(0.2)
    assert launch['init_kpoints_distance'].value == pytest.approx(0.4)
    assert launch['settings'].get_dict() == {'cmdline': ['-nk', '2']}
    assert launch['init_settings'].get_dict() == {'cmdline': ['-nk', '1']}
    assert launch['kpoints_force_parity'] is True
    assert launch['init_kpoints_force_parity'] is False
    assert pw_relax_graph.build() is None


def test_relax_init_defaults_follow_protocol_sections(aiida_code_installed) -> None:
    """Protocol defaults already differ per stage and are passed explicitly."""
    code = aiida_code_installed(label='test.parity-relax-defaults', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
    )
    assert launch['kpoints_distance'].value == pytest.approx(0.15)
    assert launch['init_kpoints_distance'].value == pytest.approx(0.3)
    assert 'settings' not in launch
    assert 'init_settings' not in launch


def test_relax_fixed_coords_preserved_per_stage(aiida_code_installed) -> None:
    """``FIXED_COORDS`` is added per stage without discarding stage settings."""
    from aiida_quantumespresso.common.types import RelaxType

    code = aiida_code_installed(label='test.parity-relax-fixed', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('base_relax', 'base_init_relax')
    overrides['base_relax']['pw']['settings'] = {'main_key': True}
    overrides['base_init_relax']['pw']['settings'] = {'init_key': True}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=overrides,
        relax_type=RelaxType.SHAPE,
    )
    settings = launch['settings'].get_dict()
    init_settings = launch['init_settings'].get_dict()
    assert settings['main_key'] is True
    assert init_settings['init_key'] is True
    assert settings['FIXED_COORDS'] == [[True, True, True]] * 2
    assert init_settings['FIXED_COORDS'] == [[True, True, True]] * 2
    assert 'init_key' not in settings
    assert 'main_key' not in init_settings


def test_runtime_config_wallclock_precedence_for_cutoff(aiida_code_installed) -> None:
    """Explicit per-scope wallclocks beat global options; untouched scopes keep the global."""
    code = aiida_code_installed(label='test.parity-precedence', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
        options={'max_wallclock_seconds': 3600},
        runtime_config={'main': {'pw_calc': {'metadata': {'options': {'max_wallclock_seconds': 555}}}}},
    )
    assert launch['parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(555 * 0.95)
    assert launch['init_parameters'].get_dict()['CONTROL']['max_seconds'] == pytest.approx(3600 * 0.95)
    assert launch['runtime_config']['main']['pw_calc']['metadata']['options']['max_wallclock_seconds'] == 555


def test_relax_protocol_disables_inheritance(aiida_code_installed) -> None:
    """The protocol helper launches with ``init_use_main_inputs=False`` and keeps absent init settings absent."""
    code = aiida_code_installed(label='test.parity-no-inherit', default_calc_job_plugin='quantumespresso.pw')
    overrides = _cutoffs('base_relax', 'base_init_relax')
    overrides['base_relax']['pw']['settings'] = {'cmdline': ['-nk', '2']}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax', structure=_silicon(), code=code, pseudos={'Si': _upf()}, overrides=overrides
    )
    assert launch['init_use_main_inputs'] is False
    assert launch['settings'].get_dict() == {'cmdline': ['-nk', '2']}
    assert 'init_settings' not in launch


def test_relax_main_kpoints_do_not_overwrite_init_density(aiida_code_installed) -> None:
    """An explicit main mesh leaves the initial protocol density intact."""
    from aiida import orm as _orm

    code = aiida_code_installed(label='test.parity-kpoints-split', default_calc_job_plugin='quantumespresso.pw')
    mesh = _orm.KpointsData()
    mesh.set_kpoints_mesh([2, 2, 2])
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
        kpoints=mesh,
    )
    assert launch['kpoints'] is mesh
    assert 'init_kpoints' not in launch
    assert launch['init_kpoints_distance'].value == pytest.approx(0.3)


def test_relax_init_kpoints_keyword(aiida_code_installed) -> None:
    """The ``init_kpoints`` keyword addresses the initial stage without touching the main mesh."""
    from aiida import orm as _orm

    code = aiida_code_installed(label='test.parity-init-kpoints', default_calc_job_plugin='quantumespresso.pw')
    init_mesh = _orm.KpointsData()
    init_mesh.set_kpoints_mesh([3, 3, 3])
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
        init_kpoints=init_mesh,
    )
    assert launch['init_kpoints'] is init_mesh
    assert launch['kpoints_distance'].value == pytest.approx(0.15)
    assert 'init_kpoints_distance' not in launch


def test_relax_shared_monitors_stay_explicit_per_stage(aiida_code_installed) -> None:
    """A shared monitors mapping resolves onto both stages with no inheritance at launch."""
    code = aiida_code_installed(label='test.parity-shared-mon', default_calc_job_plugin='quantumespresso.pw')
    monitors = {'stuck': {'entry_point': 'quantumespresso.accuracy_stuck', 'kwargs': {}}}
    launch = protocols.get_inputs_from_protocol(
        'pw_relax',
        structure=_silicon(),
        code=code,
        pseudos={'Si': _upf()},
        overrides=_cutoffs('base_relax', 'base_init_relax'),
        monitors=monitors,
    )
    assert launch['init_use_main_inputs'] is False
    assert launch['monitors'] is launch['init_monitors']
    assert pw_relax_graph.build() is None


def test_coerce_monitors_datadict_normalizes_without_mutating() -> None:
    """A ``DataDict`` boundary input normalizes onto new nodes; the input stays untouched."""
    entry_point = 'quantumespresso.accuracy_stuck'
    original = orm.DataDict({'stuck': orm.Dict({'entry_point': entry_point, 'kwargs': {'min_repeats': 3}})})
    coerced = protocols._coerce_monitors_input(original, 'monitors')
    assert coerced is not original
    assert coerced is not None
    assert coerced['stuck'].get_dict()['kwargs'] == {'min_repeats': 3, 'structured': True}
    assert original['stuck'].get_dict() == {'entry_point': entry_point, 'kwargs': {'min_repeats': 3}}
    # ``bool`` is an ``int`` subclass but never a valid repeat count; non-positive counts
    # would silently misfire the plateau check, so both fail fast at the boundary.
    with pytest.raises(ValueError, match='must be an `int`'):
        protocols._coerce_monitor_entry(
            'stuck', {'entry_point': entry_point, 'kwargs': {'min_repeats': True}}, 'monitors'
        )
    for bad in (0, -2):
        with pytest.raises(ValueError, match='must be positive'):
            protocols._coerce_monitor_entry(
                'stuck', {'entry_point': entry_point, 'kwargs': {'min_repeats': bad}}, 'monitors'
            )
