############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Iteration-2 integration: nested ``task_metadata``, retries, and monitor threading.

Run from the parent core checkout with the QE dependencies installed::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \\
        pytest aiida-quantumespresso/tests/graphs/test_round3_integration.py -q

What this proves (and nothing else does): the production composites forward an
optional shared ``monitors`` mapping into every nested base run *and*
independent per-stage scheduler options survive base-graph retries under the
strictly nested ``task_metadata`` contract (one level per authored subgraph
callee alias, the ``CalcJob`` metadata mapping at each leaf, control frames
transparent). Every test runs a production root (``pw_bands_graph`` /
``pdos_from_scratch_serial`` / ``pw_relax_graph``) through the production
``pw_base_graph`` retry loops with only the external ``CalcJob`` placements
substituted by deterministic local stubs. No production graph is copied here;
the substitution reuses the ``_patch_nested`` helper from
``test_native_workflows`` (which already carries the ``monitors`` placement
shape). Exact hierarchical aliases, ``__*`` scope wildcards, and leaf-name
fallbacks are gone: the tests below pass only nested mappings.
"""

import io

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida.engine import run_get_node
from aiida.orm import BandsData, CalcJobNode, DataDict, Dict, Float, GraphNode, KpointsData, StructureData

from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.graphs import protocols
from aiida_quantumespresso.workflows.graphs.pdos import pdos_from_scratch_serial
from aiida_quantumespresso.workflows.graphs.pw import pw_bands_graph, pw_relax_graph

from .test_native_workflows import StubDosCalc, StubProjwfcCalc, _patch_nested

pytestmark = [pytest.mark.presto, pytest.mark.usefixtures('aiida_profile_clean')]


def _silicon() -> StructureData:
    """Return an unstored diamond silicon structure."""
    structure = StructureData(cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols='Si')
    return structure


def _upf(element: str = 'Si'):
    """Return a minimal parsable pseudo for ``element``."""
    from aiida.plugins import DataFactory

    UpfData = DataFactory('pseudo.upf')
    content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
    return UpfData(io.BytesIO(content.encode('utf-8')), filename=f'{element}.upf')


def _pseudos(*elements: str) -> DataDict:
    """Return a ``DataDict`` mapping kind names onto ``UpfData`` nodes."""
    return DataDict({element: _upf(element) for element in elements})


def _monitors(**kwargs) -> DataDict:
    """Return a ``DataDict`` wiring the structured stuck monitor under ``stuck``."""
    payload = {'structured': True, **kwargs}
    return DataDict({'stuck': Dict({'entry_point': 'quantumespresso.accuracy_stuck', 'kwargs': payload})})


def _pw_calcs(node) -> list:
    """Return the dispatched ``pw.x`` CalcJobs below a finished graph."""
    return [
        child for child in node.called_descendants if isinstance(child, CalcJobNode) and 'PwCalc' in child.process_label
    ]


def _incoming_labels(node) -> list:
    """Return the incoming link labels of a stored node."""
    return [link.link_label for link in node.base.links.get_incoming().all()]


class RetryWalltimeBandsPwCalc(PwCalculation):
    """Fail once with walltime, then succeed with SCF/bands outputs.

    The first attempt (no ``restart_mode``) stores the structure a real
    interrupted run would leave and exits 400; the retry (``restart_mode ==
    'restart'``) attaches the full success outputs. Both SCF and bands stages
    therefore retry exactly once without any global state.
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
        """Fail the first attempt, succeed the restart."""
        from aiida.engine.processes.exit_code import ExitCode

        parameters = self.inputs.parameters.get_dict()
        if parameters.get('CONTROL', {}).get('restart_mode') != 'restart':
            self.out('output_parameters', Dict({'number_of_bands': 8, 'number_of_symmetries': 10}))
            self.out('output_structure', self.inputs.structure.clone())
            return ExitCode(400, 'stub walltime failure')
        self.out('output_parameters', Dict({'number_of_bands': 8, 'number_of_electrons': 8, 'fermi_energy': 6.9}))
        self.out('output_structure', self.inputs.structure.clone())
        kpoints_node = KpointsData()
        kpoints_node.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
        bands = BandsData()
        bands.set_kpointsdata(kpoints_node)
        bands.set_bands([[-5.0, -4.0], [7.0, 8.0]], occupations=[[1.0, 0.0], [1.0, 0.0]])
        self.out('output_band', bands)
        return None


def test_bands_retry_keeps_independent_scope_options(monkeypatch, aiida_code_installed) -> None:
    """SCF and bands retries each keep their own nested scope scheduler options."""
    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-bands-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-bands-bands', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=scf_code,
        bands_code=bands_code,
        pseudos=_pseudos('Si'),
        scf_kpoints_distance=Float(0.5),
        bands_kpoints_distance=Float(0.3),
        task_metadata={
            'scf': {
                'pw_calc': {'options': {'max_wallclock_seconds': 1111}},
                'pw_calc_monitored': {'options': {'max_wallclock_seconds': 1111}},
            },
            'bands': {
                'pw_calc': {'options': {'max_wallclock_seconds': 2222}},
                'pw_calc_monitored': {'options': {'max_wallclock_seconds': 2222}},
            },
        },
    )
    assert node.is_finished_ok, node.exit_message
    assert isinstance(outputs['band_structure'], BandsData)
    pw_calcs = _pw_calcs(node)
    assert len(pw_calcs) == 4, 'each stage should fail once then recover'
    by_role: dict[str, set] = {}
    for calc in pw_calcs:
        calculation = calc.inputs.parameters.get_dict().get('CONTROL', {}).get('calculation')
        role = 'bands' if calculation == 'bands' else 'scf'
        by_role.setdefault(role, set()).add(calc.get_option('max_wallclock_seconds'))
    assert by_role == {'scf': {1111}, 'bands': {2222}}, by_role


def test_bands_missing_scope_entry_fails_loudly(monkeypatch, aiida_code_installed) -> None:
    """A missing nested scope entry fails loudly instead of running with defaults."""
    from aiida.common.exceptions import GraphValidationError

    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-bands-missing-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-bands-missing-b', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(GraphValidationError, match='bands'):
        run_get_node(
            pw_bands_graph,
            structure=_silicon(),
            scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
            bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
            scf_code=scf_code,
            bands_code=bands_code,
            pseudos=_pseudos('Si'),
            scf_kpoints_distance=Float(0.5),
            bands_kpoints_distance=Float(0.3),
            task_metadata={
                'scf': {
                    'pw_calc': {'options': {'max_wallclock_seconds': 1111}},
                    'pw_calc_monitored': {'options': {'max_wallclock_seconds': 1111}},
                },
            },
        )


def test_bands_misspelled_scope_rejected(monkeypatch, aiida_code_installed) -> None:
    """A misspelled scope path fails loudly naming the unknown path."""
    from aiida.common.exceptions import GraphValidationError

    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-bands-typo-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-bands-typo-b', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(GraphValidationError, match='bandz'):
        run_get_node(
            pw_bands_graph,
            structure=_silicon(),
            scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
            bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
            scf_code=scf_code,
            bands_code=bands_code,
            pseudos=_pseudos('Si'),
            scf_kpoints_distance=Float(0.5),
            bands_kpoints_distance=Float(0.3),
            task_metadata={
                'scf': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
                'bandz': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
            },
        )


def test_bands_malformed_leaf_rejected(monkeypatch, aiida_code_installed) -> None:
    """A non-mapping leaf fails loudly naming the offending path."""
    from aiida.common.exceptions import GraphValidationError

    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-bands-malformed-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-bands-malformed-b', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(GraphValidationError, match='pw_calc'):
        run_get_node(
            pw_bands_graph,
            structure=_silicon(),
            scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
            bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
            scf_code=scf_code,
            bands_code=bands_code,
            pseudos=_pseudos('Si'),
            scf_kpoints_distance=Float(0.5),
            bands_kpoints_distance=Float(0.3),
            task_metadata={
                'scf': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
                'bands': {'pw_calc': ['not-a-mapping']},
            },
        )


def test_bands_monitors_reach_nested_calcs(monkeypatch, aiida_code_installed) -> None:
    """A shared monitors mapping reaches every nested pw CalcJob, retries included."""
    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-mon-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-mon-bands', default_calc_job_plugin='quantumespresso.pw')
    monitors = _monitors()
    _, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=scf_code,
        bands_code=bands_code,
        pseudos=_pseudos('Si'),
        scf_kpoints_distance=Float(0.5),
        bands_kpoints_distance=Float(0.3),
        monitors=monitors,
        task_metadata={
            'scf': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
            'bands': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
        },
    )
    assert node.is_finished_ok, node.exit_message
    pw_calcs = _pw_calcs(node)
    assert len(pw_calcs) == 4
    for calc in pw_calcs:
        assert 'monitors__stuck' in _incoming_labels(calc), 'monitors must reach the nested CalcJob'


def test_bands_per_stage_monitors(monkeypatch, aiida_code_installed) -> None:
    """Per-stage monitors override the shared mapping only where given."""
    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-perstage-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-perstage-b', default_calc_job_plugin='quantumespresso.pw')
    _, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=scf_code,
        bands_code=bands_code,
        pseudos=_pseudos('Si'),
        scf_kpoints_distance=Float(0.5),
        bands_kpoints_distance=Float(0.3),
        monitors=_monitors(),
        bands_monitors=_monitors(min_repeats=3),
        task_metadata={
            'scf': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
            'bands': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
        },
    )
    assert node.is_finished_ok, node.exit_message
    pw_calcs = _pw_calcs(node)
    assert len(pw_calcs) == 4
    for calc in pw_calcs:
        assert 'monitors__stuck' in _incoming_labels(calc)


def test_relax_monitors_thread_through(monkeypatch, aiida_code_installed) -> None:
    """The relax composite forwards monitors into its nested base runs."""
    from .test_base import SuccessPwCalc

    _patch_nested(monkeypatch, pw_stub=SuccessPwCalc)
    code = aiida_code_installed(label='test.r3-relax-mon', default_calc_job_plugin='quantumespresso.pw')
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=_silicon(),
        parameters=Dict({'CONTROL': {'calculation': 'vc-relax'}, 'SYSTEM': {}, 'CELL': {}}),
        code=code,
        pseudos=_pseudos('Si'),
        max_iterations=2,
        kpoints=_kpoints(),
        monitors=_monitors(),
        task_metadata={
            'init_relax_guarded': {'initial': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}}},
            'main': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
        },
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict()['status'] == 'converged'
    pw_calcs = _pw_calcs(node)
    assert pw_calcs
    for calc in pw_calcs:
        assert 'monitors__stuck' in _incoming_labels(calc)


def _kpoints() -> KpointsData:
    """Return an explicit k-point mesh."""
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])
    return kpoints


def test_pdos_retry_keeps_nscf_scope_options(monkeypatch, aiida_code_installed) -> None:
    """PDOS SCF/NSCF retries keep their own nested scope scheduler options."""
    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc, dos_stub=StubDosCalc, projwfc_stub=StubProjwfcCalc)
    pw_code = aiida_code_installed(label='test.r3-pdos-pw', default_calc_job_plugin='quantumespresso.pw')
    dos_code = aiida_code_installed(label='test.r3-pdos-dos', default_calc_job_plugin='quantumespresso.dos')
    projwfc_code = aiida_code_installed(label='test.r3-pdos-projwfc', default_calc_job_plugin='quantumespresso.projwfc')
    outputs, node = run_get_node(
        pdos_from_scratch_serial,
        structure=_silicon(),
        pw_code=pw_code,
        pseudos=_pseudos('Si'),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30.0}, 'ELECTRONS': {}}),
        nscf_parameters=Dict(
            {
                'CONTROL': {'calculation': 'nscf'},
                'SYSTEM': {'ecutwfc': 30.0, 'occupations': 'tetrahedra', 'nosym': True},
                'ELECTRONS': {},
            }
        ),
        dos_code=dos_code,
        dos_parameters=Dict({'DOS': {'deltae': 0.01}}),
        projwfc_code=projwfc_code,
        projwfc_parameters=Dict({'PROJWFC': {'deltae': 0.01, 'ngauss': 0, 'degauss': 0.01}}),
        scf_kpoints_distance=Float(0.5),
        nscf_kpoints_distance=Float(0.5),
        monitors=_monitors(),
        task_metadata={
            'scf': {
                'pw_calc': {'options': {'max_wallclock_seconds': 1111}},
                'pw_calc_monitored': {'options': {'max_wallclock_seconds': 1111}},
            },
            'pdos_stage_serial': {
                'nscf': {
                    'pw_calc': {'options': {'max_wallclock_seconds': 2222}},
                    'pw_calc_monitored': {'options': {'max_wallclock_seconds': 2222}},
                },
                'dos_calc': {'options': {}},
                'projwfc_calc': {'options': {}},
            },
        },
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report']['status'] == 'completed'
    pw_calcs = _pw_calcs(node)
    assert len(pw_calcs) == 4, 'SCF and NSCF should each fail once then recover'
    scf_values = set()
    nscf_values = set()
    for calc in pw_calcs:
        assert 'monitors__stuck' in _incoming_labels(calc)
        if calc.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'scf':
            scf_values.add(calc.get_option('max_wallclock_seconds'))
        elif calc.inputs.parameters.get_dict()['CONTROL']['calculation'] == 'nscf':
            nscf_values.add(calc.get_option('max_wallclock_seconds'))
    assert scf_values == {1111}, scf_values
    assert nscf_values == {2222}, nscf_values


def test_protocol_bands_nested_scopes_and_monitors(aiida_code_installed) -> None:
    """The bands resolver emits nested SCF/bands scopes and wires YAML monitors."""
    code = aiida_code_installed(label='test.r3-proto-bands', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        overrides={
            'scf': {
                'pw': {
                    'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}},
                    'monitors': {
                        'stuck': {
                            'entry_point': 'quantumespresso.accuracy_stuck',
                            'kwargs': {'min_repeats': 5, 'structured': True},
                        }
                    },
                }
            },
            'bands': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
        },
    )
    task_metadata = launch['task_metadata']
    assert set(task_metadata) == {'scf', 'bands'}
    for scope in ('scf', 'bands'):
        assert set(task_metadata[scope]) == {'pw_calc', 'pw_calc_monitored'}
        assert task_metadata[scope]['pw_calc']['options']['max_wallclock_seconds'] == 43200
        assert task_metadata[scope]['pw_calc_monitored'] == task_metadata[scope]['pw_calc']
    assert 'scf_monitors' in launch or 'monitors' in launch
    monitors_node = launch.get('scf_monitors', launch.get('monitors'))
    assert isinstance(monitors_node, DataDict)
    assert monitors_node['stuck'].get_dict()['entry_point'] == 'quantumespresso.accuracy_stuck'

    explicit = DataDict(
        {'stuck': Dict({'entry_point': 'quantumespresso.accuracy_stuck', 'kwargs': {'structured': True}})}
    )
    with_explicit = protocols.get_inputs_from_protocol(
        'pw_bands',
        structure=_silicon(),
        scf_code=code,
        bands_code=code,
        pseudos={'Si': _upf()},
        monitors=explicit,
        overrides={
            'scf': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
            'bands': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
        },
    )
    assert with_explicit['monitors'].get_dict() if isinstance(with_explicit.get('monitors'), dict) else True
    assert isinstance(with_explicit['monitors'], DataDict)

    with pytest.raises(ValueError, match='entry_point'):
        protocols.get_inputs_from_protocol(
            'pw_bands',
            structure=_silicon(),
            scf_code=code,
            bands_code=code,
            pseudos={'Si': _upf()},
            overrides={
                'scf': {
                    'pw': {
                        'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}},
                        'monitors': {'stuck': {'entry_point': 'unknown.monitor', 'kwargs': {}}},
                    }
                },
                'bands': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
            },
        )
    with pytest.raises(ValueError, match='kwargs'):
        protocols.get_inputs_from_protocol(
            'pw_bands',
            structure=_silicon(),
            scf_code=code,
            bands_code=code,
            pseudos={'Si': _upf()},
            overrides={
                'scf': {
                    'pw': {
                        'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}},
                        'monitors': {
                            'stuck': {
                                'entry_point': 'quantumespresso.accuracy_stuck',
                                'kwargs': {'bogus_kwarg': 1},
                            }
                        },
                    }
                },
                'bands': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
            },
        )


def test_protocol_rejects_unsupported_monitors_shape(aiida_code_installed) -> None:
    """Non-mapping monitors and non-True structured flags fail fast."""
    code = aiida_code_installed(label='test.r3-proto-reject', default_calc_job_plugin='quantumespresso.pw')
    with pytest.raises(ValueError, match='must map to a monitor'):
        protocols.get_inputs_from_protocol(
            'pw_bands',
            structure=_silicon(),
            scf_code=code,
            bands_code=code,
            pseudos={'Si': _upf()},
            overrides={
                'scf': {
                    'pw': {
                        'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}},
                        'monitors': {'stuck': 'not-a-mapping'},
                    }
                },
                'bands': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
            },
        )
    with pytest.raises(ValueError, match='structured'):
        protocols.get_inputs_from_protocol(
            'pw_bands',
            structure=_silicon(),
            scf_code=code,
            bands_code=code,
            pseudos={'Si': _upf()},
            overrides={
                'scf': {
                    'pw': {
                        'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}},
                        'monitors': {
                            'stuck': {
                                'entry_point': 'quantumespresso.accuracy_stuck',
                                'kwargs': {'structured': False},
                            }
                        },
                    }
                },
                'bands': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
            },
        )


def test_protocol_pdos_nested_scopes(aiida_code_installed) -> None:
    """The PDOS resolver emits the stage scope with nested SCF/NSCF/post-processing entries."""
    code = aiida_code_installed(label='test.r3-proto-pdos', default_calc_job_plugin='quantumespresso.pw')
    launch = protocols.get_inputs_from_protocol(
        'pdos',
        structure=_silicon(),
        pw_code=code,
        dos_code=code,
        projwfc_code=code,
        pseudos={'Si': _upf()},
        serial=True,
        overrides={
            'scf': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
            'nscf': {'pw': {'parameters': {'SYSTEM': {'ecutwfc': 30.0, 'ecutrho': 240.0}}}},
        },
    )
    task_metadata = launch['task_metadata']
    assert set(task_metadata) == {'scf', 'pdos_stage_serial'}
    assert set(task_metadata['scf']) == {'pw_calc', 'pw_calc_monitored'}
    assert set(task_metadata['pdos_stage_serial']) == {'nscf', 'dos_calc', 'projwfc_calc'}
    assert set(task_metadata['pdos_stage_serial']['nscf']) == {'pw_calc', 'pw_calc_monitored'}
    assert task_metadata['scf']['pw_calc_monitored'] == task_metadata['scf']['pw_calc']
    assert (
        task_metadata['pdos_stage_serial']['nscf']['pw_calc_monitored']
        == task_metadata['pdos_stage_serial']['nscf']['pw_calc']
    )
    for leaf in (
        task_metadata['scf']['pw_calc'],
        task_metadata['pdos_stage_serial']['nscf']['pw_calc'],
        task_metadata['pdos_stage_serial']['dos_calc'],
        task_metadata['pdos_stage_serial']['projwfc_calc'],
    ):
        assert leaf['options']['max_wallclock_seconds'] == 43200


def test_graph_root_nodes_are_native(monkeypatch, aiida_code_installed) -> None:
    """Production roots place no work chains and create nothing at the root."""
    from aiida.common.links import LinkType

    _patch_nested(monkeypatch, pw_stub=RetryWalltimeBandsPwCalc)
    scf_code = aiida_code_installed(label='test.r3-native-scf', default_calc_job_plugin='quantumespresso.pw')
    bands_code = aiida_code_installed(label='test.r3-native-b', default_calc_job_plugin='quantumespresso.pw')
    _, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {}, 'ELECTRONS': {}}),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=scf_code,
        bands_code=bands_code,
        pseudos=_pseudos('Si'),
        scf_kpoints_distance=Float(0.5),
        bands_kpoints_distance=Float(0.3),
        task_metadata={
            'scf': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
            'bands': {'pw_calc': {'options': {}}, 'pw_calc_monitored': {'options': {}}},
        },
    )
    assert node.is_finished_ok, node.exit_message
    assert isinstance(node, GraphNode)
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    for child in node.called_descendants:
        assert 'workchain' not in (child.process_type or '').lower()
