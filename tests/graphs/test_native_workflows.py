############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Full nested integration for the native QE graphs.

Run from the parent core checkout with the QE dependencies installed::

    PYTHONPATH=aiida-quantumespresso/src uv run --with qe-tools --with aiida-pseudo \\
        pytest aiida-quantumespresso/tests/graphs/test_native_workflows.py -q

What this proves (and what the sibling files do not): the production
``pw_relax_graph`` / ``pw_bands_graph`` / ``pdos_from_scratch_serial`` run
*through* the production ``pw_base_graph`` with only the external ``CalcJob``
placements substituted by deterministic local stubs. The sibling
``test_pw.py`` / ``test_pdos.py`` suites replace the whole ``pw_base_graph``
subgraph with a ``@task`` stub (good orchestration coverage, not full
integration); here the retry loop, rebinding, guards, and ``require_*``
wiring all execute for real. The substitution reuses the ``CalcJob`` stub
family from ``test_base.py`` by import and patches only
``base.pw_calc`` (plus ``pdos.dos_calc`` / ``pdos.projwfc_calc`` for PDOS),
resetting every compiled template that embeds those placements so the
production bodies recompile against the substitutes. No production graph is
copied here.

Scheduler options for the ``CalcJob`` placements travel at launch under
``task_metadata`` keyed by placement name (``pw_calc`` / ``dos_calc`` /
``projwfc_calc``), not through graph inputs.
"""

import io

import aiida.engine
import pytest

if not hasattr(aiida.engine, 'graph'):
    pytest.skip('Experimental graphs require the pinned aiida-core 3.x fork', allow_module_level=True)

from aiida.common.links import LinkType
from aiida.engine import run_get_node, task
from aiida.orm import (
    BandsData,
    CalcFunctionNode,
    CalcJobNode,
    DataDict,
    Dict,
    Float,
    GraphNode,
    KpointsData,
    StructureData,
    XyData,
)

import aiida_quantumespresso.workflows.graphs.base as base_module
import aiida_quantumespresso.workflows.graphs.pdos as pdos_module
import aiida_quantumespresso.workflows.graphs.pw as pw_module
from aiida_quantumespresso.calculations.dos import DosCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.graphs.base import PwErrors
from aiida_quantumespresso.workflows.graphs.pdos import pdos_from_scratch_serial
from aiida_quantumespresso.workflows.graphs.pw import pw_bands_graph, pw_relax_graph

from .test_base import SuccessPwCalc

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


def _kpoints() -> KpointsData:
    """Return an explicit k-point mesh."""
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])
    return kpoints


def _vc_relax_parameters() -> Dict:
    """Return ``vc-relax`` parameters, as the relax integration uses them."""
    return Dict({'CONTROL': {'calculation': 'vc-relax'}, 'SYSTEM': {}, 'CELL': {}})


def _scf_parameters() -> Dict:
    """Return SCF ``pw.parameters`` as the bands/PDOS steps use them."""
    return Dict({'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutwfc': 30.0}, 'ELECTRONS': {}})


def _nscf_parameters() -> Dict:
    """Return NSCF ``pw.parameters`` with ``tetrahedra`` occupations."""
    return Dict(
        {
            'CONTROL': {'calculation': 'nscf'},
            'SYSTEM': {'ecutwfc': 30.0, 'occupations': 'tetrahedra', 'nosym': True},
            'ELECTRONS': {},
        }
    )


def _dos_parameters() -> Dict:
    """Return ``dos.x`` parameters with only the grid step."""
    return Dict({'DOS': {'deltae': 0.01}})


def _projwfc_parameters() -> Dict:
    """Return ``projwfc.x`` parameters with only the grid step."""
    return Dict({'PROJWFC': {'deltae': 0.01, 'ngauss': 0, 'degauss': 0.01}})


class FullPwCalc(PwCalculation):
    """Trivial-execution ``PwCalculation`` that also fabricates bands and the Fermi energy.

    ``SuccessPwCalc`` (reused from ``test_base`` for the relax path) attaches only
    ``output_parameters``/``output_structure``; the bands and PDOS paths also need
    ``output_band``, ``number_of_electrons``, and ``fermi_energy`` for the
    band-count rule and the energy-window extraction. The fabricated bands span
    the ``[-5, 8]`` eV window with a Fermi energy of ``6.9`` eV, matching the
    deterministic window the assertions expect.
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
        """Attach the success outputs a real run would store."""
        self.out('output_parameters', Dict({'number_of_bands': 8, 'number_of_electrons': 8, 'fermi_energy': 6.9}))
        self.out('output_structure', self.inputs.structure.clone())
        kpoints_node = KpointsData()
        kpoints_node.set_kpoints([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
        bands = BandsData()
        bands.set_kpointsdata(kpoints_node)
        # Two bands with an empty top band: the base loop's insufficient-bands
        # guard sees an unoccupied LUMO and does not claim a restart, while the
        # window extraction still reads emin=-5.0/max=8.0 eV.
        bands.set_bands([[-5.0, -4.0], [7.0, 8.0]], occupations=[[1.0, 0.0], [1.0, 0.0]])
        self.out('output_band', bands)


class StubDosCalc(DosCalculation):
    """Trivial-execution ``DosCalculation`` echoing the resolved energy window."""

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
        """Echo the resolved ``emin``/``emax`` as a two-point total DOS."""
        import numpy as np

        block = self.inputs.parameters.get_dict().get('DOS', {})
        emin, emax = float(block.get('emin', -5.0)), float(block.get('emax', 8.0))
        total = XyData()
        total.set_x(np.linspace(emin, emax, num=2), 'energy', 'eV')
        total.set_y([np.ones(2)], 'dos', 'states/eV')
        self.out('output_parameters', Dict({'emin': emin, 'emax': emax, 'deltae': block.get('deltae', 0.01)}))
        self.out('output_dos', total)


class StubProjwfcCalc(ProjwfcCalculation):
    """Trivial-execution ``ProjwfcCalculation`` echoing the resolved window as total DOS."""

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
        """Echo the resolved ``emin``/``emax`` as the ``Dos`` output."""
        import numpy as np

        block = self.inputs.parameters.get_dict().get('PROJWFC', {})
        emin, emax = float(block.get('emin', -5.0)), float(block.get('emax', 8.0))
        total = XyData()
        total.set_x(np.linspace(emin, emax, num=2), 'energy', 'eV')
        total.set_y([np.ones(2)], 'dos', 'states/eV')
        self.out('output_parameters', Dict({'emin': emin, 'emax': emax, 'deltae': block.get('deltae', 0.01)}))
        self.out('Dos', total)


def _patch_nested(monkeypatch, pw_stub=None, dos_stub=None, projwfc_stub=None) -> None:
    """Redirect only the external ``CalcJob`` placements, keeping every production graph intact.

    The production graphs resolve placements by module-level name at compilation, so
    patching the placement attributes redirects the real retry loops without copying
    any graph body. Compiled templates are cached on the graph handles, so they are
    reset (restored afterwards by ``monkeypatch``) to force recompilation.
    """
    if pw_stub is not None:
        from aiida.orm import DataDict as _DataDict

        shaped = task(pw_stub, inputs={'pseudos': _DataDict}, errors=PwErrors)
        monkeypatch.setattr(base_module, 'pw_calc', shaped)
    if dos_stub is not None:
        monkeypatch.setattr(pdos_module, 'dos_calc', task(dos_stub))
    if projwfc_stub is not None:
        monkeypatch.setattr(pdos_module, 'projwfc_calc', task(projwfc_stub))
    for handle in (
        base_module.pw_base_graph,
        pw_module.pw_relax_graph,
        pw_module.pw_bands_graph,
        pw_module.init_relax_guarded,
        pdos_module.pdos_stage_parallel,
        pdos_module.pdos_stage_serial,
        pdos_module.pdos_from_scratch_parallel,
        pdos_module.pdos_from_scratch_serial,
        pdos_module.pdos_from_parent_parallel,
        pdos_module.pdos_from_parent_serial,
    ):
        monkeypatch.setattr(handle, '_template', None)


def _assert_native_provenance(node, pseudos: DataDict) -> None:
    """Assert the run placed no work chains, created nothing at the root, and wired pseudos through."""
    assert isinstance(node, GraphNode)
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    descendants = node.called_descendants
    assert descendants, 'the production graph should dispatch child processes'
    for child in descendants:
        assert type(child).__name__ != 'WorkChainNode'
        assert 'workchain' not in (child.process_type or '').lower()
        assert isinstance(child, (CalcJobNode, CalcFunctionNode, GraphNode))
    pw_calcs = [
        child for child in descendants if isinstance(child, CalcJobNode) and child.process_label.endswith('PwCalc')
    ]
    assert pw_calcs, 'the native base loop should place at least one pw CalcJob'
    for calc in pw_calcs:
        incoming = {link.link_label: link.node for link in calc.base.links.get_incoming().all()}
        for kind in pseudos:
            label = f'pseudos__{kind}'
            assert label in incoming, f'the general pseudos mapping must reach every pw CalcJob ({label})'
            assert incoming[label].is_stored


def test_build_production_graphs() -> None:
    """Every production graph under test validates without launching anything."""
    assert pw_relax_graph.build() is None
    assert pw_bands_graph.build() is None
    assert pdos_from_scratch_serial.build() is None


def test_no_workchain_in_production_sources() -> None:
    """The production graph modules place calculations directly: no ``WorkChain`` import."""
    for module in (base_module, pw_module, pdos_module):
        assert module.__file__ is not None
        with open(module.__file__, encoding='utf-8') as handle:
            text = handle.read()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                assert 'WorkChain' not in stripped


def test_native_relax_converged_full_nesting(monkeypatch, aiida_code_installed) -> None:
    """Production relax converges through the production base loop and one stub ``pw.x`` run."""
    _patch_nested(monkeypatch, pw_stub=SuccessPwCalc)
    structure = _silicon()
    pseudos = _pseudos('Si')
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=structure,
        parameters=_vc_relax_parameters(),
        code=aiida_code_installed(label='test.native-relax-pw', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=pseudos,
        max_iterations=5,
        meta_convergence=True,
        kpoints=_kpoints(),
        task_metadata={'pw_calc': {'options': {}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report'].get_dict()['status'] == 'converged'
    assert outputs['report'].get_dict()['attempts'] == 1
    assert outputs['output_structure'].cell == structure.cell
    _assert_native_provenance(node, pseudos)
    summary_node = next(child for child in node.called_descendants if child.process_label == 'summarize_relax')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


def test_native_bands_seekpath_full_nesting(monkeypatch, aiida_code_installed) -> None:
    """Production bands runs SCF plus bands through the production base loop with SeekPath products."""
    _patch_nested(monkeypatch, pw_stub=FullPwCalc)
    pseudos = _pseudos('Si')
    outputs, node = run_get_node(
        pw_bands_graph,
        structure=_silicon(),
        scf_parameters=_scf_parameters(),
        bands_parameters=Dict({'CONTROL': {'calculation': 'bands'}, 'SYSTEM': {}}),
        scf_code=aiida_code_installed(label='test.native-bands-scf', default_calc_job_plugin='quantumespresso.pw'),
        bands_code=aiida_code_installed(label='test.native-bands-bands', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=pseudos,
        scf_kpoints_distance=Float(0.5),
        bands_kpoints_distance=Float(0.3),
        task_metadata={'pw_calc': {'options': {}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert isinstance(outputs['band_structure'], BandsData)
    assert isinstance(outputs['primitive_structure'], StructureData)
    assert isinstance(outputs['seekpath_parameters'], Dict)
    assert outputs['band_parameters'].get_dict()['number_of_bands'] == 8
    _assert_native_provenance(node, pseudos)
    pw_calcs = [
        child
        for child in node.called_descendants
        if isinstance(child, CalcJobNode) and child.process_label.endswith('PwCalc')
    ]
    assert len(pw_calcs) == 2
    by_time = sorted(pw_calcs, key=lambda child: child.ctime)
    assert by_time[1].inputs.parent_folder.uuid == by_time[0].outputs.remote_folder.uuid
    assert outputs['band_parameters'].uuid == by_time[1].outputs.output_parameters.uuid


def test_native_pdos_serial_full_nesting_and_order(monkeypatch, aiida_code_installed) -> None:
    """Production serial PDOS runs SCF/NSCF/DOS/PROJWFC with DOS cleanup ordered before PROJWFC."""
    _patch_nested(monkeypatch, pw_stub=FullPwCalc, dos_stub=StubDosCalc, projwfc_stub=StubProjwfcCalc)
    pseudos = _pseudos('Si')
    structure = _silicon()
    outputs, node = run_get_node(
        pdos_from_scratch_serial,
        structure=structure,
        pw_code=aiida_code_installed(label='test.native-pdos-pw', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=pseudos,
        scf_parameters=_scf_parameters(),
        nscf_parameters=_nscf_parameters(),
        dos_code=aiida_code_installed(label='test.native-pdos-dos', default_calc_job_plugin='quantumespresso.dos'),
        dos_parameters=_dos_parameters(),
        projwfc_code=aiida_code_installed(
            label='test.native-pdos-projwfc', default_calc_job_plugin='quantumespresso.projwfc'
        ),
        projwfc_parameters=_projwfc_parameters(),
        scf_kpoints_distance=Float(0.5),
        nscf_kpoints_distance=Float(0.5),
        clean_workdir=True,
        task_metadata={'pw_calc': {'options': {}}, 'dos_calc': {'options': {}}, 'projwfc_calc': {'options': {}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert outputs['report']['mode'] == 'serial'
    assert outputs['report']['status'] == 'completed'
    assert outputs['report']['emin'] == -5.0
    assert outputs['report']['emax'] == 8.0
    assert isinstance(outputs['dos_output_dos'], XyData)
    assert isinstance(outputs['projwfc_dos'], XyData)
    _assert_native_provenance(node, pseudos)
    descendants = node.called_descendants
    pw_calcs = [
        child for child in descendants if isinstance(child, CalcJobNode) and child.process_label.endswith('PwCalc')
    ]
    assert len(pw_calcs) == 2
    dos_nodes = [child for child in descendants if child.process_label == 'StubDosCalc']
    projwfc_nodes = [child for child in descendants if child.process_label == 'StubProjwfcCalc']
    assert len(dos_nodes) == 1
    assert len(projwfc_nodes) == 1
    sweeps = [child for child in descendants if child.process_label == 'sweep_remote']
    assert sweeps, 'the serial graph must sweep the DOS remote before PROJWFC submits'
    assert min(sweep.ctime for sweep in sweeps) > dos_nodes[0].ctime
    assert projwfc_nodes[0].ctime > min(sweep.ctime for sweep in sweeps)
    summary_node = next(child for child in descendants if child.process_label == 'summarize_pdos')
    assert outputs['report'].uuid == summary_node.outputs.result.uuid


def test_native_relax_output_identity(monkeypatch, aiida_code_installed) -> None:
    """The relax report is the stored summary result, not a fabricated mapping."""
    _patch_nested(monkeypatch, pw_stub=SuccessPwCalc)
    outputs, node = run_get_node(
        pw_relax_graph,
        structure=_silicon(),
        parameters=_vc_relax_parameters(),
        code=aiida_code_installed(label='test.native-relax-ident', default_calc_job_plugin='quantumespresso.pw'),
        pseudos=_pseudos('Si'),
        max_iterations=5,
        kpoints=_kpoints(),
        task_metadata={'pw_calc': {'options': {}}},
    )
    assert node.is_finished_ok, node.exit_message
    assert node.base.links.get_outgoing(link_type=LinkType.CREATE).all() == []
    assert outputs['output_structure'].is_stored
    assert outputs['report'].is_stored
