"""Graph alternatives for the Quantum ESPRESSO ``base`` restart work chains.

Note: this module deliberately avoids ``from __future__ import annotations``. Task
return annotations are resolved at runtime from the (possibly pickled) function, so
they must stay real objects, not strings.

This module re-expresses the six ``*BaseWorkChain`` restart loops (``pw``, ``ph``,
``neb``, ``q2r``, ``matdyn``, ``bands``) with native graph control flow. The original
work chains in :mod:`aiida_quantumespresso.workflows` are untouched; the graphs here
are new alternatives that callers can adopt per workflow.

Shape of each translation
-------------------------

Science lives in ``@task`` collaborators (restart mathematics ported from the
corresponding ``process_handler`` methods); the ``@graph`` only orchestrates. Counters
and retry predicates stay inline, exactly like the canonical recipe in
``tests/engine/processes/graphs/test_authoring_recipe.py``. A ``while`` loop retries a
fallible ``CalcJob`` placement through its declared ``errors`` mapping, a finalizer
task assembles the next-iteration inputs, an ``ensure_*`` task fails the graph loudly
when the budget runs out, and a ``summarize_restart`` task records attempts and status
so the root returns an existing node. Handler tasks run as a chain that forwards the
upstream restart decision: only the handler matching the failure mutates the plan,
the rest pass it through, so the final plan still carries a claimed restart into the
``require_*`` guard and the node rebinding. Dropping the forwarding would silently
clear every claimed restart.

Per-workflow coverage
---------------------

* ``pw_base_graph`` places ``PwCalculation`` with ``errors=PwErrors``. Ported
  handlers: out-of-walltime full restart, diagonalization cycling (``david``/``paro``/
  ``cg`` with the tried list carried in the loop), electronic-convergence
  ``mixing_beta`` reduction, the recoverable ionic family (output-structure restart,
  ``cell_factor`` doubling, BFGS ``trust_radius``/``damp`` branches, one-shot
  rattling), and the success-path insufficient-bands check. Deliberately not ported:
  ``STOPPED_BY_MONITOR`` (the original gates on the exit *message*, which the error
  arm does not carry), ``ERROR_IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF`` and
  ``WARNING_ELECTRONIC_CONVERGENCE_NOT_REACHED`` (the original finishes *with* an
  exit code while attaching outputs; mapped-error children expose no success outputs,
  so the graph fails loudly instead of silently succeeding), scheduler
  ``max_seconds`` capping (scheduler options arrive via ``task_metadata`` and are
  constant across iterations), and the disabled empty ``exit_codes=[]`` handler.
* ``ph_base_graph`` places ``PhCalculation`` with ``errors=PhErrors``. Ported:
  scheduler-walltime ``max_seconds`` halving (only when already present, see above),
  diagonalization switch to ``cg``, out-of-walltime recover restart with the
  recover latch, and the ``nmix_ph``/``alpha_mix`` convergence chain. Not ported:
  merging of multi-recover output parameters (the recover chain is found by walking
  provenance and per-iteration outputs cannot be accumulated into one container with
  the current container types); the graph returns the last iteration outputs.
  ``qpoints_distance`` takes an explicit ``structure`` input instead of deriving it
  from the parent folder provenance.
* ``neb_base_graph`` places a static-port wrapper work chain (see below) with
  ``errors=NebErrors``. Ported: diagonalization cycling on the nested ``pw``
  parameters, interrupted-trajectory full restart, ``nstep_path`` extension from the
  initial inputs, and electronic ``mixing_beta`` reduction. ``kpoints_distance``
  builds k-points from the last image structure.
* ``q2r_base_graph``, ``matdyn_base_graph``, ``bands_base_graph`` place their
  calculation once and return its outputs. This is faithful: the originals define no
  process handlers, so a failure aborts without retry.

Mapped failures hide the failed run's outputs
---------------------------------------------

A placement that finishes with a mapped error exposes only its ``error`` arm: success
outputs read as ``None`` downstream even though the failed calculation stored them.
The legacy handlers instead read ``calculation.outputs.remote_folder``,
``output_structure``, and ``output_parameters`` directly, so any graph restart that
needs those nodes cannot execute faithfully. The ``require_*_restart_nodes`` tasks
fail such runs loudly (naming the missing node) instead of retrying degraded. In
practice only parameter-only restarts (diagonalization cycling) and success-path
restarts (whose outputs are visible) recover; walltime, ionic, and NEB remote
restarts, and every ``ph`` recover restart, abort at the guard. The ``is not None``
rebinding checks stay as defense in depth behind the guards.

Static shapes
-------------

Graph ports are static, so the dynamic ``pseudos`` namespace is closed with an
explicit single-kind example shape (``{'Si': UpfData}``). Structures with other kind
names need a copy of the graph with the corresponding entries added to the ``task``
declaration; the handler tasks themselves are kind-agnostic and reusable. ``Code``
inputs are required; scheduler options for ``CalcJob`` placements are supplied at
launch under ``task_metadata`` keyed by placement name.

The ``neb`` wrapper
--------------------

``NebCalculation`` nests its dynamic ``pseudos`` namespace inside the required
dynamic ``pw`` namespace, and the graph layer cannot close a required nested dynamic
namespace through ``task(..., inputs=...)`` (the closed parent keeps its dynamic
flag and the placement is rejected). ``NebStaticWrapper`` therefore closes the
shape at runtime instead: it takes one pseudo per the fixed example kind and fills
``pw.pseudos`` when submitting the calculation. It runs a single calculation and
propagates its exit status, so the graph retry loop observes the same declared
failures as a direct placement; the extra work chain level per iteration is the
known cost of this adapter.
"""

import typing as t

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, graph, task
from aiida.engine.processes.exit_code import ExitCode, ExitCodeEnum
from aiida.plugins import DataFactory
from aiida_quantumespresso.calculations.bands import BandsCalculation
from aiida_quantumespresso.calculations.matdyn import MatdynCalculation
from aiida_quantumespresso.calculations.neb import NebCalculation
from aiida_quantumespresso.calculations.ph import PhCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.q2r import Q2rCalculation
from aiida_quantumespresso.data.force_constants import ForceConstantsData
from aiida_quantumespresso.utils.bands import get_highest_occupied_band

UpfData = DataFactory('pseudo.upf')

# Fallback values mirror ``PwBaseWorkChain.defaults`` and ``qe_defaults``. They are
# repeated here (instead of imported) because importing the work chain modules would
# require installed entry points.
_MIXING_BETA_DEFAULT = 0.7
_MIXING_BETA_FACTOR = 0.8
_NBND_DELTA_FACTOR = 0.05
_NBND_DELTA_MINIMUM = 4
_UPSCALE_DEFAULT = 100
_TRUST_RADIUS_MIN_DEFAULT = 0.001
_TRUST_RADIUS_FACTOR = 0.1
_RATTLE_STDEV = 0.01
_RATTLE_SYMMETRY_THRESHOLD = 5
_CELL_FACTOR_DEFAULT = 2


class PwErrors(ExitCodeEnum):
    """Handled ``pw.x`` failures that the graph restart loop may recover from."""

    OUT_OF_WALLTIME = ExitCode(400, 'The calculation stopped prematurely because it ran out of walltime.')
    ELECTRONIC_CONVERGENCE_NOT_REACHED = ExitCode(
        410, 'The electronic minimization cycle did not reach self-consistency.'
    )
    COMPUTING_CHOLESKY = ExitCode(462, 'The code failed during the cholesky factorization.')
    DIAGONALIZATION_TOO_MANY_BANDS_NOT_CONVERGED = ExitCode(
        463, 'Too many bands failed to converge during the diagonalization.'
    )
    S_MATRIX_NOT_POSITIVE_DEFINITE = ExitCode(464, 'The S matrix was found to be not positive definite.')
    ZHEGVD_FAILED = ExitCode(465, 'The `zhegvd` failed in the PPCG diagonalization.')
    QR_FAILED = ExitCode(466, 'The `[Q, R] = qr(X, 0)` failed in the PPCG diagonalization.')
    EIGENVECTOR_CONVERGENCE = ExitCode(467, 'The eigenvector convergence failed.')
    BROYDEN_FACTORIZATION = ExitCode(468, 'The Broyden factorization failed.')
    IONIC_CONVERGENCE_NOT_REACHED = ExitCode(500, 'The ionic minimization cycle did not converge.')
    IONIC_CYCLE_EXCEEDED_NSTEP = ExitCode(502, 'The ionic minimization cycle exceeded the maximum number of steps.')
    IONIC_INTERRUPTED_PARTIAL_TRAJECTORY = ExitCode(503, 'The ionic cycle was interrupted with a partial trajectory.')
    IONIC_CYCLE_ELECTRONIC_CONVERGENCE_NOT_REACHED = ExitCode(
        510, 'Electronic convergence failed during an ionic minimization cycle.'
    )
    IONIC_CONVERGENCE_REACHED_FINAL_SCF_FAILED = ExitCode(511, 'Ionic convergence reached but the final SCF failed.')
    IONIC_CYCLE_BFGS_HISTORY_FAILURE = ExitCode(520, 'The ionic minimization terminated on a BFGS history failure.')
    IONIC_CYCLE_BFGS_HISTORY_AND_FINAL_SCF_FAILURE = ExitCode(
        521, 'The ionic minimization terminated on a BFGS history and final SCF failure.'
    )
    RADIAL_FFT_SIGNIFICANT_VOLUME_CONTRACTION = ExitCode(
        542, 'The cell relaxation caused a significant volume contraction.'
    )


class PhErrors(ExitCodeEnum):
    """Handled ``ph.x`` failures that the graph restart loop may recover from."""

    SCHEDULER_OUT_OF_WALLTIME = ExitCode(120, 'The job ran out of walltime.')
    OUT_OF_WALLTIME = ExitCode(400, 'The calculation stopped prematurely because it ran out of walltime.')
    CONVERGENCE_NOT_REACHED = ExitCode(410, 'Self-consistency was not reached.')
    COMPUTING_CHOLESKY = ExitCode(462, 'The code failed during the cholesky factorization.')


class NebErrors(ExitCodeEnum):
    """Handled ``neb.x`` failures that the graph restart loop may recover from."""

    OUT_OF_WALLTIME = ExitCode(400, 'The calculation stopped prematurely because it ran out of walltime.')
    ELECTRONIC_CONVERGENCE_NOT_REACHED = ExitCode(
        410, 'The electronic minimization cycle did not reach self-consistency.'
    )
    COMPUTING_CHOLESKY = ExitCode(462, 'The code failed during the cholesky factorization.')
    DIAGONALIZATION_TOO_MANY_BANDS_NOT_CONVERGED = ExitCode(
        463, 'Too many bands failed to converge during the diagonalization.'
    )
    S_MATRIX_NOT_POSITIVE_DEFINITE = ExitCode(464, 'The S matrix was found to be not positive definite.')
    ZHEGVD_FAILED = ExitCode(465, 'The `zhegvd` failed in the PPCG diagonalization.')
    QR_FAILED = ExitCode(466, 'The `[Q, R] = qr(X, 0)` failed in the PPCG diagonalization.')
    EIGENVECTOR_CONVERGENCE = ExitCode(467, 'The eigenvector convergence failed.')
    BROYDEN_FACTORIZATION = ExitCode(468, 'The Broyden factorization failed.')
    NEB_CYCLE_EXCEEDED_NSTEP = ExitCode(502, 'The NEB cycle exceeded the maximum number of steps.')
    NEB_INTERRUPTED_PARTIAL_TRAJECTORY = ExitCode(503, 'The NEB cycle was interrupted with a partial trajectory.')


class PwCarry(t.TypedDict):
    """Loop-carried ``pw`` restart state: the next inputs plus node-selection flags."""

    parameters: dict
    tried: list
    restart: bool
    rattled: bool
    use_remote_parent: bool
    clear_parent: bool


class PwRestartPlan(t.TypedDict):
    """One handler-chain pass over the ``pw`` restart state, before node selection."""

    parameters: dict
    tried: list
    restart: bool
    parent_mode: str
    structure_from_output: bool
    rattle_now: bool


class PhCarry(t.TypedDict):
    """Loop-carried ``ph`` restart state."""

    parameters: dict
    restart: bool
    recover: bool
    use_remote_parent: bool
    clear_parent: bool


class PhRestartPlan(t.TypedDict):
    """One handler-chain pass over the ``ph`` restart state, before node selection."""

    parameters: dict
    restart: bool
    recover: bool
    parent_mode: str


class NebCarry(t.TypedDict):
    """Loop-carried ``neb`` restart state."""

    parameters: dict
    pw_parameters: dict
    tried: list
    restart: bool
    use_remote_parent: bool
    clear_parent: bool


class NebRestartPlan(t.TypedDict):
    """One handler-chain pass over the ``neb`` restart state, before node selection."""

    parameters: dict
    pw_parameters: dict
    tried: list
    restart: bool
    parent_mode: str


def _pw_parameters(parameters: dict) -> dict:
    """Copy ``pw`` parameters with the electronic namelists present for mutation."""
    copied = {namelist: dict(values) for namelist, values in parameters.items()}
    copied.setdefault('CONTROL', {})
    copied.setdefault('ELECTRONS', {})
    copied.setdefault('SYSTEM', {})
    return copied


def _set_charge_density_restart(parameters: dict) -> None:
    """Mark parameters for a charge-density restart, mirroring ``RestartType.FROM_CHARGE_DENSITY``."""
    parameters['CONTROL']['restart_mode'] = 'from_scratch'
    parameters['ELECTRONS']['startingpot'] = 'file'
    parameters['ELECTRONS'].pop('startingwfc', None)


def _initial_diagonalizations(current: object) -> list:
    """List the untried diagonalization algorithms, least preferred last for ``pop``."""
    return [value for value in ('cg', 'paro', 'david') if value != str(current).lower()]


def _pass_pw_plan(
    parameters: dict,
    tried: list,
    restart: bool = False,
    parent_mode: str = 'keep',
    structure_from_output: bool = False,
    rattle_now: bool = False,
) -> PwRestartPlan:
    """Start a ``pw`` handler pass that carries the upstream restart decision forward.

    Handlers run as a chain and only the one matching the failure mutates the plan;
    every other handler must forward the decision it received, otherwise the final
    plan would silently drop a claimed restart.
    """
    return {
        'parameters': parameters,
        'tried': tried,
        'restart': restart,
        'parent_mode': parent_mode,
        'structure_from_output': structure_from_output,
        'rattle_now': rattle_now,
    }


@task
def seed_diagonalizations(parameters: dict) -> list:
    """Seed the untried diagonalization algorithms from the initial parameters."""
    current = parameters.get('ELECTRONS', {}).get('diagonalization', 'david')
    return _initial_diagonalizations(current)


@task
def handle_pw_walltime(error: PwErrors | None, parameters: dict, tried: list) -> PwRestartPlan:
    """Claim a walltime shutdown for a full restart, passing anything else through.

    Mirrors ``PwBaseWorkChain.handle_out_of_walltime``: the output structure, when the
    calculation wrote one, becomes the next input structure.
    """
    plan = _pass_pw_plan(parameters, tried)
    if error is not PwErrors.OUT_OF_WALLTIME:
        return plan
    updated = _pw_parameters(parameters)
    updated['CONTROL']['restart_mode'] = 'restart'
    plan['parameters'] = updated
    plan['restart'] = True
    plan['parent_mode'] = 'remote'
    plan['structure_from_output'] = True
    return plan


_PW_DIAGONALIZATION_ERRORS = (
    PwErrors.COMPUTING_CHOLESKY,
    PwErrors.DIAGONALIZATION_TOO_MANY_BANDS_NOT_CONVERGED,
    PwErrors.S_MATRIX_NOT_POSITIVE_DEFINITE,
    PwErrors.ZHEGVD_FAILED,
    PwErrors.QR_FAILED,
    PwErrors.EIGENVECTOR_CONVERGENCE,
    PwErrors.BROYDEN_FACTORIZATION,
)

_NEB_DIAGONALIZATION_ERRORS = (
    NebErrors.COMPUTING_CHOLESKY,
    NebErrors.DIAGONALIZATION_TOO_MANY_BANDS_NOT_CONVERGED,
    NebErrors.S_MATRIX_NOT_POSITIVE_DEFINITE,
    NebErrors.ZHEGVD_FAILED,
    NebErrors.QR_FAILED,
    NebErrors.EIGENVECTOR_CONVERGENCE,
    NebErrors.BROYDEN_FACTORIZATION,
)


@task
def handle_pw_diagonalization(
    error: PwErrors | None,
    parameters: dict,
    tried: list,
    restart: bool = False,
    parent_mode: str = 'keep',
    structure_from_output: bool = False,
    rattle_now: bool = False,
) -> PwRestartPlan:
    """Cycle the diagonalization algorithm, passing anything else through.

    Mirrors ``PwBaseWorkChain.handle_diagonalization_errors``: conjugate gradient stays
    the last resort, and once every algorithm was tried the plan requests no restart so
    the loop exits and the guard fails loudly. The upstream restart decision threads
    through untouched when this error is not claimed here.
    """
    plan = _pass_pw_plan(parameters, tried, restart, parent_mode, structure_from_output, rattle_now)
    if error not in _PW_DIAGONALIZATION_ERRORS:
        return plan
    remaining = list(tried)
    if not remaining:
        return plan
    new = remaining.pop()
    updated = _pw_parameters(parameters)
    updated['ELECTRONS']['diagonalization'] = new
    plan['parameters'] = updated
    plan['tried'] = remaining
    plan['restart'] = True
    plan['parent_mode'] = 'keep'
    return plan


@task
def handle_pw_electronic_convergence(
    error: PwErrors | None,
    parameters: dict,
    tried: list,
    restart: bool = False,
    parent_mode: str = 'keep',
    structure_from_output: bool = False,
    rattle_now: bool = False,
) -> PwRestartPlan:
    """Reduce the charge mixing and fully restart, passing anything else through.

    Mirrors ``PwBaseWorkChain.handle_electronic_convergence_not_reached``. The upstream
    restart decision threads through untouched when this error is not claimed here.
    """
    plan = _pass_pw_plan(parameters, tried, restart, parent_mode, structure_from_output, rattle_now)
    if error is not PwErrors.ELECTRONIC_CONVERGENCE_NOT_REACHED:
        return plan
    updated = _pw_parameters(parameters)
    mixing_beta = updated['ELECTRONS'].get('mixing_beta', _MIXING_BETA_DEFAULT)
    updated['ELECTRONS']['mixing_beta'] = mixing_beta * _MIXING_BETA_FACTOR
    updated['CONTROL']['restart_mode'] = 'restart'
    plan['parameters'] = updated
    plan['restart'] = True
    plan['parent_mode'] = 'remote'
    return plan


_PW_BFGS_ERRORS = (
    PwErrors.IONIC_CYCLE_BFGS_HISTORY_FAILURE,
    PwErrors.IONIC_CYCLE_BFGS_HISTORY_AND_FINAL_SCF_FAILURE,
)

_PW_IONIC_RECOVERABLE_ERRORS = (
    PwErrors.IONIC_CONVERGENCE_NOT_REACHED,
    PwErrors.IONIC_CYCLE_EXCEEDED_NSTEP,
    PwErrors.IONIC_CYCLE_BFGS_HISTORY_FAILURE,
    PwErrors.IONIC_CYCLE_BFGS_HISTORY_AND_FINAL_SCF_FAILURE,
)


@task
def handle_pw_ionic_convergence(
    error: PwErrors | None,
    parameters: dict,
    tried: list,
    rattled: bool,
    output_parameters: orm.Dict | None,
    initial_parameters: dict,
    restart: bool = False,
    parent_mode: str = 'keep',
    structure_from_output: bool = False,
    rattle_now: bool = False,
) -> PwRestartPlan:
    """Restart ionic optimizations from the output structure, passing anything else through.

    Mirrors ``PwBaseWorkChain.handle_relax_recoverable_ionic_convergence_error``,
    ``handle_vcrelax_recoverable_fft_significant_volume_contraction_error``,
    ``handle_relax_recoverable_electronic_convergence_error``,
    ``handle_ionic_interrupted_partial_trajectory`` and the BFGS-history handler. The
    BFGS branch applies to ``relax``/``vc-relax`` calculations; other calculation types
    with a BFGS failure fall through to the generic ionic restart, as in the original
    where the more specific handler returns ``False``. The upstream restart decision
    threads through untouched when this error is not claimed here.
    """
    plan = _pass_pw_plan(parameters, tried, restart, parent_mode, structure_from_output, rattle_now)
    if error is None:
        return plan
    calculation = parameters.get('CONTROL', {}).get('calculation', 'scf')
    updated = _pw_parameters(parameters)

    if error is PwErrors.RADIAL_FFT_SIGNIFICANT_VOLUME_CONTRACTION:
        cell = updated.setdefault('CELL', {})
        cell['cell_factor'] = _CELL_FACTOR_DEFAULT * cell.get('cell_factor', _CELL_FACTOR_DEFAULT)
        plan['parameters'] = updated
        plan['restart'] = True
        plan['parent_mode'] = 'none'
        plan['structure_from_output'] = True
        return plan

    if error in _PW_BFGS_ERRORS and calculation in ('relax', 'vc-relax'):
        ions = updated.setdefault('IONS', {})
        if calculation == 'relax':
            symmetries = None
            if output_parameters is not None:
                symmetries = output_parameters.get_dict().get('number_of_symmetries')
            if symmetries is not None and symmetries < _RATTLE_SYMMETRY_THRESHOLD:
                if rattled:
                    return plan
                ions['upscale'] = initial_parameters.get('IONS', {}).get('upscale', _UPSCALE_DEFAULT)
                _set_charge_density_restart(updated)
                plan['parameters'] = updated
                plan['restart'] = True
                plan['parent_mode'] = 'remote'
                plan['structure_from_output'] = True
                plan['rattle_now'] = True
                return plan
            ions['ion_dynamics'] = 'damp'
        elif calculation == 'vc-relax':
            trust_radius_min = ions.get('trust_radius_min', _TRUST_RADIUS_MIN_DEFAULT)
            if trust_radius_min > 1.0e-4:
                ions['trust_radius_ini'] = trust_radius_min
                ions['trust_radius_min'] = trust_radius_min * _TRUST_RADIUS_FACTOR
            else:
                ions['ion_dynamics'] = 'damp'
                updated.setdefault('CELL', {})['cell_dynamics'] = 'damp-w'
        _set_charge_density_restart(updated)
        plan['parameters'] = updated
        plan['restart'] = True
        plan['parent_mode'] = 'remote'
        plan['structure_from_output'] = True
        return plan

    if error in _PW_IONIC_RECOVERABLE_ERRORS:
        _set_charge_density_restart(updated)
        plan['parameters'] = updated
        plan['restart'] = True
        plan['parent_mode'] = 'remote'
        plan['structure_from_output'] = True
        return plan

    if error in (
        PwErrors.IONIC_INTERRUPTED_PARTIAL_TRAJECTORY,
        PwErrors.IONIC_CYCLE_ELECTRONIC_CONVERGENCE_NOT_REACHED,
        PwErrors.IONIC_CONVERGENCE_REACHED_FINAL_SCF_FAILED,
    ):
        if error is not PwErrors.IONIC_INTERRUPTED_PARTIAL_TRAJECTORY:
            mixing_beta = updated['ELECTRONS'].get('mixing_beta', _MIXING_BETA_DEFAULT)
            updated['ELECTRONS']['mixing_beta'] = mixing_beta * _MIXING_BETA_FACTOR
        plan['parameters'] = updated
        plan['restart'] = True
        plan['parent_mode'] = 'none'
        plan['structure_from_output'] = True
        return plan

    return plan


@task
def check_pw_sufficient_bands(
    error: PwErrors | None,
    parameters: dict,
    tried: list,
    output_band: orm.BandsData | None,
    output_parameters: orm.Dict | None,
    restart: bool = False,
    parent_mode: str = 'keep',
    structure_from_output: bool = False,
    rattle_now: bool = False,
) -> PwRestartPlan:
    """Grow ``nbnd`` and restart from the charge density when the top band is occupied.

    Mirrors ``PwBaseWorkChain.sanity_check_insufficient_bands``: only successful runs
    with smearing (no explicit ``fixed`` occupations) and a parsed band structure are
    inspected. Runs on the failure path, where ``error`` is set, forward the upstream
    restart decision untouched.
    """
    plan = _pass_pw_plan(parameters, tried, restart, parent_mode, structure_from_output, rattle_now)
    if error is not None:
        return plan
    if parameters.get('SYSTEM', {}).get('occupations') == 'fixed':
        return plan
    if output_band is None or output_parameters is None:
        return plan
    try:
        get_highest_occupied_band(output_band)
    except ValueError:
        updated = _pw_parameters(parameters)
        nbnd = output_parameters.get_dict().get('number_of_bands', 0)
        updated['SYSTEM']['nbnd'] = nbnd + max(int(nbnd * _NBND_DELTA_FACTOR), _NBND_DELTA_MINIMUM)
        _set_charge_density_restart(updated)
        plan['parameters'] = updated
        plan['restart'] = True
        plan['parent_mode'] = 'remote'
        return plan
    return plan


@task
def finalize_pw_carry(
    parameters: dict,
    tried: list,
    restart: bool,
    parent_mode: str,
    rattle_now: bool,
    rattled_before: bool,
) -> PwCarry:
    """Carry the next ``pw`` inputs; node selection stays in the graph body.

    Returning input nodes from a task would attach ``CREATE`` links to ancestors,
    which the provenance cycle check rejects. The graph body therefore rebinds the
    structure and parent folder with branches over these plain-data flags.
    """
    if parent_mode == 'remote':
        use_remote_parent = True
        clear_parent = False
    elif parent_mode == 'none':
        use_remote_parent = False
        clear_parent = True
    else:
        use_remote_parent = False
        clear_parent = False
    return {
        'parameters': parameters,
        'tried': tried,
        'restart': restart,
        'rattled': rattled_before or rattle_now,
        'use_remote_parent': use_remote_parent,
        'clear_parent': clear_parent,
    }


@task
def ensure_pw_recovered(error: PwErrors | None, restart: bool) -> None:
    """Fail the graph when the ``pw`` loop exits without a recovered run."""
    if error is not None or restart:
        raise ValueError('the pw restart loop never recovered.')


@task
def require_pw_restart_nodes(
    restart: bool,
    parent_mode: str,
    structure_from_output: bool,
    output_structure: orm.StructureData | None,
    remote_folder: orm.RemoteData | None,
) -> None:
    """Fail loudly when a claimed ``pw`` restart needs nodes a mapped failure hides.

    A placement that finishes with a mapped error exposes only its ``error`` arm: the
    success arms read as ``None`` downstream even though the failed calculation stored
    them. The legacy handlers read ``calculation.outputs.remote_folder`` and
    ``calculation.outputs.output_structure`` directly, so without this guard every
    restart needing the parent folder or the output structure would retry degraded:
    the ``is not None`` rebinding checks would silently keep the old structure and
    drop the parent while the carried parameters still claimed a restart. Fail loudly
    instead, naming the missing node. Parameter-only restarts (diagonalization
    cycling) and success-path restarts (whose outputs are visible) pass through
    untouched.
    """
    if not restart:
        return
    if structure_from_output and output_structure is None:
        msg = (
            'the pw restart handler claimed a restart from the output structure, but the failed '
            'calculation exposes no `output_structure` on its mapped error arm; retrying from the '
            'input structure would pretend recovery.'
        )
        raise ValueError(msg)
    if parent_mode == 'remote' and remote_folder is None:
        msg = (
            'the pw restart handler claimed a restart from the parent folder, but the failed '
            'calculation exposes no `remote_folder` on its mapped error arm; retrying without the '
            'parent would pretend recovery.'
        )
        raise ValueError(msg)
    return


def _pass_ph_plan(
    parameters: dict,
    restart: bool = False,
    recover: bool = False,
    parent_mode: str = 'keep',
) -> PhRestartPlan:
    """Start a ``ph`` handler pass that carries the upstream restart decision forward.

    Handlers run as a chain and only the one matching the failure mutates the plan;
    every other handler must forward the decision it received, otherwise the final
    plan would silently drop a claimed restart.
    """
    return {
        'parameters': parameters,
        'restart': restart,
        'recover': recover,
        'parent_mode': parent_mode,
    }


@task
def seed_ph_parameters(parameters: dict) -> dict:
    """Initialize ``ph`` parameters, recovering from the parent folder like the original setup."""
    prepared = dict(parameters)
    inputph = dict(prepared.get('INPUTPH', {}))
    inputph['recover'] = True
    prepared['INPUTPH'] = inputph
    return prepared


@task
def handle_ph_scheduler_walltime(error: PhErrors | None, parameters: dict) -> PhRestartPlan:
    """Halve ``max_seconds`` and recover, passing anything else through.

    Mirrors ``PhBaseWorkChain.handle_scheduler_out_of_walltime`` except that the
    scheduler walltime itself is not visible to the task: when ``max_seconds`` is absent
    the restart still recovers from the remote folder. See the module docstring.
    """
    plan = _pass_ph_plan(parameters)
    if error is not PhErrors.SCHEDULER_OUT_OF_WALLTIME:
        return plan
    updated = {block: dict(values) for block, values in parameters.items()}
    inputph = updated.setdefault('INPUTPH', {})
    if 'max_seconds' in inputph:
        inputph['max_seconds'] = inputph['max_seconds'] * 0.5
    inputph['recover'] = False
    plan['parameters'] = updated
    plan['restart'] = True
    plan['recover'] = True
    plan['parent_mode'] = 'remote'
    return plan


@task
def handle_ph_diagonalization(
    error: PhErrors | None,
    parameters: dict,
    restart: bool = False,
    recover: bool = False,
    parent_mode: str = 'keep',
) -> PhRestartPlan:
    """Switch to conjugate-gradient diagonalization, passing anything else through.

    Mirrors ``PhBaseWorkChain.handle_diagonalization_errors``: a second consecutive
    failure requests no restart so the guard fails loudly. The upstream restart
    decision threads through untouched when this error is not claimed here.
    """
    plan = _pass_ph_plan(parameters, restart, recover, parent_mode)
    if error is not PhErrors.COMPUTING_CHOLESKY:
        return plan
    if parameters.get('INPUTPH', {}).get('diagonalization') == 'cg':
        return plan
    updated = {block: dict(values) for block, values in parameters.items()}
    updated.setdefault('INPUTPH', {})['diagonalization'] = 'cg'
    plan['parameters'] = updated
    plan['restart'] = True
    plan['recover'] = False
    plan['parent_mode'] = 'keep'
    return plan


@task
def handle_ph_out_of_walltime(
    error: PhErrors | None,
    parameters: dict,
    restart: bool = False,
    recover: bool = False,
    parent_mode: str = 'keep',
) -> PhRestartPlan:
    """Recover from the remote folder, passing anything else through.

    Mirrors ``PhBaseWorkChain.handle_out_of_walltime``. The upstream restart decision
    threads through untouched when this error is not claimed here.
    """
    plan = _pass_ph_plan(parameters, restart, recover, parent_mode)
    if error is not PhErrors.OUT_OF_WALLTIME:
        return plan
    plan['restart'] = True
    plan['recover'] = True
    plan['parent_mode'] = 'remote'
    return plan


@task
def handle_ph_convergence(
    error: PhErrors | None,
    parameters: dict,
    restart: bool = False,
    recover: bool = False,
    parent_mode: str = 'keep',
) -> PhRestartPlan:
    """Tune the mixing and recover, passing anything else through.

    Mirrors ``PhBaseWorkChain.handle_convergence_not_reached``: first raise ``nmix_ph``
    to 8, then halve ``alpha_mix(1)``, finally introduce ``alpha_mix(20)``. Exhausting
    the strategies requests no restart so the guard fails loudly. The upstream restart
    decision threads through untouched when this error is not claimed here.
    """
    plan = _pass_ph_plan(parameters, restart, recover, parent_mode)
    if error is not PhErrors.CONVERGENCE_NOT_REACHED:
        return plan
    updated = {block: dict(values) for block, values in parameters.items()}
    inputph = updated.setdefault('INPUTPH', {})
    nmix_ph = inputph.get('nmix_ph', 4)
    if nmix_ph < 8:
        inputph['nmix_ph'] = 8
    else:
        if inputph.get('alpha_mix(20)') is not None:
            return plan
        alpha_mix = inputph.get('alpha_mix(1)', 0.7)
        alpha_mix_new = alpha_mix * 0.5
        if alpha_mix_new >= 0.1:
            inputph['alpha_mix(1)'] = alpha_mix_new
        else:
            inputph['alpha_mix(20)'] = 0.4
    plan['parameters'] = updated
    plan['restart'] = True
    plan['recover'] = True
    plan['parent_mode'] = 'remote'
    return plan


@task
def prepare_ph_settings(settings: dict | None, only_initialization: bool) -> dict | None:
    """Fold ``ONLY_INITIALIZATION`` into the settings, mirroring ``validate_parameters``."""
    if not only_initialization:
        return settings
    merged = dict(settings) if settings is not None else {}
    merged['ONLY_INITIALIZATION'] = True
    return merged


@task
def finalize_ph_carry(
    parameters: dict,
    restart: bool,
    recover: bool,
    parent_mode: str,
    recover_before: bool,
) -> PhCarry:
    """Carry the next ``ph`` inputs; the recover flag latches once set.

    The latch mirrors ``restart_calc`` in the original, which is never cleared, so every
    later restart recovers from the latest remote folder. Node selection stays in the
    graph body (see :func:`finalize_pw_carry`).
    """
    if parent_mode == 'remote':
        use_remote_parent = True
        clear_parent = False
    elif parent_mode == 'none':
        use_remote_parent = False
        clear_parent = True
    else:
        use_remote_parent = False
        clear_parent = False
    return {
        'parameters': parameters,
        'restart': restart,
        'recover': recover_before or recover,
        'use_remote_parent': use_remote_parent,
        'clear_parent': clear_parent,
    }


@task
def ensure_ph_recovered(error: PhErrors | None, restart: bool) -> None:
    """Fail the graph when the ``ph`` loop exits without a recovered run."""
    if error is not None or restart:
        raise ValueError('the ph restart loop never recovered.')


@task
def require_ph_restart_nodes(
    restart: bool,
    parent_mode: str,
    remote_folder: orm.RemoteData | None,
) -> None:
    """Fail loudly when a claimed ``ph`` restart needs a parent a mapped failure hides.

    A placement that finishes with a mapped error exposes only its ``error`` arm, so
    ``remote_folder`` reads as ``None`` downstream even though the failed calculation
    stored it. The legacy handlers restart from ``restart_calc.outputs.remote_folder``
    directly, so without this guard a claimed recover restart would silently reuse a
    stale parent folder with ``recover`` latched on. Fail loudly instead, naming the
    missing node. Restarts that keep the wired parent pass through untouched.
    """
    if not restart:
        return
    if parent_mode == 'remote' and remote_folder is None:
        msg = (
            'the ph restart handler claimed a recover restart from the parent folder, but the '
            'failed calculation exposes no `remote_folder` on its mapped error arm; retrying '
            'against a stale parent would pretend recovery.'
        )
        raise ValueError(msg)
    return


def _pass_neb_plan(
    parameters: dict,
    pw_parameters: dict,
    tried: list,
    restart: bool = False,
    parent_mode: str = 'keep',
) -> NebRestartPlan:
    """Start a ``neb`` handler pass that carries the upstream restart decision forward.

    Handlers run as a chain and only the one matching the failure mutates the plan;
    every other handler must forward the decision it received, otherwise the final
    plan would silently drop a claimed restart.
    """
    return {
        'parameters': parameters,
        'pw_parameters': pw_parameters,
        'tried': tried,
        'restart': restart,
        'parent_mode': parent_mode,
    }


@task
def handle_neb_diagonalization(
    error: NebErrors | None, parameters: dict, pw_parameters: dict, tried: list
) -> NebRestartPlan:
    """Cycle the nested ``pw`` diagonalization algorithm, passing anything else through.

    Mirrors ``NebBaseWorkChain.handle_diagonalization_errors``.
    """
    plan = _pass_neb_plan(parameters, pw_parameters, tried)
    if error not in _NEB_DIAGONALIZATION_ERRORS:
        return plan
    remaining = list(tried)
    if not remaining:
        return plan
    new = remaining.pop()
    updated = _pw_parameters(pw_parameters)
    updated['ELECTRONS']['diagonalization'] = new
    plan['pw_parameters'] = updated
    plan['tried'] = remaining
    plan['restart'] = True
    plan['parent_mode'] = 'keep'
    return plan


@task
def handle_neb_interrupted(
    error: NebErrors | None,
    parameters: dict,
    pw_parameters: dict,
    tried: list,
    restart: bool = False,
    parent_mode: str = 'keep',
) -> NebRestartPlan:
    """Fully restart an interrupted trajectory, passing anything else through.

    Mirrors ``NebBaseWorkChain.handle_neb_interrupted_partial_trajectory``. The upstream
    restart decision threads through untouched when this error is not claimed here.
    """
    plan = _pass_neb_plan(parameters, pw_parameters, tried, restart, parent_mode)
    if error is not NebErrors.NEB_INTERRUPTED_PARTIAL_TRAJECTORY:
        return plan
    updated = dict(parameters)
    path = dict(updated.get('PATH', {}))
    path['restart_mode'] = 'restart'
    updated['PATH'] = path
    plan['parameters'] = updated
    plan['restart'] = True
    plan['parent_mode'] = 'remote'
    return plan


@task
def handle_neb_nstep_exceeded(
    error: NebErrors | None,
    parameters: dict,
    pw_parameters: dict,
    tried: list,
    initial_parameters: dict,
    restart: bool = False,
    parent_mode: str = 'keep',
) -> NebRestartPlan:
    """Extend ``nstep_path`` and fully restart, passing anything else through.

    Mirrors ``NebBaseWorkChain.handle_neb_cycle_exceeded_nstep``: the extension is the
    input ``nstep_path`` (defaulting to 1), read from the aliased initial parameters.
    The upstream restart decision threads through untouched when this error is not
    claimed here.
    """
    plan = _pass_neb_plan(parameters, pw_parameters, tried, restart, parent_mode)
    if error is not NebErrors.NEB_CYCLE_EXCEEDED_NSTEP:
        return plan
    updated = dict(parameters)
    path = dict(updated.get('PATH', {}))
    initial_nsteps = initial_parameters.get('PATH', {}).get('nstep_path', 1)
    path['nstep_path'] = path.get('nstep_path', 1) + initial_nsteps
    path['restart_mode'] = 'restart'
    updated['PATH'] = path
    plan['parameters'] = updated
    plan['restart'] = True
    plan['parent_mode'] = 'remote'
    return plan


@task
def handle_neb_electronic_convergence(
    error: NebErrors | None,
    parameters: dict,
    pw_parameters: dict,
    tried: list,
    restart: bool = False,
    parent_mode: str = 'keep',
) -> NebRestartPlan:
    """Reduce the nested ``pw`` mixing and fully restart, passing anything else through.

    Mirrors ``NebBaseWorkChain.handle_electronic_convergence_not_reached``. The upstream
    restart decision threads through untouched when this error is not claimed here.
    """
    plan = _pass_neb_plan(parameters, pw_parameters, tried, restart, parent_mode)
    if error is not NebErrors.ELECTRONIC_CONVERGENCE_NOT_REACHED:
        return plan
    updated_pw = _pw_parameters(pw_parameters)
    mixing_beta = updated_pw['ELECTRONS'].get('mixing_beta', _MIXING_BETA_DEFAULT)
    updated_pw['ELECTRONS']['mixing_beta'] = mixing_beta * _MIXING_BETA_FACTOR
    updated = dict(parameters)
    path = dict(updated.get('PATH', {}))
    path['restart_mode'] = 'restart'
    updated['PATH'] = path
    plan['parameters'] = updated
    plan['pw_parameters'] = updated_pw
    plan['restart'] = True
    plan['parent_mode'] = 'remote'
    return plan


@task
def finalize_neb_carry(
    parameters: dict,
    pw_parameters: dict,
    tried: list,
    restart: bool,
    parent_mode: str,
) -> NebCarry:
    """Carry the next ``neb`` inputs; node selection stays in the graph body."""
    if parent_mode == 'remote':
        use_remote_parent = True
        clear_parent = False
    elif parent_mode == 'none':
        use_remote_parent = False
        clear_parent = True
    else:
        use_remote_parent = False
        clear_parent = False
    return {
        'parameters': parameters,
        'pw_parameters': pw_parameters,
        'tried': tried,
        'restart': restart,
        'use_remote_parent': use_remote_parent,
        'clear_parent': clear_parent,
    }


@task
def ensure_neb_recovered(error: NebErrors | None, restart: bool) -> None:
    """Fail the graph when the ``neb`` loop exits without a recovered run."""
    if error is not None or restart:
        raise ValueError('the neb restart loop never recovered.')


@task
def require_neb_restart_nodes(
    restart: bool,
    parent_mode: str,
    remote_folder: orm.RemoteData | None,
) -> None:
    """Fail loudly when a claimed ``neb`` restart needs a parent a mapped failure hides.

    A placement that finishes with a mapped error exposes only its ``error`` arm, so
    ``remote_folder`` reads as ``None`` downstream even though the failed calculation
    stored it. The legacy handlers fully restart from ``calculation.outputs.remote_folder``
    directly, so without this guard a claimed restart would silently drop the parent
    while the carried parameters still claimed one. Fail loudly instead, naming the
    missing node. Parameter-only restarts (diagonalization cycling) pass through
    untouched.
    """
    if not restart:
        return
    if parent_mode == 'remote' and remote_folder is None:
        msg = (
            'the neb restart handler claimed a restart from the parent folder, but the failed '
            'calculation exposes no `remote_folder` on its mapped error arm; retrying without '
            'the parent would pretend recovery.'
        )
        raise ValueError(msg)
    return


@task
def summarize_restart(attempts: int, status: str) -> dict:
    """Record the attempts taken and the final status as one report."""
    return {'attempts': attempts, 'status': status}


@task
def final_neb_image_structure(images: orm.TrajectoryData) -> orm.StructureData:
    """Return the last image structure that the k-point mesh is built from."""
    return images.get_step_structure(-1)


class NebStaticWrapper(WorkChain):
    """Single-shot static-port wrapper around ``NebCalculation`` for graph placement.

    The graph layer cannot close the required nested-dynamic ``pw`` namespace of
    ``NebCalculation`` through ``task(..., inputs=...)``. This wrapper takes one
    pseudo per the fixed example kind and fills ``pw.pseudos`` at runtime, where
    dynamic namespaces are unrestricted. It submits exactly one calculation and
    propagates its exit status, so the graph retry loop observes the same declared
    failures as a direct placement, at the cost of one work chain level per attempt.
    """

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        super().define(spec)
        spec.input('code', valid_type=orm.Code)
        spec.input('images', valid_type=orm.TrajectoryData)
        spec.input('pseudo', valid_type=UpfData)
        spec.input('parameters', valid_type=orm.Dict)
        spec.input('pw_parameters', valid_type=orm.Dict)
        spec.input('kpoints', valid_type=orm.KpointsData)
        spec.input('settings', valid_type=orm.Dict, required=False)
        spec.input('parent_folder', valid_type=orm.RemoteData, required=False)
        spec.output('remote_folder', valid_type=orm.RemoteData, required=False)
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output('output_trajectory', valid_type=orm.TrajectoryData)
        spec.output('output_mep', valid_type=orm.ArrayData)
        spec.output('iteration_array', valid_type=orm.ArrayData, required=False)
        for member in NebErrors:
            spec.exit_code(member.value.status, member.name, message=member.value.message)
        spec.outline(cls.run_neb, cls.results)

    def run_neb(self):
        """Submit one ``NebCalculation`` with the static inputs wired into ``pw``."""
        inputs = AttributeDict(
            {
                'code': self.inputs.code,
                'images': self.inputs.images,
                'parameters': self.inputs.parameters,
                'pw': {
                    'parameters': self.inputs.pw_parameters,
                    'pseudos': {'Si': self.inputs.pseudo},
                    'kpoints': self.inputs.kpoints,
                },
            }
        )
        if 'settings' in self.inputs:
            inputs.settings = self.inputs.settings
        if 'parent_folder' in self.inputs:
            inputs.parent_folder = self.inputs.parent_folder
        node = self.submit(NebCalculation, **inputs)
        return ToContext(neb=node)

    def results(self):
        """Attach the calculation outputs and propagate its exit status."""
        node = self.ctx.neb
        for name in ('remote_folder', 'output_parameters', 'output_trajectory', 'output_mep', 'iteration_array'):
            if name in node.outputs:
                self.out(name, node.outputs[name])
        if node.is_finished_ok:
            return None
        msg = node.exit_message or f'neb failed with status {node.exit_status}'
        return ExitCode(node.exit_status, msg)


@task
def make_kpoints_mesh(
    structure: orm.StructureData | None, distance: float | None, force_parity: bool
) -> orm.KpointsData:
    """Generate a uniformly spaced k-point mesh with a guaranteed minimum distance.

    This mirrors ``create_kpoints_from_distance`` as an owned task: reusing the
    ``@calcfunction`` directly is not possible because ``task`` takes undecorated
    functions. The stored node carries this task label instead. Missing inputs fail
    fast here, naming the offending value, so no separate validation task is needed.
    """
    from numpy import linalg

    from aiida.orm import KpointsData

    if distance is None:
        msg = 'neither `kpoints` nor `kpoints_distance` was provided.'
        raise ValueError(msg)
    if structure is None:
        msg = '`kpoints_distance` needs an explicit `structure` input.'
        raise ValueError(msg)

    epsilon = 1e-5

    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(distance, force_parity=force_parity)

    lengths_vector = [linalg.norm(vector) for vector in structure.cell]
    lengths_kpoint = kpoints.get_kpoints_mesh()[0]

    is_symmetric_cell = all(abs(length - lengths_vector[0]) < epsilon for length in lengths_vector)
    is_symmetric_mesh = all(length == lengths_kpoint[0] for length in lengths_kpoint)

    # If the vectors of the cell all have the same length, the kpoint mesh should be isotropic as well
    if is_symmetric_cell and not is_symmetric_mesh:
        nkpoints = max(lengths_kpoint)
        # In case the cell is not periodic in a direction, set to 1
        kpoints.set_kpoints_mesh([nkpoints if pbc else 1 for pbc in structure.pbc])

    return kpoints


@task
def rattle_positions(structure: orm.StructureData, stdev: float) -> orm.StructureData:
    """Return a clone of the structure with normally distributed displacements.

    This mirrors ``rattle_structure`` (same fixed seed) as an owned task for the
    reason documented on :func:`make_kpoints_mesh`.
    """
    import numpy as np

    rattled = structure.clone()
    positions = np.array([site.position for site in structure.sites])
    rng = np.random.RandomState(seed=42)  # same default seed as ``ase.Atoms.rattle``
    rattled.reset_sites_positions((positions + rng.normal(scale=stdev, size=positions.shape)).tolist())
    return rattled


pw_calc = task(PwCalculation, inputs={'pseudos': {'Si': UpfData}}, errors=PwErrors)
ph_calc = task(PhCalculation, errors=PhErrors)
neb_calc = task(NebStaticWrapper, errors=NebErrors)
q2r_calc = task(Q2rCalculation)
matdyn_calc = task(MatdynCalculation)
bands_calc = task(BandsCalculation)
gen_kpoints = make_kpoints_mesh
rattle_task = rattle_positions


@graph
def pw_base_graph(
    code: orm.Code,
    structure: orm.StructureData,
    pseudo: UpfData,
    parameters: dict,
    settings: dict | None = None,
    kpoints: orm.KpointsData | None = None,
    kpoints_distance: float | None = None,
    kpoints_force_parity: bool = False,
    max_iterations: int = 5,
) -> dict:
    """Run ``pw.x`` with automated restarts, returning the outputs plus a report.

    Restarts that need the failed run's outputs (walltime, ionic, electronic full
    restarts) fail loudly at ``require_pw_restart_nodes``: mapped error arms hide the
    failed calculation's ``remote_folder``/``output_structure``. Only parameter-only
    (diagonalization) and success-path (insufficient bands) restarts recover.

    The ``pseudos`` shape is fixed to one ``Si`` kind; add entries to the ``pw_calc``
    declaration for structures with other kind names. Scheduler options travel in
    ``task_metadata`` keyed by placement name at launch.
    """
    if kpoints is None:
        resolved_kpoints = gen_kpoints(
            structure=structure, distance=kpoints_distance, force_parity=kpoints_force_parity
        )
    else:
        resolved_kpoints = kpoints
    tried = seed_diagonalizations(parameters=parameters)
    run = pw_calc(
        code=code,
        structure=structure,
        pseudos={'Si': pseudo},
        parameters=parameters,
        settings=settings,
        kpoints=resolved_kpoints,
    )
    plan_walltime = handle_pw_walltime(error=run.error, parameters=parameters, tried=tried)
    plan_diagonalization = handle_pw_diagonalization(
        error=run.error,
        parameters=plan_walltime.parameters,
        tried=plan_walltime.tried,
        restart=plan_walltime.restart,
        parent_mode=plan_walltime.parent_mode,
        structure_from_output=plan_walltime.structure_from_output,
        rattle_now=plan_walltime.rattle_now,
    )
    plan_electronic = handle_pw_electronic_convergence(
        error=run.error,
        parameters=plan_diagonalization.parameters,
        tried=plan_diagonalization.tried,
        restart=plan_diagonalization.restart,
        parent_mode=plan_diagonalization.parent_mode,
        structure_from_output=plan_diagonalization.structure_from_output,
        rattle_now=plan_diagonalization.rattle_now,
    )
    plan_ionic = handle_pw_ionic_convergence(
        error=run.error,
        parameters=plan_electronic.parameters,
        tried=plan_electronic.tried,
        rattled=False,
        output_parameters=run.output_parameters,
        initial_parameters=parameters,
        restart=plan_electronic.restart,
        parent_mode=plan_electronic.parent_mode,
        structure_from_output=plan_electronic.structure_from_output,
        rattle_now=plan_electronic.rattle_now,
    )
    plan_bands = check_pw_sufficient_bands(
        error=run.error,
        parameters=plan_ionic.parameters,
        tried=plan_ionic.tried,
        output_band=run.output_band,
        output_parameters=run.output_parameters,
        restart=plan_ionic.restart,
        parent_mode=plan_ionic.parent_mode,
        structure_from_output=plan_ionic.structure_from_output,
        rattle_now=plan_ionic.rattle_now,
    )
    # Literal binding: graph bodies cannot read module constants, so the shared
    # ``_RATTLE_STDEV`` value is spelled out once here.
    stdev = 0.01
    parent = None
    if plan_ionic.rattle_now and run.output_structure is not None:
        rattled = rattle_task(structure=run.output_structure, stdev=stdev)
    else:
        rattled = structure
    guard = require_pw_restart_nodes(
        restart=plan_bands.restart,
        parent_mode=plan_bands.parent_mode,
        structure_from_output=plan_bands.structure_from_output,
        output_structure=run.output_structure,
        remote_folder=run.remote_folder,
    )
    final = finalize_pw_carry(
        parameters=plan_bands.parameters,
        tried=plan_bands.tried,
        restart=plan_bands.restart,
        parent_mode=plan_bands.parent_mode,
        rattle_now=plan_bands.rattle_now,
        rattled_before=False,
        after=guard,
    )
    if plan_bands.structure_from_output and run.output_structure is not None:
        structure = run.output_structure
    if plan_bands.rattle_now:
        structure = rattled
    if final.use_remote_parent and run.remote_folder is not None:
        parent = run.remote_folder
    if final.clear_parent:
        parent = None
    restart = plan_bands.restart
    attempts = 1
    while (run.error is not None or restart) and attempts < max_iterations:
        attempts = attempts + 1
        run = pw_calc(
            code=code,
            structure=structure,
            pseudos={'Si': pseudo},
            parameters=final.parameters,
            settings=settings,
            kpoints=resolved_kpoints,
            parent_folder=parent,
        )
        plan_walltime = handle_pw_walltime(error=run.error, parameters=final.parameters, tried=final.tried)
        plan_diagonalization = handle_pw_diagonalization(
            error=run.error,
            parameters=plan_walltime.parameters,
            tried=plan_walltime.tried,
            restart=plan_walltime.restart,
            parent_mode=plan_walltime.parent_mode,
            structure_from_output=plan_walltime.structure_from_output,
            rattle_now=plan_walltime.rattle_now,
        )
        plan_electronic = handle_pw_electronic_convergence(
            error=run.error,
            parameters=plan_diagonalization.parameters,
            tried=plan_diagonalization.tried,
            restart=plan_diagonalization.restart,
            parent_mode=plan_diagonalization.parent_mode,
            structure_from_output=plan_diagonalization.structure_from_output,
            rattle_now=plan_diagonalization.rattle_now,
        )
        plan_ionic = handle_pw_ionic_convergence(
            error=run.error,
            parameters=plan_electronic.parameters,
            tried=plan_electronic.tried,
            rattled=final.rattled,
            output_parameters=run.output_parameters,
            initial_parameters=parameters,
            restart=plan_electronic.restart,
            parent_mode=plan_electronic.parent_mode,
            structure_from_output=plan_electronic.structure_from_output,
            rattle_now=plan_electronic.rattle_now,
        )
        plan_bands = check_pw_sufficient_bands(
            error=run.error,
            parameters=plan_ionic.parameters,
            tried=plan_ionic.tried,
            output_band=run.output_band,
            output_parameters=run.output_parameters,
            restart=plan_ionic.restart,
            parent_mode=plan_ionic.parent_mode,
            structure_from_output=plan_ionic.structure_from_output,
            rattle_now=plan_ionic.rattle_now,
        )
        if plan_ionic.rattle_now and run.output_structure is not None:
            rattled = rattle_task(structure=run.output_structure, stdev=stdev)
        else:
            rattled = structure
        guard = require_pw_restart_nodes(
            restart=plan_bands.restart,
            parent_mode=plan_bands.parent_mode,
            structure_from_output=plan_bands.structure_from_output,
            output_structure=run.output_structure,
            remote_folder=run.remote_folder,
        )
        final = finalize_pw_carry(
            parameters=plan_bands.parameters,
            tried=plan_bands.tried,
            restart=plan_bands.restart,
            parent_mode=plan_bands.parent_mode,
            rattle_now=plan_bands.rattle_now,
            rattled_before=final.rattled,
            after=guard,
        )
        if plan_bands.structure_from_output and run.output_structure is not None:
            structure = run.output_structure
        if plan_bands.rattle_now:
            structure = rattled
        if final.use_remote_parent and run.remote_folder is not None:
            parent = run.remote_folder
        if final.clear_parent:
            parent = None
        restart = plan_bands.restart
    checked = ensure_pw_recovered(error=run.error, restart=restart)
    if run.error is None and not restart:
        status = 'finished'
    else:
        status = 'exhausted'
    report = summarize_restart(attempts=attempts, status=status, after=checked)
    return {
        'output_parameters': run.output_parameters,
        'output_structure': run.output_structure,
        'output_band': run.output_band,
        'output_trajectory': run.output_trajectory,
        'output_kpoints': run.output_kpoints,
        'output_atomic_occupations': run.output_atomic_occupations,
        'report': report,
    }


@graph
def ph_base_graph(
    code: orm.Code,
    parent_folder: orm.RemoteData,
    parameters: dict,
    settings: dict | None = None,
    structure: orm.StructureData | None = None,
    qpoints: orm.KpointsData | None = None,
    qpoints_distance: float | None = None,
    qpoints_force_parity: bool = False,
    only_initialization: bool = False,
    max_iterations: int = 5,
) -> dict:
    """Run ``ph.x`` with automated restarts, returning the outputs plus a report.

    Recover restarts fail loudly at ``require_ph_restart_nodes``: mapped error arms
    hide the failed calculation's ``remote_folder``, so no recover restart can run.

    ``structure`` is only read when ``qpoints_distance`` is given; the original derives
    it from the parent folder provenance instead. Multi-recover output merging is not
    ported: the last iteration outputs are returned.
    """
    if qpoints is None:
        resolved_qpoints = gen_kpoints(
            structure=structure, distance=qpoints_distance, force_parity=qpoints_force_parity
        )
    else:
        resolved_qpoints = qpoints
    prepared_settings = prepare_ph_settings(settings=settings, only_initialization=only_initialization)
    prepared = seed_ph_parameters(parameters=parameters)
    run = ph_calc(
        code=code,
        parent_folder=parent_folder,
        parameters=prepared,
        settings=prepared_settings,
        qpoints=resolved_qpoints,
    )
    plan_walltime = handle_ph_scheduler_walltime(error=run.error, parameters=prepared)
    plan_diagonalization = handle_ph_diagonalization(
        error=run.error,
        parameters=plan_walltime.parameters,
        restart=plan_walltime.restart,
        recover=plan_walltime.recover,
        parent_mode=plan_walltime.parent_mode,
    )
    plan_walltime_out = handle_ph_out_of_walltime(
        error=run.error,
        parameters=plan_diagonalization.parameters,
        restart=plan_diagonalization.restart,
        recover=plan_diagonalization.recover,
        parent_mode=plan_diagonalization.parent_mode,
    )
    plan_convergence = handle_ph_convergence(
        error=run.error,
        parameters=plan_walltime_out.parameters,
        restart=plan_walltime_out.restart,
        recover=plan_walltime_out.recover,
        parent_mode=plan_walltime_out.parent_mode,
    )
    parent = parent_folder
    guard = require_ph_restart_nodes(
        restart=plan_convergence.restart,
        parent_mode=plan_convergence.parent_mode,
        remote_folder=run.remote_folder,
    )
    final = finalize_ph_carry(
        parameters=plan_convergence.parameters,
        restart=plan_convergence.restart,
        recover=plan_convergence.recover,
        parent_mode=plan_convergence.parent_mode,
        recover_before=False,
        after=guard,
    )
    if final.use_remote_parent and run.remote_folder is not None:
        parent = run.remote_folder
    if final.clear_parent:
        parent = None
    restart = plan_convergence.restart
    attempts = 1
    while (run.error is not None or restart) and attempts < max_iterations:
        attempts = attempts + 1
        run = ph_calc(
            code=code,
            parent_folder=parent,
            parameters=final.parameters,
            settings=prepared_settings,
            qpoints=resolved_qpoints,
        )
        plan_walltime = handle_ph_scheduler_walltime(error=run.error, parameters=final.parameters)
        plan_diagonalization = handle_ph_diagonalization(
            error=run.error,
            parameters=plan_walltime.parameters,
            restart=plan_walltime.restart,
            recover=plan_walltime.recover,
            parent_mode=plan_walltime.parent_mode,
        )
        plan_walltime_out = handle_ph_out_of_walltime(
            error=run.error,
            parameters=plan_diagonalization.parameters,
            restart=plan_diagonalization.restart,
            recover=plan_diagonalization.recover,
            parent_mode=plan_diagonalization.parent_mode,
        )
        plan_convergence = handle_ph_convergence(
            error=run.error,
            parameters=plan_walltime_out.parameters,
            restart=plan_walltime_out.restart,
            recover=plan_walltime_out.recover,
            parent_mode=plan_walltime_out.parent_mode,
        )
        guard = require_ph_restart_nodes(
            restart=plan_convergence.restart,
            parent_mode=plan_convergence.parent_mode,
            remote_folder=run.remote_folder,
        )
        final = finalize_ph_carry(
            parameters=plan_convergence.parameters,
            restart=plan_convergence.restart,
            recover=plan_convergence.recover,
            parent_mode=plan_convergence.parent_mode,
            recover_before=final.recover,
            after=guard,
        )
        if final.use_remote_parent and run.remote_folder is not None:
            parent = run.remote_folder
        if final.clear_parent:
            parent = None
        restart = plan_convergence.restart
    checked = ensure_ph_recovered(error=run.error, restart=restart)
    if run.error is None and not restart:
        status = 'finished'
    else:
        status = 'exhausted'
    report = summarize_restart(attempts=attempts, status=status, after=checked)
    return {
        'output_parameters': run.output_parameters,
        'report': report,
    }


@graph
def neb_base_graph(
    code: orm.Code,
    images: orm.TrajectoryData,
    pseudo: UpfData,
    parameters: dict,
    pw_parameters: dict,
    settings: dict | None = None,
    kpoints: orm.KpointsData | None = None,
    kpoints_distance: float | None = None,
    kpoints_force_parity: bool = False,
    max_iterations: int = 5,
) -> dict:
    """Run ``neb.x`` with automated restarts, returning the outputs plus a report.

    Remote restarts fail loudly at ``require_neb_restart_nodes``: mapped error arms
    hide the failed calculation's ``remote_folder``. Only diagonalization cycling
    (parameter-only) recovers.

    The nested ``pw.pseudos`` shape is fixed to one ``Si`` kind; add entries to the
    ``neb_calc`` declaration for structures with other kind names.
    """
    last_structure = final_neb_image_structure(images=images)
    if kpoints is None:
        resolved_kpoints = gen_kpoints(
            structure=last_structure, distance=kpoints_distance, force_parity=kpoints_force_parity
        )
    else:
        resolved_kpoints = kpoints
    tried = seed_diagonalizations(parameters=pw_parameters)
    initial_parameters = parameters
    run = neb_calc(
        code=code,
        images=images,
        pseudo=pseudo,
        parameters=parameters,
        pw_parameters=pw_parameters,
        settings=settings,
        kpoints=resolved_kpoints,
    )
    plan_diagonalization = handle_neb_diagonalization(
        error=run.error, parameters=parameters, pw_parameters=pw_parameters, tried=tried
    )
    plan_interrupted = handle_neb_interrupted(
        error=run.error,
        parameters=plan_diagonalization.parameters,
        pw_parameters=plan_diagonalization.pw_parameters,
        tried=plan_diagonalization.tried,
        restart=plan_diagonalization.restart,
        parent_mode=plan_diagonalization.parent_mode,
    )
    plan_nstep = handle_neb_nstep_exceeded(
        error=run.error,
        parameters=plan_interrupted.parameters,
        pw_parameters=plan_interrupted.pw_parameters,
        tried=plan_interrupted.tried,
        initial_parameters=initial_parameters,
        restart=plan_interrupted.restart,
        parent_mode=plan_interrupted.parent_mode,
    )
    plan_electronic = handle_neb_electronic_convergence(
        error=run.error,
        parameters=plan_nstep.parameters,
        pw_parameters=plan_nstep.pw_parameters,
        tried=plan_nstep.tried,
        restart=plan_nstep.restart,
        parent_mode=plan_nstep.parent_mode,
    )
    parent = None
    guard = require_neb_restart_nodes(
        restart=plan_electronic.restart,
        parent_mode=plan_electronic.parent_mode,
        remote_folder=run.remote_folder,
    )
    final = finalize_neb_carry(
        parameters=plan_electronic.parameters,
        pw_parameters=plan_electronic.pw_parameters,
        tried=plan_electronic.tried,
        restart=plan_electronic.restart,
        parent_mode=plan_electronic.parent_mode,
        after=guard,
    )
    if final.use_remote_parent and run.remote_folder is not None:
        parent = run.remote_folder
    if final.clear_parent:
        parent = None
    restart = plan_electronic.restart
    attempts = 1
    while (run.error is not None or restart) and attempts < max_iterations:
        attempts = attempts + 1
        run = neb_calc(
            code=code,
            images=images,
            pseudo=pseudo,
            parameters=final.parameters,
            pw_parameters=final.pw_parameters,
            settings=settings,
            parent_folder=parent,
            kpoints=resolved_kpoints,
        )
        plan_diagonalization = handle_neb_diagonalization(
            error=run.error, parameters=final.parameters, pw_parameters=final.pw_parameters, tried=final.tried
        )
        plan_interrupted = handle_neb_interrupted(
            error=run.error,
            parameters=plan_diagonalization.parameters,
            pw_parameters=plan_diagonalization.pw_parameters,
            tried=plan_diagonalization.tried,
            restart=plan_diagonalization.restart,
            parent_mode=plan_diagonalization.parent_mode,
        )
        plan_nstep = handle_neb_nstep_exceeded(
            error=run.error,
            parameters=plan_interrupted.parameters,
            pw_parameters=plan_interrupted.pw_parameters,
            tried=plan_interrupted.tried,
            initial_parameters=initial_parameters,
            restart=plan_interrupted.restart,
            parent_mode=plan_interrupted.parent_mode,
        )
        plan_electronic = handle_neb_electronic_convergence(
            error=run.error,
            parameters=plan_nstep.parameters,
            pw_parameters=plan_nstep.pw_parameters,
            tried=plan_nstep.tried,
            restart=plan_nstep.restart,
            parent_mode=plan_nstep.parent_mode,
        )
        guard = require_neb_restart_nodes(
            restart=plan_electronic.restart,
            parent_mode=plan_electronic.parent_mode,
            remote_folder=run.remote_folder,
        )
        final = finalize_neb_carry(
            parameters=plan_electronic.parameters,
            pw_parameters=plan_electronic.pw_parameters,
            tried=plan_electronic.tried,
            restart=plan_electronic.restart,
            parent_mode=plan_electronic.parent_mode,
            after=guard,
        )
        if final.use_remote_parent and run.remote_folder is not None:
            parent = run.remote_folder
        if final.clear_parent:
            parent = None
        restart = plan_electronic.restart
    checked = ensure_neb_recovered(error=run.error, restart=restart)
    if run.error is None and not restart:
        status = 'finished'
    else:
        status = 'exhausted'
    report = summarize_restart(attempts=attempts, status=status, after=checked)
    return {
        'output_parameters': run.output_parameters,
        'output_trajectory': run.output_trajectory,
        'output_mep': run.output_mep,
        'iteration_array': run.iteration_array,
        'report': report,
    }


@graph
def q2r_base_graph(
    code: orm.Code,
    parent_folder: orm.RemoteData,
    parameters: dict | None = None,
    settings: dict | None = None,
) -> dict:
    """Run ``q2r.x`` once, returning the force constants and output parameters.

    The original defines no process handlers, so a failure aborts without retry and
    the graph exits with ``ERROR_TASK_FAILED``.
    """
    run = q2r_calc(code=code, parent_folder=parent_folder, parameters=parameters, settings=settings)
    return {
        'force_constants': run.force_constants,
        'output_parameters': run.output_parameters,
    }


@graph
def matdyn_base_graph(
    code: orm.Code,
    force_constants: ForceConstantsData,
    kpoints: orm.KpointsData,
    parent_folder: orm.RemoteData | None = None,
    parameters: dict | None = None,
    settings: dict | None = None,
) -> dict:
    """Run ``matdyn.x`` once, returning the outputs.

    The original defines no process handlers, so a failure aborts without retry and
    the graph exits with ``ERROR_TASK_FAILED``.
    """
    run = matdyn_calc(
        code=code,
        force_constants=force_constants,
        kpoints=kpoints,
        parent_folder=parent_folder,
        parameters=parameters,
        settings=settings,
    )
    return {
        'output_parameters': run.output_parameters,
        'output_phonon_bands': run.output_phonon_bands,
        'output_phonon_dos': run.output_phonon_dos,
    }


@graph
def bands_base_graph(
    code: orm.Code,
    parent_folder: orm.RemoteData,
    parameters: dict | None = None,
    settings: dict | None = None,
) -> dict:
    """Run ``bands.x`` once, returning the output parameters.

    The original defines no process handlers, so a failure aborts without retry and
    the graph exits with ``ERROR_TASK_FAILED``.
    """
    run = bands_calc(code=code, parent_folder=parent_folder, parameters=parameters, settings=settings)
    return {
        'output_parameters': run.output_parameters,
    }
