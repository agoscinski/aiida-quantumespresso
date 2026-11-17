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
``docs/source/topics/workflows/graphs.rst``. A ``while`` loop retries a
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
  rattling), the monitor-stuck restart (``STOPPED_BY_MONITOR`` claimed only on the
  machine-readable ``monitor_diagnostics['reason'] == 'accuracy_stuck'``, never on
  the exit message; the ``accuracy_stuck`` monitor attaches that diagnostic only in
  structured opt-in mode), and the success-path insufficient-bands check.
  Deliberately not ported:
  ``ERROR_IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF`` and
  ``WARNING_ELECTRONIC_CONVERGENCE_NOT_REACHED`` (the original finishes *with* an
  exit code while attaching outputs; a graph has no fail-with-outputs contract, so
  the graph keeps the loud ``ensure_*`` failure instead of silently succeeding with
  a non-zero status or inventing success semantics), scheduler
  ``max_seconds`` capping (deriving this from fixed launch options remains
  input-preparation work), and the disabled empty ``exit_codes=[]`` handler.
* ``ph_base_graph`` places ``PhCalculation`` with ``errors=PhErrors``. Ported:
  scheduler-walltime ``max_seconds`` halving (only when already present, see above),
  diagonalization switch to ``cg``, out-of-walltime recover restart with the
  recover latch, and the ``nmix_ph``/``alpha_mix`` convergence chain. Not ported:
  merging of multi-recover output parameters: a provenance-preserving merge remains
  QE implementation work; the graph currently returns the last iteration outputs.
  ``qpoints_distance`` takes an explicit ``structure`` input instead of deriving it
  from the parent folder provenance.
* ``neb_base_graph`` places ``NebCalculation`` directly with ``errors=NebErrors``
  (``pw.pseudos`` is a static ``DataDict`` container on the ``pw`` namespace).
  Ported: diagonalization cycling on the nested ``pw``
  parameters, interrupted-trajectory full restart, ``nstep_path`` extension from the
  initial inputs, and electronic ``mixing_beta`` reduction. ``kpoints_distance``
  builds k-points from the last image structure.
* ``q2r_base_graph``, ``matdyn_base_graph``, ``bands_base_graph`` place their
  calculation once and return its outputs. This is faithful: the originals define no
  process handlers, so a failure aborts without retry.

Failed outputs and the restart-node guards
------------------------------------------

A placement that finishes with a mapped error exposes its ``error`` arm alongside
the failed run's stored outputs: the runtime preserves them, so
``run.remote_folder`` and ``run.output_structure`` read the failed calculation's
nodes downstream, exactly like the legacy handlers reading
``calculation.outputs`` directly. Restarts that need those nodes (walltime full
restarts, ionic output-structure restarts, electronic full restarts, NEB remote
restarts, every ``ph`` recover restart) therefore rebind and recover. The
``require_*_restart_nodes`` tasks stay as defense in depth: they fail the graph
loudly (naming the missing node) when a claimed restart needs a node the failed
run genuinely never stored (for example a job killed before writing any output),
instead of retrying degraded. The ``is not None`` rebinding checks stay as
further defense in depth behind the guards.

Static pseudos and the one supported monitor
--------------------------------------------

The ``pseudos`` inputs are static ``DataDict`` containers mapping kind names onto
owned ``UpfData`` members (one entry per structure kind). CalcJobs consume the
container directly; no pseudopotential namespace declaration or expansion is needed
in the graph. Provenance links the single ``pseudos`` container, whose members keep
their identity and repository content. ``Code`` inputs are required; scheduler
options travel in ``task_metadata`` keyed by placement name.

There is no container forwarding into the ``CalcJob`` ``monitors`` namespace: the
monitored ``pw`` placement declares the one supported entry statically
(``monitors.stuck`` holding a monitor ``Dict``), and a plain ``pw`` placement
covers unmonitored runs with no monitor inputs at all. Core static declarations
are required when wired, so one handle cannot serve both shapes. The ``monitors``
graph input stays an optional ``DataDict`` so composites keep threading per-stage
mappings unchanged; a boundary task extracts the single known ``stuck`` entry
(rejecting any other key or non-``Dict`` member outright) and each ``pw`` call
site branches on its presence. Only the structured ``accuracy_stuck`` monitor can
drive recovery: the monitor-stuck handler claims ``STOPPED_BY_MONITOR`` only on
the machine-readable ``monitor_diagnostics['reason'] == 'accuracy_stuck'``.

Considered-finished 501/710 stay usable
----------------------------------------

Legacy ``501`` (ionic converged except in final SCF) and ``710`` (electronic
convergence warning with ``scf_must_converge=False``) finish *with* outputs. The
graph maps both in ``PwErrors`` and returns the failed run's stored outputs (the
runtime keeps them alongside the synthesized error) with a distinct report
status (``finished_ionic_convergence_reached_except_final_scf`` /
``finished_electronic_convergence_warning``) instead of retrying or failing.
These are implemented report outcomes; the graphs declare no numeric exit
codes, so this is not exact numeric root parity with the legacy statuses.

Native ``pw`` subgraph
----------------------

``pw_base_graph`` is the reusable restart loop for composite graphs:
``(code, structure, pseudos, parameters, settings?, kpoints?, kpoints_distance?,
kpoints_force_parity?, parent_folder?, monitors?, max_iterations?)``. ``parameters`` accepts a
plain ``dict`` or an ``orm.Dict``. ``monitors`` is an optional ``DataDict`` carrying
at most the single known ``stuck`` monitor ``Dict`` (entry point plus kwargs); the
graph resolves it to the statically declared ``monitors.stuck`` CalcJob input and
wires nothing when it is absent, so unmonitored runs carry no monitor inputs.
Structured monitors attach
a ``monitor_diagnostics`` output that the monitor-stuck handler reads. It returns the calculation outputs
(``output_parameters`` always; ``output_structure``, ``output_band``,
``output_trajectory``, ``output_kpoints``, ``output_atomic_occupations`` and
``remote_folder`` when the finished run stores them) plus a ``report`` recording
attempts and status. Outputs are existing nodes regrouped, never fabricated data.
"""

import typing as t

from aiida import orm
from aiida.engine import graph, task
from aiida.engine.processes.exit_code import ExitCode, ExitCodeEnum

from aiida_quantumespresso.calculations.bands import BandsCalculation
from aiida_quantumespresso.calculations.matdyn import MatdynCalculation
from aiida_quantumespresso.calculations.neb import NebCalculation
from aiida_quantumespresso.calculations.ph import PhCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.q2r import Q2rCalculation
from aiida_quantumespresso.data.force_constants import ForceConstantsData
from aiida_quantumespresso.tools.monitors.accuracy_stuck import ACCURACY_STUCK_REASON
from aiida_quantumespresso.utils.bands import get_highest_occupied_band

# Fallback values mirror ``PwBaseWorkChain.defaults`` and ``qe_defaults``. They are
# repeated here (instead of imported) because importing the work chain modules would
# require installed entry points.
_MIXING_BETA_DEFAULT = 0.7
_MIXING_BETA_FACTOR = 0.8
_UPSCALE_MONITOR_FACTOR = 0.3
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

    STOPPED_BY_MONITOR = ExitCode(
        150, 'The calculation was stopped by a monitor; only a recognized diagnostic restarts it.'
    )
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
    IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF = ExitCode(
        501, 'The ionic minimization cycle converged but the thresholds are exceeded in the final SCF.'
    )
    ELECTRONIC_CONVERGENCE_WARNING = ExitCode(
        710,
        'The electronic minimization cycle did not reach self-consistency, but `scf_must_converge` '
        'is `False` and/or `electron_maxstep` is 0.',
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
    keep_parent: bool
    use_remote_parent: bool


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
    keep_parent: bool
    use_remote_parent: bool


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
    keep_parent: bool
    use_remote_parent: bool


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


@task
def handle_pw_monitor_stuck(
    error: PwErrors | None,
    parameters: dict,
    tried: list,
    diagnostics: orm.Dict | None,
    restart: bool = False,
    parent_mode: str = 'keep',
    structure_from_output: bool = False,
    rattle_now: bool = False,
) -> PwRestartPlan:
    """Reduce ``IONS.upscale`` and fully restart on an accuracy-stuck monitor stop.

    Mirrors ``PwBaseWorkChain.handle_electronic_convergence_stuck`` except for the
    dispatch contract: the legacy handler gates on the exit *message* prefix, which the
    mapped error arm does not carry. This handler claims ``STOPPED_BY_MONITOR`` only
    when the failed run stored a ``monitor_diagnostics`` output whose ``reason`` is
    ``'accuracy_stuck'`` (attached by the structured ``accuracy_stuck`` monitor). An
    unrelated monitor, a legacy string-only stop without diagnostics, or any other
    failure passes the upstream restart decision through untouched, so the loop exits
    and the guard fails loudly instead of retrying the wrong recovery. Like the
    original, ``upscale`` already at its minimum requests no restart; both paths
    fully restart (``restart_mode`` set, charge-density flags dropped, parent
    rebound), with only ``relax``/``vc-relax`` additionally reducing ``upscale``.
    """
    plan = _pass_pw_plan(parameters, tried, restart, parent_mode, structure_from_output, rattle_now)
    if error is not PwErrors.STOPPED_BY_MONITOR:
        return plan
    if diagnostics is None or diagnostics.get_dict().get('reason') != ACCURACY_STUCK_REASON:
        return plan
    calculation = parameters.get('CONTROL', {}).get('calculation', 'scf')
    updated = _pw_parameters(parameters)
    if calculation in ('relax', 'vc-relax'):
        upscale = updated.get('IONS', {}).get('upscale', _UPSCALE_DEFAULT)
        if upscale <= 1:
            return plan
        updated.setdefault('IONS', {})['upscale'] = max(int(upscale * _UPSCALE_MONITOR_FACTOR), 1)
    # A full restart reuses the charge density on disk, never the input flags.
    updated['CONTROL']['restart_mode'] = 'restart'
    updated['ELECTRONS'].pop('startingpot', None)
    updated['ELECTRONS'].pop('startingwfc', None)
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
    structure and the parent folder with branches over these plain-data flags. The
    keep arm comes first so the carried ``parent`` wire stays nullable: a taken
    ``None`` arm would declare the untaken arm ``None``-only and reject a rebound
    folder at runtime.
    """
    return {
        'parameters': parameters,
        'tried': tried,
        'restart': restart,
        'rattled': rattled_before or rattle_now,
        'keep_parent': parent_mode == 'keep',
        'use_remote_parent': parent_mode == 'remote',
    }


class PwFinished(t.TypedDict):
    """Considered-finished outcome: task outputs cannot be ``None``."""

    finished: bool
    status: str


@task
def classify_pw_finished(error: PwErrors | None) -> PwFinished:
    """Return the considered-finished outcome for 501/710, else a proceed sentinel.

    Mirrors the legacy ``handle_vcrelax_converged_except_final_scf`` (501) and
    ``handle_electronic_convergence_warning`` (710): both consider the run finished
    *with* usable outputs instead of retrying. The mapped failure keeps every stored
    declared output alongside the synthesized error, so the graph returns the real
    ``run.output_*`` nodes with a distinct report status naming the outcome. Task
    outputs cannot return ``None`` (the runtime rejects the absent arm), so the
    proceed path carries ``finished=False`` with a ``'proceed'`` status the graph
    never reports; the full status strings are built here because graph bodies
    cannot format text.
    """
    if error is PwErrors.IONIC_CONVERGENCE_REACHED_EXCEPT_IN_FINAL_SCF:
        return {'finished': True, 'status': 'finished_ionic_convergence_reached_except_final_scf'}
    if error is PwErrors.ELECTRONIC_CONVERGENCE_WARNING:
        return {'finished': True, 'status': 'finished_electronic_convergence_warning'}
    return {'finished': False, 'status': 'proceed'}


@task
def ensure_pw_recovered(error: PwErrors | None, restart: bool, finished: bool = False) -> None:
    """Fail the graph when the ``pw`` loop exits without a recovered run."""
    if finished:
        return
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
    return {
        'parameters': parameters,
        'restart': restart,
        'recover': recover_before or recover,
        'keep_parent': parent_mode == 'keep',
        'use_remote_parent': parent_mode == 'remote',
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
    return {
        'parameters': parameters,
        'pw_parameters': pw_parameters,
        'tried': tried,
        'restart': restart,
        'keep_parent': parent_mode == 'keep',
        'use_remote_parent': parent_mode == 'remote',
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
    from aiida.orm import KpointsData
    from numpy import linalg

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


#: The only monitor key the native graphs wire into the ``CalcJob`` ``monitors``
#: namespace. The ``monitors`` graph inputs stay ``DataDict`` mappings so composites
#: keep threading per-stage mappings unchanged, but anything besides this single
#: entry is rejected at the boundary instead of silently dropped.
STUCK_MONITOR_KEY = 'stuck'

#: The only monitor entry point the native graphs can recover from. Recovery keys on the
#: machine-readable ``monitor_diagnostics['reason'] == 'accuracy_stuck'`` that only the structured
#: ``quantumespresso.accuracy_stuck`` monitor attaches, so any other entry point would stop the
#: calculation without ever driving recovery. Mirrored from the protocol helper's private validator
#: (``workflows.graphs.protocols``) instead of imported: this boundary module must not depend on the
#: launch helper, so the two stay in sync by convention, not by import.
KNOWN_STUCK_ENTRY_POINT = 'quantumespresso.accuracy_stuck'

#: Allowed kwargs of the known accuracy-stuck monitor, mirroring the protocol helper.
KNOWN_STUCK_KWARGS = frozenset({'min_repeats', 'structured'})

#: Top-level monitor specification fields the resolver preserves. ``entry_point`` and ``kwargs``
#: are validated strictly (see above); the rest are the ``CalcJobMonitor`` runtime fields a direct
#: launch may set (for example ``priority``). Anything else fails fast as a likely typo instead of
#: travelling silently into the placement inputs.
ALLOWED_STUCK_SPEC_KEYS = frozenset(
    {'entry_point', 'kwargs', 'priority', 'minimum_poll_interval', 'call_timestamp', 'disabled'}
)


class StuckMonitorResolution(t.TypedDict):
    """Boundary monitor outcome: task outputs cannot be ``None``."""

    present: bool
    monitor: dict


@task
def resolve_stuck_monitor(monitors: orm.DataDict | None) -> StuckMonitorResolution:
    """Extract the supported ``stuck`` monitor entry, if any, rejecting anything else.

    The ``CalcJob`` ``monitors`` namespace is dynamic, so the graph cannot declare
    an open mapping there; the monitored ``pw`` placement instead declares the one
    known entry statically (``monitors.stuck``) while the plain placement wires no
    monitor input at all. This boundary task converts the shared ``DataDict``
    mapping: ``None`` or an empty mapping resolves to ``present=False`` (the graph
    branches to the unmonitored placement), a single ``stuck`` ``Dict`` holding the
    structured ``quantumespresso.accuracy_stuck`` monitor resolves to an owned copy
    of it, and anything else fails fast naming the offending value. Only the
    structured variant (``kwargs['structured'] is True``, seeded when absent) can
    drive recovery, since the monitor-stuck handler claims ``STOPPED_BY_MONITOR``
    only on the machine-readable ``monitor_diagnostics['reason']`` the structured
    monitor attaches. Allowed ``CalcJobMonitor`` runtime fields (for example
    ``priority``) are preserved untouched on the owned copy. Task outputs cannot
    return ``None``, so the absent path carries an empty mapping the graph never
    wires. Downstream recovery still keys on the structured
    ``monitor_diagnostics['reason']``, never on this task.
    """
    if monitors is None or len(monitors) == 0:
        return {'present': False, 'monitor': {}}
    keys = list(monitors.keys())
    if keys != [STUCK_MONITOR_KEY]:
        msg = (
            'unsupported `monitors` mapping: native graphs only wire the single '
            f"'{STUCK_MONITOR_KEY}' monitor entry, got keys {keys}."
        )
        raise ValueError(msg)
    member = monitors[STUCK_MONITOR_KEY]
    if not isinstance(member, orm.Dict):
        msg = (
            f"`monitors['{STUCK_MONITOR_KEY}']` must be a `Dict` holding the monitor entry point and "
            f'kwargs, got `{type(member).__name__}`.'
        )
        raise ValueError(msg)
    spec = member.get_dict()
    for key in spec:
        if key not in ALLOWED_STUCK_SPEC_KEYS:
            msg = (
                f"`monitors['{STUCK_MONITOR_KEY}.{key}']` is not supported; expected one of "
                f'{sorted(ALLOWED_STUCK_SPEC_KEYS)}.'
            )
            raise ValueError(msg)
    entry_point = spec.get('entry_point')
    if entry_point != KNOWN_STUCK_ENTRY_POINT:
        msg = (
            f"`monitors['{STUCK_MONITOR_KEY}'].entry_point` must be `{KNOWN_STUCK_ENTRY_POINT}`, "
            f'got `{entry_point!r}`: only the structured accuracy-stuck monitor can drive recovery.'
        )
        raise ValueError(msg)
    kwargs = spec.get('kwargs', {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        msg = f"`monitors['{STUCK_MONITOR_KEY}'].kwargs` must be a mapping, got `{type(kwargs).__name__}`."
        raise ValueError(msg)
    for key in kwargs:
        if key not in KNOWN_STUCK_KWARGS:
            msg = (
                f"`monitors['{STUCK_MONITOR_KEY}'].kwargs.{key}` is not supported; expected one of "
                f'{sorted(KNOWN_STUCK_KWARGS)}.'
            )
            raise ValueError(msg)
    if 'min_repeats' in kwargs and (
        isinstance(kwargs['min_repeats'], bool) or not isinstance(kwargs['min_repeats'], int)
    ):
        msg = (
            f"`monitors['{STUCK_MONITOR_KEY}'].kwargs.min_repeats` must be an `int`, got "
            f'`{type(kwargs["min_repeats"]).__name__}`.'
        )
        raise ValueError(msg)
    if 'min_repeats' in kwargs and kwargs['min_repeats'] <= 0:
        msg = f"`monitors['{STUCK_MONITOR_KEY}'].kwargs.min_repeats` must be positive."
        raise ValueError(msg)
    if 'structured' in kwargs and not isinstance(kwargs['structured'], bool):
        msg = (
            f"`monitors['{STUCK_MONITOR_KEY}'].kwargs.structured` must be a `bool`, got "
            f'`{type(kwargs["structured"]).__name__}`.'
        )
        raise ValueError(msg)
    if kwargs.get('structured', True) is not True:
        msg = (
            f"`monitors['{STUCK_MONITOR_KEY}'].kwargs.structured` must be `True`: unstructured stops "
            'carry no diagnostics for the monitor-stuck handler to match on.'
        )
        raise ValueError(msg)
    resolved = dict(spec)
    if 'kwargs' not in resolved or 'structured' not in kwargs:
        resolved_kwargs = dict(kwargs)
        resolved_kwargs.setdefault('structured', True)
        resolved['kwargs'] = resolved_kwargs
        # Returning the member itself would hand back an input-owned node, so the
        # placement wires an owned copy, exactly like the node-rebinding tasks above.
        return {'present': True, 'monitor': orm.Dict(resolved)}
    # Returning the member itself would hand back an input-owned node, so the
    # placement wires an owned copy, exactly like the node-rebinding tasks above.
    return {'present': True, 'monitor': member.clone()}


pw_calc = task(PwCalculation, errors=PwErrors)
#: Monitored ``pw`` placement: the one supported entry is declared statically as
#: ``monitors.stuck`` holding a monitor ``Dict``. Core static declarations are
#: required when wired, so unmonitored runs use plain ``pw_calc`` (no monitor
#: inputs at all) while monitored runs use this handle. Both share ``PwErrors``,
#: including the considered-finished 501/710 members whose stored outputs stay
#: available downstream alongside the synthesized error.
pw_calc_monitored = task(PwCalculation, inputs={'monitors': {'stuck': orm.Dict}}, errors=PwErrors)
ph_calc = task(PhCalculation, errors=PhErrors)
neb_calc = task(NebCalculation, errors=NebErrors)
q2r_calc = task(Q2rCalculation)
matdyn_calc = task(MatdynCalculation)
bands_calc = task(BandsCalculation)
gen_kpoints = make_kpoints_mesh
rattle_task = rattle_positions


@graph
def pw_base_graph(
    code: orm.Code,
    structure: orm.StructureData,
    pseudos: orm.DataDict,
    parameters: dict,
    settings: dict | None = None,
    kpoints: orm.KpointsData | None = None,
    kpoints_distance: float | None = None,
    kpoints_force_parity: bool = False,
    parent_folder: orm.RemoteData | None = None,
    monitors: orm.DataDict | None = None,
    max_iterations: int = 5,
) -> dict:
    """Run ``pw.x`` with automated restarts, returning the outputs plus a report.

    This is the reusable native restart loop for composite graphs: restarts that
    need the failed run's outputs (walltime full restarts, ionic output-structure
    restarts, electronic full restarts, monitor-stuck upscale restarts) rebind the
    preserved ``remote_folder`` and ``output_structure`` and recover;
    ``require_pw_restart_nodes`` fails loudly
    only when the failed run genuinely never stored the needed node.
    ``parameters`` accepts a plain ``dict`` or an ``orm.Dict``. Scheduler options
    travel in ``task_metadata`` keyed by placement name at launch. ``monitors`` is an
    optional ``DataDict`` carrying at most the single known ``stuck`` monitor entry;
    omit it (or pass an empty mapping) and the placements run unmonitored with no
    monitor inputs wired. Any other entry fails fast in ``resolve_stuck_monitor``.
    """
    if kpoints is None:
        resolved_kpoints = gen_kpoints(
            structure=structure, distance=kpoints_distance, force_parity=kpoints_force_parity
        )
    else:
        resolved_kpoints = kpoints
    tried = seed_diagonalizations(parameters=parameters)
    stuck = resolve_stuck_monitor(monitors=monitors)
    if stuck.present:
        run = pw_calc_monitored(
            code=code,
            structure=structure,
            pseudos=pseudos,
            parameters=parameters,
            settings=settings,
            kpoints=resolved_kpoints,
            parent_folder=parent_folder,
            monitors={'stuck': stuck.monitor},
        )
    else:
        run = pw_calc(
            code=code,
            structure=structure,
            pseudos=pseudos,
            parameters=parameters,
            settings=settings,
            kpoints=resolved_kpoints,
            parent_folder=parent_folder,
        )
    finished = classify_pw_finished(error=run.error)
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
    plan_monitor = handle_pw_monitor_stuck(
        error=run.error,
        parameters=plan_electronic.parameters,
        tried=plan_electronic.tried,
        diagnostics=run.monitor_diagnostics,
        restart=plan_electronic.restart,
        parent_mode=plan_electronic.parent_mode,
        structure_from_output=plan_electronic.structure_from_output,
        rattle_now=plan_electronic.rattle_now,
    )
    plan_ionic = handle_pw_ionic_convergence(
        error=run.error,
        parameters=plan_monitor.parameters,
        tried=plan_monitor.tried,
        rattled=False,
        output_parameters=run.output_parameters,
        initial_parameters=parameters,
        restart=plan_monitor.restart,
        parent_mode=plan_monitor.parent_mode,
        structure_from_output=plan_monitor.structure_from_output,
        rattle_now=plan_monitor.rattle_now,
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
    parent = parent_folder
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
    if final.keep_parent:
        parent = parent  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
    elif final.use_remote_parent:
        parent = run.remote_folder
    else:
        parent = None
    restart = plan_bands.restart
    attempts = 1
    while ((run.error is not None and not finished.finished) or restart) and attempts < max_iterations:
        attempts = attempts + 1
        if stuck.present:
            run = pw_calc_monitored(
                code=code,
                structure=structure,
                pseudos=pseudos,
                parameters=final.parameters,
                settings=settings,
                kpoints=resolved_kpoints,
                parent_folder=parent,
                monitors={'stuck': stuck.monitor},
            )
        else:
            run = pw_calc(
                code=code,
                structure=structure,
                pseudos=pseudos,
                parameters=final.parameters,
                settings=settings,
                kpoints=resolved_kpoints,
                parent_folder=parent,
            )
        finished = classify_pw_finished(error=run.error)
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
        plan_monitor = handle_pw_monitor_stuck(
            error=run.error,
            parameters=plan_electronic.parameters,
            tried=plan_electronic.tried,
            diagnostics=run.monitor_diagnostics,
            restart=plan_electronic.restart,
            parent_mode=plan_electronic.parent_mode,
            structure_from_output=plan_electronic.structure_from_output,
            rattle_now=plan_electronic.rattle_now,
        )
        plan_ionic = handle_pw_ionic_convergence(
            error=run.error,
            parameters=plan_monitor.parameters,
            tried=plan_monitor.tried,
            rattled=final.rattled,
            output_parameters=run.output_parameters,
            initial_parameters=parameters,
            restart=plan_monitor.restart,
            parent_mode=plan_monitor.parent_mode,
            structure_from_output=plan_monitor.structure_from_output,
            rattle_now=plan_monitor.rattle_now,
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
        if final.keep_parent:
            parent = parent  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        elif final.use_remote_parent:
            parent = run.remote_folder
        else:
            parent = None
        restart = plan_bands.restart
    checked = ensure_pw_recovered(error=run.error, restart=restart, finished=finished.finished)
    if finished.finished:
        status = finished.status
    elif run.error is None and not restart:
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
        'remote_folder': run.remote_folder,
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
    if final.keep_parent:
        parent = parent  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
    elif final.use_remote_parent:
        parent = run.remote_folder
    else:
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
        if final.keep_parent:
            parent = parent  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        elif final.use_remote_parent:
            parent = run.remote_folder
        else:
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
    pseudos: orm.DataDict,
    parameters: dict,
    pw_parameters: dict,
    settings: dict | None = None,
    kpoints: orm.KpointsData | None = None,
    kpoints_distance: float | None = None,
    kpoints_force_parity: bool = False,
    parent_folder: orm.RemoteData | None = None,
    max_iterations: int = 5,
) -> dict:
    """Run ``neb.x`` with automated restarts, returning the outputs plus a report.

    Remote restarts rebind the preserved ``remote_folder`` and recover;
    ``require_neb_restart_nodes`` fails loudly only when the failed run genuinely
    never stored one. ``parameters`` accepts a plain ``dict`` or an ``orm.Dict``.
    Scheduler options travel in ``task_metadata`` keyed by placement name at
    launch.
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
        parameters=parameters,
        pw={
            'parameters': pw_parameters,
            'pseudos': pseudos,
            'kpoints': resolved_kpoints,
        },
        settings=settings,
        parent_folder=parent_folder,
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
    parent = parent_folder
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
    if final.keep_parent:
        parent = parent  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
    elif final.use_remote_parent:
        parent = run.remote_folder
    else:
        parent = None
    restart = plan_electronic.restart
    attempts = 1
    while (run.error is not None or restart) and attempts < max_iterations:
        attempts = attempts + 1
        run = neb_calc(
            code=code,
            images=images,
            parameters=final.parameters,
            pw={
                'parameters': final.pw_parameters,
                'pseudos': pseudos,
                'kpoints': resolved_kpoints,
            },
            settings=settings,
            parent_folder=parent,
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
        if final.keep_parent:
            parent = parent  # noqa: PLW0127 - keep arm rebinds the same wire to stay nullable
        elif final.use_remote_parent:
            parent = run.remote_folder
        else:
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
        'remote_folder': run.remote_folder,
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
