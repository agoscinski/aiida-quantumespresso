############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Plain protocol helpers for the native PW/PDOS graphs.

:func:`get_protocol_inputs` returns the raw merged YAML protocol settings as
plain data, exactly as the legacy work chains consume them. It never resolves
pseudos, cutoffs, or thresholds.

:func:`get_inputs_from_protocol` translates that YAML nesting into launch-ready
native graph arguments plus a ``runtime_config`` mapping. It resolves one
pseudo set (explicit ``UpfData`` nodes or an ``aiida-pseudo`` family with
recommended cutoffs), per-atom ``conv_thr``/``etot_conv_thr`` thresholds,
k-point distances, the requested electronic/spin settings, and the optional
``monitors`` mappings. It reuses the
pure science helpers (:func:`get_magnetization
<aiida_quantumespresso.workflows.protocols.utils.get_magnetization>`,
fixed-coordinates handling) and never imports or calls a ``WorkChain``
subclass or a ``ProcessBuilder``.

Scheduler options are not graph inputs. The helper lifts every
``pw.metadata.options``/``dos.metadata.options``/``projwfc.metadata.options``
section into the ``runtime_config`` launch mapping as a strictly nested mapping,
and merges the ``options`` keyword into every placement. Each authored subgraph
callee alias adds one level (for example ``{'scf': {'pw_calc': ...},
'bands': {...}}`` for the bands root); the ``CalcJob`` placement holds
the process input mapping (with a ``metadata`` key) at the leaf. Branch and loop control frames are
transparent, and loop iterations reuse their placement's configuration, so no
generated suffixes or per-iteration keys appear. The protocol supplies
scheduler options for every ``CalcJob`` placement. Unknown paths and
non-mapping leaves fail at launch. See
``reports/STATUS.md`` for the scope contract and the
authored aliases of each production root.

The QE ``CONTROL.max_seconds`` cutoff is prepared per ``pw`` placement at launch:
when the resolved ``CONTROL`` parameters carry no explicit ``max_seconds`` and the final
``runtime_config`` leaf holds ``metadata.options.max_wallclock_seconds``, the helper
seeds ``0.95`` of the wallclock (mirroring ``PwBaseWorkChain.prepare_process``); an explicit
``max_seconds`` always wins and a placement without a wallclock gets no cutoff. The cutoff is
constant across base-loop retries, matching the fixed scheduler options.

Monitors travel as graph inputs, not scheduler options. Each ``pw.monitors``
YAML section holds a mapping of monitor names onto ``{'entry_point': ...,
'kwargs': {...}}`` entries; only the ``stuck`` monitor key with the known
``quantumespresso.accuracy_stuck``
entry point is accepted and unknown fields fail fast. The helper converts each
entry into an ``orm.Dict`` collected in an ``orm.DataDict`` and forwards it
into every nested base run through its authored alias (``scf``/``bands``,
``initial``/``main``, ``scf``/``nscf``; shared plus per-stage overrides).

What is deliberately not resolved (fail fast, never silently dropped):

* ``dry_run`` and ``serial_clean`` overrides are rejected;
  ``serial_clean`` maps to the explicit ``serial`` keyword.
* ``parallelization`` sections are rejected; the native base graph has no
  parallelization port.
* An inner base ``max_iterations`` other than the graph default (5) is
  rejected; only the outer budget is exposed.
* The relax graph carries per-stage k-point meshes, settings mappings, and ``kpoints_force_parity``
  values for the initial and the main relax. The graph defaults (``init_use_main_inputs=True``)
  fall back to the main value when an ``init_*`` input is unset, preserving direct launches; the
  protocol helper launches with ``init_use_main_inputs=False`` and passes each stage's independently
  resolved values with no inheritance (an absent ``init_settings`` stays absent), while their scheduler
  options resolve independently per scope (``initial`` versus ``main``).
* ``RelaxType.VOLUME``/``POSITIONS_VOLUME`` and 2D structures outside the x-y
  plane are rejected, mirroring the legacy builder.
"""

import copy
import enum
import hashlib
import math
import pathlib
import typing as t
import warnings

import yaml
from aiida import orm
from aiida.common import exceptions
from aiida.engine import ExecutionEnvironment, PreparedRun
from aiida.plugins import GroupFactory

from aiida_quantumespresso.common.types import ElectronicType, RelaxType, SpinType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin, get_magnetization

__all__ = (
    'PROTOCOL_KINDS',
    'available_protocols',
    'get_default_protocol',
    'get_inputs_from_protocol',
    'prepare_ph_inputs',
    'prepare_run_from_protocol',
    'get_protocol_filepath',
    'get_protocol_inputs',
    'pdos_protocol_runtime_config',
    'runtime_config_for_placements',
)

PROTOCOL_KINDS = ('pw_relax', 'pw_bands', 'pdos')
"""Protocol families with a YAML file in ``workflows/protocols``."""

_RELATIVE_FILES: dict[str, pathlib.Path] = {
    'pw_relax': pathlib.Path('pw') / 'relax.yaml',
    'pw_bands': pathlib.Path('pw') / 'bands.yaml',
    'pdos': pathlib.Path('pdos.yaml'),
}

_ALIASES = {'moderate': 'balanced', 'precise': 'stringent'}

_BASE_PROTOCOL_FILE = pathlib.Path('pw') / 'base.yaml'

#: Keys a caller may override per base (``pw.x``) section. ``clean_workdir``
#: is intentionally absent: only the root-level value reaches the graph.
_BASE_SECTION_KEYS = frozenset(
    {'meta_parameters', 'pseudo_family', 'pw', 'kpoints_distance', 'kpoints_force_parity', 'max_iterations'}
)

#: Keys allowed inside a base section ``pw`` mapping.
_BASE_PW_KEYS = frozenset({'parameters', 'settings', 'metadata', 'monitors'})

#: Top-level keys allowed per graph family after merging.
_TOP_LEVEL_KEYS: dict[str, frozenset] = {
    'pw_relax': frozenset(
        {'clean_workdir', 'max_meta_convergence_iterations', 'meta_convergence', 'base_relax', 'base_init_relax'}
    ),
    'pw_bands': frozenset({'clean_workdir', 'nbands_factor', 'scf', 'bands', 'bands_kpoints_distance'}),
    'pdos': frozenset({'clean_workdir', 'scf', 'nscf', 'dos', 'projwfc', 'nbands_factor', 'energy_range_vs_fermi'}),
}

#: Legacy knobs with no native-graph consumer; mentioning them is always an error.
_REJECTED_OVERRIDES = {
    'dry_run': 'the native graphs have no `dry_run` mode; inspect the generated inputs instead.',
    'serial_clean': 'pass `serial=True` to select the serial PDOS root instead of the `serial_clean` flag.',
}

#: The only monitor entry point the protocol helper wires structurally.
_KNOWN_MONITOR_ENTRY_POINT = 'quantumespresso.accuracy_stuck'

#: The only monitor key the native graphs wire into the ``CalcJob`` ``monitors`` namespace.
_STUCK_MONITOR_KEY = 'stuck'

#: Allowed kwargs of the known accuracy-stuck monitor.
_KNOWN_MONITOR_KWARGS = frozenset({'min_repeats', 'structured'})

#: The inner restart budget the native base graph always runs with.
_BASE_MAX_ITERATIONS = 5

#: Fraction of the scheduler wallclock reserved for the QE ``CONTROL.max_seconds`` cutoff.
#: Mirrors ``PwBaseWorkChain.defaults.delta_factor_max_seconds``: the legacy ``prepare_process``
#: sets ``max_seconds`` to this fraction of ``metadata.options.max_wallclock_seconds`` on every
#: attempt when ``CONTROL.max_seconds`` is absent. The native graphs fix scheduler options at
#: launch, so the launch helper prepares the same cutoff per CalcJob placement up front.
_DELTA_FACTOR_MAX_SECONDS = 0.95

#: Scheduler default resources follow the legacy builder per placement code.
_DEFAULT_RESOURCES = (
    'core.direct',
    'core.slurm',
    'core.pbspro',
    'core.torque',
    'hyperqueue',
)


def get_protocol_filepath(kind: str) -> pathlib.Path:
    """Return the YAML protocol file for one graph family.

    :param kind: one of ``'pw_relax'``, ``'pw_bands'``, ``'pdos'``.
    :return: the path to the YAML file.
    :raises ValueError: for an unknown kind.
    """
    try:
        relative = _RELATIVE_FILES[kind]
    except KeyError:
        msg = f'Unknown protocol kind {kind!r}; expected one of {sorted(_RELATIVE_FILES)}.'
        raise ValueError(msg) from None
    return pathlib.Path(__file__).resolve().parent.parent / 'protocols' / relative


def _load_protocol_file(kind: str) -> dict:
    """Return the parsed YAML mapping for one graph family."""
    with get_protocol_filepath(kind).open() as handle:
        return yaml.safe_load(handle)


def _load_base_protocol(protocol: str | None) -> tuple[dict, str]:
    """Return the merged base (``pw.x``) protocol defaults and the resolved protocol name."""
    with (pathlib.Path(__file__).resolve().parent.parent / 'protocols' / _BASE_PROTOCOL_FILE).open() as handle:
        data = yaml.safe_load(handle)
    protocol = protocol or data['default_protocol']
    try:
        protocol_inputs = data['protocols'][protocol]
    except KeyError:
        alias = _ALIASES.get(protocol)
        if alias is None or alias not in data['protocols']:
            msg = f'`{protocol}` is not a valid protocol. Call `available_protocols` to show available protocols.'
            raise ValueError(msg) from None
        protocol_inputs = data['protocols'][alias]
        protocol = alias
    merged = _recursive_merge(data['default_inputs'], protocol_inputs)
    merged.pop('description', None)
    return merged, protocol


def _recursive_merge(left: dict, right: dict) -> dict:
    """Recursively merge two mappings, with ``right`` winning on conflicts."""
    merged = copy.deepcopy(left)
    for key, value in copy.deepcopy(right).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def available_protocols(kind: str) -> dict[str, dict[str, str]]:
    """Return the available protocols and their descriptions for one graph family.

    :param kind: one of ``'pw_relax'``, ``'pw_bands'``, ``'pdos'``.
    :return: mapping of protocol name to ``{'description': ...}``.
    """
    data = _load_protocol_file(kind)
    return {name: {'description': values['description']} for name, values in data['protocols'].items()}


def get_default_protocol(kind: str) -> str:
    """Return the default protocol name for one graph family.

    :param kind: one of ``'pw_relax'``, ``'pw_bands'``, ``'pdos'``.
    :return: the default protocol name.
    """
    return _load_protocol_file(kind)['default_protocol']


def get_protocol_inputs(
    kind: str,
    protocol: str | None = None,
    overrides: dict | pathlib.Path | None = None,
) -> dict:
    """Return the merged protocol inputs for one graph family as plain data.

    The result merges ``default_inputs``, the named protocol, and ``overrides``
    with the same nesting as the YAML files. Values stay plain (``dict``/``list``/
    scalars); the caller resolves parameter mappings to ``Dict`` nodes and
    resolves pseudos separately.

    :param kind: one of ``'pw_relax'``, ``'pw_bands'``, ``'pdos'``.
    :param protocol: the protocol name or alias (``'moderate'``/``'precise'``); defaults to the YAML default.
    :param overrides: an overrides mapping (same nesting as the YAML) or a path to a YAML file holding one.
    :return: the merged plain-data inputs.
    :raises ValueError: for an unknown protocol.
    """
    data = _load_protocol_file(kind)
    protocol = protocol or data['default_protocol']
    try:
        protocol_inputs = data['protocols'][protocol]
    except KeyError:
        alias = _ALIASES.get(protocol)
        if alias is None or alias not in data['protocols']:
            msg = f'`{protocol}` is not a valid protocol. Call `available_protocols` to show available protocols.'
            raise ValueError(msg) from None
        protocol_inputs = data['protocols'][alias]
    merged = _recursive_merge(data['default_inputs'], protocol_inputs)
    merged.pop('description', None)

    if isinstance(overrides, pathlib.Path):
        with overrides.open() as handle:
            overrides = yaml.safe_load(handle)

    if overrides:
        merged = _recursive_merge(merged, t.cast(dict, overrides))
    return merged


def _as_overrides(overrides: dict | pathlib.Path | None) -> dict:
    """Return the overrides as a plain mapping, loading a YAML path when given."""
    if overrides is None:
        return {}
    if isinstance(overrides, pathlib.Path):
        with overrides.open() as handle:
            loaded = yaml.safe_load(handle)
        return loaded or {}
    if not isinstance(overrides, dict):
        msg = f'`overrides` must be a mapping or a YAML path, got `{type(overrides).__name__}`.'
        raise ValueError(msg)
    return overrides


def _reject_node_overrides(mapping: t.Any, path: str = 'overrides') -> None:
    """Fail fast when overrides carry ORM nodes, which must travel as helper keywords."""
    if isinstance(mapping, orm.Node):
        msg = (
            f'`{path}` holds an ORM node; pass nodes as `get_inputs_from_protocol` keywords '
            '(`pseudos`, `kpoints`, `parent_folder`, ...) instead of inside `overrides`.'
        )
        raise ValueError(msg)
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            _reject_node_overrides(value, f'{path}.{key}')


def _reject_legacy_knobs(mapping: t.Any, path: str = 'overrides') -> None:
    """Fail fast when overrides request legacy behavior the native graphs do not offer."""
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if key in _REJECTED_OVERRIDES:
                msg = f'`{path}.{key}` is not supported: {_REJECTED_OVERRIDES[key]}'
                raise ValueError(msg)
            _reject_legacy_knobs(value, f'{path}.{key}')


def _check_allowed(mapping: dict, allowed: frozenset[str], where: str) -> None:
    """Fail fast on override keys with no native-graph consumer."""
    for key in mapping:
        if key not in allowed:
            msg = f'`{where}.{key}` is not a supported protocol option for the native graphs.'
            raise ValueError(msg)


def _coerce_monitor_entry(name: t.Any, value: t.Any, where: str) -> orm.Dict:
    """Validate one monitor entry and return it as an unstored ``Dict``."""
    if str(name) != _STUCK_MONITOR_KEY:
        msg = f'`{where}.{name}` is not supported; only the `{_STUCK_MONITOR_KEY}` monitor key is wired structurally.'
        raise ValueError(msg)
    if not isinstance(value, (dict, orm.Dict)):
        msg = f'`{where}.{name}` must map to a monitor mapping, got `{type(value).__name__}`.'
        raise ValueError(msg)
    mapping = dict(value.get_dict() if isinstance(value, orm.Dict) else value)
    allowed = {'entry_point', 'kwargs'}
    for key in mapping:
        if key not in allowed:
            msg = f'`{where}.{name}.{key}` is not supported; expected one of {sorted(allowed)}.'
            raise ValueError(msg)
    entry_point = mapping.get('entry_point')
    if entry_point != _KNOWN_MONITOR_ENTRY_POINT:
        msg = (
            f'`{where}.{name}.entry_point` must be `{_KNOWN_MONITOR_ENTRY_POINT}`, got `{entry_point!r}`: '
            'only the accuracy-stuck monitor is wired structurally.'
        )
        raise ValueError(msg)
    kwargs = mapping.get('kwargs', {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        msg = f'`{where}.{name}.kwargs` must be a mapping, got `{type(kwargs).__name__}`.'
        raise ValueError(msg)
    for key in kwargs:
        if key not in _KNOWN_MONITOR_KWARGS:
            msg = f'`{where}.{name}.kwargs.{key}` is not supported; expected one of {sorted(_KNOWN_MONITOR_KWARGS)}.'
            raise ValueError(msg)
    if 'min_repeats' in kwargs:
        min_repeats = kwargs['min_repeats']
        if isinstance(min_repeats, bool) or not isinstance(min_repeats, int):
            msg = f'`{where}.{name}.kwargs.min_repeats` must be an `int`, got `{type(min_repeats).__name__}`.'
            raise ValueError(msg)
        if min_repeats <= 0:
            msg = f'`{where}.{name}.kwargs.min_repeats` must be positive, got `{min_repeats!r}`.'
            raise ValueError(msg)
    if 'structured' in kwargs and not isinstance(kwargs['structured'], bool):
        msg = f'`{where}.{name}.kwargs.structured` must be a `bool`, got `{type(kwargs["structured"]).__name__}`.'
        raise ValueError(msg)
    resolved_kwargs = dict(kwargs)
    resolved_kwargs.setdefault('structured', True)
    if resolved_kwargs['structured'] is not True:
        msg = f'`{where}.{name}.kwargs.structured` must be `True`: unstructured stops carry no diagnostics.'
        raise ValueError(msg)
    return orm.Dict({'entry_point': entry_point, 'kwargs': resolved_kwargs})


def _monitors_from_yaml(value: t.Any, where: str) -> orm.DataDict | None:
    """Convert a ``pw.monitors`` YAML mapping into an unstored ``DataDict``."""
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = f'`{where}` must map monitor names onto monitor entries, got `{type(value).__name__}`.'
        raise ValueError(msg)
    if not value:
        return None
    entries = {str(name): _coerce_monitor_entry(str(name), entry, where) for name, entry in value.items()}
    return orm.DataDict(entries)


def _coerce_monitors_input(value: t.Any, name: str) -> orm.DataDict | None:
    """Coerce an explicit monitors boundary input into an unstored ``DataDict``."""
    if value is None:
        return None
    if isinstance(value, orm.DataDict):
        normalized = {}
        for label, node in value.items():
            if not isinstance(node, orm.Dict):
                msg = f'`{name}[{label!r}]` must be a `Dict`, got `{type(node).__name__}`.'
                raise ValueError(msg)
            normalized[label] = _coerce_monitor_entry(label, node, name)
        return orm.DataDict(normalized)
    if isinstance(value, dict):
        if not value:
            return None
        entries = {str(label): _coerce_monitor_entry(str(label), entry, name) for label, entry in value.items()}
        return orm.DataDict(entries)
    msg = f'`{name}` must be a `DataDict` or a plain mapping, got `{type(value).__name__}`.'
    raise ValueError(msg)


def _as_enum(value: t.Any, enum_type: type[enum.Enum], name: str) -> enum.Enum:
    """Return the enum member for an enum or its value, failing fast otherwise."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        valid = sorted(member.value for member in enum_type)
        msg = f'`{name}` must be one of {valid}, got `{value}`.'
        raise ValueError(msg) from None


def _resolve_code(code: t.Any, name: str) -> orm.Code:
    """Return the code node for a code or its label, failing fast otherwise."""
    if isinstance(code, str):
        code = orm.load_code(code)
    if not isinstance(code, orm.Code):
        msg = f'`{name}` must be a code or a code label, got `{type(code).__name__ if code is not None else None}`.'
        raise ValueError(msg)
    return code


def _normalize_pseudos(pseudos: dict | orm.DataDict, structure: orm.StructureData) -> dict[str, orm.Node]:
    """Return the explicit pseudos as a kind-to-node mapping after validating coverage."""
    if isinstance(pseudos, (orm.DataDict, dict)):
        mapping = dict(pseudos)
    else:
        msg = f'`pseudos` must map kind names to pseudo nodes, got `{type(pseudos).__name__}`.'
        raise ValueError(msg)
    if sorted(mapping) != sorted(structure.get_kind_names()):
        msg = f'`pseudos` needs one value for each of the {len(structure.kinds)} kinds.'
        raise ValueError(msg)
    for kind, node in mapping.items():
        if not isinstance(node, orm.Node):
            msg = f'`pseudos[{kind!r}]` must be a pseudo node, got `{type(node).__name__}`.'
            raise ValueError(msg)
        if getattr(node, 'z_valence', None) is None:
            msg = f'`pseudos[{kind!r}]` exposes no `z_valence`; spin handling needs it.'
            raise ValueError(msg)
    return mapping


def _query_pseudo_family(label: str) -> t.Any:
    """Return the installed pseudo family for a label, failing fast when missing or ambiguous."""
    PseudoPotentialFamily = GroupFactory('pseudo.family')
    query = orm.QueryBuilder().append(PseudoPotentialFamily, filters={'label': label})
    try:
        return query.one()[0]
    except exceptions.NotExistent as exception:
        msg = (
            f'required pseudo family `{label}` is not installed. Use `aiida-pseudo install` to install it, '
            'or pass explicit `pseudos`.'
        )
        raise ValueError(msg) from exception
    except exceptions.MultipleObjectsError as exception:
        matches = ', '.join(f'`{family}`' for family in query.all(flat=True))
        msg = (
            f'the label `{label}` matches more than one installed pseudo family: {matches}. '
            'Delete or relabel all but one, or pass explicit `pseudos` instead.'
        )
        raise ValueError(msg) from exception


def _family_cutoffs(family: t.Any, structure: orm.StructureData) -> tuple[float, float]:
    """Return the recommended cutoffs of a pseudo family in Ry, failing fast when unavailable."""
    try:
        from aiida_pseudo.groups.mixins import RecommendedCutoffMixin
    except ImportError as exception:
        msg = 'resolving pseudo cutoffs needs the `aiida-pseudo` package to be installed.'
        raise ValueError(msg) from exception
    if not isinstance(family, RecommendedCutoffMixin):
        msg = (
            f'pseudo family `{family.label}` cannot recommend cutoffs. Provide both `ecutwfc` and '
            '`ecutrho` in the `overrides`, or use a family that recommends them.'
        )
        raise ValueError(msg)
    try:
        return family.get_recommended_cutoffs(structure=structure, unit='Ry')
    except ValueError as exception:
        msg = (
            f'failed to obtain recommended cutoffs for pseudo family `{family.label}`: {exception} '
            'If the family does not define any, specify both `ecutwfc` and `ecutrho` in the `overrides`.'
        )
        raise ValueError(msg) from exception


def _resolve_section_pseudos(
    *,
    structure: orm.StructureData,
    family_label: str,
    explicit: dict[str, orm.Node] | None,
    parameters: dict,
    section_path: str,
) -> dict[str, orm.Node]:
    """Return the pseudo nodes for one base section and ensure its cutoffs are set.

    Explicit nodes win and require both cutoffs in the section parameters;
    otherwise the family recommends them, or the section overrides must carry
    them.
    """
    system = parameters.setdefault('SYSTEM', {})
    if explicit is not None:
        if 'ecutwfc' not in system or 'ecutrho' not in system:
            msg = (
                'When passing explicit `pseudos`, both `ecutwfc` and `ecutrho` cutoffs should be '
                f'provided in the `{section_path}.pw.parameters.SYSTEM` overrides.'
            )
            raise ValueError(msg)
        return dict(explicit)
    if 'ecutwfc' not in system or 'ecutrho' not in system:
        family = _query_pseudo_family(family_label)
        system['ecutwfc'], system['ecutrho'] = _family_cutoffs(family, structure)
    else:
        family = _query_pseudo_family(family_label)
    return {kind.name: family.get_pseudos(structure=structure)[kind.name] for kind in structure.kinds}


def _apply_science_parameters(
    *,
    parameters: dict,
    structure: orm.StructureData,
    pseudos: dict[str, orm.Node],
    meta_parameters: dict,
    electronic_type: ElectronicType,
    spin_type: SpinType,
    initial_magnetic_moments: dict | None,
    absolute_overrides: dict,
) -> dict:
    """Apply thresholds, periodicity, and electronic/spin settings, with caller overrides absolute last."""
    try:
        conv_thr = meta_parameters['conv_thr_per_atom']
        etot_conv_thr = meta_parameters['etot_conv_thr_per_atom']
    except KeyError as exception:
        msg = f'`meta_parameters` must define `conv_thr_per_atom` and `etot_conv_thr_per_atom`: {meta_parameters}.'
        raise ValueError(msg) from exception
    natoms = len(structure.sites)
    parameters.setdefault('CONTROL', {})['etot_conv_thr'] = natoms * etot_conv_thr
    parameters.setdefault('ELECTRONS', {})['conv_thr'] = natoms * conv_thr
    parameters.setdefault('SYSTEM', {})

    pbc = structure.pbc
    if pbc != (True, True, True):
        if pbc.count(True) == 2 and pbc != (True, True, False):
            msg = f'2D-periodic structures must be periodic in the x-y plane, got `{pbc}`.'
            raise ValueError(msg)
        warnings.warn(
            'This protocol was developed for fully periodic (i.e. 3D) systems. Use `overrides` to provide '
            'any relevant keywords for handling aperiodicity, and proceed with caution.'
        )
    if pbc == (True, True, False):
        parameters['SYSTEM'] = {**parameters['SYSTEM'], 'assume_isolated': '2D'}

    if electronic_type is ElectronicType.INSULATOR:
        parameters['SYSTEM']['occupations'] = 'fixed'
        parameters['SYSTEM'].pop('smearing', None)
        parameters['SYSTEM'].pop('degauss', None)

    if spin_type is not SpinType.NONE:
        magnetization = get_magnetization(
            structure=structure,
            z_valences={kind.name: pseudos[kind.name].z_valence for kind in structure.kinds},
            initial_magnetic_moments=initial_magnetic_moments,
            spin_type=spin_type,
        )
        parameters['SYSTEM']['starting_magnetization'] = magnetization['starting_magnetization']
        if spin_type is SpinType.COLLINEAR:
            parameters['SYSTEM']['nspin'] = 2
        if spin_type in (SpinType.SPIN_ORBIT, SpinType.NON_COLLINEAR):
            parameters['SYSTEM']['angle1'] = magnetization['angle1']
            parameters['SYSTEM']['angle2'] = magnetization['angle2']
            parameters['SYSTEM']['noncolin'] = True
            parameters['SYSTEM']['nspin'] = 4
        if spin_type is SpinType.SPIN_ORBIT:
            parameters['SYSTEM']['lspinorb'] = True

    if absolute_overrides:
        parameters = _recursive_merge(parameters, absolute_overrides)
        if parameters.get('SYSTEM', {}).get('tot_magnetization') is not None:
            parameters.setdefault('SYSTEM', {}).pop('starting_magnetization', None)
    return parameters


def _resolve_base_section(
    *,
    section: dict,
    section_path: str,
    base_defaults: dict,
    structure: orm.StructureData,
    family_label: str,
    explicit_pseudos: dict[str, orm.Node] | None,
    electronic_type: ElectronicType,
    spin_type: SpinType,
    initial_magnetic_moments: dict | None,
    absolute_param_overrides: dict,
) -> dict:
    """Resolve one ``pw.x`` base section into graph-ready pieces.

    Merges the section over the base protocol defaults, resolves pseudos and
    cutoffs, applies per-atom thresholds and electronic/spin settings, and
    splits scheduler options out of the graph inputs.
    """
    _check_allowed(section, _BASE_SECTION_KEYS, section_path)
    merged = _recursive_merge(base_defaults, section)
    if 'clean_workdir' in section:
        msg = f'`{section_path}.clean_workdir` is ignored; set the root-level `clean_workdir` instead.'
        raise ValueError(msg)
    max_iterations = merged.get('max_iterations', _BASE_MAX_ITERATIONS)
    if max_iterations != _BASE_MAX_ITERATIONS:
        msg = (
            f'`{section_path}.max_iterations` is `{max_iterations}` but the native base graph always '
            f'runs an inner budget of {_BASE_MAX_ITERATIONS}; only the outer budget is exposed.'
        )
        raise ValueError(msg)

    meta_parameters = merged.get('meta_parameters', {})
    pw = merged.get('pw', {})
    if 'parallelization' in pw:
        msg = f'`{section_path}.pw.parallelization` is not supported: the native base graph has no port for it.'
        raise ValueError(msg)
    _check_allowed(pw, _BASE_PW_KEYS, f'{section_path}.pw')
    metadata = pw.get('metadata', {})
    _check_allowed(metadata, frozenset({'options'}), f'{section_path}.pw.metadata')
    parameters = copy.deepcopy(pw.get('parameters', {}))
    parameters.setdefault('CONTROL', {})
    parameters.setdefault('ELECTRONS', {})
    parameters.setdefault('SYSTEM', {})

    pseudos = _resolve_section_pseudos(
        structure=structure,
        family_label=family_label,
        explicit=explicit_pseudos,
        parameters=parameters,
        section_path=section_path,
    )
    parameters = _apply_science_parameters(
        parameters=parameters,
        structure=structure,
        pseudos=pseudos,
        meta_parameters=meta_parameters,
        electronic_type=electronic_type,
        spin_type=spin_type,
        initial_magnetic_moments=initial_magnetic_moments,
        absolute_overrides=copy.deepcopy(absolute_param_overrides),
    )
    return {
        'parameters': parameters,
        'settings': copy.deepcopy(pw.get('settings')),
        'metadata_options': copy.deepcopy(metadata.get('options', {})),
        'kpoints_distance': merged.get('kpoints_distance'),
        'kpoints_force_parity': bool(merged.get('kpoints_force_parity', False)),
        'pseudos': pseudos,
        'monitors': _monitors_from_yaml(pw.get('monitors'), f'{section_path}.pw.monitors'),
    }


def _resolve_family_label(
    *,
    kind_sections: dict[str, dict],
    base_default: str,
    pseudo_family: str | None,
    spin_type: SpinType,
) -> str:
    """Return the single pseudo family shared by every base section of one launch."""
    if pseudo_family is not None:
        return pseudo_family
    labels = {section.get('pseudo_family', base_default) for section in kind_sections.values()}
    if len(labels) > 1:
        msg = (
            f'base sections resolve different pseudo families {sorted(labels)} but the native graphs '
            'share one `pseudos` input; pass a single `pseudo_family` keyword instead.'
        )
        raise ValueError(msg)
    label = next(iter(labels))
    if spin_type is SpinType.SPIN_ORBIT and not any('pseudo_family' in section for section in kind_sections.values()):
        return 'PseudoDojo/0.4/PBEsol/FR/standard/upf'
    return label


def _placement_metadata(section_options: dict, global_options: dict | None, code: orm.Code | None) -> dict:
    """Merge section scheduler options with global ones and scheduler defaults for one placement."""
    merged = _recursive_merge(copy.deepcopy(section_options), copy.deepcopy(global_options or {}))
    computer = code.computer if code is not None else None
    if computer is not None and computer.scheduler_type:
        merged = ProtocolMixin.set_default_resources(merged, computer.scheduler_type)
    return {'metadata': {'options': merged}}


def _pw_scope_metadata(section_options: dict, global_options: dict | None, code: orm.Code | None) -> dict:
    """Return the scheduler metadata for the single PW placement."""
    return {'pw_calc': _placement_metadata(section_options, global_options, code)}


def _normalize_pw_leaves(node: t.Any, path: str = 'runtime_config') -> t.Any:
    """Translate old monitored placement keys in QE protocol overrides."""
    if not isinstance(node, dict):
        return node
    scope = {key: _normalize_pw_leaves(value, f'{path}.{key}') for key, value in node.items()}
    if 'pw_calc_monitored' in scope:
        legacy = scope.pop('pw_calc_monitored')
        if 'pw_calc' in scope and scope['pw_calc'] != legacy:
            msg = f'`{path}` holds conflicting `pw_calc` and `pw_calc_monitored` overrides.'
            raise ValueError(msg)
        scope['pw_calc'] = legacy
    return scope


def _placement_options(runtime_config: dict | None, *path: str) -> dict:
    """Return the scheduler ``options`` leaf of a merged ``runtime_config`` mapping at one placement path."""
    node: t.Any = runtime_config or {}
    for key in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    if not isinstance(node, dict):
        return {}
    metadata = node.get('metadata', {})
    options = metadata.get('options', {}) if isinstance(metadata, dict) else {}
    return dict(options) if isinstance(options, dict) else {}


def _ensure_max_seconds(
    parameters: dict, options: dict | None, *, where: str = 'placement', namelist: str = 'CONTROL'
) -> dict:
    """Derive ``CONTROL.max_seconds`` from a fixed scheduler wallclock, preserving an explicit value.

    Mirrors ``PwBaseWorkChain.prepare_process`` without per-attempt scheduler metadata: when
    ``CONTROL.max_seconds`` is already set (YAML default, caller override) it wins; otherwise a
    present ``options['max_wallclock_seconds']`` seeds ``0.95`` of itself so the QE cutoff stops
    the SCF loop just before the scheduler kills the job. No wallclock means no cutoff.

    A present but non-numeric wallclock is an actionable error, not a silent no-op: the
    ``CalcJob`` spec requires ``metadata.options.max_wallclock_seconds`` to be an ``int``,
    so an invalid limit would fail loudly at launch while a silently skipped cutoff would
    leave the run without the QE stop that the wallclock promised.
    """
    control = parameters.setdefault(namelist, {})
    if 'max_seconds' in control:
        return parameters
    if not options:
        return parameters
    wallclock = options.get('max_wallclock_seconds')
    if wallclock is None:
        return parameters
    if isinstance(wallclock, bool):
        msg = (
            f'`{where}` holds `max_wallclock_seconds={wallclock!r}`: the wallclock must be a '
            'positive number of seconds, matching `metadata.options.max_wallclock_seconds`.'
        )
        raise ValueError(msg)
    try:
        wallclock_value = float(wallclock)
    except (TypeError, ValueError):
        msg = (
            f'`{where}` holds `max_wallclock_seconds={wallclock!r}`: the wallclock must be a '
            'positive number of seconds, matching `metadata.options.max_wallclock_seconds`.'
        )
        raise ValueError(msg) from None
    if not math.isfinite(wallclock_value) or wallclock_value <= 0:
        msg = (
            f'`{where}` holds `max_wallclock_seconds={wallclock!r}`: the wallclock must be a '
            'positive number of seconds, matching `metadata.options.max_wallclock_seconds`.'
        )
        raise ValueError(msg)
    control['max_seconds'] = wallclock_value * _DELTA_FACTOR_MAX_SECONDS
    return parameters


def _apply_relax_type(
    *,
    parameters: dict,
    init_parameters: dict,
    settings: dict | None,
    init_settings: dict | None,
    structure: orm.StructureData,
    relax_type: RelaxType,
) -> tuple[dict | None, dict | None]:
    """Map a relax type onto calculation modes, cell constraints, and fixed coordinates per stage."""
    if relax_type in (RelaxType.VOLUME, RelaxType.POSITIONS_VOLUME):
        msg = f'relax type `{relax_type.value}` is not supported.'
        raise ValueError(msg)
    for params in (parameters, init_parameters):
        if relax_type is RelaxType.NONE:
            params['CONTROL']['calculation'] = 'scf'
            params.pop('CELL', None)
        elif relax_type is RelaxType.POSITIONS:
            params['CONTROL']['calculation'] = 'relax'
            params.pop('CELL', None)
        else:
            params['CONTROL']['calculation'] = 'vc-relax'
    cell = parameters.setdefault('CELL', {})
    init_cell = init_parameters.setdefault('CELL', {})
    if relax_type in (RelaxType.SHAPE, RelaxType.POSITIONS_SHAPE):
        cell['cell_dofree'] = 'shape'
        init_cell['cell_dofree'] = 'shape'
    if relax_type in (RelaxType.CELL, RelaxType.POSITIONS_CELL):
        pbc_map = {
            (True, True, True): 'all',
            (True, False, False): 'x',
            (False, True, False): 'y',
            (False, False, True): 'z',
            (True, True, False): '2Dxy',
        }
        if structure.pbc not in pbc_map:
            msg = f'Structures with periodic boundary conditions `{structure.pbc}` are not supported.'
            raise ValueError(msg)
        cell['cell_dofree'] = pbc_map[structure.pbc]
        init_cell['cell_dofree'] = pbc_map[structure.pbc]
    if parameters['CONTROL']['calculation'] == 'scf':
        parameters.pop('CELL', None)
    if init_parameters['CONTROL']['calculation'] == 'scf':
        init_parameters.pop('CELL', None)
    if relax_type in (RelaxType.VOLUME, RelaxType.SHAPE, RelaxType.CELL):

        def _with_fixed(mapping: dict | None) -> dict:
            fixed = dict(mapping) if mapping is not None else {}
            fixed['FIXED_COORDS'] = [[True, True, True]] * len(structure.sites)
            return fixed

        return _with_fixed(settings), _with_fixed(init_settings)
    return settings, init_settings


def _as_nodes(*, parameters: dict, settings: dict | None, distance: float | None) -> tuple:
    """Wrap resolved plain mappings into unstored graph input nodes."""
    nodes: tuple = (orm.Dict(parameters),)
    nodes += (orm.Dict(settings),) if settings is not None else (None,)
    nodes += (orm.Float(distance),) if distance is not None else (None,)
    return nodes


def get_inputs_from_protocol(
    kind: str,
    *,
    structure: orm.StructureData,
    protocol: str | None = None,
    overrides: dict | pathlib.Path | None = None,
    code: t.Any = None,
    scf_code: t.Any = None,
    bands_code: t.Any = None,
    pw_code: t.Any = None,
    dos_code: t.Any = None,
    projwfc_code: t.Any = None,
    pseudos: dict | orm.DataDict | None = None,
    pseudo_family: str | None = None,
    electronic_type: ElectronicType | str = ElectronicType.METAL,
    spin_type: SpinType | str = SpinType.NONE,
    initial_magnetic_moments: dict | None = None,
    relax_type: RelaxType | str = RelaxType.POSITIONS_CELL,
    kpoints: orm.KpointsData | None = None,
    init_kpoints: orm.KpointsData | None = None,
    scf_kpoints: orm.KpointsData | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    nscf_kpoints: orm.KpointsData | None = None,
    parent_folder: orm.RemoteData | None = None,
    parent_output_parameters: orm.Dict | dict | None = None,
    serial: bool = False,
    options: dict | None = None,
    runtime_config: dict | None = None,
    monitors: orm.DataDict | dict | None = None,
    scf_monitors: orm.DataDict | dict | None = None,
    bands_monitors: orm.DataDict | dict | None = None,
    nscf_monitors: orm.DataDict | dict | None = None,
    init_monitors: orm.DataDict | dict | None = None,
) -> dict:
    """Translate protocol YAML nesting into launch-ready native graph inputs.

    The returned mapping splats directly into :func:`run_get_node`/:func:`submit`
    for the matching production root (``pw_relax_graph``, ``pw_bands_graph``,
    or a ``pdos_from_*`` root): graph arguments as plain values and ORM nodes,
    plus a strictly nested ``runtime_config`` entry (one level per authored
    subgraph callee alias, the ``CalcJob`` metadata mapping at each leaf).
    Pseudo, cutoff,
    threshold, k-point, and spin resolution mirrors
    ``PwBaseWorkChain.get_builder_from_protocol`` without importing it.

    :param kind: one of ``'pw_relax'``, ``'pw_bands'``, ``'pdos'``.
    :param structure: the structure to launch on.
    :param protocol: the protocol name or alias; defaults to the YAML default.
    :param overrides: plain-data overrides with the YAML nesting, or a YAML path. ORM nodes are
        rejected here; pass them as keywords.
    :param code: the ``quantumespresso.pw`` code for ``'pw_relax'``.
    :param scf_code: the SCF ``quantumespresso.pw`` code for ``'pw_bands'``.
    :param bands_code: the bands ``quantumespresso.pw`` code for ``'pw_bands'``.
    :param pw_code: the ``quantumespresso.pw`` code for ``'pdos'``.
    :param dos_code: the ``quantumespresso.dos`` code for ``'pdos'``.
    :param projwfc_code: the ``quantumespresso.projwfc`` code for ``'pdos'``.
    :param pseudos: explicit mapping of kind names onto pseudo nodes, shared by every section.
    :param pseudo_family: pseudo family label shared by every section; defaults to the protocol value.
    :param electronic_type: ``ElectronicType.METAL`` or ``ElectronicType.INSULATOR``.
    :param spin_type: the requested ``SpinType``; resolved through :func:`get_magnetization`.
    :param initial_magnetic_moments: per-kind initial moments for spin-polarized runs.
    :param relax_type: the ``RelaxType`` for ``'pw_relax'``.
    :param kpoints: explicit mesh for the main relax run; never overwrites the initial density.
    :param init_kpoints: explicit mesh for the initial relax run, if any.
    :param scf_kpoints: explicit SCF mesh for ``'pw_bands'`` or ``'pdos'``.
    :param bands_kpoints: explicit bands path for ``'pw_bands'``; conflicts with the protocol distance.
    :param nscf_kpoints: explicit NSCF mesh for ``'pdos'``.
    :param parent_folder: when given for ``'pdos'``, build from-parent inputs instead of from-scratch.
    :param parent_output_parameters: parent ``output_parameters`` for the PDOS band-count rule.
    :param serial: select the serial PDOS root (and its ``runtime_config`` stage scope).
    :param options: scheduler options merged into every CalcJob placement.
    :param runtime_config: explicit placement metadata merged last, winning on conflicts.
    :param monitors: shared monitors forwarded into every nested ``pw_base_graph`` unless a
        per-stage override wins.
    :param scf_monitors: optional SCF monitors for ``'pw_bands'``/``'pdos'``, defaulting to ``monitors``.
    :param bands_monitors: optional bands monitors for ``'pw_bands'``, defaulting to ``monitors``.
    :param nscf_monitors: optional NSCF monitors for ``'pdos'``, defaulting to ``monitors``.
    :param init_monitors: optional initial-relax monitors for ``'pw_relax'``, defaulting to ``monitors``.
    :return: the launch inputs, including ``runtime_config``.
    :raises ValueError: for unsupported protocol options, conflicting shared inputs, or missing codes.
    """
    if kind not in PROTOCOL_KINDS:
        msg = f'Unknown protocol kind {kind!r}; expected one of {sorted(PROTOCOL_KINDS)}.'
        raise ValueError(msg)
    if not isinstance(structure, orm.StructureData):
        msg = f'`structure` must be a `StructureData`, got `{type(structure).__name__}`.'
        raise ValueError(msg)
    electronic_type = _as_enum(electronic_type, ElectronicType, 'electronic_type')
    spin_type = _as_enum(spin_type, SpinType, 'spin_type')
    if electronic_type not in (ElectronicType.METAL, ElectronicType.INSULATOR):
        msg = f'electronic type `{electronic_type.value}` is not supported.'
        raise ValueError(msg)
    if initial_magnetic_moments is not None and spin_type is SpinType.NONE:
        msg = '`initial_magnetic_moments` is specified but spin type `SpinType.NONE` is incompatible.'
        raise ValueError(msg)

    raw_overrides = _as_overrides(overrides)
    _reject_node_overrides(raw_overrides)
    _reject_legacy_knobs(raw_overrides)
    merged = get_protocol_inputs(kind, protocol, raw_overrides if raw_overrides else None)
    _check_allowed(merged, _TOP_LEVEL_KEYS[kind], kind)
    if kind != 'pdos' and merged.get('clean_workdir', False):
        msg = (
            f'`{kind}.clean_workdir=True` has no in-graph consumer; '
            'call cleanup_graph_workdirs(node) explicitly after the graph terminates.'
        )
        raise ValueError(msg)
    if kind != 'pw_relax':
        for section in ('scf', 'bands', 'nscf'):
            if (merged.get(section) or {}).get('kpoints_force_parity', False):
                msg = f'`{kind}.{section}.kpoints_force_parity=True` is not routed by this composite graph.'
                raise ValueError(msg)
    resolved_protocol = protocol or get_default_protocol(kind)
    base_defaults, _ = _load_base_protocol(_ALIASES.get(resolved_protocol, resolved_protocol))

    explicit_pseudos = _normalize_pseudos(pseudos, structure) if pseudos is not None else None
    for label, nodes in (
        ('kpoints', kpoints),
        ('init_kpoints', init_kpoints),
        ('scf_kpoints', scf_kpoints),
        ('bands_kpoints', bands_kpoints),
        ('nscf_kpoints', nscf_kpoints),
    ):
        if nodes is not None and not isinstance(nodes, orm.KpointsData):
            msg = f'`{label}` must be a `KpointsData` or `None`, got `{type(nodes).__name__}`.'
            raise ValueError(msg)
    if parent_folder is not None and not isinstance(parent_folder, orm.RemoteData):
        msg = f'`parent_folder` must be a `RemoteData` or `None`, got `{type(parent_folder).__name__}`.'
        raise ValueError(msg)
    if parent_output_parameters is not None and not isinstance(parent_output_parameters, (orm.Dict, dict)):
        msg = (
            '`parent_output_parameters` must be a `Dict` or a plain mapping, '
            f'got `{type(parent_output_parameters).__name__}`.'
        )
        raise ValueError(msg)

    if kind == 'pw_relax':
        return _relax_launch_inputs(
            merged=merged,
            raw_overrides=raw_overrides,
            base_defaults=base_defaults,
            structure=structure,
            protocol_code=_resolve_code(code, 'code') if code is not None else None,
            explicit_pseudos=explicit_pseudos,
            pseudo_family=pseudo_family,
            electronic_type=electronic_type,
            spin_type=spin_type,
            initial_magnetic_moments=initial_magnetic_moments,
            relax_type=_as_enum(relax_type, RelaxType, 'relax_type'),
            kpoints=kpoints,
            init_kpoints=init_kpoints,
            options=options,
            runtime_config=runtime_config,
            monitors=_coerce_monitors_input(monitors, 'monitors'),
            init_monitors=_coerce_monitors_input(init_monitors, 'init_monitors'),
        )
    if kind == 'pw_bands':
        return _bands_launch_inputs(
            merged=merged,
            raw_overrides=raw_overrides,
            base_defaults=base_defaults,
            structure=structure,
            scf_code=_resolve_code(scf_code, 'scf_code') if scf_code is not None else None,
            bands_code=_resolve_code(bands_code, 'bands_code') if bands_code is not None else None,
            explicit_pseudos=explicit_pseudos,
            pseudo_family=pseudo_family,
            electronic_type=electronic_type,
            spin_type=spin_type,
            initial_magnetic_moments=initial_magnetic_moments,
            scf_kpoints=scf_kpoints,
            bands_kpoints=bands_kpoints,
            options=options,
            runtime_config=runtime_config,
            monitors=_coerce_monitors_input(monitors, 'monitors'),
            scf_monitors=_coerce_monitors_input(scf_monitors, 'scf_monitors'),
            bands_monitors=_coerce_monitors_input(bands_monitors, 'bands_monitors'),
        )
    return _pdos_launch_inputs(
        merged=merged,
        raw_overrides=raw_overrides,
        base_defaults=base_defaults,
        structure=structure,
        pw_code=_resolve_code(pw_code, 'pw_code') if pw_code is not None else None,
        dos_code=_resolve_code(dos_code, 'dos_code') if dos_code is not None else None,
        projwfc_code=_resolve_code(projwfc_code, 'projwfc_code') if projwfc_code is not None else None,
        explicit_pseudos=explicit_pseudos,
        pseudo_family=pseudo_family,
        electronic_type=electronic_type,
        spin_type=spin_type,
        initial_magnetic_moments=initial_magnetic_moments,
        scf_kpoints=scf_kpoints,
        nscf_kpoints=nscf_kpoints,
        parent_folder=parent_folder,
        parent_output_parameters=parent_output_parameters,
        serial=serial,
        options=options,
        runtime_config=runtime_config,
        monitors=_coerce_monitors_input(monitors, 'monitors'),
        scf_monitors=_coerce_monitors_input(scf_monitors, 'scf_monitors'),
        nscf_monitors=_coerce_monitors_input(nscf_monitors, 'nscf_monitors'),
    )


def prepare_run_from_protocol(
    kind: str,
    *,
    protocol: str | None = None,
    overrides: dict | pathlib.Path | None = None,
    execution_policy: dict | None = None,
    **inputs: t.Any,
) -> PreparedRun:
    """Resolve a native QE protocol and attach its execution policy and provenance.

    :param kind: Graph protocol family, e.g. ``'pw_bands'``.
    :param protocol: Protocol name, defaulting to the YAML default.
    :param overrides: Explicit scientific protocol overrides, or a YAML path.
    :param execution_policy: Opt-in policies keyed by the same placements as runtime configuration.
    :param inputs: Arguments accepted by :func:`get_inputs_from_protocol`.
    :return: A payload for launching the matching native graph with ``prepared_run=``.
    """
    name = get_default_protocol(kind) if protocol is None else protocol
    raw_overrides = _as_overrides(overrides)
    launch = get_inputs_from_protocol(kind, protocol=name, overrides=raw_overrides, **inputs)
    resolved_defaults = get_protocol_inputs(kind, name)
    digest = hashlib.sha256(yaml.safe_dump(resolved_defaults, sort_keys=True).encode()).hexdigest()
    runtime_config = launch.pop('runtime_config')
    return PreparedRun(
        inputs=launch,
        execution_environment=ExecutionEnvironment(
            runtime_config=runtime_config, execution_policy=execution_policy or {}
        ),
        protocol=f'quantumespresso.{kind}.{name}',
        version=f'sha256:{digest}',
        overrides=raw_overrides,
    )


def prepare_ph_inputs(
    *,
    code: orm.Code,
    parent_folder: orm.RemoteData,
    parameters: orm.Dict | dict,
    options: dict | None = None,
    structure: orm.StructureData | None = None,
    qpoints: orm.KpointsData | None = None,
    qpoints_distance: float | None = None,
    **inputs: t.Any,
) -> dict:
    """Prepare fixed scheduler and q-point inputs for ``ph_base_graph``.

    :param code: Executable for the PH calculation.
    :param parent_folder: Parent calculation folder.
    :param parameters: Calculation parameters; explicit ``INPUTPH.max_seconds`` is retained.
    :param options: Scheduler options for the PH placement.
    :param structure: Explicit structure for generating the mesh; otherwise use the parent structure.
    :param qpoints: Explicit q-points, mutually exclusive with ``qpoints_distance``.
    :param qpoints_distance: Target mesh spacing.
    :param inputs: Additional PH graph inputs.
    :return: A mapping ready to launch ``ph_base_graph``.
    """
    if (qpoints is None) == (qpoints_distance is None):
        msg = 'Specify exactly one of `qpoints` and `qpoints_distance`.'
        raise ValueError(msg)
    if qpoints_distance is not None and structure is None:
        creator = parent_folder.creator
        if creator is not None:
            structure = getattr(creator.outputs, 'output_structure', None)
            if structure is None:
                structure = getattr(creator.inputs, 'structure', None)
        if not isinstance(structure, orm.StructureData):
            msg = 'Cannot resolve the PH mesh structure from the parent folder; pass `structure` explicitly.'
            raise ValueError(msg)
    metadata = _placement_metadata({}, options, code)
    prepared_parameters = parameters.get_dict() if isinstance(parameters, orm.Dict) else copy.deepcopy(parameters)
    _ensure_max_seconds(prepared_parameters, metadata['metadata']['options'], where='ph_calc', namelist='INPUTPH')
    return {
        **inputs,
        'code': code,
        'parent_folder': parent_folder,
        'parameters': prepared_parameters,
        'structure': structure,
        'qpoints': qpoints,
        'qpoints_distance': qpoints_distance,
        'runtime_config': {'ph_calc': metadata},
    }


def _section_param_overrides(raw_overrides: dict, *path: str) -> dict:
    """Return the absolute ``pw.parameters`` overrides at a nested override path, if any."""
    node: t.Any = raw_overrides
    for key in (*path, 'pw', 'parameters'):
        if not isinstance(node, dict) or key not in node:
            return {}
        node = node[key]
    return copy.deepcopy(node) if isinstance(node, dict) else {}


def _resolve_sections(
    *,
    names: tuple[str, ...],
    merged: dict,
    raw_overrides: dict,
    base_defaults: dict,
    structure: orm.StructureData,
    family_label: str,
    explicit_pseudos: dict[str, orm.Node] | None,
    electronic_type: ElectronicType,
    spin_type: SpinType,
    initial_magnetic_moments: dict | None,
) -> dict[str, dict]:
    """Resolve every named base section of one launch with one shared pseudo set."""
    return {
        name: _resolve_base_section(
            section=merged.get(name) or {},
            section_path=f'{name}',
            base_defaults=base_defaults,
            structure=structure,
            family_label=family_label,
            explicit_pseudos=explicit_pseudos,
            electronic_type=electronic_type,
            spin_type=spin_type,
            initial_magnetic_moments=initial_magnetic_moments,
            absolute_param_overrides=_section_param_overrides(raw_overrides, name),
        )
        for name in names
    }


def _merge_runtime_config(*mappings: dict | None) -> dict:
    """Merge placement metadata mappings left to right, with the rightmost winning.

    Legacy monitored placement keys are normalized before merging.
    """
    merged: dict[str, dict] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            msg = '`runtime_config` must be a mapping of authored placement names.'
            raise ValueError(msg)
        mapping = _normalize_pw_leaves(mapping)
        for placement, metadata in mapping.items():
            if not isinstance(metadata, dict):
                msg = f'`runtime_config.{placement}` must be a mapping.'
                raise ValueError(msg)
            merged[placement] = _recursive_merge(merged.get(placement, {}), copy.deepcopy(metadata))
    return merged


def _relax_launch_inputs(
    *,
    merged: dict,
    raw_overrides: dict,
    base_defaults: dict,
    structure: orm.StructureData,
    protocol_code: orm.Code | None,
    explicit_pseudos: dict[str, orm.Node] | None,
    pseudo_family: str | None,
    electronic_type: ElectronicType,
    spin_type: SpinType,
    initial_magnetic_moments: dict | None,
    relax_type: RelaxType,
    kpoints: orm.KpointsData | None,
    init_kpoints: orm.KpointsData | None,
    options: dict | None,
    runtime_config: dict | None,
    monitors: orm.DataDict | None = None,
    init_monitors: orm.DataDict | None = None,
) -> dict:
    """Build the launch inputs of ``pw_relax_graph`` from merged relax protocol inputs."""
    if protocol_code is None:
        msg = '`code` is required for the `pw_relax` protocols.'
        raise ValueError(msg)
    sections = _resolve_sections(
        names=('base_relax', 'base_init_relax'),
        merged=merged,
        raw_overrides=raw_overrides,
        base_defaults=base_defaults,
        structure=structure,
        family_label=_resolve_family_label(
            kind_sections={name: merged.get(name) or {} for name in ('base_relax', 'base_init_relax')},
            base_default=base_defaults.get('pseudo_family'),
            pseudo_family=pseudo_family,
            spin_type=spin_type,
        ),
        explicit_pseudos=explicit_pseudos,
        electronic_type=electronic_type,
        spin_type=spin_type,
        initial_magnetic_moments=initial_magnetic_moments,
    )
    main, init = sections['base_relax'], sections['base_init_relax']

    settings, init_settings = _apply_relax_type(
        parameters=main['parameters'],
        init_parameters=init['parameters'],
        settings=main['settings'],
        init_settings=init['settings'],
        structure=structure,
        relax_type=relax_type,
    )
    runtime_config_final = _merge_runtime_config(
        {
            'init_relax_guarded': {'initial': _pw_scope_metadata(init['metadata_options'], options, protocol_code)},
            'main': _pw_scope_metadata(main['metadata_options'], options, protocol_code),
        },
        runtime_config,
    )
    _ensure_max_seconds(
        main['parameters'], _placement_options(runtime_config_final, 'main', 'pw_calc'), where='main.pw_calc'
    )
    _ensure_max_seconds(
        init['parameters'],
        _placement_options(runtime_config_final, 'init_relax_guarded', 'initial', 'pw_calc'),
        where='init_relax_guarded.initial.pw_calc',
    )
    parameters, settings_node, distance_node = _as_nodes(
        parameters=main['parameters'], settings=settings, distance=main['kpoints_distance']
    )
    _, init_settings_node, init_distance_node = _as_nodes(
        parameters=init['parameters'], settings=init_settings, distance=init['kpoints_distance']
    )
    launch: dict[str, t.Any] = {
        'structure': structure,
        'parameters': parameters,
        'init_parameters': orm.Dict(init['parameters']),
        'run_init_relax': True,
        'init_use_main_inputs': False,
        'code': protocol_code,
        'pseudos': orm.DataDict(main['pseudos']),
        'max_iterations': int(merged.get('max_meta_convergence_iterations', 5)),
        'meta_convergence': bool(merged.get('meta_convergence', True)),
        'kpoints_force_parity': main['kpoints_force_parity'],
        'init_kpoints_force_parity': init['kpoints_force_parity'],
    }
    if kpoints is not None:
        launch['kpoints'] = kpoints
    elif distance_node is not None:
        launch['kpoints_distance'] = distance_node
    if init_kpoints is not None:
        launch['init_kpoints'] = init_kpoints
    elif init_distance_node is not None:
        launch['init_kpoints_distance'] = init_distance_node
    if settings_node is not None:
        launch['settings'] = settings_node
    if init_settings_node is not None:
        launch['init_settings'] = init_settings_node
    shared = monitors
    yaml_main = main['monitors']
    yaml_init = init['monitors']
    eff_main = shared if shared is not None else yaml_main
    if init_monitors is not None:
        eff_init = init_monitors
    elif shared is not None:
        eff_init = shared
    else:
        eff_init = yaml_init
    if eff_main is not None:
        launch['monitors'] = eff_main
    if eff_init is not None:
        launch['init_monitors'] = eff_init
    launch['runtime_config'] = runtime_config_final
    return launch


def _bands_launch_inputs(
    *,
    merged: dict,
    raw_overrides: dict,
    base_defaults: dict,
    structure: orm.StructureData,
    scf_code: orm.Code | None,
    bands_code: orm.Code | None,
    explicit_pseudos: dict[str, orm.Node] | None,
    pseudo_family: str | None,
    electronic_type: ElectronicType,
    spin_type: SpinType,
    initial_magnetic_moments: dict | None,
    scf_kpoints: orm.KpointsData | None,
    bands_kpoints: orm.KpointsData | None,
    options: dict | None,
    runtime_config: dict | None,
    monitors: orm.DataDict | None = None,
    scf_monitors: orm.DataDict | None = None,
    bands_monitors: orm.DataDict | None = None,
) -> dict:
    """Build the launch inputs of ``pw_bands_graph`` from merged bands protocol inputs."""
    if scf_code is None or bands_code is None:
        msg = '`scf_code` and `bands_code` are required for the `pw_bands` protocols.'
        raise ValueError(msg)
    for name in ('scf', 'bands'):
        section_raw = (raw_overrides.get(name) or {}) if isinstance(raw_overrides.get(name), dict) else {}
        for key in ('kpoints_distance', 'kpoints_force_parity'):
            if name == 'bands' and key in section_raw:
                msg = f'`{name}.{key}` has no native-graph consumer; use the SCF mesh or `bands_kpoints`.'
                raise ValueError(msg)
    sections = _resolve_sections(
        names=('scf', 'bands'),
        merged=merged,
        raw_overrides=raw_overrides,
        base_defaults=base_defaults,
        structure=structure,
        family_label=_resolve_family_label(
            kind_sections={name: merged.get(name) or {} for name in ('scf', 'bands')},
            base_default=base_defaults.get('pseudo_family'),
            pseudo_family=pseudo_family,
            spin_type=spin_type,
        ),
        explicit_pseudos=explicit_pseudos,
        electronic_type=electronic_type,
        spin_type=spin_type,
        initial_magnetic_moments=initial_magnetic_moments,
    )
    scf, bands = sections['scf'], sections['bands']
    if bands_kpoints is not None and merged.get('bands_kpoints_distance') is not None:
        msg = 'Cannot specify both `bands_kpoints` and `bands_kpoints_distance`.'
        raise ValueError(msg)

    scf_metadata = _pw_scope_metadata(scf['metadata_options'], options, scf_code)
    bands_metadata = _pw_scope_metadata(bands['metadata_options'], options, bands_code)
    runtime_config_final = _merge_runtime_config(
        {
            'scf': scf_metadata,
            'bands': bands_metadata,
        },
        runtime_config,
    )
    _ensure_max_seconds(
        scf['parameters'], _placement_options(runtime_config_final, 'scf', 'pw_calc'), where='scf.pw_calc'
    )
    _ensure_max_seconds(
        bands['parameters'], _placement_options(runtime_config_final, 'bands', 'pw_calc'), where='bands.pw_calc'
    )
    scf_parameters, scf_settings, scf_distance = _as_nodes(
        parameters=scf['parameters'], settings=scf['settings'], distance=scf['kpoints_distance']
    )
    bands_parameters, bands_settings, _ = _as_nodes(
        parameters=bands['parameters'], settings=bands['settings'], distance=None
    )
    launch: dict[str, t.Any] = {
        'structure': structure,
        'scf_parameters': scf_parameters,
        'bands_parameters': bands_parameters,
        'scf_code': scf_code,
        'bands_code': bands_code,
        'pseudos': orm.DataDict(scf['pseudos']),
        'nbands_factor': orm.Float(merged['nbands_factor']) if merged.get('nbands_factor') is not None else None,
    }
    if scf_kpoints is not None:
        launch['scf_kpoints'] = scf_kpoints
    elif scf_distance is not None:
        launch['scf_kpoints_distance'] = scf_distance
    if bands_kpoints is not None:
        launch['bands_kpoints'] = bands_kpoints
    elif merged.get('bands_kpoints_distance') is not None:
        launch['bands_kpoints_distance'] = orm.Float(merged['bands_kpoints_distance'])
    if scf_settings is not None:
        launch['scf_settings'] = scf_settings
    if bands_settings is not None:
        launch['bands_settings'] = bands_settings
    yaml_scf_monitors = scf['monitors']
    yaml_bands_monitors = bands['monitors']
    if monitors is not None:
        launch['monitors'] = monitors
        if scf_monitors is not None and scf_monitors is not monitors:
            launch['scf_monitors'] = scf_monitors
        if bands_monitors is not None and bands_monitors is not monitors:
            launch['bands_monitors'] = bands_monitors
    else:
        eff_scf = scf_monitors if scf_monitors is not None else yaml_scf_monitors
        eff_bands = bands_monitors if bands_monitors is not None else yaml_bands_monitors
        if eff_scf is not None and eff_scf is eff_bands:
            launch['monitors'] = eff_scf
        else:
            if eff_scf is not None:
                launch['scf_monitors'] = eff_scf
            if eff_bands is not None:
                launch['bands_monitors'] = eff_bands
    launch['runtime_config'] = runtime_config_final
    return launch


def _post_section(merged: dict, name: str, path: str) -> dict:
    """Validate and return a DOS/PROJWFC protocol section with only supported keys."""
    section = merged.get(name) or {}
    _check_allowed(section, frozenset({'parameters', 'settings', 'metadata'}), f'{path}.{name}')
    metadata = section.get('metadata', {})
    _check_allowed(metadata, frozenset({'options'}), f'{path}.{name}.metadata')
    return {
        'parameters': copy.deepcopy(section.get('parameters', {})),
        'settings': copy.deepcopy(section.get('settings')),
        'metadata_options': copy.deepcopy(metadata.get('options', {})),
    }


def _checked_energy_range(value: t.Any) -> list | None:
    """Validate the PDOS energy window override, failing fast on malformed ranges."""
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(entry, (float, int)) for entry in value)
    ):
        msg = f'`energy_range_vs_fermi` should be a pair of numbers, but got: {value}'
        raise ValueError(msg)
    return [value[0], value[1]]


def _pdos_launch_inputs(
    *,
    merged: dict,
    raw_overrides: dict,
    base_defaults: dict,
    structure: orm.StructureData,
    pw_code: orm.Code | None,
    dos_code: orm.Code | None,
    projwfc_code: orm.Code | None,
    explicit_pseudos: dict[str, orm.Node] | None,
    pseudo_family: str | None,
    electronic_type: ElectronicType,
    spin_type: SpinType,
    initial_magnetic_moments: dict | None,
    scf_kpoints: orm.KpointsData | None,
    nscf_kpoints: orm.KpointsData | None,
    parent_folder: orm.RemoteData | None,
    parent_output_parameters: orm.Dict | dict | None,
    serial: bool,
    options: dict | None,
    runtime_config: dict | None,
    monitors: orm.DataDict | None = None,
    scf_monitors: orm.DataDict | None = None,
    nscf_monitors: orm.DataDict | None = None,
) -> dict:
    """Build the launch inputs of a ``pdos_from_*`` root from merged PDOS protocol inputs."""
    if pw_code is None or dos_code is None or projwfc_code is None:
        msg = '`pw_code`, `dos_code`, and `projwfc_code` are required for the `pdos` protocols.'
        raise ValueError(msg)
    from_parent = parent_folder is not None
    if from_parent and ('scf' in raw_overrides or scf_kpoints is not None):
        msg = 'from-parent PDOS takes no `scf` inputs; drop the `scf` overrides or run from scratch.'
        raise ValueError(msg)
    names = ('nscf',) if from_parent else ('scf', 'nscf')
    sections = _resolve_sections(
        names=names,
        merged=merged,
        raw_overrides=raw_overrides,
        base_defaults=base_defaults,
        structure=structure,
        family_label=_resolve_family_label(
            kind_sections={name: merged.get(name) or {} for name in names},
            base_default=base_defaults.get('pseudo_family'),
            pseudo_family=pseudo_family,
            spin_type=spin_type,
        ),
        explicit_pseudos=explicit_pseudos,
        electronic_type=electronic_type,
        spin_type=spin_type,
        initial_magnetic_moments=initial_magnetic_moments,
    )
    for name in names:
        sections[name]['parameters'].get('SYSTEM', {}).pop('smearing', None)
        sections[name]['parameters'].get('SYSTEM', {}).pop('degauss', None)
    nscf = sections['nscf']
    dos = _post_section(merged, 'dos', 'pdos')
    projwfc = _post_section(merged, 'projwfc', 'pdos')

    stage = 'pdos_stage_serial' if serial else 'pdos_stage_parallel'
    launch: dict[str, t.Any] = {
        'structure': structure,
        'pw_code': pw_code,
        'pseudos': orm.DataDict(nscf['pseudos']),
        'nscf_parameters': orm.Dict(nscf['parameters']),
        'dos_code': dos_code,
        'dos_parameters': orm.Dict(dos['parameters']),
        'projwfc_code': projwfc_code,
        'projwfc_parameters': orm.Dict(projwfc['parameters']),
        'energy_range_vs_fermi': _checked_energy_range(merged.get('energy_range_vs_fermi')),
        'nbands_factor': merged.get('nbands_factor'),
        'clean_workdir': bool(merged.get('clean_workdir', False)),
    }
    if not from_parent:
        scf = sections['scf']
        launch['scf_parameters'] = orm.Dict(scf['parameters'])
        if scf_kpoints is not None:
            launch['scf_kpoints'] = scf_kpoints
        elif scf['kpoints_distance'] is not None:
            launch['scf_kpoints_distance'] = orm.Float(scf['kpoints_distance'])
    else:
        launch['parent_folder'] = parent_folder
        if parent_output_parameters is not None:
            launch['parent_output_parameters'] = (
                parent_output_parameters
                if isinstance(parent_output_parameters, orm.Dict)
                else orm.Dict(parent_output_parameters)
            )
    if nscf_kpoints is not None:
        launch['nscf_kpoints'] = nscf_kpoints
    elif nscf['kpoints_distance'] is not None:
        launch['nscf_kpoints_distance'] = orm.Float(nscf['kpoints_distance'])
    if dos['settings'] is not None:
        launch['dos_settings'] = orm.Dict(dos['settings'])
    if projwfc['settings'] is not None:
        launch['projwfc_settings'] = orm.Dict(projwfc['settings'])
    scf_options = _pw_scope_metadata(sections['scf']['metadata_options'], options, pw_code) if not from_parent else None
    nscf_metadata = _pw_scope_metadata(nscf['metadata_options'], options, pw_code)
    dos_metadata = _placement_metadata(dos['metadata_options'], options, dos_code)
    projwfc_metadata = _placement_metadata(projwfc['metadata_options'], options, projwfc_code)
    yaml_scf_monitors = sections['scf']['monitors'] if not from_parent else None
    yaml_nscf_monitors = nscf['monitors']
    if from_parent:
        eff_parent = (
            nscf_monitors if nscf_monitors is not None else (monitors if monitors is not None else yaml_nscf_monitors)
        )
        if scf_monitors is not None:
            msg = '`scf_monitors` has no consumer for from-parent PDOS; pass `monitors` or `nscf_monitors`.'
            raise ValueError(msg)
        if eff_parent is not None:
            launch['monitors'] = eff_parent
    elif monitors is not None:
        launch['monitors'] = monitors
        if scf_monitors is not None and scf_monitors is not monitors:
            launch['scf_monitors'] = scf_monitors
        if nscf_monitors is not None and nscf_monitors is not monitors:
            launch['nscf_monitors'] = nscf_monitors
    else:
        eff_scf = scf_monitors if scf_monitors is not None else yaml_scf_monitors
        eff_nscf = nscf_monitors if nscf_monitors is not None else yaml_nscf_monitors
        if eff_scf is not None and eff_scf is eff_nscf:
            launch['monitors'] = eff_scf
        else:
            if eff_scf is not None:
                launch['scf_monitors'] = eff_scf
            if eff_nscf is not None:
                launch['nscf_monitors'] = eff_nscf
    stage_metadata: dict = {
        'nscf': nscf_metadata,
        'dos_calc': dos_metadata,
        'projwfc_calc': projwfc_metadata,
    }
    hierarchical = {stage: stage_metadata}
    if not from_parent:
        hierarchical['scf'] = scf_options
    runtime_config_final = _merge_runtime_config(hierarchical, runtime_config)
    _ensure_max_seconds(
        nscf['parameters'],
        _placement_options(runtime_config_final, stage, 'nscf', 'pw_calc'),
        where=f'{stage}.nscf.pw_calc',
    )
    launch['nscf_parameters'] = orm.Dict(nscf['parameters'])
    if not from_parent:
        _ensure_max_seconds(
            sections['scf']['parameters'],
            _placement_options(runtime_config_final, 'scf', 'pw_calc'),
            where='scf.pw_calc',
        )
        launch['scf_parameters'] = orm.Dict(sections['scf']['parameters'])
    launch['runtime_config'] = runtime_config_final
    return launch


def runtime_config_for_placements(**placements: dict | None) -> dict[str, dict]:
    """Form a ``runtime_config`` launch mapping from per-placement scheduler mappings.

    Each keyword names one scope of the structured ``runtime_config`` mapping: an
    authored subgraph callee alias (for example ``scf``) whose value is the nested
    metadata subtree below that scope, or — when launching a base graph directly
    — the ``CalcJob`` placement alias itself (for example ``pw_calc``) whose
    value is a process input mapping (for example ``{'metadata': {'options': {...}}}``).
    Entries with ``None`` are omitted; explicit empty mappings are retained::

        runtime_config_for_placements(scf={'pw_calc': {'metadata': {'options': scf_options}}},
                                  bands={'pw_calc': {'metadata': {'options': bands_options}}})

    :param placements: per-scope metadata subtrees keyed by scope name.
    :return: the ``runtime_config`` mapping to pass at launch.
    """
    return {name: dict(metadata) for name, metadata in placements.items() if metadata is not None}


def pdos_protocol_runtime_config(protocol_inputs: dict) -> dict[str, dict]:
    """Extract the ``dos``/``projwfc`` scheduler options of PDOS protocol inputs.

    The PDOS YAML carries ``dos.metadata.options`` and
    ``projwfc.metadata.options``; this helper lifts them into the nested
    ``dos_calc``/``projwfc_calc`` leaf fragment of the stage scope. ``pw``
    scheduler options are intentionally left out: the nested SCF/NSCF scopes
    take theirs under their own ``'pw_calc'`` leaves at launch. Nest the
    fragment under the stage scope to form a launch mapping::

        {stage: {'nscf': {'pw_calc': {'metadata': {'options': nscf_options}}},
                 **pdos_protocol_runtime_config(inputs)}}

    :param protocol_inputs: the merged mapping from :func:`get_protocol_inputs`.
    :return: the ``dos_calc``/``projwfc_calc`` leaf fragment of a PDOS stage scope.
    """
    runtime_config: dict[str, dict] = {}
    for placement, section in (('dos_calc', 'dos'), ('projwfc_calc', 'projwfc')):
        options = (protocol_inputs.get(section, {}) or {}).get('metadata', {}).get('options')
        if options:
            runtime_config[placement] = {'metadata': {'options': copy.deepcopy(options)}}
    return runtime_config
