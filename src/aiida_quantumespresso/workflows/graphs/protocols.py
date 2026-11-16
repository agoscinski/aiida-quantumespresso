############################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file      #
# For further information please visit http://www.aiida.net               #
############################################################################
"""Plain protocol helpers for the native PW/PDOS graphs.

The helpers return plain configuration mappings (``dict`` objects), never a
``ProcessBuilder``; protocol settings are not yet fully translated graph launch inputs. They read the same YAML protocol files the legacy work
chains consume but do not import any ``WorkChain`` subclass: pseudo-family and
meta-parameter (``conv_thr``/``etot_conv_thr``) resolution still lives in
``PwBaseWorkChain.get_builder_from_protocol`` and is deliberately not
re-implemented here. Callers that need fully resolved ``pw.parameters`` should
resolve pseudos and thresholds first, then pass explicit ``Dict`` nodes to the
graphs; the mappings below only merge protocol defaults, the named protocol,
and caller overrides.

Scheduler options are not graph inputs. Use :func:`task_metadata_for_options`
to form the ``task_metadata`` launch mapping keyed by placement name::

    run_get_node(
        pw_relax_graph, ..., task_metadata=task_metadata_for_options(pw_calc={'options': {...}})
    )

Only the helpers implemented here are supported: YAML loading/merging,
``available_protocols``, ``get_protocol_inputs``, and ``task_metadata``
formation. There is no ``get_builder_from_protocol``: building a legacy
builder would require importing the legacy work chains.
"""

import copy
import pathlib
import typing as t

import yaml

__all__ = (
    'PROTOCOL_KINDS',
    'available_protocols',
    'get_default_protocol',
    'get_protocol_filepath',
    'get_protocol_inputs',
    'pdos_protocol_task_metadata',
    'task_metadata_for_options',
)

PROTOCOL_KINDS = ('pw_relax', 'pw_bands', 'pdos')
"""Protocol families with a YAML file in ``workflows/protocols``."""

_RELATIVE_FILES: dict[str, pathlib.Path] = {
    'pw_relax': pathlib.Path('pw') / 'relax.yaml',
    'pw_bands': pathlib.Path('pw') / 'bands.yaml',
    'pdos': pathlib.Path('pdos.yaml'),
}

_ALIASES = {'moderate': 'balanced', 'precise': 'stringent'}


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
    scalars); the caller converts parameter mappings to ``Dict`` nodes and
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


def task_metadata_for_options(**placements: dict | None) -> dict[str, dict]:
    """Form a ``task_metadata`` launch mapping from per-placement scheduler mappings.

    Each keyword names a graph placement (for example ``pw_calc``, ``dos_calc``,
    ``projwfc_calc``) and each value is the metadata mapping for that placement
    (for example ``{'options': {...}}``). Placements with ``None`` or empty
    mappings are left out so the launch only carries what was requested.

    :param placements: per-placement metadata mappings keyed by placement name.
    :return: the ``task_metadata`` mapping to pass at launch.
    """
    return {name: dict(metadata) for name, metadata in placements.items() if metadata}


def pdos_protocol_task_metadata(protocol_inputs: dict) -> dict[str, dict]:
    """Extract the ``dos``/``projwfc`` scheduler options of PDOS protocol inputs.

    The PDOS YAML carries ``dos.metadata.options`` and
    ``projwfc.metadata.options``; this helper lifts them into a
    ``task_metadata`` mapping keyed by the graph placement names. ``pw`` XC
    options are intentionally left out: the native base graph takes scheduler
    options under its own ``'pw_calc'`` key at launch.

    :param protocol_inputs: the merged mapping from :func:`get_protocol_inputs`.
    :return: the ``task_metadata`` mapping for the ``dos_calc``/``projwfc_calc`` placements.
    """
    task_metadata: dict[str, dict] = {}
    for placement, section in (('dos_calc', 'dos'), ('projwfc_calc', 'projwfc')):
        options = (protocol_inputs.get(section, {}) or {}).get('metadata', {}).get('options')
        if options:
            task_metadata[placement] = {'options': copy.deepcopy(options)}
    return task_metadata
